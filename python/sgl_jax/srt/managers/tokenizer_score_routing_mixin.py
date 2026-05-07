"""Tokenizer score routing and scheduler ingress helpers."""

from __future__ import annotations

import asyncio
import zlib

from sgl_jax.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.tokenizer_score_common import ReqState


class TokenizerScoreRoutingMixin:
    def _scheduler_sender_fan_out(self) -> int:
        return int(getattr(self.send_to_scheduler, "fan_out", 1) or 1)

    def _score_lane_scheduler_index(self, cache_handle: str | None) -> int | None:
        fan_out = self._scheduler_sender_fan_out()
        if fan_out <= 1 or not cache_handle:
            return None
        return zlib.crc32(cache_handle.encode("utf-8")) % fan_out

    def _can_use_local_request_ingress(self, tokenized_obj) -> bool:
        if not callable(getattr(self, "local_request_submitter", None)):
            return False
        if self._scheduler_sender_fan_out() > 1:
            return False
        return bool(getattr(tokenized_obj, "cache_for_scoring", False))

    def _send_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        tokenized_obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput,
        created_time: float | None = None,
    ):
        try:
            caller_loop = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        use_local_ingress = self._can_use_local_request_ingress(tokenized_obj)
        expected_finish_count = 1
        if use_local_ingress:
            state = ReqState(
                [],
                False,
                asyncio.Event(),
                obj,
                created_time=created_time,
                event_loop=caller_loop,
                expected_finish_count=expected_finish_count,
            )
            rid_key = obj.rid[0] if isinstance(obj.rid, list) else obj.rid
            self.rid_to_state[rid_key] = state
            self.local_request_submitter(tokenized_obj)
            return state
        if (
            bool(getattr(tokenized_obj, "cache_for_scoring", False))
            and self._scheduler_sender_fan_out() > 1
        ):
            self.send_to_scheduler.send_pyobj_all(tokenized_obj)
            expected_finish_count = self._scheduler_sender_fan_out()
        else:
            scheduler_idx = self._score_lane_scheduler_index(
                getattr(tokenized_obj, "extend_from_cache", None)
            )
            if scheduler_idx is not None:
                self.send_to_scheduler.send_pyobj_to(scheduler_idx, tokenized_obj)
            else:
                self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = ReqState(
            [],
            False,
            asyncio.Event(),
            obj,
            created_time=created_time,
            event_loop=caller_loop,
            expected_finish_count=expected_finish_count,
        )
        # Handle rid being a list (single element) or string
        rid_key = obj.rid[0] if isinstance(obj.rid, list) else obj.rid
        self.rid_to_state[rid_key] = state
        return state
