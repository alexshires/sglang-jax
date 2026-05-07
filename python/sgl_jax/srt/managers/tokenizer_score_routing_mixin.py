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
        expected_finish_count = 1
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

    def _send_batch_requests(
        self,
        objs: list[GenerateReqInput | EmbeddingReqInput],
        tokenized_objs: list[TokenizedGenerateReqInput | TokenizedEmbeddingReqInput],
        created_time: float | None = None,
    ) -> list[ReqState]:
        if len(objs) != len(tokenized_objs):
            raise ValueError("objs and tokenized_objs must have the same length")

        try:
            caller_loop = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None

        scheduler_idx = None
        if tokenized_objs:
            extend_handles = {
                getattr(tokenized_obj, "extend_from_cache", None)
                for tokenized_obj in tokenized_objs
            }
            if len(extend_handles) == 1:
                scheduler_idx = self._score_lane_scheduler_index(next(iter(extend_handles)))

        payload = tokenized_objs[0] if len(tokenized_objs) == 1 else tokenized_objs
        if scheduler_idx is not None:
            self.send_to_scheduler.send_pyobj_to(scheduler_idx, payload)
        else:
            self.send_to_scheduler.send_pyobj(payload)

        states = []
        for obj in objs:
            state = ReqState(
                [],
                False,
                asyncio.Event(),
                obj,
                created_time=created_time,
                event_loop=caller_loop,
            )
            rid_key = obj.rid[0] if isinstance(obj.rid, list) else obj.rid
            self.rid_to_state[rid_key] = state
            states.append(state)
        return states
