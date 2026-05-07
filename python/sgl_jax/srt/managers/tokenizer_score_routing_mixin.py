"""Tokenizer score routing and scheduler ingress helpers."""

from __future__ import annotations

import asyncio
import logging
import os
import zlib
from http import HTTPStatus

from sgl_jax.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.tokenizer_score_common import ReqState

logger = logging.getLogger(__name__)


class TokenizerScoreRoutingMixin:
    def _scheduler_sender_fan_out(self) -> int:
        return int(getattr(self.send_to_scheduler, "fan_out", 1) or 1)

    def _score_lane_scheduler_index(self, cache_handle: str | None) -> int | None:
        fan_out = self._scheduler_sender_fan_out()
        if fan_out <= 1 or not cache_handle:
            return None
        return zlib.crc32(cache_handle.encode("utf-8")) % fan_out

    def _can_use_local_score_rpc(self, total_items: int | None = None) -> bool:
        if not callable(getattr(self, "local_rpc_submitter", None)):
            return False
        if self._scheduler_sender_fan_out() > 1:
            return False
        if total_items is None:
            return True
        min_items = int(getattr(self.server_args, "multi_item_score_local_rpc_min_items", 256) or 0)
        if min_items <= 0:
            min_items = 256
        return int(total_items) >= min_items

    def _can_use_local_request_ingress(self, tokenized_obj) -> bool:
        if not callable(getattr(self, "local_request_submitter", None)):
            return False
        if self._scheduler_sender_fan_out() > 1:
            return False
        return bool(getattr(tokenized_obj, "cache_for_scoring", False))

    async def _submit_local_score_rpc(self, req_obj, timeout: float | None = None):
        submitter = getattr(self, "local_rpc_submitter", None)
        if not callable(submitter):
            raise RuntimeError("TokenizerManager local_rpc_submitter is not configured.")

        future = submitter(req_obj)
        wait_coro = asyncio.wrap_future(future)
        if timeout is not None and timeout > 0:
            result = await asyncio.wait_for(wait_coro, timeout=timeout)
        else:
            result = await wait_coro
        return [result]

    def _send_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        tokenized_obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput,
        created_time: float | None = None,
    ):
        self._raise_if_scheduler_unavailable()
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

    def _send_batch_requests(
        self,
        objs: list[GenerateReqInput | EmbeddingReqInput],
        tokenized_objs: list[TokenizedGenerateReqInput | TokenizedEmbeddingReqInput],
        created_time: float | None = None,
    ) -> list[ReqState]:
        if len(objs) != len(tokenized_objs):
            raise ValueError("objs and tokenized_objs must have the same length")
        if not objs:
            return []

        self._raise_if_scheduler_unavailable()
        try:
            caller_loop = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        if len(tokenized_objs) == 1 and self._can_use_local_request_ingress(tokenized_objs[0]):
            states: list[ReqState] = []
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
            self.local_request_submitter(tokenized_objs[0])
            return states

        payload = tokenized_objs[0] if len(tokenized_objs) == 1 else tokenized_objs
        scheduler_idx = None
        if tokenized_objs:
            extend_from_cache = getattr(tokenized_objs[0], "extend_from_cache", None)
            if extend_from_cache and all(
                getattr(tokenized_obj, "extend_from_cache", None) == extend_from_cache
                for tokenized_obj in tokenized_objs
            ):
                scheduler_idx = self._score_lane_scheduler_index(extend_from_cache)
        if scheduler_idx is not None:
            self.send_to_scheduler.send_pyobj_to(scheduler_idx, payload)
        else:
            self.send_to_scheduler.send_pyobj(payload)

        states: list[ReqState] = []
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

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _build_scheduler_unavailable_message(self) -> str | None:
        if not self.scheduler_pids:
            return None
        dead_pids = [pid for pid in self.scheduler_pids if not self._is_process_alive(pid)]
        if not dead_pids:
            return None
        return (
            "Scheduler subprocess is unavailable "
            f"(dead pid(s): {', '.join(str(pid) for pid in dead_pids)}). "
            "Please restart the server."
        )

    def _fail_pending_requests(self, message: str) -> None:
        for rid, state in list(self.rid_to_state.items()):
            if state.finished:
                continue
            state.finished = True
            state.out_list.append(
                {
                    "text": "",
                    "meta_info": {
                        "id": rid,
                        "finish_reason": {
                            "type": "abort",
                            "message": message,
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                        },
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    },
                }
            )
            self._notify_state_event(state)
            self.rid_to_state.pop(rid, None)

    def _mark_scheduler_unavailable(self, message: str) -> None:
        if self.scheduler_unavailable_error is None:
            logger.error(message)
        self.scheduler_unavailable_error = message
        self.health_check_failed = True
        self._fail_pending_requests(message)

    def _check_and_handle_scheduler_health(self) -> bool:
        if self.scheduler_unavailable_error is not None:
            return False
        message = self._build_scheduler_unavailable_message()
        if message is None:
            return True
        self._mark_scheduler_unavailable(message)
        return False

    def _raise_if_scheduler_unavailable(self) -> None:
        if self._check_and_handle_scheduler_health():
            return
        raise ValueError(
            self.scheduler_unavailable_error
            or "Scheduler subprocess is unavailable. Please restart the server."
        )
