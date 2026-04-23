"""Tokenizer scoring helper types."""

# ruff: noqa: F401
import asyncio
import dataclasses
import logging
import math
import os
import time
import uuid
import zlib
from http import HTTPStatus
from typing import Any

from sgl_jax.srt.managers.io_struct import (
    GenerateReqInput,
    EmbeddingReqInput,
    ReleaseScoringCacheReqInput,
    ReleaseScoringCacheReqOutput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.validation import (
    ValidationError,
    validate_score_request,
)

logger = logging.getLogger(__name__)

def _stable_softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    if total == 0:
        return [0.0 for _ in exp_values]
    return [value / total for value in exp_values]

class _SchedulerSender:
    """Route tokenizer requests to one or more scheduler sockets."""

    def __init__(self, senders: list[Any]):
        if not senders:
            raise ValueError("At least one scheduler sender is required.")
        self._senders = list(senders)
        self._rr_index = 0

    @property
    def fan_out(self) -> int:
        return len(self._senders)

    def send_pyobj(self, obj):
        sender = self._senders[self._rr_index % len(self._senders)]
        self._rr_index = (self._rr_index + 1) % len(self._senders)
        sender.send_pyobj(obj)

    def send_pyobj_to(self, scheduler_idx: int, obj):
        self._senders[scheduler_idx].send_pyobj(obj)

    def send_pyobj_all(self, obj):
        for sender in self._senders:
            sender.send_pyobj(obj)

@dataclasses.dataclass
class _CorrelatedWaiter[T]:
    event: asyncio.Event
    values: list[T]

class _CorrelatedCommunicator[T]:
    """Allow multiple in-flight RPCs by correlating responses with `rid`."""

    def __init__(self, sender, fan_out: int):
        self._sender = sender
        self._fan_out = fan_out
        self._pending: dict[str, _CorrelatedWaiter[T]] = {}

    async def __call__(
        self,
        obj,
        timeout: float | None = None,
        scheduler_idx: int | None = None,
        broadcast: bool = False,
    ):
        rid = getattr(obj, "rid", None)
        if not rid:
            raise ValueError(
                "Correlated communicator requires request objects with non-empty `rid`."
            )
        if rid in self._pending:
            raise RuntimeError(f"Duplicate in-flight correlated request rid={rid!r}.")

        waiter = _CorrelatedWaiter(event=asyncio.Event(), values=[])
        self._pending[rid] = waiter
        try:
            if obj is not None:
                if broadcast and hasattr(self._sender, "send_pyobj_all"):
                    self._sender.send_pyobj_all(obj)
                elif scheduler_idx is not None and hasattr(self._sender, "send_pyobj_to"):
                    self._sender.send_pyobj_to(scheduler_idx, obj)
                else:
                    self._sender.send_pyobj(obj)

            wait_coro = waiter.event.wait()
            if timeout is not None and timeout > 0:
                await asyncio.wait_for(wait_coro, timeout=timeout)
            else:
                await wait_coro
            return list(waiter.values)
        finally:
            self._pending.pop(rid, None)

    def handle_recv(self, recv_obj: T):
        rid = getattr(recv_obj, "rid", None)
        if not rid:
            logger.warning(
                "Dropping correlated communicator response missing rid. type=%s",
                type(recv_obj).__name__,
            )
            return

        waiter = self._pending.get(rid)
        if waiter is None:
            logger.warning(
                "Dropping correlated communicator response with no active waiter. rid=%s type=%s",
                rid,
                type(recv_obj).__name__,
            )
            return

        waiter.values.append(recv_obj)
        if len(waiter.values) >= self._fan_out:
            waiter.event.set()

__all__ = [name for name in globals() if not name.startswith("__")]
