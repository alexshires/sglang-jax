"""Tokenizer scoring helper types."""

import asyncio
import dataclasses
import math
from typing import Any

from sgl_jax.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
)


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: list[dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: GenerateReqInput | EmbeddingReqInput

    # For metrics
    created_time: float
    event_loop: asyncio.AbstractEventLoop | None = None
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    last_completion_tokens: int = 1

    # For streaming output
    last_output_offset: int = 0
    expected_finish_count: int = 1
    observed_finish_count: int = 0

    text: str = ""
    output_ids: list[int] = dataclasses.field(default_factory=list)
    input_token_logprobs_val: list[float] = dataclasses.field(default_factory=list)
    input_token_logprobs_idx: list[int] = dataclasses.field(default_factory=list)
    output_token_logprobs_val: list[float] = dataclasses.field(default_factory=list)
    output_token_logprobs_idx: list[int] = dataclasses.field(default_factory=list)
    input_top_logprobs_val: list[list[float]] = dataclasses.field(default_factory=list)
    input_top_logprobs_idx: list[list[int]] = dataclasses.field(default_factory=list)
    output_top_logprobs_val: list[list[float]] = dataclasses.field(default_factory=list)
    output_top_logprobs_idx: list[list[int]] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_val: list = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_idx: list = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_val: list = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_idx: list = dataclasses.field(default_factory=list)


def _stable_softmax(values: list[float]) -> list[float]:
    # Score requests use short label lists; full-vocab softmax stays on device.
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
        if not 0 <= scheduler_idx < len(self._senders):
            raise IndexError(
                f"scheduler_idx={scheduler_idx} out of range for {len(self._senders)} senders"
            )
        self._senders[scheduler_idx].send_pyobj(obj)

    def send_pyobj_all(self, obj):
        for sender in self._senders:
            sender.send_pyobj(obj)
