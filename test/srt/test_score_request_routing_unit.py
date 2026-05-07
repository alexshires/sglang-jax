import asyncio
from types import SimpleNamespace

from sgl_jax.srt.managers.tokenizer_manager import _Communicator
from sgl_jax.srt.managers.tokenizer_score_routing_mixin import (
    TokenizerScoreRoutingMixin,
)


class _DummyTokenizerManager(TokenizerScoreRoutingMixin):
    def __init__(self, fan_out: int = 1):
        self.send_to_scheduler = _FakeSchedulerSender(fan_out)
        self.rid_to_state = {}

    def _notify_state_event(self, state):
        state.event.set()


class _FakeSchedulerSender:
    def __init__(self, fan_out: int):
        self.fan_out = fan_out
        self.sent = []
        self.sent_to = []
        self.sent_all = []

    def send_pyobj(self, obj):
        self.sent.append(obj)

    def send_pyobj_to(self, scheduler_idx, obj):
        self.sent_to.append((scheduler_idx, obj))

    def send_pyobj_all(self, obj):
        self.sent_all.append(obj)


def test_score_lane_scheduler_index_uses_cache_handle_hash_when_fan_out_enabled():
    manager = _DummyTokenizerManager(fan_out=4)
    index = manager._score_lane_scheduler_index("cache-handle")
    assert index is not None
    assert 0 <= index < 4


def test_send_one_request_broadcasts_cache_prefill_to_all_scheduler_lanes():
    manager = _DummyTokenizerManager(fan_out=3)
    obj = SimpleNamespace(rid="rid-prefill")
    tokenized_obj = SimpleNamespace(cache_for_scoring=True, extend_from_cache=None)

    state = manager._send_one_request(obj, tokenized_obj, created_time=1.0)

    assert manager.send_to_scheduler.sent_all == [tokenized_obj]
    assert manager.send_to_scheduler.sent == []
    assert manager.send_to_scheduler.sent_to == []
    assert state.expected_finish_count == 3
    assert manager.rid_to_state["rid-prefill"] is state


def test_send_one_request_routes_cache_extend_to_hashed_scheduler_lane():
    manager = _DummyTokenizerManager(fan_out=4)
    obj = SimpleNamespace(rid="rid-extend")
    tokenized_obj = SimpleNamespace(cache_for_scoring=False, extend_from_cache="cache-handle")
    expected_idx = manager._score_lane_scheduler_index("cache-handle")

    state = manager._send_one_request(obj, tokenized_obj, created_time=1.0)

    assert manager.send_to_scheduler.sent_to == [(expected_idx, tokenized_obj)]
    assert manager.send_to_scheduler.sent == []
    assert manager.send_to_scheduler.sent_all == []
    assert state.expected_finish_count == 1


def test_communicator_broadcasts_by_default_for_multi_lane_waiters():
    async def run_test():
        sender = _FakeSchedulerSender(fan_out=2)
        communicator = _Communicator(sender, fan_out=2)

        task = asyncio.create_task(communicator("payload"))
        await asyncio.sleep(0)

        assert sender.sent_all == ["payload"]
        assert sender.sent == []
        communicator.handle_recv("lane-0")
        communicator.handle_recv("lane-1")
        assert await task == ["lane-0", "lane-1"]

    asyncio.run(run_test())
