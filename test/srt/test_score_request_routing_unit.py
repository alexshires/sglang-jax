import asyncio
from types import SimpleNamespace

from sgl_jax.srt.managers.tokenizer_score_common import ReqState
from sgl_jax.srt.managers.tokenizer_score_routing_mixin import (
    TokenizerScoreRoutingMixin,
)


class _DummyTokenizerManager(TokenizerScoreRoutingMixin):
    def __init__(self, fan_out: int = 1):
        self.send_to_scheduler = _FakeSchedulerSender(fan_out)
        self.rid_to_state = {}
        self.scheduler_pids = []
        self.scheduler_unavailable_error = None
        self.health_check_failed = False
        self.server_args = SimpleNamespace(multi_item_score_local_rpc_min_items=8)
        self.local_rpc_submitter = None
        self.local_request_submitter = None

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


def test_local_score_rpc_requires_threshold_and_single_scheduler_lane():
    manager = _DummyTokenizerManager()
    manager.local_rpc_submitter = lambda _req: None

    assert manager._can_use_local_score_rpc(total_items=8) is True
    assert manager._can_use_local_score_rpc(total_items=7) is False


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


def test_send_batch_requests_routes_shared_cache_extends_to_one_lane():
    manager = _DummyTokenizerManager(fan_out=4)
    objs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
    tokenized_objs = [
        SimpleNamespace(cache_for_scoring=False, extend_from_cache="cache-handle"),
        SimpleNamespace(cache_for_scoring=False, extend_from_cache="cache-handle"),
    ]
    expected_idx = manager._score_lane_scheduler_index("cache-handle")

    states = manager._send_batch_requests(objs, tokenized_objs, created_time=1.0)

    assert manager.send_to_scheduler.sent_to == [(expected_idx, tokenized_objs)]
    assert manager.send_to_scheduler.sent == []
    assert len(states) == 2
    assert manager.rid_to_state["rid-a"] is states[0]
    assert manager.rid_to_state["rid-b"] is states[1]


def test_scheduler_health_failure_marks_pending_requests_unavailable():
    manager = _DummyTokenizerManager()
    manager.scheduler_pids = [-1]
    state = ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=SimpleNamespace(rid="rid-dead"),
        created_time=1.0,
    )
    manager.rid_to_state["rid-dead"] = state

    assert manager._check_and_handle_scheduler_health() is False

    assert manager.health_check_failed is True
    assert manager.scheduler_unavailable_error is not None
    assert "dead pid" in manager.scheduler_unavailable_error
    assert state.finished is True
    assert state.event.is_set()
    assert state.out_list[0]["meta_info"]["finish_reason"]["status_code"] == 503
    assert "rid-dead" not in manager.rid_to_state
