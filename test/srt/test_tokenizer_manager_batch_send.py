import asyncio
import zlib
from types import SimpleNamespace

import pytest
import zmq

from sgl_jax.srt.managers.io_struct import (
    ReleaseScoringCacheReqInput,
    ScoreFromCacheReqInput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.scheduler_scoring_state_mixin import (
    SchedulerScoringStateMixin,
)
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.managers.tokenizer_score_common import ReqState
from sgl_jax.srt.managers.tokenizer_score_routing_mixin import (
    TokenizerScoreRoutingMixin,
)


class _NoopSender:
    def __init__(self, fan_out: int = 1):
        self.fan_out = fan_out
        self.calls = []

    def send_pyobj(self, obj):
        self.calls.append(("default", obj))

    def send_pyobj_to(self, scheduler_idx, obj):
        self.calls.append((scheduler_idx, obj))

    def send_pyobj_all(self, obj):
        self.calls.append(("all", obj))


class _FakeBatchSendManager(TokenizerScoreRoutingMixin):
    def __init__(self, fan_out: int = 1):
        self.rid_to_state = {}
        self.send_to_scheduler = _NoopSender(fan_out=fan_out)

    def _raise_if_scheduler_unavailable(self):
        return None


class _FakeBatchRequestContainer:
    def __init__(self, requests: list[SimpleNamespace], stream: bool = False):
        self._requests = requests
        self.batch_size = len(requests)
        self.stream = stream
        self.parallel_sample_num = 1

    def __getitem__(self, index: int) -> SimpleNamespace:
        return self._requests[index]


class _FakeBatchHandleManager:
    _handle_batch_request = TokenizerManager._handle_batch_request

    def __init__(self, enable_batch_send: bool, enable_batch_encode: bool):
        self.server_args = SimpleNamespace(
            enable_tokenizer_batch_encode=enable_batch_encode,
            enable_tokenizer_batch_send=enable_batch_send,
        )
        self.sent_single = []
        self.sent_batch = []

    def _validate_batch_tokenization_constraints(self, batch_size, obj):
        del batch_size, obj
        return None

    async def _batch_tokenize_and_process(self, batch_size: int, obj):
        del obj
        return [SimpleNamespace(tokenized_idx=i) for i in range(batch_size)]

    async def _tokenize_one_request(self, obj):
        return SimpleNamespace(tokenized_single=obj.rid)

    def _send_one_request(self, obj, tokenized_obj, created_time=None):
        self.sent_single.append((obj.rid, tokenized_obj, created_time))
        return ReqState([], True, asyncio.Event(), obj, created_time=created_time)

    def _send_batch_requests(self, objs, tokenized_objs, created_time=None):
        self.sent_batch.append(
            ([obj.rid for obj in objs], tokenized_objs, created_time)
        )
        return [
            ReqState([], True, asyncio.Event(), obj, created_time=created_time)
            for obj in objs
        ]

    async def _wait_one_response(self, obj, state, request=None):
        del state, request
        yield {"meta_info": {"id": obj.rid}, "text": "", "index": 0}


class _FakeIngressSocket:
    def __init__(self, payloads: list):
        self.payloads = list(payloads)

    def recv_pyobj(self, flags=None):
        del flags
        if self.payloads:
            return self.payloads.pop(0)
        raise zmq.ZMQError()


class _FakeSchedulerIngress:
    recv_requests = SchedulerScoringStateMixin.recv_requests

    def __init__(self, tokenizer_payloads: list, rpc_payloads: list):
        self.node_rank = 0
        self.nnodes = 1
        self.recv_from_tokenizer = _FakeIngressSocket(tokenizer_payloads)
        self.recv_from_rpc = _FakeIngressSocket(rpc_payloads)
        self.ingress_recv_calls = 0
        self.ingress_nonempty_calls = 0
        self.ingress_max_batch_size = 0
        self.ingress_tokenizer_frames = 0
        self.ingress_rpc_frames = 0
        self.ingress_tokenizer_messages = 0
        self.ingress_rpc_messages = 0
        self.ingress_batch_size_histogram = {
            "eq_0": 0,
            "eq_1": 0,
            "2_to_4": 0,
            "5_to_16": 0,
            "gt_16": 0,
        }
        self.ingress_score_paths = {
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }
        self.ingress_score_path_frames = {
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }


def test_send_batch_requests_sends_one_zmq_frame_and_tracks_all_states():
    manager = _FakeBatchSendManager()
    reqs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
    tokenized_objs = [SimpleNamespace(tok=1), SimpleNamespace(tok=2)]

    states = manager._send_batch_requests(reqs, tokenized_objs, created_time=1.0)

    assert len(states) == 2
    assert set(manager.rid_to_state) == {"rid-a", "rid-b"}
    assert manager.send_to_scheduler.calls == [("default", tokenized_objs)]


def test_send_batch_requests_returns_no_states_for_empty_batch():
    manager = _FakeBatchSendManager()

    states = manager._send_batch_requests([], [], created_time=0.0)

    assert states == []
    assert manager.rid_to_state == {}
    assert manager.send_to_scheduler.calls == []


def test_send_batch_requests_unwraps_single_element_payload():
    manager = _FakeBatchSendManager()
    req = SimpleNamespace(rid="rid-a")
    tokenized_obj = SimpleNamespace(tok=1)

    states = manager._send_batch_requests([req], [tokenized_obj], created_time=1.0)

    assert len(states) == 1
    assert manager.rid_to_state == {"rid-a": states[0]}
    assert manager.send_to_scheduler.calls == [("default", tokenized_obj)]


def test_send_batch_requests_rejects_length_mismatch():
    manager = _FakeBatchSendManager()

    with pytest.raises(ValueError, match="same length"):
        manager._send_batch_requests(
            [SimpleNamespace(rid="rid-a")], [], created_time=0.0
        )


def test_send_batch_requests_routes_uniform_extend_cache_batch_to_lane():
    manager = _FakeBatchSendManager(fan_out=4)
    cache_handle = "cache-handle-1"
    reqs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
    tokenized_objs = [
        SimpleNamespace(extend_from_cache=cache_handle),
        SimpleNamespace(extend_from_cache=cache_handle),
    ]

    states = manager._send_batch_requests(reqs, tokenized_objs, created_time=1.0)

    expected_idx = zlib.crc32(cache_handle.encode("utf-8")) % 4
    assert len(states) == 2
    assert manager.send_to_scheduler.calls == [(expected_idx, tokenized_objs)]


def test_send_batch_requests_falls_back_for_heterogeneous_extend_handles():
    manager = _FakeBatchSendManager(fan_out=4)
    reqs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
    tokenized_objs = [
        SimpleNamespace(extend_from_cache="cache-a"),
        SimpleNamespace(extend_from_cache="cache-b"),
    ]

    states = manager._send_batch_requests(reqs, tokenized_objs, created_time=1.0)

    expected_calls = [
        (zlib.crc32(b"cache-a") % 4, tokenized_objs[0]),
        (zlib.crc32(b"cache-b") % 4, tokenized_objs[1]),
    ]
    assert len(states) == 2
    assert manager.send_to_scheduler.calls == expected_calls


def test_send_batch_requests_falls_back_for_cache_for_scoring_broadcast():
    manager = _FakeBatchSendManager(fan_out=4)
    reqs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
    tokenized_objs = [
        SimpleNamespace(cache_for_scoring=True),
        SimpleNamespace(cache_for_scoring=True),
    ]

    states = manager._send_batch_requests(reqs, tokenized_objs, created_time=1.0)

    assert [state.expected_finish_count for state in states] == [4, 4]
    assert manager.send_to_scheduler.calls == [
        ("all", tokenized_objs[0]),
        ("all", tokenized_objs[1]),
    ]


@pytest.mark.parametrize("enable_batch_send", [False, True])
def test_handle_batch_request_switches_between_single_and_batch_send(
    enable_batch_send: bool,
):
    manager = _FakeBatchHandleManager(
        enable_batch_send=enable_batch_send,
        enable_batch_encode=True,
    )
    obj = _FakeBatchRequestContainer(
        [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")],
        stream=False,
    )

    async def _collect():
        outputs = []
        async for out in manager._handle_batch_request(
            obj, request=None, created_time=0.0
        ):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_collect())

    assert len(outputs) == 1
    assert len(outputs[0]) == 2
    if enable_batch_send:
        assert len(manager.sent_batch) == 1
        assert manager.sent_single == []
    else:
        assert manager.sent_batch == []
        assert len(manager.sent_single) == 2


def test_scheduler_recv_requests_unpacks_list_payloads_and_tracks_ingress_metrics():
    tokenizer_payload = [
        TokenizedGenerateReqInput(
            rid="tok-1",
            input_ids=[1, 2],
            sampling_params={},
            cache_for_scoring=True,
        ),
        TokenizedGenerateReqInput(
            rid="tok-2",
            input_ids=[1, 3],
            sampling_params={},
            extend_from_cache="cache-handle-1",
        ),
    ]
    rpc_payload = [
        ScoreFromCacheReqInput(
            rid="rpc-1",
            cache_handle="cache-handle-1",
            items_2d=[[7, 8]],
            label_token_ids=[198],
        ),
        ReleaseScoringCacheReqInput(rid="rpc-2"),
    ]
    scheduler = _FakeSchedulerIngress(
        tokenizer_payloads=[tokenizer_payload],
        rpc_payloads=[rpc_payload],
    )

    recv_reqs = scheduler.recv_requests()

    assert len(recv_reqs) == 4
    assert scheduler.ingress_tokenizer_frames == 1
    assert scheduler.ingress_rpc_frames == 1
    assert scheduler.ingress_tokenizer_messages == 2
    assert scheduler.ingress_rpc_messages == 2
    assert scheduler.ingress_nonempty_calls == 1
    assert scheduler.ingress_max_batch_size == 4
    assert scheduler.ingress_batch_size_histogram["2_to_4"] == 1
    assert scheduler.ingress_score_paths["tokenizer_cache_for_scoring"] == 1
    assert scheduler.ingress_score_paths["tokenizer_extend_from_cache"] == 1
    assert scheduler.ingress_score_paths["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_paths["rpc_release_scoring_cache"] == 1
