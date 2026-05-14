import asyncio
import logging
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqOutput
from sgl_jax.srt.managers.tokenizer_score_cache_mixin import TokenizerScoreCacheMixin
from sgl_jax.srt.managers.tokenizer_score_common import _CorrelatedCommunicator
from sgl_jax.srt.validation import ValidationError


class _FakeTokenizer:
    def __len__(self):
        return 100

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) % 50 + 1 for char in text]


class _DummyScoreCacheManager(TokenizerScoreCacheMixin):
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.server_args = SimpleNamespace(
            multi_item_score_from_cache_v2_items_per_step=16,
            multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget=False,
            multi_item_score_from_cache_v2_token_budget=0,
            multi_item_score_from_cache_v2_min_items_per_step=1,
            multi_item_enable_score_from_cache_v2=True,
            multi_item_extend_batch_size=2,
            multi_item_score_fastpath_log_metrics=False,
        )
        self.generated_requests = []
        self.fastpath_calls = []
        self.released_handles = []
        self.release_started = None
        self.release_continue = None
        self.release_result = True
        self.release_exception = None
        self.asyncio_tasks = set()
        self.score_fastpath_attempted = 0
        self.score_fastpath_succeeded = 0
        self.score_fastpath_fallback = 0
        self.score_fastpath_fallback_reasons = {}
        self.fastpath_response = ScoreFromCacheReqOutput(
            success=True,
            scores=[[0.2, 0.8], [0.3, 0.7]],
        )
        self.batched_extend_calls = []

    async def generate_request(self, req):
        self.generated_requests.append(req)
        if False:
            yield {}

    async def _score_from_cache_fastpath_v2(self, **kwargs):
        self.fastpath_calls.append(kwargs)
        return self.fastpath_response

    async def _batched_extend_score_with_metrics(
        self,
        cache_handle,
        items,
        label_token_ids,
        apply_softmax=False,
    ):
        self.batched_extend_calls.append(
            {
                "cache_handle": cache_handle,
                "items": items,
                "label_token_ids": label_token_ids,
                "apply_softmax": apply_softmax,
            }
        )
        return (
            [[0.4, 0.6] for _ in items],
            {
                "dispatch_count": 1,
                "queue_wait_s": 0.0,
                "device_compute_s": 0.0,
                "host_orchestration_s": 0.0,
                "lifecycle_requests_sent": len(items),
                "lifecycle_results_received": len(items),
            },
        )

    async def _release_cache(self, cache_handle):
        if self.release_exception is not None:
            raise self.release_exception
        if self.release_started is not None:
            self.release_started.set()
        if self.release_continue is not None:
            await self.release_continue.wait()
        self.released_handles.append(cache_handle)
        return self.release_result


async def _drain_asyncio_tasks(manager):
    tasks = tuple(manager.asyncio_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)


def test_prefill_scoring_cache_builds_prefill_only_request():
    manager = _DummyScoreCacheManager()

    cache_handle = asyncio.run(manager.prefill_scoring_cache("hi"))

    assert len(cache_handle) == 32
    assert len(manager.generated_requests) == 1
    req = manager.generated_requests[0]
    assert req.input_ids == manager.tokenizer.encode("hi", add_special_tokens=False)
    assert req.sampling_params == {"max_new_tokens": 1}
    assert req.return_logprob is False
    assert req.cache_for_scoring is True
    assert req.is_single is True
    assert req.rid == cache_handle


def test_score_from_cache_uses_fastpath_with_normalized_items():
    manager = _DummyScoreCacheManager()

    scores = asyncio.run(
        manager.score_from_cache(
            "cache-1",
            items=["a", "bc"],
            label_token_ids=[1, 2],
            apply_softmax=True,
        )
    )

    assert scores == [[0.2, 0.8], [0.3, 0.7]]
    assert len(manager.fastpath_calls) == 1
    call = manager.fastpath_calls[0]
    assert call["cache_handle"] == "cache-1"
    assert call["items"] == [
        manager.tokenizer.encode("a", add_special_tokens=False),
        manager.tokenizer.encode("bc", add_special_tokens=False),
    ]
    assert call["label_token_ids"] == [1, 2]
    assert call["apply_softmax"] is True
    assert call["items_per_step"] == 16


def test_score_from_cache_rejects_empty_cache_handle():
    manager = _DummyScoreCacheManager()

    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(manager.score_from_cache("", items=[[1]], label_token_ids=[1]))

    assert exc_info.value.param == "cache_handle"
    assert exc_info.value.code == "invalid_cache_handle"


def test_release_scoring_cache_validates_then_delegates():
    manager = _DummyScoreCacheManager()

    assert asyncio.run(manager.release_scoring_cache("cache-1")) is True
    assert manager.released_handles == ["cache-1"]

    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(manager.release_scoring_cache(""))
    assert exc_info.value.param == "cache_handle"
    assert exc_info.value.code == "invalid_cache_handle"


class _FakeSender:
    def __init__(self):
        self.round_robin_requests = []
        self.lane_requests = []
        self.broadcast_requests = []

    def send_pyobj(self, obj):
        self.round_robin_requests.append(obj)

    def send_pyobj_to(self, scheduler_idx, obj):
        self.lane_requests.append((scheduler_idx, obj))

    def send_pyobj_all(self, obj):
        self.broadcast_requests.append(obj)


def test_correlated_communicator_single_lane_waits_for_one_response():
    async def run_test():
        sender = _FakeSender()
        communicator = _CorrelatedCommunicator(sender, fan_out=2)
        req = SimpleNamespace(rid="rid-lane")

        task = asyncio.create_task(communicator(req, scheduler_idx=1, timeout=1.0))
        await asyncio.sleep(0)

        assert sender.lane_requests == [(1, req)]
        communicator.handle_recv(SimpleNamespace(rid="rid-lane", value="lane-result"))

        outputs = await task
        assert [output.value for output in outputs] == ["lane-result"]

    asyncio.run(run_test())


def test_correlated_communicator_broadcast_waits_for_all_responses():
    async def run_test():
        sender = _FakeSender()
        communicator = _CorrelatedCommunicator(sender, fan_out=2)
        req = SimpleNamespace(rid="rid-broadcast")

        task = asyncio.create_task(communicator(req, broadcast=True, timeout=1.0))
        await asyncio.sleep(0)

        assert sender.broadcast_requests == [req]
        communicator.handle_recv(SimpleNamespace(rid="rid-broadcast", value="lane-0"))
        await asyncio.sleep(0)
        assert not task.done()

        communicator.handle_recv(SimpleNamespace(rid="rid-broadcast", value="lane-1"))
        outputs = await task
        assert [output.value for output in outputs] == ["lane-0", "lane-1"]

    asyncio.run(run_test())


def test_correlated_communicator_correlates_by_rid():
    async def run_test():
        sender = _FakeSender()
        communicator = _CorrelatedCommunicator(sender, fan_out=2)

        task_a = asyncio.create_task(
            communicator(SimpleNamespace(rid="rid-a"), scheduler_idx=0, timeout=1.0)
        )
        task_b = asyncio.create_task(
            communicator(SimpleNamespace(rid="rid-b"), scheduler_idx=1, timeout=1.0)
        )
        await asyncio.sleep(0)

        communicator.handle_recv(SimpleNamespace(rid="rid-b", value="b"))
        communicator.handle_recv(SimpleNamespace(rid="rid-a", value="a"))

        outputs_a = await task_a
        outputs_b = await task_b
        assert [output.value for output in outputs_a] == ["a"]
        assert [output.value for output in outputs_b] == ["b"]

    asyncio.run(run_test())


def test_correlated_communicator_rejects_duplicate_in_flight_rid():
    async def run_test():
        sender = _FakeSender()
        communicator = _CorrelatedCommunicator(sender, fan_out=1)
        task = asyncio.create_task(
            communicator(SimpleNamespace(rid="same-rid"), scheduler_idx=0, timeout=1.0)
        )
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="Duplicate in-flight"):
            await communicator(SimpleNamespace(rid="same-rid"), scheduler_idx=0, timeout=1.0)

        communicator.handle_recv(SimpleNamespace(rid="same-rid", value="ok"))
        outputs = await task
        assert [output.value for output in outputs] == ["ok"]

    asyncio.run(run_test())


def test_partition_score_from_cache_v2_items_balances_token_loads():
    manager = _DummyScoreCacheManager()
    items = [
        [0] * 10,
        [1] * 2,
        [2] * 8,
        [3],
        [4],
    ]

    partitions = manager._partition_score_from_cache_v2_items(items, scheduler_count=2)

    assert [[idx for idx, _ in partition] for partition in partitions] == [
        [0, 3],
        [1, 2, 4],
    ]


def test_resolve_score_from_cache_v2_items_per_step_uses_token_budget():
    manager = _DummyScoreCacheManager()
    manager.server_args.multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget = True
    manager.server_args.multi_item_score_from_cache_v2_token_budget = 25
    manager.server_args.multi_item_score_from_cache_v2_min_items_per_step = 3

    items_per_step, max_total_tokens, token_budget = (
        manager._resolve_score_from_cache_v2_items_per_step(
            query_tokens=[1, 2, 3, 4],
            items=[[5] * 6, [6] * 2],
        )
    )

    assert items_per_step == 3
    assert max_total_tokens == 10
    assert token_budget == 25


def test_score_prefill_extend_updates_fastpath_success_counters():
    manager = _DummyScoreCacheManager()

    async def run_test():
        scores = await manager.score_prefill_extend(
            query_tokens=[1, 2],
            item_tokens_list=[[3], [4]],
            label_token_ids=[5, 6],
        )
        await _drain_asyncio_tasks(manager)
        return scores

    scores = asyncio.run(run_test())

    assert scores == [[0.2, 0.8], [0.3, 0.7]]
    assert manager.score_fastpath_attempted == 1
    assert manager.score_fastpath_succeeded == 1
    assert manager.score_fastpath_fallback == 0
    assert len(manager.released_handles) == 1


def test_score_prefill_extend_records_fastpath_fallback_counter():
    manager = _DummyScoreCacheManager()
    manager.fastpath_response = ScoreFromCacheReqOutput(
        success=False,
        scores=[],
        fallback_reason="test_fallback",
        error_msg="forced fallback",
    )

    async def run_test():
        scores = await manager.score_prefill_extend(
            query_tokens=[1, 2],
            item_tokens_list=[[3], [4], [5]],
            label_token_ids=[6, 7],
        )
        await _drain_asyncio_tasks(manager)
        return scores

    scores = asyncio.run(run_test())

    assert scores == [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]
    assert manager.score_fastpath_attempted == 1
    assert manager.score_fastpath_succeeded == 0
    assert manager.score_fastpath_fallback == 1
    assert manager.score_fastpath_fallback_reasons == {"test_fallback": 1}
    assert len(manager.batched_extend_calls) == 2
    assert len(manager.released_handles) == 1


def test_score_prefill_extend_returns_before_background_release_completes():
    async def run_test():
        manager = _DummyScoreCacheManager()
        manager.release_started = asyncio.Event()
        manager.release_continue = asyncio.Event()

        scores = await manager.score_prefill_extend(
            query_tokens=[1, 2],
            item_tokens_list=[[3], [4]],
            label_token_ids=[5, 6],
        )

        assert scores == [[0.2, 0.8], [0.3, 0.7]]
        assert len(manager.asyncio_tasks) == 1
        await asyncio.wait_for(manager.release_started.wait(), timeout=1.0)
        assert manager.released_handles == []

        manager.release_continue.set()
        await _drain_asyncio_tasks(manager)
        assert manager.released_handles == [manager.generated_requests[0].rid]
        assert manager.asyncio_tasks == set()

    asyncio.run(run_test())


def test_release_cache_background_logs_failed_release_and_cleans_task(caplog):
    async def run_test():
        manager = _DummyScoreCacheManager()
        manager.release_result = False

        with caplog.at_level(logging.WARNING):
            manager._release_cache_background("cache-fail")
            assert len(manager.asyncio_tasks) == 1
            await _drain_asyncio_tasks(manager)

        assert manager.released_handles == ["cache-fail"]
        assert manager.asyncio_tasks == set()
        assert "was not cleanly released" in caplog.text

    asyncio.run(run_test())


def test_release_cache_background_logs_unexpected_task_exception(caplog):
    async def run_test():
        manager = _DummyScoreCacheManager()
        manager.release_exception = RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            manager._release_cache_background("cache-error")
            await _drain_asyncio_tasks(manager)

        assert manager.released_handles == []
        assert manager.asyncio_tasks == set()
        assert "Unexpected failure in background prefill+extend cache release task" in caplog.text

    asyncio.run(run_test())
