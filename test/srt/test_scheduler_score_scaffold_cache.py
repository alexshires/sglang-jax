from types import SimpleNamespace

from jax import numpy as jnp

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqInput
from sgl_jax.srt.managers.scheduler_scoring_cache_mixin import SchedulerScoringCacheMixin
from sgl_jax.srt.managers.scheduler_scoring_dispatch_mixin import (
    SchedulerScoringDispatchMixin,
)
from sgl_jax.srt.managers.scheduler_scoring_execute_mixin import (
    SchedulerScoringExecuteMixin,
)
from sgl_jax.srt.managers.scheduler_scoring_state_mixin import SchedulerScoringStateMixin


class _DummyScheduler(
    SchedulerScoringCacheMixin,
    SchedulerScoringDispatchMixin,
    SchedulerScoringExecuteMixin,
    SchedulerScoringStateMixin,
):
    pass


class _DirectHappyPathScheduler(_DummyScheduler):
    def _score_from_cache_v2_use_direct_label_only(self, *, label_only_logprob: bool) -> bool:
        return bool(label_only_logprob)

    def _resolve_score_from_cache_v2_items_per_step(
        self,
        *,
        requested_items_per_step,
        default_items_per_step,
        effective_capacity,
        total_items,
        lane_name,
    ):
        return min(max(1, requested_items_per_step), max(1, effective_capacity))

    def _score_from_cache_v2_topology_dispatch_policy(
        self,
        *,
        lane_name,
        prefix_len,
        requested_items_per_step,
        effective_items_per_step,
        effective_capacity,
        total_items,
        requested_token_budget,
        max_total_tokens,
    ):
        return effective_items_per_step, requested_token_budget, 1, "unit-test"

    def _build_score_from_cache_v2_chunk_plan(
        self,
        items_2d,
        items_per_step,
        *,
        prefix_len=0,
        token_budget=0,
    ):
        return [
            ([2, 0], [items_2d[2], items_2d[0]]),
            ([1], [items_2d[1]]),
        ]

    def _run_score_from_cache_v2_direct_chunk_label_only(
        self,
        *,
        cache_handle,
        chunk_items,
        label_token_ids,
        label_token_ids_arr,
        apply_softmax,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids,
        cached_extra_key,
    ):
        self.chunk_calls.append([list(item) for item in chunk_items])
        scores = [[float(item[0]), float(item[0]) + 0.5] for item in chunk_items]
        return jnp.asarray(scores, dtype=jnp.float32), 0.1, 0.02


def test_unpack_scoring_cache_entry_accepts_current_format():
    scheduler = _DummyScheduler()

    current = scheduler._unpack_scoring_cache_entry(("node", "swa", [1], [2], "k", 3.5))
    assert current == ("node", "swa", [1], [2], "k", 3.5)


def test_scoring_cache_metrics_snapshot_tracks_hits_and_misses():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        SimpleNamespace(multi_item_prefill_extend_cache_timeout=1.0)
    )
    scheduler.scoring_cache_nodes = {"rid-1": ("node", "swa", [1], [2], "k", 0.0)}
    scheduler.scoring_cache_prefix_handles_by_key = {("k", (1,)): {"rid-1"}}
    scheduler.scoring_cache_handle_to_prefix_key = {"rid-1": ("k", (1,))}
    scheduler.scoring_cache_handles_created = 1

    scheduler._record_scoring_cache_lookup(path="extend", hit=True, lane_name="short")
    scheduler._record_scoring_cache_lookup(path="extend", hit=False, lane_name="long")

    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["lookup_queries"] == 2
    assert metrics["lookup_hits"] == 1
    assert metrics["lookup_misses"] == 1
    assert metrics["lookup_by_path"]["extend"]["hits"] == 1


def test_score_from_cache_v2_scaffold_reports_validation_failure():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        SimpleNamespace(multi_item_prefill_extend_cache_timeout=1.0)
    )
    scheduler.enable_overlap = False
    scheduler.server_args = type("ServerArgs", (), {})()

    result = scheduler.score_from_cache_v2(ScoreFromCacheReqInput(rid="rid-1"))

    assert result.success is False
    assert result.fallback_reason == "missing_cache_handle"
    assert scheduler.score_from_cache_v2_attempted == 1
    assert scheduler.score_from_cache_v2_fallback == 1


def test_score_from_cache_v2_success_reassembles_direct_chunks_in_original_order():
    scheduler = _DirectHappyPathScheduler()
    scheduler.init_scoring_state(
        SimpleNamespace(multi_item_prefill_extend_cache_timeout=0.0)
    )
    scheduler.enable_overlap = False
    scheduler.server_args = SimpleNamespace(
        device="cpu",
        max_running_requests=4,
        multi_item_score_label_only_logprob=True,
        multi_item_score_from_cache_v2_items_per_step=2,
        multi_item_score_from_cache_v2_token_budget=0,
    )
    scheduler.model_config = SimpleNamespace(vocab_size=1000)
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 4)
    scheduler.scoring_cache_nodes = {
        "cache-1": ("node", None, [101, 102, 103], [11, 12, 13], "extra", 0.0)
    }
    scheduler.chunk_calls = []

    result = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            rid="rid-2",
            cache_handle="cache-1",
            items_2d=[[10], [20, 21], [30]],
            label_token_ids=[1, 2],
            apply_softmax=True,
            items_per_step=2,
        )
    )

    assert result.success is True
    assert result.scores == [[10.0, 10.5], [20.0, 20.5], [30.0, 30.5]]
    assert result.dispatch_count == 2
    assert result.lifecycle_requests_sent == 0
    assert result.lifecycle_results_received == 0
    assert result.effective_items_per_step == 3
    assert result.topology_name == "unit-test"
    assert scheduler.chunk_calls == [[[30], [10]], [[20, 21]]]
    assert scheduler.score_from_cache_v2_attempted == 1
    assert scheduler.score_from_cache_v2_succeeded == 1
    assert scheduler.score_from_cache_v2_fallback == 0
