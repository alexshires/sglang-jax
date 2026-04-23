from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqInput
from sgl_jax.srt.managers.scheduler_scoring_cache_mixin import SchedulerScoringCacheMixin
from sgl_jax.srt.managers.scheduler_scoring_execute_mixin import (
    SchedulerScoringExecuteMixin,
)


class _DummyScheduler(SchedulerScoringCacheMixin, SchedulerScoringExecuteMixin):
    pass


def test_unpack_scoring_cache_entry_accepts_current_and_legacy_formats():
    scheduler = _DummyScheduler()

    current = scheduler._unpack_scoring_cache_entry(("node", "swa", [1], [2], "k", 3.5))
    assert current == ("node", "swa", [1], [2], "k", 3.5)

    legacy = scheduler._unpack_scoring_cache_entry(("node", "swa", [1], [2], "k"))
    assert legacy == ("node", "swa", [1], [2], "k", 0.0)


def test_scoring_cache_metrics_snapshot_tracks_hits_and_misses():
    scheduler = _DummyScheduler()
    scheduler.scoring_cache_nodes = {"rid-1": ("node", "swa", [1], [2], "k", 0.0)}
    scheduler.scoring_cache_prefix_handles_by_key = {("k", (1,)): {"rid-1"}}
    scheduler.scoring_cache_handle_to_prefix_key = {"rid-1": ("k", (1,))}
    scheduler.scoring_cache_handles_created = 1
    scheduler.scoring_cache_handles_released = 0
    scheduler.scoring_cache_handles_released_manual = 0
    scheduler.scoring_cache_handles_released_expired = 0
    scheduler.scoring_cache_handles_released_other = 0
    scheduler.scoring_cache_handles_missing_node = 0
    scheduler.scoring_cache_lookup_queries = 0
    scheduler.scoring_cache_lookup_hits = 0
    scheduler.scoring_cache_lookup_misses = 0
    scheduler.scoring_cache_lookup_by_path = {}
    scheduler.scoring_cache_lookup_by_lane = {}

    scheduler._record_scoring_cache_lookup(path="extend", hit=True, lane_name="short")
    scheduler._record_scoring_cache_lookup(path="extend", hit=False, lane_name="long")

    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["lookup_queries"] == 2
    assert metrics["lookup_hits"] == 1
    assert metrics["lookup_misses"] == 1
    assert metrics["lookup_by_path"]["extend"]["hits"] == 1


def test_score_from_cache_v2_scaffold_returns_not_enabled():
    scheduler = _DummyScheduler()
    scheduler.score_from_cache_v2_attempted = 0
    scheduler.score_from_cache_v2_fallback = 0
    scheduler.score_from_cache_v2_fallback_reasons = {}

    result = scheduler.score_from_cache_v2(ScoreFromCacheReqInput(rid="rid-1"))

    assert result.success is False
    assert result.fallback_reason == "not_enabled"
    assert scheduler.score_from_cache_v2_attempted == 1
    assert scheduler.score_from_cache_v2_fallback == 1
