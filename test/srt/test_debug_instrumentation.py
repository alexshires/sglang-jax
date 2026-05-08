import pytest

from sgl_jax.srt.managers.scheduler_scoring_cache_mixin import SchedulerScoringCacheMixin
from sgl_jax.srt.managers.scheduler_scoring_dispatch_mixin import SchedulerScoringDispatchMixin
from sgl_jax.srt.managers.scheduler_scoring_state_mixin import SchedulerScoringStateMixin
from sgl_jax.srt.server_args import ServerArgs


class _FakeInstrumentationScheduler(
    SchedulerScoringDispatchMixin,
    SchedulerScoringCacheMixin,
    SchedulerScoringStateMixin,
):
    pass


def _make_scheduler() -> _FakeInstrumentationScheduler:
    scheduler = _FakeInstrumentationScheduler()
    server_args = ServerArgs(
        model_path="dummy-model",
        multi_item_prefill_extend_cache_timeout=60.0,
    )
    scheduler.init_scoring_state(server_args)
    return scheduler


def test_scoring_cache_metrics_snapshot_tracks_path_and_lane_breakdown():
    scheduler = _make_scheduler()
    scheduler.scoring_cache_nodes["cache-ok"] = ("node", "swa", [1, 2], [0, 1], "key", 0.0)

    scheduler._record_scoring_cache_lookup(
        path="score_from_cache_v2",
        hit=True,
        lane_name="short",
    )
    scheduler._record_scoring_cache_lookup(
        path="score_from_cache_v2",
        hit=False,
        lane_name="long",
    )

    metrics = scheduler._scoring_cache_metrics_snapshot()

    assert metrics["active_handles"] == 1
    assert metrics["lookup_queries"] == 2
    assert metrics["lookup_hits"] == 1
    assert metrics["lookup_misses"] == 1
    assert metrics["lookup_hit_rate"] == 0.5
    assert metrics["lookup_by_path"]["score_from_cache_v2"] == {
        "queries": 2,
        "hits": 1,
        "misses": 1,
    }
    assert metrics["lookup_by_lane"]["score_from_cache_v2"]["short"] == {
        "queries": 1,
        "hits": 1,
        "misses": 0,
    }
    assert metrics["lookup_by_lane"]["score_from_cache_v2"]["long"] == {
        "queries": 1,
        "hits": 0,
        "misses": 1,
    }


def test_score_from_cache_v2_timing_counters_accumulate_totals_and_maxima():
    scheduler = _make_scheduler()

    scheduler._record_score_from_cache_v2_timing(
        queue_wait_s=0.01,
        device_compute_s=0.02,
        host_orchestration_s=0.03,
    )
    scheduler._record_score_from_cache_v2_timing(
        queue_wait_s=0.04,
        device_compute_s=0.01,
        host_orchestration_s=0.02,
    )

    assert scheduler.score_from_cache_v2_queue_wait_s_total == pytest.approx(0.05)
    assert scheduler.score_from_cache_v2_device_compute_s_total == pytest.approx(0.03)
    assert scheduler.score_from_cache_v2_host_orchestration_s_total == pytest.approx(0.05)
    assert scheduler.score_from_cache_v2_queue_wait_s_max == pytest.approx(0.04)
    assert scheduler.score_from_cache_v2_device_compute_s_max == pytest.approx(0.02)
    assert scheduler.score_from_cache_v2_host_orchestration_s_max == pytest.approx(0.03)


def test_scoring_cache_handle_release_counters_track_reasons():
    scheduler = _make_scheduler()

    scheduler._record_scoring_cache_handle_created()
    scheduler._record_scoring_cache_handle_released("manual")
    scheduler._record_scoring_cache_handle_released("expired")
    scheduler._record_scoring_cache_handle_released("runtime_error")

    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["handles_created"] == 1
    assert metrics["handles_released_total"] == 3
    assert metrics["handles_released_manual"] == 1
    assert metrics["handles_released_expired"] == 1
    assert metrics["handles_released_other"] == 1


def test_unknown_scoring_cache_lane_normalizes_to_default_and_warns_once(caplog):
    scheduler = _make_scheduler()

    scheduler._record_scoring_cache_lookup(
        path="score_from_cache_v2",
        hit=True,
        lane_name="weird",
    )
    scheduler._record_scoring_cache_lookup(
        path="score_from_cache_v2",
        hit=False,
        lane_name="weird",
    )

    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["lookup_by_lane"]["score_from_cache_v2"]["default"] == {
        "queries": 2,
        "hits": 1,
        "misses": 1,
    }
    assert sum("Unknown scoring-cache lane" in record.message for record in caplog.records) == 1


def test_score_from_cache_v2_fallback_reasons_accumulate():
    scheduler = _make_scheduler()

    scheduler._record_score_from_cache_v2_fallback("timeout")
    scheduler._record_score_from_cache_v2_fallback("timeout")
    scheduler._record_score_from_cache_v2_fallback("runtime_exception")

    assert scheduler.score_from_cache_v2_fallback == 3
    assert scheduler.score_from_cache_v2_fallback_reasons == {
        "timeout": 2,
        "runtime_exception": 1,
    }
