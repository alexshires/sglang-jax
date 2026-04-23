from types import SimpleNamespace

from sgl_jax.srt.managers.scheduler_scoring_state_mixin import SchedulerScoringStateMixin


class _DummyScheduler(SchedulerScoringStateMixin):
    def _detect_score_scheduler_topology_name(self) -> str:
        return "test-topology"


def test_init_scoring_state_sets_cache_and_ingress_counters():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        SimpleNamespace(
            multi_item_prefill_extend_cache_timeout=12.5,
            score_scheduler_global_microbatch_window_ms=2.0,
            score_scheduler_global_microbatch_poll_interval_ms=0.25,
            score_scheduler_short_prompt_tokens_threshold=1024,
            score_scheduler_short_lane_max_inflight=8,
            score_scheduler_long_lane_max_inflight=4,
            score_scheduler_enable_lane_isolation=True,
            score_scheduler_lane_isolation_short_burst=3,
            score_scheduler_lane_isolation_long_burst=2,
            score_scheduler_dynamic_items_per_step_enable=True,
            score_scheduler_dynamic_items_per_step_pressure_threshold=32,
            score_scheduler_dynamic_items_per_step_short_lane_bias=0.9,
            score_scheduler_dynamic_items_per_step_long_lane_bias=0.6,
            score_scheduler_dynamic_items_per_step_short_lane_min=12,
            score_scheduler_dynamic_items_per_step_long_lane_min=6,
            score_scheduler_cache_admission_bias_enable=True,
            score_scheduler_cache_admission_bias_require_hit=False,
        )
    )

    assert scheduler.scoring_cache_timeout == 12.5
    assert scheduler.ingress_recv_calls == 0
    assert scheduler.score_from_cache_v2_attempted == 0
    assert scheduler.score_scheduler_global_microbatch_window_s == 0.002
    assert scheduler.score_scheduler_enable_lane_isolation is True
    assert scheduler.score_scheduler_dynamic_items_per_step_enable is True
    assert scheduler.score_scheduler_cache_admission_bias_require_hit is False


def test_init_scoring_state_clamps_lane_controls_to_positive_values():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        SimpleNamespace(
            multi_item_prefill_extend_cache_timeout=0.0,
            score_scheduler_global_microbatch_window_ms=-1.0,
            score_scheduler_global_microbatch_poll_interval_ms=0.0,
            score_scheduler_short_prompt_tokens_threshold=0,
            score_scheduler_short_lane_max_inflight=-1,
            score_scheduler_long_lane_max_inflight=-1,
            score_scheduler_enable_lane_isolation=False,
            score_scheduler_lane_isolation_short_burst=0,
            score_scheduler_lane_isolation_long_burst=0,
            score_scheduler_dynamic_items_per_step_enable=False,
            score_scheduler_dynamic_items_per_step_pressure_threshold=0,
            score_scheduler_dynamic_items_per_step_short_lane_bias=0.0,
            score_scheduler_dynamic_items_per_step_long_lane_bias=0.0,
            score_scheduler_dynamic_items_per_step_short_lane_min=0,
            score_scheduler_dynamic_items_per_step_long_lane_min=0,
            score_scheduler_cache_admission_bias_enable=False,
            score_scheduler_cache_admission_bias_require_hit=True,
        )
    )

    assert scheduler.score_scheduler_global_microbatch_window_s == 0.0
    assert scheduler.score_scheduler_global_microbatch_poll_s == 0.0001
    assert scheduler.score_scheduler_short_prompt_tokens_threshold == 1
    assert scheduler.score_scheduler_lane_isolation_short_burst == 1
    assert scheduler.score_scheduler_dynamic_items_per_step_short_lane_min == 1
