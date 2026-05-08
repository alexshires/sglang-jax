from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqInput, TokenizedGenerateReqInput
from sgl_jax.srt.managers.scheduler_scoring_state_mixin import SchedulerScoringStateMixin


class _DummyScheduler(SchedulerScoringStateMixin):
    def _detect_score_scheduler_topology_name(self) -> str:
        return "test-topology"

    def _scoring_cache_metrics_snapshot(self):
        return {"queries": self.scoring_cache_lookup_queries}


class _ForwardMode:
    def __init__(self, is_extend: bool):
        self._is_extend = is_extend

    def is_extend(self) -> bool:
        return self._is_extend


def _server_args(**overrides):
    values = {
        "multi_item_prefill_extend_cache_timeout": 12.5,
        "score_scheduler_global_microbatch_window_ms": 2.0,
        "score_scheduler_global_microbatch_poll_interval_ms": 0.25,
        "score_scheduler_short_prompt_tokens_threshold": 1024,
        "score_scheduler_short_lane_max_inflight": 8,
        "score_scheduler_long_lane_max_inflight": 4,
        "score_scheduler_enable_lane_isolation": True,
        "score_scheduler_lane_isolation_short_burst": 3,
        "score_scheduler_lane_isolation_long_burst": 2,
        "score_scheduler_dynamic_items_per_step_enable": True,
        "score_scheduler_dynamic_items_per_step_pressure_threshold": 32,
        "score_scheduler_dynamic_items_per_step_short_lane_bias": 0.9,
        "score_scheduler_dynamic_items_per_step_long_lane_bias": 0.6,
        "score_scheduler_dynamic_items_per_step_short_lane_min": 12,
        "score_scheduler_dynamic_items_per_step_long_lane_min": 6,
        "score_scheduler_cache_admission_bias_enable": True,
        "score_scheduler_cache_admission_bias_require_hit": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _req(
    rid: str,
    *,
    tokens: list[int] | None = None,
    cache_for_scoring: bool = False,
    extend_from_cache: str | None = None,
    return_logprob: bool = False,
    max_new_tokens: int = 1,
    extra_key: str | None = None,
):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=tokens or [],
        cache_for_scoring=cache_for_scoring,
        extend_from_cache=extend_from_cache,
        return_logprob=return_logprob,
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        extra_key=extra_key,
    )


def test_init_scoring_state_sets_cache_and_ingress_counters():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(_server_args())

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
        _server_args(
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


def test_init_scoring_state_caps_microbatch_window_to_safe_bound():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(_server_args(score_scheduler_global_microbatch_window_ms=5000.0))

    assert scheduler.score_scheduler_global_microbatch_window_s == 0.1


def test_add_scoring_internal_state_preserves_explicit_zero_values():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(_server_args())
    scheduler.score_scheduler_global_microbatch_poll_s = 0.0
    scheduler.score_scheduler_lane_isolation_short_burst = 0
    scheduler.score_scheduler_dynamic_items_per_step_short_lane_bias = 0.0

    ret = {}
    scheduler.add_scoring_internal_state(ret)

    assert ret["ingress_metrics"]["score_coalescing"]["poll_interval_s"] == 0.0
    admission = ret["score_scheduler_admission_metrics"]
    assert admission["lane_isolation_short_burst"] == 0
    assert admission["dynamic_items_per_step"]["short_lane_bias"] == 0.0


def test_admission_lane_and_caps_follow_score_prompt_length():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        _server_args(
            score_scheduler_short_prompt_tokens_threshold=3,
            score_scheduler_short_lane_max_inflight=2,
            score_scheduler_long_lane_max_inflight=5,
        )
    )

    assert (
        SchedulerScoringStateMixin._admission_lane(
            scheduler, _req("short", tokens=[1, 2, 3], cache_for_scoring=True)
        )
        == "short"
    )
    assert (
        SchedulerScoringStateMixin._admission_lane(
            scheduler, _req("long", tokens=[1, 2, 3, 4], cache_for_scoring=True)
        )
        == "long"
    )
    assert SchedulerScoringStateMixin._admission_lane(scheduler, _req("default")) == "default"
    assert SchedulerScoringStateMixin._lane_cap(scheduler, "short") == 2
    assert SchedulerScoringStateMixin._lane_cap(scheduler, "long") == 5
    assert SchedulerScoringStateMixin._lane_cap(scheduler, "default") == 0


def test_cache_admission_priority_orders_cache_reuse_paths():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        _server_args(score_scheduler_cache_admission_bias_enable=True)
    )
    scheduler.scoring_cache_nodes["hit-handle"] = object()
    scheduler.scoring_cache_prefix_handles_by_key[("tenant", (1, 2))] = {"existing-handle"}

    extend_hit = _req("extend-hit", extend_from_cache="hit-handle")
    cache_prefix_hit = _req(
        "cache-prefix-hit",
        tokens=[1, 2],
        cache_for_scoring=True,
        extra_key="tenant",
    )
    extend_miss = _req("extend-miss", extend_from_cache="missing-handle")
    normal = _req("normal")

    assert SchedulerScoringStateMixin._cache_admission_priority(scheduler, extend_hit) == 3
    assert SchedulerScoringStateMixin._cache_admission_priority(scheduler, cache_prefix_hit) == 2
    assert SchedulerScoringStateMixin._cache_admission_priority(scheduler, extend_miss) == 1
    assert SchedulerScoringStateMixin._cache_admission_priority(scheduler, normal) == 0

    scheduler.score_scheduler_cache_admission_bias_require_hit = True
    assert SchedulerScoringStateMixin._cache_admission_priority(scheduler, extend_miss) == 0


def test_iter_waiting_queue_uses_lane_weighted_order():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        _server_args(
            score_scheduler_short_prompt_tokens_threshold=2,
            score_scheduler_enable_lane_isolation=True,
            score_scheduler_lane_isolation_short_burst=2,
            score_scheduler_lane_isolation_long_burst=1,
            score_scheduler_cache_admission_bias_enable=False,
        )
    )
    reqs = [
        _req("long-1", tokens=[1, 2, 3], cache_for_scoring=True),
        _req("default"),
        _req("short-1", tokens=[1], cache_for_scoring=True),
        _req("short-2", tokens=[1, 2], cache_for_scoring=True),
        _req("long-2", tokens=[1, 2, 3, 4], cache_for_scoring=True),
    ]

    ordered = SchedulerScoringStateMixin._iter_waiting_queue(scheduler, reqs)

    assert [req.rid for req in ordered] == [
        "short-1",
        "short-2",
        "default",
        "long-1",
        "long-2",
    ]
    assert scheduler.score_scheduler_lane_isolation_rounds == 1
    assert scheduler.score_scheduler_lane_isolation_selected["short"] == 2
    assert scheduler.score_scheduler_lane_isolation_selected["default"] == 1
    assert scheduler.score_scheduler_lane_isolation_selected["long"] == 2


def test_iter_waiting_queue_applies_cache_bias_within_lane():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        _server_args(
            score_scheduler_enable_lane_isolation=False,
            score_scheduler_cache_admission_bias_enable=True,
            score_scheduler_cache_admission_bias_require_hit=False,
        )
    )
    scheduler.scoring_cache_nodes["hit-handle"] = object()
    reqs = [
        _req("miss", extend_from_cache="missing-handle"),
        _req("hit", extend_from_cache="hit-handle"),
        _req("normal"),
    ]

    ordered = SchedulerScoringStateMixin._iter_waiting_queue(scheduler, reqs)

    assert [req.rid for req in ordered] == ["hit", "miss", "normal"]
    assert scheduler.score_scheduler_cache_admission_candidates["default"] == 2
    assert scheduler.score_scheduler_cache_admission_promoted["default"] == 1


def test_can_skip_sample_for_prefill_batch_requires_cache_only_extend():
    batch = SimpleNamespace(
        is_prefill_only=True,
        forward_mode=_ForwardMode(True),
        return_logprob=False,
        return_output_logprob_only=False,
        reqs=[_req("cache", cache_for_scoring=True)],
    )

    assert SchedulerScoringStateMixin._can_skip_sample_for_prefill_batch(batch) is True

    batch.return_logprob = True
    assert SchedulerScoringStateMixin._can_skip_sample_for_prefill_batch(batch) is False
    batch.return_logprob = False
    batch.reqs = [_req("non-cache")]
    assert SchedulerScoringStateMixin._can_skip_sample_for_prefill_batch(batch) is False


def test_normalize_scoring_cache_prefix_key_handles_empty_and_numpy_inputs():
    assert SchedulerScoringStateMixin._normalize_scoring_cache_prefix_key(None, None) is None
    assert SchedulerScoringStateMixin._normalize_scoring_cache_prefix_key([], None) is None

    assert SchedulerScoringStateMixin._normalize_scoring_cache_prefix_key(
        np.asarray([1, 2], dtype=np.int32), "tenant"
    ) == ("tenant", (1, 2))


def test_unpack_ingress_payload_only_expands_known_scheduler_request_batches():
    tokenized = TokenizedGenerateReqInput(rid="rid-1")
    score_req = ScoreFromCacheReqInput(rid="rid-2", cache_handle="handle")
    tuple_payload = ("rid", {"not": "a scheduler request"})

    assert SchedulerScoringStateMixin._unpack_ingress_payload([tokenized, score_req]) == [
        tokenized,
        score_req,
    ]
    assert SchedulerScoringStateMixin._unpack_ingress_payload(tuple_payload) == [tuple_payload]
