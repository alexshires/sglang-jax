from types import SimpleNamespace

from sgl_jax.srt.managers.scheduler_scoring_dispatch_mixin import (
    SchedulerScoringDispatchMixin,
)


class _DummyScheduler(SchedulerScoringDispatchMixin):
    @staticmethod
    def _score_scheduler_queue_pressure(req_owner):
        return getattr(req_owner, "queue_pressure", 0)

    @staticmethod
    def _lane_counter(req_owner, attr_name: str):
        counter = getattr(req_owner, attr_name, None)
        if not isinstance(counter, dict):
            counter = {"default": 0, "short": 0, "long": 0}
            setattr(req_owner, attr_name, counter)
        return counter


def _score_req(**overrides):
    values = {
        "rid": "rid-1",
        "cache_handle": "handle",
        "items_2d": [[1, 2]],
        "label_token_ids": [3, 4],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_topology_dispatch_policy_boosts_long_lane_on_v6e_8():
    scheduler = _DummyScheduler()
    scheduler.server_args = SimpleNamespace()
    scheduler.mesh = SimpleNamespace(shape={"data": 4})
    scheduler.score_scheduler_topology_name = "TPU v6e-8"

    items_per_step, token_budget, replica_lanes, topology_name = (
        scheduler._score_from_cache_v2_topology_dispatch_policy(
            lane_name="long",
            prefix_len=8,
            requested_items_per_step=4,
            effective_items_per_step=4,
            effective_capacity=16,
            total_items=16,
            requested_token_budget=0,
            max_total_tokens=32,
        )
    )

    assert items_per_step == 16
    assert token_budget == 640
    assert replica_lanes == 4
    assert topology_name == "TPU v6e-8"


def test_direct_token_ids_auto_gate_uses_page_size_threshold():
    scheduler = _DummyScheduler()
    scheduler.page_size = 16
    scheduler.server_args = SimpleNamespace(
        multi_item_score_direct_token_ids_logprob_only=False,
        multi_item_score_direct_token_ids_logprob_only_auto=True,
        multi_item_score_direct_token_ids_logprob_only_auto_max_page_size=32,
        multi_item_score_direct_token_ids_logprob_only_auto_max_running_requests=0,
        max_running_requests=128,
    )

    assert scheduler._score_from_cache_v2_use_direct_token_ids_logprob_only() is True


def test_direct_label_only_toggle_requires_flag():
    scheduler = _DummyScheduler()
    scheduler.server_args = SimpleNamespace(multi_item_score_direct_label_only=True)
    assert scheduler._score_from_cache_v2_use_direct_label_only(label_only_logprob=True) is True
    assert scheduler._score_from_cache_v2_use_direct_label_only(label_only_logprob=False) is False


def test_direct_token_ids_auto_gate_respects_zero_thresholds():
    scheduler = _DummyScheduler()
    scheduler.page_size = 16
    scheduler.server_args = SimpleNamespace(
        multi_item_score_direct_token_ids_logprob_only=False,
        multi_item_score_direct_token_ids_logprob_only_auto=True,
        multi_item_score_direct_token_ids_logprob_only_auto_max_page_size=0,
        multi_item_score_direct_token_ids_logprob_only_auto_max_running_requests=0,
        max_running_requests=8,
    )

    assert scheduler._score_from_cache_v2_use_direct_token_ids_logprob_only() is False


def test_chunk_plan_count_path_sorts_by_length_and_preserves_ties():
    items = [[1], [2, 3, 4], [5, 6], [7, 8]]

    plan = SchedulerScoringDispatchMixin._build_score_from_cache_v2_chunk_plan(
        items,
        items_per_step=2,
    )

    assert [(indices, chunk_items) for indices, chunk_items in plan] == [
        ([1, 2], [[2, 3, 4], [5, 6]]),
        ([3, 0], [[7, 8], [1]]),
    ]


def test_chunk_plan_token_budget_packs_without_losing_original_indices():
    items = [[1, 2, 3, 4], [5, 6, 7], [8, 9], [10]]

    plan = SchedulerScoringDispatchMixin._build_score_from_cache_v2_chunk_plan(
        items,
        items_per_step=2,
        prefix_len=2,
        token_budget=7,
    )

    assert [(indices, chunk_items) for indices, chunk_items in plan] == [
        ([0], [[1, 2, 3, 4]]),
        ([1], [[5, 6, 7]]),
        ([2, 3], [[8, 9], [10]]),
    ]


def test_chunk_plan_token_budget_gives_oversized_items_single_chunks():
    items = [[1, 2], [3, 4]]

    plan = SchedulerScoringDispatchMixin._build_score_from_cache_v2_chunk_plan(
        items,
        items_per_step=2,
        prefix_len=2,
        token_budget=3,
    )

    assert [(indices, chunk_items) for indices, chunk_items in plan] == [
        ([0], [[1, 2]]),
        ([1], [[3, 4]]),
    ]


def test_resolve_items_per_step_applies_dynamic_pressure_and_lane_bias():
    scheduler = _DummyScheduler()
    scheduler.queue_pressure = 20
    scheduler.score_scheduler_dynamic_items_per_step_enable = True
    scheduler.score_scheduler_dynamic_items_per_step_pressure_threshold = 10
    scheduler.score_scheduler_dynamic_items_per_step_long_lane_bias = 0.5
    scheduler.score_scheduler_dynamic_items_per_step_long_lane_min = 2

    result = scheduler._resolve_score_from_cache_v2_items_per_step(
        requested_items_per_step=8,
        default_items_per_step=8,
        effective_capacity=8,
        total_items=16,
        lane_name="long",
    )

    assert result == 2
    assert scheduler.score_scheduler_dynamic_items_per_step_requests == 1
    assert scheduler.score_scheduler_dynamic_items_per_step_applied_by_lane["long"] == 1


def test_score_from_cache_validate_items_checks_item_token_range():
    scheduler = _DummyScheduler()
    scheduler.model_config = SimpleNamespace(vocab_size=10)

    assert scheduler._score_from_cache_v2_validate_items(_score_req())[0] is True

    ok, reason, message = scheduler._score_from_cache_v2_validate_items(
        _score_req(items_2d=[[1, 10]])
    )
    assert ok is False
    assert reason == "unsupported_shape"
    assert "items_2d[0] token ids" in message

    ok, _, message = scheduler._score_from_cache_v2_validate_items(
        _score_req(items_2d=[[1, -1]])
    )
    assert ok is False
    assert "items_2d[0] token ids" in message


def test_direct_hot_shape_applies_batch_token_and_cache_loc_padding():
    scheduler = _DummyScheduler()
    scheduler.page_size = 16
    scheduler.server_args = SimpleNamespace(
        multi_item_score_direct_hot_shape_bs=4,
        multi_item_score_direct_hot_shape_tokens=128,
        multi_item_score_direct_hot_shape_token_rounding=64,
        multi_item_score_direct_hot_shape_token_rounding_min_hot_tokens=64,
    )

    assert scheduler._score_from_cache_v2_resolve_direct_hot_shape(
        real_bs=2,
        real_input_tokens=33,
        real_cache_loc_tokens=40,
        max_seq_len=17,
    ) == (4, 64, 128)


def test_probs_from_logprobs_handles_infinities_and_softmax():
    raw = SchedulerScoringDispatchMixin._score_from_cache_v2_probs_from_logprobs(
        [0.0, float("-inf")],
        apply_softmax=False,
    )
    assert raw == [1.0, 0.0]

    softmax = SchedulerScoringDispatchMixin._score_from_cache_v2_probs_from_logprobs(
        [0.0, -1.0, float("-inf")],
        apply_softmax=True,
    )
    assert abs(sum(softmax) - 1.0) < 1e-9
    assert softmax[-1] == 0.0

    assert SchedulerScoringDispatchMixin._score_from_cache_v2_probs_from_logprobs(
        [float("-inf"), float("-inf")],
        apply_softmax=True,
    ) == [0.0, 0.0]
