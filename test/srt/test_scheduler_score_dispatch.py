from types import SimpleNamespace

from sgl_jax.srt.managers.scheduler_scoring_dispatch_mixin import (
    SchedulerScoringDispatchMixin,
)


class _DummyScheduler(SchedulerScoringDispatchMixin):
    pass


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

    assert items_per_step > 4
    assert token_budget > 0
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
