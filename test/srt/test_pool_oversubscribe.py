from types import SimpleNamespace

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqInput, ScoreFromCacheReqOutput
from sgl_jax.srt.managers.scheduler_scoring_execute_mixin import SchedulerScoringExecuteMixin


class _FakeScheduler(SchedulerScoringExecuteMixin):
    def __init__(self, *, allow_reqpool_oversubscribe: bool):
        self.enable_overlap = False
        self.server_args = SimpleNamespace(
            max_running_requests=24,
            multi_item_score_label_only_logprob=False,
            multi_item_score_from_cache_v2_items_per_step=64,
            multi_item_score_from_cache_v2_token_budget=0,
            score_v2_allow_reqpool_oversubscribe=allow_reqpool_oversubscribe,
        )
        self.req_to_token_pool = SimpleNamespace(available_size=lambda: 25)
        self.scoring_cache_nodes = {"cache-ok": ("node", "swa", [1, 2, 3], [0, 1, 2], "key", 0.0)}
        self.score_from_cache_v2_attempted = 0
        self.score_from_cache_v2_succeeded = 0
        self.score_from_cache_v2_fallback = 0
        self.score_from_cache_v2_fallback_reasons = {}
        self.score_from_cache_v2_queue_wait_s_total = 0.0
        self.score_from_cache_v2_device_compute_s_total = 0.0
        self.score_from_cache_v2_host_orchestration_s_total = 0.0
        self.score_from_cache_v2_queue_wait_s_max = 0.0
        self.score_from_cache_v2_device_compute_s_max = 0.0
        self.score_from_cache_v2_host_orchestration_s_max = 0.0
        self.chunk_sizes: list[int] = []

    def _score_from_cache_v2_validate_items(self, recv_req):
        return True, None, ""

    def _score_from_cache_v2_fallback_output(
        self,
        recv_req,
        reason,
        error_msg,
        dispatch_count=0,
        queue_wait_s=0.0,
        device_compute_s=0.0,
        host_orchestration_s=0.0,
    ):
        self.score_from_cache_v2_fallback += 1
        self.score_from_cache_v2_fallback_reasons[reason] = (
            self.score_from_cache_v2_fallback_reasons.get(reason, 0) + 1
        )
        return ScoreFromCacheReqOutput(
            rid=recv_req.rid,
            success=False,
            fallback_reason=reason,
            error_msg=error_msg,
            dispatch_count=dispatch_count,
            queue_wait_s=queue_wait_s,
            device_compute_s=device_compute_s,
            host_orchestration_s=host_orchestration_s,
        )

    def _evict_expired_scoring_cache_nodes(self):
        return 0

    def _record_scoring_cache_lookup(self, path, hit, lane_name="default"):
        return None

    def _unpack_scoring_cache_entry(self, entry):
        return entry

    @staticmethod
    def _score_scheduler_lane_from_prefix_len(req_owner, prefix_len: int) -> str:
        return "short"

    def _score_from_cache_v2_use_direct_label_only(self, *, label_only_logprob: bool) -> bool:
        return False

    def _resolve_score_from_cache_v2_items_per_step(
        self,
        requested_items_per_step,
        default_items_per_step,
        effective_capacity,
        total_items,
        lane_name,
    ):
        del default_items_per_step, lane_name
        return max(1, min(requested_items_per_step, effective_capacity, total_items))

    def _score_from_cache_v2_topology_dispatch_policy(
        self,
        lane_name,
        prefix_len,
        requested_items_per_step,
        effective_items_per_step,
        effective_capacity,
        total_items,
        requested_token_budget,
        max_total_tokens,
    ):
        del (
            lane_name,
            prefix_len,
            requested_items_per_step,
            effective_capacity,
            total_items,
            requested_token_budget,
            max_total_tokens,
        )
        return effective_items_per_step, 0, 1, "test"

    @staticmethod
    def _estimate_score_from_cache_v2_words(prefix_len: int, items: list[list[int]]) -> int:
        return prefix_len * max(1, len(items))

    @staticmethod
    def _build_score_from_cache_v2_chunk_plan(
        items_2d,
        items_per_step,
        prefix_len,
        token_budget,
    ):
        del prefix_len, token_budget
        return [
            (list(range(start, min(start + items_per_step, len(items_2d)))), items_2d[start:start + items_per_step])
            for start in range(0, len(items_2d), items_per_step)
        ]

    def _touch_scoring_cache_entry(self, rid: str):
        return None

    def _run_score_from_cache_v2_chunk(
        self,
        *,
        cache_handle,
        chunk_items,
        label_token_ids,
        apply_softmax,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids,
        cached_extra_key,
    ):
        del (
            cache_handle,
            apply_softmax,
            cached_last_node,
            cached_prefix_indices,
            prefix_ids,
            cached_extra_key,
        )
        self.chunk_sizes.append(len(chunk_items))
        return [[0.1] * len(label_token_ids) for _ in chunk_items], 0.0, 0.0

    def _record_score_from_cache_v2_timing(
        self,
        *,
        queue_wait_s,
        device_compute_s,
        host_orchestration_s,
    ):
        self.score_from_cache_v2_queue_wait_s_total += queue_wait_s
        self.score_from_cache_v2_device_compute_s_total += device_compute_s
        self.score_from_cache_v2_host_orchestration_s_total += host_orchestration_s
        self.score_from_cache_v2_queue_wait_s_max = max(
            self.score_from_cache_v2_queue_wait_s_max,
            queue_wait_s,
        )
        self.score_from_cache_v2_device_compute_s_max = max(
            self.score_from_cache_v2_device_compute_s_max,
            device_compute_s,
        )
        self.score_from_cache_v2_host_orchestration_s_max = max(
            self.score_from_cache_v2_host_orchestration_s_max,
            host_orchestration_s,
        )


def _make_req(item_count: int) -> ScoreFromCacheReqInput:
    return ScoreFromCacheReqInput(
        rid="test-rid",
        cache_handle="cache-ok",
        items_2d=[[i, i + 1] for i in range(item_count)],
        label_token_ids=[11, 17],
        items_per_step=64,
        apply_softmax=False,
    )


def test_score_from_cache_v2_clamps_to_max_running_requests_by_default():
    scheduler = _FakeScheduler(allow_reqpool_oversubscribe=False)

    out = scheduler.score_from_cache_v2(_make_req(item_count=50))

    assert out.success is True
    assert out.effective_items_per_step == 24
    assert out.dispatch_count == 3
    assert scheduler.chunk_sizes == [24, 24, 2]


def test_score_from_cache_v2_can_oversubscribe_to_live_req_pool_size():
    scheduler = _FakeScheduler(allow_reqpool_oversubscribe=True)

    out = scheduler.score_from_cache_v2(_make_req(item_count=50))

    assert out.success is True
    assert out.effective_items_per_step == 25
    assert out.dispatch_count == 2
    assert scheduler.chunk_sizes == [25, 25]
