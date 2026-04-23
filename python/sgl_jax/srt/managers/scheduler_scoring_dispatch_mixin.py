"""Scheduler score dispatch and chunk planning helpers."""

from sgl_jax.srt.managers.scheduler_scoring_common import *


class SchedulerScoringDispatchMixin:
    def _detect_score_scheduler_topology_name(self) -> str:
        return ""

    def _score_direct_warmup_spec(self) -> SimpleNamespace | None:
        return None

    def _run_score_direct_label_only_warmup(self) -> None:
        return None

    def _score_from_cache_v2_use_direct_label_only(self, *, label_only_logprob: bool) -> bool:
        return False

    def _resolve_score_from_cache_v2_items_per_step(
        self,
        *,
        requested_items_per_step: int,
        default_items_per_step: int,
        effective_capacity: int,
        total_items: int,
        lane_name: str,
    ) -> int:
        return max(1, min(requested_items_per_step, default_items_per_step, effective_capacity))
