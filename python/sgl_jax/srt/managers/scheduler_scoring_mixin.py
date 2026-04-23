"""Aggregate scoring mixin for the scheduler."""

from sgl_jax.srt.managers.scheduler_scoring_cache_mixin import SchedulerScoringCacheMixin
from sgl_jax.srt.managers.scheduler_scoring_direct_mixin import SchedulerScoringDirectMixin
from sgl_jax.srt.managers.scheduler_scoring_dispatch_mixin import SchedulerScoringDispatchMixin
from sgl_jax.srt.managers.scheduler_scoring_execute_mixin import SchedulerScoringExecuteMixin
from sgl_jax.srt.managers.scheduler_scoring_state_mixin import SchedulerScoringStateMixin


class SchedulerScoringMixin(
    SchedulerScoringExecuteMixin,
    SchedulerScoringDirectMixin,
    SchedulerScoringDispatchMixin,
    SchedulerScoringCacheMixin,
    SchedulerScoringStateMixin,
):
    pass
