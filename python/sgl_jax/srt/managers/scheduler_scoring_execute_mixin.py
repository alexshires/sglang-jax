"""Scheduler score-from-cache request execution."""

from sgl_jax.srt.managers.scheduler_scoring_common import *


class SchedulerScoringExecuteMixin:
    def score_from_cache_v2(self, recv_req: ScoreFromCacheReqInput) -> ScoreFromCacheReqOutput:
        self.score_from_cache_v2_attempted += 1
        self.score_from_cache_v2_fallback += 1
        self.score_from_cache_v2_fallback_reasons["not_enabled"] = (
            self.score_from_cache_v2_fallback_reasons.get("not_enabled", 0) + 1
        )
        return ScoreFromCacheReqOutput(
            rid=recv_req.rid,
            success=False,
            scores=[],
            fallback_reason="not_enabled",
            error_msg="score-from-cache v2 is not enabled in this stack slice.",
        )
