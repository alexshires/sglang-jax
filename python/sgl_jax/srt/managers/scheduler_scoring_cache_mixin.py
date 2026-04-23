"""Scheduler scoring cache lifecycle helpers."""

from sgl_jax.srt.managers.scheduler_scoring_common import *


class SchedulerScoringCacheMixin:
    def _scoring_cache_metrics_snapshot(self) -> dict:
        return {"active_handles": len(getattr(self, "scoring_cache_nodes", {}))}

    def _evict_expired_scoring_cache_nodes(self, now: float | None = None) -> int:
        return 0

    def _resolve_extend_from_cache(
        self, recv_req: TokenizedGenerateReqInput
    ) -> tuple[tuple | None, str | None]:
        return None, None

    def _record_scoring_cache_handle_created(self) -> None:
        pass

    def _release_scoring_cache_nodes(self, rid_prefix: str | None, abort_all: bool) -> int:
        return 0

    def release_scoring_cache(
        self, recv_req: ReleaseScoringCacheReqInput
    ) -> ReleaseScoringCacheReqOutput:
        return ReleaseScoringCacheReqOutput(rid=recv_req.rid, success=True, released_items=0)
