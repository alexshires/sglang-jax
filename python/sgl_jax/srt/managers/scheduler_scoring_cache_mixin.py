"""Scheduler scoring cache lifecycle helpers."""

import logging
import time

import numpy as np

from sgl_jax.srt.managers.io_struct import (
    ReleaseScoringCacheReqInput,
    ReleaseScoringCacheReqOutput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache

logger = logging.getLogger(__name__)


class SchedulerScoringCacheMixin:
    def _unpack_scoring_cache_entry(self, entry):
        if len(entry) == 6:
            return entry
        raise RuntimeError(f"Invalid scoring cache entry format (len={len(entry)}).")

    def _register_scoring_cache_handle(
        self,
        rid: str,
        input_ids: list[int] | tuple[int, ...] | np.ndarray | None,
        extra_key: str | None,
    ) -> tuple[str, tuple[int, ...]] | None:
        prefix_key = self._normalize_scoring_cache_prefix_key(input_ids, extra_key)
        if prefix_key is None:
            return None
        handles = self.scoring_cache_prefix_handles_by_key.setdefault(prefix_key, set())
        handles.add(rid)
        self.scoring_cache_handle_to_prefix_key[rid] = prefix_key
        return prefix_key

    def _unregister_scoring_cache_handle(self, rid: str) -> None:
        prefix_key = self.scoring_cache_handle_to_prefix_key.pop(rid, None)
        if prefix_key is None:
            return
        handles = self.scoring_cache_prefix_handles_by_key.get(prefix_key)
        if handles is None:
            return
        handles.discard(rid)
        if not handles:
            self.scoring_cache_prefix_handles_by_key.pop(prefix_key, None)

    def _record_scoring_cache_lookup(
        self,
        path: str,
        hit: bool,
        lane_name: str = "default",
    ) -> None:
        self.scoring_cache_lookup_queries += 1
        if hit:
            self.scoring_cache_lookup_hits += 1
        else:
            self.scoring_cache_lookup_misses += 1

        bucket = self.scoring_cache_lookup_by_path.setdefault(
            path,
            {"queries": 0, "hits": 0, "misses": 0},
        )
        bucket["queries"] += 1
        if hit:
            bucket["hits"] += 1
        else:
            bucket["misses"] += 1

        if lane_name in {"default", "short", "long"}:
            normalized_lane = lane_name
        else:
            normalized_lane = "default"
            warned_lanes = getattr(self, "_warned_scoring_cache_lanes", set())
            if lane_name not in warned_lanes:
                logger.warning(
                    "Unknown scoring-cache lane %r; recording metrics in default lane.",
                    lane_name,
                )
                warned_lanes.add(lane_name)
                self._warned_scoring_cache_lanes = warned_lanes
        by_lane = self.scoring_cache_lookup_by_lane.setdefault(path, {})
        lane_bucket = by_lane.setdefault(
            normalized_lane,
            {"queries": 0, "hits": 0, "misses": 0},
        )
        lane_bucket["queries"] += 1
        if hit:
            lane_bucket["hits"] += 1
        else:
            lane_bucket["misses"] += 1

    def _record_scoring_cache_handle_created(self) -> None:
        self.scoring_cache_handles_created += 1

    def _record_scoring_cache_handle_released(self, reason: str) -> None:
        self.scoring_cache_handles_released += 1
        if reason == "manual":
            self.scoring_cache_handles_released_manual += 1
        elif reason == "expired":
            self.scoring_cache_handles_released_expired += 1
        else:
            self.scoring_cache_handles_released_other += 1

    def _scoring_cache_metrics_snapshot(self) -> dict:
        query_total = self.scoring_cache_lookup_queries
        hit_total = self.scoring_cache_lookup_hits
        miss_total = self.scoring_cache_lookup_misses
        hit_rate = float(hit_total / query_total) if query_total > 0 else 0.0
        return {
            "active_handles": len(self.scoring_cache_nodes),
            "active_prefix_keys": len(self.scoring_cache_prefix_handles_by_key),
            "handles_created": self.scoring_cache_handles_created,
            "handles_released_total": self.scoring_cache_handles_released,
            "handles_released_manual": self.scoring_cache_handles_released_manual,
            "handles_released_expired": self.scoring_cache_handles_released_expired,
            "handles_released_other": self.scoring_cache_handles_released_other,
            "handles_missing_node": self.scoring_cache_handles_missing_node,
            "release_failures": self.scoring_cache_release_failures,
            "lookup_queries": query_total,
            "lookup_hits": hit_total,
            "lookup_misses": miss_total,
            "lookup_hit_rate": hit_rate,
            "lookup_by_path": {
                path: dict(stats) for path, stats in self.scoring_cache_lookup_by_path.items()
            },
            "lookup_by_lane": {
                path: {lane: dict(stats) for lane, stats in lane_stats.items()}
                for path, lane_stats in self.scoring_cache_lookup_by_lane.items()
            },
        }

    def _release_scoring_cache_entry(self, rid: str, entry, reason: str) -> None:
        self._unregister_scoring_cache_handle(rid)
        node, swa_uuid, *_ = self._unpack_scoring_cache_entry(entry)
        self._record_scoring_cache_handle_released(reason)
        if node is None:
            self.scoring_cache_handles_missing_node += 1
            logger.warning("Scoring cache entry rid=%s has no radix node (%s).", rid, reason)
            return
        try:
            if isinstance(self.tree_cache, SWARadixCache):
                self.tree_cache.dec_lock_ref(node, swa_uuid)
            else:
                self.tree_cache.dec_lock_ref(node)
        except Exception:
            self.scoring_cache_release_failures += 1
            logger.exception(
                "Failed to decrement scoring-cache lock ref for rid=%s (%s).",
                rid,
                reason,
            )

    def _touch_scoring_cache_entry(self, rid: str, now: float | None = None):
        entry = self.scoring_cache_nodes.get(rid)
        if entry is None:
            return
        node, swa_uuid, input_ids, prefix_indices, extra_key, _ = self._unpack_scoring_cache_entry(
            entry
        )
        self.scoring_cache_nodes[rid] = (
            node,
            swa_uuid,
            input_ids,
            prefix_indices,
            extra_key,
            time.monotonic() if now is None else now,
        )

    def _evict_expired_scoring_cache_nodes(self, now: float | None = None) -> int:
        timeout = self.scoring_cache_timeout
        if timeout <= 0:
            return 0

        now_ts = time.monotonic() if now is None else now
        # Throttle GC to avoid walking the dict too often.
        if now is None and now_ts - self._last_scoring_cache_gc < 0.5:
            return 0
        if now is None:
            self._last_scoring_cache_gc = now_ts

        expired_rids: list[str] = []
        for rid, entry in self.scoring_cache_nodes.items():
            *_, last_access_ts = self._unpack_scoring_cache_entry(entry)
            if now_ts - last_access_ts > timeout:
                expired_rids.append(rid)

        for rid in expired_rids:
            entry = self.scoring_cache_nodes.pop(rid, None)
            if entry is None:
                continue
            self._release_scoring_cache_entry(rid, entry, reason="expired")

        if expired_rids:
            logger.info("Evicted %d expired scoring cache handles.", len(expired_rids))
        return len(expired_rids)

    def _resolve_extend_from_cache(
        self, recv_req: TokenizedGenerateReqInput
    ) -> tuple[tuple | None, str | None]:
        if not recv_req.extend_from_cache:
            return None, None

        self._evict_expired_scoring_cache_nodes()
        entry = self.scoring_cache_nodes.get(recv_req.extend_from_cache)
        if entry is None:
            miss_lane = self._score_scheduler_lane_from_prefix_len(
                self,
                len(getattr(recv_req, "input_ids", []) or [])
            )
            self._record_scoring_cache_lookup(path="extend", hit=False, lane_name=miss_lane)
            err = (
                f"Missing scoring cache handle '{recv_req.extend_from_cache}'. "
                "The cached prefix may have expired or been released."
            )
            logger.warning("Prefill+extend scheduler: %s", err)
            return None, err

        cached_last_node, _, prefix_ids, prefix_indices, cached_extra_key, _ = (
            self._unpack_scoring_cache_entry(entry)
        )
        hit_lane = self._score_scheduler_lane_from_prefix_len(self, len(prefix_indices))
        self._record_scoring_cache_lookup(path="extend", hit=True, lane_name=hit_lane)

        item_ids = recv_req.input_ids or []
        merged_input_ids = prefix_ids + item_ids
        cached_prefix_len = len(prefix_indices)
        suffix_len = max(0, len(item_ids))
        merged_extra_key = cached_extra_key if recv_req.extra_key is None else recv_req.extra_key
        self._touch_scoring_cache_entry(recv_req.extend_from_cache)
        logger.debug(
            "Prefill+extend scheduler: extend request rid=%s handle=%s prefix_tokens=%d cached_prefix=%d item_tokens=%d merged_input_tokens=%d max_new_tokens=%s",
            recv_req.rid,
            recv_req.extend_from_cache,
            len(prefix_ids),
            cached_prefix_len,
            suffix_len,
            len(merged_input_ids),
            recv_req.sampling_params.max_new_tokens,
        )
        return (cached_last_node, prefix_indices, merged_input_ids, merged_extra_key), None

    def _release_scoring_cache_nodes(self, rid_prefix: str | None, abort_all: bool) -> int:
        released = 0
        self._evict_expired_scoring_cache_nodes()
        if not abort_all and not rid_prefix:
            return released

        rids_to_remove = []
        for rid in self.scoring_cache_nodes:
            if abort_all or (rid_prefix and rid.startswith(rid_prefix)):
                rids_to_remove.append(rid)

        for rid in rids_to_remove:
            entry = self.scoring_cache_nodes.pop(rid, None)
            if entry is None:
                continue
            self._release_scoring_cache_entry(rid, entry, reason="manual")
            released += 1
            logger.debug("Released cached node for rid=%s", rid)
        return released

    def release_scoring_cache(
        self, recv_req: ReleaseScoringCacheReqInput
    ) -> ReleaseScoringCacheReqOutput:
        released = self._release_scoring_cache_nodes(recv_req.rid, abort_all=False)
        return ReleaseScoringCacheReqOutput(
            rid=recv_req.rid,
            success=True,
            released_items=released,
        )
