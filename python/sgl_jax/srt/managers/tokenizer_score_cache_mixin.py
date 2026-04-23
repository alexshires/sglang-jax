"""Tokenizer score cache and fastpath helpers."""

from __future__ import annotations

import asyncio
import logging
import uuid

from sgl_jax.srt.managers.io_struct import (
    ReleaseScoringCacheReqInput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
)
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class TokenizerScoreCacheMixin:
    async def _release_cache(self, cache_handle: str):
        """Release the cached query."""
        self.auto_create_handle_loop()
        logger.debug("Prefill+extend: releasing cache handle=%s", cache_handle)
        timeout_s = float(
            getattr(self.server_args, "multi_item_prefill_extend_cache_timeout", 60.0)
        )
        scheduler_fan_out = self._scheduler_sender_fan_out()
        try:
            req = ReleaseScoringCacheReqInput(rid=cache_handle)
            if self._can_use_local_score_rpc():
                outputs = await self._submit_local_score_rpc(
                    req,
                    timeout=timeout_s if timeout_s > 0 else None,
                )
            else:
                outputs = await self.release_scoring_cache_communicator(
                    req,
                    timeout=timeout_s if timeout_s > 0 else None,
                    broadcast=scheduler_fan_out > 1,
                )
        except TimeoutError:
            logger.error(
                "Timed out releasing prefill+extend cache handle=%s (timeout=%.2fs).",
                cache_handle,
                timeout_s,
            )
            return False
        except Exception:
            logger.exception(
                "Unexpected failure while releasing prefill+extend cache handle=%s.",
                cache_handle,
            )
            return False

        if not outputs:
            logger.warning("Release scoring cache returned no output for handle=%s", cache_handle)
            return False

        for out in outputs:
            if not out.success:
                logger.error(
                    "Failed to release scoring cache handle=%s: %s",
                    cache_handle,
                    out.error_msg,
                )
                return False
            logger.debug(
                "Prefill+extend: released cache handle=%s released_items=%d",
                cache_handle,
                out.released_items,
            )
        return True

    def _record_score_fastpath_fallback(self, reason: str):
        self.score_fastpath_fallback += 1
        self.score_fastpath_fallback_reasons[reason] = (
            self.score_fastpath_fallback_reasons.get(reason, 0) + 1
        )

    def _resolve_score_from_cache_v2_items_per_step(
        self,
        query_tokens: list[int],
        items: list[list[int]],
    ) -> tuple[int, int, int]:
        default_items_per_step = int(
            getattr(
                self.server_args,
                "multi_item_score_from_cache_v2_items_per_step",
                ServerArgs.multi_item_score_from_cache_v2_items_per_step,
            )
        )
        if default_items_per_step <= 0:
            default_items_per_step = 1

        adaptive_enabled = bool(
            getattr(
                self.server_args,
                "multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget",
                False,
            )
        )
        token_budget = int(
            getattr(
                self.server_args,
                "multi_item_score_from_cache_v2_token_budget",
                ServerArgs.multi_item_score_from_cache_v2_token_budget,
            )
        )
        if not adaptive_enabled or token_budget <= 0:
            return default_items_per_step, 0, max(0, token_budget)

        max_total_tokens = max((len(query_tokens) + len(item) for item in items), default=0)
        if max_total_tokens <= 0:
            return default_items_per_step, 0, token_budget

        min_items_per_step = int(
            getattr(
                self.server_args,
                "multi_item_score_from_cache_v2_min_items_per_step",
                ServerArgs.multi_item_score_from_cache_v2_min_items_per_step,
            )
        )
        if min_items_per_step <= 0:
            min_items_per_step = 1

        budget_items_per_step = max(1, token_budget // max_total_tokens)
        resolved_items_per_step = min(
            default_items_per_step,
            max(min_items_per_step, budget_items_per_step),
        )
        return resolved_items_per_step, max_total_tokens, token_budget

    def _partition_score_from_cache_v2_items(
        self,
        items: list[list[int]],
        scheduler_count: int,
    ) -> list[list[tuple[int, list[int]]]]:
        partitions = [[] for _ in range(max(1, scheduler_count))]
        lane_token_loads = [0] * len(partitions)
        for item_idx, item_tokens in enumerate(items):
            lane_idx = min(
                range(len(partitions)),
                key=lambda idx: (lane_token_loads[idx], len(partitions[idx]), idx),
            )
            partitions[lane_idx].append((item_idx, item_tokens))
            lane_token_loads[lane_idx] += len(item_tokens)
        return partitions

    async def _score_from_cache_fastpath_v2(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
        items_per_step: int | None = None,
        token_budget: int = 0,
        max_total_tokens: int = 0,
    ) -> ScoreFromCacheReqOutput:
        self.auto_create_handle_loop()
        req_rid = f"scorev2-{uuid.uuid4().hex}"
        timeout_s = float(
            getattr(self.server_args, "multi_item_prefill_extend_cache_timeout", 60.0)
        )
        if items_per_step is None or items_per_step <= 0:
            items_per_step = int(
                getattr(
                    self.server_args,
                    "multi_item_score_from_cache_v2_items_per_step",
                    ServerArgs.multi_item_score_from_cache_v2_items_per_step,
                )
            )
        if items_per_step <= 0:
            items_per_step = 1
        scheduler_count = self._scheduler_sender_fan_out()
        request_token_budget = max(0, int(token_budget or 0))
        request_max_total_tokens = max(0, int(max_total_tokens or 0))

        if scheduler_count <= 1:
            req = ScoreFromCacheReqInput(
                rid=req_rid,
                cache_handle=cache_handle,
                items_2d=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                items_per_step=items_per_step,
                token_budget=request_token_budget,
                max_total_tokens=request_max_total_tokens,
            )
            if self._can_use_local_score_rpc(total_items=len(items)):
                outputs = await self._submit_local_score_rpc(
                    req,
                    timeout=timeout_s if timeout_s > 0 else None,
                )
            else:
                outputs = await self.score_from_cache_v2_communicator(
                    req,
                    timeout=timeout_s if timeout_s > 0 else None,
                )
            if not outputs:
                return ScoreFromCacheReqOutput(
                    success=False,
                    scores=[],
                    fallback_reason="no_scheduler_response",
                    error_msg="No score-from-cache v2 response from scheduler.",
                )
            return outputs[0]

        item_partitions = self._partition_score_from_cache_v2_items(items, scheduler_count)

        async def _dispatch_partition(
            lane_idx: int,
            partition: list[tuple[int, list[int]]],
        ) -> tuple[list[int], ScoreFromCacheReqOutput]:
            partition_indices = [item_idx for item_idx, _ in partition]
            partition_items = [item_tokens for _, item_tokens in partition]
            if not partition_items:
                return (
                    [],
                    ScoreFromCacheReqOutput(
                        rid=f"{req_rid}-lane{lane_idx}",
                        success=True,
                        scores=[],
                        effective_items_per_step=0,
                        dispatch_token_budget=0,
                        replica_lane_count=0,
                    ),
                )
            partition_budget = (
                request_token_budget // scheduler_count if request_token_budget > 0 else 0
            )
            outputs = await self.score_from_cache_v2_communicator(
                ScoreFromCacheReqInput(
                    rid=f"{req_rid}-lane{lane_idx}",
                    cache_handle=cache_handle,
                    items_2d=partition_items,
                    label_token_ids=label_token_ids,
                    apply_softmax=apply_softmax,
                    items_per_step=items_per_step,
                    token_budget=partition_budget,
                    max_total_tokens=request_max_total_tokens,
                ),
                timeout=timeout_s if timeout_s > 0 else None,
                scheduler_idx=lane_idx,
            )
            if not outputs:
                return (
                    partition_indices,
                    ScoreFromCacheReqOutput(
                        rid=f"{req_rid}-lane{lane_idx}",
                        success=False,
                        scores=[],
                        fallback_reason="no_scheduler_response",
                        error_msg="No score-from-cache v2 response from scheduler lane.",
                    ),
                )
            return partition_indices, outputs[0]

        lane_results = await asyncio.gather(
            *[
                _dispatch_partition(lane_idx, partition)
                for lane_idx, partition in enumerate(item_partitions)
                if partition
            ]
        )
        merged_scores: list[list[float] | None] = [None] * len(items)
        aggregate = ScoreFromCacheReqOutput(
            rid=req_rid,
            success=True,
            scores=[],
            dispatch_count=0,
            lifecycle_requests_sent=0,
            lifecycle_results_received=0,
            queue_wait_s=0.0,
            device_compute_s=0.0,
            host_orchestration_s=0.0,
            effective_items_per_step=0,
            dispatch_token_budget=0,
            replica_lane_count=len(lane_results),
            topology_name="",
        )
        topology_names: list[str] = []
        for partition_indices, lane_output in lane_results:
            if not lane_output.success:
                return lane_output
            if len(lane_output.scores) != len(partition_indices):
                return ScoreFromCacheReqOutput(
                    rid=req_rid,
                    success=False,
                    scores=[],
                    fallback_reason="invalid_response_count",
                    error_msg=(
                        "Score-from-cache v2 replica lane returned wrong score count: "
                        f"{len(lane_output.scores)} != {len(partition_indices)}."
                    ),
                )
            for item_idx, item_scores in zip(partition_indices, lane_output.scores):
                merged_scores[item_idx] = item_scores
            aggregate.dispatch_count += int(lane_output.dispatch_count)
            aggregate.lifecycle_requests_sent += int(lane_output.lifecycle_requests_sent)
            aggregate.lifecycle_results_received += int(lane_output.lifecycle_results_received)
            aggregate.queue_wait_s = max(
                aggregate.queue_wait_s,
                float(lane_output.queue_wait_s),
            )
            aggregate.device_compute_s = max(
                aggregate.device_compute_s,
                float(lane_output.device_compute_s),
            )
            aggregate.host_orchestration_s = max(
                aggregate.host_orchestration_s,
                float(lane_output.host_orchestration_s),
            )
            aggregate.effective_items_per_step += int(lane_output.effective_items_per_step or 0)
            aggregate.dispatch_token_budget += int(lane_output.dispatch_token_budget or 0)
            if lane_output.topology_name:
                topology_names.append(lane_output.topology_name)

        if any(score is None for score in merged_scores):
            return ScoreFromCacheReqOutput(
                rid=req_rid,
                success=False,
                scores=[],
                fallback_reason="missing_partition_scores",
                error_msg="Replica-lane score aggregation left gaps in the merged result.",
            )

        aggregate.scores = [score for score in merged_scores if score is not None]
        unique_topologies = sorted(set(topology_names))
        if unique_topologies:
            topology_name = unique_topologies[0]
            if len(unique_topologies) > 1:
                topology_name = ",".join(unique_topologies)
            aggregate.topology_name = f"{topology_name} replicated x{len(lane_results)}"
        return aggregate

    def _maybe_log_score_path_metrics(self, metrics: dict):
        if not getattr(self.server_args, "multi_item_score_fastpath_log_metrics", False):
            return
        logger.info(
            "ScorePathMetrics path=%s items=%d dispatches=%d lifecycle_sent=%d lifecycle_recv=%d "
            "queue_wait_s=%.6f device_compute_s=%.6f host_orchestration_s=%.6f "
            "fastpath_attempted=%s fastpath_succeeded=%s fastpath_fallback_reason=%s "
            "fastpath_items_per_step=%d fastpath_token_budget=%d fastpath_max_total_tokens=%d "
            "fastpath_replica_lanes=%d fastpath_topology=%s",
            metrics.get("path", "unknown"),
            int(metrics.get("items", 0)),
            int(metrics.get("dispatch_count", 0)),
            int(metrics.get("lifecycle_requests_sent", 0)),
            int(metrics.get("lifecycle_results_received", 0)),
            float(metrics.get("queue_wait_s", 0.0)),
            float(metrics.get("device_compute_s", 0.0)),
            float(metrics.get("host_orchestration_s", 0.0)),
            bool(metrics.get("fastpath_attempted", False)),
            bool(metrics.get("fastpath_succeeded", False)),
            metrics.get("fastpath_fallback_reason"),
            int(metrics.get("fastpath_items_per_step", 0)),
            int(metrics.get("fastpath_token_budget", 0)),
            int(metrics.get("fastpath_max_total_tokens", 0)),
            int(metrics.get("fastpath_replica_lanes", 1)),
            metrics.get("fastpath_topology", ""),
        )

    async def score_prefill_extend(
        self,
        query_tokens: list[int],
        item_tokens_list: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        """
        Score items using prefill+extend strategy.
        """
        if not item_tokens_list:
            return []

        logger.debug(
            "Prefill+extend: begin scoring query_tokens=%d items=%d",
            len(query_tokens),
            len(item_tokens_list),
        )
        metrics = {
            "path": "prefill_extend_baseline",
            "items": len(item_tokens_list),
            "dispatch_count": 0,
            "lifecycle_requests_sent": 0,
            "lifecycle_results_received": 0,
            "queue_wait_s": 0.0,
            "device_compute_s": 0.0,
            "host_orchestration_s": 0.0,
            "fastpath_attempted": False,
            "fastpath_succeeded": False,
            "fastpath_fallback_reason": None,
            "fastpath_items_per_step": 0,
            "fastpath_token_budget": 0,
            "fastpath_max_total_tokens": 0,
            "fastpath_replica_lanes": 1,
            "fastpath_topology": "",
        }
        # Step 1: Prefill query and get cache handle
        cache_handle = await self._prefill_and_cache(query_tokens)
        metrics["lifecycle_requests_sent"] += 1
        metrics["lifecycle_results_received"] += 1

        try:
            if getattr(self.server_args, "multi_item_enable_score_from_cache_v2", False):
                metrics["fastpath_attempted"] = True
                self.score_fastpath_attempted += 1
                try:
                    (
                        resolved_items_per_step,
                        resolved_max_total_tokens,
                        resolved_token_budget,
                    ) = self._resolve_score_from_cache_v2_items_per_step(
                        query_tokens=query_tokens,
                        items=item_tokens_list,
                    )
                    metrics["fastpath_items_per_step"] = int(resolved_items_per_step)
                    metrics["fastpath_token_budget"] = int(resolved_token_budget)
                    metrics["fastpath_max_total_tokens"] = int(resolved_max_total_tokens)
                    fastpath_out = await self._score_from_cache_fastpath_v2(
                        cache_handle=cache_handle,
                        items=item_tokens_list,
                        label_token_ids=label_token_ids,
                        apply_softmax=apply_softmax,
                        items_per_step=resolved_items_per_step,
                        token_budget=resolved_token_budget,
                        max_total_tokens=resolved_max_total_tokens,
                    )
                    metrics["lifecycle_requests_sent"] += 1
                    metrics["lifecycle_results_received"] += 1
                except TimeoutError:
                    fastpath_out = ScoreFromCacheReqOutput(
                        success=False,
                        scores=[],
                        fallback_reason="timeout",
                        error_msg="Timed out waiting for score-from-cache v2 response.",
                    )
                except Exception:
                    logger.exception("Fastpath v2 request failed before scheduler response.")
                    fastpath_out = ScoreFromCacheReqOutput(
                        success=False,
                        scores=[],
                        fallback_reason="runtime_exception",
                        error_msg="Fastpath v2 communicator exception.",
                    )

                fallback_reason = None
                fallback_error_msg = fastpath_out.error_msg
                if fastpath_out.success:
                    if len(fastpath_out.scores) != len(item_tokens_list):
                        fallback_reason = "invalid_response_count"
                        fallback_error_msg = (
                            "Fastpath v2 returned wrong score count: "
                            f"{len(fastpath_out.scores)} != {len(item_tokens_list)}."
                        )
                    else:
                        self.score_fastpath_succeeded += 1
                        metrics["path"] = "score_from_cache_v2"
                        metrics["fastpath_succeeded"] = True
                        metrics["fastpath_items_per_step"] = int(
                            fastpath_out.effective_items_per_step or resolved_items_per_step
                        )
                        metrics["fastpath_token_budget"] = int(
                            fastpath_out.dispatch_token_budget or resolved_token_budget
                        )
                        metrics["fastpath_replica_lanes"] = int(
                            fastpath_out.replica_lane_count or 1
                        )
                        metrics["fastpath_topology"] = fastpath_out.topology_name or ""
                        metrics["dispatch_count"] += int(fastpath_out.dispatch_count)
                        metrics["queue_wait_s"] += float(fastpath_out.queue_wait_s)
                        metrics["device_compute_s"] += float(fastpath_out.device_compute_s)
                        metrics["host_orchestration_s"] += float(fastpath_out.host_orchestration_s)
                        metrics["lifecycle_requests_sent"] += int(
                            fastpath_out.lifecycle_requests_sent
                        )
                        metrics["lifecycle_results_received"] += int(
                            fastpath_out.lifecycle_results_received
                        )
                        self._maybe_log_score_path_metrics(metrics)
                        return fastpath_out.scores
                else:
                    fallback_reason = fastpath_out.fallback_reason or "runtime_exception"

                if fallback_reason is not None:
                    metrics["fastpath_fallback_reason"] = fallback_reason
                    self._record_score_fastpath_fallback(fallback_reason)
                    logger.warning(
                        "Fastpath v2 falling back to baseline: reason=%s error=%s",
                        fallback_reason,
                        fallback_error_msg,
                    )

            # Step 2: Process items in batches
            all_scores = []
            batch_size = int(getattr(self.server_args, "multi_item_extend_batch_size", 32))
            if batch_size <= 0:
                batch_size = len(item_tokens_list) or 1

            for i in range(0, len(item_tokens_list), batch_size):
                batch = item_tokens_list[i : i + batch_size]
                logger.debug(
                    "Prefill+extend: processing batch start=%d size=%d total=%d",
                    i,
                    len(batch),
                    len(item_tokens_list),
                )
                # Keep extend batch shape stable to avoid extra compile on trailing
                # partial batches (e.g., 10 items with batch size 4 -> 4,4,2).
                # We drop padded scores after the call.
                padded_batch = batch
                padded_count = 0
                if len(batch) < batch_size and len(batch) > 0:
                    padded_count = batch_size - len(batch)
                    padded_batch = batch + [batch[-1]] * padded_count
                batch_scores, batch_metrics = await self._batched_extend_score_with_metrics(
                    cache_handle=cache_handle,
                    items=padded_batch,
                    label_token_ids=label_token_ids,
                    apply_softmax=apply_softmax,
                )
                if padded_count > 0:
                    batch_scores = batch_scores[: len(batch)]
                all_scores.extend(batch_scores)
                metrics["dispatch_count"] += int(batch_metrics["dispatch_count"])
                metrics["queue_wait_s"] += float(batch_metrics["queue_wait_s"])
                metrics["device_compute_s"] += float(batch_metrics["device_compute_s"])
                metrics["host_orchestration_s"] += float(batch_metrics["host_orchestration_s"])
                # Only real items should contribute to lifecycle counters.
                real_items = len(batch)
                metrics["lifecycle_requests_sent"] += real_items
                metrics["lifecycle_results_received"] += real_items

            logger.debug("Prefill+extend: complete items=%d", len(item_tokens_list))
            self._maybe_log_score_path_metrics(metrics)
            return all_scores
        finally:
            # Step 3: Release cache
            released = await self._release_cache(cache_handle)
            if not released:
                logger.warning(
                    "Prefill+extend cache handle=%s was not cleanly released.", cache_handle
                )
