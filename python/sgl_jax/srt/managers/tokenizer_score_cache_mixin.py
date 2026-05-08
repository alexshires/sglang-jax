"""Tokenizer score cache and fastpath helpers."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid

from sgl_jax.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseScoringCacheReqInput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
)
from sgl_jax.srt.managers.tokenizer_score_common import _stable_softmax
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.validation import ValidationError

logger = logging.getLogger(__name__)


class TokenizerScoreCacheMixin:
    def _normalize_score_query_tokens(self, query: str | list[int] | None) -> list[int]:
        if query is None:
            raise ValidationError("query is required", param="query", code="missing_query")
        if isinstance(query, str):
            if len(query) == 0:
                raise ValidationError("query cannot be empty", param="query", code="empty_query")
            if self.tokenizer is None:
                raise ValidationError(
                    "Tokenizer is required for text scoring.",
                    param="query",
                    code="tokenizer_required",
                )
            return self.tokenizer.encode(query, add_special_tokens=False)
        if not isinstance(query, list):
            raise ValidationError(
                f"query must be a string or list of integers, got {type(query).__name__}",
                param="query",
                code="invalid_query_type",
            )
        if len(query) == 0:
            raise ValidationError("query token list cannot be empty", param="query", code="empty_query")
        if any(not isinstance(token_id, int) for token_id in query):
            raise ValidationError(
                "query contains non-integer values in token input mode",
                param="query",
                code="invalid_token_id_type",
            )
        return query

    def _normalize_score_item_tokens(
        self,
        items: str | list[str] | list[list[int]] | None,
    ) -> list[list[int]]:
        if items is None:
            raise ValidationError("items is required", param="items", code="missing_items")
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            raise ValidationError(
                "items must be a list of strings or list of token ID lists",
                param="items",
                code="invalid_items_type",
            )
        if len(items) == 0:
            raise ValidationError("items cannot be empty", param="items", code="empty_items")
        if isinstance(items[0], str):
            if self.tokenizer is None:
                raise ValidationError(
                    "Tokenizer is required for text scoring.",
                    param="items",
                    code="tokenizer_required",
                )
            for idx, item in enumerate(items):
                if not isinstance(item, str):
                    raise ValidationError(
                        f"items[{idx}] must be a string when using text input mode",
                        param="items",
                        code="mixed_item_types",
                    )
            return [self.tokenizer.encode(item, add_special_tokens=False) for item in items]

        normalized_items: list[list[int]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, list):
                raise ValidationError(
                    f"items[{idx}] must be a list of integers when using token input mode",
                    param="items",
                    code="invalid_item_type",
                )
            if len(item) == 0:
                raise ValidationError(
                    f"items[{idx}] must contain at least one token",
                    param="items",
                    code="empty_item",
                )
            if any(not isinstance(token_id, int) for token_id in item):
                raise ValidationError(
                    f"items[{idx}] contains non-integer values",
                    param="items",
                    code="invalid_token_id_type",
                )
            normalized_items.append(item)
        return normalized_items

    def _validate_label_token_ids_for_score(
        self,
        label_token_ids: list[int] | None,
        *,
        apply_softmax: bool,
    ) -> None:
        if label_token_ids is None:
            raise ValidationError(
                "label_token_ids is required",
                param="label_token_ids",
                code="missing_label_token_ids",
            )
        if not isinstance(label_token_ids, list) or len(label_token_ids) == 0:
            raise ValidationError(
                "label_token_ids must be a non-empty list of integers",
                param="label_token_ids",
                code="invalid_label_token_ids",
            )
        if not isinstance(apply_softmax, bool):
            raise ValidationError(
                f"apply_softmax must be a boolean, got {type(apply_softmax).__name__}",
                param="apply_softmax",
                code="invalid_apply_softmax",
            )
        vocab_size = len(self.tokenizer) if self.tokenizer is not None else None
        for idx, token_id in enumerate(label_token_ids):
            if not isinstance(token_id, int):
                raise ValidationError(
                    f"label_token_ids[{idx}] must be an integer",
                    param="label_token_ids",
                    code="invalid_token_id_type",
                )
            if token_id < 0:
                raise ValidationError(
                    f"label_token_ids[{idx}] is negative ({token_id})",
                    error_type="invalid_value_error",
                    param="label_token_ids",
                    code="token_id_negative",
                )
            if vocab_size is not None and token_id >= vocab_size:
                raise ValidationError(
                    f"label_token_ids[{idx}] ({token_id}) exceeds vocabulary size ({vocab_size})",
                    error_type="invalid_value_error",
                    param="label_token_ids",
                    code="token_id_exceeds_vocab",
                )

    async def prefill_scoring_cache(self, query: str | list[int] | None = None) -> str:
        query_tokens = self._normalize_score_query_tokens(query)
        return await self._prefill_and_cache(query_tokens)

    async def score_from_cache(
        self,
        cache_handle: str,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        if not isinstance(cache_handle, str) or len(cache_handle) == 0:
            raise ValidationError(
                "cache_handle must be a non-empty string",
                param="cache_handle",
                code="invalid_cache_handle",
            )
        item_tokens_list = self._normalize_score_item_tokens(items)
        self._validate_label_token_ids_for_score(
            label_token_ids,
            apply_softmax=apply_softmax,
        )
        (
            resolved_items_per_step,
            resolved_max_total_tokens,
            resolved_token_budget,
        ) = self._resolve_score_from_cache_v2_items_per_step(
            query_tokens=[],
            items=item_tokens_list,
        )
        fastpath_out = await self._score_from_cache_fastpath_v2(
            cache_handle=cache_handle,
            items=item_tokens_list,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            items_per_step=resolved_items_per_step,
            token_budget=resolved_token_budget,
            max_total_tokens=resolved_max_total_tokens,
        )
        if not fastpath_out.success:
            error_msg = fastpath_out.error_msg or fastpath_out.fallback_reason or "unknown_error"
            raise RuntimeError(f"score_from_cache failed: {error_msg}")
        if len(fastpath_out.scores) != len(item_tokens_list):
            raise RuntimeError(
                "score_from_cache returned wrong score count: "
                f"{len(fastpath_out.scores)} != {len(item_tokens_list)}."
            )
        return fastpath_out.scores

    async def release_scoring_cache(self, cache_handle: str) -> bool:
        if not isinstance(cache_handle, str) or len(cache_handle) == 0:
            raise ValidationError(
                "cache_handle must be a non-empty string",
                param="cache_handle",
                code="invalid_cache_handle",
            )
        return await self._release_cache(cache_handle)

    async def _prefill_and_cache(self, query_tokens: list[int]) -> str:
        cache_handle = uuid.uuid4().hex
        logger.debug(
            "Prefill+extend: starting prefill cache request rid=%s query_tokens=%d",
            cache_handle,
            len(query_tokens),
        )
        req = GenerateReqInput(
            input_ids=query_tokens,
            sampling_params={"max_new_tokens": 1},
            return_logprob=False,
            cache_for_scoring=True,
            is_single=True,
            rid=cache_handle,
        )
        async for _ in self.generate_request(req):
            pass
        logger.debug("Prefill+extend: prefill cache ready rid=%s", cache_handle)
        return cache_handle

    async def _batched_extend_score(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        scores, _ = await self._batched_extend_score_with_metrics(
            cache_handle=cache_handle,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
        )
        return scores

    async def _batched_extend_score_with_metrics(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool = False,
    ) -> tuple[list[list[float]], dict[str, float | int]]:
        if not items:
            return (
                [],
                {
                    "dispatch_count": 0,
                    "queue_wait_s": 0.0,
                    "device_compute_s": 0.0,
                    "host_orchestration_s": 0.0,
                    "lifecycle_requests_sent": 0,
                    "lifecycle_results_received": 0,
                },
            )
        logger.debug(
            "Prefill+extend: scoring extend batch handle=%s batch_items=%d",
            cache_handle,
            len(items),
        )
        requests = GenerateReqInput(
            input_ids=items,
            sampling_params={"max_new_tokens": 1},
            return_logprob=True,
            return_output_logprob_only=False,
            token_ids_logprob=label_token_ids,
            extend_from_cache=cache_handle,
            stream=False,
        )
        results = []
        async for res in self.generate_request(requests):
            if isinstance(res, list):
                results.extend(res)
            else:
                results.append(res)
        if all("index" in result for result in results):
            results.sort(key=lambda x: x["index"])
        if len(results) != len(items):
            raise RuntimeError(
                f"Expected {len(items)} extend results for cache handle {cache_handle}, "
                f"but got {len(results)}."
            )

        scores = []
        for result in results:
            meta_info = result.get("meta_info", {})
            finish_reason = meta_info.get("finish_reason")
            if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
                raise RuntimeError(
                    "Prefill+extend extend request aborted for "
                    f"{meta_info.get('id', '<unknown>')}: {finish_reason}"
                )
            output_logprobs = meta_info.get("output_token_ids_logprobs", [])
            if not output_logprobs or not output_logprobs[0]:
                raise RuntimeError(
                    "output_token_ids_logprobs is empty for prefill+extend request "
                    f"{meta_info.get('id', '<unknown>')}."
                )
            logprobs_map = {}
            for logprob, token_id, _ in output_logprobs[0]:
                if token_id in label_token_ids:
                    logprobs_map[token_id] = logprob
            item_scores = [
                logprobs_map.get(token_id, float("-inf")) for token_id in label_token_ids
            ]
            if all(score == float("-inf") for score in item_scores):
                raise RuntimeError(
                    "No requested label token IDs were found in output_token_ids_logprobs for "
                    f"{meta_info.get('id', '<unknown>')}."
                )
            if apply_softmax:
                scores.append(_stable_softmax(item_scores))
            else:
                scores.append([math.exp(x) if x != float("-inf") else 0.0 for x in item_scores])

        logger.debug(
            "Prefill+extend: completed extend batch handle=%s batch_items=%d",
            cache_handle,
            len(items),
        )
        return (
            scores,
            {
                "dispatch_count": 1,
                "queue_wait_s": 0.0,
                "device_compute_s": 0.0,
                "host_orchestration_s": 0.0,
                "lifecycle_requests_sent": len(items),
                "lifecycle_results_received": len(results),
            },
        )

    def _release_cache_background(self, cache_handle: str) -> None:
        async def _release_and_log() -> None:
            try:
                released = await self._release_cache(cache_handle)
                if not released:
                    logger.warning(
                        "Prefill+extend cache handle=%s was not cleanly released.",
                        cache_handle,
                    )
            except Exception:
                logger.exception(
                    "Unexpected failure in background prefill+extend cache release "
                    "task for handle=%s.",
                    cache_handle,
                )

        task = asyncio.create_task(_release_and_log())
        self.asyncio_tasks.add(task)
        task.add_done_callback(self.asyncio_tasks.discard)

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
            self._release_cache_background(cache_handle)
