"""Tokenizer score request API helpers."""

from __future__ import annotations

from sgl_jax.srt.managers.tokenizer_score_common import *


class TokenizerScoreApiMixin:
    def _normalize_score_query_tokens(self, query: str | list[int] | None) -> list[int]:
        if query is None:
            raise ValueError("query is required")

        if isinstance(query, str):
            if len(query) == 0:
                raise ValueError("query cannot be empty")
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text scoring.")
            return self.tokenizer.encode(query, add_special_tokens=False)

        if not isinstance(query, list):
            raise ValueError(
                f"query must be a string or list of integers, got {type(query).__name__}"
            )
        if len(query) == 0:
            raise ValueError("query token list cannot be empty")
        if any(not isinstance(token_id, int) for token_id in query):
            raise ValueError("query contains non-integer values in token input mode")
        return query

    def _normalize_score_item_tokens(
        self,
        items: str | list[str] | list[list[int]] | None,
    ) -> list[list[int]]:
        if items is None:
            raise ValueError("items is required")

        if isinstance(items, str):
            items = [items]

        if not isinstance(items, list):
            raise ValueError("items must be a list of strings or list of token ID lists")
        if len(items) == 0:
            raise ValueError("items cannot be empty. At least one item is required.")

        if isinstance(items[0], str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text scoring.")
            for idx, item in enumerate(items):
                if not isinstance(item, str):
                    raise ValueError(f"items[{idx}] must be a string when using text input mode")
            return [self.tokenizer.encode(item, add_special_tokens=False) for item in items]

        normalized_items: list[list[int]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, list):
                raise ValueError(
                    f"items[{idx}] must be a list of integers when using token input mode"
                )
            if any(not isinstance(token_id, int) for token_id in item):
                raise ValueError(
                    f"items[{idx}] contains non-integer values. All token IDs must be integers."
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
            raise ValueError("label_token_ids is required")
        if not isinstance(label_token_ids, list):
            raise ValueError("label_token_ids must be a list of integers")
        if len(label_token_ids) == 0:
            raise ValueError("label_token_ids cannot be empty. At least one token ID is required.")
        if not isinstance(apply_softmax, bool):
            raise ValueError(f"apply_softmax must be a boolean, got {type(apply_softmax).__name__}")

        vocab_size = len(self.tokenizer) if self.tokenizer is not None else None
        for idx, token_id in enumerate(label_token_ids):
            if not isinstance(token_id, int):
                raise ValueError(
                    f"label_token_ids[{idx}] must be an integer, got {type(token_id).__name__}"
                )
            if token_id < 0:
                raise ValueError(
                    f"label_token_ids[{idx}] is negative ({token_id}). "
                    "Token IDs must be non-negative."
                )
            if vocab_size is not None and token_id >= vocab_size:
                raise ValueError(
                    f"label_token_ids[{idx}] ({token_id}) exceeds vocabulary size ({vocab_size})"
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
            raise ValueError("cache_handle must be a non-empty string")

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
            raise ValueError("cache_handle must be a non-empty string")
        return await self._release_cache(cache_handle)

    async def score_request(
        self,
        query: str | list[int] | None = None,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        request: Any | None = None,
    ) -> list[list[float]]:
        """
        See Engine.score() for more details.
        """
        logger.debug(
            "Score request: query_type=%s, items_len=%s, label_token_ids=%s, "
            "apply_softmax=%s, item_first=%s",
            type(query),
            len(items) if items is not None else 0,
            label_token_ids,
            apply_softmax,
            item_first,
        )
        # Comprehensive validation per RFC-006
        vocab_size = len(self.tokenizer) if self.tokenizer is not None else None
        try:
            validate_score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                vocab_size=vocab_size,
            )
        except ValidationError as e:
            raise ValueError(e.message) from e

        max_multi_item_count = int(getattr(self.server_args, "max_multi_item_count", 0) or 0)
        if (
            max_multi_item_count > 0
            and isinstance(items, list)
            and len(items) > max_multi_item_count
        ):
            raise ValueError(f"Too many items for scoring: {len(items)} > {max_multi_item_count}")

        if getattr(self.server_args, "multi_item_enable_prefill_extend", False):
            query_tokens = self._normalize_score_query_tokens(query)
            item_tokens_list = self._normalize_score_item_tokens(items)

            if item_first:
                logger.warning("Ignoring item_first=True for prefill+extend strategy.")

            return await self.score_prefill_extend(
                query_tokens=query_tokens,
                item_tokens_list=item_tokens_list,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
            )

        def _convert_logprobs(logprobs_data: list) -> list[float]:
            logprobs = {}
            for logprob, token_id, _ in logprobs_data:
                if token_id in label_token_ids:
                    logprobs[token_id] = logprob
            score_list = [logprobs.get(token_id, float("-inf")) for token_id in label_token_ids]
            if apply_softmax:
                return _stable_softmax(score_list)
            return [math.exp(x) if x != float("-inf") else 0.0 for x in score_list]

        # Handle string or tokenized query/items in single-item mode
        if isinstance(query, str) and (
            isinstance(items, str)
            or (isinstance(items, list) and (not items or isinstance(items[0], str)))
        ):
            # Both query and items are text
            items_list = [items] if isinstance(items, str) else items
            if item_first:
                prompts = [f"{item}{query}" for item in items_list]
            else:
                prompts = [f"{query}{item}" for item in items_list]
            batch_request = GenerateReqInput(
                text=prompts,
                return_logprob=True,
                token_ids_logprob=label_token_ids,
                stream=False,
                sampling_params={"max_new_tokens": 0},  # Prefill-only: no generation needed
            )
            logger.debug(
                "Scoring text prompts: num_items=%d, first_prompt_len=%d",
                len(prompts),
                len(prompts[0]),
            )
        elif (
            isinstance(query, list)
            and isinstance(items, list)
            and items
            and isinstance(items[0], list)
        ):
            # Both query and items are token IDs
            if item_first:
                input_ids_list = [item + query for item in items]
            else:
                input_ids_list = [query + item for item in items]
            batch_request = GenerateReqInput(
                input_ids=input_ids_list,
                return_logprob=True,
                token_ids_logprob=label_token_ids,
                stream=False,
                sampling_params={"max_new_tokens": 0},  # Prefill-only: no generation needed
            )
            logger.debug(
                "Scoring token IDs: num_items=%d, first_ids_len=%d",
                len(input_ids_list),
                len(input_ids_list[0]),
            )
        else:
            raise ValueError("Invalid combination of query/items types for score_request.")

        results = await self.generate_request(batch_request, request).__anext__()
        scores = []

        for result in results:
            output_logprobs = result["meta_info"].get("output_token_ids_logprobs", [])
            if not output_logprobs or len(output_logprobs) == 0:
                raise RuntimeError(
                    f"output_token_ids_logprobs is empty for request "
                    f"{result['meta_info'].get('id', '<unknown>')}. "
                    "This indicates token_ids_logprobs were not computed properly."
                )
            scores.append(_convert_logprobs(output_logprobs[0]))

        return scores

    async def _prefill_and_cache(self, query_tokens: list[int]) -> str:
        """Prefill query and return handle to cached KV."""
        cache_handle = uuid.uuid4().hex
        logger.debug(
            "Prefill+extend: starting prefill cache request rid=%s query_tokens=%d",
            cache_handle,
            len(query_tokens),
        )
        req = GenerateReqInput(
            # Use a single request (flat token list), not a batch-of-1. This keeps
            # the cache handle stable and avoids rid suffix rewrites during normalize.
            input_ids=query_tokens,
            sampling_params={"max_new_tokens": 0},  # Prefill only
            return_logprob=False,
            cache_for_scoring=True,  # New flag
            is_single=True,
            rid=cache_handle,
        )

        # Execute request
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
        """Score items by extending from cached prefix."""
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
            sampling_params={"max_new_tokens": 0},
            return_logprob=True,
            return_output_logprob_only=False,
            token_ids_logprob=label_token_ids,
            extend_from_cache=cache_handle,
            stream=False,
        )

        results = []
        async for res in self.generate_request(requests):
            # res is a list of results for the batch
            if isinstance(res, list):
                results.extend(res)
            else:
                results.append(res)

        # Sort results by index when present so scores align with request order.
        if all("index" in result for result in results):
            results.sort(key=lambda x: x["index"])

        if len(results) != len(items):
            raise RuntimeError(
                f"Expected {len(items)} extend results for cache handle {cache_handle}, "
                f"but got {len(results)}."
            )

        scores = []
        scheduler_dispatch_counts = []
        scheduler_queue_wait = []
        scheduler_device_compute = []
        scheduler_host_overhead = []
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
            if meta_info.get("scheduler_dispatch_count") is not None:
                scheduler_dispatch_counts.append(int(meta_info["scheduler_dispatch_count"]))
            if meta_info.get("scheduler_queue_wait_s") is not None:
                scheduler_queue_wait.append(float(meta_info["scheduler_queue_wait_s"]))
            if meta_info.get("scheduler_device_compute_s") is not None:
                scheduler_device_compute.append(float(meta_info["scheduler_device_compute_s"]))
            if meta_info.get("scheduler_host_overhead_s") is not None:
                scheduler_host_overhead.append(float(meta_info["scheduler_host_overhead_s"]))

        logger.debug(
            "Prefill+extend: completed extend batch handle=%s batch_items=%d",
            cache_handle,
            len(items),
        )
        return (
            scores,
            {
                "dispatch_count": (
                    max(scheduler_dispatch_counts) if scheduler_dispatch_counts else 1
                ),
                "queue_wait_s": (max(scheduler_queue_wait) if scheduler_queue_wait else 0.0),
                "device_compute_s": (
                    max(scheduler_device_compute) if scheduler_device_compute else 0.0
                ),
                "host_orchestration_s": (
                    max(scheduler_host_overhead) if scheduler_host_overhead else 0.0
                ),
                "lifecycle_requests_sent": len(items),
                "lifecycle_results_received": len(results),
            },
        )
