import logging
import uuid

from sgl_jax.srt.managers.io_struct import GenerateReqInput, ScoreFromCacheReqInput, ScoreFromCacheReqOutput

logger = logging.getLogger(__name__)


class TokenizerScoringMixin:
    """Mixin for TokenizerManager to handle scoring requests."""

    async def _prefill_and_cache(self, query_tokens: list[int]) -> str:
        """Prefill the query and cache it, returning a handle."""
        # This method needs to create a special request to the scheduler
        # to prefill the prompt and keep it in cache.
        # In vLLM/SGLang, this is often done by sending a regular request
        # with max_new_tokens=0 and a flag to keep the cache.

        rid = f"prefill-cache-{uuid.uuid4().hex}"
        batch_request = GenerateReqInput(
            input_ids=[query_tokens],
            return_logprob=False,
            stream=False,
            sampling_params={"max_new_tokens": 0},
            cache_for_scoring=True,
            rid=rid,
        )

        logger.debug(
            "Prefill+extend: sending prefill request rid=%s query_len=%d",
            rid,
            len(query_tokens),
        )
        # Use generate_request entry point to handle the prefill
        results_gen = self.generate_request(batch_request, None)
        async for results in results_gen:
            for result in results:
                if result["meta_info"]["finish_reason"] is not None:
                    logger.debug("Prefill+extend: prefill completed for rid=%s", rid)
                    return result["meta_info"]["id"]

        raise RuntimeError(f"Prefill failed to yield a result for rid={rid}")

    async def _score_from_cache_fastpath_v2(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
    ) -> ScoreFromCacheReqOutput:
        """Construct and send ScoreFromCacheReqInput to scheduler."""
        req_rid = f"scorev2-{uuid.uuid4().hex}"
        timeout_s = float(
            getattr(
                self.server_args,
                "multi_item_prefill_extend_cache_timeout",
                60.0,
            )
        )
        items_per_step = int(
            getattr(
                self.server_args,
                "multi_item_score_from_cache_v2_items_per_step",
                64,
            )
        )

        logger.debug(
            "Fastpath v2: sending score request rid=%s handle=%s items=%d",
            req_rid,
            cache_handle,
            len(items),
        )

        # This relies on score_from_cache_v2_communicator being initialized in TokenizerManager.__init__
        outputs = await self.score_from_cache_v2_communicator(
            ScoreFromCacheReqInput(
                rid=req_rid,
                cache_handle=cache_handle,
                items_2d=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                items_per_step=items_per_step,
            )
        )

        if not outputs:
            return ScoreFromCacheReqOutput(
                success=False,
                scores=[],
                fallback_reason="no_scheduler_response",
                error_msg="No score-from-cache v2 response from scheduler.",
            )
        return outputs[0]

    async def score_prefill_extend(
        self,
        query_tokens: list[int],
        item_tokens_list: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        """Orchestrate the prefill and extend steps for scoring."""
        if not item_tokens_list:
            return []

        # Step 1: Prefill query and get cache handle
        cache_handle = await self._prefill_and_cache(query_tokens)

        # Step 2: Send fastpath scoring request
        try:
            fastpath_out = await self._score_from_cache_fastpath_v2(
                cache_handle=cache_handle,
                items=item_tokens_list,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
            )
            if fastpath_out.success:
                return fastpath_out.scores
            else:
                logger.error(
                    "Scoring fastpath failed: %s",
                    fastpath_out.error_msg,
                )
                raise RuntimeError(f"Scoring failed: {fastpath_out.error_msg}")
        except Exception as e:
            logger.exception("Error during score_prefill_extend")
            raise e
