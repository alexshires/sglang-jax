"""Tokenizer score request API helpers."""

from __future__ import annotations

import logging
import math
from typing import Any

from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.tokenizer_score_common import _stable_softmax
from sgl_jax.srt.validation import ValidationError, validate_score_request

logger = logging.getLogger(__name__)


class TokenizerScoreApiMixin:
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
            type(query).__name__,
            len(items) if items is not None else 0,
            label_token_ids,
            apply_softmax,
            item_first,
        )

        vocab_size = len(self.tokenizer) if self.tokenizer is not None else None
        validate_score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            vocab_size=vocab_size,
        )

        max_multi_item_count = int(getattr(self.server_args, "max_multi_item_count", 0) or 0)
        if (
            max_multi_item_count > 0
            and isinstance(items, list)
            and len(items) > max_multi_item_count
        ):
            raise ValidationError(
                message=f"Too many items for scoring: {len(items)} > {max_multi_item_count}",
                error_type="invalid_value_error",
                param="items",
                code="too_many_items",
            )

        if isinstance(query, str) and (
            isinstance(items, str)
            or (isinstance(items, list) and (not items or isinstance(items[0], str)))
        ):
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
                sampling_params={"max_new_tokens": 1},
            )
        elif (
            isinstance(query, list)
            and isinstance(items, list)
            and items
            and isinstance(items[0], list)
        ):
            if item_first:
                input_ids_list = [item + query for item in items]
            else:
                input_ids_list = [query + item for item in items]
            batch_request = GenerateReqInput(
                input_ids=input_ids_list,
                return_logprob=True,
                token_ids_logprob=label_token_ids,
                stream=False,
                sampling_params={"max_new_tokens": 1},
            )
        else:
            raise ValueError("Invalid combination of query/items types for score_request.")

        results = await self.generate_request(batch_request, request).__anext__()
        scores = []

        for result in results:
            logprobs = {}
            for logprob, token_id, _ in result["meta_info"].get("output_token_ids_logprobs", [])[0]:
                if token_id in label_token_ids:
                    logprobs[token_id] = logprob

            score_list = [logprobs.get(token_id, float("-inf")) for token_id in label_token_ids]
            if apply_softmax:
                score_list = _stable_softmax(score_list)
            else:
                score_list = [math.exp(x) if x != float("-inf") else 0.0 for x in score_list]

            scores.append(score_list)

        return scores
