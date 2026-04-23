"""Tokenizer score request API helpers."""

from __future__ import annotations

from sgl_jax.srt.managers.tokenizer_score_common import *


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
        if label_token_ids is None:
            raise ValueError("label_token_ids must be provided")

        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
            for token_id in label_token_ids:
                if token_id >= vocab_size:
                    raise ValueError(
                        f"Token ID {token_id} is out of vocabulary (vocab size: {vocab_size})"
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
