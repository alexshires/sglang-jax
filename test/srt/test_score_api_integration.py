"""Integration tests for score request construction."""

import unittest
from unittest.mock import patch

import jax

from sgl_jax.test.score_test_utils import (
    ScoreAPITestCase,
    ScoreTestConfig,
    assert_scores_shape,
    assert_scores_valid,
    get_label_token_ids,
)


def _skip_if_no_tpu() -> None:
    if not any(device.platform == "tpu" for device in jax.devices()):
        raise unittest.SkipTest("Score API integration tests require TPU.")


class TestScoreAPIIntegration(ScoreAPITestCase):
    config = ScoreTestConfig(
        mem_fraction_static=0.7,
        max_running_requests=16,
        precompile_bs_paddings=[16],
        precompile_token_paddings=[1024],
    )

    @classmethod
    def setUpClass(cls):
        _skip_if_no_tpu()
        super().setUpClass()

    def test_score_request_construction(self):
        """Score requests should request selective logprobs without streaming."""
        captured_requests = []
        original_gen = self.engine.tokenizer_manager.generate_request

        async def mock_generate_request(req, request=None):
            captured_requests.append(req)
            async for result in original_gen(req, request):
                yield result

        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B", " C"])
        with patch.object(
            self.engine.tokenizer_manager,
            "generate_request",
            side_effect=mock_generate_request,
        ):
            scores = self.engine.score(
                query="What is the capital of",
                items=["France", "Germany"],
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )

        assert_scores_shape(scores, expected_items=2, expected_labels=3, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)
        self.assertGreater(len(captured_requests), 0, "Expected a score request.")

        request = captured_requests[0]
        sampling_params = request.sampling_params
        if isinstance(sampling_params, dict):
            max_new_tokens = sampling_params.get("max_new_tokens", 0)
        elif isinstance(sampling_params, list):
            max_new_tokens = sampling_params[0].get("max_new_tokens", 0)
        else:
            max_new_tokens = getattr(sampling_params, "max_new_tokens", 0)

        self.assertLessEqual(
            max_new_tokens,
            1,
            "score requests should avoid decode-phase generation",
        )
        self.assertTrue(request.return_logprob, "score requests must return logprobs")
        self.assertFalse(request.stream, "score requests should not stream")

        token_ids_logprob = request.token_ids_logprob
        if (
            isinstance(token_ids_logprob, list)
            and token_ids_logprob
            and isinstance(token_ids_logprob[0], list)
        ):
            for item_token_ids in token_ids_logprob:
                self.assertEqual(item_token_ids, label_token_ids)
        else:
            self.assertEqual(token_ids_logprob, label_token_ids)
