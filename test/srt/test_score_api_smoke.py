"""Smoke tests for `Engine.score(...)`."""

import unittest

import jax

from sgl_jax.test.score_test_utils import (
    ScoreAPITestCase,
    ScoreTestConfig,
    assert_scores_shape,
    assert_scores_valid,
    get_single_token_id,
)


def _skip_if_no_tpu() -> None:
    if not any(device.platform == "tpu" for device in jax.devices()):
        raise unittest.SkipTest("Score API smoke tests require TPU.")


class TestScoreAPISmoke(ScoreAPITestCase):
    config = ScoreTestConfig(
        mem_fraction_static=0.7,
        max_running_requests=8,
        precompile_bs_paddings=[8],
        precompile_token_paddings=[1024],
    )

    @classmethod
    def setUpClass(cls):
        _skip_if_no_tpu()
        super().setUpClass()

    def test_score_text_input_smoke(self):
        scores = self.engine.score(
            query="The capital of France is",
            items=[" Paris", " London"],
            label_token_ids=[
                get_single_token_id(self.tokenizer, " A"),
                get_single_token_id(self.tokenizer, " B"),
            ],
            apply_softmax=True,
        )

        assert_scores_shape(scores, expected_items=2, expected_labels=2, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_score_token_input_smoke(self):
        scores = self.engine.score(
            query=self.tokenizer.encode("The answer is", add_special_tokens=False),
            items=[
                self.tokenizer.encode(" yes", add_special_tokens=False),
                self.tokenizer.encode(" no", add_special_tokens=False),
            ],
            label_token_ids=[
                get_single_token_id(self.tokenizer, " A"),
                get_single_token_id(self.tokenizer, " B"),
            ],
            apply_softmax=True,
        )

        assert_scores_shape(scores, expected_items=2, expected_labels=2, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)
