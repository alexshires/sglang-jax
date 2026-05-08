"""Smoke coverage for multi-item prefill+extend scoring."""

import unittest

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
        raise unittest.SkipTest("multi-item prefill+extend smoke tests require TPU.")


class TestMultiItemPrefillExtendSmoke(ScoreAPITestCase):
    config = ScoreTestConfig(
        random_seed=42,
        mem_fraction_static=0.7,
        precompile_bs_paddings=[1, 2, 4],
    )

    @classmethod
    def setUpClass(cls):
        _skip_if_no_tpu()
        cls.engine = cls._build_prefill_extend_engine()
        cls.tokenizer = cls.engine.tokenizer_manager.tokenizer

    @classmethod
    def _build_prefill_extend_engine(cls):
        from sgl_jax.srt.entrypoints.engine import Engine

        return Engine(
            model_path=cls.config.model_name,
            trust_remote_code=True,
            tp_size=cls.config.tp_size,
            device=cls.config.device,
            random_seed=cls.config.random_seed,
            mem_fraction_static=cls.config.mem_fraction_static,
            multi_item_enable_prefill_extend=True,
            enable_scoring_cache=True,
            disable_radix_cache=False,
            multi_item_extend_batch_size=4,
            log_requests=True,
            enable_deterministic_sampling=True,
            precompile_bs_paddings=cls.config.precompile_bs_paddings,
            download_dir=cls.config.download_dir,
            dtype=cls.config.dtype,
        )

    def test_prefill_extend_flow(self):
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        scores = self.engine.score(
            query="What is the capital of France?",
            items=["Paris", "London", "Berlin"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        assert_scores_shape(scores, expected_items=3, expected_labels=2, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_prefill_extend_batching(self):
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        scores = self.engine.score(
            query="Rank these numbers:",
            items=[str(i) for i in range(10)],
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        assert_scores_shape(scores, expected_items=10, expected_labels=2, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)
