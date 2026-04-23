"""
Focused smoke coverage for multi-item prefill+extend scoring.
"""

import os
import unittest

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)


class TestMultiItemPrefillExtendFocus(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=TEST_MODEL_NAME,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=42,
            mem_fraction_static=0.7,
            multi_item_enable_prefill_extend=True,
            enable_scoring_cache=True,
            disable_radix_cache=False,
            multi_item_extend_batch_size=4,
            log_requests=True,
            enable_deterministic_sampling=True,
            precompile_bs_paddings=[1, 2, 4],
            download_dir="/dev/shm",
            dtype="bfloat16",
        )

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_prefill_extend_flow(self):
        scores = self.engine.score(
            query="What is the capital of France?",
            items=["Paris", "London", "Berlin"],
            label_token_ids=[100, 200],
            apply_softmax=True,
        )
        self.assertEqual(len(scores), 3)
        for item_scores in scores:
            self.assertEqual(len(item_scores), 2)
            self.assertAlmostEqual(sum(item_scores), 1.0, places=5)

    def test_prefill_extend_batching(self):
        scores = self.engine.score(
            query="Rank these numbers:",
            items=[str(i) for i in range(10)],
            label_token_ids=[15, 16],
            apply_softmax=True,
        )
        self.assertEqual(len(scores), 10)


if __name__ == "__main__":
    unittest.main()
