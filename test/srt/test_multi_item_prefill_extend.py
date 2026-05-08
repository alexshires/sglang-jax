"""
Test multi-item scoring with prefill+extend strategy (Workstream B).
"""

import os
import tempfile
import unittest

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.score_test_utils import get_label_token_ids, get_tokenizer
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
DOWNLOAD_DIR = os.getenv("SGLANG_TEST_DOWNLOAD_DIR", tempfile.gettempdir())


def _skip_if_no_tpu() -> None:
    if not any(device.platform == "tpu" for device in jax.devices()):
        raise unittest.SkipTest("Multi-item prefill+extend tests require TPU.")


def _assert_score_rows_close(
    test_case: unittest.TestCase,
    actual: list[float],
    expected: list[float],
    *,
    delta: float = 1e-4,
) -> None:
    test_case.assertEqual(len(actual), len(expected))
    for got, want in zip(actual, expected, strict=True):
        test_case.assertAlmostEqual(got, want, delta=delta)


class TestMultiItemPrefillExtend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _skip_if_no_tpu()
        cls.model_path = TEST_MODEL_NAME
        cls.tokenizer = get_tokenizer(cls.model_path)
        # Initialize engine with prefill+extend enabled and radix cache configuration
        cls.engine = Engine(
            model_path=cls.model_path,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=42,
            mem_fraction_static=0.7,
            # Critical flags for prefill+extend
            multi_item_enable_prefill_extend=True,
            enable_scoring_cache=True,
            # We don't need disable_radix_cache=True here because enable_scoring_cache overrides intent
            # but usually multi-item sets it.
            # Let's set it to False (default) but enable_scoring_cache=True should handle it.
            disable_radix_cache=False,
            multi_item_extend_batch_size=4,
            log_requests=True,
            enable_deterministic_sampling=True,
            precompile_bs_paddings=[1, 2, 4],
            # Standard args
            download_dir=DOWNLOAD_DIR,
            dtype="bfloat16",
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_prefill_extend_flow(self):
        """Test the basic flow of prefill+extend scoring."""
        query = "What is the capital of France?"
        items = ["Paris", "London", "Berlin"]
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), len(items))
        for item_idx, item_scores in enumerate(scores):
            self.assertEqual(len(item_scores), len(label_token_ids))
            self.assertAlmostEqual(sum(item_scores), 1.0, places=5)
            single_score = self.engine.score(
                query=query,
                items=[items[item_idx]],
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )
            _assert_score_rows_close(self, item_scores, single_score[0])

    def test_prefill_extend_batching(self):
        """Test with more items than extend batch size."""
        query = "Rank these numbers:"
        items = [str(i) for i in range(10)]
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        # Batch size is 4 (set in setUpClass)
        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), 10)
        for item_scores in scores:
            self.assertEqual(len(item_scores), len(label_token_ids))
            self.assertAlmostEqual(sum(item_scores), 1.0, places=5)

        for item_idx in (0, 4, 9):
            single_score = self.engine.score(
                query=query,
                items=[items[item_idx]],
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )
            _assert_score_rows_close(self, scores[item_idx], single_score[0])


if __name__ == "__main__":
    unittest.main()
