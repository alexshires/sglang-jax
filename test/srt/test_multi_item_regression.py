"""
Regression checks for multi-item score API correctness.
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
QUERY_TEXT = "Classify the statement:"
BASE_ITEMS = [
    "This answer is correct.",
    "This answer is wrong.",
    "The statement is uncertain.",
    "The response is incomplete.",
    "The evidence supports the claim.",
]


def _skip_if_no_tpu() -> None:
    if not any(device.platform == "tpu" for device in jax.devices()):
        raise unittest.SkipTest("Multi-item regression tests require TPU.")


def _max_abs_diff(vec_a: list[float], vec_b: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(vec_a, vec_b, strict=True))


def _build_score_engine() -> Engine:
    return Engine(
        model_path=TEST_MODEL_NAME,
        trust_remote_code=True,
        tp_size=1,
        device="tpu",
        random_seed=3,
        node_rank=0,
        mem_fraction_static=0.6,
        download_dir=DOWNLOAD_DIR,
        dtype="bfloat16",
        skip_server_warmup=True,
        attention_backend="fa",
        precompile_token_paddings=[1024],
        precompile_bs_paddings=[1, 2, 4, 8, 16],
        max_running_requests=16,
        page_size=64,
        log_requests=False,
        enable_deterministic_sampling=True,
        disable_radix_cache=False,
        enable_scoring_cache=True,
        multi_item_enable_prefill_extend=True,
        multi_item_extend_batch_size=8,
    )


class TestMultiItemRegression(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _skip_if_no_tpu()
        cls.tokenizer = get_tokenizer(TEST_MODEL_NAME)

    def test_prefill_extend_flow_no_regression(self):
        engine = _build_score_engine()
        try:
            items = [BASE_ITEMS[i % len(BASE_ITEMS)] for i in range(16)]
            label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

            first_scores = engine.score(
                query=QUERY_TEXT,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )
            second_scores = engine.score(
                query=QUERY_TEXT,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )

            self.assertEqual(len(first_scores), len(items))
            self.assertEqual(len(second_scores), len(items))
            for score_vec in first_scores + second_scores:
                self.assertAlmostEqual(sum(score_vec), 1.0, places=5)

            max_replay_diff = max(
                _max_abs_diff(first_scores[i], second_scores[i])
                for i in range(len(items))
            )
            self.assertLessEqual(
                max_replay_diff,
                1e-4,
                f"Prefill+extend replay drift exceeded tolerance: {max_replay_diff}",
            )
        finally:
            engine.shutdown()
            jax.clear_caches()


if __name__ == "__main__":
    unittest.main()
