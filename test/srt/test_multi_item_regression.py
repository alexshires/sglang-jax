"""
Regression checks for multi-item score API correctness.
"""

import unittest

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import CustomTestCase

TEST_MODEL_NAME = "/models/Qwen/Qwen3-0.6B"
LABEL_TOKEN_IDS = [9834, 902]
QUERY_IDS = [1957, 1437, 25975, 25]
BASE_ITEMS = [
    [358, 2948, 419, 1985, 13],
    [1096, 374, 17478, 323, 38123, 13],
    [1084, 4278, 438, 3601, 13],
    [56938, 4271, 323, 4937, 9691, 13],
    [2806, 5802, 279, 3349, 13],
]


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
        download_dir="/dev/shm",
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
    def test_prefill_extend_flow_no_regression(self):
        engine = _build_score_engine()
        try:
            items = [BASE_ITEMS[i % len(BASE_ITEMS)] for i in range(16)]

            first_scores = engine.score(
                query=QUERY_IDS,
                items=items,
                label_token_ids=LABEL_TOKEN_IDS,
                apply_softmax=True,
            )
            second_scores = engine.score(
                query=QUERY_IDS,
                items=items,
                label_token_ids=LABEL_TOKEN_IDS,
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
