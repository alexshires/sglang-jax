import os

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)


def _single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{text!r} tokenizes to {len(token_ids)} tokens")
    return token_ids[0]


class TestScoreAPISmoke(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=TEST_MODEL_NAME,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            enable_single_process=True,
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.7,
            chunked_prefill_size=1024,
            download_dir="/dev/shm",
            dtype="bfloat16",
            precompile_bs_paddings=[8],
            max_running_requests=8,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024],
            page_size=64,
            log_requests=False,
            enable_deterministic_sampling=True,
        )
        cls.tokenizer = get_tokenizer(TEST_MODEL_NAME, trust_remote_code=True)

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_score_text_input_smoke(self):
        scores = self.engine.score(
            query="The capital of France is",
            items=[" Paris", " London"],
            label_token_ids=[
                _single_token_id(self.tokenizer, " A"),
                _single_token_id(self.tokenizer, " B"),
            ],
            apply_softmax=True,
        )
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(scores[0]), 2)
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=5)

    def test_score_token_input_smoke(self):
        scores = self.engine.score(
            query=self.tokenizer.encode("The answer is", add_special_tokens=False),
            items=[
                self.tokenizer.encode(" yes", add_special_tokens=False),
                self.tokenizer.encode(" no", add_special_tokens=False),
            ],
            label_token_ids=[
                _single_token_id(self.tokenizer, " A"),
                _single_token_id(self.tokenizer, " B"),
            ],
            apply_softmax=True,
        )
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(scores[1]), 2)
