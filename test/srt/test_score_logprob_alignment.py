import math
import os
import unittest
from types import SimpleNamespace

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)


def _has_tpu() -> bool:
    return any(device.platform == "tpu" for device in jax.devices())


class _DummyScheduler(SchedulerOutputProcessorMixin):
    """Minimal host for add_input_logprob_return_values."""


class TestInputLogprobAlignment(CustomTestCase):
    def test_input_logprobs_align_to_input_token_positions(self):
        scheduler = _DummyScheduler()
        scheduler.model_config = SimpleNamespace(vocab_size=32000)

        token_ids_logprob = [1, 2]
        req = SimpleNamespace(
            input_token_logprobs=None,
            temp_input_top_logprobs_val=None,
            temp_input_top_logprobs_idx=None,
            temp_input_token_ids_logprobs_val=None,
            temp_input_token_ids_logprobs_idx=None,
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None,
            input_token_ids_logprobs_idx=None,
            top_logprobs_num=2,
            token_ids_logprob=token_ids_logprob,
            origin_input_ids=[101, 31998, 31999, 32005, 301, 9],
            logprob_start_len=0,
            return_logprob=True,
        )
        # Shape is [batch][token_position][requested_token_id].
        input_token_ids_logprobs_val = [
            [[f"v{token_pos}:{token_id}" for token_id in token_ids_logprob] for token_pos in range(6)]
        ]
        input_token_ids_logprobs_idx = [
            [[f"i{token_pos}:{token_id}" for token_id in token_ids_logprob] for token_pos in range(6)]
        ]
        input_top_logprobs_val = [
            [[f"top-v{token_pos}:{rank}" for rank in range(2)] for token_pos in range(6)]
        ]
        input_top_logprobs_idx = [
            [[f"top-i{token_pos}:{rank}" for rank in range(2)] for token_pos in range(6)]
        ]
        output = LogitsProcessorOutput(
            next_token_logits=None,
            input_token_logprobs=[0, 1, 2, 3, 4, 5],
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
        )

        scheduler.add_input_logprob_return_values(
            i=0,
            req=req,
            output=output,
            logprob_pt=0,
            num_input_logprobs=6,
            last_prefill_chunk=True,
        )

        self.assertIsNone(req.input_token_logprobs)
        self.assertEqual(req.input_token_logprobs_val, [None, 0, 1, 2, 3, 4])
        self.assertEqual(req.input_token_logprobs_idx, [101, 31998, 0, 0, 301, 9])
        self.assertEqual(req.input_top_logprobs_val, [None, *input_top_logprobs_val[0][:-1]])
        self.assertEqual(req.input_top_logprobs_idx, [None, *input_top_logprobs_idx[0][:-1]])
        self.assertEqual(
            req.input_token_ids_logprobs_val,
            [None, *input_token_ids_logprobs_val[0][:-1]],
        )
        self.assertEqual(
            req.input_token_ids_logprobs_idx,
            [None, *input_token_ids_logprobs_idx[0][:-1]],
        )
        self.assertIsNone(req.temp_input_top_logprobs_val)
        self.assertIsNone(req.temp_input_top_logprobs_idx)
        self.assertIsNone(req.temp_input_token_ids_logprobs_val)
        self.assertIsNone(req.temp_input_token_ids_logprobs_idx)


class TestScoreEndpointSmoke(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not _has_tpu():
            raise unittest.SkipTest("score endpoint smoke test requires TPU")

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
            download_dir="/tmp",
            dtype="bfloat16",
            precompile_bs_paddings=[8],
            max_running_requests=8,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024],
            page_size=64,
            log_level="warning",
            log_requests=False,
            enable_deterministic_sampling=True,
        )

    @classmethod
    def tearDownClass(cls):
        engine = getattr(cls, "engine", None)
        if engine is not None:
            engine.shutdown()
        jax.clear_caches()

    def test_score_softmax_smoke(self):
        scores = self.engine.score(
            query="The test was",
            items=[f"item {i}" for i in range(4)],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
        )
        self.assertEqual(len(scores), 4)
        for score_list in scores:
            self.assertEqual(len(score_list), 3)
            self.assertTrue(all(math.isfinite(score) for score in score_list))
            self.assertTrue(all(0.0 <= score <= 1.0 for score in score_list))
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)
