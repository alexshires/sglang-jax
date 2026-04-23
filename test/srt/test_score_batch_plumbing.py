import os
from types import SimpleNamespace

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)


class _DummyScheduler(SchedulerOutputProcessorMixin):
    pass


def test_input_logprobs_align_to_input_token_positions():
    scheduler = _DummyScheduler()
    scheduler.model_config = SimpleNamespace(vocab_size=32000)

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
        top_logprobs_num=0,
        token_ids_logprob=[1, 2],
        origin_input_ids=[101, 9, 201, 9, 301, 9],
        logprob_start_len=0,
        return_logprob=True,
    )
    output = LogitsProcessorOutput(
        next_token_logits=None,
        input_token_logprobs=[0, 1, 2, 3, 4, 5],
        input_token_ids_logprobs_val=[[["v0"], ["v1"], ["v2"], ["v3"], ["v4"], ["v5"]]],
        input_token_ids_logprobs_idx=[[["i0"], ["i1"], ["i2"], ["i3"], ["i4"], ["i5"]]],
    )

    scheduler.add_input_logprob_return_values(
        i=0,
        req=req,
        output=output,
        logprob_pt=0,
        num_input_logprobs=6,
        last_prefill_chunk=True,
    )

    assert req.input_token_logprobs_val == [None, 0, 1, 2, 3, 4]


class TestScoreBatchPlumbing(CustomTestCase):
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

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_score_batch_smoke(self):
        scores = self.engine.score(
            query="The test was",
            items=[f"item {i}" for i in range(4)],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
        )
        self.assertEqual(len(scores), 4)
        for score_list in scores:
            self.assertEqual(len(score_list), 3)
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)
