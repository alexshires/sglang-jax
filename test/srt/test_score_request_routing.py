import os
from unittest.mock import patch

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)


class TestScoreRequestRouting(CustomTestCase):
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
            precompile_bs_paddings=[16],
            max_running_requests=16,
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

    def test_score_request_uses_prefill_only_routing(self):
        captured_requests = []
        original_generate_request = self.engine.tokenizer_manager.generate_request

        async def capture_generate_request(req, request=None):
            captured_requests.append(req)
            async for result in original_generate_request(req, request):
                yield result

        with patch.object(
            self.engine.tokenizer_manager,
            "generate_request",
            side_effect=capture_generate_request,
        ):
            scores = self.engine.score(
                query="What is the capital of",
                items=["France", "Germany"],
                label_token_ids=[1, 2, 3],
                apply_softmax=True,
            )

        self.assertEqual(len(scores), 2)
        self.assertGreater(len(captured_requests), 0)

        request = captured_requests[0]
        if isinstance(request.sampling_params, dict):
            max_new_tokens = request.sampling_params.get("max_new_tokens", 0)
        elif isinstance(request.sampling_params, list):
            max_new_tokens = request.sampling_params[0].get("max_new_tokens", 0)
        else:
            max_new_tokens = getattr(request.sampling_params, "max_new_tokens", 0)

        self.assertLessEqual(max_new_tokens, 1)
        self.assertTrue(request.return_logprob)
        self.assertFalse(request.stream)

        if (
            isinstance(request.token_ids_logprob, list)
            and request.token_ids_logprob
            and isinstance(request.token_ids_logprob[0], list)
        ):
            for item_token_ids in request.token_ids_logprob:
                self.assertEqual(item_token_ids, [1, 2, 3])
        else:
            self.assertEqual(request.token_ids_logprob, [1, 2, 3])
