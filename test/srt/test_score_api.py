
import unittest
import asyncio
from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEEPSEEK_R1_DISTILL_QWEN_1_5B

# Use the same default config as test_logprobs.py
DEFAULT_ENGINE_CONFIG = {
    "model_path": DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    "random_seed": 27,
    "device": "tpu",
    "chunked_prefill_size": 8192,
    "dtype": "bfloat16",
    "max_running_requests": 64,
    "page_size": 1, # Use 1 for simpler memory management in tests? test_logprobs uses 64
    "max_total_tokens": 32000,
    "precompile_token_paddings": [8192],
    "precompile_bs_paddings": [1, 64],
    "use_sort_for_toppk_minp": True,
    "mem_fraction_static": 0.8,
    "disable_overlap_schedule": True,
    "trust_remote_code": True,
    "skip_server_warmup": True,
    "tp_size": 1,
    "enable_precision_tracer": False, # Disable for simpler test
    "log_level": "info",
}

class TestScoreAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Launching SGLang-Jax Engine with {DEEPSEEK_R1_DISTILL_QWEN_1_5B}...")
        # Ensure JIT cache is set
        import os
        os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"
        cls.engine = Engine(**DEFAULT_ENGINE_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_score_request(self):
        # We need to access tokenizer_manager directly as Engine.generate doesn't expose score_request yet
        # But Engine has tokenizer_manager
        
        # Define inputs
        query = "The capital of France is"
        items = [" Paris", " London", " Berlin"]
        
        # Run async test
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.run_scoring(query, items))
        
        print(f"Scoring results: {results}")
        
        # Validation
        self.assertEqual(len(results), 3)
        # Paris should have highest score (least negative logprob)
        paris_score = results[0]
        london_score = results[1]
        berlin_score = results[2]
        
        self.assertGreater(paris_score, london_score)
        self.assertGreater(paris_score, berlin_score)
        
        # Check against manual logprob calculation using generate
        # (Optional, but good for verification)
        
    async def run_scoring(self, query, items):
        return await self.engine.tokenizer_manager.score_request(
            query=query,
            items=items,
        )

if __name__ == "__main__":
    unittest.main()
