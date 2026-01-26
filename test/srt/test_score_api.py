
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

    def test_score_request_with_labels(self):
        # Test the "Old Mode" - Next Token Classification
        query = "The color of the sky is"
        # We want to check prob of " blue" vs " green"
        items = [""] # Items are empty because we are just checking next token of query? 
        # Or items are the candidates...
        # In old API: query="The color of the sky is", items=[" blue", " green"], item_first=False
        # prompts = ["The color of the sky is blue", "The color of the sky is green"]
        # Wait, if I want to check just the next token, I should use items as candidates.
        
        # Le's try the exact usage from the old docstring if available, or just standard usage.
        # If I want to classify "blue" vs "green":
        # query = "The color of the sky is"
        # items = [""]
        # label_token_ids = [token_id_blue, token_id_green]
        
        # Actually proper usage for classification with labels often implies we just want to know 
        # probability of specific tokens at the end of the prompt.
        
        # Let's rely on how `score_request` constructs prompts:
        # prompts = [f"{query}{item}" for item in items_list] (if item_first=False)
        # If items=[""], then prompt is just query.
        
        query = "The color of the sky is"
        items = [""]
        tokenizer = self.engine.tokenizer_manager.tokenizer
        blue_id = tokenizer.encode(" blue")[0]
        green_id = tokenizer.encode(" green")[0]
        label_token_ids = [blue_id, green_id]
        
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.engine.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True
            )
        )
        
        print(f"Label Scoring results: {results}")
        
        # Results should be a list of lists: [[score_blue, score_green]]
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)
        
        blue_score = results[0][0]
        green_score = results[0][1]
        
        # Blue should be higher than green for "The color of the sky is"
        self.assertGreater(blue_score, green_score)


if __name__ == "__main__":
    unittest.main()
