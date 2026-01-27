
import threading
import unittest
import asyncio
from types import SimpleNamespace

from sgl_jax.test.test_utils import (
    DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    CustomTestCase,
    popen_launch_server,
    kill_process_tree,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
)
from sgl_jax.bench_score import run_benchmark

class TestBenchScore(CustomTestCase):
    def test_score_throughput(self):
        model = DEEPSEEK_R1_DISTILL_QWEN_1_5B
        base_url = DEFAULT_URL_FOR_TEST
        
        # Launch server
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            # Use configurations suitable for v5e or generic
            other_args=[
                "--precompile-token-paddings", "2048", 
                "--precompile-bs-paddings", "1,32,128", 
                "--page-size", "128",
                "--mem-fraction-static", "0.8",
                "--chunked-prefill-size", "32768",
            ],
            check_cache_miss=False
        )

        try:
            # Run benchmark
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                num_prompts=100,
                query_len=128,
                mode="classification",
                num_items=2,
                item_len=1,
                max_concurrency=128
            )
            
            print(f"Running benchmark on {model}...")
            res = asyncio.run(run_benchmark(args))
            print(f"Benchmark Result: {res}")
            
            self.assertGreater(res["throughput"], 0)
            
        finally:
            kill_process_tree(process.pid)

if __name__ == "__main__":
    unittest.main()
