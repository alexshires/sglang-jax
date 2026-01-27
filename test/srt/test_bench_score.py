
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
            timeout=600,
            # Use configurations suitable for v5e or generic
            other_args=[
                "--precompile-bs-paddings", "1", "32", "128", "1024",
                "--precompile-token-paddings", "256", "1024", "2048", "4096", "8192", 
                "--chunked-prefill-size", "4096",
                "--page-size", "128",
                "--mem-fraction-static", "0.8",
                "--schedule-conservativeness", "1.0",
                "--max-running-requests", "1024",
                "--enable-mixed-chunk",
                "--context-length", "4096",
                "--disable-radix-cache",
                "--disable-overlap-schedule",
            ],
            check_cache_miss=False
        )

        try:
            # Run benchmark
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                num_prompts=2000,
                query_len=128,
                mode="classification",
                num_items=2,
                item_len=1,
                max_concurrency=1024
            )
            args.no_logprobs = True
            args.enable_flashinfer = False            
            print(f"Running benchmark on {model}...")
            res = asyncio.run(run_benchmark(args))
            print(f"Benchmark Result: {res}")
            
            self.assertGreater(res["throughput"], 0)
            
        finally:
            kill_process_tree(process.pid)

if __name__ == "__main__":
    unittest.main()
