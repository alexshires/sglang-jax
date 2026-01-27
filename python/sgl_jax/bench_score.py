"""
Benchmark online serving for Scoring API.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    error: str = ""

async def async_request_score(
    api_url: str,
    payload: dict,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession() as session:
        output = RequestFuncOutput()
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    await response.read() # Read body to complete request
                    output.success = True
                    output.latency = time.perf_counter() - st
                else:
                    output.error = f"{response.reason}: {await response.text()}"
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
        
        if pbar:
            pbar.update(1)
        return output

async def run_benchmark(args):
    print(f"Benchmarking with {args.num_prompts} requests...")
    api_url = f"{args.base_url}/v1/score"
    print(f"Target API URL: {api_url}")
    
    # Generate requests
    requests = []
    # ... (rest of generation)
    for _ in range(args.num_prompts):
        # random query
        query_len = args.query_len
        # naive way: just use repeating token
        query_ids = [100] * query_len 
        
        if args.mode == "classification":
             # use label_token_ids
             payload = {
                 "model": args.model,
                 "query": query_ids,
                 "label_token_ids": [100] + [101] * (args.num_items - 1) if args.num_items > 0 else [100],
                 "items": [[]],
                 "apply_softmax": True,
             }
        else:
             # use items
             num_items = args.num_items
             item_len = args.item_len
             items = [[101] * item_len for _ in range(num_items)]
             payload = {
                 "model": args.model,
                 "query": query_ids,
                 "items": items,
             }
        requests.append(payload)

    if requests:
        print(f"Sample Payload (0): {json.dumps(requests[0])}")
        
        # Warmup
        print("Warmup: Sending 1 request to trigger compilation...")
        try:
             async with aiohttp.ClientSession() as session:
                async with session.post(url=api_url, json=requests[0]) as response:
                    await response.text()
        except Exception as e:
            print(f"Warmup failed (ignored): {e}")
        print("Warmup done.")

    pbar = tqdm(total=args.num_prompts)
    tasks = []
    
    # Simple semaphore for concurrency
    sem = asyncio.Semaphore(args.max_concurrency)
    
    async def bound_req(req):
        async with sem:
            return await async_request_score(api_url, req, pbar)
            
    start_time = time.perf_counter()
    tasks = [asyncio.create_task(bound_req(req)) for req in requests]
    outputs = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time
    pbar.close()
    
    # Calculate metrics
    latencies = [out.latency for out in outputs if out.success]
    success_count = len(latencies)
    
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {success_count / total_time:.2f} requests/s")
    if latencies:
        print(f"Avg Latency: {np.mean(latencies)*1000:.2f} ms")
        print(f"P50 Latency: {np.percentile(latencies, 50)*1000:.2f} ms")
        print(f"P99 Latency: {np.percentile(latencies, 99)*1000:.2f} ms")
    else:
        print("No successful requests.")
        failed_outputs = [out for out in outputs if not out.success]
        if failed_outputs:
             print(f"First error: {failed_outputs[0].error}")
             print(f"Total failures: {len(failed_outputs)}")

    return {
        "throughput": success_count / total_time if total_time > 0 else 0,
        "avg_latency": np.mean(latencies) if latencies else 0,
        "p50_latency": np.percentile(latencies, 50) if latencies else 0,
        "p99_latency": np.percentile(latencies, 99) if latencies else 0,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--query-len", type=int, default=128)
    parser.add_argument("--mode", type=str, default="classification", choices=["classification", "items"])
    parser.add_argument("--num-items", type=int, default=2)
    parser.add_argument("--item-len", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=64)
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args))
