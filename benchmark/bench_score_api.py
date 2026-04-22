# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import logging
import os
import time
import unittest
from typing import Any

import requests

# Configuration
API_URL = os.getenv("SGLANG_API_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("SGLANG_MODEL_NAME", "qwen3-0-6b")
LABEL_TOKEN_IDS = [9693, 2152]  # Default label IDs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dummy_text(target_tokens):
    """Generates a string of approximately `target_tokens` length."""
    word = "test "
    multiplier = max(1, target_tokens)
    return (word * multiplier).strip()


def snapshot_server_info(base_url: str) -> dict[str, Any]:
    try:
        resp = requests.post(f"{base_url}/v1/get_internal_state", json={}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"Failed to snapshot server info: {e}")
    return {}


def diff_nested(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    diff = {}
    for k, v in after.items():
        if k in before:
            if isinstance(v, (int, float)) and isinstance(before[k], (int, float)):
                diff[k] = v - before[k]
            elif isinstance(v, dict) and isinstance(before[k], dict):
                sub_diff = diff_nested(v, before[k])
                if sub_diff:
                    diff[k] = sub_diff
        else:
            diff[k] = v
    return diff


class TestScoreAPIBench(unittest.TestCase):

    def _run_http_bench(
        self,
        name: str,
        query_len: int,
        num_items: int,
        item_len: int,
        num_requests: int = 20,
        concurrency: int = 2,
    ):
        logger.info(f"\n🚀 Starting Benchmark: {name} ({num_requests} requests, {num_items} items)")

        query_text = generate_dummy_text(query_len)
        items_list = [f"{i}_{generate_dummy_text(item_len)}" for i in range(num_items)]

        payload = {
            "query": query_text,
            "items": items_list,
            "label_token_ids": LABEL_TOKEN_IDS,
            "model": MODEL_NAME,
            "apply_softmax": True,
        }

        # Warm-up
        logger.info("🔥 Warming up...")
        try:
            resp = requests.post(f"{API_URL}/v1/score", json=payload, timeout=300)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
            self.fail(f"Warm-up failed: {e}")

        before_info = snapshot_server_info(API_URL)
        start_time = time.perf_counter()
        successful_requests = 0

        def send_request():
            try:
                resp = requests.post(f"{API_URL}/v1/score", json=payload, timeout=300)
                resp.raise_for_status()
                return True
            except Exception as e:
                logger.debug(f"Request failed: {e}")
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request) for _ in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful_requests += 1

        total_time = time.perf_counter() - start_time
        after_info = snapshot_server_info(API_URL)

        # Telemetry
        telemetry = ""
        try:
            counter_delta = diff_nested(after_info, before_info)
            metrics = after_info.get("score_from_cache_v2_metrics", {})
            totals = metrics.get("timing_totals_s", {})
            h_time = totals.get("host_orchestration", 0.0)
            d_time = totals.get("device_compute", 0.0)
            if (h_time + d_time) > 0:
                telemetry = f"HostOverhead: {(h_time / (h_time + d_time)) * 100:.1f}%"

            cache_metrics = counter_delta.get("scoring_cache_metrics", {})
            hits = cache_metrics.get("cache_hits", 0.0)
            misses = cache_metrics.get("cache_misses", 0.0)
            if (hits + misses) > 0:
                hit_rate = (hits / (hits + misses)) * 100
                telemetry += f" | CacheHitRate: {hit_rate:.1f}% ({int(hits)}/{int(hits+misses)})"
        except Exception as e:
            logger.debug(f"Telemetry extraction failed: {e}")

        rps = successful_requests / total_time
        items_per_second = (successful_requests * num_items) / total_time

        logger.info("\n" + "=" * 40)
        logger.info(f"📊 RESULTS for {name}")
        logger.info("=" * 40)
        logger.info(f"Total Time:       {total_time:.4f} s")
        logger.info(f"Success Rate:     {successful_requests}/{num_requests}")
        logger.info(f"RPS:              {rps:.2f}")
        logger.info(f"IPS:              {items_per_second:.2f}")
        logger.info(f"Telemetry:        {telemetry}")
        logger.info("=" * 40)

        self.assertGreater(successful_requests, 0, "All requests failed!")

    def test_scan_p250_n10_i10(self):
        """Small Baseline."""
        self._run_http_bench("scan_p250_n10_i10", 250, 10, 10)

    def test_scan_p250_n500_i10(self):
        """High Fanout."""
        self._run_http_bench("scan_p250_n500_i10", 250, 500, 10)

    def test_scan_p2000_n10_i80(self):
        """Long Context."""
        self._run_http_bench("scan_p2000_n10_i80", 2000, 10, 80)

    def test_scan_p1900_n500_i10(self):
        """Large Prompt, High Fanout (Was Scenario 1)."""
        self._run_http_bench("scan_p1900_n500_i10", 1900, 500, 10)

    def test_scan_p2000_n500_i20(self):
        """Large Prompt, High Fanout, Medium Items (Was Scenario 2)."""
        self._run_http_bench("scan_p2000_n500_i20", 2000, 500, 20)

    def test_scan_p1000_n100_i26(self):
        """Medium Scan 1."""
        self._run_http_bench("scan_p1000_n100_i26", 1000, 100, 26)

    def test_scan_p1000_n250_i50(self):
        """Medium Scan 2."""
        self._run_http_bench("scan_p1000_n250_i50", 1000, 250, 50)


if __name__ == "__main__":
    unittest.main()
