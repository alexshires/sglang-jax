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

import dataclasses
import unittest
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqOutput
from sgl_jax.srt.managers.scheduler_scoring_mixin import SchedulerScoringMixin


class _FakeSchedulerScoreFromCacheV2(SchedulerScoringMixin):

    def __init__(self):
        self.enable_overlap = False
        self.server_args = SimpleNamespace(
            max_running_requests=0,
            score_v2_allow_reqpool_oversubscribe=False,
        )
        self.scoring_cache_nodes = {"cache-ok": ("node", "swa", [], [], "key", 0.0)}
        self.scoring_cache_handles_created = 1
        self.scoring_cache_handles_released = 0
        self.scoring_cache_handles_released_manual = 0
        self.scoring_cache_handles_released_expired = 0

        self.score_from_cache_v2_attempted = 0
        self.score_from_cache_v2_succeeded = 0
        self.score_from_cache_v2_fallback = 0
        self.score_from_cache_v2_fallback_reasons = {}
        self.score_from_cache_v2_queue_wait_s_total = 0.0
        self.score_from_cache_v2_device_compute_s_total = 0.0
        self.score_from_cache_v2_host_orchestration_s_total = 0.0
        self.score_from_cache_v2_queue_wait_s_max = 0.0
        self.score_from_cache_v2_device_compute_s_max = 0.0
        self.score_from_cache_v2_host_orchestration_s_max = 0.0

    def _run_score_from_cache_v2_chunk(self, **kwargs):
        chunk_items = kwargs.get("chunk_items", [])
        # Return dummy scores and timing
        return [[0.1] * len(kwargs.get("label_token_ids", [])) for _ in chunk_items], 0.01, 0.02

    def _touch_scoring_cache_entry(self, rid):
        pass

    def score_from_cache_v2(self, recv_req):
        # Simulate lookup!
        hit = recv_req.cache_handle == "cache-ok"
        self._record_scoring_cache_lookup(path="score_from_cache_v2", hit=hit)
        
        if not hit:
            self._record_score_from_cache_v2_fallback("missing_cache_handle")
            return ScoreFromCacheReqOutput(rid=recv_req.rid, success=False, fallback_reason="missing_cache_handle")
            
        # Simulate timing recording (as in scheduler.py)
        self._record_score_from_cache_v2_timing(attempted=1, succeeded=1, queue_wait_s=0.0, device_compute_s=0.01, host_orchestration_s=0.02)
        
        return super().score_from_cache_v2(recv_req)

    def _record_score_from_cache_v2_fallback(self, reason):
        self.score_from_cache_v2_attempted += 1
        self.score_from_cache_v2_fallback += 1
        self.score_from_cache_v2_fallback_reasons[reason] = (
            self.score_from_cache_v2_fallback_reasons.get(reason, 0) + 1
        )

    def _record_score_from_cache_v2_timing(
        self, attempted, succeeded, queue_wait_s, device_compute_s, host_orchestration_s
    ):
        self.score_from_cache_v2_attempted += attempted
        self.score_from_cache_v2_succeeded += succeeded
        self.score_from_cache_v2_queue_wait_s_total += queue_wait_s
        self.score_from_cache_v2_device_compute_s_total += device_compute_s
        self.score_from_cache_v2_host_orchestration_s_total += host_orchestration_s
        self.score_from_cache_v2_queue_wait_s_max = max(
            self.score_from_cache_v2_queue_wait_s_max, queue_wait_s
        )
        self.score_from_cache_v2_device_compute_s_max = max(
            self.score_from_cache_v2_device_compute_s_max, device_compute_s
        )
        self.score_from_cache_v2_host_orchestration_s_max = max(
            self.score_from_cache_v2_host_orchestration_s_max, host_orchestration_s
        )


@dataclasses.dataclass
class ScoreFromCacheReqInput:
    cache_handle: str
    items_2d: list[list[int]]
    label_token_ids: list[int]
    items_per_step: int
    apply_softmax: bool
    prefix_ids: list = dataclasses.field(default_factory=list)
    prefix_indices: list = dataclasses.field(default_factory=list)
    cached_last_node: object = None
    cached_extra_key: object = None
    rid: str = "test-rid"


def _parity_metrics(
    baseline_scores: list[list[float]],
    fastpath_scores: list[list[float]],
) -> tuple[float, float]:
    diffs = []
    for base_row, fast_row in zip(baseline_scores, fastpath_scores):
        diffs.extend(abs(a - b) for a, b in zip(base_row, fast_row))
    return max(diffs), sum(diffs) / len(diffs)


class TestDebugInstrumentation(unittest.TestCase):

    def test_score_from_cache_v2_updates_scoring_cache_lookup_counters_on_hit(self):
        scheduler = _FakeSchedulerScoreFromCacheV2()
        out = scheduler.score_from_cache_v2(
            ScoreFromCacheReqInput(
                cache_handle="cache-ok",
                items_2d=[[1] * 20 for _ in range(4)],
                label_token_ids=[9454, 2753],
                items_per_step=4,
                apply_softmax=False,
            )
        )

        assert out.success is True
        metrics = scheduler._scoring_cache_metrics_snapshot()
        assert metrics["lookup_queries"] == 1
        assert metrics["lookup_hits"] == 1
        assert metrics["lookup_misses"] == 0
        assert metrics["lookup_by_path"]["score_from_cache_v2"]["queries"] == 1
        assert metrics["lookup_by_path"]["score_from_cache_v2"]["hits"] == 1
        assert metrics["lookup_hit_rate"] == 1.0

    def test_score_from_cache_v2_updates_scoring_cache_lookup_counters_on_miss(self):
        scheduler = _FakeSchedulerScoreFromCacheV2()
        out = scheduler.score_from_cache_v2(
            ScoreFromCacheReqInput(
                cache_handle="cache-missing",
                items_2d=[[1] * 20 for _ in range(4)],
                label_token_ids=[9454, 2753],
                items_per_step=4,
                apply_softmax=False,
            )
        )

        assert out.success is False
        assert out.fallback_reason == "missing_cache_handle"
        metrics = scheduler._scoring_cache_metrics_snapshot()
        assert metrics["lookup_queries"] == 1
        assert metrics["lookup_hits"] == 0
        assert metrics["lookup_misses"] == 1
        assert metrics["lookup_by_path"]["score_from_cache_v2"]["queries"] == 1
        assert metrics["lookup_by_path"]["score_from_cache_v2"]["misses"] == 1
        assert metrics["lookup_hit_rate"] == 0.0

    def test_score_from_cache_v2_timing_counters_are_recorded(self):
        scheduler = _FakeSchedulerScoreFromCacheV2()
        out = scheduler.score_from_cache_v2(
            ScoreFromCacheReqInput(
                cache_handle="cache-ok",
                items_2d=[[1] * 20 for _ in range(4)],
                label_token_ids=[9454, 2753],
                items_per_step=4,
                apply_softmax=False,
            )
        )

        assert out.success is True
        assert scheduler.score_from_cache_v2_attempted == 1
        assert scheduler.score_from_cache_v2_succeeded == 1
        assert scheduler.score_from_cache_v2_queue_wait_s_total >= 0.0
        assert scheduler.score_from_cache_v2_device_compute_s_total == pytest.approx(0.01)
        assert scheduler.score_from_cache_v2_host_orchestration_s_total == pytest.approx(0.02)
        assert scheduler.score_from_cache_v2_device_compute_s_max == pytest.approx(0.01)
        assert scheduler.score_from_cache_v2_host_orchestration_s_max == pytest.approx(0.02)

        before_queue_wait = scheduler.score_from_cache_v2_queue_wait_s_total
        miss_out = scheduler.score_from_cache_v2(
            ScoreFromCacheReqInput(
                cache_handle="cache-missing",
                items_2d=[[1] * 20 for _ in range(4)],
                label_token_ids=[9454, 2753],
                items_per_step=4,
                apply_softmax=False,
            )
        )

        assert miss_out.success is False
        assert scheduler.score_from_cache_v2_attempted == 2
        assert scheduler.score_from_cache_v2_fallback == 1
        assert scheduler.score_from_cache_v2_queue_wait_s_total >= before_queue_wait
        # Missing-cache fallback records zero compute overhead.
        assert scheduler.score_from_cache_v2_device_compute_s_total == pytest.approx(0.01)
        assert scheduler.score_from_cache_v2_host_orchestration_s_total == pytest.approx(0.02)

    def test_score_from_cache_v2_parity_metric_threshold(self):
        baseline_scores = [[0.1, 0.9], [0.3, 0.7], [0.8, 0.2]]
        fastpath_scores = [[0.1000004, 0.8999996], [0.3000001, 0.6999999], [0.8, 0.2]]
        max_abs_diff, mean_abs_diff = _parity_metrics(baseline_scores, fastpath_scores)
        assert max_abs_diff < 1e-3
        assert mean_abs_diff < 5e-4


if __name__ == "__main__":
    unittest.main()
