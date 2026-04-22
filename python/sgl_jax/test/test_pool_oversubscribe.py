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

from sgl_jax.srt.managers.scheduler_scoring_mixin import SchedulerScoringMixin


class _FakeSchedulerScoreFromCacheV2(SchedulerScoringMixin):

    def __init__(self):
        self.enable_overlap = False
        self.server_args = SimpleNamespace(
            max_running_requests=24,
            score_v2_allow_reqpool_oversubscribe=False,
        )
        self.req_to_token_pool = SimpleNamespace(available_size=lambda: 25)
        self.chunk_calls = []

    def _run_score_from_cache_v2_chunk(self, **kwargs):
        chunk_items = kwargs.get("chunk_items", [])
        self.chunk_calls.append(chunk_items)
        # Return dummy scores and timing
        return (
            [[0.1] * len(kwargs.get("label_token_ids", [])) for _ in chunk_items],
            0.1,
            0.05,
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


class TestPoolOversubscribe(unittest.TestCase):

    def test_score_from_cache_v2_reqpool_oversubscribe_flag_uses_available_slots(
        self,
    ):
        scheduler = _FakeSchedulerScoreFromCacheV2()
        scheduler.server_args.max_running_requests = 24
        scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 25)

        # Enable oversubscribe!
        scheduler.server_args.score_v2_allow_reqpool_oversubscribe = True

        items = [[i] * 20 for i in range(50)]
        out = scheduler.score_from_cache_v2(
            ScoreFromCacheReqInput(
                cache_handle="cache-ok",
                items_2d=items,
                label_token_ids=[9454, 2753],
                items_per_step=64,
                apply_softmax=False,
            )
        )

        self.assertEqual(out.dispatch_count, 2)
        self.assertEqual([len(chunk) for chunk in scheduler.chunk_calls], [25, 25])


if __name__ == "__main__":
    unittest.main()
