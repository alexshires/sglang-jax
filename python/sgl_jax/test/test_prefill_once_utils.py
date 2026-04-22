import unittest
from types import SimpleNamespace

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.scheduler_scoring_mixin import SchedulerScoringMixin
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class DummyScheduler(SchedulerScoringMixin):
    def __init__(self):
        self.model_config = SimpleNamespace(hf_eos_token_id=[2], vocab_size=32000)
        self.tokenizer = SimpleNamespace()
        self.max_req_input_len = 4096
        self.server_args = SimpleNamespace(allow_auto_truncate=False)
        self.tree_cache = SimpleNamespace(
            match_prefix=lambda key: ([0], SimpleNamespace(), SimpleNamespace(), 0)
        )


class TestPrefillOnceUtils(unittest.TestCase):
    def test_build_score_from_cache_v2_chunk_reqs(self):
        scheduler = DummyScheduler()

        cache_handle = "test_cache_handle"
        chunk_items = [[3, 4], [5, 6]]
        label_token_ids = [7, 8]
        cached_last_node = SimpleNamespace()
        cached_prefix_indices = [0, 1]
        prefix_ids = [1, 2]
        cached_extra_key = "extra_key"
        return_label_logprobs = True

        reqs = scheduler._build_score_from_cache_v2_chunk_reqs(
            cache_handle=cache_handle,
            chunk_items=chunk_items,
            label_token_ids=label_token_ids,
            cached_last_node=cached_last_node,
            cached_prefix_indices=cached_prefix_indices,
            prefix_ids=prefix_ids,
            cached_extra_key=cached_extra_key,
            return_label_logprobs=return_label_logprobs,
        )

        self.assertEqual(len(reqs), 2)

        # Check first request
        req0 = reqs[0]
        self.assertEqual(req0.origin_input_ids, [1, 2, 3, 4])
        self.assertEqual(req0.return_logprob, True)
        self.assertEqual(req0.token_ids_logprob, [7, 8])
        print(f"DEBUG: req0.extend_from_cache = {req0.extend_from_cache}")
        print(f"DEBUG: req0 attributes = {dir(req0)}")
        self.assertEqual(req0.extend_from_cache, cache_handle)
        self.assertEqual(req0.cached_last_node, cached_last_node)
        self.assertEqual(req0.cached_prefix_indices, [0, 1])
        self.assertEqual(req0.logprob_start_len, 3)  # len([1,2,3,4]) - 1

        # Check second request
        req1 = reqs[1]
        self.assertEqual(req1.origin_input_ids, [1, 2, 5, 6])
        self.assertEqual(req1.extend_from_cache, cache_handle)
        self.assertEqual(req1.logprob_start_len, 3)

    def test_req_extend_from_cache(self):
        req = Req(
            rid="test",
            origin_input_text="text",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
            extend_from_cache="test_cache_handle",
        )
        print(f"DEBUG in simple test: req.extend_from_cache = {req.extend_from_cache}")
        self.assertEqual(req.extend_from_cache, "test_cache_handle")


if __name__ == "__main__":
    unittest.main()
