import unittest
from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, Req
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class TestReqClass(unittest.TestCase):
    def test_req_init(self):
        """Test that Req initializes all fields correctly."""
        sampling_params = SamplingParams(max_new_tokens=10)
        req = Req(
            rid="test_rid",
            origin_input_text="test text",
            origin_input_ids=[1, 2, 3],
            sampling_params=sampling_params,
            return_logprob=True,
            stream=True,
            vocab_size=32000,
        )

        self.assertEqual(req.rid, "test_rid")
        self.assertEqual(req.origin_input_text, "test text")
        self.assertEqual(req.origin_input_ids, [1, 2, 3])
        self.assertEqual(req.sampling_params, sampling_params)
        self.assertEqual(req.return_logprob, True)
        self.assertEqual(req.stream, True)
        self.assertEqual(req.vocab_size, 32000)
        self.assertEqual(req.output_ids, [])
        self.assertEqual(req.fill_ids, [])

    def test_req_init_next_round_input_no_cache(self):
        """Test init_next_round_input without tree_cache."""
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        req.output_ids = [3, 4]

        req.init_next_round_input(tree_cache=None)

        self.assertEqual(req.fill_ids, [1, 2, 3, 4])

    def test_req_init_next_round_input_with_extend_cache(self):
        """Test init_next_round_input with extend_from_cache."""
        cached_last_node = SimpleNamespace()
        cached_prefix_indices = [0, 1]
        cached_last_host_node = SimpleNamespace()

        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
            extend_from_cache="test_handle",
        )
        req.cached_last_node = cached_last_node
        req.cached_prefix_indices = cached_prefix_indices
        req.cached_last_host_node = cached_last_host_node
        req.cached_host_hit_length = 5

        tree_cache = SimpleNamespace()  # Should not be called

        req.init_next_round_input(tree_cache=tree_cache)

        self.assertEqual(req.prefix_indices, cached_prefix_indices)
        self.assertEqual(req.last_node, cached_last_node)
        self.assertEqual(req.last_host_node, cached_last_host_node)
        self.assertEqual(req.host_hit_length, 5)

    def test_req_init_next_round_input_with_cache_miss(self):
        """Test init_next_round_input falling through to match_prefix."""
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
        )

        dummy_node = SimpleNamespace()
        dummy_host_node = SimpleNamespace()

        class MockTreeCache:
            def match_prefix(self, key):
                return ([0, 1], dummy_node, dummy_host_node, 2)

        tree_cache = MockTreeCache()

        # We need to mock adjust_max_prefix_ids because it might use self.fill_ids or similar
        req.adjust_max_prefix_ids = lambda: [1, 2]

        req.init_next_round_input(tree_cache=tree_cache)

        self.assertEqual(req.prefix_indices, [0, 1])
        self.assertEqual(req.last_node, dummy_node)
        self.assertEqual(req.last_host_node, dummy_host_node)
        self.assertEqual(req.host_hit_length, 2)

    def test_req_finished(self):
        """Test finished() method."""
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        self.assertFalse(req.finished())

        req.finished_reason = "STOP"
        self.assertTrue(req.finished())

    def test_req_seqlen(self):
        """Test seqlen property."""
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        req.output_ids = [3, 4]
        self.assertEqual(req.seqlen, 4)

    def test_req_adjust_max_prefix_ids(self):
        """Test adjust_max_prefix_ids."""
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(),
        )
        self.assertEqual(req.adjust_max_prefix_ids(), [1, 2, 3])

    def test_req_init_incremental_detokenize(self):
        """Test init_incremental_detokenize."""
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        req.init_incremental_detokenize()
        self.assertIsNotNone(req.surr_offset)
        self.assertIsNotNone(req.read_offset)

    def test_req_check_finished_max_tokens(self):
        """Test check_finished with max tokens."""
        sampling_params = SamplingParams(max_new_tokens=2)
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=sampling_params,
        )
        req.output_ids = [3, 4]
        req.check_finished()
        self.assertTrue(req.finished())

    def test_req_reset_for_retract(self):
        """Test reset_for_retract."""
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        req.output_ids = [3, 4]
        req.finished_reason = "STOP"
        req.reset_for_retract()
        self.assertTrue(req.is_retracted)

    def test_req_set_finish_with_abort(self):
        """Test set_finish_with_abort."""
        req = Req(
            rid="test_rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        req.set_finish_with_abort("User aborted")
        self.assertIsNotNone(req.to_finish)


class TestModelWorkerBatch(unittest.TestCase):

    def test_batch_init(self):
        """Test that ModelWorkerBatch initializes correctly."""
        batch = ModelWorkerBatch(
            bid=1,
            forward_mode=ForwardMode.EXTEND,
            input_ids=np.array([1, 2]),
            real_input_ids_len=2,
            seq_lens=np.array([2]),
            out_cache_loc=np.array([0, 1]),
            req_pool_indices=np.array([0]),
            sampling_info=SimpleNamespace(),
            positions=np.array([0, 1]),
            cache_loc=np.array([0, 1]),
            return_logprob=False,
            return_output_logprob_only=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            real_bs=1,
        )
        self.assertEqual(batch.bid, 1)
        self.assertEqual(batch.real_bs, 1)


if __name__ == "__main__":
    unittest.main()
