import argparse
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.managers.scheduler_scoring_direct_mixin import (
    SchedulerScoringDirectMixin,
    _compute_label_only_logprobs,
    _compute_label_only_logprobs_log_softmax,
    _compute_label_only_scores_from_logprobs,
    _compute_label_only_scores_fused,
)
from sgl_jax.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sgl_jax.srt.server_args import ServerArgs


class _DummyDirectWarmupScheduler(SchedulerScoringDirectMixin):
    def _score_from_cache_v2_use_direct_label_only(self, *, label_only_logprob: bool) -> bool:
        return bool(label_only_logprob)

    def _score_from_cache_v2_use_direct_token_ids_logprob_only(self) -> bool:
        return False


def test_label_only_direct_score_args_parse():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--multi-item-score-label-only-logprob",
            "--no-multi-item-score-label-only-fused-kernel",
            "--multi-item-score-direct-label-only",
            "--multi-item-score-direct-hot-shape-bs",
            "128",
            "--multi-item-score-direct-hot-shape-tokens",
            "4096",
            "--multi-item-score-direct-token-ids-logprob-only-auto",
            "--multi-item-score-direct-warmup-prefix-len",
            "650",
            "--multi-item-score-direct-warmup-item-len",
            "53",
            "--multi-item-score-direct-warmup-label-count",
            "4",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_score_label_only_logprob is True
    assert server_args.multi_item_score_label_only_fused_kernel is False
    assert server_args.multi_item_score_direct_label_only is True
    assert server_args.multi_item_score_direct_hot_shape_bs == 128
    assert server_args.multi_item_score_direct_hot_shape_tokens == 4096
    assert server_args.multi_item_score_direct_token_ids_logprob_only_auto is True
    assert server_args.multi_item_score_direct_warmup_prefix_len == 650
    assert server_args.multi_item_score_direct_warmup_item_len == 53
    assert server_args.multi_item_score_direct_warmup_label_count == 4


def test_label_only_fused_kernel_default_true():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_score_label_only_fused_kernel is True


def test_direct_label_only_requires_label_only_logprob():
    server_args = ServerArgs(
        model_path="dummy-model",
        multi_item_score_direct_label_only=True,
        multi_item_score_label_only_logprob=False,
    )
    with pytest.raises(AssertionError, match="label-only logprob"):
        server_args.check_server_args()


def test_direct_warmup_requires_positive_shapes():
    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=True,
        multi_item_enable_prefill_extend=True,
        multi_item_enable_score_from_cache_v2=True,
        multi_item_score_label_only_logprob=True,
        multi_item_score_direct_label_only=True,
        multi_item_score_direct_warmup_enable=True,
        multi_item_score_direct_warmup_prefix_len=0,
        multi_item_score_direct_warmup_item_len=53,
    )
    with pytest.raises(AssertionError, match="warmup-prefix-len"):
        server_args.check_server_args()


def test_direct_warmup_accepts_items_per_step_batch_fallback():
    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=True,
        multi_item_enable_prefill_extend=True,
        multi_item_enable_score_from_cache_v2=True,
        multi_item_score_label_only_logprob=True,
        multi_item_score_direct_label_only=True,
        multi_item_score_direct_warmup_enable=True,
        multi_item_score_direct_warmup_prefix_len=5,
        multi_item_score_direct_warmup_item_len=3,
        multi_item_score_direct_warmup_batch_size=0,
        multi_item_score_direct_hot_shape_bs=0,
        multi_item_score_from_cache_v2_items_per_step=7,
    )
    server_args.check_server_args()


def test_direct_token_ids_logprob_only_chunk_size_must_be_positive():
    server_args = ServerArgs(
        model_path="dummy-model",
        multi_item_score_direct_token_ids_logprob_only_chunk_size=0,
    )
    with pytest.raises(AssertionError, match="chunk-size must be positive"):
        server_args.check_server_args()


def test_direct_warmup_spec_uses_zero_as_zero():
    scheduler = _DummyDirectWarmupScheduler()
    scheduler.server_args = SimpleNamespace(
        multi_item_score_label_only_logprob=True,
        multi_item_score_direct_warmup_enable=True,
        multi_item_score_direct_warmup_batch_size=0,
        multi_item_score_direct_hot_shape_bs=0,
        multi_item_score_from_cache_v2_items_per_step=7,
        multi_item_score_direct_warmup_prefix_len=5,
        multi_item_score_direct_warmup_item_len=3,
        multi_item_score_direct_warmup_label_count=2,
        multi_item_score_direct_warmup_apply_softmax=True,
    )

    spec = scheduler._score_direct_warmup_spec()

    assert spec.batch_size == 7
    assert spec.prefix_len == 5
    assert spec.item_len == 3
    assert spec.label_count == 2
    assert spec.apply_softmax is True


def test_direct_warmup_failure_does_not_raise():
    scheduler = _DummyDirectWarmupScheduler()
    scheduler.server_args = SimpleNamespace(
        multi_item_score_label_only_logprob=True,
        multi_item_score_direct_warmup_enable=True,
        multi_item_score_direct_warmup_batch_size=1,
        multi_item_score_direct_hot_shape_bs=0,
        multi_item_score_from_cache_v2_items_per_step=0,
        multi_item_score_direct_warmup_prefix_len=5,
        multi_item_score_direct_warmup_item_len=3,
        multi_item_score_direct_warmup_label_count=2,
        multi_item_score_direct_warmup_apply_softmax=False,
    )
    scheduler.model_config = SimpleNamespace(vocab_size=128, hf_eos_token_id=set())

    def _fail_materialize(_prefix_ids):
        raise RuntimeError("synthetic warmup failure")

    # Raise before the chunk runner so this test isolates the non-fatal warmup
    # wrapper rather than requiring a fully mocked scheduler host.
    scheduler._materialize_score_direct_warmup_prefix = _fail_materialize

    scheduler._run_score_direct_label_only_warmup()


def test_direct_empty_chunk_returns_empty_score_array():
    scheduler = _DummyDirectWarmupScheduler()
    scheduler.server_args = SimpleNamespace()

    scores, device_s, host_s = scheduler._run_score_from_cache_v2_direct_chunk_label_only(
        cache_handle="empty",
        chunk_items=[],
        label_token_ids=[1, 2],
        label_token_ids_arr=jnp.asarray([1, 2], dtype=jnp.int32),
        apply_softmax=False,
        cached_last_node=None,
        cached_prefix_indices=np.asarray([11, 12], dtype=np.int32),
        prefix_ids=[101, 102],
        cached_extra_key=None,
    )

    assert isinstance(scores, jax.Array)
    assert scores.shape == (0, 2)
    assert device_s == 0.0
    assert host_s == 0.0


def test_direct_freeable_cache_locs_skip_padding_sentinels():
    locs = SchedulerScoringDirectMixin._score_direct_freeable_cache_locs(
        np.asarray([-1, 0, 1, 16], dtype=np.int32)
    )

    np.testing.assert_array_equal(locs, np.asarray([1, 16], dtype=np.int32))


def test_kv_allocators_reserve_zero_slot():
    token_allocator = TokenToKVPoolAllocator(size=8, kvcache=object())
    np.testing.assert_array_equal(token_allocator.alloc(3), np.asarray([1, 2, 3]))

    paged_allocator = PagedTokenToKVPoolAllocator(
        size=64,
        page_size=16,
        kvcache=object(),
    )
    np.testing.assert_array_equal(paged_allocator.alloc(16), np.arange(16, 32))


def test_label_only_kernels_match_numpy_reference():
    logits = jnp.asarray(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.0, -2.0, 3.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    label_ids = jnp.asarray([0, 2, 3], dtype=jnp.int32)
    logits_np = np.asarray(logits, dtype=np.float64)
    label_ids_np = np.asarray(label_ids, dtype=np.int32)
    logsumexp = np.log(np.sum(np.exp(logits_np), axis=-1, keepdims=True))
    label_logprobs_ref = logits_np[:, label_ids_np] - logsumexp
    label_probs_ref = np.exp(label_logprobs_ref)
    row_max = np.max(label_probs_ref, axis=-1, keepdims=True)
    label_softmax_ref = np.exp(label_probs_ref - row_max)
    label_softmax_ref = label_softmax_ref / np.sum(
        label_softmax_ref,
        axis=-1,
        keepdims=True,
    )

    baseline = _compute_label_only_logprobs(logits, label_ids, None)
    log_softmax = _compute_label_only_logprobs_log_softmax(logits, label_ids, None)
    fused_probs = _compute_label_only_scores_fused(logits, label_ids, False, None)
    fused_softmax = _compute_label_only_scores_fused(logits, label_ids, True, None)
    from_logprobs = _compute_label_only_scores_from_logprobs(baseline, False)
    from_logprobs_softmax = _compute_label_only_scores_from_logprobs(baseline, True)

    np.testing.assert_allclose(np.asarray(baseline), label_logprobs_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(log_softmax), label_logprobs_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fused_probs), label_probs_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(from_logprobs), label_probs_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fused_softmax), label_softmax_ref, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(from_logprobs_softmax),
        label_softmax_ref,
        rtol=1e-6,
    )
