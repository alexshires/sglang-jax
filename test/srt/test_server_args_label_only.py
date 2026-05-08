import argparse
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.scheduler_scoring_direct_mixin import SchedulerScoringDirectMixin
from sgl_jax.srt.server_args import ServerArgs


class _DummyDirectWarmupScheduler(SchedulerScoringDirectMixin):
    def _score_from_cache_v2_use_direct_label_only(self, *, label_only_logprob: bool) -> bool:
        return bool(label_only_logprob)


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

    scheduler._materialize_score_direct_warmup_prefix = _fail_materialize

    scheduler._run_score_direct_label_only_warmup()
