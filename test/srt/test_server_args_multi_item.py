import argparse
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.scheduler_scoring_state_mixin import SchedulerScoringStateMixin
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.model_executor.model_runner import _unpack_prefill_body_only_outputs
from sgl_jax.srt.server_args import ServerArgs


def test_multi_item_core_args_parse():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--max-multi-item-count",
            "128",
            "--multi-item-enable-prefill-extend",
            "--multi-item-extend-batch-size",
            "16",
            "--enable-scoring-cache",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.max_multi_item_count == 128
    assert server_args.multi_item_enable_prefill_extend is True
    assert server_args.multi_item_extend_batch_size == 16
    assert server_args.enable_scoring_cache is True


def test_prefill_cache_body_only_args_parse():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--enable-scoring-cache",
            "--disable-overlap-schedule",
            "--multi-item-enable-prefill-extend",
            "--multi-item-score-prefill-cache-body-only",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_score_prefill_cache_body_only is True
    assert server_args.disable_overlap_schedule is True


def test_reusable_prefill_cache_args_parse():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--enable-scoring-cache",
            "--multi-item-enable-prefill-extend",
            "--multi-item-score-reuse-prefill-cache-by-prefix",
            "--multi-item-score-reuse-prefill-cache-ttl",
            "12.5",
            "--multi-item-score-reuse-prefill-cache-max-entries",
            "32",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_score_reuse_prefill_cache_by_prefix is True
    assert server_args.multi_item_score_reuse_prefill_cache_ttl == 12.5
    assert server_args.multi_item_score_reuse_prefill_cache_max_entries == 32


def test_prefill_extend_requires_scoring_cache():
    server_args = ServerArgs(
        model_path="dummy-model",
        multi_item_enable_prefill_extend=True,
        enable_scoring_cache=False,
    )
    with pytest.raises(AssertionError, match="scoring cache"):
        server_args.check_server_args()


def test_prefill_extend_allows_scoring_cache():
    server_args = ServerArgs(
        model_path="dummy-model",
        multi_item_enable_prefill_extend=True,
        enable_scoring_cache=True,
    )
    server_args.check_server_args()


def test_score_from_cache_v2_args_parse():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--enable-scoring-cache",
            "--multi-item-enable-prefill-extend",
            "--multi-item-enable-score-from-cache-v2",
            "--multi-item-score-from-cache-v2-items-per-step",
            "32",
            "--multi-item-score-from-cache-v2-adaptive-chunk-by-token-budget",
            "--multi-item-score-from-cache-v2-token-budget",
            "4096",
            "--multi-item-score-from-cache-v2-min-items-per-step",
            "4",
            "--multi-item-score-fastpath-log-metrics",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_enable_score_from_cache_v2 is True
    assert server_args.multi_item_score_from_cache_v2_items_per_step == 32
    assert server_args.multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget is True
    assert server_args.multi_item_score_from_cache_v2_token_budget == 4096
    assert server_args.multi_item_score_from_cache_v2_min_items_per_step == 4
    assert server_args.multi_item_score_fastpath_log_metrics is True


def test_score_from_cache_v2_requires_prefill_extend():
    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=True,
        multi_item_enable_score_from_cache_v2=True,
        multi_item_enable_prefill_extend=False,
    )
    with pytest.raises(AssertionError, match="prefill\\+extend"):
        server_args.check_server_args()


def test_adaptive_score_from_cache_v2_requires_token_budget():
    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=True,
        multi_item_enable_prefill_extend=True,
        multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget=True,
        multi_item_score_from_cache_v2_token_budget=0,
    )
    with pytest.raises(AssertionError, match="token-budget"):
        server_args.check_server_args()


def test_prefill_cache_body_only_requires_non_overlap():
    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=True,
        multi_item_enable_prefill_extend=True,
        multi_item_score_prefill_cache_body_only=True,
        disable_overlap_schedule=False,
    )
    with pytest.raises(AssertionError, match="non-overlap scheduling"):
        server_args.check_server_args()


def test_prefill_cache_body_only_requires_prefill_extend():
    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=True,
        multi_item_enable_prefill_extend=False,
        multi_item_score_prefill_cache_body_only=True,
        disable_overlap_schedule=True,
    )
    with pytest.raises(AssertionError, match=r"prefill\+extend"):
        server_args.check_server_args()


def test_prefill_cache_body_only_skip_logits_gate():
    owner = SimpleNamespace(
        server_args=SimpleNamespace(multi_item_score_prefill_cache_body_only=True)
    )
    batch = SimpleNamespace(
        is_prefill_only=True,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        return_output_logprob_only=False,
        reqs=[SimpleNamespace(cache_for_scoring=True)],
    )
    assert SchedulerScoringStateMixin._can_skip_logits_for_prefill_batch(owner, batch)

    batch.return_logprob = True
    assert not SchedulerScoringStateMixin._can_skip_logits_for_prefill_batch(owner, batch)


def test_prefill_cache_body_only_skip_plan_gates_sampler_skip_to_flag():
    batch = SimpleNamespace(
        is_prefill_only=True,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        return_output_logprob_only=False,
        reqs=[SimpleNamespace(cache_for_scoring=True)],
    )

    owner = SimpleNamespace(
        server_args=SimpleNamespace(multi_item_score_prefill_cache_body_only=False)
    )
    assert SchedulerScoringStateMixin._can_skip_sample_for_prefill_batch(batch)
    assert SchedulerScoringStateMixin._score_prefill_cache_skip_plan(owner, batch) == (
        False,
        False,
    )

    owner.server_args.multi_item_score_prefill_cache_body_only = True
    assert SchedulerScoringStateMixin._score_prefill_cache_skip_plan(owner, batch) == (
        True,
        True,
    )


def test_prefill_body_only_unpack_handles_common_model_shapes():
    hidden = object()
    kv = object()
    callbacks = object()
    topk = object()

    assert _unpack_prefill_body_only_outputs(
        (hidden, kv), body_returns_topk_ids=False
    ) == (kv, None)
    assert _unpack_prefill_body_only_outputs(
        (hidden, kv, callbacks), body_returns_topk_ids=False
    ) == (kv, None)
    assert _unpack_prefill_body_only_outputs(
        (hidden, kv, topk), body_returns_topk_ids=True
    ) == (kv, topk)
    assert _unpack_prefill_body_only_outputs(
        (hidden, object(), kv, callbacks), body_returns_topk_ids=False
    ) == (kv, None)


def test_prefill_body_only_unpack_rejects_unknown_shape():
    with pytest.raises(ValueError, match="expected model body to return 2, 3, or 4"):
        _unpack_prefill_body_only_outputs((object(),), body_returns_topk_ids=False)


def test_reusable_prefill_cache_requires_prefill_extend():
    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=True,
        multi_item_score_reuse_prefill_cache_by_prefix=True,
        multi_item_enable_prefill_extend=False,
    )
    with pytest.raises(AssertionError, match=r"prefill\+extend"):
        server_args.check_server_args()


def test_reusable_prefill_cache_requires_scoring_cache():
    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=False,
        multi_item_enable_prefill_extend=True,
        multi_item_score_reuse_prefill_cache_by_prefix=True,
    )
    with pytest.raises(AssertionError, match="scoring cache"):
        server_args.check_server_args()


def test_reusable_prefill_cache_args_validation():
    server_args = ServerArgs(
        model_path="dummy-model",
        multi_item_score_reuse_prefill_cache_ttl=-1.0,
    )
    with pytest.raises(AssertionError, match="ttl must be non-negative"):
        server_args.check_server_args()

    server_args = ServerArgs(
        model_path="dummy-model",
        multi_item_score_reuse_prefill_cache_max_entries=0,
    )
    with pytest.raises(AssertionError, match="max-entries must be positive"):
        server_args.check_server_args()

    server_args = ServerArgs(
        model_path="dummy-model",
        enable_scoring_cache=True,
        multi_item_enable_prefill_extend=True,
        multi_item_score_reuse_prefill_cache_by_prefix=True,
        multi_item_prefill_extend_cache_timeout=1.0,
        multi_item_score_reuse_prefill_cache_ttl=2.0,
    )
    with pytest.raises(AssertionError, match="must not exceed"):
        server_args.check_server_args()
