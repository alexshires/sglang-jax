import argparse

import pytest

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
