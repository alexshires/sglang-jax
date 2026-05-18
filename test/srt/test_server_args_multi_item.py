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
