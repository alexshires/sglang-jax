import argparse

import pytest

from sgl_jax.srt.server_args import ServerArgs


def test_enable_gc_freeze_flag_parsing():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model", "--enable-gc-freeze"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.enable_gc_freeze is True


def test_enable_gc_freeze_default_false():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.enable_gc_freeze is False


def test_gc_freeze_rollback_flag_parsing():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model", "--gc-freeze-rollback"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.gc_freeze_rollback is True


def test_gc_freeze_rollback_default_false():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.gc_freeze_rollback is False


def test_enable_tokenizer_batch_send_flag_parsing():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model", "--enable-tokenizer-batch-send"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.enable_tokenizer_batch_send is True


def test_enable_tokenizer_batch_send_default_false():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.enable_tokenizer_batch_send is False


def test_score_v2_adaptive_chunk_args_defaults():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget is False
    assert server_args.multi_item_score_from_cache_v2_token_budget == 0
    assert server_args.multi_item_score_from_cache_v2_min_items_per_step == 1


def test_score_v2_adaptive_chunk_args_parsing():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--multi-item-score-from-cache-v2-adaptive-chunk-by-token-budget",
            "--multi-item-score-from-cache-v2-token-budget",
            "32768",
            "--multi-item-score-from-cache-v2-min-items-per-step",
            "8",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget is True
    assert server_args.multi_item_score_from_cache_v2_token_budget == 32768
    assert server_args.multi_item_score_from_cache_v2_min_items_per_step == 8


def test_score_v2_adaptive_chunk_requires_positive_token_budget_when_enabled():
    with pytest.raises(AssertionError):
        server_args = ServerArgs(
            model_path="dummy-model",
            multi_item_scoring_delimiter=151643,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            attention_backend="fa",
            multi_item_enable_prefill_extend=True,
            multi_item_enable_score_from_cache_v2=True,
            enable_scoring_cache=True,
            multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget=True,
            multi_item_score_from_cache_v2_token_budget=0,
        )
        server_args.check_server_args()


def test_score_scheduler_controls_parse():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--score-scheduler-global-microbatch-window-ms",
            "2.5",
            "--score-scheduler-global-microbatch-poll-interval-ms",
            "0.25",
            "--score-scheduler-short-prompt-tokens-threshold",
            "3072",
            "--score-scheduler-short-lane-max-inflight",
            "48",
            "--score-scheduler-long-lane-max-inflight",
            "32",
            "--score-scheduler-enable-lane-isolation",
            "--score-scheduler-lane-isolation-short-burst",
            "3",
            "--score-scheduler-lane-isolation-long-burst",
            "2",
            "--score-scheduler-dynamic-items-per-step-enable",
            "--score-scheduler-dynamic-items-per-step-pressure-threshold",
            "96",
            "--score-scheduler-dynamic-items-per-step-short-lane-bias",
            "0.9",
            "--score-scheduler-dynamic-items-per-step-long-lane-bias",
            "0.6",
            "--score-scheduler-dynamic-items-per-step-short-lane-min",
            "40",
            "--score-scheduler-dynamic-items-per-step-long-lane-min",
            "20",
            "--score-scheduler-cache-admission-bias-enable",
            "--score-scheduler-cache-admission-bias-require-hit",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.score_scheduler_global_microbatch_window_ms == pytest.approx(2.5)
    assert server_args.score_scheduler_global_microbatch_poll_interval_ms == pytest.approx(0.25)
    assert server_args.score_scheduler_short_prompt_tokens_threshold == 3072
    assert server_args.score_scheduler_short_lane_max_inflight == 48
    assert server_args.score_scheduler_long_lane_max_inflight == 32
    assert server_args.score_scheduler_enable_lane_isolation is True
    assert server_args.score_scheduler_lane_isolation_short_burst == 3
    assert server_args.score_scheduler_lane_isolation_long_burst == 2
    assert server_args.score_scheduler_dynamic_items_per_step_enable is True
    assert server_args.score_scheduler_dynamic_items_per_step_pressure_threshold == 96
    assert server_args.score_scheduler_dynamic_items_per_step_short_lane_bias == pytest.approx(0.9)
    assert server_args.score_scheduler_dynamic_items_per_step_long_lane_bias == pytest.approx(0.6)
    assert server_args.score_scheduler_dynamic_items_per_step_short_lane_min == 40
    assert server_args.score_scheduler_dynamic_items_per_step_long_lane_min == 20
    assert server_args.score_scheduler_cache_admission_bias_enable is True
    assert server_args.score_scheduler_cache_admission_bias_require_hit is True


def test_score_scheduler_cache_bias_default_require_hit_true():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--score-scheduler-cache-admission-bias-enable",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.score_scheduler_cache_admission_bias_enable is True
    assert server_args.score_scheduler_cache_admission_bias_require_hit is True


def test_score_scheduler_cache_bias_can_disable_require_hit():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--score-scheduler-cache-admission-bias-enable",
            "--no-score-scheduler-cache-admission-bias-require-hit",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.score_scheduler_cache_admission_bias_enable is True
    assert server_args.score_scheduler_cache_admission_bias_require_hit is False


def test_score_scheduler_controls_reject_negative_window():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--score-scheduler-global-microbatch-window-ms",
            "-1",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    with pytest.raises(AssertionError, match="score-scheduler-global-microbatch-window-ms"):
        server_args.check_server_args()


def test_score_scheduler_controls_reject_non_positive_lane_isolation_burst():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--score-scheduler-lane-isolation-short-burst",
            "0",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    with pytest.raises(AssertionError, match="score-scheduler-lane-isolation-short-burst"):
        server_args.check_server_args()


def test_score_scheduler_controls_reject_non_positive_dynamic_pressure_threshold():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--score-scheduler-dynamic-items-per-step-pressure-threshold",
            "0",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    with pytest.raises(
        AssertionError,
        match="score-scheduler-dynamic-items-per-step-pressure-threshold",
    ):
        server_args.check_server_args()


def test_multi_item_label_only_fused_kernel_default_true():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy-model"])
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_score_label_only_fused_kernel is True


def test_multi_item_label_only_fused_kernel_can_disable():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "dummy-model",
            "--no-multi-item-score-label-only-fused-kernel",
        ]
    )
    server_args = ServerArgs.from_cli_args(args)
    assert server_args.multi_item_score_label_only_fused_kernel is False
