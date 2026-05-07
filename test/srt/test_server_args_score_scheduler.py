import argparse

import pytest

from sgl_jax.srt.server_args import ServerArgs


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
            "--no-score-scheduler-cache-admission-bias-require-hit",
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
    assert server_args.score_scheduler_cache_admission_bias_require_hit is False


def test_score_scheduler_controls_reject_negative_window():
    server_args = ServerArgs(
        model_path="dummy-model",
        score_scheduler_global_microbatch_window_ms=-1.0,
    )
    with pytest.raises(AssertionError, match="global-microbatch-window-ms"):
        server_args.check_server_args()


def test_score_scheduler_controls_reject_non_positive_lane_isolation_burst():
    server_args = ServerArgs(
        model_path="dummy-model",
        score_scheduler_lane_isolation_short_burst=0,
    )
    with pytest.raises(AssertionError, match="lane-isolation-short-burst"):
        server_args.check_server_args()


def test_score_scheduler_controls_reject_non_positive_dynamic_pressure_threshold():
    server_args = ServerArgs(
        model_path="dummy-model",
        score_scheduler_dynamic_items_per_step_pressure_threshold=0,
    )
    with pytest.raises(AssertionError, match="dynamic-items-per-step-pressure-threshold"):
        server_args.check_server_args()
