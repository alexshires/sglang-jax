"""Scheduler scoring state, ingress, and admission helpers."""

import concurrent.futures as futures
import queue

import zmq

from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.scheduler_scoring_common import _LocalSchedulerRpcEnvelope
from sgl_jax.srt.server_args import ServerArgs


class SchedulerScoringStateMixin:
    def init_scoring_state(self, server_args: ServerArgs) -> None:
        self.local_rpc_queue: queue.SimpleQueue[_LocalSchedulerRpcEnvelope] = queue.SimpleQueue()

        # Workstream B: Store cached nodes for prefill+extend
        # Map:
        # rid -> (
        #   last_node,
        #   swa_uuid_for_lock,
        #   input_ids,
        #   prefix_indices,
        #   extra_key,
        #   last_access_ts,
        # )
        self.scoring_cache_nodes = {}
        self.scoring_cache_timeout = float(server_args.multi_item_prefill_extend_cache_timeout)
        self._last_scoring_cache_gc = 0.0
        # Scoring-cache counters (vLLM-style query/hit/miss accounting).
        self.scoring_cache_lookup_queries = 0
        self.scoring_cache_lookup_hits = 0
        self.scoring_cache_lookup_misses = 0
        self.scoring_cache_lookup_by_path: dict[str, dict[str, int]] = {
            "extend": {"queries": 0, "hits": 0, "misses": 0},
            "score_from_cache_v2": {"queries": 0, "hits": 0, "misses": 0},
            "cache_for_scoring": {"queries": 0, "hits": 0, "misses": 0},
        }
        self.scoring_cache_lookup_by_lane: dict[str, dict[str, dict[str, int]]] = {
            "extend": {
                "default": {"queries": 0, "hits": 0, "misses": 0},
                "short": {"queries": 0, "hits": 0, "misses": 0},
                "long": {"queries": 0, "hits": 0, "misses": 0},
            },
            "score_from_cache_v2": {
                "default": {"queries": 0, "hits": 0, "misses": 0},
                "short": {"queries": 0, "hits": 0, "misses": 0},
                "long": {"queries": 0, "hits": 0, "misses": 0},
            },
            "cache_for_scoring": {
                "default": {"queries": 0, "hits": 0, "misses": 0},
                "short": {"queries": 0, "hits": 0, "misses": 0},
                "long": {"queries": 0, "hits": 0, "misses": 0},
            },
        }
        # Prefix-level registry for cross-request scoring-cache reuse telemetry/admission.
        self.scoring_cache_prefix_handles_by_key: dict[tuple[str, tuple[int, ...]], set[str]] = {}
        self.scoring_cache_handle_to_prefix_key: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.scoring_cache_handles_created = 0
        self.scoring_cache_handles_released = 0
        self.scoring_cache_handles_released_manual = 0
        self.scoring_cache_handles_released_expired = 0
        self.scoring_cache_handles_released_other = 0
        self.scoring_cache_handles_missing_node = 0
        self.scoring_cache_release_failures = 0
        self._warned_scoring_cache_lanes = set()
        # Ingress message metrics for tokenizer->scheduler and rpc->scheduler paths.
        self.ingress_recv_calls = 0
        self.ingress_nonempty_calls = 0
        self.ingress_max_batch_size = 0
        self.ingress_tokenizer_frames = 0
        self.ingress_rpc_frames = 0
        # "messages" here counts logical requests seen by scheduler after unpack.
        self.ingress_tokenizer_messages = 0
        self.ingress_rpc_messages = 0
        self.ingress_batch_size_histogram = {
            "eq_0": 0,
            "eq_1": 0,
            "2_to_4": 0,
            "5_to_16": 0,
            "gt_16": 0,
        }
        self.ingress_score_paths = {
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }
        # Number of socket frames that carried each scoring path.
        self.ingress_score_path_frames = {
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }
        # Fastpath v2 score-from-cache counters.
        self.score_from_cache_v2_attempted = 0
        self.score_from_cache_v2_succeeded = 0
        self.score_from_cache_v2_fallback = 0
        self.score_from_cache_v2_fallback_reasons: dict[str, int] = {}
        self.score_from_cache_v2_queue_wait_s_total = 0.0
        self.score_from_cache_v2_device_compute_s_total = 0.0
        self.score_from_cache_v2_host_orchestration_s_total = 0.0
        self.score_from_cache_v2_queue_wait_s_max = 0.0
        self.score_from_cache_v2_device_compute_s_max = 0.0
        self.score_from_cache_v2_host_orchestration_s_max = 0.0
        self.score_label_only_fused_kernel_calls = 0
        self.score_label_only_fused_kernel_softmax_calls = 0
        self.score_label_only_legacy_kernel_calls = 0
        self.score_label_only_token_ids_only_calls = 0
        # Score ingress coalescing + lane fairness controls.
        self.score_scheduler_global_microbatch_window_s = max(
            0.0,
            float(getattr(server_args, "score_scheduler_global_microbatch_window_ms", 0.0))
            / 1000.0,
        )
        self.score_scheduler_global_microbatch_poll_s = max(
            0.0001,
            float(getattr(server_args, "score_scheduler_global_microbatch_poll_interval_ms", 0.5))
            / 1000.0,
        )
        self.score_scheduler_short_prompt_tokens_threshold = max(
            1,
            int(getattr(server_args, "score_scheduler_short_prompt_tokens_threshold", 2048)),
        )
        self.score_scheduler_short_lane_max_inflight = max(
            0,
            int(getattr(server_args, "score_scheduler_short_lane_max_inflight", 0)),
        )
        self.score_scheduler_long_lane_max_inflight = max(
            0,
            int(getattr(server_args, "score_scheduler_long_lane_max_inflight", 0)),
        )
        self.score_scheduler_enable_lane_isolation = bool(
            getattr(server_args, "score_scheduler_enable_lane_isolation", False)
        )
        self.score_scheduler_lane_isolation_short_burst = max(
            1,
            int(getattr(server_args, "score_scheduler_lane_isolation_short_burst", 2)),
        )
        self.score_scheduler_lane_isolation_long_burst = max(
            1,
            int(getattr(server_args, "score_scheduler_lane_isolation_long_burst", 1)),
        )
        self.score_scheduler_dynamic_items_per_step_enable = bool(
            getattr(server_args, "score_scheduler_dynamic_items_per_step_enable", False)
        )
        self.score_scheduler_dynamic_items_per_step_pressure_threshold = max(
            1,
            int(
                getattr(
                    server_args,
                    "score_scheduler_dynamic_items_per_step_pressure_threshold",
                    64,
                )
            ),
        )
        self.score_scheduler_dynamic_items_per_step_short_lane_bias = max(
            0.1,
            float(
                getattr(
                    server_args,
                    "score_scheduler_dynamic_items_per_step_short_lane_bias",
                    1.0,
                )
            ),
        )
        self.score_scheduler_dynamic_items_per_step_long_lane_bias = max(
            0.1,
            float(
                getattr(
                    server_args,
                    "score_scheduler_dynamic_items_per_step_long_lane_bias",
                    0.75,
                )
            ),
        )
        self.score_scheduler_dynamic_items_per_step_short_lane_min = max(
            1,
            int(
                getattr(
                    server_args,
                    "score_scheduler_dynamic_items_per_step_short_lane_min",
                    32,
                )
            ),
        )
        self.score_scheduler_dynamic_items_per_step_long_lane_min = max(
            1,
            int(
                getattr(
                    server_args,
                    "score_scheduler_dynamic_items_per_step_long_lane_min",
                    16,
                )
            ),
        )
        self.score_scheduler_cache_admission_bias_enable = bool(
            getattr(server_args, "score_scheduler_cache_admission_bias_enable", False)
        )
        self.score_scheduler_cache_admission_bias_require_hit = bool(
            getattr(server_args, "score_scheduler_cache_admission_bias_require_hit", True)
        )
        self.score_scheduler_microbatch_windows = 0
        self.score_scheduler_microbatch_added_requests = 0
        self.score_scheduler_microbatch_max_added_requests = 0
        self.score_scheduler_lane_admission_attempted = 0
        self.score_scheduler_lane_admission_admitted = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_lane_admission_skipped = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_lane_inflight_max = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_lane_isolation_rounds = 0
        self.score_scheduler_lane_isolation_selected = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_lane_isolation_empty_turns = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_lane_waiting_max = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_dynamic_items_per_step_requests = 0
        self.score_scheduler_dynamic_items_per_step_requested_total = 0
        self.score_scheduler_dynamic_items_per_step_effective_total = 0
        self.score_scheduler_dynamic_items_per_step_max_queue_pressure = 0
        self.score_scheduler_dynamic_items_per_step_requested_by_lane = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_dynamic_items_per_step_effective_by_lane = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_dynamic_items_per_step_applied_by_lane = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_cache_admission_candidates = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_cache_admission_promoted = {
            "default": 0,
            "short": 0,
            "long": 0,
        }
        self.score_scheduler_topology_name = self._detect_score_scheduler_topology_name()

    def add_scoring_internal_state(self, ret: dict) -> None:
        score_from_cache_v2_attempted = self.score_from_cache_v2_attempted
        score_timing_totals_s = {
            "queue_wait": self.score_from_cache_v2_queue_wait_s_total,
            "device_compute": self.score_from_cache_v2_device_compute_s_total,
            "host_orchestration": self.score_from_cache_v2_host_orchestration_s_total,
        }
        score_timing_max_s = {
            "queue_wait": self.score_from_cache_v2_queue_wait_s_max,
            "device_compute": self.score_from_cache_v2_device_compute_s_max,
            "host_orchestration": self.score_from_cache_v2_host_orchestration_s_max,
        }
        if score_from_cache_v2_attempted > 0:
            score_timing_mean_s = {
                "queue_wait": (
                    self.score_from_cache_v2_queue_wait_s_total / score_from_cache_v2_attempted
                ),
                "device_compute": (
                    self.score_from_cache_v2_device_compute_s_total / score_from_cache_v2_attempted
                ),
                "host_orchestration": (
                    self.score_from_cache_v2_host_orchestration_s_total
                    / score_from_cache_v2_attempted
                ),
            }
        else:
            score_timing_mean_s = {
                "queue_wait": 0.0,
                "device_compute": 0.0,
                "host_orchestration": 0.0,
            }

        ret["score_from_cache_v2_metrics"] = {
            "attempted": score_from_cache_v2_attempted,
            "succeeded": self.score_from_cache_v2_succeeded,
            "fallback": self.score_from_cache_v2_fallback,
            "fallback_reasons": dict(self.score_from_cache_v2_fallback_reasons),
            "label_only_fused_kernel_calls": int(
                getattr(self, "score_label_only_fused_kernel_calls", 0) or 0
            ),
            "label_only_fused_kernel_softmax_calls": int(
                getattr(self, "score_label_only_fused_kernel_softmax_calls", 0) or 0
            ),
            "label_only_legacy_kernel_calls": int(
                getattr(self, "score_label_only_legacy_kernel_calls", 0) or 0
            ),
            "label_only_token_ids_only_calls": int(
                getattr(self, "score_label_only_token_ids_only_calls", 0) or 0
            ),
            "timing_totals_s": score_timing_totals_s,
            "timing_mean_s": score_timing_mean_s,
            "timing_max_s": score_timing_max_s,
        }
        ret["scoring_cache_metrics"] = self._scoring_cache_metrics_snapshot()
        score_path_messages = dict(self.ingress_score_paths)
        score_path_frames = dict(self.ingress_score_path_frames)
        score_path_messages_per_frame = {}
        for path_name, path_message_count in score_path_messages.items():
            path_frame_count = score_path_frames.get(path_name, 0)
            score_path_messages_per_frame[path_name] = (
                float(path_message_count / path_frame_count) if path_frame_count > 0 else 0.0
            )
        ret["ingress_metrics"] = {
            "recv_calls": self.ingress_recv_calls,
            "nonempty_calls": self.ingress_nonempty_calls,
            "max_batch_size": self.ingress_max_batch_size,
            "tokenizer_frames": self.ingress_tokenizer_frames,
            "rpc_frames": self.ingress_rpc_frames,
            "tokenizer_messages": self.ingress_tokenizer_messages,
            "rpc_messages": self.ingress_rpc_messages,
            "tokenizer_messages_per_frame": (
                float(self.ingress_tokenizer_messages / self.ingress_tokenizer_frames)
                if self.ingress_tokenizer_frames > 0
                else 0.0
            ),
            "rpc_messages_per_frame": (
                float(self.ingress_rpc_messages / self.ingress_rpc_frames)
                if self.ingress_rpc_frames > 0
                else 0.0
            ),
            "batch_size_histogram": dict(self.ingress_batch_size_histogram),
            "score_path_messages": score_path_messages,
            "score_path_frames": score_path_frames,
            "score_path_messages_per_frame": score_path_messages_per_frame,
            "score_coalescing": {
                "window_s": float(
                    getattr(self, "score_scheduler_global_microbatch_window_s", 0.0) or 0.0
                ),
                "poll_interval_s": float(
                    getattr(self, "score_scheduler_global_microbatch_poll_s", 0.0005) or 0.0005
                ),
                "windows": int(getattr(self, "score_scheduler_microbatch_windows", 0) or 0),
                "added_requests_total": int(
                    getattr(self, "score_scheduler_microbatch_added_requests", 0) or 0
                ),
                "max_added_requests": int(
                    getattr(self, "score_scheduler_microbatch_max_added_requests", 0) or 0
                ),
            },
        }
        ret["score_scheduler_admission_metrics"] = {
            "short_prompt_tokens_threshold": int(
                getattr(self, "score_scheduler_short_prompt_tokens_threshold", 2048) or 2048
            ),
            "short_lane_max_inflight": int(
                getattr(self, "score_scheduler_short_lane_max_inflight", 0) or 0
            ),
            "long_lane_max_inflight": int(
                getattr(self, "score_scheduler_long_lane_max_inflight", 0) or 0
            ),
            "lane_isolation_enabled": bool(
                getattr(self, "score_scheduler_enable_lane_isolation", False)
            ),
            "lane_isolation_short_burst": int(
                getattr(self, "score_scheduler_lane_isolation_short_burst", 2) or 2
            ),
            "lane_isolation_long_burst": int(
                getattr(self, "score_scheduler_lane_isolation_long_burst", 1) or 1
            ),
            "lane_isolation_rounds": int(
                getattr(self, "score_scheduler_lane_isolation_rounds", 0) or 0
            ),
            "attempted": int(getattr(self, "score_scheduler_lane_admission_attempted", 0) or 0),
            "admitted_by_lane": dict(
                SchedulerScoringStateMixin._lane_counter(
                    self, "score_scheduler_lane_admission_admitted"
                )
            ),
            "skipped_by_lane": dict(
                SchedulerScoringStateMixin._lane_counter(
                    self, "score_scheduler_lane_admission_skipped"
                )
            ),
            "max_inflight_by_lane": dict(
                SchedulerScoringStateMixin._lane_counter(self, "score_scheduler_lane_inflight_max")
            ),
            "lane_isolation_selected_by_lane": dict(
                SchedulerScoringStateMixin._lane_counter(
                    self, "score_scheduler_lane_isolation_selected"
                )
            ),
            "lane_isolation_empty_turns_by_lane": dict(
                SchedulerScoringStateMixin._lane_counter(
                    self, "score_scheduler_lane_isolation_empty_turns"
                )
            ),
            "max_waiting_by_lane": dict(
                SchedulerScoringStateMixin._lane_counter(self, "score_scheduler_lane_waiting_max")
            ),
            "dynamic_items_per_step": {
                "enabled": bool(
                    getattr(self, "score_scheduler_dynamic_items_per_step_enable", False)
                ),
                "pressure_threshold": int(
                    getattr(self, "score_scheduler_dynamic_items_per_step_pressure_threshold", 64)
                    or 64
                ),
                "short_lane_bias": float(
                    getattr(self, "score_scheduler_dynamic_items_per_step_short_lane_bias", 1.0)
                    or 1.0
                ),
                "long_lane_bias": float(
                    getattr(self, "score_scheduler_dynamic_items_per_step_long_lane_bias", 0.75)
                    or 0.75
                ),
                "short_lane_min": int(
                    getattr(self, "score_scheduler_dynamic_items_per_step_short_lane_min", 32) or 32
                ),
                "long_lane_min": int(
                    getattr(self, "score_scheduler_dynamic_items_per_step_long_lane_min", 16) or 16
                ),
                "requests": int(
                    getattr(self, "score_scheduler_dynamic_items_per_step_requests", 0) or 0
                ),
                "requested_total": int(
                    getattr(self, "score_scheduler_dynamic_items_per_step_requested_total", 0) or 0
                ),
                "effective_total": int(
                    getattr(self, "score_scheduler_dynamic_items_per_step_effective_total", 0) or 0
                ),
                "max_queue_pressure": int(
                    getattr(self, "score_scheduler_dynamic_items_per_step_max_queue_pressure", 0)
                    or 0
                ),
                "requested_by_lane": dict(
                    SchedulerScoringStateMixin._lane_counter(
                        self, "score_scheduler_dynamic_items_per_step_requested_by_lane"
                    )
                ),
                "effective_by_lane": dict(
                    SchedulerScoringStateMixin._lane_counter(
                        self, "score_scheduler_dynamic_items_per_step_effective_by_lane"
                    )
                ),
                "applied_by_lane": dict(
                    SchedulerScoringStateMixin._lane_counter(
                        self, "score_scheduler_dynamic_items_per_step_applied_by_lane"
                    )
                ),
            },
            "cache_admission_bias": {
                "enabled": bool(
                    getattr(self, "score_scheduler_cache_admission_bias_enable", False)
                ),
                "require_hit": bool(
                    getattr(self, "score_scheduler_cache_admission_bias_require_hit", True)
                ),
                "candidates_by_lane": dict(
                    SchedulerScoringStateMixin._lane_counter(
                        self, "score_scheduler_cache_admission_candidates"
                    )
                ),
                "promoted_by_lane": dict(
                    SchedulerScoringStateMixin._lane_counter(
                        self, "score_scheduler_cache_admission_promoted"
                    )
                ),
            },
        }

    @staticmethod
    def _is_score_path_req(req: Req) -> bool:
        if bool(getattr(req, "cache_for_scoring", False)):
            return True
        if bool(getattr(req, "extend_from_cache", None)):
            return True
        if not bool(getattr(req, "return_logprob", False)):
            return False
        sampling_params = getattr(req, "sampling_params", None)
        max_new_tokens = getattr(sampling_params, "max_new_tokens", None)
        return int(max_new_tokens or 0) <= 0

    @staticmethod
    def _can_skip_sample_for_prefill_batch(batch: ScheduleBatch | None) -> bool:
        if batch is None or not bool(getattr(batch, "is_prefill_only", False)):
            return False
        forward_mode = getattr(batch, "forward_mode", None)
        if forward_mode is None or not bool(forward_mode.is_extend()):
            return False
        if bool(getattr(batch, "return_logprob", False)):
            return False
        if bool(getattr(batch, "return_output_logprob_only", False)):
            return False
        reqs = getattr(batch, "reqs", None) or []
        return len(reqs) > 0 and all(bool(getattr(req, "cache_for_scoring", False)) for req in reqs)

    @staticmethod
    def _admission_lane(req_owner, req: Req) -> str:
        if not SchedulerScoringStateMixin._is_score_path_req(req):
            return "default"
        threshold = max(
            1,
            int(getattr(req_owner, "score_scheduler_short_prompt_tokens_threshold", 2048) or 2048),
        )
        prompt_tokens = len(getattr(req, "origin_input_ids", []) or [])
        return "short" if prompt_tokens <= threshold else "long"

    @staticmethod
    def _lane_cap(req_owner, lane: str) -> int:
        if lane == "short":
            return max(
                0, int(getattr(req_owner, "score_scheduler_short_lane_max_inflight", 0) or 0)
            )
        if lane == "long":
            return max(0, int(getattr(req_owner, "score_scheduler_long_lane_max_inflight", 0) or 0))
        return 0

    @staticmethod
    def _lane_counter(req_owner, attr_name: str) -> dict[str, int]:
        counter = getattr(req_owner, attr_name, None)
        if not isinstance(counter, dict):
            counter = {}
            setattr(req_owner, attr_name, counter)
        counter.setdefault("default", 0)
        counter.setdefault("short", 0)
        counter.setdefault("long", 0)
        return counter

    @staticmethod
    def _running_lane_counts(req_owner) -> dict[str, int]:
        counts = {"default": 0, "short": 0, "long": 0}
        running_batch = getattr(req_owner, "running_batch", None)
        running_reqs = getattr(running_batch, "reqs", []) if running_batch is not None else []
        for req in running_reqs:
            lane = SchedulerScoringStateMixin._admission_lane(req_owner, req)
            counts[lane] = counts.get(lane, 0) + 1
        return counts

    @staticmethod
    def _waiting_lane_counts(req_owner, waiting_queue: list[Req]) -> dict[str, int]:
        counts = {"default": 0, "short": 0, "long": 0}
        for req in waiting_queue:
            lane = SchedulerScoringStateMixin._admission_lane(req_owner, req)
            counts[lane] = counts.get(lane, 0) + 1
        return counts

    @staticmethod
    def _normalize_scoring_cache_prefix_key(
        input_ids: list[int] | tuple[int, ...] | np.ndarray | None,
        extra_key: str | None,
    ) -> tuple[str, tuple[int, ...]] | None:
        if input_ids is None:
            return None
        token_list = input_ids.tolist() if isinstance(input_ids, np.ndarray) else list(input_ids)
        if not token_list:
            return None
        normalized_extra_key = "" if extra_key is None else str(extra_key)
        return normalized_extra_key, tuple(int(tok) for tok in token_list)

    @staticmethod
    def _cache_admission_priority(req_owner, req: Req) -> int:
        if not bool(getattr(req_owner, "score_scheduler_cache_admission_bias_enable", False)):
            return 0

        extend_handle = getattr(req, "extend_from_cache", None)
        if isinstance(extend_handle, str) and extend_handle:
            scoring_cache_nodes = getattr(req_owner, "scoring_cache_nodes", {})
            if extend_handle in scoring_cache_nodes:
                return 3
            if bool(getattr(req_owner, "score_scheduler_cache_admission_bias_require_hit", True)):
                return 0
            return 1

        if not bool(getattr(req, "cache_for_scoring", False)):
            return 0

        prefix_key = req_owner._normalize_scoring_cache_prefix_key(
            getattr(req, "origin_input_ids", None),
            getattr(req, "extra_key", None),
        )
        if prefix_key is None:
            return 0
        prefix_registry = getattr(req_owner, "scoring_cache_prefix_handles_by_key", {})
        return 2 if prefix_key in prefix_registry else 0

    @staticmethod
    def _iter_waiting_queue(req_owner, waiting_queue: list[Req]) -> list[Req]:
        bias_enabled = bool(
            getattr(req_owner, "score_scheduler_cache_admission_bias_enable", False)
        )
        lane_bias_candidates = SchedulerScoringStateMixin._lane_counter(
            req_owner,
            "score_scheduler_cache_admission_candidates",
        )
        lane_bias_promoted = SchedulerScoringStateMixin._lane_counter(
            req_owner,
            "score_scheduler_cache_admission_promoted",
        )

        def _apply_cache_bias(
            lane_name: str,
            queue_items: list[Req] | deque[Req],
        ) -> deque[Req]:
            items = list(queue_items)
            if not bias_enabled or not items:
                return deque(items)

            scored = []
            for idx, lane_req in enumerate(items):
                priority = SchedulerScoringStateMixin._cache_admission_priority(req_owner, lane_req)
                scored.append((idx, priority, lane_req))
                if priority > 0:
                    lane_bias_candidates[lane_name] = lane_bias_candidates.get(lane_name, 0) + 1

            scored.sort(key=lambda x: (-x[1], x[0]))
            for new_idx, (old_idx, priority, _) in enumerate(scored):
                if priority > 0 and new_idx < old_idx:
                    lane_bias_promoted[lane_name] = lane_bias_promoted.get(lane_name, 0) + 1

            return deque([req for _, _, req in scored])

        if not bool(getattr(req_owner, "score_scheduler_enable_lane_isolation", False)):
            return list(_apply_cache_bias("default", waiting_queue))

        lane_queues = {
            "default": deque(),
            "short": deque(),
            "long": deque(),
        }
        for req in waiting_queue:
            lane_queues[SchedulerScoringStateMixin._admission_lane(req_owner, req)].append(req)

        for lane_name in ("default", "short", "long"):
            lane_queues[lane_name] = _apply_cache_bias(lane_name, lane_queues[lane_name])

        lane_waiting_max = SchedulerScoringStateMixin._lane_counter(
            req_owner, "score_scheduler_lane_waiting_max"
        )
        for lane_name, lane_queue in lane_queues.items():
            lane_waiting_max[lane_name] = max(lane_waiting_max.get(lane_name, 0), len(lane_queue))

        lane_selected = SchedulerScoringStateMixin._lane_counter(
            req_owner,
            "score_scheduler_lane_isolation_selected",
        )
        lane_empty_turns = SchedulerScoringStateMixin._lane_counter(
            req_owner,
            "score_scheduler_lane_isolation_empty_turns",
        )
        short_burst = max(
            1,
            int(getattr(req_owner, "score_scheduler_lane_isolation_short_burst", 2) or 2),
        )
        long_burst = max(
            1,
            int(getattr(req_owner, "score_scheduler_lane_isolation_long_burst", 1) or 1),
        )

        ordered_waiting_queue: list[Req] = []
        # Short-first weighted round robin keeps short-lane score traffic from
        # being dominated by long-lane queue depth, while still admitting long/default.
        lane_plan = (("short", short_burst), ("default", 1), ("long", long_burst))
        while any(lane_queues[lane_name] for lane_name in lane_queues):
            round_made_progress = False
            for lane_name, burst in lane_plan:
                if not lane_queues[lane_name]:
                    lane_empty_turns[lane_name] = lane_empty_turns.get(lane_name, 0) + 1
                    continue
                for _ in range(burst):
                    if not lane_queues[lane_name]:
                        break
                    ordered_waiting_queue.append(lane_queues[lane_name].popleft())
                    lane_selected[lane_name] = lane_selected.get(lane_name, 0) + 1
                    round_made_progress = True
            if not round_made_progress:
                break

        req_owner.score_scheduler_lane_isolation_rounds = (
            int(getattr(req_owner, "score_scheduler_lane_isolation_rounds", 0)) + 1
        )
        return ordered_waiting_queue

    @staticmethod
    def _score_scheduler_queue_pressure(req_owner) -> int:
        waiting_queue = getattr(req_owner, "waiting_queue", []) or []
        running_batch = getattr(req_owner, "running_batch", None)
        running_reqs = getattr(running_batch, "reqs", []) if running_batch is not None else []
        return max(0, len(waiting_queue)) + max(0, len(running_reqs))

    @staticmethod
    def _score_scheduler_lane_from_prefix_len(req_owner, prefix_len: int) -> str:
        threshold = max(
            1,
            int(getattr(req_owner, "score_scheduler_short_prompt_tokens_threshold", 2048) or 2048),
        )
        return "short" if int(prefix_len) <= threshold else "long"

    def recv_requests(self) -> list[Req]:
        """Receive results at node_rank = 0 and broadcast it to all other Node ranks."""
        self.ingress_recv_calls += 1
        if self.node_rank == 0:
            recv_reqs = []
            tokenizer_frame_count = 0
            rpc_frame_count = 0
            tokenizer_req_count = 0
            rpc_req_count = 0
            score_path_counts = self.ingress_score_paths
            score_path_frames = self.ingress_score_path_frames

            def _drain_ingress_once() -> None:
                nonlocal tokenizer_frame_count, rpc_frame_count, tokenizer_req_count, rpc_req_count
                while True:
                    try:
                        recv_obj = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    tokenizer_frame_count += 1
                    unpacked_reqs = (
                        list(recv_obj) if isinstance(recv_obj, (list, tuple)) else [recv_obj]
                    )
                    recv_reqs.extend(unpacked_reqs)
                    tokenizer_req_count += len(unpacked_reqs)
                    tokenizer_frame_paths = {
                        "tokenizer_cache_for_scoring": False,
                        "tokenizer_extend_from_cache": False,
                        "rpc_score_from_cache_v2": False,
                        "rpc_release_scoring_cache": False,
                    }
                    for recv_req in unpacked_reqs:
                        if isinstance(recv_req, TokenizedGenerateReqInput):
                            if bool(getattr(recv_req, "cache_for_scoring", False)):
                                score_path_counts["tokenizer_cache_for_scoring"] += 1
                                tokenizer_frame_paths["tokenizer_cache_for_scoring"] = True
                            if bool(getattr(recv_req, "extend_from_cache", None)):
                                score_path_counts["tokenizer_extend_from_cache"] += 1
                                tokenizer_frame_paths["tokenizer_extend_from_cache"] = True
                        elif isinstance(recv_req, ScoreFromCacheReqInput):
                            # Score fastpath requests are sent by tokenizer manager over the
                            # tokenizer socket.
                            score_path_counts["rpc_score_from_cache_v2"] += 1
                            tokenizer_frame_paths["rpc_score_from_cache_v2"] = True
                        elif isinstance(recv_req, ReleaseScoringCacheReqInput):
                            # Cache release requests are also sent over the tokenizer socket.
                            score_path_counts["rpc_release_scoring_cache"] += 1
                            tokenizer_frame_paths["rpc_release_scoring_cache"] = True
                    for path, seen in tokenizer_frame_paths.items():
                        if seen:
                            score_path_frames[path] += 1

                while True:
                    try:
                        recv_obj = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    rpc_frame_count += 1
                    unpacked_reqs = (
                        list(recv_obj) if isinstance(recv_obj, (list, tuple)) else [recv_obj]
                    )
                    recv_reqs.extend(unpacked_reqs)
                    rpc_req_count += len(unpacked_reqs)
                    rpc_frame_paths = {
                        "rpc_score_from_cache_v2": False,
                        "rpc_release_scoring_cache": False,
                    }
                    for recv_rpc in unpacked_reqs:
                        if isinstance(recv_rpc, ScoreFromCacheReqInput):
                            score_path_counts["rpc_score_from_cache_v2"] += 1
                            rpc_frame_paths["rpc_score_from_cache_v2"] = True
                        elif isinstance(recv_rpc, ReleaseScoringCacheReqInput):
                            score_path_counts["rpc_release_scoring_cache"] += 1
                            rpc_frame_paths["rpc_release_scoring_cache"] = True
                    for path, seen in rpc_frame_paths.items():
                        if seen:
                            score_path_frames[path] += 1

                local_rpc_queue = getattr(self, "local_rpc_queue", None)
                if local_rpc_queue is not None:
                    while True:
                        try:
                            recv_local = local_rpc_queue.get_nowait()
                        except queue.Empty:
                            break
                        rpc_frame_count += 1
                        recv_reqs.append(recv_local)
                        rpc_req_count += 1
                        rpc_frame_paths = {
                            "rpc_score_from_cache_v2": False,
                            "rpc_release_scoring_cache": False,
                        }
                        local_req = recv_local.req_obj
                        if isinstance(local_req, ScoreFromCacheReqInput):
                            score_path_counts["rpc_score_from_cache_v2"] += 1
                            rpc_frame_paths["rpc_score_from_cache_v2"] = True
                        elif isinstance(local_req, ReleaseScoringCacheReqInput):
                            score_path_counts["rpc_release_scoring_cache"] += 1
                            rpc_frame_paths["rpc_release_scoring_cache"] = True
                        for path, seen in rpc_frame_paths.items():
                            if seen:
                                score_path_frames[path] += 1

            _drain_ingress_once()
            initial_batch_size = tokenizer_req_count + rpc_req_count
            coalesce_window_s = max(
                0.0,
                float(getattr(self, "score_scheduler_global_microbatch_window_s", 0.0) or 0.0),
            )
            coalesce_poll_s = max(
                0.0001,
                float(getattr(self, "score_scheduler_global_microbatch_poll_s", 0.0005) or 0.0005),
            )
            if initial_batch_size > 0 and coalesce_window_s > 0:
                self.score_scheduler_microbatch_windows = (
                    int(getattr(self, "score_scheduler_microbatch_windows", 0)) + 1
                )
                deadline = time.perf_counter() + coalesce_window_s
                while True:
                    remaining_s = deadline - time.perf_counter()
                    if remaining_s <= 0:
                        break
                    time.sleep(min(coalesce_poll_s, remaining_s))
                    _drain_ingress_once()
                added = (tokenizer_req_count + rpc_req_count) - initial_batch_size
                if added > 0:
                    self.score_scheduler_microbatch_added_requests = (
                        int(getattr(self, "score_scheduler_microbatch_added_requests", 0)) + added
                    )
                    self.score_scheduler_microbatch_max_added_requests = max(
                        int(getattr(self, "score_scheduler_microbatch_max_added_requests", 0)),
                        added,
                    )

            self.ingress_tokenizer_frames += tokenizer_frame_count
            self.ingress_rpc_frames += rpc_frame_count
            self.ingress_tokenizer_messages += tokenizer_req_count
            self.ingress_rpc_messages += rpc_req_count
            batch_size = tokenizer_req_count + rpc_req_count
            if batch_size > 0:
                self.ingress_nonempty_calls += 1
                if batch_size > self.ingress_max_batch_size:
                    self.ingress_max_batch_size = batch_size
            if batch_size == 0:
                self.ingress_batch_size_histogram["eq_0"] += 1
            elif batch_size == 1:
                self.ingress_batch_size_histogram["eq_1"] += 1
            elif batch_size <= 4:
                self.ingress_batch_size_histogram["2_to_4"] += 1
            elif batch_size <= 16:
                self.ingress_batch_size_histogram["5_to_16"] += 1
            else:
                self.ingress_batch_size_histogram["gt_16"] += 1
        else:
            recv_reqs = None

        if self.nnodes > 1:
            recv_reqs = self.broadcast_pyobj(recv_reqs)
        return recv_reqs

    def submit_local_rpc(self, req_obj) -> futures.Future:
        future: futures.Future = futures.Future()
        self.local_rpc_queue.put(_LocalSchedulerRpcEnvelope(req_obj=req_obj, result_future=future))
        return future

    def submit_local_request(self, req_obj) -> None:
        self.local_rpc_queue.put(_LocalSchedulerRpcEnvelope(req_obj=req_obj, result_future=None))

    def process_input_requests(self, recv_reqs: list):
        self._evict_expired_scoring_cache_nodes()
        for recv_req in recv_reqs:
            local_result_future = None
            dispatch_req = recv_req
            if isinstance(recv_req, _LocalSchedulerRpcEnvelope):
                local_result_future = recv_req.result_future
                dispatch_req = recv_req.req_obj

            try:
                output = self._request_dispatcher(dispatch_req)
            except Exception as exc:
                if local_result_future is not None and not local_result_future.done():
                    local_result_future.set_exception(exc)
                raise

            if local_result_future is not None:
                if not local_result_future.done():
                    local_result_future.set_result(output)
                continue

            if output is not None:
                if self._comm_backend is not None:
                    self._comm_backend.send_pyobj(output)
                else:
                    self.send_to_tokenizer.send_pyobj(output)
