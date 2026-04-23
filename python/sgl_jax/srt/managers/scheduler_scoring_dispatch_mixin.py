"""Scheduler score dispatch and chunk planning helpers."""

from sgl_jax.srt.managers.scheduler_scoring_common import *


class SchedulerScoringDispatchMixin:
    def _detect_score_scheduler_topology_name(self) -> str:
        try:
            logical_device_count = int(
                os.environ.get(
                    "SGLANG_LOGICAL_DEVICE_COUNT",
                    len(getattr(self.server_args, "device_indexes", []) or []),
                )
                or len(jax.devices())
            )
            return str(get_device_name(num_devices=logical_device_count) or "")
        except Exception:
            return ""

    def _score_from_cache_v2_replica_lane_count(self) -> int:
        mesh = getattr(self, "mesh", None)
        mesh_shape = getattr(mesh, "shape", None)
        if mesh_shape is None:
            return 1
        return max(1, int(mesh_shape.get("data", 1) or 1))

    def _score_from_cache_v2_topology_dispatch_policy(
        self,
        *,
        lane_name: str,
        prefix_len: int,
        requested_items_per_step: int,
        effective_items_per_step: int,
        effective_capacity: int,
        total_items: int,
        requested_token_budget: int,
        max_total_tokens: int,
    ) -> tuple[int, int, int, str]:
        topology_name = str(getattr(self, "score_scheduler_topology_name", "") or "")
        replica_lane_count = self._score_from_cache_v2_replica_lane_count()
        dispatch_token_budget = max(0, int(requested_token_budget or 0))

        if (
            topology_name != "TPU v6e-8"
            or replica_lane_count <= 1
            or effective_capacity <= 1
            or total_items <= 1
        ):
            return (
                effective_items_per_step,
                dispatch_token_budget,
                replica_lane_count,
                topology_name,
            )

        lane_scale = replica_lane_count if lane_name == "long" else max(1, replica_lane_count - 1)
        boosted_items_per_step = max(
            effective_items_per_step,
            min(
                max(1, total_items),
                max(1, effective_capacity),
                max(1, int(effective_items_per_step or 1)) * lane_scale,
            ),
        )
        max_total_tokens = max(0, int(max_total_tokens or 0))
        prefix_len = max(0, int(prefix_len or 0))
        if max_total_tokens > 0:
            boosted_token_budget = (prefix_len + max_total_tokens) * boosted_items_per_step
            dispatch_token_budget = max(dispatch_token_budget, boosted_token_budget)

        return boosted_items_per_step, dispatch_token_budget, replica_lane_count, topology_name

    def _score_from_cache_v2_use_direct_label_only(self, *, label_only_logprob: bool) -> bool:
        return bool(
            label_only_logprob
            and getattr(self.server_args, "multi_item_score_direct_label_only", False)
        )

    def _score_from_cache_v2_use_direct_token_ids_logprob_only(self) -> bool:
        if bool(
            getattr(
                self.server_args,
                "multi_item_score_direct_token_ids_logprob_only",
                False,
            )
        ):
            return True
        if not bool(
            getattr(
                self.server_args,
                "multi_item_score_direct_token_ids_logprob_only_auto",
                False,
            )
        ):
            return False

        max_page_size = max(
            0,
            int(
                getattr(
                    self.server_args,
                    "multi_item_score_direct_token_ids_logprob_only_auto_max_page_size",
                    0,
                )
                or 0
            ),
        )
        max_running_requests = max(
            0,
            int(getattr(self.server_args, "max_running_requests", 0) or 0),
        )
        max_running_requests_threshold = max(
            0,
            int(
                getattr(
                    self.server_args,
                    "multi_item_score_direct_token_ids_logprob_only_auto_max_running_requests",
                    0,
                )
                or 0
            ),
        )

        return bool(
            (max_page_size > 0 and int(self.page_size) <= max_page_size)
            or (
                max_running_requests_threshold > 0
                and max_running_requests > 0
                and max_running_requests <= max_running_requests_threshold
            )
        )

    def _score_from_cache_v2_resolve_direct_token_ids_logprob_only_chunk_size(
        self,
        *,
        direct_token_ids_logprob_only: bool,
        real_bs: int,
        prefix_len: int,
        max_seq_len: int,
    ) -> int:
        chunk_size = max(
            1,
            int(
                getattr(
                    self.server_args,
                    "multi_item_score_direct_token_ids_logprob_only_chunk_size",
                    4096,
                )
                or 4096
            ),
        )
        if not direct_token_ids_logprob_only:
            return chunk_size

        short_prompt_tokens_threshold = max(
            0,
            int(
                getattr(
                    self.server_args,
                    "score_scheduler_short_prompt_tokens_threshold",
                    2048,
                )
                or 2048
            ),
        )
        if (
            short_prompt_tokens_threshold > 0
            and prefix_len <= short_prompt_tokens_threshold
            and real_bs >= SCORE_V2_DIRECT_TOKEN_IDS_LOGPROB_ONLY_LARGE_CHUNK_MIN_BS
            and max_seq_len <= SCORE_V2_DIRECT_TOKEN_IDS_LOGPROB_ONLY_LARGE_CHUNK_MAX_SEQ_LEN
        ):
            return max(
                chunk_size,
                SCORE_V2_DIRECT_TOKEN_IDS_LOGPROB_ONLY_LARGE_CHUNK_SIZE,
            )
        return chunk_size

    def _score_from_cache_v2_resolve_direct_hot_shape(
        self,
        *,
        real_bs: int,
        real_input_tokens: int,
        real_cache_loc_tokens: int,
        max_seq_len: int,
    ) -> tuple[int, int, int]:
        hot_bs = max(
            0,
            int(getattr(self.server_args, "multi_item_score_direct_hot_shape_bs", 0) or 0),
        )
        hot_tokens = max(
            0,
            int(getattr(self.server_args, "multi_item_score_direct_hot_shape_tokens", 0) or 0),
        )
        hot_token_rounding = max(
            0,
            int(
                getattr(
                    self.server_args,
                    "multi_item_score_direct_hot_shape_token_rounding",
                    0,
                )
                or 0
            ),
        )
        hot_token_rounding_min_hot_tokens = max(
            0,
            int(
                getattr(
                    self.server_args,
                    "multi_item_score_direct_hot_shape_token_rounding_min_hot_tokens",
                    0,
                )
                or 0
            ),
        )

        padded_bs = max(int(real_bs), hot_bs)
        padded_input_tokens = max(int(real_input_tokens), hot_tokens)
        if (
            hot_token_rounding > 0
            and padded_input_tokens > int(real_input_tokens)
            and (
                hot_token_rounding_min_hot_tokens <= 0
                or hot_tokens >= hot_token_rounding_min_hot_tokens
            )
        ):
            rounded_real_input_tokens = (
                (max(1, int(real_input_tokens)) + hot_token_rounding - 1) // hot_token_rounding
            ) * hot_token_rounding
            if int(padded_input_tokens) - int(rounded_real_input_tokens) >= hot_token_rounding:
                padded_input_tokens = max(
                    int(real_input_tokens),
                    min(int(padded_input_tokens), int(rounded_real_input_tokens)),
                )
        padded_cache_loc_tokens = int(real_cache_loc_tokens)
        if hot_bs > 0:
            aligned_hot_seq_len = (
                (max(1, int(max_seq_len)) + self.page_size - 1) // self.page_size
            ) * self.page_size
            padded_cache_loc_tokens = max(
                padded_cache_loc_tokens,
                int(hot_bs) * int(aligned_hot_seq_len),
            )

        return padded_bs, padded_input_tokens, padded_cache_loc_tokens

    def _score_direct_warmup_spec(self) -> SimpleNamespace | None:
        if not self._score_from_cache_v2_use_direct_label_only(
            label_only_logprob=bool(
                getattr(self.server_args, "multi_item_score_label_only_logprob", False)
            )
        ):
            return None
        if not bool(getattr(self.server_args, "multi_item_score_direct_warmup_enable", False)):
            return None

        batch_size = max(
            0,
            int(getattr(self.server_args, "multi_item_score_direct_warmup_batch_size", 0) or 0),
        )
        if batch_size <= 0:
            batch_size = max(
                0,
                int(getattr(self.server_args, "multi_item_score_direct_hot_shape_bs", 0) or 0),
            )
        if batch_size <= 0:
            batch_size = max(
                0,
                int(
                    getattr(self.server_args, "multi_item_score_from_cache_v2_items_per_step", 0)
                    or 0
                ),
            )

        return SimpleNamespace(
            prefix_len=max(
                0,
                int(getattr(self.server_args, "multi_item_score_direct_warmup_prefix_len", 0) or 0),
            ),
            item_len=max(
                0,
                int(getattr(self.server_args, "multi_item_score_direct_warmup_item_len", 0) or 0),
            ),
            batch_size=batch_size,
            label_count=max(
                1,
                int(
                    getattr(self.server_args, "multi_item_score_direct_warmup_label_count", 1) or 1
                ),
            ),
            apply_softmax=bool(
                getattr(self.server_args, "multi_item_score_direct_warmup_apply_softmax", False)
            ),
        )

    def _score_direct_warmup_token_ids(self, length: int, *, offset: int) -> list[int]:
        if length <= 0:
            return []
        vocab_size = max(2, int(self.model_config.vocab_size))
        eos_token_ids = set(getattr(self.model_config, "hf_eos_token_id", set()) or set())
        token_ids: list[int] = []
        candidate = max(1, 1024 + offset * 131)
        while len(token_ids) < length:
            token_id = int(candidate % vocab_size)
            candidate += 1
            if token_id in eos_token_ids:
                continue
            token_ids.append(token_id)
        return token_ids

    def _score_direct_warmup_label_token_ids(self, label_count: int) -> list[int]:
        vocab_size = max(2, int(self.model_config.vocab_size))
        eos_token_ids = set(getattr(self.model_config, "hf_eos_token_id", set()) or set())
        label_ids: list[int] = []
        seen: set[int] = set()
        candidate = 17
        while len(label_ids) < label_count:
            token_id = int(candidate % vocab_size)
            candidate += 1
            if token_id in eos_token_ids or token_id in seen:
                continue
            seen.add(token_id)
            label_ids.append(token_id)
        return label_ids

    def _materialize_score_direct_warmup_prefix(
        self,
        prefix_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        prefix_len = len(prefix_ids)
        if prefix_len <= 0:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty

        aligned_prefix_tokens = int(prefix_len)
        if self.page_size > 1:
            aligned_prefix_tokens = (
                (aligned_prefix_tokens + self.page_size - 1) // self.page_size
            ) * self.page_size

        prefix_alloc = np.asarray(
            alloc_token_slots(self.tree_cache, aligned_prefix_tokens),
            dtype=np.int32,
        )
        prefix_cache_loc = prefix_alloc[:prefix_len].astype(np.int32, copy=True)

        input_ids_cpu = np.asarray(prefix_ids, dtype=np.int32)
        positions_cpu = np.arange(prefix_len, dtype=np.int32)
        seq_lens_cpu = np.asarray([prefix_len], dtype=np.int32)
        extend_seq_lens_cpu = np.asarray([prefix_len], dtype=np.int32)
        extend_prefix_lens_cpu = np.zeros(1, dtype=np.int32)
        extend_start_loc = np.zeros(1, dtype=np.int32)
        extend_logprob_start_lens = np.zeros(1, dtype=np.int32)
        cache_loc_cpu = np.zeros(aligned_prefix_tokens, dtype=np.int32)
        cache_loc_cpu[:prefix_len] = prefix_cache_loc

        batch = ModelWorkerBatch(
            bid=acc_global_bid(),
            forward_mode=ForwardMode.EXTEND,
            input_ids=input_ids_cpu,
            real_input_ids_len=prefix_len,
            seq_lens=seq_lens_cpu,
            out_cache_loc=prefix_cache_loc,
            req_pool_indices=np.asarray([0], dtype=np.int32),
            sampling_info=SamplingBatchInfo.generate_for_precompile_all_greedy(
                1,
                vocab_size=self.model_config.vocab_size,
            ),
            positions=positions_cpu,
            extend_start_loc=extend_start_loc,
            cache_loc=cache_loc_cpu,
            return_logprob=False,
            return_output_logprob_only=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            is_prefill_only=True,
            extend_seq_lens=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens_cpu,
            extend_logprob_start_lens=extend_logprob_start_lens,
            extend_input_logprob_token_ids=np.empty((0,), dtype=np.int32),
            real_bs=1,
            lora_ids=["0"],
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        logits_output, _, _ = self.tp_worker.forward_batch_generation(
            model_worker_batch=batch,
            launch_done=None,
            skip_sample=True,
            sampling_metadata=None,
        )
        if logits_output is not None and logits_output.next_token_logits is not None:
            logits_output.next_token_logits.block_until_ready()
        return prefix_cache_loc, prefix_alloc

    def _run_score_direct_label_only_warmup(self) -> None:
        warmup = self._score_direct_warmup_spec()
        if warmup is None:
            return

        prefix_ids = self._score_direct_warmup_token_ids(warmup.prefix_len, offset=0)
        chunk_items = [
            self._score_direct_warmup_token_ids(warmup.item_len, offset=idx + 1)
            for idx in range(warmup.batch_size)
        ]
        label_token_ids = self._score_direct_warmup_label_token_ids(warmup.label_count)
        label_token_ids_arr = jnp.asarray(label_token_ids, dtype=jnp.int32)

        prefix_alloc = np.empty((0,), dtype=np.int32)
        prefix_indices = np.empty((0,), dtype=np.int32)
        warmup_start = time.perf_counter()
        try:
            prefix_indices, prefix_alloc = self._materialize_score_direct_warmup_prefix(prefix_ids)
            scores, device_compute_s, host_overhead_s = (
                self._run_score_from_cache_v2_direct_chunk_label_only(
                    cache_handle="__score-direct-warmup__",
                    chunk_items=chunk_items,
                    label_token_ids=label_token_ids,
                    label_token_ids_arr=label_token_ids_arr,
                    apply_softmax=warmup.apply_softmax,
                    cached_last_node=None,
                    cached_prefix_indices=prefix_indices,
                    prefix_ids=prefix_ids,
                    cached_extra_key=None,
                )
            )
            logger.info(
                "[Scheduler] Direct bulk score warmup complete. prefix_len=%d batch=%d item_len=%d labels=%d apply_softmax=%s total_s=%.3f device_s=%.3f host_s=%.3f score_rows=%d",
                warmup.prefix_len,
                warmup.batch_size,
                warmup.item_len,
                warmup.label_count,
                warmup.apply_softmax,
                max(0.0, time.perf_counter() - warmup_start),
                max(0.0, device_compute_s),
                max(0.0, host_overhead_s),
                len(scores),
            )
        finally:
            if prefix_alloc.size > 0:
                self.token_to_kv_pool_allocator.free(prefix_alloc)

    def _resolve_score_from_cache_v2_items_per_step(
        self,
        *,
        requested_items_per_step: int,
        default_items_per_step: int,
        effective_capacity: int,
        total_items: int,
        lane_name: str,
    ) -> int:
        base_items_per_step = max(
            1,
            min(
                requested_items_per_step,
                default_items_per_step,
                effective_capacity,
                max(1, total_items),
            ),
        )
        if not bool(getattr(self, "score_scheduler_dynamic_items_per_step_enable", False)):
            return base_items_per_step

        queue_pressure = self._score_scheduler_queue_pressure(self)
        pressure_threshold = max(
            1,
            int(
                getattr(
                    self,
                    "score_scheduler_dynamic_items_per_step_pressure_threshold",
                    64,
                )
                or 64
            ),
        )
        pressure_scale = min(1.0, float(pressure_threshold) / float(max(queue_pressure, 1)))
        if lane_name == "long":
            lane_bias = max(
                0.1,
                float(
                    getattr(
                        self,
                        "score_scheduler_dynamic_items_per_step_long_lane_bias",
                        0.75,
                    )
                    or 0.75
                ),
            )
            lane_floor = max(
                1,
                int(
                    getattr(
                        self,
                        "score_scheduler_dynamic_items_per_step_long_lane_min",
                        16,
                    )
                    or 16
                ),
            )
        else:
            lane_bias = max(
                0.1,
                float(
                    getattr(
                        self,
                        "score_scheduler_dynamic_items_per_step_short_lane_bias",
                        1.0,
                    )
                    or 1.0
                ),
            )
            lane_floor = max(
                1,
                int(
                    getattr(
                        self,
                        "score_scheduler_dynamic_items_per_step_short_lane_min",
                        32,
                    )
                    or 32
                ),
            )
        lane_floor = min(lane_floor, base_items_per_step)
        target_items_per_step = int(
            math.ceil(float(base_items_per_step) * pressure_scale * lane_bias)
        )
        effective_items_per_step = max(
            1,
            min(base_items_per_step, max(lane_floor, target_items_per_step)),
        )

        self.score_scheduler_dynamic_items_per_step_requests = (
            int(getattr(self, "score_scheduler_dynamic_items_per_step_requests", 0)) + 1
        )
        self.score_scheduler_dynamic_items_per_step_requested_total = (
            int(getattr(self, "score_scheduler_dynamic_items_per_step_requested_total", 0))
            + base_items_per_step
        )
        self.score_scheduler_dynamic_items_per_step_effective_total = (
            int(getattr(self, "score_scheduler_dynamic_items_per_step_effective_total", 0))
            + effective_items_per_step
        )
        self.score_scheduler_dynamic_items_per_step_max_queue_pressure = max(
            int(getattr(self, "score_scheduler_dynamic_items_per_step_max_queue_pressure", 0)),
            queue_pressure,
        )
        requested_by_lane = self._lane_counter(
            self,
            "score_scheduler_dynamic_items_per_step_requested_by_lane",
        )
        requested_by_lane[lane_name] = requested_by_lane.get(lane_name, 0) + base_items_per_step
        effective_by_lane = self._lane_counter(
            self,
            "score_scheduler_dynamic_items_per_step_effective_by_lane",
        )
        effective_by_lane[lane_name] = (
            effective_by_lane.get(lane_name, 0) + effective_items_per_step
        )
        if effective_items_per_step != base_items_per_step:
            applied_by_lane = self._lane_counter(
                self,
                "score_scheduler_dynamic_items_per_step_applied_by_lane",
            )
            applied_by_lane[lane_name] = applied_by_lane.get(lane_name, 0) + 1

        return effective_items_per_step

    def _record_score_from_cache_v2_fallback(self, reason: str):
        self.score_from_cache_v2_fallback += 1
        self.score_from_cache_v2_fallback_reasons[reason] = (
            self.score_from_cache_v2_fallback_reasons.get(reason, 0) + 1
        )

    def _record_score_from_cache_v2_timing(
        self,
        queue_wait_s: float,
        device_compute_s: float,
        host_orchestration_s: float,
    ) -> None:
        queue_wait_s = max(0.0, float(queue_wait_s))
        device_compute_s = max(0.0, float(device_compute_s))
        host_orchestration_s = max(0.0, float(host_orchestration_s))
        self.score_from_cache_v2_queue_wait_s_total += queue_wait_s
        self.score_from_cache_v2_device_compute_s_total += device_compute_s
        self.score_from_cache_v2_host_orchestration_s_total += host_orchestration_s
        self.score_from_cache_v2_queue_wait_s_max = max(
            self.score_from_cache_v2_queue_wait_s_max,
            queue_wait_s,
        )
        self.score_from_cache_v2_device_compute_s_max = max(
            self.score_from_cache_v2_device_compute_s_max,
            device_compute_s,
        )
        self.score_from_cache_v2_host_orchestration_s_max = max(
            self.score_from_cache_v2_host_orchestration_s_max,
            host_orchestration_s,
        )

    def _score_from_cache_v2_fallback_output(
        self,
        recv_req: ScoreFromCacheReqInput,
        reason: str,
        error_msg: str = "",
        dispatch_count: int = 0,
        queue_wait_s: float = 0.0,
        device_compute_s: float = 0.0,
        host_orchestration_s: float = 0.0,
    ) -> ScoreFromCacheReqOutput:
        self._record_score_from_cache_v2_fallback(reason)
        self._record_score_from_cache_v2_timing(
            queue_wait_s=queue_wait_s,
            device_compute_s=device_compute_s,
            host_orchestration_s=host_orchestration_s,
        )
        return ScoreFromCacheReqOutput(
            rid=recv_req.rid,
            success=False,
            scores=[],
            fallback_reason=reason,
            error_msg=error_msg,
            dispatch_count=dispatch_count,
            lifecycle_requests_sent=0,
            lifecycle_results_received=0,
            queue_wait_s=max(0.0, float(queue_wait_s)),
            device_compute_s=device_compute_s,
            host_orchestration_s=host_orchestration_s,
        )

    def _score_from_cache_v2_validate_items(
        self, recv_req: ScoreFromCacheReqInput
    ) -> tuple[bool, str, str]:
        if not recv_req.cache_handle:
            return False, "missing_cache_handle", "cache_handle must be non-empty."
        if not isinstance(recv_req.items_2d, list):
            return False, "unsupported_shape", "items_2d must be a list of token lists."
        if not isinstance(recv_req.label_token_ids, list) or len(recv_req.label_token_ids) == 0:
            return False, "unsupported_shape", "label_token_ids must be a non-empty list."
        if any((not isinstance(token_id, int)) for token_id in recv_req.label_token_ids):
            return False, "unsupported_shape", "label_token_ids must contain ints."
        for token_id in recv_req.label_token_ids:
            if token_id < 0 or token_id >= self.model_config.vocab_size:
                return (
                    False,
                    "unsupported_shape",
                    f"label_token_ids must be in [0, {self.model_config.vocab_size - 1}].",
                )
        for idx, item in enumerate(recv_req.items_2d):
            if not isinstance(item, list):
                return (
                    False,
                    "unsupported_shape",
                    f"items_2d[{idx}] must be a list of token ids.",
                )
            if len(item) == 0:
                return (
                    False,
                    "unsupported_shape",
                    f"items_2d[{idx}] must contain at least one token.",
                )
            if any((not isinstance(token_id, int)) for token_id in item):
                return (
                    False,
                    "unsupported_shape",
                    f"items_2d[{idx}] must contain ints.",
                )
        return True, "", ""

    @staticmethod
    def _score_from_cache_v2_probs_from_logprobs(
        row_logprobs: list[float], apply_softmax: bool
    ) -> list[float]:
        if apply_softmax:
            finite_vals = [x for x in row_logprobs if x != float("-inf")]
            if not finite_vals:
                return [0.0 for _ in row_logprobs]
            max_logprob = max(finite_vals)
            exps = [math.exp(x - max_logprob) if x != float("-inf") else 0.0 for x in row_logprobs]
            denom = sum(exps)
            if denom <= 0:
                return [0.0 for _ in row_logprobs]
            return [x / denom for x in exps]
        return [math.exp(x) if x != float("-inf") else 0.0 for x in row_logprobs]

    @staticmethod
    def _label_only_parity_metrics(
        baseline_logprobs: np.ndarray,
        candidate_logprobs: np.ndarray,
    ) -> tuple[float, float]:
        if baseline_logprobs.shape != candidate_logprobs.shape:
            return float("inf"), float("inf")
        diffs = np.abs(baseline_logprobs.astype(np.float64) - candidate_logprobs.astype(np.float64))
        if diffs.size == 0:
            return 0.0, 0.0
        return float(np.max(diffs)), float(np.mean(diffs))

    @staticmethod
    def _estimate_score_from_cache_v2_words(prefix_len: int, items: list[list[int]]) -> int:
        # Conservative host-side int32-sized tensor estimate for this chunk.
        total_item_tokens = sum(len(item) for item in items)
        total_fill_tokens = sum(prefix_len + len(item) for item in items)
        max_item_len = max((len(item) for item in items), default=0)
        bs = len(items)
        # Terms loosely track main arrays: flat input ids, seq/prefix/extend lengths,
        # req_to_token writes, and token-id-logprob tensors.
        return (
            total_item_tokens
            + total_fill_tokens
            + (3 * bs)
            + (bs * max_item_len)
            + (bs * prefix_len)
        )

    @staticmethod
    def _build_score_from_cache_v2_chunk_plan(
        items_2d: list[list[int]],
        items_per_step: int,
        *,
        prefix_len: int = 0,
        token_budget: int = 0,
    ) -> list[tuple[list[int], list[list[int]]]]:
        if items_per_step <= 0:
            items_per_step = 1

        # Stable length-aware packing: longer items first, original order tie-break.
        indexed_items = list(enumerate(items_2d))
        indexed_items.sort(key=lambda pair: (-len(pair[1]), pair[0]))

        dispatch_token_budget = max(0, int(token_budget or 0))
        if dispatch_token_budget > 0:
            chunk_plan: list[tuple[list[int], list[list[int]]]] = []
            chunk_token_totals: list[int] = []
            for idx, item in indexed_items:
                item_total_tokens = max(1, int(prefix_len) + len(item))
                placed = False
                for chunk_idx, (chunk_indices, chunk_items) in enumerate(chunk_plan):
                    if len(chunk_items) >= items_per_step:
                        continue
                    if chunk_token_totals[chunk_idx] + item_total_tokens > dispatch_token_budget:
                        continue
                    chunk_indices.append(idx)
                    chunk_items.append(item)
                    chunk_token_totals[chunk_idx] += item_total_tokens
                    placed = True
                    break
                if placed:
                    continue
                chunk_plan.append(([idx], [item]))
                chunk_token_totals.append(item_total_tokens)
            return chunk_plan

        chunk_plan: list[tuple[list[int], list[list[int]]]] = []
        for start in range(0, len(indexed_items), items_per_step):
            chunk_pairs = indexed_items[start : start + items_per_step]
            chunk_indices = [idx for idx, _ in chunk_pairs]
            chunk_items = [item for _, item in chunk_pairs]
            chunk_plan.append((chunk_indices, chunk_items))
        return chunk_plan

    def _release_score_from_cache_v2_chunk_reqs(
        self,
        reqs: list[Req],
        batch: ScheduleBatch | None = None,
    ) -> None:
        if batch is not None:
            try:
                out_cache_loc = getattr(batch, "out_cache_loc", None)
                if out_cache_loc is not None:
                    out_cache_loc_arr = np.asarray(out_cache_loc, dtype=np.int32)
                    if out_cache_loc_arr.size > 0:
                        self.token_to_kv_pool_allocator.free(out_cache_loc_arr)
            except Exception:
                logger.exception("Fastpath v2 cleanup failed while freeing chunk KV slots.")

            try:
                req_pool_indices = getattr(batch, "req_pool_indices", None)
                if req_pool_indices is not None:
                    req_pool_indices_list = np.asarray(req_pool_indices, dtype=np.int32).tolist()
                    if req_pool_indices_list:
                        self.req_to_token_pool.free(req_pool_indices_list)
            except Exception:
                logger.exception("Fastpath v2 cleanup failed while freeing chunk req slots.")

            for req in reqs:
                req.req_pool_idx = None
            return

        for req in reqs:
            if req.req_pool_idx is None:
                continue
            try:
                pre_len = len(req.prefix_indices)
                seq_len = pre_len + max(0, req.extend_input_len)
                if seq_len > 0:
                    token_locs = self.req_to_token_pool.read(req.req_pool_idx, seq_len)
                    token_locs = token_locs[pre_len:seq_len]
                    token_locs = token_locs[token_locs != 0]
                    if len(token_locs) > 0:
                        self.token_to_kv_pool_allocator.free(token_locs)
            except Exception:
                logger.exception(
                    "Fastpath v2 cleanup failed while freeing KV tokens for rid=%s.",
                    req.rid,
                )
            try:
                self.req_to_token_pool.free(req.req_pool_idx)
            except Exception:
                logger.exception(
                    "Fastpath v2 cleanup failed while freeing req slot for rid=%s.",
                    req.rid,
                )
            req.req_pool_idx = None

    def _run_score_from_cache_v2_chunk(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
    ) -> tuple[list[list[float]], float, float]:
        batch: ScheduleBatch | None = None
        reqs = self._build_score_from_cache_v2_chunk_reqs(
            cache_handle=cache_handle,
            chunk_items=chunk_items,
            label_token_ids=label_token_ids,
            cached_last_node=cached_last_node,
            cached_prefix_indices=cached_prefix_indices,
            prefix_ids=prefix_ids,
            cached_extra_key=cached_extra_key,
            return_label_logprobs=True,
        )

        try:
            batch = ScheduleBatch.init_new(
                reqs=reqs,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                model_config=self.model_config,
                enable_overlap=self.enable_overlap,
                spec_algorithm=self.spec_algorithm,
                enable_custom_logit_processor=False,
                chunked_req=None,
                mesh=self.mesh,
            )
            batch.prepare_for_extend()
            batch.bid = acc_global_bid()
            result = self.run_batch(batch)

            if result.logits_output is None:
                raise RuntimeError("Missing logits output from score-from-cache v2 chunk.")

            logprob_vals = result.logits_output.next_token_token_ids_logprobs_val
            logprob_idxs = result.logits_output.next_token_token_ids_logprobs_idx
            if logprob_vals is None or logprob_idxs is None:
                raise RuntimeError(
                    "Missing token_ids_logprobs tensors from score-from-cache v2 chunk."
                )
            logprob_vals = np.asarray(jax.device_get(logprob_vals), dtype=np.float64)
            logprob_idxs = np.asarray(jax.device_get(logprob_idxs), dtype=np.int32)
            if logprob_vals.ndim != 2 or logprob_idxs.shape != logprob_vals.shape:
                raise RuntimeError(
                    f"Unexpected token_ids_logprobs shape: vals={logprob_vals.shape}, idxs={logprob_idxs.shape}."
                )
            if logprob_vals.shape[0] != len(reqs):
                raise RuntimeError(
                    f"Chunk output rows ({logprob_vals.shape[0]}) != request count ({len(reqs)})."
                )

            scores: list[list[float]] = []
            for row_vals, row_idxs in zip(logprob_vals, logprob_idxs):
                row_logprobs: list[float] = []
                for token_id in label_token_ids:
                    match = np.where(row_idxs == token_id)[0]
                    if len(match) == 0:
                        row_logprobs.append(float("-inf"))
                    else:
                        row_logprobs.append(float(row_vals[int(match[0])]))
                scores.append(
                    self._score_from_cache_v2_probs_from_logprobs(
                        row_logprobs=row_logprobs,
                        apply_softmax=apply_softmax,
                    )
                )

            chunk_device_compute_s = reqs[0].device_compute_time_s if reqs else 0.0
            chunk_host_overhead_s = reqs[0].host_overhead_time_s if reqs else 0.0
            return scores, chunk_device_compute_s, chunk_host_overhead_s
        finally:
            self._release_score_from_cache_v2_chunk_reqs(reqs, batch=batch)

    def _build_score_from_cache_v2_chunk_reqs(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
        return_label_logprobs: bool,
    ) -> list[Req]:
        reqs: list[Req] = []
        chunk_uid = time.time_ns()
        for local_idx, item_ids in enumerate(chunk_items):
            sampling_params = SamplingParams(max_new_tokens=0)
            sampling_params.stop_strs = []
            sampling_params.stop_str_max_len = 0

            rid = f"{cache_handle}-scorev2-{chunk_uid}-{local_idx}"
            req = Req(
                rid=rid,
                origin_input_text=None,
                origin_input_ids=prefix_ids + item_ids,
                sampling_params=sampling_params,
                return_logprob=return_label_logprobs,
                return_output_logprob_only=False,
                top_logprobs_num=0,
                token_ids_logprob=label_token_ids if return_label_logprobs else None,
                stream=False,
                extra_key=cached_extra_key,
                eos_token_ids=self.model_config.hf_eos_token_id,
                vocab_size=self.model_config.vocab_size,
                cache_for_scoring=False,
                extend_from_cache=cache_handle,
            )
            req.tokenizer = self.tokenizer
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.cached_last_node = cached_last_node
            req.cached_last_host_node = cached_last_node
            req.cached_prefix_indices = cached_prefix_indices
            req.cached_host_hit_length = 0

            error_msg = validate_input_length(
                req,
                self.max_req_input_len,
                self.server_args.allow_auto_truncate,
            )
            if error_msg:
                raise ValueError(error_msg)
            req.init_next_round_input(self.tree_cache)
            reqs.append(req)
        return reqs
