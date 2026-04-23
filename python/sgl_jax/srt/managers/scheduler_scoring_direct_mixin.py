"""Scheduler label-only scoring kernels."""

from sgl_jax.srt.managers.scheduler_scoring_common import *


class SchedulerScoringDirectMixin:
    def _run_score_from_cache_v2_direct_chunk_label_only(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        label_token_ids_arr: jax.Array,
        apply_softmax: bool,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
    ) -> tuple[jax.Array, float, float]:
        batch: ModelWorkerBatch | None = None
        out_cache_loc: np.ndarray | None = None
        try:
            chunk_wall_start = time.perf_counter()
            direct_token_ids_logprob_only = (
                self._score_from_cache_v2_use_direct_token_ids_logprob_only()
            )
            prefix_indices_np = np.asarray(cached_prefix_indices, dtype=np.int32)
            prefix_len = int(prefix_indices_np.shape[0])
            real_bs = len(chunk_items)
            extend_lens = np.asarray([len(item) for item in chunk_items], dtype=np.int32)
            if real_bs == 0:
                return [], 0.0, 0.0

            extend_num_tokens = int(np.sum(extend_lens, dtype=np.int64))
            if extend_num_tokens <= 0:
                raise RuntimeError("Direct score-from-cache v2 chunk has no extend tokens.")

            seq_lens = extend_lens + prefix_len
            max_seq_len = int(np.max(seq_lens))
            direct_token_ids_logprob_only_chunk_size = (
                self._score_from_cache_v2_resolve_direct_token_ids_logprob_only_chunk_size(
                    direct_token_ids_logprob_only=direct_token_ids_logprob_only,
                    real_bs=real_bs,
                    prefix_len=prefix_len,
                    max_seq_len=max_seq_len,
                )
            )

            if self.page_size == 1:
                out_cache_loc = alloc_token_slots(self.tree_cache, extend_num_tokens)
            else:
                last_loc = np.full(
                    real_bs,
                    int(prefix_indices_np[prefix_len - 1]) if prefix_len > 0 else 0,
                    dtype=np.int32,
                )
                out_cache_loc = alloc_paged_token_slots_extend(
                    self.tree_cache,
                    [prefix_len] * real_bs,
                    seq_lens.tolist(),
                    last_loc.tolist(),
                    extend_num_tokens,
                )
            out_cache_loc = np.asarray(out_cache_loc, dtype=np.int32)

            input_ids_cpu = np.empty(extend_num_tokens, dtype=np.int32)
            positions_cpu = np.empty(extend_num_tokens, dtype=np.int32)
            extend_start_loc = np.empty(real_bs, dtype=np.int32)
            token_pt = 0
            for req_idx, (item, extend_len) in enumerate(
                zip(chunk_items, extend_lens, strict=True)
            ):
                extend_len_i = int(extend_len)
                extend_start_loc[req_idx] = token_pt
                if extend_len_i <= 0:
                    continue
                item_arr = np.asarray(item, dtype=np.int32)
                next_token_pt = token_pt + extend_len_i
                input_ids_cpu[token_pt:next_token_pt] = item_arr
                positions_cpu[token_pt:next_token_pt] = np.arange(
                    prefix_len,
                    prefix_len + extend_len_i,
                    dtype=np.int32,
                )
                token_pt = next_token_pt

            aligned_seq_lens = ((seq_lens + self.page_size - 1) // self.page_size) * self.page_size
            real_cache_loc_tokens = int(np.sum(aligned_seq_lens, dtype=np.int64))
            padded_bs, padded_input_tokens, padded_cache_loc_tokens = (
                self._score_from_cache_v2_resolve_direct_hot_shape(
                    real_bs=real_bs,
                    real_input_tokens=int(input_ids_cpu.shape[0]),
                    real_cache_loc_tokens=real_cache_loc_tokens,
                    max_seq_len=max_seq_len,
                )
            )

            cache_loc_cpu = np.zeros(padded_cache_loc_tokens, dtype=np.int32)
            token_pt = 0
            cache_pt = 0
            for extend_len, aligned_len in zip(extend_lens, aligned_seq_lens, strict=True):
                extend_len_i = int(extend_len)
                aligned_len_i = int(aligned_len)
                if prefix_len > 0:
                    cache_loc_cpu[cache_pt : cache_pt + prefix_len] = prefix_indices_np
                if extend_len_i > 0:
                    cache_loc_cpu[cache_pt + prefix_len : cache_pt + prefix_len + extend_len_i] = (
                        out_cache_loc[token_pt : token_pt + extend_len_i]
                    )
                token_pt += extend_len_i
                cache_pt += aligned_len_i

            if padded_input_tokens > input_ids_cpu.shape[0]:
                pad = padded_input_tokens - input_ids_cpu.shape[0]
                input_ids_cpu = np.concatenate(
                    [input_ids_cpu, np.zeros(pad, dtype=np.int32)],
                    axis=0,
                )
                positions_cpu = np.concatenate(
                    [positions_cpu, np.zeros(pad, dtype=np.int32)],
                    axis=0,
                )
                out_cache_loc = np.concatenate(
                    [out_cache_loc, np.full(pad, -1, dtype=np.int32)],
                    axis=0,
                )

            seq_lens_cpu = seq_lens.astype(np.int32, copy=False)
            extend_seq_lens_cpu = extend_lens.astype(np.int32, copy=False)
            extend_prefix_lens_cpu = np.full(real_bs, prefix_len, dtype=np.int32)
            req_pool_indices = np.arange(real_bs, dtype=np.int32)
            extend_logprob_start_lens = np.zeros(real_bs, dtype=np.int32)

            if padded_bs > real_bs:
                bs_pad = padded_bs - real_bs
                seq_lens_cpu = np.concatenate(
                    [seq_lens_cpu, np.zeros(bs_pad, dtype=np.int32)],
                    axis=0,
                )
                extend_seq_lens_cpu = np.concatenate(
                    [extend_seq_lens_cpu, np.zeros(bs_pad, dtype=np.int32)],
                    axis=0,
                )
                extend_prefix_lens_cpu = np.concatenate(
                    [extend_prefix_lens_cpu, np.zeros(bs_pad, dtype=np.int32)],
                    axis=0,
                )
                extend_start_loc = np.concatenate(
                    [
                        extend_start_loc,
                        np.full(bs_pad, extend_num_tokens, dtype=np.int32),
                    ],
                    axis=0,
                )
                req_pool_indices = np.concatenate(
                    [req_pool_indices, np.full(bs_pad, -1, dtype=np.int32)],
                    axis=0,
                )
                extend_logprob_start_lens = np.concatenate(
                    [extend_logprob_start_lens, np.zeros(bs_pad, dtype=np.int32)],
                    axis=0,
                )

            batch = ModelWorkerBatch(
                bid=acc_global_bid(),
                forward_mode=ForwardMode.EXTEND,
                input_ids=input_ids_cpu,
                real_input_ids_len=extend_num_tokens,
                seq_lens=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                req_pool_indices=req_pool_indices,
                sampling_info=SamplingBatchInfo.generate_for_precompile_all_greedy(
                    padded_bs,
                    vocab_size=self.model_config.vocab_size,
                ),
                positions=positions_cpu,
                extend_start_loc=extend_start_loc,
                cache_loc=cache_loc_cpu,
                return_logprob=False,
                return_output_logprob_only=False,
                top_logprobs_nums=None,
                token_ids_logprobs=(
                    None if direct_token_ids_logprob_only else [list(label_token_ids)] * padded_bs
                ),
                is_prefill_only=True,
                extend_seq_lens=extend_seq_lens_cpu,
                extend_prefix_lens=extend_prefix_lens_cpu,
                extend_logprob_start_lens=extend_logprob_start_lens,
                extend_input_logprob_token_ids=np.empty((0,), dtype=np.int32),
                real_bs=real_bs,
                lora_ids=["0"] * padded_bs,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                next_token_token_ids_logprob_only=direct_token_ids_logprob_only,
                next_token_token_ids_logprob_only_chunk_size=direct_token_ids_logprob_only_chunk_size,
                next_token_shared_token_ids=(
                    np.asarray(label_token_ids, dtype=np.int32)
                    if direct_token_ids_logprob_only
                    else None
                ),
            )

            forward_start = time.perf_counter()
            logits_output, _, _ = self.tp_worker.forward_batch_generation(
                model_worker_batch=batch,
                launch_done=None,
                skip_sample=True,
                sampling_metadata=None,
            )

            direct_label_logprobs = None
            if (
                logits_output is not None
                and logits_output.next_token_token_ids_logprobs_val is not None
            ):
                direct_label_logprobs = logits_output.next_token_token_ids_logprobs_val[:real_bs, :]
            elif (
                logits_output is None
                or logits_output.next_token_logits is None
                or logits_output.next_token_logits.shape[-1] == 0
            ):
                raise RuntimeError(
                    "Missing score tensors from direct score-from-cache v2 label-only chunk."
                )

            next_token_logits = (
                None
                if direct_label_logprobs is not None
                else logits_output.next_token_logits[:real_bs, :]
            )
            out_sharding = NamedSharding(self.mesh, P(None, None))
            legacy_kernel_mode = SCORE_V2_LABEL_ONLY_KERNEL_MODE or "baseline"
            if legacy_kernel_mode not in {"baseline", "", "log_softmax"}:
                raise RuntimeError(
                    "Unsupported SGLANG_SCORE_LABEL_ONLY_KERNEL_MODE="
                    f"{SCORE_V2_LABEL_ONLY_KERNEL_MODE!r}."
                )

            fused_kernel_enabled = bool(
                getattr(self.server_args, "multi_item_score_label_only_fused_kernel", True)
            )
            if legacy_kernel_mode not in {"baseline", ""}:
                fused_kernel_enabled = False

            if direct_label_logprobs is not None:
                self.score_label_only_token_ids_only_calls += 1
                scores_dev = _compute_label_only_scores_from_logprobs(
                    direct_label_logprobs,
                    bool(apply_softmax),
                )
                scores_dev.block_until_ready()
                forward_end = time.perf_counter()
            elif fused_kernel_enabled:
                self.score_label_only_fused_kernel_calls += 1
                if apply_softmax:
                    self.score_label_only_fused_kernel_softmax_calls += 1
                scores_dev = _compute_label_only_scores_fused(
                    next_token_logits,
                    label_token_ids_arr,
                    bool(apply_softmax),
                    out_sharding,
                )
                scores_dev.block_until_ready()
                forward_end = time.perf_counter()
            else:
                self.score_label_only_legacy_kernel_calls += 1
                if legacy_kernel_mode == "log_softmax":
                    row_logprobs_dev = _compute_label_only_logprobs_log_softmax(
                        next_token_logits,
                        label_token_ids_arr,
                        out_sharding,
                    )
                else:
                    row_logprobs_dev = _compute_label_only_logprobs(
                        next_token_logits,
                        label_token_ids_arr,
                        out_sharding,
                    )
                if apply_softmax:
                    scores_dev = jax.nn.softmax(
                        row_logprobs_dev.astype(jnp.float32),
                        axis=-1,
                    )
                else:
                    scores_dev = jnp.exp(row_logprobs_dev.astype(jnp.float32))
                scores_dev.block_until_ready()
                forward_end = time.perf_counter()

            scores_dev = scores_dev[:real_bs, :]
            if scores_dev.ndim != 2:
                raise RuntimeError(f"Unexpected label-only score shape: {scores_dev.shape}.")
            if scores_dev.shape[0] != real_bs:
                raise RuntimeError(
                    f"Chunk output rows ({scores_dev.shape[0]}) != request count ({real_bs})."
                )
            if scores_dev.shape[1] != len(label_token_ids):
                raise RuntimeError(
                    f"Chunk output labels ({scores_dev.shape[1]}) != requested label count ({len(label_token_ids)})."
                )

            chunk_device_compute_s = max(0.0, forward_end - forward_start)
            chunk_total_s = max(0.0, time.perf_counter() - chunk_wall_start)
            chunk_host_overhead_s = max(0.0, chunk_total_s - chunk_device_compute_s)
            return scores_dev, chunk_device_compute_s, chunk_host_overhead_s
        finally:
            if out_cache_loc is not None:
                try:
                    valid_out_cache_loc = np.asarray(out_cache_loc, dtype=np.int32)
                    valid_out_cache_loc = valid_out_cache_loc[valid_out_cache_loc > 0]
                    if valid_out_cache_loc.size > 0:
                        self.token_to_kv_pool_allocator.free(valid_out_cache_loc)
                except Exception:
                    logger.exception(
                        "Direct fastpath v2 cleanup failed while freeing chunk KV slots."
                    )

    def _run_score_from_cache_v2_chunk_label_only(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        label_token_ids_arr: jax.Array,
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
            return_label_logprobs=False,
        )
        try:
            chunk_wall_start = time.perf_counter()
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
            (
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
            ) = self.tp_worker.get_precompile_paddings()
            model_worker_batch = batch.get_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
            )

            forward_start = time.perf_counter()
            logits_output, _, _ = self.tp_worker.forward_batch_generation(
                model_worker_batch=model_worker_batch,
                launch_done=None,
                skip_sample=True,
                sampling_metadata=None,
            )

            if (
                logits_output is None
                or logits_output.next_token_logits is None
                or logits_output.next_token_logits.shape[-1] == 0
            ):
                raise RuntimeError(
                    "Missing next_token_logits from score-from-cache v2 label-only chunk."
                )

            next_token_logits = logits_output.next_token_logits[: model_worker_batch.real_bs, :]
            out_sharding = NamedSharding(self.mesh, P(None, None))
            legacy_kernel_mode = SCORE_V2_LABEL_ONLY_KERNEL_MODE or "baseline"
            if legacy_kernel_mode not in {"baseline", "", "log_softmax"}:
                raise RuntimeError(
                    "Unsupported SGLANG_SCORE_LABEL_ONLY_KERNEL_MODE="
                    f"{SCORE_V2_LABEL_ONLY_KERNEL_MODE!r}."
                )

            fused_kernel_enabled = bool(
                getattr(self.server_args, "multi_item_score_label_only_fused_kernel", True)
            )
            # Preserve explicit kernel-mode requests by forcing the legacy path.
            if legacy_kernel_mode not in {"baseline", ""}:
                fused_kernel_enabled = False

            if fused_kernel_enabled:
                self.score_label_only_fused_kernel_calls += 1
                if apply_softmax:
                    self.score_label_only_fused_kernel_softmax_calls += 1
                scores_dev = _compute_label_only_scores_fused(
                    next_token_logits,
                    label_token_ids_arr,
                    bool(apply_softmax),
                    out_sharding,
                )
                scores_dev.block_until_ready()
                forward_end = time.perf_counter()
                scores_np = np.asarray(jax.device_get(scores_dev), dtype=np.float64)
            else:
                self.score_label_only_legacy_kernel_calls += 1
                if legacy_kernel_mode == "log_softmax":
                    row_logprobs_dev = _compute_label_only_logprobs_log_softmax(
                        next_token_logits,
                        label_token_ids_arr,
                        out_sharding,
                    )
                else:
                    row_logprobs_dev = _compute_label_only_logprobs(
                        next_token_logits,
                        label_token_ids_arr,
                        out_sharding,
                    )
                row_logprobs_dev.block_until_ready()
                forward_end = time.perf_counter()
                row_logprobs = np.asarray(jax.device_get(row_logprobs_dev), dtype=np.float32)

                if row_logprobs.ndim != 2:
                    raise RuntimeError(
                        f"Unexpected label-only logprob shape: {row_logprobs.shape}."
                    )
                if row_logprobs.shape[0] != len(reqs):
                    raise RuntimeError(
                        f"Chunk output rows ({row_logprobs.shape[0]}) != request count ({len(reqs)})."
                    )
                if row_logprobs.shape[1] != len(label_token_ids):
                    raise RuntimeError(
                        f"Chunk output labels ({row_logprobs.shape[1]}) != requested label count ({len(label_token_ids)})."
                    )

                if legacy_kernel_mode == "log_softmax" and SCORE_V2_LABEL_ONLY_PARITY_CHECK:
                    baseline_logprobs_dev = _compute_label_only_logprobs(
                        next_token_logits,
                        label_token_ids_arr,
                        out_sharding,
                    )
                    baseline_logprobs_dev.block_until_ready()
                    baseline_logprobs = np.asarray(
                        jax.device_get(baseline_logprobs_dev), dtype=np.float32
                    )
                    parity_max_abs_diff, parity_mean_abs_diff = self._label_only_parity_metrics(
                        baseline_logprobs=baseline_logprobs,
                        candidate_logprobs=row_logprobs,
                    )
                    if (
                        parity_max_abs_diff > SCORE_V2_LABEL_ONLY_PARITY_MAX_ABS_DIFF
                        or parity_mean_abs_diff > SCORE_V2_LABEL_ONLY_PARITY_MEAN_ABS_DIFF
                    ):
                        raise RuntimeError(
                            "Label-only kernel parity check failed: "
                            f"max_abs_diff={parity_max_abs_diff:.6g} "
                            f"(threshold={SCORE_V2_LABEL_ONLY_PARITY_MAX_ABS_DIFF:.6g}), "
                            f"mean_abs_diff={parity_mean_abs_diff:.6g} "
                            f"(threshold={SCORE_V2_LABEL_ONLY_PARITY_MEAN_ABS_DIFF:.6g})."
                        )

                # Legacy path: produce token probabilities from logprobs and apply
                # optional softmax over labels on host.
                token_prob_vals = np.exp(row_logprobs.astype(np.float64))
                if apply_softmax:
                    row_max = np.max(token_prob_vals, axis=1, keepdims=True)
                    stable = token_prob_vals - row_max
                    exp_vals = np.exp(stable)
                    denom = np.sum(exp_vals, axis=1, keepdims=True)
                    scores_np = exp_vals / denom
                else:
                    scores_np = token_prob_vals

            if scores_np.ndim != 2:
                raise RuntimeError(f"Unexpected label-only score shape: {scores_np.shape}.")
            if scores_np.shape[0] != len(reqs):
                raise RuntimeError(
                    f"Chunk output rows ({scores_np.shape[0]}) != request count ({len(reqs)})."
                )
            if scores_np.shape[1] != len(label_token_ids):
                raise RuntimeError(
                    f"Chunk output labels ({scores_np.shape[1]}) != requested label count ({len(label_token_ids)})."
                )

            scores = scores_np.tolist()

            chunk_device_compute_s = max(0.0, forward_end - forward_start)
            chunk_total_s = max(0.0, time.perf_counter() - chunk_wall_start)
            chunk_host_overhead_s = max(0.0, chunk_total_s - chunk_device_compute_s)
            return scores, chunk_device_compute_s, chunk_host_overhead_s
        finally:
            self._release_score_from_cache_v2_chunk_reqs(reqs, batch=batch)
