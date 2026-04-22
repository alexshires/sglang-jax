import logging
import math
import os
import time
from sgl_jax.srt.managers.scoring_utils import _compute_label_only_logprobs

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqOutput
from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch, acc_global_bid
from sgl_jax.srt.managers.utils import validate_input_length
from sgl_jax.srt.sampling.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


def print_mem(label):
    process = psutil.Process(os.getpid())
    logger.info("MEM %s: %.2f MB", label, process.memory_info().rss / 1024 / 1024)


class SchedulerScoringMixin:

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
            )
            req.extend_from_cache = cache_handle
            req.tokenizer = self.tokenizer
            req.logprob_start_len = len(req.origin_input_ids) - 1
            assert (
                req.extend_from_cache is not None
            ), f"DEBUG: extend_from_cache is None in Mixin for rid {req.rid}"
            req.cached_last_node = cached_last_node
            req.cached_last_host_node = cached_last_node
            req.cached_prefix_indices = cached_prefix_indices
            req.cached_host_hit_length = 0
            reqs.append(req)
        return reqs

    def score_from_cache_v2(self, recv_req):
        # Simple sequential chunking loop
        print_mem("score_from_cache_v2 start")
        score_start = time.perf_counter()
        total_items = len(recv_req.items_2d)
        items_per_step = getattr(recv_req, "items_per_step", 16) or 16

        max_running_requests = int(getattr(self.server_args, "max_running_requests", 0) or 0)
        if max_running_requests > 0 and not getattr(self.server_args, "score_v2_allow_reqpool_oversubscribe", False):
            items_per_step = min(items_per_step, max_running_requests)

        req_to_token_pool = getattr(self, "req_to_token_pool", None)
        if req_to_token_pool is not None and hasattr(req_to_token_pool, "available_size"):
            try:
                req_pool_available = int(req_to_token_pool.available_size())
                items_per_step = min(items_per_step, req_pool_available)
            except Exception:
                pass

        entry = self.scoring_cache_nodes.get(recv_req.cache_handle)
        if entry is None:
            raise ValueError(f"Cache handle {recv_req.cache_handle} not found or expired.")

        cached_last_node, _, prefix_ids, prefix_indices, cached_extra_key, _ = (
            self._unpack_scoring_cache_entry(entry)
        )

        all_scores = []
        dispatch_count = 0
        device_compute_s = 0.0
        host_orchestration_s = 0.0

        first_dispatch_started = False
        for start in range(0, total_items, items_per_step):
            chunk_items = recv_req.items_2d[start : start + items_per_step]
            if not chunk_items:
                continue
            if not first_dispatch_started:
                queue_wait_s = max(0.0, time.perf_counter() - score_start)
                first_dispatch_started = True

            chunk_host_start = time.perf_counter()
            print_mem(f"Before chunk run {start}")

            chunk_scores, chunk_device_compute_s, chunk_host_overhead_s = (
                self._run_score_from_cache_v2_chunk(
                    cache_handle=recv_req.cache_handle,
                    chunk_items=chunk_items,
                    label_token_ids=recv_req.label_token_ids,
                    apply_softmax=recv_req.apply_softmax,
                    cached_last_node=cached_last_node,
                    cached_prefix_indices=prefix_indices,
                    prefix_ids=prefix_ids,
                    cached_extra_key=cached_extra_key,
                )
            )
            print_mem(f"After chunk run {start}")

            all_scores.extend(chunk_scores)
            dispatch_count += 1
            device_compute_s += max(0.0, chunk_device_compute_s)
            chunk_total = max(0.0, time.perf_counter() - chunk_host_start)
            host_orchestration_s += max(
                0.0,
                max(chunk_host_overhead_s, chunk_total - chunk_device_compute_s),
            )

        return ScoreFromCacheReqOutput(
            rid=recv_req.rid,
            scores=all_scores,
            device_compute_s=device_compute_s,
            host_orchestration_s=host_orchestration_s,
            queue_wait_s=queue_wait_s,
            dispatch_count=dispatch_count,
        )

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
            return_label_logprobs=False,
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

            logits_output_dict, _, _ = self.tp_worker.forward_batch_generation(
                model_worker_batch, skip_sample=True, sampling_metadata=None
            )
            
            from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
            logits_output = LogitsProcessorOutput(**logits_output_dict)

            if logits_output.next_token_logits is None:
                # Logprobs were pre-computed on device!
                logprob_vals = np.asarray(jax.device_get(logits_output.next_token_logprobs), dtype=np.float64)
            else:
                # Compute on host using JAX math
                from jax.sharding import NamedSharding
                from jax.sharding import PartitionSpec as P

                next_token_logits = logits_output.next_token_logits[: len(reqs), :]
                label_token_ids_arr = jnp.asarray(label_token_ids, dtype=jnp.int32)
                out_sharding = NamedSharding(self.mesh, P(None, None))

                row_logprobs_dev = _compute_label_only_logprobs(
                    next_token_logits, label_token_ids_arr, out_sharding
                )
                logprob_vals = np.asarray(jax.device_get(row_logprobs_dev), dtype=np.float64)

            if logprob_vals.shape[0] != len(reqs):
                raise RuntimeError(
                    f"Chunk output rows ({logprob_vals.shape[0]}) != request count ({len(reqs)})."
                )

            scores: list[list[float]] = []
            for row in logprob_vals:
                scores.append(
                    self._score_from_cache_v2_probs_from_logprobs(
                        row_logprobs=row.tolist(),
                        apply_softmax=apply_softmax,
                    )
                )

            chunk_device_compute_s = reqs[0].device_compute_time_s if reqs else 0.0
            chunk_host_overhead_s = reqs[0].host_overhead_time_s if reqs else 0.0
            return scores, chunk_device_compute_s, chunk_host_overhead_s
        finally:
            self._release_score_from_cache_v2_chunk_reqs(reqs, batch=batch)

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
                # For PR 1a, we don't use extend_from_cache optimization yet!
                # extend_from_cache=cache_handle,
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

    def _record_scoring_cache_lookup(self, path: str, hit: bool) -> None:
        if not hasattr(self, "scoring_cache_lookup_queries"):
            self.scoring_cache_lookup_queries = 0
            self.scoring_cache_lookup_hits = 0
            self.scoring_cache_lookup_misses = 0
            self.scoring_cache_lookup_by_path = {}

        self.scoring_cache_lookup_queries += 1
        if hit:
            self.scoring_cache_lookup_hits += 1
        else:
            self.scoring_cache_lookup_misses += 1

        bucket = self.scoring_cache_lookup_by_path.setdefault(
            path,
            {"queries": 0, "hits": 0, "misses": 0},
        )
        bucket["queries"] += 1
        if hit:
            bucket["hits"] += 1
        else:
            bucket["misses"] += 1

    def _scoring_cache_metrics_snapshot(self) -> dict:
        query_total = getattr(self, "scoring_cache_lookup_queries", 0)
        hit_total = getattr(self, "scoring_cache_lookup_hits", 0)
        miss_total = getattr(self, "scoring_cache_lookup_misses", 0)
        hit_rate = float(hit_total / query_total) if query_total > 0 else 0.0
        
        scoring_cache_nodes = getattr(self, "scoring_cache_nodes", {})
        
        return {
            "active_handles": len(scoring_cache_nodes),
            "handles_created": getattr(self, "scoring_cache_handles_created", 0),
            "handles_released_total": getattr(self, "scoring_cache_handles_released", 0),
            "handles_released_manual": getattr(self, "scoring_cache_handles_released_manual", 0),
            "handles_released_expired": getattr(self, "scoring_cache_handles_released_expired", 0),
            "lookup_queries": query_total,
            "lookup_hits": hit_total,
            "lookup_misses": miss_total,
            "lookup_hit_rate": hit_rate,
            "lookup_by_path": getattr(self, "scoring_cache_lookup_by_path", {}),
        }
