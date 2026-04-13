"""A scheduler that manages a tensor parallel TPU worker."""

import concurrent.futures as futures
import dataclasses
import faulthandler
import gc
import logging
import math
import os
import pickle
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import numpy as np
import pathwaysutils
import psutil
import setproctitle
import zmq
from jax import numpy as jnp
from jax.scipy import special as jsp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.global_config import global_config
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes import (
    set_logical_device_count_override,
)
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseScoringCacheReqInput,
    ReleaseScoringCacheReqOutput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerBatch,
    Req,
    ScheduleBatch,
    acc_global_bid,
    global_server_args_dict,
)
from sgl_jax.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sgl_jax.srt.managers.scheduler_metrics_mixin import SchedulerMetricsMixin
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.managers.scheduler_profiler_mixing import SchedulerProfilerMixin
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient
from sgl_jax.srt.managers.utils import validate_input_length
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sgl_jax.srt.mem_cache.radix_cache import RadixCache
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.multimodal.tokenizer_utils import resolve_tokenizer_subdir
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import (
    configure_logger,
    get_bool_env_var,
    get_zmq_socket,
    kill_itself_when_parent_died,
    pyspy_dump_schedulers,
    set_random_seed,
)
from sgl_jax.srt.utils.jax_utils import get_device_name
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")
RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")
GRAMMAR_TIMEOUT = float(os.environ.get("SGLANG_GRAMMAR_TIMEOUT", 300))
SCORE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE = get_bool_env_var(
    "SGLANG_SCORE_FROM_CACHE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE"
)
SCORE_V2_LABEL_ONLY_KERNEL_MODE = (
    os.environ.get("SGLANG_SCORE_LABEL_ONLY_KERNEL_MODE", "baseline").strip().lower()
)
SCORE_V2_LABEL_ONLY_PARITY_CHECK = get_bool_env_var("SGLANG_SCORE_LABEL_ONLY_PARITY_CHECK")
SCORE_V2_LABEL_ONLY_PARITY_MAX_ABS_DIFF = float(
    os.environ.get("SGLANG_SCORE_LABEL_ONLY_PARITY_MAX_ABS_DIFF", "1e-3")
)
SCORE_V2_LABEL_ONLY_PARITY_MEAN_ABS_DIFF = float(
    os.environ.get("SGLANG_SCORE_LABEL_ONLY_PARITY_MEAN_ABS_DIFF", "5e-4")
)
SCORE_V2_DIRECT_TOKEN_IDS_LOGPROB_ONLY_LARGE_CHUNK_SIZE = 16384
SCORE_V2_DIRECT_TOKEN_IDS_LOGPROB_ONLY_LARGE_CHUNK_MIN_BS = 256
SCORE_V2_DIRECT_TOKEN_IDS_LOGPROB_ONLY_LARGE_CHUNK_MAX_SEQ_LEN = 4096


class SyncError(Exception):
    pass


def _set_scheduler_logical_device_count(
    server_args: ServerArgs,
    *,
    update_env: bool,
) -> None:
    logical_device_count = None
    if server_args.device_indexes is not None:
        logical_device_count = len(server_args.device_indexes)
    set_logical_device_count_override(logical_device_count)
    if not update_env:
        return
    if logical_device_count is None:
        os.environ.pop("SGLANG_LOGICAL_DEVICE_COUNT", None)
    else:
        os.environ["SGLANG_LOGICAL_DEVICE_COUNT"] = str(logical_device_count)


class SendDataError(Exception):
    pass


class ReceiveDataError(Exception):
    pass


@jax.jit(static_argnums=(2,))
def _compute_label_only_logprobs(next_token_logits, label_token_ids_arr, out_sharding):
    """Compute target-only logprobs for [batch, vocab] logits."""
    logits_f32 = next_token_logits.astype(jnp.float32)
    label_logits = logits_f32.at[:, label_token_ids_arr].get(out_sharding=out_sharding)
    normalizer = jsp.logsumexp(logits_f32, axis=-1, keepdims=True)
    return label_logits - normalizer


@jax.jit(static_argnums=(2,))
def _compute_label_only_logprobs_log_softmax(next_token_logits, label_token_ids_arr, out_sharding):
    """Alternative label-only kernel: full log-softmax then gather labels."""
    logits_f32 = next_token_logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
    return log_probs.at[:, label_token_ids_arr].get(out_sharding=out_sharding)


@jax.jit(static_argnums=(2, 3))
def _compute_label_only_scores_fused(
    next_token_logits,
    label_token_ids_arr,
    apply_softmax: bool,
    out_sharding,
):
    """Compute label-only probabilities directly on device for score fastpath."""
    logits_f32 = next_token_logits.astype(jnp.float32)
    label_logits = logits_f32.at[:, label_token_ids_arr].get(out_sharding=out_sharding)
    normalizer = jsp.logsumexp(logits_f32, axis=-1, keepdims=True)
    label_probs = jnp.exp(label_logits - normalizer)
    if apply_softmax:
        return jax.nn.softmax(label_probs, axis=-1)
    return label_probs


@jax.jit(static_argnums=(1,))
def _compute_label_only_scores_from_logprobs(label_logprobs, apply_softmax: bool):
    label_probs = jnp.exp(label_logprobs.astype(jnp.float32))
    if apply_softmax:
        return jax.nn.softmax(label_probs, axis=-1)
    return label_probs


@dataclass
class GenerationBatchResult:
    logits_output: LogitsProcessorOutput | None
    next_token_ids: list[int] | None  # on device
    extend_input_len_per_req: list[int]
    extend_logprob_start_len_per_req: list[int]
    bid: int
    cache_miss_count: int
    # relay path: forward stream -> next step forward
    next_draft_input: EagleDraftInput | None = None

    allocate_lens: np.ndarray | None = None
    num_accepted_tokens: int | None = None
    accept_lens: np.ndarray | None = None


@dataclass
class _LocalSchedulerRpcEnvelope:
    req_obj: object
    result_future: futures.Future | None = None


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
):
    """
    A scheduler that manages a tensor parallel TPU worker, which managaes fixed multi TPU devices.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs = None,
        communication_backend: CommunicationBackend = None,
        mesh: jax.sharding.Mesh = None,
        model_class: None = None,
        stage_sub_dir: str | None = None,
        precompile_params: dict | None = None,
    ):
        if stage_sub_dir is not None:
            server_args = dataclasses.replace(server_args)
            server_args.model_sub_dir = stage_sub_dir
        # set jit cache
        jit_cache_dir = os.getenv("JAX_COMPILATION_CACHE_DIR", None)
        if jit_cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", jit_cache_dir)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
            from jax.experimental.compilation_cache import compilation_cache as cc

            cc.set_cache_dir(jit_cache_dir)

        # Parse args
        self.server_args = server_args
        self.node_rank = server_args.node_rank
        self.nnodes = server_args.nnodes
        if port_args is not None:
            self.pub_sub_addr = port_args.pub_sub_addr
            self.pub_sub_sync_addr = port_args.pub_sub_sync_addr
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.max_seq_len = server_args.max_seq_len
        self.page_size = server_args.page_size
        self.enable_overlap = not server_args.disable_overlap_schedule
        if server_args.multimodal:
            logger.info("Multimodal mode enabled, disabling overlap schedule")
            self.enable_overlap = False
        self.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)

        # LoRA configurations
        self.lora_paths = server_args.lora_paths
        self.max_loras_per_batch = server_args.max_loras_per_batch

        # Init inter-process communication
        context = zmq.Context(2)
        self._comm_backend = None
        self.local_rpc_queue: queue.SimpleQueue[_LocalSchedulerRpcEnvelope] = queue.SimpleQueue()

        if self.node_rank == 0:
            # todo: support multi host
            if communication_backend is not None:
                self._comm_backend = communication_backend
            else:
                self.recv_from_tokenizer = get_zmq_socket(
                    context, zmq.PULL, port_args.scheduler_input_ipc_name, False
                )
                self.send_to_tokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )

                if server_args.skip_tokenizer_init:
                    # Directly send to the TokenizerManager
                    self.send_to_detokenizer = get_zmq_socket(
                        context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                    )
                else:
                    # Send to the DetokenizerManager
                    self.send_to_detokenizer = get_zmq_socket(
                        context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                    )

                self.recv_from_rpc = get_zmq_socket(
                    context, zmq.DEALER, port_args.rpc_ipc_name, False
                )
                if self.nnodes > 1:
                    self.publisher = get_zmq_socket(context, zmq.PUB, self.pub_sub_addr, bind=True)
                    self.publisher_sync = get_zmq_socket(
                        context, zmq.REP, self.pub_sub_sync_addr, bind=True
                    )
                    self.num_subscribers = self.nnodes - 1
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            if self.nnodes > 1:
                self.subscriber = get_zmq_socket(context, zmq.SUB, self.pub_sub_addr, bind=False)
                self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                self.subscriber.setsockopt(zmq.RCVTIMEO, 5000)
                self.subscriber_sync = get_zmq_socket(
                    context, zmq.REQ, self.pub_sub_sync_addr, bind=False
                )

        if self.nnodes > 1:
            self.sync_pub_sub()

        # Init tokenizer
        self.init_tokenizer()

        # Init grammar backend for structured output
        self.grammar_backend = None
        self.grammar_queue: list[Req] = []  # Requests waiting for grammar compilation
        if not server_args.skip_tokenizer_init and not server_args.multimodal:
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.model_config.vocab_size,
                self.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # init distribution
        if self.nnodes > 1:
            jax.distributed.initialize(server_args.dist_init_addr, self.nnodes, self.node_rank)

        platform = os.getenv("JAX_PLATFORMS", None)
        if platform == "proxy":
            pathwaysutils.initialize()
        if mesh is not None:
            self.mesh = mesh
        else:
            self.mesh = create_device_mesh(
                ici_parallelism=[-1, self.tp_size],
                dcn_parallelism=[1, 1],
                device_indexes=server_args.device_indexes,
            )

        if server_args.moe_backend == "fused":
            mesh_ep_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            if server_args.ep_size != mesh_ep_size:
                logger.warning(
                    "moe_backend='fused' uses EP size = mesh(data*tensor)=%d, but --ep-size=%d. "
                    "If you expected separate EP and TP (e.g. ep_size=%d, tp_size=%d), note that the "
                    "fused MoE kernel currently treats the full 2D mesh as its EP group.",
                    mesh_ep_size,
                    server_args.ep_size,
                    server_args.ep_size,
                    server_args.tp_size,
                )

        TpWorkerClass = ModelWorkerClient if self.enable_overlap else ModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            mesh=self.mesh,
            model_class=model_class,
            precompile_params=precompile_params,
        )

        # launch draft worker
        if self.spec_algorithm is not None and self.spec_algorithm.is_eagle():
            from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker

            self.draft_worker = EAGLEWorker(
                server_args=server_args,
                target_worker=self.tp_worker,
            )

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,  # total requests
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            _,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()

        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)

        self.is_hybrid = self.tp_worker.is_hybrid
        if self.is_hybrid:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: list[Req] = []
        # The aborted requests
        self.aborted_reqs: dict[str, Req] = {}
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: ScheduleBatch | None = None
        # The last forward batch
        self.last_batch: ScheduleBatch | None = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_prefill_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.accept_token = 0
        self.spec_num_forward_ct = 0
        self.draft_token = 0
        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init pause/continue state
        self._engine_paused = False

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
            "tokenizer_multi_item_packed": 0,
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }
        # Number of socket frames that carried each scoring path.
        self.ingress_score_path_frames = {
            "tokenizer_multi_item_packed": 0,
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

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
        )
        assert server_args.schedule_conservativeness >= 0, "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        self.init_profier()

        self.init_metrics()

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (AbortReq, self.abort_request),
                (ProfileReq, self.profile),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ReleaseScoringCacheReqInput, self.release_scoring_cache),
                (ScoreFromCacheReqInput, self.score_from_cache_v2),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
            ]
        )

        if not server_args.disable_precompile:
            if self.spec_algorithm is None or self.spec_algorithm.is_none():
                logger.info("[Scheduler] Begins to run worker precompile.")
                self.tp_worker.run_precompile()
                logger.info("[Scheduler] Completes worker precompile.")
            else:
                logger.info("[Scheduler] Begins to run spec_decode worker precompile.")
                self.draft_worker.run_spec_decode_precompile()
                logger.info("[Scheduler] Completes spec_decode worker precompile.")

        if self._score_direct_warmup_spec() is not None:
            logger.info("[Scheduler] Begins direct bulk score warmup.")
            self._run_score_direct_label_only_warmup()
            logger.info("[Scheduler] Completes direct bulk score warmup.")

    def sync_pub(self):
        logger.info(
            "[Publisher %s] Begins to synchronize, wait %s Subscribers",
            self.node_rank,
            self.nnodes - 1,
        )
        ready_count = 0
        try:
            while ready_count < self.num_subscribers:
                message = self.publisher_sync.recv_string()
                if message == "READY":
                    ready_count += 1
                    logger.info(
                        "[Publisher %s] receives %s READY signal",
                        self.node_rank,
                        ready_count,
                    )
                    self.publisher_sync.send_string("ACK")
                else:
                    self.publisher_sync.send_string("NACK")
        except zmq.Again:
            logger.error("[Publisher %s] Fails to synchronize due to timeout", self.node_rank)
            return False
        except Exception as e:
            logger.error("[Publisher %s] Encounters error: %s", self.node_rank, e)
            return False
        logger.info("[Publisher %s] Succeeds to synchronize!", self.node_rank)
        return True

    def sync_sub(self):
        logger.info("[Subscriber %s] Begins to synchronize", self.node_rank)
        try:
            self.subscriber_sync.send_string("READY")
            ack = self.subscriber_sync.recv_string()
            if ack == "ACK":
                logger.info("[Subscriber %s] Succeeds to synchronizes!", self.node_rank)
                return True
            else:
                logger.error(
                    "[Subscriber %s] Fails to synchroinze with ack: %s",
                    self.node_rank,
                    ack,
                )
                return False
        except Exception as e:
            logger.error("[Subscriber %s] Fails to synchronize with error: %s", self.node_rank, e)
            return False

    def sync_pub_sub(self):
        success = self.sync_pub() if self.node_rank == 0 else self.sync_sub()
        if not success:
            raise SyncError("Fail to synchronize between publisher and subscribers")

    def init_tokenizer(self):
        server_args = self.server_args
        self.model_config = ModelConfig.from_server_args(server_args)
        self.is_generation = self.model_config.is_generation
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            tokenizer_subdir = ""
            if server_args.multimodal:
                tokenizer_subdir = resolve_tokenizer_subdir(
                    server_args.model_path, server_args.tokenizer_path
                )
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                sub_dir=tokenizer_subdir,
            )

    def init_memory_pool_and_cache(self):
        server_args = self.server_args
        self.req_to_token_pool, self.token_to_kv_pool_allocator = self.tp_worker.get_memory_pool()

        if server_args.chunked_prefill_size is not None and server_args.disable_radix_cache:
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
            )
        elif self.is_hybrid:
            self.tree_cache = SWARadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                sliding_window_size=self.sliding_window_size,
                page_size=self.page_size,
                disable=server_args.disable_radix_cache,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
                disable=server_args.disable_radix_cache,
                kv_head_num=self.model_config.get_num_kv_heads(self.tp_size),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                max_seq_len=server_args.max_seq_len,
                is_eagle=self.spec_algorithm is not None and self.spec_algorithm.is_eagle(),
            )

        self.decode_mem_cache_buf_multiplier = 1

    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            self.process_input_requests(recv_reqs)

            # Skip batch processing when engine is paused
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.check_tree_cache()
                self.new_token_ratio = self.init_new_token_ratio

                # Elegant wait if idle
                if self._comm_backend is not None:
                    self._comm_backend.wait_for_new_requests(0.001)

            self.last_batch = batch

    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and Accelerator computation."""
        self.result_queue = deque()

        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            self.process_input_requests(recv_reqs)

            # Skip batch processing when engine is paused
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                batch.launch_done = threading.Event()
                with jax.profiler.TraceAnnotation("run_batch"):
                    result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    with jax.profiler.TraceAnnotation("process_batch_result"):
                        self.process_batch_result(tmp_batch, None, batch.launch_done)

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                # NOTE: we should use current launched batch's launch_done event Instead of the last batch's
                self.process_batch_result(
                    tmp_batch, tmp_result, batch.launch_done if batch else None
                )
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.check_tree_cache()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    def run_publisher(self, recv_reqs):
        retry_count = 0
        while retry_count < 3:
            try:
                serialized_data = pickle.dumps(recv_reqs)
                self.publisher.send(serialized_data)
                return True
            except Exception as e:
                logger.error(
                    "[Publisher %s] Fails to send data with error: %s",
                    self.node_rank,
                    e,
                )
        return False

    def run_subscriber(self):
        retry_count = 0
        while retry_count < 3:
            try:
                serialized_data = self.subscriber.recv()
                return pickle.loads(serialized_data)
            except zmq.Again:
                logger.error(
                    "[Subscriber %s] Fails to receive data with timeout, and try again",
                    self.node_rank,
                )
            except Exception as e:
                logger.error(
                    "[Subscriber %s] Fails to receive or deserialize with error: %s, and try again",
                    self.node_rank,
                    e,
                )
        return None

    def broadcast_pyobj(self, recv_reqs):
        if self.node_rank == 0:
            if not self.run_publisher(recv_reqs):
                raise SendDataError(f"[Publisher {self.node_rank}] Fails to send data")
        else:
            recv_reqs = self.run_subscriber()
            if recv_reqs is None:
                raise ReceiveDataError(f"[Subscriber {self.node_rank}] Fails to receive data")
        return recv_reqs

    @staticmethod
    def _is_score_path_req(req: Req) -> bool:
        if bool(getattr(req, "is_multi_item_scoring", False)):
            return True
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
        if not Scheduler._is_score_path_req(req):
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
            lane = Scheduler._admission_lane(req_owner, req)
            counts[lane] = counts.get(lane, 0) + 1
        return counts

    @staticmethod
    def _waiting_lane_counts(req_owner, waiting_queue: list[Req]) -> dict[str, int]:
        counts = {"default": 0, "short": 0, "long": 0}
        for req in waiting_queue:
            lane = Scheduler._admission_lane(req_owner, req)
            counts[lane] = counts.get(lane, 0) + 1
        return counts

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

        prefix_key = Scheduler._normalize_scoring_cache_prefix_key(
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
        lane_bias_candidates = Scheduler._lane_counter(
            req_owner,
            "score_scheduler_cache_admission_candidates",
        )
        lane_bias_promoted = Scheduler._lane_counter(
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
                priority = Scheduler._cache_admission_priority(req_owner, lane_req)
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
            lane_queues[Scheduler._admission_lane(req_owner, req)].append(req)

        for lane_name in ("default", "short", "long"):
            lane_queues[lane_name] = _apply_cache_bias(lane_name, lane_queues[lane_name])

        lane_waiting_max = Scheduler._lane_counter(req_owner, "score_scheduler_lane_waiting_max")
        for lane_name, lane_queue in lane_queues.items():
            lane_waiting_max[lane_name] = max(lane_waiting_max.get(lane_name, 0), len(lane_queue))

        lane_selected = Scheduler._lane_counter(
            req_owner,
            "score_scheduler_lane_isolation_selected",
        )
        lane_empty_turns = Scheduler._lane_counter(
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
            multi_item_scoring_flags=np.zeros(1, dtype=np.bool_),
            multi_item_scoring_delimiter=None,
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

        queue_pressure = Scheduler._score_scheduler_queue_pressure(self)
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
        requested_by_lane = Scheduler._lane_counter(
            self,
            "score_scheduler_dynamic_items_per_step_requested_by_lane",
        )
        requested_by_lane[lane_name] = requested_by_lane.get(lane_name, 0) + base_items_per_step
        effective_by_lane = Scheduler._lane_counter(
            self,
            "score_scheduler_dynamic_items_per_step_effective_by_lane",
        )
        effective_by_lane[lane_name] = (
            effective_by_lane.get(lane_name, 0) + effective_items_per_step
        )
        if effective_items_per_step != base_items_per_step:
            applied_by_lane = Scheduler._lane_counter(
                self,
                "score_scheduler_dynamic_items_per_step_applied_by_lane",
            )
            applied_by_lane[lane_name] = applied_by_lane.get(lane_name, 0) + 1

        return effective_items_per_step

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
                        "tokenizer_multi_item_packed": False,
                        "tokenizer_cache_for_scoring": False,
                        "tokenizer_extend_from_cache": False,
                        "rpc_score_from_cache_v2": False,
                        "rpc_release_scoring_cache": False,
                    }
                    for recv_req in unpacked_reqs:
                        if isinstance(recv_req, TokenizedGenerateReqInput):
                            if bool(getattr(recv_req, "is_multi_item_scoring", False)):
                                score_path_counts["tokenizer_multi_item_packed"] += 1
                                tokenizer_frame_paths["tokenizer_multi_item_packed"] = True
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

    def _unpack_scoring_cache_entry(self, entry):
        # Backward-compatible unpack for entries created before `last_access_ts`
        # was added.
        if len(entry) == 6:
            return entry
        if len(entry) == 5:
            node, swa_uuid, input_ids, prefix_indices, extra_key = entry
            return node, swa_uuid, input_ids, prefix_indices, extra_key, 0.0
        raise RuntimeError(f"Invalid scoring cache entry format (len={len(entry)}).")

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

    def _register_scoring_cache_handle(
        self,
        rid: str,
        input_ids: list[int] | tuple[int, ...] | np.ndarray | None,
        extra_key: str | None,
    ) -> tuple[str, tuple[int, ...]] | None:
        prefix_key = Scheduler._normalize_scoring_cache_prefix_key(input_ids, extra_key)
        if prefix_key is None:
            return None
        handles = self.scoring_cache_prefix_handles_by_key.setdefault(prefix_key, set())
        handles.add(rid)
        self.scoring_cache_handle_to_prefix_key[rid] = prefix_key
        return prefix_key

    def _unregister_scoring_cache_handle(self, rid: str) -> None:
        prefix_key = self.scoring_cache_handle_to_prefix_key.pop(rid, None)
        if prefix_key is None:
            return
        handles = self.scoring_cache_prefix_handles_by_key.get(prefix_key)
        if handles is None:
            return
        handles.discard(rid)
        if not handles:
            self.scoring_cache_prefix_handles_by_key.pop(prefix_key, None)

    def _record_scoring_cache_lookup(
        self,
        path: str,
        hit: bool,
        lane_name: str = "default",
    ) -> None:
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

        normalized_lane = lane_name if lane_name in {"default", "short", "long"} else "default"
        by_lane = self.scoring_cache_lookup_by_lane.setdefault(path, {})
        lane_bucket = by_lane.setdefault(
            normalized_lane,
            {"queries": 0, "hits": 0, "misses": 0},
        )
        lane_bucket["queries"] += 1
        if hit:
            lane_bucket["hits"] += 1
        else:
            lane_bucket["misses"] += 1

    def _record_scoring_cache_handle_created(self) -> None:
        self.scoring_cache_handles_created += 1

    def _record_scoring_cache_handle_released(self, reason: str) -> None:
        self.scoring_cache_handles_released += 1
        if reason == "manual":
            self.scoring_cache_handles_released_manual += 1
        elif reason == "expired":
            self.scoring_cache_handles_released_expired += 1
        else:
            self.scoring_cache_handles_released_other += 1

    def _scoring_cache_metrics_snapshot(self) -> dict:
        query_total = self.scoring_cache_lookup_queries
        hit_total = self.scoring_cache_lookup_hits
        miss_total = self.scoring_cache_lookup_misses
        hit_rate = float(hit_total / query_total) if query_total > 0 else 0.0
        return {
            "active_handles": len(self.scoring_cache_nodes),
            "active_prefix_keys": len(self.scoring_cache_prefix_handles_by_key),
            "handles_created": self.scoring_cache_handles_created,
            "handles_released_total": self.scoring_cache_handles_released,
            "handles_released_manual": self.scoring_cache_handles_released_manual,
            "handles_released_expired": self.scoring_cache_handles_released_expired,
            "handles_released_other": self.scoring_cache_handles_released_other,
            "handles_missing_node": self.scoring_cache_handles_missing_node,
            "lookup_queries": query_total,
            "lookup_hits": hit_total,
            "lookup_misses": miss_total,
            "lookup_hit_rate": hit_rate,
            "lookup_by_path": {
                path: dict(stats) for path, stats in self.scoring_cache_lookup_by_path.items()
            },
            "lookup_by_lane": {
                path: {lane: dict(stats) for lane, stats in lane_stats.items()}
                for path, lane_stats in self.scoring_cache_lookup_by_lane.items()
            },
        }

    def _release_scoring_cache_entry(self, rid: str, entry, reason: str) -> None:
        self._unregister_scoring_cache_handle(rid)
        node, swa_uuid, *_ = self._unpack_scoring_cache_entry(entry)
        self._record_scoring_cache_handle_released(reason)
        if node is None:
            self.scoring_cache_handles_missing_node += 1
            logger.warning("Scoring cache entry rid=%s has no radix node (%s).", rid, reason)
            return
        try:
            if isinstance(self.tree_cache, SWARadixCache):
                self.tree_cache.dec_lock_ref(node, swa_uuid)
            else:
                self.tree_cache.dec_lock_ref(node)
        except Exception:
            logger.exception(
                "Failed to decrement scoring-cache lock ref for rid=%s (%s).",
                rid,
                reason,
            )

    def _touch_scoring_cache_entry(self, rid: str, now: float | None = None):
        entry = self.scoring_cache_nodes.get(rid)
        if entry is None:
            return
        node, swa_uuid, input_ids, prefix_indices, extra_key, _ = self._unpack_scoring_cache_entry(
            entry
        )
        self.scoring_cache_nodes[rid] = (
            node,
            swa_uuid,
            input_ids,
            prefix_indices,
            extra_key,
            time.monotonic() if now is None else now,
        )

    def _evict_expired_scoring_cache_nodes(self, now: float | None = None) -> int:
        timeout = self.scoring_cache_timeout
        if timeout <= 0:
            return 0

        now_ts = time.monotonic() if now is None else now
        # Throttle GC to avoid walking the dict too often.
        if now is None and now_ts - self._last_scoring_cache_gc < 0.5:
            return 0
        self._last_scoring_cache_gc = now_ts

        expired_rids: list[str] = []
        for rid, entry in self.scoring_cache_nodes.items():
            *_, last_access_ts = self._unpack_scoring_cache_entry(entry)
            if now_ts - last_access_ts > timeout:
                expired_rids.append(rid)

        for rid in expired_rids:
            entry = self.scoring_cache_nodes.pop(rid, None)
            if entry is None:
                continue
            self._release_scoring_cache_entry(rid, entry, reason="expired")

        if expired_rids:
            logger.info("Evicted %d expired scoring cache handles.", len(expired_rids))
        return len(expired_rids)

    def _resolve_extend_from_cache(
        self, recv_req: TokenizedGenerateReqInput
    ) -> tuple[tuple | None, str | None]:
        if not recv_req.extend_from_cache:
            return None, None

        self._evict_expired_scoring_cache_nodes()
        entry = self.scoring_cache_nodes.get(recv_req.extend_from_cache)
        if entry is None:
            miss_lane = Scheduler._score_scheduler_lane_from_prefix_len(
                self, len(getattr(recv_req, "input_ids", []) or [])
            )
            self._record_scoring_cache_lookup(path="extend", hit=False, lane_name=miss_lane)
            err = (
                f"Missing scoring cache handle '{recv_req.extend_from_cache}'. "
                "The cached prefix may have expired or been released."
            )
            logger.warning("Prefill+extend scheduler: %s", err)
            return None, err

        cached_last_node, _, prefix_ids, prefix_indices, cached_extra_key, _ = (
            self._unpack_scoring_cache_entry(entry)
        )
        hit_lane = Scheduler._score_scheduler_lane_from_prefix_len(self, len(prefix_indices))
        self._record_scoring_cache_lookup(path="extend", hit=True, lane_name=hit_lane)

        item_ids = recv_req.input_ids or []
        recv_req.input_ids = prefix_ids + item_ids
        cached_prefix_len = len(prefix_indices)
        suffix_len = max(0, len(item_ids))
        if recv_req.extra_key is None:
            recv_req.extra_key = cached_extra_key
        self._touch_scoring_cache_entry(recv_req.extend_from_cache)
        logger.debug(
            "Prefill+extend scheduler: extend request rid=%s handle=%s prefix_tokens=%d cached_prefix=%d item_tokens=%d merged_input_tokens=%d max_new_tokens=%s",
            recv_req.rid,
            recv_req.extend_from_cache,
            len(prefix_ids),
            cached_prefix_len,
            suffix_len,
            len(recv_req.input_ids),
            recv_req.sampling_params.max_new_tokens,
        )
        return (cached_last_node, prefix_indices), None

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
                is_multi_item_scoring=False,
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
            multi_item_flags = np.zeros(padded_bs, dtype=np.bool_)

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
                multi_item_scoring_flags=multi_item_flags,
                multi_item_scoring_delimiter=None,
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

    def score_from_cache_v2(self, recv_req: ScoreFromCacheReqInput) -> ScoreFromCacheReqOutput:
        self.score_from_cache_v2_attempted += 1
        dispatch_count = 0
        queue_wait_s = 0.0
        device_compute_s = 0.0
        host_orchestration_s = 0.0
        score_start = time.perf_counter()

        try:
            if self.enable_overlap:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="unsupported_scheduler_mode",
                    error_msg="score-from-cache v2 does not support overlap schedule.",
                )

            is_valid, fallback_reason, error_msg = self._score_from_cache_v2_validate_items(
                recv_req
            )
            if not is_valid:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason=fallback_reason,
                    error_msg=error_msg,
                )

            self._evict_expired_scoring_cache_nodes()
            entry = self.scoring_cache_nodes.get(recv_req.cache_handle)
            if entry is None:
                self._record_scoring_cache_lookup(
                    path="score_from_cache_v2",
                    hit=False,
                    lane_name="default",
                )
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="missing_cache_handle",
                    error_msg=(
                        f"Missing scoring cache handle '{recv_req.cache_handle}'. "
                        "The cached prefix may have expired or been released."
                    ),
                )
            cached_last_node, _, prefix_ids, prefix_indices, cached_extra_key, _ = (
                self._unpack_scoring_cache_entry(entry)
            )
            lane_name = Scheduler._score_scheduler_lane_from_prefix_len(self, len(prefix_indices))
            self._record_scoring_cache_lookup(
                path="score_from_cache_v2",
                hit=True,
                lane_name=lane_name,
            )
            if cached_last_node is None:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="missing_cache_handle",
                    error_msg=f"Scoring cache handle '{recv_req.cache_handle}' has no radix node.",
                )

            label_only_logprob = bool(
                getattr(self.server_args, "multi_item_score_label_only_logprob", False)
            )
            use_direct_label_only = self._score_from_cache_v2_use_direct_label_only(
                label_only_logprob=label_only_logprob
            )
            if label_only_logprob:
                backend = str(getattr(self.server_args, "device", "")).lower()
                if backend not in {"tpu", "gpu", "cuda", "cpu"}:
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="unsupported_backend",
                        error_msg=(
                            "Label-only logprob fastpath requires TPU/GPU/CPU backend, "
                            f"got device={backend!r}."
                        ),
                    )

            items_per_step = int(recv_req.items_per_step or 0)
            default_items_per_step = int(
                getattr(self.server_args, "multi_item_score_from_cache_v2_items_per_step", 64)
            )
            if default_items_per_step <= 0:
                default_items_per_step = 1
            if use_direct_label_only:
                default_items_per_step = max(default_items_per_step, len(recv_req.items_2d))
            if items_per_step <= 0:
                items_per_step = default_items_per_step
            requested_items_per_step = max(1, items_per_step)
            if use_direct_label_only:
                requested_items_per_step = max(requested_items_per_step, len(recv_req.items_2d))
            requested_token_budget = max(
                0,
                int(recv_req.token_budget or 0),
                int(
                    getattr(self.server_args, "multi_item_score_from_cache_v2_token_budget", 0) or 0
                ),
            )
            # Keep chunk size within request-slot capacity so large configured values
            # (e.g., 64 with max_running_requests=24) do not trigger alloc_req_slots failures.
            capacity_caps: list[int] = []
            max_running_requests = int(getattr(self.server_args, "max_running_requests", 0) or 0)
            if (
                not use_direct_label_only
                and max_running_requests > 0
                and not SCORE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE
            ):
                capacity_caps.append(max_running_requests)
            req_to_token_pool = getattr(self, "req_to_token_pool", None)
            if (
                not use_direct_label_only
                and req_to_token_pool is not None
                and hasattr(req_to_token_pool, "available_size")
            ):
                try:
                    req_pool_available = int(req_to_token_pool.available_size())
                except Exception:
                    req_pool_available = 0
                if req_pool_available > 0:
                    capacity_caps.append(req_pool_available)
            effective_capacity = min(capacity_caps) if capacity_caps else requested_items_per_step
            if effective_capacity <= 0:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="req_slot_exhausted",
                    error_msg=(
                        "Fastpath v2 requires at least one free request slot "
                        f"(requested_items_per_step={requested_items_per_step})."
                    ),
                    dispatch_count=dispatch_count,
                    queue_wait_s=queue_wait_s,
                    device_compute_s=device_compute_s,
                    host_orchestration_s=host_orchestration_s,
                )
            items_per_step = self._resolve_score_from_cache_v2_items_per_step(
                requested_items_per_step=requested_items_per_step,
                default_items_per_step=default_items_per_step,
                effective_capacity=effective_capacity,
                total_items=len(recv_req.items_2d),
                lane_name=lane_name,
            )
            total_items = len(recv_req.items_2d)
            (
                items_per_step,
                dispatch_token_budget,
                replica_lane_count,
                topology_name,
            ) = self._score_from_cache_v2_topology_dispatch_policy(
                lane_name=lane_name,
                prefix_len=len(prefix_ids),
                requested_items_per_step=requested_items_per_step,
                effective_items_per_step=items_per_step,
                effective_capacity=effective_capacity,
                total_items=total_items,
                requested_token_budget=requested_token_budget,
                max_total_tokens=max(
                    int(recv_req.max_total_tokens or 0),
                    max((len(item) for item in recv_req.items_2d), default=0),
                ),
            )
            if total_items == 0:
                self.score_from_cache_v2_succeeded += 1
                self._record_score_from_cache_v2_timing(
                    queue_wait_s=0.0,
                    device_compute_s=0.0,
                    host_orchestration_s=0.0,
                )
                return ScoreFromCacheReqOutput(
                    rid=recv_req.rid,
                    success=True,
                    scores=[],
                    fallback_reason=None,
                    error_msg="",
                    dispatch_count=0,
                    lifecycle_requests_sent=0,
                    lifecycle_results_received=0,
                    queue_wait_s=0.0,
                    device_compute_s=0.0,
                    host_orchestration_s=0.0,
                    effective_items_per_step=items_per_step,
                    dispatch_token_budget=dispatch_token_budget,
                    replica_lane_count=replica_lane_count,
                    topology_name=topology_name,
                )

            label_token_ids_arr = None
            if label_only_logprob:
                label_token_ids_arr = jnp.asarray(recv_req.label_token_ids, dtype=jnp.int32)
            chunk_plan = self._build_score_from_cache_v2_chunk_plan(
                recv_req.items_2d,
                items_per_step,
                prefix_len=len(prefix_ids),
                token_budget=dispatch_token_budget,
            )

            for _, chunk_items in chunk_plan:
                if not chunk_items:
                    continue

                int32_max = np.iinfo(np.int32).max
                max_seq_len = max((len(prefix_ids) + len(item) for item in chunk_items), default=0)
                estimated_words = self._estimate_score_from_cache_v2_words(
                    prefix_len=len(prefix_ids),
                    items=chunk_items,
                )
                if max_seq_len >= int32_max or estimated_words >= int(int32_max * 0.9):
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="size_guard",
                        error_msg=(
                            "Fastpath v2 size guard triggered. "
                            f"max_seq_len={max_seq_len}, estimated_words={estimated_words}"
                        ),
                        dispatch_count=dispatch_count,
                        queue_wait_s=queue_wait_s,
                        device_compute_s=device_compute_s,
                        host_orchestration_s=host_orchestration_s,
                    )

            self._touch_scoring_cache_entry(recv_req.cache_handle)

            all_scores: list[list[float] | None] = [None] * total_items
            deferred_direct_chunks: list[tuple[list[int], jax.Array]] = []
            first_dispatch_started = False
            for chunk_indices, chunk_items in chunk_plan:
                if not chunk_items:
                    continue
                if not first_dispatch_started:
                    queue_wait_s = max(0.0, time.perf_counter() - score_start)
                    first_dispatch_started = True
                chunk_host_start = time.perf_counter()
                if label_only_logprob:
                    if use_direct_label_only:
                        chunk_scores, chunk_device_compute_s, chunk_host_overhead_s = (
                            self._run_score_from_cache_v2_direct_chunk_label_only(
                                cache_handle=recv_req.cache_handle,
                                chunk_items=chunk_items,
                                label_token_ids=recv_req.label_token_ids,
                                label_token_ids_arr=label_token_ids_arr,
                                apply_softmax=recv_req.apply_softmax,
                                cached_last_node=cached_last_node,
                                cached_prefix_indices=prefix_indices,
                                prefix_ids=prefix_ids,
                                cached_extra_key=cached_extra_key,
                            )
                        )
                    else:
                        chunk_scores, chunk_device_compute_s, chunk_host_overhead_s = (
                            self._run_score_from_cache_v2_chunk_label_only(
                                cache_handle=recv_req.cache_handle,
                                chunk_items=chunk_items,
                                label_token_ids=recv_req.label_token_ids,
                                label_token_ids_arr=label_token_ids_arr,
                                apply_softmax=recv_req.apply_softmax,
                                cached_last_node=cached_last_node,
                                cached_prefix_indices=prefix_indices,
                                prefix_ids=prefix_ids,
                                cached_extra_key=cached_extra_key,
                            )
                        )
                else:
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
                chunk_score_count = (
                    int(chunk_scores.shape[0])
                    if isinstance(chunk_scores, jax.Array)
                    else len(chunk_scores)
                )
                if chunk_score_count != len(chunk_indices):
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="runtime_exception",
                        error_msg=(
                            "score-from-cache v2 chunk output count mismatch: "
                            f"scores={chunk_score_count}, chunk_items={len(chunk_indices)}."
                        ),
                        dispatch_count=dispatch_count,
                        queue_wait_s=queue_wait_s,
                        device_compute_s=device_compute_s,
                        host_orchestration_s=host_orchestration_s,
                    )
                if isinstance(chunk_scores, jax.Array):
                    deferred_direct_chunks.append((list(chunk_indices), chunk_scores))
                else:
                    for original_idx, score_row in zip(chunk_indices, chunk_scores, strict=True):
                        all_scores[original_idx] = score_row
                dispatch_count += 1
                device_compute_s += max(0.0, chunk_device_compute_s)
                # host_orchestration_s excludes device time by design.
                chunk_total = max(0.0, time.perf_counter() - chunk_host_start)
                host_orchestration_s += max(
                    0.0,
                    max(chunk_host_overhead_s, chunk_total - chunk_device_compute_s),
                )

            if deferred_direct_chunks:
                materialize_start = time.perf_counter()
                merged_chunk_scores = jnp.concatenate(
                    [chunk_scores for _, chunk_scores in deferred_direct_chunks],
                    axis=0,
                )
                merged_scores_np = np.asarray(jax.device_get(merged_chunk_scores))
                merged_indices = [
                    original_idx
                    for chunk_indices, _ in deferred_direct_chunks
                    for original_idx in chunk_indices
                ]
                if merged_scores_np.shape[0] != len(merged_indices):
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="runtime_exception",
                        error_msg=(
                            "score-from-cache v2 merged direct output count mismatch: "
                            f"scores={merged_scores_np.shape[0]}, chunk_items={len(merged_indices)}."
                        ),
                        dispatch_count=dispatch_count,
                        queue_wait_s=queue_wait_s,
                        device_compute_s=device_compute_s,
                        host_orchestration_s=host_orchestration_s,
                    )
                for original_idx, score_row in zip(
                    merged_indices,
                    merged_scores_np,
                    strict=True,
                ):
                    all_scores[original_idx] = score_row.tolist()
                host_orchestration_s += max(0.0, time.perf_counter() - materialize_start)

            if any(score_row is None for score_row in all_scores):
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="runtime_exception",
                    error_msg=(
                        "score-from-cache v2 failed to assemble scores in original order "
                        f"({sum(score_row is not None for score_row in all_scores)} / {total_items})."
                    ),
                    dispatch_count=dispatch_count,
                    queue_wait_s=queue_wait_s,
                    device_compute_s=device_compute_s,
                    host_orchestration_s=host_orchestration_s,
                )
            ordered_scores = [score_row for score_row in all_scores if score_row is not None]

            self.score_from_cache_v2_succeeded += 1
            self._record_score_from_cache_v2_timing(
                queue_wait_s=queue_wait_s,
                device_compute_s=device_compute_s,
                host_orchestration_s=host_orchestration_s,
            )
            return ScoreFromCacheReqOutput(
                rid=recv_req.rid,
                success=True,
                scores=ordered_scores,
                fallback_reason=None,
                error_msg="",
                dispatch_count=dispatch_count,
                lifecycle_requests_sent=0,
                lifecycle_results_received=0,
                queue_wait_s=queue_wait_s,
                device_compute_s=device_compute_s,
                host_orchestration_s=host_orchestration_s,
                effective_items_per_step=items_per_step,
                dispatch_token_budget=dispatch_token_budget,
                replica_lane_count=replica_lane_count,
                topology_name=topology_name,
            )
        except Exception as e:
            logger.exception("score-from-cache v2 failed; falling back to baseline path.")
            return self._score_from_cache_v2_fallback_output(
                recv_req,
                reason="runtime_exception",
                error_msg=str(e),
                dispatch_count=dispatch_count,
                queue_wait_s=queue_wait_s,
                device_compute_s=device_compute_s,
                host_orchestration_s=host_orchestration_s,
            )

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        if self.server_args.log_requests:
            logger.debug(
                "Handle request: rid=%s, max_new_tokens=%s, token_ids_logprob=%s",
                recv_req.rid,
                recv_req.sampling_params.max_new_tokens,
                recv_req.token_ids_logprob,
            )

        cached_prefix_ctx, cache_lookup_error = self._resolve_extend_from_cache(recv_req)

        # Create a new request
        req = Req(
            recv_req.rid,
            recv_req.text,
            recv_req.input_ids,
            recv_req.sampling_params,
            return_logprob=recv_req.return_logprob,
            return_output_logprob_only=recv_req.return_output_logprob_only,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            lora_id=recv_req.lora_id,
            extra_key=recv_req.extra_key,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
            return_routed_experts=recv_req.return_routed_experts,
            return_hidden_states=recv_req.return_hidden_states,
            is_multi_item_scoring=recv_req.is_multi_item_scoring,
            multi_item_scoring_delimiter=recv_req.multi_item_scoring_delimiter,
            multi_item_algorithm=getattr(recv_req, "multi_item_algorithm", None),
            multi_item_mask_mode=getattr(recv_req, "multi_item_mask_mode", None),
            cache_for_scoring=recv_req.cache_for_scoring,
            extend_from_cache=recv_req.extend_from_cache,
        )
        req.tokenizer = self.tokenizer
        if cache_lookup_error is not None:
            req.set_finish_with_abort(cache_lookup_error)
            self._add_request_to_queue(req)
            return

        if cached_prefix_ctx is not None:
            cached_last_node, cached_prefix_indices = cached_prefix_ctx
            req.cached_last_node = cached_last_node
            req.cached_last_host_node = cached_last_node
            req.cached_prefix_indices = cached_prefix_indices
            req.cached_host_hit_length = 0
        if hasattr(recv_req, "mm_inputs") and recv_req.mm_inputs:
            req.mm_inputs = recv_req.mm_inputs
            multimodal_embedding = recv_req.mm_inputs.get("multimodal_embedding")
            req.multimodal_embedding = multimodal_embedding
            if (
                recv_req.mm_inputs.get("deepstack_visual_pos_mask") is not None
                and recv_req.mm_inputs.get("deepstack_visual_embedding") is not None
            ):
                req.apply_for_deepstack = True
                req.deepstack_visual_pos_mask = recv_req.mm_inputs.get("deepstack_visual_pos_mask")
                req.deepstack_visual_embedding = recv_req.mm_inputs.get(
                    "deepstack_visual_embedding"
                )
        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        if recv_req.logprob_start_len == -1 or not recv_req.return_logprob:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len >= len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none or the current grammar backend isn’t compatible with the model’s tokenizer"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(key)
                req.grammar = value

                if not cache_hit:
                    req.grammar_key = key
                    add_to_grammar_queue = True
                else:
                    if value is INVALID_GRAMMAR_OBJ:  # We hit a cached invalid grammar.
                        error_msg = f"Invalid grammar request with cache hit: {key=}"
                        req.set_finish_with_abort(error_msg)

        if add_to_grammar_queue:
            req.queue_time_start = time.perf_counter()
            self.grammar_queue.append(req)
        else:
            self._add_request_to_queue(req)

    def move_ready_grammar_requests(self):
        """Poll grammar futures and move ready requests to waiting queue."""
        if not self.grammar_queue:
            return

        num_ready_reqs = 0
        num_timeout_reqs = 0

        for req in self.grammar_queue:
            try:
                if req.finished():  # Aborted by AbortReq
                    num_ready_reqs += 1
                    continue

                # Poll with short timeout
                req.grammar = req.grammar.result(timeout=0.03)
                # Cache the compiled grammar
                if self.grammar_backend and req.grammar_key:
                    self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())

                # Check if compilation resulted in invalid grammar
                if req.grammar is INVALID_GRAMMAR_OBJ:
                    req.set_finish_with_abort(f"Invalid grammar request: key={req.grammar_key}")

                num_ready_reqs += 1
            except futures._base.TimeoutError:
                req.grammar_wait_ct += 1
                # Check if we've exceeded the timeout
                if req.grammar_wait_ct > GRAMMAR_TIMEOUT / 0.03:
                    num_timeout_reqs = 1
                break

        # Handle timeout requests: cancel and mark as failed
        for i in range(num_ready_reqs, num_ready_reqs + num_timeout_reqs):
            req = self.grammar_queue[i]
            req.grammar.cancel()
            error_msg = f"Grammar preprocessing timed out for {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)
            # Cache as invalid to avoid retrying
            if self.grammar_backend and req.grammar_key:
                self.grammar_backend.set_cache(req.grammar_key, INVALID_GRAMMAR_OBJ)
        num_ready_reqs += num_timeout_reqs

        # Move ready requests to waiting queue
        self._extend_requests_to_queue(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(global_server_args_dict)
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "kvcache": round(self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2),
            "token_capacity": int(self.max_total_num_tokens),
        }

        # state for pause/continue generation
        ret["engine_paused"] = self._engine_paused
        ret["waiting_queue_size"] = len(self.waiting_queue)
        ret["running_batch_size"] = (
            0 if self.running_batch.is_empty() else len(self.running_batch.reqs)
        )
        ret["prefill_decode_size"] = ret["waiting_queue_size"] + ret["running_batch_size"]
        ret["waiting_queue_rids"] = [req.rid for req in self.waiting_queue]
        ret["running_batch_rids"] = (
            [req.rid for req in self.running_batch.reqs]
            if not self.running_batch.is_empty()
            else []
        )

        # scheduling state
        ret["cur_batch_is_none"] = self.cur_batch is None
        ret["last_batch_is_none"] = self.last_batch is None
        ret["chunked_req_is_none"] = self.chunked_req is None

        # request cache stat
        if isinstance(self.tree_cache, ChunkCache):
            ret["tree_cache_size"] = 0
        else:
            ret["tree_cache_size"] = (
                self.tree_cache.total_size() if self.tree_cache is not None else 0
            )
        if self.req_to_token_pool is not None:
            ret["req_to_token_pool_total"] = self.req_to_token_pool.size
            ret["req_to_token_pool_available"] = self.req_to_token_pool.available_size()
            ret["req_to_token_pool_used"] = (
                self.req_to_token_pool.size - self.req_to_token_pool.available_size()
            )
        else:
            ret["req_to_token_pool_total"] = 0
            ret["req_to_token_pool_available"] = 0
            ret["req_to_token_pool_used"] = 0

        # physical kv cache stat
        ret["available_kv_tokens"] = self.token_to_kv_pool_allocator.available_size()

        # counters
        ret["num_generated_tokens"] = self.num_generated_tokens
        ret["forward_ct_decode"] = self.forward_ct_decode
        ret["new_token_ratio"] = self.new_token_ratio
        ret["init_new_token_ratio"] = self.init_new_token_ratio
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
                Scheduler._lane_counter(self, "score_scheduler_lane_admission_admitted")
            ),
            "skipped_by_lane": dict(
                Scheduler._lane_counter(self, "score_scheduler_lane_admission_skipped")
            ),
            "max_inflight_by_lane": dict(
                Scheduler._lane_counter(self, "score_scheduler_lane_inflight_max")
            ),
            "lane_isolation_selected_by_lane": dict(
                Scheduler._lane_counter(self, "score_scheduler_lane_isolation_selected")
            ),
            "lane_isolation_empty_turns_by_lane": dict(
                Scheduler._lane_counter(self, "score_scheduler_lane_isolation_empty_turns")
            ),
            "max_waiting_by_lane": dict(
                Scheduler._lane_counter(self, "score_scheduler_lane_waiting_max")
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
                    Scheduler._lane_counter(
                        self, "score_scheduler_dynamic_items_per_step_requested_by_lane"
                    )
                ),
                "effective_by_lane": dict(
                    Scheduler._lane_counter(
                        self, "score_scheduler_dynamic_items_per_step_effective_by_lane"
                    )
                ),
                "applied_by_lane": dict(
                    Scheduler._lane_counter(
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
                    Scheduler._lane_counter(self, "score_scheduler_cache_admission_candidates")
                ),
                "promoted_by_lane": dict(
                    Scheduler._lane_counter(self, "score_scheduler_cache_admission_promoted")
                ),
            },
        }

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        """Handle internal state updates, including precision tracer configuration"""
        success = True
        error_msg = ""

        try:
            if "precision_tracer" in recv_req.state_data:
                tracer_config = recv_req.state_data["precision_tracer"]

                # Update precision_tracer state in this process
                if "trace_active" in tracer_config:
                    logger.info(
                        "[SCHEDULER] check trace_active: %s",
                        precision_tracer.get_trace_active(),
                    )
                    precision_tracer._trace_active = tracer_config["trace_active"]
                    logger.info(
                        "[SCHEDULER] Updated trace_active to: %s",
                        precision_tracer._trace_active,
                    )

                    # Reset counters when starting trace
                    if tracer_config["trace_active"]:
                        precision_tracer._request_counter = 0
                        precision_tracer._completed_requests_count = 0
                        precision_tracer._request_traces = {}
                        logger.info("[SCHEDULER] Reset request_counter, completed_count and traces")

                if "max_requests" in tracer_config:
                    precision_tracer._max_requests = tracer_config["max_requests"]
                    logger.info(
                        "[SCHEDULER] Updated max_requests to: %s",
                        precision_tracer._max_requests,
                    )

                if "output_file" in tracer_config:
                    precision_tracer._trace_output_file = tracer_config["output_file"]
                    logger.info(
                        "[SCHEDULER] Updated output_file to: %s",
                        precision_tracer._trace_output_file,
                    )

                if "save_tensor" in tracer_config:
                    precision_tracer._save_tensor = tracer_config["save_tensor"]
                    logger.info(
                        "[SCHEDULER] Updated save_tensor to: %s",
                        precision_tracer._save_tensor,
                    )

                logger.info("[SCHEDULER] Precision tracer state updated: %s", tracer_config)

        except Exception as e:
            success = False
            error_msg = str(e)
            logger.info("[SCHEDULER] Error updating internal state: %s", error_msg)

        return SetInternalStateReqOutput(
            request_id=recv_req.request_id, success=success, error_msg=error_msg
        )

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        success, error_msg, flushed_items = self.flush_cache()
        return FlushCacheReqOutput(
            rid=recv_req.rid,
            error_msg=error_msg,
            success=success,
            flushed_items=flushed_items,
        )

    def _can_flush_cache(self) -> tuple[bool, str]:
        """Return whether cache flush can proceed and an optional error message."""

        def _batch_size(batch: ScheduleBatch | None) -> int:
            if batch is None:
                return 0
            return 0 if batch.is_empty() else batch.batch_size()

        waiting_reqs = len(self.waiting_queue)
        running_reqs = _batch_size(self.running_batch)
        current_batch_reqs = _batch_size(self.cur_batch)
        last_batch_reqs = _batch_size(self.last_batch)
        chunked_pending = self.chunked_req is not None
        pending_results = len(getattr(self, "result_queue", ())) if self.enable_overlap else 0

        has_pending = (
            waiting_reqs > 0
            or running_reqs > 0
            or current_batch_reqs > 0
            or last_batch_reqs > 0
            or chunked_pending
            or pending_results > 0
        )

        if has_pending:
            msg = (
                "Cache not flushed because there are pending requests. "
                f"waiting={waiting_reqs}, running={running_reqs}, "
                f"cur_batch={current_batch_reqs}, last_batch={last_batch_reqs}, "
                f"chunked={chunked_pending}, pending_results={pending_results}"
            )
            return False, msg

        return True, ""

    def flush_cache(self) -> tuple[bool, str, int]:
        can_flush, message = self._can_flush_cache()
        if not can_flush:
            logger.warning(message)
            return False, message, 0

        # Reset scheduling state
        self.cur_batch = None
        self.last_batch = None
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.chunked_req = None
        if self.enable_overlap:
            self.result_queue = deque()

        # Clear cache-related state
        if self.tree_cache is not None:
            self.tree_cache.reset()
        if self.req_to_token_pool is not None:
            self.req_to_token_pool.clear()
        if self.token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator.clear()
        if self.grammar_backend is not None:
            self.grammar_backend.reset()

        self.num_generated_tokens = 0
        self.forward_ct_decode = 0
        self.new_token_ratio = self.init_new_token_ratio

        flushed_items = (
            self.token_to_kv_pool_allocator.available_size()
            if self.token_to_kv_pool_allocator is not None
            else 0
        )

        logger.info("Cache flushed successfully!")
        return True, "", flushed_items

    def _add_request_to_queue(self, req: Req):
        req.queue_time_start = time.perf_counter()
        req.queue_time_end = None
        self.waiting_queue.append(req)

    def _extend_requests_to_queue(self, reqs: list[Req], is_retracted: bool = False):
        if is_retracted:
            now = time.perf_counter()
            for req in reqs:
                req.queue_time_start = now
                req.queue_time_end = None
        self.waiting_queue.extend(reqs)

    def check_memory(self):
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                _,
                _,
                full_available_size,
                full_evictable_size,
                swa_available_size,
                swa_evictable_size,
            ) = self._get_swa_token_info()
            # Strict mode: require perfect accounting with no tolerance
            full_protected = self.tree_cache.full_protected_size()
            swa_protected = self.tree_cache.swa_protected_size()
            memory_leak = (
                full_available_size + full_evictable_size + full_protected
            ) != self.full_tokens_per_layer or (
                swa_available_size + swa_evictable_size + swa_protected
            ) != self.swa_tokens_per_layer
            token_msg = (
                f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, full_protected={full_protected} (used={full_num_used})\n"
                f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, swa_protected={swa_protected} (used={swa_num_used})\n"
            )
        else:
            _, _, available_size, evictable_size = self._get_token_info()
            protected_size = self.tree_cache.protected_size()
            memory_leak = (
                available_size + evictable_size + protected_size
            ) != self.max_total_num_tokens
            token_msg = f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}\n"

        if memory_leak:
            msg = f"token_to_kv_pool_allocator memory leak detected! {token_msg}"
            raise ValueError(msg)

        req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise ValueError(msg)

    def check_tree_cache(self):
        if self.is_hybrid and isinstance(self.tree_cache, SWARadixCache):
            self.tree_cache.sanity_check()

    def _get_token_info(self):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_swa_token_info(self):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (full_available_size + full_evictable_size)
        swa_num_used = self.swa_tokens_per_layer - (swa_available_size + swa_evictable_size)
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        chunked_req_to_exclude = set()
        if self.chunked_req:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
            self.tree_cache.cache_unfinished_req(self.chunked_req)
            # chunked request keeps its rid but will get a new req_pool_idx
            self.req_to_token_pool.free(self.chunked_req.req_pool_idx)

        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        new_batch = self.get_new_batch_prefill()

        # if new_batch is not None:
        if new_batch:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        return ret

    def get_new_batch_prefill(self) -> ScheduleBatch | None:
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # `batch_is_full` is a soft throttle flag. If nothing is running, clear it so
        # prefill admission can resume and we don't get stuck in a full-but-idle state.
        if self.running_batch.is_empty() and self.running_batch.batch_is_full:
            self.running_batch.batch_is_full = False

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        if running_bs >= self.max_running_requests:
            self.running_batch.batch_is_full = True
            return None

        # ReqToTokenPool slots gate how many requests can enter EXTEND in this round.
        # Under prefill+extend scoring, a single user request can fan out into many
        # internal requests. If we ignore current slot pressure here, prepare_for_extend()
        # can raise and kill the scheduler process.
        req_slots_budget = self.req_to_token_pool.available_size()
        if req_slots_budget <= 0:
            self.running_batch.batch_is_full = True
            logger.debug("Deferring prefill: no req slots available in ReqToTokenPool.")
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        # Collect existing LoRA IDs in the running batch if LoRA is enabled
        if self.lora_paths is not None:
            lora_set = (
                set([req.lora_id for req in self.running_batch.reqs])
                if self.running_batch is not None
                else set([])
            )

        lane_inflight = Scheduler._running_lane_counts(self)
        lane_waiting_counts = Scheduler._waiting_lane_counts(self, self.waiting_queue)
        lane_waiting_max = Scheduler._lane_counter(self, "score_scheduler_lane_waiting_max")
        for lane_name, lane_count in lane_waiting_counts.items():
            lane_waiting_max[lane_name] = max(lane_waiting_max.get(lane_name, 0), lane_count)
        lane_inflight_max = Scheduler._lane_counter(self, "score_scheduler_lane_inflight_max")
        for lane_name, lane_count in lane_inflight.items():
            lane_inflight_max[lane_name] = max(lane_inflight_max.get(lane_name, 0), lane_count)
        lane_admitted = Scheduler._lane_counter(self, "score_scheduler_lane_admission_admitted")
        lane_skipped = Scheduler._lane_counter(self, "score_scheduler_lane_admission_skipped")
        ordered_waiting_queue = Scheduler._iter_waiting_queue(self, self.waiting_queue)

        # Get requests from the waiting queue to a new prefill batch
        for req in ordered_waiting_queue:
            if len(adder.can_run_list) >= req_slots_budget:
                self.running_batch.batch_is_full = True
                break

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.running_batch.batch_is_full = True
                break

            lane_name = Scheduler._admission_lane(self, req)
            self.score_scheduler_lane_admission_attempted = (
                int(getattr(self, "score_scheduler_lane_admission_attempted", 0)) + 1
            )
            lane_cap = Scheduler._lane_cap(self, lane_name)
            if lane_cap > 0 and lane_inflight.get(lane_name, 0) >= lane_cap:
                lane_skipped[lane_name] = lane_skipped.get(lane_name, 0) + 1
                continue

            # Check LoRA constraint: ensure we don't exceed max_loras_per_batch
            if (
                self.lora_paths is not None
                and len(
                    lora_set | set([req.lora_id for req in adder.can_run_list]) | set([req.lora_id])
                )
                > self.max_loras_per_batch
            ):
                break

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(req)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

            lane_inflight[lane_name] = lane_inflight.get(lane_name, 0) + 1
            lane_inflight_max[lane_name] = max(
                lane_inflight_max.get(lane_name, 0),
                lane_inflight[lane_name],
            )
            lane_admitted[lane_name] = lane_admitted.get(lane_name, 0) + 1

        # Update waiting queue
        can_run_list: list[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        admit_ts = time.perf_counter()
        for req in can_run_list:
            if req.queue_time_start is None:
                continue
            req.queue_time_end = admit_ts
            req.queue_wait_time_s += max(0.0, req.queue_time_end - req.queue_time_start)
            req.queue_time_start = None

        self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            enable_custom_logit_processor=False,
            chunked_req=self.chunked_req,
            mesh=self.mesh,
            spec_algorithm=self.spec_algorithm,
        )

        new_batch.prepare_for_extend()

        # Update waiting queue and chunked request state only after we
        # successfully allocate req slots in prepare_for_extend().
        self.waiting_queue = [x for x in self.waiting_queue if x not in set(can_run_list)]

        if adder.new_chunked_req is not None and adder.new_chunked_req in set(can_run_list):
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs

            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full, mesh=self.mesh
            )
        else:
            new_batch.decoding_reqs = None

        new_batch.bid = acc_global_bid()

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
            num_retracted_reqs = len(retracted_reqs)
            self.new_token_ratio = new_token_ratio

            logger.info(
                "KV cache pool is full. Retract requests. #retracted_reqs: %d, #new_token_ratio: %.4f -> %.4f",
                num_retracted_reqs,
                old_ratio,
                self.new_token_ratio,
            )

            self._extend_requests_to_queue(retracted_reqs, is_retracted=True)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch arrays
        batch.prepare_for_decode()
        return batch

    def run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run a batch."""
        self.forward_ct += 1

        if self.server_args.log_requests:
            logger.debug(
                "Run batch: mode=%s, bs=%d, return_logprob=%s",
                batch.forward_mode,
                batch.batch_size(),
                batch.return_logprob,
            )

        # Whether to run the profiler
        self._profile_batch_predicate(batch)

        # Run forward
        assert self.is_generation
        batch_wall_start = time.perf_counter()
        forward_start = time.perf_counter()
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self.tp_worker.get_precompile_paddings()
        if self.spec_algorithm is None or self.spec_algorithm.is_none():
            model_worker_batch = batch.get_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
            )
            skip_sample = self._can_skip_sample_for_prefill_batch(batch)

            if self.enable_overlap:
                with jax.profiler.TraceAnnotation(
                    f"forward_batch_generation_overlap {self.forward_ct}"
                ):

                    logits_output, next_token_ids, cache_miss_count = (
                        self.tp_worker.forward_batch_generation(
                            model_worker_batch,
                            sampling_metadata=None,
                            skip_sample=skip_sample,
                        )
                    )
                next_token_ids = next_token_ids[: model_worker_batch.real_bs]
            else:
                logits_output, next_token_ids_device, cache_miss_count = (
                    self.tp_worker.forward_batch_generation(
                        model_worker_batch,
                        skip_sample=skip_sample,
                        sampling_metadata=None,
                    )
                )
                if skip_sample:
                    next_token_ids = []
                else:
                    next_token_ids = np.array(jax.device_get(next_token_ids_device))[
                        : model_worker_batch.real_bs
                    ]
        else:
            model_worker_batch = batch.get_spec_model_worker_batch(
                precompile_token_paddings,
                precompile_bs_paddings,
                precompile_cache_loc_paddings,
                self.page_size,
                self.server_args.enable_static_lora,
            )
            batch_output = self.draft_worker.forward_batch_speculative_generation(
                model_worker_batch
            )
            if batch_output.accept_lens is not None:
                # Decode
                batch.seq_lens = batch.seq_lens + batch_output.accept_lens
            else:
                # Prefill
                batch.seq_lens = batch.seq_lens + 1
            batch.spec_info = batch_output.next_draft_input
            next_token_ids = batch_output.next_token_ids
            logits_output = batch_output.logits_output
            cache_miss_count = batch_output.cache_miss_count
        forward_end = time.perf_counter()
        batch_wall_end = time.perf_counter()
        bid = model_worker_batch.bid
        batch.output_ids = next_token_ids

        device_compute_s = max(0.0, forward_end - forward_start)
        host_overhead_s = max(0.0, (batch_wall_end - batch_wall_start) - device_compute_s)
        for req in batch.reqs:
            req.device_compute_time_s += device_compute_s
            req.host_overhead_time_s += host_overhead_s
            req.scheduler_dispatch_count += 1

        # These 2 values are needed for processing the output, but the values can be
        # modified by overlap schedule. So we have to copy them here so that
        # we can use the correct values in output processing.
        if batch.return_logprob:
            extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
        else:
            extend_input_len_per_req = None
        if batch.return_logprob:
            extend_logprob_start_len_per_req = [req.extend_logprob_start_len for req in batch.reqs]
        else:
            extend_logprob_start_len_per_req = None

        ret = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=(
                next_token_ids.tolist()
                if hasattr(next_token_ids, "tolist")
                else list(next_token_ids)
            ),
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            bid=bid,
            cache_miss_count=cache_miss_count,
        )
        if self.spec_algorithm is not None and self.spec_algorithm.is_eagle():
            assert isinstance(batch_output.next_draft_input, EagleDraftInput)
            ret.next_draft_input = batch_output.next_draft_input
            ret.accept_lens = batch_output.accept_lens
            ret.allocate_lens = batch_output.allocate_lens
        return ret

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: threading.Event | None = None,
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result, launch_done)
        elif batch.forward_mode.is_extend():
            self.process_batch_result_prefill(batch, result, launch_done)
        elif batch.forward_mode.is_idle():
            if self.enable_overlap:
                self.tp_worker.resolve_last_batch_result(launch_done)
                self.set_next_batch_sampling_info_done(batch)
        elif batch.forward_mode.is_dummy_first():
            self.set_next_batch_sampling_info_done(batch)

    def get_idle_batch(self):
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.server_args.enable_custom_logit_processor,
            self.mesh,
            spec_algorithm=self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def set_next_batch_sampling_info_done(self, batch: ScheduleBatch):
        if batch.next_batch_sampling_info:
            # Update grammar vocab masks for next batch in overlap mode
            if batch.next_batch_sampling_info.grammars is not None:
                batch.next_batch_sampling_info.update_grammar_vocab_mask()
            batch.next_batch_sampling_info.sampling_info_done.set()

    def watchdog_thread(self):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if current > self.watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = current
            time.sleep(self.watchdog_timeout // 2)

        pyspy_dump_schedulers()
        logger.error("Watchdog timeout (watchdog_timeout=%s)", self.watchdog_timeout)
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        # Wait for some time so that the parent process can print the error.
        time.sleep(5)
        self.parent_process.send_signal(signal.SIGQUIT)

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            abort_out = AbortReq(rid=req.rid)
            if self._comm_backend is not None:
                self._comm_backend.send_pyobj(abort_out)
            else:
                self.send_to_tokenizer.send_pyobj(abort_out)
            logger.debug("Abort queued request. rid=%s", req.rid)

        # Delete the requests in the grammar queue
        for req in self.grammar_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug("Abort grammar queue request. rid=%s", req.rid)
                if req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (recv_req.abort_all or req.rid.startswith(recv_req.rid)):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug("Abort running request. rid=%s", req.rid)
                req.to_finish = FINISH_ABORT()

        # Abort method 4: Release cached nodes for prefill+extend
        self._release_scoring_cache_nodes(recv_req.rid, recv_req.abort_all)

    def _release_scoring_cache_nodes(self, rid_prefix: str | None, abort_all: bool) -> int:
        released = 0
        self._evict_expired_scoring_cache_nodes()
        if not abort_all and not rid_prefix:
            return released

        rids_to_remove = []
        for rid in self.scoring_cache_nodes:
            if abort_all or (rid_prefix and rid.startswith(rid_prefix)):
                rids_to_remove.append(rid)

        for rid in rids_to_remove:
            entry = self.scoring_cache_nodes.pop(rid, None)
            if entry is None:
                continue
            self._release_scoring_cache_entry(rid, entry, reason="manual")
            released += 1
            logger.debug("Released cached node for rid=%s", rid)
        return released

    def release_scoring_cache(
        self, recv_req: ReleaseScoringCacheReqInput
    ) -> ReleaseScoringCacheReqOutput:
        released = self._release_scoring_cache_nodes(recv_req.rid, abort_all=False)
        return ReleaseScoringCacheReqOutput(
            rid=recv_req.rid,
            success=True,
            released_items=released,
        )

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True

        # finish all in-flight request; in overlap mode, last_batch is running
        if self.enable_overlap and self.last_batch:
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
            self.last_batch = None
            self.cur_batch = None

        if recv_req.mode == "retract":
            self.running_batch.filter_batch()
            if len(self.running_batch.reqs) != 0:
                # clear the kv cache
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None
            logger.info("Paused generation retracted")
        elif recv_req.mode == "in_place":
            logger.info("Paused generation in place")

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        self._engine_paused = False
        logger.info("Generation continued")


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: int | None,
    pipe_writer,
):
    def maybe_freeze_gc_after_warmup():
        if not getattr(server_args, "enable_gc_freeze", False):
            return
        if not hasattr(gc, "freeze"):
            logger.warning(
                "GC freeze requested but gc.freeze is unavailable on this Python runtime."
            )
            return
        try:
            freeze_before = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
            collected = gc.collect()
            gc.freeze()
            freeze_after = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
            logger.info(
                "Applied gc.freeze after warmup/precompile. collected=%d freeze_before=%d freeze_after=%d gc_count=%s",
                collected,
                freeze_before,
                freeze_after,
                gc.get_count(),
            )
            if getattr(server_args, "gc_freeze_rollback", False):
                if hasattr(gc, "unfreeze"):
                    gc.unfreeze()
                    rollback_count = (
                        gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
                    )
                    logger.warning(
                        "Rolled back gc.freeze due to --gc-freeze-rollback. freeze_count_after_rollback=%d gc_count=%s",
                        rollback_count,
                        gc.get_count(),
                    )
                else:
                    logger.warning(
                        "GC freeze rollback requested but gc.unfreeze is unavailable on this Python runtime."
                    )
        except Exception:
            logger.exception("Failed to apply gc.freeze after warmup/precompile.")

    # Generate the prefix
    prefix = ""
    if server_args.nnodes > 1:
        prefix += f" NP{server_args.node_rank}"
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"

    _set_scheduler_logical_device_count(server_args, update_env=True)

    # Config the process
    kill_itself_when_parent_died()
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args)
        maybe_freeze_gc_after_warmup()
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()

    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)


def run_scheduler_loop_thread_after_create(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: int | None = None,
):
    def maybe_freeze_gc_after_warmup():
        if not getattr(server_args, "enable_gc_freeze", False):
            return
        if not hasattr(gc, "freeze"):
            logger.warning(
                "GC freeze requested but gc.freeze is unavailable on this Python runtime."
            )
            return
        try:
            freeze_before = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
            collected = gc.collect()
            gc.freeze()
            freeze_after = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
            logger.info(
                "Applied gc.freeze after warmup/precompile. collected=%d freeze_before=%d freeze_after=%d gc_count=%s",
                collected,
                freeze_before,
                freeze_after,
                gc.get_count(),
            )
            if getattr(server_args, "gc_freeze_rollback", False):
                if hasattr(gc, "unfreeze"):
                    gc.unfreeze()
                    rollback_count = (
                        gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else -1
                    )
                    logger.warning(
                        "Rolled back gc.freeze due to --gc-freeze-rollback. freeze_count_after_rollback=%d gc_count=%s",
                        rollback_count,
                        gc.get_count(),
                    )
                else:
                    logger.warning(
                        "GC freeze rollback requested but gc.unfreeze is unavailable on this Python runtime."
                    )
        except Exception:
            logger.exception("Failed to apply gc.freeze after warmup/precompile.")

    current_process = psutil.Process()
    # Create a scheduler and run the event loop
    try:
        _set_scheduler_logical_device_count(server_args, update_env=False)
        scheduler = Scheduler(server_args, port_args)
        maybe_freeze_gc_after_warmup()
        scheduler_thread = threading.Thread(
            target=scheduler_loop_after_create,
            args=(server_args, scheduler, dp_rank),
            daemon=True,
        )
        scheduler_thread.start()
        return {
            "status": "ready",
            "max_total_num_tokens": scheduler.max_total_num_tokens,
            "max_req_input_len": scheduler.max_req_input_len,
            "scheduler": scheduler,
            "scheduler_thread": scheduler_thread,
        }
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        current_process.send_signal(signal.SIGQUIT)


def scheduler_loop_after_create(server_args, scheduler, dp_rank: int | None = None):
    # Generate the prefix
    prefix = ""
    if server_args.nnodes > 1:
        prefix += f" NP{server_args.node_rank}"
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"

    # Config the process
    current_thread = threading.current_thread()
    current_thread.name = f"sglang::scheduler{prefix.replace(' ', '_')}"
    faulthandler.enable()
    current_process = psutil.Process()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    try:
        _set_scheduler_logical_device_count(server_args, update_env=False)
        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        current_process.send_signal(signal.SIGQUIT)
