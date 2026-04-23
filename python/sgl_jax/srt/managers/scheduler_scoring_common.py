"""Scoring-specific scheduler helpers."""

# ruff: noqa: F401

import concurrent.futures as futures
import logging
import math
import os
import queue
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import numpy as np
import zmq
from jax import numpy as jnp
from jax.scipy import special as jsp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.managers.io_struct import (
    ReleaseScoringCacheReqInput,
    ReleaseScoringCacheReqOutput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    Req,
    ScheduleBatch,
    acc_global_bid,
)
from sgl_jax.srt.managers.utils import validate_input_length
from sgl_jax.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import get_device_name

logger = logging.getLogger(__name__)

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
class _LocalSchedulerRpcEnvelope:
    req_obj: object
    result_future: futures.Future | None = None

__all__ = [name for name in globals() if not name.startswith("__")]
