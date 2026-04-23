import asyncio
import concurrent.futures as futures
import math
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import zmq
from jax.sharding import Mesh

import sgl_jax.srt.entrypoints.engine as engine_module
import sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes as tuned_block_sizes
import sgl_jax.srt.layers.sampler as sampler_module
import sgl_jax.srt.managers.scheduler as scheduler_module
from sgl_jax.srt.entrypoints.engine import _resolve_dp_scheduler_device_partitions
from sgl_jax.srt.layers.logits_processor import (
    LogitsProcessor,
    _compute_next_token_shared_token_ids_logprobs_chunked,
    _compute_next_token_token_ids_logprobs_chunked,
)
from sgl_jax.srt.layers.sampler import Sampler
from sgl_jax.srt.layers.sampler import (
    get_token_ids_logprobs as sampler_get_token_ids_logprobs,
)
from sgl_jax.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseScoringCacheReqInput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sgl_jax.srt.server_args import PortArgs, ServerArgs


class _HashableNamespace(SimpleNamespace):
    __hash__ = object.__hash__


class _FakeCreateTokenizedManager:
    _create_tokenized_object = TokenizerManager._create_tokenized_object

    def __init__(self):
        self.preferred_sampling_params = None
        self.tokenizer = None
        self.model_config = SimpleNamespace(vocab_size=32000)


class _FakePrefillManager:
    _prefill_and_cache = TokenizerManager._prefill_and_cache

    def __init__(self):
        self.seen_requests = []

    async def generate_request(self, req, request=None):
        del request
        self.seen_requests.append(req)
        yield {"meta_info": {"id": req.rid}}


class _FakeExtendManager:
    _batched_extend_score = TokenizerManager._batched_extend_score
    _batched_extend_score_with_metrics = TokenizerManager._batched_extend_score_with_metrics

    def __init__(self):
        self.seen_request = None

    async def generate_request(self, req, request=None):
        del request
        self.seen_request = req
        yield [
            {
                "index": 0,
                "meta_info": {
                    "output_token_ids_logprobs": [
                        [
                            (math.log(0.9), 10, None),
                            (math.log(0.1), 20, None),
                        ]
                    ]
                },
            },
            {
                "index": 1,
                "meta_info": {
                    "output_token_ids_logprobs": [
                        [
                            (math.log(0.2), 10, None),
                            (math.log(0.8), 20, None),
                        ]
                    ]
                },
            },
        ]


class _FakeExtendMissingLogprobsManager:
    _batched_extend_score = TokenizerManager._batched_extend_score
    _batched_extend_score_with_metrics = TokenizerManager._batched_extend_score_with_metrics

    async def generate_request(self, req, request=None):
        del req, request
        yield [{"index": 0, "meta_info": {"output_token_ids_logprobs": []}}]


class _FakeReleaseCacheCommunicator:
    def __init__(self, outputs=None, delay_s: float = 0.0, raise_exc: Exception | None = None):
        self.outputs = outputs
        self.delay_s = delay_s
        self.raise_exc = raise_exc

    async def __call__(self, req, timeout=None, scheduler_idx=None, broadcast=False):
        del req, timeout, scheduler_idx, broadcast
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.outputs


class _FakeReleaseCacheManager:
    _release_cache = TokenizerManager._release_cache
    _can_use_local_score_rpc = TokenizerManager._can_use_local_score_rpc
    _submit_local_score_rpc = TokenizerManager._submit_local_score_rpc

    def __init__(self, communicator, timeout_s: float):
        self.release_scoring_cache_communicator = communicator
        self.server_args = SimpleNamespace(multi_item_prefill_extend_cache_timeout=timeout_s)
        self.local_rpc_submitter = None
        self.local_request_submitter = None

    def auto_create_handle_loop(self):
        return None

    def _scheduler_sender_fan_out(self):
        return 1


class _FailingCommunicator:
    async def __call__(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("Communicator should not be used when local RPC is enabled.")


class _FakeLocalScoreRpcManager:
    _score_from_cache_fastpath_v2 = TokenizerManager._score_from_cache_fastpath_v2
    _release_cache = TokenizerManager._release_cache
    _can_use_local_score_rpc = TokenizerManager._can_use_local_score_rpc
    _submit_local_score_rpc = TokenizerManager._submit_local_score_rpc

    def __init__(self):
        self.server_args = SimpleNamespace(
            multi_item_prefill_extend_cache_timeout=1.0,
            multi_item_score_from_cache_v2_items_per_step=64,
            multi_item_score_local_rpc_min_items=1,
        )
        self.score_from_cache_v2_communicator = _FailingCommunicator()
        self.release_scoring_cache_communicator = _FailingCommunicator()
        self.local_rpc_requests = []
        self.local_rpc_submitter = self._submit_local_rpc
        self.local_request_submitter = None

    def auto_create_handle_loop(self):
        return None

    def _scheduler_sender_fan_out(self):
        return 1

    def _submit_local_rpc(self, req):
        self.local_rpc_requests.append(req)
        future = futures.Future()
        if isinstance(req, ScoreFromCacheReqInput):
            future.set_result(
                ScoreFromCacheReqOutput(
                    rid=req.rid,
                    success=True,
                    scores=[[0.25], [0.75]],
                    effective_items_per_step=req.items_per_step,
                )
            )
        elif isinstance(req, ReleaseScoringCacheReqInput):
            future.set_result(
                SimpleNamespace(
                    rid=req.rid,
                    success=True,
                    released_items=1,
                    error_msg="",
                )
            )
        else:
            future.set_exception(AssertionError(f"Unexpected local RPC request: {req!r}"))
        return future


class _FakeLocalRequestIngressManager:
    _send_one_request = TokenizerManager._send_one_request
    _send_batch_requests = TokenizerManager._send_batch_requests
    _can_use_local_request_ingress = TokenizerManager._can_use_local_request_ingress

    def __init__(self):
        self.server_args = SimpleNamespace()
        self.local_request_submitter = self._submit_local_request
        self.local_rpc_submitter = None
        self.rid_to_state = {}
        self.local_requests = []
        self.send_to_scheduler = SimpleNamespace(
            send_pyobj=lambda _: (_ for _ in ()).throw(
                AssertionError("send_to_scheduler.send_pyobj should not be used.")
            ),
            send_pyobj_all=lambda _: (_ for _ in ()).throw(
                AssertionError("send_to_scheduler.send_pyobj_all should not be used.")
            ),
            send_pyobj_to=lambda *_: (_ for _ in ()).throw(
                AssertionError("send_to_scheduler.send_pyobj_to should not be used.")
            ),
        )

    def _raise_if_scheduler_unavailable(self):
        return None

    def _scheduler_sender_fan_out(self):
        return 1

    def _submit_local_request(self, req):
        self.local_requests.append((req, sorted(self.rid_to_state.keys())))


class _FakeStandardSendManager:
    _send_one_request = TokenizerManager._send_one_request
    _can_use_local_request_ingress = TokenizerManager._can_use_local_request_ingress

    def __init__(self):
        self.server_args = SimpleNamespace()
        self.local_request_submitter = None
        self.local_rpc_submitter = None
        self.rid_to_state = {}
        self.sent = []
        self.send_to_scheduler = SimpleNamespace(
            send_pyobj=self.sent.append,
            send_pyobj_all=lambda obj: (_ for _ in ()).throw(
                AssertionError(f"send_pyobj_all unexpectedly called with {obj!r}")
            ),
            send_pyobj_to=lambda *_: (_ for _ in ()).throw(
                AssertionError("send_pyobj_to unexpectedly called")
            ),
        )

    def _raise_if_scheduler_unavailable(self):
        return None

    def _scheduler_sender_fan_out(self):
        return 1


class _FakePublicScoringCacheManager:
    prefill_scoring_cache = TokenizerManager.prefill_scoring_cache
    score_from_cache = TokenizerManager.score_from_cache
    release_scoring_cache = TokenizerManager.release_scoring_cache
    _normalize_score_query_tokens = TokenizerManager._normalize_score_query_tokens
    _normalize_score_item_tokens = TokenizerManager._normalize_score_item_tokens
    _validate_label_token_ids_for_score = TokenizerManager._validate_label_token_ids_for_score
    _resolve_score_from_cache_v2_items_per_step = (
        TokenizerManager._resolve_score_from_cache_v2_items_per_step
    )

    def __init__(self):
        class _FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [len(text), int(add_special_tokens)]

            def __len__(self):
                return 32000

        self.tokenizer = _FakeTokenizer()
        self.server_args = SimpleNamespace(
            multi_item_score_from_cache_v2_items_per_step=64,
            multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget=False,
            multi_item_score_from_cache_v2_token_budget=0,
            multi_item_score_from_cache_v2_min_items_per_step=1,
        )
        self.prefill_calls = []
        self.release_calls = []
        self.last_score_kwargs = None

    async def _prefill_and_cache(self, query_tokens):
        self.prefill_calls.append(list(query_tokens))
        return "cache-handle-public"

    async def _score_from_cache_fastpath_v2(self, **kwargs):
        self.last_score_kwargs = kwargs
        return ScoreFromCacheReqOutput(
            success=True,
            scores=[[float(idx)] for idx, _ in enumerate(kwargs["items"])],
        )

    async def _release_cache(self, cache_handle):
        self.release_calls.append(cache_handle)
        return True


class _FakePrefillOnlyTreeCache:
    def __init__(self):
        self.finished_rids = []
        self.unfinished_rids = []
        self.locked_nodes = []

    def cache_finished_req(self, req):
        self.finished_rids.append(req.rid)

    def cache_unfinished_req(self, req):
        self.unfinished_rids.append(req.rid)

    def match_prefix(self, key):
        del key
        return np.arange(3, dtype=np.int32), "cached-node", None, 0

    def inc_lock_ref(self, node):
        self.locked_nodes.append(node)


class _FakePrefillOnlyReq:
    def __init__(self, rid: str):
        self.rid = rid
        self.is_retracted = False
        self.is_chunked = 0
        self.latest_bid = None
        self.output_ids = []
        self.finished_reason = None
        self.to_finish = None
        self.cache_for_scoring = True
        self.origin_input_ids = [101, 102, 103]
        self.extra_key = None
        self.return_output_logprob_only = False
        self.token_ids_logprob = None
        self.return_logprob = False
        self.return_hidden_states = False
        self.hidden_states = []
        self.grammar = None
        self.device_compute_time_s = 0.0
        self.host_overhead_time_s = 0.0
        self.scheduler_dispatch_count = 0

    def finished(self) -> bool:
        return self.finished_reason is not None

    def check_finished(self, new_accepted_len: int = 1):
        del new_accepted_len
        self.finished_reason = {"type": "length", "length": 0}


class _FakeSchedulerPrefillNoSample:
    process_batch_result_prefill = SchedulerOutputProcessorMixin.process_batch_result_prefill
    _can_skip_sample_for_prefill_batch = staticmethod(Scheduler._can_skip_sample_for_prefill_batch)
    _normalize_scoring_cache_prefix_key = staticmethod(
        Scheduler._normalize_scoring_cache_prefix_key
    )
    _register_scoring_cache_handle = Scheduler._register_scoring_cache_handle
    _record_scoring_cache_lookup = Scheduler._record_scoring_cache_lookup

    def __init__(self):
        self.is_generation = True
        self.enable_overlap = False
        self.is_mixed_chunk = False
        self.tree_cache = _FakePrefillOnlyTreeCache()
        self.scoring_cache_nodes = {}
        self.scoring_cache_prefix_handles_by_key = {}
        self.scoring_cache_handle_to_prefix_key = {}
        self.scoring_cache_handles_created = 0
        self.scoring_cache_lookup_queries = 0
        self.scoring_cache_lookup_hits = 0
        self.scoring_cache_lookup_misses = 0
        self.scoring_cache_lookup_by_path = {
            "extend": {"queries": 0, "hits": 0, "misses": 0},
            "score_from_cache_v2": {"queries": 0, "hits": 0, "misses": 0},
            "cache_for_scoring": {"queries": 0, "hits": 0, "misses": 0},
        }
        self.scoring_cache_lookup_by_lane = {
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
        self.stream_calls = []

    def maybe_collect_routed_experts(self, req):
        del req

    def set_next_batch_sampling_info_done(self, batch):
        del batch

    def stream_output(
        self,
        reqs,
        return_logprob,
        return_output_logprob_only,
        skip_stream_req,
        cache_miss_count,
    ):
        self.stream_calls.append(
            (
                [req.rid for req in reqs],
                return_logprob,
                return_output_logprob_only,
                skip_stream_req,
                cache_miss_count,
            )
        )

    def _score_scheduler_lane_from_prefix_len(self, _scheduler, prefix_len: int) -> str:
        del _scheduler, prefix_len
        return "default"

    def _record_scoring_cache_handle_created(self) -> None:
        self.scoring_cache_handles_created += 1


class _FakeSchedulerDirectWarmup:
    _score_from_cache_v2_use_direct_label_only = (
        Scheduler._score_from_cache_v2_use_direct_label_only
    )
    _score_direct_warmup_spec = Scheduler._score_direct_warmup_spec

    def __init__(self):
        self.server_args = SimpleNamespace(
            multi_item_score_label_only_logprob=True,
            multi_item_score_direct_label_only=True,
            multi_item_score_direct_warmup_enable=True,
            multi_item_score_direct_warmup_prefix_len=2000,
            multi_item_score_direct_warmup_item_len=20,
            multi_item_score_direct_warmup_batch_size=0,
            multi_item_score_direct_warmup_label_count=2,
            multi_item_score_direct_warmup_apply_softmax=True,
            multi_item_score_direct_hot_shape_bs=512,
            multi_item_score_from_cache_v2_items_per_step=64,
        )


class _NoopSender:
    def __init__(self):
        self.calls = []
        self.fan_out = 1

    def send_pyobj(self, obj):
        self.calls.append(obj)

    def send_pyobj_to(self, scheduler_idx: int, obj):
        self.calls.append((scheduler_idx, obj))

    def send_pyobj_all(self, obj):
        self.calls.append(("all", obj))


class _FakeSchedulerLivenessManager:
    _send_one_request = TokenizerManager._send_one_request
    _send_batch_requests = TokenizerManager._send_batch_requests
    _wait_one_response = TokenizerManager._wait_one_response
    _build_scheduler_unavailable_message = TokenizerManager._build_scheduler_unavailable_message
    _fail_pending_requests = TokenizerManager._fail_pending_requests
    _mark_scheduler_unavailable = TokenizerManager._mark_scheduler_unavailable
    _check_scheduler_health = TokenizerManager._check_scheduler_health
    _raise_if_scheduler_unavailable = TokenizerManager._raise_if_scheduler_unavailable
    _scheduler_sender_fan_out = TokenizerManager._scheduler_sender_fan_out
    _score_lane_scheduler_index = TokenizerManager._score_lane_scheduler_index

    def __init__(self):
        self.wait_timeout = 0.01
        self.scheduler_pids = [4321]
        self.scheduler_unavailable_error = None
        self.health_check_failed = False
        self.rid_to_state = {}
        self.send_to_scheduler = _NoopSender()
        self.log_requests = False

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        del pid
        return False


class _FakeBatchRequestContainer:
    def __init__(self, requests: list[SimpleNamespace], stream: bool = False):
        self._requests = requests
        self.batch_size = len(requests)
        self.stream = stream
        self.parallel_sample_num = 1

    def __getitem__(self, index: int) -> SimpleNamespace:
        return self._requests[index]


class _FakeBatchHandleManager:
    _handle_batch_request = TokenizerManager._handle_batch_request

    def __init__(self, enable_batch_send: bool, enable_batch_encode: bool):
        self.server_args = SimpleNamespace(
            enable_tokenizer_batch_encode=enable_batch_encode,
            enable_tokenizer_batch_send=enable_batch_send,
        )
        self.sent_single = []
        self.sent_batch = []

    def _validate_batch_tokenization_constraints(self, batch_size, obj):
        del batch_size, obj
        return None

    async def _batch_tokenize_and_process(self, batch_size: int, obj):
        del obj
        return [SimpleNamespace(tokenized_idx=i) for i in range(batch_size)]

    async def _tokenize_one_request(self, obj):
        return SimpleNamespace(tokenized_single=obj.rid)

    def _send_one_request(self, obj, tokenized_obj, created_time=None):
        self.sent_single.append((obj.rid, tokenized_obj))
        return ReqState([], True, asyncio.Event(), obj, created_time=created_time)

    def _send_batch_requests(self, objs, tokenized_objs, created_time=None):
        self.sent_batch.append(([obj.rid for obj in objs], tokenized_objs))
        return [ReqState([], True, asyncio.Event(), obj, created_time=created_time) for obj in objs]

    async def _wait_one_response(self, obj, state, request=None):
        del state, request
        yield {"meta_info": {"id": obj.rid}, "text": "", "index": 0}


class _FakeIngressSocket:
    def __init__(self, payloads: list):
        self.payloads = list(payloads)

    def recv_pyobj(self, flags=None):
        del flags
        if self.payloads:
            return self.payloads.pop(0)
        raise zmq.ZMQError()


class _FakeDelayedIngressSocket:
    def __init__(self, first_payload, delayed_payload, release_after_calls: int):
        self.first_payload = first_payload
        self.delayed_payload = delayed_payload
        self.release_after_calls = release_after_calls
        self.calls = 0

    def recv_pyobj(self, flags=None):
        del flags
        self.calls += 1
        if self.first_payload is not None:
            payload = self.first_payload
            self.first_payload = None
            return payload
        if self.calls >= self.release_after_calls and self.delayed_payload is not None:
            payload = self.delayed_payload
            self.delayed_payload = None
            return payload
        raise zmq.ZMQError()


class _FakeSchedulerIngress:
    recv_requests = Scheduler.recv_requests

    def __init__(
        self,
        tokenizer_payloads: list,
        rpc_payloads: list,
        *,
        tokenizer_socket=None,
        rpc_socket=None,
        coalesce_window_s: float = 0.0,
        coalesce_poll_s: float = 0.0005,
    ):
        self.node_rank = 0
        self.nnodes = 1
        self.recv_from_tokenizer = tokenizer_socket or _FakeIngressSocket(tokenizer_payloads)
        self.recv_from_rpc = rpc_socket or _FakeIngressSocket(rpc_payloads)
        self.ingress_recv_calls = 0
        self.ingress_nonempty_calls = 0
        self.ingress_max_batch_size = 0
        self.ingress_tokenizer_frames = 0
        self.ingress_rpc_frames = 0
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
        self.score_scheduler_global_microbatch_window_s = coalesce_window_s
        self.score_scheduler_global_microbatch_poll_s = coalesce_poll_s
        self.score_scheduler_microbatch_windows = 0
        self.score_scheduler_microbatch_added_requests = 0
        self.score_scheduler_microbatch_max_added_requests = 0
        self.ingress_score_path_frames = {
            "tokenizer_multi_item_packed": 0,
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }


def test_create_tokenized_object_keeps_prefill_extend_fields():
    manager = _FakeCreateTokenizedManager()
    req = GenerateReqInput(
        rid="req-1",
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        token_ids_logprob=[10, 20],
        cache_for_scoring=True,
        extend_from_cache="cache-handle-1",
        is_single=True,
    )
    req.normalize_batch_and_arguments()

    tokenized = manager._create_tokenized_object(req, input_text=None, input_ids=req.input_ids)

    assert tokenized.cache_for_scoring is True
    assert tokenized.extend_from_cache == "cache-handle-1"


def test_prefill_and_cache_uses_single_request_and_stable_handle():
    manager = _FakePrefillManager()
    handle = asyncio.run(manager._prefill_and_cache([11, 12, 13]))

    assert len(manager.seen_requests) == 1
    req = manager.seen_requests[0]
    assert req.input_ids == [11, 12, 13]
    assert req.is_single is True
    assert req.cache_for_scoring is True
    assert isinstance(req.rid, str)
    assert handle == req.rid


def test_batched_extend_score_passes_cache_handle_and_scores_items():
    manager = _FakeExtendManager()
    scores = asyncio.run(
        manager._batched_extend_score(
            cache_handle="cache-handle-xyz",
            items=[[1], [2]],
            label_token_ids=[10, 20],
            apply_softmax=False,
        )
    )

    assert manager.seen_request is not None
    assert manager.seen_request.extend_from_cache == "cache-handle-xyz"
    assert manager.seen_request.input_ids == [[1], [2]]
    assert manager.seen_request.return_logprob is True
    assert manager.seen_request.return_output_logprob_only is False
    assert manager.seen_request.token_ids_logprob == [10, 20]
    assert manager.seen_request.logprob_start_len is None
    assert len(scores) == 2
    assert scores[0] == pytest.approx([0.9, 0.1])
    assert scores[1] == pytest.approx([0.2, 0.8])


def test_batched_extend_score_raises_when_output_logprobs_missing():
    manager = _FakeExtendMissingLogprobsManager()
    with pytest.raises(RuntimeError, match="output_token_ids_logprobs is empty"):
        asyncio.run(
            manager._batched_extend_score(
                cache_handle="cache-handle-xyz",
                items=[[1]],
                label_token_ids=[10, 20],
                apply_softmax=False,
            )
        )


def test_release_cache_returns_true_on_success():
    manager = _FakeReleaseCacheManager(
        communicator=_FakeReleaseCacheCommunicator(
            outputs=[SimpleNamespace(success=True, released_items=1, error_msg="")]
        ),
        timeout_s=1.0,
    )
    assert asyncio.run(manager._release_cache("cache-handle-ok")) is True


def test_release_cache_times_out_and_returns_false():
    manager = _FakeReleaseCacheManager(
        communicator=_FakeReleaseCacheCommunicator(outputs=[], delay_s=0.2),
        timeout_s=0.01,
    )
    assert asyncio.run(manager._release_cache("cache-handle-timeout")) is False


def test_release_cache_uses_local_rpc_submitter_when_available():
    manager = _FakeLocalScoreRpcManager()

    released = asyncio.run(manager._release_cache("cache-handle-local"))

    assert released is True
    assert len(manager.local_rpc_requests) == 1
    assert isinstance(manager.local_rpc_requests[0], ReleaseScoringCacheReqInput)
    assert manager.local_rpc_requests[0].rid == "cache-handle-local"


def test_score_from_cache_fastpath_uses_local_rpc_submitter_when_available():
    manager = _FakeLocalScoreRpcManager()

    out = asyncio.run(
        manager._score_from_cache_fastpath_v2(
            cache_handle="cache-handle-local",
            items=[[1, 2], [3, 4]],
            label_token_ids=[198],
            apply_softmax=False,
            items_per_step=64,
            token_budget=0,
            max_total_tokens=0,
        )
    )

    assert out.success is True
    assert out.scores == [[0.25], [0.75]]
    assert len(manager.local_rpc_requests) == 1
    req = manager.local_rpc_requests[0]
    assert isinstance(req, ScoreFromCacheReqInput)
    assert req.cache_handle == "cache-handle-local"
    assert req.items_2d == [[1, 2], [3, 4]]
    assert req.label_token_ids == [198]
    assert req.items_per_step == 64


def test_local_score_rpc_threshold_blocks_small_batches():
    manager = _FakeLocalScoreRpcManager()
    manager.server_args.multi_item_score_local_rpc_min_items = 256

    assert manager._can_use_local_score_rpc(total_items=80) is False
    assert manager._can_use_local_score_rpc(total_items=500) is True


def test_public_prefill_scoring_cache_tokenizes_text_query():
    manager = _FakePublicScoringCacheManager()

    handle = asyncio.run(manager.prefill_scoring_cache("abcd"))

    assert handle == "cache-handle-public"
    assert manager.prefill_calls == [[4, 0]]


def test_public_score_from_cache_uses_fastpath_directly():
    manager = _FakePublicScoringCacheManager()

    scores = asyncio.run(
        manager.score_from_cache(
            cache_handle="cache-handle-public",
            items=["aa", "bbbb"],
            label_token_ids=[10, 20],
            apply_softmax=False,
        )
    )

    assert scores == [[0.0], [1.0]]
    assert manager.last_score_kwargs is not None
    assert manager.last_score_kwargs["cache_handle"] == "cache-handle-public"
    assert manager.last_score_kwargs["items"] == [[2, 0], [4, 0]]
    assert manager.last_score_kwargs["items_per_step"] == 64


def test_public_release_scoring_cache_delegates_to_internal_release():
    manager = _FakePublicScoringCacheManager()

    released = asyncio.run(manager.release_scoring_cache("cache-handle-public"))

    assert released is True
    assert manager.release_calls == ["cache-handle-public"]


def test_process_batch_result_prefill_finishes_cache_for_scoring_without_output_token():
    scheduler = _FakeSchedulerPrefillNoSample()
    req = _FakePrefillOnlyReq("prefill-cache")
    batch = SimpleNamespace(
        reqs=[req],
        bid=17,
        return_output_logprob_only=False,
        return_logprob=False,
        is_prefill_only=True,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        decoding_reqs=None,
        cache_miss_count=0,
        spec_info=None,
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(
            next_token_logprobs=None,
            input_token_logprobs=None,
            hidden_states=None,
            next_token_token_ids_logprobs_val=None,
            next_token_token_ids_logprobs_idx=None,
        ),
        next_token_ids=[],
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        cache_miss_count=0,
        bid=17,
        next_draft_input=None,
    )

    scheduler.process_batch_result_prefill(batch, result)

    assert req.output_ids == []
    assert req.finished() is True
    assert scheduler.tree_cache.finished_rids == ["prefill-cache"]
    assert scheduler.scoring_cache_nodes["prefill-cache"][0] == "cached-node"
    assert scheduler.scoring_cache_handles_created == 1


def test_score_direct_warmup_spec_prefers_hot_shape_batch_size():
    scheduler = _FakeSchedulerDirectWarmup()

    warmup = scheduler._score_direct_warmup_spec()

    assert warmup is not None
    assert warmup.prefix_len == 2000
    assert warmup.item_len == 20
    assert warmup.batch_size == 512
    assert warmup.label_count == 2
    assert warmup.apply_softmax is True


def test_process_input_requests_sets_local_rpc_future_without_tokenizer_send():
    future = futures.Future()
    req = ScoreFromCacheReqInput(
        rid="scorev2-local",
        cache_handle="cache-handle-local",
        items_2d=[[1, 2]],
        label_token_ids=[198],
    )
    output = ScoreFromCacheReqOutput(rid=req.rid, success=True, scores=[[0.5]])

    class _FakeLocalSchedulerDispatch:
        process_input_requests = Scheduler.process_input_requests

        def __init__(self):
            self.sent = []
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=self.sent.append)
            self._request_dispatcher = lambda recv_req: output

        def _evict_expired_scoring_cache_nodes(self):
            return None

    scheduler = _FakeLocalSchedulerDispatch()
    envelope = scheduler_module._LocalSchedulerRpcEnvelope(req_obj=req, result_future=future)

    scheduler.process_input_requests([envelope])

    assert future.result(timeout=0.0) == output
    assert scheduler.sent == []


def test_score_from_cache_v2_resolve_direct_hot_shape_clamps_tokens_by_rounding():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.page_size = 32
    scheduler.server_args.multi_item_score_direct_hot_shape_bs = 512
    scheduler.server_args.multi_item_score_direct_hot_shape_tokens = 8192
    scheduler.server_args.multi_item_score_direct_hot_shape_token_rounding = 512
    scheduler.server_args.multi_item_score_direct_hot_shape_token_rounding_min_hot_tokens = 8192

    padded_bs, padded_input_tokens, padded_cache_loc_tokens = (
        scheduler._score_from_cache_v2_resolve_direct_hot_shape(
            real_bs=500,
            real_input_tokens=5000,
            real_cache_loc_tokens=592000,
            max_seq_len=1160,
        )
    )

    assert padded_bs == 512
    assert padded_input_tokens == 5120
    assert padded_cache_loc_tokens == 606208


def test_score_from_cache_v2_resolve_direct_hot_shape_rounding_disabled_keeps_hot_tokens():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.page_size = 32
    scheduler.server_args.multi_item_score_direct_hot_shape_bs = 512
    scheduler.server_args.multi_item_score_direct_hot_shape_tokens = 8192

    padded_bs, padded_input_tokens, padded_cache_loc_tokens = (
        scheduler._score_from_cache_v2_resolve_direct_hot_shape(
            real_bs=500,
            real_input_tokens=5000,
            real_cache_loc_tokens=592000,
            max_seq_len=1160,
        )
    )

    assert padded_bs == 512
    assert padded_input_tokens == 8192
    assert padded_cache_loc_tokens == 606208


def test_score_from_cache_v2_resolve_direct_hot_shape_rounding_requires_real_savings():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.page_size = 32
    scheduler.server_args.multi_item_score_direct_hot_shape_bs = 64
    scheduler.server_args.multi_item_score_direct_hot_shape_tokens = 4096
    scheduler.server_args.multi_item_score_direct_hot_shape_token_rounding = 512

    padded_bs, padded_input_tokens, padded_cache_loc_tokens = (
        scheduler._score_from_cache_v2_resolve_direct_hot_shape(
            real_bs=50,
            real_input_tokens=4000,
            real_cache_loc_tokens=62400,
            max_seq_len=1230,
        )
    )

    assert padded_bs == 64
    assert padded_input_tokens == 4096
    assert padded_cache_loc_tokens == 79872


def test_score_from_cache_v2_resolve_direct_hot_shape_rounding_floor_keeps_smaller_hot_shape():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.page_size = 32
    scheduler.server_args.multi_item_score_direct_hot_shape_bs = 128
    scheduler.server_args.multi_item_score_direct_hot_shape_tokens = 4096
    scheduler.server_args.multi_item_score_direct_hot_shape_token_rounding = 512
    scheduler.server_args.multi_item_score_direct_hot_shape_token_rounding_min_hot_tokens = 8192

    padded_bs, padded_input_tokens, padded_cache_loc_tokens = (
        scheduler._score_from_cache_v2_resolve_direct_hot_shape(
            real_bs=100,
            real_input_tokens=2600,
            real_cache_loc_tokens=170752,
            max_seq_len=1026,
        )
    )

    assert padded_bs == 128
    assert padded_input_tokens == 4096
    assert padded_cache_loc_tokens == 170752


def test_send_one_request_uses_local_request_submitter_for_cache_prefill():
    manager = _FakeLocalRequestIngressManager()
    obj = GenerateReqInput(
        input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 0}, rid="prefill-local"
    )
    tokenized_obj = TokenizedGenerateReqInput(
        rid="prefill-local",
        input_ids=[1, 2, 3],
        sampling_params=obj.sampling_params,
        stream=False,
        cache_for_scoring=True,
    )

    state = manager._send_one_request(obj, tokenized_obj)

    assert state is manager.rid_to_state["prefill-local"]
    assert manager.local_requests == [(tokenized_obj, ["prefill-local"])]


def test_send_one_request_keeps_standard_path_for_non_cache_requests():
    manager = _FakeStandardSendManager()
    obj = GenerateReqInput(input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 0}, rid="normal")
    tokenized_obj = TokenizedGenerateReqInput(
        rid="normal",
        input_ids=[1, 2, 3],
        sampling_params=obj.sampling_params,
        stream=False,
        cache_for_scoring=False,
    )

    state = manager._send_one_request(obj, tokenized_obj)

    assert state is manager.rid_to_state["normal"]
    assert manager.sent == [tokenized_obj]


class _FakeReqToTokenPool:
    def __init__(self, available_size: int):
        self._available_size = available_size

    def available_size(self) -> int:
        return self._available_size


class _FakeRunningBatch:
    def __init__(self):
        self.batch_is_full = False
        self.reqs = []

    def is_empty(self) -> bool:
        return len(self.reqs) == 0


class _FakeSchedulerCacheOps:
    _unpack_scoring_cache_entry = Scheduler._unpack_scoring_cache_entry
    _normalize_scoring_cache_prefix_key = Scheduler._normalize_scoring_cache_prefix_key
    _register_scoring_cache_handle = Scheduler._register_scoring_cache_handle
    _unregister_scoring_cache_handle = Scheduler._unregister_scoring_cache_handle
    _release_scoring_cache_entry = Scheduler._release_scoring_cache_entry
    _touch_scoring_cache_entry = Scheduler._touch_scoring_cache_entry
    _evict_expired_scoring_cache_nodes = Scheduler._evict_expired_scoring_cache_nodes
    _resolve_extend_from_cache = Scheduler._resolve_extend_from_cache
    _record_scoring_cache_lookup = Scheduler._record_scoring_cache_lookup
    _record_scoring_cache_handle_released = Scheduler._record_scoring_cache_handle_released

    def __init__(self, timeout_s: float):
        self.scoring_cache_timeout = timeout_s
        self._last_scoring_cache_gc = 0.0
        self.scoring_cache_nodes = {}
        self.scoring_cache_lookup_queries = 0
        self.scoring_cache_lookup_hits = 0
        self.scoring_cache_lookup_misses = 0
        self.scoring_cache_lookup_by_path = {
            "extend": {"queries": 0, "hits": 0, "misses": 0},
            "score_from_cache_v2": {"queries": 0, "hits": 0, "misses": 0},
            "cache_for_scoring": {"queries": 0, "hits": 0, "misses": 0},
        }
        self.scoring_cache_lookup_by_lane = {
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
        self.scoring_cache_prefix_handles_by_key = {}
        self.scoring_cache_handle_to_prefix_key = {}
        self.score_scheduler_short_prompt_tokens_threshold = 8
        self.scoring_cache_handles_released = 0
        self.scoring_cache_handles_released_manual = 0
        self.scoring_cache_handles_released_expired = 0
        self.scoring_cache_handles_released_other = 0
        self.scoring_cache_handles_missing_node = 0
        self.tree_cache = SimpleNamespace(dec_lock_ref=lambda *args, **kwargs: None)


def test_resolve_extend_from_cache_missing_handle_returns_error():
    scheduler = _FakeSchedulerCacheOps(timeout_s=60.0)
    recv_req = SimpleNamespace(
        rid="req-missing",
        input_ids=[101, 102],
        extra_key=None,
        extend_from_cache="missing-handle",
        sampling_params=SimpleNamespace(max_new_tokens=0),
    )

    cached_prefix_ctx, err = scheduler._resolve_extend_from_cache(recv_req)

    assert cached_prefix_ctx is None
    assert "Missing scoring cache handle" in err
    assert recv_req.input_ids == [101, 102]


def test_resolve_extend_from_cache_merges_prefix_and_suffix():
    scheduler = _FakeSchedulerCacheOps(timeout_s=0.0)
    scheduler.scoring_cache_nodes["cache-1"] = (
        "node",
        None,
        [1, 2, 3],
        np.array([0, 1, 2], dtype=np.int32),
        "extra-key",
        0.0,
    )
    recv_req = SimpleNamespace(
        rid="req-hit",
        input_ids=[11, 12],
        extra_key=None,
        extend_from_cache="cache-1",
        sampling_params=SimpleNamespace(max_new_tokens=0),
    )

    cached_prefix_ctx, err = scheduler._resolve_extend_from_cache(recv_req)

    assert err is None
    assert cached_prefix_ctx is not None
    assert recv_req.input_ids == [1, 2, 3, 11, 12]
    assert recv_req.extra_key == "extra-key"


def test_evict_expired_scoring_cache_nodes_removes_stale_entries():
    scheduler = _FakeSchedulerCacheOps(timeout_s=10.0)
    scheduler.scoring_cache_nodes["cache-stale"] = (
        None,
        None,
        [1, 2, 3],
        np.array([0, 1, 2], dtype=np.int32),
        None,
        0.0,
    )

    removed = scheduler._evict_expired_scoring_cache_nodes(now=20.0)
    assert removed == 1
    assert "cache-stale" not in scheduler.scoring_cache_nodes


def test_record_scoring_cache_lookup_tracks_lane_metrics():
    scheduler = _FakeSchedulerCacheOps(timeout_s=0.0)
    scheduler._record_scoring_cache_lookup(path="extend", hit=True, lane_name="short")
    scheduler._record_scoring_cache_lookup(path="extend", hit=False, lane_name="long")

    assert scheduler.scoring_cache_lookup_by_lane["extend"]["short"]["queries"] == 1
    assert scheduler.scoring_cache_lookup_by_lane["extend"]["short"]["hits"] == 1
    assert scheduler.scoring_cache_lookup_by_lane["extend"]["long"]["queries"] == 1
    assert scheduler.scoring_cache_lookup_by_lane["extend"]["long"]["misses"] == 1


def test_scheduler_req_slot_exhaustion_does_not_stick_batch_full():
    """Scheduler should defer prefill when req slots are exhausted without deadlocking future rounds."""
    scheduler = SimpleNamespace(
        grammar_queue=[],
        move_ready_grammar_requests=lambda: None,
        running_batch=_FakeRunningBatch(),
        waiting_queue=[object()],
        chunked_req=None,
        max_running_requests=8,
        req_to_token_pool=_FakeReqToTokenPool(available_size=0),
    )

    new_batch = Scheduler.get_new_batch_prefill(scheduler)
    assert new_batch is None
    assert scheduler.running_batch.batch_is_full is True

    # Simulate a later scheduler round after pressure is relieved.
    scheduler.req_to_token_pool._available_size = 1
    scheduler.waiting_queue = []
    new_batch = Scheduler.get_new_batch_prefill(scheduler)

    assert new_batch is None
    # Important for liveness: soft throttle clears when the running batch is idle.
    assert scheduler.running_batch.batch_is_full is False


def test_scheduler_admission_lane_classification_for_score_traffic():
    scheduler = SimpleNamespace(score_scheduler_short_prompt_tokens_threshold=8)
    short_req = SimpleNamespace(
        origin_input_ids=[1] * 4,
        return_logprob=True,
        sampling_params=SimpleNamespace(max_new_tokens=0),
        is_multi_item_scoring=False,
        cache_for_scoring=True,
        extend_from_cache=None,
    )
    long_req = SimpleNamespace(
        origin_input_ids=[1] * 12,
        return_logprob=True,
        sampling_params=SimpleNamespace(max_new_tokens=0),
        is_multi_item_scoring=False,
        cache_for_scoring=True,
        extend_from_cache=None,
    )
    gen_req = SimpleNamespace(
        origin_input_ids=[1] * 4,
        return_logprob=False,
        sampling_params=SimpleNamespace(max_new_tokens=16),
        is_multi_item_scoring=False,
        cache_for_scoring=False,
        extend_from_cache=None,
    )

    assert Scheduler._admission_lane(scheduler, short_req) == "short"
    assert Scheduler._admission_lane(scheduler, long_req) == "long"
    assert Scheduler._admission_lane(scheduler, gen_req) == "default"


def test_scheduler_recv_requests_score_coalescing_adds_delayed_frame():
    first = TokenizedGenerateReqInput(
        rid="tok-now",
        input_ids=[1, 2],
        sampling_params={},
        cache_for_scoring=True,
    )
    delayed = TokenizedGenerateReqInput(
        rid="tok-later",
        input_ids=[1, 3],
        sampling_params={},
        cache_for_scoring=True,
    )
    scheduler = _FakeSchedulerIngress(
        tokenizer_payloads=[],
        rpc_payloads=[],
        tokenizer_socket=_FakeDelayedIngressSocket(
            first_payload=first,
            delayed_payload=delayed,
            release_after_calls=4,
        ),
        coalesce_window_s=0.002,
        coalesce_poll_s=0.0001,
    )

    recv_reqs = scheduler.recv_requests()

    assert [req.rid for req in recv_reqs] == ["tok-now", "tok-later"]
    assert scheduler.score_scheduler_microbatch_windows == 1
    assert scheduler.score_scheduler_microbatch_added_requests == 1
    assert scheduler.score_scheduler_microbatch_max_added_requests == 1


def test_scheduler_iter_waiting_queue_lane_isolation_prioritizes_short():
    scheduler = SimpleNamespace(
        score_scheduler_enable_lane_isolation=True,
        score_scheduler_lane_isolation_short_burst=1,
        score_scheduler_lane_isolation_long_burst=1,
        score_scheduler_short_prompt_tokens_threshold=8,
    )

    def _make_score_req(rid: str, prompt_len: int):
        return SimpleNamespace(
            rid=rid,
            origin_input_ids=[1] * prompt_len,
            return_logprob=True,
            sampling_params=SimpleNamespace(max_new_tokens=0),
            is_multi_item_scoring=False,
            cache_for_scoring=True,
            extend_from_cache=None,
        )

    waiting_queue = [
        _make_score_req("long-a", 16),
        _make_score_req("long-b", 12),
        _make_score_req("short-a", 4),
        _make_score_req("short-b", 6),
    ]

    ordered = Scheduler._iter_waiting_queue(scheduler, waiting_queue)

    assert [req.rid for req in ordered] == ["short-a", "long-a", "short-b", "long-b"]
    assert scheduler.score_scheduler_lane_isolation_rounds == 1
    assert scheduler.score_scheduler_lane_isolation_selected["short"] == 2
    assert scheduler.score_scheduler_lane_isolation_selected["long"] == 2


def test_scheduler_iter_waiting_queue_cache_bias_prioritizes_cache_hits():
    scheduler = SimpleNamespace(
        score_scheduler_enable_lane_isolation=False,
        score_scheduler_cache_admission_bias_enable=True,
        score_scheduler_cache_admission_bias_require_hit=True,
        scoring_cache_nodes={"cache-hit": object()},
        scoring_cache_prefix_handles_by_key={("", (1, 2, 3)): {"cache-prefix"}},
        score_scheduler_cache_admission_candidates={"default": 0, "short": 0, "long": 0},
        score_scheduler_cache_admission_promoted={"default": 0, "short": 0, "long": 0},
    )
    req_miss = _HashableNamespace(
        rid="miss",
        extend_from_cache="cache-miss",
        cache_for_scoring=False,
        origin_input_ids=[1, 2],
        extra_key=None,
        return_logprob=True,
        sampling_params=SimpleNamespace(max_new_tokens=0),
        is_multi_item_scoring=False,
    )
    req_hit = _HashableNamespace(
        rid="hit",
        extend_from_cache="cache-hit",
        cache_for_scoring=False,
        origin_input_ids=[1, 2],
        extra_key=None,
        return_logprob=True,
        sampling_params=SimpleNamespace(max_new_tokens=0),
        is_multi_item_scoring=False,
    )
    req_prefix = _HashableNamespace(
        rid="prefix",
        extend_from_cache=None,
        cache_for_scoring=True,
        origin_input_ids=[1, 2, 3],
        extra_key=None,
        return_logprob=True,
        sampling_params=SimpleNamespace(max_new_tokens=0),
        is_multi_item_scoring=False,
    )

    ordered = Scheduler._iter_waiting_queue(scheduler, [req_miss, req_prefix, req_hit])

    assert [req.rid for req in ordered] == ["hit", "prefix", "miss"]
    assert scheduler.score_scheduler_cache_admission_candidates["default"] == 2
    assert scheduler.score_scheduler_cache_admission_promoted["default"] >= 1


def test_scheduler_get_new_batch_prefill_respects_lane_caps(monkeypatch):
    class _Adder:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.can_run_list = []
            self.new_chunked_req = None

        def add_chunked_req(self, req):
            return req

        def add_one_req(self, req):
            self.can_run_list.append(req)
            return scheduler_module.AddReqResult.CONTINUE

    class _NewBatch:
        def __init__(self, reqs):
            self.reqs = reqs
            self.return_logprob = False
            self.bid = None
            self.decoding_reqs = None

        def prepare_for_extend(self):
            return None

    class _ScheduleBatchProxy:
        @staticmethod
        def init_new(reqs, *args, **kwargs):
            del args, kwargs
            return _NewBatch(reqs)

    monkeypatch.setattr(scheduler_module, "PrefillAdder", _Adder)
    monkeypatch.setattr(scheduler_module, "ScheduleBatch", _ScheduleBatchProxy)

    def _make_score_req(rid: str, prompt_len: int):
        req = _HashableNamespace(
            rid=rid,
            origin_input_ids=[1] * prompt_len,
            return_logprob=True,
            sampling_params=SimpleNamespace(max_new_tokens=0),
            is_multi_item_scoring=False,
            cache_for_scoring=True,
            extend_from_cache=None,
            lora_id=None,
            queue_time_start=None,
            queue_time_end=None,
            queue_wait_time_s=0.0,
        )
        req.init_next_round_input = lambda tree_cache: None
        return req

    waiting_long = _make_score_req("waiting-long", 16)
    waiting_short = _make_score_req("waiting-short", 4)
    running_long = _make_score_req("running-long", 16)

    scheduler = SimpleNamespace(
        grammar_queue=[],
        move_ready_grammar_requests=lambda: None,
        running_batch=_FakeRunningBatch(),
        waiting_queue=[waiting_long, waiting_short],
        chunked_req=None,
        max_running_requests=8,
        req_to_token_pool=_FakeReqToTokenPool(available_size=8),
        policy=SimpleNamespace(calc_priority=lambda waiting_queue: False),
        page_size=1,
        tree_cache=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(),
        new_token_ratio=1.0,
        max_prefill_tokens=4096,
        chunked_prefill_size=None,
        is_mixed_chunk=False,
        lora_paths=None,
        log_prefill_stats=lambda adder, can_run_list, running_bs: None,
        model_config=SimpleNamespace(),
        enable_overlap=False,
        mesh=None,
        spec_algorithm=None,
        score_scheduler_short_prompt_tokens_threshold=8,
        score_scheduler_short_lane_max_inflight=1,
        score_scheduler_long_lane_max_inflight=1,
        score_scheduler_lane_admission_attempted=0,
        score_scheduler_lane_admission_admitted={"default": 0, "short": 0, "long": 0},
        score_scheduler_lane_admission_skipped={"default": 0, "short": 0, "long": 0},
        score_scheduler_lane_inflight_max={"default": 0, "short": 0, "long": 0},
    )
    scheduler.running_batch.reqs = [running_long]

    batch = Scheduler.get_new_batch_prefill(scheduler)

    assert batch is not None
    assert [req.rid for req in batch.reqs] == ["waiting-short"]
    assert [req.rid for req in scheduler.waiting_queue] == ["waiting-long"]
    assert scheduler.score_scheduler_lane_admission_admitted["short"] == 1
    assert scheduler.score_scheduler_lane_admission_skipped["long"] == 1


def test_scheduler_get_new_batch_prefill_lane_isolation_avoids_long_hol(monkeypatch):
    class _Adder:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.can_run_list = []
            self.new_chunked_req = None

        def add_chunked_req(self, req):
            return req

        def add_one_req(self, req):
            self.can_run_list.append(req)
            return scheduler_module.AddReqResult.CONTINUE

    class _NewBatch:
        def __init__(self, reqs):
            self.reqs = reqs
            self.return_logprob = False
            self.bid = None
            self.decoding_reqs = None

        def prepare_for_extend(self):
            return None

    class _ScheduleBatchProxy:
        @staticmethod
        def init_new(reqs, *args, **kwargs):
            del args, kwargs
            return _NewBatch(reqs)

    monkeypatch.setattr(scheduler_module, "PrefillAdder", _Adder)
    monkeypatch.setattr(scheduler_module, "ScheduleBatch", _ScheduleBatchProxy)

    def _make_score_req(rid: str, prompt_len: int):
        req = _HashableNamespace(
            rid=rid,
            origin_input_ids=[1] * prompt_len,
            return_logprob=True,
            sampling_params=SimpleNamespace(max_new_tokens=0),
            is_multi_item_scoring=False,
            cache_for_scoring=True,
            extend_from_cache=None,
            lora_id=None,
            queue_time_start=None,
            queue_time_end=None,
            queue_wait_time_s=0.0,
        )
        req.init_next_round_input = lambda tree_cache: None
        return req

    waiting_long_a = _make_score_req("waiting-long-a", 16)
    waiting_long_b = _make_score_req("waiting-long-b", 16)
    waiting_short_a = _make_score_req("waiting-short-a", 4)
    waiting_short_b = _make_score_req("waiting-short-b", 4)

    scheduler = SimpleNamespace(
        grammar_queue=[],
        move_ready_grammar_requests=lambda: None,
        running_batch=_FakeRunningBatch(),
        waiting_queue=[waiting_long_a, waiting_long_b, waiting_short_a, waiting_short_b],
        chunked_req=None,
        max_running_requests=8,
        req_to_token_pool=_FakeReqToTokenPool(available_size=2),
        policy=SimpleNamespace(calc_priority=lambda waiting_queue: False),
        page_size=1,
        tree_cache=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(),
        new_token_ratio=1.0,
        max_prefill_tokens=4096,
        chunked_prefill_size=None,
        is_mixed_chunk=False,
        lora_paths=None,
        log_prefill_stats=lambda adder, can_run_list, running_bs: None,
        model_config=SimpleNamespace(),
        enable_overlap=False,
        mesh=None,
        spec_algorithm=None,
        score_scheduler_short_prompt_tokens_threshold=8,
        score_scheduler_short_lane_max_inflight=0,
        score_scheduler_long_lane_max_inflight=0,
        score_scheduler_enable_lane_isolation=True,
        score_scheduler_lane_isolation_short_burst=1,
        score_scheduler_lane_isolation_long_burst=1,
        score_scheduler_lane_admission_attempted=0,
        score_scheduler_lane_admission_admitted={"default": 0, "short": 0, "long": 0},
        score_scheduler_lane_admission_skipped={"default": 0, "short": 0, "long": 0},
        score_scheduler_lane_inflight_max={"default": 0, "short": 0, "long": 0},
        score_scheduler_lane_isolation_selected={"default": 0, "short": 0, "long": 0},
        score_scheduler_lane_isolation_empty_turns={"default": 0, "short": 0, "long": 0},
        score_scheduler_lane_waiting_max={"default": 0, "short": 0, "long": 0},
        score_scheduler_lane_isolation_rounds=0,
    )

    batch = Scheduler.get_new_batch_prefill(scheduler)

    assert batch is not None
    assert [req.rid for req in batch.reqs] == ["waiting-short-a", "waiting-long-a"]
    assert [req.rid for req in scheduler.waiting_queue] == ["waiting-long-b", "waiting-short-b"]
    assert scheduler.score_scheduler_lane_admission_admitted["short"] == 1
    assert scheduler.score_scheduler_lane_admission_admitted["long"] == 1
    assert scheduler.score_scheduler_lane_isolation_selected["short"] == 2
    assert scheduler.score_scheduler_lane_isolation_selected["long"] == 2


def test_wait_one_response_fails_fast_when_scheduler_dies():
    manager = _FakeSchedulerLivenessManager()
    req_obj = SimpleNamespace(stream=False, rid="rid-1")
    state = ReqState([], False, asyncio.Event(), req_obj, created_time=0.0)
    manager.rid_to_state["rid-1"] = state

    async def _await_next():
        gen = manager._wait_one_response(req_obj, state, request=None)
        return await gen.__anext__()

    with pytest.raises(ValueError, match="Scheduler subprocess is unavailable"):
        asyncio.run(_await_next())

    assert manager.health_check_failed is True
    assert manager.scheduler_unavailable_error is not None
    assert state.finished is True
    assert state.event.is_set()


def test_send_one_request_fails_fast_when_scheduler_unavailable():
    manager = _FakeSchedulerLivenessManager()
    manager.scheduler_unavailable_error = "Scheduler subprocess is unavailable. Please restart."
    req = GenerateReqInput(
        rid="rid-2",
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 0},
        is_single=True,
    )
    req.normalize_batch_and_arguments()

    with pytest.raises(ValueError, match="Scheduler subprocess is unavailable"):
        manager._send_one_request(req, tokenized_obj=SimpleNamespace(), created_time=0.0)

    assert manager.send_to_scheduler.calls == []


def test_send_one_request_broadcasts_cache_for_scoring_across_scheduler_lanes():
    manager = _FakeSchedulerLivenessManager()
    manager.scheduler_pids = []
    manager.send_to_scheduler.fan_out = 2
    req = GenerateReqInput(
        rid="rid-cache",
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 0},
        cache_for_scoring=True,
        is_single=True,
    )
    req.normalize_batch_and_arguments()
    tokenized_obj = SimpleNamespace(cache_for_scoring=True, extend_from_cache=None)

    state = manager._send_one_request(req, tokenized_obj=tokenized_obj, created_time=0.0)

    assert manager.send_to_scheduler.calls == [("all", tokenized_obj)]
    assert state.expected_finish_count == 2
    assert state.observed_finish_count == 0


def test_send_batch_requests_sends_single_payload_and_tracks_all_states():
    manager = _FakeSchedulerLivenessManager()
    manager.scheduler_pids = []
    reqs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
    tokenized_objs = [SimpleNamespace(tok=1), SimpleNamespace(tok=2)]

    states = manager._send_batch_requests(reqs, tokenized_objs, created_time=1.0)

    assert len(states) == 2
    assert set(manager.rid_to_state.keys()) == {"rid-a", "rid-b"}
    assert len(manager.send_to_scheduler.calls) == 1
    assert manager.send_to_scheduler.calls[0] == tokenized_objs


def test_send_batch_requests_routes_shared_extend_cache_to_stable_lane():
    manager = _FakeSchedulerLivenessManager()
    manager.scheduler_pids = []
    manager.send_to_scheduler.fan_out = 2
    reqs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
    tokenized_objs = [
        SimpleNamespace(tok=1, extend_from_cache="cache-1"),
        SimpleNamespace(tok=2, extend_from_cache="cache-1"),
    ]

    manager._send_batch_requests(reqs, tokenized_objs, created_time=1.0)

    assert len(manager.send_to_scheduler.calls) == 1
    scheduler_idx, payload = manager.send_to_scheduler.calls[0]
    assert scheduler_idx in {0, 1}
    assert payload == tokenized_objs


def test_send_batch_requests_raises_on_length_mismatch():
    manager = _FakeSchedulerLivenessManager()
    manager.scheduler_pids = []
    with pytest.raises(ValueError, match="same length"):
        manager._send_batch_requests([SimpleNamespace(rid="rid-a")], [], created_time=0.0)


def test_resolve_dp_scheduler_device_partitions_splits_exact_tp_sized_lanes():
    server_args = SimpleNamespace(dp_size=2, tp_size=4, device_indexes=None)

    partitions = _resolve_dp_scheduler_device_partitions(server_args, list(range(8)))

    assert partitions == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_resolve_dp_scheduler_device_partitions_rejects_mismatched_device_count():
    server_args = SimpleNamespace(dp_size=2, tp_size=4, device_indexes=None)

    with pytest.raises(ValueError, match="exact device partition"):
        _resolve_dp_scheduler_device_partitions(server_args, list(range(6)))


def test_build_scheduler_launch_plan_avoids_parent_tpu_device_probe(monkeypatch):
    server_args = ServerArgs(
        model_path="Qwen/Qwen3-0.6B",
        device="tpu",
        dp_size=1,
        tp_size=8,
        enable_single_process=False,
    )
    port_args = PortArgs.init_new(server_args)

    monkeypatch.setattr(
        engine_module.jax,
        "devices",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("parent jax.devices() should not be called for TPU multiprocess")
        ),
    )

    plan = engine_module._build_scheduler_launch_plan(server_args, port_args)

    assert len(plan) == 1
    lane_server_args, lane_port_args, dp_rank = plan[0]
    assert dp_rank == 0
    assert lane_server_args.device_indexes == list(range(8))
    assert lane_port_args == port_args


def test_launch_threads_supports_dp_scheduler_lanes(monkeypatch):
    lane_calls = []
    tokenizer_port_args_seen = []
    detokenizer_args_seen = []
    template_models_seen = []

    server_args = SimpleNamespace(
        enable_single_process=True,
        model_path="Qwen/Qwen3-0.6B",
        tokenizer_path=None,
        dp_size=2,
        tp_size=4,
        device_indexes=list(range(8)),
        node_rank=0,
        host="127.0.0.1",
        port=30001,
        check_server_args=lambda: None,
    )
    port_args = SimpleNamespace(tokenizer_ipc_name="tok", detokenizer_ipc_name="detok")

    monkeypatch.setattr(engine_module, "configure_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module, "_set_envs_and_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        engine_module,
        "prepare_model_and_tokenizer",
        lambda model_path, tokenizer_path: (model_path, tokenizer_path),
    )
    monkeypatch.setattr(
        engine_module,
        "_build_scheduler_launch_plan",
        lambda server_args, port_args: [
            (
                SimpleNamespace(
                    model_path=server_args.model_path,
                    tokenizer_path=server_args.tokenizer_path,
                    device_indexes=[0, 1, 2, 3],
                    node_rank=server_args.node_rank,
                    enable_single_process=True,
                ),
                port_args,
                0,
            ),
            (
                SimpleNamespace(
                    model_path=server_args.model_path,
                    tokenizer_path=server_args.tokenizer_path,
                    device_indexes=[4, 5, 6, 7],
                    node_rank=server_args.node_rank,
                    enable_single_process=True,
                ),
                SimpleNamespace(
                    tokenizer_ipc_name="tok",
                    detokenizer_ipc_name="detok",
                    scheduler_input_ipc_name="sched-1",
                ),
                1,
            ),
        ],
    )

    def _fake_run_scheduler_loop_thread_after_create(
        lane_server_args, lane_port_args, dp_rank=None
    ):
        lane_calls.append((dp_rank, list(lane_server_args.device_indexes), lane_port_args))
        return {
            "status": "ready",
            "max_total_num_tokens": 32768,
            "max_req_input_len": 4096,
            "scheduler": SimpleNamespace(),
            "scheduler_thread": SimpleNamespace(name=f"lane-{dp_rank}", join=lambda: None),
        }

    monkeypatch.setattr(
        engine_module,
        "run_scheduler_loop_thread_after_create",
        _fake_run_scheduler_loop_thread_after_create,
    )

    class _FakeTokenizerManager:
        def __init__(self, launch_server_args, launch_port_args):
            tokenizer_port_args_seen.append((launch_server_args, launch_port_args))
            self.scheduler_pids = []
            self.max_req_input_len = None

    monkeypatch.setattr(engine_module, "TokenizerManager", _FakeTokenizerManager)

    class _FakeTemplateManager:
        def initialize_templates(self, model_path):
            template_models_seen.append(model_path)

    monkeypatch.setattr(engine_module, "TemplateManager", _FakeTemplateManager)

    def _fake_run_detokenizer_thread(*args):
        detokenizer_args_seen.append(args)

    monkeypatch.setattr(engine_module, "run_detokenizer_thread", _fake_run_detokenizer_thread)

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.name = "fake-detokenizer"

        def start(self):
            if self.target is not None:
                self.target(*self.args)

        def join(self):
            return None

    monkeypatch.setattr(engine_module.threading, "Thread", _FakeThread)

    tokenizer_manager, template_manager, scheduler_info = engine_module._launch_threads(
        server_args,
        port_args,
    )

    assert len(lane_calls) == 2
    assert lane_calls[0][0] == 0
    assert lane_calls[0][1] == [0, 1, 2, 3]
    assert lane_calls[1][0] == 1
    assert lane_calls[1][1] == [4, 5, 6, 7]
    assert len(detokenizer_args_seen) == 1
    assert len(tokenizer_port_args_seen) == 1
    assert tokenizer_port_args_seen[0][1] == [lane_calls[0][2], lane_calls[1][2]]
    assert tokenizer_manager.max_req_input_len == 4096
    assert template_models_seen == ["Qwen/Qwen3-0.6B"]
    assert scheduler_info["max_total_num_tokens"] == 32768


def test_handle_batch_request_uses_single_send_when_batch_send_enabled():
    manager = _FakeBatchHandleManager(enable_batch_send=True, enable_batch_encode=True)
    obj = _FakeBatchRequestContainer(
        [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")],
        stream=False,
    )

    async def _collect():
        outputs = []
        async for out in manager._handle_batch_request(obj, request=None, created_time=0.0):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_collect())
    assert len(outputs) == 1
    assert len(outputs[0]) == 2
    assert len(manager.sent_batch) == 1
    assert manager.sent_single == []


def test_handle_batch_request_uses_per_request_send_when_batch_send_disabled():
    manager = _FakeBatchHandleManager(enable_batch_send=False, enable_batch_encode=True)
    obj = _FakeBatchRequestContainer(
        [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")],
        stream=False,
    )

    async def _collect():
        outputs = []
        async for out in manager._handle_batch_request(obj, request=None, created_time=0.0):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_collect())
    assert len(outputs) == 1
    assert len(outputs[0]) == 2
    assert manager.sent_batch == []
    assert len(manager.sent_single) == 2


def test_handle_batch_request_uses_single_send_without_batch_encode():
    manager = _FakeBatchHandleManager(enable_batch_send=True, enable_batch_encode=False)
    obj = _FakeBatchRequestContainer(
        [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")],
        stream=False,
    )

    async def _collect():
        outputs = []
        async for out in manager._handle_batch_request(obj, request=None, created_time=0.0):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_collect())
    assert len(outputs) == 1
    assert len(outputs[0]) == 2
    assert len(manager.sent_batch) == 1
    assert manager.sent_single == []


def test_scheduler_recv_requests_unpacks_list_payload_into_logical_batch():
    tokenizer_payload = [
        TokenizedGenerateReqInput(
            rid="tok-1",
            input_ids=[1, 2],
            sampling_params={},
            cache_for_scoring=True,
            is_multi_item_scoring=True,
        ),
        TokenizedGenerateReqInput(
            rid="tok-2",
            input_ids=[1, 3],
            sampling_params={},
            extend_from_cache="cache-handle-1",
        ),
    ]
    rpc_payload = [
        ScoreFromCacheReqInput(
            rid="rpc-1",
            cache_handle="cache-handle-1",
            items_2d=[[7, 8]],
            label_token_ids=[198],
        ),
        ReleaseScoringCacheReqInput(rid="rpc-2"),
    ]
    scheduler = _FakeSchedulerIngress(
        tokenizer_payloads=[tokenizer_payload],
        rpc_payloads=[rpc_payload],
    )

    recv_reqs = scheduler.recv_requests()

    assert len(recv_reqs) == 4
    assert scheduler.ingress_tokenizer_frames == 1
    assert scheduler.ingress_rpc_frames == 1
    assert scheduler.ingress_tokenizer_messages == 2
    assert scheduler.ingress_rpc_messages == 2
    assert scheduler.ingress_nonempty_calls == 1
    assert scheduler.ingress_max_batch_size == 4
    assert scheduler.ingress_batch_size_histogram["2_to_4"] == 1
    assert scheduler.ingress_score_paths["tokenizer_multi_item_packed"] == 1
    assert scheduler.ingress_score_paths["tokenizer_cache_for_scoring"] == 1
    assert scheduler.ingress_score_paths["tokenizer_extend_from_cache"] == 1
    assert scheduler.ingress_score_paths["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_paths["rpc_release_scoring_cache"] == 1
    assert scheduler.ingress_score_path_frames["tokenizer_multi_item_packed"] == 1
    assert scheduler.ingress_score_path_frames["tokenizer_cache_for_scoring"] == 1
    assert scheduler.ingress_score_path_frames["tokenizer_extend_from_cache"] == 1
    assert scheduler.ingress_score_path_frames["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_path_frames["rpc_release_scoring_cache"] == 1


def test_scheduler_recv_requests_counts_score_control_reqs_on_tokenizer_socket():
    tokenizer_payload = [
        ScoreFromCacheReqInput(
            rid="score-1",
            cache_handle="cache-handle-2",
            items_2d=[[7, 8]],
            label_token_ids=[198],
        ),
        ReleaseScoringCacheReqInput(rid="release-1"),
    ]
    scheduler = _FakeSchedulerIngress(
        tokenizer_payloads=[tokenizer_payload],
        rpc_payloads=[],
    )

    recv_reqs = scheduler.recv_requests()

    assert len(recv_reqs) == 2
    assert scheduler.ingress_tokenizer_frames == 1
    assert scheduler.ingress_rpc_frames == 0
    assert scheduler.ingress_tokenizer_messages == 2
    assert scheduler.ingress_rpc_messages == 0
    assert scheduler.ingress_score_paths["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_paths["rpc_release_scoring_cache"] == 1
    assert scheduler.ingress_score_path_frames["rpc_score_from_cache_v2"] == 1
    assert scheduler.ingress_score_path_frames["rpc_release_scoring_cache"] == 1


def test_mark_scheduler_unavailable_aborts_all_pending_requests():
    manager = _FakeSchedulerLivenessManager()
    state1 = ReqState([], False, asyncio.Event(), SimpleNamespace(stream=False), created_time=0.0)
    state2 = ReqState([], False, asyncio.Event(), SimpleNamespace(stream=False), created_time=0.0)
    manager.rid_to_state["rid-a"] = state1
    manager.rid_to_state["rid-b"] = state2

    manager._mark_scheduler_unavailable(
        "Scheduler subprocess is unavailable (dead pid(s): 4321). Please restart the server."
    )

    assert manager.health_check_failed is True
    assert manager.rid_to_state == {}
    for state in (state1, state2):
        assert state.finished is True
        assert state.event.is_set()
        finish_reason = state.out_list[-1]["meta_info"]["finish_reason"]
        assert finish_reason["type"] == "abort"
        assert "Scheduler subprocess is unavailable" in finish_reason["message"]


def test_token_ids_logprobs_handles_ragged_prefill_lengths():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    all_logprobs = jax.numpy.array(
        [
            [0.10, 0.20, 0.70],
            [0.15, 0.25, 0.60],
            [0.20, 0.30, 0.50],
        ],
        dtype=jax.numpy.float32,
    )
    logits_metadata = SimpleNamespace(
        token_ids_logprobs=[[0, 2], [0, 2], [0, 2]],
        extend_logprob_pruned_lens_cpu=[0, 2, 1],
    )

    vals, idxs = LogitsProcessor.get_token_ids_logprobs(all_logprobs, logits_metadata, mesh)

    assert vals.shape == (3, 2, 2)
    assert idxs.shape == (3, 2, 2)
    np.testing.assert_array_equal(np.array(idxs[0]), np.array([[-1, -1], [-1, -1]]))
    np.testing.assert_allclose(np.array(vals[1]), np.array([[0.10, 0.70], [0.15, 0.60]]))
    np.testing.assert_allclose(np.array(vals[2][0]), np.array([0.20, 0.50]))
    np.testing.assert_array_equal(np.array(idxs[2][1]), np.array([-1, -1]))


def test_input_logprob_slicing_handles_nested_lists():
    req = SimpleNamespace(
        is_multi_item_scoring=False,
        multi_item_scoring_delimiter=None,
        input_token_logprobs=[],
        temp_input_top_logprobs_val=[],
        temp_input_top_logprobs_idx=[],
        temp_input_token_ids_logprobs_val=[],
        temp_input_token_ids_logprobs_idx=[],
        input_token_logprobs_val=None,
        top_logprobs_num=2,
        token_ids_logprob=[101, 202],
    )
    output = SimpleNamespace(
        input_token_logprobs=[0.1, 0.2, 0.3, 0.4],
        input_top_logprobs_val=[
            [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0], [0.6, 0.4, 0.0]]
        ],
        input_top_logprobs_idx=[[[11, 12, -1], [21, 22, -1], [31, 32, -1], [41, 42, -1]]],
        input_token_ids_logprobs_val=[
            [[0.6, 0.4, 0.0], [0.7, 0.3, 0.0], [0.8, 0.2, 0.0], [0.9, 0.1, 0.0]]
        ],
        input_token_ids_logprobs_idx=[
            [[101, 202, -1], [101, 202, -1], [101, 202, -1], [101, 202, -1]]
        ],
    )

    SchedulerOutputProcessorMixin.add_input_logprob_return_values(
        self=SimpleNamespace(),
        i=0,
        req=req,
        output=output,
        logprob_pt=0,
        num_input_logprobs=2,
        last_prefill_chunk=False,
    )

    assert req.input_token_logprobs == [0.1, 0.2]
    assert req.temp_input_top_logprobs_val == [[[0.9, 0.1], [0.8, 0.2]]]
    assert req.temp_input_top_logprobs_idx == [[[11, 12], [21, 22]]]
    assert req.temp_input_token_ids_logprobs_val == [[[0.6, 0.4], [0.7, 0.3]]]
    assert req.temp_input_token_ids_logprobs_idx == [[[101, 202], [101, 202]]]


def test_sampler_token_ids_logprobs_handles_none_entries():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    logprobs = jax.numpy.array(
        [
            [0.20, 0.30, 0.50],
            [0.60, 0.10, 0.30],
        ],
        dtype=jax.numpy.float32,
    )

    vals, idxs = sampler_get_token_ids_logprobs(logprobs, [None, [0, 2]], mesh)

    assert vals.shape == (2, 2)
    assert idxs.shape == (2, 2)
    np.testing.assert_array_equal(np.array(idxs[0]), np.array([-1, -1]))
    np.testing.assert_allclose(np.array(vals[1]), np.array([0.60, 0.30]))
    np.testing.assert_array_equal(np.array(idxs[1]), np.array([0, 2]))


def test_sampler_regular_sampling_reshards_logits_for_sampler_layout(monkeypatch):
    from jax import numpy as jnp

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    sampler = Sampler(mesh=mesh)
    observed = {}

    def _fake_reshard(value, sharding):
        observed["sharding"] = sharding
        return value

    monkeypatch.setattr(jax.sharding, "reshard", _fake_reshard)
    monkeypatch.setattr(
        sampler_module,
        "top_k_top_p_min_p_sampling_from_probs_jax",
        lambda args, use_sort_for_toppk_minp: jnp.array([[1]], dtype=jnp.int32),
    )

    sampling_metadata = SimpleNamespace(
        temperatures=jnp.array([[1.0]], dtype=jnp.float32),
        top_ks=jnp.array([1], dtype=jnp.int32),
        top_ps=jnp.array([1.0], dtype=jnp.float32),
        min_ps=jnp.array([0.0], dtype=jnp.float32),
        positions=jnp.array([0], dtype=jnp.int32),
        sampling_seeds=jnp.array([0], dtype=jnp.int32),
        need_min_p_sampling=jnp.array([False]),
    )

    batch_next_token_ids, logprobs = sampler._regular_sampling(
        (
            jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float32),
            sampling_metadata,
            jax.random.PRNGKey(0),
            False,
        )
    )

    assert observed["sharding"] == jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(None, None),
    )
    assert np.array(batch_next_token_ids).shape == (1, 1)
    assert np.array(logprobs).shape == (1, 3)


class _FakeScorePrefillExtendManager:
    score_prefill_extend = TokenizerManager.score_prefill_extend
    _record_score_fastpath_fallback = TokenizerManager._record_score_fastpath_fallback
    _resolve_score_from_cache_v2_items_per_step = (
        TokenizerManager._resolve_score_from_cache_v2_items_per_step
    )

    def __init__(
        self, fastpath_enabled: bool, fastpath_output: ScoreFromCacheReqOutput | Exception
    ):
        self.server_args = SimpleNamespace(
            multi_item_extend_batch_size=64,
            multi_item_enable_score_from_cache_v2=fastpath_enabled,
            multi_item_score_fastpath_log_metrics=True,
            multi_item_score_from_cache_v2_items_per_step=64,
            multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget=False,
            multi_item_score_from_cache_v2_token_budget=0,
            multi_item_score_from_cache_v2_min_items_per_step=1,
        )
        self.fastpath_output = fastpath_output
        self.prefill_calls = 0
        self.fastpath_calls = 0
        self.fastpath_items_per_step_seen = 0
        self.fastpath_token_budget_seen = 0
        self.fastpath_max_total_tokens_seen = 0
        self.baseline_calls: list[list[list[int]]] = []
        self.logged_metrics = []
        self.score_fastpath_attempted = 0
        self.score_fastpath_succeeded = 0
        self.score_fastpath_fallback = 0
        self.score_fastpath_fallback_reasons = {}

    async def _prefill_and_cache(self, query_tokens: list[int]) -> str:
        del query_tokens
        self.prefill_calls += 1
        return "cache-handle"

    async def _score_from_cache_fastpath_v2(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
        items_per_step: int | None = None,
        token_budget: int = 0,
        max_total_tokens: int = 0,
    ) -> ScoreFromCacheReqOutput:
        del cache_handle, items, label_token_ids, apply_softmax
        self.fastpath_calls += 1
        self.fastpath_items_per_step_seen = int(items_per_step or 0)
        self.fastpath_token_budget_seen = int(token_budget or 0)
        self.fastpath_max_total_tokens_seen = int(max_total_tokens or 0)
        if isinstance(self.fastpath_output, Exception):
            raise self.fastpath_output
        return self.fastpath_output

    async def _batched_extend_score_with_metrics(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
    ) -> tuple[list[list[float]], dict[str, float | int]]:
        del cache_handle, label_token_ids, apply_softmax
        self.baseline_calls.append(items)
        scores = [[float(item[0]), float(item[0]) + 0.1] for item in items]
        metrics = {
            "dispatch_count": 1,
            "queue_wait_s": 0.001,
            "device_compute_s": 0.002,
            "host_orchestration_s": 0.003,
            "lifecycle_requests_sent": len(items),
            "lifecycle_results_received": len(items),
        }
        return scores, metrics

    async def _release_cache(self, cache_handle: str) -> bool:
        del cache_handle
        return True

    def _maybe_log_score_path_metrics(self, metrics: dict):
        self.logged_metrics.append(metrics)


class _FakeSchedulerScoreFromCacheV2:
    score_from_cache_v2 = Scheduler.score_from_cache_v2
    _score_from_cache_v2_use_direct_label_only = (
        Scheduler._score_from_cache_v2_use_direct_label_only
    )
    _score_from_cache_v2_use_direct_token_ids_logprob_only = (
        Scheduler._score_from_cache_v2_use_direct_token_ids_logprob_only
    )
    _score_from_cache_v2_resolve_direct_hot_shape = (
        Scheduler._score_from_cache_v2_resolve_direct_hot_shape
    )
    _resolve_score_from_cache_v2_items_per_step = (
        Scheduler._resolve_score_from_cache_v2_items_per_step
    )
    _score_from_cache_v2_replica_lane_count = Scheduler._score_from_cache_v2_replica_lane_count
    _score_from_cache_v2_topology_dispatch_policy = (
        Scheduler._score_from_cache_v2_topology_dispatch_policy
    )
    _normalize_scoring_cache_prefix_key = Scheduler._normalize_scoring_cache_prefix_key
    _register_scoring_cache_handle = Scheduler._register_scoring_cache_handle
    _unregister_scoring_cache_handle = Scheduler._unregister_scoring_cache_handle
    _score_from_cache_v2_validate_items = Scheduler._score_from_cache_v2_validate_items
    _score_from_cache_v2_fallback_output = Scheduler._score_from_cache_v2_fallback_output
    _record_score_from_cache_v2_fallback = Scheduler._record_score_from_cache_v2_fallback
    _record_score_from_cache_v2_timing = Scheduler._record_score_from_cache_v2_timing
    _record_scoring_cache_lookup = Scheduler._record_scoring_cache_lookup
    _scoring_cache_metrics_snapshot = Scheduler._scoring_cache_metrics_snapshot
    _estimate_score_from_cache_v2_words = Scheduler._estimate_score_from_cache_v2_words
    _label_only_parity_metrics = Scheduler._label_only_parity_metrics
    _build_score_from_cache_v2_chunk_plan = staticmethod(
        Scheduler._build_score_from_cache_v2_chunk_plan
    )
    _touch_scoring_cache_entry = Scheduler._touch_scoring_cache_entry
    _unpack_scoring_cache_entry = Scheduler._unpack_scoring_cache_entry

    def __init__(self):
        self.enable_overlap = False
        self.server_args = SimpleNamespace(
            multi_item_score_from_cache_v2_items_per_step=64,
            multi_item_score_from_cache_v2_token_budget=0,
            multi_item_score_label_only_logprob=False,
            multi_item_score_label_only_fused_kernel=True,
            multi_item_score_direct_label_only=False,
            multi_item_score_direct_hot_shape_bs=0,
            multi_item_score_direct_hot_shape_tokens=0,
            multi_item_score_direct_hot_shape_token_rounding=0,
            multi_item_score_direct_token_ids_logprob_only_auto=False,
            multi_item_score_direct_token_ids_logprob_only_auto_max_page_size=32,
            multi_item_score_direct_token_ids_logprob_only_auto_max_running_requests=32,
            allow_auto_truncate=False,
            max_running_requests=1024,
            device="tpu",
        )
        self.req_to_token_pool = SimpleNamespace(available_size=lambda: 1024)
        self.model_config = SimpleNamespace(hf_eos_token_id={2}, vocab_size=32000)
        self.max_req_len = 32768
        self.max_req_input_len = 32768
        self.scoring_cache_nodes = {
            "cache-ok": (
                "node",
                None,
                [101] * 2000,
                np.arange(2000, dtype=np.int32),
                None,
                0.0,
            )
        }
        self.scoring_cache_timeout = 0.0
        self._last_scoring_cache_gc = 0.0
        self.score_from_cache_v2_attempted = 0
        self.score_from_cache_v2_succeeded = 0
        self.score_from_cache_v2_fallback = 0
        self.score_from_cache_v2_fallback_reasons = {}
        self.score_from_cache_v2_queue_wait_s_total = 0.0
        self.score_from_cache_v2_device_compute_s_total = 0.0
        self.score_from_cache_v2_host_orchestration_s_total = 0.0
        self.score_from_cache_v2_queue_wait_s_max = 0.0
        self.score_from_cache_v2_device_compute_s_max = 0.0
        self.score_from_cache_v2_host_orchestration_s_max = 0.0
        self.scoring_cache_lookup_queries = 0
        self.scoring_cache_lookup_hits = 0
        self.scoring_cache_lookup_misses = 0
        self.scoring_cache_lookup_by_path = {
            "extend": {"queries": 0, "hits": 0, "misses": 0},
            "score_from_cache_v2": {"queries": 0, "hits": 0, "misses": 0},
            "cache_for_scoring": {"queries": 0, "hits": 0, "misses": 0},
        }
        self.scoring_cache_lookup_by_lane = {
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
        self.scoring_cache_prefix_handles_by_key = {}
        self.scoring_cache_handle_to_prefix_key = {}
        self.score_scheduler_short_prompt_tokens_threshold = 400
        self.score_scheduler_topology_name = ""
        self.scoring_cache_handles_created = 0
        self.scoring_cache_handles_released = 0
        self.scoring_cache_handles_released_manual = 0
        self.scoring_cache_handles_released_expired = 0
        self.scoring_cache_handles_released_other = 0
        self.scoring_cache_handles_missing_node = 0
        self.chunk_calls = []
        self.label_only_chunk_calls = []
        self.direct_label_only_chunk_calls = []
        self.label_only_chunk_fused_flags = []
        self.fail_next_chunk = False
        self.force_estimated_words = None
        self.mesh = SimpleNamespace(shape={"data": 1, "tensor": 1})
        self.page_size = 64

    def _evict_expired_scoring_cache_nodes(self):
        return 0

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
        del (
            cache_handle,
            label_token_ids,
            apply_softmax,
            cached_last_node,
            cached_prefix_indices,
            prefix_ids,
            cached_extra_key,
        )
        self.chunk_calls.append([item[0] for item in chunk_items])
        if self.fail_next_chunk:
            self.fail_next_chunk = False
            raise RuntimeError("synthetic chunk failure")
        return (
            [[float(item[0]), float(item[0]) + 1.0] for item in chunk_items],
            0.01,
            0.02,
        )

    def _estimate_score_from_cache_v2_words(self, prefix_len: int, items: list[list[int]]) -> int:
        if self.force_estimated_words is not None:
            return self.force_estimated_words
        return Scheduler._estimate_score_from_cache_v2_words(prefix_len, items)

    def _run_score_from_cache_v2_chunk_label_only(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        label_token_ids_arr,
        apply_softmax: bool,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
    ) -> tuple[list[list[float]], float, float]:
        del (
            cache_handle,
            label_token_ids,
            label_token_ids_arr,
            apply_softmax,
            cached_last_node,
            cached_prefix_indices,
            prefix_ids,
            cached_extra_key,
        )
        self.label_only_chunk_calls.append([item[0] for item in chunk_items])
        self.label_only_chunk_fused_flags.append(
            bool(getattr(self.server_args, "multi_item_score_label_only_fused_kernel", True))
        )
        return (
            [[float(item[0]), float(item[0]) + 0.5] for item in chunk_items],
            0.02,
            0.01,
        )

    def _run_score_from_cache_v2_direct_chunk_label_only(
        self,
        cache_handle: str,
        chunk_items: list[list[int]],
        label_token_ids: list[int],
        label_token_ids_arr,
        apply_softmax: bool,
        cached_last_node,
        cached_prefix_indices,
        prefix_ids: list[int],
        cached_extra_key: str | None,
    ) -> tuple[list[list[float]], float, float]:
        del (
            cache_handle,
            label_token_ids,
            label_token_ids_arr,
            apply_softmax,
            cached_last_node,
            cached_prefix_indices,
            prefix_ids,
            cached_extra_key,
        )
        self.direct_label_only_chunk_calls.append([item[0] for item in chunk_items])
        return (
            jax.numpy.asarray(
                [[float(item[0]), float(item[0]) + 0.25] for item in chunk_items],
                dtype=jax.numpy.float32,
            ),
            0.03,
            0.01,
        )


class _FakeSchedulerDirectLabelOnlyRunner:
    _run_score_from_cache_v2_direct_chunk_label_only = (
        Scheduler._run_score_from_cache_v2_direct_chunk_label_only
    )
    _score_from_cache_v2_use_direct_token_ids_logprob_only = (
        Scheduler._score_from_cache_v2_use_direct_token_ids_logprob_only
    )
    _score_from_cache_v2_resolve_direct_token_ids_logprob_only_chunk_size = (
        Scheduler._score_from_cache_v2_resolve_direct_token_ids_logprob_only_chunk_size
    )
    _score_from_cache_v2_resolve_direct_hot_shape = (
        Scheduler._score_from_cache_v2_resolve_direct_hot_shape
    )

    def __init__(self, logits_output):
        self.tree_cache = object()
        self.page_size = 64
        self.mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
        self.model_config = SimpleNamespace(vocab_size=32000)
        self.server_args = SimpleNamespace(
            multi_item_score_direct_hot_shape_bs=0,
            multi_item_score_direct_hot_shape_tokens=0,
            multi_item_score_direct_hot_shape_token_rounding=0,
            multi_item_score_direct_token_ids_logprob_only=False,
            multi_item_score_direct_token_ids_logprob_only_auto=False,
            multi_item_score_direct_token_ids_logprob_only_auto_max_page_size=32,
            multi_item_score_direct_token_ids_logprob_only_auto_max_running_requests=32,
            multi_item_score_direct_token_ids_logprob_only_chunk_size=4096,
            score_scheduler_short_prompt_tokens_threshold=2048,
            max_running_requests=64,
            multi_item_score_label_only_fused_kernel=True,
        )
        self._logits_output = logits_output
        self.tp_worker = SimpleNamespace(forward_batch_generation=self._forward_batch_generation)
        self.score_label_only_token_ids_only_calls = 0
        self.score_label_only_fused_kernel_calls = 0
        self.score_label_only_fused_kernel_softmax_calls = 0
        self.score_label_only_legacy_kernel_calls = 0
        self.seen_model_worker_batch = None
        self.freed_token_slots = []
        self.token_to_kv_pool_allocator = SimpleNamespace(free=self._free_token_slots)

    def _forward_batch_generation(
        self,
        model_worker_batch,
        launch_done=None,
        skip_sample=False,
        sampling_metadata=None,
    ):
        del launch_done, skip_sample, sampling_metadata
        self.seen_model_worker_batch = model_worker_batch
        return self._logits_output, None, None

    def _free_token_slots(self, token_slots):
        self.freed_token_slots.append(np.asarray(token_slots, dtype=np.int32).tolist())


def _parity_metrics(
    baseline_scores: list[list[float]],
    fastpath_scores: list[list[float]],
) -> tuple[float, float]:
    diffs = []
    for base_row, fast_row in zip(baseline_scores, fastpath_scores):
        diffs.extend(abs(a - b) for a, b in zip(base_row, fast_row))
    return max(diffs), sum(diffs) / len(diffs)


class _FakeScoreFromCacheV2Sizer:
    _resolve_score_from_cache_v2_items_per_step = (
        TokenizerManager._resolve_score_from_cache_v2_items_per_step
    )
    _scheduler_sender_fan_out = TokenizerManager._scheduler_sender_fan_out
    _partition_score_from_cache_v2_items = TokenizerManager._partition_score_from_cache_v2_items
    _score_from_cache_fastpath_v2 = TokenizerManager._score_from_cache_fastpath_v2

    def __init__(self, scheduler_fan_out: int = 1):
        self.server_args = SimpleNamespace(
            multi_item_prefill_extend_cache_timeout=60.0,
            multi_item_score_from_cache_v2_items_per_step=64,
            multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget=False,
            multi_item_score_from_cache_v2_token_budget=0,
            multi_item_score_from_cache_v2_min_items_per_step=1,
        )
        self.last_request = None
        self.last_timeout = None
        self.last_items_per_step = None
        self.last_scheduler_idx = None
        self.requests_by_scheduler = []
        self.send_to_scheduler = SimpleNamespace(fan_out=scheduler_fan_out)
        self.score_from_cache_v2_communicator = self._communicator

    def auto_create_handle_loop(self):
        return None

    async def _communicator(self, req, timeout=None, scheduler_idx=None, broadcast=False):
        del broadcast
        self.last_request = req
        self.last_timeout = timeout
        self.last_items_per_step = req.items_per_step
        self.last_scheduler_idx = scheduler_idx
        self.requests_by_scheduler.append((scheduler_idx, req))
        if scheduler_idx is None:
            scores = [[0.1, 0.9] for _ in req.items_2d]
        else:
            scores = [[float(scheduler_idx), float(item[0])] for item in req.items_2d]
        return [
            ScoreFromCacheReqOutput(
                success=True,
                scores=scores,
                effective_items_per_step=req.items_per_step,
                dispatch_token_budget=req.token_budget,
                replica_lane_count=1,
                topology_name=f"lane-{scheduler_idx if scheduler_idx is not None else 0}",
            )
        ]


def test_score_prefill_extend_fastpath_v2_500x20_order_and_count():
    expected_scores = [[float(i), float(i) + 0.5] for i in range(500)]
    manager = _FakeScorePrefillExtendManager(
        fastpath_enabled=True,
        fastpath_output=ScoreFromCacheReqOutput(
            success=True,
            scores=expected_scores,
            dispatch_count=8,
            queue_wait_s=0.01,
            device_compute_s=0.2,
            host_orchestration_s=0.05,
        ),
    )
    query_tokens = [11] * 2000
    items = [[i] * 20 for i in range(500)]

    scores = asyncio.run(
        manager.score_prefill_extend(
            query_tokens=query_tokens,
            item_tokens_list=items,
            label_token_ids=[9454, 2753],
            apply_softmax=False,
        )
    )

    assert len(scores) == 500
    assert scores == expected_scores
    assert manager.prefill_calls == 1
    assert manager.fastpath_calls == 1
    assert manager.baseline_calls == []
    assert manager.score_fastpath_attempted == 1
    assert manager.score_fastpath_succeeded == 1
    assert manager.score_fastpath_fallback == 0


def test_score_prefill_extend_fastpath_exception_falls_back_and_recovers():
    manager = _FakeScorePrefillExtendManager(
        fastpath_enabled=True,
        fastpath_output=RuntimeError("synthetic fastpath communicator failure"),
    )
    query_tokens = [7] * 2000
    items = [[i] * 20 for i in range(500)]

    scores = asyncio.run(
        manager.score_prefill_extend(
            query_tokens=query_tokens,
            item_tokens_list=items,
            label_token_ids=[9454, 2753],
            apply_softmax=False,
        )
    )

    assert len(scores) == 500
    assert manager.fastpath_calls == 1
    assert len(manager.baseline_calls) > 0
    assert manager.score_fastpath_attempted == 1
    assert manager.score_fastpath_succeeded == 0
    assert manager.score_fastpath_fallback == 1
    assert manager.score_fastpath_fallback_reasons.get("runtime_exception") == 1

    # Recovery sanity: a second request still succeeds.
    scores_2 = asyncio.run(
        manager.score_prefill_extend(
            query_tokens=query_tokens,
            item_tokens_list=items[:10],
            label_token_ids=[9454, 2753],
            apply_softmax=False,
        )
    )
    assert len(scores_2) == 10


def test_score_prefill_extend_fastpath_uses_token_budget_adaptive_items_per_step():
    expected_scores = [[float(i), float(i) + 0.5] for i in range(50)]
    manager = _FakeScorePrefillExtendManager(
        fastpath_enabled=True,
        fastpath_output=ScoreFromCacheReqOutput(
            success=True,
            scores=expected_scores,
            dispatch_count=7,
            queue_wait_s=0.01,
            device_compute_s=0.2,
            host_orchestration_s=0.05,
        ),
    )
    manager.server_args.multi_item_score_from_cache_v2_items_per_step = 64
    manager.server_args.multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget = True
    manager.server_args.multi_item_score_from_cache_v2_token_budget = 2000
    manager.server_args.multi_item_score_from_cache_v2_min_items_per_step = 8

    query_tokens = [11] * 420
    items = [[i] * 55 for i in range(50)]
    scores = asyncio.run(
        manager.score_prefill_extend(
            query_tokens=query_tokens,
            item_tokens_list=items,
            label_token_ids=[9454, 2753],
            apply_softmax=False,
        )
    )

    assert scores == expected_scores
    # budget=2000, max_total_tokens=475 => floor=4 -> clamped by min_items_per_step=8.
    assert manager.fastpath_items_per_step_seen == 8


def test_score_from_cache_v2_resolver_keeps_default_when_adaptive_disabled():
    manager = _FakeScoreFromCacheV2Sizer()
    manager.server_args.multi_item_score_from_cache_v2_items_per_step = 64

    items_per_step, max_total_tokens, token_budget = (
        manager._resolve_score_from_cache_v2_items_per_step(
            query_tokens=[1] * 420,
            items=[[2] * 55 for _ in range(8)],
        )
    )

    assert items_per_step == 64
    assert max_total_tokens == 0
    assert token_budget == 0


def test_score_from_cache_v2_resolver_applies_token_budget_and_floor():
    manager = _FakeScoreFromCacheV2Sizer()
    manager.server_args.multi_item_score_from_cache_v2_items_per_step = 64
    manager.server_args.multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget = True
    manager.server_args.multi_item_score_from_cache_v2_token_budget = 20000
    manager.server_args.multi_item_score_from_cache_v2_min_items_per_step = 8

    items_per_step, max_total_tokens, token_budget = (
        manager._resolve_score_from_cache_v2_items_per_step(
            query_tokens=[1] * 420,
            items=[[2] * 55 for _ in range(8)],
        )
    )

    assert max_total_tokens == 475
    assert token_budget == 20000
    assert items_per_step == 42


def test_score_from_cache_v2_fastpath_request_uses_resolved_items_per_step():
    manager = _FakeScoreFromCacheV2Sizer()
    manager.server_args.multi_item_score_from_cache_v2_items_per_step = 64

    out = asyncio.run(
        manager._score_from_cache_fastpath_v2(
            cache_handle="cache-handle",
            items=[[1] * 55 for _ in range(8)],
            label_token_ids=[9454, 2753],
            apply_softmax=False,
            items_per_step=17,
        )
    )

    assert out.success is True
    assert manager.last_items_per_step == 17
    assert manager.last_request is not None
    assert manager.last_request.items_per_step == 17
    assert manager.last_request.token_budget == 0
    assert manager.last_request.max_total_tokens == 0


def test_score_from_cache_v2_fastpath_request_includes_budget_metadata():
    manager = _FakeScoreFromCacheV2Sizer()

    out = asyncio.run(
        manager._score_from_cache_fastpath_v2(
            cache_handle="cache-handle",
            items=[[1] * 55 for _ in range(8)],
            label_token_ids=[9454, 2753],
            apply_softmax=False,
            items_per_step=17,
            token_budget=8192,
            max_total_tokens=475,
        )
    )

    assert out.success is True
    assert manager.last_request is not None
    assert manager.last_request.token_budget == 8192
    assert manager.last_request.max_total_tokens == 475


def test_score_from_cache_v2_fastpath_partitions_work_across_scheduler_lanes():
    manager = _FakeScoreFromCacheV2Sizer(scheduler_fan_out=2)

    out = asyncio.run(
        manager._score_from_cache_fastpath_v2(
            cache_handle="cache-handle",
            items=[[10], [20], [30], [40]],
            label_token_ids=[9454, 2753],
            apply_softmax=False,
            items_per_step=17,
            token_budget=8000,
            max_total_tokens=475,
        )
    )

    assert out.success is True
    assert out.replica_lane_count == 2
    assert out.effective_items_per_step == 34
    assert out.dispatch_token_budget == 8000
    assert out.topology_name == "lane-0,lane-1 replicated x2"
    assert out.scores == [[0.0, 10.0], [1.0, 20.0], [0.0, 30.0], [1.0, 40.0]]
    assert [scheduler_idx for scheduler_idx, _ in manager.requests_by_scheduler] == [0, 1]


def test_score_from_cache_v2_chunk_loop_dispatches_multiple_steps():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    items = [[i] * 20 for i in range(150)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 3
    assert len(out.scores) == 150
    assert out.scores[0] == [0.0, 1.0]
    assert out.scores[-1] == [149.0, 150.0]
    assert len(scheduler.chunk_calls) == 3
    assert scheduler.score_from_cache_v2_succeeded == 1


def test_score_from_cache_v2_length_aware_packing_preserves_output_order():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    items = [
        [100] * 2,
        [200] * 5,
        [300] * 3,
        [400] * 5,
        [500] * 1,
    ]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=2,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 3
    assert scheduler.chunk_calls == [[200, 400], [300, 100], [500]]
    assert out.scores == [
        [100.0, 101.0],
        [200.0, 201.0],
        [300.0, 301.0],
        [400.0, 401.0],
        [500.0, 501.0],
    ]


def test_score_from_cache_v2_caps_items_per_step_by_req_slots():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.max_running_requests = 24
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 25)
    items = [[i] * 20 for i in range(50)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 3
    assert [len(chunk) for chunk in scheduler.chunk_calls] == [24, 24, 2]


def test_score_from_cache_v2_reqpool_oversubscribe_flag_uses_available_slots(monkeypatch):
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.max_running_requests = 24
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 25)
    monkeypatch.setattr(
        scheduler_module,
        "SCORE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE",
        True,
    )
    items = [[i] * 20 for i in range(50)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 2
    assert [len(chunk) for chunk in scheduler.chunk_calls] == [25, 25]


def test_score_from_cache_v2_dynamic_items_per_step_reduces_long_lane_chunk_size():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.score_scheduler_dynamic_items_per_step_enable = True
    scheduler.score_scheduler_dynamic_items_per_step_pressure_threshold = 64
    scheduler.score_scheduler_dynamic_items_per_step_short_lane_bias = 1.0
    scheduler.score_scheduler_dynamic_items_per_step_long_lane_bias = 0.5
    scheduler.score_scheduler_dynamic_items_per_step_short_lane_min = 32
    scheduler.score_scheduler_dynamic_items_per_step_long_lane_min = 16
    scheduler.score_scheduler_short_prompt_tokens_threshold = 400
    scheduler.waiting_queue = [_HashableNamespace(rid=f"w{i}") for i in range(128)]
    scheduler.running_batch = SimpleNamespace(
        reqs=[_HashableNamespace(rid=f"r{i}") for i in range(32)]
    )

    items = [[i] * 20 for i in range(40)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 3
    assert [len(chunk) for chunk in scheduler.chunk_calls] == [16, 16, 8]
    assert scheduler.score_scheduler_dynamic_items_per_step_requests == 1
    assert scheduler.score_scheduler_dynamic_items_per_step_requested_by_lane["long"] == 40
    assert scheduler.score_scheduler_dynamic_items_per_step_effective_by_lane["long"] == 16
    assert scheduler.score_scheduler_dynamic_items_per_step_applied_by_lane["long"] == 1


def test_score_from_cache_v2_dynamic_items_per_step_preserves_short_lane_floor():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.score_scheduler_dynamic_items_per_step_enable = True
    scheduler.score_scheduler_dynamic_items_per_step_pressure_threshold = 64
    scheduler.score_scheduler_dynamic_items_per_step_short_lane_bias = 1.0
    scheduler.score_scheduler_dynamic_items_per_step_long_lane_bias = 0.5
    scheduler.score_scheduler_dynamic_items_per_step_short_lane_min = 24
    scheduler.score_scheduler_dynamic_items_per_step_long_lane_min = 16
    scheduler.score_scheduler_short_prompt_tokens_threshold = 4000
    scheduler.waiting_queue = [_HashableNamespace(rid=f"w{i}") for i in range(96)]
    scheduler.running_batch = SimpleNamespace(
        reqs=[_HashableNamespace(rid=f"r{i}") for i in range(32)]
    )

    items = [[i] * 20 for i in range(60)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 2
    assert [len(chunk) for chunk in scheduler.chunk_calls] == [30, 30]
    assert scheduler.score_scheduler_dynamic_items_per_step_requested_by_lane["short"] == 60
    assert scheduler.score_scheduler_dynamic_items_per_step_effective_by_lane["short"] == 30
    assert scheduler.score_scheduler_dynamic_items_per_step_applied_by_lane["short"] == 1


def test_score_from_cache_v2_v6e8_replica_lanes_boost_long_dispatch_width():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.score_scheduler_topology_name = "TPU v6e-8"
    scheduler.mesh = SimpleNamespace(shape={"data": 4, "tensor": 2})
    items = [[i] * 20 for i in range(64)]

    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=16,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 1
    assert out.effective_items_per_step == 64
    assert out.replica_lane_count == 4
    assert out.topology_name == "TPU v6e-8"
    assert [len(chunk) for chunk in scheduler.chunk_calls] == [64]


def test_score_from_cache_v2_direct_label_only_bypasses_req_slot_caps():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.multi_item_score_label_only_logprob = True
    scheduler.server_args.multi_item_score_direct_label_only = True
    scheduler.server_args.max_running_requests = 24
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 24)
    items = [[i] * 20 for i in range(500)]

    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=64,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 1
    assert out.effective_items_per_step == 500
    assert scheduler.chunk_calls == []
    assert scheduler.label_only_chunk_calls == []
    assert [len(chunk) for chunk in scheduler.direct_label_only_chunk_calls] == [500]
    assert out.scores[0] == [0.0, 0.25]
    assert out.scores[-1] == [499.0, 499.25]


def test_direct_label_only_chunk_defaults_to_fused_scores(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "alloc_paged_token_slots_extend",
        lambda tree_cache, prefix_lens, seq_lens, last_loc, extend_num_tokens: np.arange(
            1, extend_num_tokens + 1, dtype=np.int32
        ),
    )
    logits_output = SimpleNamespace(
        next_token_logits=jax.numpy.log(
            jax.numpy.array([[0.2, 0.8], [0.6, 0.4]], dtype=jax.numpy.float32)
        ),
        next_token_token_ids_logprobs_val=None,
        next_token_token_ids_logprobs_idx=None,
    )
    scheduler = _FakeSchedulerDirectLabelOnlyRunner(logits_output=logits_output)

    scores, device_s, host_s = scheduler._run_score_from_cache_v2_direct_chunk_label_only(
        cache_handle="cache-ok",
        chunk_items=[[101, 102], [201]],
        label_token_ids=[0, 1],
        label_token_ids_arr=jax.numpy.array([0, 1], dtype=jax.numpy.int32),
        apply_softmax=False,
        cached_last_node=None,
        cached_prefix_indices=np.array([7, 8, 9], dtype=np.int32),
        prefix_ids=[7, 8, 9],
        cached_extra_key=None,
    )

    np.testing.assert_allclose(
        np.asarray(scores),
        np.asarray([[0.2, 0.8], [0.6, 0.4]]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert scheduler.score_label_only_token_ids_only_calls == 0
    assert scheduler.score_label_only_fused_kernel_calls == 1
    assert scheduler.score_label_only_legacy_kernel_calls == 0
    assert scheduler.seen_model_worker_batch is not None
    assert scheduler.seen_model_worker_batch.next_token_token_ids_logprob_only is False
    assert scheduler.seen_model_worker_batch.next_token_shared_token_ids is None
    assert scheduler.seen_model_worker_batch.token_ids_logprobs == [[0, 1], [0, 1]]
    assert scheduler.freed_token_slots == [[1, 2, 3]]
    assert device_s >= 0.0
    assert host_s >= 0.0


def test_direct_label_only_chunk_opt_in_uses_token_id_only_logprobs(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "alloc_paged_token_slots_extend",
        lambda tree_cache, prefix_lens, seq_lens, last_loc, extend_num_tokens: np.arange(
            1, extend_num_tokens + 1, dtype=np.int32
        ),
    )
    logits_output = SimpleNamespace(
        next_token_logits=jax.numpy.zeros((2, 0), dtype=jax.numpy.float32),
        next_token_token_ids_logprobs_val=jax.numpy.log(
            jax.numpy.array([[0.2, 0.8], [0.6, 0.4]], dtype=jax.numpy.float32)
        ),
        next_token_token_ids_logprobs_idx=jax.numpy.array(
            [[10, 20], [10, 20]],
            dtype=jax.numpy.int32,
        ),
    )
    scheduler = _FakeSchedulerDirectLabelOnlyRunner(logits_output=logits_output)
    scheduler.server_args.multi_item_score_direct_token_ids_logprob_only = True
    scheduler.server_args.multi_item_score_direct_token_ids_logprob_only_chunk_size = 8192

    scores, device_s, host_s = scheduler._run_score_from_cache_v2_direct_chunk_label_only(
        cache_handle="cache-ok",
        chunk_items=[[101, 102], [201]],
        label_token_ids=[10, 20],
        label_token_ids_arr=jax.numpy.array([10, 20], dtype=jax.numpy.int32),
        apply_softmax=False,
        cached_last_node=None,
        cached_prefix_indices=np.array([7, 8, 9], dtype=np.int32),
        prefix_ids=[7, 8, 9],
        cached_extra_key=None,
    )

    np.testing.assert_allclose(
        np.asarray(scores),
        np.asarray([[0.2, 0.8], [0.6, 0.4]]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert scheduler.score_label_only_token_ids_only_calls == 1
    assert scheduler.score_label_only_fused_kernel_calls == 0
    assert scheduler.score_label_only_legacy_kernel_calls == 0
    assert scheduler.seen_model_worker_batch is not None
    assert scheduler.seen_model_worker_batch.next_token_token_ids_logprob_only is True
    assert scheduler.seen_model_worker_batch.next_token_token_ids_logprob_only_chunk_size == 8192
    np.testing.assert_array_equal(
        scheduler.seen_model_worker_batch.next_token_shared_token_ids,
        np.asarray([10, 20], dtype=np.int32),
    )
    assert scheduler.seen_model_worker_batch.token_ids_logprobs is None
    assert scheduler.freed_token_slots == [[1, 2, 3]]
    assert device_s >= 0.0
    assert host_s >= 0.0


def test_direct_label_only_chunk_auto_gate_uses_token_id_only_for_small_page_size(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "alloc_paged_token_slots_extend",
        lambda tree_cache, prefix_lens, seq_lens, last_loc, extend_num_tokens: np.arange(
            1, extend_num_tokens + 1, dtype=np.int32
        ),
    )
    logits_output = SimpleNamespace(
        next_token_logits=jax.numpy.zeros((2, 0), dtype=jax.numpy.float32),
        next_token_token_ids_logprobs_val=jax.numpy.log(
            jax.numpy.array([[0.3, 0.7], [0.4, 0.6]], dtype=jax.numpy.float32)
        ),
        next_token_token_ids_logprobs_idx=jax.numpy.array(
            [[10, 20], [10, 20]],
            dtype=jax.numpy.int32,
        ),
    )
    scheduler = _FakeSchedulerDirectLabelOnlyRunner(logits_output=logits_output)
    scheduler.page_size = 32
    scheduler.server_args.multi_item_score_direct_token_ids_logprob_only_auto = True

    scores, _, _ = scheduler._run_score_from_cache_v2_direct_chunk_label_only(
        cache_handle="cache-ok",
        chunk_items=[[101, 102], [201]],
        label_token_ids=[10, 20],
        label_token_ids_arr=jax.numpy.array([10, 20], dtype=jax.numpy.int32),
        apply_softmax=False,
        cached_last_node=None,
        cached_prefix_indices=np.array([7, 8, 9], dtype=np.int32),
        prefix_ids=[7, 8, 9],
        cached_extra_key=None,
    )

    np.testing.assert_allclose(
        np.asarray(scores),
        np.asarray([[0.3, 0.7], [0.4, 0.6]]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert scheduler.score_label_only_token_ids_only_calls == 1
    assert scheduler.score_label_only_fused_kernel_calls == 0
    assert scheduler.seen_model_worker_batch.next_token_token_ids_logprob_only is True


def test_direct_label_only_chunk_auto_gate_stays_fused_for_large_page_size_and_mrr(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "alloc_paged_token_slots_extend",
        lambda tree_cache, prefix_lens, seq_lens, last_loc, extend_num_tokens: np.arange(
            1, extend_num_tokens + 1, dtype=np.int32
        ),
    )
    logits_output = SimpleNamespace(
        next_token_logits=jax.numpy.log(
            jax.numpy.array([[0.2, 0.8], [0.6, 0.4]], dtype=jax.numpy.float32)
        ),
        next_token_token_ids_logprobs_val=None,
        next_token_token_ids_logprobs_idx=None,
    )
    scheduler = _FakeSchedulerDirectLabelOnlyRunner(logits_output=logits_output)
    scheduler.page_size = 64
    scheduler.server_args.max_running_requests = 64
    scheduler.server_args.multi_item_score_direct_token_ids_logprob_only_auto = True

    scores, _, _ = scheduler._run_score_from_cache_v2_direct_chunk_label_only(
        cache_handle="cache-ok",
        chunk_items=[[101, 102], [201]],
        label_token_ids=[0, 1],
        label_token_ids_arr=jax.numpy.array([0, 1], dtype=jax.numpy.int32),
        apply_softmax=False,
        cached_last_node=None,
        cached_prefix_indices=np.array([7, 8, 9], dtype=np.int32),
        prefix_ids=[7, 8, 9],
        cached_extra_key=None,
    )

    np.testing.assert_allclose(
        np.asarray(scores),
        np.asarray([[0.2, 0.8], [0.6, 0.4]]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_direct_label_only_chunk_auto_promotes_large_chunk_size_for_short_high_fanout(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "alloc_paged_token_slots_extend",
        lambda tree_cache, prefix_lens, seq_lens, last_loc, extend_num_tokens: np.arange(
            1, extend_num_tokens + 1, dtype=np.int32
        ),
    )
    logits_output = SimpleNamespace(
        next_token_logits=jax.numpy.zeros((256, 0), dtype=jax.numpy.float32),
        next_token_token_ids_logprobs_val=jax.numpy.log(
            jax.numpy.tile(
                jax.numpy.array([[0.3, 0.7]], dtype=jax.numpy.float32),
                (256, 1),
            )
        ),
        next_token_token_ids_logprobs_idx=jax.numpy.tile(
            jax.numpy.array([[10, 20]], dtype=jax.numpy.int32),
            (256, 1),
        ),
    )
    scheduler = _FakeSchedulerDirectLabelOnlyRunner(logits_output=logits_output)
    scheduler.page_size = 32
    scheduler.server_args.multi_item_score_direct_token_ids_logprob_only_auto = True
    scheduler.server_args.multi_item_score_direct_token_ids_logprob_only_chunk_size = 4096

    scheduler._run_score_from_cache_v2_direct_chunk_label_only(
        cache_handle="cache-ok",
        chunk_items=[[101, 102]] * 256,
        label_token_ids=[10, 20],
        label_token_ids_arr=jax.numpy.array([10, 20], dtype=jax.numpy.int32),
        apply_softmax=False,
        cached_last_node=None,
        cached_prefix_indices=np.arange(512, dtype=np.int32),
        prefix_ids=list(range(512)),
        cached_extra_key=None,
    )

    assert scheduler.seen_model_worker_batch is not None
    assert scheduler.seen_model_worker_batch.next_token_token_ids_logprob_only is True
    assert scheduler.seen_model_worker_batch.next_token_token_ids_logprob_only_chunk_size == 16384


def test_direct_label_only_chunk_keeps_base_chunk_size_for_long_seq(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "alloc_paged_token_slots_extend",
        lambda tree_cache, prefix_lens, seq_lens, last_loc, extend_num_tokens: np.arange(
            1, extend_num_tokens + 1, dtype=np.int32
        ),
    )
    logits_output = SimpleNamespace(
        next_token_logits=jax.numpy.zeros((256, 0), dtype=jax.numpy.float32),
        next_token_token_ids_logprobs_val=jax.numpy.log(
            jax.numpy.tile(
                jax.numpy.array([[0.25, 0.75]], dtype=jax.numpy.float32),
                (256, 1),
            )
        ),
        next_token_token_ids_logprobs_idx=jax.numpy.tile(
            jax.numpy.array([[10, 20]], dtype=jax.numpy.int32),
            (256, 1),
        ),
    )
    scheduler = _FakeSchedulerDirectLabelOnlyRunner(logits_output=logits_output)
    scheduler.page_size = 32
    scheduler.server_args.multi_item_score_direct_token_ids_logprob_only_auto = True
    scheduler.server_args.multi_item_score_direct_token_ids_logprob_only_chunk_size = 4096

    scheduler._run_score_from_cache_v2_direct_chunk_label_only(
        cache_handle="cache-ok",
        chunk_items=[[101, 102]] * 256,
        label_token_ids=[10, 20],
        label_token_ids_arr=jax.numpy.array([10, 20], dtype=jax.numpy.int32),
        apply_softmax=False,
        cached_last_node=None,
        cached_prefix_indices=np.arange(4090, dtype=np.int32),
        prefix_ids=list(range(4090)),
        cached_extra_key=None,
    )

    assert scheduler.seen_model_worker_batch is not None
    assert scheduler.seen_model_worker_batch.next_token_token_ids_logprob_only is True
    assert scheduler.seen_model_worker_batch.next_token_token_ids_logprob_only_chunk_size == 4096
    assert scheduler.score_label_only_token_ids_only_calls == 1
    assert scheduler.score_label_only_fused_kernel_calls == 0


def test_score_from_cache_v2_chunk_plan_honors_dispatch_token_budget():
    chunk_plan = Scheduler._build_score_from_cache_v2_chunk_plan(
        [
            [100] * 60,
            [200] * 50,
            [300] * 50,
            [400] * 40,
        ],
        4,
        prefix_len=100,
        token_budget=300,
    )

    assert [chunk_indices for chunk_indices, _ in chunk_plan] == [[0, 3], [1, 2]]
    assert [[len(item) for item in chunk_items] for _, chunk_items in chunk_plan] == [
        [60, 40],
        [50, 50],
    ]


def test_v6e8_replica_lane_page64_override(monkeypatch):
    monkeypatch.setattr(tuned_block_sizes, "get_tpu_version", lambda: 6)
    monkeypatch.setattr(
        tuned_block_sizes,
        "get_device_name",
        lambda num_devices=None: "TPU v6e-8" if num_devices == 8 else "TPU v6e",
    )
    monkeypatch.setattr(
        tuned_block_sizes.jax,
        "devices",
        lambda: [SimpleNamespace(device_kind="TPU v6e") for _ in range(8)],
    )

    bkv_p, bq = tuned_block_sizes.get_tuned_block_sizes(
        q_dtype=np.dtype("bfloat16"),
        kv_dtype=np.dtype("bfloat16"),
        actual_num_q_heads=16,
        actual_num_kv_heads=8,
        head_dim=128,
        page_size=64,
        max_num_tokens=8192,
        pages_per_seq=64,
    )

    assert (bkv_p, bq) == (32, 128)


def test_v6e8_replica_lane_page64_override_honors_thread_local_logical_device_count(
    monkeypatch,
):
    observed_num_devices = []

    monkeypatch.setattr(tuned_block_sizes, "get_tpu_version", lambda: 6)

    def _fake_get_device_name(num_devices=None):
        observed_num_devices.append(num_devices)
        return "TPU v6e-8" if num_devices == 8 else "TPU v6e"

    monkeypatch.setattr(tuned_block_sizes, "get_device_name", _fake_get_device_name)
    monkeypatch.setattr(
        tuned_block_sizes.jax,
        "devices",
        lambda: [SimpleNamespace(device_kind="TPU v6e") for _ in range(8)],
    )
    monkeypatch.delenv("SGLANG_LOGICAL_DEVICE_COUNT", raising=False)
    tuned_block_sizes.set_logical_device_count_override(4)
    try:
        bkv_p, bq = tuned_block_sizes.get_tuned_block_sizes(
            q_dtype=np.dtype("bfloat16"),
            kv_dtype=np.dtype("bfloat16"),
            actual_num_q_heads=16,
            actual_num_kv_heads=8,
            head_dim=128,
            page_size=64,
            max_num_tokens=8192,
            pages_per_seq=64,
        )
    finally:
        tuned_block_sizes.set_logical_device_count_override(None)

    assert 4 in observed_num_devices


def test_score_from_cache_v2_size_guard_fallback():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.force_estimated_words = np.iinfo(np.int32).max
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is False
    assert out.fallback_reason == "size_guard"
    assert scheduler.score_from_cache_v2_fallback == 1
    assert scheduler.score_from_cache_v2_fallback_reasons.get("size_guard") == 1


def test_score_from_cache_v2_runtime_exception_does_not_poison_future_requests():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.fail_next_chunk = True
    first = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(8)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )
    assert first.success is False
    assert first.fallback_reason == "runtime_exception"

    second = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[2] * 20 for _ in range(8)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )
    assert second.success is True
    assert len(second.scores) == 8


def test_score_from_cache_v2_label_only_uses_dedicated_chunk_runner():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.multi_item_score_label_only_logprob = True
    items = [[i] * 20 for i in range(10)]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 3
    assert [len(chunk) for chunk in scheduler.label_only_chunk_calls] == [4, 4, 2]
    assert scheduler.label_only_chunk_fused_flags == [True, True, True]
    assert scheduler.chunk_calls == []


def test_score_from_cache_v2_label_only_length_aware_packing_preserves_output_order():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.multi_item_score_label_only_logprob = True
    items = [
        [10] * 1,
        [20] * 4,
        [30] * 2,
        [40] * 4,
    ]
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=items,
            label_token_ids=[9454, 2753],
            items_per_step=2,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert out.dispatch_count == 2
    assert scheduler.label_only_chunk_calls == [[20, 40], [30, 10]]
    assert out.scores == [
        [10.0, 10.5],
        [20.0, 20.5],
        [30.0, 30.5],
        [40.0, 40.5],
    ]


def test_score_from_cache_v2_label_only_can_disable_fused_kernel():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.multi_item_score_label_only_logprob = True
    scheduler.server_args.multi_item_score_label_only_fused_kernel = False
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[i] * 20 for i in range(10)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert scheduler.label_only_chunk_fused_flags == [False, False, False]


def test_label_only_fused_kernel_matches_legacy_math():
    from jax import numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    out_sharding = NamedSharding(mesh, P(None, None))
    logits = jnp.array(
        [[1.5, -0.2, 0.8, 2.1], [0.4, 1.2, -1.0, 0.7]],
        dtype=jnp.float32,
    )
    label_ids = jnp.asarray([0, 3], dtype=jnp.int32)

    legacy_logprobs = scheduler_module._compute_label_only_logprobs(
        logits,
        label_ids,
        out_sharding,
    )
    legacy_probs = jnp.exp(legacy_logprobs)
    fused_probs = scheduler_module._compute_label_only_scores_fused(
        logits,
        label_ids,
        False,
        out_sharding,
    )
    fused_softmax = scheduler_module._compute_label_only_scores_fused(
        logits,
        label_ids,
        True,
        out_sharding,
    )
    legacy_softmax = jax.nn.softmax(legacy_probs, axis=-1)

    np.testing.assert_allclose(
        np.asarray(legacy_probs),
        np.asarray(fused_probs),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(legacy_softmax),
        np.asarray(fused_softmax),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("soft_cap", [0.0, 1.75])
def test_chunked_next_token_token_ids_logprobs_matches_full_logits_math(soft_cap):
    hidden_states = jax.numpy.array(
        [[1.25, -0.5, 0.75], [0.2, 1.1, -1.3]],
        dtype=jax.numpy.float32,
    )
    embedding = jax.numpy.array(
        [
            [0.5, -0.2, 1.0],
            [-0.3, 0.7, 0.4],
            [1.2, 0.1, -0.6],
            [0.8, -1.1, 0.2],
            [-0.9, 0.3, 0.5],
        ],
        dtype=jax.numpy.float32,
    )
    token_ids = jax.numpy.array(
        [[0, 4, 0], [2, 1, 0]],
        dtype=jax.numpy.int32,
    )
    token_mask = jax.numpy.array(
        [[True, True, False], [True, True, False]],
        dtype=jax.numpy.bool_,
    )
    safe_token_ids = jax.numpy.where(token_mask, token_ids, 0)
    selected_embeddings = embedding.at[safe_token_ids].get()

    selected_logprobs = _compute_next_token_token_ids_logprobs_chunked(
        hidden_states,
        embedding,
        selected_embeddings,
        token_mask,
        chunk_size=4,
        soft_cap=soft_cap,
    )

    full_logits = jax.numpy.dot(hidden_states, embedding.T)
    if soft_cap:
        full_logits = soft_cap * jax.numpy.tanh(full_logits / soft_cap)
    full_logprobs = jax.nn.log_softmax(full_logits.astype(jax.numpy.float32), axis=-1)
    expected = np.take_along_axis(
        np.asarray(full_logprobs),
        np.asarray(token_ids),
        axis=1,
    )
    expected = np.where(np.asarray(token_mask), expected, 0.0)

    np.testing.assert_allclose(
        np.asarray(selected_logprobs),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("soft_cap", [0.0, 1.75])
def test_chunked_next_token_shared_token_ids_logprobs_matches_full_logits_math(soft_cap):
    hidden_states = jax.numpy.array(
        [[1.25, -0.5, 0.75], [0.2, 1.1, -1.3]],
        dtype=jax.numpy.float32,
    )
    embedding = jax.numpy.array(
        [
            [0.5, -0.2, 1.0],
            [-0.3, 0.7, 0.4],
            [1.2, 0.1, -0.6],
            [0.8, -1.1, 0.2],
            [-0.9, 0.3, 0.5],
        ],
        dtype=jax.numpy.float32,
    )
    shared_token_ids = jax.numpy.array([0, 4], dtype=jax.numpy.int32)
    selected_embeddings = embedding.at[shared_token_ids].get()

    selected_logprobs = _compute_next_token_shared_token_ids_logprobs_chunked(
        hidden_states,
        embedding,
        selected_embeddings,
        chunk_size=4,
        soft_cap=soft_cap,
    )

    full_logits = jax.numpy.dot(hidden_states, embedding.T)
    if soft_cap:
        full_logits = soft_cap * jax.numpy.tanh(full_logits / soft_cap)
    full_logprobs = jax.nn.log_softmax(full_logits.astype(jax.numpy.float32), axis=-1)
    expected = np.take_along_axis(
        np.asarray(full_logprobs),
        np.asarray(shared_token_ids)[None, :].repeat(hidden_states.shape[0], axis=0),
        axis=1,
    )

    np.testing.assert_allclose(
        np.asarray(selected_logprobs),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_label_only_scores_from_logprobs_matches_fused_math():
    from jax import numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    out_sharding = NamedSharding(mesh, P(None, None))
    logits = jnp.array(
        [[1.5, -0.2, 0.8, 2.1], [0.4, 1.2, -1.0, 0.7]],
        dtype=jnp.float32,
    )
    label_ids = jnp.asarray([0, 3], dtype=jnp.int32)

    label_logprobs = scheduler_module._compute_label_only_logprobs(
        logits,
        label_ids,
        out_sharding,
    )
    from_logprobs = scheduler_module._compute_label_only_scores_from_logprobs(
        label_logprobs,
        False,
    )
    from_logprobs_softmax = scheduler_module._compute_label_only_scores_from_logprobs(
        label_logprobs,
        True,
    )
    fused_probs = scheduler_module._compute_label_only_scores_fused(
        logits,
        label_ids,
        False,
        out_sharding,
    )
    fused_softmax = scheduler_module._compute_label_only_scores_fused(
        logits,
        label_ids,
        True,
        out_sharding,
    )

    np.testing.assert_allclose(
        np.asarray(from_logprobs),
        np.asarray(fused_probs),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(from_logprobs_softmax),
        np.asarray(fused_softmax),
        rtol=1e-6,
        atol=1e-6,
    )


def test_score_from_cache_v2_label_only_rejects_unsupported_backend():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    scheduler.server_args.multi_item_score_label_only_logprob = True
    scheduler.server_args.device = "metal"
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is False
    assert out.fallback_reason == "unsupported_backend"


def test_score_from_cache_v2_updates_scoring_cache_lookup_counters_on_hit():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is True
    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["lookup_queries"] == 1
    assert metrics["lookup_hits"] == 1
    assert metrics["lookup_misses"] == 0
    assert metrics["lookup_by_path"]["score_from_cache_v2"]["queries"] == 1
    assert metrics["lookup_by_path"]["score_from_cache_v2"]["hits"] == 1
    assert metrics["lookup_hit_rate"] == 1.0


def test_score_from_cache_v2_updates_scoring_cache_lookup_counters_on_miss():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-missing",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is False
    assert out.fallback_reason == "missing_cache_handle"
    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["lookup_queries"] == 1
    assert metrics["lookup_hits"] == 0
    assert metrics["lookup_misses"] == 1
    assert metrics["lookup_by_path"]["score_from_cache_v2"]["queries"] == 1
    assert metrics["lookup_by_path"]["score_from_cache_v2"]["misses"] == 1
    assert metrics["lookup_hit_rate"] == 0.0


def test_score_from_cache_v2_timing_counters_are_recorded():
    scheduler = _FakeSchedulerScoreFromCacheV2()
    out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-ok",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert out.success is True
    assert scheduler.score_from_cache_v2_attempted == 1
    assert scheduler.score_from_cache_v2_succeeded == 1
    assert scheduler.score_from_cache_v2_queue_wait_s_total >= 0.0
    assert scheduler.score_from_cache_v2_device_compute_s_total == pytest.approx(0.01)
    assert scheduler.score_from_cache_v2_host_orchestration_s_total == pytest.approx(0.02)
    assert scheduler.score_from_cache_v2_device_compute_s_max == pytest.approx(0.01)
    assert scheduler.score_from_cache_v2_host_orchestration_s_max == pytest.approx(0.02)

    before_queue_wait = scheduler.score_from_cache_v2_queue_wait_s_total
    miss_out = scheduler.score_from_cache_v2(
        ScoreFromCacheReqInput(
            cache_handle="cache-missing",
            items_2d=[[1] * 20 for _ in range(4)],
            label_token_ids=[9454, 2753],
            items_per_step=4,
            apply_softmax=False,
        )
    )

    assert miss_out.success is False
    assert scheduler.score_from_cache_v2_attempted == 2
    assert scheduler.score_from_cache_v2_fallback == 1
    assert scheduler.score_from_cache_v2_queue_wait_s_total >= before_queue_wait
    # Missing-cache fallback records zero compute overhead.
    assert scheduler.score_from_cache_v2_device_compute_s_total == pytest.approx(0.01)
    assert scheduler.score_from_cache_v2_host_orchestration_s_total == pytest.approx(0.02)


def test_score_from_cache_v2_parity_metric_threshold():
    baseline_scores = [[0.1, 0.9], [0.3, 0.7], [0.8, 0.2]]
    fastpath_scores = [[0.1000004, 0.8999996], [0.3000001, 0.6999999], [0.8, 0.2]]
    max_abs_diff, mean_abs_diff = _parity_metrics(baseline_scores, fastpath_scores)
    assert max_abs_diff < 1e-3
    assert mean_abs_diff < 5e-4


def test_label_only_parity_metrics_returns_expected_values():
    baseline = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    candidate = np.array([[0.1005, 0.2002], [0.2998, 0.4004]], dtype=np.float32)
    max_abs_diff, mean_abs_diff = Scheduler._label_only_parity_metrics(
        baseline_logprobs=baseline,
        candidate_logprobs=candidate,
    )
    assert max_abs_diff == pytest.approx(0.0005, abs=1e-7)
    assert mean_abs_diff == pytest.approx(0.000325, abs=1e-7)


def test_label_only_parity_metrics_shape_mismatch_returns_inf():
    baseline = np.array([[0.1, 0.2]], dtype=np.float32)
    candidate = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    max_abs_diff, mean_abs_diff = Scheduler._label_only_parity_metrics(
        baseline_logprobs=baseline,
        candidate_logprobs=candidate,
    )
    assert math.isinf(max_abs_diff)
    assert math.isinf(mean_abs_diff)
