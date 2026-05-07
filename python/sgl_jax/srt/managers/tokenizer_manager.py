"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import contextlib
import copy
import dataclasses
import json
import logging
import os
import pickle
import signal
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any

import fastapi
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.lora.lora_registry import LoRARegistry
from sgl_jax.srt.managers.tokenizer_score_api_mixin import TokenizerScoreApiMixin
from sgl_jax.srt.managers.tokenizer_score_cache_mixin import TokenizerScoreCacheMixin
from sgl_jax.srt.managers.tokenizer_score_common import (
    _CorrelatedCommunicator,
    _SchedulerSender,
    ReqState,
)
from sgl_jax.srt.managers.tokenizer_score_routing_mixin import TokenizerScoreRoutingMixin
from sgl_jax.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    EmbeddingReqInput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GenerateReqInput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    HealthCheckOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ReleaseScoringCacheReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ScoreFromCacheReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.multimodal.tokenizer_utils import resolve_tokenizer_subdir
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    dataclass_to_string_truncated,
    get_bool_env_var,
    get_zmq_socket,
    kill_process_tree,
)
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


class TokenizerManager(
    TokenizerScoreApiMixin,
    TokenizerScoreCacheMixin,
    TokenizerScoreRoutingMixin,
):
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs | list[PortArgs],
    ):
        # Parse args
        self.server_args = server_args
        self.log_requests = server_args.log_requests
        self.log_requests_level = server_args.log_requests_level
        self.preferred_sampling_params = (
            json.loads(server_args.preferred_sampling_params)
            if server_args.preferred_sampling_params
            else None
        )
        self.crash_dump_folder = server_args.crash_dump_folder
        self.crash_dump_performed = False  # Flag to ensure dump is only called once
        self.event_loop = None  # Store the event loop to use

        # Init inter-process communication
        scheduler_port_args = port_args if isinstance(port_args, list) else [port_args]
        if not scheduler_port_args:
            raise ValueError("TokenizerManager requires at least one PortArgs entry.")
        primary_port_args = scheduler_port_args[0]
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, primary_port_args.tokenizer_ipc_name, True
        )
        scheduler_senders = [
            get_zmq_socket(context, zmq.PUSH, scheduler_port_arg.scheduler_input_ipc_name, True)
            for scheduler_port_arg in scheduler_port_args
        ]
        self.send_to_scheduler = _SchedulerSender(scheduler_senders)
        self.scheduler_port_count = self.send_to_scheduler.fan_out
        self.score_replica_lane_count = self.scheduler_port_count

        self.send_to_rpc = get_zmq_socket(context, zmq.DEALER, primary_port_args.rpc_ipc_name, True)

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        if not server_args.multimodal:
            self.model_config = ModelConfig.from_server_args(server_args)
            self.is_generation = self.model_config.is_generation
            self.context_len = self.model_config.context_len
            self.image_token_id = self.model_config.image_token_id
        else:
            self.model_config = None
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()
        self._updating = False
        self._cond = asyncio.Condition()

        self.mm_processor = None

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
        # Store states
        self.no_create_loop = False
        self.rid_to_state: dict[str, ReqState] = {}
        self.health_check_failed = False
        self.gracefully_exit = False
        self.last_receive_tstamp = 0
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_request_list: list[tuple] = []
        self.crash_dump_request_list: deque[tuple] = deque()
        self.log_request_metadata = self.get_log_request_metadata()
        self.session_futures = {}  # session_id -> asyncio event
        self.max_req_input_len = None
        self.asyncio_tasks = set()

        # For load balancing
        self.current_load = 0
        self.current_load_lock = asyncio.Lock()

        # Communicators
        self.release_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.resume_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.flush_cache_communicator = _Communicator(self.send_to_scheduler, server_args.dp_size)
        self.release_scoring_cache_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.score_from_cache_v2_communicator = _CorrelatedCommunicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.profile_communicator = _Communicator(self.send_to_scheduler, server_args.dp_size)
        self.get_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.set_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.local_rpc_submitter = None
        self.local_request_submitter = None
        self.score_fastpath_attempted = 0
        self.score_fastpath_succeeded = 0
        self.score_fastpath_fallback = 0
        self.score_fastpath_fallback_reasons: dict[str, int] = {}

        # LoRA
        self.lora_registry = LoRARegistry(self.server_args.lora_paths)

        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (
                        BatchStrOut,
                        BatchEmbeddingOut,
                        BatchTokenIDOut,
                    ),
                    self._handle_batch_output,
                ),
                (AbortReq, self._handle_abort_req),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    ReleaseMemoryOccupationReqOutput,
                    self.release_memory_occupation_communicator.handle_recv,
                ),
                (
                    ResumeMemoryOccupationReqOutput,
                    self.resume_memory_occupation_communicator.handle_recv,
                ),
                (
                    FlushCacheReqOutput,
                    self.flush_cache_communicator.handle_recv,
                ),
                (
                    ReleaseScoringCacheReqOutput,
                    self.release_scoring_cache_communicator.handle_recv,
                ),
                (
                    ScoreFromCacheReqOutput,
                    self.score_from_cache_v2_communicator.handle_recv,
                ),
                (
                    ProfileReqOutput,
                    self.profile_communicator.handle_recv,
                ),
                (
                    GetInternalStateReqOutput,
                    self.get_internal_state_communicator.handle_recv,
                ),
                (
                    SetInternalStateReqOutput,
                    self.set_internal_state_communicator.handle_recv,
                ),
                (HealthCheckOutput, lambda x: None),
            ]
        )
        self.wait_timeout = int(os.environ.get("SGLANG_WAIT_TIMEOUT", "4"))
        self.scheduler_pids: list[int] = []
        self.scheduler_unavailable_error: str | None = None

    async def generate_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        request: fastapi.Request | None = None,
    ):

        created_time = time.time()
        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        self.auto_create_handle_loop()
        obj.normalize_batch_and_arguments()

        # Acquire LoRA ID if lora_path is provided
        if isinstance(obj, GenerateReqInput) and self.server_args.enable_lora and obj.lora_path:
            obj.lora_id = await self.lora_registry.acquire(obj.lora_path)

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                "Receive: obj=%s",
                dataclass_to_string_truncated(obj, max_length, skip_names=skip_names),
            )

        if obj.is_single:
            tokenized_obj = await self._tokenize_one_request(obj)
            state = self._send_one_request(obj, tokenized_obj, created_time)
            async for response in self._wait_one_response(obj, state, request):
                yield response
        else:
            async for response in self._handle_batch_request(obj, request, created_time):
                yield response

    async def _tokenize_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ):
        """Tokenize one request."""

        # Tokenize
        input_text = obj.text
        input_ids = obj.input_ids
        if input_ids is None and input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but input_text requires tokenization"
                )
            encoded = self.tokenizer(input_text)
            input_ids = encoded["input_ids"]
        self._validate_one_request(obj, input_ids)
        return self._create_tokenized_object(obj, input_text, input_ids)

    def _validate_one_request(
        self, obj: GenerateReqInput | EmbeddingReqInput, input_ids: list[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""

        input_token_num = len(input_ids) if input_ids is not None else 0
        # Check if input alone exceeds context length
        if input_token_num >= self.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Check total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if max_new_tokens is not None and (max_new_tokens + input_token_num) >= self.context_len:
            total_tokens = max_new_tokens + input_token_num
            error_msg = (
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{max_new_tokens} tokens for the completion. Please reduce the number "
                f"of tokens in the input messages or the completion to fit within the limit."
            )
            raise ValueError(error_msg)

    def _validate_input_ids_in_vocab(self, input_ids: list[int], vocab_size: int) -> None:
        if any(id >= vocab_size for id in input_ids):
            raise ValueError(
                f"The input_ids {input_ids} contains values greater than the vocab size ({vocab_size})."
            )

    def _create_tokenized_object(
        self,
        obj: GenerateReqInput,
        input_text: str,
        input_ids: list[int],
    ) -> TokenizedGenerateReqInput:
        """Create a tokenized request object from common parameters."""
        # Parse sampling parameters
        # Note: if there are preferred sampling params, we use them if they are not
        # explicitly passed in sampling_params
        if self.preferred_sampling_params:
            sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
        else:
            sampling_kwargs = obj.sampling_params
        sampling_params = SamplingParams(**sampling_kwargs)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        # Build return object

        tokenized_obj = TokenizedGenerateReqInput(
            rid=obj.rid,
            text=input_text,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=obj.return_logprob,
            return_output_logprob_only=obj.return_output_logprob_only,
            logprob_start_len=obj.logprob_start_len,
            top_logprobs_num=obj.top_logprobs_num,
            token_ids_logprob=obj.token_ids_logprob,
            stream=obj.stream,
            lora_id=obj.lora_id,
            extra_key=obj.extra_key,
            return_routed_experts=obj.return_routed_experts,
            cache_for_scoring=bool(obj.cache_for_scoring),
            extend_from_cache=obj.extend_from_cache,
        )
        # note: When only `return_logprob` is specified, we assume that only the output probability is required.
        if (
            tokenized_obj.return_logprob
            and (obj.logprob_start_len is None or obj.logprob_start_len == -1)
            and (obj.top_logprobs_num == 0 or obj.top_logprobs_num is None)
            and obj.token_ids_logprob is None
        ):
            tokenized_obj.return_logprob = False
            obj.return_output_logprob_only = True
            tokenized_obj.return_output_logprob_only = True

        return tokenized_obj

    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: GenerateReqInput
    ) -> list[TokenizedGenerateReqInput | TokenizedEmbeddingReqInput]:
        """Handle batch tokenization for text inputs only."""
        logger.debug("Starting batch tokenization for %s text requests", batch_size)

        # Collect requests and texts
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # Batch tokenize all texts
        encoded = self.tokenizer(texts)
        input_ids_list = encoded["input_ids"]

        # Process all requests
        tokenized_objs = []
        for i, req in enumerate(requests):
            # self._validate_token_len(obj[i], input_ids_list[i])
            tokenized_objs.append(self._create_tokenized_object(req, req.text, input_ids_list[i]))
        logger.debug("Completed batch processing for %s requests", batch_size)
        return tokenized_objs

    def _validate_batch_tokenization_constraints(
        self, batch_size: int, obj: GenerateReqInput | EmbeddingReqInput
    ) -> None:
        """Validate constraints for batch tokenization processing."""
        for i in range(batch_size):
            if self.is_generation and obj[i].contains_mm_input():
                raise ValueError(
                    "For multimodal input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )

    def _notify_state_event(self, state: ReqState) -> None:
        """Thread-safe wrapper around state.event.set().

        If enable_engine_loop_run_forever_daemon was enabled, handle_loop would run on the daemon_loop thread, but the asyncio.Event's
        internal Future belongs to the eval_loop (the loop that called
        _send_one_request).  Calling fut.set_result() from the wrong thread
        does not wake up eval_loop's selector.  call_soon_threadsafe writes to
        the self-pipe so the selector returns from epoll_wait immediately.
        """
        loop = state.event_loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                # RuntimeError: loop is already closed (request timed-out / cancelled).
                loop.call_soon_threadsafe(state.event.set)
        else:
            state.event.set()

    async def _wait_one_response(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        state: ReqState,
        request: fastapi.Request | None = None,
    ):
        """Wait for the response of one request."""
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
            except TimeoutError:
                if request is not None and await request.is_disconnected():
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    try:
                        raise ValueError(
                            f"Request is disconnected from the client side (type 1). Abort request rid={obj.rid}"
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Request is disconnected from the client side (type 1). Abort request rid={obj.rid}"
                        ) from e
                if not self._check_scheduler_health():
                    raise ValueError(
                        self.scheduler_unavailable_error
                        or "Scheduler subprocess is unavailable. Please restart the server."
                    ) from None
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                    logger.info(msg)

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if finish_reason.get("type") == "abort":
                        raise ValueError(
                            finish_reason.get("message") or "Request aborted by scheduler."
                        )

                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    # Abort the request for disconnected requests (non-streaming, running)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 3). Abort request {obj.rid=}"
                    )

    async def _handle_batch_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        request: fastapi.Request | None = None,
        created_time: float | None = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self.server_args.enable_tokenizer_batch_encode:
                # Validate batch tokenization constraints
                self._validate_batch_tokenization_constraints(batch_size, obj)

                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)
                batched_objs = [obj[i] for i in range(batch_size)]
                if getattr(self.server_args, "enable_tokenizer_batch_send", False):
                    states = self._send_batch_requests(
                        batched_objs,
                        tokenized_objs,
                        created_time,
                    )
                    for tmp_obj, state in zip(batched_objs, states, strict=True):
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
                else:
                    for tmp_obj, tokenized_obj in zip(batched_objs, tokenized_objs, strict=True):
                        state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
            else:
                # Sequential tokenization and processing
                batched_objs = [obj[i] for i in range(batch_size)]
                tokenized_objs = []
                for tmp_obj in batched_objs:
                    tokenized_objs.append(await self._tokenize_one_request(tmp_obj))

                if getattr(self.server_args, "enable_tokenizer_batch_send", False):
                    states = self._send_batch_requests(
                        batched_objs,
                        tokenized_objs,
                        created_time,
                    )
                    for tmp_obj, state in zip(batched_objs, states, strict=True):
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
                else:
                    for tmp_obj, tokenized_obj in zip(batched_objs, tokenized_objs, strict=True):
                        state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, state, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, state, request))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(task_map.keys(), return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    async def flush_cache(self) -> FlushCacheReqOutput:
        self.auto_create_handle_loop()
        return (await self.flush_cache_communicator(FlushCacheReqInput()))[0]

    def abort_request(self, rid: str = "", abort_all: bool = False):
        if not abort_all and rid not in self.rid_to_state:
            return
        req = AbortReq(rid=rid, abort_all=abort_all)
        self.send_to_scheduler.send_pyobj(req)

    async def start_profile(
        self,
        output_dir: str | None = None,
        start_step: int | None = None,
        num_steps: int | None = None,
        host_tracer_level: int | None = None,
        python_tracer_level: int | None = None,
        stage_id: int | None = None,
        profile_by_stage: bool = False,
        profile_stages: list[str] | None = None,
    ):
        self.auto_create_handle_loop()
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=output_dir,
            start_step=start_step,
            num_steps=num_steps,
            host_tracer_level=host_tracer_level,
            python_tracer_level=python_tracer_level,
            profile_id=str(time.time()),
            stage_id=stage_id,
            profile_by_stage=profile_by_stage,
            profile_stages=profile_stages,
        )
        return await self._execute_profile(req)

    async def stop_profile(self):
        self.auto_create_handle_loop()
        req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def get_profile_status(self):
        self.auto_create_handle_loop()
        req = ProfileReq(type=ProfileReqType.GET_STATUS)
        return await self._execute_profile(req)

    async def _execute_profile(self, req: ProfileReq):
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    async def pause_generation(self, obj: PauseGenerationReqInput):
        async with self.is_pause_cond:
            self.is_pause = True
            if obj.mode != "abort":
                await self.send_to_scheduler.send_pyobj(obj)
            else:
                # use len(self.rid_to_state) == 0 to ensure all requests are aborted
                while True:
                    self.abort_request(abort_all=True)
                    if len(self.rid_to_state) == 0:
                        break
                    await asyncio.sleep(0.1)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        async with self.is_pause_cond:
            self.is_pause = False
            await self.send_to_scheduler.send_pyobj(obj)
            self.is_pause_cond.notify_all()

    async def release_memory_occupation(
        self,
        obj: ReleaseMemoryOccupationReqInput,
        request: fastapi.Request | None = None,
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self,
        obj: ResumeMemoryOccupationReqInput,
        request: fastapi.Request | None = None,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def open_session(self, obj: OpenSessionReqInput, request: fastapi.Request | None = None):
        self.auto_create_handle_loop()

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        self.send_to_scheduler.send_pyobj(obj)

        self.session_futures[obj.session_id] = asyncio.Future()
        session_id = await self.session_futures[obj.session_id]
        del self.session_futures[obj.session_id]
        return session_id

    async def close_session(
        self, obj: CloseSessionReqInput, request: fastapi.Request | None = None
    ):
        await self.send_to_scheduler.send_pyobj(obj)

    async def get_internal_state(self) -> list[dict[Any, Any]]:
        self.auto_create_handle_loop()
        req = GetInternalStateReq()
        responses: list[GetInternalStateReqOutput] = await self.get_internal_state_communicator(req)
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def get_load(self) -> dict:
        if not self.current_load_lock.locked():
            async with self.current_load_lock:
                internal_state = await self.get_internal_state()
                self.current_load = internal_state[0]["load"]
        return {"load": self.current_load}

    async def set_internal_state(self, obj: SetInternalStateReq) -> SetInternalStateReqOutput:
        self.auto_create_handle_loop()
        responses: list[SetInternalStateReqOutput] = await self.set_internal_state_communicator(obj)
        return (
            responses[0]
            if responses
            else SetInternalStateReqOutput(
                request_id=obj.request_id,
                success=False,
                error_msg="No response from scheduler",
            )
        )

    def get_log_request_metadata(self):
        max_length = None
        skip_names = None
        out_skip_names = None
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = set(
                    [
                        "text",
                        "input_ids",
                        "input_embeds",
                        "image_data",
                        "audio_data",
                        "lora_path",
                        "sampling_params",
                    ]
                )
                out_skip_names = set(
                    [
                        "text",
                        "output_ids",
                        "embedding",
                    ]
                )
            elif self.log_requests_level == 1:
                max_length = 1 << 30
                skip_names = set(
                    [
                        "text",
                        "input_ids",
                        "input_embeds",
                        "image_data",
                        "audio_data",
                        "lora_path",
                    ]
                )
                out_skip_names = set(
                    [
                        "text",
                        "output_ids",
                        "embedding",
                    ]
                )
            elif self.log_requests_level == 2:
                max_length = 2048
            elif self.log_requests_level == 3:
                max_length = 1 << 30
            else:
                raise ValueError(f"Invalid --log-requests-level: {self.log_requests_level=}")
        return max_length, skip_names, out_skip_names

    def configure_logging(self, obj: ConfigureLoggingReq):
        if obj.log_requests is not None:
            self.log_requests = obj.log_requests
        if obj.log_requests_level is not None:
            self.log_requests_level = obj.log_requests_level
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        if obj.crash_dump_folder is not None:
            self.crash_dump_folder = obj.crash_dump_folder
        self.log_request_metadata = self.get_log_request_metadata()

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(2)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def auto_create_handle_loop(self):
        if self.no_create_loop:
            return

        self.no_create_loop = True
        # Use the provided event loop if available, otherwise get the current one
        loop = self.event_loop if self.event_loop is not None else asyncio.get_event_loop()

        try:
            current_running_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_running_loop = None

        if current_running_loop == loop:
            task = loop.create_task(print_exception_wrapper(self.handle_loop))
            self.asyncio_tasks.add(task)
        else:
            asyncio.run_coroutine_threadsafe(print_exception_wrapper(self.handle_loop), loop)

        # We cannot add signal handler when the tokenizer manager is not in
        # the main thread due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = SignalHandler(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(signal.SIGQUIT, signal_handler.running_phase_sigquit_handler)
        else:
            logger.warning(
                "Signal handler is not added because the tokenizer manager is "
                "not in the main thread. This disables graceful shutdown of the "
                "tokenizer manager when SIGTERM is received."
            )
        self.asyncio_tasks.add(loop.create_task(print_exception_wrapper(self.sigterm_watchdog)))

    def dump_requests_before_crash(self):
        if self.crash_dump_performed:
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return
        logger.error(
            "Dumping requests before crash. crash_dump_folder=%s",
            self.crash_dump_folder,
        )
        self.crash_dump_performed = True
        if not self.crash_dump_folder:
            return

        data_to_dump = []
        if self.crash_dump_request_list:
            data_to_dump.extend(self.crash_dump_request_list)

        # Add unfinished requests from rid_to_state
        unfinished_requests = []
        for rid, state in self.rid_to_state.items():
            if not state.finished:
                unfinished_requests.append((state.obj, {}, state.created_time, time.time()))
        if unfinished_requests:
            data_to_dump.extend(unfinished_requests)

        if not data_to_dump:
            return

        filename = os.path.join(
            self.crash_dump_folder,
            os.getenv("HOSTNAME", None),
            f"crash_dump_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl",
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Include server_args in the dump
        data_to_dump_with_server_args = {
            "server_args": self.server_args,
            "requests": data_to_dump,
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_dump_with_server_args, f)
        logger.error(
            "Dumped %d finished and %d unfinished requests before crash to %s",
            len(self.crash_dump_request_list),
            len(unfinished_requests),
            filename,
        )

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)

            if self.health_check_failed:
                # if health check failed, we should exit immediately
                logger.error(
                    "Signal SIGTERM received while health check failed. Exiting... remaining number of requests: %d",
                    remain_num_req,
                )
                self.dump_requests_before_crash()
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                # if force shutdown flag set, exit immediately
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting... remaining number of requests: %d",
                    remain_num_req,
                )
                break

            logger.info(
                "Gracefully exiting... remaining number of requests %d",
                remain_num_req,
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                self.dump_requests_before_crash()
                break

        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    async def handle_loop(self):
        """The event loop that handles requests"""
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = time.perf_counter()

    def _handle_batch_output(
        self,
        recv_obj: BatchStrOut | BatchEmbeddingOut | BatchTokenIDOut,
    ):
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                logger.error(
                    "Received output for rid=%s but the state was deleted in TokenizerManager.",
                    rid,
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
            }

            if getattr(state.obj, "return_logprob", False) or getattr(
                state.obj, "return_output_logprob_only", False
            ):
                self.convert_logprob_style(
                    meta_info,
                    state,
                    state.obj.top_logprobs_num,
                    state.obj.token_ids_logprob,
                    state.obj.return_text_in_logprobs and not self.server_args.skip_tokenizer_init,
                    recv_obj,
                    i,
                )

            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if getattr(recv_obj, "output_routed_experts", None):
                meta_info["routed_experts"] = recv_obj.output_routed_experts[i]

            if getattr(recv_obj, "cache_miss_count", None) is not None:
                if (
                    get_bool_env_var("SGLANG_JAX_ENABLE_CACHE_MISS_CHECK")
                    and recv_obj.cache_miss_count > 0
                ):
                    raise RuntimeError(
                        f"Cache miss occurred {recv_obj.cache_miss_count} times, please check if the precompile logic covers the current scenario"
                    )
                meta_info["cache_miss_count"] = recv_obj.cache_miss_count
            if getattr(recv_obj, "scheduler_queue_wait_s", None) is not None:
                meta_info["scheduler_queue_wait_s"] = recv_obj.scheduler_queue_wait_s[i]
            if getattr(recv_obj, "scheduler_device_compute_s", None) is not None:
                meta_info["scheduler_device_compute_s"] = recv_obj.scheduler_device_compute_s[i]
            if getattr(recv_obj, "scheduler_host_overhead_s", None) is not None:
                meta_info["scheduler_host_overhead_s"] = recv_obj.scheduler_host_overhead_s[i]
            if getattr(recv_obj, "scheduler_dispatch_count", None) is not None:
                meta_info["scheduler_dispatch_count"] = recv_obj.scheduler_dispatch_count[i]

            if isinstance(recv_obj, BatchStrOut):
                state.text += recv_obj.output_strs[i]
                state.output_ids += recv_obj.output_ids[i]
                out_dict = {
                    "text": state.text,
                    "output_ids": state.output_ids,
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchTokenIDOut):
                if self.server_args.stream_output and state.obj.stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
            else:
                assert isinstance(recv_obj, BatchEmbeddingOut)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            finished_reason = recv_obj.finished_reasons[i] is not None
            if finished_reason:
                state.observed_finish_count += 1
            state.finished = finished_reason and state.observed_finish_count >= max(
                1, state.expected_finish_count
            )
            if state.finished:
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time
                # Release LoRA ID if it was acquired
                # Note: Only GenerateReqInput supports LoRA, not EmbeddingReqInput
                if (
                    isinstance(state.obj, GenerateReqInput)
                    and self.server_args.enable_lora
                    and state.obj.lora_id
                ):
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))
                del self.rid_to_state[rid]

            state.out_list.append(out_dict)
            self._notify_state_event(state)

            # Log metrics and dump
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)
            if self.crash_dump_folder and state.finished and state.obj.log_metrics:
                self.record_request_for_crash_dump(state, out_dict)

    def convert_logprob_style(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: list[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ):
        if state.obj.return_output_logprob_only:
            state.output_token_logprobs_val.extend(
                recv_obj.output_token_logprobs_val[recv_obj_index]
            )
            state.output_token_logprobs_idx.extend(
                recv_obj.output_token_logprobs_idx[recv_obj_index]
            )
            meta_info["output_token_logprobs"] = self.detokenize_logprob_tokens(
                state.output_token_logprobs_val,
                state.output_token_logprobs_idx,
                return_text_in_logprobs,
            )
            if (
                token_ids_logprob is not None
                and recv_obj.output_token_ids_logprobs_val is not None
                and len(recv_obj.output_token_ids_logprobs_val) > 0
            ):
                state.output_token_ids_logprobs_val.extend(
                    recv_obj.output_token_ids_logprobs_val[recv_obj_index]
                )
                state.output_token_ids_logprobs_idx.extend(
                    recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
                )
                meta_info["output_token_ids_logprobs"] = self.detokenize_top_logprobs_tokens(
                    state.output_token_ids_logprobs_val,
                    state.output_token_ids_logprobs_idx,
                    return_text_in_logprobs,
                )
            return
        if recv_obj.input_token_logprobs_val is None:
            return
        if len(recv_obj.input_token_logprobs_val) > 0:
            state.input_token_logprobs_val.extend(recv_obj.input_token_logprobs_val[recv_obj_index])
            state.input_token_logprobs_idx.extend(recv_obj.input_token_logprobs_idx[recv_obj_index])
        state.output_token_logprobs_val.extend(recv_obj.output_token_logprobs_val[recv_obj_index])
        state.output_token_logprobs_idx.extend(recv_obj.output_token_logprobs_idx[recv_obj_index])
        meta_info["input_token_logprobs"] = self.detokenize_logprob_tokens(
            state.input_token_logprobs_val,
            state.input_token_logprobs_idx,
            return_text_in_logprobs,
        )
        meta_info["output_token_logprobs"] = self.detokenize_logprob_tokens(
            state.output_token_logprobs_val,
            state.output_token_logprobs_idx,
            return_text_in_logprobs,
        )

        if top_logprobs_num > 0:
            if len(recv_obj.input_top_logprobs_val) > 0:
                state.input_top_logprobs_val.extend(recv_obj.input_top_logprobs_val[recv_obj_index])
                state.input_top_logprobs_idx.extend(recv_obj.input_top_logprobs_idx[recv_obj_index])
            state.output_top_logprobs_val.extend(recv_obj.output_top_logprobs_val[recv_obj_index])
            state.output_top_logprobs_idx.extend(recv_obj.output_top_logprobs_idx[recv_obj_index])
            meta_info["input_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.input_top_logprobs_val,
                state.input_top_logprobs_idx,
                return_text_in_logprobs,
            )
            meta_info["output_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.output_top_logprobs_val,
                state.output_top_logprobs_idx,
                return_text_in_logprobs,
            )

        if token_ids_logprob is not None:
            if len(recv_obj.input_token_ids_logprobs_val) > 0:
                state.input_token_ids_logprobs_val.extend(
                    recv_obj.input_token_ids_logprobs_val[recv_obj_index]
                )
                state.input_token_ids_logprobs_idx.extend(
                    recv_obj.input_token_ids_logprobs_idx[recv_obj_index]
                )
            state.output_token_ids_logprobs_val.extend(
                recv_obj.output_token_ids_logprobs_val[recv_obj_index]
            )
            state.output_token_ids_logprobs_idx.extend(
                recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
            )
            meta_info["input_token_ids_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.input_token_ids_logprobs_val,
                state.input_token_ids_logprobs_idx,
                return_text_in_logprobs,
            )
            meta_info["output_token_ids_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.output_token_ids_logprobs_val,
                state.output_token_ids_logprobs_idx,
                return_text_in_logprobs,
            )

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):
            if token_logprobs_val[i]:
                ret.append(
                    self.detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret

    def dump_requests(self, state: ReqState, out_dict: dict):
        self.dump_request_list.append((state.obj, out_dict, state.created_time, time.time()))

        if len(self.dump_request_list) >= self.dump_requests_threshold:
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            logger.info("Dump %s requests to %s", len(self.dump_request_list), filename)

            to_dump = self.dump_request_list
            self.dump_request_list = []

            to_dump_with_server_args = {
                "server_args": self.server_args,
                "requests": to_dump,
            }

            def background_task():
                os.makedirs(self.dump_requests_folder, exist_ok=True)
                with open(filename, "wb") as f:
                    pickle.dump(to_dump_with_server_args, f)

            # Schedule the task to run in the background without awaiting it
            asyncio.create_task(asyncio.to_thread(background_task))

    def record_request_for_crash_dump(self, state: ReqState, out_dict: dict):
        current_time = time.time()
        self.crash_dump_request_list.append((state.obj, out_dict, state.created_time, current_time))
        # Remove requests older than 5 minutes based on finish time
        while (
            self.crash_dump_request_list
            and current_time - self.crash_dump_request_list[0][3] >= 300
        ):
            self.crash_dump_request_list.popleft()

    def _handle_abort_req(self, recv_obj):
        state = self.rid_to_state[recv_obj.rid]
        state.finished = True
        state.out_list.append(
            {
                "text": "",
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": {
                        "type": "abort",
                        "message": "Abort before prefill",
                    },
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            }
        )
        notify_state_event = getattr(self, "_notify_state_event", None)
        if notify_state_event is not None:
            notify_state_event(state)
        else:
            state.event.set()

    def _handle_open_session_req_output(self, recv_obj):
        self.session_futures[recv_obj.session_id].set_result(
            recv_obj.session_id if recv_obj.success else None
        )


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("TokenizerManager hit an exception: %s", traceback)
        if hasattr(func, "__self__") and isinstance(func.__self__, TokenizerManager):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


class SignalHandler:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def sigterm_handler(self, signum=None, frame=None):
        logger.warning(
            "SIGTERM received. signum=%s frame=%s. Draining requests and shutting down...",
            signum,
            frame,
        )
        self.tokenizer_manager.gracefully_exit = True

    def running_phase_sigquit_handler(self, signum=None, frame=None):
        logger.error("Received sigquit from a child process. It usually means the child failed.")
        self.tokenizer_manager.dump_requests_before_crash()
        kill_process_tree(os.getpid())


@dataclasses.dataclass
class _Communicator[T]:
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    def __init__(self, sender, fan_out: int):
        self._sender = sender
        self._fan_out = fan_out
        self._lock = asyncio.Lock()
        self._result_event: asyncio.Event | None = None
        self._result_values: list[T] | None = None

    async def __call__(
        self,
        obj,
        timeout: float | None = None,
        scheduler_idx: int | None = None,
        broadcast: bool = False,
    ):
        async with self._lock:
            if self._result_event is not None or self._result_values is not None:
                raise RuntimeError(
                    "Communicator received a new call while a previous call is still active."
                )

            self._result_event = asyncio.Event()
            self._result_values = []
            try:
                if obj is not None:
                    if broadcast and hasattr(self._sender, "send_pyobj_all"):
                        self._sender.send_pyobj_all(obj)
                    elif scheduler_idx is not None and hasattr(self._sender, "send_pyobj_to"):
                        self._sender.send_pyobj_to(scheduler_idx, obj)
                    else:
                        self._sender.send_pyobj(obj)

                wait_coro = self._result_event.wait()
                if timeout is not None and timeout > 0:
                    await asyncio.wait_for(wait_coro, timeout=timeout)
                else:
                    await wait_coro

                return list(self._result_values)
            finally:
                self._result_event = None
                self._result_values = None

    def handle_recv(self, recv_obj: T):
        if self._result_values is None or self._result_event is None:
            logger.warning(
                "Dropping communicator response with no active waiter. type=%s",
                type(recv_obj).__name__,
            )
            return
        self._result_values.append(recv_obj)
        if len(self._result_values) >= self._fan_out:
            self._result_event.set()
