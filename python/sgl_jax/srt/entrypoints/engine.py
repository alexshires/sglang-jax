"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements python APIs for the inference engine.
"""

import asyncio
import atexit
import contextlib
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import jax
import uvloop
import zmq
import zmq.asyncio
from flax import nnx

from sgl_jax.srt.utils.common_utils import (
    SUPPORTED_LORA_TARGET_MODULES,
    get_or_create_loop,
)
from sgl_jax.utils import traverse_and_update

# ruff: noqa: E402
# Fix a bug of Python threading
threading._register_atexit = lambda *args, **kwargs: None

from sgl_jax.srt.entrypoints.EngineBase import EngineBase
from sgl_jax.srt.hf_transformers_utils import get_generation_config
from sgl_jax.srt.managers.detokenizer_manager import (
    run_detokenizer_process,
    run_detokenizer_thread,
)
from sgl_jax.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sgl_jax.srt.managers.scheduler import (
    run_scheduler_loop_thread_after_create,
    run_scheduler_process,
)
from sgl_jax.srt.managers.template_manager import TemplateManager
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_process_tree,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
    set_ulimit,
)
from sgl_jax.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def _resolve_dp_scheduler_device_partitions(
    server_args: ServerArgs,
    available_device_ids: list[int],
) -> list[list[int]]:
    dp_size = int(getattr(server_args, "dp_size", 1) or 1)
    if dp_size <= 1:
        device_indexes = list(server_args.device_indexes or available_device_ids)
        return [device_indexes]

    tensor_parallel_size = int(getattr(server_args, "tp_size", 1) or 1)
    if tensor_parallel_size <= 0:
        raise ValueError("tp_size must be positive when dp_size > 1.")

    requested_device_ids = list(server_args.device_indexes or available_device_ids)
    total_required_devices = dp_size * tensor_parallel_size
    if len(requested_device_ids) != total_required_devices:
        raise ValueError(
            "dp_size serving requires an exact device partition: "
            f"got {len(requested_device_ids)} device(s), need {total_required_devices} "
            f"for dp_size={dp_size} and tp_size={tensor_parallel_size}."
        )

    return [
        requested_device_ids[start : start + tensor_parallel_size]
        for start in range(0, total_required_devices, tensor_parallel_size)
    ]


def _build_scheduler_launch_plan(
    server_args: ServerArgs,
    port_args: PortArgs,
) -> list[tuple[ServerArgs, PortArgs, int]]:
    if server_args.device == "tpu" and not server_args.enable_single_process:
        # Avoid parent-side TPU PJRT initialization before scheduler subprocesses
        # spawn. For single-host TPU serving the worker ids are dense 0..N-1, so
        # tp_size * dp_size is enough to seed the launch partitioning here.
        total_required_devices = max(
            1,
            int(getattr(server_args, "tp_size", 1) or 1)
            * int(getattr(server_args, "dp_size", 1) or 1),
        )
        available_device_ids = list(range(total_required_devices))
    else:
        available_device_ids = [device.id for device in jax.devices()]
    device_partitions = _resolve_dp_scheduler_device_partitions(server_args, available_device_ids)
    plan: list[tuple[ServerArgs, PortArgs, int]] = []
    for dp_rank, device_indexes in enumerate(device_partitions):
        lane_server_args = dataclasses.replace(
            server_args,
            device_indexes=list(device_indexes),
            dp_size=1,
        )
        if dp_rank == 0:
            lane_port_args = port_args
        else:
            lane_port_args = dataclasses.replace(
                PortArgs.init_new(server_args),
                tokenizer_ipc_name=port_args.tokenizer_ipc_name,
                detokenizer_ipc_name=port_args.detokenizer_ipc_name,
            )
        plan.append((lane_server_args, lane_port_args, dp_rank))
    return plan


class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"

            if kwargs.get("multimodal", False):
                from sgl_jax.srt.multimodal.common.ServerArgs import (
                    MultimodalServerArgs,
                )

                server_args = MultimodalServerArgs(**kwargs)
            else:
                server_args = ServerArgs(**kwargs)

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        self.port_args = PortArgs.init_new(server_args)
        self.server_args = server_args
        logger.info("server_args=%s", server_args)

        # Launch subprocesses or threads
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses_or_threads(
            server_args=server_args,
            port_args=self.port_args,
        )
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.scheduler_info = scheduler_info
        self.default_sampling_params: dict[str, Any] | None = None
        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(context, zmq.DEALER, self.port_args.rpc_ipc_name, True)

        if self.server_args.enable_engine_loop_run_forever_daemon:
            import queue

            result_queue = queue.Queue()

            def run_loop_forever():
                loop = get_or_create_loop()
                result_queue.put(loop)
                loop.run_forever()

            loop_thread = threading.Thread(target=run_loop_forever, daemon=True)
            loop_thread.start()
            self.loop = result_queue.get()
            self.tokenizer_manager.event_loop = self.loop
        else:
            self.loop = get_or_create_loop()

    def generate(
        self,
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: list[list[int]] | list[int] | None = None,
        return_logprob: list[bool] | bool | None = False,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        stream: bool = False,
        lora_path: list[str] | str | None = None,
        return_routed_experts: list[bool] | bool | None = False,
    ) -> dict | Iterator[dict]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            stream=stream,
            lora_path=lora_path,
            return_routed_experts=return_routed_experts,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = self.loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = self.loop.run_until_complete(generator.__anext__())
            return ret

    async def async_generate(
        self,
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: list[list[int]] | list[int] | None = None,
        return_logprob: list[bool] | bool | None = False,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        stream: bool = False,
        lora_path: list[str] | str | None = None,
        return_routed_experts: list[bool] | bool | None = False,
    ) -> dict | AsyncIterator[dict]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            stream=stream,
            lora_path=lora_path,
            return_routed_experts=return_routed_experts,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True:
            return generator
        else:
            return await generator.__anext__()

    def apply_dummy_lora_ab_buffer(self, target_modules: list | None = None):
        if target_modules is None or len(target_modules) == 0:
            logger.warning("No %v is specified, so skip to apply", target_modules)
            return

        if "all" in target_modules:
            target_modules = SUPPORTED_LORA_TARGET_MODULES

        logger.info("Applying dummy LoRA buffers to modules: %v", target_modules)

        model_runner = self.scheduler_info["scheduler"].tp_worker.worker.model_runner
        model_state = nnx.split(model_runner.model)[1]
        new_state = traverse_and_update(model_state, target_modules)
        self.scheduler_info["scheduler"].tp_worker.worker.model_runner.model_state_leaves, _ = (
            jax.tree_util.tree_flatten(new_state)
        )

    def encode(
        self,
        prompt: str | list[str] | list[dict] | list[list[dict]],
    ) -> dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret

    async def async_encode(
        self,
        prompt: str | list[str] | list[dict] | list[list[dict]],
    ) -> dict:
        """
        Asynchronous version of encode method.

        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        return await generator.__anext__()

    def rerank(
        self,
        prompt: list[list[str]],
    ) -> dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, is_cross_encoder_request=True)
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret

    def shutdown(self):
        """Shutdown the engine"""
        with contextlib.suppress(ValueError, RuntimeError):
            logger.debug("Shutting down engine (pid=%d)...", os.getpid())

        kill_process_tree(os.getpid(), include_parent=False)

        if (
            hasattr(self, "server_args")
            and self.server_args.enable_single_process
            and hasattr(self, "send_to_rpc")
        ):
            self.send_to_rpc.close()

        with contextlib.suppress(ValueError, RuntimeError):
            logger.debug("Engine shutdown complete.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    async def async_flush_cache(self):
        """
        Descriptioin: requests will be sent to tokenizer manager. It will flush all cache: tree_cache, req_to_token_pool, token_to_kv_pool_allocator(free physical cache through allocator)
        """
        return await self.tokenizer_manager.flush_cache()

    async def async_pause_generation(self, mode: str = "retract"):
        """
        Input: the pause generation mode: ["abort", "retract", "in-place"]

        Description: Deal with requests according to mode. Now support abort, in_place and retract.
        """
        obj = PauseGenerationReqInput(mode=mode)
        return await self.tokenizer_manager.pause_generation(obj)

    async def async_continue_generation(self):
        """
        Description: continue previous paused generation
        """
        obj = ContinueGenerationReqInput()
        return await self.tokenizer_manager.continue_generation(obj)

    async def async_get_server_info(self):
        internal_states = await self.tokenizer_manager.get_internal_state()
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self.scheduler_info,
            "internal_states": internal_states,
            "version": __version__,
        }

    def flush_cache(self):
        """
        Descriptioin: requests will be sent to tokenizer manager. It will flush all cache: tree_cache, req_to_token_pool, token_to_kv_pool_allocator(free physical cache through allocator)
        """
        return self.loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def pause_generation(self, mode: str = "retract"):
        """
        Input: the pause generation mode: ["abort", "retract", "in-place"]

        Description: Deal with requests according to mode. Now support abort, in_place and retract.
        """
        obj = PauseGenerationReqInput(mode=mode)
        return self.loop.run_until_complete(self.tokenizer_manager.pause_generation(obj))

    def continue_generation(self):
        """
        Description: continue previous paused generation
        """
        obj = ContinueGenerationReqInput()
        return self.loop.run_until_complete(self.tokenizer_manager.continue_generation(obj))

    # abort request is sync, therefore do not need event loop
    def abort_request(self, rid: str | None = None, abort_all: bool = False):
        """
        Description: Abort a request.

        Input: rid is request id, abort_all determines whether abort all requests
        """
        self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)

    def start_profile(self):
        self.loop.run_until_complete(self.tokenizer_manager.start_profile())

    def stop_profile(self):
        self.loop.run_until_complete(self.tokenizer_manager.stop_profile())

    def get_server_info(self):
        internal_states = self.loop.run_until_complete(self.tokenizer_manager.get_internal_state())
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self.scheduler_info,
            "internal_states": internal_states,
            "version": __version__,
        }

    def release_memory_occupation(self, tags: list[str] | None = None):
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return self.loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)
        )

    def resume_memory_occupation(self, tags: list[str] | None = None):
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        return self.loop.run_until_complete(
            self.tokenizer_manager.resume_memory_occupation(obj, None)
        )

    def score(
        self,
        query: str | list[int] | None = None,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> list[list[float]]:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair. For example:
        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223] # Token IDs for "Yes" and "No"
        item_first = False

        This would pass the following prompts to the model:
        "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        "<|user|>Is the following city the capital of France? London <|assistant|>"
        "<|user|>Is the following city the capital of France? Berlin <|assistant|>"
        The api would then return the probabilities of the model producing "Yes" and "No" as the next token.
        The output would look like:
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]


        Args:
            query: The query text or pre-tokenized query token IDs. Must be provided.
            items: The item text(s) or pre-tokenized item token IDs. Must be provided.
            label_token_ids: List of token IDs to compute probabilities for. If None, no token probabilities will be computed.
            apply_softmax: Whether to normalize probabilities using softmax.
            item_first: If True, prepend items to query. Otherwise append items to query.

        Returns:
            List of dictionaries mapping token IDs to their probabilities for each item.
            Each dictionary in the list corresponds to one item input.

        Raises:
            ValueError: If query is not provided, or if items is not provided,
                      or if token IDs are out of vocabulary, or if logprobs are not available for the specified tokens.
        """
        return self.loop.run_until_complete(
            self.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                request=None,
            )
        )

    def prefill_scoring_cache(self, query: str | list[int] | None = None) -> str:
        """Prefill a query once and return a reusable scoring cache handle."""
        return self.loop.run_until_complete(
            self.tokenizer_manager.prefill_scoring_cache(query=query)
        )

    async def async_prefill_scoring_cache(
        self,
        query: str | list[int] | None = None,
    ) -> str:
        """Asynchronous version of prefill_scoring_cache()."""
        return await self.tokenizer_manager.prefill_scoring_cache(query=query)

    def score_from_cache(
        self,
        cache_handle: str,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        """Score items against a previously prefetched query cache handle."""
        return self.loop.run_until_complete(
            self.tokenizer_manager.score_from_cache(
                cache_handle=cache_handle,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
            )
        )

    async def async_score_from_cache(
        self,
        cache_handle: str,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        """Asynchronous version of score_from_cache()."""
        return await self.tokenizer_manager.score_from_cache(
            cache_handle=cache_handle,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
        )

    def release_scoring_cache(self, cache_handle: str) -> bool:
        """Release a reusable scoring cache handle."""
        return self.loop.run_until_complete(
            self.tokenizer_manager.release_scoring_cache(cache_handle)
        )

    async def async_release_scoring_cache(self, cache_handle: str) -> bool:
        """Asynchronous version of release_scoring_cache()."""
        return await self.tokenizer_manager.release_scoring_cache(cache_handle)

    async def async_score(
        self,
        query: str | list[int] | None = None,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> list[list[float]]:
        """
        Asynchronous version of score method.

        See score() for detailed documentation.
        """
        return await self.tokenizer_manager.score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            request=None,
        )

    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:
            config = get_generation_config(
                self.server_args.model_path,
                self.server_args.trust_remote_code,
                self.server_args.revision,
            )
            if config is not None:
                self.default_sampling_params = config.to_diff_dict()
            else:
                self.default_sampling_params = {}

            if self.server_args.preferred_sampling_params is not None:
                self.default_sampling_params.update(
                    json.loads(self.server_args.preferred_sampling_params)
                )

            available_params = [
                "repetition_penalty",
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "max_new_tokens",
            ]
            if any(p in self.default_sampling_params for p in available_params):
                diff_sampling_param = {
                    p: self.default_sampling_params.get(p)
                    for p in available_params
                    if self.default_sampling_params.get(p) is not None
                }
                self.default_sampling_params = diff_sampling_param

            else:
                self.default_sampling_params = {}

        if self.default_sampling_params:
            return SamplingParams(**self.default_sampling_params)
        return SamplingParams()


def _set_envs_and_config(server_args):
    # Set ulimit
    set_ulimit()

    def sigchld_handler(signum, frame):
        try:
            pid, exitcode = os.waitpid(0, os.WNOHANG)
        except ChildProcessError:
            return  # child process could already be reaped, ignore if no child process exists
        if exitcode != 0:
            logger.warning(
                "Child process unexpectedly failed with exitcode=%s. pid=%s",
                exitcode,
                pid,
            )
            logger.warning("Child process pid=%s frame=%s", pid, frame)

    signal.signal(signal.SIGCHLD, sigchld_handler)

    # Register the signal handler.
    # The child processes will send SIGQUIT to this process when any error happens
    # This process then clean up the whole process tree
    def sigquit_handler(signum, frame):
        logger.error("Received sigquit from a child process. It usually means the child failed.")
        kill_process_tree(os.getpid())

    signal.signal(signal.SIGQUIT, sigquit_handler)
    if not server_args.enable_single_process:
        # Set mp start method
        mp.set_start_method("spawn", force=True)
    else:
        ## close resource tracker process
        from multiprocessing import resource_tracker

        resource_tracker._resource_tracker._fd = -1


def _launch_subprocesses(
    server_args, port_args: PortArgs | None = None
) -> tuple[TokenizerManager, TemplateManager, dict]:
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info("server_args=%s", server_args)

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    scheduler_pipe_readers = []
    scheduler_launch_plan = _build_scheduler_launch_plan(server_args, port_args)
    for lane_server_args, lane_port_args, dp_rank in scheduler_launch_plan:
        reader, writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=run_scheduler_process,
            args=(
                lane_server_args,
                lane_port_args,
                dp_rank,
                writer,
            ),
        )
        # with memory_saver_adapter.configure_subprocess():
        proc.start()
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                "Scheduler or DataParallelController %s terminated with %s",
                proc.pid,
                proc.exitcode,
            )
        return None, None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    tokenizer_port_args = [lane_port_args for _, lane_port_args, _ in scheduler_launch_plan]
    tokenizer_manager = TokenizerManager(server_args, tokenizer_port_args)
    tokenizer_manager.scheduler_pids = [
        proc.pid for proc in scheduler_procs if getattr(proc, "pid", None) is not None
    ]

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        model_path=server_args.model_path,
    )

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                "Scheduler lane %s is dead. Please check if there are relevant logs.",
                i,
            )
            scheduler_procs[i].join()
            logger.error("Exit code: %s", scheduler_procs[i].exitcode)
            raise

        if data["status"] != "ready":
            raise RuntimeError("Initialization failed. Please see the error messages above.")
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    if tokenizer_manager.scheduler_pids:
        scheduler_info["scheduler_pids"] = list(tokenizer_manager.scheduler_pids)
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_info


def _launch_threads(
    server_args, port_args: PortArgs | None = None
) -> tuple[TokenizerManager, TemplateManager, dict]:
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info("server_args=%s", server_args)
    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_infos = []
    scheduler_pipe_readers = []
    scheduler_launch_plan = _build_scheduler_launch_plan(server_args, port_args)
    for lane_server_args, lane_port_args, dp_rank in scheduler_launch_plan:
        scheduler_info = run_scheduler_loop_thread_after_create(
            lane_server_args,
            lane_port_args,
            dp_rank=dp_rank,
        )
        scheduler_infos.append(scheduler_info)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for scheduler_info in scheduler_infos:
            scheduler_thread = scheduler_info.get("scheduler_thread")
            if scheduler_thread is None:
                continue
            scheduler_thread.join()
            logger.error("Scheduler or DataParallelController %s terminated", scheduler_thread.name)
        return None, None, None

    # Launch detokenizer thread
    detoken_thread = threading.Thread(
        target=run_detokenizer_thread,
        args=(
            server_args,
            port_args,
        ),
        daemon=True,
    )
    detoken_thread.start()

    # Launch tokenizer process
    tokenizer_port_args = [lane_port_args for _, lane_port_args, _ in scheduler_launch_plan]
    tokenizer_manager = TokenizerManager(server_args, tokenizer_port_args)
    tokenizer_manager.scheduler_pids = []

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        model_path=server_args.model_path,
    )

    # Wait for the model to finish loading
    for i in range(len(scheduler_infos)):
        if scheduler_infos[i]["status"] != "ready":
            raise RuntimeError("Initialization failed. Please see the error messages above.")

    # Assume all schedulers have the same scheduler_info
    assert len(scheduler_infos) > 0, "scheduler_infos is empty"
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    if len(scheduler_infos) == 1:
        scheduler = scheduler_info.get("scheduler")
        if scheduler is not None and hasattr(scheduler, "submit_local_rpc"):
            tokenizer_manager.local_rpc_submitter = scheduler.submit_local_rpc
        if scheduler is not None and hasattr(scheduler, "submit_local_request"):
            tokenizer_manager.local_request_submitter = scheduler.submit_local_request
    return tokenizer_manager, template_manager, scheduler_info


def _launch_subprocesses_or_threads(
    server_args, port_args: PortArgs | None = None
) -> tuple[TokenizerManager, TemplateManager, dict]:
    if server_args.enable_single_process:
        return _launch_threads(server_args, port_args)
    else:
        return _launch_subprocesses(server_args, port_args)
