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
        self.scoring_cache_nodes = {}
        self.scoring_cache_timeout = float(server_args.multi_item_prefill_extend_cache_timeout)
        self._last_scoring_cache_gc = 0.0
        self.scoring_cache_prefix_handles_by_key = {}
        self.scoring_cache_handle_to_prefix_key = {}
        self.scoring_cache_handles_created = 0
        self.scoring_cache_handles_released = 0
        self.scoring_cache_handles_released_manual = 0
        self.scoring_cache_handles_released_expired = 0
        self.scoring_cache_handles_released_other = 0
        self.scoring_cache_handles_missing_node = 0
        self.scoring_cache_release_failures = 0
        self.scoring_cache_lookup_queries = 0
        self.scoring_cache_lookup_hits = 0
        self.scoring_cache_lookup_misses = 0
        self.scoring_cache_lookup_by_path = {
            "extend": {"queries": 0, "hits": 0, "misses": 0},
            "score_from_cache_v2": {"queries": 0, "hits": 0, "misses": 0},
            "cache_for_scoring": {"queries": 0, "hits": 0, "misses": 0},
        }
        self.scoring_cache_lookup_by_lane = {
            path: {
                "default": {"queries": 0, "hits": 0, "misses": 0},
                "short": {"queries": 0, "hits": 0, "misses": 0},
                "long": {"queries": 0, "hits": 0, "misses": 0},
            }
            for path in self.scoring_cache_lookup_by_path
        }
        self._warned_scoring_cache_lanes = set()
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
        self.ingress_score_paths = {}
        self.ingress_score_path_frames = {}
        self.score_from_cache_v2_attempted = 0
        self.score_from_cache_v2_succeeded = 0
        self.score_from_cache_v2_fallback = 0
        self.score_from_cache_v2_fallback_reasons = {}
        self.score_scheduler_topology_name = ""

    def add_scoring_internal_state(self, ret: dict) -> None:
        ret["score_from_cache_v2_metrics"] = {
            "attempted": self.score_from_cache_v2_attempted,
            "succeeded": self.score_from_cache_v2_succeeded,
            "fallback": self.score_from_cache_v2_fallback,
            "fallback_reasons": dict(self.score_from_cache_v2_fallback_reasons),
        }

    @staticmethod
    def _can_skip_sample_for_prefill_batch(batch: ScheduleBatch | None) -> bool:
        return False

    @staticmethod
    def _lane_counter(req_owner, attr_name: str) -> dict[str, int]:
        counter = getattr(req_owner, attr_name, None)
        if not isinstance(counter, dict):
            counter = {"default": 0, "short": 0, "long": 0}
            setattr(req_owner, attr_name, counter)
        return counter

    @staticmethod
    def _running_lane_counts(req_owner) -> dict[str, int]:
        return {"default": 0, "short": 0, "long": 0}

    @staticmethod
    def _waiting_lane_counts(req_owner, waiting_queue: list[Req]) -> dict[str, int]:
        return {"default": len(waiting_queue), "short": 0, "long": 0}

    @staticmethod
    def _iter_waiting_queue(req_owner, waiting_queue: list[Req]) -> list[Req]:
        return list(waiting_queue)

    @staticmethod
    def _admission_lane(req_owner, req: Req) -> str:
        return "default"

    @staticmethod
    def _lane_cap(req_owner, lane: str) -> int:
        return 0

    @staticmethod
    def _normalize_scoring_cache_prefix_key(input_ids, extra_key):
        return None

    def _score_scheduler_lane_from_prefix_len(self, prefix_len: int) -> str:
        return "default"

    def recv_requests(self) -> list[Req]:
        recv_reqs = []
        if self.node_rank == 0:
            while True:
                try:
                    recv_reqs.append(self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK))
                except zmq.ZMQError:
                    break
            while True:
                try:
                    recv_reqs.append(self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK))
                except zmq.ZMQError:
                    break
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
        for recv_req in recv_reqs:
            output = self._request_dispatcher(recv_req)
            if output is not None:
                if self._comm_backend is not None:
                    self._comm_backend.send_pyobj(output)
                else:
                    self.send_to_tokenizer.send_pyobj(output)
