# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import unittest
from types import SimpleNamespace

import pytest
import zmq

from sgl_jax.srt.managers.io_struct import (
    ReleaseScoringCacheReqInput,
    ScoreFromCacheReqInput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.managers.tokenizer_manager import ReqState, TokenizerManager


class _NoopSender:

    def __init__(self):
        self.calls = []

    def send_pyobj(self, obj):
        self.calls.append(obj)


class _FakeSchedulerLivenessManager:
    _send_batch_requests = TokenizerManager._send_batch_requests

    def __init__(self):
        self.wait_timeout = 0.01
        self.scheduler_pids = [4321]
        self.scheduler_unavailable_error = None
        self.health_check_failed = False
        self.rid_to_state = {}
        self.send_to_scheduler = _NoopSender()
        self.log_requests = False

    def _raise_if_scheduler_unavailable(self):
        if self.scheduler_unavailable_error:
            raise ValueError(self.scheduler_unavailable_error)

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True




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


class _FakeSchedulerIngress:
    recv_requests = Scheduler.recv_requests

    def __init__(self, tokenizer_payloads: list, rpc_payloads: list):
        self.node_rank = 0
        self.nnodes = 1
        self.recv_from_tokenizer = _FakeIngressSocket(tokenizer_payloads)
        self.recv_from_rpc = _FakeIngressSocket(rpc_payloads)
        self.ingress_recv_calls = 0
        self.ingress_nonempty_calls = 0
        self.ingress_max_batch_size = 0
        self.ingress_tokenizer_frames = 0
        self.ingress_rpc_frames = 0
        self.ingress_tokenizer_messages = 0
        self.ingress_rpc_messages = 0
        self.ingress_batch_size_histogram = {"2_to_4": 0}
        self.ingress_score_paths = {
            "tokenizer_multi_item_packed": 0,
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }
        self.ingress_score_path_frames = {
            "tokenizer_cache_for_scoring": 0,
            "tokenizer_extend_from_cache": 0,
            "rpc_score_from_cache_v2": 0,
            "rpc_release_scoring_cache": 0,
        }


class TestTokenizerManagerBatchSend(unittest.TestCase):

    def test_send_batch_requests_sends_single_payload_and_tracks_all_states(self):
        manager = _FakeSchedulerLivenessManager()
        manager.scheduler_pids = []
        reqs = [SimpleNamespace(rid="rid-a"), SimpleNamespace(rid="rid-b")]
        tokenized_objs = [SimpleNamespace(tok=1), SimpleNamespace(tok=2)]

        states = manager._send_batch_requests(reqs, tokenized_objs, created_time=1.0)

        assert len(states) == 2
        assert set(manager.rid_to_state.keys()) == {"rid-a", "rid-b"}
        assert len(manager.send_to_scheduler.calls) == 1
        assert manager.send_to_scheduler.calls[0] == tokenized_objs

    def test_send_batch_requests_raises_on_length_mismatch(self):
        manager = _FakeSchedulerLivenessManager()
        manager.scheduler_pids = []
        with pytest.raises(ValueError, match="same length"):
            manager._send_batch_requests([SimpleNamespace(rid="rid-a")], [], created_time=0.0)

    def test_handle_batch_request_uses_single_send_when_batch_send_enabled(self):
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

    def test_handle_batch_request_uses_per_request_send_when_batch_send_disabled(self):
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

    def test_handle_batch_request_uses_single_send_without_batch_encode(self):
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

    def test_scheduler_recv_requests_unpacks_list_payload_into_logical_batch(self):
        tokenizer_payload = [
            TokenizedGenerateReqInput(
                rid="tok-1",
                input_ids=[1, 2],
                sampling_params={},
                cache_for_scoring=True,
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
        assert scheduler.ingress_score_paths["tokenizer_cache_for_scoring"] == 1
        assert scheduler.ingress_score_paths["tokenizer_extend_from_cache"] == 1


if __name__ == "__main__":
    unittest.main()
