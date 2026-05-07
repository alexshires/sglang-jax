import asyncio
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqOutput
from sgl_jax.srt.managers.tokenizer_score_cache_mixin import TokenizerScoreCacheMixin
from sgl_jax.srt.validation import ValidationError


class _FakeTokenizer:
    def __len__(self):
        return 100

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) % 50 + 1 for char in text]


class _DummyScoreCacheManager(TokenizerScoreCacheMixin):
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.server_args = SimpleNamespace(
            multi_item_score_from_cache_v2_items_per_step=16,
            multi_item_score_from_cache_v2_adaptive_chunk_by_token_budget=False,
            multi_item_score_from_cache_v2_token_budget=0,
            multi_item_score_from_cache_v2_min_items_per_step=1,
        )
        self.generated_requests = []
        self.fastpath_calls = []
        self.released_handles = []

    async def generate_request(self, req):
        self.generated_requests.append(req)
        if False:
            yield {}

    async def _score_from_cache_fastpath_v2(self, **kwargs):
        self.fastpath_calls.append(kwargs)
        return ScoreFromCacheReqOutput(
            success=True,
            scores=[[0.2, 0.8], [0.3, 0.7]],
        )

    async def _release_cache(self, cache_handle):
        self.released_handles.append(cache_handle)
        return True


def test_prefill_scoring_cache_builds_prefill_only_request():
    manager = _DummyScoreCacheManager()

    cache_handle = asyncio.run(manager.prefill_scoring_cache("hi"))

    assert len(cache_handle) == 32
    assert len(manager.generated_requests) == 1
    req = manager.generated_requests[0]
    assert req.input_ids == manager.tokenizer.encode("hi", add_special_tokens=False)
    assert req.sampling_params == {"max_new_tokens": 0}
    assert req.return_logprob is False
    assert req.cache_for_scoring is True
    assert req.is_single is True
    assert req.rid == cache_handle


def test_score_from_cache_uses_fastpath_with_normalized_items():
    manager = _DummyScoreCacheManager()

    scores = asyncio.run(
        manager.score_from_cache(
            "cache-1",
            items=["a", "bc"],
            label_token_ids=[1, 2],
            apply_softmax=True,
        )
    )

    assert scores == [[0.2, 0.8], [0.3, 0.7]]
    assert len(manager.fastpath_calls) == 1
    call = manager.fastpath_calls[0]
    assert call["cache_handle"] == "cache-1"
    assert call["items"] == [
        manager.tokenizer.encode("a", add_special_tokens=False),
        manager.tokenizer.encode("bc", add_special_tokens=False),
    ]
    assert call["label_token_ids"] == [1, 2]
    assert call["apply_softmax"] is True
    assert call["items_per_step"] == 16


def test_score_from_cache_rejects_empty_cache_handle():
    manager = _DummyScoreCacheManager()

    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(manager.score_from_cache("", items=[[1]], label_token_ids=[1]))

    assert exc_info.value.param == "cache_handle"
    assert exc_info.value.code == "invalid_cache_handle"


def test_release_scoring_cache_validates_then_delegates():
    manager = _DummyScoreCacheManager()

    assert asyncio.run(manager.release_scoring_cache("cache-1")) is True
    assert manager.released_handles == ["cache-1"]

    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(manager.release_scoring_cache(""))
    assert exc_info.value.param == "cache_handle"
    assert exc_info.value.code == "invalid_cache_handle"
