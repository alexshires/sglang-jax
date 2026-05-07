import asyncio
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.tokenizer_score_api_mixin import TokenizerScoreApiMixin
from sgl_jax.srt.validation import ValidationError


class _FakeTokenizer:
    def __len__(self):
        return 100


class _DummyScoreApi(TokenizerScoreApiMixin):
    def __init__(self, max_multi_item_count: int = 512):
        self.server_args = SimpleNamespace(max_multi_item_count=max_multi_item_count)
        self.tokenizer = _FakeTokenizer()
        self.last_batch_request = None

    def generate_request(self, batch_request, request=None):
        self.last_batch_request = batch_request

        async def _gen():
            yield [
                {
                    "meta_info": {
                        "id": "score-test",
                        "output_token_ids_logprobs": [[(0.0, 1, "label")]],
                    }
                }
            ]

        return _gen()


def test_score_request_raises_structured_error_for_too_many_items():
    manager = _DummyScoreApi(max_multi_item_count=1)

    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(manager.score_request("q", ["a", "b"], [1]))

    error = exc_info.value
    assert error.message == "Too many items for scoring: 2 > 1"
    assert error.error_type == "invalid_value_error"
    assert error.param == "items"
    assert error.code == "too_many_items"


def test_score_request_preserves_one_token_score_probe():
    manager = _DummyScoreApi()

    scores = asyncio.run(manager.score_request("q", ["a"], [1]))

    assert scores == [[1.0]]
    assert manager.last_batch_request.sampling_params["max_new_tokens"] == 1
