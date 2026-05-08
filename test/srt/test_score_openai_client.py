"""OpenAI-client compatibility tests for `/v1/score`."""

import os

import jax
import pytest
import requests

from sgl_jax.test.score_test_utils import (
    get_label_token_ids,
    get_tokenizer,
)
from sgl_jax.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)

pytest.importorskip("openai", reason="openai package required for these tests")
from openai import APIStatusError, OpenAI  # noqa: E402

TEST_MODEL_NAME = os.getenv("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)


def _skip_if_no_tpu() -> None:
    if not any(device.platform == "tpu" for device in jax.devices()):
        pytest.skip("Score OpenAI-client tests require TPU.")


@pytest.fixture(scope="module")
def tokenizer(score_server_url):
    return get_tokenizer(TEST_MODEL_NAME)


@pytest.fixture(scope="module")
def score_server_url():
    _skip_if_no_tpu()

    try:
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate", timeout=1)
        if response.status_code == 200:
            yield DEFAULT_URL_FOR_TEST
            return
    except requests.RequestException:
        pass

    server_process = popen_launch_server(
        model=TEST_MODEL_NAME,
        base_url=DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=[
            "--mem-fraction-static",
            "0.7",
        ],
        env={"JAX_PLATFORMS": "tpu"},
        device="tpu",
        check_cache_miss=False,
    )
    try:
        yield DEFAULT_URL_FOR_TEST
    finally:
        kill_process_tree(server_process.pid)


@pytest.fixture()
def score_client(score_server_url):
    return OpenAI(
        base_url=f"{score_server_url}/v1",
        api_key="test-key",
        timeout=600.0,
    )


def _label_ids(tokenizer, labels: list[str]) -> list[int]:
    return get_label_token_ids(tokenizer, labels)


def _assert_scores_shape(response, expected_items: int, expected_labels: int) -> None:
    assert response["object"] == "scoring"
    assert response["model"] == TEST_MODEL_NAME
    assert len(response["scores"]) == expected_items
    for row in response["scores"]:
        assert len(row) == expected_labels


def _assert_softmax_rows(response) -> None:
    for row in response["scores"]:
        assert all(0.0 <= score <= 1.0 for score in row)
        assert abs(sum(row) - 1.0) <= 1e-5


def _assert_status_error(exc_info, expected_status: int, expected_param: str) -> None:
    error = exc_info.value
    assert error.status_code == expected_status
    body = error.response.json()
    assert body["object"] == "error"
    assert body["code"] == expected_status
    assert body["param"] == expected_param


class TestOpenAIClientVersion:
    def test_openai_client_version_minimum(self):
        import openai

        major = int(openai.__version__.split(".")[0])
        assert major >= 1, f"Requires openai>=1.0.0, got {openai.__version__}"

    def test_openai_client_has_post_method(self):
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
        assert callable(client.post)


class TestScoreOpenAIClient:
    def test_score_with_openai_client_post(self, score_client, tokenizer):
        response = score_client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "The capital of France is",
                "items": [" Paris", " London", " Berlin"],
                "label_token_ids": _label_ids(tokenizer, [" A", " B", " C"]),
                "apply_softmax": True,
                "item_first": False,
            },
            cast_to=object,
        )

        _assert_scores_shape(response, expected_items=3, expected_labels=3)
        _assert_softmax_rows(response)

    def test_score_without_softmax_openai_client(self, score_client, tokenizer):
        response = score_client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Test query",
                "items": [" option1", " option2"],
                "label_token_ids": _label_ids(tokenizer, [" A", " B"]),
                "apply_softmax": False,
            },
            cast_to=object,
        )

        _assert_scores_shape(response, expected_items=2, expected_labels=2)
        for row in response["scores"]:
            assert all(isinstance(score, (int, float)) for score in row)

    def test_score_response_contains_usage(self, score_client, tokenizer):
        response = score_client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Test",
                "items": [" A"],
                "label_token_ids": _label_ids(tokenizer, [" A"]),
                "apply_softmax": True,
            },
            cast_to=object,
        )

        assert "usage" in response

    def test_score_with_item_first_flag(self, score_client, tokenizer):
        response = score_client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": " continues the story",
                "items": ["Once upon a time", "In a galaxy far away"],
                "label_token_ids": _label_ids(tokenizer, [" A", " B"]),
                "apply_softmax": True,
                "item_first": True,
            },
            cast_to=object,
        )

        _assert_scores_shape(response, expected_items=2, expected_labels=2)
        _assert_softmax_rows(response)

    def test_large_batch_openai_client(self, score_client, tokenizer):
        items = [f" item{idx}" for idx in range(20)]
        response = score_client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Score these items:",
                "items": items,
                "label_token_ids": _label_ids(tokenizer, [" A", " B"]),
                "apply_softmax": True,
            },
            cast_to=object,
        )

        _assert_scores_shape(response, expected_items=20, expected_labels=2)
        _assert_softmax_rows(response)

    def test_unicode_content_openai_client(self, score_client, tokenizer):
        response = score_client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "Translate: こんにちは",
                "items": [" Hello", " Goodbye", " Thanks"],
                "label_token_ids": _label_ids(tokenizer, [" A", " B", " C"]),
                "apply_softmax": True,
            },
            cast_to=object,
        )

        _assert_scores_shape(response, expected_items=3, expected_labels=3)
        _assert_softmax_rows(response)

    def test_empty_string_item_openai_client(self, score_client, tokenizer):
        response = score_client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": "The answer is",
                "items": [""],
                "label_token_ids": _label_ids(tokenizer, [" A", " B"]),
                "apply_softmax": True,
            },
            cast_to=object,
        )

        _assert_scores_shape(response, expected_items=1, expected_labels=2)
        _assert_softmax_rows(response)

    def test_token_input_mode_openai_client(self, score_client, tokenizer):
        response = score_client.post(
            "/score",
            body={
                "model": TEST_MODEL_NAME,
                "query": tokenizer.encode("The answer is", add_special_tokens=False),
                "items": [
                    tokenizer.encode(" yes", add_special_tokens=False),
                    tokenizer.encode(" no", add_special_tokens=False),
                ],
                "label_token_ids": _label_ids(tokenizer, [" A", " B"]),
                "apply_softmax": False,
            },
            cast_to=object,
        )

        _assert_scores_shape(response, expected_items=2, expected_labels=2)


class TestScoreOpenAIClientErrors:
    def test_empty_items_returns_bad_request(self, score_client, tokenizer):
        with pytest.raises(APIStatusError) as exc_info:
            score_client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "Test",
                    "items": [],
                    "label_token_ids": _label_ids(tokenizer, [" A"]),
                },
                cast_to=object,
            )

        _assert_status_error(exc_info, expected_status=400, expected_param="items")

    def test_empty_query_returns_bad_request(self, score_client, tokenizer):
        with pytest.raises(APIStatusError) as exc_info:
            score_client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "",
                    "items": [" test"],
                    "label_token_ids": _label_ids(tokenizer, [" A"]),
                },
                cast_to=object,
            )

        _assert_status_error(exc_info, expected_status=400, expected_param="query")

    def test_missing_required_field_returns_bad_request(self, score_client):
        with pytest.raises(APIStatusError) as exc_info:
            score_client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "Test",
                },
                cast_to=object,
            )

        _assert_status_error(exc_info, expected_status=400, expected_param="items")

    def test_invalid_type_returns_bad_request(self, score_client, tokenizer):
        with pytest.raises(APIStatusError) as exc_info:
            score_client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "Test",
                    "items": "not a list",
                    "label_token_ids": _label_ids(tokenizer, [" A"]),
                },
                cast_to=object,
            )

        _assert_status_error(exc_info, expected_status=400, expected_param="items")

    def test_negative_token_id_returns_unprocessable(self, score_client):
        with pytest.raises(APIStatusError) as exc_info:
            score_client.post(
                "/score",
                body={
                    "model": TEST_MODEL_NAME,
                    "query": "Test",
                    "items": [" test"],
                    "label_token_ids": [-1],
                },
                cast_to=object,
            )

        _assert_status_error(
            exc_info,
            expected_status=422,
            expected_param="label_token_ids",
        )
