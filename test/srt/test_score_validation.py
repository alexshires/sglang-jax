"""
Tests for Score API validation module.

These tests validate the input validation logic for the Score API,
ensuring proper error messages and codes for invalid requests.

Design Document: sglang-jax-dev-scripts/rfcs/006-error-handling-api-contract.md

Usage:
    python -m pytest test/srt/test_score_validation.py -v
"""

import pytest

from sgl_jax.srt.validation import (
    ValidationError,
    validate_score_request,
)


class TestValidationError:
    def test_basic_creation(self):
        error = ValidationError(
            message="test error",
            error_type="invalid_request_error",
            param="query",
            code="test_code",
        )
        assert error.message == "test error"
        assert error.error_type == "invalid_request_error"
        assert error.param == "query"
        assert error.code == "test_code"
        assert str(error) == "test error"

    def test_to_dict_full(self):
        error = ValidationError(
            message="test error",
            error_type="invalid_request_error",
            param="query",
            code="test_code",
        )
        result = error.to_dict()
        assert result == {
            "error": {
                "message": "test error",
                "type": "invalid_request_error",
                "param": "query",
                "code": "test_code",
            }
        }

    def test_to_dict_minimal(self):
        error = ValidationError(
            message="test error",
            error_type="invalid_request_error",
        )
        result = error.to_dict()
        assert result == {
            "error": {
                "message": "test error",
                "type": "invalid_request_error",
            }
        }

    def test_http_status_400(self):
        error = ValidationError(
            message="test",
            error_type="invalid_request_error",
            code="empty_query",
        )
        assert error.get_http_status() == 400

    def test_http_status_422_vocab(self):
        error = ValidationError(
            message="test",
            error_type="invalid_value_error",
            code="token_id_exceeds_vocab",
        )
        assert error.get_http_status() == 422

    def test_http_status_422_negative(self):
        error = ValidationError(
            message="test",
            error_type="invalid_value_error",
            code="token_id_negative",
        )
        assert error.get_http_status() == 422


class TestValidateScoreRequest:
    # Query validation tests

    def test_query_missing(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=None,
                items=["test"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "missing_query"
        assert exc_info.value.param == "query"

    def test_query_empty_string(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="",
                items=["test"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_query"

    def test_query_empty_list(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[],
                items=[[1, 2]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_query"

    def test_query_invalid_type(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=123,
                items=["test"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_query_type"

    def test_query_list_with_non_integers(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[1, "two", 3],
                items=[[1, 2]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_token_id_type"

    # Items validation tests

    def test_items_missing(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=None,
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "missing_items"

    def test_items_not_list(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items="not a list",
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_items_type"

    def test_items_empty_list(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=[],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_items"

    def test_items_mixed_types_with_query(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="text query",
                items=[[1, 2, 3]],  # Token mode items with text query
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "mixed_input_types"

    def test_items_inconsistent_types_text_mode(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["valid", 123, "also valid"],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_items_type"

    def test_items_inconsistent_types_token_mode(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[1, 2, 3],
                items=[[1, 2], "not a list", [3, 4]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_items_type"

    def test_items_token_mode_non_integer(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[1, 2, 3],
                items=[[1, 2], [3, "four"]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "invalid_token_id_type"

    def test_items_token_mode_empty_item(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query=[1, 2, 3],
                items=[[1, 2], [], [3]],
                label_token_ids=[1, 2],
            )
        assert exc_info.value.code == "empty_item"
        assert exc_info.value.param == "items"

    def test_label_token_ids_missing(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=None,
            )
        assert exc_info.value.code == "missing_label_token_ids"

    def test_label_token_ids_not_list(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=123,
            )
        assert exc_info.value.code == "invalid_label_token_ids_type"

    def test_label_token_ids_empty(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[],
            )
        assert exc_info.value.code == "empty_label_token_ids"

    def test_label_token_ids_non_integer(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, "two", 3],
            )
        assert exc_info.value.code == "invalid_token_id_type"

    def test_label_token_ids_negative(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, -5, 3],
            )
        assert exc_info.value.code == "token_id_negative"
        assert exc_info.value.get_http_status() == 422

    def test_label_token_ids_exceeds_vocab(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 50000, 3],
                vocab_size=32000,
            )
        assert exc_info.value.code == "token_id_exceeds_vocab"
        assert exc_info.value.get_http_status() == 422

    # Boolean parameter validation tests

    def test_apply_softmax_not_boolean(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                apply_softmax="true",
            )
        assert exc_info.value.code == "invalid_apply_softmax_type"

    def test_apply_softmax_int_not_boolean(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                apply_softmax=1,
            )
        assert exc_info.value.code == "invalid_apply_softmax_type"

    def test_item_first_not_boolean(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                item_first=1,
            )
        assert exc_info.value.code == "invalid_item_first_type"

    def test_item_first_string_not_boolean(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_score_request(
                query="test",
                items=["item"],
                label_token_ids=[1, 2],
                item_first="true",
            )
        assert exc_info.value.code == "invalid_item_first_type"

    # Valid request tests

    def test_valid_text_mode_request(self):
        # Should not raise
        validate_score_request(
            query="What is the answer?",
            items=[" yes", " no", " maybe"],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
            item_first=False,
        )

    def test_valid_token_mode_request(self):
        # Should not raise
        validate_score_request(
            query=[1, 2, 3, 4],
            items=[[5, 6], [7, 8], [9, 10]],
            label_token_ids=[100, 200, 300],
            apply_softmax=False,
            item_first=True,
        )

    def test_valid_request_with_vocab_size(self):
        # Should not raise
        validate_score_request(
            query="test",
            items=["item"],
            label_token_ids=[100, 200, 300],
            vocab_size=32000,
        )

    def test_valid_single_item(self):
        # Should not raise
        validate_score_request(
            query="test",
            items=["single item"],
            label_token_ids=[1],
        )

    def test_valid_empty_string_item(self):
        # Should not raise - empty items are valid for scoring candidates
        validate_score_request(
            query="The answer is",
            items=[""],  # Empty item is valid
            label_token_ids=[1, 2],
        )
