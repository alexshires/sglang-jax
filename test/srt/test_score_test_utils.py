"""
Tests for Score API test utilities.

These tests validate the helper functions and assertions in score_test_utils,
ensuring they work correctly for Score API test development.

Design Document: sglang-jax-dev-scripts/rfcs/003-score-api-comprehensive-test-suite.md

Usage:
    python -m pytest test/srt/test_score_test_utils.py -v
"""

import unittest
import tempfile

import pytest

import sgl_jax.test.score_test_utils as score_utils
from sgl_jax.test.score_test_utils import (
    ScoreTestConfig,
    assert_scores_match,
    assert_scores_shape,
    assert_scores_valid,
    generate_test_items,
    get_label_token_ids,
    get_single_token_id,
    get_tokenizer,
    should_run_hf_reference,
)


class TestScoreTestConfig:
    """Tests for ScoreTestConfig dataclass."""

    def test_default_values(self, monkeypatch):
        """Test that default values are sensible."""
        monkeypatch.delenv("SGLANG_TEST_DEVICE", raising=False)
        monkeypatch.delenv("SGLANG_TEST_DOWNLOAD_DIR", raising=False)
        config = ScoreTestConfig()
        assert config.device == "tpu"
        assert config.tp_size == 1
        assert config.dtype == "bfloat16"
        assert config.download_dir == tempfile.gettempdir()
        assert config.tolerance == 0.01
        assert isinstance(config.precompile_bs_paddings, list)

    def test_environment_defaults(self, monkeypatch):
        """Test model, device, and download directory environment defaults."""
        monkeypatch.setenv("SGLANG_TEST_MODEL", "env/model")
        monkeypatch.setenv("SGLANG_TEST_DEVICE", "cpu")
        monkeypatch.setenv("SGLANG_TEST_DOWNLOAD_DIR", "/tmp/score-tests")

        config = ScoreTestConfig()

        assert config.model_name == "env/model"
        assert config.device == "cpu"
        assert config.download_dir == "/tmp/score-tests"

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = ScoreTestConfig(
            model_name="custom/model",
            device="cpu",
            tp_size=4,
            tolerance=0.05,
        )
        assert config.model_name == "custom/model"
        assert config.device == "cpu"
        assert config.tp_size == 4
        assert config.tolerance == 0.05


class _FixedTokenizer:
    def __init__(self, token_ids_by_text):
        self.token_ids_by_text = token_ids_by_text

    def encode(self, text, add_special_tokens=False):
        return list(self.token_ids_by_text[text])


class TestGetTokenizer:
    """Tests for get_tokenizer factory."""

    def test_uses_env_model_when_model_omitted(self, monkeypatch):
        calls = []

        def fake_from_pretrained(model_name, trust_remote_code):
            calls.append((model_name, trust_remote_code))
            return object()

        monkeypatch.setenv("SGLANG_TEST_MODEL", "env/model")
        monkeypatch.setattr(score_utils.AutoTokenizer, "from_pretrained", fake_from_pretrained)

        get_tokenizer()

        assert calls == [("env/model", True)]

    def test_accepts_config(self, monkeypatch):
        calls = []

        def fake_from_pretrained(model_name, trust_remote_code):
            calls.append((model_name, trust_remote_code))
            return object()

        monkeypatch.setattr(score_utils.AutoTokenizer, "from_pretrained", fake_from_pretrained)

        get_tokenizer(ScoreTestConfig(model_name="config/model"))

        assert calls == [("config/model", True)]


class TestGetSingleTokenId:
    """Tests for get_single_token_id function."""

    def test_raises_on_multi_token(self):
        """Test that multi-token text raises ValueError."""
        tokenizer = _FixedTokenizer({"hello world": [1, 2, 3]})

        with pytest.raises(ValueError) as exc_info:
            get_single_token_id(tokenizer, "hello world")
        assert "tokenizes to 3 tokens" in str(exc_info.value)

    def test_raises_on_empty(self):
        """Test that empty tokenization raises ValueError."""
        tokenizer = _FixedTokenizer({"": []})

        with pytest.raises(ValueError) as exc_info:
            get_single_token_id(tokenizer, "")
        assert "tokenizes to 0 tokens" in str(exc_info.value)

    def test_returns_single_token(self):
        """Test that single-token text returns the token ID."""
        tokenizer = _FixedTokenizer({" yes": [42]})
        result = get_single_token_id(tokenizer, " yes")
        assert result == 42


class TestGetLabelTokenIds:
    """Tests for get_label_token_ids function."""

    def test_multiple_tokens(self):
        """Test getting IDs for multiple tokens."""
        tokenizer = _FixedTokenizer({" a": [97], " b": [98], " c": [99]})
        result = get_label_token_ids(tokenizer, [" a", " b", " c"])
        assert result == [ord("a"), ord("b"), ord("c")]

    def test_raises_on_any_multi_token(self):
        """Test that any multi-token text raises ValueError."""
        tokenizer = _FixedTokenizer({" yes": [1], " multi": [1, 2], " no": [2]})

        with pytest.raises(ValueError):
            get_label_token_ids(tokenizer, [" yes", " multi", " no"])


class TestAssertScoresShape:
    """Tests for assert_scores_shape function."""

    def test_correct_shape(self):
        """Test that correct shape passes."""
        scores = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]
        # Should not raise
        assert_scores_shape(scores, expected_items=2, expected_labels=3)

    def test_wrong_items_count(self):
        """Test that wrong items count raises."""
        scores = [[0.5, 0.5]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_shape(scores, expected_items=2, expected_labels=2)
        assert "Expected 2 items" in str(exc_info.value)

    def test_wrong_labels_count(self):
        """Test that wrong labels count raises."""
        scores = [[0.5, 0.5], [0.3, 0.7]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_shape(scores, expected_items=2, expected_labels=3)
        assert "expected 3 labels" in str(exc_info.value)

    def test_with_test_case(self):
        """Test using TestCase for assertions."""
        tc = unittest.TestCase()
        tc.maxDiff = None

        scores = [[0.5, 0.5], [0.3, 0.7]]
        # Should not raise
        assert_scores_shape(scores, expected_items=2, expected_labels=2, test_case=tc)


class TestAssertScoresValid:
    """Tests for assert_scores_valid function."""

    def test_valid_probabilities(self):
        """Test that valid probabilities pass."""
        scores = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]
        # Should not raise
        assert_scores_valid(scores, apply_softmax=True)

    def test_probabilities_dont_sum_to_one(self):
        """Test that probabilities not summing to 1 raise."""
        scores = [[0.5, 0.3, 0.3]]  # Sums to 1.1
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_valid(scores, apply_softmax=True)
        assert "expected 1.0" in str(exc_info.value)

    def test_negative_probability(self):
        """Test that negative probability raises."""
        scores = [[0.5, -0.2, 0.7]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_valid(scores, apply_softmax=True)
        assert "negative" in str(exc_info.value)

    def test_probability_exceeds_one(self):
        """Test that probability > 1 raises."""
        scores = [[1.5, -0.3, -0.2]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_valid(scores, apply_softmax=True)
        assert "exceeds 1" in str(exc_info.value)

    def test_nan_score(self):
        """Test that NaN score raises."""
        scores = [[0.5, float("nan"), 0.5]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_valid(scores, apply_softmax=True)
        assert "not finite" in str(exc_info.value)

    def test_inf_score(self):
        """Test that Inf score raises."""
        scores = [[float("inf"), 0.0, 0.0]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_valid(scores, apply_softmax=True)
        assert "not finite" in str(exc_info.value)

    def test_valid_logprobs(self):
        """Test that valid logprobs pass (apply_softmax=False)."""
        scores = [[-1.5, -2.3, -0.5], [-10.0, -0.1, -5.0]]
        # Should not raise - logprobs are typically negative
        assert_scores_valid(scores, apply_softmax=False)

    def test_logprobs_allow_negative(self):
        """Test that negative values are allowed for logprobs."""
        scores = [[-100.0, -200.0, -50.0]]
        # Should not raise
        assert_scores_valid(scores, apply_softmax=False)

    def test_logprobs_check_finite(self):
        """Test that NaN is still caught for logprobs."""
        scores = [[-1.0, float("nan"), -2.0]]
        with pytest.raises(AssertionError):
            assert_scores_valid(scores, apply_softmax=False)

    def test_logprobs_reject_negative_inf(self):
        """Test that -inf is rejected for logprobs."""
        scores = [[-1.0, float("-inf"), -2.0]]
        with pytest.raises(AssertionError):
            assert_scores_valid(scores, apply_softmax=False)

    def test_tolerance_applies_with_test_case(self):
        """Test unittest assertions respect the tolerance argument."""
        tc = unittest.TestCase()

        assert_scores_valid(
            [[0.5002, 0.4999]],
            apply_softmax=True,
            test_case=tc,
            tolerance=1e-3,
        )
        with pytest.raises(AssertionError):
            assert_scores_valid(
                [[0.5002, 0.4999]],
                apply_softmax=True,
                test_case=tc,
                tolerance=1e-6,
            )


class TestAssertScoresMatch:
    """Tests for assert_scores_match function."""

    def test_matching_scores(self):
        """Test that matching scores pass."""
        expected = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]
        actual = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]
        # Should not raise
        assert_scores_match(expected, actual)

    def test_within_tolerance(self):
        """Test that scores within tolerance pass."""
        expected = [[0.50, 0.30, 0.20]]
        actual = [[0.505, 0.298, 0.197]]  # Within 1% tolerance
        # Should not raise
        assert_scores_match(expected, actual, tolerance=0.01)

    def test_exceeds_tolerance(self):
        """Test that scores exceeding tolerance raise."""
        expected = [[0.50, 0.30, 0.20]]
        actual = [[0.55, 0.25, 0.20]]  # 5% difference
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_match(expected, actual, tolerance=0.01)
        assert "diff" in str(exc_info.value)

    def test_different_item_count(self):
        """Test that different item counts raise."""
        expected = [[0.5, 0.5]]
        actual = [[0.5, 0.5], [0.3, 0.7]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_match(expected, actual)
        assert "Expected 1 items" in str(exc_info.value)

    def test_different_label_count(self):
        """Test that different label counts raise."""
        expected = [[0.5, 0.3, 0.2]]
        actual = [[0.5, 0.5]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_match(expected, actual)
        assert "expected 3 scores" in str(exc_info.value)

    def test_with_case_name(self):
        """Test that case name appears in error message."""
        expected = [[0.5]]
        actual = [[0.9]]
        with pytest.raises(AssertionError) as exc_info:
            assert_scores_match(expected, actual, case_name="my_test")
        assert "my_test" in str(exc_info.value)


class TestGenerateTestItems:
    """Tests for generate_test_items function."""

    def test_generates_correct_count(self):
        """Test that correct number of items are generated."""
        items = generate_test_items(5)
        assert len(items) == 5

    def test_default_prefix(self):
        """Test default prefix is used."""
        items = generate_test_items(3)
        assert items[0] == " item0"
        assert items[1] == " item1"
        assert items[2] == " item2"

    def test_custom_prefix(self):
        """Test custom prefix is used."""
        items = generate_test_items(2, prefix=" test")
        assert items[0] == " test0"
        assert items[1] == " test1"

    def test_zero_items(self):
        """Test generating zero items."""
        items = generate_test_items(0)
        assert items == []

    def test_one_item(self):
        """Test generating one item."""
        items = generate_test_items(1)
        assert items == [" item0"]


class TestSkipHelpers:
    """Tests for skip helper predicates."""

    def test_should_run_hf_reference(self, monkeypatch):
        monkeypatch.delenv("SGLANG_JAX_RUN_HF_REFERENCE", raising=False)
        assert should_run_hf_reference() is False

        monkeypatch.setenv("SGLANG_JAX_RUN_HF_REFERENCE", "1")
        assert should_run_hf_reference() is True
