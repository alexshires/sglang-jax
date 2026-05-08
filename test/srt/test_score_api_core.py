"""
Core engine tests for Score API.

These tests exercise `Engine.score(...)` against a real TPU-backed engine.
They skip on non-TPU hosts so the file is safe to collect in local CPU runs.
"""

import unittest

import jax

from sgl_jax.test.score_test_utils import (
    ScoreAPITestCase,
    ScoreTestConfig,
    assert_scores_shape,
    assert_scores_valid,
    get_label_token_ids,
)


def _skip_if_no_tpu() -> None:
    if not any(device.platform == "tpu" for device in jax.devices()):
        raise unittest.SkipTest("Score API core tests require TPU.")


class TestScoreAPICore(ScoreAPITestCase):
    """Core engine tests for Score API."""

    config = ScoreTestConfig(
        mem_fraction_static=0.7,
        max_running_requests=8,
        precompile_bs_paddings=[8],
        precompile_token_paddings=[1024],
        log_level="debug",
    )

    @classmethod
    def setUpClass(cls):
        _skip_if_no_tpu()
        super().setUpClass()

    def test_score_text_input(self):
        """Test Score API with text query and items."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B", " C"])

        scores = self.engine.score(
            query="The capital of France is",
            items=[" Paris", " London", " Berlin"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
            item_first=False,
        )

        assert_scores_shape(scores, expected_items=3, expected_labels=3, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_score_token_input(self):
        """Test Score API with token ID inputs."""
        query_tokens = self.tokenizer.encode("The answer is", add_special_tokens=False)
        item1_tokens = self.tokenizer.encode(" yes", add_special_tokens=False)
        item2_tokens = self.tokenizer.encode(" no", add_special_tokens=False)
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        scores = self.engine.score(
            query=query_tokens,
            items=[item1_tokens, item2_tokens],
            label_token_ids=label_token_ids,
            apply_softmax=True,
            item_first=False,
        )

        assert_scores_shape(scores, expected_items=2, expected_labels=2, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_score_apply_softmax_true(self):
        """Test Score API with apply_softmax=True."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" X", " Y", " Z"])

        scores = self.engine.score(
            query="Test query",
            items=[" option1", " option2", " option3"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        assert_scores_shape(scores, expected_items=3, expected_labels=3, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_score_apply_softmax_false(self):
        """Test Score API with apply_softmax=False."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B", " C"])

        scores = self.engine.score(
            query="Test query",
            items=[" option1", " option2"],
            label_token_ids=label_token_ids,
            apply_softmax=False,
        )

        assert_scores_shape(scores, expected_items=2, expected_labels=3, test_case=self)
        assert_scores_valid(scores, apply_softmax=False, test_case=self)

    def test_score_item_first_false(self):
        """Test Score API with query+item order."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        scores = self.engine.score(
            query=" is the answer",
            items=["Yes", "No"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
            item_first=False,
        )

        assert_scores_shape(scores, expected_items=2, expected_labels=2, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_score_item_first_true(self):
        """Test Score API with item+query order."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        scores_item_first = self.engine.score(
            query=" is the answer",
            items=["Yes", "No"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
            item_first=True,
        )
        scores_query_first = self.engine.score(
            query=" is the answer",
            items=["Yes", "No"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
            item_first=False,
        )

        assert_scores_shape(scores_item_first, expected_items=2, expected_labels=2, test_case=self)
        assert_scores_valid(scores_item_first, apply_softmax=True, test_case=self)
        self.assertEqual(len(scores_item_first), len(scores_query_first))

    def test_score_batch_handling(self):
        """Test Score API with various batch sizes."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        for batch_size in [1, 2, 4, 8]:
            with self.subTest(batch_size=batch_size):
                scores = self.engine.score(
                    query="Test query for batch handling",
                    items=[f" item{i}" for i in range(batch_size)],
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                )

                assert_scores_shape(
                    scores, expected_items=batch_size, expected_labels=2, test_case=self
                )
                assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_score_single_item(self):
        """Test Score API with a single item."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" X", " Y", " Z"])

        scores = self.engine.score(
            query="Single item test",
            items=[" only_item"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        assert_scores_shape(scores, expected_items=1, expected_labels=3, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_score_different_label_token_counts(self):
        """Test Score API with different label token counts."""
        candidate_labels = [" A", " B", " C", " D", " E", " F", " G", " H"]

        for num_labels in [1, 2, 4, 8]:
            with self.subTest(num_labels=num_labels):
                label_token_ids = get_label_token_ids(
                    self.tokenizer,
                    candidate_labels[:num_labels],
                )

                scores = self.engine.score(
                    query="Test with varying labels",
                    items=[" A", " B"],
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                )

                assert_scores_shape(
                    scores, expected_items=2, expected_labels=num_labels, test_case=self
                )
                assert_scores_valid(scores, apply_softmax=True, test_case=self)

    def test_score_determinism(self):
        """Test that same input produces identical scores."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])
        kwargs = dict(
            query="Determinism test query",
            items=[" option1", " option2"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        scores1 = self.engine.score(**kwargs)
        scores2 = self.engine.score(**kwargs)

        self.assertEqual(len(scores1), len(scores2))
        for i, (s1, s2) in enumerate(zip(scores1, scores2)):
            self.assertEqual(len(s1), len(s2))
            for j, (v1, v2) in enumerate(zip(s1, s2)):
                self.assertAlmostEqual(
                    v1,
                    v2,
                    places=5,
                    msg=f"Score [{i}][{j}]: {v1} != {v2} (non-deterministic)",
                )

    def test_score_default_params(self):
        """Test Score API with default parameters."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B"])

        scores = self.engine.score(
            query="Default params test",
            items=[" test"],
            label_token_ids=label_token_ids,
        )

        assert_scores_shape(scores, expected_items=1, expected_labels=2, test_case=self)
        assert_scores_valid(scores, apply_softmax=False, test_case=self)

    def test_score_numerical_stability(self):
        """Test repeated score calls are numerically stable."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" X", " Y"])
        all_scores = [
            self.engine.score(
                query="Numerical stability test",
                items=[" A", " B", " C"],
                label_token_ids=label_token_ids,
                apply_softmax=True,
            )
            for _ in range(3)
        ]

        for run_idx in range(1, len(all_scores)):
            for item_idx in range(len(all_scores[0])):
                for label_idx in range(len(all_scores[0][0])):
                    v0 = all_scores[0][item_idx][label_idx]
                    v1 = all_scores[run_idx][item_idx][label_idx]
                    self.assertAlmostEqual(
                        v0,
                        v1,
                        places=5,
                        msg=(
                            f"Run {run_idx}: Score [{item_idx}][{label_idx}] "
                            f"unstable: {v0} vs {v1}"
                        ),
                    )

    def test_score_extreme_values(self):
        """Test long inputs do not produce invalid scores."""
        label_token_ids = get_label_token_ids(self.tokenizer, [" X", " Y"])

        scores = self.engine.score(
            query="This is a test " * 50,
            items=[" A", " B"],
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        assert_scores_shape(scores, expected_items=2, expected_labels=2, test_case=self)
        assert_scores_valid(scores, apply_softmax=True, test_case=self)
