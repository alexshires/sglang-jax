import unittest

import numpy as np

from sgl_jax.srt.managers.schedule_batch import _build_multi_item_extend_positions


class TestMultiItemUtils(unittest.TestCase):
    def test_build_multi_item_extend_positions(self):
        # Query: [1, 2, 3] (len 3)
        # Delimiter: 99
        # Item 1: [4, 5]
        # Item 2: [6]
        # Sequence: [1, 2, 3, 99, 4, 5, 99, 6, 99]

        tokens = np.array([1, 2, 3, 99, 4, 5, 99, 6, 99], dtype=np.int32)
        delimiter_token_id = 99

        expected_positions = np.array([0, 1, 2, 3, 4, 5, 3, 4, 3], dtype=np.int32)

        positions = _build_multi_item_extend_positions(tokens, delimiter_token_id, np.int32)

        np.testing.assert_array_equal(positions, expected_positions)

    def test_build_multi_item_extend_positions_no_delimiter(self):
        tokens = np.array([1, 2, 3], dtype=np.int32)
        delimiter_token_id = 99

        with self.assertRaises(ValueError):
            _build_multi_item_extend_positions(tokens, delimiter_token_id, np.int32)

    def test_build_multi_item_extend_positions_empty_query(self):
        # Starts with delimiter
        tokens = np.array([99, 1, 2], dtype=np.int32)
        delimiter_token_id = 99

        with self.assertRaises(ValueError):
            _build_multi_item_extend_positions(tokens, delimiter_token_id, np.int32)


if __name__ == "__main__":
    unittest.main()
