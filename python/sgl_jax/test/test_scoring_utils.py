import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding

from sgl_jax.srt.managers.scoring_utils import _compute_label_only_logprobs


class TestScoringUtils(unittest.TestCase):
    def test_compute_label_only_logprobs(self):
        # Force CPU for test
        device = jax.devices("cpu")[0]
        sharding = SingleDeviceSharding(device)

        # Logits: [2, 3]
        next_token_logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
        # Target labels: [0, 2]
        label_token_ids_arr = jnp.array([0, 2], dtype=jnp.int32)

        # Expected logsumexp:
        # row 0: log(e^1 + e^2 + e^3) = 3.4076
        # row 1: log(e^4 + e^5 + e^6) = 6.4076

        # Expected logprobs:
        # row 0, label 0: 1.0 - 3.4076 = -2.4076
        # row 0, label 2: 3.0 - 3.4076 = -0.4076
        # row 1, label 0: 4.0 - 6.4076 = -2.4076
        # row 1, label 2: 6.0 - 6.4076 = -0.4076

        expected_logprobs = jnp.array([[-2.4076, -0.4076], [-2.4076, -0.4076]], dtype=jnp.float32)

        # Run function
        logprobs = _compute_label_only_logprobs(next_token_logits, label_token_ids_arr, sharding)

        # Compare
        np.testing.assert_allclose(logprobs, expected_logprobs, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
