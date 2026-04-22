import jax
import jax.nn as jnn


def _compute_label_only_logprobs(next_token_logits, label_token_ids_arr, out_sharding):
    """TPU-optimized label-only logprob computation.

    Math: logprob[label] = logit[label] - logsumexp(logits)
    """
    # next_token_logits: [bs, vocab]
    # label_token_ids_arr: [num_labels]

    # 1. Get logits for our labels
    # row_logits: [bs, num_labels]
    label_logits = next_token_logits[:, label_token_ids_arr]

    # 2. Get normalizer (logsumexp across entire vocab)
    # normalizer: [bs, 1]
    normalizer = jnn.logsumexp(next_token_logits, axis=-1, keepdims=True)

    # 3. Compute logprobs
    row_logprobs = label_logits - normalizer

    # jax.device_put with sharding to keep it on TPU
    return jax.lax.with_sharding_constraint(row_logprobs, out_sharding)
