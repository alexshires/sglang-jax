import json
import os

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

MODEL_PATH = "/models/meta-llama/Llama-3.2-1B"


def get_token_ids(tokenizer, tokens):
    token_ids = []
    for t in tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        token_ids.append(ids[0])
    return token_ids


def generate_scores():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = FlaxAutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)

    tokens = [" to", " the"]
    label_token_ids = get_token_ids(tokenizer, tokens)
    print(f"Label token IDs: {label_token_ids}")

    test_cases = [
        {
            "name": "default case",
            "query": "I pledge allegiance",
            "items": ["", " to"],
            "item_first": False,
        },
        {
            "name": "item_first case",
            "query": " is a city",
            "items": ["Tokyo", "Japan"],
            "item_first": True,
        },
    ]

    results = {}

    for case in test_cases:
        print(f"Processing {case['name']}...")
        scores = []
        for item in case["items"]:
            full_text = f"{item}{case['query']}" if case["item_first"] else f"{case['query']}{item}"
            inputs = tokenizer(full_text, return_tensors="np")
            outputs = model(**inputs)
            last_token_logits = outputs.logits[0, -1]
            target_logits = last_token_logits[jnp.asarray(label_token_ids)]
            target_probs = jax.nn.softmax(target_logits, axis=-1)
            probs = [float(target_probs[i]) for i in range(len(label_token_ids))]
            scores.append(probs)

        results[case["name"]] = {
            "query": case["query"],
            "items": case["items"],
            "item_first": case["item_first"],
            "scores": scores,
            "label_token_ids": label_token_ids,
            "tokens": tokens,
        }

    os.makedirs("python/sgl_jax/test/data", exist_ok=True)
    output_path = "python/sgl_jax/test/data/llama_3_2_1b_golden_scores.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved golden scores to {output_path}")


if __name__ == "__main__":
    generate_scores()
