import pytest
from jax import numpy as jnp

from sgl_jax.srt.kernels.ragged_paged_attention import tuned_block_sizes as tbs


@pytest.fixture(autouse=True)
def reset_rpa_tuning_state(monkeypatch):
    tbs._LOGGED_RPA_POLICY_KEYS.clear()
    monkeypatch.delenv("SGLANG_RPA_KERNEL_V11", raising=False)
    yield
    tbs._LOGGED_RPA_POLICY_KEYS.clear()


def _install_fake_v6e(monkeypatch):
    monkeypatch.setattr(tbs, "get_tpu_version", lambda: 6)
    monkeypatch.setattr(tbs, "get_device_name", lambda: "TPU v6e")


def _get_tuned(*, page_size=64, max_num_tokens=2048, pages_per_seq=99):
    return tbs.get_tuned_block_sizes(
        jnp.bfloat16,
        jnp.bfloat16,
        actual_num_q_heads=16,
        actual_num_kv_heads=8,
        head_dim=128,
        page_size=page_size,
        max_num_tokens=max_num_tokens,
        pages_per_seq=pages_per_seq,
    )


def test_exact_tuned_block_size_wins_before_fallback(monkeypatch):
    _install_fake_v6e(monkeypatch)
    monkeypatch.setattr(
        tbs,
        "TUNED_BLOCK_SIZES",
        {
            "TPU v6e": {
                ("bfloat16", "bfloat16", 16, 8, 128, 64, 2048): (7, 77),
                ("bfloat16", "bfloat16", 16, 8, 128, 128, 4096): (16, 96),
            }
        },
    )

    assert _get_tuned(page_size=64, max_num_tokens=2048) == (7, 77)


def test_page_size_fallback_scales_kv_pages_from_neighbor(monkeypatch):
    _install_fake_v6e(monkeypatch)
    monkeypatch.setattr(
        tbs,
        "TUNED_BLOCK_SIZES",
        {
            "TPU v6e": {
                ("bfloat16", "bfloat16", 16, 8, 128, 128, 4096): (16, 96),
            }
        },
    )

    assert _get_tuned(page_size=64, max_num_tokens=2048) == (32, 96)


def test_missing_tuned_candidates_falls_back_to_default(monkeypatch):
    _install_fake_v6e(monkeypatch)
    monkeypatch.setattr(tbs, "TUNED_BLOCK_SIZES", {"TPU v6e": {}})

    assert _get_tuned(page_size=64, max_num_tokens=128) == (16, 32)
