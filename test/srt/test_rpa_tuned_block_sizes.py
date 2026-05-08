import pytest
from jax import numpy as jnp

from sgl_jax.srt.kernels.ragged_paged_attention import tuned_block_sizes as tbs


@pytest.fixture(autouse=True)
def reset_rpa_tuning_state(monkeypatch):
    tbs.set_logical_device_count_override(None)
    tbs._LOGGED_RPA_POLICY_KEYS.clear()
    monkeypatch.delenv("SGLANG_LOGICAL_DEVICE_COUNT", raising=False)
    monkeypatch.delenv("SGLANG_RPA_KERNEL_V11", raising=False)
    monkeypatch.delenv("SGLANG_RPA_V6E8_PAGE64_REPLICA_OVERRIDE", raising=False)
    yield
    tbs.set_logical_device_count_override(None)
    tbs._LOGGED_RPA_POLICY_KEYS.clear()


def _install_fake_v6e(monkeypatch):
    monkeypatch.setattr(tbs, "get_tpu_version", lambda: 6)

    def fake_get_device_name(num_devices=None):
        if num_devices is not None:
            return "TPU v6e-8" if int(num_devices) == 8 else "TPU v6e"
        return "TPU v6e"

    monkeypatch.setattr(tbs, "get_device_name", fake_get_device_name)
    tbs.set_logical_device_count_override(8)


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


def test_v6e8_page64_replica_override_is_opt_in(monkeypatch):
    _install_fake_v6e(monkeypatch)
    monkeypatch.setattr(tbs, "TUNED_BLOCK_SIZES", {"TPU v6e": {}})

    assert _get_tuned(page_size=64, max_num_tokens=128) == (16, 32)

    monkeypatch.setenv("SGLANG_RPA_V6E8_PAGE64_REPLICA_OVERRIDE", "1")
    assert _get_tuned(page_size=64, max_num_tokens=128) == (32, 96)


def test_logical_device_count_override_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("SGLANG_LOGICAL_DEVICE_COUNT", "4")
    assert tbs.get_logical_device_count() == 4

    tbs.set_logical_device_count_override(8)
    assert tbs.get_logical_device_count() == 8

    tbs.set_logical_device_count_override(None)
    assert tbs.get_logical_device_count() == 4
