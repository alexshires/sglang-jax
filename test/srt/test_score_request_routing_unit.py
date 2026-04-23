from types import SimpleNamespace

from sgl_jax.srt.managers.tokenizer_score_routing_mixin import (
    TokenizerScoreRoutingMixin,
)


class _DummyTokenizerManager(TokenizerScoreRoutingMixin):
    pass


def test_score_lane_scheduler_index_uses_cache_handle_hash_when_fan_out_enabled():
    manager = _DummyTokenizerManager()
    manager.send_to_scheduler = SimpleNamespace(fan_out=4)
    index = manager._score_lane_scheduler_index("cache-handle")
    assert index is not None
    assert 0 <= index < 4


def test_local_score_rpc_requires_threshold_and_single_scheduler_lane():
    manager = _DummyTokenizerManager()
    manager.local_rpc_submitter = lambda _req: None
    manager.send_to_scheduler = SimpleNamespace(fan_out=1)
    manager.server_args = SimpleNamespace(multi_item_score_local_rpc_min_items=8)

    assert manager._can_use_local_score_rpc(total_items=8) is True
    assert manager._can_use_local_score_rpc(total_items=7) is False
