import asyncio
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.tokenizer_score_cache_mixin import TokenizerScoreCacheMixin


class _FakeScoreCacheManager(TokenizerScoreCacheMixin):
    def __init__(
        self,
        *,
        enabled: bool,
        ttl: float = 0.0,
        max_entries: int = 128,
    ):
        self.server_args = SimpleNamespace(
            multi_item_score_reuse_prefill_cache_by_prefix=enabled,
            multi_item_score_reuse_prefill_cache_ttl=ttl,
            multi_item_score_reuse_prefill_cache_max_entries=max_entries,
        )
        self.asyncio_tasks = set()
        self.prefill_gate: asyncio.Event | None = None
        self.prefill_exception: Exception | None = None
        self.prefill_calls: list[list[int]] = []
        self.release_calls: list[str] = []

    async def _prefill_and_cache(self, query_tokens: list[int]) -> str:
        self.prefill_calls.append(list(query_tokens))
        handle = f"handle-{len(self.prefill_calls)}"
        if self.prefill_gate is not None:
            await self.prefill_gate.wait()
        if self.prefill_exception is not None:
            raise self.prefill_exception
        await asyncio.sleep(0)
        return handle

    async def _release_cache(self, cache_handle: str) -> bool:
        self.release_calls.append(cache_handle)
        return True


def test_reusable_prefill_cache_disabled_uses_direct_lifecycle():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=False)

        first = await manager._acquire_prefill_cache_handle([1, 2, 3])
        second = await manager._acquire_prefill_cache_handle([1, 2, 3])

        assert first == "handle-1"
        assert second == "handle-2"
        assert manager.prefill_calls == [[1, 2, 3], [1, 2, 3]]
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "disabled": 2
        }

        assert await manager._release_prefill_cache_handle(first)
        assert await manager._release_prefill_cache_handle(second)
        assert manager.release_calls == ["handle-1", "handle-2"]

    asyncio.run(_run())


def test_reusable_prefill_cache_coalesces_in_flight_prefix_prefill():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.0)
        manager.prefill_gate = asyncio.Event()

        first_task = asyncio.create_task(
            manager._acquire_prefill_cache_handle([4, 5, 6])
        )
        await asyncio.sleep(0)
        second_task = asyncio.create_task(
            manager._acquire_prefill_cache_handle([4, 5, 6])
        )
        await asyncio.sleep(0)

        assert manager.prefill_calls == [[4, 5, 6]]
        manager.prefill_gate.set()

        first, second = await asyncio.gather(first_task, second_task)
        assert first == "handle-1"
        assert second == "handle-1"
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "hit_inflight": 1,
        }

        assert await manager._release_prefill_cache_handle(first)
        assert manager.release_calls == []
        assert await manager._release_prefill_cache_handle(second)
        assert manager.release_calls == ["handle-1"]
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "hit_inflight": 1,
            "released_now": 1,
        }

    asyncio.run(_run())


def test_reusable_prefill_cache_cancelled_waiter_does_not_cancel_shared_prefill():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.0)
        manager.prefill_gate = asyncio.Event()

        first_task = asyncio.create_task(
            manager._acquire_prefill_cache_handle([4, 5, 6])
        )
        second_task = asyncio.create_task(
            manager._acquire_prefill_cache_handle([4, 5, 6])
        )
        await asyncio.sleep(0)

        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task

        manager.prefill_gate.set()
        second = await second_task

        assert second == "handle-1"
        assert manager.prefill_calls == [[4, 5, 6]]
        assert manager.release_calls == []
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "hit_inflight": 1,
        }

        assert await manager._release_prefill_cache_handle(second)
        assert manager.release_calls == ["handle-1"]

    asyncio.run(_run())


def test_reusable_prefill_cache_cancelled_only_waiter_releases_unused_handle():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.0)
        manager.prefill_gate = asyncio.Event()

        task = asyncio.create_task(manager._acquire_prefill_cache_handle([12, 13]))
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        manager.prefill_gate.set()
        await asyncio.gather(*list(manager.asyncio_tasks), return_exceptions=True)

        _, cache, handle_to_key = manager._score_reusable_prefill_cache_state()
        assert cache == {}
        assert handle_to_key == {}
        assert manager.release_calls == ["handle-1"]
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "released_unused": 1,
        }

    asyncio.run(_run())


def test_reusable_prefill_cache_reuses_ready_handle_until_last_release():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.0)

        first = await manager._acquire_prefill_cache_handle([7, 8])
        second = await manager._acquire_prefill_cache_handle([7, 8])
        other = await manager._acquire_prefill_cache_handle([9])

        assert first == "handle-1"
        assert second == "handle-1"
        assert other == "handle-2"
        assert manager.prefill_calls == [[7, 8], [9]]
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 2,
            "hit_ready": 1,
        }

        assert await manager._release_prefill_cache_handle(first)
        assert manager.release_calls == []
        assert await manager._release_prefill_cache_handle(second)
        assert manager.release_calls == ["handle-1"]
        assert await manager._release_prefill_cache_handle(other)
        assert manager.release_calls == ["handle-1", "handle-2"]

    asyncio.run(_run())


def test_reusable_prefill_cache_idle_release_skips_reacquired_generation():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.01)

        first = await manager._acquire_prefill_cache_handle([10, 11])
        assert await manager._release_prefill_cache_handle(first)

        second = await manager._acquire_prefill_cache_handle([10, 11])
        assert second == first

        await asyncio.sleep(0.03)
        assert manager.release_calls == []

        assert await manager._release_prefill_cache_handle(second)
        await asyncio.sleep(0.03)
        assert manager.release_calls == ["handle-1"]
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "hit_ready": 1,
            "released_idle": 1,
        }

    asyncio.run(_run())


def test_reusable_prefill_cache_idle_release_expires_without_reacquire():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.01)

        handle = await manager._acquire_prefill_cache_handle([14, 15])
        assert handle == "handle-1"
        assert await manager._release_prefill_cache_handle(handle)

        await asyncio.sleep(0.03)

        _, cache, handle_to_key = manager._score_reusable_prefill_cache_state()
        assert cache == {}
        assert handle_to_key == {}
        assert manager.release_calls == ["handle-1"]
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "released_idle": 1,
        }

    asyncio.run(_run())


def test_reusable_prefill_cache_max_entries_bypasses_new_keys():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.0, max_entries=1)

        first = await manager._acquire_prefill_cache_handle([1])
        second = await manager._acquire_prefill_cache_handle([2])

        assert first == "handle-1"
        assert second == "handle-2"
        assert manager.prefill_calls == [[1], [2]]
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "bypass_full": 1,
        }

        _, cache, handle_to_key = manager._score_reusable_prefill_cache_state()
        assert list(cache.keys()) == [(1,)]
        assert handle_to_key == {"handle-1": (1,)}

        assert await manager._release_prefill_cache_handle(second)
        assert manager.release_calls == ["handle-2"]
        assert await manager._release_prefill_cache_handle(first)
        assert manager.release_calls == ["handle-2", "handle-1"]
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "bypass_full": 1,
            "released_now": 2,
        }

    asyncio.run(_run())


def test_reusable_prefill_cache_max_entries_counts_inflight_entries():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.0, max_entries=1)
        manager.prefill_gate = asyncio.Event()

        first_task = asyncio.create_task(manager._acquire_prefill_cache_handle([1]))
        await asyncio.sleep(0)
        second_task = asyncio.create_task(manager._acquire_prefill_cache_handle([2]))
        await asyncio.sleep(0)

        assert manager.prefill_calls == [[1], [2]]
        metrics = manager.score_reusable_prefill_cache_metrics()
        assert metrics["entries"] == 1
        assert metrics["inflight_entries"] == 1
        assert metrics["events"] == {"new": 1, "bypass_full": 1}

        manager.prefill_gate.set()
        first, second = await asyncio.gather(first_task, second_task)
        assert first == "handle-1"
        assert second == "handle-2"

        assert await manager._release_prefill_cache_handle(first)
        assert await manager._release_prefill_cache_handle(second)
        assert manager.release_calls == ["handle-1", "handle-2"]

    asyncio.run(_run())


def test_reusable_prefill_cache_prefill_failure_propagates_to_all_waiters():
    async def _run():
        manager = _FakeScoreCacheManager(enabled=True, ttl=0.0)
        manager.prefill_gate = asyncio.Event()
        manager.prefill_exception = RuntimeError("prefill failed")

        first_task = asyncio.create_task(manager._acquire_prefill_cache_handle([4, 5]))
        await asyncio.sleep(0)
        second_task = asyncio.create_task(manager._acquire_prefill_cache_handle([4, 5]))
        await asyncio.sleep(0)

        manager.prefill_gate.set()

        with pytest.raises(RuntimeError, match="prefill failed"):
            await first_task
        with pytest.raises(RuntimeError, match="prefill failed"):
            await second_task

        _, cache, handle_to_key = manager._score_reusable_prefill_cache_state()
        assert cache == {}
        assert handle_to_key == {}
        assert manager.release_calls == []
        assert manager.score_reusable_prefill_cache_metrics()["events"] == {
            "new": 1,
            "hit_inflight": 1,
            "prefill_failed": 2,
        }

    asyncio.run(_run())
