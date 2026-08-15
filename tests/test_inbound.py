from __future__ import annotations

import asyncio

import pytest

from sidekick.inbound import DurableInboundPool


@pytest.mark.asyncio
async def test_pool_rechecks_queue_after_a_consumed_wakeup() -> None:
    rechecked = asyncio.Event()

    class ScheduledStore:
        async def next_pending_ai_work_at(self, _source_id):
            return None

    class WakeupDuringIdleWorker:
        calls = 0
        pool: DurableInboundPool | None = None

        async def process_one(self, _handler):
            self.calls += 1
            if self.calls == 1:
                assert self.pool is not None
                self.pool.notify()
                return "idle"
            rechecked.set()
            await asyncio.Future()

    class UnusedHandler:
        async def handle(self, _message, *, attested_origin=None):
            return False

    worker = WakeupDuringIdleWorker()
    pool = DurableInboundPool(
        worker,
        ScheduledStore(),
        "test-source",
        UnusedHandler(),
        concurrency=1,
    )
    worker.pool = pool
    pool.start()
    try:
        await asyncio.wait_for(rechecked.wait(), timeout=1)
    finally:
        await pool.close()
