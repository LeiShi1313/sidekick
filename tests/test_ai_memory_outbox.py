from __future__ import annotations

import pytest

from sidekick.ai_memory_ingestion import (
    MemoryIngestionBusyError,
    MemoryOutboxDrainResult,
)
from sidekick.ai_memory_outbox import (
    MemoryOutboxScheduler,
    MemoryOutboxSchedulerSettings,
)


def test_memory_outbox_scheduler_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("SIDEKICK_MEMORY_OUTBOX_POLL_SECONDS", "3.5")
    monkeypatch.setenv("SIDEKICK_MEMORY_OUTBOX_CONCURRENCY", "4")
    monkeypatch.setenv("SIDEKICK_MEMORY_OUTBOX_SCOPE_BATCH_SIZE", "12")

    settings = MemoryOutboxSchedulerSettings.from_env()

    assert settings.poll_interval_seconds == 3.5
    assert settings.concurrency == 4
    assert settings.scope_batch_size == 12


@pytest.mark.asyncio
async def test_memory_outbox_scheduler_drains_due_scopes_independently():
    class Store:
        calls = []

        async def list_due_memory_outbox_scopes(self, *, due_at, limit):
            self.calls.append((due_at, limit))
            return ("telegram:chat:-1001", "qq:group:700")

    class Runner:
        calls = []

        async def run_memory_outbox_scope(self, scope_id):
            self.calls.append(scope_id)
            if scope_id.startswith("qq:"):
                raise MemoryIngestionBusyError("leased")
            return MemoryOutboxDrainResult(
                documents_attempted=2,
                documents_failed=1,
                documents_dead_lettered=1,
            )

    store = Store()
    runner = Runner()
    scheduler = MemoryOutboxScheduler(
        runner=runner,
        store=store,
        settings=MemoryOutboxSchedulerSettings(
            concurrency=2,
            scope_batch_size=10,
        ),
        clock=lambda: 1_800_000_000,
    )

    result = await scheduler.run_once()

    assert store.calls == [(1_800_000_000, 10)]
    assert runner.calls == ["telegram:chat:-1001", "qq:group:700"]
    assert result.scopes_succeeded == 1
    assert result.scopes_busy == 1
    assert result.documents_attempted == 2
    assert result.documents_failed == 1
    assert result.documents_dead_lettered == 1
