from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from sidekick.ai import AIStateRepository
from sidekick.ai_memory_ingestion import (
    MemoryIngestionBusyError,
    MemoryOutboxDrainResult,
)


class MemoryOutboxRunner(Protocol):
    async def run_memory_outbox_scope(
        self,
        scope_id: str,
    ) -> MemoryOutboxDrainResult: ...


@dataclass(frozen=True, slots=True)
class MemoryOutboxSchedulerSettings:
    poll_interval_seconds: float = 10
    concurrency: int = 2
    scope_batch_size: int = 20

    def __post_init__(self) -> None:
        if (
            self.poll_interval_seconds <= 0
            or self.concurrency < 1
            or self.scope_batch_size < 1
        ):
            raise ValueError("Memory outbox scheduler limits must be positive")

    @classmethod
    def from_env(cls) -> MemoryOutboxSchedulerSettings:
        return cls(
            poll_interval_seconds=float(
                os.environ.get("SIDEKICK_MEMORY_OUTBOX_POLL_SECONDS", "10")
            ),
            concurrency=int(
                os.environ.get("SIDEKICK_MEMORY_OUTBOX_CONCURRENCY", "2")
            ),
            scope_batch_size=int(
                os.environ.get("SIDEKICK_MEMORY_OUTBOX_SCOPE_BATCH_SIZE", "20")
            ),
        )


@dataclass(frozen=True, slots=True)
class MemoryOutboxScheduleResult:
    scopes_seen: int
    scopes_succeeded: int
    scopes_failed: int
    scopes_busy: int
    documents_attempted: int
    documents_failed: int
    documents_dead_lettered: int
    scopes_pending: bool


class MemoryOutboxScheduler:
    def __init__(
        self,
        *,
        runner: MemoryOutboxRunner,
        store: AIStateRepository,
        settings: MemoryOutboxSchedulerSettings = MemoryOutboxSchedulerSettings(),
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        logger: Any | None = None,
    ):
        self._runner = runner
        self._store = store
        self._settings = settings
        self._clock = clock
        self._sleep = sleep
        self._logger = logger
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run_forever(),
            name="sidekick-memory-outbox-scheduler",
        )

    async def close(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def run_once(self) -> MemoryOutboxScheduleResult:
        scopes = await self._store.list_due_memory_outbox_scopes(
            due_at=self._clock(),
            limit=self._settings.scope_batch_size,
        )
        semaphore = asyncio.Semaphore(self._settings.concurrency)

        async def run(
            scope_id: str,
        ) -> tuple[str, MemoryOutboxDrainResult | None]:
            async with semaphore:
                try:
                    result = await self._runner.run_memory_outbox_scope(scope_id)
                except MemoryIngestionBusyError:
                    return "busy", None
                except Exception as exc:
                    if self._logger is not None:
                        self._logger.warning(
                            "Memory outbox scope failed (error=%s)",
                            type(exc).__name__,
                        )
                    return "failed", None
            return "succeeded", result

        results = await asyncio.gather(*(run(scope_id) for scope_id in scopes))
        succeeded = sum(status == "succeeded" for status, _ in results)
        failed = sum(status == "failed" for status, _ in results)
        busy = sum(status == "busy" for status, _ in results)
        deliveries = tuple(result for _, result in results if result is not None)
        schedule_result = MemoryOutboxScheduleResult(
            scopes_seen=len(scopes),
            scopes_succeeded=succeeded,
            scopes_failed=failed,
            scopes_busy=busy,
            documents_attempted=sum(
                result.documents_attempted for result in deliveries
            ),
            documents_failed=sum(result.documents_failed for result in deliveries),
            documents_dead_lettered=sum(
                result.documents_dead_lettered for result in deliveries
            ),
            scopes_pending=len(scopes) >= self._settings.scope_batch_size,
        )
        if self._logger is not None and (
            schedule_result.documents_attempted
            or schedule_result.scopes_failed
            or schedule_result.scopes_busy
        ):
            self._logger.info(
                "Memory outbox cycle complete "
                "(scopes=%s, succeeded=%s, failed=%s, busy=%s, "
                "documents=%s, delivery_failures=%s, dead_letters=%s)",
                schedule_result.scopes_seen,
                schedule_result.scopes_succeeded,
                schedule_result.scopes_failed,
                schedule_result.scopes_busy,
                schedule_result.documents_attempted,
                schedule_result.documents_failed,
                schedule_result.documents_dead_lettered,
            )
        return schedule_result

    async def _run_forever(self) -> None:
        while True:
            try:
                result = await self.run_once()
            except Exception as exc:
                if self._logger is not None:
                    self._logger.error(
                        "Memory outbox orchestration failed (error=%s)",
                        type(exc).__name__,
                    )
                result = None
            await self._sleep(
                0
                if result is not None and result.scopes_pending
                else self._settings.poll_interval_seconds
            )
