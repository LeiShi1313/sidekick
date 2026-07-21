from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable

from croniter import croniter

from sidekick.ai import AIStateRepository, MemoryDreamRunner
from sidekick.ai_memory_ingestion import MemoryIngestionBusyError
from sidekick.chat.identity import IdentityCodec


@dataclass(frozen=True, slots=True)
class DreamSettings:
    lookback: timedelta = timedelta(hours=24)
    overlap: timedelta = timedelta(minutes=10)
    cycle_budget_seconds: float = 50
    scope_timeout_seconds: float = 300

    def __post_init__(self) -> None:
        if self.lookback <= timedelta(0):
            raise ValueError("Dream lookback must be positive")
        if self.overlap < timedelta(0):
            raise ValueError("Dream overlap cannot be negative")
        if self.cycle_budget_seconds <= 0:
            raise ValueError("Dream cycle budget must be positive")
        if self.scope_timeout_seconds <= 0:
            raise ValueError("Dream scope timeout must be positive")

    @classmethod
    def from_env(cls) -> DreamSettings:
        return cls(
            lookback=timedelta(
                hours=float(
                    os.environ.get("SIDEKICK_MEMORY_DREAM_LOOKBACK_HOURS", "24")
                )
            ),
            overlap=timedelta(
                seconds=float(
                    os.environ.get("SIDEKICK_MEMORY_DREAM_OVERLAP_SECONDS", "600")
                )
            ),
            cycle_budget_seconds=float(
                os.environ.get("SIDEKICK_MEMORY_DREAM_CYCLE_BUDGET_SECONDS", "50")
            ),
            scope_timeout_seconds=float(
                os.environ.get("SIDEKICK_MEMORY_DREAM_SCOPE_TIMEOUT_SECONDS", "300")
            ),
        )


@dataclass(frozen=True, slots=True)
class DreamSchedulerSettings:
    cron: str | None = "0 * * * *"
    concurrency: int = 2
    scope_batch_size: int = 20

    def __post_init__(self) -> None:
        if self.cron is not None:
            if len(self.cron.split()) != 5 or not croniter.is_valid(self.cron):
                raise ValueError("Dream schedule must be a valid five-field cron")
        if self.concurrency < 1 or self.scope_batch_size < 1:
            raise ValueError("Dream scheduler limits must be positive")

    @classmethod
    def from_env(cls) -> DreamSchedulerSettings:
        raw_cron = os.environ.get("SIDEKICK_MEMORY_DREAM_CRON", "0 * * * *").strip()
        cron = None if raw_cron.casefold() in {"", "off", "disabled"} else raw_cron
        return cls(
            cron=cron,
            concurrency=int(os.environ.get("SIDEKICK_MEMORY_DREAM_CONCURRENCY", "2")),
            scope_batch_size=int(
                os.environ.get("SIDEKICK_MEMORY_DREAM_SCOPE_BATCH_SIZE", "20")
            ),
        )


@dataclass(frozen=True, slots=True)
class DreamScheduleResult:
    scopes_seen: int
    scopes_succeeded: int
    scopes_failed: int
    scopes_busy: int


class DreamScheduler:
    def __init__(
        self,
        *,
        scanner: MemoryDreamRunner,
        store: AIStateRepository,
        identity_codec: IdentityCodec,
        settings: DreamSchedulerSettings = DreamSchedulerSettings(),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        logger: Any | None = None,
    ):
        self._scanner = scanner
        self._store = store
        self._identity_codec = identity_codec
        self._settings = settings
        self._clock = clock
        self._sleep = sleep
        self._logger = logger
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._settings.cron is None or self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run_forever(),
            name="sidekick-memory-dream-scheduler",
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

    async def run_once(self) -> DreamScheduleResult:
        scopes = tuple(
            (scope_id, chat_id)
            for scope_id in await self._store.list_memory_dream_scopes()
            if isinstance(
                chat_id := self._identity_codec.parse_scope_id(scope_id),
                int,
            )
        )
        semaphore = asyncio.Semaphore(self._settings.concurrency)
        succeeded = 0
        failed = 0
        busy = 0

        async def run(scope_id: str, chat_id: int) -> str:
            async with semaphore:
                try:
                    await self._scanner.run_scope(chat_id)
                except MemoryIngestionBusyError:
                    return "busy"
                except Exception as exc:
                    if self._logger is not None:
                        self._logger.warning(
                            "Scheduled Dream Cycle failed "
                            "(scope_id=%s, error=%s): %s",
                            scope_id,
                            type(exc).__name__,
                            exc,
                        )
                    return "failed"
            return "succeeded"

        for start in range(0, len(scopes), self._settings.scope_batch_size):
            results = await asyncio.gather(
                *(
                    run(scope_id, chat_id)
                    for scope_id, chat_id in scopes[
                        start : start + self._settings.scope_batch_size
                    ]
                )
            )
            succeeded += results.count("succeeded")
            failed += results.count("failed")
            busy += results.count("busy")

        result = DreamScheduleResult(
            scopes_seen=len(scopes),
            scopes_succeeded=succeeded,
            scopes_failed=failed,
            scopes_busy=busy,
        )
        if self._logger is not None:
            self._logger.info(
                "Scheduled Dream Cycle complete "
                "(scopes=%s, succeeded=%s, failed=%s, busy=%s)",
                result.scopes_seen,
                result.scopes_succeeded,
                result.scopes_failed,
                result.scopes_busy,
            )
        return result

    async def _run_forever(self) -> None:
        assert self._settings.cron is not None
        while True:
            now = self._clock()
            if now.tzinfo is None:
                now = now.replace(tzinfo=UTC)
            next_run = croniter(self._settings.cron, now).get_next(datetime)
            await self._sleep(max(0.0, (next_run - now).total_seconds()))
            try:
                await self.run_once()
            except Exception as exc:
                if self._logger is not None:
                    self._logger.exception(
                        "Scheduled Dream Cycle orchestration failed "
                        "(error=%s): %s",
                        type(exc).__name__,
                        exc,
                    )
