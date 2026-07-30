from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from sidekick.ai import AIStateRepository
from sidekick.ai_memory_ingestion import (
    ContinuousMemoryResult,
    MemoryIngestionBusyError,
)
from sidekick.chat.identity import ExternalId, IdentityCodec


class ContinuousMemoryRunner(Protocol):
    async def run_continuous_scope(
        self,
        chat_id: ExternalId,
    ) -> ContinuousMemoryResult: ...


@dataclass(frozen=True, slots=True)
class ContinuousMemorySchedulerSettings:
    poll_interval_seconds: float = 10
    concurrency: int = 2
    scope_batch_size: int = 20

    def __post_init__(self) -> None:
        if (
            self.poll_interval_seconds <= 0
            or self.concurrency < 1
            or self.scope_batch_size < 1
        ):
            raise ValueError("Continuous memory scheduler limits must be positive")

    @classmethod
    def from_env(cls) -> ContinuousMemorySchedulerSettings:
        return cls(
            poll_interval_seconds=float(
                os.environ.get("SIDEKICK_MEMORY_CONTINUOUS_POLL_SECONDS", "10")
            ),
            concurrency=int(
                os.environ.get("SIDEKICK_MEMORY_CONTINUOUS_CONCURRENCY", "2")
            ),
            scope_batch_size=int(
                os.environ.get("SIDEKICK_MEMORY_CONTINUOUS_SCOPE_BATCH_SIZE", "20")
            ),
        )


@dataclass(frozen=True, slots=True)
class ContinuousMemoryScheduleResult:
    scopes_seen: int
    scopes_succeeded: int
    scopes_failed: int
    scopes_busy: int
    scopes_pending: int
    messages_seen: int
    messages_retained: int


class ContinuousMemoryScheduler:
    def __init__(
        self,
        *,
        runner: ContinuousMemoryRunner,
        store: AIStateRepository,
        identity_codec: IdentityCodec,
        settings: ContinuousMemorySchedulerSettings = (
            ContinuousMemorySchedulerSettings()
        ),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        logger: Any | None = None,
    ):
        self._runner = runner
        self._store = store
        self._identity_codec = identity_codec
        self._settings = settings
        self._sleep = sleep
        self._logger = logger
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run_forever(),
            name="sidekick-continuous-memory-scheduler",
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

    async def run_once(self) -> ContinuousMemoryScheduleResult:
        scopes = tuple(
            (scope_id, chat_id)
            for scope_id in await self._store.list_continuous_memory_scopes()
            if (chat_id := self._identity_codec.parse_scope_id(scope_id)) is not None
        )
        semaphore = asyncio.Semaphore(self._settings.concurrency)
        succeeded = 0
        failed = 0
        busy = 0
        pending = 0
        messages_seen = 0
        messages_retained = 0

        async def run(
            scope_id: str,
            chat_id: ExternalId,
        ) -> tuple[str, ContinuousMemoryResult | None]:
            async with semaphore:
                try:
                    result = await self._runner.run_continuous_scope(chat_id)
                except MemoryIngestionBusyError:
                    return "busy", None
                except Exception as exc:
                    if self._logger is not None:
                        self._logger.warning(
                            "Continuous memory ingestion failed "
                            "(scope_id=%s, error=%s): %s",
                            scope_id,
                            type(exc).__name__,
                            exc,
                        )
                    return "failed", None
            return "succeeded", result

        for start in range(0, len(scopes), self._settings.scope_batch_size):
            results = await asyncio.gather(
                *(
                    run(scope_id, chat_id)
                    for scope_id, chat_id in scopes[
                        start : start + self._settings.scope_batch_size
                    ]
                )
            )
            for status, result in results:
                succeeded += status == "succeeded"
                failed += status == "failed"
                busy += status == "busy"
                if result is None:
                    continue
                messages_seen += result.messages_seen
                messages_retained += result.messages_retained
                pending += not result.caught_up

        schedule_result = ContinuousMemoryScheduleResult(
            scopes_seen=len(scopes),
            scopes_succeeded=succeeded,
            scopes_failed=failed,
            scopes_busy=busy,
            scopes_pending=pending,
            messages_seen=messages_seen,
            messages_retained=messages_retained,
        )
        if self._logger is not None and (
            schedule_result.messages_seen
            or schedule_result.scopes_failed
            or schedule_result.scopes_busy
            or schedule_result.scopes_pending
        ):
            self._logger.info(
                "Continuous memory cycle complete "
                "(scopes=%s, succeeded=%s, failed=%s, busy=%s, pending=%s, "
                "messages=%s, retained=%s)",
                schedule_result.scopes_seen,
                schedule_result.scopes_succeeded,
                schedule_result.scopes_failed,
                schedule_result.scopes_busy,
                schedule_result.scopes_pending,
                schedule_result.messages_seen,
                schedule_result.messages_retained,
            )
        return schedule_result

    async def _run_forever(self) -> None:
        while True:
            try:
                result = await self.run_once()
            except Exception as exc:
                if self._logger is not None:
                    self._logger.exception(
                        "Continuous memory orchestration failed "
                        "(error=%s): %s",
                        type(exc).__name__,
                        exc,
                    )
                result = None
            if result is not None and result.scopes_pending:
                await self._sleep(0)
            else:
                await self._sleep(self._settings.poll_interval_seconds)
