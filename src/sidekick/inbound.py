from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any, Callable, Generic, Literal, Protocol, TypeVar

from sidekick.ai import ReplyTarget
from sidekick.chat.identity import ExternalId
from sidekick.chat.provenance import MessageOrigin


InboundWorkKind = Literal["message", "message_remove"]
InboundWorkerResult = Literal[
    "idle",
    "completed",
    "ignored",
    "duplicate",
    "recalled",
    "deferred",
    "unavailable",
    "stale",
    "failed",
]
InboundCompletion = Literal["completed", "ignored", "recalled", "failed"]
InboundExecutionStart = Literal["started", "duplicate", "stale"]
InboundDeferral = Literal["pending", "unavailable", "stale"]
InboundSourceState = Literal["present", "recalled"]


class InboundWork(Protocol):
    chat_id: ExternalId
    message_id: ExternalId
    kind: InboundWorkKind
    attempt_count: int
    last_error_code: str | None
    attested_origin: MessageOrigin | None
    lease_id: str | None


class InboundWorkStore(Protocol):
    async def claim_pending_ai_work(
        self,
        source_id: str,
        *,
        now: float | None = None,
    ) -> InboundWork | None: ...

    async def resolve_pending_ai_removal(self, work: InboundWork) -> bool: ...

    async def begin_pending_ai_execution(
        self,
        work: InboundWork,
        *,
        version: str,
        now: float | None = None,
    ) -> InboundExecutionStart: ...

    async def complete_pending_ai_work(
        self,
        work: InboundWork,
        *,
        version: str,
        outcome: InboundCompletion,
        now: float | None = None,
    ) -> bool: ...

    async def defer_pending_ai_work(
        self,
        work: InboundWork,
        *,
        error_code: str,
        retry_at: float,
        max_attempts: int | None,
        now: float | None = None,
    ) -> InboundDeferral: ...

    async def release_pending_ai_work(self, work: InboundWork) -> None: ...

    async def mark_pending_ai_execution_unknown(
        self,
        work: InboundWork,
        *,
        version: str,
        now: float | None = None,
    ) -> bool: ...


class InboundWorkSchedule(Protocol):
    async def next_pending_ai_work_at(
        self,
        source_id: str,
    ) -> float | None: ...


_SourcePayload = TypeVar("_SourcePayload")


@dataclass(frozen=True, slots=True)
class InboundSourceRevision(Generic[_SourcePayload]):
    version: str
    state: InboundSourceState
    payload: _SourcePayload | None = None
    attested_origin: MessageOrigin | None = None

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("Inbound source revision cannot be empty")
        if self.state == "present" and self.payload is None:
            raise ValueError("Present inbound source revision requires a payload")
        if self.state == "recalled" and self.payload is not None:
            raise ValueError("Recalled inbound source revision cannot have a payload")
        if self.attested_origin is not None and not isinstance(
            self.attested_origin,
            MessageOrigin,
        ):
            raise ValueError("Inbound source origin is invalid")


class InboundMessageSource(Protocol[_SourcePayload]):
    async def fetch(
        self,
        work: InboundWork,
    ) -> InboundSourceRevision[_SourcePayload]: ...

    async def materialize(self, payload: _SourcePayload) -> ReplyTarget | None: ...


class InboundSourceUnavailable(Exception):
    def __init__(self, code: str, *, max_attempts: int | None) -> None:
        if not code:
            raise ValueError("Inbound source error code cannot be empty")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("Inbound source attempts must be positive")
        super().__init__(code)
        self.code = code
        self.max_attempts = max_attempts


class InboundMessageHandler(Protocol):
    async def handle(
        self,
        message: ReplyTarget,
        *,
        attested_origin: MessageOrigin | None = None,
    ) -> bool: ...


class InboundWorker(Protocol):
    async def process_one(
        self,
        handler: InboundMessageHandler,
    ) -> InboundWorkerResult: ...


class DurableInboundWorker(Generic[_SourcePayload]):
    RETRY_BASE_SECONDS = 2.0
    RETRY_MAX_SECONDS = 300.0

    def __init__(
        self,
        source: InboundMessageSource[_SourcePayload],
        store: InboundWorkStore,
        source_id: str,
        *,
        clock: Callable[[], float] = time.time,
        logger: Any | None = None,
    ) -> None:
        if not source_id:
            raise ValueError("Inbound source ID cannot be empty")
        self._source = source
        self._store = store
        self._source_id = source_id
        self._clock = clock
        self._logger = logger
        self._active_work: dict[
            tuple[ExternalId, ExternalId], asyncio.Task[Any]
        ] = {}
        self._recall_cancellations: set[asyncio.Task[Any]] = set()

    def cancel_message(
        self,
        chat_id: ExternalId,
        message_id: ExternalId,
    ) -> None:
        task = self._active_work.get((chat_id, message_id))
        if task is None or task.done():
            return
        self._recall_cancellations.add(task)
        task.cancel()

    async def process_one(
        self,
        handler: InboundMessageHandler,
    ) -> InboundWorkerResult:
        work = await self._store.claim_pending_ai_work(
            self._source_id,
            now=self._clock(),
        )
        if work is None:
            return "idle"
        if work.kind == "message_remove":
            resolved = await self._store.resolve_pending_ai_removal(work)
            return "recalled" if resolved else "stale"

        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Inbound work requires an asyncio task")
        work_key = (work.chat_id, work.message_id)
        self._active_work[work_key] = task
        execution_version: str | None = None
        try:
            try:
                revision = await self._source.fetch(work)
            except InboundSourceUnavailable as exc:
                return await self._defer(
                    work,
                    error_code=exc.code,
                    max_attempts=exc.max_attempts,
                )

            begin = await self._store.begin_pending_ai_execution(
                work,
                version=revision.version,
                now=self._clock(),
            )
            if begin == "stale":
                return "stale"
            if begin == "duplicate":
                await self._store.complete_pending_ai_work(
                    work,
                    version=revision.version,
                    outcome="ignored",
                    now=self._clock(),
                )
                return "duplicate"
            execution_version = revision.version
            if revision.state == "recalled":
                await self._store.complete_pending_ai_work(
                    work,
                    version=revision.version,
                    outcome="recalled",
                    now=self._clock(),
                )
                return "recalled"

            assert revision.payload is not None
            try:
                message = await self._source.materialize(revision.payload)
            except Exception as exc:
                await self._store.complete_pending_ai_work(
                    work,
                    version=revision.version,
                    outcome="failed",
                    now=self._clock(),
                )
                self._log_failure("message context", exc, work)
                return "failed"
            if message is None:
                await self._store.complete_pending_ai_work(
                    work,
                    version=revision.version,
                    outcome="ignored",
                    now=self._clock(),
                )
                return "ignored"
            try:
                handled = await handler.handle(
                    message,
                    attested_origin=revision.attested_origin,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._store.complete_pending_ai_work(
                    work,
                    version=revision.version,
                    outcome="failed",
                    now=self._clock(),
                )
                self._log_failure("handler", exc, work)
                return "failed"
            outcome: Literal["completed", "ignored"] = (
                "completed" if handled else "ignored"
            )
            await self._store.complete_pending_ai_work(
                work,
                version=revision.version,
                outcome=outcome,
                now=self._clock(),
            )
            return outcome
        except asyncio.CancelledError:
            if execution_version is None:
                cleanup = self._store.release_pending_ai_work(work)
            else:
                cleanup = self._store.mark_pending_ai_execution_unknown(
                    work,
                    version=execution_version,
                    now=self._clock(),
                )
            await asyncio.shield(cleanup)
            if task in self._recall_cancellations:
                return "stale"
            raise
        finally:
            if self._active_work.get(work_key) is task:
                self._active_work.pop(work_key, None)
            self._recall_cancellations.discard(task)

    async def _defer(
        self,
        work: InboundWork,
        *,
        error_code: str,
        max_attempts: int | None,
    ) -> InboundWorkerResult:
        now = self._clock()
        prior_attempts = (
            work.attempt_count if work.last_error_code == error_code else 0
        )
        status = await self._store.defer_pending_ai_work(
            work,
            error_code=error_code,
            retry_at=now + self._retry_delay(prior_attempts),
            max_attempts=max_attempts,
            now=now,
        )
        if status == "stale":
            return "stale"
        return "unavailable" if status == "unavailable" else "deferred"

    @classmethod
    def _retry_delay(cls, attempt_count: int) -> float:
        return min(
            cls.RETRY_MAX_SECONDS,
            cls.RETRY_BASE_SECONDS * (2 ** min(attempt_count, 16)),
        )

    def _log_failure(
        self,
        stage: str,
        exc: Exception,
        work: InboundWork,
    ) -> None:
        if self._logger is not None:
            self._logger.error(
                "Inbound AI %s failed (%s; source=%s; message=%s)",
                stage,
                type(exc).__name__,
                self._source_id,
                work.message_id,
            )


class DurableInboundPool:
    def __init__(
        self,
        worker: InboundWorker,
        store: InboundWorkSchedule,
        source_id: str,
        handler: InboundMessageHandler,
        *,
        concurrency: int,
        clock: Callable[[], float] = time.time,
        logger: Any | None = None,
    ) -> None:
        if not source_id:
            raise ValueError("Inbound source ID cannot be empty")
        if concurrency < 1:
            raise ValueError("Inbound worker concurrency must be positive")
        self._worker = worker
        self._store = store
        self._source_id = source_id
        self._handler = handler
        self._concurrency = concurrency
        self._clock = clock
        self._logger = logger
        self._work_available = asyncio.Event()
        self._tasks: tuple[asyncio.Task[None], ...] = ()

    def start(self) -> None:
        if self._tasks:
            raise RuntimeError("Inbound worker pool is already running")
        self._tasks = tuple(
            asyncio.create_task(
                self._run_worker(),
                name=f"inbound-ai-worker-{self._source_id}-{index}",
            )
            for index in range(self._concurrency)
        )

    def notify(self) -> None:
        self._work_available.set()

    async def close(self) -> None:
        tasks, self._tasks = self._tasks, ()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_worker(self) -> None:
        while True:
            self._work_available.clear()
            processing = asyncio.create_task(
                self._worker.process_one(self._handler)
            )
            try:
                result = await processing
            except asyncio.CancelledError:
                processing.cancel()
                await asyncio.gather(processing, return_exceptions=True)
                raise
            except Exception as exc:
                self._log_failure(exc)
                result = "idle"
            if result != "idle":
                continue
            next_attempt_at = await self._store.next_pending_ai_work_at(
                self._source_id
            )
            timeout = (
                max(0.0, next_attempt_at - self._clock())
                if next_attempt_at is not None
                else None
            )
            if timeout == 0:
                continue
            try:
                await asyncio.wait_for(
                    self._work_available.wait(),
                    timeout=timeout,
                )
            except TimeoutError:
                pass

    def _log_failure(self, exc: Exception) -> None:
        if self._logger is not None:
            self._logger.error(
                "Inbound AI worker failed (%s; source=%s)",
                type(exc).__name__,
                self._source_id,
                exc_info=True,
            )
