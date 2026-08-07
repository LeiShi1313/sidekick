from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

import aiohttp

from sidekick.ai import (
    AIStateRepository,
    HumanObservation,
    MemoryDreamResult,
    PromptBuilder,
    ReplyTarget,
    _chat_memory_episode,
    _memory_message_text,
    _memory_cursor,
    _message_datetime,
    _record_episode_labels,
)
from sidekick.ai_attachments import attachment_metadata_only, message_has_attachment
from sidekick.chat.commands import (
    MAX_MEMORY_BACKFILL_MESSAGES,
    MemoryBackfillCommand,
)
from sidekick.chat.identity import ExternalId, IdentityCodec
from sidekick.ai_memory import (
    MemoryClient,
    MemoryClientError,
    MemoryDocumentReceipt,
    MemoryEpisode,
    retain_episodes_once,
)
from sidekick.ai_memory_segments import (
    MemorySegmentationSettings,
    PendingMemoryDocument,
    merge_pending_document,
    pending_document_accepts,
    segment_accepts,
)


class MemoryMessageSource(Protocol):
    async def fetch_window(
        self,
        chat_id: ExternalId,
        *,
        since: datetime,
        until: datetime,
        limit: int,
    ) -> tuple[ReplyTarget, ...]: ...

    async def fetch_message(
        self,
        chat_id: ExternalId,
        message_id: ExternalId,
    ) -> ReplyTarget | None: ...

    async def fetch_after(
        self,
        chat_id: ExternalId,
        *,
        after_message_id: ExternalId,
        until: datetime,
        limit: int,
    ) -> tuple[ReplyTarget, ...]: ...


class DreamScanSettings(Protocol):
    lookback: timedelta
    overlap: timedelta
    cycle_budget_seconds: float
    scope_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class MemoryIngestionSettings:
    settlement_delay: timedelta = timedelta(seconds=30)
    max_messages: int = 500
    max_thread_messages: int = 100
    segmentation: MemorySegmentationSettings = MemorySegmentationSettings()
    retain_concurrency: int = 4
    preprocess_concurrency: int = 12
    lease_seconds: float = 3_600
    retry_attempts: int = 3
    max_retry_delay: float = 30
    delivery_max_attempts: int = 5
    delivery_retry_base_delay: float = 30
    delivery_retry_max_delay: float = 3_600
    delivery_cycle_documents: int = 100

    def __post_init__(self) -> None:
        if self.settlement_delay < timedelta(0):
            raise ValueError("Memory settlement delay cannot be negative")
        if (
            self.max_messages < 1
            or self.max_thread_messages < 1
            or self.retain_concurrency < 1
            or self.preprocess_concurrency < 1
        ):
            raise ValueError("Memory ingestion limits must be positive")
        if self.lease_seconds <= 0:
            raise ValueError("Memory ingestion lease duration must be positive")
        if self.retry_attempts < 1 or self.max_retry_delay < 0:
            raise ValueError("Memory ingestion retry settings are invalid")
        if (
            self.delivery_max_attempts < 1
            or self.delivery_retry_base_delay < 0
            or self.delivery_retry_max_delay < self.delivery_retry_base_delay
            or self.delivery_cycle_documents < 1
        ):
            raise ValueError("Memory outbox delivery settings are invalid")

    @classmethod
    def from_env(cls) -> MemoryIngestionSettings:
        return cls(
            settlement_delay=timedelta(
                seconds=float(
                    os.environ.get(
                        "SIDEKICK_MEMORY_INGESTION_SETTLEMENT_SECONDS",
                        "30",
                    )
                )
            ),
            max_messages=int(
                os.environ.get("SIDEKICK_MEMORY_INGESTION_MAX_MESSAGES", "500")
            ),
            max_thread_messages=int(
                os.environ.get(
                    "SIDEKICK_MEMORY_INGESTION_MAX_THREAD_MESSAGES",
                    "100",
                )
            ),
            segmentation=MemorySegmentationSettings.from_env(),
            retain_concurrency=int(
                os.environ.get(
                    "SIDEKICK_MEMORY_INGESTION_RETAIN_CONCURRENCY",
                    "4",
                )
            ),
            preprocess_concurrency=int(
                os.environ.get(
                    "SIDEKICK_MEMORY_INGESTION_PREPROCESS_CONCURRENCY",
                    "12",
                )
            ),
            lease_seconds=float(
                os.environ.get("SIDEKICK_MEMORY_INGESTION_LEASE_SECONDS", "3600")
            ),
            retry_attempts=int(
                os.environ.get("SIDEKICK_MEMORY_INGESTION_RETRY_ATTEMPTS", "3")
            ),
            max_retry_delay=float(
                os.environ.get(
                    "SIDEKICK_MEMORY_INGESTION_MAX_RETRY_DELAY",
                    "30",
                )
            ),
            delivery_max_attempts=int(
                os.environ.get("SIDEKICK_MEMORY_OUTBOX_MAX_ATTEMPTS", "5")
            ),
            delivery_retry_base_delay=float(
                os.environ.get("SIDEKICK_MEMORY_OUTBOX_RETRY_BASE_SECONDS", "30")
            ),
            delivery_retry_max_delay=float(
                os.environ.get("SIDEKICK_MEMORY_OUTBOX_RETRY_MAX_SECONDS", "3600")
            ),
            delivery_cycle_documents=int(
                os.environ.get("SIDEKICK_MEMORY_OUTBOX_CYCLE_DOCUMENTS", "100")
            ),
        )


@dataclass(frozen=True, slots=True)
class ContinuousMemoryResult:
    messages_seen: int
    messages_retained: int
    documents_created: int
    documents_unchanged: int
    caught_up: bool


@dataclass(frozen=True, slots=True)
class MemoryOutboxDrainResult:
    documents_attempted: int = 0
    messages_retained: int = 0
    documents_created: int = 0
    documents_unchanged: int = 0
    documents_failed: int = 0
    documents_dead_lettered: int = 0


@dataclass(frozen=True, slots=True)
class PreparedMemoryDocument:
    episode: MemoryEpisode
    window_message_ids: frozenset[ExternalId]


class MemoryIngestionBusyError(RuntimeError):
    pass


class DreamCycleTimeoutError(TimeoutError):
    pass


class DreamBackfillLimitError(RuntimeError):
    pass


class MemoryThreadLimitError(RuntimeError):
    pass


def _memory_session_document_id(
    identity_codec: IdentityCodec,
    chat_id: ExternalId,
    root_message_id: ExternalId,
    started_at: datetime,
) -> str:
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    stamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{identity_codec.source}:memory-session:{chat_id}:{stamp}:{root_message_id}"


def _memory_session_prefix(
    identity_codec: IdentityCodec,
    chat_id: ExternalId,
) -> str:
    return f"{identity_codec.source}:memory-session:{chat_id}:"


def _legacy_memory_session_prefix(
    identity_codec: IdentityCodec,
    chat_id: ExternalId,
) -> str:
    return f"{identity_codec.source}:dream-session:{chat_id}:"


def _is_memory_session_document(
    identity_codec: IdentityCodec,
    document_id: str,
) -> bool:
    return document_id.startswith(
        (
            f"{identity_codec.source}:memory-session:",
            f"{identity_codec.source}:dream-session:",
        )
    )


def _session_document_sort_key(document_id: str) -> tuple[str, str]:
    parts = document_id.rsplit(":", 2)
    started_at = parts[-2] if len(parts) == 3 else ""
    return started_at, document_id


def _completed_message_prefix(
    messages: tuple[ReplyTarget, ...],
    documents: tuple[PreparedMemoryDocument, ...],
    completed_document_ids: set[str],
) -> tuple[ReplyTarget, ...]:
    owner_by_message_id = {
        message_id: document.episode.document_id
        for document in documents
        for message_id in document.window_message_ids
    }
    completed: list[ReplyTarget] = []
    for message in messages:
        document_id = owner_by_message_id.get(message.id)
        if document_id is not None and document_id not in completed_document_ids:
            break
        completed.append(message)
    return tuple(completed)


class ChatMemoryIngestor:
    def __init__(
        self,
        *,
        source: MemoryMessageSource,
        store: AIStateRepository,
        memory: MemoryClient,
        prompt_builder: PromptBuilder,
        dream_settings: DreamScanSettings,
        ingestion_settings: MemoryIngestionSettings = MemoryIngestionSettings(),
        identity_codec: IdentityCodec | None = None,
        source_retry_delay: Callable[[Exception], float | None] | None = None,
        album_document_id: (
            Callable[[ExternalId, ReplyTarget], str | None] | None
        ) = None,
        clock: Any = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        logger: Any | None = None,
    ):
        self._source = source
        self._store = store
        self._memory = memory
        self._prompt_builder = prompt_builder
        self._dream_settings = dream_settings
        self._ingestion_settings = ingestion_settings
        self._identity_codec = identity_codec or prompt_builder.identity_codec
        self._source_retry_delay = source_retry_delay
        self._album_document_id = album_document_id
        self._clock = clock
        self._sleep = sleep
        self._monotonic = monotonic
        self._logger = logger
        self._locks: dict[str, asyncio.Lock] = {}
        self._lease_owner = uuid4().hex

    @property
    def identity_codec(self) -> IdentityCodec:
        return self._identity_codec

    async def run_scope(self, chat_id: ExternalId) -> MemoryDreamResult:
        try:
            return await self._run_bounded(
                lambda: self._run_exclusive(
                    chat_id,
                    lambda: self._run_scope(chat_id),
                ),
                timeout_seconds=self._dream_settings.scope_timeout_seconds,
            )
        except DreamCycleTimeoutError as exc:
            await self._store.record_memory_dream_failure(
                self._identity_codec.scope_id(chat_id),
                failed_at=self._clock(),
                error=type(exc).__name__,
            )
            raise

    async def _run_bounded(
        self,
        operation: Callable[[], Awaitable[MemoryDreamResult]],
        *,
        timeout_seconds: float,
    ) -> MemoryDreamResult:
        work = asyncio.create_task(operation())
        timeout = asyncio.create_task(asyncio.sleep(timeout_seconds))
        tasks: set[asyncio.Task[Any]] = {work, timeout}
        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if work in done:
                return await work
            raise DreamCycleTimeoutError(
                f"Dream Cycle exceeded its {timeout_seconds:g}-second scope timeout"
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run_backfill(
        self,
        chat_id: ExternalId,
        request: MemoryBackfillCommand,
    ) -> MemoryDreamResult:
        return await self._run_exclusive(
            chat_id,
            lambda: self._run_backfill(chat_id, request),
        )

    async def run_continuous_scope(
        self,
        chat_id: ExternalId,
    ) -> ContinuousMemoryResult:
        return await self._run_exclusive(
            chat_id,
            lambda: self._run_continuous_scope(chat_id),
        )

    async def run_memory_outbox_scope(
        self,
        scope_id: str,
    ) -> MemoryOutboxDrainResult:
        return await self._run_scope_exclusive(
            scope_id,
            lambda: self._drain_memory_outbox(scope_id),
        )

    async def _run_exclusive(
        self,
        chat_id: ExternalId,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        return await self._run_scope_exclusive(
            self._identity_codec.scope_id(chat_id),
            operation,
        )

    async def _run_scope_exclusive(
        self,
        scope_id: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        lock = self._locks.setdefault(scope_id, asyncio.Lock())
        async with lock:
            acquired_at = self._clock()
            acquired = await self._store.acquire_memory_dream_lease(
                scope_id,
                owner=self._lease_owner,
                acquired_at=acquired_at,
                lease_seconds=self._ingestion_settings.lease_seconds,
            )
            if not acquired:
                raise MemoryIngestionBusyError(
                    "Another memory ingestion operation is already running for this chat"
                )
            work = asyncio.create_task(operation())
            heartbeat = asyncio.create_task(self._renew_lease(scope_id))
            tasks: set[asyncio.Task[Any]] = {work, heartbeat}
            try:
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if heartbeat in done:
                    await heartbeat
                    raise AssertionError(
                        "Memory ingestion lease heartbeat stopped unexpectedly"
                    )
                return await work
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await self._store.release_memory_dream_lease(
                    scope_id,
                    owner=self._lease_owner,
                )

    async def _run_continuous_scope(
        self,
        chat_id: ExternalId,
    ) -> ContinuousMemoryResult:
        scope_id = self._identity_codec.scope_id(chat_id)
        scope = await self._store.get_memory_scope_state(scope_id)
        if not scope.continuous_enabled:
            raise ValueError("Continuous memory is disabled for this chat")
        if scope.continuous_cursor_message_id is None:
            raise ValueError("Continuous memory cursor is not initialized")
        attempted_at = self._clock()
        await self._store.record_continuous_memory_attempt(scope_id, attempted_at)
        until = (
            datetime.fromtimestamp(attempted_at, UTC)
            - self._ingestion_settings.settlement_delay
        )

        try:
            earlier = await self._drain_memory_outbox(scope_id)
            # Back off source scanning while a retryable sealed document remains.
            # Dead letters are explicitly excluded, so one poison document never
            # prevents later chat history from progressing.
            if await self._store.has_retryable_memory_outbox_documents(scope_id):
                return ContinuousMemoryResult(
                    messages_seen=0,
                    messages_retained=earlier.messages_retained,
                    documents_created=earlier.documents_created,
                    documents_unchanged=earlier.documents_unchanged,
                    caught_up=True,
                )
            pending = tuple(
                item.document
                for item in await self._store.list_memory_outbox_documents(
                    scope_id,
                    pipeline="continuous",
                )
                if not item.document.sealed
            )
            messages = await self._retry_source(
                lambda: self._source.fetch_after(
                    chat_id,
                    after_message_id=scope.continuous_cursor_message_id,
                    until=until,
                    limit=self._ingestion_settings.max_messages,
                )
            )
            prepared = await self._prepare_documents(chat_id, messages)
            pending = self._stage_continuous_documents(
                chat_id,
                pending,
                prepared,
            )
            pending = self._seal_due_pending_documents(
                pending,
                watermark=until,
            )
            # Persist the boundary before the remote write so retries cannot reopen a
            # document that was already quiet (or full) when retention failed.
            scanned_until_at = (
                _message_datetime(messages[-1]).timestamp()
                if len(messages) >= self._ingestion_settings.max_messages
                else until.timestamp()
            )
            await self._store.stage_continuous_memory_scan(
                scope_id,
                pending,
                cursor_message_id=_memory_cursor(messages[-1]) if messages else None,
                scanned_until_at=scanned_until_at,
                succeeded_at=self._clock(),
            )
            delivered = await self._drain_memory_outbox(scope_id)
            return ContinuousMemoryResult(
                messages_seen=len(messages),
                messages_retained=(
                    earlier.messages_retained + delivered.messages_retained
                ),
                documents_created=(
                    earlier.documents_created + delivered.documents_created
                ),
                documents_unchanged=(
                    earlier.documents_unchanged + delivered.documents_unchanged
                ),
                caught_up=len(messages) < self._ingestion_settings.max_messages,
            )
        except Exception as exc:
            await self._store.record_continuous_memory_failure(
                scope_id,
                failed_at=self._clock(),
                error=type(exc).__name__,
            )
            raise

    def _stage_continuous_documents(
        self,
        chat_id: ExternalId,
        pending: tuple[PendingMemoryDocument, ...],
        prepared: tuple[PreparedMemoryDocument, ...],
    ) -> tuple[PendingMemoryDocument, ...]:
        by_id = {document.episode.document_id: document for document in pending}
        ordered_ids = [document.episode.document_id for document in pending]
        open_session_id: str | None = None
        for document in pending:
            document_id = document.episode.document_id
            if document.sealed or not self._is_session_document(document_id):
                continue
            if open_session_id is not None:
                previous = by_id[open_session_id]
                by_id[open_session_id] = PendingMemoryDocument(
                    episode=previous.episode,
                    staged_source_ids=previous.staged_source_ids,
                    sealed=True,
                )
            open_session_id = document_id

        for prepared_document in prepared:
            episode = prepared_document.episode
            window_source_ids = {
                self._identity_codec.message_source_id(chat_id, message_id)
                for message_id in prepared_document.window_message_ids
            }
            staged_source_ids = tuple(
                event.source_id
                for event in episode.events
                if event.source_id in window_source_ids
            )
            existing = by_id.get(episode.document_id)
            if existing is not None:
                by_id[episode.document_id] = merge_pending_document(
                    existing,
                    episode,
                    staged_source_ids,
                )
                if self._is_session_document(episode.document_id):
                    open_session_id = episode.document_id
                continue

            if self._is_session_document(episode.document_id):
                open_session = (
                    by_id.get(open_session_id) if open_session_id is not None else None
                )
                if (
                    open_session is not None
                    and not open_session.sealed
                    and pending_document_accepts(
                        open_session,
                        episode,
                        self._ingestion_settings.segmentation,
                    )
                ):
                    by_id[open_session_id] = merge_pending_document(
                        open_session,
                        episode,
                        staged_source_ids,
                    )
                    continue
                if open_session is not None:
                    by_id[open_session_id] = PendingMemoryDocument(
                        episode=open_session.episode,
                        staged_source_ids=open_session.staged_source_ids,
                        sealed=True,
                    )
                open_session_id = episode.document_id

            by_id[episode.document_id] = PendingMemoryDocument(
                episode=episode,
                staged_source_ids=staged_source_ids,
            )
            ordered_ids.append(episode.document_id)

        return tuple(by_id[document_id] for document_id in ordered_ids)

    def _sealed_outbox_documents(
        self,
        chat_id: ExternalId,
        prepared: tuple[PreparedMemoryDocument, ...],
    ) -> tuple[PendingMemoryDocument, ...]:
        documents: list[PendingMemoryDocument] = []
        for prepared_document in prepared:
            window_source_ids = {
                self._identity_codec.message_source_id(chat_id, message_id)
                for message_id in prepared_document.window_message_ids
            }
            documents.append(
                PendingMemoryDocument(
                    episode=prepared_document.episode,
                    staged_source_ids=tuple(
                        event.source_id
                        for event in prepared_document.episode.events
                        if event.source_id in window_source_ids
                    ),
                    sealed=True,
                )
            )
        return tuple(documents)

    def _seal_due_pending_documents(
        self,
        pending: tuple[PendingMemoryDocument, ...],
        *,
        watermark: datetime,
    ) -> tuple[PendingMemoryDocument, ...]:
        return tuple(
            (
                PendingMemoryDocument(
                    episode=document.episode,
                    staged_source_ids=document.staged_source_ids,
                    sealed=True,
                )
                if not document.sealed
                and watermark - document.last_event_at
                >= self._ingestion_settings.segmentation.idle_gap
                else document
            )
            for document in pending
        )

    async def _drain_memory_outbox(
        self,
        scope_id: str,
        *,
        deadline: float | None = None,
    ) -> MemoryOutboxDrainResult:
        documents_attempted = 0
        messages_retained = 0
        documents_created = 0
        documents_unchanged = 0
        documents_failed = 0
        documents_dead_lettered = 0
        while (
            documents_attempted
            < self._ingestion_settings.delivery_cycle_documents
        ):
            if deadline is not None and self._monotonic() >= deadline:
                break
            batch = await self._store.list_due_memory_outbox_documents(
                scope_id,
                due_at=self._clock(),
                limit=min(
                    self._ingestion_settings.retain_concurrency,
                    self._ingestion_settings.delivery_cycle_documents
                    - documents_attempted,
                ),
            )
            if not batch:
                break
            results = await asyncio.gather(
                *(
                    self._retry_memory(
                        lambda item=item: retain_episodes_once(
                            self._memory,
                            self._store,
                            (item.document.episode,),
                        )
                    )
                    for item in batch
                ),
                return_exceptions=True,
            )
            documents_attempted += len(batch)
            completed: list[tuple[str, float]] = []
            for item, created in zip(batch, results, strict=True):
                if isinstance(created, BaseException):
                    if not isinstance(created, Exception):
                        raise created
                    failed_at = self._clock()
                    attempt_count = item.attempt_count + 1
                    dead_lettered_at = (
                        failed_at
                        if attempt_count
                        >= self._ingestion_settings.delivery_max_attempts
                        else None
                    )
                    retry_delay = min(
                        self._ingestion_settings.delivery_retry_base_delay
                        * (2 ** min(attempt_count - 1, 30)),
                        self._ingestion_settings.delivery_retry_max_delay,
                    )
                    await self._store.record_memory_outbox_failure(
                        scope_id,
                        item.document.episode.document_id,
                        attempt_count=attempt_count,
                        attempted_at=failed_at,
                        next_attempt_at=failed_at + retry_delay,
                        dead_lettered_at=dead_lettered_at,
                        error=type(created).__name__,
                    )
                    documents_failed += 1
                    documents_dead_lettered += dead_lettered_at is not None
                    if self._logger is not None:
                        self._logger.warning(
                            "Memory outbox delivery failed "
                            "(attempt=%s, dead_lettered=%s, error=%s)",
                            attempt_count,
                            dead_lettered_at is not None,
                            type(created).__name__,
                        )
                    continue
                completed.append(
                    (
                        item.document.episode.document_id,
                        item.document.last_event_at.timestamp(),
                    )
                )
                messages_retained += len(item.document.staged_source_ids)
                documents_created += int(created[0])
                documents_unchanged += int(not created[0])
            await self._store.complete_memory_outbox_documents(
                scope_id,
                tuple(completed),
                retained_at=self._clock(),
            )
        return MemoryOutboxDrainResult(
            documents_attempted=documents_attempted,
            messages_retained=messages_retained,
            documents_created=documents_created,
            documents_unchanged=documents_unchanged,
            documents_failed=documents_failed,
            documents_dead_lettered=documents_dead_lettered,
        )

    def _is_session_document(self, document_id: str) -> bool:
        return _is_memory_session_document(
            self._identity_codec,
            document_id,
        )

    async def _latest_session_receipt(
        self,
        scope_id: str,
        chat_id: ExternalId,
    ) -> tuple[str, MemoryDocumentReceipt] | None:
        candidates = await asyncio.gather(
            self._store.get_latest_memory_document_receipt(
                scope_id,
                _memory_session_prefix(self._identity_codec, chat_id),
            ),
            self._store.get_latest_memory_document_receipt(
                scope_id,
                _legacy_memory_session_prefix(self._identity_codec, chat_id),
            ),
        )
        prefixes = (
            _memory_session_prefix(self._identity_codec, chat_id),
            _legacy_memory_session_prefix(self._identity_codec, chat_id),
        )
        outbox_candidates = tuple(
            (
                item.document.episode.document_id,
                MemoryDocumentReceipt(
                    content_hash=item.document.episode.content_hash,
                    event_versions=item.document.episode.event_versions,
                ),
            )
            for item in await self._store.list_memory_outbox_documents(scope_id)
            if item.document.episode.document_id.startswith(prefixes)
        )
        available = (
            *(candidate for candidate in candidates if candidate is not None),
            *outbox_candidates,
        )
        if not available:
            return None
        return max(
            available,
            key=lambda candidate: _session_document_sort_key(candidate[0]),
        )

    async def _renew_lease(self, scope_id: str) -> None:
        interval = max(0.05, self._ingestion_settings.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            renewed_at = self._clock()
            if not await self._store.renew_memory_dream_lease(
                scope_id,
                owner=self._lease_owner,
                renewed_at=renewed_at,
                lease_seconds=self._ingestion_settings.lease_seconds,
            ):
                raise MemoryIngestionBusyError("Memory ingestion lease was lost")

    async def _run_backfill(
        self,
        chat_id: ExternalId,
        request: MemoryBackfillCommand,
    ) -> MemoryDreamResult:
        until = (
            datetime.fromtimestamp(self._clock(), UTC)
            - self._ingestion_settings.settlement_delay
        )
        if request.mode == "days":
            since = until - timedelta(days=request.value)
            limit = MAX_MEMORY_BACKFILL_MESSAGES + 1
        else:
            since = datetime.min.replace(tzinfo=UTC)
            limit = request.value
        messages = await self._retry_source(
            lambda: self._source.fetch_window(
                chat_id,
                since=since,
                until=until,
                limit=limit,
            )
        )
        if request.mode == "days" and len(messages) > MAX_MEMORY_BACKFILL_MESSAGES:
            raise DreamBackfillLimitError(
                "Memory backfill exceeds the 5,000-message limit; use message mode "
                "or request a shorter day range"
            )
        result, _, _ = await self._retain_threads(chat_id, messages)
        return result

    async def _run_scope(self, chat_id: ExternalId) -> MemoryDreamResult:
        deadline = self._monotonic() + self._dream_settings.cycle_budget_seconds
        scope_id = self._identity_codec.scope_id(chat_id)
        scope = await self._store.get_memory_scope_state(scope_id)
        if scope.continuous_enabled:
            raise ValueError("Continuous memory overrides Dream for this chat")
        if not scope.dream_enabled:
            raise ValueError("Dream is disabled for this chat")
        attempted_at = self._clock()
        await self._store.record_memory_dream_attempt(scope_id, attempted_at)
        state = await self._store.get_memory_dream_state(scope_id)
        until = (
            datetime.fromtimestamp(attempted_at, UTC)
            - self._ingestion_settings.settlement_delay
        )
        since = (
            datetime.fromtimestamp(state.scanned_until_at, UTC)
            - self._dream_settings.overlap
            if state.scanned_until_at is not None
            else until - self._dream_settings.lookback
        )

        try:
            messages = await self._retry_source(
                lambda: self._source.fetch_window(
                    chat_id,
                    since=since,
                    until=until,
                    limit=self._ingestion_settings.max_messages,
                )
            )
            documents = await self._prepare_documents(chat_id, messages)
            pending = self._sealed_outbox_documents(chat_id, documents)
            scanned_until_at = until.timestamp()
            # The source checkpoint and sealed documents commit together before
            # Hindsight is called. A remote outage can delay retention, but it can
            # no longer force Dream to rescan or block later history.
            await self._store.stage_dream_memory_scan(
                scope_id,
                pending,
                cursor_message_id=_memory_cursor(messages[-1]) if messages else None,
                scanned_until_at=scanned_until_at,
                succeeded_at=self._clock(),
            )
            delivered = await self._drain_memory_outbox(
                scope_id,
                deadline=deadline,
            )
            result = MemoryDreamResult(
                messages_seen=len(messages),
                messages_retained=delivered.messages_retained,
                documents_created=delivered.documents_created,
                documents_unchanged=delivered.documents_unchanged,
            )
            if self._logger is not None and result.messages_seen:
                self._logger.info(
                    "Dream scan staged "
                    "(messages=%s, documents=%s, retained=%s, "
                    "failed=%s, dead_lettered=%s)",
                    result.messages_seen,
                    len(pending),
                    delivered.documents_created
                    + delivered.documents_unchanged,
                    delivered.documents_failed,
                    delivered.documents_dead_lettered,
                )
            return result
        except Exception as exc:
            await self._store.record_memory_dream_failure(
                scope_id,
                failed_at=self._clock(),
                error=type(exc).__name__,
            )
            raise

    async def _retain_threads(
        self,
        chat_id: ExternalId,
        messages: tuple[ReplyTarget, ...],
        *,
        deadline: float | None = None,
        checkpoint: Callable[[ExternalId | None, float], Awaitable[None]] | None = None,
    ) -> tuple[MemoryDreamResult, ExternalId | None, bool]:
        started_at = self._monotonic()
        documents = await self._prepare_documents(chat_id, messages)
        prepared_at = self._monotonic()
        documents_created = 0
        documents_unchanged = 0
        retained_window_ids: set[ExternalId] = set()
        completed_document_ids: set[str] = set()
        complete = True
        cursor: ExternalId | None = None
        for start in range(
            0,
            len(documents),
            self._ingestion_settings.retain_concurrency,
        ):
            batch = documents[
                start : start + self._ingestion_settings.retain_concurrency
            ]
            results = await asyncio.gather(
                *(
                    self._retry_memory(
                        lambda document=document: retain_episodes_once(
                            self._memory,
                            self._store,
                            (document.episode,),
                        )
                    )
                    for document in batch
                ),
                return_exceptions=True,
            )
            first_error: BaseException | None = None
            for document, created in zip(batch, results, strict=True):
                if isinstance(created, BaseException):
                    if first_error is None:
                        first_error = created
                    continue
                documents_created += int(created[0])
                documents_unchanged += int(not created[0])
                retained_window_ids.update(document.window_message_ids)
                completed_document_ids.add(document.episode.document_id)
            if checkpoint is not None:
                prefix = _completed_message_prefix(
                    messages,
                    documents,
                    completed_document_ids,
                )
                if prefix:
                    cursor = _memory_cursor(prefix[-1])
                    await checkpoint(
                        cursor,
                        _message_datetime(prefix[-1]).timestamp(),
                    )
            if first_error is not None:
                raise first_error
            if (
                deadline is not None
                and start + len(batch) < len(documents)
                and self._monotonic() >= deadline
            ):
                complete = False
                break
        if complete and messages:
            cursor = _memory_cursor(messages[-1])
            if not documents and checkpoint is not None:
                await checkpoint(
                    cursor,
                    _message_datetime(messages[-1]).timestamp(),
                )
        result = MemoryDreamResult(
            messages_seen=len(messages),
            messages_retained=len(retained_window_ids),
            documents_created=documents_created,
            documents_unchanged=documents_unchanged,
        )
        if self._logger is not None and result.messages_seen:
            finished_at = self._monotonic()
            self._logger.info(
                "Dream retention complete "
                "(messages=%s, documents=%s, created=%s, "
                "unchanged=%s, complete=%s, prepare_seconds=%.3f, "
                "retain_seconds=%.3f)",
                result.messages_seen,
                len(documents),
                result.documents_created,
                result.documents_unchanged,
                complete,
                prepared_at - started_at,
                finished_at - prepared_at,
            )
        return result, cursor, complete

    async def _prepare_documents(
        self,
        chat_id: ExternalId,
        messages: tuple[ReplyTarget, ...],
    ) -> tuple[PreparedMemoryDocument, ...]:
        window_ids = {message.id for message in messages}
        window_positions = {message.id: index for index, message in enumerate(messages)}
        known = {message.id: message for message in messages}
        root_groups: dict[ExternalId, dict[ExternalId, ReplyTarget]] = {}
        fixed_document_ids: dict[ExternalId, str] = {}
        album_roots: dict[int, ExternalId] = {}
        for message in messages:
            chain = await self._load_chain(chat_id, message, known)
            if not chain:
                continue
            channel_document_id = (
                self._album_document_id(chat_id, message)
                if len(chain) == 1 and self._album_document_id is not None
                else None
            )
            grouped_id = getattr(message, "grouped_id", None)
            if channel_document_id is not None and isinstance(grouped_id, int):
                root_id = album_roots.setdefault(grouped_id, message.id)
            else:
                root_id = chain[0].id
            if channel_document_id is not None:
                fixed_document_ids[root_id] = channel_document_id
            group = root_groups.setdefault(root_id, {})
            for item in chain:
                group[item.id] = item

        scope_id = self._identity_codec.scope_id(chat_id)
        source_ids = tuple(
            self._identity_codec.message_source_id(chat_id, message_id)
            for grouped in root_groups.values()
            for message_id in grouped
        )
        source_documents = await self._store.find_memory_document_ids_for_sources(
            scope_id,
            source_ids,
        )

        assigned_documents: dict[ExternalId, str] = {}
        for root_id, grouped in root_groups.items():
            root_source_id = self._identity_codec.message_source_id(
                chat_id,
                root_id,
            )
            document_id = source_documents.get(root_source_id)
            if document_id is None:
                for message_id in grouped:
                    source_id = self._identity_codec.message_source_id(
                        chat_id,
                        message_id,
                    )
                    document_id = source_documents.get(source_id)
                    if document_id is not None:
                        break
            if document_id is None:
                document_id = fixed_document_ids.get(root_id)
            if document_id is not None:
                assigned_documents[root_id] = document_id

        assigned_document_ids = tuple(dict.fromkeys(assigned_documents.values()))
        receipts = await self._store.get_memory_document_receipts(
            scope_id,
            assigned_document_ids,
        )
        assigned_ids = set(assigned_document_ids)
        for item in await self._store.list_memory_outbox_documents(scope_id):
            episode = item.document.episode
            if episode.document_id in assigned_ids:
                receipts[episode.document_id] = MemoryDocumentReceipt(
                    content_hash=episode.content_hash,
                    event_versions=episode.event_versions,
                )

        document_groups: dict[str, dict[ExternalId, ReplyTarget]] = {}
        for root_id, document_id in assigned_documents.items():
            grouped = root_groups[root_id]
            if len(grouped) > self._ingestion_settings.max_thread_messages:
                raise MemoryThreadLimitError(
                    f"Thread {root_id} exceeds the configured memory thread bound"
                )
            document_group = document_groups.setdefault(document_id, {})
            document_group.update(grouped)

        unassigned_root_ids = tuple(
            root_id for root_id in root_groups if root_id not in assigned_documents
        )
        open_document_id: str | None = None
        candidate_document_id: str | None = None
        loaded_candidate_only = False
        if unassigned_root_ids:
            latest = await self._latest_session_receipt(scope_id, chat_id)
            if latest is not None:
                open_document_id, receipt = latest
                candidate_document_id = open_document_id
                receipts[open_document_id] = receipt
                loaded_candidate_only = open_document_id not in document_groups
                document_groups.setdefault(open_document_id, {})

        await self._hydrate_previous_events(
            chat_id,
            document_groups,
            receipts,
            known,
        )
        observation_by_id = await self._build_observations(
            chat_id,
            tuple(
                sorted(
                    {
                        message.id: message
                        for group in (
                            *document_groups.values(),
                            *(root_groups[root_id] for root_id in unassigned_root_ids),
                        )
                        for message in group.values()
                    }.values(),
                    key=lambda message: (_message_datetime(message), message.id),
                )
            ),
        )

        open_observations = (
            tuple(
                observation_by_id[message_id]
                for message_id in document_groups[open_document_id]
                if message_id in observation_by_id
            )
            if open_document_id is not None
            else ()
        )
        candidate_appended = False
        unassigned_roots: list[
            tuple[
                ExternalId,
                dict[ExternalId, ReplyTarget],
                tuple[HumanObservation, ...],
            ]
        ] = []
        for root_id in unassigned_root_ids:
            grouped = root_groups[root_id]
            if len(grouped) > self._ingestion_settings.max_thread_messages:
                raise MemoryThreadLimitError(
                    f"Thread {root_id} exceeds the configured memory thread bound"
                )
            observations = tuple(
                sorted(
                    (
                        observation_by_id[message_id]
                        for message_id in grouped
                        if message_id in observation_by_id
                    ),
                    key=lambda observation: (
                        observation.occurred_at,
                        observation.message_id,
                    ),
                )
            )
            if observations:
                unassigned_roots.append((root_id, grouped, observations))
        unassigned_roots.sort(key=lambda item: (item[2][0].occurred_at, str(item[0])))

        for root_id, grouped, observations in unassigned_roots:
            if open_document_id is not None and segment_accepts(
                open_observations,
                observations,
                self._ingestion_settings.segmentation,
            ):
                document_id = open_document_id
                candidate_appended = (
                    candidate_appended or document_id == candidate_document_id
                )
            else:
                document_id = _memory_session_document_id(
                    self._identity_codec,
                    chat_id,
                    root_id,
                    observations[0].occurred_at,
                )
                open_document_id = document_id
                open_observations = ()
            document_groups.setdefault(document_id, {}).update(grouped)
            open_observations = tuple(
                {
                    observation.message_id: observation
                    for observation in (*open_observations, *observations)
                }.values()
            )

        if loaded_candidate_only and not candidate_appended:
            assert candidate_document_id is not None
            document_groups.pop(candidate_document_id, None)

        documents: list[PreparedMemoryDocument] = []
        for document_id, grouped in document_groups.items():
            ordered = sorted(
                grouped.values(),
                key=lambda message: (_message_datetime(message), message.id),
            )
            if (
                document_id.startswith(f"{self._identity_codec.source}:thread:")
                and len(ordered) > self._ingestion_settings.max_thread_messages
            ):
                raise MemoryThreadLimitError(
                    f"Document {document_id} exceeds the memory thread bound"
                )
            observations = tuple(
                observation_by_id[message.id]
                for message in ordered
                if message.id in observation_by_id
            )
            if not observations:
                continue
            episode = _chat_memory_episode(
                self._identity_codec,
                chat_id,
                observations,
                document_id=document_id,
            )
            await _record_episode_labels(self._store, episode)
            window_message_ids = frozenset(
                observation.message_id
                for observation in observations
                if observation.message_id in window_ids
            )
            if not window_message_ids:
                continue
            documents.append(
                PreparedMemoryDocument(
                    episode=episode,
                    window_message_ids=window_message_ids,
                )
            )
        documents.sort(
            key=lambda document: min(
                (
                    window_positions[message_id]
                    for message_id in document.window_message_ids
                ),
                default=len(messages),
            )
        )
        return tuple(documents)

    async def _hydrate_previous_events(
        self,
        chat_id: ExternalId,
        document_groups: dict[str, dict[ExternalId, ReplyTarget]],
        receipts: dict[str, MemoryDocumentReceipt],
        known: dict[ExternalId, ReplyTarget],
    ) -> None:
        previous_by_document: dict[str, tuple[ExternalId, ...]] = {}
        missing_ids: set[ExternalId] = set()
        for document_id, grouped in document_groups.items():
            receipt = receipts.get(document_id)
            if receipt is None:
                continue
            previous_ids: list[ExternalId] = []
            for source_id, _ in receipt.event_versions:
                parsed = self._identity_codec.parse_message_source_id(source_id)
                if parsed is None:
                    continue
                source_chat_id, message_id = parsed
                if source_chat_id != chat_id:
                    continue
                if message_id not in grouped:
                    previous_ids.append(message_id)
                    if message_id not in known:
                        missing_ids.add(message_id)
            if document_id.startswith(f"{self._identity_codec.source}:thread:"):
                limit = self._ingestion_settings.max_thread_messages
            elif _is_memory_session_document(self._identity_codec, document_id):
                limit = max(
                    self._ingestion_settings.max_thread_messages,
                    self._ingestion_settings.segmentation.max_events,
                )
            else:
                # Legacy ID-packed documents can be larger than new sessions.
                limit = max(
                    self._ingestion_settings.max_messages,
                    self._ingestion_settings.max_thread_messages,
                )
            if len(grouped) + len(previous_ids) > limit:
                raise MemoryThreadLimitError(
                    f"Document {document_id} exceeds its configured memory bound"
                )
            previous_by_document[document_id] = tuple(previous_ids)

        semaphore = asyncio.Semaphore(self._ingestion_settings.preprocess_concurrency)

        async def fetch(message_id: ExternalId) -> ReplyTarget | None:
            async with semaphore:
                return await self._retry_source(
                    lambda: self._source.fetch_message(chat_id, message_id)
                )

        if missing_ids:
            fetched = await asyncio.gather(
                *(fetch(message_id) for message_id in sorted(missing_ids, key=str))
            )
            for message in fetched:
                if message is not None:
                    known[message.id] = message

        for document_id, previous_ids in previous_by_document.items():
            grouped = document_groups[document_id]
            for message_id in previous_ids:
                message = known.get(message_id)
                if message is not None:
                    grouped[message.id] = message

    async def _build_observations(
        self,
        chat_id: ExternalId,
        messages: tuple[ReplyTarget, ...],
    ) -> dict[ExternalId, HumanObservation]:
        message_ids = tuple(message.id for message in messages)
        scope_id = self._identity_codec.scope_id(chat_id)
        excluded_ids, answer_ids = await asyncio.gather(
            self._store.get_memory_excluded_message_ids(scope_id, message_ids),
            self._store.get_ai_answer_message_ids(scope_id, message_ids),
        )
        semaphore = asyncio.Semaphore(self._ingestion_settings.preprocess_concurrency)
        identity_semaphore = asyncio.Semaphore(
            self._ingestion_settings.preprocess_concurrency
        )
        identity_tasks: dict[ExternalId, asyncio.Task[Any]] = {}
        album_attachment_representatives: dict[int, ExternalId] = {}
        for message in messages:
            grouped_id = getattr(message, "grouped_id", None)
            if (
                bool(getattr(message, "post", False))
                and isinstance(grouped_id, int)
                and message_has_attachment(message)
            ):
                album_attachment_representatives.setdefault(grouped_id, message.id)

        async def resolve_identity(message: ReplyTarget) -> Any:
            async with identity_semaphore:
                return await self._prompt_builder.resolve_identity(message)

        def identity_for(message: ReplyTarget) -> asyncio.Task[Any]:
            assert message.sender_id is not None
            task = identity_tasks.get(message.sender_id)
            if task is None:
                task = asyncio.create_task(resolve_identity(message))
                identity_tasks[message.sender_id] = task
            return task

        async def build(message: ReplyTarget) -> HumanObservation | None:
            if (
                message.sender_id is None
                or message.id in excluded_ids
                or message.id in answer_ids
            ):
                return None
            async with semaphore:
                text = _memory_message_text(message.raw_text or "")
                grouped_id = getattr(message, "grouped_id", None)
                representative_id = (
                    album_attachment_representatives.get(grouped_id)
                    if isinstance(grouped_id, int)
                    else None
                )
                if representative_id is not None and representative_id != message.id:
                    attachment = attachment_metadata_only(
                        message,
                        reason="another item in this media album was analyzed",
                    )
                else:
                    attachment = await self._prompt_builder.describe_attachment(message)
                observation_text = self._prompt_builder.build_observation_text(
                    text,
                    attachment,
                )
                if not observation_text:
                    return None
                identity = await identity_for(message)
                if not identity.is_memory_source:
                    return None
                mentioned_users = await self._prompt_builder.resolve_mentions(message)
                return HumanObservation(
                    message_id=message.id,
                    sender_id=message.sender_id,
                    text=observation_text,
                    occurred_at=_message_datetime(message),
                    mentioned_at=_message_datetime(message),
                    identity=identity,
                    reply_to_message_id=message.reply_to_msg_id,
                    mentioned_users=mentioned_users,
                    metadata=self._prompt_builder.resolve_metadata(message),
                )

        observations = await asyncio.gather(*(build(message) for message in messages))
        return {
            observation.message_id: observation
            for observation in observations
            if observation is not None
        }

    async def _load_chain(
        self,
        chat_id: ExternalId,
        message: ReplyTarget,
        known: dict[ExternalId, ReplyTarget],
    ) -> tuple[ReplyTarget, ...]:
        newest_first: list[ReplyTarget] = []
        seen: set[ExternalId] = set()
        current: ReplyTarget | None = message
        while current is not None:
            if current.id in seen:
                break
            if len(newest_first) >= self._ingestion_settings.max_thread_messages:
                raise MemoryThreadLimitError(
                    f"Reply chain at message {message.id} exceeds the configured bound"
                )
            seen.add(current.id)
            newest_first.append(current)
            parent_id = current.reply_to_msg_id
            if parent_id is None:
                break
            parent = known.get(parent_id)
            if parent is None:
                parent = await self._retry_source(
                    lambda: self._source.fetch_message(chat_id, parent_id)
                )
                if parent is not None:
                    known[parent.id] = parent
            current = parent
        return tuple(reversed(newest_first))

    async def _retry_source(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        for attempt in range(1, self._ingestion_settings.retry_attempts + 1):
            try:
                return await operation()
            except Exception as exc:
                delay = (
                    self._source_retry_delay(exc)
                    if self._source_retry_delay is not None
                    else None
                )
                if delay is None or attempt >= self._ingestion_settings.retry_attempts:
                    raise
                delay = min(
                    max(0.0, delay),
                    self._ingestion_settings.max_retry_delay,
                )
                if self._logger is not None:
                    self._logger.warning(
                        "Memory source request backpressured; retrying in %.1fs",
                        delay,
                    )
                await self._sleep(delay)
        raise AssertionError("unreachable")

    async def _retry_memory(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        for attempt in range(1, self._ingestion_settings.retry_attempts + 1):
            try:
                return await operation()
            except MemoryClientError as exc:
                if exc.status not in {429, 502, 503, 504}:
                    raise
                error = exc
                retry_after = exc.retry_after
            except (TimeoutError, aiohttp.ClientConnectionError) as exc:
                error = exc
                retry_after = None
            if attempt >= self._ingestion_settings.retry_attempts:
                raise error
            delay = min(
                retry_after if retry_after is not None else 2 ** (attempt - 1),
                self._ingestion_settings.max_retry_delay,
            )
            if self._logger is not None:
                self._logger.warning(
                    "Memory retention backpressured; retrying in %.1fs",
                    delay,
                )
            await self._sleep(delay)
        raise AssertionError("unreachable")
