from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
import time
from typing import Concatenate, Literal, ParamSpec, TypeVar, cast
from uuid import uuid4

import aiosqlite

from sidekick.chat.identity import ExternalId
from sidekick.chat.provenance import MessageOrigin
from sidekick.inbound import (
    InboundCompletion,
    InboundDeferral,
    InboundExecutionStart,
    InboundWorkKind,
)


_P = ParamSpec("_P")
_R = TypeVar("_R")
_TERMINAL_RETENTION_SECONDS = 30 * 24 * 60 * 60
InboundWorkStatus = Literal["pending", "unavailable", "failed_unknown"]
InboundRevisionStatus = Literal[
    "running",
    "completed",
    "ignored",
    "recalled",
    "failed",
    "failed_unknown",
]
GenerationJobStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "completed",
    "ignored",
    "failed",
    "cancelled",
    "source_unavailable",
    "superseded",
    "failed_unknown",
]
GenerationPromotion = Literal[
    "queued",
    "waiting",
    "updated",
    "duplicate",
    "principal_queue_full",
    "stale",
]
GenerationStart = Literal["started", "stale"]


@dataclass(frozen=True, slots=True)
class StoredInboundWork:
    source_id: str
    chat_id: ExternalId
    message_id: ExternalId
    trigger_cursor: ExternalId
    kind: InboundWorkKind
    status: InboundWorkStatus
    attempt_count: int
    next_attempt_at: float
    last_error_code: str | None
    attested_origin: MessageOrigin | None
    lease_id: str | None
    lease_trigger_cursor: ExternalId | None
    current_version: str | None
    acceptance_sequence: int
    updated_at: float


@dataclass(frozen=True, slots=True)
class StoredGenerationJob:
    job_id: str
    queue_sequence: int
    source_id: str
    chat_id: ExternalId
    message_id: ExternalId
    trigger_cursor: ExternalId
    kind: InboundWorkKind
    expected_version: str
    principal_actor_id: str
    scope_id: str
    is_owner: bool
    status: GenerationJobStatus
    attempt_count: int
    eligible_at: float
    last_error_code: str | None
    attested_origin: MessageOrigin | None
    lease_id: str | None
    started_at: float | None
    finished_at: float | None
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class GenerationQueueSnapshot:
    pending_intake: int
    queued: int
    active: int
    failed_unknown: int
    oldest_pending_intake_at: float | None
    oldest_queued_at: float | None


def _serialized(
    method: Callable[
        Concatenate[SQLiteInboundWorkStore, _P],
        Awaitable[_R],
    ],
) -> Callable[
    Concatenate[SQLiteInboundWorkStore, _P],
    Awaitable[_R],
]:
    @wraps(method)
    async def locked(
        self: SQLiteInboundWorkStore,
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        async with self._access_lock:
            try:
                return await method(self, *args, **kwargs)
            except BaseException:
                if self._connection is not None:
                    await _rollback_quietly(self._connection)
                raise

    return locked


async def _rollback_quietly(connection: aiosqlite.Connection) -> None:
    try:
        await asyncio.shield(connection.rollback())
    except BaseException:
        pass


class SQLiteInboundWorkStore:
    def __init__(self, path: Path):
        self.path = Path(path).resolve(strict=False)
        self._connection: aiosqlite.Connection | None = None
        self._access_lock = asyncio.Lock()

    @_serialized
    async def connect(self) -> SQLiteInboundWorkStore:
        if self._connection is not None:
            raise RuntimeError("Inbound work store is already connected")
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.path.parent.chmod(0o700)
        connection = await aiosqlite.connect(self.path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA busy_timeout=30000")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ai_inbound_sources (
                source_id TEXT PRIMARY KEY,
                epoch TEXT NOT NULL,
                cursor BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ai_inbound_work (
                source_id TEXT NOT NULL,
                chat_id BLOB NOT NULL,
                message_id BLOB NOT NULL,
                trigger_cursor BLOB NOT NULL,
                kind TEXT NOT NULL
                    CHECK (kind IN ('message', 'message_remove')),
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'unavailable', 'failed_unknown'
                    )),
                attempt_count INTEGER NOT NULL DEFAULT 0
                    CHECK (attempt_count >= 0),
                next_attempt_at REAL NOT NULL DEFAULT 0,
                last_error_code TEXT,
                attested_origin TEXT CHECK (
                    attested_origin IS NULL OR attested_origin IN (
                        'incoming', 'manual-outgoing'
                    )
                ),
                lease_id TEXT,
                lease_trigger_cursor BLOB,
                current_version TEXT,
                acceptance_sequence INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (source_id, chat_id, message_id),
                FOREIGN KEY (source_id)
                    REFERENCES ai_inbound_sources(source_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ai_inbound_work_due
            ON ai_inbound_work (
                source_id, status, next_attempt_at, acceptance_sequence
            );
            CREATE TABLE IF NOT EXISTS ai_inbound_acceptance_counters (
                source_id TEXT PRIMARY KEY,
                last_sequence INTEGER NOT NULL DEFAULT 0
                    CHECK (last_sequence >= 0),
                FOREIGN KEY (source_id)
                    REFERENCES ai_inbound_sources(source_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS ai_inbound_revisions (
                source_id TEXT NOT NULL,
                chat_id BLOB NOT NULL,
                message_id BLOB NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN (
                    'running', 'completed', 'ignored', 'recalled', 'failed',
                    'failed_unknown'
                )),
                started_at REAL NOT NULL,
                finished_at REAL,
                PRIMARY KEY (source_id, chat_id, message_id, version),
                FOREIGN KEY (source_id)
                    REFERENCES ai_inbound_sources(source_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS ai_generation_jobs (
                queue_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL,
                chat_id BLOB NOT NULL,
                message_id BLOB NOT NULL,
                trigger_cursor BLOB NOT NULL,
                kind TEXT NOT NULL DEFAULT 'message'
                    CHECK (kind = 'message'),
                expected_version TEXT NOT NULL,
                principal_actor_id TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                is_owner INTEGER NOT NULL CHECK (is_owner IN (0, 1)),
                status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN (
                    'queued', 'running', 'cancel_requested', 'completed',
                    'ignored', 'failed', 'cancelled', 'source_unavailable',
                    'superseded', 'failed_unknown'
                )),
                attempt_count INTEGER NOT NULL DEFAULT 0
                    CHECK (attempt_count >= 0),
                eligible_at REAL NOT NULL DEFAULT 0,
                last_error_code TEXT,
                attested_origin TEXT CHECK (
                    attested_origin IS NULL OR attested_origin IN (
                        'incoming', 'manual-outgoing'
                    )
                ),
                lease_id TEXT,
                started_at REAL,
                finished_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (source_id, chat_id, message_id, expected_version),
                FOREIGN KEY (source_id)
                    REFERENCES ai_inbound_sources(source_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ai_generation_jobs_due
            ON ai_generation_jobs (
                source_id, status, eligible_at, queue_sequence
            );
            CREATE INDEX IF NOT EXISTS ai_generation_jobs_principal
            ON ai_generation_jobs (
                source_id, principal_actor_id, status, queue_sequence
            );
            """
        )
        inbound_columns = {
            str(row["name"])
            async for row in await connection.execute(
                "PRAGMA table_info(ai_inbound_work)"
            )
        }
        if "acceptance_sequence" not in inbound_columns:
            await connection.execute(
                "ALTER TABLE ai_inbound_work "
                "ADD COLUMN acceptance_sequence INTEGER NOT NULL DEFAULT 0"
            )
        # The origin/main schema had no acceptance sequence. Preserve its
        # timestamp order before the shared intake lane uses this column as
        # its FIFO authority.
        await connection.execute(
            """
            WITH affected_sources AS (
                SELECT DISTINCT source_id
                FROM ai_inbound_work
                WHERE acceptance_sequence = 0
            ), ranked AS (
                SELECT rowid,
                    ROW_NUMBER() OVER (
                        PARTITION BY source_id
                        ORDER BY updated_at, chat_id, message_id
                    ) AS acceptance_sequence
                FROM ai_inbound_work
                WHERE source_id IN (SELECT source_id FROM affected_sources)
            )
            UPDATE ai_inbound_work
            SET acceptance_sequence = (
                SELECT ranked.acceptance_sequence
                FROM ranked
                WHERE ranked.rowid = ai_inbound_work.rowid
            )
            WHERE source_id IN (SELECT source_id FROM affected_sources)
            """
        )
        # CREATE INDEX IF NOT EXISTS does not update the definition on an
        # existing database, so rebuild this internal scheduling index after
        # the additive column migration.
        await connection.execute("DROP INDEX IF EXISTS ai_inbound_work_due")
        await connection.execute(
            "CREATE INDEX ai_inbound_work_due ON ai_inbound_work "
            "(source_id, status, next_attempt_at, acceptance_sequence)"
        )
        await connection.execute(
            """
            DELETE FROM ai_generation_jobs
            WHERE status IN (
                'completed', 'ignored', 'failed', 'cancelled',
                'source_unavailable', 'superseded', 'failed_unknown'
            )
              AND updated_at < ?
            """,
            (time.time() - _TERMINAL_RETENTION_SECONDS,),
        )
        await connection.commit()
        self._connection = connection
        self.path.chmod(0o600)
        return self

    @_serialized
    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @_serialized
    async def initialize_source(
        self,
        source_id: str,
        *,
        epoch: str,
        initial_cursor: ExternalId,
    ) -> ExternalId:
        _validate_source_identity(source_id, epoch)
        _validate_external_id(initial_cursor, "initial cursor")
        connection = self._require_connection()
        now = time.time()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = await connection.execute(
                "SELECT epoch, cursor FROM ai_inbound_sources WHERE source_id = ?",
                (source_id,),
            )
            existing = await cursor.fetchone()
            if existing is not None and str(existing["epoch"]) == epoch:
                await connection.execute(
                    """
                    INSERT INTO ai_inbound_acceptance_counters (
                        source_id, last_sequence
                    )
                    SELECT ?, COALESCE(MAX(acceptance_sequence), 0)
                    FROM ai_inbound_work
                    WHERE source_id = ?
                    ON CONFLICT(source_id) DO UPDATE SET
                        last_sequence = MAX(
                            ai_inbound_acceptance_counters.last_sequence,
                            excluded.last_sequence
                        )
                    """,
                    (source_id, source_id),
                )
                await connection.commit()
                return _external_id(existing["cursor"], "stored source cursor")
            if existing is not None:
                await connection.execute(
                    "DELETE FROM ai_generation_jobs WHERE source_id = ?",
                    (source_id,),
                )
                await connection.execute(
                    "DELETE FROM ai_inbound_work WHERE source_id = ?",
                    (source_id,),
                )
                await connection.execute(
                    "DELETE FROM ai_inbound_revisions WHERE source_id = ?",
                    (source_id,),
                )
                await connection.execute(
                    "DELETE FROM ai_inbound_acceptance_counters WHERE source_id = ?",
                    (source_id,),
                )
            await connection.execute(
                """
                INSERT INTO ai_inbound_sources (
                    source_id, epoch, cursor, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    epoch = excluded.epoch,
                    cursor = excluded.cursor,
                    updated_at = excluded.updated_at
                """,
                (source_id, epoch, initial_cursor, now),
            )
            await connection.execute(
                """
                INSERT INTO ai_inbound_acceptance_counters (
                    source_id, last_sequence
                ) VALUES (?, 0)
                ON CONFLICT(source_id) DO UPDATE SET last_sequence = 0
                """,
                (source_id,),
            )
            await connection.commit()
            return initial_cursor
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def get_cursor(self, source_id: str) -> ExternalId:
        cursor = await self._require_connection().execute(
            "SELECT cursor FROM ai_inbound_sources WHERE source_id = ?",
            (source_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("Inbound source is not initialized")
        return _external_id(row["cursor"], "stored source cursor")

    @_serialized
    async def acknowledge_event(
        self,
        source_id: str,
        cursor: ExternalId,
    ) -> None:
        _validate_external_id(cursor, "source cursor")
        connection = self._require_connection()
        result = await connection.execute(
            """
            UPDATE ai_inbound_sources
            SET cursor = ?, updated_at = ?
            WHERE source_id = ?
            """,
            (cursor, time.time(), source_id),
        )
        if result.rowcount != 1:
            await connection.rollback()
            raise RuntimeError("Inbound source is not initialized")
        await connection.commit()

    @_serialized
    async def accept_pending_ai_event(
        self,
        source_id: str,
        *,
        cursor: ExternalId,
        chat_id: ExternalId,
        message_id: ExternalId,
        kind: InboundWorkKind,
        attested_origin: MessageOrigin | None,
    ) -> None:
        _validate_external_id(cursor, "source cursor")
        _validate_external_id(chat_id, "chat ID")
        _validate_external_id(message_id, "message ID")
        if kind not in {"message", "message_remove"}:
            raise ValueError("Inbound work kind is invalid")
        if attested_origin not in {
            None,
            MessageOrigin.INCOMING,
            MessageOrigin.MANUAL_OUTGOING,
        }:
            raise ValueError("Durable inbound origin is not executable")
        connection = self._require_connection()
        now = time.time()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            incremented = await connection.execute(
                """
                UPDATE ai_inbound_acceptance_counters
                SET last_sequence = last_sequence + 1
                WHERE source_id = ?
                """,
                (source_id,),
            )
            if incremented.rowcount != 1:
                raise RuntimeError("Inbound source is not initialized")
            sequence_cursor = await connection.execute(
                """
                SELECT last_sequence
                FROM ai_inbound_acceptance_counters
                WHERE source_id = ?
                """,
                (source_id,),
            )
            sequence_row = await sequence_cursor.fetchone()
            assert sequence_row is not None
            acceptance_sequence = int(sequence_row["last_sequence"])
            await connection.execute(
                """
                INSERT INTO ai_inbound_work (
                    source_id, chat_id, message_id, trigger_cursor, kind,
                    status, attempt_count, next_attempt_at, attested_origin,
                    acceptance_sequence, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, ?, ?, ?)
                ON CONFLICT(source_id, chat_id, message_id) DO UPDATE SET
                    trigger_cursor = excluded.trigger_cursor,
                    kind = excluded.kind,
                    status = 'pending',
                    attempt_count = 0,
                    next_attempt_at = 0,
                    last_error_code = NULL,
                    attested_origin = excluded.attested_origin,
                    lease_id = NULL,
                    lease_trigger_cursor = NULL,
                    current_version = NULL,
                    acceptance_sequence = excluded.acceptance_sequence,
                    updated_at = excluded.updated_at
                WHERE ai_inbound_work.trigger_cursor
                    IS NOT excluded.trigger_cursor
                """,
                (
                    source_id,
                    chat_id,
                    message_id,
                    cursor,
                    kind,
                    (attested_origin.value if attested_origin is not None else None),
                    acceptance_sequence,
                    now,
                ),
            )
            updated = await connection.execute(
                """
                UPDATE ai_inbound_sources
                SET cursor = ?, updated_at = ?
                WHERE source_id = ?
                """,
                (cursor, now, source_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError("Inbound source is not initialized")
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def get_pending_ai_work(
        self,
        source_id: str,
        chat_id: ExternalId,
        message_id: ExternalId,
    ) -> StoredInboundWork | None:
        return await self._get_pending_ai_work(source_id, chat_id, message_id)

    @_serialized
    async def has_live_ai_message(
        self,
        source_id: str,
        chat_id: ExternalId,
        message_id: ExternalId,
    ) -> bool:
        cursor = await self._require_connection().execute(
            """
            SELECT 1
            WHERE EXISTS (
                SELECT 1
                FROM ai_inbound_work
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
            ) OR EXISTS (
                SELECT 1
                FROM ai_generation_jobs
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND status IN ('queued', 'running', 'cancel_requested')
            )
            """,
            (
                source_id,
                chat_id,
                message_id,
                source_id,
                chat_id,
                message_id,
            ),
        )
        return await cursor.fetchone() is not None

    async def _get_pending_ai_work(
        self,
        source_id: str,
        chat_id: ExternalId,
        message_id: ExternalId,
    ) -> StoredInboundWork | None:
        cursor = await self._require_connection().execute(
            """
            SELECT * FROM ai_inbound_work
            WHERE source_id = ? AND chat_id = ? AND message_id = ?
            """,
            (source_id, chat_id, message_id),
        )
        row = await cursor.fetchone()
        return _work_from_row(row) if row is not None else None

    @_serialized
    async def claim_pending_ai_work(
        self,
        source_id: str,
        *,
        now: float | None = None,
    ) -> StoredInboundWork | None:
        claimed_at = time.time() if now is None else now
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = await connection.execute(
                """
                SELECT * FROM ai_inbound_work
                WHERE source_id = ? AND status = 'pending'
                  AND lease_id IS NULL AND next_attempt_at <= ?
                ORDER BY acceptance_sequence, chat_id, message_id
                LIMIT 1
                """,
                (source_id, claimed_at),
            )
            row = await cursor.fetchone()
            if row is None:
                await connection.commit()
                return None
            lease_id = uuid4().hex
            await connection.execute(
                """
                UPDATE ai_inbound_work
                SET lease_id = ?, lease_trigger_cursor = trigger_cursor
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND lease_id IS NULL
                """,
                (
                    lease_id,
                    source_id,
                    row["chat_id"],
                    row["message_id"],
                ),
            )
            await connection.commit()
            return await self._get_pending_ai_work(
                source_id,
                row["chat_id"],
                row["message_id"],
            )
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def begin_pending_ai_execution(
        self,
        work: StoredInboundWork,
        *,
        version: str,
        supersede_queued_generation: bool = False,
        now: float | None = None,
    ) -> InboundExecutionStart:
        _validate_claimed_work(work)
        _validate_version(version)
        started_at = time.time() if now is None else now
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            pending = await self._get_pending_ai_work(
                work.source_id,
                work.chat_id,
                work.message_id,
            )
            if (
                pending is None
                or pending.lease_id != work.lease_id
                or pending.trigger_cursor != work.trigger_cursor
            ):
                await self._release_pending_ai_claim(work)
                await connection.commit()
                return "stale"
            cursor = await connection.execute(
                """
                SELECT status FROM ai_inbound_revisions
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND version = ?
                """,
                (work.source_id, work.chat_id, work.message_id, version),
            )
            duplicate = await cursor.fetchone() is not None
            if not duplicate:
                await connection.execute(
                    """
                    INSERT INTO ai_inbound_revisions (
                        source_id, chat_id, message_id, version, status,
                        started_at
                    ) VALUES (?, ?, ?, ?, 'running', ?)
                    """,
                    (
                        work.source_id,
                        work.chat_id,
                        work.message_id,
                        version,
                        started_at,
                    ),
                )
            if supersede_queued_generation:
                await connection.execute(
                    """
                    UPDATE ai_generation_jobs
                    SET status = 'superseded',
                        last_error_code = 'SOURCE_REVISION_CHANGED',
                        lease_id = NULL, finished_at = ?, updated_at = ?
                    WHERE source_id = ? AND chat_id = ? AND message_id = ?
                      AND status = 'queued'
                    """,
                    (
                        started_at,
                        started_at,
                        work.source_id,
                        work.chat_id,
                        work.message_id,
                    ),
                )
            await connection.execute(
                """
                UPDATE ai_inbound_work
                SET current_version = ?
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND lease_id = ? AND trigger_cursor IS ?
                """,
                (
                    version,
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                    work.lease_id,
                    work.trigger_cursor,
                ),
            )
            await connection.commit()
            return "duplicate" if duplicate else "started"
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def complete_pending_ai_work(
        self,
        work: StoredInboundWork,
        *,
        version: str,
        outcome: InboundCompletion,
        now: float | None = None,
    ) -> bool:
        _validate_claimed_work(work)
        _validate_version(version)
        if outcome not in {"completed", "ignored", "recalled", "failed"}:
            raise ValueError("Inbound completion outcome is invalid")
        finished_at = time.time() if now is None else now
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.execute(
                """
                UPDATE ai_inbound_revisions
                SET status = ?, finished_at = ?
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND version = ? AND status = 'running'
                """,
                (
                    outcome,
                    finished_at,
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                    version,
                ),
            )
            deleted = await connection.execute(
                """
                DELETE FROM ai_inbound_work
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND lease_id = ? AND trigger_cursor IS ?
                """,
                (
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                    work.lease_id,
                    work.trigger_cursor,
                ),
            )
            if deleted.rowcount != 1:
                await self._release_pending_ai_claim(work)
            await connection.commit()
            return deleted.rowcount == 1
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def promote_pending_ai_generation(
        self,
        work: StoredInboundWork,
        *,
        version: str,
        principal_actor_id: str,
        scope_id: str,
        is_owner: bool,
        eligible_at: float,
        now: float | None = None,
    ) -> GenerationPromotion:
        """Atomically hand one source revision to the generation queue."""
        _validate_claimed_work(work)
        _validate_version(version)
        _validate_workflow_identity(principal_actor_id, "principal actor ID")
        _validate_workflow_identity(scope_id, "scope ID")
        promoted_at = time.time() if now is None else now
        if eligible_at < 0:
            raise ValueError("Generation eligibility cannot be negative")
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            pending = await self._get_pending_ai_work(
                work.source_id,
                work.chat_id,
                work.message_id,
            )
            if (
                pending is None
                or pending.lease_id != work.lease_id
                or pending.trigger_cursor != work.trigger_cursor
            ):
                await self._release_pending_ai_claim(work)
                await connection.commit()
                return "stale"

            revision_cursor = await connection.execute(
                """
                SELECT 1 FROM ai_inbound_revisions
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND version = ?
                """,
                (work.source_id, work.chat_id, work.message_id, version),
            )
            if await revision_cursor.fetchone() is not None:
                await connection.execute(
                    """
                    UPDATE ai_generation_jobs
                    SET trigger_cursor = ?, lease_id = NULL, updated_at = ?
                    WHERE source_id = ? AND chat_id = ? AND message_id = ?
                      AND status = 'queued' AND expected_version = ?
                    """,
                    (
                        work.trigger_cursor,
                        promoted_at,
                        work.source_id,
                        work.chat_id,
                        work.message_id,
                        version,
                    ),
                )
                await connection.execute(
                    """
                    UPDATE ai_generation_jobs
                    SET status = 'superseded',
                        last_error_code = 'SOURCE_REVISION_CHANGED',
                        lease_id = NULL, finished_at = ?, updated_at = ?
                    WHERE source_id = ? AND chat_id = ? AND message_id = ?
                      AND status = 'queued' AND expected_version <> ?
                    """,
                    (
                        promoted_at,
                        promoted_at,
                        work.source_id,
                        work.chat_id,
                        work.message_id,
                        version,
                    ),
                )
                await connection.execute(
                    """
                    DELETE FROM ai_inbound_work
                    WHERE source_id = ? AND chat_id = ? AND message_id = ?
                      AND lease_id = ? AND trigger_cursor IS ?
                    """,
                    (
                        work.source_id,
                        work.chat_id,
                        work.message_id,
                        work.lease_id,
                        work.trigger_cursor,
                    ),
                )
                await connection.commit()
                return "duplicate"

            queued_cursor = await connection.execute(
                """
                SELECT job_id FROM ai_generation_jobs
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND status = 'queued'
                ORDER BY queue_sequence
                LIMIT 1
                """,
                (work.source_id, work.chat_id, work.message_id),
            )
            queued = await queued_cursor.fetchone()
            if queued is None:
                active_for_principal = await connection.execute(
                    """
                    SELECT 1 FROM ai_generation_jobs
                    WHERE source_id = ? AND principal_actor_id = ?
                      AND status IN ('running', 'cancel_requested')
                    LIMIT 1
                    """,
                    (work.source_id, principal_actor_id),
                )
                has_active = await active_for_principal.fetchone() is not None
                queued_count = 0
                if not is_owner:
                    queued_for_principal = await connection.execute(
                        """
                        SELECT COUNT(*) AS queued_count
                        FROM ai_generation_jobs
                        WHERE source_id = ? AND principal_actor_id = ?
                          AND status = 'queued'
                        """,
                        (work.source_id, principal_actor_id),
                    )
                    queued_row = await queued_for_principal.fetchone()
                    assert queued_row is not None
                    queued_count = int(queued_row["queued_count"])
                    # One claimable head plus one waiting tail, or one tail
                    # behind an already active job.
                    queued_limit = 1 if has_active else 2
                    if queued_count >= queued_limit:
                        await self._consume_claimed_inbound_revision(
                            work,
                            version=version,
                            outcome="failed",
                            finished_at=promoted_at,
                        )
                        await connection.commit()
                        return "principal_queue_full"

                promotion: GenerationPromotion = (
                    "waiting"
                    if (
                        (not is_owner and (has_active or queued_count > 0))
                        or eligible_at > promoted_at
                    )
                    else "queued"
                )
                await connection.execute(
                    """
                    INSERT INTO ai_generation_jobs (
                        job_id, source_id, chat_id, message_id,
                        trigger_cursor, kind, expected_version,
                        principal_actor_id, scope_id, is_owner, status,
                        eligible_at, attested_origin, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, 'message', ?, ?, ?, ?, 'queued',
                        ?, ?, ?, ?
                    )
                    """,
                    (
                        uuid4().hex,
                        work.source_id,
                        work.chat_id,
                        work.message_id,
                        work.trigger_cursor,
                        version,
                        principal_actor_id,
                        scope_id,
                        int(is_owner),
                        eligible_at,
                        (
                            work.attested_origin.value
                            if work.attested_origin is not None
                            else None
                        ),
                        promoted_at,
                        promoted_at,
                    ),
                )
            else:
                promotion = "updated"
                await connection.execute(
                    """
                    UPDATE ai_generation_jobs
                    SET trigger_cursor = ?, expected_version = ?,
                        principal_actor_id = ?, scope_id = ?, is_owner = ?,
                        attempt_count = 0, eligible_at = ?,
                        last_error_code = NULL, attested_origin = ?,
                        lease_id = NULL, updated_at = ?
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (
                        work.trigger_cursor,
                        version,
                        principal_actor_id,
                        scope_id,
                        int(is_owner),
                        eligible_at,
                        (
                            work.attested_origin.value
                            if work.attested_origin is not None
                            else None
                        ),
                        promoted_at,
                        str(queued["job_id"]),
                    ),
                )

            await self._consume_claimed_inbound_revision(
                work,
                version=version,
                outcome="completed",
                finished_at=promoted_at,
            )
            await connection.commit()
            return promotion
        except BaseException:
            await connection.rollback()
            raise

    async def _consume_claimed_inbound_revision(
        self,
        work: StoredInboundWork,
        *,
        version: str,
        outcome: Literal["completed", "failed"],
        finished_at: float,
    ) -> None:
        connection = self._require_connection()
        await connection.execute(
            """
            INSERT INTO ai_inbound_revisions (
                source_id, chat_id, message_id, version, status,
                started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work.source_id,
                work.chat_id,
                work.message_id,
                version,
                outcome,
                finished_at,
                finished_at,
            ),
        )
        deleted = await connection.execute(
            """
            DELETE FROM ai_inbound_work
            WHERE source_id = ? AND chat_id = ? AND message_id = ?
              AND lease_id = ? AND trigger_cursor IS ?
            """,
            (
                work.source_id,
                work.chat_id,
                work.message_id,
                work.lease_id,
                work.trigger_cursor,
            ),
        )
        if deleted.rowcount != 1:
            raise RuntimeError("Inbound generation promotion lost its source lease")

    @_serialized
    async def claim_pending_ai_generation(
        self,
        source_id: str,
        *,
        now: float | None = None,
    ) -> StoredGenerationJob | None:
        claimed_at = time.time() if now is None else now
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = await connection.execute(
                """
                SELECT candidate.*
                FROM ai_generation_jobs AS candidate
                WHERE candidate.source_id = ?
                  AND candidate.status = 'queued'
                  AND candidate.lease_id IS NULL
                  AND candidate.eligible_at <= ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM ai_generation_jobs AS same_message
                      WHERE same_message.source_id = candidate.source_id
                        AND same_message.chat_id = candidate.chat_id
                        AND same_message.message_id = candidate.message_id
                        AND same_message.queue_sequence <
                            candidate.queue_sequence
                        AND same_message.status IN (
                            'queued', 'running', 'cancel_requested'
                        )
                  )
                  AND (candidate.is_owner = 1 OR NOT EXISTS (
                      SELECT 1
                      FROM ai_generation_jobs AS earlier
                      WHERE earlier.source_id = candidate.source_id
                        AND earlier.principal_actor_id =
                            candidate.principal_actor_id
                        AND earlier.queue_sequence <
                            candidate.queue_sequence
                        AND earlier.status IN (
                            'queued', 'running', 'cancel_requested'
                        )
                  ))
                ORDER BY candidate.queue_sequence
                LIMIT 1
                """,
                (source_id, claimed_at),
            )
            row = await cursor.fetchone()
            if row is None:
                await connection.commit()
                return None
            lease_id = uuid4().hex
            updated = await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET lease_id = ?, updated_at = ?
                WHERE job_id = ? AND status = 'queued' AND lease_id IS NULL
                """,
                (lease_id, claimed_at, str(row["job_id"])),
            )
            if updated.rowcount != 1:
                await connection.commit()
                return None
            await connection.commit()
            return await self._get_generation_job(str(row["job_id"]))
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def begin_ai_generation(
        self,
        job: StoredGenerationJob,
        *,
        now: float | None = None,
    ) -> GenerationStart:
        _validate_claimed_generation(job)
        started_at = time.time() if now is None else now
        connection = self._require_connection()
        result = await connection.execute(
            """
            UPDATE ai_generation_jobs
            SET status = 'running', started_at = ?, updated_at = ?
            WHERE job_id = ? AND status = 'queued' AND lease_id = ?
              AND expected_version = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM ai_inbound_work AS newer
                  WHERE newer.source_id = ai_generation_jobs.source_id
                    AND newer.chat_id = ai_generation_jobs.chat_id
                    AND newer.message_id = ai_generation_jobs.message_id
                    AND newer.trigger_cursor IS NOT
                        ai_generation_jobs.trigger_cursor
              )
            """,
            (
                started_at,
                started_at,
                job.job_id,
                job.lease_id,
                job.expected_version,
            ),
        )
        await connection.commit()
        return "started" if result.rowcount == 1 else "stale"

    @_serialized
    async def complete_ai_generation(
        self,
        job: StoredGenerationJob,
        *,
        outcome: Literal[
            "completed",
            "ignored",
            "failed",
            "cancelled",
            "source_unavailable",
            "superseded",
        ],
        error_code: str | None = None,
        require_source_current: bool = False,
        now: float | None = None,
    ) -> bool:
        _validate_claimed_generation(job)
        if error_code is not None:
            _validate_error_code(error_code)
        finished_at = time.time() if now is None else now
        connection = self._require_connection()
        current_guard = (
            """
              AND NOT EXISTS (
                  SELECT 1
                  FROM ai_inbound_work AS newer
                  WHERE newer.source_id = ai_generation_jobs.source_id
                    AND newer.chat_id = ai_generation_jobs.chat_id
                    AND newer.message_id = ai_generation_jobs.message_id
                    AND newer.trigger_cursor IS NOT
                        ai_generation_jobs.trigger_cursor
              )
            """
            if require_source_current
            else ""
        )
        result = await connection.execute(
            f"""
            UPDATE ai_generation_jobs
            SET status = ?, last_error_code = ?, lease_id = NULL,
                finished_at = ?, updated_at = ?
            WHERE job_id = ? AND lease_id = ?
              AND status IN ('queued', 'running', 'cancel_requested')
              {current_guard}
            """,
            (
                outcome,
                error_code,
                finished_at,
                finished_at,
                job.job_id,
                job.lease_id,
            ),
        )
        await connection.commit()
        return result.rowcount == 1

    @_serialized
    async def defer_ai_generation(
        self,
        job: StoredGenerationJob,
        *,
        error_code: str,
        eligible_at: float,
        require_source_current: bool = False,
        now: float | None = None,
    ) -> bool:
        _validate_claimed_generation(job)
        _validate_error_code(error_code)
        deferred_at = time.time() if now is None else now
        if eligible_at < deferred_at:
            raise ValueError("Generation retry cannot be scheduled in the past")
        attempt_count = (
            job.attempt_count + 1 if job.last_error_code == error_code else 1
        )
        connection = self._require_connection()
        current_guard = (
            """
              AND NOT EXISTS (
                  SELECT 1
                  FROM ai_inbound_work AS newer
                  WHERE newer.source_id = ai_generation_jobs.source_id
                    AND newer.chat_id = ai_generation_jobs.chat_id
                    AND newer.message_id = ai_generation_jobs.message_id
                    AND newer.trigger_cursor IS NOT
                        ai_generation_jobs.trigger_cursor
              )
            """
            if require_source_current
            else ""
        )
        result = await connection.execute(
            f"""
            UPDATE ai_generation_jobs
            SET attempt_count = ?, eligible_at = ?,
                last_error_code = ?, lease_id = NULL, updated_at = ?
            WHERE job_id = ? AND status = 'queued' AND lease_id = ?
              {current_guard}
            """,
            (
                attempt_count,
                eligible_at,
                error_code,
                deferred_at,
                job.job_id,
                job.lease_id,
            ),
        )
        await connection.commit()
        return result.rowcount == 1

    @_serialized
    async def release_ai_generation(self, job: StoredGenerationJob) -> None:
        _validate_claimed_generation(job)
        await self._require_connection().execute(
            """
            UPDATE ai_generation_jobs
            SET lease_id = NULL
            WHERE job_id = ? AND status = 'queued' AND lease_id = ?
            """,
            (job.job_id, job.lease_id),
        )
        await self._require_connection().commit()

    @_serialized
    async def mark_ai_generation_unknown(
        self,
        job: StoredGenerationJob,
        *,
        error_code: str = "EXECUTION_OUTCOME_UNKNOWN",
        now: float | None = None,
    ) -> bool:
        _validate_claimed_generation(job)
        _validate_error_code(error_code)
        failed_at = time.time() if now is None else now
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            result = await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET status = 'failed_unknown', last_error_code = ?,
                    lease_id = NULL, finished_at = ?, updated_at = ?
                WHERE job_id = ? AND lease_id = ?
                  AND status IN ('running', 'cancel_requested')
                """,
                (
                    error_code,
                    failed_at,
                    failed_at,
                    job.job_id,
                    job.lease_id,
                ),
            )
            await connection.commit()
            return result.rowcount == 1
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def request_ai_generation_cancellation(
        self,
        source_id: str,
        principal_actor_id: str,
        *,
        now: float | None = None,
    ) -> tuple[int, int]:
        _validate_workflow_identity(principal_actor_id, "principal actor ID")
        cancelled_at = time.time() if now is None else now
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.execute(
                """
                UPDATE ai_inbound_work
                SET status = 'unavailable',
                    last_error_code = 'USER_CANCELLED',
                    lease_id = NULL, lease_trigger_cursor = NULL,
                    current_version = NULL, updated_at = ?
                WHERE source_id = ? AND status = 'pending'
                  AND EXISTS (
                      SELECT 1
                      FROM ai_generation_jobs AS known_generation
                      WHERE known_generation.source_id =
                              ai_inbound_work.source_id
                        AND known_generation.chat_id = ai_inbound_work.chat_id
                        AND known_generation.message_id =
                              ai_inbound_work.message_id
                        AND known_generation.principal_actor_id = ?
                        AND known_generation.status IN (
                            'queued', 'running', 'cancel_requested'
                        )
                  )
                """,
                (cancelled_at, source_id, principal_actor_id),
            )
            queued = await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET status = 'cancelled', last_error_code = 'USER_CANCELLED',
                    lease_id = NULL, finished_at = ?, updated_at = ?
                WHERE source_id = ? AND principal_actor_id = ?
                  AND status = 'queued'
                """,
                (
                    cancelled_at,
                    cancelled_at,
                    source_id,
                    principal_actor_id,
                ),
            )
            running = await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET status = 'cancel_requested',
                    last_error_code = 'USER_CANCELLED', updated_at = ?
                WHERE source_id = ? AND principal_actor_id = ?
                  AND status = 'running'
                """,
                (cancelled_at, source_id, principal_actor_id),
            )
            await connection.commit()
            return queued.rowcount, running.rowcount
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def supersede_queued_ai_generation(
        self,
        source_id: str,
        chat_id: ExternalId,
        message_id: ExternalId,
        *,
        now: float | None = None,
    ) -> int:
        superseded_at = time.time() if now is None else now
        result = await self._require_connection().execute(
            """
            UPDATE ai_generation_jobs
            SET status = 'superseded',
                last_error_code = 'SOURCE_REVISION_CHANGED',
                lease_id = NULL, finished_at = ?, updated_at = ?
            WHERE source_id = ? AND chat_id = ? AND message_id = ?
              AND status = 'queued'
            """,
            (
                superseded_at,
                superseded_at,
                source_id,
                chat_id,
                message_id,
            ),
        )
        await self._require_connection().commit()
        return result.rowcount

    @_serialized
    async def reschedule_ai_generation_scope(
        self,
        source_id: str,
        scope_id: str,
        *,
        now: float | None = None,
    ) -> int:
        _validate_workflow_identity(scope_id, "scope ID")
        reconsider_at = time.time() if now is None else now
        result = await self._require_connection().execute(
            """
            UPDATE ai_generation_jobs
            SET eligible_at = ?, updated_at = ?
            WHERE source_id = ? AND scope_id = ? AND status = 'queued'
              AND (last_error_code IS NULL OR last_error_code = 'COOLDOWN')
            """,
            (reconsider_at, reconsider_at, source_id, scope_id),
        )
        await self._require_connection().commit()
        return result.rowcount

    @_serialized
    async def next_pending_ai_generation_at(self, source_id: str) -> float | None:
        cursor = await self._require_connection().execute(
            """
            SELECT MIN(candidate.eligible_at) AS eligible_at
            FROM ai_generation_jobs AS candidate
            WHERE candidate.source_id = ?
              AND candidate.status = 'queued'
              AND candidate.lease_id IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM ai_generation_jobs AS same_message
                  WHERE same_message.source_id = candidate.source_id
                    AND same_message.chat_id = candidate.chat_id
                    AND same_message.message_id = candidate.message_id
                    AND same_message.queue_sequence < candidate.queue_sequence
                    AND same_message.status IN (
                        'queued', 'running', 'cancel_requested'
                    )
              )
              AND (candidate.is_owner = 1 OR NOT EXISTS (
                  SELECT 1
                  FROM ai_generation_jobs AS earlier
                  WHERE earlier.source_id = candidate.source_id
                    AND earlier.principal_actor_id =
                        candidate.principal_actor_id
                    AND earlier.queue_sequence < candidate.queue_sequence
                    AND earlier.status IN (
                        'queued', 'running', 'cancel_requested'
                    )
              ))
            """,
            (source_id,),
        )
        row = await cursor.fetchone()
        return (
            float(row["eligible_at"])
            if row is not None and row["eligible_at"] is not None
            else None
        )

    @_serialized
    async def get_ai_generation_job(self, job_id: str) -> StoredGenerationJob | None:
        return await self._get_generation_job(job_id)

    @_serialized
    async def get_ai_generation_queue_snapshot(
        self,
        source_id: str,
    ) -> GenerationQueueSnapshot:
        cursor = await self._require_connection().execute(
            """
            WITH generation AS (
                SELECT
                    SUM(
                        CASE WHEN status = 'queued' THEN 1 ELSE 0 END
                    ) AS queued,
                    SUM(
                        CASE WHEN status IN ('running', 'cancel_requested')
                        THEN 1 ELSE 0 END
                    ) AS active,
                    SUM(
                        CASE WHEN status = 'failed_unknown' THEN 1 ELSE 0 END
                    ) AS failed_unknown,
                    MIN(
                        CASE WHEN status = 'queued'
                        THEN created_at ELSE NULL END
                    ) AS oldest_queued_at
                FROM ai_generation_jobs
                WHERE source_id = ?
            ), intake AS (
                SELECT
                    SUM(
                        CASE WHEN status = 'pending' THEN 1 ELSE 0 END
                    ) AS pending,
                    SUM(
                        CASE WHEN status = 'failed_unknown' THEN 1 ELSE 0 END
                    ) AS failed_unknown,
                    MIN(
                        CASE WHEN status = 'pending'
                        THEN updated_at ELSE NULL END
                    ) AS oldest_pending_at
                FROM ai_inbound_work
                WHERE source_id = ?
            )
            SELECT
                intake.pending AS pending_intake,
                generation.queued AS queued,
                generation.active AS active,
                COALESCE(generation.failed_unknown, 0)
                    + COALESCE(intake.failed_unknown, 0) AS failed_unknown,
                intake.oldest_pending_at AS oldest_pending_intake_at,
                generation.oldest_queued_at AS oldest_queued_at
            FROM generation, intake
            """,
            (source_id, source_id),
        )
        row = await cursor.fetchone()
        assert row is not None
        return GenerationQueueSnapshot(
            pending_intake=int(row["pending_intake"] or 0),
            queued=int(row["queued"] or 0),
            active=int(row["active"] or 0),
            failed_unknown=int(row["failed_unknown"] or 0),
            oldest_pending_intake_at=(
                float(row["oldest_pending_intake_at"])
                if row["oldest_pending_intake_at"] is not None
                else None
            ),
            oldest_queued_at=(
                float(row["oldest_queued_at"])
                if row["oldest_queued_at"] is not None
                else None
            ),
        )

    async def _get_generation_job(
        self,
        job_id: str,
    ) -> StoredGenerationJob | None:
        cursor = await self._require_connection().execute(
            "SELECT * FROM ai_generation_jobs WHERE job_id = ?",
            (job_id,),
        )
        row = await cursor.fetchone()
        return _generation_job_from_row(row) if row is not None else None

    @_serialized
    async def defer_pending_ai_work(
        self,
        work: StoredInboundWork,
        *,
        error_code: str,
        retry_at: float,
        max_attempts: int | None,
        supersede_queued_generation_on_unavailable: bool = False,
        now: float | None = None,
    ) -> InboundDeferral:
        _validate_claimed_work(work)
        _validate_error_code(error_code)
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("Inbound work attempts must be positive")
        failed_at = time.time() if now is None else now
        if retry_at < failed_at:
            raise ValueError("Inbound retry cannot be scheduled in the past")
        attempt_count = (
            work.attempt_count + 1 if work.last_error_code == error_code else 1
        )
        status: Literal["pending", "unavailable"] = "pending"
        if max_attempts is not None and attempt_count >= max_attempts:
            status = "unavailable"
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            result = await connection.execute(
                """
                UPDATE ai_inbound_work
                SET status = ?, attempt_count = ?, next_attempt_at = ?,
                    last_error_code = ?, lease_id = NULL,
                    lease_trigger_cursor = NULL, current_version = NULL,
                    updated_at = ?
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND trigger_cursor IS ? AND lease_id = ?
                """,
                (
                    status,
                    attempt_count,
                    retry_at,
                    error_code,
                    failed_at,
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                    work.trigger_cursor,
                    work.lease_id,
                ),
            )
            if result.rowcount != 1:
                await self._release_pending_ai_claim(work)
                await connection.commit()
                return "stale"
            if status == "unavailable" and supersede_queued_generation_on_unavailable:
                await connection.execute(
                    """
                    UPDATE ai_generation_jobs
                    SET status = 'superseded',
                        last_error_code = 'SOURCE_REVISION_CHANGED',
                        lease_id = NULL, finished_at = ?, updated_at = ?
                    WHERE source_id = ? AND chat_id = ? AND message_id = ?
                      AND status = 'queued'
                    """,
                    (
                        failed_at,
                        failed_at,
                        work.source_id,
                        work.chat_id,
                        work.message_id,
                    ),
                )
            await connection.commit()
            return status
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def next_pending_ai_work_at(self, source_id: str) -> float | None:
        cursor = await self._require_connection().execute(
            """
            SELECT MIN(next_attempt_at) AS next_attempt_at
            FROM ai_inbound_work
            WHERE source_id = ? AND status = 'pending'
              AND lease_id IS NULL
            """,
            (source_id,),
        )
        row = await cursor.fetchone()
        return (
            float(row["next_attempt_at"])
            if row is not None and row["next_attempt_at"] is not None
            else None
        )

    @_serialized
    async def resolve_pending_ai_removal(self, work: StoredInboundWork) -> bool:
        _validate_claimed_work(work)
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            removed_at = time.time()
            deleted = await connection.execute(
                """
                DELETE FROM ai_inbound_work
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND trigger_cursor IS ? AND lease_id = ?
                """,
                (
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                    work.trigger_cursor,
                    work.lease_id,
                ),
            )
            if deleted.rowcount != 1:
                await self._release_pending_ai_claim(work)
                await connection.commit()
                return False
            await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET status = 'cancelled',
                    last_error_code = 'SOURCE_RECALLED', lease_id = NULL,
                    finished_at = ?, updated_at = ?
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND status = 'queued'
                """,
                (
                    removed_at,
                    removed_at,
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                ),
            )
            await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET status = 'cancel_requested',
                    last_error_code = 'SOURCE_RECALLED', updated_at = ?
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND status = 'running'
                """,
                (
                    removed_at,
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                ),
            )
            await connection.commit()
            return True
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def release_pending_ai_work(self, work: StoredInboundWork) -> None:
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await self._release_pending_ai_claim(work)
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def mark_pending_ai_execution_unknown(
        self,
        work: StoredInboundWork,
        *,
        version: str,
        now: float | None = None,
    ) -> bool:
        _validate_claimed_work(work)
        _validate_version(version)
        failed_at = time.time() if now is None else now
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.execute(
                """
                UPDATE ai_inbound_revisions
                SET status = 'failed_unknown', finished_at = ?
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND version = ? AND status = 'running'
                """,
                (
                    failed_at,
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                    version,
                ),
            )
            updated = await connection.execute(
                """
                UPDATE ai_inbound_work
                SET status = 'failed_unknown',
                    last_error_code = 'EXECUTION_OUTCOME_UNKNOWN',
                    lease_id = NULL, lease_trigger_cursor = NULL,
                    current_version = ?, updated_at = ?
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                  AND trigger_cursor IS ? AND lease_id = ?
                """,
                (
                    version,
                    failed_at,
                    work.source_id,
                    work.chat_id,
                    work.message_id,
                    work.trigger_cursor,
                    work.lease_id,
                ),
            )
            if updated.rowcount != 1:
                await self._release_pending_ai_claim(work)
            await connection.commit()
            return updated.rowcount == 1
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def recover_pending_ai_work(
        self,
        source_id: str,
        *,
        now: float | None = None,
    ) -> None:
        recovered_at = time.time() if now is None else now
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET status = 'failed_unknown',
                    last_error_code = 'ADAPTER_RESTARTED', lease_id = NULL,
                    finished_at = ?, updated_at = ?
                WHERE source_id = ?
                  AND status IN ('running', 'cancel_requested')
                """,
                (recovered_at, recovered_at, source_id),
            )
            await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET lease_id = NULL, updated_at = ?
                WHERE source_id = ? AND status = 'queued'
                  AND lease_id IS NOT NULL
                """,
                (recovered_at, source_id),
            )
            await connection.execute(
                """
                UPDATE ai_inbound_revisions
                SET status = 'failed_unknown', finished_at = ?
                WHERE source_id = ? AND status = 'running'
                """,
                (recovered_at, source_id),
            )
            await connection.execute(
                """
                UPDATE ai_inbound_work
                SET status = CASE
                        WHEN trigger_cursor IS lease_trigger_cursor
                          AND current_version IS NOT NULL
                        THEN 'failed_unknown'
                        ELSE 'pending'
                    END,
                    last_error_code = CASE
                        WHEN trigger_cursor IS lease_trigger_cursor
                          AND current_version IS NOT NULL
                        THEN 'EXECUTION_OUTCOME_UNKNOWN'
                        ELSE NULL
                    END,
                    lease_id = NULL,
                    lease_trigger_cursor = NULL,
                    current_version = CASE
                        WHEN trigger_cursor IS lease_trigger_cursor
                        THEN current_version
                        ELSE NULL
                    END,
                    updated_at = ?
                WHERE source_id = ? AND lease_id IS NOT NULL
                """,
                (recovered_at, source_id),
            )
            await connection.execute(
                """
                UPDATE ai_generation_jobs
                SET status = 'superseded',
                    last_error_code = 'SOURCE_REVISION_CHANGED',
                    lease_id = NULL, finished_at = ?, updated_at = ?
                WHERE source_id = ? AND status = 'queued'
                  AND EXISTS (
                      SELECT 1
                      FROM ai_inbound_work AS terminal_inbound
                      WHERE terminal_inbound.source_id =
                              ai_generation_jobs.source_id
                        AND terminal_inbound.chat_id =
                              ai_generation_jobs.chat_id
                        AND terminal_inbound.message_id =
                              ai_generation_jobs.message_id
                        AND terminal_inbound.status IN (
                            'unavailable', 'failed_unknown'
                        )
                  )
                """,
                (recovered_at, recovered_at, source_id),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def get_processed_revision_status(
        self,
        source_id: str,
        chat_id: ExternalId,
        message_id: ExternalId,
        version: str,
    ) -> InboundRevisionStatus | None:
        cursor = await self._require_connection().execute(
            """
            SELECT status FROM ai_inbound_revisions
            WHERE source_id = ? AND chat_id = ? AND message_id = ?
              AND version = ?
            """,
            (source_id, chat_id, message_id, version),
        )
        row = await cursor.fetchone()
        return (
            cast(InboundRevisionStatus, str(row["status"])) if row is not None else None
        )

    async def _release_pending_ai_claim(self, work: StoredInboundWork) -> None:
        await self._require_connection().execute(
            """
            UPDATE ai_inbound_work
            SET lease_id = NULL, lease_trigger_cursor = NULL,
                current_version = NULL
            WHERE source_id = ? AND chat_id = ? AND message_id = ?
              AND lease_id = ?
            """,
            (
                work.source_id,
                work.chat_id,
                work.message_id,
                work.lease_id,
            ),
        )

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Inbound work store is not connected")
        return self._connection


def _work_from_row(row: aiosqlite.Row) -> StoredInboundWork:
    raw_origin = row["attested_origin"]
    return StoredInboundWork(
        source_id=str(row["source_id"]),
        chat_id=_external_id(row["chat_id"], "stored chat ID"),
        message_id=_external_id(row["message_id"], "stored message ID"),
        trigger_cursor=_external_id(
            row["trigger_cursor"],
            "stored trigger cursor",
        ),
        kind=cast(InboundWorkKind, str(row["kind"])),
        status=cast(InboundWorkStatus, str(row["status"])),
        attempt_count=int(row["attempt_count"]),
        next_attempt_at=float(row["next_attempt_at"]),
        last_error_code=(
            str(row["last_error_code"]) if row["last_error_code"] is not None else None
        ),
        attested_origin=(
            MessageOrigin(str(raw_origin)) if raw_origin is not None else None
        ),
        lease_id=str(row["lease_id"]) if row["lease_id"] is not None else None,
        lease_trigger_cursor=(
            _external_id(row["lease_trigger_cursor"], "stored lease cursor")
            if row["lease_trigger_cursor"] is not None
            else None
        ),
        current_version=(
            str(row["current_version"]) if row["current_version"] is not None else None
        ),
        acceptance_sequence=int(row["acceptance_sequence"]),
        updated_at=float(row["updated_at"]),
    )


def _generation_job_from_row(row: aiosqlite.Row) -> StoredGenerationJob:
    raw_origin = row["attested_origin"]
    return StoredGenerationJob(
        job_id=str(row["job_id"]),
        queue_sequence=int(row["queue_sequence"]),
        source_id=str(row["source_id"]),
        chat_id=_external_id(row["chat_id"], "stored generation chat ID"),
        message_id=_external_id(
            row["message_id"],
            "stored generation message ID",
        ),
        trigger_cursor=_external_id(
            row["trigger_cursor"],
            "stored generation trigger cursor",
        ),
        kind=cast(InboundWorkKind, str(row["kind"])),
        expected_version=str(row["expected_version"]),
        principal_actor_id=str(row["principal_actor_id"]),
        scope_id=str(row["scope_id"]),
        is_owner=bool(row["is_owner"]),
        status=cast(GenerationJobStatus, str(row["status"])),
        attempt_count=int(row["attempt_count"]),
        eligible_at=float(row["eligible_at"]),
        last_error_code=(
            str(row["last_error_code"]) if row["last_error_code"] is not None else None
        ),
        attested_origin=(
            MessageOrigin(str(raw_origin)) if raw_origin is not None else None
        ),
        lease_id=str(row["lease_id"]) if row["lease_id"] is not None else None,
        started_at=(
            float(row["started_at"]) if row["started_at"] is not None else None
        ),
        finished_at=(
            float(row["finished_at"]) if row["finished_at"] is not None else None
        ),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _validate_source_identity(source_id: str, epoch: str) -> None:
    if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 256:
        raise ValueError("Inbound source ID is invalid")
    if not isinstance(epoch, str) or not epoch.strip() or len(epoch) > 256:
        raise ValueError("Inbound source epoch is invalid")


def _validate_external_id(value: ExternalId, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Inbound {label} is invalid")
    if isinstance(value, str) and not value:
        raise ValueError(f"Inbound {label} cannot be empty")


def _external_id(value: object, label: str) -> ExternalId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise RuntimeError(f"Inbound {label} is corrupt")
    if isinstance(value, str) and not value:
        raise RuntimeError(f"Inbound {label} is corrupt")
    return value


def _validate_claimed_work(work: StoredInboundWork) -> None:
    if not isinstance(work, StoredInboundWork) or work.lease_id is None:
        raise ValueError("Claimed inbound work is required")


def _validate_claimed_generation(job: StoredGenerationJob) -> None:
    if not isinstance(job, StoredGenerationJob) or job.lease_id is None:
        raise ValueError("Claimed generation job is required")


def _validate_workflow_identity(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"Generation {label} is invalid")


def _validate_version(version: str) -> None:
    if not isinstance(version, str) or not version or len(version) > 512:
        raise ValueError("Inbound source version is invalid")


def _validate_error_code(error_code: str) -> None:
    if not isinstance(error_code, str) or not error_code or len(error_code) > 128:
        raise ValueError("Inbound source error code is invalid")
