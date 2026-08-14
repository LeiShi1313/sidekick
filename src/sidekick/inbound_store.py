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
InboundWorkStatus = Literal["pending", "unavailable", "failed_unknown"]
InboundRevisionStatus = Literal[
    "running",
    "completed",
    "ignored",
    "recalled",
    "failed",
    "failed_unknown",
]


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
    updated_at: float


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
                updated_at REAL NOT NULL,
                PRIMARY KEY (source_id, chat_id, message_id),
                FOREIGN KEY (source_id)
                    REFERENCES ai_inbound_sources(source_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ai_inbound_work_due
            ON ai_inbound_work (
                source_id, status, next_attempt_at, updated_at
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
            """
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
                await connection.commit()
                return _external_id(existing["cursor"], "stored source cursor")
            if existing is not None:
                await connection.execute(
                    "DELETE FROM ai_inbound_work WHERE source_id = ?",
                    (source_id,),
                )
                await connection.execute(
                    "DELETE FROM ai_inbound_revisions WHERE source_id = ?",
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
            await connection.execute(
                """
                INSERT INTO ai_inbound_work (
                    source_id, chat_id, message_id, trigger_cursor, kind,
                    status, attempt_count, next_attempt_at, attested_origin,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, ?, ?)
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
                    (
                        attested_origin.value
                        if attested_origin is not None
                        else None
                    ),
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
                ORDER BY updated_at, chat_id, message_id
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
    async def defer_pending_ai_work(
        self,
        work: StoredInboundWork,
        *,
        error_code: str,
        retry_at: float,
        max_attempts: int | None,
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
            work.attempt_count + 1
            if work.last_error_code == error_code
            else 1
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
            return deleted.rowcount == 1
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
            cast(InboundRevisionStatus, str(row["status"]))
            if row is not None
            else None
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
            str(row["last_error_code"])
            if row["last_error_code"] is not None
            else None
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
            str(row["current_version"])
            if row["current_version"] is not None
            else None
        ),
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


def _validate_version(version: str) -> None:
    if not isinstance(version, str) or not version or len(version) > 512:
        raise ValueError("Inbound source version is invalid")


def _validate_error_code(error_code: str) -> None:
    if not isinstance(error_code, str) or not error_code or len(error_code) > 128:
        raise ValueError("Inbound source error code is invalid")
