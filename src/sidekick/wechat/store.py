from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
import fcntl
from functools import wraps
import os
from pathlib import Path
import time
from typing import Concatenate, Literal, ParamSpec, TypeVar
from uuid import uuid4

import aiosqlite

from sidekick.wechat.api import (
    WeChatAPIContractError,
    WeChatChat,
    WeChatChatList,
    WeChatConnectorMessage,
    WeChatEvent,
    WeChatGroupMemberList,
    WeChatMessageList,
    WeChatSession,
    WeChatUser,
    WeChatUserList,
)
from sidekick.wechat.message import WeChatMessage


_P = ParamSpec("_P")
_R = TypeVar("_R")
_MAX_PENDING_GENERATED_SENDS = 4_096
_MAX_GENERATED_SEND_LEASES_PER_REQUEST = 8


@dataclass(frozen=True, slots=True)
class WeChatStoredChat:
    chat_id: str
    chat_type: str
    display_name: str | None
    last_observed_at: float


@dataclass(frozen=True, slots=True)
class WeChatGeneratedSendReservation:
    request_id: str
    chat_id: str
    fingerprint: bytes
    reconciliation_attempts: int = 0


def _serialized(
    method: Callable[
        Concatenate[WeChatStateRepository, _P],
        Awaitable[_R],
    ],
) -> Callable[
    Concatenate[WeChatStateRepository, _P],
    Awaitable[_R],
]:
    @wraps(method)
    async def locked(
        self: WeChatStateRepository,
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        async with self._access_lock:
            return await method(self, *args, **kwargs)

    return locked


async def _rollback_quietly(connection: aiosqlite.Connection) -> None:
    try:
        await asyncio.shield(connection.rollback())
    except BaseException:
        pass


async def _release_generated_send_lease(
    connection: aiosqlite.Connection,
    connector_key: str,
    account_id: str,
    request_id: str,
    lease_id: str,
) -> None:
    async def cleanup() -> None:
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                """
                DELETE FROM wechat_generated_send_leases
                WHERE connector_key = ? AND account_id = ?
                  AND request_id = ? AND lease_id = ?
                """,
                (connector_key, account_id, request_id, lease_id),
            )
            await connection.execute(
                """
                DELETE FROM wechat_generated_send_reservations
                WHERE connector_key = ? AND account_id = ? AND request_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM wechat_generated_send_leases
                      WHERE connector_key = ? AND account_id = ?
                        AND request_id = ?
                  )
                """,
                (
                    connector_key,
                    account_id,
                    request_id,
                    connector_key,
                    account_id,
                    request_id,
                ),
            )
            await connection.execute("COMMIT")
        except BaseException:
            await _rollback_quietly(connection)

    await asyncio.shield(cleanup())


class WeChatStateRepository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._connection: aiosqlite.Connection | None = None
        self._adapter_lock_fd: int | None = None
        # aiosqlite serializes statements, not multi-statement transactions.
        # Readers share this lock so they cannot observe an in-progress refresh.
        self._access_lock = asyncio.Lock()

    @_serialized
    async def connect(self) -> WeChatStateRepository:
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.path.parent.chmod(0o700)
        connection = await aiosqlite.connect(self.path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS wechat_connectors (
                connector_key TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                self_display_name TEXT,
                connection_generation INTEGER NOT NULL,
                cursor TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                snapshot_cursor TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS wechat_chats (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                chat_type TEXT NOT NULL,
                display_name TEXT,
                connection_generation INTEGER NOT NULL,
                snapshot_id TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (connector_key, account_id, chat_id)
            );
            CREATE TABLE IF NOT EXISTS wechat_users (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                display_name TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (connector_key, account_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS wechat_group_members (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                display_name TEXT,
                nickname TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (connector_key, account_id, group_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS wechat_messages (
                local_order INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_order INTEGER NOT NULL,
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                message_type TEXT NOT NULL,
                media_id TEXT,
                sender_id TEXT NOT NULL,
                reply_to_message_id TEXT,
                content TEXT NOT NULL,
                content_redacted INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                source TEXT,
                sequence TEXT,
                removed INTEGER NOT NULL DEFAULT 0,
                last_event_cursor TEXT NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (connector_key, account_id, chat_id, message_id)
            );
            CREATE INDEX IF NOT EXISTS wechat_messages_by_chat_order
            ON wechat_messages (
                connector_key, account_id, chat_id, local_order
            );
            CREATE TABLE IF NOT EXISTS wechat_processed_messages (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                processed_at REAL NOT NULL,
                PRIMARY KEY (connector_key, account_id, chat_id, message_id)
            );
            CREATE TABLE IF NOT EXISTS wechat_generated_send_reservations (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                fingerprint BLOB NOT NULL CHECK (length(fingerprint) = 32),
                created_at REAL NOT NULL,
                reconciliation_attempts INTEGER NOT NULL DEFAULT 0
                    CHECK (reconciliation_attempts >= 0),
                next_reconciliation_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (connector_key, account_id, request_id)
            );
            CREATE INDEX IF NOT EXISTS wechat_generated_sends_by_candidate
            ON wechat_generated_send_reservations (
                connector_key, account_id, chat_id, fingerprint
            );
            CREATE TABLE IF NOT EXISTS wechat_generated_send_leases (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'send'
                    CHECK (kind IN ('send', 'reconcile')),
                created_at REAL NOT NULL,
                PRIMARY KEY (
                    connector_key, account_id, request_id, lease_id
                )
            );
            CREATE INDEX IF NOT EXISTS wechat_generated_send_leases_by_request
            ON wechat_generated_send_leases (
                connector_key, account_id, request_id
            );
            CREATE TABLE IF NOT EXISTS wechat_generated_messages (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                confirmed_at REAL NOT NULL,
                PRIMARY KEY (
                    connector_key, account_id, chat_id, message_id
                )
            );
            CREATE TABLE IF NOT EXISTS wechat_attachment_send_attempts (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                trigger_message_id TEXT NOT NULL,
                payload_fingerprint TEXT NOT NULL,
                attempt INTEGER NOT NULL CHECK (attempt >= 0),
                PRIMARY KEY (
                    connector_key,
                    account_id,
                    chat_id,
                    trigger_message_id,
                    payload_fingerprint
                )
            );
            CREATE TABLE IF NOT EXISTS wechat_projection_counters (
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                last_memory_order INTEGER NOT NULL,
                PRIMARY KEY (connector_key, account_id)
            );
            """
        )
        message_columns = {
            str(row["name"])
            async for row in await connection.execute(
                "PRAGMA table_info(wechat_messages)"
            )
        }
        if "memory_order" not in message_columns:
            await connection.execute(
                "ALTER TABLE wechat_messages ADD COLUMN memory_order INTEGER"
            )
        if "media_id" not in message_columns:
            await connection.execute(
                "ALTER TABLE wechat_messages ADD COLUMN media_id TEXT"
            )
        generated_send_columns = {
            str(row["name"])
            async for row in await connection.execute(
                "PRAGMA table_info(wechat_generated_send_reservations)"
            )
        }
        if "reconciliation_attempts" not in generated_send_columns:
            await connection.execute(
                """
                ALTER TABLE wechat_generated_send_reservations
                ADD COLUMN reconciliation_attempts INTEGER NOT NULL DEFAULT 0
                """
            )
        if "next_reconciliation_at" not in generated_send_columns:
            await connection.execute(
                """
                ALTER TABLE wechat_generated_send_reservations
                ADD COLUMN next_reconciliation_at REAL NOT NULL DEFAULT 0
                """
            )
        generated_send_lease_columns = {
            str(row["name"])
            async for row in await connection.execute(
                "PRAGMA table_info(wechat_generated_send_leases)"
            )
        }
        if "kind" not in generated_send_lease_columns:
            await connection.execute(
                """
                ALTER TABLE wechat_generated_send_leases
                ADD COLUMN kind TEXT NOT NULL DEFAULT 'send'
                """
            )
        await connection.execute(
            "UPDATE wechat_messages SET memory_order = local_order "
            "WHERE memory_order IS NULL"
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS wechat_messages_by_chat_memory_order
            ON wechat_messages (
                connector_key, account_id, chat_id, memory_order
            )
            """
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS wechat_generated_sends_due
            ON wechat_generated_send_reservations (
                connector_key, account_id, next_reconciliation_at, created_at
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO wechat_projection_counters (
                connector_key, account_id, last_memory_order
            )
            SELECT connector_key, account_id, MAX(memory_order)
            FROM wechat_messages
            GROUP BY connector_key, account_id
            ON CONFLICT(connector_key, account_id) DO UPDATE SET
                last_memory_order = MAX(
                    wechat_projection_counters.last_memory_order,
                    excluded.last_memory_order
                )
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
        self._release_adapter_ownership()

    @_serialized
    async def acquire_adapter_ownership(self) -> None:
        """Exclusively own adapter mutations for this state database."""
        self._require_connection()
        if self._adapter_lock_fd is not None:
            raise RuntimeError("WeChat adapter ownership is already held")
        lock_path = self.path.with_name(f"{self.path.name}.adapter.lock")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RuntimeError(
                f"WeChat adapter is already active for {self.path}"
            ) from exc
        except BaseException:
            os.close(descriptor)
            raise
        self._adapter_lock_fd = descriptor

    def _release_adapter_ownership(self) -> None:
        descriptor = self._adapter_lock_fd
        if descriptor is None:
            return
        self._adapter_lock_fd = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @_serialized
    async def bootstrap(
        self,
        *,
        connector_key: str,
        session: WeChatSession,
        chats: WeChatChatList,
        messages: WeChatMessageList,
    ) -> None:
        connection = self._require_connection()
        cursor = await connection.execute(
            "SELECT account_id, cursor FROM wechat_connectors WHERE connector_key = ?",
            (connector_key,),
        )
        existing = await cursor.fetchone()
        durable_cursor = (
            str(existing["cursor"])
            if existing is not None and existing["account_id"] == session.self_id
            else messages.cursor
        )
        now = time.time()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.execute(
                """
                INSERT INTO wechat_connectors (
                    connector_key, account_id, self_display_name,
                    connection_generation, cursor, snapshot_id,
                    snapshot_cursor, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(connector_key) DO UPDATE SET
                    account_id = excluded.account_id,
                    self_display_name = excluded.self_display_name,
                    connection_generation = excluded.connection_generation,
                    cursor = excluded.cursor,
                    snapshot_id = excluded.snapshot_id,
                    snapshot_cursor = excluded.snapshot_cursor,
                    updated_at = excluded.updated_at
                """,
                (
                    connector_key,
                    session.self_id,
                    session.display_name,
                    session.connection_generation,
                    durable_cursor,
                    chats.snapshot.id,
                    chats.cursor,
                    now,
                ),
            )
            await self._replace_chats(
                connector_key,
                session.self_id,
                session.connection_generation,
                chats,
                now=now,
            )
            for message in messages.messages:
                await self._upsert_message(
                    connector_key,
                    session.self_id,
                    message,
                    cursor=messages.cursor,
                    now=now,
                )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def refresh_chats(
        self,
        connector_key: str,
        chats: WeChatChatList,
    ) -> None:
        connection = self._require_connection()
        state = await self._connector_state(connector_key)
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await self._replace_chats(
                connector_key,
                str(state["account_id"]),
                int(state["connection_generation"]),
                chats,
                now=time.time(),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def refresh_users(
        self,
        connector_key: str,
        users: WeChatUserList,
    ) -> None:
        connection = self._require_connection()
        state = await self._connector_state(connector_key)
        account_id = str(state["account_id"])
        now = time.time()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.executemany(
                """
                INSERT INTO wechat_users (
                    connector_key, account_id, user_id, display_name, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(connector_key, account_id, user_id) DO UPDATE SET
                    display_name = COALESCE(
                        excluded.display_name,
                        wechat_users.display_name
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    (
                        connector_key,
                        account_id,
                        user.id,
                        user.display_name,
                        now,
                    )
                    for user in users.users
                ),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def refresh_user(
        self,
        connector_key: str,
        user_id: str,
        user: WeChatUser | None,
    ) -> None:
        if user is not None and user.id != user_id:
            raise ValueError("WeChat user refresh returned a different user")
        connection = self._require_connection()
        state = await self._connector_state(connector_key)
        account_id = str(state["account_id"])
        if user is None:
            return
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.execute(
                """
                INSERT INTO wechat_users (
                    connector_key, account_id, user_id,
                    display_name, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(connector_key, account_id, user_id) DO UPDATE SET
                    display_name = COALESCE(
                        excluded.display_name,
                        wechat_users.display_name
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    connector_key,
                    account_id,
                    user.id,
                    user.display_name,
                    time.time(),
                ),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def refresh_group_members(
        self,
        connector_key: str,
        group_id: str,
        members: WeChatGroupMemberList,
        *,
        replace_aliases: bool = False,
    ) -> None:
        if members.group_id != group_id or any(
            member.group_id != group_id for member in members.members
        ):
            raise ValueError("WeChat member refresh returned a different group")
        connection = self._require_connection()
        state = await self._connector_state(connector_key)
        account_id = str(state["account_id"])
        generation = int(state["connection_generation"])
        if members.snapshot_current and not members.snapshot_complete:
            raise WeChatAPIContractError(
                "WeChat current group member snapshot is not complete"
            )
        if members.snapshot_current and (
            members.snapshot_connection_generation != generation
        ):
            raise WeChatAPIContractError(
                "WeChat group member snapshot generation is stale"
            )
        membership_authoritative = (
            members.snapshot_complete and members.snapshot_current
        )
        replace_nicknames = replace_aliases
        now = time.time()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            existing_user_ids: set[str] = set()
            if membership_authoritative:
                cursor = await connection.execute(
                    """
                    SELECT user_id
                    FROM wechat_group_members
                    WHERE connector_key = ? AND account_id = ? AND group_id = ?
                    """,
                    (connector_key, account_id, group_id),
                )
                existing_user_ids = {
                    str(row["user_id"]) async for row in cursor
                }
                await cursor.close()
            if replace_aliases:
                await connection.execute(
                    """
                    UPDATE wechat_group_members
                    SET nickname = NULL, updated_at = ?
                    WHERE connector_key = ? AND account_id = ? AND group_id = ?
                      AND nickname IS NOT NULL
                    """,
                    (now, connector_key, account_id, group_id),
                )
            await connection.executemany(
                """
                INSERT INTO wechat_group_members (
                    connector_key, account_id, group_id, user_id,
                    display_name, nickname, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    connector_key, account_id, group_id, user_id
                ) DO UPDATE SET
                    display_name = COALESCE(
                        excluded.display_name,
                        wechat_group_members.display_name
                    ),
                    nickname = CASE
                        WHEN ? THEN excluded.nickname
                        ELSE COALESCE(
                            excluded.nickname,
                            wechat_group_members.nickname
                        )
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    (
                        connector_key,
                        account_id,
                        group_id,
                        member.user_id,
                        member.display_name,
                        member.nickname,
                        now,
                        replace_nicknames,
                    )
                    for member in members.members
                ),
            )
            if membership_authoritative:
                retained_user_ids = {
                    member.user_id for member in members.members
                }
                await connection.executemany(
                    """
                    DELETE FROM wechat_group_members
                    WHERE connector_key = ? AND account_id = ?
                      AND group_id = ? AND user_id = ?
                    """,
                    (
                        (connector_key, account_id, group_id, user_id)
                        for user_id in existing_user_ids - retained_user_ids
                    ),
                )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def project_event(
        self,
        connector_key: str,
        event: WeChatEvent,
    ) -> WeChatMessage | None:
        state = await self._connector_state(connector_key)
        generation = int(state["connection_generation"])
        if (
            event.connection_generation is not None
            and event.connection_generation != generation
        ):
            return None
        account_id = str(state["account_id"])
        connection = self._require_connection()
        if event.name == "message":
            message = event.message()
            await self._upsert_message(
                connector_key,
                account_id,
                message,
                cursor=event.cursor,
                now=time.time(),
            )
            await connection.commit()
            return await self._get_message(
                connector_key,
                message.chat_id,
                message.id,
                include_unsupported=True,
            )
        if event.name == "message_remove":
            chat_id, message_id = event.removed_message()
            memory_order = await self._next_memory_order(
                connector_key,
                account_id,
            )
            now = time.time()
            await connection.execute(
                """
                UPDATE wechat_messages
                SET memory_order = CASE
                        WHEN removed = 0 THEN ?
                        ELSE memory_order
                    END,
                    removed = 1,
                    last_event_cursor = ?,
                    updated_at = ?
                WHERE connector_key = ? AND account_id = ?
                  AND chat_id = ? AND message_id = ?
                """,
                (
                    memory_order,
                    event.cursor,
                    now,
                    connector_key,
                    account_id,
                    chat_id,
                    message_id,
                ),
            )
            await connection.execute(
                """
                UPDATE wechat_chats
                SET updated_at = MAX(updated_at, ?)
                WHERE connector_key = ? AND account_id = ? AND chat_id = ?
                """,
                (now, connector_key, account_id, chat_id),
            )
            await connection.commit()
        return None

    @_serialized
    async def acknowledge_event(
        self,
        connector_key: str,
        cursor: str,
        *,
        processed_message: WeChatMessage | None = None,
    ) -> None:
        connection = self._require_connection()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            if processed_message is not None:
                if processed_message.connector_key != connector_key:
                    raise ValueError(
                        "Processed WeChat message belongs to another connector"
                    )
                await connection.execute(
                    """
                    INSERT OR IGNORE INTO wechat_processed_messages (
                        connector_key, account_id, chat_id, message_id, processed_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        connector_key,
                        processed_message.account_id,
                        processed_message.chat_id,
                        processed_message.id,
                        time.time(),
                    ),
                )
            result = await connection.execute(
                "UPDATE wechat_connectors SET cursor = ?, updated_at = ? "
                "WHERE connector_key = ?",
                (cursor, time.time(), connector_key),
            )
            if result.rowcount != 1:
                raise RuntimeError("WeChat connector state is unavailable")
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    @_serialized
    async def is_processed(self, message: WeChatMessage) -> bool:
        cursor = await self._require_connection().execute(
            """
            SELECT 1 FROM wechat_processed_messages
            WHERE connector_key = ? AND account_id = ?
              AND chat_id = ? AND message_id = ?
            """,
            (
                message.connector_key,
                message.account_id,
                message.chat_id,
                message.id,
            ),
        )
        return await cursor.fetchone() is not None

    @_serialized
    async def mark_processed_identity(
        self,
        connector_key: str,
        account_id: str,
        chat_id: str,
        message_id: str,
    ) -> None:
        connection = self._require_connection()
        await connection.execute(
            """
            INSERT OR IGNORE INTO wechat_processed_messages (
                connector_key, account_id, chat_id, message_id, processed_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (connector_key, account_id, chat_id, message_id, time.time()),
        )
        await connection.commit()

    @_serialized
    async def reserve_generated_send(
        self,
        connector_key: str,
        account_id: str,
        chat_id: str,
        request_id: str,
        fingerprint: bytes,
    ) -> str:
        if not isinstance(fingerprint, bytes) or len(fingerprint) != 32:
            raise ValueError("WeChat generated-send fingerprint is invalid")
        connection = self._require_connection()
        lease_id = uuid4().hex
        lease_inserted = False
        try:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                """
                SELECT chat_id, fingerprint
                FROM wechat_generated_send_reservations
                WHERE connector_key = ? AND account_id = ? AND request_id = ?
                """,
                (connector_key, account_id, request_id),
            )
            existing = await cursor.fetchone()
            if existing is not None:
                if (
                    str(existing["chat_id"]) != chat_id
                    or bytes(existing["fingerprint"]) != fingerprint
                ):
                    raise RuntimeError(
                        "WeChat generated-send request ID was reused for another payload"
                    )
            else:
                cursor = await connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM wechat_generated_send_reservations
                    WHERE connector_key = ? AND account_id = ?
                    """,
                    (connector_key, account_id),
                )
                row = await cursor.fetchone()
                if (
                    row is not None
                    and int(row["count"]) >= _MAX_PENDING_GENERATED_SENDS
                ):
                    raise RuntimeError(
                        "WeChat generated-send reservation capacity reached"
                    )
                await connection.execute(
                    """
                    INSERT INTO wechat_generated_send_reservations (
                        connector_key, account_id, request_id, chat_id,
                        fingerprint, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        connector_key,
                        account_id,
                        request_id,
                        chat_id,
                        fingerprint,
                        time.time(),
                    ),
                )
            cursor = await connection.execute(
                """
                SELECT 1
                FROM wechat_generated_send_leases
                WHERE connector_key = ? AND account_id = ? AND request_id = ?
                  AND kind = 'reconcile'
                LIMIT 1
                """,
                (connector_key, account_id, request_id),
            )
            if await cursor.fetchone() is not None:
                raise RuntimeError("WeChat generated send is being reconciled")
            cursor = await connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM wechat_generated_send_leases
                WHERE connector_key = ? AND account_id = ? AND request_id = ?
                  AND kind = 'send'
                """,
                (connector_key, account_id, request_id),
            )
            row = await cursor.fetchone()
            if (
                row is not None
                and int(row["count"]) >= _MAX_GENERATED_SEND_LEASES_PER_REQUEST
            ):
                raise RuntimeError(
                    "WeChat generated-send active caller capacity reached"
                )
            await connection.execute(
                """
                INSERT INTO wechat_generated_send_leases (
                    connector_key, account_id, request_id,
                    lease_id, kind, created_at
                ) VALUES (?, ?, ?, ?, 'send', ?)
                """,
                (
                    connector_key,
                    account_id,
                    request_id,
                    lease_id,
                    time.time(),
                ),
            )
            lease_inserted = True
            await connection.commit()
            return lease_id
        except BaseException:
            await _rollback_quietly(connection)
            if lease_inserted:
                await _release_generated_send_lease(
                    connection,
                    connector_key,
                    account_id,
                    request_id,
                    lease_id,
                )
            raise

    @_serialized
    async def defer_generated_send(
        self,
        connector_key: str,
        account_id: str,
        request_id: str,
        lease_id: str,
    ) -> None:
        """Detach a completed caller while retaining its unknown send."""
        connection = self._require_connection()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                """
                DELETE FROM wechat_generated_send_leases
                WHERE connector_key = ? AND account_id = ?
                  AND request_id = ? AND lease_id = ?
                """,
                (connector_key, account_id, request_id, lease_id),
            )
            await connection.commit()
        except BaseException:
            await _rollback_quietly(connection)
            raise

    @_serialized
    async def recover_generated_send_leases(
        self,
        connector_key: str,
    ) -> None:
        """Release stale leases after exclusive adapter startup."""
        if self._adapter_lock_fd is None:
            raise RuntimeError(
                "WeChat generated-send recovery requires adapter ownership"
            )
        connection = self._require_connection()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                """
                DELETE FROM wechat_generated_send_leases
                WHERE connector_key = ?
                """,
                (connector_key,),
            )
            await connection.commit()
        except BaseException:
            await _rollback_quietly(connection)
            raise

    @_serialized
    async def claim_generated_send_reconciliation(
        self,
        connector_key: str,
        account_id: str,
        request_id: str,
    ) -> str | None:
        connection = self._require_connection()
        lease_id = uuid4().hex
        try:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                """
                INSERT INTO wechat_generated_send_leases (
                    connector_key, account_id, request_id,
                    lease_id, kind, created_at
                )
                SELECT connector_key, account_id, request_id,
                       ?, 'reconcile', ?
                FROM wechat_generated_send_reservations AS reservations
                WHERE connector_key = ? AND account_id = ? AND request_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM wechat_generated_send_leases AS leases
                      WHERE leases.connector_key = reservations.connector_key
                        AND leases.account_id = reservations.account_id
                        AND leases.request_id = reservations.request_id
                  )
                """,
                (lease_id, time.time(), connector_key, account_id, request_id),
            )
            claimed = cursor.rowcount == 1
            await connection.commit()
            return lease_id if claimed else None
        except BaseException:
            await _rollback_quietly(connection)
            raise

    @_serialized
    async def fail_generated_send(
        self,
        connector_key: str,
        account_id: str,
        request_id: str,
        lease_id: str,
    ) -> bool:
        connection = self._require_connection()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                """
                DELETE FROM wechat_generated_send_leases
                WHERE connector_key = ? AND account_id = ?
                  AND request_id = ? AND lease_id = ?
                """,
                (connector_key, account_id, request_id, lease_id),
            )
            owned = cursor.rowcount == 1
            if owned:
                await connection.execute(
                    """
                    DELETE FROM wechat_generated_send_reservations
                    WHERE connector_key = ? AND account_id = ? AND request_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM wechat_generated_send_leases
                          WHERE connector_key = ? AND account_id = ?
                            AND request_id = ?
                      )
                    """,
                    (
                        connector_key,
                        account_id,
                        request_id,
                        connector_key,
                        account_id,
                        request_id,
                    ),
                )
            await connection.commit()
            return owned
        except BaseException:
            await _rollback_quietly(connection)
            raise

    @_serialized
    async def confirm_generated_send(
        self,
        connector_key: str,
        account_id: str,
        chat_id: str,
        request_id: str,
        message_id: str,
        lease_id: str,
    ) -> bool:
        connection = self._require_connection()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                """
                SELECT 1
                FROM wechat_generated_send_leases
                WHERE connector_key = ? AND account_id = ?
                  AND request_id = ? AND lease_id = ?
                LIMIT 1
                """,
                (connector_key, account_id, request_id, lease_id),
            )
            if await cursor.fetchone() is None:
                await connection.commit()
                return False
            await connection.execute(
                """
                INSERT OR IGNORE INTO wechat_processed_messages (
                    connector_key, account_id, chat_id, message_id, processed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (connector_key, account_id, chat_id, message_id, time.time()),
            )
            await connection.execute(
                """
                INSERT OR IGNORE INTO wechat_generated_messages (
                    connector_key, account_id, chat_id, message_id, confirmed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (connector_key, account_id, chat_id, message_id, time.time()),
            )
            await connection.execute(
                """
                DELETE FROM wechat_generated_send_leases
                WHERE connector_key = ? AND account_id = ? AND request_id = ?
                """,
                (connector_key, account_id, request_id),
            )
            await connection.execute(
                """
                DELETE FROM wechat_generated_send_reservations
                WHERE connector_key = ? AND account_id = ? AND request_id = ?
                """,
                (connector_key, account_id, request_id),
            )
            await connection.commit()
            return True
        except BaseException:
            await _rollback_quietly(connection)
            raise

    @_serialized
    async def generated_message_provenance(
        self,
        message: WeChatMessage,
    ) -> Literal["confirmed", "candidate"] | None:
        cursor = await self._require_connection().execute(
            """
            SELECT CASE
                WHEN EXISTS (
                    SELECT 1 FROM wechat_generated_messages
                    WHERE connector_key = ? AND account_id = ?
                      AND chat_id = ? AND message_id = ?
                ) THEN 'confirmed'
                WHEN EXISTS (
                    SELECT 1 FROM wechat_generated_send_reservations
                    WHERE connector_key = ? AND account_id = ?
                      AND chat_id = ?
                ) THEN 'candidate'
                ELSE NULL
            END AS provenance
            """,
            (
                message.connector_key,
                message.account_id,
                message.chat_id,
                message.id,
                message.connector_key,
                message.account_id,
                message.chat_id,
            ),
        )
        row = await cursor.fetchone()
        provenance = row["provenance"] if row is not None else None
        if provenance in {"confirmed", "candidate"}:
            return provenance
        return None

    @_serialized
    async def list_generated_send_reservations(
        self,
        connector_key: str,
        account_id: str,
    ) -> tuple[WeChatGeneratedSendReservation, ...]:
        cursor = await self._require_connection().execute(
            """
            SELECT request_id, chat_id, fingerprint, reconciliation_attempts
            FROM wechat_generated_send_reservations
            WHERE connector_key = ? AND account_id = ?
            ORDER BY created_at, request_id
            """,
            (connector_key, account_id),
        )
        return tuple(
            WeChatGeneratedSendReservation(
                request_id=str(row["request_id"]),
                chat_id=str(row["chat_id"]),
                fingerprint=bytes(row["fingerprint"]),
                reconciliation_attempts=int(row["reconciliation_attempts"]),
            )
            for row in await cursor.fetchall()
        )

    @_serialized
    async def list_due_generated_send_reservations(
        self,
        connector_key: str,
        account_id: str,
        *,
        now: float,
        limit: int,
    ) -> tuple[WeChatGeneratedSendReservation, ...]:
        if limit < 1:
            raise ValueError("WeChat reconciliation limit must be positive")
        cursor = await self._require_connection().execute(
            """
            SELECT request_id, chat_id, fingerprint, reconciliation_attempts
            FROM wechat_generated_send_reservations AS reservations
            WHERE connector_key = ? AND account_id = ?
              AND next_reconciliation_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM wechat_generated_send_leases AS leases
                  WHERE leases.connector_key = reservations.connector_key
                    AND leases.account_id = reservations.account_id
                    AND leases.request_id = reservations.request_id
              )
            ORDER BY next_reconciliation_at, created_at, request_id
            LIMIT ?
            """,
            (connector_key, account_id, now, limit),
        )
        return tuple(
            WeChatGeneratedSendReservation(
                request_id=str(row["request_id"]),
                chat_id=str(row["chat_id"]),
                fingerprint=bytes(row["fingerprint"]),
                reconciliation_attempts=int(row["reconciliation_attempts"]),
            )
            for row in await cursor.fetchall()
        )

    @_serialized
    async def defer_generated_send_reconciliation(
        self,
        connector_key: str,
        account_id: str,
        request_id: str,
        lease_id: str,
        *,
        next_attempt_at: float,
    ) -> bool:
        connection = self._require_connection()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                """
                UPDATE wechat_generated_send_reservations
                SET reconciliation_attempts = reconciliation_attempts + 1,
                    next_reconciliation_at = ?
                WHERE connector_key = ? AND account_id = ? AND request_id = ?
                  AND EXISTS (
                      SELECT 1 FROM wechat_generated_send_leases
                      WHERE connector_key = ? AND account_id = ?
                        AND request_id = ? AND lease_id = ?
                        AND kind = 'reconcile'
                  )
                """,
                (
                    next_attempt_at,
                    connector_key,
                    account_id,
                    request_id,
                    connector_key,
                    account_id,
                    request_id,
                    lease_id,
                ),
            )
            owned = cursor.rowcount == 1
            await connection.execute(
                """
                DELETE FROM wechat_generated_send_leases
                WHERE connector_key = ? AND account_id = ?
                  AND request_id = ? AND lease_id = ? AND kind = 'reconcile'
                """,
                (connector_key, account_id, request_id, lease_id),
            )
            await connection.commit()
            return owned
        except BaseException:
            await _rollback_quietly(connection)
            raise

    @_serialized
    async def count_generated_send_reservations(
        self,
        connector_key: str,
        account_id: str,
    ) -> int:
        cursor = await self._require_connection().execute(
            """
            SELECT COUNT(*) AS count
            FROM wechat_generated_send_reservations
            WHERE connector_key = ? AND account_id = ?
            """,
            (connector_key, account_id),
        )
        row = await cursor.fetchone()
        return int(row["count"]) if row is not None else 0

    @_serialized
    async def count_connector_generated_send_reservations(
        self,
        connector_key: str,
    ) -> int:
        cursor = await self._require_connection().execute(
            """
            SELECT COUNT(*) AS count
            FROM wechat_generated_send_reservations
            WHERE connector_key = ?
            """,
            (connector_key,),
        )
        row = await cursor.fetchone()
        return int(row["count"]) if row is not None else 0

    @_serialized
    async def get_attachment_send_attempt(
        self,
        connector_key: str,
        account_id: str,
        chat_id: str,
        trigger_message_id: str,
        payload_fingerprint: str,
    ) -> int:
        cursor = await self._require_connection().execute(
            """
            SELECT attempt FROM wechat_attachment_send_attempts
            WHERE connector_key = ? AND account_id = ? AND chat_id = ?
              AND trigger_message_id = ? AND payload_fingerprint = ?
            """,
            (
                connector_key,
                account_id,
                chat_id,
                trigger_message_id,
                payload_fingerprint,
            ),
        )
        row = await cursor.fetchone()
        return int(row["attempt"]) if row is not None else 0

    @_serialized
    async def advance_attachment_send_attempt(
        self,
        connector_key: str,
        account_id: str,
        chat_id: str,
        trigger_message_id: str,
        payload_fingerprint: str,
        *,
        expected_attempt: int,
    ) -> None:
        if expected_attempt < 0:
            raise ValueError("WeChat attachment send attempt cannot be negative")
        connection = self._require_connection()
        await connection.execute(
            """
            INSERT INTO wechat_attachment_send_attempts (
                connector_key, account_id, chat_id, trigger_message_id,
                payload_fingerprint, attempt
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                connector_key, account_id, chat_id, trigger_message_id,
                payload_fingerprint
            ) DO UPDATE SET
                attempt = MAX(
                    wechat_attachment_send_attempts.attempt,
                    excluded.attempt
                )
            """,
            (
                connector_key,
                account_id,
                chat_id,
                trigger_message_id,
                payload_fingerprint,
                expected_attempt + 1,
            ),
        )
        await connection.commit()

    @_serialized
    async def get_cursor(self, connector_key: str) -> str:
        state = await self._connector_state(connector_key)
        return str(state["cursor"])

    @_serialized
    async def get_account_id(self, connector_key: str) -> str:
        state = await self._connector_state(connector_key)
        return str(state["account_id"])

    @_serialized
    async def get_generation(self, connector_key: str) -> int:
        state = await self._connector_state(connector_key)
        return int(state["connection_generation"])

    @_serialized
    async def get_message(
        self,
        connector_key: str,
        chat_id: str,
        message_id: str,
    ) -> WeChatMessage | None:
        return await self._get_message(
            connector_key,
            chat_id,
            message_id,
            include_unsupported=False,
        )

    @_serialized
    async def get_reply_message(
        self,
        connector_key: str,
        chat_id: str,
        message_id: str,
    ) -> WeChatMessage | None:
        return await self._get_message(
            connector_key,
            chat_id,
            message_id,
            include_unsupported=False,
            include_quoted_images=True,
        )

    async def _get_message(
        self,
        connector_key: str,
        chat_id: str,
        message_id: str,
        *,
        include_unsupported: bool,
        include_quoted_images: bool = False,
    ) -> WeChatMessage | None:
        cursor = await self._require_connection().execute(
            self._message_select(
                include_unsupported=include_unsupported,
                include_quoted_images=include_quoted_images,
            )
            + " AND messages.chat_id = ? AND messages.message_id = ?",
            (connector_key, chat_id, message_id),
        )
        row = await cursor.fetchone()
        return WeChatMessage.from_row(row) if row is not None else None

    @_serialized
    async def get_chat(
        self,
        connector_key: str,
        chat_id: str,
    ) -> WeChatChat | None:
        cursor = await self._require_connection().execute(
            """
            SELECT chats.chat_id, chats.chat_type, chats.display_name
            FROM wechat_chats AS chats
            JOIN wechat_connectors AS connectors
              ON connectors.connector_key = chats.connector_key
             AND connectors.account_id = chats.account_id
            WHERE chats.connector_key = ? AND chats.chat_id = ?
            """,
            (connector_key, chat_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return WeChatChat(
            id=str(row["chat_id"]),
            type=str(row["chat_type"]),
            display_name=(
                str(row["display_name"]) if row["display_name"] is not None else None
            ),
        )

    @_serialized
    async def list_chats(
        self,
        connector_key: str,
    ) -> tuple[WeChatStoredChat, ...]:
        cursor = await self._require_connection().execute(
            """
            SELECT
                chats.chat_id,
                chats.chat_type,
                chats.display_name,
                chats.updated_at AS last_observed_at
            FROM wechat_chats AS chats
            JOIN wechat_connectors AS connectors
              ON connectors.connector_key = chats.connector_key
             AND connectors.account_id = chats.account_id
            WHERE chats.connector_key = ?
            ORDER BY chats.chat_id
            """,
            (connector_key,),
        )
        rows = await cursor.fetchall()
        return tuple(
            WeChatStoredChat(
                chat_id=str(row["chat_id"]),
                chat_type=str(row["chat_type"]),
                display_name=(
                    str(row["display_name"])
                    if row["display_name"] is not None
                    else None
                ),
                last_observed_at=float(row["last_observed_at"]),
            )
            for row in rows
        )

    @_serialized
    async def get_latest_memory_cursor(
        self,
        connector_key: str,
        chat_id: str,
    ) -> int:
        cursor = await self._require_connection().execute(
            """
            SELECT MAX(messages.memory_order) AS memory_order
            FROM wechat_messages AS messages
            JOIN wechat_connectors AS connectors
              ON connectors.connector_key = messages.connector_key
             AND connectors.account_id = messages.account_id
            WHERE messages.connector_key = ? AND messages.chat_id = ?
              AND messages.removed = 0
            """,
            (connector_key, chat_id),
        )
        row = await cursor.fetchone()
        return (
            int(row["memory_order"])
            if row is not None and row["memory_order"] is not None
            else 0
        )

    @_serialized
    async def fetch_recent(
        self,
        connector_key: str,
        chat_id: str,
        before_message_id: str,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        if limit < 1:
            return ()
        connection = self._require_connection()
        anchor_cursor = await connection.execute(
            """
            SELECT messages.local_order
            FROM wechat_messages AS messages
            JOIN wechat_connectors AS connectors
              ON connectors.connector_key = messages.connector_key
             AND connectors.account_id = messages.account_id
            WHERE messages.connector_key = ? AND messages.chat_id = ?
              AND messages.message_id = ? AND messages.removed = 0
            """,
            (connector_key, chat_id, before_message_id),
        )
        anchor = await anchor_cursor.fetchone()
        if anchor is None:
            return ()
        cursor = await connection.execute(
            self._message_select()
            + " AND messages.chat_id = ? AND messages.local_order < ?"
            + " ORDER BY messages.local_order DESC LIMIT ?",
            (connector_key, chat_id, int(anchor["local_order"]), limit),
        )
        rows = await cursor.fetchall()
        return tuple(WeChatMessage.from_row(row) for row in reversed(rows))

    @_serialized
    async def fetch_memory_window(
        self,
        connector_key: str,
        chat_id: str,
        *,
        since: datetime,
        until: datetime,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        if limit < 1:
            return ()
        if limit > 5_001:
            raise ValueError("WeChat memory window exceeds the supported bound")
        cursor = await self._require_connection().execute(
            self._message_select()
            + " AND messages.chat_id = ?"
            + " AND messages.timestamp >= ? AND messages.timestamp <= ?"
            + " ORDER BY messages.timestamp DESC, messages.local_order DESC"
            + " LIMIT ?",
            (
                connector_key,
                chat_id,
                since.timestamp(),
                until.timestamp(),
                limit,
            ),
        )
        rows = await cursor.fetchall()
        return tuple(WeChatMessage.from_row(row) for row in reversed(rows))

    @_serialized
    async def fetch_memory_after(
        self,
        connector_key: str,
        chat_id: str,
        *,
        after_memory_order: int,
        until: datetime,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        if after_memory_order < 0:
            raise ValueError("WeChat memory cursor cannot be negative")
        if limit < 1:
            return ()
        if limit > 5_001:
            raise ValueError("WeChat memory window exceeds the supported bound")
        cursor = await self._require_connection().execute(
            self._message_select()
            + " AND messages.chat_id = ? AND messages.memory_order > ?"
            + " ORDER BY messages.memory_order LIMIT ?",
            (connector_key, chat_id, after_memory_order, limit),
        )
        messages: list[WeChatMessage] = []
        async for row in cursor:
            message = WeChatMessage.from_row(row)
            if message.date > until:
                break
            messages.append(message)
        return tuple(messages)

    @_serialized
    async def count_messages(self, connector_key: str) -> int:
        cursor = await self._require_connection().execute(
            """
            SELECT COUNT(*) AS count
            FROM wechat_messages AS messages
            JOIN wechat_connectors AS connectors
              ON connectors.connector_key = messages.connector_key
             AND connectors.account_id = messages.account_id
            WHERE messages.connector_key = ?
            """,
            (connector_key,),
        )
        row = await cursor.fetchone()
        return int(row["count"])

    async def _replace_chats(
        self,
        connector_key: str,
        account_id: str,
        generation: int,
        chats: WeChatChatList,
        *,
        now: float,
    ) -> None:
        if chats.snapshot.connection_generation != generation:
            raise WeChatAPIContractError("WeChat chat snapshot generation is stale")
        connection = self._require_connection()
        await connection.execute(
            "DELETE FROM wechat_chats WHERE connector_key = ? AND account_id = ?",
            (connector_key, account_id),
        )
        await connection.executemany(
            """
            INSERT INTO wechat_chats (
                connector_key, account_id, chat_id, chat_type, display_name,
                connection_generation, snapshot_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    connector_key,
                    account_id,
                    chat.id,
                    chat.type,
                    chat.display_name,
                    generation,
                    chats.snapshot.id,
                    now,
                )
                for chat in chats.chats
            ),
        )

    async def _upsert_message(
        self,
        connector_key: str,
        account_id: str,
        message: WeChatConnectorMessage,
        *,
        cursor: str,
        now: float,
    ) -> None:
        connection = self._require_connection()
        memory_order = await self._next_memory_order(connector_key, account_id)
        await connection.execute(
            """
            INSERT INTO wechat_messages (
                memory_order, connector_key, account_id, chat_id, message_id,
                direction, message_type, media_id, sender_id,
                reply_to_message_id, content, content_redacted, timestamp,
                source, sequence, last_event_cursor, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(connector_key, account_id, chat_id, message_id)
            DO UPDATE SET
                memory_order = CASE
                    WHEN wechat_messages.direction IS NOT excluded.direction
                      OR wechat_messages.message_type IS NOT excluded.message_type
                      OR (
                          excluded.media_id IS NOT NULL
                          AND wechat_messages.media_id IS NOT excluded.media_id
                      )
                      OR wechat_messages.sender_id IS NOT excluded.sender_id
                      OR (
                          excluded.reply_to_message_id IS NOT NULL
                          AND wechat_messages.reply_to_message_id
                              IS NOT excluded.reply_to_message_id
                      )
                      OR (
                          excluded.content <> ''
                          AND (
                              wechat_messages.content IS NOT excluded.content
                              OR wechat_messages.content_redacted
                                  IS NOT excluded.content_redacted
                          )
                      )
                      OR (
                          excluded.content_redacted = 1
                          AND wechat_messages.content_redacted != 1
                      )
                      OR (
                          excluded.timestamp > 0
                          AND wechat_messages.timestamp IS NOT excluded.timestamp
                      )
                      OR (
                          excluded.source IS NOT NULL
                          AND wechat_messages.source IS NOT excluded.source
                      )
                      OR (
                          excluded.sequence IS NOT NULL
                          AND wechat_messages.sequence IS NOT excluded.sequence
                      )
                    THEN excluded.memory_order
                    ELSE wechat_messages.memory_order
                END,
                direction = excluded.direction,
                message_type = excluded.message_type,
                media_id = COALESCE(excluded.media_id, wechat_messages.media_id),
                sender_id = excluded.sender_id,
                reply_to_message_id = COALESCE(
                    excluded.reply_to_message_id,
                    wechat_messages.reply_to_message_id
                ),
                content = CASE
                    WHEN excluded.content_redacted = 1 THEN ''
                    WHEN excluded.content <> '' THEN excluded.content
                    ELSE wechat_messages.content
                END,
                content_redacted = CASE
                    WHEN excluded.content_redacted = 1 THEN 1
                    WHEN excluded.content <> '' THEN excluded.content_redacted
                    ELSE wechat_messages.content_redacted
                END,
                timestamp = CASE
                    WHEN excluded.timestamp > 0 THEN excluded.timestamp
                    ELSE wechat_messages.timestamp
                END,
                source = COALESCE(excluded.source, wechat_messages.source),
                sequence = COALESCE(excluded.sequence, wechat_messages.sequence),
                last_event_cursor = excluded.last_event_cursor,
                updated_at = excluded.updated_at
            """,
            (
                memory_order,
                connector_key,
                account_id,
                message.chat_id,
                message.id,
                message.direction,
                message.message_type,
                message.media_id,
                message.sender_id,
                message.reply_to_message_id,
                message.display_content,
                int(message.content_redacted),
                message.timestamp,
                message.source,
                message.sequence,
                cursor,
                now,
            ),
        )
        await connection.execute(
            """
            UPDATE wechat_chats
            SET updated_at = MAX(updated_at, ?)
            WHERE connector_key = ? AND account_id = ? AND chat_id = ?
            """,
            (now, connector_key, account_id, message.chat_id),
        )

    async def _next_memory_order(
        self,
        connector_key: str,
        account_id: str,
    ) -> int:
        cursor = await self._require_connection().execute(
            """
            INSERT INTO wechat_projection_counters (
                connector_key, account_id, last_memory_order
            ) VALUES (?, ?, 1)
            ON CONFLICT(connector_key, account_id) DO UPDATE SET
                last_memory_order = last_memory_order + 1
            RETURNING last_memory_order
            """,
            (connector_key, account_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("WeChat projection counter did not advance")
        return int(row["last_memory_order"])

    async def _connector_state(self, connector_key: str) -> aiosqlite.Row:
        cursor = await self._require_connection().execute(
            "SELECT * FROM wechat_connectors WHERE connector_key = ?",
            (connector_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("WeChat connector has not been bootstrapped")
        return row

    @staticmethod
    def _message_select(
        *,
        include_unsupported: bool = False,
        include_quoted_images: bool = False,
    ) -> str:
        query = """
            SELECT
                messages.*,
                connectors.account_id AS self_id,
                chats.chat_type,
                chats.display_name AS chat_display_name,
                COALESCE(
                    CASE
                        WHEN chats.chat_type = 'group'
                        THEN group_members.nickname
                    END,
                    users.display_name,
                    CASE
                        WHEN chats.chat_type = 'group'
                        THEN group_members.display_name
                    END,
                    CASE
                        WHEN messages.sender_id = connectors.account_id
                        THEN connectors.self_display_name
                    END,
                    messages.sender_id
                ) AS sender_display_name
            FROM wechat_messages AS messages
            JOIN wechat_connectors AS connectors
              ON connectors.connector_key = messages.connector_key
             AND connectors.account_id = messages.account_id
            JOIN wechat_chats AS chats
              ON chats.connector_key = messages.connector_key
             AND chats.account_id = messages.account_id
             AND chats.chat_id = messages.chat_id
            LEFT JOIN wechat_users AS users
              ON users.connector_key = messages.connector_key
             AND users.account_id = messages.account_id
             AND users.user_id = messages.sender_id
            LEFT JOIN wechat_group_members AS group_members
              ON group_members.connector_key = messages.connector_key
             AND group_members.account_id = messages.account_id
             AND group_members.group_id = messages.chat_id
             AND group_members.user_id = messages.sender_id
            WHERE messages.connector_key = ? AND messages.removed = 0
        """
        if include_unsupported:
            return query
        if include_quoted_images:
            return (
                query
                + " AND ((messages.content_redacted = 0"
                + " AND messages.message_type IN ('text', 'chat_history'))"
                + " OR (messages.message_type = 'image'"
                + " AND messages.media_id IS NOT NULL))"
            )
        return (
            query
            + " AND messages.content_redacted = 0"
            + " AND messages.message_type IN ('text', 'chat_history')"
        )

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("WeChat state repository is not connected")
        return self._connection
