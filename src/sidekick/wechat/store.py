from __future__ import annotations

from pathlib import Path
import time

import aiosqlite

from sidekick.wechat.api import (
    WeChatAPIContractError,
    WeChatChatList,
    WeChatConnectorMessage,
    WeChatEvent,
    WeChatMessageList,
    WeChatSession,
)
from sidekick.wechat.message import WeChatMessage


class WeChatStateRepository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._connection: aiosqlite.Connection | None = None

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
            CREATE TABLE IF NOT EXISTS wechat_messages (
                local_order INTEGER PRIMARY KEY AUTOINCREMENT,
                connector_key TEXT NOT NULL,
                account_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                message_type TEXT NOT NULL,
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
            """
        )
        await connection.commit()
        self._connection = connection
        self.path.chmod(0o600)
        return self

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

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
            "SELECT account_id, cursor FROM wechat_connectors "
            "WHERE connector_key = ?",
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
            return await self.get_message(
                connector_key,
                message.chat_id,
                message.id,
            )
        if event.name == "message_remove":
            payload = event.payload
            if payload.get("status") != "recalled":
                raise WeChatAPIContractError("Malformed WeChat message removal")
            chat_id = _event_id(payload.get("chatId"), "chatId")
            message_id = _event_id(payload.get("id"), "id")
            await connection.execute(
                """
                UPDATE wechat_messages
                SET removed = 1, last_event_cursor = ?, updated_at = ?
                WHERE connector_key = ? AND account_id = ?
                  AND chat_id = ? AND message_id = ?
                """,
                (
                    event.cursor,
                    time.time(),
                    connector_key,
                    account_id,
                    chat_id,
                    message_id,
                ),
            )
            await connection.commit()
        return None

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
                    raise ValueError("Processed WeChat message belongs to another connector")
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

    async def get_cursor(self, connector_key: str) -> str:
        state = await self._connector_state(connector_key)
        return str(state["cursor"])

    async def get_account_id(self, connector_key: str) -> str:
        state = await self._connector_state(connector_key)
        return str(state["account_id"])

    async def get_generation(self, connector_key: str) -> int:
        state = await self._connector_state(connector_key)
        return int(state["connection_generation"])

    async def get_message(
        self,
        connector_key: str,
        chat_id: str,
        message_id: str,
    ) -> WeChatMessage | None:
        cursor = await self._require_connection().execute(
            self._message_select()
            + " AND messages.chat_id = ? AND messages.message_id = ?",
            (connector_key, chat_id, message_id),
        )
        row = await cursor.fetchone()
        return WeChatMessage.from_row(row) if row is not None else None

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
        await self._require_connection().execute(
            """
            INSERT INTO wechat_messages (
                connector_key, account_id, chat_id, message_id, direction,
                message_type, sender_id, reply_to_message_id, content,
                content_redacted, timestamp, source, sequence,
                last_event_cursor, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(connector_key, account_id, chat_id, message_id)
            DO UPDATE SET
                direction = excluded.direction,
                message_type = excluded.message_type,
                sender_id = excluded.sender_id,
                reply_to_message_id = COALESCE(
                    excluded.reply_to_message_id,
                    wechat_messages.reply_to_message_id
                ),
                content = CASE
                    WHEN excluded.content <> '' THEN excluded.content
                    ELSE wechat_messages.content
                END,
                content_redacted = CASE
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
                connector_key,
                account_id,
                message.chat_id,
                message.id,
                message.direction,
                message.message_type,
                message.sender_id,
                message.reply_to_message_id,
                message.content,
                int(message.content_redacted),
                message.timestamp,
                message.source,
                message.sequence,
                cursor,
                now,
            ),
        )

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
    def _message_select() -> str:
        return """
            SELECT
                messages.*,
                connectors.account_id AS self_id,
                chats.chat_type,
                chats.display_name AS chat_display_name
            FROM wechat_messages AS messages
            JOIN wechat_connectors AS connectors
              ON connectors.connector_key = messages.connector_key
             AND connectors.account_id = messages.account_id
            JOIN wechat_chats AS chats
              ON chats.connector_key = messages.connector_key
             AND chats.account_id = messages.account_id
             AND chats.chat_id = messages.chat_id
            WHERE messages.connector_key = ? AND messages.removed = 0
        """

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("WeChat state repository is not connected")
        return self._connection


def _event_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise WeChatAPIContractError(f"WeChat removal {field} is invalid")
    return value
