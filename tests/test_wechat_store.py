from __future__ import annotations

import asyncio
import sqlite3

import pytest

from sidekick.wechat.ai import (
    WECHAT_IDENTITY_CODEC,
    WeChatIdentityCodec,
)
from sidekick.wechat.api import (
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatSession,
)
from sidekick.wechat.store import WeChatStateRepository


CONNECTOR_KEY = "http://127.0.0.1:18188"
ACCOUNT_ID = "wxid_self"
GROUP_ID = "56825427596@chatroom"


class PausingWeChatStateRepository(WeChatStateRepository):
    def __init__(self, path):
        super().__init__(path)
        self.pause_chat_refresh = False
        self.chat_refresh_deleted = asyncio.Event()
        self.resume_chat_refresh = asyncio.Event()

    async def _replace_chats(
        self,
        connector_key: str,
        account_id: str,
        generation: int,
        chats: WeChatChatList,
        *,
        now: float,
    ) -> None:
        if self.pause_chat_refresh:
            await self._require_connection().execute(
                "DELETE FROM wechat_chats WHERE connector_key = ? AND account_id = ?",
                (connector_key, account_id),
            )
            self.chat_refresh_deleted.set()
            await self.resume_chat_refresh.wait()
        await super()._replace_chats(
            connector_key,
            account_id,
            generation,
            chats,
            now=now,
        )


@pytest.mark.asyncio
async def test_wechat_store_does_not_create_retired_message_or_ingress_tables(
    tmp_path,
) -> None:
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        cursor = await store._require_connection().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        tables = {str(row["name"]) async for row in cursor}

        assert tables.isdisjoint(
            {
                "wechat_messages",
                "wechat_pending_ai_work",
                "wechat_processed_message_revisions",
                "wechat_processed_messages",
                "wechat_projection_counters",
            }
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_store_leaves_existing_legacy_messages_untouched(tmp_path) -> None:
    state_path = tmp_path / "wechat.db"
    connection = sqlite3.connect(state_path)
    connection.executescript(
        """
        CREATE TABLE wechat_messages (
            local_order INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_order INTEGER,
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
        INSERT INTO wechat_messages (
            memory_order, connector_key, account_id, chat_id, message_id,
            direction, message_type, sender_id, content, content_redacted,
            timestamp, removed, last_event_cursor, updated_at
        ) VALUES (
            NULL, 'connector', 'account', 'chat', 'message',
            'in', 'text', 'sender', 'legacy body', 0,
            1, 0, 'cursor', 1
        );
        """
    )
    connection.commit()
    connection.close()

    store = await WeChatStateRepository(state_path).connect()
    await store.close()

    connection = sqlite3.connect(state_path)
    try:
        row = connection.execute(
            "SELECT memory_order, content FROM wechat_messages"
        ).fetchone()
        assert row == (None, "legacy body")
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_wechat_store_upgrades_existing_generated_send_queue(tmp_path) -> None:
    state_path = tmp_path / "wechat.db"
    connection = sqlite3.connect(state_path)
    connection.executescript(
        """
        CREATE TABLE wechat_generated_send_reservations (
            connector_key TEXT NOT NULL,
            account_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            fingerprint BLOB NOT NULL CHECK (length(fingerprint) = 32),
            created_at REAL NOT NULL,
            PRIMARY KEY (connector_key, account_id, request_id)
        );
        CREATE TABLE wechat_generated_send_leases (
            connector_key TEXT NOT NULL,
            account_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            lease_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (connector_key, account_id, request_id, lease_id)
        );
        """
    )
    connection.execute(
        """
        INSERT INTO wechat_generated_send_reservations (
            connector_key, account_id, request_id, chat_id,
            fingerprint, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (CONNECTOR_KEY, ACCOUNT_ID, "request-existing", GROUP_ID, b"x" * 32, 1.0),
    )
    connection.execute(
        """
        INSERT INTO wechat_generated_send_leases (
            connector_key, account_id, request_id, lease_id, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (CONNECTOR_KEY, ACCOUNT_ID, "request-existing", "stale-lease", 1.0),
    )
    connection.commit()
    connection.close()

    store = await WeChatStateRepository(state_path).connect()
    try:
        await store.acquire_adapter_ownership()
        await store.recover_generated_send_leases(CONNECTOR_KEY)
        reservations = await store.list_due_generated_send_reservations(
            CONNECTOR_KEY,
            ACCOUNT_ID,
            now=1.0,
            limit=1,
        )

        assert len(reservations) == 1
        assert reservations[0].request_id == "request-existing"
        assert reservations[0].reconciliation_attempts == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_adapter_ownership_is_exclusive_and_released_on_close(
    tmp_path,
) -> None:
    state_path = tmp_path / "wechat.db"
    first = WeChatStateRepository(state_path)
    second = WeChatStateRepository(state_path)
    try:
        await first.acquire_adapter_ownership()
        await first.connect()
        with pytest.raises(RuntimeError, match="already active"):
            await second.acquire_adapter_ownership()
        with pytest.raises(RuntimeError, match="requires adapter ownership"):
            await second.recover_generated_send_leases(CONNECTOR_KEY)

        await first.close()
        await second.acquire_adapter_ownership()
        await second.connect()
    finally:
        await second.close()
        await first.close()


@pytest.mark.asyncio
async def test_wechat_adapter_ownership_rejects_symlink_alias(tmp_path) -> None:
    state_path = tmp_path / "wechat.db"
    owner = await WeChatStateRepository(state_path).connect()
    await owner.acquire_adapter_ownership()
    alias_path = tmp_path / "wechat-alias.db"
    alias_path.symlink_to(state_path.name)
    alias = WeChatStateRepository(alias_path)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            await alias.acquire_adapter_ownership()
    finally:
        await alias.close()
        await owner.close()


@pytest.mark.asyncio
async def test_wechat_adapter_ownership_rejects_hard_link_alias(tmp_path) -> None:
    state_path = tmp_path / "wechat.db"
    owner = await WeChatStateRepository(state_path).connect()
    await owner.acquire_adapter_ownership()
    alias_path = tmp_path / "wechat-hard-link.db"
    alias_path.hardlink_to(state_path)
    alias = WeChatStateRepository(alias_path)
    try:
        with pytest.raises(RuntimeError, match="hard-linked"):
            await alias.acquire_adapter_ownership()
    finally:
        await alias.close()
        await owner.close()


@pytest.mark.asyncio
async def test_generated_send_lease_recovery_rolls_back_on_cancellation(
    tmp_path,
    monkeypatch,
) -> None:
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    await store.acquire_adapter_ownership()
    lease_id = await store.reserve_generated_send(
        CONNECTOR_KEY,
        ACCOUNT_ID,
        GROUP_ID,
        "request-cancelled-recovery",
        b"x" * 32,
    )
    connection = store._require_connection()
    original_commit = connection.commit

    async def cancelled_commit() -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(connection, "commit", cancelled_commit)
    try:
        with pytest.raises(asyncio.CancelledError):
            await store.recover_generated_send_leases(CONNECTOR_KEY)

        monkeypatch.setattr(connection, "commit", original_commit)
        assert connection.in_transaction is False
        await store.defer_generated_send(
            CONNECTOR_KEY,
            ACCOUNT_ID,
            "request-cancelled-recovery",
            lease_id,
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_stale_reconciliation_lease_cannot_mutate_replacement(
    tmp_path,
) -> None:
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    await store.acquire_adapter_ownership()
    original_lease = await store.reserve_generated_send(
        CONNECTOR_KEY,
        ACCOUNT_ID,
        GROUP_ID,
        "request-replaced",
        b"x" * 32,
    )
    await store.defer_generated_send(
        CONNECTOR_KEY,
        ACCOUNT_ID,
        "request-replaced",
        original_lease,
    )
    stale_lease = await store.claim_generated_send_reconciliation(
        CONNECTOR_KEY,
        ACCOUNT_ID,
        "request-replaced",
    )
    assert stale_lease is not None
    await store.recover_generated_send_leases(CONNECTOR_KEY)
    replacement_lease = await store.reserve_generated_send(
        CONNECTOR_KEY,
        ACCOUNT_ID,
        GROUP_ID,
        "request-replaced",
        b"x" * 32,
    )
    try:
        assert await store.confirm_generated_send(
            CONNECTOR_KEY,
            ACCOUNT_ID,
            GROUP_ID,
            "request-replaced",
            "message-stale",
            stale_lease,
        ) is False
        assert await store.defer_generated_send_reconciliation(
            CONNECTOR_KEY,
            ACCOUNT_ID,
            "request-replaced",
            stale_lease,
            next_attempt_at=123.0,
        ) is False

        reservations = await store.list_generated_send_reservations(
            CONNECTOR_KEY,
            ACCOUNT_ID,
        )
        assert len(reservations) == 1
        assert reservations[0].reconciliation_attempts == 0
        cursor = await store._require_connection().execute(
            """
            SELECT lease_id
            FROM wechat_generated_send_leases
            WHERE connector_key = ? AND account_id = ? AND request_id = ?
            """,
            (CONNECTOR_KEY, ACCOUNT_ID, "request-replaced"),
        )
        row = await cursor.fetchone()
        assert row is not None and row["lease_id"] == replacement_lease
        cursor = await store._require_connection().execute(
            """
            SELECT 1
            FROM wechat_generated_messages
            WHERE connector_key = ? AND account_id = ?
              AND chat_id = ? AND message_id = ?
            """,
            (CONNECTOR_KEY, ACCOUNT_ID, GROUP_ID, "message-stale"),
        )
        assert await cursor.fetchone() is None
    finally:
        await store.defer_generated_send(
            CONNECTOR_KEY,
            ACCOUNT_ID,
            "request-replaced",
            replacement_lease,
        )
        await store.close()




def session(
    *, generation: int = 41, cursor: str = "bootstrap-session"
) -> WeChatSession:
    return WeChatSession(
        status="logged_in",
        self_id=ACCOUNT_ID,
        display_name="Sidekick",
        hook_connected=True,
        connection_generation=generation,
        content_redacted=False,
        cursor=cursor,
    )


def chat_list(
    *, generation: int = 41, cursor: str = "bootstrap-chats"
) -> WeChatChatList:
    return WeChatChatList(
        chats=(WeChatChat(id=GROUP_ID, type="group", display_name="Example group"),),
        snapshot=WeChatChatSnapshot(
            id=f"snapshot-{generation}",
            complete=True,
            current=True,
            count=1,
            cursor=cursor,
            connection_generation=generation,
        ),
        cursor=cursor,
    )




def test_wechat_identity_codec_round_trips_opaque_ids() -> None:
    scope_id = WECHAT_IDENTITY_CODEC.scope_id(GROUP_ID)

    assert scope_id == "wechat:chat:56825427596%40chatroom"
    assert WECHAT_IDENTITY_CODEC.parse_scope_id(scope_id) == GROUP_ID
    assert WECHAT_IDENTITY_CODEC.actor_id("wxid_alice") == ("wechat:user:wxid_alice")
    assert (
        WECHAT_IDENTITY_CODEC.message_source_id(
            GROUP_ID,
            "4159667620982040828",
        )
        == "wechat:message:56825427596%40chatroom:4159667620982040828"
    )
    assert WECHAT_IDENTITY_CODEC.parse_message_source_id(
        "wechat:message:56825427596%40chatroom:4159667620982040828"
    ) == (GROUP_ID, "4159667620982040828")


def test_wechat_account_identity_codec_isolates_memory_banks() -> None:
    first = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    second = WeChatIdentityCodec(account_id="wxid_other")
    scope_id = first.scope_id(GROUP_ID)
    source_id = first.message_source_id(GROUP_ID, "4159667620982040828")

    assert scope_id == ("wechat:account:wxid_self:chat:56825427596%40chatroom")
    assert first.parse_scope_id(scope_id) == GROUP_ID
    assert second.parse_scope_id(scope_id) is None
    assert first.parse_message_source_id(source_id) == (
        GROUP_ID,
        "4159667620982040828",
    )
    assert second.parse_message_source_id(source_id) is None




@pytest.mark.asyncio
async def test_wechat_store_hides_partial_chat_refreshes_from_concurrent_reads(
    tmp_path,
) -> None:
    store = await PausingWeChatStateRepository(tmp_path / "wechat.db").connect()
    refresh_task = None
    read_task = None
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
        )
        store.pause_chat_refresh = True
        refresh_task = asyncio.create_task(
            store.refresh_chats(CONNECTOR_KEY, chat_list(cursor="refresh-chats"))
        )
        await store.chat_refresh_deleted.wait()

        read_task = asyncio.create_task(store.get_chat(CONNECTOR_KEY, GROUP_ID))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(read_task), timeout=0.1)

        store.resume_chat_refresh.set()
        await refresh_task
        chat = await read_task
        assert chat is not None
        assert chat.id == GROUP_ID
    finally:
        store.resume_chat_refresh.set()
        if refresh_task is not None:
            await asyncio.gather(refresh_task, return_exceptions=True)
        if read_task is not None:
            await asyncio.gather(read_task, return_exceptions=True)
        await store.close()
