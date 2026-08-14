from __future__ import annotations

import pytest

from sidekick.wechat.api import (
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatSession,
)
from sidekick.wechat.store import WeChatStateRepository


CONNECTOR_KEY = "http://wechat-connector:18188"
CHAT_ID = "56825427596@chatroom"
MESSAGE_ID = "4159667620982040828"


def session() -> WeChatSession:
    return WeChatSession(
        status="logged_in",
        self_id="wxid_self",
        display_name="Sidekick",
        hook_connected=True,
        connection_generation=41,
        content_redacted=False,
        cursor="bootstrap-cursor",
    )


def chats() -> WeChatChatList:
    return WeChatChatList(
        chats=(
            WeChatChat(
                id=CHAT_ID,
                type="group",
                display_name="Example group",
            ),
        ),
        snapshot=WeChatChatSnapshot(
            id="snapshot-41",
            complete=True,
            current=True,
            count=1,
            cursor="bootstrap-cursor",
            connection_generation=41,
        ),
        cursor="bootstrap-cursor",
    )


async def open_store(path) -> WeChatStateRepository:
    store = await WeChatStateRepository(path).connect()
    await store.bootstrap(
        connector_key=CONNECTOR_KEY,
        session=session(),
        chats=chats(),
    )
    return store


@pytest.mark.asyncio
async def test_cursor_and_pending_reference_commit_atomically_without_history_write(
    tmp_path,
) -> None:
    store = await open_store(tmp_path / "wechat.db")
    try:
        await store.accept_pending_ai_event(
            CONNECTOR_KEY,
            cursor="event-11",
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
        )

        pending = await store.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.trigger_cursor == "event-11"
        assert pending.kind == "message"
        assert await store.get_cursor(CONNECTOR_KEY) == "event-11"
        assert await store.count_messages(CONNECTOR_KEY) == 0

        connection = store._require_connection()
        await connection.execute(
            """
            CREATE TRIGGER reject_connector_cursor_update
            BEFORE UPDATE OF cursor ON wechat_connectors
            BEGIN
                SELECT RAISE(ABORT, 'cursor rejected');
            END
            """
        )
        await connection.commit()
        with pytest.raises(Exception, match="cursor rejected"):
            await store.accept_pending_ai_event(
                CONNECTOR_KEY,
                cursor="event-12",
                chat_id=CHAT_ID,
                message_id="4159667620982040829",
                kind="message",
            )

        assert await store.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            "4159667620982040829",
        ) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_replay_is_idempotent_and_newer_cursor_survives_stale_worker(
    tmp_path,
) -> None:
    store = await open_store(tmp_path / "wechat.db")
    try:
        await store.accept_pending_ai_event(
            CONNECTOR_KEY,
            cursor="event-11",
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
        )
        claimed = await store.claim_pending_ai_work(CONNECTOR_KEY, now=100)
        assert claimed is not None

        await store.accept_pending_ai_event(
            CONNECTOR_KEY,
            cursor="event-11",
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
        )
        replayed = await store.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert replayed == claimed

        await store.accept_pending_ai_event(
            CONNECTOR_KEY,
            cursor="event-12",
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message_remove",
        )
        assert await store.complete_pending_ai_work(
            claimed,
            version="mv1:present",
            outcome="completed",
            now=101,
        ) is False

        pending = await store.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.trigger_cursor == "event-12"
        assert pending.kind == "message_remove"
        assert pending.lease_id is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_processed_revision_is_deduplicated_by_semantic_version(tmp_path) -> None:
    store = await open_store(tmp_path / "wechat.db")
    try:
        await store.accept_pending_ai_event(
            CONNECTOR_KEY,
            cursor="event-11",
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
        )
        first = await store.claim_pending_ai_work(CONNECTOR_KEY, now=100)
        assert first is not None
        assert await store.begin_pending_ai_execution(
            first,
            version="mv1:same",
            now=100,
        ) == "started"
        assert await store.complete_pending_ai_work(
            first,
            version="mv1:same",
            outcome="completed",
            now=101,
        ) is True

        await store.accept_pending_ai_event(
            CONNECTOR_KEY,
            cursor="event-12",
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
        )
        replay = await store.claim_pending_ai_work(CONNECTOR_KEY, now=102)
        assert replay is not None
        assert await store.begin_pending_ai_execution(
            replay,
            version="mv1:same",
            now=102,
        ) == "duplicate"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_restart_does_not_replay_an_execution_with_unknown_side_effects(
    tmp_path,
) -> None:
    path = tmp_path / "wechat.db"
    first_store = await open_store(path)
    await first_store.accept_pending_ai_event(
        CONNECTOR_KEY,
        cursor="event-11",
        chat_id=CHAT_ID,
        message_id=MESSAGE_ID,
        kind="message",
    )
    claimed = await first_store.claim_pending_ai_work(CONNECTOR_KEY, now=100)
    assert claimed is not None
    assert await first_store.begin_pending_ai_execution(
        claimed,
        version="mv1:running",
        now=100,
    ) == "started"
    await first_store.close()

    restarted = await WeChatStateRepository(path).connect()
    try:
        await restarted.recover_pending_ai_work(CONNECTOR_KEY, now=200)

        pending = await restarted.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.status == "failed_unknown"
        assert await restarted.get_processed_revision_status(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
            "mv1:running",
        ) == "failed_unknown"
        assert await restarted.claim_pending_ai_work(
            CONNECTOR_KEY,
            now=201,
        ) is None
    finally:
        await restarted.close()
