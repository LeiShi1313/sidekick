from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import sqlite3

import pytest

from sidekick.wechat.ai import (
    WECHAT_IDENTITY_CODEC,
    WeChatHistorySource,
    WeChatIdentityCodec,
    WeChatMemoryScopeTargetResolver,
)
from sidekick.wechat.api import (
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatConnectorMessage,
    WeChatEvent,
    WeChatGroupMember,
    WeChatGroupMemberList,
    WeChatMessageList,
    WeChatSession,
    WeChatUser,
    WeChatUserList,
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


def connector_message(
    message_id: str,
    content: str,
    *,
    chat_id: str = GROUP_ID,
    direction: str = "in",
    sender_id: str = "wxid_alice",
    reply_to_message_id: str | None = None,
    timestamp: int = 1_783_772_734,
    message_type: str = "text",
    content_redacted: bool = False,
    media_id: str | None = None,
) -> WeChatConnectorMessage:
    return WeChatConnectorMessage(
        id=message_id,
        chat_id=chat_id,
        direction=direction,
        message_type=message_type,
        sender_id=sender_id,
        reply_to_message_id=reply_to_message_id,
        content=content,
        content_redacted=content_redacted,
        timestamp=timestamp,
        source="wechat+localdb",
        sequence=None,
        media_id=media_id,
    )


@pytest.mark.asyncio
async def test_wechat_store_resolves_group_scoped_sender_labels(tmp_path) -> None:
    other_group_id = "56825427597@chatroom"
    direct_chat_id = "wxid_alice"
    chats = (
        WeChatChat(id=GROUP_ID, type="group", display_name="First group"),
        WeChatChat(id=other_group_id, type="group", display_name="Second group"),
        WeChatChat(id=direct_chat_id, type="direct", display_name="Alice"),
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=WeChatChatList(
                chats=chats,
                snapshot=WeChatChatSnapshot(
                    id="identity-snapshot",
                    complete=True,
                    current=True,
                    count=len(chats),
                    cursor="identity-chats",
                    connection_generation=41,
                ),
                cursor="identity-chats",
            ),
            messages=WeChatMessageList(
                messages=(
                    connector_message("1001", "first", chat_id=GROUP_ID),
                    connector_message("1002", "second", chat_id=other_group_id),
                    connector_message("1003", "direct", chat_id=direct_chat_id),
                    connector_message(
                        "1004",
                        "member fallback",
                        chat_id=GROUP_ID,
                        sender_id="wxid_bob",
                    ),
                    connector_message(
                        "1005",
                        "identifier fallback",
                        chat_id=other_group_id,
                        sender_id="wxid_unknown",
                    ),
                ),
                cursor="identity-messages",
            ),
        )
        await store.refresh_users(
            CONNECTOR_KEY,
            WeChatUserList(
                users=(WeChatUser(id="wxid_alice", display_name="Alice Global"),),
                cursor="users-1",
            ),
        )
        await store.refresh_group_members(
            CONNECTOR_KEY,
            GROUP_ID,
            WeChatGroupMemberList(
                group_id=GROUP_ID,
                members=(
                    WeChatGroupMember(
                        group_id=GROUP_ID,
                        user_id="wxid_alice",
                        display_name="Alice Stale",
                        nickname="Alice in First",
                    ),
                    WeChatGroupMember(
                        group_id=GROUP_ID,
                        user_id="wxid_bob",
                        display_name="Bob Member",
                        nickname=None,
                    ),
                ),
                cursor="members-first",
            ),
        )
        await store.refresh_group_members(
            CONNECTOR_KEY,
            other_group_id,
            WeChatGroupMemberList(
                group_id=other_group_id,
                members=(
                    WeChatGroupMember(
                        group_id=other_group_id,
                        user_id="wxid_alice",
                        display_name="Alice Stale",
                        nickname=None,
                    ),
                ),
                cursor="members-second",
            ),
        )

        resolved = []
        for chat_id, message_id in (
            (GROUP_ID, "1001"),
            (other_group_id, "1002"),
            (direct_chat_id, "1003"),
            (GROUP_ID, "1004"),
            (other_group_id, "1005"),
        ):
            message = await store.get_message(CONNECTOR_KEY, chat_id, message_id)
            assert message is not None
            resolved.append(message.sender_display_name)

        assert resolved == [
            "Alice in First",
            "Alice Global",
            "Alice Global",
            "Bob Member",
            "wxid_unknown",
        ]

        await store.refresh_user(
            CONNECTOR_KEY,
            "wxid_alice",
            WeChatUser(id="wxid_alice", display_name="Alice Renamed"),
        )
        group_message = await store.get_message(
            CONNECTOR_KEY,
            other_group_id,
            "1002",
        )
        direct_message = await store.get_message(
            CONNECTOR_KEY,
            direct_chat_id,
            "1003",
        )
        assert group_message is not None
        assert direct_message is not None
        assert group_message.sender_display_name == "Alice Renamed"
        assert direct_message.sender_display_name == "Alice Renamed"
    finally:
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


def event(payload: dict[str, object], *, cursor: str) -> WeChatEvent:
    return WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": cursor,
            "event": "message",
            "connectionGeneration": 41,
            **payload,
        }
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
async def test_wechat_store_upserts_revisions_and_acks_processing_atomically(tmp_path):
    path = tmp_path / "wechat.db"
    oldest_id = "3159667620982040828"
    command_id = "4159667620982040828"
    store = await WeChatStateRepository(path).connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(
                messages=(
                    connector_message(oldest_id, "earlier", timestamp=1_783_772_700),
                    connector_message(command_id, "/ai hello"),
                ),
                cursor="bootstrap-messages",
            ),
        )

        command = await store.get_message(
            CONNECTOR_KEY,
            GROUP_ID,
            command_id,
        )
        assert command is not None
        assert command.id == command_id
        assert command.chat_id == GROUP_ID
        assert command.chat_type == "group"
        assert command.scope_display_name == "Example group"
        assert await store.get_cursor(CONNECTOR_KEY) == "bootstrap-messages"
        stored_chats = await store.list_chats(CONNECTOR_KEY)
        assert len(stored_chats) == 1
        assert stored_chats[0].chat_id == GROUP_ID
        assert stored_chats[0].chat_type == "group"
        assert stored_chats[0].display_name == "Example group"
        assert stored_chats[0].last_observed_at > 0

        revision = event(
            {
                "id": command_id,
                "chatId": GROUP_ID,
                "direction": "in",
                "messageType": "text",
                "senderId": "wxid_alice",
                "replyToMessageId": oldest_id,
                "content": "/ai hello",
                "timestamp": 1_783_772_734,
                "source": "wechat+localdb",
            },
            cursor="opaque-revision",
        )
        projected = await store.project_event(CONNECTOR_KEY, revision)

        assert projected is not None
        assert projected.reply_to_msg_id == oldest_id
        assert await store.get_cursor(CONNECTOR_KEY) == "bootstrap-messages"
        assert await store.is_processed(projected) is False

        await store.acknowledge_event(
            CONNECTOR_KEY,
            revision.cursor,
            processed_message=projected,
        )

        assert await store.get_cursor(CONNECTOR_KEY) == "opaque-revision"
        assert await store.is_processed(projected) is True
        assert await store.count_messages(CONNECTOR_KEY) == 2

        history = WeChatHistorySource(store, CONNECTOR_KEY)
        recent = await history.fetch_recent(command, before=command, limit=10)
        parent = await history.fetch_message(GROUP_ID, oldest_id)

        assert [message.id for message in recent] == [oldest_id]
        assert parent is not None
        assert parent.raw_text == "earlier"
    finally:
        await store.close()

    restarted = await WeChatStateRepository(path).connect()
    try:
        command = await restarted.get_message(
            CONNECTOR_KEY,
            GROUP_ID,
            command_id,
        )
        assert command is not None
        assert await restarted.get_cursor(CONNECTOR_KEY) == "opaque-revision"
        assert await restarted.is_processed(command) is True
    finally:
        await restarted.close()


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
            messages=WeChatMessageList(messages=(), cursor="bootstrap-messages"),
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


@pytest.mark.asyncio
async def test_wechat_memory_source_reads_stored_windows_and_late_revisions(tmp_path):
    oldest_id = "3159667620982040828"
    latest_id = "4159667620982040828"
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(
                messages=(
                    connector_message(oldest_id, "earlier", timestamp=1_783_772_700),
                    connector_message(latest_id, "latest", timestamp=1_783_772_734),
                ),
                cursor="bootstrap-messages",
            ),
        )
        source = WeChatHistorySource(store, CONNECTOR_KEY)
        latest = await store.get_message(CONNECTOR_KEY, GROUP_ID, latest_id)
        assert latest is not None

        window = await source.fetch_window(
            GROUP_ID,
            since=datetime.fromtimestamp(1_783_772_699, UTC),
            until=datetime.fromtimestamp(1_783_772_735, UTC),
            limit=1,
        )

        assert [message.id for message in window] == [latest_id]

        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(
                messages=(
                    connector_message(oldest_id, "earlier", timestamp=1_783_772_700),
                    connector_message(latest_id, "latest", timestamp=1_783_772_734),
                ),
                cursor="second-bootstrap",
            ),
        )
        unchanged = await store.get_message(
            CONNECTOR_KEY,
            GROUP_ID,
            latest_id,
        )
        assert unchanged is not None
        assert unchanged.memory_cursor == latest.memory_cursor

        revision = event(
            {
                "id": oldest_id,
                "chatId": GROUP_ID,
                "direction": "in",
                "messageType": "text",
                "senderId": "wxid_alice",
                "content": "earlier, corrected",
                "timestamp": 1_783_772_700,
                "source": "wechat+localdb",
            },
            cursor="late-revision",
        )
        revised = await store.project_event(CONNECTOR_KEY, revision)
        assert revised is not None

        after = await source.fetch_after(
            GROUP_ID,
            after_message_id=latest.memory_cursor,
            until=datetime.fromtimestamp(1_783_772_735, UTC),
            limit=10,
        )

        assert [message.id for message in after] == [oldest_id]
        assert after[0].raw_text == "earlier, corrected"
        assert after[0].memory_cursor > latest.memory_cursor

        target = await WeChatMemoryScopeTargetResolver(
            store,
            CONNECTOR_KEY,
        ).resolve(GROUP_ID, include_latest_message=True)
        assert target.chat_id == GROUP_ID
        assert target.display_name == "Example group"
        assert target.latest_message_id == revised.memory_cursor
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_history_excludes_redacted_and_unsupported_messages(tmp_path):
    visible_id = "3159667620982040828"
    redacted_id = "4159667620982040828"
    image_id = "5159667620982040828"
    anchor_id = "6159667620982040828"
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(
                messages=(
                    connector_message(visible_id, "visible", timestamp=1_783_772_700),
                    connector_message(
                        redacted_id,
                        "must not escape",
                        timestamp=1_783_772_701,
                        content_redacted=True,
                    ),
                    connector_message(
                        image_id,
                        "unsupported image metadata",
                        timestamp=1_783_772_702,
                        message_type="image",
                    ),
                    connector_message(anchor_id, "/ai summarize"),
                ),
                cursor="bootstrap-messages",
            ),
        )
        source = WeChatHistorySource(store, CONNECTOR_KEY)
        anchor = await source.fetch_message(GROUP_ID, anchor_id)
        assert anchor is not None

        recent = await source.fetch_recent(anchor, before=anchor, limit=10)
        window = await source.fetch_window(
            GROUP_ID,
            since=datetime.fromtimestamp(1_783_772_699, UTC),
            until=datetime.fromtimestamp(1_783_772_735, UTC),
            limit=10,
        )

        assert [message.id for message in recent] == [visible_id]
        assert [message.id for message in window] == [visible_id, anchor_id]
        assert await source.fetch_message(GROUP_ID, redacted_id) is None
        assert await source.fetch_message(GROUP_ID, image_id) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_store_promotes_shared_history_enrichment_in_place(tmp_path):
    message_id = "5159667620982040828"
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(
                messages=(
                    connector_message(
                        message_id,
                        "[Chat history] Team history",
                        message_type="app",
                    ),
                ),
                cursor="bootstrap-messages",
            ),
        )
        correction = event(
            {
                "id": message_id,
                "chatId": GROUP_ID,
                "direction": "in",
                "messageType": "chat_history",
                "senderId": "wxid_alice",
                "content": "Team history",
                "sharedChatHistory": {
                    "title": "Team history",
                    "itemCount": 2,
                    "items": [
                        {"kind": "text", "senderName": "Alice", "content": "Hi"},
                        {"kind": "image", "senderName": "Bob"},
                    ],
                },
                "timestamp": 1_783_772_734,
                "source": "wechat+localdb",
            },
            cursor="shared-history-correction",
        )

        revised = await store.project_event(CONNECTOR_KEY, correction)
        visible = await store.get_message(CONNECTOR_KEY, GROUP_ID, message_id)
        quoted = await store.get_reply_message(CONNECTOR_KEY, GROUP_ID, message_id)
        memory_window = await WeChatHistorySource(
            store,
            CONNECTOR_KEY,
        ).fetch_window(
            GROUP_ID,
            since=datetime.fromtimestamp(1_783_772_700, UTC),
            until=datetime.fromtimestamp(1_783_772_735, UTC),
            limit=10,
        )

        assert revised is not None
        assert revised.message_type == "chat_history"
        assert revised.raw_text == (
            "[Forwarded chat history]\n"
            "Team history\n"
            "Alice: Hi\n"
            "Bob: [Image]"
        )
        assert visible is not None
        assert quoted is not None
        assert [message.id for message in memory_window] == [message_id]
        assert await store.count_messages(CONNECTOR_KEY) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_store_persists_images_for_direct_reply_lookup_only(tmp_path):
    path = tmp_path / "wechat.db"
    image_id = "5159667620982040828"
    media_id = "0123456789abcdef0123456789abcdef"
    store = await WeChatStateRepository(path).connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(
                messages=(
                    connector_message(
                        image_id,
                        "",
                        message_type="image",
                        content_redacted=True,
                        media_id=media_id,
                    ),
                ),
                cursor="bootstrap-messages",
            ),
        )

        assert await store.get_message(CONNECTOR_KEY, GROUP_ID, image_id) is None
        quoted = await store.get_reply_message(CONNECTOR_KEY, GROUP_ID, image_id)

        assert quoted is not None
        assert quoted.message_type == "image"
        assert quoted.media_id == media_id
    finally:
        await store.close()

    restarted = await WeChatStateRepository(path).connect()
    try:
        quoted = await restarted.get_reply_message(
            CONNECTOR_KEY,
            GROUP_ID,
            image_id,
        )
        assert quoted is not None
        assert quoted.media_id == media_id
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_wechat_store_applies_empty_redaction_revision(tmp_path):
    message_id = "3159667620982040828"
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(
                messages=(connector_message(message_id, "private text"),),
                cursor="bootstrap-messages",
            ),
        )
        redaction = event(
            {
                "id": message_id,
                "chatId": GROUP_ID,
                "direction": "in",
                "messageType": "text",
                "senderId": "wxid_alice",
                "contentRedacted": True,
                "timestamp": 1_783_772_734,
                "source": "wechat+hook",
            },
            cursor="redacted-revision",
        )

        projected = await store.project_event(CONNECTOR_KEY, redaction)
        visible = await WeChatHistorySource(store, CONNECTOR_KEY).fetch_message(
            GROUP_ID,
            message_id,
        )

        assert projected is not None
        assert projected.content_redacted is True
        assert projected.raw_text == ""
        assert visible is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_store_migrates_existing_messages_to_memory_order(tmp_path):
    path = tmp_path / "wechat.db"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE wechat_messages (
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
            INSERT INTO wechat_messages (
                connector_key, account_id, chat_id, message_id, direction,
                message_type, sender_id, content, content_redacted, timestamp,
                source, last_event_cursor, updated_at
            ) VALUES (
                'http://127.0.0.1:18188', 'wxid_self',
                '56825427596@chatroom', '4159667620982040828', 'in',
                'text', 'wxid_alice', 'existing', 0, 1783772734,
                'wechat+localdb', 'legacy-cursor', 1783772734
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    store = await WeChatStateRepository(path).connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(
                messages=(connector_message("4159667620982040828", "existing"),),
                cursor="bootstrap-messages",
            ),
        )
        message = await store.get_message(
            CONNECTOR_KEY,
            GROUP_ID,
            "4159667620982040828",
        )

        assert message is not None
        assert message.memory_cursor == 1
        assert message.media_id is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_store_preserves_existing_cursor_on_same_account_bootstrap(
    tmp_path,
):
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(messages=(), cursor="initial-cut"),
        )
        await store.acknowledge_event(CONNECTOR_KEY, "durable-progress")

        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(cursor="new-chat-cut"),
            messages=WeChatMessageList(messages=(), cursor="new-message-cut"),
        )

        assert await store.get_cursor(CONNECTOR_KEY) == "durable-progress"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_store_rebaselines_cursor_when_endpoint_account_changes(tmp_path):
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=session(),
            chats=chat_list(),
            messages=WeChatMessageList(messages=(), cursor="account-one"),
        )
        await store.acknowledge_event(CONNECTOR_KEY, "account-one-progress")

        replacement = WeChatSession(
            status="logged_in",
            self_id="wxid_other_account",
            display_name="Other",
            hook_connected=True,
            connection_generation=1,
            content_redacted=False,
            cursor="replacement-session",
        )
        await store.bootstrap(
            connector_key=CONNECTOR_KEY,
            session=replacement,
            chats=chat_list(generation=1, cursor="replacement-chats"),
            messages=WeChatMessageList(messages=(), cursor="replacement-messages"),
        )

        assert await store.get_cursor(CONNECTOR_KEY) == "replacement-messages"
        assert await store.get_account_id(CONNECTOR_KEY) == "wxid_other_account"
    finally:
        await store.close()
