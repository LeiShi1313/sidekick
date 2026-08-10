from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from sidekick.ai import AISettings, AIStateRepository
from sidekick.chat.provenance import MessageOrigin, message_fingerprint
from sidekick.chat.output_policy import MAINLAND_MESSAGING_POLICY_ID
from sidekick.channel_status import ChannelOpsSettings
from sidekick.plugins.base import command_registry
from sidekick.plugins.wechat_ai import WeChatAI, WeChatRuntimeSettings
from sidekick.wechat.ai import WeChatChatTransport, WeChatQuotedImageDescriber
from sidekick.wechat.api import (
    WeChatCapabilities,
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
from sidekick.wechat.service import (
    WeChatEventPump,
    bootstrap_wechat_channel,
)
from sidekick.wechat.store import WeChatStateRepository


CONNECTOR_KEY = "http://127.0.0.1:18188"
ACCOUNT_ID = "wxid_self"
CHAT_ID = "56825427596@chatroom"


def test_wechat_runtime_settings_and_cli_command(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "wechat.db"
    monkeypatch.setenv("SIDEKICK_WECHAT_URL", "http://127.0.0.1:18189/")
    monkeypatch.setenv("SIDEKICK_WECHAT_TOKEN", "test-token")
    monkeypatch.setenv("SIDEKICK_WECHAT_STATE_PATH", str(state_path))
    monkeypatch.setenv("SIDEKICK_WECHAT_RECONNECT_DELAY", "1.5")

    settings = WeChatRuntimeSettings.from_env()

    assert settings.connector_url == "http://127.0.0.1:18189"
    assert settings.token == "test-token"
    assert settings.state_path == state_path
    assert settings.reconnect_delay == 1.5
    assert command_registry.as_fire_commands()["wechat"]["ai"]


@pytest.mark.asyncio
async def test_duplicate_wechat_adapter_fails_before_opening_ai_state(
    monkeypatch,
    tmp_path,
) -> None:
    state_path = tmp_path / "wechat.db"
    owner = await WeChatStateRepository(state_path).connect()
    await owner.acquire_adapter_ownership()

    class RecordingAIStore:
        def __init__(self) -> None:
            self.connect_calls = 0

        async def connect(self) -> None:
            self.connect_calls += 1

        async def close(self) -> None:
            pass

    class AsyncCloser:
        async def close(self) -> None:
            pass

    class AdapterStatus:
        def update(self, **_values) -> None:
            pass

    ai_store = RecordingAIStore()
    plugin = object.__new__(WeChatAI)
    plugin._wechat_store = WeChatStateRepository(state_path)
    plugin._ai_store = ai_store
    plugin._client = SimpleNamespace(
        base_url=CONNECTOR_KEY,
        close=AsyncCloser().close,
    )
    plugin._gateway = AsyncCloser()
    plugin._memory = None
    plugin._ops_server = AsyncCloser()
    plugin._adapter_status = AdapterStatus()
    plugin._channel_runtime = None
    plugin._generated_send_reconciliation_task = None
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *_args: None)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            await plugin._run()
        assert ai_store.connect_calls == 0
        assert plugin._wechat_store._connection is None
    finally:
        await owner.close()


def test_wechat_channel_runtime_wires_account_scoped_memory(tmp_path) -> None:
    memory = object()
    plugin = object.__new__(WeChatAI)
    plugin._settings = AISettings(
        agent_url="http://agent.invalid",
        agent_token="test-agent-token-that-is-long-enough",
        state_path=tmp_path / "ai.db",
    )
    plugin._ops_settings = ChannelOpsSettings(
        instance_id="wechat-test",
        token="channel-ops-token-that-is-long-enough",
    )
    plugin._client = SimpleNamespace(base_url=CONNECTOR_KEY)
    plugin._wechat_store = WeChatStateRepository(tmp_path / "wechat.db")
    plugin._ai_store = AIStateRepository(tmp_path / "ai.db")
    plugin._gateway = object()
    plugin._memory = memory
    plugin.logger = logging.getLogger("test-wechat-memory-runtime")

    connector = FakeConnectorClient()
    runtime = plugin._build_channel_runtime(
        SimpleNamespace(
            session=connector.session,
            capabilities=connector.capabilities,
        )
    )

    assert runtime.handler._memory is memory
    assert runtime.handler._dream_runner is runtime.memory_ingestor
    assert runtime.handler._memory_scope_resolver is not None
    assert runtime.dream_scheduler is not None
    assert runtime.continuous_scheduler is not None
    assert runtime.outbox_scheduler is not None
    assert runtime.identity_codec.scope_id(CHAT_ID) == (
        "wechat:account:wxid_self:chat:56825427596%40chatroom"
    )
    assert isinstance(
        runtime.handler._prompt_builder.quoted_attachment_describer,
        WeChatQuotedImageDescriber,
    )
    assert runtime.handler._transport._native_reply_ready is True
    assert MAINLAND_MESSAGING_POLICY_ID in (
        runtime.handler._prompt_builder.system_prompt
    )
    assert (
        runtime.handler._responder._output_policy.policy_id
        == MAINLAND_MESSAGING_POLICY_ID
    )


@pytest.mark.asyncio
async def test_wechat_channel_runtime_restarts_memory_schedulers() -> None:
    calls: list[str] = []

    class Scheduler:
        def __init__(self, name: str):
            self.name = name

        def start(self) -> None:
            calls.append(f"start:{self.name}")

        async def close(self) -> None:
            calls.append(f"close:{self.name}")

    class Transport:
        async def reconcile_pending(self, account_id: str) -> None:
            calls.append(f"reconcile:{account_id}")

    old = SimpleNamespace(
        continuous_scheduler=Scheduler("old-continuous"),
        dream_scheduler=Scheduler("old-dream"),
        outbox_scheduler=Scheduler("old-outbox"),
    )
    new_handler = object()
    new = SimpleNamespace(
        handler=new_handler,
        transport=Transport(),
        continuous_scheduler=Scheduler("new-continuous"),
        dream_scheduler=Scheduler("new-dream"),
        outbox_scheduler=Scheduler("new-outbox"),
    )
    plugin = object.__new__(WeChatAI)
    plugin._channel_runtime = old
    plugin._build_channel_runtime = lambda _bootstrap: new

    handler = await plugin._activate_channel_runtime(
        SimpleNamespace(session=SimpleNamespace(self_id="wxid_self"))
    )
    await plugin._close_channel_runtime()

    assert handler is new_handler
    assert calls == [
        "close:old-continuous",
        "close:old-dream",
        "close:old-outbox",
        "start:new-continuous",
        "start:new-dream",
        "start:new-outbox",
        "reconcile:wxid_self",
        "close:new-continuous",
        "close:new-dream",
        "close:new-outbox",
    ]


@pytest.mark.asyncio
async def test_wechat_channel_activation_does_not_wait_for_reconciliation() -> None:
    reconcile_started = asyncio.Event()
    reconcile_cancelled = asyncio.Event()

    class Transport:
        async def reconcile_pending(self, _account_id: str) -> None:
            reconcile_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                reconcile_cancelled.set()

    runtime = SimpleNamespace(
        handler=object(),
        transport=Transport(),
        continuous_scheduler=None,
        dream_scheduler=None,
        outbox_scheduler=None,
    )
    plugin = object.__new__(WeChatAI)
    plugin._channel_runtime = None
    plugin._build_channel_runtime = lambda _bootstrap: runtime

    handler = await asyncio.wait_for(
        plugin._activate_channel_runtime(
            SimpleNamespace(session=SimpleNamespace(self_id="wxid_self"))
        ),
        timeout=0.1,
    )
    await asyncio.wait_for(reconcile_started.wait(), timeout=0.1)
    await plugin._close_channel_runtime()

    assert handler is runtime.handler
    assert reconcile_cancelled.is_set()


class FakeConnectorClient:
    def __init__(self, events=()):
        self.session = WeChatSession(
            status="logged_in",
            self_id=ACCOUNT_ID,
            display_name="Sidekick",
            hook_connected=True,
            connection_generation=41,
            content_redacted=False,
            cursor="10",
        )
        self.capabilities = WeChatCapabilities(
            receive_text=True,
            receive_shared_chat_history=True,
            stable_inbound_message_ids=True,
            send_text=True,
            send_reply=True,
            send_native_reply=True,
            request_idempotency=True,
            outbound_stable_message_id=True,
            websocket=True,
            cursor=True,
            replay=True,
            durable_cursor=True,
            text_send_ready=True,
            reply_send_ready=True,
            connection_generation=41,
            history=False,
            inbound_image_download=True,
            request_original_image=True,
        )
        self.chats = WeChatChatList(
            chats=(WeChatChat(id=CHAT_ID, type="group", display_name="Example"),),
            snapshot=WeChatChatSnapshot(
                id="snapshot-41",
                complete=True,
                current=True,
                count=1,
                cursor="10",
                connection_generation=41,
            ),
            cursor="10",
        )
        self.messages = WeChatMessageList(messages=(), cursor="10")
        self.users = WeChatUserList(users=(), cursor="10")
        self.group_members = {
            CHAT_ID: WeChatGroupMemberList(
                group_id=CHAT_ID,
                members=(),
                cursor="10",
                snapshot_complete=False,
                snapshot_current=False,
                snapshot_connection_generation=None,
            )
        }
        self.user_details: dict[str, WeChatUser | None] = {}
        self.event_values = list(events)
        self.after_values: list[str] = []
        self.chat_reads = 0
        self.user_reads = 0
        self.user_detail_reads: list[str] = []
        self.group_member_reads: list[str] = []

    async def get_session(self):
        return self.session

    async def get_capabilities(self):
        return self.capabilities

    async def get_chats(self):
        self.chat_reads += 1
        return self.chats

    async def get_messages(self, *, limit):
        assert limit == 1_000
        return self.messages

    async def get_users(self):
        self.user_reads += 1
        return self.users

    async def get_user(self, user_id):
        self.user_detail_reads.append(user_id)
        if user_id in self.user_details:
            return self.user_details[user_id]
        return next((user for user in self.users.users if user.id == user_id), None)

    async def get_group_members(self, group_id):
        self.group_member_reads.append(group_id)
        return self.group_members[group_id]

    async def events(self, *, after):
        self.after_values.append(after)
        for event in self.event_values:
            yield event


class RecordingHandler:
    def __init__(self, error: Exception | None = None):
        self.messages = []
        self.error = error

    async def handle(self, message):
        self.messages.append(message)
        if self.error is not None:
            raise self.error
        return True


def message_event(
    *,
    cursor: str,
    message_id: str = "4159667620982040828",
    content: str = "/ai hello",
    reply_to: str | None = None,
    connection_generation: int = 41,
    message_type: str = "text",
    content_redacted: bool = False,
    sender_id: str | None = ACCOUNT_ID,
    direction: str = "out",
    shared_chat_history: dict[str, object] | None = None,
) -> WeChatEvent:
    payload = {
        "schemaVersion": "wechat-bridge/v1alpha1",
        "cursor": cursor,
        "event": "message",
        "id": message_id,
        "chatId": CHAT_ID,
        "direction": direction,
        "messageType": message_type,
        "content": content,
        "timestamp": 1_783_772_734,
        "connectionGeneration": connection_generation,
    }
    if sender_id is not None:
        payload["senderId"] = sender_id
    if content_redacted:
        payload["contentRedacted"] = True
    if reply_to is not None:
        payload["replyToMessageId"] = reply_to
    if shared_chat_history is not None:
        payload["sharedChatHistory"] = shared_chat_history
    return WeChatEvent.parse(payload)


@pytest.mark.asyncio
async def test_wechat_bootstrap_hydrates_readable_sender_labels(tmp_path) -> None:
    client = FakeConnectorClient()
    client.messages = WeChatMessageList(
        messages=(
            WeChatConnectorMessage(
                id="4159667620982040828",
                chat_id=CHAT_ID,
                direction="in",
                message_type="text",
                sender_id="wxid_alice",
                reply_to_message_id=None,
                content="The launch is Monday",
                content_redacted=False,
                timestamp=1_783_772_734,
                source="wechat+localdb",
                sequence=None,
            ),
        ),
        cursor="10",
    )
    client.users = WeChatUserList(
        users=(WeChatUser(id="wxid_alice", display_name="Alice Global"),),
        cursor="10",
    )
    client.group_members[CHAT_ID] = WeChatGroupMemberList(
        group_id=CHAT_ID,
        members=(
            WeChatGroupMember(
                group_id=CHAT_ID,
                user_id="wxid_alice",
                display_name="Alice Global",
                nickname="项目阿丽",
            ),
        ),
        cursor="10",
        snapshot_complete=False,
        snapshot_current=False,
        snapshot_connection_generation=None,
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        message = await store.get_message(
            CONNECTOR_KEY,
            CHAT_ID,
            "4159667620982040828",
        )

        assert message is not None
        assert message.sender_id == "wxid_alice"
        assert message.sender_display_name == "项目阿丽"
        assert client.user_reads == 1
        assert client.group_member_reads == [CHAT_ID]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_bootstrap_falls_back_to_sender_id_when_identity_is_unavailable(
    tmp_path,
) -> None:
    class UnavailableIdentityClient(FakeConnectorClient):
        async def get_users(self):
            raise ConnectionError("user directory unavailable")

        async def get_group_members(self, group_id):
            raise ConnectionError(f"member directory unavailable for {group_id}")

    client = UnavailableIdentityClient()
    client.messages = WeChatMessageList(
        messages=(
            WeChatConnectorMessage(
                id="4159667620982040828",
                chat_id=CHAT_ID,
                direction="in",
                message_type="text",
                sender_id="wxid_alice",
                reply_to_message_id=None,
                content="The launch is Monday",
                content_redacted=False,
                timestamp=1_783_772_734,
                source="wechat+localdb",
                sequence=None,
            ),
        ),
        cursor="10",
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        message = await store.get_message(
            CONNECTOR_KEY,
            CHAT_ID,
            "4159667620982040828",
        )

        assert message is not None
        assert message.sender_display_name == "wxid_alice"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_refreshes_identity_before_ack(tmp_path) -> None:
    profile_event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "11",
            "event": "user_profile",
            "status": "changed",
            "id": "sha256:profile-1",
            "userId": "wxid_alice",
            "connectionGeneration": 41,
        }
    )
    member_begin = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "12",
            "event": "group_member_snapshot",
            "id": "members-2",
            "groupId": CHAT_ID,
            "status": "begin",
            "connectionGeneration": 41,
        }
    )
    member_row = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "13",
            "event": "group_member",
            "groupId": CHAT_ID,
            "userId": "wxid_alice",
            "connectionGeneration": 41,
            "raw": {"mode": "cache_snapshot", "snapshotId": "members-2"},
        }
    )
    member_end = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "14",
            "event": "group_member_snapshot",
            "id": "members-2",
            "groupId": CHAT_ID,
            "status": "end",
            "connectionGeneration": 41,
        }
    )
    client = FakeConnectorClient((profile_event, member_begin, member_row, member_end))
    client.messages = WeChatMessageList(
        messages=(
            WeChatConnectorMessage(
                id="4159667620982040828",
                chat_id=CHAT_ID,
                direction="in",
                message_type="text",
                sender_id="wxid_alice",
                reply_to_message_id=None,
                content="The launch is Monday",
                content_redacted=False,
                timestamp=1_783_772_734,
                source="wechat+localdb",
                sequence=None,
            ),
        ),
        cursor="10",
    )
    client.users = WeChatUserList(
        users=(WeChatUser(id="wxid_alice", display_name="Alice Old"),),
        cursor="10",
    )
    client.group_members[CHAT_ID] = WeChatGroupMemberList(
        group_id=CHAT_ID,
        members=(
            WeChatGroupMember(
                group_id=CHAT_ID,
                user_id="wxid_alice",
                display_name="Alice Old",
                nickname=None,
            ),
        ),
        cursor="10",
        snapshot_complete=False,
        snapshot_current=False,
        snapshot_connection_generation=None,
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        client.user_details["wxid_alice"] = WeChatUser(
            id="wxid_alice",
            display_name="Alice Renamed",
        )
        client.group_members[CHAT_ID] = WeChatGroupMemberList(
            group_id=CHAT_ID,
            members=(
                WeChatGroupMember(
                    group_id=CHAT_ID,
                    user_id="wxid_alice",
                    display_name="Alice Renamed",
                    nickname="New Group Alias",
                ),
            ),
            cursor="14",
            snapshot_complete=False,
            snapshot_current=False,
            snapshot_connection_generation=None,
        )

        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(RecordingHandler(), asyncio.Event())
        message = await store.get_message(
            CONNECTOR_KEY,
            CHAT_ID,
            "4159667620982040828",
        )

        assert result == "reconnect"
        assert message is not None
        assert message.sender_display_name == "New Group Alias"
        assert client.user_detail_reads == ["wxid_alice"]
        assert client.group_member_reads == [CHAT_ID, CHAT_ID]
        assert await store.get_cursor(CONNECTOR_KEY) == "14"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_clears_group_alias_directory_before_ack(
    tmp_path,
) -> None:
    directory_event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "11",
            "event": "group_member_directory",
            "connectionGeneration": 41,
            "status": "changed",
            "source": "wechat+localdb-contact",
            "id": "sha256:directory-2",
            "groupId": CHAT_ID,
        }
    )
    client = FakeConnectorClient((directory_event,))
    client.messages = WeChatMessageList(
        messages=(
            WeChatConnectorMessage(
                id="4159667620982040828",
                chat_id=CHAT_ID,
                direction="in",
                message_type="text",
                sender_id="wxid_alice",
                reply_to_message_id=None,
                content="The launch is Monday",
                content_redacted=False,
                timestamp=1_783_772_734,
                source="wechat+localdb",
                sequence=None,
            ),
            WeChatConnectorMessage(
                id="4159667620982040829",
                chat_id=CHAT_ID,
                direction="in",
                message_type="text",
                sender_id="wxid_bob",
                reply_to_message_id=None,
                content="The review is Tuesday",
                content_redacted=False,
                timestamp=1_783_772_735,
                source="wechat+localdb",
                sequence=None,
            ),
        ),
        cursor="10",
    )
    client.users = WeChatUserList(
        users=(WeChatUser(id="wxid_alice", display_name="Alice Global"),),
        cursor="10",
    )
    client.group_members[CHAT_ID] = WeChatGroupMemberList(
        group_id=CHAT_ID,
        members=(
            WeChatGroupMember(
                group_id=CHAT_ID,
                user_id="wxid_alice",
                display_name="Alice Global",
                nickname="Old Group Alias",
            ),
            WeChatGroupMember(
                group_id=CHAT_ID,
                user_id="wxid_bob",
                display_name="Bob Member",
                nickname="Old Omitted Alias",
            ),
        ),
        cursor="10",
        snapshot_complete=True,
        snapshot_current=True,
        snapshot_connection_generation=41,
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        client.group_members[CHAT_ID] = WeChatGroupMemberList(
            group_id=CHAT_ID,
            members=(
                WeChatGroupMember(
                    group_id=CHAT_ID,
                    user_id="wxid_alice",
                    display_name="Alice Global",
                    nickname=None,
                ),
            ),
            cursor="11",
            snapshot_complete=False,
            snapshot_current=False,
            snapshot_connection_generation=None,
        )

        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(RecordingHandler(), asyncio.Event())
        alice_message = await store.get_message(
            CONNECTOR_KEY,
            CHAT_ID,
            "4159667620982040828",
        )
        bob_message = await store.get_message(
            CONNECTOR_KEY,
            CHAT_ID,
            "4159667620982040829",
        )

        assert result == "reconnect"
        assert alice_message is not None
        assert bob_message is not None
        assert alice_message.sender_display_name == "Alice Global"
        assert bob_message.sender_display_name == "Bob Member"
        assert client.group_member_reads == [CHAT_ID, CHAT_ID]
        assert await store.get_cursor(CONNECTOR_KEY) == "11"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_refreshes_once_at_member_delta_end(tmp_path) -> None:
    events = tuple(
        WeChatEvent.parse(
            {
                "schemaVersion": "wechat-bridge/v1alpha1",
                "cursor": str(11 + index),
                "event": "group_member",
                "groupId": CHAT_ID,
                "userId": f"wxid_member_{index}",
                "connectionGeneration": 41,
                "raw": {
                    "mode": "delta",
                    "deltaId": "delta-1",
                    "deltaIndex": str(index),
                    "responseCount": "3",
                },
            }
        )
        for index in range(3)
    )
    client = FakeConnectorClient(events)
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)

        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(RecordingHandler(), asyncio.Event())

        assert result == "reconnect"
        assert client.group_member_reads == [CHAT_ID, CHAT_ID]
        assert await store.get_cursor(CONNECTOR_KEY) == "13"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_does_not_ack_failed_profile_refresh(tmp_path) -> None:
    class FailingProfileClient(FakeConnectorClient):
        async def get_user(self, user_id):
            self.user_detail_reads.append(user_id)
            raise ConnectionError("profile refresh failed")

    profile_event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "11",
            "event": "user_profile",
            "status": "changed",
            "id": "sha256:profile-1",
            "userId": "wxid_alice",
            "connectionGeneration": 41,
        }
    )
    client = FailingProfileClient((profile_event,))
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        with pytest.raises(ConnectionError, match="profile refresh failed"):
            await WeChatEventPump(
                client,
                store,
                CONNECTOR_KEY,
                bootstrap,
            ).run(RecordingHandler(), asyncio.Event())

        assert client.user_detail_reads == ["wxid_alice"]
        assert await store.get_cursor(CONNECTOR_KEY) == "10"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_handles_each_message_once_and_then_acks(tmp_path):
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = FakeConnectorClient((message_event(cursor="11"),))
    handler = RecordingHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(handler, asyncio.Event())

        assert result == "reconnect"
        assert client.after_values == ["10"]
        assert [message.id for message in handler.messages] == ["4159667620982040828"]
        assert await store.get_cursor(CONNECTOR_KEY) == "11"

        replay_client = FakeConnectorClient(
            (message_event(cursor="12", reply_to="3159667620982040828"),)
        )
        replay_result = await WeChatEventPump(
            replay_client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(handler, asyncio.Event())

        assert replay_result == "reconnect"
        assert len(handler.messages) == 1
        assert await store.get_cursor(CONNECTOR_KEY) == "12"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_unknown_send_does_not_block_event_ingress(tmp_path) -> None:
    events = tuple(
        message_event(
            cursor=str(cursor),
            message_id=f"41596676209820408{cursor}",
            content=f"/ai outgoing {cursor}",
        )
        for cursor in range(11, 19)
    ) + (
        message_event(
            cursor="19",
            message_id="4159667620982040819",
            content="/ai incoming",
            sender_id="wxid_alice",
            direction="in",
        ),
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = FakeConnectorClient(events)
    origins: list[MessageOrigin] = []
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        await store.reserve_generated_send(
            CONNECTOR_KEY,
            ACCOUNT_ID,
            CHAT_ID,
            "request-with-unknown-outcome",
            message_fingerprint(
                text="generated",
                reply_to_message_id=None,
                has_attachment=False,
            ).digest,
        )
        transport = WeChatChatTransport(
            client,  # type: ignore[arg-type]
            store,
            CONNECTOR_KEY,
            native_reply_ready=False,
        )

        class ClassifyingHandler:
            async def handle(self, message):
                origins.append(await transport.classify_origin(message))
                return True

        result = await asyncio.wait_for(
            WeChatEventPump(
                client,
                store,
                CONNECTOR_KEY,
                bootstrap,
            ).run(ClassifyingHandler(), asyncio.Event()),
            timeout=0.5,
        )

        assert result == "reconnect"
        assert origins == [MessageOrigin.INDETERMINATE] * 8 + [
            MessageOrigin.INCOMING
        ]
        assert await store.get_cursor(CONNECTOR_KEY) == "19"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_dispatches_textual_quoted_app_messages(
    tmp_path,
) -> None:
    client = FakeConnectorClient(
        (
            message_event(
                cursor="11",
                content="/ai explain this",
                reply_to="3159667620982040828",
                message_type="app",
            ),
            message_event(
                cursor="12",
                message_id="4159667620982040829",
                content="continue the answer",
                reply_to="3159667620982040829",
                message_type="app",
            ),
            message_event(
                cursor="13",
                message_id="4159667620982040830",
                content="/ai not a quoted reply",
                message_type="app",
            ),
        )
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    handler = RecordingHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(handler, asyncio.Event())

        assert result == "reconnect"
        assert [message.id for message in handler.messages] == [
            "4159667620982040828",
            "4159667620982040829",
        ]
        assert [message.reply_to_msg_id for message in handler.messages] == [
            "3159667620982040828",
            "3159667620982040829",
        ]
        assert await store.get_cursor(CONNECTOR_KEY) == "13"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_dispatches_shared_chat_history(tmp_path) -> None:
    client = FakeConnectorClient(
        (
            message_event(
                cursor="11",
                content="Team history",
                message_type="chat_history",
                shared_chat_history={
                    "title": "Team history",
                    "itemCount": 1,
                    "items": [
                        {"kind": "text", "senderName": "Alice", "content": "Hi"},
                    ],
                },
            ),
        )
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    handler = RecordingHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(handler, asyncio.Event())

        assert result == "reconnect"
        assert len(handler.messages) == 1
        assert handler.messages[0].message_type == "chat_history"
        assert handler.messages[0].raw_text == (
            "[Forwarded chat history]\nTeam history\nAlice: Hi"
        )
        assert await store.get_cursor(CONNECTOR_KEY) == "11"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_ignores_unsupported_messages_without_chat_refresh(
    tmp_path,
) -> None:
    client = FakeConnectorClient(
        (
            message_event(
                cursor="11",
                content="redacted",
                content_redacted=True,
            ),
            message_event(
                cursor="12",
                message_id="4159667620982040829",
                content="image metadata",
                message_type="image",
            ),
        )
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    handler = RecordingHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(handler, asyncio.Event())

        assert result == "reconnect"
        assert handler.messages == []
        assert client.chat_reads == 1
        assert await store.get_cursor(CONNECTOR_KEY) == "12"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_acks_senderless_unsupported_message(tmp_path) -> None:
    client = FakeConnectorClient(
        (
            message_event(
                cursor="11",
                message_type="app",
                sender_id=None,
            ),
        )
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    handler = RecordingHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(handler, asyncio.Event())

        assert result == "reconnect"
        assert handler.messages == []
        assert client.chat_reads == 1
        assert await store.get_cursor(CONNECTOR_KEY) == "11"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_does_not_ack_failed_handler(tmp_path):
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = FakeConnectorClient((message_event(cursor="11"),))
    handler = RecordingHandler(RuntimeError("handler failed"))
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        with pytest.raises(RuntimeError, match="handler failed"):
            await WeChatEventPump(
                client,
                store,
                CONNECTOR_KEY,
                bootstrap,
            ).run(handler, asyncio.Event())

        assert await store.get_cursor(CONNECTOR_KEY) == "10"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_acks_disconnect_and_requests_rebootstrap(tmp_path):
    disconnect = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "11",
            "event": "hook_connection",
            "status": "disconnected",
            "connectionGeneration": 41,
        }
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = FakeConnectorClient((disconnect,))
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(RecordingHandler(), asyncio.Event())

        assert result == "rebootstrap"
        assert await store.get_cursor(CONNECTOR_KEY) == "11"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_replays_new_generation_event_after_rebootstrap(
    tmp_path,
):
    event = message_event(cursor="11", connection_generation=42)
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = FakeConnectorClient((event,))
    handler = RecordingHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)

        result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(handler, asyncio.Event())

        assert result == "rebootstrap"
        assert await store.get_cursor(CONNECTOR_KEY) == "10"
        assert handler.messages == []

        client.session = WeChatSession(
            status="logged_in",
            self_id=ACCOUNT_ID,
            display_name="Sidekick",
            hook_connected=True,
            connection_generation=42,
            content_redacted=False,
            cursor="11",
        )
        client.capabilities = WeChatCapabilities(
            receive_text=True,
            receive_shared_chat_history=True,
            stable_inbound_message_ids=True,
            send_text=True,
            send_reply=True,
            send_native_reply=True,
            request_idempotency=True,
            outbound_stable_message_id=True,
            websocket=True,
            cursor=True,
            replay=True,
            durable_cursor=True,
            text_send_ready=True,
            reply_send_ready=True,
            connection_generation=42,
            history=False,
            inbound_image_download=True,
            request_original_image=True,
        )
        client.chats = WeChatChatList(
            chats=client.chats.chats,
            snapshot=WeChatChatSnapshot(
                id="snapshot-42",
                complete=True,
                current=True,
                count=1,
                cursor="11",
                connection_generation=42,
            ),
            cursor="11",
        )
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)

        replay_result = await WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap,
        ).run(handler, asyncio.Event())

        assert replay_result == "reconnect"
        assert [message.id for message in handler.messages] == [event.payload["id"]]
        assert await store.get_cursor(CONNECTOR_KEY) == "11"
        assert client.after_values == ["10", "10"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_stops_without_waiting_for_an_event(tmp_path):
    class BlockingClient(FakeConnectorClient):
        async def events(self, *, after):
            self.after_values.append(after)
            await asyncio.Event().wait()
            yield  # pragma: no cover

    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = BlockingClient()
    stop = asyncio.Event()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        task = asyncio.create_task(
            WeChatEventPump(
                client,
                store,
                CONNECTOR_KEY,
                bootstrap,
            ).run(RecordingHandler(), stop)
        )
        await asyncio.sleep(0)
        stop.set()

        assert await asyncio.wait_for(task, timeout=1) == "stopped"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_reads_cancel_while_ai_request_is_running(tmp_path):
    class CancellableHandler:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.messages = []

        async def handle(self, message):
            self.messages.append(message.raw_text)
            if message.raw_text == "/ai long task":
                self.started.set()
                await self.release.wait()
            elif message.raw_text == "/ai_cancel":
                await self.started.wait()
                self.release.set()
            return True

    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = FakeConnectorClient(
        (
            message_event(cursor="11", content="/ai long task"),
            message_event(
                cursor="12",
                message_id="4159667620982040829",
                content="/ai_cancel",
            ),
        )
    )
    handler = CancellableHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        result = await asyncio.wait_for(
            WeChatEventPump(
                client,
                store,
                CONNECTOR_KEY,
                bootstrap,
                handler_concurrency=2,
            ).run(handler, asyncio.Event()),
            timeout=1,
        )

        assert result == "reconnect"
        assert handler.messages == ["/ai long task", "/ai_cancel"]
        assert await store.get_cursor(CONNECTOR_KEY) == "12"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_commits_concurrent_handlers_in_cursor_order(tmp_path):
    class OrderedHandler:
        def __init__(self):
            self.first_release = asyncio.Event()
            self.second_done = asyncio.Event()

        async def handle(self, message):
            if message.raw_text == "/ai first":
                await self.first_release.wait()
            else:
                self.second_done.set()
            return True

    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = FakeConnectorClient(
        (
            message_event(cursor="11", content="/ai first"),
            message_event(
                cursor="12",
                message_id="4159667620982040829",
                content="/ai second",
            ),
        )
    )
    handler = OrderedHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        task = asyncio.create_task(
            WeChatEventPump(
                client,
                store,
                CONNECTOR_KEY,
                bootstrap,
                handler_concurrency=2,
            ).run(handler, asyncio.Event())
        )
        await asyncio.wait_for(handler.second_done.wait(), timeout=1)

        assert await store.get_cursor(CONNECTOR_KEY) == "10"

        handler.first_release.set()
        assert await asyncio.wait_for(task, timeout=1) == "reconnect"
        assert await store.get_cursor(CONNECTOR_KEY) == "12"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_does_not_repeat_later_completed_handler(
    tmp_path,
) -> None:
    class FailFirstOnceHandler:
        def __init__(self):
            self.calls: list[str] = []
            self.second_done = asyncio.Event()
            self.failed = False

        async def handle(self, message):
            self.calls.append(message.raw_text)
            if message.raw_text == "/ai first" and not self.failed:
                await self.second_done.wait()
                self.failed = True
                raise RuntimeError("first handler failed")
            if message.raw_text == "/ai second":
                self.second_done.set()
            return True

    events = (
        message_event(cursor="11", content="/ai first"),
        message_event(
            cursor="12",
            message_id="4159667620982040829",
            content="/ai second",
        ),
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    handler = FailFirstOnceHandler()
    first_client = FakeConnectorClient(events)
    try:
        bootstrap = await bootstrap_wechat_channel(
            first_client,
            store,
            CONNECTOR_KEY,
        )
        with pytest.raises(RuntimeError, match="first handler failed"):
            await WeChatEventPump(
                first_client,
                store,
                CONNECTOR_KEY,
                bootstrap,
                handler_concurrency=2,
            ).run(handler, asyncio.Event())

        assert await store.get_cursor(CONNECTOR_KEY) == "10"

        replay_client = FakeConnectorClient(events)
        assert (
            await WeChatEventPump(
                replay_client,
                store,
                CONNECTOR_KEY,
                bootstrap,
                handler_concurrency=2,
            ).run(handler, asyncio.Event())
            == "reconnect"
        )

        assert handler.calls == ["/ai first", "/ai second", "/ai first"]
        assert await store.get_cursor(CONNECTOR_KEY) == "12"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_event_pump_stops_dispatch_at_rebootstrap_boundary(tmp_path):
    class BlockingHandler:
        def __init__(self):
            self.started = asyncio.Event()
            self.later_started = asyncio.Event()
            self.release = asyncio.Event()
            self.messages = []

        async def handle(self, message):
            self.messages.append(message.raw_text)
            if message.raw_text == "/ai long task":
                self.started.set()
                await self.release.wait()
            else:
                self.later_started.set()
            return True

    session_update = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "12",
            "event": "session_update",
            "connectionGeneration": 41,
        }
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    client = FakeConnectorClient(
        (
            message_event(cursor="11", content="/ai long task"),
            session_update,
            message_event(
                cursor="13",
                message_id="4159667620982040829",
                content="/ai must wait for the new session",
            ),
        )
    )
    handler = BlockingHandler()
    try:
        bootstrap = await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
        task = asyncio.create_task(
            WeChatEventPump(
                client,
                store,
                CONNECTOR_KEY,
                bootstrap,
                handler_concurrency=3,
            ).run(handler, asyncio.Event())
        )
        await asyncio.wait_for(handler.started.wait(), timeout=1)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(handler.later_started.wait(), timeout=0.05)

        assert handler.messages == ["/ai long task"]

        handler.release.set()
        assert await asyncio.wait_for(task, timeout=1) == "rebootstrap"
        assert await store.get_cursor(CONNECTOR_KEY) == "12"
    finally:
        await store.close()
