from __future__ import annotations

import asyncio

import pytest

from sidekick.wechat.api import (
    WeChatCapabilities,
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatEvent,
    WeChatMessageList,
    WeChatSession,
)
from sidekick.plugins.base import command_registry
from sidekick.plugins.wechat_ai import WeChatRuntimeSettings
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
            stable_inbound_message_ids=True,
            send_text=True,
            request_idempotency=True,
            outbound_stable_message_id=True,
            websocket=True,
            cursor=True,
            replay=True,
            durable_cursor=True,
            text_send_ready=True,
            connection_generation=41,
            history=False,
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
        self.event_values = list(events)
        self.after_values: list[str] = []

    async def get_session(self):
        return self.session

    async def get_capabilities(self):
        return self.capabilities

    async def get_chats(self):
        return self.chats

    async def get_messages(self, *, limit):
        assert limit == 1_000
        return self.messages

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


def message_event(*, cursor: str, reply_to: str | None = None) -> WeChatEvent:
    payload = {
        "schemaVersion": "wechat-bridge/v1alpha1",
        "cursor": cursor,
        "event": "message",
        "id": "4159667620982040828",
        "chatId": CHAT_ID,
        "direction": "out",
        "messageType": "text",
        "senderId": ACCOUNT_ID,
        "content": "/ai hello",
        "timestamp": 1_783_772_734,
        "connectionGeneration": 41,
    }
    if reply_to is not None:
        payload["replyToMessageId"] = reply_to
    return WeChatEvent.parse(payload)


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
        assert [message.id for message in handler.messages] == [
            "4159667620982040828"
        ]
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
