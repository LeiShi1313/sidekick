from __future__ import annotations

import asyncio
from dataclasses import replace
import logging
from types import SimpleNamespace

import pytest

from sidekick.ai import AISettings, AIStateRepository
from sidekick.chat.output_policy import MAINLAND_MESSAGING_POLICY_ID
from sidekick.channel_status import ChannelOpsSettings
from sidekick.plugins.base import command_registry
from sidekick.plugins.wechat_ai import WeChatAI, WeChatRuntimeSettings
from sidekick.wechat.ai import WeChatQuotedImageDescriber
from sidekick.wechat.api import (
    WeChatAPIContractError,
    WeChatCapabilities,
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatGroupMemberList,
    WeChatSession,
    WeChatUserList,
)
from sidekick.wechat.service import bootstrap_wechat_channel
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
    def __init__(self) -> None:
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
            fetch_observed_messages=True,
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
        self.users = WeChatUserList(users=(), cursor="10")
        self.group_members = WeChatGroupMemberList(
            group_id=CHAT_ID,
            members=(),
            cursor="10",
            snapshot_complete=False,
            snapshot_current=False,
            snapshot_connection_generation=None,
        )
        self.legacy_message_reads = 0

    async def get_session(self):
        return self.session

    async def get_capabilities(self):
        return self.capabilities

    async def get_chats(self):
        return self.chats

    async def get_messages(self, *, limit):
        self.legacy_message_reads += 1
        raise AssertionError(f"legacy history was requested with limit={limit}")

    async def get_users(self):
        return self.users

    async def get_user(self, _user_id):
        return None

    async def get_group_members(self, _group_id):
        return self.group_members

    async def events(self, *, after):
        if False:
            yield after


@pytest.mark.asyncio
async def test_wechat_bootstrap_uses_session_cursor_without_legacy_history(
    tmp_path,
) -> None:
    client = FakeConnectorClient()
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        bootstrap = await bootstrap_wechat_channel(
            client,
            store,
            CONNECTOR_KEY,
        )

        assert bootstrap.session.cursor == "10"
        assert await store.get_cursor(CONNECTOR_KEY) == "10"
        assert await store.count_messages(CONNECTOR_KEY) == 0
        assert client.legacy_message_reads == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_bootstrap_requires_observed_message_capability(
    tmp_path,
) -> None:
    client = FakeConnectorClient()
    client.capabilities = replace(
        client.capabilities,
        fetch_observed_messages=False,
    )
    store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        with pytest.raises(
            WeChatAPIContractError,
            match="fetchObservedMessages",
        ):
            await bootstrap_wechat_channel(client, store, CONNECTOR_KEY)
    finally:
        await store.close()
