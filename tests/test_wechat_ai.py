from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from sidekick.ai import (
    AIConversationHandler,
    AIResponder,
    AIStateRepository,
    AgentEvent,
    AgentRunRequest,
    PromptBuilder,
)
from sidekick.ai_continuous_memory import ContinuousMemoryScheduler
from sidekick.ai_memory_ingestion import ContinuousMemoryResult
from sidekick.wechat.ai import (
    WECHAT_IDENTITY_CODEC,
    WeChatChatTransport,
    WeChatHistorySource,
    WeChatMessageIdentityResolver,
    WeChatMessageMentionResolver,
)
from sidekick.wechat.api import (
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatConnectorMessage,
    WeChatEvent,
    WeChatMessageList,
    WeChatSendOperation,
    WeChatSendOutcomeUnknown,
    WeChatSession,
)
from sidekick.wechat.store import WeChatStateRepository


CONNECTOR_KEY = "http://127.0.0.1:18188"
ACCOUNT_ID = "wxid_self"
GROUP_ID = "56825427596@chatroom"


class RecordingConnectorClient:
    def __init__(self, responses: tuple[object, ...]):
        self.responses = list(responses)
        self.calls: list[dict[str, str]] = []

    async def send_text_and_wait(self, *, request_id, to, content):
        self.calls.append(
            {"request_id": request_id, "to": to, "content": content}
        )
        if not self.responses:
            raise AssertionError("No WeChat send response prepared")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FinalGateway:
    def __init__(self, answer: str = "final answer"):
        self.answer = answer
        self.requests: list[AgentRunRequest] = []

    async def run(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        self.requests.append(request)
        yield AgentEvent(
            type="run_started",
            run_id=request.run_id,
            session_id="session-1",
        )
        yield AgentEvent(type="text_delta", delta="partial", reset=True)
        yield AgentEvent(
            type="run_completed",
            session_id="session-1",
            entry_id="entry-1",
            answer=self.answer,
        )

    async def cancel(self, _run_id: str) -> bool:
        return True


def submitted(
    request_id: str = "placeholder",
    message_id: str = "7158246912028861544",
) -> WeChatSendOperation:
    return WeChatSendOperation(
        request_id=request_id,
        status="submitted",
        message_id=message_id,
        error_code=None,
        to=GROUP_ID,
    )


async def bootstrap_store(path, *, trigger_text="/ai hello", direction="out"):
    store = await WeChatStateRepository(path).connect()
    trigger = WeChatConnectorMessage(
        id="4159667620982040828",
        chat_id=GROUP_ID,
        direction=direction,
        message_type="text",
        sender_id=ACCOUNT_ID if direction == "out" else "wxid_alice",
        reply_to_message_id=None,
        content=trigger_text,
        content_redacted=False,
        timestamp=1_783_772_734,
        source="wechat+localdb",
        sequence=None,
    )
    await store.bootstrap(
        connector_key=CONNECTOR_KEY,
        session=WeChatSession(
            status="logged_in",
            self_id=ACCOUNT_ID,
            display_name="Sidekick",
            hook_connected=True,
            connection_generation=41,
            content_redacted=False,
            cursor="10",
        ),
        chats=WeChatChatList(
            chats=(
                WeChatChat(
                    id=GROUP_ID,
                    type="group",
                    display_name="Example group",
                ),
            ),
            snapshot=WeChatChatSnapshot(
                id="snapshot-41",
                complete=True,
                current=True,
                count=1,
                cursor="10",
                connection_generation=41,
            ),
            cursor="10",
        ),
        messages=WeChatMessageList(messages=(trigger,), cursor="10"),
    )
    observed = await store.get_message(CONNECTOR_KEY, GROUP_ID, trigger.id)
    assert observed is not None
    return store, observed


@pytest.mark.asyncio
async def test_wechat_responder_defers_placeholder_and_sends_one_bounded_final(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    answer = "你" * 2_000
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(client, store, CONNECTOR_KEY)
    responder = AIResponder(
        FinalGateway(answer),
        initial_status=None,
        max_output_chars=10_000,
        transport=transport,
    )
    request = AgentRunRequest(
        run_id="run-1",
        session_id=None,
        parent_entry_id=None,
        prompt="hello",
        context=(),
        system_prompt="system",
        tool_policy="owner",
    )
    try:
        result = await responder.answer(trigger, request)
    finally:
        await store.close()

    assert result.succeeded is True
    assert result.message.id == "7158246912028861544"
    assert len(client.calls) == 1
    assert client.calls[0]["content"].endswith("...")
    assert len(client.calls[0]["content"].encode("utf-8")) <= 4_095
    assert result.text == answer


@pytest.mark.asyncio
async def test_wechat_transport_uses_stable_request_id_for_same_trigger(tmp_path) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    client = RecordingConnectorClient(
        (
            submitted(message_id="7158246912028861544"),
            submitted(message_id="7158246912028861544"),
        )
    )
    transport = WeChatChatTransport(client, store, CONNECTOR_KEY)
    try:
        first = await transport.reply(trigger, "Access is open.", presentation="plain")
        second = await transport.reply(trigger, "Access is open.", presentation="plain")
    finally:
        await store.close()

    assert first.id == second.id == "7158246912028861544"
    assert client.calls[0]["request_id"] == client.calls[1]["request_id"]
    assert client.calls[0]["request_id"].startswith("sidekick.wechat.reply.")


@pytest.mark.asyncio
async def test_wechat_responder_does_not_resubmit_unknown_outcome(tmp_path) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    operation = WeChatSendOperation(
        request_id="placeholder",
        status="unknown",
        message_id=None,
        error_code="SEND_OUTCOME_UNKNOWN",
        to=GROUP_ID,
    )
    client = RecordingConnectorClient(
        (WeChatSendOutcomeUnknown(operation, "outcome unknown"),)
    )
    responder = AIResponder(
        FinalGateway(),
        initial_status=None,
        transport=WeChatChatTransport(client, store, CONNECTOR_KEY),
    )
    request = AgentRunRequest(
        run_id="run-1",
        session_id=None,
        parent_entry_id=None,
        prompt="hello",
        context=(),
        system_prompt="system",
        tool_policy="owner",
    )
    try:
        result = await responder.answer(trigger, request)
    finally:
        await store.close()

    assert result.succeeded is False
    assert result.text == "AI request failed. Try again later."
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_wechat_responder_does_not_change_payload_after_transport_error(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    client = RecordingConnectorClient((ConnectionError("connection lost"),))
    responder = AIResponder(
        FinalGateway(),
        initial_status=None,
        transport=WeChatChatTransport(client, store, CONNECTOR_KEY),
    )
    request = AgentRunRequest(
        run_id="run-1",
        session_id=None,
        parent_entry_id=None,
        prompt="hello",
        context=(),
        system_prompt="system",
        tool_policy="owner",
    )
    try:
        result = await responder.answer(trigger, request)
    finally:
        await store.close()

    assert result.succeeded is False
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_wechat_conversation_handler_runs_ai_and_persists_opaque_answer_id(
    tmp_path,
) -> None:
    wechat_store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(client, wechat_store, CONNECTOR_KEY)
    history = WeChatHistorySource(wechat_store, CONNECTOR_KEY)
    gateway = FinalGateway("hello from Sidekick")
    prompt_builder = PromptBuilder(
        transport=transport,
        history_source=history,
        identity_resolver=WeChatMessageIdentityResolver(),
        mention_resolver=WeChatMessageMentionResolver(),
        identity_codec=WECHAT_IDENTITY_CODEC,
    )
    handler = AIConversationHandler(
        owner_id=ACCOUNT_ID,
        responder=AIResponder(
            gateway,
            initial_status=None,
            transport=transport,
        ),
        store=ai_store,
        prompt_builder=prompt_builder,
        transport=transport,
        identity_codec=WECHAT_IDENTITY_CODEC,
    )
    try:
        handled = await handler.handle(trigger)
        marker = await ai_store.get_answer(
            WECHAT_IDENTITY_CODEC.scope_id(GROUP_ID),
            "7158246912028861544",
        )
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert handled is True
    assert marker is not None
    assert marker.trigger_message_id == "4159667620982040828"
    assert marker.answer_message_id == "7158246912028861544"
    assert gateway.requests[0].prompt == "hello"
    assert client.calls[0]["content"] == "hello from Sidekick"

    outbound_echo = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "11",
            "event": "message",
            "id": "7158246912028861544",
            "chatId": GROUP_ID,
            "direction": "out",
            "messageType": "text",
            "senderId": ACCOUNT_ID,
            "content": "/ai this must not trigger",
            "timestamp": 1_783_772_735,
            "connectionGeneration": 41,
        }
    )
    replay_store = await WeChatStateRepository(tmp_path / "wechat.db").connect()
    try:
        echoed_message = await replay_store.project_event(
            CONNECTOR_KEY,
            outbound_echo,
        )
        assert echoed_message is not None
        assert await replay_store.is_processed(echoed_message) is True
    finally:
        await replay_store.close()


@pytest.mark.asyncio
async def test_continuous_memory_scheduler_accepts_wechat_scope_ids(tmp_path) -> None:
    class RecordingScanner:
        def __init__(self):
            self.calls: list[str] = []

        async def run_continuous_scope(self, chat_id):
            self.calls.append(chat_id)
            return ContinuousMemoryResult(
                messages_seen=1,
                messages_retained=1,
                documents_created=1,
                documents_unchanged=0,
                caught_up=True,
            )

    store = await AIStateRepository(tmp_path / "ai.db").connect()
    scanner = RecordingScanner()
    await store.set_continuous_memory_enabled(
        WECHAT_IDENTITY_CODEC.scope_id(GROUP_ID),
        True,
        cursor_message_id="4159667620982040828",
    )
    try:
        result = await ContinuousMemoryScheduler(
            runner=scanner,
            store=store,
            identity_codec=WECHAT_IDENTITY_CODEC,
        ).run_once()
    finally:
        await store.close()

    assert scanner.calls == [GROUP_ID]
    assert result.scopes_seen == 1
    assert result.scopes_succeeded == 1
