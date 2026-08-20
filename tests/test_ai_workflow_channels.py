from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from sidekick.ai import (
    AIConversationHandler,
    AIWorkflowCancellation,
    PromptBuilder,
)
from sidekick.ai_workflow import AIWorkflow
from sidekick.chat.provenance import MessageOrigin
from sidekick.inbound import InboundSourceRevision
from sidekick.inbound_store import SQLiteInboundWorkStore
from sidekick.onebot.ai import OneBotDirectory
from sidekick.plugins.ai import TelegramAI
from sidekick.plugins.onebot_ai import OneBotAI
from sidekick.wechat.api import WeChatAPIError, WeChatEvent, WeChatObservedMessage
from sidekick.wechat.service import WeChatEventPump, WeChatObservedMessageSource


class RecordingWorkflow:
    def __init__(self) -> None:
        self.accepted: list[dict[str, object]] = []

    async def accept(self, **reference: object) -> None:
        self.accepted.append(reference)


class MutableOriginTransport:
    def __init__(self, origin: MessageOrigin) -> None:
        self.origin = origin

    async def classify_origin(self, _message: object) -> MessageOrigin:
        return self.origin


class ObservableInboundStore(SQLiteInboundWorkStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.promoted: asyncio.Queue[int | str] = asyncio.Queue()

    async def promote_pending_ai_generation(self, work, **options):
        result = await super().promote_pending_ai_generation(work, **options)
        await self.promoted.put(work.message_id)
        return result


class ReferenceSource:
    def __init__(self, messages: dict[int, SimpleNamespace]) -> None:
        self.messages = messages

    async def fetch(self, work) -> InboundSourceRevision[SimpleNamespace]:
        message = self.messages[work.message_id]
        return InboundSourceRevision(
            version=f"message:{message.id}:v1",
            state="present",
            payload=message,
            attested_origin=work.attested_origin,
        )

    async def materialize(self, message: SimpleNamespace) -> SimpleNamespace:
        return message


class PrefixStore:
    async def get_ai_command_prefix(self, _scope_id: str) -> None:
        return None


class LaneHandler:
    def __init__(self, classifier: AIConversationHandler) -> None:
        self.classifier = classifier
        self.control = None
        self.first_generation_started = asyncio.Event()
        self.first_generation_release = asyncio.Event()
        self.first_generation_finished = asyncio.Event()
        self.control_finished = asyncio.Event()
        self.generation_messages: list[int] = []
        self.cancellation = AIWorkflowCancellation()
        self.workflow_notices: list[tuple[int, str]] = []

    def bind_workflow_control(self, control) -> None:
        self.control = control

    def unbind_workflow_control(self, control) -> None:
        if self.control is control:
            self.control = None

    async def classify(self, message, *, attested_origin=None):
        return await self.classifier.classify(
            message,
            attested_origin=attested_origin,
        )

    async def generation_eligible_at(self, _classification) -> float:
        return 0

    async def handle(
        self,
        message,
        *,
        attested_origin=None,
        workflow_admitted: bool = False,
    ) -> bool:
        assert attested_origin is MessageOrigin.INCOMING
        if message.raw_text == "/ai_cancel":
            assert not workflow_admitted
            assert self.control is not None
            self.cancellation = await self.control.cancel_generations(
                "chat:actor:1",
                interrupt_running=False,
            )
            self.control_finished.set()
            return True

        assert workflow_admitted
        self.generation_messages.append(message.id)
        if message.id == 1:
            self.first_generation_started.set()
            await self.first_generation_release.wait()
            self.first_generation_finished.set()
        return True

    async def reply_workflow_notice(self, message, notice: str) -> None:
        self.workflow_notices.append((message.id, notice))


def workflow_message(message_id: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        chat_id=10,
        sender_id=1,
        raw_text=text,
        reply_to_msg_id=None,
    )


@pytest.mark.asyncio
async def test_control_lane_bypasses_blocked_generation_and_cancels_fifo_tail(
    tmp_path,
) -> None:
    messages = {
        1: workflow_message(1, "/ai first"),
        2: workflow_message(2, "/ai second"),
        3: workflow_message(3, "/ai_cancel"),
    }
    store = await ObservableInboundStore(tmp_path / "ai.db").connect()
    await store.initialize_source("channel-test", epoch="account-1", initial_cursor=0)
    classifier = AIConversationHandler(
        owner_id=1,
        responder=object(),
        store=PrefixStore(),
        prompt_builder=PromptBuilder(),
        transport=object(),
    )
    handler = LaneHandler(classifier)
    workflow = AIWorkflow(
        ReferenceSource(messages),
        store,
        "channel-test",
        handler,
        generation_concurrency=1,
    )
    workflow.start()
    try:
        await workflow.accept(
            cursor=1,
            chat_id=10,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        await asyncio.wait_for(handler.first_generation_started.wait(), timeout=1)

        await workflow.accept(
            cursor=2,
            chat_id=10,
            message_id=2,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        while await asyncio.wait_for(store.promoted.get(), timeout=1) != 2:
            pass

        await workflow.accept(
            cursor=3,
            chat_id=10,
            message_id=3,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        await asyncio.wait_for(handler.control_finished.wait(), timeout=1)

        assert handler.cancellation == AIWorkflowCancellation(queued=1, running=1)
        assert handler.workflow_notices == []
        assert handler.generation_messages == [1]

        handler.first_generation_release.set()
        await asyncio.wait_for(handler.first_generation_finished.wait(), timeout=1)
        await asyncio.sleep(0)
        assert handler.generation_messages == [1]
    finally:
        handler.first_generation_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_telegram_preserves_candidate_filter_and_attested_origin() -> None:
    workflow = RecordingWorkflow()
    transport = MutableOriginTransport(MessageOrigin.INCOMING)
    plugin = object.__new__(TelegramAI)
    plugin._handler = object()
    plugin._transport = transport
    plugin._ai_workflow = workflow
    plugin._owner_id = 999
    plugin.logger = logging.getLogger("test-telegram-ai-workflow")

    ambient = SimpleNamespace(
        id=40,
        chat_id=-1001,
        peer_id=None,
        raw_text="ambient chat",
        reply_to_msg_id=None,
    )
    await plugin._on_message(SimpleNamespace(message=ambient))

    candidate = SimpleNamespace(
        id=41,
        chat_id=-1001,
        peer_id=None,
        raw_text="!ai_access open",
        reply_to_msg_id=None,
    )
    transport.origin = MessageOrigin.SIDEKICK_GENERATED
    await plugin._on_message(SimpleNamespace(message=candidate))

    transport.origin = MessageOrigin.MANUAL_OUTGOING
    await plugin._on_message(SimpleNamespace(message=candidate))

    assert workflow.accepted == [
        {
            "cursor": 41,
            "chat_id": -1001,
            "message_id": 41,
            "kind": "message",
            "attested_origin": MessageOrigin.MANUAL_OUTGOING,
        }
    ]


def onebot_group_event(
    *,
    message_id: int,
    text: str,
    post_type: str = "message",
    sender_id: int = 42,
) -> dict[str, object]:
    return {
        "post_type": post_type,
        "message_type": "group",
        "self_id": 99,
        "user_id": sender_id,
        "group_id": 700,
        "message_id": message_id,
        "time": 1_700_000_000,
        "sender": {
            "user_id": sender_id,
            "nickname": "Alice",
            "card": "Alice Card",
            "role": "member",
        },
        "message": [{"type": "text", "data": {"text": text}}],
        "raw_message": text,
    }


@pytest.mark.asyncio
async def test_qq_preserves_candidate_filter_and_generated_echo_guard() -> None:
    workflow = RecordingWorkflow()
    transport = MutableOriginTransport(MessageOrigin.INCOMING)
    plugin = object.__new__(OneBotAI)
    plugin._runtime = SimpleNamespace(self_id=99)
    plugin._bridge = SimpleNamespace()
    plugin._directory = OneBotDirectory()
    plugin._handler = object()
    plugin._transport = transport
    plugin._ai_workflow = workflow
    plugin.logger = logging.getLogger("test-qq-ai-workflow")

    await plugin._on_event(onebot_group_event(message_id=100, text="ambient chat"))

    transport.origin = MessageOrigin.SIDEKICK_GENERATED
    await plugin._on_event(
        onebot_group_event(
            message_id=101,
            text="/ai hello",
            post_type="message_sent",
            sender_id=99,
        )
    )

    transport.origin = MessageOrigin.MANUAL_OUTGOING
    await plugin._on_event(
        onebot_group_event(
            message_id=102,
            text="$ask hello",
            post_type="message_sent",
            sender_id=99,
        )
    )

    assert workflow.accepted == [
        {
            "cursor": 102,
            "chat_id": 700,
            "message_id": 102,
            "kind": "message",
            "attested_origin": MessageOrigin.MANUAL_OUTGOING,
        }
    ]


def wechat_message_event(
    *,
    cursor: str,
    message_id: str,
    content: str,
    reply_to_message_id: str | None = None,
) -> WeChatEvent:
    payload = {
        "schemaVersion": "wechat-bridge/v1alpha1",
        "cursor": cursor,
        "event": "message",
        "connectionGeneration": 41,
        "id": message_id,
        "chatId": "56825427596@chatroom",
        "direction": "in",
        "messageType": "text",
        "senderId": "wxid_alice",
        "content": content,
        "timestamp": 1_700_000_010,
    }
    if reply_to_message_id is not None:
        payload["replyToMessageId"] = reply_to_message_id
    return WeChatEvent.parse(payload)


@pytest.mark.asyncio
async def test_wechat_pump_advances_cursor_for_ambient_without_admitting_workflow(
    tmp_path,
) -> None:
    store = await SQLiteInboundWorkStore(tmp_path / "ai.db").connect()
    await store.initialize_source(
        "wechat-test",
        epoch="wxid_self",
        initial_cursor="bootstrap-cursor",
    )
    workflow = RecordingWorkflow()
    pump = WeChatEventPump(
        object(),
        object(),
        store,
        connector_key="http://wechat-connector:18188",
        source_id="wechat-test",
        bootstrap=SimpleNamespace(session=SimpleNamespace(connection_generation=41)),
    )
    ambient = wechat_message_event(
        cursor="event-10",
        message_id="4159667620982040827",
        content="please discuss /ai later, but do not invoke it",
    )

    try:
        assert not await pump._accept_event(ambient, workflow)
        assert await store.get_cursor("wechat-test") == "event-10"
        assert workflow.accepted == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wechat_pump_passes_only_source_references_for_commands_replies_and_recall() -> (
    None
):
    class Inbox:
        def __init__(self) -> None:
            self.acknowledged: list[str] = []

        async def acknowledge_event(self, _source_id: str, cursor: str) -> None:
            self.acknowledged.append(cursor)

    inbox = Inbox()
    workflow = RecordingWorkflow()
    pump = WeChatEventPump(
        object(),
        object(),
        inbox,
        connector_key="http://wechat-connector:18188",
        source_id="wechat-test",
        bootstrap=SimpleNamespace(session=SimpleNamespace(connection_generation=41)),
    )
    command = wechat_message_event(
        cursor="event-11",
        message_id="4159667620982040828",
        content="  $ask connector-owned secret",
    )
    reply = wechat_message_event(
        cursor="event-12",
        message_id="4159667620982040829",
        content="continue from that answer",
        reply_to_message_id="4159667620982040828",
    )
    recall = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "event-13",
            "event": "message_remove",
            "connectionGeneration": 41,
            "status": "recalled",
            "chatId": "56825427596@chatroom",
            "id": "4159667620982040828",
        }
    )

    assert not await pump._accept_event(command, workflow)
    assert not await pump._accept_event(reply, workflow)
    assert not await pump._accept_event(recall, workflow)

    assert workflow.accepted == [
        {
            "cursor": "event-11",
            "chat_id": "56825427596@chatroom",
            "message_id": "4159667620982040828",
            "kind": "message",
            "attested_origin": None,
        },
        {
            "cursor": "event-12",
            "chat_id": "56825427596@chatroom",
            "message_id": "4159667620982040829",
            "kind": "message",
            "attested_origin": None,
        },
        {
            "cursor": "event-13",
            "chat_id": "56825427596@chatroom",
            "message_id": "4159667620982040828",
            "kind": "message_remove",
            "attested_origin": None,
        },
    ]
    assert inbox.acknowledged == []


class ObservedClient:
    def __init__(self, results: list[WeChatObservedMessage | Exception]) -> None:
        self.results = results

    async def get_observed_message(
        self,
        _chat_id: str,
        _message_id: str,
    ) -> WeChatObservedMessage:
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class EphemeralWeChatDirectory:
    async def message_from_observation(
        self,
        _connector_key: str,
        observed: WeChatObservedMessage,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=observed.id,
            chat_id=observed.chat_id,
            sender_id=observed.sender_id,
            raw_text=observed.display_content,
            reply_to_msg_id=observed.reply_to_message_id,
            content_redacted=observed.content_redacted,
            message_type=observed.message_type,
        )


class WeChatGenerationHandler:
    def __init__(self) -> None:
        self.generated: list[str] = []
        self.workflow_notices: list[tuple[str, str]] = []

    async def classify(self, _message, *, attested_origin=None):
        assert attested_origin is None
        return SimpleNamespace(
            disposition="generation",
            principal_actor_id="wechat:user:wxid_alice",
            scope_id="wechat:group:56825427596@chatroom",
            is_owner=False,
        )

    async def generation_eligible_at(self, _classification) -> float:
        return 0

    async def handle(
        self,
        message,
        *,
        attested_origin=None,
        workflow_admitted: bool = False,
    ) -> bool:
        assert attested_origin is None
        assert workflow_admitted
        self.generated.append(message.raw_text)
        return True

    async def reply_workflow_notice(self, message, notice: str) -> None:
        self.workflow_notices.append((message.id, notice))


def observed_present(content: str) -> WeChatObservedMessage:
    return WeChatObservedMessage.parse(
        {
            "id": "4159667620982040828",
            "chatId": "56825427596@chatroom",
            "state": "present",
            "version": "mv1:present",
            "direction": "in",
            "messageType": "text",
            "content": content,
            "senderId": "wxid_alice",
            "timestamp": 1_700_000_010,
            "orderTimestamp": 1_700_000_000,
            "observedAt": 1_786_651_200,
            "source": "wechat+localdb",
        }
    )


def observed_recalled() -> WeChatObservedMessage:
    return WeChatObservedMessage.parse(
        {
            "id": "4159667620982040828",
            "chatId": "56825427596@chatroom",
            "state": "recalled",
            "version": "mv1:recalled",
            "orderTimestamp": 1_700_000_000,
            "observedAt": 1_786_651_200,
            "source": "wechat+localdb",
        }
    )


@pytest.mark.asyncio
async def test_wechat_generation_refetches_partial_source_and_never_persists_payload(
    tmp_path,
) -> None:
    marker = "/ai connector-only payload 7b1e79f4"
    now = 100.0
    client = ObservedClient(
        [
            observed_present(marker),
            WeChatAPIError(503, "MESSAGE_HISTORY_NOT_READY", "not ready"),
            observed_recalled(),
        ]
    )
    source = WeChatObservedMessageSource(
        client,
        EphemeralWeChatDirectory(),
        "http://wechat-connector:18188",
    )
    database = tmp_path / "ai.db"
    store = await SQLiteInboundWorkStore(database).connect()
    await store.initialize_source(
        "wechat-test",
        epoch="wxid_self",
        initial_cursor="bootstrap-cursor",
    )
    handler = WeChatGenerationHandler()
    workflow = AIWorkflow(
        source,
        store,
        "wechat-test",
        handler,
        generation_concurrency=1,
        clock=lambda: now,
    )
    try:
        await workflow.accept(
            cursor="event-11",
            chat_id="56825427596@chatroom",
            message_id="4159667620982040828",
            kind="message",
            attested_origin=None,
        )
        assert await workflow._process_intake_one() == "queued"
        assert await workflow._process_generation_one() == "deferred"

        for sqlite_file in tmp_path.glob("ai.db*"):
            assert marker.encode() not in sqlite_file.read_bytes()

        now = 102.0
        assert await workflow._process_generation_one() == "recalled"
        assert handler.generated == []
    finally:
        await store.close()
