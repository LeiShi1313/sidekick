from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from io import BytesIO
import struct
import zlib

import pytest
from PIL import Image

from sidekick.ai import (
    AIConversationHandler,
    AIResponder,
    AIStateRepository,
    AgentEvent,
    AgentIdentityAnchor,
    AgentRequestIdentity,
    AgentRunOrigin,
    AgentRunRequest,
    PromptBuilder,
)
from sidekick.ai_continuous_memory import ContinuousMemoryScheduler
from sidekick.ai_attachments import AttachmentAnalysisRequest
from sidekick.ai_dream import DreamSettings
from sidekick.ai_memory import MemoryRetainResult
from sidekick.ai_memory_ingestion import (
    ChatMemoryIngestor,
    ContinuousMemoryResult,
    MemoryIngestionSettings,
)
from sidekick.ai_memory_segments import MemorySegmentationSettings
from sidekick.chat.attachments import OutboundAttachment
from sidekick.wechat.ai import (
    WECHAT_IDENTITY_CODEC,
    WeChatChatTransport,
    WeChatHistorySource,
    WeChatIdentityCodec,
    WeChatMessageIdentityResolver,
    WeChatMessageMentionResolver,
    WeChatQuotedImageDescriber,
)
from sidekick.wechat.api import (
    WeChatAPIError,
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatConnectorMessage,
    WeChatDownloadedImage,
    WeChatEvent,
    WeChatGroupMember,
    WeChatGroupMemberList,
    WeChatMessageList,
    WeChatSendFailed,
    WeChatSendOperation,
    WeChatSendOutcomeUnknown,
    WeChatSession,
    WeChatUser,
    WeChatUserList,
)
from sidekick.wechat.message import WeChatMessage
from sidekick.wechat.store import WeChatStateRepository


CONNECTOR_KEY = "http://127.0.0.1:18188"
ACCOUNT_ID = "wxid_self"
GROUP_ID = "56825427596@chatroom"


class RecordingConnectorClient:
    def __init__(self, responses: tuple[object, ...]):
        self.responses = list(responses)
        self.calls: list[dict[str, str | None]] = []
        self.attachment_calls: list[dict[str, object]] = []

    async def send_text_and_wait(
        self,
        *,
        request_id,
        to,
        content,
        reply_to_message_id,
    ):
        self.calls.append(
            {
                "request_id": request_id,
                "to": to,
                "content": content,
                "reply_to_message_id": reply_to_message_id,
            }
        )
        if not self.responses:
            raise AssertionError("No WeChat send response prepared")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def send_attachment_and_wait(
        self,
        *,
        request_id,
        to,
        attachment,
    ):
        self.attachment_calls.append(
            {
                "request_id": request_id,
                "to": to,
                "attachment": attachment,
            }
        )
        if not self.responses:
            raise AssertionError("No WeChat send response prepared")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingMediaConnectorClient(RecordingConnectorClient):
    def __init__(
        self,
        responses: tuple[object, ...],
        *,
        original: object,
        preview: object,
    ):
        super().__init__(responses)
        self.original = original
        self.preview = preview
        self.media_calls: list[tuple[str, str]] = []

    async def download_original_image(
        self,
        *,
        request_id,
        chat_id,
        message_id,
        media_id,
    ):
        assert request_id.startswith("sidekick.wechat.original.")
        assert chat_id == GROUP_ID
        assert message_id == "4159667620982040828"
        self.media_calls.append(("original", media_id))
        if isinstance(self.original, Exception):
            raise self.original
        return self.original

    async def download_image_preview(self, *, media_id):
        self.media_calls.append(("preview", media_id))
        if isinstance(self.preview, Exception):
            raise self.preview
        return self.preview


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


class ImageFinalGateway(FinalGateway):
    def __init__(self, description: str = "A sign says high resolution."):
        super().__init__("I used the quoted image.")
        self.description = description
        self.attachment_requests: list[AttachmentAnalysisRequest] = []

    async def describe_attachment(self, request: AttachmentAnalysisRequest) -> str:
        self.attachment_requests.append(request)
        return self.description


class RecordingMemory:
    def __init__(self):
        self.episodes = []

    async def retain_many(self, episodes, *, update_mode="replace"):
        self.episodes.extend(episodes)
        return MemoryRetainResult(accepted=True, items_count=len(episodes))


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


async def bootstrap_store(
    path,
    *,
    trigger_text="/ai hello",
    direction="out",
    message_type="text",
    media_id=None,
    content_redacted=False,
):
    store = await WeChatStateRepository(path).connect()
    trigger = WeChatConnectorMessage(
        id="4159667620982040828",
        chat_id=GROUP_ID,
        direction=direction,
        message_type=message_type,
        sender_id=ACCOUNT_ID if direction == "out" else "wxid_alice",
        reply_to_message_id=None,
        content=trigger_text,
        content_redacted=content_redacted,
        timestamp=1_783_772_734,
        source="wechat+localdb",
        sequence=None,
        media_id=media_id,
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
    observed = await (
        store.get_reply_message(CONNECTOR_KEY, GROUP_ID, trigger.id)
        if message_type == "image"
        else store.get_message(CONNECTOR_KEY, GROUP_ID, trigger.id)
    )
    assert observed is not None
    return store, observed


def image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2_000, 1_000), (255, 255, 255)).save(output, format="PNG")
    return output.getvalue()


def oversized_png_header() -> bytes:
    dimensions = struct.pack(">IIBBBBB", 5_001, 5_000, 8, 2, 0, 0, 0)

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", dimensions) + chunk(b"IEND", b"")


async def project_quoted_reply(
    store: WeChatStateRepository,
    *,
    reply_to: str,
    content: str = "/ai explain this",
) -> WeChatMessage:
    event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "quoted-command",
            "event": "message",
            "connectionGeneration": 41,
            "id": "5159667620982040828",
            "chatId": GROUP_ID,
            "direction": "out",
            "messageType": "app",
            "senderId": ACCOUNT_ID,
            "replyToMessageId": reply_to,
            "content": content,
            "timestamp": 1_783_772_735,
            "source": "wechat+message-reconciler",
        }
    )
    message = await store.project_event(CONNECTOR_KEY, event)
    assert message is not None
    return message


@pytest.mark.asyncio
async def test_wechat_responder_defers_placeholder_and_sends_one_bounded_final(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    answer = "你" * 2_000
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
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
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("wechat:user:test", "Tester"),
            anchors=(AgentIdentityAnchor("wechat:user:test", "Tester"),),
        ),
        origin=AgentRunOrigin("wechat:chat:test", "wechat-test"),
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
async def test_wechat_transport_uses_stable_request_id_for_same_trigger(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    client = RecordingConnectorClient(
        (
            submitted(message_id="7158246912028861544"),
            submitted(message_id="7158246912028861544"),
        )
    )
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    try:
        first = await transport.reply(trigger, "Access is open.", presentation="plain")
        second = await transport.reply(trigger, "Access is open.", presentation="plain")
    finally:
        await store.close()

    assert first.id == second.id == "7158246912028861544"
    assert client.calls[0]["request_id"] == client.calls[1]["request_id"]
    assert client.calls[0]["request_id"].startswith("sidekick.wechat.reply.")


@pytest.mark.asyncio
@pytest.mark.parametrize("display_as", ("image", "file"))
async def test_wechat_transport_replies_with_one_stably_identified_attachment(
    tmp_path,
    display_as,
    make_png,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    client = RecordingConnectorClient(
        (
            submitted(message_id="7158246912028861544"),
            submitted(message_id="7158246912028861544"),
        )
    )
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    attachment = OutboundAttachment(
        data=make_png() if display_as == "image" else b"attachment-bytes",
        filename="answer.png" if display_as == "image" else "answer.txt",
        mime_type="image/png" if display_as == "image" else "text/plain",
        display_as=display_as,
    )
    try:
        first = await transport.reply_attachment(trigger, attachment)
        second = await transport.reply_attachment(trigger, attachment)
    finally:
        await store.close()

    assert first is second is None
    assert client.attachment_calls[0] == client.attachment_calls[1]
    assert client.attachment_calls[0]["request_id"].startswith(
        "sidekick.wechat.attachment."
    )
    assert client.attachment_calls[0]["to"] == trigger.chat_id
    assert client.attachment_calls[0]["attachment"] is attachment


@pytest.mark.asyncio
async def test_wechat_attachment_retry_rotates_after_terminal_failure(
    tmp_path,
) -> None:
    state_path = tmp_path / "wechat.db"
    store, trigger = await bootstrap_store(state_path)
    failure = WeChatSendFailed(
        WeChatSendOperation(
            request_id="placeholder",
            status="failed",
            message_id=None,
            error_code="SEND_FAILED",
            to=GROUP_ID,
        ),
        "send failed",
    )
    client = RecordingConnectorClient((failure, submitted()))
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    attachment = OutboundAttachment(
        data=b"report",
        filename="report.txt",
        mime_type="text/plain",
        display_as="file",
    )
    try:
        with pytest.raises(WeChatSendFailed):
            await transport.reply_attachment(trigger, attachment)
        await transport.reply_attachment(trigger, attachment)
    finally:
        await store.close()

    restarted_store = await WeChatStateRepository(state_path).connect()
    restarted_client = RecordingConnectorClient((submitted(),))
    restarted_transport = WeChatChatTransport(
        restarted_client,
        restarted_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    try:
        await restarted_transport.reply_attachment(trigger, attachment)
    finally:
        await restarted_store.close()

    assert (
        client.attachment_calls[0]["request_id"]
        != client.attachment_calls[1]["request_id"]
    )
    assert (
        restarted_client.attachment_calls[0]["request_id"]
        == client.attachment_calls[1]["request_id"]
    )


@pytest.mark.asyncio
async def test_wechat_attachment_retry_preserves_id_after_unknown_outcome(
    tmp_path,
) -> None:
    state_path = tmp_path / "wechat.db"
    store, trigger = await bootstrap_store(state_path)
    unknown = WeChatSendOutcomeUnknown(
        WeChatSendOperation(
            request_id="placeholder",
            status="unknown",
            message_id=None,
            error_code="CONFIRMATION_TIMEOUT",
            to=GROUP_ID,
        ),
        "send outcome unknown",
    )
    client = RecordingConnectorClient((unknown,))
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    attachment = OutboundAttachment(
        data=b"report",
        filename="report.txt",
        mime_type="text/plain",
        display_as="file",
    )
    try:
        with pytest.raises(WeChatSendOutcomeUnknown):
            await transport.reply_attachment(trigger, attachment)
    finally:
        await store.close()

    restarted_store = await WeChatStateRepository(state_path).connect()
    restarted_client = RecordingConnectorClient((submitted(),))
    restarted_transport = WeChatChatTransport(
        restarted_client,
        restarted_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    try:
        await restarted_transport.reply_attachment(trigger, attachment)
    finally:
        await restarted_store.close()

    assert (
        client.attachment_calls[0]["request_id"]
        == restarted_client.attachment_calls[0]["request_id"]
    )


@pytest.mark.asyncio
async def test_wechat_transport_quotes_eligible_trigger_when_reply_is_ready(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=True,
    )
    try:
        await transport.reply(trigger, "Quoted answer.", presentation="plain")
    finally:
        await store.close()

    assert client.calls[0]["reply_to_message_id"] == trigger.id


@pytest.mark.asyncio
async def test_wechat_transport_falls_back_after_quote_send_failed(tmp_path) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    failed_operation = WeChatSendOperation(
        request_id="placeholder",
        status="failed",
        message_id=None,
        error_code="SEND_NOT_READY",
        to=GROUP_ID,
    )
    client = RecordingConnectorClient(
        (
            WeChatSendFailed(failed_operation, "quote send failed"),
            submitted(message_id="8158246912028861544"),
        )
    )
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=True,
    )
    try:
        sent = await transport.reply(
            trigger,
            "Plain fallback answer.",
            presentation="plain",
        )
    finally:
        await store.close()

    assert sent.id == "8158246912028861544"
    assert [call["reply_to_message_id"] for call in client.calls] == [
        trigger.id,
        None,
    ]
    assert client.calls[1]["request_id"] == f"{client.calls[0]['request_id']}.plain"
    assert client.calls[0]["content"] == client.calls[1]["content"]


@pytest.mark.parametrize(
    ("status", "code"),
    (
        (422, "REPLY_UNSUPPORTED"),
        (503, "SEND_NOT_READY"),
        (501, "SEND_UNAVAILABLE"),
    ),
)
@pytest.mark.asyncio
async def test_wechat_transport_falls_back_after_safe_quote_rejection(
    tmp_path,
    status,
    code,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    client = RecordingConnectorClient(
        (
            WeChatAPIError(status, code, "quote rejected before activation"),
            submitted(message_id="8158246912028861544"),
        )
    )
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=True,
    )
    try:
        await transport.reply(trigger, "Plain fallback answer.", presentation="plain")
    finally:
        await store.close()

    assert [call["reply_to_message_id"] for call in client.calls] == [
        trigger.id,
        None,
    ]


@pytest.mark.asyncio
async def test_wechat_transport_does_not_fallback_after_quote_outcome_unknown(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    unknown_operation = WeChatSendOperation(
        request_id="placeholder",
        status="unknown",
        message_id=None,
        error_code="SEND_OUTCOME_UNKNOWN",
        to=GROUP_ID,
    )
    client = RecordingConnectorClient(
        (WeChatSendOutcomeUnknown(unknown_operation, "outcome unknown"),)
    )
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=True,
    )
    try:
        with pytest.raises(WeChatSendOutcomeUnknown):
            await transport.reply(trigger, "Possibly sent.", presentation="plain")
    finally:
        await store.close()

    assert len(client.calls) == 1
    assert client.calls[0]["reply_to_message_id"] == trigger.id


@pytest.mark.asyncio
async def test_wechat_transport_stops_after_plain_fallback_becomes_unknown(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    failed_operation = WeChatSendOperation(
        request_id="placeholder",
        status="failed",
        message_id=None,
        error_code="SEND_NOT_READY",
        to=GROUP_ID,
    )
    unknown_operation = WeChatSendOperation(
        request_id="placeholder.plain",
        status="unknown",
        message_id=None,
        error_code="SEND_OUTCOME_UNKNOWN",
        to=GROUP_ID,
    )
    client = RecordingConnectorClient(
        (
            WeChatSendFailed(failed_operation, "quote send failed"),
            WeChatSendOutcomeUnknown(unknown_operation, "fallback outcome unknown"),
        )
    )
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=True,
    )
    try:
        with pytest.raises(WeChatSendOutcomeUnknown):
            await transport.reply(trigger, "Possibly sent plainly.", presentation="plain")
    finally:
        await store.close()

    assert len(client.calls) == 2
    assert [call["reply_to_message_id"] for call in client.calls] == [
        trigger.id,
        None,
    ]


@pytest.mark.asyncio
async def test_wechat_transport_sends_plain_text_when_reply_is_not_ready(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    try:
        await transport.reply(trigger, "Plain answer.", presentation="plain")
    finally:
        await store.close()

    assert client.calls[0]["reply_to_message_id"] is None


@pytest.mark.asyncio
async def test_wechat_transport_does_not_quote_ineligible_trigger(tmp_path) -> None:
    store, trigger = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="/ai hello\nwith more detail",
    )
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=True,
    )
    try:
        await transport.reply(trigger, "Plain answer.", presentation="plain")
    finally:
        await store.close()

    assert client.calls[0]["reply_to_message_id"] is None


@pytest.mark.asyncio
async def test_wechat_transport_sends_followup_when_reply_cannot_be_edited(
    tmp_path,
) -> None:
    store, trigger = await bootstrap_store(tmp_path / "wechat.db")
    client = RecordingConnectorClient(
        (
            submitted(message_id="7158246912028861544"),
            submitted(message_id="8158246912028861544"),
        )
    )
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    try:
        progress = await transport.reply(
            trigger,
            "Backfilling stored WeChat messages...",
            presentation="plain",
        )
        updated = await transport.update(
            progress,
            "Memory backfill complete.",
            presentation="plain",
            wait=True,
        )
    finally:
        await store.close()

    assert updated is True
    assert [call["content"] for call in client.calls] == [
        "Backfilling stored WeChat messages...",
        "Memory backfill complete.",
    ]
    assert client.calls[0]["request_id"] != client.calls[1]["request_id"]


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
        transport=WeChatChatTransport(
            client,
            store,
            CONNECTOR_KEY,
            native_reply_ready=False,
        ),
    )
    request = AgentRunRequest(
        run_id="run-1",
        session_id=None,
        parent_entry_id=None,
        prompt="hello",
        context=(),
        system_prompt="system",
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("wechat:user:test", "Tester"),
            anchors=(AgentIdentityAnchor("wechat:user:test", "Tester"),),
        ),
        origin=AgentRunOrigin("wechat:chat:test", "wechat-test"),
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
        transport=WeChatChatTransport(
            client,
            store,
            CONNECTOR_KEY,
            native_reply_ready=False,
        ),
    )
    request = AgentRunRequest(
        run_id="run-1",
        session_id=None,
        parent_entry_id=None,
        prompt="hello",
        context=(),
        system_prompt="system",
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("wechat:user:test", "Tester"),
            anchors=(AgentIdentityAnchor("wechat:user:test", "Tester"),),
        ),
        origin=AgentRunOrigin("wechat:chat:test", "wechat-test"),
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
    transport = WeChatChatTransport(
        client,
        wechat_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    history = WeChatHistorySource(wechat_store, CONNECTOR_KEY)
    gateway = FinalGateway("hello from Sidekick")
    memory = RecordingMemory()
    identity_codec = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    prompt_builder = PromptBuilder(
        transport=transport,
        history_source=history,
        identity_resolver=WeChatMessageIdentityResolver(identity_codec),
        mention_resolver=WeChatMessageMentionResolver(),
        identity_codec=identity_codec,
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
        memory=memory,
        transport=transport,
        identity_codec=identity_codec,
    )
    try:
        handled = await handler.handle(trigger)
        marker = await ai_store.get_answer(
            identity_codec.scope_id(GROUP_ID),
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
    assert gateway.requests[0].memory is not None
    assert gateway.requests[0].memory.primary_bank_id == identity_codec.scope_id(
        GROUP_ID
    )
    assert memory.episodes[0].scope_id == identity_codec.scope_id(GROUP_ID)
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
async def test_wechat_conversation_handler_uses_quoted_message_as_context(
    tmp_path,
) -> None:
    wechat_store, target = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="The bridge uses a native hook and local projection.",
        direction="in",
    )
    command = await project_quoted_reply(wechat_store, reply_to=target.id)
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        wechat_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    gateway = FinalGateway("Here is how it works.")
    identity_codec = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    handler = AIConversationHandler(
        owner_id=ACCOUNT_ID,
        responder=AIResponder(
            gateway,
            initial_status=None,
            transport=transport,
        ),
        store=ai_store,
        prompt_builder=PromptBuilder(
            transport=transport,
            history_source=WeChatHistorySource(wechat_store, CONNECTOR_KEY),
            identity_resolver=WeChatMessageIdentityResolver(identity_codec),
            mention_resolver=WeChatMessageMentionResolver(),
            identity_codec=identity_codec,
        ),
        transport=transport,
        identity_codec=identity_codec,
    )
    try:
        handled = await handler.handle(command)
        marker = await ai_store.get_answer(
            identity_codec.scope_id(GROUP_ID),
            "7158246912028861544",
        )
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert handled is True
    assert gateway.requests[0].prompt == "explain this"
    assert len(gateway.requests[0].context) == 1
    assert "The bridge uses a native hook and local projection." in (
        gateway.requests[0].context[0].text
    )
    assert marker is not None
    assert marker.agent_session_id == "session-1"
    assert marker.agent_entry_id == "entry-1"


@pytest.mark.asyncio
async def test_wechat_conversation_handler_uses_quoted_original_image_as_context(
    tmp_path,
) -> None:
    media_id = "0123456789abcdef0123456789abcdef"
    raw_image = image_bytes()
    wechat_store, target = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="",
        direction="in",
        message_type="image",
        media_id=media_id,
        content_redacted=True,
    )
    command = await project_quoted_reply(wechat_store, reply_to=target.id)
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    client = RecordingMediaConnectorClient(
        (submitted(),),
        original=WeChatDownloadedImage(
            data=raw_image,
            mime_type="image/png",
            variant="original",
        ),
        preview=AssertionError("preview should not be requested"),
    )
    transport = WeChatChatTransport(
        client,
        wechat_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    gateway = ImageFinalGateway()
    identity_codec = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    handler = AIConversationHandler(
        owner_id=ACCOUNT_ID,
        responder=AIResponder(gateway, initial_status=None, transport=transport),
        store=ai_store,
        prompt_builder=PromptBuilder(
            transport=transport,
            history_source=WeChatHistorySource(wechat_store, CONNECTOR_KEY),
            quoted_attachment_describer=WeChatQuotedImageDescriber(
                client,
                gateway,
                request_original=True,
                download_preview=True,
            ),
            identity_resolver=WeChatMessageIdentityResolver(identity_codec),
            mention_resolver=WeChatMessageMentionResolver(),
            identity_codec=identity_codec,
        ),
        transport=transport,
        identity_codec=identity_codec,
    )
    try:
        handled = await handler.handle(command)
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert handled is True
    assert client.media_calls == [("original", media_id)]
    assert len(gateway.attachment_requests) == 1
    attachment_request = gateway.attachment_requests[0]
    assert attachment_request.kind == "image"
    assert attachment_request.mime_type == "image/jpeg"
    assert attachment_request.data is not None
    with Image.open(BytesIO(attachment_request.data)) as normalized:
        assert max(normalized.size) == 1_600
    assert len(gateway.requests[0].context) == 1
    assert "A sign says high resolution." in gateway.requests[0].context[0].text
    assert len(gateway.requests[0].images) == 1
    assert gateway.requests[0].images[0].data == attachment_request.data


@pytest.mark.asyncio
async def test_wechat_quoted_image_falls_back_to_preview_when_original_fails(
    tmp_path,
) -> None:
    media_id = "0123456789abcdef0123456789abcdef"
    wechat_store, target = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="",
        direction="in",
        message_type="image",
        media_id=media_id,
    )
    gateway = ImageFinalGateway("A readable low-resolution sign.")
    client = RecordingMediaConnectorClient(
        (),
        original=WeChatAPIError(404, "ORIGINAL_IMAGE_NOT_FOUND", "not found"),
        preview=WeChatDownloadedImage(
            data=image_bytes(),
            mime_type="image/png",
            variant="preview",
        ),
    )
    describer = WeChatQuotedImageDescriber(
        client,
        gateway,
        request_original=True,
        download_preview=True,
    )
    try:
        result = await describer.describe(target)
    finally:
        await wechat_store.close()

    assert result is not None
    assert "readable low-resolution sign" in result.context_text
    assert result.model_image is not None
    assert result.model_image.mime_type == "image/jpeg"
    assert client.media_calls == [
        ("original", media_id),
        ("preview", media_id),
    ]


@pytest.mark.asyncio
async def test_wechat_quoted_image_falls_back_when_original_cannot_normalize(
    tmp_path,
) -> None:
    media_id = "0123456789abcdef0123456789abcdef"
    wechat_store, target = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="",
        direction="in",
        message_type="image",
        media_id=media_id,
    )
    gateway = ImageFinalGateway("A usable preview.")
    client = RecordingMediaConnectorClient(
        (),
        original=WeChatDownloadedImage(
            data=oversized_png_header(),
            mime_type="image/png",
            variant="original",
        ),
        preview=WeChatDownloadedImage(
            data=image_bytes(),
            mime_type="image/png",
            variant="preview",
        ),
    )
    describer = WeChatQuotedImageDescriber(
        client,
        gateway,
        request_original=True,
        download_preview=True,
    )
    try:
        result = await describer.describe(target)
    finally:
        await wechat_store.close()

    assert result is not None
    assert result.model_image is not None
    assert "usable preview" in result.context_text.lower()
    assert client.media_calls == [
        ("original", media_id),
        ("preview", media_id),
    ]
    assert len(gateway.attachment_requests) == 1


@pytest.mark.asyncio
async def test_wechat_quoted_image_uses_preview_without_original_capability(
    tmp_path,
) -> None:
    media_id = "0123456789abcdef0123456789abcdef"
    wechat_store, target = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="",
        direction="in",
        message_type="image",
        media_id=media_id,
    )
    gateway = ImageFinalGateway("A preview-only description.")
    client = RecordingMediaConnectorClient(
        (),
        original=AssertionError("original should be capability-gated"),
        preview=WeChatDownloadedImage(
            data=image_bytes(),
            mime_type="image/png",
            variant="preview",
        ),
    )
    describer = WeChatQuotedImageDescriber(
        client,
        gateway,
        request_original=False,
        download_preview=True,
    )
    try:
        result = await describer.describe(target)
    finally:
        await wechat_store.close()

    assert result is not None
    assert "preview-only description" in result.context_text
    assert result.model_image is not None
    assert client.media_calls == [("preview", media_id)]


@pytest.mark.asyncio
async def test_wechat_quoted_image_reports_when_no_image_variant_is_available(
    tmp_path,
) -> None:
    media_id = "0123456789abcdef0123456789abcdef"
    wechat_store, target = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="",
        direction="in",
        message_type="image",
        media_id=media_id,
    )
    gateway = ImageFinalGateway()
    client = RecordingMediaConnectorClient(
        (),
        original=AssertionError("original should be capability-gated"),
        preview=AssertionError("preview should be capability-gated"),
    )
    describer = WeChatQuotedImageDescriber(
        client,
        gateway,
        request_original=False,
        download_preview=False,
    )
    try:
        result = await describer.describe(target)
    finally:
        await wechat_store.close()

    assert result is not None
    assert "quoted image content is unavailable" in result.context_text.lower()
    assert result.model_image is None
    assert client.media_calls == []
    assert gateway.attachment_requests == []


@pytest.mark.asyncio
async def test_wechat_conversation_handler_treats_quoted_lookup_as_best_effort(
    tmp_path,
    monkeypatch,
) -> None:
    wechat_store, _ = await bootstrap_store(tmp_path / "wechat.db")
    command = await project_quoted_reply(
        wechat_store,
        reply_to="3159667620982040828",
    )

    async def unavailable_reply(*_args, **_kwargs):
        raise RuntimeError("projection unavailable")

    monkeypatch.setattr(wechat_store, "get_reply_message", unavailable_reply)
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        wechat_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    gateway = FinalGateway("Answer without quoted context.")
    identity_codec = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    handler = AIConversationHandler(
        owner_id=ACCOUNT_ID,
        responder=AIResponder(
            gateway,
            initial_status=None,
            transport=transport,
        ),
        store=ai_store,
        prompt_builder=PromptBuilder(
            transport=transport,
            history_source=WeChatHistorySource(wechat_store, CONNECTOR_KEY),
            identity_resolver=WeChatMessageIdentityResolver(identity_codec),
            mention_resolver=WeChatMessageMentionResolver(),
            identity_codec=identity_codec,
        ),
        transport=transport,
        identity_codec=identity_codec,
    )
    try:
        handled = await handler.handle(command)
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert handled is True
    assert gateway.requests[0].prompt == "explain this"
    assert gateway.requests[0].context == ()
    assert client.calls[0]["content"] == "Answer without quoted context."


@pytest.mark.asyncio
async def test_wechat_manual_memory_command_uses_refreshed_group_alias(
    tmp_path,
) -> None:
    wechat_store, target = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="Remember the launch date is Friday",
        direction="in",
    )
    await wechat_store.refresh_users(
        CONNECTOR_KEY,
        WeChatUserList(
            users=(WeChatUser(id="wxid_alice", display_name="Alice Global"),),
            cursor="users-10",
        ),
    )
    await wechat_store.refresh_group_members(
        CONNECTOR_KEY,
        GROUP_ID,
        WeChatGroupMemberList(
            group_id=GROUP_ID,
            members=(
                WeChatGroupMember(
                    group_id=GROUP_ID,
                    user_id="wxid_alice",
                    display_name="Alice Global",
                    nickname="旧项目阿丽",
                ),
            ),
            cursor="members-10",
            snapshot_complete=False,
            snapshot_current=False,
            snapshot_connection_generation=None,
        ),
    )
    await wechat_store.refresh_group_members(
        CONNECTOR_KEY,
        GROUP_ID,
        WeChatGroupMemberList(
            group_id=GROUP_ID,
            members=(
                WeChatGroupMember(
                    group_id=GROUP_ID,
                    user_id="wxid_alice",
                    display_name="Alice Global",
                    nickname="项目阿丽",
                ),
            ),
            cursor="members-11",
            snapshot_complete=True,
            snapshot_current=True,
            snapshot_connection_generation=41,
        ),
    )
    command_event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "manual-memory-command",
            "event": "message",
            "connectionGeneration": 41,
            "id": "5159667620982040828",
            "chatId": GROUP_ID,
            "direction": "out",
            "messageType": "text",
            "senderId": ACCOUNT_ID,
            "replyToMessageId": target.id,
            "content": "/ai_memory",
            "timestamp": 1_783_772_735,
            "source": "wechat+hook",
        }
    )
    command = await wechat_store.project_event(CONNECTOR_KEY, command_event)
    assert command is not None
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        wechat_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    history = WeChatHistorySource(wechat_store, CONNECTOR_KEY)
    memory = RecordingMemory()
    identity_codec = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    handler = AIConversationHandler(
        owner_id=ACCOUNT_ID,
        responder=AIResponder(
            FinalGateway(),
            initial_status=None,
            transport=transport,
        ),
        store=ai_store,
        prompt_builder=PromptBuilder(
            transport=transport,
            history_source=history,
            identity_resolver=WeChatMessageIdentityResolver(identity_codec),
            mention_resolver=WeChatMessageMentionResolver(),
            identity_codec=identity_codec,
        ),
        memory=memory,
        memory_command_delete_delay=0,
        transport=transport,
        identity_codec=identity_codec,
    )
    try:
        handled = await handler.handle(command)
        await asyncio.sleep(0)
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert handled is True
    assert client.calls[0]["content"] == "Memory stored from reply chain: 1 message."
    assert memory.episodes[0].events[0].source_id == (
        identity_codec.message_source_id(GROUP_ID, target.id)
    )
    assert memory.episodes[0].events[0].actor_id == identity_codec.actor_id(
        "wxid_alice"
    )
    assert memory.episodes[0].events[0].actor_display_name == "项目阿丽"
    assert memory.episodes[0].events[0].text == ("Remember the launch date is Friday")


@pytest.mark.asyncio
async def test_wechat_backfill_reports_and_ingests_only_locally_stored_history(
    tmp_path,
) -> None:
    caveat = (
        "WeChat backfill covers only messages already observed and stored by Sidekick."
    )
    wechat_store, target = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="The migration window starts at 22:00",
        direction="in",
    )
    command_event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "backfill-command",
            "event": "message",
            "connectionGeneration": 41,
            "id": "5159667620982040828",
            "chatId": GROUP_ID,
            "direction": "out",
            "messageType": "text",
            "senderId": ACCOUNT_ID,
            "content": "/ai_memory_backfill messages 10",
            "timestamp": 1_783_772_735,
            "source": "wechat+hook",
        }
    )
    command = await wechat_store.project_event(CONNECTOR_KEY, command_event)
    assert command is not None
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    client = RecordingConnectorClient(
        (
            submitted(message_id="7158246912028861544"),
            submitted(message_id="8158246912028861544"),
        )
    )
    transport = WeChatChatTransport(
        client,
        wechat_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    history = WeChatHistorySource(wechat_store, CONNECTOR_KEY)
    memory = RecordingMemory()
    identity_codec = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    prompt_builder = PromptBuilder(
        transport=transport,
        history_source=history,
        identity_resolver=WeChatMessageIdentityResolver(identity_codec),
        mention_resolver=WeChatMessageMentionResolver(),
        identity_codec=identity_codec,
    )
    scanner = ChatMemoryIngestor(
        source=history,
        store=ai_store,
        memory=memory,
        prompt_builder=prompt_builder,
        dream_settings=DreamSettings(),
        ingestion_settings=MemoryIngestionSettings(
            settlement_delay=timedelta(0),
        ),
        identity_codec=identity_codec,
        clock=lambda: 1_783_773_000,
    )
    handler = AIConversationHandler(
        owner_id=ACCOUNT_ID,
        responder=AIResponder(
            FinalGateway(),
            initial_status=None,
            transport=transport,
        ),
        store=ai_store,
        prompt_builder=prompt_builder,
        memory=memory,
        dream_runner=scanner,
        memory_backfill_caveat=caveat,
        memory_command_delete_delay=0,
        transport=transport,
        identity_codec=identity_codec,
    )
    try:
        handled = await handler.handle(command)
        final_reply_excluded = await ai_store.is_memory_excluded_message(
            identity_codec.scope_id(GROUP_ID),
            "8158246912028861544",
        )
        await asyncio.sleep(0)
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert handled is True
    assert final_reply_excluded is True
    assert caveat in client.calls[0]["content"]
    assert caveat in client.calls[1]["content"]
    assert "scanned 2 messages; retained 1" in client.calls[1]["content"]
    assert [event.source_id for event in memory.episodes[0].events] == [
        identity_codec.message_source_id(GROUP_ID, target.id)
    ]


@pytest.mark.asyncio
async def test_wechat_memory_enable_starts_after_command_projection_cursor(
    tmp_path,
) -> None:
    class NoopDreamRunner:
        async def run_scope(self, _chat_id):
            raise AssertionError("Dream should not run while enabling memory")

        async def run_backfill(self, _chat_id, _request):
            raise AssertionError("Backfill should not run while enabling memory")

    wechat_store, command = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="/ai_memory_enable",
        direction="out",
    )
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        wechat_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    history = WeChatHistorySource(wechat_store, CONNECTOR_KEY)
    memory = RecordingMemory()
    identity_codec = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    handler = AIConversationHandler(
        owner_id=ACCOUNT_ID,
        responder=AIResponder(
            FinalGateway(),
            initial_status=None,
            transport=transport,
        ),
        store=ai_store,
        prompt_builder=PromptBuilder(
            transport=transport,
            history_source=history,
            identity_resolver=WeChatMessageIdentityResolver(identity_codec),
            mention_resolver=WeChatMessageMentionResolver(),
            identity_codec=identity_codec,
        ),
        memory=memory,
        dream_runner=NoopDreamRunner(),
        memory_command_delete_delay=0,
        transport=transport,
        identity_codec=identity_codec,
    )
    scope_id = identity_codec.scope_id(GROUP_ID)
    try:
        handled = await handler.handle(command)
        state = await ai_store.get_memory_scope_state(scope_id)
        await asyncio.sleep(0)
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert handled is True
    assert state.continuous_enabled is True
    assert state.continuous_cursor_message_id == command.memory_cursor
    assert client.calls[0]["content"] == (
        "Continuous memory enabled for this chat. New messages will be remembered."
    )


@pytest.mark.asyncio
async def test_wechat_memory_enable_reports_when_hindsight_is_disabled(
    tmp_path,
) -> None:
    wechat_store, command = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="/ai_memory_enable",
        direction="out",
    )
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    client = RecordingConnectorClient((submitted(),))
    transport = WeChatChatTransport(
        client,
        wechat_store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    history = WeChatHistorySource(wechat_store, CONNECTOR_KEY)
    identity_codec = WeChatIdentityCodec(account_id=ACCOUNT_ID)
    handler = AIConversationHandler(
        owner_id=ACCOUNT_ID,
        responder=AIResponder(
            FinalGateway(),
            initial_status=None,
            transport=transport,
        ),
        store=ai_store,
        prompt_builder=PromptBuilder(
            transport=transport,
            history_source=history,
            identity_resolver=WeChatMessageIdentityResolver(identity_codec),
            mention_resolver=WeChatMessageMentionResolver(),
            identity_codec=identity_codec,
        ),
        memory=None,
        dream_runner=None,
        memory_command_delete_delay=0,
        transport=transport,
        identity_codec=identity_codec,
    )
    scope_id = identity_codec.scope_id(GROUP_ID)
    try:
        handled = await handler.handle(command)
        state = await ai_store.get_memory_scope_state(scope_id)
        await asyncio.sleep(0)
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert handled is True
    assert state.continuous_enabled is False
    assert client.calls[0]["content"] == (
        "Memory ingestion is unavailable because Hindsight is disabled."
    )


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


@pytest.mark.asyncio
async def test_wechat_continuous_memory_checkpoints_projection_cursor(tmp_path) -> None:
    wechat_store, trigger = await bootstrap_store(
        tmp_path / "wechat.db",
        trigger_text="project decision",
        direction="in",
    )
    ai_store = await AIStateRepository(tmp_path / "ai.db").connect()
    memory = RecordingMemory()
    source = WeChatHistorySource(wechat_store, CONNECTOR_KEY)
    scanner = ChatMemoryIngestor(
        source=source,
        store=ai_store,
        memory=memory,
        prompt_builder=PromptBuilder(
            identity_resolver=WeChatMessageIdentityResolver(),
            mention_resolver=WeChatMessageMentionResolver(),
            history_source=source,
            identity_codec=WECHAT_IDENTITY_CODEC,
        ),
        dream_settings=DreamSettings(),
        ingestion_settings=MemoryIngestionSettings(
            settlement_delay=timedelta(0),
            segmentation=MemorySegmentationSettings(
                idle_gap=timedelta(seconds=1),
            ),
        ),
        identity_codec=WECHAT_IDENTITY_CODEC,
        clock=lambda: 1_783_773_000,
    )
    scope_id = WECHAT_IDENTITY_CODEC.scope_id(GROUP_ID)
    await ai_store.set_continuous_memory_enabled(
        scope_id,
        True,
        cursor_message_id=0,
    )
    try:
        initial_result = await scanner.run_continuous_scope(GROUP_ID)
        initial_state = await ai_store.get_memory_scope_state(scope_id)

        revision = WeChatEvent.parse(
            {
                "schemaVersion": "wechat-bridge/v1alpha1",
                "cursor": "late-revision",
                "event": "message",
                "connectionGeneration": 41,
                "id": trigger.id,
                "chatId": GROUP_ID,
                "direction": "in",
                "messageType": "text",
                "senderId": "wxid_alice",
                "content": "project decision, corrected",
                "timestamp": 1_783_772_734,
                "source": "wechat+localdb",
            }
        )
        corrected = await wechat_store.project_event(CONNECTOR_KEY, revision)
        assert corrected is not None

        revised_result = await scanner.run_continuous_scope(GROUP_ID)
        revised_state = await ai_store.get_memory_scope_state(scope_id)
    finally:
        await ai_store.close()
        await wechat_store.close()

    assert initial_result.messages_seen == 1
    assert initial_result.messages_retained == 1
    assert initial_state.continuous_cursor_message_id == trigger.memory_cursor
    assert revised_result.messages_seen == 1
    assert revised_result.messages_retained == 1
    assert revised_state.continuous_cursor_message_id == corrected.memory_cursor
    assert corrected.memory_cursor > trigger.memory_cursor
    assert [event.source_id for event in memory.episodes[0].events] == [
        WECHAT_IDENTITY_CODEC.message_source_id(GROUP_ID, trigger.id)
    ]
    assert memory.episodes[-1].events[0].text == "project decision, corrected"
