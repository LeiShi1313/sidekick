from __future__ import annotations

import asyncio
import base64
import logging
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp.test_utils import TestServer
from PIL import Image

from sidekick.ai import AIConversationHandler, AIResponder, AISettings, PromptBuilder
from sidekick.ai_attachments import (
    AttachmentAnalysisRequest,
    ChatAttachmentDescriber,
)

from sidekick.chat.attachments import OutboundAttachment
from sidekick.chat.output_policy import MAINLAND_MESSAGING_POLICY_ID
from sidekick.chat.provenance import GeneratedMessageTracker, MessageOrigin
from sidekick.channel_status import ChannelOpsSettings
from sidekick.onebot.ai import (
    QQ_IDENTITY_CODEC,
    OneBotChatTransport,
    OneBotDirectory,
    OneBotDirectorySourceResolver,
    OneBotHistorySource,
    OneBotInboundMessageSource,
    OneBotMessageIdentityResolver,
    OneBotMessageMentionResolver,
)
from sidekick.onebot.client import (
    OneBotActionError,
    OneBotReverseWebSocket,
)
from sidekick.onebot.message import (
    OneBotMessage,
    OneBotMessageError,
)
from sidekick.plugins.onebot_ai import OneBotAI


class RecordingActionClient:
    def __init__(self, responses=()):
        self.calls = []
        self.responses = list(responses)

    async def call(self, action, params=None, *, timeout=None):
        self.calls.append((action, params or {}, timeout))
        if not self.responses:
            raise AssertionError(f"No response prepared for {action}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingAttachmentGateway:
    def __init__(self):
        self.requests: list[AttachmentAnalysisRequest] = []

    async def describe_attachment(self, request: AttachmentAnalysisRequest) -> str:
        self.requests.append(request)
        return "A red image."


@pytest.mark.asyncio
async def test_onebot_runtime_applies_mainland_output_policy(tmp_path) -> None:
    class SetupStore:
        async def connect(self):
            return self

    class SetupInboundStore(SetupStore):
        async def initialize_source(self, *_args, **_kwargs):
            return 0

        async def recover_pending_ai_work(self, _source_id):
            return None

    class SetupBridge:
        def set_event_handler(self, handler):
            self.event_handler = handler

    plugin = object.__new__(OneBotAI)
    plugin._runtime = SimpleNamespace(self_id=99)
    plugin._settings = AISettings(
        agent_url="http://agent.invalid",
        agent_token="test-agent-token-that-is-long-enough",
        state_path=tmp_path / "ai.db",
    )
    plugin._ops_settings = ChannelOpsSettings(
        instance_id="qq-test",
        token="channel-ops-token-that-is-long-enough",
    )
    plugin._gateway = object()
    plugin._store = SetupStore()
    plugin._inbound_store = SetupInboundStore()
    plugin._memory = None
    plugin._bridge = SetupBridge()
    plugin._directory = OneBotDirectory()
    plugin._dream_scheduler = None
    plugin._continuous_scheduler = None
    plugin._memory_outbox_scheduler = None
    plugin.logger = logging.getLogger("test-onebot-output-policy")

    await plugin._setup()

    assert MAINLAND_MESSAGING_POLICY_ID in (
        plugin._handler._prompt_builder.system_prompt
    )
    assert (
        plugin._handler._responder._output_policy.policy_id
        == MAINLAND_MESSAGING_POLICY_ID
    )
    assert plugin._handler._responder._initial_status is None


def group_event(
    *,
    message_id=101,
    sender_id=42,
    group_id=700,
    text="/ai hello",
    segments=None,
    post_type="message",
    timestamp=1_700_000_000,
):
    return {
        "post_type": post_type,
        "message_type": "group",
        "self_id": 99,
        "user_id": sender_id,
        "group_id": group_id,
        "group_name": "Dog Food Filter",
        "message_id": message_id,
        "time": timestamp,
        "sender": {
            "user_id": sender_id,
            "nickname": "Alice",
            "card": "Alice Card",
            "role": "member",
        },
        "message": segments or [{"type": "text", "data": {"text": text}}],
        "raw_message": text,
    }


def private_event(
    *,
    message_id=201,
    sender_id=42,
    target_id=42,
    text="/ai hello",
    post_type="message",
):
    return {
        "post_type": post_type,
        "message_type": "private",
        "self_id": 99,
        "user_id": sender_id,
        "target_id": target_id,
        "message_id": message_id,
        "time": 1_700_000_000,
        "sender": {
            "user_id": sender_id,
            "nickname": "Cherry",
            "card": "",
        },
        "message": [{"type": "text", "data": {"text": text}}],
        "raw_message": text,
    }


def test_qq_identity_codec_separates_group_and_private_scopes():
    assert QQ_IDENTITY_CODEC.actor_id(42) == "qq:user:42"
    assert QQ_IDENTITY_CODEC.scope_id(700) == "qq:group:700"
    assert QQ_IDENTITY_CODEC.scope_id(-42) == "qq:private:42"
    assert QQ_IDENTITY_CODEC.parse_scope_id("qq:group:700") == 700
    assert QQ_IDENTITY_CODEC.parse_scope_id("qq:private:42") == -42
    assert QQ_IDENTITY_CODEC.message_source_id(-42, 9) == "qq:message:private:42:9"


@pytest.mark.asyncio
async def test_onebot_directory_exposes_only_its_cached_channel_inventory() -> None:
    directory = OneBotDirectory()
    client = RecordingActionClient(
        responses=[
            [{"user_id": 42, "remark": "Cherry", "nickname": "Old name"}],
            [{"group_id": 700, "group_name": "Dog Food Filter"}],
        ]
    )
    await directory.refresh(client)

    channels = await directory.list_channels()

    assert [(item.scope_id, item.display_name, item.chat_kind) for item in channels] == [
        ("qq:group:700", "Dog Food Filter", "GROUP"),
        ("qq:private:42", "Cherry", "DIRECT"),
    ]
    assert all(item.last_observed_at is not None for item in channels)
    assert len(client.calls) == 2
    await directory.list_channels()
    assert len(client.calls) == 2


def test_onebot_transport_distinguishes_group_messages():
    transport = OneBotChatTransport(RecordingActionClient())

    group = OneBotMessage.from_payload(
        group_event(),
        action_client=RecordingActionClient(),
    )
    private = OneBotMessage.from_payload(
        private_event(),
        action_client=RecordingActionClient(),
    )

    assert transport.is_group(group) is True
    assert transport.is_group(private) is False


@pytest.mark.asyncio
async def test_qq_directory_resolves_current_and_numeric_groups():
    client = RecordingActionClient(
        responses=[
            {"group_id": 700, "group_name": "Dog Food Filter"},
            {"group_id": 800, "group_name": "Arch Linux"},
        ]
    )
    resolver = OneBotDirectorySourceResolver(client)
    message = OneBotMessage.from_payload(
        group_event(group_id=700),
        action_client=client,
    )

    current = await resolver.resolve_publication(message, "群内筛选信息")
    explicit = await resolver.resolve_publication(message, "800 Linux 中文群")

    assert current.source.bank_id == "qq:group:700"
    assert current.source.display_name == "Dog Food Filter"
    assert current.description == "群内筛选信息"
    assert explicit.source.bank_id == "qq:group:800"
    assert explicit.source.display_name == "Arch Linux"
    assert explicit.description == "Linux 中文群"
    assert [call[0] for call in client.calls] == ["get_group_info", "get_group_info"]


@pytest.mark.asyncio
async def test_qq_directory_rejects_private_and_nonnumeric_selectors():
    client = RecordingActionClient()
    resolver = OneBotDirectorySourceResolver(client)
    private = OneBotMessage.from_payload(
        private_event(),
        action_client=client,
    )

    with pytest.raises(ValueError, match="QQ group"):
        await resolver.resolve_publication(private, "")
    with pytest.raises(ValueError, match="numeric QQ group"):
        await resolver.resolve_bank(private, "@Seele_Leaks")


def test_onebot_message_normalizes_reply_mentions_and_attachment_metadata():
    action_client = RecordingActionClient()
    payload = group_event(
        segments=[
            {"type": "reply", "data": {"id": "88"}},
            {"type": "text", "data": {"text": "/ai ask "}},
            {"type": "at", "data": {"qq": "123", "name": "Bob"}},
            {
                "type": "image",
                "data": {
                    "file": "photo.jpg",
                    "url": "https://example.test/photo.jpg",
                    "file_size": "321",
                    "summary": "[image]",
                },
            },
        ]
    )

    message = OneBotMessage.from_payload(payload, action_client=action_client)

    assert message.id == 101
    assert message.chat_id == 700
    assert message.sender_id == 42
    assert message.reply_to_msg_id == 88
    assert message.raw_text == "/ai ask @Bob"
    assert message.date == datetime.fromtimestamp(1_700_000_000, UTC)
    assert message.out is False
    assert message.file is not None
    assert message.file.name == "photo.jpg"
    assert message.file.size == 321
    assert not hasattr(message.file, "data")


@pytest.mark.asyncio
async def test_onebot_attachment_bytes_are_fetched_on_demand_only():
    action_client = RecordingActionClient(responses=[{"base64": "aGVsbG8="}])
    message = OneBotMessage.from_payload(
        group_event(
            segments=[
                {
                    "type": "image",
                    "data": {"file": "opaque-image-token"},
                }
            ]
        ),
        action_client=action_client,
    )

    assert message.file is not None
    assert await message.download_media(file=bytes) == b"hello"
    assert action_client.calls[0][0] == "get_file"
    assert not hasattr(message.file, "data")


@pytest.mark.asyncio
async def test_onebot_unknown_size_image_uses_its_bounded_downloader() -> None:
    output = BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(output, format="PNG")
    action_client = RecordingActionClient(
        responses=[{"base64": base64.b64encode(output.getvalue()).decode("ascii")}]
    )
    message = OneBotMessage.from_payload(
        group_event(
            segments=[
                {"type": "text", "data": {"text": "/ai describe"}},
                {"type": "image", "data": {"file": "opaque-image-token"}},
            ]
        ),
        action_client=action_client,
    )
    gateway = RecordingAttachmentGateway()

    result = await ChatAttachmentDescriber(
        gateway,
        allow_unknown_size=True,
    ).describe(message)

    assert result is not None
    assert result.model_image is not None
    assert result.model_image.mime_type == "image/jpeg"
    assert len(gateway.requests) == 1
    assert action_client.calls[0][0] == "get_file"


def test_onebot_private_self_message_uses_target_as_conversation_scope():
    message = OneBotMessage.from_payload(
        private_event(
            sender_id=99,
            target_id=42,
            post_type="message_sent",
        ),
        action_client=RecordingActionClient(),
    )

    assert message.chat_id == -42
    assert message.sender_id == 99
    assert message.out is True


@pytest.mark.asyncio
async def test_onebot_plugin_rejects_events_for_another_account() -> None:
    payload = group_event()
    payload["self_id"] = 100
    plugin = object.__new__(OneBotAI)
    plugin._runtime = SimpleNamespace(self_id=99)
    plugin._bridge = RecordingActionClient()
    plugin._directory = OneBotDirectory()
    plugin._handler = object()
    plugin.logger = logging.getLogger("test-onebot-account-boundary")

    await plugin._on_event(payload)



@pytest.mark.asyncio
async def test_onebot_plugin_persists_only_candidate_identity_and_origin() -> None:
    class Workflow:
        def __init__(self):
            self.accepted = []

        async def accept(self, **work):
            self.accepted.append(work)

    class Transport:
        async def classify_origin(self, _message):
            return MessageOrigin.INCOMING

    plugin = object.__new__(OneBotAI)
    plugin._runtime = SimpleNamespace(self_id=99)
    plugin._ops_settings = SimpleNamespace(instance_id="qq-test")
    plugin._bridge = RecordingActionClient()
    plugin._directory = OneBotDirectory()
    plugin._handler = object()
    plugin._transport = Transport()
    plugin._ai_workflow = Workflow()
    plugin.logger = logging.getLogger("test-onebot-durable-accept")

    await plugin._on_event(group_event(text="ambient chat"))

    assert plugin._ai_workflow.accepted == []

    await plugin._on_event(group_event())

    assert plugin._ai_workflow.accepted == [
        {
            "cursor": 101,
            "chat_id": 700,
            "message_id": 101,
            "kind": "message",
            "attested_origin": MessageOrigin.INCOMING,
        }
    ]


@pytest.mark.asyncio
async def test_onebot_plugin_never_queues_generated_echo() -> None:
    class RejectingWorkflow:
        async def accept(self, **_kwargs):
            raise AssertionError("generated echo must not be persisted")

    class Transport:
        async def classify_origin(self, _message):
            return MessageOrigin.SIDEKICK_GENERATED

    plugin = object.__new__(OneBotAI)
    plugin._runtime = SimpleNamespace(self_id=99)
    plugin._ops_settings = SimpleNamespace(instance_id="qq-test")
    plugin._bridge = RecordingActionClient()
    plugin._directory = OneBotDirectory()
    plugin._handler = object()
    plugin._transport = Transport()
    plugin._ai_workflow = RejectingWorkflow()
    plugin.logger = logging.getLogger("test-onebot-generated-drop")

    await plugin._on_event(group_event(post_type="message_sent", sender_id=99))


@pytest.mark.asyncio
async def test_onebot_inbound_source_refetches_exact_message_and_keeps_origin() -> None:
    client = RecordingActionClient(responses=[group_event()])
    source = OneBotInboundMessageSource(
        client,
        self_id=99,
        directory=OneBotDirectory(),
    )
    work = SimpleNamespace(
        chat_id=700,
        message_id=101,
        attested_origin=MessageOrigin.MANUAL_OUTGOING,
    )

    revision = await source.fetch(work)
    message = await source.materialize(revision.payload)

    assert revision.version == "onebot:v1:101"
    assert revision.attested_origin is MessageOrigin.MANUAL_OUTGOING
    assert message is not None
    assert message.id == 101
    assert client.calls == [("get_msg", {"message_id": "101"}, None)]


def test_onebot_private_history_uses_explicit_peer_when_target_is_absent():
    payload = private_event(
        sender_id=99,
        target_id=42,
        post_type="message_sent",
    )
    payload.pop("target_id")

    message = OneBotMessage.from_payload(
        payload,
        action_client=RecordingActionClient(),
        private_peer_id=42,
    )

    assert message.chat_id == -42


@pytest.mark.parametrize(
    "patch",
    [
        {"message_id": "not-a-number"},
        {"message": "CQ string is not accepted"},
        {"message_type": "guild"},
        {"group_id": None},
    ],
)
def test_onebot_message_rejects_malformed_external_events(patch):
    payload = group_event()
    payload.update(patch)

    with pytest.raises(OneBotMessageError):
        OneBotMessage.from_payload(
            payload,
            action_client=RecordingActionClient(),
        )


@pytest.mark.asyncio
async def test_onebot_transport_defers_placeholder_and_sends_one_final_reply():
    action_client = RecordingActionClient(responses=[{"message_id": 502}])
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=action_client,
    )
    transport = OneBotChatTransport(action_client)

    sent = await transport.draft_reply(trigger)
    await transport.delete(sent)
    assert action_client.calls == []

    streamed = await transport.update(
        sent,
        "partial",
        presentation="agent",
        wait=False,
    )
    finalized = await transport.update(
        sent,
        "**Final**",
        presentation="agent",
        wait=True,
    )

    assert streamed is False
    assert finalized is True
    assert sent.id == 502
    assert sent.text == "Final"
    assert [call[0] for call in action_client.calls] == ["send_group_msg"]
    assert action_client.calls[0][1]["message"][1] == {
        "type": "text",
        "data": {"text": "Final"},
    }


@pytest.mark.asyncio
async def test_onebot_transport_replaces_placeholder_with_one_final_reply():
    action_client = RecordingActionClient(
        responses=[
            {"message_id": 501},
            {"message_id": 502},
            None,
        ]
    )
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=action_client,
    )
    transport = OneBotChatTransport(action_client)

    sent = await transport.reply(trigger, "**Thinking...**", presentation="plain")
    streamed = await transport.update(
        sent,
        "partial",
        presentation="agent",
        wait=False,
    )
    finalized = await transport.update(
        sent,
        (
            "**Final** with [the docs](https://example.com/docs), `code`, "
            "and 这是**重点**内容。"
        ),
        presentation="agent",
        wait=True,
    )

    assert streamed is False
    assert finalized is True
    assert sent.id == 502
    assert sent.text == (
        "Final with the docs (https://example.com/docs), code, "
        "and 这是重点内容。"
    )
    assert [call[0] for call in action_client.calls] == [
        "send_group_msg",
        "send_group_msg",
        "delete_msg",
    ]
    assert action_client.calls[0][1]["message"][0] == {
        "type": "reply",
        "data": {"id": "101"},
    }
    assert action_client.calls[0][1]["message"][1] == {
        "type": "text",
        "data": {"text": "**Thinking...**"},
    }
    assert action_client.calls[1][1]["message"][1] == {
        "type": "text",
        "data": {
            "text": (
                "Final with the docs (https://example.com/docs), code, "
                "and 这是重点内容。"
            )
        },
    }


@pytest.mark.asyncio
async def test_onebot_transport_retries_reply_lookup_timeout_without_reply():
    action_client = RecordingActionClient(
        responses=[
            OneBotActionError(
                "send_group_msg",
                1200,
                "invoke timeout, wrapperSession.getMsgService().getMsgsByMsgId",
            ),
            {"message_id": 502},
        ]
    )
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=action_client,
    )

    sent = await OneBotChatTransport(action_client).reply(
        trigger,
        "Final answer",
        presentation="agent",
    )

    assert sent.id == 502
    assert action_client.calls[1][1]["message"] == [
        {"type": "text", "data": {"text": "Final answer"}}
    ]


@pytest.mark.asyncio
async def test_onebot_transport_does_not_retry_send_timeout():
    action_client = RecordingActionClient(
        responses=[
            OneBotActionError(
                "send_group_msg",
                1200,
                "invoke timeout, wrapperSession.getMsgService().sendMsg",
            )
        ]
    )
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=action_client,
    )

    with pytest.raises(OneBotActionError, match="sendMsg"):
        await OneBotChatTransport(action_client).reply(
            trigger,
            "Final answer",
            presentation="agent",
        )

    assert len(action_client.calls) == 1


@pytest.mark.asyncio
async def test_onebot_transport_suppresses_echo_that_arrives_before_send_receipt():
    class RacingActionClient:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def call(self, action, params=None, *, timeout=None):
            assert action == "send_group_msg"
            self.started.set()
            await self.release.wait()
            return {"message_id": 501}

    client = RacingActionClient()
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=client,
    )
    transport = OneBotChatTransport(client)
    sending = asyncio.create_task(
        transport.reply(trigger, "/ai must not run", presentation="agent")
    )
    await client.started.wait()
    echoed = OneBotMessage.from_payload(
        group_event(
            message_id=501,
            sender_id=99,
            text="/ai must not run",
            post_type="message_sent",
            segments=[
                {"type": "reply", "data": {"id": str(trigger.id)}},
                {"type": "text", "data": {"text": "/ai must not run"}},
            ],
        ),
        action_client=client,
    )
    classification = asyncio.create_task(transport.classify_origin(echoed))
    await asyncio.sleep(0)

    client.release.set()

    await sending
    assert await classification is MessageOrigin.SIDEKICK_GENERATED


@pytest.mark.asyncio
async def test_onebot_confirmations_do_not_accumulate_when_self_echoes_are_off():
    action_client = RecordingActionClient(
        responses=[{"message_id": 501}, {"message_id": 502}]
    )
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=action_client,
    )
    transport = OneBotChatTransport(action_client)
    transport._generated_messages = GeneratedMessageTracker(max_confirmed=1)

    await transport.reply(trigger, "first", presentation="plain")
    await transport.reply(trigger, "second", presentation="plain")

    assert len(action_client.calls) == 2


@pytest.mark.asyncio
async def test_onebot_disconnected_send_does_not_quarantine_manual_messages():
    bridge = OneBotReverseWebSocket(token="secret", self_id=99)
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=bridge,
    )
    transport = OneBotChatTransport(bridge)

    with pytest.raises(ConnectionError, match="NapCat is not connected"):
        await transport.reply(trigger, "not dispatched", presentation="plain")

    manual = OneBotMessage.from_payload(
        group_event(
            message_id=777,
            sender_id=99,
            text="/ai manual request",
            post_type="message_sent",
        ),
        action_client=bridge,
    )
    assert await transport.classify_origin(manual) is MessageOrigin.MANUAL_OUTGOING


@pytest.mark.asyncio
async def test_onebot_transport_preserves_manual_outgoing_message_by_exact_id():
    action_client = RecordingActionClient(responses=[{"message_id": 501}])
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=action_client,
    )
    transport = OneBotChatTransport(action_client)
    await transport.reply(trigger, "generated", presentation="agent")
    manual = OneBotMessage.from_payload(
        group_event(
            message_id=777,
            sender_id=99,
            text="/ai manual request",
            post_type="message_sent",
        ),
        action_client=action_client,
    )

    assert await transport.classify_origin(manual) is MessageOrigin.MANUAL_OUTGOING


@pytest.mark.asyncio
async def test_onebot_transport_does_not_trust_cross_account_message_sent() -> None:
    message = OneBotMessage.from_payload(
        group_event(
            message_id=777,
            sender_id=42,
            text="/ai_prefix /forged",
            post_type="message_sent",
        ),
        action_client=RecordingActionClient(),
    )

    assert (
        await OneBotChatTransport(RecordingActionClient()).classify_origin(message)
        is MessageOrigin.INCOMING
    )


@pytest.mark.asyncio
async def test_onebot_transport_replies_with_an_inline_image(make_png) -> None:
    action_client = RecordingActionClient(responses=[{"message_id": 503}])
    trigger = OneBotMessage.from_payload(group_event(), action_client=action_client)
    attachment = OutboundAttachment(
        data=make_png(),
        filename="answer.png",
        mime_type="image/png",
        display_as="image",
    )

    result = await OneBotChatTransport(action_client).reply_attachment(
        trigger,
        attachment,
    )

    assert result is not None
    assert result.id == 503
    assert result.text is None
    action, params, timeout = action_client.calls[0]
    assert action == "send_group_msg"
    assert timeout == 120
    assert params == {
        "group_id": "700",
        "message": [
            {"type": "reply", "data": {"id": "101"}},
            {
                "type": "image",
                "data": {
                    "file": "base64://"
                    + base64.b64encode(attachment.data).decode("ascii")
                },
            },
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_kind", ("group", "private"))
async def test_onebot_transport_uploads_a_file(chat_kind) -> None:
    action_client = RecordingActionClient(responses=[{"file_id": "file-503"}])
    trigger = OneBotMessage.from_payload(
        group_event() if chat_kind == "group" else private_event(),
        action_client=action_client,
    )
    attachment = OutboundAttachment(
        data=b"report-bytes",
        filename="report.txt",
        mime_type="text/plain",
        display_as="file",
    )

    result = await OneBotChatTransport(action_client).reply_attachment(
        trigger,
        attachment,
    )

    assert result is None
    action, params, timeout = action_client.calls[0]
    target = {"group_id": "700"} if chat_kind == "group" else {"user_id": "42"}
    assert action == (
        "upload_group_file" if chat_kind == "group" else "upload_private_file"
    )
    assert timeout == 120
    assert params == {
        **target,
        "file": "base64://"
        + base64.b64encode(attachment.data).decode("ascii"),
        "name": "report.txt",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    ({"file_id": "file-503"}, {"file_id": None}, None),
)
async def test_onebot_file_upload_without_message_id_releases_chat(response) -> None:
    action_client = RecordingActionClient(responses=[response])
    trigger = OneBotMessage.from_payload(
        group_event(),
        action_client=action_client,
    )
    transport = OneBotChatTransport(action_client)
    attachment = OutboundAttachment(
        data=b"report-bytes",
        filename="report.txt",
        mime_type="text/plain",
        display_as="file",
    )

    assert await transport.reply_attachment(trigger, attachment) is None
    echoed = OneBotMessage.from_payload(
        group_event(
            message_id=503,
            sender_id=99,
            text="",
            post_type="message_sent",
            segments=[
                {
                    "type": "file",
                    "data": {"file": "file-503", "name": "report.txt"},
                }
            ],
        ),
        action_client=action_client,
    )

    assert await transport.classify_origin(echoed) is MessageOrigin.MANUAL_OUTGOING
    handler = AIConversationHandler(
        owner_id=99,
        responder=AIResponder(object(), transport=transport),  # type: ignore[arg-type]
        store=object(),  # type: ignore[arg-type]
        prompt_builder=PromptBuilder(identity_codec=QQ_IDENTITY_CODEC),
        transport=transport,
        identity_codec=QQ_IDENTITY_CODEC,
    )
    assert await handler.handle(echoed) is False

    manual = OneBotMessage.from_payload(
        group_event(
            message_id=504,
            sender_id=99,
            text="/ai manual request",
            post_type="message_sent",
        ),
        action_client=action_client,
    )
    assert await transport.classify_origin(manual) is MessageOrigin.MANUAL_OUTGOING


@pytest.mark.asyncio
async def test_onebot_unknown_file_upload_outcome_quarantines_chat() -> None:
    action_client = RecordingActionClient(responses=[ConnectionError("lost")])
    trigger = OneBotMessage.from_payload(group_event(), action_client=action_client)
    transport = OneBotChatTransport(action_client)
    attachment = OutboundAttachment(
        data=b"report-bytes",
        filename="report.txt",
        mime_type="text/plain",
        display_as="file",
    )

    with pytest.raises(ConnectionError, match="lost"):
        await transport.reply_attachment(trigger, attachment)

    manual = OneBotMessage.from_payload(
        group_event(
            message_id=504,
            sender_id=99,
            text="/ai manual request",
            post_type="message_sent",
        ),
        action_client=action_client,
    )
    assert await transport.classify_origin(manual) is MessageOrigin.INDETERMINATE


@pytest.mark.asyncio
async def test_onebot_transport_fetches_reply_and_deletes_by_action():
    action_client = RecordingActionClient(
        responses=[
            group_event(message_id=88, text="parent"),
            None,
        ]
    )
    message = OneBotMessage.from_payload(
        group_event(
            segments=[
                {"type": "reply", "data": {"id": "88"}},
                {"type": "text", "data": {"text": "/ai follow up"}},
            ]
        ),
        action_client=action_client,
    )
    transport = OneBotChatTransport(action_client)

    parent = await transport.get_reply(message)
    await transport.delete(message)

    assert parent is not None
    assert parent.id == 88
    assert [call[0] for call in action_client.calls] == ["get_msg", "delete_msg"]


@pytest.mark.asyncio
async def test_private_reply_lookup_supplies_conversation_peer_to_get_msg():
    parent = private_event(
        message_id=88,
        sender_id=99,
        target_id=42,
        text="parent",
        post_type="message_sent",
    )
    parent.pop("target_id")
    action_client = RecordingActionClient(responses=[parent])
    message = OneBotMessage.from_payload(
        {
            **private_event(
                sender_id=42,
                target_id=42,
                text="/ai follow up",
            ),
            "message": [
                {"type": "reply", "data": {"id": "88"}},
                {"type": "text", "data": {"text": "/ai follow up"}},
            ],
        },
        action_client=action_client,
    )

    fetched = await OneBotChatTransport(action_client).get_reply(message)

    assert fetched is not None
    assert fetched.chat_id == -42


@pytest.mark.asyncio
async def test_onebot_history_uses_transport_order_and_explicit_anchor():
    action_client = RecordingActionClient(
        responses=[
            {
                "messages": [
                    group_event(
                        message_id=2_000_000_000,
                        text="older",
                    ),
                    group_event(
                        message_id=3,
                        text="newer",
                    ),
                    group_event(
                        message_id=90,
                        text="Replied-to anchor",
                    ),
                ]
            }
        ]
    )
    trigger = OneBotMessage.from_payload(
        group_event(message_id=100, text="/ai2 summarize"),
        action_client=action_client,
    )
    anchor = OneBotMessage.from_payload(
        group_event(message_id=90, text="Replied-to anchor"),
        action_client=action_client,
    )

    messages = await OneBotHistorySource(action_client).fetch_recent(
        trigger,
        before=anchor,
        limit=2,
    )

    assert [message.id for message in messages] == [2_000_000_000, 3]
    action, params, _ = action_client.calls[0]
    assert action == "get_group_msg_history"
    assert params["group_id"] == "700"
    assert params["message_seq"] == "90"
    assert params["reverse_order"] is True


@pytest.mark.asyncio
async def test_onebot_history_resumes_from_latest_window_when_cursor_is_gone():
    action_client = RecordingActionClient(
        responses=[
            OneBotActionError("get_msg", 1404, "message not found"),
            {
                "messages": [
                    group_event(message_id=501, text="available one"),
                    group_event(message_id=502, text="available two"),
                ]
            },
        ]
    )

    messages = await OneBotHistorySource(action_client).fetch_after(
        700,
        after_message_id=400,
        until=datetime.max.replace(tzinfo=UTC),
        limit=2,
    )

    assert [message.id for message in messages] == [501, 502]


@pytest.mark.asyncio
async def test_onebot_history_pages_past_unsettled_messages_for_window():
    cutoff = datetime.fromtimestamp(1_700_000_000, UTC)
    action_client = RecordingActionClient(
        responses=[
            {
                "messages": [
                    group_event(
                        message_id=501,
                        text="unsettled one",
                        timestamp=1_700_000_010,
                    ),
                    group_event(
                        message_id=502,
                        text="unsettled two",
                        timestamp=1_700_000_011,
                    ),
                ]
            },
            {
                "messages": [
                    group_event(
                        message_id=500,
                        text="settled",
                        timestamp=1_699_999_999,
                    ),
                    group_event(
                        message_id=501,
                        text="unsettled one",
                        timestamp=1_700_000_010,
                    ),
                ]
            },
        ]
    )

    messages = await OneBotHistorySource(action_client).fetch_window(
        700,
        since=datetime.fromtimestamp(1_699_999_000, UTC),
        until=cutoff,
        limit=1,
    )

    assert [message.id for message in messages] == [500]
    assert action_client.calls[1][1]["message_seq"] == "501"
    assert action_client.calls[1][1]["reverse_order"] is True


@pytest.mark.asyncio
async def test_onebot_identity_and_mentions_use_display_labels_when_available():
    message = OneBotMessage.from_payload(
        group_event(
            segments=[
                {"type": "text", "data": {"text": "hello "}},
                {"type": "at", "data": {"qq": "123", "name": "Bob"}},
            ]
        ),
        action_client=RecordingActionClient(),
    )

    identity = await OneBotMessageIdentityResolver().resolve(message)
    mentions = await OneBotMessageMentionResolver().resolve(message)

    assert identity.subject_id == "qq:user:42"
    assert identity.subject_display_name == "Alice Card"
    assert identity.scope_display_name == "Dog Food Filter"
    assert [(item.user_id, item.display_name) for item in mentions] == [(123, "Bob")]


@pytest.mark.asyncio
async def test_reverse_websocket_authenticates_and_correlates_actions():
    seen = []
    received = asyncio.Event()

    async def on_event(payload):
        seen.append(payload)
        received.set()

    bridge = OneBotReverseWebSocket(
        token="secret",
        self_id=99,
        event_handler=on_event,
    )
    async with TestServer(bridge.application) as server:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as rejected:
                await session.ws_connect(
                    server.make_url("/onebot"),
                    headers={"Authorization": "Bearer wrong", "X-Self-ID": "99"},
                )
            assert rejected.value.status == 401

            websocket = await session.ws_connect(
                server.make_url("/onebot"),
                headers={
                    "Authorization": "Bearer secret",
                    "X-Self-ID": "99",
                },
            )
            await websocket.send_json(group_event())
            await asyncio.wait_for(received.wait(), timeout=1)

            pending = asyncio.create_task(bridge.call("get_status", {}, timeout=1))
            action = await websocket.receive_json(timeout=1)
            await websocket.send_json(
                {
                    "status": "ok",
                    "retcode": 0,
                    "data": {"online": True},
                    "echo": action["echo"],
                }
            )

            assert await pending == {"online": True}
            assert seen[0]["message_id"] == 101
            await websocket.close()
    await bridge.close()


@pytest.mark.asyncio
async def test_reverse_websocket_dispatches_events_concurrently():
    first_started = asyncio.Event()
    second_seen = asyncio.Event()
    release_first = asyncio.Event()

    async def on_event(payload):
        if payload["message_id"] == 101:
            first_started.set()
            await release_first.wait()
        else:
            second_seen.set()

    bridge = OneBotReverseWebSocket(
        token="secret",
        self_id=99,
        event_handler=on_event,
        event_concurrency=2,
    )
    async with TestServer(bridge.application) as server:
        async with aiohttp.ClientSession() as session:
            websocket = await session.ws_connect(
                server.make_url("/onebot"),
                headers={
                    "Authorization": "Bearer secret",
                    "X-Self-ID": "99",
                },
            )
            await websocket.send_json(group_event(message_id=101))
            await asyncio.wait_for(first_started.wait(), timeout=1)
            await websocket.send_json(group_event(message_id=102))
            await asyncio.wait_for(second_seen.wait(), timeout=1)
            release_first.set()
            await websocket.close()
    await bridge.close()


@pytest.mark.asyncio
async def test_reverse_websocket_surfaces_action_failures_without_payload_leaks():
    bridge = OneBotReverseWebSocket(token="secret", self_id=99)
    async with TestServer(bridge.application) as server:
        async with aiohttp.ClientSession() as session:
            websocket = await session.ws_connect(
                server.make_url("/onebot"),
                headers={
                    "Authorization": "Bearer secret",
                    "X-Self-ID": "99",
                },
            )
            pending = asyncio.create_task(
                bridge.call("get_msg", {"message_id": "1"}, timeout=1)
            )
            action = await websocket.receive_json(timeout=1)
            await websocket.send_json(
                {
                    "status": "failed",
                    "retcode": 1404,
                    "message": "message not found",
                    "data": {"private": "must not be copied into the error"},
                    "echo": action["echo"],
                }
            )

            with pytest.raises(OneBotActionError, match="message not found") as exc:
                await pending
            assert "private" not in str(exc.value)
            await websocket.close()
    await bridge.close()
