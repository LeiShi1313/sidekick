from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Protocol
from urllib.parse import quote, unquote

from sidekick.ai import MemoryScopeTarget, MessageIdentity, MentionedUser, ReplyTarget
from sidekick.chat.formatting import markdown_to_plain_text
from sidekick.chat.identity import ExternalId, IdentityCodec
from sidekick.chat.transport import ChatPresentation, SentMessage
from sidekick.wechat.api import (
    MAX_TEXT_BYTES,
    WeChatSendFailed,
    WeChatSendOperation,
    WeChatSendOutcomeUnknown,
)
from sidekick.wechat.message import WeChatMessage
from sidekick.wechat.store import WeChatStateRepository


@dataclass(frozen=True, slots=True)
class WeChatIdentityCodec:
    source: str = "wechat"
    account_id: str | None = None

    def __post_init__(self) -> None:
        if self.source != "wechat":
            raise ValueError("WeChat identity source must be 'wechat'")
        if self.account_id is not None:
            _component(self.account_id)

    def actor_id(self, actor_id: ExternalId) -> str:
        return f"{self._identity_prefix()}user:{_component(actor_id)}"

    def scope_id(self, scope_id: ExternalId) -> str:
        return f"{self._identity_prefix()}chat:{_component(scope_id)}"

    def parse_scope_id(self, scope_id: str) -> ExternalId | None:
        prefix = f"{self._identity_prefix()}chat:"
        if not scope_id.startswith(prefix):
            return None
        encoded = scope_id.removeprefix(prefix)
        if not encoded:
            return None
        decoded = unquote(encoded)
        if not decoded or quote(decoded, safe="-_.~") != encoded:
            return None
        return decoded

    def message_source_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str:
        return (
            f"{self._identity_prefix()}message:{_component(scope_id)}:"
            f"{_component(message_id)}"
        )

    def parse_message_source_id(
        self,
        source_id: str,
    ) -> tuple[ExternalId, ExternalId] | None:
        prefix = f"{self._identity_prefix()}message:"
        if not source_id.startswith(prefix):
            return None
        parts = source_id.removeprefix(prefix).split(":")
        if len(parts) != 2:
            return None
        scope_id = _decoded_component(parts[0])
        message_id = _decoded_component(parts[1])
        if scope_id is None or message_id is None:
            return None
        return scope_id, message_id

    def thread_document_id(
        self,
        scope_id: ExternalId,
        root_message_id: ExternalId,
    ) -> str:
        return (
            f"{self._identity_prefix()}thread:{_component(scope_id)}:"
            f"{_component(root_message_id)}"
        )

    def revision_document_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str:
        return (
            f"{self._identity_prefix()}revision:{_component(scope_id)}:"
            f"{_component(message_id)}"
        )

    def _identity_prefix(self) -> str:
        if self.account_id is None:
            return "wechat:"
        return f"wechat:account:{_component(self.account_id)}:"


WECHAT_IDENTITY_CODEC: IdentityCodec = WeChatIdentityCodec()


class WeChatTextSender(Protocol):
    async def send_text_and_wait(
        self,
        *,
        request_id: str,
        to: str,
        content: str,
    ) -> WeChatSendOperation: ...


@dataclass(slots=True)
class WeChatSentMessage:
    id: str
    text: str | None
    trigger: WeChatMessage
    request_id: str
    sent: bool = False
    failed: bool = False
    uncertain: bool = False


class WeChatChatTransport:
    def __init__(
        self,
        client: WeChatTextSender,
        store: WeChatStateRepository,
        connector_key: str,
        *,
        logger: Any | None = None,
    ):
        self._client = client
        self._store = store
        self._connector_key = connector_key
        self._logger = logger

    async def draft_reply(self, message: Any) -> WeChatSentMessage:
        trigger = self._trigger(message)
        request_id = _request_id(trigger, "answer")
        return WeChatSentMessage(
            id=f"draft:{request_id}",
            text=None,
            trigger=trigger,
            request_id=request_id,
        )

    async def get_reply(self, message: Any) -> WeChatMessage | None:
        if not isinstance(message, WeChatMessage) or message.reply_to_msg_id is None:
            return None
        return await self._store.get_message(
            self._connector_key,
            message.chat_id,
            message.reply_to_msg_id,
        )

    async def reply(
        self,
        message: Any,
        text: str,
        *,
        presentation: ChatPresentation,
    ) -> SentMessage:
        trigger = self._trigger(message)
        rendered = self._render(text, presentation)
        sent = WeChatSentMessage(
            id=f"draft:{_request_id(trigger, 'reply')}",
            text=None,
            trigger=trigger,
            request_id=_request_id(trigger, "reply"),
        )
        await self._send(sent, rendered)
        return sent

    async def update(
        self,
        message: SentMessage,
        text: str,
        *,
        presentation: ChatPresentation,
        wait: bool,
    ) -> bool:
        if not isinstance(message, WeChatSentMessage):
            raise RuntimeError("WeChat transport requires a WeChat sent message")
        if not wait:
            return False
        rendered = self._render(text, presentation)
        if message.uncertain:
            message.text = rendered
            return True
        if message.sent:
            if message.text == rendered:
                return True
            content_fingerprint = sha256(rendered.encode("utf-8")).hexdigest()[:16]
            message.request_id = _request_id(
                message.trigger,
                f"update.{content_fingerprint}",
            )
            message.sent = False
        if message.failed:
            message.request_id = _request_id(message.trigger, "failure")
            message.failed = False
        await self._send(message, rendered)
        return True

    async def delete(self, _message: Any) -> None:
        # Deletion is deliberately not mapped to WeChat Recall. Recall has a
        # narrower capability and uncertainty contract than local cleanup.
        return None

    def is_outgoing(self, message: Any) -> bool:
        return bool(getattr(message, "is_outgoing", getattr(message, "out", False)))

    def is_group(self, message: Any) -> bool:
        return getattr(message, "chat_type", None) == "group"

    async def _send(self, message: WeChatSentMessage, text: str) -> None:
        try:
            operation = await self._client.send_text_and_wait(
                request_id=message.request_id,
                to=message.trigger.chat_id,
                content=text,
            )
        except WeChatSendOutcomeUnknown:
            message.uncertain = True
            message.text = text
            raise
        except WeChatSendFailed:
            message.failed = True
            message.text = text
            raise
        except Exception:
            # A transport failure can happen after the connector accepted the
            # request. Keep the original ID/payload reserved and never replace
            # it with an error message under that ID.
            message.uncertain = True
            message.text = text
            raise
        assert operation.message_id is not None
        await self._store.mark_processed_identity(
            self._connector_key,
            message.trigger.account_id,
            message.trigger.chat_id,
            operation.message_id,
        )
        message.id = operation.message_id
        message.text = text
        message.sent = True

    @staticmethod
    def _trigger(message: Any) -> WeChatMessage:
        if not isinstance(message, WeChatMessage):
            raise RuntimeError("WeChat transport requires a WeChat message")
        return message

    @staticmethod
    def _render(text: str, presentation: ChatPresentation) -> str:
        rendered = markdown_to_plain_text(text) if presentation == "agent" else text
        return _truncate_utf8(rendered, MAX_TEXT_BYTES)


class WeChatMessageIdentityResolver:
    def __init__(self, identity_codec: IdentityCodec = WECHAT_IDENTITY_CODEC):
        self._identity_codec = identity_codec

    async def resolve(self, message: ReplyTarget) -> MessageIdentity:
        return MessageIdentity(
            subject_id=(
                self._identity_codec.actor_id(message.sender_id)
                if message.sender_id is not None
                else None
            ),
            subject_display_name=getattr(message, "sender_display_name", None),
            scope_display_name=getattr(message, "scope_display_name", None),
            is_human=message.sender_id is not None,
        )


class WeChatMessageMentionResolver:
    async def resolve(self, _message: ReplyTarget) -> tuple[MentionedUser, ...]:
        return ()


class WeChatHistorySource:
    def __init__(self, store: WeChatStateRepository, connector_key: str):
        self._store = store
        self._connector_key = connector_key

    async def fetch_recent(
        self,
        trigger: ReplyTarget,
        *,
        before: ReplyTarget,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        if (
            not isinstance(trigger.chat_id, str)
            or before.chat_id != trigger.chat_id
            or not isinstance(before.id, str)
        ):
            return ()
        return await self._store.fetch_recent(
            self._connector_key,
            trigger.chat_id,
            before.id,
            limit,
        )

    async def fetch_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> WeChatMessage | None:
        return await self._store.get_message(
            self._connector_key,
            chat_id,
            message_id,
        )

    async def fetch_window(
        self,
        chat_id: str,
        *,
        since: datetime,
        until: datetime,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        return await self._store.fetch_memory_window(
            self._connector_key,
            chat_id,
            since=since,
            until=until,
            limit=limit,
        )

    async def fetch_after(
        self,
        chat_id: str,
        *,
        after_message_id: ExternalId,
        until: datetime,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        if isinstance(after_message_id, bool) or not isinstance(
            after_message_id,
            int,
        ):
            raise ValueError("WeChat continuous memory cursor is invalid")
        return await self._store.fetch_memory_after(
            self._connector_key,
            chat_id,
            after_memory_order=after_message_id,
            until=until,
            limit=limit,
        )


class WeChatMemoryScopeTargetResolver:
    def __init__(self, store: WeChatStateRepository, connector_key: str):
        self._store = store
        self._connector_key = connector_key

    async def resolve(
        self,
        target: str,
        *,
        include_latest_message: bool = False,
    ) -> MemoryScopeTarget:
        chat_id = target.strip()
        if not chat_id or chat_id != target:
            raise ValueError("WeChat target must be an exact stored chat ID")
        chat = await self._store.get_chat(self._connector_key, chat_id)
        if chat is None:
            raise ValueError("WeChat chat is not present in the local projection")
        latest_memory_cursor = (
            await self._store.get_latest_memory_cursor(
                self._connector_key,
                chat_id,
            )
            if include_latest_message
            else 0
        )
        return MemoryScopeTarget(
            chat_id=chat.id,
            display_name=chat.display_name,
            latest_message_id=latest_memory_cursor,
        )


def _component(value: ExternalId) -> str:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid WeChat IDs")
    normalized = str(value)
    if not normalized or normalized != normalized.strip():
        raise ValueError("WeChat IDs cannot be empty or padded")
    return quote(normalized, safe="-_.~")


def _decoded_component(value: str) -> str | None:
    decoded = unquote(value)
    if not decoded or quote(decoded, safe="-_.~") != value:
        return None
    return decoded


def _request_id(trigger: WeChatMessage, purpose: str) -> str:
    fingerprint = "\0".join(
        (trigger.account_id, trigger.chat_id, trigger.id, purpose)
    ).encode("utf-8")
    return f"sidekick.wechat.{purpose}.{sha256(fingerprint).hexdigest()[:40]}"


def _truncate_utf8(text: str, maximum: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum:
        return text
    prefix = encoded[: maximum - 3].decode("utf-8", errors="ignore")
    return f"{prefix}..."
