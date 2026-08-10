from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import json
import re
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import aiohttp
from aiohttp import WSMsgType

from sidekick.chat.attachments import OutboundAttachment


API_VERSION = "v1alpha1"
API_VERSION_HEADER = "X-WeChat-Bridge-API-Version"
EVENT_SCHEMA_VERSION = "wechat-bridge/v1alpha1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_MEDIA_BYTES = 20 * 1024 * 1024
MAX_TEXT_BYTES = 4_095
MAX_SHARED_CHAT_HISTORY_TEXT_BYTES = 16 * 1024
MAX_NATIVE_MESSAGE_ID = "18446744073709551615"
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
CANONICAL_WECHAT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,159}")
MEDIA_ID_RE = re.compile(r"[0-9a-f]{32}")
MAX_OPAQUE_ID_BYTES = 4_096
IMAGE_MIME_TYPES = frozenset(
    {
        "image/bmp",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    }
)
SHARED_CHAT_HISTORY_KINDS = frozenset(
    {
        "text",
        "image",
        "voice",
        "video",
        "sticker",
        "file",
        "link",
        "location",
        "app",
        "system",
        "unknown",
    }
)
MAX_SHARED_CHAT_HISTORY_ITEMS = 256
MAX_PROJECTED_SHARED_CHAT_HISTORY_ITEMS = 100
MAX_SHARED_CHAT_HISTORY_BYTES = 48 * 1024
MAX_SHARED_CHAT_HISTORY_LABEL_CHARS = 240
MAX_SHARED_CHAT_HISTORY_CONTENT_CHARS = 4_096
MAX_GROUP_MEMBER_DELTA_ROWS = 100_000


class WeChatAPIError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        super().__init__(f"WeChat connector {code} ({status}): {message}")


class WeChatAPIContractError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WeChatSession:
    status: str
    self_id: str
    display_name: str | None
    hook_connected: bool
    connection_generation: int
    content_redacted: bool
    cursor: str

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatSession:
        status = _required_enum(
            payload, "status", {"unknown", "logged_in", "logged_out"}
        )
        return cls(
            status=status,
            self_id=_required_wechat_id(payload, "selfId"),
            display_name=_optional_text(payload, "displayName"),
            hook_connected=_required_bool(payload, "hookConnected"),
            connection_generation=_required_positive_int(
                payload,
                "connectionGeneration",
            ),
            content_redacted=_required_bool(payload, "contentRedacted"),
            cursor=_required_id(payload, "cursor"),
        )

    def require_current_login(self) -> None:
        if self.status != "logged_in" or not self.hook_connected:
            raise WeChatAPIContractError(
                "WeChat connector is not logged in and connected"
            )
        if self.content_redacted:
            raise WeChatAPIContractError(
                "WeChat event content is redacted; /ai commands cannot be observed"
            )


@dataclass(frozen=True, slots=True)
class WeChatCapabilities:
    receive_text: bool
    receive_shared_chat_history: bool
    stable_inbound_message_ids: bool
    send_text: bool
    send_reply: bool
    send_native_reply: bool
    request_idempotency: bool
    outbound_stable_message_id: bool
    websocket: bool
    cursor: bool
    replay: bool
    durable_cursor: bool
    text_send_ready: bool
    reply_send_ready: bool
    connection_generation: int
    history: bool
    inbound_image_download: bool
    request_original_image: bool

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatCapabilities:
        if payload.get("apiVersion") != API_VERSION:
            raise WeChatAPIContractError(
                "WeChat capabilities API version is unsupported"
            )
        messages = _required_object(payload, "messages")
        events = _required_object(payload, "events")
        media = _required_object(payload, "media")
        runtime = _required_object(payload, "runtime")
        sync = _required_object(payload, "sync")
        return cls(
            receive_text=_required_bool(messages, "receiveText"),
            receive_shared_chat_history=_required_bool(
                messages,
                "receiveSharedChatHistory",
            ),
            stable_inbound_message_ids=_required_bool(
                messages,
                "stableInboundMessageIds",
            ),
            send_text=_required_bool(messages, "sendText"),
            send_reply=_required_bool(messages, "sendReply"),
            send_native_reply=_required_bool(messages, "sendNativeReply"),
            request_idempotency=_required_bool(messages, "requestIdempotency"),
            outbound_stable_message_id=_required_bool(
                messages,
                "outboundStableMessageId",
            ),
            websocket=_required_bool(events, "websocket"),
            cursor=_required_bool(events, "cursor"),
            replay=_required_bool(events, "replay"),
            durable_cursor=_required_bool(events, "durableCursor"),
            text_send_ready=_required_bool(runtime, "textSendReady"),
            reply_send_ready=_required_bool(runtime, "replySendReady"),
            connection_generation=_required_positive_int(
                runtime,
                "connectionGeneration",
            ),
            history=_required_bool(sync, "history"),
            inbound_image_download=_required_bool(
                media,
                "inboundImageDownload",
            ),
            request_original_image=_required_bool(media, "requestOriginalImage"),
        )

    @property
    def native_reply_ready(self) -> bool:
        return self.send_reply and self.send_native_reply and self.reply_send_ready

    def require_ai_channel(self) -> None:
        required = {
            "messages.receiveText": self.receive_text,
            "messages.stableInboundMessageIds": self.stable_inbound_message_ids,
            "messages.sendText": self.send_text,
            "messages.requestIdempotency": self.request_idempotency,
            "messages.outboundStableMessageId": self.outbound_stable_message_id,
            "events.websocket": self.websocket,
            "events.cursor": self.cursor,
            "events.replay": self.replay,
            "events.durableCursor": self.durable_cursor,
            "runtime.textSendReady": self.text_send_ready,
        }
        unavailable = [name for name, ready in required.items() if not ready]
        if unavailable:
            raise WeChatAPIContractError(
                "WeChat AI channel requirements are unavailable: "
                + ", ".join(unavailable)
            )


@dataclass(frozen=True, slots=True)
class WeChatChat:
    id: str
    type: str
    display_name: str | None

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatChat:
        return cls(
            id=_required_wechat_id(payload, "id"),
            type=_required_enum(payload, "type", {"direct", "group"}),
            display_name=_optional_text(payload, "displayName"),
        )


@dataclass(frozen=True, slots=True)
class WeChatChatSnapshot:
    id: str
    complete: bool
    current: bool
    count: int
    cursor: str
    connection_generation: int

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatChatSnapshot:
        return cls(
            id=_required_id(payload, "id"),
            complete=_required_bool(payload, "complete"),
            current=_required_bool(payload, "current"),
            count=_required_nonnegative_int(payload, "count"),
            cursor=_required_id(payload, "cursor"),
            connection_generation=_required_positive_int(
                payload,
                "connectionGeneration",
            ),
        )


@dataclass(frozen=True, slots=True)
class WeChatChatList:
    chats: tuple[WeChatChat, ...]
    snapshot: WeChatChatSnapshot
    cursor: str

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatChatList:
        rows = _required_array(payload, "data")
        chats = tuple(WeChatChat.parse(_object(row, "chat")) for row in rows)
        snapshot = WeChatChatSnapshot.parse(_required_object(payload, "snapshot"))
        if snapshot.count != len(chats):
            raise WeChatAPIContractError("WeChat chat snapshot count is inconsistent")
        if len({chat.id for chat in chats}) != len(chats):
            raise WeChatAPIContractError("WeChat chat snapshot contains duplicate IDs")
        return cls(
            chats=chats,
            snapshot=snapshot,
            cursor=_required_id(payload, "cursor"),
        )

    def require_current(self, generation: int) -> None:
        if not self.snapshot.complete or not self.snapshot.current:
            raise WeChatAPIContractError(
                "WeChat chat snapshot is not complete and current"
            )
        if self.snapshot.connection_generation != generation:
            raise WeChatAPIContractError("WeChat chat snapshot generation is stale")


@dataclass(frozen=True, slots=True)
class WeChatUser:
    id: str
    display_name: str | None

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatUser:
        _required_bool(payload, "isPartial")
        return cls(
            id=_required_user_id(payload, "id"),
            display_name=_optional_text(payload, "displayName"),
        )


@dataclass(frozen=True, slots=True)
class WeChatUserList:
    users: tuple[WeChatUser, ...]
    cursor: str

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatUserList:
        if not _required_bool(payload, "isPartial"):
            raise WeChatAPIContractError("WeChat user directory must be partial")
        _required_text(payload, "source")
        users: list[WeChatUser] = []
        user_ids: set[str] = set()
        for value in _required_array(payload, "data"):
            row = _object(value, "user")
            try:
                user_id = _user_id(row.get("id"), "id")
            except WeChatAPIContractError:
                continue
            if user_id in user_ids:
                raise WeChatAPIContractError(
                    "WeChat user directory contains duplicate IDs"
                )
            user_ids.add(user_id)
            users.append(WeChatUser.parse(row))
        return cls(users=tuple(users), cursor=_required_id(payload, "cursor"))


@dataclass(frozen=True, slots=True)
class WeChatGroupMember:
    group_id: str
    user_id: str
    display_name: str | None
    nickname: str | None

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatGroupMember:
        _required_bool(payload, "isPartial")
        return cls(
            group_id=_required_group_id(payload, "groupId"),
            user_id=_required_user_id(payload, "userId"),
            display_name=_optional_text(payload, "displayName"),
            nickname=_optional_text(payload, "nickname"),
        )


@dataclass(frozen=True, slots=True)
class WeChatGroupMemberList:
    group_id: str
    members: tuple[WeChatGroupMember, ...]
    cursor: str
    snapshot_complete: bool
    snapshot_current: bool
    snapshot_connection_generation: int | None

    @classmethod
    def parse(
        cls,
        payload: Mapping[str, Any],
        *,
        group_id: str,
    ) -> WeChatGroupMemberList:
        target_group_id = _group_id(group_id, "group_id")
        if not _required_bool(payload, "isPartial"):
            raise WeChatAPIContractError(
                "WeChat group member directory must be partial"
            )
        _required_text(payload, "source")
        rows = _required_array(payload, "data")
        members: list[WeChatGroupMember] = []
        for value in rows:
            row = _object(value, "group member")
            try:
                _user_id(row.get("userId"), "userId")
            except WeChatAPIContractError:
                continue
            members.append(WeChatGroupMember.parse(row))
        if any(member.group_id != target_group_id for member in members):
            raise WeChatAPIContractError(
                "WeChat group member directory contains a different group"
            )
        if len({member.user_id for member in members}) != len(members):
            raise WeChatAPIContractError(
                "WeChat group member directory contains duplicate user IDs"
            )
        snapshot = _required_object(payload, "snapshot")
        if _required_group_id(snapshot, "groupId") != target_group_id:
            raise WeChatAPIContractError(
                "WeChat group member snapshot belongs to a different group"
            )
        complete = _required_bool(snapshot, "complete")
        current = _required_bool(snapshot, "current")
        count = _required_nonnegative_int(snapshot, "count")
        if current and not complete:
            raise WeChatAPIContractError(
                "WeChat current group member snapshot must be complete"
            )
        if complete and current and count != len(rows):
            raise WeChatAPIContractError(
                "WeChat group member snapshot count is inconsistent"
            )
        connection_generation = _optional_positive_int(
            snapshot,
            "connectionGeneration",
        )
        if current and connection_generation is None:
            raise WeChatAPIContractError(
                "WeChat current group member snapshot generation is missing"
            )
        return cls(
            group_id=target_group_id,
            members=tuple(members),
            cursor=_required_id(payload, "cursor"),
            snapshot_complete=complete,
            snapshot_current=current,
            snapshot_connection_generation=connection_generation,
        )


@dataclass(frozen=True, slots=True)
class WeChatSharedChatHistory:
    title: str
    item_count: int
    text: str

    @classmethod
    def parse(cls, value: Any) -> WeChatSharedChatHistory:
        payload = _shared_history_object(value, "object")
        title = _shared_history_label(payload.get("title"), allow_empty=False)
        item_count = _shared_history_nonnegative_int(
            payload.get("itemCount"),
            "item count",
        )
        rows = payload.get("items")
        if not isinstance(rows, list):
            raise _shared_history_error("items must be an array")
        truncated = payload.get("truncated", False)
        if not isinstance(truncated, bool):
            raise _shared_history_error("truncated must be a boolean")
        if (
            item_count < 1
            or item_count > MAX_SHARED_CHAT_HISTORY_ITEMS
            or not rows
            or len(rows) > MAX_PROJECTED_SHARED_CHAT_HISTORY_ITEMS
            or len(rows) > item_count
            or truncated != (len(rows) < item_count)
        ):
            raise _shared_history_error("item counts are inconsistent")

        total_bytes = _shared_history_utf8_size(title)
        formatted_items: list[tuple[str, ...]] = []
        for row in rows:
            item = _shared_history_object(row, "item")
            kind = _shared_history_kind(item.get("kind"))
            sender = _shared_history_label(
                item.get("senderName", ""),
                allow_empty=True,
            )
            content = _shared_history_text(item.get("content", ""))
            _shared_history_nonnegative_int(
                item.get("timestamp", 0),
                "item timestamp",
            )
            item_lines = [_format_shared_history_part(kind, sender, content)]
            total_bytes += sum(
                _shared_history_utf8_size(value)
                for value in (kind, sender, content)
            ) + 32
            if item.get("reply") is not None:
                reply = _shared_history_object(item["reply"], "reply")
                reply_kind = _shared_history_kind(reply.get("kind"))
                reply_sender = _shared_history_label(
                    reply.get("senderName", ""),
                    allow_empty=True,
                )
                reply_content = _shared_history_text(reply.get("content", ""))
                if not reply_sender and not reply_content:
                    raise _shared_history_error(
                        "reply must contain a sender or content"
                    )
                item_lines.append(
                    "  ↳ "
                    + _format_shared_history_part(
                        reply_kind,
                        reply_sender,
                        reply_content,
                    )
                )
                total_bytes += sum(
                    _shared_history_utf8_size(value)
                    for value in (reply_kind, reply_sender, reply_content)
                ) + 16
            if total_bytes > MAX_SHARED_CHAT_HISTORY_BYTES:
                raise _shared_history_error("payload is oversized")
            formatted_items.append(tuple(item_lines))

        lines = ["[Forwarded chat history]", title]
        rendered_items = 0
        for item_lines in formatted_items:
            remaining = item_count - (rendered_items + 1)
            candidate = [*lines, *item_lines]
            if remaining > 0:
                candidate.append(_shared_history_footer(remaining))
            if _shared_history_utf8_size("\n".join(candidate)) > (
                MAX_SHARED_CHAT_HISTORY_TEXT_BYTES
            ):
                break
            lines.extend(item_lines)
            rendered_items += 1
        remaining = item_count - rendered_items
        if remaining > 0:
            lines.append(_shared_history_footer(remaining))
        return cls(title=title, item_count=item_count, text="\n".join(lines))


@dataclass(frozen=True, slots=True)
class WeChatConnectorMessage:
    id: str
    chat_id: str
    direction: str
    message_type: str
    sender_id: str
    reply_to_message_id: str | None
    content: str
    content_redacted: bool
    timestamp: int
    source: str | None
    sequence: str | None
    media_id: str | None = None
    shared_chat_history: WeChatSharedChatHistory | None = None

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatConnectorMessage:
        message_id = _required_native_message_id(payload, "id")
        reply_id = _optional_native_message_id(payload, "replyToMessageId")
        message_type = _required_text(payload, "messageType")
        content_redacted = _optional_bool(payload, "contentRedacted", False)
        has_shared_history = "sharedChatHistory" in payload
        if message_type == "chat_history":
            if content_redacted:
                if has_shared_history:
                    raise _shared_history_error(
                        "redacted messages must omit the object"
                    )
                shared_chat_history = None
            else:
                if not has_shared_history:
                    raise _shared_history_error("object is required")
                shared_chat_history = WeChatSharedChatHistory.parse(
                    payload.get("sharedChatHistory")
                )
        else:
            if has_shared_history:
                raise _shared_history_error(
                    "object is only valid for chat_history messages"
                )
            shared_chat_history = None
        sequence = payload.get("seq")
        if sequence is not None:
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
            ):
                raise WeChatAPIContractError("WeChat message seq is invalid")
            sequence_text = str(sequence)
        else:
            sequence_text = None
        return cls(
            id=message_id,
            chat_id=_required_wechat_id(payload, "chatId"),
            direction=_required_enum(payload, "direction", {"in", "out"}),
            message_type=message_type,
            sender_id=_required_wechat_id(payload, "senderId"),
            reply_to_message_id=reply_id,
            content=_optional_raw_text(payload, "content") or "",
            content_redacted=content_redacted,
            timestamp=_required_nonnegative_int(payload, "timestamp"),
            source=_optional_text(payload, "source"),
            sequence=sequence_text,
            media_id=_optional_message_media_id(payload),
            shared_chat_history=shared_chat_history,
        )

    @property
    def display_content(self) -> str:
        if self.shared_chat_history is not None and not self.content_redacted:
            return self.shared_chat_history.text
        return self.content


@dataclass(frozen=True, slots=True)
class WeChatDownloadedImage:
    data: bytes
    mime_type: str
    variant: Literal["original", "preview"]


@dataclass(frozen=True, slots=True)
class _WeChatOriginalImage:
    request_id: str
    chat_id: str
    message_id: str
    media_id: str
    mime_type: str
    size: int
    download_url: str

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> _WeChatOriginalImage:
        _required_enum(payload, "status", {"available"})
        media = _required_object(payload, "media")
        _required_enum(media, "variant", {"original"})
        size = _required_positive_int(media, "size")
        if size > MAX_MEDIA_BYTES:
            raise WeChatAPIContractError(
                "WeChat original image metadata is oversized"
            )
        return cls(
            request_id=_required_id(payload, "requestId"),
            chat_id=_required_wechat_id(payload, "chatId"),
            message_id=_required_native_message_id(payload, "messageId"),
            media_id=_required_media_id(media),
            mime_type=_required_image_mime_type(media, "mimeType"),
            size=size,
            download_url=_required_text(media, "downloadUrl"),
        )


@dataclass(frozen=True, slots=True)
class WeChatMessageList:
    messages: tuple[WeChatConnectorMessage, ...]
    cursor: str

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatMessageList:
        rows = _required_array(payload, "data")
        messages: list[WeChatConnectorMessage] = []
        for row in rows:
            message = _object(row, "message")
            if _is_senderless_unsupported_message(message):
                continue
            messages.append(WeChatConnectorMessage.parse(message))
        return cls(
            messages=tuple(messages),
            cursor=_required_id(payload, "cursor"),
        )


@dataclass(frozen=True, slots=True)
class WeChatEvent:
    cursor: str
    name: str
    connection_generation: int | None
    payload: Mapping[str, Any]

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatEvent:
        if payload.get("schemaVersion") != EVENT_SCHEMA_VERSION:
            raise WeChatAPIContractError("WeChat event schema version is unsupported")
        generation = payload.get("connectionGeneration")
        if generation is not None and (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            raise WeChatAPIContractError("WeChat event generation is invalid")
        return cls(
            cursor=_required_id(payload, "cursor"),
            name=_required_text(payload, "event"),
            connection_generation=generation,
            payload=dict(payload),
        )

    def message(self) -> WeChatConnectorMessage:
        if self.name != "message":
            raise WeChatAPIContractError("WeChat event is not a message")
        return WeChatConnectorMessage.parse(self.payload)

    def is_senderless_unsupported_message(self) -> bool:
        return self.name == "message" and _is_senderless_unsupported_message(
            self.payload
        )

    def removed_message(self) -> tuple[str, str]:
        if self.name != "message_remove" or self.payload.get("status") != "recalled":
            raise WeChatAPIContractError("Malformed WeChat message removal")
        return (
            _required_wechat_id(self.payload, "chatId"),
            _required_native_message_id(self.payload, "id"),
        )

    def changed_user_id(self) -> str:
        if self.name != "user_profile" or self.payload.get("status") != "changed":
            raise WeChatAPIContractError("Malformed WeChat user profile event")
        _required_id(self.payload, "id")
        return _required_user_id(self.payload, "userId")

    def invalidated_group_id(self) -> str | None:
        if self.name == "group_member_directory":
            if self.connection_generation is None:
                raise WeChatAPIContractError(
                    "WeChat group member directory generation is missing"
                )
            _required_enum(self.payload, "status", {"changed"})
            _required_enum(
                self.payload,
                "source",
                {"wechat+localdb-contact"},
            )
            _required_id(self.payload, "id")
            return _required_group_id(self.payload, "groupId")
        if self.name == "group_member_snapshot":
            status = _required_enum(self.payload, "status", {"begin", "end"})
            group_id = _required_group_id(self.payload, "groupId")
            return group_id if status == "end" else None
        if self.name == "group_member":
            group_id = _required_group_id(self.payload, "groupId")
            raw = _required_object(self.payload, "raw")
            mode = _required_enum(raw, "mode", {"cache_snapshot", "delta"})
            if mode == "cache_snapshot":
                return None
            _required_id(raw, "deltaId")
            response_count = _required_bounded_decimal(
                raw,
                "responseCount",
                minimum=1,
                maximum=MAX_GROUP_MEMBER_DELTA_ROWS,
            )
            delta_index = _required_bounded_decimal(
                raw,
                "deltaIndex",
                minimum=0,
                maximum=response_count - 1,
            )
            return group_id if delta_index == response_count - 1 else None
        raise WeChatAPIContractError("Event is not a WeChat group member event")


@dataclass(frozen=True, slots=True)
class WeChatSendOperation:
    request_id: str
    status: str
    message_id: str | None
    error_code: str | None
    to: str | None

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatSendOperation:
        return cls(
            request_id=_required_id(payload, "requestId"),
            status=_required_enum(
                payload,
                "status",
                {"queued", "dispatching", "submitted", "failed", "unknown"},
            ),
            message_id=_optional_native_message_id(payload, "messageId"),
            error_code=_optional_text(payload, "errorCode"),
            to=_optional_wechat_id(payload, "to"),
        )


class WeChatSendError(RuntimeError):
    def __init__(self, operation: WeChatSendOperation, message: str):
        self.operation = operation
        super().__init__(message)


class WeChatSendFailed(WeChatSendError):
    pass


class WeChatSendOutcomeUnknown(WeChatSendError):
    pass


class WeChatConnectorClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout: float = 30,
        send_poll_interval: float = 0.25,
        send_settle_timeout: float = 30,
    ):
        self.base_url = _normalize_base_url(base_url)
        if timeout <= 0 or send_poll_interval < 0 or send_settle_timeout <= 0:
            raise ValueError("WeChat connector timeouts must be positive")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._send_poll_interval = send_poll_interval
        self._send_settle_timeout = send_settle_timeout
        self._session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def get_session(self) -> WeChatSession:
        payload = await self._request_json("GET", "/session")
        return WeChatSession.parse(payload)

    async def get_capabilities(self) -> WeChatCapabilities:
        payload = await self._request_json("GET", "/capabilities")
        return WeChatCapabilities.parse(payload)

    async def get_chats(self) -> WeChatChatList:
        payload = await self._request_json("GET", "/chats")
        return WeChatChatList.parse(payload)

    async def get_users(self) -> WeChatUserList:
        payload = await self._request_json("GET", "/users")
        return WeChatUserList.parse(payload)

    async def get_user(self, user_id: str) -> WeChatUser | None:
        target_user_id = _user_id(user_id, "user_id")
        try:
            payload = await self._request_json(
                "GET",
                f"/users/{quote(target_user_id, safe='')}",
            )
        except WeChatAPIError as exc:
            if exc.status == 404 and exc.code == "NOT_FOUND":
                return None
            raise
        user = WeChatUser.parse(payload)
        if user.id != target_user_id:
            raise WeChatAPIContractError(
                "WeChat user response returned a different user"
            )
        return user

    async def get_group_members(self, group_id: str) -> WeChatGroupMemberList:
        target_group_id = _group_id(group_id, "group_id")
        payload = await self._request_json(
            "GET",
            f"/groups/{quote(target_group_id, safe='')}/members",
        )
        return WeChatGroupMemberList.parse(payload, group_id=target_group_id)

    async def get_messages(
        self,
        *,
        chat_id: str | None = None,
        limit: int = 100,
    ) -> WeChatMessageList:
        if not 1 <= limit <= 1_000:
            raise ValueError("WeChat message limit must be between 1 and 1000")
        params = {"limit": str(limit)}
        if chat_id is not None:
            params["chatId"] = _canonical_wechat_id(chat_id, "chat_id")
        payload = await self._request_json("GET", "/messages", params=params)
        return WeChatMessageList.parse(payload)

    async def download_original_image(
        self,
        *,
        request_id: str,
        chat_id: str,
        message_id: str,
        media_id: str,
    ) -> WeChatDownloadedImage:
        # Connector contract: exact tuple, separate original variant, fixed URL.
        # https://github.com/LeiShi1313/wechat-linux-bridge/blob/cc208c06d41bd3c3d1ec8e60bf22b58d2300099d/docs/wechat-connector-api.md#L2117-L2173
        if REQUEST_ID_RE.fullmatch(request_id) is None:
            raise ValueError("WeChat request ID must be 1-128 URL-safe characters")
        target_chat = _canonical_wechat_id(chat_id, "chat_id")
        target_message = _native_message_id(
            _external_id(message_id, "message_id"),
            "message_id",
        )
        target_media = _media_id(media_id, "media_id")
        path = f"/media/{target_media}/original"
        payload = await self._request_json(
            "POST",
            path,
            json_body={
                "requestId": request_id,
                "chatId": target_chat,
                "messageId": target_message,
            },
            allow_redirects=False,
        )
        original = _WeChatOriginalImage.parse(payload)
        if original.request_id != request_id:
            raise WeChatAPIContractError(
                "WeChat original image returned a different request ID"
            )
        if original.chat_id != target_chat or original.message_id != target_message:
            raise WeChatAPIContractError(
                "WeChat original image returned a different message identity"
            )
        if original.media_id != target_media or original.download_url != path:
            raise WeChatAPIContractError(
                "WeChat original image returned a different media identity"
            )
        return await self._request_image(
            path,
            variant="original",
            expected_mime_type=original.mime_type,
            expected_size=original.size,
        )

    async def download_image_preview(
        self,
        *,
        media_id: str,
    ) -> WeChatDownloadedImage:
        target_media = _media_id(media_id, "media_id")
        return await self._request_image(
            f"/media/{target_media}",
            variant="preview",
        )

    async def send_text_and_wait(
        self,
        *,
        request_id: str,
        to: str,
        content: str,
        reply_to_message_id: str | None,
    ) -> WeChatSendOperation:
        if REQUEST_ID_RE.fullmatch(request_id) is None:
            raise ValueError("WeChat request ID must be 1-128 URL-safe characters")
        target = _canonical_wechat_id(to, "to")
        if not isinstance(content, str) or not content:
            raise ValueError("WeChat text content cannot be empty")
        if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("WeChat text content exceeds 4095 UTF-8 bytes")
        body = {
            "requestId": request_id,
            "to": target,
            "content": content,
        }
        if reply_to_message_id is not None:
            body["replyToMessageId"] = _native_message_id(
                _external_id(reply_to_message_id, "reply_to_message_id"),
                "reply_to_message_id",
            )
        payload = await self._request_json(
            "POST",
            "/messages/text",
            json_body=body,
            expected_statuses=frozenset({200, 202}),
        )
        operation = WeChatSendOperation.parse(payload)
        self._validate_send_identity(operation, request_id=request_id, to=target)
        return await self._wait_for_send(operation, request_id=request_id, to=target)

    async def send_attachment_and_wait(
        self,
        *,
        request_id: str,
        to: str,
        attachment: OutboundAttachment,
    ) -> WeChatSendOperation:
        if REQUEST_ID_RE.fullmatch(request_id) is None:
            raise ValueError("WeChat request ID must be 1-128 URL-safe characters")
        target = _canonical_wechat_id(to, "to")
        form = aiohttp.FormData()
        form.add_field("requestId", request_id)
        form.add_field("to", target)
        form.add_field(
            "file",
            attachment.data,
            filename=attachment.filename,
            content_type=attachment.mime_type,
        )
        payload = await self._request_json(
            "POST",
            f"/messages/{attachment.display_as}",
            multipart_body=form,
            expected_statuses=frozenset({200, 202}),
        )
        operation = WeChatSendOperation.parse(payload)
        self._validate_send_identity(operation, request_id=request_id, to=target)
        return await self._wait_for_send(operation, request_id=request_id, to=target)

    async def reconcile_send_and_wait(
        self,
        *,
        request_id: str,
        to: str,
    ) -> WeChatSendOperation:
        if REQUEST_ID_RE.fullmatch(request_id) is None:
            raise ValueError("WeChat request ID must be 1-128 URL-safe characters")
        target = _canonical_wechat_id(to, "to")
        payload = await self._request_json(
            "GET",
            f"/sends/{quote(request_id, safe='')}",
        )
        operation = WeChatSendOperation.parse(payload)
        self._validate_send_identity(
            operation,
            request_id=request_id,
            to=target,
        )
        return await self._wait_for_send(
            operation,
            request_id=request_id,
            to=target,
        )

    async def _wait_for_send(
        self,
        operation: WeChatSendOperation,
        *,
        request_id: str,
        to: str,
    ) -> WeChatSendOperation:
        deadline = asyncio.get_running_loop().time() + self._send_settle_timeout
        while operation.status in {"queued", "dispatching"}:
            if asyncio.get_running_loop().time() >= deadline:
                raise WeChatSendOutcomeUnknown(
                    operation,
                    "WeChat send settlement timed out; reuse the original request ID",
                )
            await asyncio.sleep(self._send_poll_interval)
            payload = await self._request_json(
                "GET",
                f"/sends/{quote(request_id, safe='')}",
            )
            operation = WeChatSendOperation.parse(payload)
            self._validate_send_identity(
                operation,
                request_id=request_id,
                to=to,
            )
        if operation.status == "submitted":
            if operation.message_id is None:
                raise WeChatAPIContractError(
                    "WeChat submitted send has no stable outbound message ID"
                )
            return operation
        if operation.status == "failed":
            raise WeChatSendFailed(
                operation,
                "WeChat send failed"
                + (f" ({operation.error_code})" if operation.error_code else ""),
            )
        raise WeChatSendOutcomeUnknown(
            operation,
            "WeChat send outcome is unknown; never retry under a new request ID",
        )

    async def events(self, *, after: str) -> AsyncIterator[WeChatEvent]:
        cursor = _external_id(after, "after cursor")
        session = self._get_session()
        async with session.ws_connect(
            f"{self.base_url}/events",
            params={"after": cursor},
            headers=self._headers,
            heartbeat=45,
            max_msg_size=1 * 1024 * 1024,
            compress=0,
        ) as websocket:
            async for incoming in websocket:
                if incoming.type == WSMsgType.TEXT:
                    try:
                        decoded = json.loads(incoming.data)
                    except json.JSONDecodeError as exc:
                        raise WeChatAPIContractError(
                            "WeChat event stream returned malformed JSON"
                        ) from exc
                    yield WeChatEvent.parse(_object(decoded, "event"))
                    continue
                if incoming.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                    break
                if incoming.type == WSMsgType.ERROR:
                    error = websocket.exception()
                    raise ConnectionError("WeChat event stream failed") from error
                raise WeChatAPIContractError(
                    "WeChat event stream returned an unsupported frame"
                )

    @staticmethod
    def _validate_send_identity(
        operation: WeChatSendOperation,
        *,
        request_id: str,
        to: str,
    ) -> None:
        if operation.request_id != request_id:
            raise WeChatAPIContractError("WeChat send returned a different request ID")
        if operation.to is not None and operation.to != to:
            raise WeChatAPIContractError("WeChat send returned a different target")

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        multipart_body: aiohttp.FormData | None = None,
        expected_statuses: frozenset[int] = frozenset({200}),
        allow_redirects: bool = True,
    ) -> Mapping[str, Any]:
        if json_body is not None and multipart_body is not None:
            raise ValueError("WeChat request cannot combine JSON and multipart bodies")
        session = self._get_session()
        async with session.request(
            method,
            f"{self.base_url}{path}",
            params=params,
            json=json_body,
            data=multipart_body,
            headers=self._headers,
            allow_redirects=allow_redirects,
        ) as response:
            body = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                body.extend(chunk)
                if len(body) > MAX_JSON_BYTES:
                    raise WeChatAPIContractError("WeChat connector JSON is oversized")
            raw = bytes(body)
            if response.headers.get(API_VERSION_HEADER) != API_VERSION:
                raise WeChatAPIContractError(
                    "WeChat connector API version header is missing or unsupported"
                )
            try:
                decoded = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WeChatAPIContractError(
                    "WeChat connector returned malformed JSON"
                ) from exc
            payload = _object(decoded, "response")
            if response.status not in expected_statuses:
                error = payload.get("error")
                error_object = error if isinstance(error, dict) else {}
                code = error_object.get("code")
                message = error_object.get("message")
                raise WeChatAPIError(
                    response.status,
                    code if isinstance(code, str) and code else "HTTP_ERROR",
                    message
                    if isinstance(message, str) and message
                    else "request failed",
                )
            return payload

    async def _request_image(
        self,
        path: str,
        *,
        variant: Literal["original", "preview"],
        expected_mime_type: str | None = None,
        expected_size: int | None = None,
    ) -> WeChatDownloadedImage:
        # Successful binary media intentionally has no API-version header.
        # https://github.com/LeiShi1313/wechat-linux-bridge/blob/cc208c06d41bd3c3d1ec8e60bf22b58d2300099d/docs/wechat-connector-api.md#L1052-L1063
        session = self._get_session()
        async with session.get(
            f"{self.base_url}{path}",
            headers=self._headers,
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                await self._raise_image_response_error(response)
            if (
                response.content_length is not None
                and response.content_length > MAX_MEDIA_BYTES
            ):
                raise WeChatAPIContractError("WeChat connector image is oversized")
            if (
                expected_size is not None
                and response.content_length is not None
                and response.content_length != expected_size
            ):
                raise WeChatAPIContractError(
                    "WeChat connector image size differs from its metadata"
                )
            mime_type = response.content_type.lower()
            if mime_type not in IMAGE_MIME_TYPES:
                raise WeChatAPIContractError(
                    "WeChat connector returned an unsupported image type"
                )
            if expected_mime_type is not None and mime_type != expected_mime_type:
                raise WeChatAPIContractError(
                    "WeChat connector image type differs from its metadata"
                )
            body = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                body.extend(chunk)
                if len(body) > MAX_MEDIA_BYTES:
                    raise WeChatAPIContractError(
                        "WeChat connector image is oversized"
                    )
            data = bytes(body)
            if not data:
                raise WeChatAPIContractError("WeChat connector image is empty")
            if expected_size is not None and len(data) != expected_size:
                raise WeChatAPIContractError(
                    "WeChat connector image size differs from its metadata"
                )
            return WeChatDownloadedImage(
                data=data,
                mime_type=mime_type,
                variant=variant,
            )

    async def _raise_image_response_error(
        self,
        response: aiohttp.ClientResponse,
    ) -> None:
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            body.extend(chunk)
            if len(body) > MAX_JSON_BYTES:
                raise WeChatAPIContractError("WeChat connector JSON is oversized")
        if response.headers.get(API_VERSION_HEADER) != API_VERSION:
            raise WeChatAPIContractError(
                "WeChat connector API version header is missing or unsupported"
            )
        try:
            payload = _object(json.loads(bytes(body)), "response")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeChatAPIContractError(
                "WeChat connector returned malformed JSON"
            ) from exc
        error = payload.get("error")
        error_object = error if isinstance(error, dict) else {}
        code = error_object.get("code")
        message = error_object.get("message")
        raise WeChatAPIError(
            response.status,
            code if isinstance(code, str) and code else "HTTP_ERROR",
            message if isinstance(message, str) and message else "request failed",
        )

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session


def _normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("WeChat connector URL must be an HTTP(S) origin")
    return normalized


def _shared_history_error(detail: str) -> WeChatAPIContractError:
    return WeChatAPIContractError(f"WeChat shared chat history {detail}")


def _shared_history_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _shared_history_error(f"{field} must be an object")
    return value


def _shared_history_utf8_size(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise _shared_history_error("contains invalid Unicode") from exc


def _shared_history_label(value: Any, *, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise _shared_history_error("label must be a string")
    if not value:
        if allow_empty:
            return ""
        raise _shared_history_error("label must not be empty")
    if value.strip() != value or len(value) > MAX_SHARED_CHAT_HISTORY_LABEL_CHARS:
        raise _shared_history_error("label is invalid or oversized")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise _shared_history_error("label contains control characters")
    _shared_history_utf8_size(value)
    return value


def _shared_history_text(value: Any) -> str:
    if not isinstance(value, str):
        raise _shared_history_error("content must be a string")
    if not value:
        return ""
    if value.strip() != value or len(value) > MAX_SHARED_CHAT_HISTORY_CONTENT_CHARS:
        raise _shared_history_error("content is invalid or oversized")
    if any(
        (ord(character) < 0x20 and character not in "\n\r\t")
        or ord(character) == 0x7F
        for character in value
    ):
        raise _shared_history_error("content contains control characters")
    _shared_history_utf8_size(value)
    return value


def _shared_history_kind(value: Any) -> str:
    if not isinstance(value, str) or value not in SHARED_CHAT_HISTORY_KINDS:
        raise _shared_history_error("item kind is unsupported")
    return value


def _shared_history_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _shared_history_error(f"{field} must be a non-negative integer")
    return value


def _format_shared_history_part(kind: str, sender_name: str, content: str) -> str:
    prefixes = {
        "image": "[Image]",
        "voice": "[Voice]",
        "video": "[Video]",
        "sticker": "[Sticker]",
        "file": "[File]",
        "link": "[Link]",
        "location": "[Location]",
        "app": "[App]",
        "system": "[System]",
        "unknown": "[Unknown]",
    }
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    prefix = prefixes.get(kind, "")
    if prefix:
        body = f"{prefix} {normalized}" if normalized else prefix
    else:
        body = normalized or "[Text]"
    return f"{sender_name}: {body}" if sender_name else body


def _shared_history_footer(remaining: int) -> str:
    noun = "item" if remaining == 1 else "items"
    return f"… {remaining} more {noun} not included"


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise WeChatAPIContractError(f"WeChat {field} must be an object")
    return value


def _required_object(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    return _object(payload.get(field), field)


def _required_array(payload: Mapping[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise WeChatAPIContractError(f"WeChat {field} must be an array")
    return value


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value or value != value.strip():
        raise WeChatAPIContractError(f"WeChat {field} must be a non-empty string")
    return value


def _optional_raw_text(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise WeChatAPIContractError(f"WeChat {field} must be a string")
    return value


def _optional_text(payload: Mapping[str, Any], field: str) -> str | None:
    value = _optional_raw_text(payload, field)
    if value is None:
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _external_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise WeChatAPIContractError(f"WeChat {field} must be an opaque string ID")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise WeChatAPIContractError(f"WeChat {field} contains control characters")
    if len(value.encode("utf-8")) > MAX_OPAQUE_ID_BYTES:
        raise WeChatAPIContractError(f"WeChat {field} is oversized")
    return value


def _canonical_wechat_id(value: Any, field: str) -> str:
    external_id = _external_id(value, field)
    if (
        CANONICAL_WECHAT_ID_RE.fullmatch(external_id) is None
        or external_id == "@chatroom"
    ):
        raise WeChatAPIContractError(f"WeChat {field} must be a canonical WeChat ID")
    return external_id


def _required_id(payload: Mapping[str, Any], field: str) -> str:
    return _external_id(payload.get(field), field)


def _required_wechat_id(payload: Mapping[str, Any], field: str) -> str:
    return _canonical_wechat_id(payload.get(field), field)


def _user_id(value: Any, field: str) -> str:
    user_id = _canonical_wechat_id(value, field)
    if user_id.endswith("@chatroom"):
        raise WeChatAPIContractError(f"WeChat {field} must be a user ID")
    return user_id


def _required_user_id(payload: Mapping[str, Any], field: str) -> str:
    return _user_id(payload.get(field), field)


def _group_id(value: Any, field: str) -> str:
    group_id = _canonical_wechat_id(value, field)
    if not group_id.endswith("@chatroom"):
        raise WeChatAPIContractError(f"WeChat {field} must be a group ID")
    return group_id


def _required_group_id(payload: Mapping[str, Any], field: str) -> str:
    return _group_id(payload.get(field), field)


def _media_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or MEDIA_ID_RE.fullmatch(value) is None:
        raise WeChatAPIContractError(
            f"WeChat {field} must be exactly 32 lowercase hexadecimal characters"
        )
    return value


def _required_media_id(payload: Mapping[str, Any]) -> str:
    return _media_id(payload.get("id"), "media.id")


def _optional_message_media_id(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("media")
    if value is None:
        return None
    return _required_media_id(_object(value, "media"))


def _required_image_mime_type(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = _required_text(payload, field)
    if value not in IMAGE_MIME_TYPES:
        raise WeChatAPIContractError(
            f"WeChat {field} has an unsupported image type"
        )
    return value


def _is_senderless_unsupported_message(payload: Mapping[str, Any]) -> bool:
    message_type = payload.get("messageType")
    return (
        "senderId" not in payload
        and isinstance(message_type, str)
        and bool(message_type)
        and message_type == message_type.strip()
        and message_type != "text"
    )


def _optional_id(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    return None if value is None else _external_id(value, field)


def _optional_wechat_id(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    return None if value is None else _canonical_wechat_id(value, field)


def _required_native_message_id(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    return _native_message_id(_required_id(payload, field), field)


def _optional_native_message_id(
    payload: Mapping[str, Any],
    field: str,
) -> str | None:
    value = _optional_id(payload, field)
    return None if value is None else _native_message_id(value, field)


def _native_message_id(value: str, field: str) -> str:
    if (
        not value.isascii()
        or not value.isdecimal()
        or value[0] == "0"
        or len(value) > len(MAX_NATIVE_MESSAGE_ID)
        or (len(value) == len(MAX_NATIVE_MESSAGE_ID) and value > MAX_NATIVE_MESSAGE_ID)
    ):
        raise WeChatAPIContractError(
            f"WeChat {field} must be a canonical non-zero decimal message id"
        )
    return value


def _required_bool(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise WeChatAPIContractError(f"WeChat {field} must be a boolean")
    return value


def _optional_bool(payload: Mapping[str, Any], field: str, default: bool) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise WeChatAPIContractError(f"WeChat {field} must be a boolean")
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WeChatAPIContractError(f"WeChat {field} must be a non-negative integer")
    return value


def _required_positive_int(payload: Mapping[str, Any], field: str) -> int:
    value = _required_nonnegative_int(payload, field)
    if value < 1:
        raise WeChatAPIContractError(f"WeChat {field} must be positive")
    return value


def _optional_positive_int(
    payload: Mapping[str, Any],
    field: str,
) -> int | None:
    value = payload.get(field)
    return None if value is None else _required_positive_int(payload, field)


def _required_bounded_decimal(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(field)
    maximum_text = str(maximum)
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or (len(value) > 1 and value[0] == "0")
        or len(value) > len(maximum_text)
        or (len(value) == len(maximum_text) and value > maximum_text)
    ):
        raise WeChatAPIContractError(
            f"WeChat {field} must be a canonical decimal between "
            f"{minimum} and {maximum}"
        )
    parsed = int(value)
    if parsed < minimum:
        raise WeChatAPIContractError(
            f"WeChat {field} must be a canonical decimal between "
            f"{minimum} and {maximum}"
        )
    return parsed


def _required_enum(
    payload: Mapping[str, Any],
    field: str,
    values: set[str],
) -> str:
    value = _required_text(payload, field)
    if value not in values:
        raise WeChatAPIContractError(f"WeChat {field} has an unsupported value")
    return value
