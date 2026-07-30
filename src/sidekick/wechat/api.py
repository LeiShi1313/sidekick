from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import json
import re
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp
from aiohttp import WSMsgType


API_VERSION = "v1alpha1"
API_VERSION_HEADER = "X-WeChat-Bridge-API-Version"
EVENT_SCHEMA_VERSION = "wechat-bridge/v1alpha1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_TEXT_BYTES = 4_095
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")


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
        status = _required_enum(payload, "status", {"unknown", "logged_in", "logged_out"})
        return cls(
            status=status,
            self_id=_required_id(payload, "selfId"),
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
            raise WeChatAPIContractError("WeChat connector is not logged in and connected")
        if self.content_redacted:
            raise WeChatAPIContractError(
                "WeChat event content is redacted; /ai commands cannot be observed"
            )


@dataclass(frozen=True, slots=True)
class WeChatCapabilities:
    receive_text: bool
    stable_inbound_message_ids: bool
    send_text: bool
    request_idempotency: bool
    outbound_stable_message_id: bool
    websocket: bool
    cursor: bool
    replay: bool
    durable_cursor: bool
    text_send_ready: bool
    connection_generation: int
    history: bool

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatCapabilities:
        if payload.get("apiVersion") != API_VERSION:
            raise WeChatAPIContractError("WeChat capabilities API version is unsupported")
        messages = _required_object(payload, "messages")
        events = _required_object(payload, "events")
        runtime = _required_object(payload, "runtime")
        sync = _required_object(payload, "sync")
        return cls(
            receive_text=_required_bool(messages, "receiveText"),
            stable_inbound_message_ids=_required_bool(
                messages,
                "stableInboundMessageIds",
            ),
            send_text=_required_bool(messages, "sendText"),
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
            connection_generation=_required_positive_int(
                runtime,
                "connectionGeneration",
            ),
            history=_required_bool(sync, "history"),
        )

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
            id=_required_id(payload, "id"),
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
            raise WeChatAPIContractError("WeChat chat snapshot is not complete and current")
        if self.snapshot.connection_generation != generation:
            raise WeChatAPIContractError("WeChat chat snapshot generation is stale")


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

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatConnectorMessage:
        message_id = _required_id(payload, "id")
        if not message_id.isascii() or not message_id.isdecimal() or int(message_id) < 1:
            raise WeChatAPIContractError("WeChat message id must be a decimal string")
        reply_id = _optional_id(payload, "replyToMessageId")
        if reply_id is not None and (
            not reply_id.isascii() or not reply_id.isdecimal() or int(reply_id) < 1
        ):
            raise WeChatAPIContractError(
                "WeChat replyToMessageId must be a decimal string"
            )
        sequence = payload.get("seq")
        if sequence is not None:
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                raise WeChatAPIContractError("WeChat message seq is invalid")
            sequence_text = str(sequence)
        else:
            sequence_text = None
        return cls(
            id=message_id,
            chat_id=_required_id(payload, "chatId"),
            direction=_required_enum(payload, "direction", {"in", "out"}),
            message_type=_required_text(payload, "messageType"),
            sender_id=_required_id(payload, "senderId"),
            reply_to_message_id=reply_id,
            content=_optional_raw_text(payload, "content") or "",
            content_redacted=_optional_bool(payload, "contentRedacted", False),
            timestamp=_required_nonnegative_int(payload, "timestamp"),
            source=_optional_text(payload, "source"),
            sequence=sequence_text,
        )


@dataclass(frozen=True, slots=True)
class WeChatMessageList:
    messages: tuple[WeChatConnectorMessage, ...]
    cursor: str

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatMessageList:
        rows = _required_array(payload, "data")
        return cls(
            messages=tuple(
                WeChatConnectorMessage.parse(_object(row, "message")) for row in rows
            ),
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


@dataclass(frozen=True, slots=True)
class WeChatSendOperation:
    request_id: str
    status: str
    message_id: str | None
    error_code: str | None
    to: str | None

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> WeChatSendOperation:
        message_id = _optional_id(payload, "messageId")
        if message_id is not None and (
            not message_id.isascii()
            or not message_id.isdecimal()
            or int(message_id) < 1
        ):
            raise WeChatAPIContractError("WeChat outbound message id is invalid")
        return cls(
            request_id=_required_id(payload, "requestId"),
            status=_required_enum(
                payload,
                "status",
                {"queued", "dispatching", "submitted", "failed", "unknown"},
            ),
            message_id=message_id,
            error_code=_optional_text(payload, "errorCode"),
            to=_optional_id(payload, "to"),
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
            params["chatId"] = _external_id(chat_id, "chat_id")
        payload = await self._request_json("GET", "/messages", params=params)
        return WeChatMessageList.parse(payload)

    async def send_text_and_wait(
        self,
        *,
        request_id: str,
        to: str,
        content: str,
    ) -> WeChatSendOperation:
        if REQUEST_ID_RE.fullmatch(request_id) is None:
            raise ValueError("WeChat request ID must be 1-128 URL-safe characters")
        target = _external_id(to, "to")
        if not isinstance(content, str) or not content:
            raise ValueError("WeChat text content cannot be empty")
        if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("WeChat text content exceeds 4095 UTF-8 bytes")
        payload = await self._request_json(
            "POST",
            "/messages/text",
            json_body={
                "requestId": request_id,
                "to": target,
                "content": content,
            },
            expected_statuses=frozenset({200, 202}),
        )
        operation = WeChatSendOperation.parse(payload)
        self._validate_send_identity(operation, request_id=request_id, to=target)
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
                to=target,
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
        expected_statuses: frozenset[int] = frozenset({200}),
    ) -> Mapping[str, Any]:
        session = self._get_session()
        async with session.request(
            method,
            f"{self.base_url}{path}",
            params=params,
            json=json_body,
            headers=self._headers,
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
                raise WeChatAPIContractError("WeChat connector returned malformed JSON") from exc
            payload = _object(decoded, "response")
            if response.status not in expected_statuses:
                error = payload.get("error")
                error_object = error if isinstance(error, dict) else {}
                code = error_object.get("code")
                message = error_object.get("message")
                raise WeChatAPIError(
                    response.status,
                    code if isinstance(code, str) and code else "HTTP_ERROR",
                    message if isinstance(message, str) and message else "request failed",
                )
            return payload

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
    return value


def _required_id(payload: Mapping[str, Any], field: str) -> str:
    return _external_id(payload.get(field), field)


def _optional_id(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    return None if value is None else _external_id(value, field)


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


def _required_enum(
    payload: Mapping[str, Any],
    field: str,
    values: set[str],
) -> str:
    value = _required_text(payload, field)
    if value not in values:
        raise WeChatAPIContractError(f"WeChat {field} has an unsupported value")
    return value
