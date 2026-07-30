from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from sidekick.wechat.api import (
    WeChatAPIContractError,
    WeChatConnectorMessage,
    WeChatConnectorClient,
    WeChatEvent,
    WeChatSendOperation,
    WeChatSendOutcomeUnknown,
)


API_HEADERS = {"X-WeChat-Bridge-API-Version": "v1alpha1"}


def connector_message_payload(message_id: str) -> dict[str, object]:
    return {
        "id": message_id,
        "chatId": "filehelper",
        "direction": "in",
        "messageType": "text",
        "senderId": "wxid_alice",
        "content": "hello",
        "timestamp": 1_783_772_734,
    }


def json_response(payload: object, *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status, headers=API_HEADERS)


@pytest.mark.parametrize(
    "message_id",
    ("0", "01", "18446744073709551616"),
)
def test_wechat_client_rejects_noncanonical_message_ids(message_id: str) -> None:
    with pytest.raises(WeChatAPIContractError, match="message id"):
        WeChatConnectorMessage.parse(connector_message_payload(message_id))

    with pytest.raises(WeChatAPIContractError, match="message id"):
        WeChatSendOperation.parse(
            {
                "requestId": "request-1",
                "status": "submitted",
                "messageId": message_id,
            }
        )


def test_wechat_client_preserves_maximum_uint64_message_id_as_text() -> None:
    message_id = "18446744073709551615"

    message = WeChatConnectorMessage.parse(connector_message_payload(message_id))
    operation = WeChatSendOperation.parse(
        {
            "requestId": "request-1",
            "status": "submitted",
            "messageId": message_id,
        }
    )

    assert message.id == message_id
    assert operation.message_id == message_id


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("chatId", "room with spaces"),
        ("senderId", "wxid_alice/../../other"),
        ("chatId", f"a{'b' * 160}"),
    ),
)
def test_wechat_client_rejects_noncanonical_wechat_ids(
    field: str,
    value: str,
) -> None:
    payload = connector_message_payload("4159667620982040828")
    payload[field] = value

    with pytest.raises(WeChatAPIContractError, match=field):
        WeChatConnectorMessage.parse(payload)


def test_wechat_client_validates_recall_event_identity() -> None:
    malformed = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "11",
            "event": "message_remove",
            "connectionGeneration": 41,
            "status": "recalled",
            "chatId": "not a canonical/chat",
            "id": "4159667620982040828",
        }
    )

    with pytest.raises(WeChatAPIContractError, match="chatId"):
        malformed.removed_message()


@pytest.mark.asyncio
async def test_wechat_client_validates_bootstrap_and_waits_for_stable_send() -> None:
    posted: list[dict[str, object]] = []
    send_reads = 0

    async def session(_request: web.Request) -> web.Response:
        return json_response(
            {
                "status": "logged_in",
                "selfId": "wxid_self",
                "displayName": "Sidekick",
                "hookConnected": True,
                "connectionGeneration": 41,
                "contentRedacted": False,
                "cursor": "108739",
            }
        )

    async def capabilities(_request: web.Request) -> web.Response:
        return json_response(
            {
                "apiVersion": "v1alpha1",
                "messages": {
                    "receiveText": True,
                    "stableInboundMessageIds": True,
                    "sendText": True,
                    "requestIdempotency": True,
                    "outboundStableMessageId": True,
                },
                "events": {
                    "websocket": True,
                    "cursor": True,
                    "replay": True,
                    "durableCursor": True,
                },
                "runtime": {
                    "hookConnected": True,
                    "connectionGeneration": 41,
                    "sessionStatus": "logged_in",
                    "textSendReady": True,
                },
                "sync": {"history": False},
            }
        )

    async def chats(_request: web.Request) -> web.Response:
        return json_response(
            {
                "data": [
                    {
                        "id": "56825427596@chatroom",
                        "type": "group",
                        "displayName": "Example group",
                        "isPartial": True,
                    }
                ],
                "isPartial": True,
                "source": "native-chat-snapshot",
                "snapshot": {
                    "id": "snapshot-41",
                    "complete": True,
                    "current": True,
                    "count": 1,
                    "cursor": "108738",
                    "connectionGeneration": 41,
                },
                "cursor": "108739",
            }
        )

    async def messages(request: web.Request) -> web.Response:
        assert request.query == {"limit": "1000"}
        return json_response(
            {
                "data": [
                    {
                        "id": "4159667620982040828",
                        "chatId": "56825427596@chatroom",
                        "direction": "in",
                        "messageType": "text",
                        "from": "56825427596@chatroom",
                        "to": "wxid_self",
                        "senderId": "wxid_alice",
                        "content": "/ai hello",
                        "timestamp": 1_783_772_734,
                        "source": "wechat+localdb",
                        "isPartial": True,
                    }
                ],
                "isPartial": True,
                "source": "learned-hook-events",
                "cursor": "108739",
            }
        )

    async def send_text(request: web.Request) -> web.Response:
        posted.append(await request.json())
        return json_response(
            {
                "requestId": "sidekick.wechat.request-1",
                "status": "queued",
                "to": "56825427596@chatroom",
                "messageType": "text",
                "method": "bridge-send-queue",
            },
            status=202,
        )

    async def get_send(_request: web.Request) -> web.Response:
        nonlocal send_reads
        send_reads += 1
        return json_response(
            {
                "requestId": "sidekick.wechat.request-1",
                "status": "submitted",
                "messageId": "7158246912028861544",
                "to": "56825427596@chatroom",
                "messageType": "text",
                "method": "gui-fallback-text",
            }
        )

    app = web.Application()
    app.router.add_get("/session", session)
    app.router.add_get("/capabilities", capabilities)
    app.router.add_get("/chats", chats)
    app.router.add_get("/messages", messages)
    app.router.add_post("/messages/text", send_text)
    app.router.add_get("/sends/{request_id}", get_send)

    async with TestServer(app) as server:
        client = WeChatConnectorClient(
            str(server.make_url("/")),
            token="bridge-secret",
            send_poll_interval=0,
        )
        try:
            observed_session = await client.get_session()
            observed_capabilities = await client.get_capabilities()
            observed_chats = await client.get_chats()
            observed_messages = await client.get_messages(limit=1000)
            operation = await client.send_text_and_wait(
                request_id="sidekick.wechat.request-1",
                to="56825427596@chatroom",
                content="final answer",
            )
        finally:
            await client.close()

    assert observed_session.self_id == "wxid_self"
    assert observed_session.connection_generation == 41
    observed_capabilities.require_ai_channel()
    assert observed_chats.snapshot.id == "snapshot-41"
    assert observed_chats.chats[0].id == "56825427596@chatroom"
    assert observed_messages.messages[0].id == "4159667620982040828"
    assert operation.message_id == "7158246912028861544"
    assert send_reads == 1
    assert posted == [
        {
            "requestId": "sidekick.wechat.request-1",
            "to": "56825427596@chatroom",
            "content": "final answer",
        }
    ]


@pytest.mark.asyncio
async def test_wechat_client_rejects_unversioned_json() -> None:
    async def session(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "logged_in",
                "selfId": "wxid_self",
                "hookConnected": True,
                "connectionGeneration": 1,
                "contentRedacted": False,
                "cursor": "1",
            }
        )

    app = web.Application()
    app.router.add_get("/session", session)
    async with TestServer(app) as server:
        client = WeChatConnectorClient(str(server.make_url("/")))
        try:
            with pytest.raises(WeChatAPIContractError, match="API version"):
                await client.get_session()
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_wechat_client_reads_complete_chunked_json_response() -> None:
    payload = json.dumps(
        {
            "data": [
                {
                    "id": "filehelper",
                    "type": "direct",
                    "displayName": "File Transfer",
                }
            ],
            "snapshot": {
                "id": "snapshot-1",
                "complete": True,
                "current": True,
                "count": 1,
                "cursor": "10",
                "connectionGeneration": 1,
            },
            "cursor": "10",
        }
    ).encode()

    async def chats(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={**API_HEADERS, "Content-Type": "application/json"}
        )
        await response.prepare(request)
        await response.write(payload[:32])
        await asyncio.sleep(0.01)
        await response.write(payload[32:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/chats", chats)
    async with TestServer(app) as server:
        client = WeChatConnectorClient(str(server.make_url("/")))
        try:
            observed = await client.get_chats()
        finally:
            await client.close()

    assert observed.cursor == "10"
    assert observed.chats[0].id == "filehelper"


@pytest.mark.asyncio
async def test_wechat_client_never_retries_unknown_send_under_a_new_id() -> None:
    posts = 0

    async def send_text(_request: web.Request) -> web.Response:
        nonlocal posts
        posts += 1
        return json_response(
            {
                "requestId": "sidekick.wechat.unknown-1",
                "status": "unknown",
                "errorCode": "SEND_OUTCOME_UNKNOWN",
                "to": "filehelper",
                "messageType": "text",
            }
        )

    app = web.Application()
    app.router.add_post("/messages/text", send_text)
    async with TestServer(app) as server:
        client = WeChatConnectorClient(str(server.make_url("/")))
        try:
            with pytest.raises(WeChatSendOutcomeUnknown) as raised:
                await client.send_text_and_wait(
                    request_id="sidekick.wechat.unknown-1",
                    to="filehelper",
                    content="possibly sent",
                )
        finally:
            await client.close()

    assert raised.value.operation.request_id == "sidekick.wechat.unknown-1"
    assert posts == 1


@pytest.mark.asyncio
async def test_wechat_client_replays_events_after_opaque_cursor() -> None:
    observed_after: list[str] = []

    async def events(request: web.Request) -> web.StreamResponse:
        observed_after.append(request.query["after"])
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        await websocket.send_json(
            {
                "schemaVersion": "wechat-bridge/v1alpha1",
                "cursor": "opaque-next",
                "event": "message",
                "id": "4159667620982040828",
                "chatId": "filehelper",
                "direction": "in",
                "messageType": "text",
                "senderId": "wxid_alice",
                "content": "/ai hello",
                "timestamp": 1_783_772_734,
                "connectionGeneration": 41,
            }
        )
        await websocket.close()
        return websocket

    app = web.Application()
    app.router.add_get("/events", events)
    async with TestServer(app) as server:
        client = WeChatConnectorClient(str(server.make_url("/")))
        try:
            stream: AsyncIterator = client.events(after="opaque-current")
            event = await stream.__anext__()
            await stream.aclose()
        finally:
            await client.close()

    assert event.cursor == "opaque-next"
    assert event.name == "message"
    assert event.payload["id"] == "4159667620982040828"
    assert observed_after == ["opaque-current"]
