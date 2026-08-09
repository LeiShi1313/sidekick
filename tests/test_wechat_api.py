from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from sidekick.wechat.api import (
    MAX_MEDIA_BYTES,
    WeChatAPIContractError,
    WeChatConnectorMessage,
    WeChatConnectorClient,
    WeChatEvent,
    WeChatGroupMemberList,
    WeChatMessageList,
    WeChatSendOperation,
    WeChatSendOutcomeUnknown,
    WeChatUser,
    WeChatUserList,
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


def test_wechat_client_parses_readable_user_and_group_member_names() -> None:
    users = WeChatUserList.parse(
        {
            "data": [
                {
                    "id": "wxid_alice",
                    "displayName": " Alice Global ",
                    "source": "learned-hook-events",
                    "isPartial": True,
                },
                {
                    "id": "56825427596@chatroom",
                    "displayName": "Project group",
                    "source": "learned-message-events",
                    "isPartial": True,
                },
                {
                    "id": "_1234567",
                    "displayName": "Unsupported learned identity",
                    "source": "native-group-member-snapshot",
                    "isPartial": True,
                }
            ],
            "isPartial": True,
            "source": "learned-hook-events",
            "cursor": "users-10",
        }
    )
    members = WeChatGroupMemberList.parse(
        {
            "data": [
                {
                    "groupId": "56825427596@chatroom",
                    "userId": "wxid_alice",
                    "displayName": "Alice Global",
                    "nickname": " Alice in this group ",
                    "source": "wechat+0x5cf0af0",
                    "isPartial": False,
                }
            ],
            "isPartial": True,
            "source": "native-group-member-snapshot",
            "snapshot": {
                "id": "members-10",
                "groupId": "56825427596@chatroom",
                "complete": True,
                "current": True,
                "count": 1,
                "connectionGeneration": 41,
            },
            "cursor": "members-10",
        },
        group_id="56825427596@chatroom",
    )

    assert users.users == (
        WeChatUser(id="wxid_alice", display_name="Alice Global"),
    )
    assert members.members[0].display_name == "Alice Global"
    assert members.members[0].nickname == "Alice in this group"
    assert members.snapshot_complete is True
    assert members.snapshot_current is True
    assert members.snapshot_connection_generation == 41


@pytest.mark.parametrize(
    ("identity_id", "expected"),
    (
        ("56825427596@chatroom", "must be a user ID"),
        ("_1234567", "must be a canonical WeChat ID"),
    ),
)
def test_wechat_user_profile_rejects_non_user_identity(
    identity_id: str,
    expected: str,
) -> None:
    with pytest.raises(WeChatAPIContractError, match=expected):
        WeChatUser.parse(
            {
                "id": identity_id,
                "displayName": "Not a user",
                "isPartial": True,
            }
        )


@pytest.mark.parametrize(
    ("complete", "connection_generation", "expected"),
    (
        (False, 41, "must be complete"),
        (True, None, "generation is missing"),
    ),
)
def test_wechat_client_rejects_invalid_current_group_member_snapshots(
    complete: bool,
    connection_generation: int | None,
    expected: str,
) -> None:
    snapshot: dict[str, object] = {
        "groupId": "56825427596@chatroom",
        "complete": complete,
        "current": True,
        "count": 0,
    }
    if connection_generation is not None:
        snapshot["connectionGeneration"] = connection_generation

    with pytest.raises(WeChatAPIContractError, match=expected):
        WeChatGroupMemberList.parse(
            {
                "data": [],
                "isPartial": True,
                "source": "native-group-member-snapshot",
                "snapshot": snapshot,
                "cursor": "members-10",
            },
            group_id="56825427596@chatroom",
        )


def test_wechat_client_rejects_group_members_from_another_group() -> None:
    with pytest.raises(WeChatAPIContractError, match="different group"):
        WeChatGroupMemberList.parse(
            {
                "data": [
                    {
                        "groupId": "56825427597@chatroom",
                        "userId": "wxid_alice",
                        "isPartial": True,
                    }
                ],
                "isPartial": True,
                "source": "learned-hook-events",
                "snapshot": {
                    "groupId": "56825427596@chatroom",
                    "complete": False,
                    "current": False,
                    "count": 0,
                },
                "cursor": "members-10",
            },
            group_id="56825427596@chatroom",
        )


@pytest.mark.asyncio
async def test_wechat_client_reads_identity_directories_and_missing_users() -> None:
    requested: list[str] = []

    async def users(_request: web.Request) -> web.Response:
        return json_response(
            {
                "data": [
                    {
                        "id": "wxid_alice",
                        "displayName": "Alice",
                        "isPartial": True,
                    }
                ],
                "isPartial": True,
                "source": "learned-hook-events",
                "cursor": "users-10",
            }
        )

    async def user(request: web.Request) -> web.Response:
        user_id = request.match_info["user_id"]
        requested.append(user_id)
        if user_id == "wxid_missing":
            return json_response(
                {"error": {"code": "NOT_FOUND", "message": "User not found"}},
                status=404,
            )
        return json_response(
            {
                "id": user_id,
                "displayName": "Alice Updated",
                "isPartial": True,
            }
        )

    async def members(request: web.Request) -> web.Response:
        group_id = request.match_info["group_id"]
        requested.append(group_id)
        return json_response(
            {
                "data": [
                    {
                        "groupId": group_id,
                        "userId": "wxid_alice",
                        "nickname": "Group Alice",
                        "isPartial": True,
                    }
                ],
                "isPartial": True,
                "source": "learned-hook-events",
                "snapshot": {
                    "groupId": group_id,
                    "complete": False,
                    "current": False,
                    "count": 0,
                },
                "cursor": "members-10",
            }
        )

    app = web.Application()
    app.router.add_get("/users", users)
    app.router.add_get("/users/{user_id}", user)
    app.router.add_get("/groups/{group_id}/members", members)
    async with TestServer(app) as server:
        client = WeChatConnectorClient(str(server.make_url("/")))
        try:
            observed_users = await client.get_users()
            observed_user = await client.get_user("wxid_alice")
            missing_user = await client.get_user("wxid_missing")
            observed_members = await client.get_group_members(
                "56825427596@chatroom"
            )
        finally:
            await client.close()

    assert observed_users.users[0].display_name == "Alice"
    assert observed_user is not None
    assert observed_user.display_name == "Alice Updated"
    assert missing_user is None
    assert observed_members.members[0].nickname == "Group Alice"
    assert requested == [
        "wxid_alice",
        "wxid_missing",
        "56825427596@chatroom",
    ]


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


def test_wechat_client_parses_image_media_identity() -> None:
    payload = {
        **connector_message_payload("4159667620982040828"),
        "messageType": "image",
        "content": "",
        "contentRedacted": True,
        "media": {
            "id": "0123456789abcdef0123456789abcdef",
            "state": "available",
        },
    }

    message = WeChatConnectorMessage.parse(payload)

    assert message.media_id == "0123456789abcdef0123456789abcdef"
    assert message.content_redacted is True


def test_wechat_client_parses_shared_chat_history_as_bounded_text() -> None:
    payload = {
        **connector_message_payload("4159667620982040828"),
        "messageType": "chat_history",
        "content": "Team history",
        "sharedChatHistory": {
            "title": "Team history",
            "itemCount": 3,
            "items": [
                {
                    "kind": "text",
                    "senderName": "Alice",
                    "timestamp": 1_700_000_001,
                    "content": "Hello",
                },
                {
                    "kind": "video",
                    "senderName": "Bob",
                    "timestamp": 1_700_000_002,
                    "content": "Demo clip",
                    "reply": {
                        "kind": "image",
                        "senderName": "Carol",
                        "content": "Earlier image",
                    },
                },
            ],
            "truncated": True,
        },
    }

    message = WeChatConnectorMessage.parse(payload)

    assert message.content == "Team history"
    assert message.shared_chat_history is not None
    assert message.shared_chat_history.item_count == 3
    assert message.display_content == (
        "[Forwarded chat history]\n"
        "Team history\n"
        "Alice: Hello\n"
        "Bob: [Video] Demo clip\n"
        "  ↳ Carol: [Image] Earlier image\n"
        "… 1 more item not included"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda history: history["items"][0].update(kind="script"),
        lambda history: history["items"][0].update(senderName="Alice\nAdmin"),
        lambda history: history.update(itemCount=257),
        lambda history: history.update(truncated=False),
    ),
)
def test_wechat_client_rejects_invalid_shared_chat_history(mutation) -> None:
    history = {
        "title": "Team history",
        "itemCount": 2,
        "items": [
            {"kind": "text", "senderName": "Alice", "content": "Hello"},
        ],
        "truncated": True,
    }
    mutation(history)
    payload = {
        **connector_message_payload("4159667620982040828"),
        "messageType": "chat_history",
        "content": "Team history",
        "sharedChatHistory": history,
    }

    with pytest.raises(WeChatAPIContractError, match="shared chat history"):
        WeChatConnectorMessage.parse(payload)


def test_wechat_client_rejects_shared_history_on_other_message_types() -> None:
    payload = connector_message_payload("4159667620982040828")
    payload["sharedChatHistory"] = {
        "title": "Team history",
        "itemCount": 1,
        "items": [{"kind": "text", "content": "Hello"}],
    }

    with pytest.raises(WeChatAPIContractError, match="shared chat history"):
        WeChatConnectorMessage.parse(payload)


def test_wechat_client_bounds_rendered_shared_chat_history() -> None:
    payload = {
        **connector_message_payload("4159667620982040828"),
        "messageType": "chat_history",
        "content": "Large history",
        "sharedChatHistory": {
            "title": "Large history",
            "itemCount": 8,
            "items": [
                {
                    "kind": "text",
                    "senderName": f"User {index}",
                    "content": "界" * 1_500,
                }
                for index in range(8)
            ],
        },
    }

    message = WeChatConnectorMessage.parse(payload)

    assert len(message.display_content.encode("utf-8")) <= 16 * 1024
    assert "more items not included" in message.display_content


@pytest.mark.parametrize(
    "media_id",
    (
        "0123456789ABCDEF0123456789ABCDEF",
        "0123456789abcdef",
        "../../0123456789abcdef0123456789ab",
    ),
)
def test_wechat_client_rejects_noncanonical_media_ids(media_id: str) -> None:
    payload = {
        **connector_message_payload("4159667620982040828"),
        "messageType": "image",
        "media": {"id": media_id, "state": "available"},
    }

    with pytest.raises(WeChatAPIContractError, match="media.id"):
        WeChatConnectorMessage.parse(payload)


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


@pytest.mark.parametrize(
    ("event_name", "status", "raw", "expected"),
    (
        ("group_member_snapshot", "begin", None, None),
        (
            "group_member_snapshot",
            "end",
            None,
            "56825427596@chatroom",
        ),
        ("group_member", None, {"mode": "cache_snapshot"}, None),
        (
            "group_member",
            None,
            {
                "mode": "delta",
                "deltaId": "delta-1",
                "deltaIndex": "0",
                "responseCount": "2",
            },
            None,
        ),
        (
            "group_member",
            None,
            {
                "mode": "delta",
                "deltaId": "delta-1",
                "deltaIndex": "1",
                "responseCount": "2",
            },
            "56825427596@chatroom",
        ),
    ),
)
def test_wechat_identity_events_identify_directory_invalidations(
    event_name: str,
    status: str | None,
    raw: dict[str, str] | None,
    expected: str | None,
) -> None:
    payload: dict[str, object] = {
        "schemaVersion": "wechat-bridge/v1alpha1",
        "cursor": "11",
        "event": event_name,
        "connectionGeneration": 41,
        "groupId": "56825427596@chatroom",
    }
    if status is not None:
        payload["status"] = status
    if raw is not None:
        payload["raw"] = raw

    event = WeChatEvent.parse(payload)

    assert event.invalidated_group_id() == expected


@pytest.mark.parametrize(
    "raw",
    (
        {"mode": "delta", "deltaIndex": "0", "responseCount": "1"},
        {
            "mode": "delta",
            "deltaId": "delta-1",
            "deltaIndex": "0",
            "responseCount": "0",
        },
        {
            "mode": "delta",
            "deltaId": "delta-1",
            "deltaIndex": "2",
            "responseCount": "2",
        },
        {
            "mode": "delta",
            "deltaId": "delta-1",
            "deltaIndex": "00",
            "responseCount": "1",
        },
    ),
)
def test_wechat_member_delta_rejects_invalid_batch_metadata(
    raw: dict[str, str],
) -> None:
    event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "11",
            "event": "group_member",
            "connectionGeneration": 41,
            "groupId": "56825427596@chatroom",
            "raw": raw,
        }
    )

    with pytest.raises(WeChatAPIContractError):
        event.invalidated_group_id()


def test_wechat_user_profile_event_requires_canonical_user_identity() -> None:
    event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "11",
            "event": "user_profile",
            "status": "changed",
            "id": "sha256:profile-1",
            "userId": "56825427596@chatroom",
            "connectionGeneration": 41,
        }
    )

    with pytest.raises(WeChatAPIContractError, match="userId"):
        event.changed_user_id()


def test_wechat_message_list_skips_senderless_unsupported_history_rows() -> None:
    text = connector_message_payload("4159667620982040828")
    app = {
        **connector_message_payload("4159667620982040829"),
        "messageType": "app",
    }
    unknown = {
        **connector_message_payload("4159667620982040830"),
        "messageType": "unknown",
    }
    app.pop("senderId")
    unknown.pop("senderId")

    messages = WeChatMessageList.parse(
        {"data": [app, text, unknown], "cursor": "108739"}
    )

    assert [message.id for message in messages.messages] == [text["id"]]


def test_wechat_message_list_rejects_senderless_text_history_row() -> None:
    text = connector_message_payload("4159667620982040828")
    text.pop("senderId")

    with pytest.raises(WeChatAPIContractError, match="senderId"):
        WeChatMessageList.parse({"data": [text], "cursor": "108739"})


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
                    "receiveSharedChatHistory": True,
                    "stableInboundMessageIds": True,
                    "sendText": True,
                    "sendReply": True,
                    "sendNativeReply": True,
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
                    "replySendReady": True,
                },
                "media": {
                    "inboundImageDownload": True,
                    "requestOriginalImage": True,
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
                reply_to_message_id="4159667620982040828",
            )
        finally:
            await client.close()

    assert observed_session.self_id == "wxid_self"
    assert observed_session.connection_generation == 41
    observed_capabilities.require_ai_channel()
    assert observed_capabilities.inbound_image_download is True
    assert observed_capabilities.request_original_image is True
    assert observed_capabilities.receive_shared_chat_history is True
    assert observed_capabilities.native_reply_ready is True
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
            "replyToMessageId": "4159667620982040828",
        }
    ]


@pytest.mark.asyncio
async def test_wechat_client_downloads_exact_original_image() -> None:
    media_id = "0123456789abcdef0123456789abcdef"
    image_bytes = b"full-resolution-image"
    posted: list[dict[str, object]] = []

    async def request_original(request: web.Request) -> web.Response:
        assert request.match_info["media_id"] == media_id
        assert request.headers["Authorization"] == "Bearer bridge-secret"
        posted.append(await request.json())
        return json_response(
            {
                "requestId": "sidekick.wechat.original.request-1",
                "status": "available",
                "chatId": "56825427596@chatroom",
                "messageId": "4159667620982040828",
                "media": {
                    "id": media_id,
                    "variant": "original",
                    "mimeType": "image/png",
                    "size": len(image_bytes),
                    "downloadUrl": f"/media/{media_id}/original",
                },
            }
        )

    async def download_original(request: web.Request) -> web.Response:
        assert request.headers["Authorization"] == "Bearer bridge-secret"
        return web.Response(body=image_bytes, content_type="image/png")

    app = web.Application()
    app.router.add_post("/media/{media_id}/original", request_original)
    app.router.add_get("/media/{media_id}/original", download_original)
    async with TestServer(app) as server:
        client = WeChatConnectorClient(
            str(server.make_url("/")),
            token="bridge-secret",
        )
        try:
            image = await client.download_original_image(
                request_id="sidekick.wechat.original.request-1",
                chat_id="56825427596@chatroom",
                message_id="4159667620982040828",
                media_id=media_id,
            )
        finally:
            await client.close()

    assert image.data == image_bytes
    assert image.mime_type == "image/png"
    assert image.variant == "original"
    assert posted == [
        {
            "requestId": "sidekick.wechat.original.request-1",
            "chatId": "56825427596@chatroom",
            "messageId": "4159667620982040828",
        }
    ]


@pytest.mark.asyncio
async def test_wechat_client_rejects_oversized_preview_before_reading_body() -> None:
    media_id = "0123456789abcdef0123456789abcdef"

    async def oversized_preview(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "image/jpeg",
                "Content-Length": str(MAX_MEDIA_BYTES + 1),
            }
        )
        await response.prepare(request)
        return response

    app = web.Application()
    app.router.add_get("/media/{media_id}", oversized_preview)
    async with TestServer(app) as server:
        client = WeChatConnectorClient(str(server.make_url("/")))
        try:
            with pytest.raises(WeChatAPIContractError, match="oversized"):
                await client.download_image_preview(media_id=media_id)
        finally:
            await client.close()


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
                    reply_to_message_id=None,
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
