from __future__ import annotations

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from sidekick.wechat.api import (
    WeChatAPIContractError,
    WeChatConnectorClient,
    WeChatObservedMessage,
    WeChatObservedMessagePage,
)


API_HEADERS = {"X-WeChat-Bridge-API-Version": "v1alpha1"}
CHAT_ID = "56825427596@chatroom"
MESSAGE_ID = "4159667620982040828"


def observed_message_payload(
    message_id: str = MESSAGE_ID,
) -> dict[str, object]:
    return {
        "id": message_id,
        "chatId": CHAT_ID,
        "state": "present",
        "version": "mv1:current",
        "direction": "in",
        "messageType": "chat_history",
        "content": "Team history",
        "senderId": "wxid_alice",
        "senderDisplayName": "Alice",
        "senderGroupAlias": "Team Alice",
        "timestamp": 1_700_000_010,
        "orderTimestamp": 1_700_000_000,
        "replyToMessageId": "4159667620982040800",
        "sharedChatHistory": {
            "title": "Team history",
            "itemCount": 1,
            "items": [
                {
                    "kind": "text",
                    "senderName": "Bob",
                    "content": "Hello",
                    "timestamp": 1_700_000_000,
                }
            ],
        },
        "observedAt": 1_786_651_200,
        "source": "wechat+localdb",
    }


def observed_page_payload() -> dict[str, object]:
    return {
        "data": [observed_message_payload()],
        "page": {
            "before": MESSAGE_ID,
            "after": MESSAGE_ID,
            "hasMoreBefore": True,
            "hasMoreAfter": False,
        },
        "coverage": {
            "kind": "partial",
            "oldestAvailable": {
                "id": "100",
                "orderTimestamp": 1_690_000_000,
            },
            "newestAvailable": {
                "id": MESSAGE_ID,
                "orderTimestamp": 1_700_000_000,
            },
            "mayHaveUnobservedMessages": True,
            "observationGaps": [
                {
                    "source": "local_db",
                    "reason": "chat_not_covered",
                    "from": 1_786_651_200,
                }
            ],
            "gapsAreExhaustive": False,
            "sources": ["local_db", "wechat+localdb"],
        },
    }


def test_observed_message_keeps_stable_order_and_readable_sender_identity() -> None:
    message = WeChatObservedMessage.parse(observed_message_payload())

    assert message.id == MESSAGE_ID
    assert message.timestamp == 1_700_000_010
    assert message.order_timestamp == 1_700_000_000
    assert message.sender_label == "Team Alice"
    assert message.display_content == (
        "[Forwarded chat history]\nTeam history\nBob: Hello"
    )


def test_observed_recall_is_a_payload_free_tombstone() -> None:
    message = WeChatObservedMessage.parse(
        {
            "id": MESSAGE_ID,
            "chatId": CHAT_ID,
            "state": "recalled",
            "version": "mv1:recalled",
            "orderTimestamp": 1_700_000_000,
            "observedAt": 1_786_651_200,
            "source": "wechat+localdb",
        }
    )

    assert message.state == "recalled"
    assert message.sender_id is None
    assert message.display_content == ""


def test_observed_recall_rejects_stale_content() -> None:
    payload = {
        "id": MESSAGE_ID,
        "chatId": CHAT_ID,
        "state": "recalled",
        "version": "mv1:recalled",
        "orderTimestamp": 1_700_000_000,
        "observedAt": 1_786_651_200,
        "source": "wechat+localdb",
        "content": "stale",
    }

    with pytest.raises(WeChatAPIContractError, match="recalled.*payload"):
        WeChatObservedMessage.parse(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", "complete"),
        ("mayHaveUnobservedMessages", False),
        ("gapsAreExhaustive", True),
    ),
)
def test_observed_page_cannot_claim_complete_history(
    field: str,
    value: object,
) -> None:
    payload = observed_page_payload()
    coverage = payload["coverage"]
    assert isinstance(coverage, dict)
    coverage[field] = value

    with pytest.raises(WeChatAPIContractError, match="coverage"):
        WeChatObservedMessagePage.parse(payload)


@pytest.mark.asyncio
async def test_client_fetches_exact_message_and_bounded_partial_page() -> None:
    requests: list[tuple[str, dict[str, str]]] = []

    async def exact(request: web.Request) -> web.Response:
        requests.append((request.path, dict(request.query)))
        return web.json_response(
            observed_message_payload(request.match_info["message_id"]),
            headers=API_HEADERS,
        )

    async def page(request: web.Request) -> web.Response:
        requests.append((request.path, dict(request.query)))
        return web.json_response(observed_page_payload(), headers=API_HEADERS)

    app = web.Application()
    app.router.add_get(
        "/chats/{chat_id}/messages/{message_id}",
        exact,
    )
    app.router.add_get("/chats/{chat_id}/messages", page)
    async with TestServer(app) as server:
        client = WeChatConnectorClient(str(server.make_url("/")))
        try:
            message = await client.get_observed_message(CHAT_ID, MESSAGE_ID)
            observed_page = await client.get_observed_messages(
                CHAT_ID,
                before=MESSAGE_ID,
                since=1_690_000_000,
                until=1_700_000_000,
                limit=100,
            )
        finally:
            await client.close()

    assert message.version == "mv1:current"
    assert observed_page.coverage.kind == "partial"
    assert observed_page.page.has_more_before is True
    assert requests == [
        (f"/chats/{CHAT_ID}/messages/{MESSAGE_ID}", {}),
        (
            f"/chats/{CHAT_ID}/messages",
            {
                "before": MESSAGE_ID,
                "since": "1690000000",
                "until": "1700000000",
                "limit": "100",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_client_rejects_ambiguous_or_unbounded_observed_pages() -> None:
    client = WeChatConnectorClient("http://127.0.0.1:1")
    try:
        with pytest.raises(ValueError, match="before.*after"):
            await client.get_observed_messages(
                CHAT_ID,
                before=MESSAGE_ID,
                after=MESSAGE_ID,
            )
        with pytest.raises(ValueError, match="between 1 and 100"):
            await client.get_observed_messages(CHAT_ID, limit=101)
    finally:
        await client.close()
