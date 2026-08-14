from __future__ import annotations

import asyncio

import pytest

from sidekick.wechat.api import (
    WeChatAPIError,
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatObservedMessage,
    WeChatSession,
)
from sidekick.wechat.service import WeChatPendingAIWorker
from sidekick.wechat.store import WeChatStateRepository


CONNECTOR_KEY = "http://wechat-connector:18188"
CHAT_ID = "56825427596@chatroom"
MESSAGE_ID = "4159667620982040828"


def present_message(*, version: str = "mv1:present") -> WeChatObservedMessage:
    return WeChatObservedMessage.parse(
        {
            "id": MESSAGE_ID,
            "chatId": CHAT_ID,
            "state": "present",
            "version": version,
            "direction": "in",
            "messageType": "text",
            "content": "/ai hello",
            "senderId": "wxid_alice",
            "senderDisplayName": "Alice",
            "senderGroupAlias": "Team Alice",
            "timestamp": 1_700_000_010,
            "orderTimestamp": 1_700_000_000,
            "observedAt": 1_786_651_200,
            "source": "wechat+localdb",
        }
    )


def recalled_message() -> WeChatObservedMessage:
    return WeChatObservedMessage.parse(
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


async def open_store(path) -> WeChatStateRepository:
    store = await WeChatStateRepository(path).connect()
    await store.bootstrap(
        connector_key=CONNECTOR_KEY,
        session=WeChatSession(
            status="logged_in",
            self_id="wxid_self",
            display_name="Sidekick",
            hook_connected=True,
            connection_generation=41,
            content_redacted=False,
            cursor="bootstrap-cursor",
        ),
        chats=WeChatChatList(
            chats=(
                WeChatChat(
                    id=CHAT_ID,
                    type="group",
                    display_name="Example group",
                ),
            ),
            snapshot=WeChatChatSnapshot(
                id="snapshot-41",
                complete=True,
                current=True,
                count=1,
                cursor="bootstrap-cursor",
                connection_generation=41,
            ),
            cursor="bootstrap-cursor",
        ),
    )
    return store


class FakeObservedClient:
    def __init__(self, results: list[WeChatObservedMessage | Exception]):
        self.results = results
        self.requests: list[tuple[str, str]] = []

    async def get_observed_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> WeChatObservedMessage:
        self.requests.append((chat_id, message_id))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class RecordingHandler:
    def __init__(self, *, handled: bool = True):
        self.handled = handled
        self.messages = []

    async def handle(self, message) -> bool:
        self.messages.append(message)
        return self.handled


async def enqueue(store: WeChatStateRepository, *, kind: str = "message") -> None:
    await store.accept_pending_ai_event(
        CONNECTOR_KEY,
        cursor="event-11",
        chat_id=CHAT_ID,
        message_id=MESSAGE_ID,
        kind=kind,
    )


@pytest.mark.asyncio
async def test_worker_fetches_authoritative_message_and_uses_connector_label(
    tmp_path,
) -> None:
    store = await open_store(tmp_path / "wechat.db")
    await enqueue(store)
    client = FakeObservedClient([present_message()])
    handler = RecordingHandler()
    try:
        result = await WeChatPendingAIWorker(
            client,
            store,
            CONNECTOR_KEY,
        ).process_one(handler)

        assert result == "completed"
        assert client.requests == [(CHAT_ID, MESSAGE_ID)]
        assert len(handler.messages) == 1
        message = handler.messages[0]
        assert message.raw_text == "/ai hello"
        assert message.memory_cursor == MESSAGE_ID
        assert message.sender_display_name == "Team Alice"
        assert message.scope_display_name == "Example group"
        assert await store.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
        ) is None
        assert await store.count_messages(CONNECTOR_KEY) == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_worker_retries_503_but_bounds_not_observed_404(tmp_path) -> None:
    store = await open_store(tmp_path / "wechat.db")
    await enqueue(store)
    now = 100.0
    client = FakeObservedClient(
        [
            WeChatAPIError(503, "MESSAGE_HISTORY_NOT_READY", "not ready"),
            WeChatAPIError(404, "MESSAGE_NOT_OBSERVED", "not observed"),
            WeChatAPIError(404, "MESSAGE_NOT_OBSERVED", "not observed"),
        ]
    )
    worker = WeChatPendingAIWorker(
        client,
        store,
        CONNECTOR_KEY,
        not_observed_attempts=2,
        clock=lambda: now,
    )
    handler = RecordingHandler()
    try:
        assert await worker.process_one(handler) == "deferred"
        now = 102
        assert await worker.process_one(handler) == "deferred"
        now = 106
        assert await worker.process_one(handler) == "unavailable"

        pending = await store.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.status == "unavailable"
        assert pending.last_error_code == "MESSAGE_NOT_OBSERVED"
        assert handler.messages == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_worker_resolves_removal_without_fetching_or_generating(tmp_path) -> None:
    store = await open_store(tmp_path / "wechat.db")
    await enqueue(store, kind="message_remove")
    client = FakeObservedClient([])
    handler = RecordingHandler()
    try:
        assert await WeChatPendingAIWorker(
            client,
            store,
            CONNECTOR_KEY,
        ).process_one(handler) == "recalled"
        assert client.requests == []
        assert handler.messages == []
        assert await store.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
        ) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_worker_treats_fetched_recall_as_terminal_without_content(
    tmp_path,
) -> None:
    store = await open_store(tmp_path / "wechat.db")
    await enqueue(store)
    handler = RecordingHandler()
    try:
        assert await WeChatPendingAIWorker(
            FakeObservedClient([recalled_message()]),
            store,
            CONNECTOR_KEY,
        ).process_one(handler) == "recalled"
        assert handler.messages == []
        assert await store.get_processed_revision_status(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
            "mv1:recalled",
        ) == "recalled"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancelled_worker_marks_started_execution_unknown_instead_of_retrying(
    tmp_path,
) -> None:
    entered = asyncio.Event()

    class BlockingHandler:
        async def handle(self, _message) -> bool:
            entered.set()
            await asyncio.Future()
            return True

    store = await open_store(tmp_path / "wechat.db")
    await enqueue(store)
    worker = WeChatPendingAIWorker(
        FakeObservedClient([present_message(version="mv1:running")]),
        store,
        CONNECTOR_KEY,
    )
    task = asyncio.create_task(worker.process_one(BlockingHandler()))
    await entered.wait()
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task

        pending = await store.get_pending_ai_work(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.status == "failed_unknown"
        assert await store.claim_pending_ai_work(CONNECTOR_KEY) is None
    finally:
        await store.close()
