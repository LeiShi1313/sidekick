from __future__ import annotations

import asyncio

import pytest

from sidekick.inbound import DurableInboundWorker
from sidekick.wechat.api import (
    WeChatAPIError,
    WeChatCapabilities,
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatObservedMessage,
    WeChatEvent,
    WeChatSession,
)
from sidekick.wechat.service import (
    WeChatBootstrap,
    WeChatEventPump,
    WeChatObservedMessageSource,
)
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


def shared_history_message() -> WeChatObservedMessage:
    return WeChatObservedMessage.parse(
        {
            "id": MESSAGE_ID,
            "chatId": CHAT_ID,
            "state": "present",
            "version": "mv1:shared-history",
            "direction": "in",
            "messageType": "chat_history",
            "content": "Team history",
            "senderId": "wxid_alice",
            "senderDisplayName": "Alice",
            "senderGroupAlias": "Team Alice",
            "timestamp": 1_700_000_010,
            "orderTimestamp": 1_700_000_000,
            "sharedChatHistory": {
                "title": "Team history",
                "itemCount": 1,
                "items": [
                    {
                        "kind": "text",
                        "senderName": "Bob",
                        "content": "Hello",
                    }
                ],
            },
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


def inbound_worker(
    client,
    store,
    *,
    not_observed_attempts: int = 3,
    clock=None,
) -> DurableInboundWorker:
    source = WeChatObservedMessageSource(
        client,
        store,
        CONNECTOR_KEY,
        not_observed_attempts=not_observed_attempts,
    )
    options = {} if clock is None else {"clock": clock}
    return DurableInboundWorker(
        source,
        store,
        CONNECTOR_KEY,
        **options,
    )


async def enqueue(store: WeChatStateRepository, *, kind: str = "message") -> None:
    await store.accept_pending_ai_event(
        CONNECTOR_KEY,
        cursor="event-11",
        chat_id=CHAT_ID,
        message_id=MESSAGE_ID,
        kind=kind,
    )


def bootstrap() -> WeChatBootstrap:
    return WeChatBootstrap(
        session=WeChatSession(
            status="logged_in",
            self_id="wxid_self",
            display_name="Sidekick",
            hook_connected=True,
            connection_generation=41,
            content_redacted=False,
            cursor="bootstrap-cursor",
        ),
        capabilities=WeChatCapabilities(
            receive_text=True,
            receive_shared_chat_history=True,
            stable_inbound_message_ids=True,
            send_text=True,
            send_reply=True,
            send_native_reply=True,
            request_idempotency=True,
            outbound_stable_message_id=True,
            websocket=True,
            cursor=True,
            replay=True,
            durable_cursor=True,
            text_send_ready=True,
            reply_send_ready=True,
            connection_generation=41,
            history=False,
            inbound_image_download=True,
            request_original_image=True,
            fetch_observed_messages=True,
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


def message_event(
    *,
    cursor: str,
    message_id: str = MESSAGE_ID,
) -> WeChatEvent:
    return WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": cursor,
            "event": "message",
            "connectionGeneration": 41,
            "id": message_id,
            "chatId": CHAT_ID,
            "direction": "in",
            "messageType": "text",
            "senderId": "wxid_alice",
            "content": "/ai hello",
            "timestamp": 1_700_000_010,
        }
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
        result = await inbound_worker(client, store).process_one(handler)

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
    worker = inbound_worker(
        client,
        store,
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
        assert await inbound_worker(client, store).process_one(handler) == "recalled"
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
        assert await inbound_worker(
            FakeObservedClient([recalled_message()]),
            store,
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
    worker = inbound_worker(
        FakeObservedClient([present_message(version="mv1:running")]),
        store,
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


class StreamingObservedClient(FakeObservedClient):
    def __init__(self, events, results):
        super().__init__(results)
        self._events = tuple(events)
        self.after_values: list[str] = []
        self.keep_open = asyncio.Event()

    async def events(self, *, after: str):
        self.after_values.append(after)
        for event in self._events:
            yield event
        await self.keep_open.wait()

    async def get_chats(self):
        return bootstrap().chats

    async def get_user(self, _user_id):
        return None

    async def get_group_members(self, _group_id):
        raise AssertionError("group directory should not be read")


async def wait_until(predicate) -> None:
    for _ in range(100):
        if await predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


@pytest.mark.asyncio
async def test_event_pump_drops_poison_but_advances_to_later_ai_command(
    tmp_path,
) -> None:
    poison = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "1249405",
            "event": "message",
            "connectionGeneration": 41,
            "id": "4159667620982040800",
            "chatId": CHAT_ID,
            "direction": "in",
            "messageType": "system",
            "sharedChatHistory": {"title": "stale"},
            "timestamp": 1_700_000_000,
        }
    )
    valid = message_event(cursor="1249406")
    client = StreamingObservedClient((poison, valid), [present_message()])
    store = await open_store(tmp_path / "wechat.db")
    handler = RecordingHandler()
    stop = asyncio.Event()
    task = asyncio.create_task(
        WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap(),
            handler_concurrency=1,
        ).run(handler, stop)
    )
    try:
        await wait_until(_async_predicate(lambda: bool(handler.messages)))

        assert await store.get_cursor(CONNECTOR_KEY) == "1249406"
        assert [message.id for message in handler.messages] == [MESSAGE_ID]
        assert await store.count_messages(CONNECTOR_KEY) == 0
    finally:
        stop.set()
        assert await asyncio.wait_for(task, timeout=1) == "stopped"
        await store.close()


@pytest.mark.asyncio
async def test_event_pump_dispatches_valid_shared_chat_history(tmp_path) -> None:
    event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "event-11",
            "event": "message",
            "connectionGeneration": 41,
            "id": MESSAGE_ID,
            "chatId": CHAT_ID,
            "direction": "in",
            "messageType": "chat_history",
            "senderId": "wxid_alice",
            "content": "Team history",
            "sharedChatHistory": {
                "title": "Team history",
                "itemCount": 1,
                "items": [
                    {
                        "kind": "text",
                        "senderName": "Bob",
                        "content": "Hello",
                    }
                ],
            },
            "timestamp": 1_700_000_010,
        }
    )
    client = StreamingObservedClient((event,), [shared_history_message()])
    store = await open_store(tmp_path / "wechat.db")
    handler = RecordingHandler()
    stop = asyncio.Event()
    task = asyncio.create_task(
        WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap(),
            handler_concurrency=1,
        ).run(handler, stop)
    )
    try:
        await wait_until(_async_predicate(lambda: bool(handler.messages)))

        assert handler.messages[0].message_type == "chat_history"
        assert handler.messages[0].raw_text == (
            "[Forwarded chat history]\nTeam history\nBob: Hello"
        )
        assert await store.get_cursor(CONNECTOR_KEY) == "event-11"
    finally:
        stop.set()
        assert await asyncio.wait_for(task, timeout=1) == "stopped"
        await store.close()


@pytest.mark.asyncio
async def test_null_shared_history_does_not_match_the_poison_guard(
    tmp_path,
    caplog,
) -> None:
    payload = dict(message_event(cursor="event-11").payload)
    payload["sharedChatHistory"] = None
    client = StreamingObservedClient((WeChatEvent.parse(payload),), [])
    store = await open_store(tmp_path / "wechat.db")
    stop = asyncio.Event()
    task = asyncio.create_task(
        WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap(),
            handler_concurrency=1,
        ).run(RecordingHandler(), stop)
    )
    try:
        await wait_until(
            _async_predicate_from_async(
                store.get_cursor,
                CONNECTOR_KEY,
                expected="event-11",
            )
        )

        assert client.requests == []
        assert (
            "Dropped inconsistent WeChat shared chat history event"
            not in caplog.text
        )
        assert "Dropped malformed WeChat message event" in caplog.text
    finally:
        stop.set()
        assert await asyncio.wait_for(task, timeout=1) == "stopped"
        await store.close()


@pytest.mark.asyncio
async def test_event_cursor_is_not_blocked_by_running_ai_handler(tmp_path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingHandler:
        async def handle(self, _message) -> bool:
            started.set()
            await release.wait()
            return True

    later = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "event-12",
            "event": "unsupported_future_event",
            "connectionGeneration": 41,
        }
    )
    client = StreamingObservedClient(
        (message_event(cursor="event-11"), later),
        [present_message()],
    )
    store = await open_store(tmp_path / "wechat.db")
    stop = asyncio.Event()
    task = asyncio.create_task(
        WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap(),
            handler_concurrency=1,
        ).run(BlockingHandler(), stop)
    )
    try:
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
        except TimeoutError as exc:
            pending = await store.get_pending_ai_work(
                CONNECTOR_KEY,
                CHAT_ID,
                MESSAGE_ID,
            )
            raise AssertionError(
                f"handler did not start; pending={pending}; "
                f"requests={client.requests}; pump_done={task.done()}"
            ) from exc
        await wait_until(
            _async_predicate_from_async(
                store.get_cursor,
                CONNECTOR_KEY,
                expected="event-12",
            )
        )
        assert not release.is_set()
        release.set()
    finally:
        release.set()
        stop.set()
        result = await asyncio.wait_for(task, timeout=1)
        assert result == "stopped"
        await store.close()


@pytest.mark.asyncio
async def test_chat_directory_refresh_runs_after_cursor_advance(tmp_path) -> None:
    chat_event = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "event-12",
            "event": "chat",
            "connectionGeneration": 41,
            "id": CHAT_ID,
        }
    )
    renamed_chats = WeChatChatList(
        chats=(
            WeChatChat(
                id=CHAT_ID,
                type="group",
                display_name="Renamed group",
            ),
        ),
        snapshot=WeChatChatSnapshot(
            id="snapshot-42",
            complete=True,
            current=True,
            count=1,
            cursor="event-12",
            connection_generation=41,
        ),
        cursor="event-12",
    )

    class RefreshClient(StreamingObservedClient):
        async def get_chats(self):
            return renamed_chats

    client = RefreshClient((chat_event,), [])
    store = await open_store(tmp_path / "wechat.db")
    stop = asyncio.Event()
    task = asyncio.create_task(
        WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap(),
            handler_concurrency=1,
        ).run(RecordingHandler(), stop)
    )

    async def directory_refreshed() -> bool:
        chat = await store.get_chat(CONNECTOR_KEY, CHAT_ID)
        return chat is not None and chat.display_name == "Renamed group"

    try:
        await wait_until(directory_refreshed)
        assert await store.get_cursor(CONNECTOR_KEY) == "event-12"
    finally:
        stop.set()
        assert await asyncio.wait_for(task, timeout=1) == "stopped"
        await store.close()


@pytest.mark.asyncio
async def test_worker_rechecks_queue_after_clearing_a_consumed_wakeup(
    tmp_path,
) -> None:
    store = await open_store(tmp_path / "wechat.db")
    pump = WeChatEventPump(
        StreamingObservedClient((), []),
        store,
        CONNECTOR_KEY,
        bootstrap(),
        handler_concurrency=1,
    )
    rechecked = asyncio.Event()

    class WakeupDuringIdleWorker:
        calls = 0

        async def process_one(self, _handler):
            self.calls += 1
            if self.calls == 1:
                pump._work_available.set()
                return "idle"
            rechecked.set()
            await asyncio.Future()

    task = asyncio.create_task(
        pump._run_worker(WakeupDuringIdleWorker(), RecordingHandler())
    )
    try:
        await asyncio.wait_for(rechecked.wait(), timeout=1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()


@pytest.mark.asyncio
async def test_removal_event_cancels_in_flight_generation_and_resolves_work(
    tmp_path,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    emit_removal = asyncio.Event()

    class BlockingHandler:
        async def handle(self, _message) -> bool:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    removal = WeChatEvent.parse(
        {
            "schemaVersion": "wechat-bridge/v1alpha1",
            "cursor": "event-12",
            "event": "message_remove",
            "connectionGeneration": 41,
            "status": "recalled",
            "chatId": CHAT_ID,
            "id": MESSAGE_ID,
        }
    )

    class RemovalClient(StreamingObservedClient):
        async def events(self, *, after: str):
            self.after_values.append(after)
            yield message_event(cursor="event-11")
            await emit_removal.wait()
            yield removal
            await self.keep_open.wait()

    client = RemovalClient((), [present_message(version="mv1:before-recall")])
    store = await open_store(tmp_path / "wechat.db")
    stop = asyncio.Event()
    task = asyncio.create_task(
        WeChatEventPump(
            client,
            store,
            CONNECTOR_KEY,
            bootstrap(),
            handler_concurrency=1,
        ).run(BlockingHandler(), stop)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        emit_removal.set()
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        await wait_until(
            _async_predicate_from_async(
                store.get_cursor,
                CONNECTOR_KEY,
                expected="event-12",
            )
        )

        async def removal_finished() -> bool:
            return await store.get_pending_ai_work(
                CONNECTOR_KEY,
                CHAT_ID,
                MESSAGE_ID,
            ) is None

        await wait_until(removal_finished)
        assert await store.get_processed_revision_status(
            CONNECTOR_KEY,
            CHAT_ID,
            MESSAGE_ID,
            "mv1:before-recall",
        ) == "failed_unknown"
    finally:
        emit_removal.set()
        stop.set()
        assert await asyncio.wait_for(task, timeout=1) == "stopped"
        await store.close()


def _async_predicate(predicate):
    async def evaluate() -> bool:
        return bool(predicate())

    return evaluate


def _async_predicate_from_async(function, *args, expected):
    async def evaluate() -> bool:
        return await function(*args) == expected

    return evaluate
