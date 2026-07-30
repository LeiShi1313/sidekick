from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from sidekick.ai import ReplyTarget
from sidekick.wechat.api import (
    WeChatAPIContractError,
    WeChatCapabilities,
    WeChatChatList,
    WeChatEvent,
    WeChatMessageList,
    WeChatSession,
)
from sidekick.wechat.message import WeChatMessage
from sidekick.wechat.store import WeChatStateRepository


PumpResult = Literal["stopped", "reconnect", "rebootstrap"]


class WeChatInboundHandler(Protocol):
    async def handle(self, message: ReplyTarget) -> bool: ...


class WeChatBootstrapClient(Protocol):
    async def get_session(self) -> WeChatSession: ...

    async def get_capabilities(self) -> WeChatCapabilities: ...

    async def get_chats(self) -> WeChatChatList: ...

    async def get_messages(self, *, limit: int) -> WeChatMessageList: ...

    def events(self, *, after: str) -> AsyncIterator[WeChatEvent]: ...


@dataclass(frozen=True, slots=True)
class WeChatBootstrap:
    session: WeChatSession
    capabilities: WeChatCapabilities
    chats: WeChatChatList
    messages: WeChatMessageList


@dataclass(slots=True)
class _PendingEvent:
    event: WeChatEvent
    operation: asyncio.Task[WeChatMessage | None] | None = None
    commit_cursor: bool = True
    rebootstrap: bool = False


async def bootstrap_wechat_channel(
    client: WeChatBootstrapClient,
    store: WeChatStateRepository,
    connector_key: str,
) -> WeChatBootstrap:
    capabilities = await client.get_capabilities()
    capabilities.require_ai_channel()
    session = await client.get_session()
    session.require_current_login()
    if capabilities.connection_generation != session.connection_generation:
        raise WeChatAPIContractError(
            "WeChat capabilities and session generations do not match"
        )
    chats = await client.get_chats()
    chats.require_current(session.connection_generation)
    messages = await client.get_messages(limit=1_000)
    await store.bootstrap(
        connector_key=connector_key,
        session=session,
        chats=chats,
        messages=messages,
    )
    return WeChatBootstrap(
        session=session,
        capabilities=capabilities,
        chats=chats,
        messages=messages,
    )


class WeChatEventPump:
    def __init__(
        self,
        client: WeChatBootstrapClient,
        store: WeChatStateRepository,
        connector_key: str,
        bootstrap: WeChatBootstrap,
        *,
        handler_concurrency: int = 8,
    ):
        if handler_concurrency < 1:
            raise ValueError("WeChat handler concurrency must be positive")
        self._client = client
        self._store = store
        self._connector_key = connector_key
        self._bootstrap = bootstrap
        self._handler_concurrency = handler_concurrency

    async def run(
        self,
        handler: WeChatInboundHandler,
        stop: asyncio.Event,
    ) -> PumpResult:
        after = await self._store.get_cursor(self._connector_key)
        stream = self._client.events(after=after)
        iterator = stream.__aiter__()
        pending: deque[_PendingEvent] = deque()
        next_event: asyncio.Task[WeChatEvent] | None = None
        stopped = asyncio.create_task(stop.wait())
        stream_ended = False
        try:
            while True:
                result = await self._ack_ready_events(pending)
                if result is not None:
                    return result
                if stop.is_set():
                    return "stopped"
                if stream_ended and not pending:
                    return "reconnect"
                if (
                    next_event is None
                    and not stream_ended
                    and len(pending) < self._handler_concurrency
                ):
                    next_event = asyncio.create_task(anext(iterator))

                wait_for: set[asyncio.Task[object]] = {stopped}
                if next_event is not None:
                    wait_for.add(next_event)
                if pending and pending[0].operation is not None:
                    wait_for.add(pending[0].operation)
                done, _ = await asyncio.wait(
                    wait_for,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stopped in done:
                    return "stopped"
                if next_event is not None and next_event in done:
                    try:
                        event = next_event.result()
                    except StopAsyncIteration:
                        stream_ended = True
                    else:
                        prepared = await self._prepare_event(handler, event)
                        pending.append(prepared)
                        if prepared.rebootstrap:
                            # Do not dispatch anything beyond an account/session
                            # boundary using the old bootstrap state.
                            stream_ended = True
                    next_event = None
        finally:
            if next_event is not None:
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
            operations = [
                item.operation
                for item in pending
                if item.operation is not None and not item.operation.done()
            ]
            for operation in operations:
                operation.cancel()
            if operations:
                await asyncio.gather(*operations, return_exceptions=True)
            stopped.cancel()
            await asyncio.gather(stopped, return_exceptions=True)
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()

    async def _prepare_event(
        self,
        handler: WeChatInboundHandler,
        event: WeChatEvent,
    ) -> _PendingEvent:
        generation = self._bootstrap.session.connection_generation
        if (
            event.connection_generation is not None
            and event.connection_generation != generation
        ):
            return _PendingEvent(
                event=event,
                commit_cursor=event.connection_generation < generation,
                rebootstrap=True,
            )

        if event.name == "hook_connection":
            return _PendingEvent(
                event=event,
                rebootstrap=event.payload.get("status") == "disconnected",
            )

        if event.name == "session_update":
            return _PendingEvent(event=event, rebootstrap=True)

        if event.name in {"chat", "chat_snapshot"}:
            await self._refresh_chats(generation)
            return _PendingEvent(event=event)

        if event.name == "message":
            message = await self._store.project_event(self._connector_key, event)
            if message is None:
                await self._refresh_chats(generation)
                message = await self._store.project_event(self._connector_key, event)
            if message is not None and _dispatchable(message):
                if not await self._store.is_processed(message):
                    return _PendingEvent(
                        event=event,
                        operation=asyncio.create_task(
                            self._handle_message(handler, message)
                        ),
                    )
            return _PendingEvent(event=event)

        if event.name == "message_remove":
            await self._store.project_event(self._connector_key, event)

        return _PendingEvent(event=event)

    async def _ack_ready_events(
        self,
        pending: deque[_PendingEvent],
    ) -> Literal["rebootstrap"] | None:
        while pending:
            current = pending[0]
            operation = current.operation
            if operation is not None and not operation.done():
                return None
            processed = operation.result() if operation is not None else None
            if current.commit_cursor:
                await self._store.acknowledge_event(
                    self._connector_key,
                    current.event.cursor,
                    processed_message=processed,
                )
            pending.popleft()
            if current.rebootstrap:
                return "rebootstrap"
        return None

    @staticmethod
    async def _handle_message(
        handler: WeChatInboundHandler,
        message: WeChatMessage,
    ) -> WeChatMessage:
        await handler.handle(message)
        return message

    async def _refresh_chats(self, generation: int) -> None:
        chats = await self._client.get_chats()
        chats.require_current(generation)
        await self._store.refresh_chats(self._connector_key, chats)


def _dispatchable(message: WeChatMessage) -> bool:
    return (
        message.message_type == "text"
        and not message.content_redacted
        and bool(message.raw_text.strip())
    )
