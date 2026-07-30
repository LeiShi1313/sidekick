from __future__ import annotations

import asyncio
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
    ):
        self._client = client
        self._store = store
        self._connector_key = connector_key
        self._bootstrap = bootstrap

    async def run(
        self,
        handler: WeChatInboundHandler,
        stop: asyncio.Event,
    ) -> PumpResult:
        after = await self._store.get_cursor(self._connector_key)
        stream = self._client.events(after=after)
        iterator = stream.__aiter__()
        try:
            while True:
                next_event = asyncio.create_task(anext(iterator))
                stopped = asyncio.create_task(stop.wait())
                done, _ = await asyncio.wait(
                    {next_event, stopped},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stopped in done:
                    next_event.cancel()
                    await asyncio.gather(next_event, return_exceptions=True)
                    return "stopped"
                stopped.cancel()
                await asyncio.gather(stopped, return_exceptions=True)
                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    return "reconnect"
                result = await self._handle_event(handler, event)
                if result is not None:
                    return result
        finally:
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()

    async def _handle_event(
        self,
        handler: WeChatInboundHandler,
        event: WeChatEvent,
    ) -> Literal["rebootstrap"] | None:
        generation = self._bootstrap.session.connection_generation
        if (
            event.connection_generation is not None
            and event.connection_generation != generation
        ):
            await self._store.acknowledge_event(
                self._connector_key,
                event.cursor,
            )
            return "rebootstrap"

        if event.name == "hook_connection":
            await self._store.acknowledge_event(
                self._connector_key,
                event.cursor,
            )
            if event.payload.get("status") == "disconnected":
                return "rebootstrap"
            return None

        if event.name == "session_update":
            await self._store.acknowledge_event(
                self._connector_key,
                event.cursor,
            )
            return "rebootstrap"

        if event.name in {"chat", "chat_snapshot"}:
            await self._refresh_chats(generation)
            await self._store.acknowledge_event(
                self._connector_key,
                event.cursor,
            )
            return None

        if event.name == "message":
            message = await self._store.project_event(self._connector_key, event)
            if message is None:
                await self._refresh_chats(generation)
                message = await self._store.project_event(self._connector_key, event)
            processed = None
            if message is not None and _dispatchable(message):
                if not await self._store.is_processed(message):
                    await handler.handle(message)
                    processed = message
            await self._store.acknowledge_event(
                self._connector_key,
                event.cursor,
                processed_message=processed,
            )
            return None

        if event.name == "message_remove":
            await self._store.project_event(self._connector_key, event)

        await self._store.acknowledge_event(
            self._connector_key,
            event.cursor,
        )
        return None

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
