from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
import time
from typing import Literal, Protocol

import aiohttp

from sidekick.ai import ReplyTarget
from sidekick.inbound import (
    DurableInboundWorker,
    InboundMessageHandler,
    InboundSourceRevision,
    InboundSourceUnavailable,
    InboundWork,
)
from sidekick.wechat.api import (
    WeChatAPIError,
    WeChatAPIContractError,
    WeChatCapabilities,
    WeChatChatList,
    WeChatEvent,
    WeChatObservedMessage,
    WeChatSession,
)
from sidekick.wechat.message import WeChatMessage
from sidekick.wechat.store import WeChatStateRepository


PumpResult = Literal["stopped", "reconnect", "rebootstrap"]
_LOGGER = logging.getLogger(__name__)


class WeChatObservedMessageClient(Protocol):
    async def get_observed_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> WeChatObservedMessage: ...


class WeChatObservedMessageSource:
    def __init__(
        self,
        client: WeChatObservedMessageClient,
        store: WeChatStateRepository,
        connector_key: str,
        *,
        not_observed_attempts: int = 3,
    ) -> None:
        if not_observed_attempts < 1:
            raise ValueError("WeChat not-observed attempts must be positive")
        self._client = client
        self._store = store
        self._connector_key = connector_key
        self._not_observed_attempts = not_observed_attempts

    async def fetch(
        self,
        work: InboundWork,
    ) -> InboundSourceRevision[WeChatObservedMessage]:
        if not isinstance(work.chat_id, str) or not isinstance(work.message_id, str):
            raise ValueError("WeChat work requires string message identity")
        try:
            observed = await self._client.get_observed_message(
                work.chat_id,
                work.message_id,
            )
        except WeChatAPIError as exc:
            if exc.status == 404 and exc.code == "MESSAGE_NOT_OBSERVED":
                max_attempts = self._not_observed_attempts
            elif exc.status == 503 and exc.code == "MESSAGE_HISTORY_NOT_READY":
                max_attempts = None
            else:
                max_attempts = 1
            raise InboundSourceUnavailable(
                exc.code,
                max_attempts=max_attempts,
            ) from exc
        except (TimeoutError, ConnectionError, aiohttp.ClientError) as exc:
            raise InboundSourceUnavailable(
                type(exc).__name__,
                max_attempts=None,
            ) from exc

        if observed.state == "recalled":
            return InboundSourceRevision(
                version=observed.version,
                state="recalled",
            )
        return InboundSourceRevision(
            version=observed.version,
            state="present",
            payload=observed,
        )

    async def materialize(
        self,
        observed: WeChatObservedMessage,
    ) -> ReplyTarget | None:
        message = await self._store.message_from_observation(
            self._connector_key,
            observed,
        )
        return message if _dispatchable(message) else None


class WeChatBootstrapClient(Protocol):
    async def get_session(self) -> WeChatSession: ...

    async def get_capabilities(self) -> WeChatCapabilities: ...

    async def get_chats(self) -> WeChatChatList: ...

    def events(self, *, after: str) -> AsyncIterator[WeChatEvent]: ...


@dataclass(frozen=True, slots=True)
class WeChatBootstrap:
    session: WeChatSession
    capabilities: WeChatCapabilities
    chats: WeChatChatList


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
    await store.bootstrap(
        connector_key=connector_key,
        session=session,
        chats=chats,
    )
    return WeChatBootstrap(
        session=session,
        capabilities=capabilities,
        chats=chats,
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
        self._work_available = asyncio.Event()
        self._directory_tasks: set[asyncio.Task[None]] = set()

    async def run(
        self,
        handler: InboundMessageHandler,
        stop: asyncio.Event,
    ) -> PumpResult:
        await self._store.recover_pending_ai_work(self._connector_key)
        source = WeChatObservedMessageSource(
            self._client,
            self._store,
            self._connector_key,
        )
        worker = DurableInboundWorker(
            source,
            self._store,
            self._connector_key,
            logger=_LOGGER,
        )
        worker_tasks = tuple(
            asyncio.create_task(
                self._run_worker(worker, handler),
                name=f"wechat-ai-worker-{index}",
            )
            for index in range(self._handler_concurrency)
        )
        self._work_available.set()
        after = await self._store.get_cursor(self._connector_key)
        stream = self._client.events(after=after)
        iterator = stream.__aiter__()
        next_event: asyncio.Task[WeChatEvent] | None = None
        stopped = asyncio.create_task(stop.wait())
        try:
            while True:
                if stop.is_set():
                    return "stopped"
                if next_event is None:
                    next_event = asyncio.create_task(iterator.__anext__())
                done, _ = await asyncio.wait(
                    {stopped, next_event},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stopped in done:
                    return "stopped"
                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    return "reconnect"
                next_event = None
                if await self._accept_event(event, worker):
                    return "rebootstrap"
        finally:
            if next_event is not None:
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            for task in self._directory_tasks:
                task.cancel()
            if self._directory_tasks:
                await asyncio.gather(
                    *self._directory_tasks,
                    return_exceptions=True,
                )
            self._directory_tasks.clear()
            stopped.cancel()
            await asyncio.gather(stopped, return_exceptions=True)
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()

    async def _accept_event(
        self,
        event: WeChatEvent,
        worker: DurableInboundWorker[WeChatObservedMessage],
    ) -> bool:
        generation = self._bootstrap.session.connection_generation
        if (
            event.connection_generation is not None
            and event.connection_generation != generation
        ):
            if event.connection_generation < generation:
                await self._store.acknowledge_event(
                    self._connector_key,
                    event.cursor,
                )
            return True

        if event.name == "hook_connection":
            await self._store.acknowledge_event(
                self._connector_key,
                event.cursor,
            )
            return event.payload.get("status") == "disconnected"

        if event.name == "session_update":
            await self._store.acknowledge_event(
                self._connector_key,
                event.cursor,
            )
            return True

        if event.name == "message":
            if _is_inconsistent_shared_chat_history(event):
                _LOGGER.warning(
                    "Dropped inconsistent WeChat shared chat history event",
                    extra={
                        "wechat_event_cursor": event.cursor,
                        "wechat_message_id": event.payload.get("id"),
                        "wechat_message_type": event.payload.get("messageType"),
                    },
                )
                await self._store.acknowledge_event(
                    self._connector_key,
                    event.cursor,
                )
                return False
            if event.is_senderless_unsupported_message():
                await self._store.acknowledge_event(
                    self._connector_key,
                    event.cursor,
                )
                return False
            try:
                message = event.message()
            except WeChatAPIContractError:
                self._log_dropped_message(event)
                await self._store.acknowledge_event(
                    self._connector_key,
                    event.cursor,
                )
                return False
            await self._store.accept_pending_ai_event(
                self._connector_key,
                cursor=event.cursor,
                chat_id=message.chat_id,
                message_id=message.id,
                kind="message",
            )
            self._work_available.set()
            return False

        if event.name == "message_remove":
            try:
                chat_id, message_id = event.removed_message()
            except WeChatAPIContractError:
                self._log_dropped_message(event)
                await self._store.acknowledge_event(
                    self._connector_key,
                    event.cursor,
                )
                return False
            await self._store.accept_pending_ai_event(
                self._connector_key,
                cursor=event.cursor,
                chat_id=chat_id,
                message_id=message_id,
                kind="message_remove",
            )
            worker.cancel_message(chat_id, message_id)
            self._work_available.set()
            return False

        await self._store.acknowledge_event(
            self._connector_key,
            event.cursor,
        )
        self._schedule_directory_refresh(event, generation)
        return False

    async def _run_worker(
        self,
        worker: DurableInboundWorker[WeChatObservedMessage],
        handler: InboundMessageHandler,
    ) -> None:
        while True:
            self._work_available.clear()
            processing = asyncio.create_task(worker.process_one(handler))
            try:
                result = await processing
            except asyncio.CancelledError:
                processing.cancel()
                await asyncio.gather(processing, return_exceptions=True)
                raise
            except Exception as exc:
                _LOGGER.error(
                    "WeChat pending AI worker failed (%s)",
                    type(exc).__name__,
                    exc_info=True,
                )
                result = "idle"
            if result != "idle":
                continue
            next_attempt_at = await self._store.next_pending_ai_work_at(
                self._connector_key
            )
            timeout = (
                max(0.0, next_attempt_at - time.time())
                if next_attempt_at is not None
                else None
            )
            if timeout == 0:
                continue
            try:
                await asyncio.wait_for(
                    self._work_available.wait(),
                    timeout=timeout,
                )
            except TimeoutError:
                pass

    def _schedule_directory_refresh(
        self,
        event: WeChatEvent,
        generation: int,
    ) -> None:
        if event.name not in {"chat", "chat_snapshot"}:
            return
        task = asyncio.create_task(
            self._refresh_directory_event(event, generation),
            name=f"wechat-directory-refresh-{event.cursor}",
        )
        self._directory_tasks.add(task)
        task.add_done_callback(self._directory_tasks.discard)

    async def _refresh_directory_event(
        self,
        event: WeChatEvent,
        generation: int,
    ) -> None:
        try:
            await self._refresh_chats(generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOGGER.warning(
                "WeChat directory refresh deferred (%s; cursor=%s)",
                type(exc).__name__,
                event.cursor,
            )

    @staticmethod
    def _log_dropped_message(event: WeChatEvent) -> None:
        _LOGGER.warning(
            "Dropped malformed WeChat message event",
            extra={
                "wechat_event_cursor": event.cursor,
                "wechat_message_id": event.payload.get("id"),
                "wechat_message_type": event.payload.get("messageType"),
            },
        )

    async def _refresh_chats(self, generation: int) -> None:
        chats = await self._client.get_chats()
        chats.require_current(generation)
        await self._store.refresh_chats(self._connector_key, chats)


def _dispatchable(message: WeChatMessage) -> bool:
    return (
        not message.content_redacted
        and bool(message.raw_text.strip())
        and (
            message.message_type in {"text", "chat_history"}
            or (
                message.message_type == "app"
                and message.reply_to_msg_id is not None
            )
        )
    )


def _is_inconsistent_shared_chat_history(event: WeChatEvent) -> bool:
    return (
        event.name == "message"
        and event.payload.get("sharedChatHistory") is not None
        and event.payload.get("messageType") != "chat_history"
    )
