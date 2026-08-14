from __future__ import annotations

import asyncio
import aiohttp
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Literal, Protocol

from sidekick.ai import ReplyTarget
from sidekick.wechat.api import (
    WeChatAPIError,
    WeChatAPIContractError,
    WeChatCapabilities,
    WeChatChatList,
    WeChatEvent,
    WeChatGroupMemberList,
    WeChatMessageList,
    WeChatObservedMessage,
    WeChatSession,
    WeChatUser,
    WeChatUserList,
)
from sidekick.wechat.message import WeChatMessage
from sidekick.wechat.store import WeChatStateRepository
from sidekick.wechat.store import WeChatPendingAIWork


PumpResult = Literal["stopped", "reconnect", "rebootstrap"]
_IDENTITY_DIRECTORY_CONCURRENCY = 8
_LOGGER = logging.getLogger(__name__)


class WeChatInboundHandler(Protocol):
    async def handle(self, message: ReplyTarget) -> bool: ...


class WeChatObservedMessageClient(Protocol):
    async def get_observed_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> WeChatObservedMessage: ...


WorkerResult = Literal[
    "idle",
    "completed",
    "ignored",
    "duplicate",
    "recalled",
    "deferred",
    "unavailable",
    "stale",
    "failed",
]


class WeChatPendingAIWorker:
    RETRY_BASE_SECONDS = 2.0
    RETRY_MAX_SECONDS = 300.0

    def __init__(
        self,
        client: WeChatObservedMessageClient,
        store: WeChatStateRepository,
        connector_key: str,
        *,
        not_observed_attempts: int = 3,
        clock: Callable[[], float] = time.time,
        logger: Any | None = None,
    ):
        if not_observed_attempts < 1:
            raise ValueError("WeChat not-observed attempts must be positive")
        self._client = client
        self._store = store
        self._connector_key = connector_key
        self._not_observed_attempts = not_observed_attempts
        self._clock = clock
        self._logger = logger

    async def process_one(self, handler: WeChatInboundHandler) -> WorkerResult:
        work = await self._store.claim_pending_ai_work(
            self._connector_key,
            now=self._clock(),
        )
        if work is None:
            return "idle"
        if work.kind == "message_remove":
            resolved = await self._store.resolve_pending_ai_removal(work)
            return "recalled" if resolved else "stale"

        execution_version: str | None = None
        try:
            try:
                observed = await self._client.get_observed_message(
                    work.chat_id,
                    work.message_id,
                )
            except WeChatAPIError as exc:
                return await self._handle_api_error(work, exc)
            except (TimeoutError, ConnectionError, aiohttp.ClientError) as exc:
                return await self._defer(
                    work,
                    error_code=type(exc).__name__,
                    max_attempts=None,
                )

            begin = await self._store.begin_pending_ai_execution(
                work,
                version=observed.version,
                now=self._clock(),
            )
            if begin == "stale":
                return "stale"
            if begin == "duplicate":
                await self._store.complete_pending_ai_work(
                    work,
                    version=observed.version,
                    outcome="ignored",
                    now=self._clock(),
                )
                return "duplicate"
            execution_version = observed.version
            if observed.state == "recalled":
                await self._store.complete_pending_ai_work(
                    work,
                    version=observed.version,
                    outcome="recalled",
                    now=self._clock(),
                )
                return "recalled"

            try:
                message = await self._store.message_from_observation(
                    self._connector_key,
                    observed,
                )
            except Exception as exc:
                await self._store.complete_pending_ai_work(
                    work,
                    version=observed.version,
                    outcome="failed",
                    now=self._clock(),
                )
                self._log_failure("message context", exc, work)
                return "failed"
            if not _dispatchable(message):
                await self._store.complete_pending_ai_work(
                    work,
                    version=observed.version,
                    outcome="ignored",
                    now=self._clock(),
                )
                return "ignored"
            try:
                handled = await handler.handle(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._store.complete_pending_ai_work(
                    work,
                    version=observed.version,
                    outcome="failed",
                    now=self._clock(),
                )
                self._log_failure("handler", exc, work)
                return "failed"
            outcome: Literal["completed", "ignored"] = (
                "completed" if handled else "ignored"
            )
            await self._store.complete_pending_ai_work(
                work,
                version=observed.version,
                outcome=outcome,
                now=self._clock(),
            )
            return outcome
        except asyncio.CancelledError:
            if execution_version is None:
                cleanup = self._store.release_pending_ai_work(work)
            else:
                cleanup = self._store.mark_pending_ai_execution_unknown(
                    work,
                    version=execution_version,
                    now=self._clock(),
                )
            await asyncio.shield(cleanup)
            raise

    async def _handle_api_error(
        self,
        work: WeChatPendingAIWork,
        exc: WeChatAPIError,
    ) -> WorkerResult:
        if exc.status == 404 and exc.code == "MESSAGE_NOT_OBSERVED":
            return await self._defer(
                work,
                error_code=exc.code,
                max_attempts=self._not_observed_attempts,
            )
        if exc.status == 503 and exc.code == "MESSAGE_HISTORY_NOT_READY":
            return await self._defer(
                work,
                error_code=exc.code,
                max_attempts=None,
            )
        return await self._defer(
            work,
            error_code=exc.code,
            max_attempts=1,
        )

    async def _defer(
        self,
        work: WeChatPendingAIWork,
        *,
        error_code: str,
        max_attempts: int | None,
    ) -> WorkerResult:
        now = self._clock()
        prior_attempts = (
            work.attempt_count if work.last_error_code == error_code else 0
        )
        status = await self._store.defer_pending_ai_work(
            work,
            error_code=error_code,
            retry_at=now + self._retry_delay(prior_attempts),
            max_attempts=max_attempts,
            now=now,
        )
        if status == "stale":
            return "stale"
        return "unavailable" if status == "unavailable" else "deferred"

    @classmethod
    def _retry_delay(cls, attempt_count: int) -> float:
        return min(
            cls.RETRY_MAX_SECONDS,
            cls.RETRY_BASE_SECONDS * (2 ** min(attempt_count, 16)),
        )

    def _log_failure(
        self,
        stage: str,
        exc: Exception,
        work: WeChatPendingAIWork,
    ) -> None:
        if self._logger is not None:
            self._logger.error(
                "WeChat pending AI %s failed (%s; message=%s)",
                stage,
                type(exc).__name__,
                work.message_id,
            )


class WeChatBootstrapClient(Protocol):
    async def get_session(self) -> WeChatSession: ...

    async def get_capabilities(self) -> WeChatCapabilities: ...

    async def get_chats(self) -> WeChatChatList: ...

    async def get_messages(self, *, limit: int) -> WeChatMessageList: ...

    async def get_users(self) -> WeChatUserList: ...

    async def get_user(self, user_id: str) -> WeChatUser | None: ...

    async def get_group_members(self, group_id: str) -> WeChatGroupMemberList: ...

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
    processed_persisted: bool = False
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
    await _hydrate_identity_directories(client, store, connector_key, chats)
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
                    next_event = asyncio.create_task(iterator.__anext__())

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

        if event.name == "user_profile":
            user_id = event.changed_user_id()
            user = await self._client.get_user(user_id)
            await self._store.refresh_user(self._connector_key, user_id, user)
            return _PendingEvent(event=event)

        if event.name in {
            "group_member",
            "group_member_snapshot",
            "group_member_directory",
        }:
            group_id = event.invalidated_group_id()
            if group_id is not None:
                await self._refresh_group_members(
                    group_id,
                    replace_aliases=event.name == "group_member_directory",
                )
            return _PendingEvent(event=event)

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
                return _PendingEvent(event=event)
            if event.is_senderless_unsupported_message():
                return _PendingEvent(event=event)
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
        await self._persist_out_of_order_completions(pending)
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
                    processed_message=(
                        None if current.processed_persisted else processed
                    ),
                )
            pending.popleft()
            if current.rebootstrap:
                return "rebootstrap"
        return None

    async def _persist_out_of_order_completions(
        self,
        pending: deque[_PendingEvent],
    ) -> None:
        for current in tuple(pending)[1:]:
            operation = current.operation
            if (
                operation is None
                or not operation.done()
                or operation.cancelled()
                or current.processed_persisted
                or operation.exception() is not None
            ):
                continue
            message = operation.result()
            if message is None:
                continue
            await self._store.mark_processed_identity(
                self._connector_key,
                message.account_id,
                message.chat_id,
                message.id,
            )
            current.processed_persisted = True

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

    async def _refresh_group_members(
        self,
        group_id: str,
        *,
        replace_aliases: bool = False,
    ) -> None:
        members = await self._client.get_group_members(group_id)
        await self._store.refresh_group_members(
            self._connector_key,
            group_id,
            members,
            replace_aliases=replace_aliases,
        )


async def _hydrate_identity_directories(
    client: WeChatBootstrapClient,
    store: WeChatStateRepository,
    connector_key: str,
    chats: WeChatChatList,
) -> None:
    semaphore = asyncio.Semaphore(_IDENTITY_DIRECTORY_CONCURRENCY)

    async def read_users() -> WeChatUserList | None:
        try:
            return await client.get_users()
        except Exception:
            _LOGGER.warning(
                "Could not refresh the optional WeChat user directory",
                exc_info=True,
            )
            return None

    async def read_members(group_id: str) -> WeChatGroupMemberList | None:
        async with semaphore:
            try:
                return await client.get_group_members(group_id)
            except Exception:
                _LOGGER.warning(
                    "Could not refresh the optional WeChat group member directory",
                    extra={"wechat_group_id": group_id},
                    exc_info=True,
                )
                return None

    users_task = asyncio.create_task(read_users())
    group_ids = tuple(chat.id for chat in chats.chats if chat.type == "group")
    member_tasks = tuple(
        asyncio.create_task(read_members(group_id)) for group_id in group_ids
    )
    directory_results = await asyncio.gather(users_task, *member_tasks)
    users = directory_results[0]
    member_lists = directory_results[1:]

    if users is not None:
        await store.refresh_users(connector_key, users)
    for group_id, members in zip(group_ids, member_lists, strict=True):
        if members is not None:
            await store.refresh_group_members(connector_key, group_id, members)


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
