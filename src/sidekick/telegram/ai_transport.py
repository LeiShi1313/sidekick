from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from telethon.errors import FloodWaitError, MessageNotModifiedError
from telethon.extensions import markdown as telegram_markdown
from telethon.tl import functions as telegram_functions
from telethon.tl import types as telegram_types

from sidekick.chat.formatting import (
    PORTABLE_LINK_RE,
    has_streamable_markdown_content,
    sanitize_rich_markdown,
)
from sidekick.chat.transport import ChatPresentation, SentMessage


TelegramResponseFormat = Literal["regular_entities", "rich_markdown"]

_TELEGRAM_MARKDOWN_DELIMITERS = {
    "```": telegram_types.MessageEntityPre,
    "**": telegram_types.MessageEntityBold,
    "~~": telegram_types.MessageEntityStrike,
    "*": telegram_types.MessageEntityItalic,
    "`": telegram_types.MessageEntityCode,
}

_COLLAPSE_AFTER_CHARS = 700
_COLLAPSE_AFTER_NEWLINES = 10


@dataclass(frozen=True, slots=True)
class _TelegramUpdateSnapshot:
    text: str
    presentation: ChatPresentation


@dataclass(slots=True)
class _TelegramUpdateState:
    message: SentMessage
    pending: _TelegramUpdateSnapshot | None = None
    published: _TelegramUpdateSnapshot | None = None
    waiting_for_first_agent_update: bool = True
    buffered_agent_text: str | None = None
    initial_timer: asyncio.Task[None] | None = None
    worker: asyncio.Task[None] | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    error: Exception | None = None
    active: bool = True


def select_telegram_response_format(
    *,
    is_bot_account: bool,
    rich_messages_available: bool,
) -> TelegramResponseFormat:
    if is_bot_account and rich_messages_available:
        return "rich_markdown"
    return "regular_entities"


class TelegramEditLimiter:
    def __init__(
        self,
        minimum_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        logger: Any | None = None,
    ):
        self._minimum_interval = max(0.0, minimum_interval)
        self._clock = clock
        self._sleep = sleep or asyncio.sleep
        self._logger = logger
        self._lock = asyncio.Lock()
        self._next_edit_at = 0.0

    async def run(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        wait: bool,
    ) -> bool:
        if not wait and self._lock.locked():
            return False

        async with self._lock:
            while True:
                now = self._clock()
                delay = self._next_edit_at - now
                if delay > 0:
                    if not wait:
                        return False
                    await self._sleep(delay)
                    now = self._next_edit_at
                    self._next_edit_at = 0.0

                try:
                    await operation()
                except FloodWaitError as exc:
                    seconds = max(0.0, float(exc.seconds))
                    self._next_edit_at = now + seconds
                    if self._logger is not None:
                        self._logger.warning(
                            "Telegram edit rate limited; waiting %.0f seconds",
                            seconds,
                        )
                    if not wait:
                        return False
                    continue
                except MessageNotModifiedError:
                    self._next_edit_at = now + self._minimum_interval
                    return True
                except Exception:
                    self._next_edit_at = now + self._minimum_interval
                    raise

                self._next_edit_at = now + self._minimum_interval
                return True


class TelegramChatTransport:
    INITIAL_STREAM_CHARS = 100
    INITIAL_STREAM_BOUNDARY_CHARS = 50
    INITIAL_STREAM_DELAY = 1.0

    def __init__(
        self,
        response_format: TelegramResponseFormat = "regular_entities",
        *,
        edit_limiter: TelegramEditLimiter | None = None,
        edit_cadence: float = 4.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        logger: Any | None = None,
    ):
        self._response_format = response_format
        self._sleep = sleep or asyncio.sleep
        self._logger = logger
        self._update_states: dict[int, _TelegramUpdateState] = {}
        self._edit_limiter = edit_limiter or TelegramEditLimiter(
            edit_cadence,
            clock=clock,
            sleep=sleep,
            logger=logger,
        )

    async def get_reply(self, message: Any) -> Any | None:
        operation = getattr(message, "get_reply_message", None)
        return await operation() if callable(operation) else None

    async def reply(
        self,
        message: Any,
        text: str,
        *,
        presentation: ChatPresentation,
    ) -> SentMessage:
        operation = getattr(message, "reply", None)
        if not callable(operation):
            raise RuntimeError("Telegram message cannot be replied to")
        return await operation(text, parse_mode=None)

    async def update(
        self,
        message: SentMessage,
        text: str,
        *,
        presentation: ChatPresentation,
        wait: bool,
    ) -> bool:
        key = id(message)
        state = self._update_states.get(key)
        if state is None:
            state = _TelegramUpdateState(message=message)
            self._update_states[key] = state
        if wait:
            await self._cancel_initial_timer(state)
            if (
                presentation == "agent"
                and not self._rendered_agent_text(text).strip()
            ):
                return False
            snapshot, _ = await self._publish(
                state,
                text,
                presentation,
                wait=True,
            )
            try:
                await self._wait_until_published(state, snapshot)
            finally:
                state.active = False
                if self._update_states.get(key) is state:
                    self._update_states.pop(key, None)
            return True

        if presentation == "plain":
            await self._cancel_initial_timer(state)
            state.waiting_for_first_agent_update = True
            state.buffered_agent_text = None
            _, emitted = await self._publish(
                state,
                text,
                presentation,
                wait=False,
            )
            return emitted

        return await self._offer_agent_update(state, text)

    async def delete(self, message: Any) -> None:
        operation = getattr(message, "delete", None)
        if callable(operation):
            await operation()

    def is_outgoing(self, message: Any) -> bool:
        return bool(getattr(message, "out", False))

    def is_group(self, message: Any) -> bool:
        return bool(getattr(message, "is_group", False))

    async def _offer_agent_update(
        self,
        state: _TelegramUpdateState,
        text: str,
    ) -> bool:
        state.buffered_agent_text = text
        rendered = self._streamable_agent_text(text)
        if not rendered.strip():
            return False
        if not state.waiting_for_first_agent_update:
            _, emitted = await self._publish(
                state,
                text,
                "agent",
                wait=False,
            )
            return emitted

        if self._initial_update_ready(rendered):
            state.waiting_for_first_agent_update = False
            await self._cancel_initial_timer(state)
            _, emitted = await self._publish(
                state,
                text,
                "agent",
                wait=False,
            )
            return emitted

        if state.initial_timer is None:
            state.initial_timer = asyncio.create_task(
                self._release_initial_update(state)
            )
        return False

    def _initial_update_ready(self, rendered: str) -> bool:
        # Telegram does not expose a local renderer for rich-message Markdown.
        # Its source length can be dominated by hidden link targets, so use the
        # time-based release instead of risking a tiny first visible edit.
        if self._response_format == "rich_markdown":
            return False
        content = rendered.strip()
        if len(content) >= self.INITIAL_STREAM_CHARS:
            return True
        if len(content) < self.INITIAL_STREAM_BOUNDARY_CHARS:
            return False
        return (
            rendered.endswith("\n")
            or content.endswith((".", "!", "?", "。", "！", "？"))
        )

    async def _release_initial_update(
        self,
        state: _TelegramUpdateState,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            await self._sleep(self.INITIAL_STREAM_DELAY)
            if state.initial_timer is current_task:
                state.initial_timer = None
            if (
                not state.active
                or not state.waiting_for_first_agent_update
                or state.buffered_agent_text is None
            ):
                return
            text = state.buffered_agent_text
            if not self._streamable_agent_text(text).strip():
                return
            state.waiting_for_first_agent_update = False
            await self._publish(
                state,
                text,
                "agent",
                wait=False,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._record_background_error(state, "initial stream update", exc)
        finally:
            if state.initial_timer is current_task:
                state.initial_timer = None

    async def _cancel_initial_timer(self, state: _TelegramUpdateState) -> None:
        task = state.initial_timer
        state.initial_timer = None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _publish(
        self,
        state: _TelegramUpdateState,
        text: str,
        presentation: ChatPresentation,
        *,
        wait: bool,
    ) -> tuple[_TelegramUpdateSnapshot, bool]:
        snapshot = _TelegramUpdateSnapshot(text, presentation)
        if state.pending == snapshot:
            return snapshot, False
        if state.pending is None and state.published == snapshot:
            return snapshot, True
        state.pending = snapshot

        if state.worker is None:
            state.error = None
            emitted = await self._emit_pending(state, wait=wait)
            if state.pending is not None:
                state.worker = asyncio.create_task(self._drain_updates(state))
            return snapshot, emitted and state.published == snapshot
        return snapshot, False

    async def _drain_updates(
        self,
        state: _TelegramUpdateState,
    ) -> None:
        try:
            while state.active and state.pending is not None:
                await self._emit_pending(state, wait=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_background_error(state, "streaming update", exc)
        finally:
            state.worker = None

    async def _emit_pending(
        self,
        state: _TelegramUpdateState,
        *,
        wait: bool,
    ) -> bool:
        emitted: _TelegramUpdateSnapshot | None = None

        async def edit() -> None:
            nonlocal emitted
            snapshot = state.pending
            if snapshot is None:
                return
            emitted = snapshot
            await self._edit_snapshot(state.message, snapshot)

        applied = await self._edit_limiter.run(edit, wait=wait)
        if not applied or emitted is None:
            return False
        state.published = emitted
        if state.pending == emitted:
            state.pending = None
        self._notify_state_changed(state)
        return True

    async def _wait_until_published(
        self,
        state: _TelegramUpdateState,
        snapshot: _TelegramUpdateSnapshot,
    ) -> None:
        while state.published != snapshot or state.pending is not None:
            if state.error is not None:
                raise state.error
            changed = state.changed
            if state.published == snapshot and state.pending is None:
                return
            await changed.wait()

    @staticmethod
    def _notify_state_changed(state: _TelegramUpdateState) -> None:
        changed = state.changed
        state.changed = asyncio.Event()
        changed.set()

    def _record_background_error(
        self,
        state: _TelegramUpdateState,
        operation: str,
        exc: Exception,
    ) -> None:
        state.error = exc
        self._notify_state_changed(state)
        if self._logger is not None:
            self._logger.exception(
                "Telegram %s failed (%s)",
                operation,
                type(exc).__name__,
            )

    def _rendered_agent_text(self, text: str) -> str:
        if self._response_format == "regular_entities":
            rendered, _ = _parse_agent_markdown(text)
            return rendered
        return text

    def _streamable_agent_text(self, text: str) -> str:
        rendered = self._rendered_agent_text(text)
        return rendered if has_streamable_markdown_content(rendered) else ""

    async def _edit_snapshot(
        self,
        message: SentMessage,
        snapshot: _TelegramUpdateSnapshot,
    ) -> None:
        if snapshot.presentation == "plain":
            await message.edit(snapshot.text, parse_mode=None)
            return
        if self._response_format == "regular_entities":
            rendered, entities = _parse_agent_markdown(snapshot.text)
            if not rendered.strip():
                return
            if (
                len(rendered) > _COLLAPSE_AFTER_CHARS
                or rendered.count("\n") >= _COLLAPSE_AFTER_NEWLINES
            ):
                entities.insert(
                    0,
                    telegram_types.MessageEntityBlockquote(
                        offset=0,
                        length=len(rendered.encode("utf-16-le")) // 2,
                        collapsed=True,
                    ),
                )
            await message.edit(
                rendered,
                parse_mode=None,
                formatting_entities=entities,
            )
            return

        text = sanitize_rich_markdown(snapshot.text)
        if not snapshot.text.strip():
            return
        client = getattr(message, "client", None)
        get_input_chat = getattr(message, "get_input_chat", None)
        if client is None or not callable(get_input_chat):
            raise RuntimeError("Telegram rich-message editing is unavailable")
        peer = await get_input_chat()
        if peer is None:
            raise RuntimeError("Telegram rich-message peer is unavailable")

        await client(
            telegram_functions.messages.EditMessageRequest(
                peer=peer,
                id=message.id,
                rich_message=telegram_types.InputRichMessageMarkdown(markdown=text),
            )
        )


def _parse_agent_markdown(text: str) -> tuple[str, list[Any]]:
    return telegram_markdown.parse(
        text,
        delimiters=_TELEGRAM_MARKDOWN_DELIMITERS,
        url_re=PORTABLE_LINK_RE,
    )
