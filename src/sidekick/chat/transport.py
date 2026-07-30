from __future__ import annotations

from typing import Any, Literal, Protocol

from sidekick.chat.identity import ExternalId


ChatPresentation = Literal["plain", "agent"]


class SentMessage(Protocol):
    id: ExternalId
    text: str | None


class ChatTransport(Protocol):
    async def get_reply(self, message: Any) -> Any | None: ...

    async def reply(
        self,
        message: Any,
        text: str,
        *,
        presentation: ChatPresentation,
    ) -> SentMessage: ...

    async def update(
        self,
        message: SentMessage,
        text: str,
        *,
        presentation: ChatPresentation,
        wait: bool,
    ) -> bool: ...

    async def delete(self, message: Any) -> None: ...

    def is_outgoing(self, message: Any) -> bool: ...

    def is_group(self, message: Any) -> bool: ...


class ObjectChatTransport:
    """Adapter for SDK message objects exposing reply/edit/delete methods."""

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
            raise RuntimeError("Chat transport cannot reply to this message")
        return await operation(text)

    async def update(
        self,
        message: SentMessage,
        text: str,
        *,
        presentation: ChatPresentation,
        wait: bool,
    ) -> bool:
        operation = getattr(message, "edit", None)
        if not callable(operation):
            raise RuntimeError("Chat transport cannot update this message")
        await operation(text)
        return True

    async def delete(self, message: Any) -> None:
        operation = getattr(message, "delete", None)
        if callable(operation):
            await operation()

    def is_outgoing(self, message: Any) -> bool:
        outgoing = getattr(message, "is_outgoing", None)
        if outgoing is None:
            outgoing = getattr(message, "out", False)
        return bool(outgoing)

    def is_group(self, message: Any) -> bool:
        group = getattr(message, "is_group", None)
        if group is not None:
            return bool(group)
        return getattr(message, "message_type", None) == "group"
