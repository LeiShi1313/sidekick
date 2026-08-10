from __future__ import annotations

from io import BytesIO
from typing import Any, Literal, Protocol

from sidekick.chat.attachments import OutboundAttachment
from sidekick.chat.identity import ExternalId
from sidekick.chat.provenance import (
    GeneratedMessageTracker,
    MessageOrigin,
    message_fingerprint,
    observed_message_fingerprint,
)


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

    async def reply_attachment(
        self,
        message: Any,
        attachment: OutboundAttachment,
    ) -> SentMessage | None: ...

    async def update(
        self,
        message: SentMessage,
        text: str,
        *,
        presentation: ChatPresentation,
        wait: bool,
    ) -> bool: ...

    async def delete(self, message: Any) -> None: ...

    async def classify_origin(self, message: Any) -> MessageOrigin:
        """Attest origin; MANUAL_OUTGOING must mean authenticated local input."""
        ...

    def is_group(self, message: Any) -> bool: ...


class ObjectChatTransport:
    """Adapter for SDK message objects exposing reply/edit/delete methods."""

    def __init__(self) -> None:
        self._generated_messages = GeneratedMessageTracker()

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
        fingerprint = message_fingerprint(
            text=text,
            reply_to_message_id=message.id,
            has_attachment=False,
        )
        with self._generated_messages.reserve(
            message.chat_id,
            fingerprint,
        ) as reservation:
            sent = await operation(text)
            reservation.confirm(sent.id)
            return sent

    async def reply_attachment(
        self,
        message: Any,
        attachment: OutboundAttachment,
    ) -> SentMessage | None:
        operation = getattr(message, "reply", None)
        if not callable(operation):
            raise RuntimeError("Chat transport cannot reply to this message")
        upload = BytesIO(attachment.data)
        upload.name = attachment.filename
        try:
            fingerprint = message_fingerprint(
                text=None,
                reply_to_message_id=message.id,
                has_attachment=True,
            )
            with self._generated_messages.reserve(
                message.chat_id,
                fingerprint,
            ) as reservation:
                sent = await operation(
                    file=upload,
                    force_document=attachment.display_as == "file",
                )
                if sent is not None:
                    reservation.confirm(sent.id)
                else:
                    reservation.uncertain()
                return sent
        finally:
            upload.close()

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

    async def classify_origin(self, message: Any) -> MessageOrigin:
        outgoing = getattr(message, "is_outgoing", None)
        if outgoing is None:
            outgoing = getattr(message, "out", False)
        self_id = getattr(message, "self_id", None)
        if self_id is not None and getattr(message, "sender_id", None) != self_id:
            outgoing = False
        return await self._generated_messages.classify(
            chat_id=message.chat_id,
            message_id=message.id,
            outgoing=bool(outgoing),
            fingerprint=observed_message_fingerprint(message),
        )

    def is_group(self, message: Any) -> bool:
        group = getattr(message, "is_group", None)
        if group is not None:
            return bool(group)
        return getattr(message, "message_type", None) == "group"
