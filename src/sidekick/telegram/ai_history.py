from __future__ import annotations

from datetime import datetime
from typing import Any

from telethon.errors import FloodWaitError, RPCError

from sidekick.ai import ReplyTarget, _message_datetime
from sidekick.chat.provenance import observed_message_fingerprint
from sidekick.inbound import (
    InboundSourceRevision,
    InboundSourceUnavailable,
    InboundWork,
)


class TelegramInboundMessageSource:
    def __init__(self, client: Any):
        self._client = client

    async def fetch(
        self,
        work: InboundWork,
    ) -> InboundSourceRevision[Any]:
        if (
            isinstance(work.chat_id, bool)
            or not isinstance(work.chat_id, int)
            or isinstance(work.message_id, bool)
            or not isinstance(work.message_id, int)
        ):
            raise ValueError("Telegram work requires integer message identity")
        try:
            message = await self._client.get_messages(
                work.chat_id,
                ids=work.message_id,
            )
        except FloodWaitError as exc:
            raise InboundSourceUnavailable(
                "TELEGRAM_FLOOD_WAIT",
                max_attempts=None,
                retry_after_seconds=float(exc.seconds),
            ) from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise InboundSourceUnavailable(
                type(exc).__name__,
                max_attempts=None,
            ) from exc
        except RPCError as exc:
            raise InboundSourceUnavailable(
                f"TELEGRAM_{type(exc).__name__}",
                max_attempts=1,
            ) from exc
        if isinstance(message, list):
            message = message[0] if message else None
        if (
            message is None
            or getattr(message, "chat_id", None) != work.chat_id
            or getattr(message, "id", None) != work.message_id
        ):
            raise InboundSourceUnavailable(
                "TELEGRAM_MESSAGE_UNAVAILABLE",
                max_attempts=1,
            )
        edited_at = getattr(message, "edit_date", None)
        edit_version = (
            edited_at.isoformat()
            if isinstance(edited_at, datetime)
            else "original"
        )
        fingerprint = observed_message_fingerprint(message).digest.hex()
        return InboundSourceRevision(
            version=(
                f"telegram:v1:{work.message_id}:{edit_version}:{fingerprint}"
            ),
            state="present",
            payload=message,
            attested_origin=work.attested_origin,
        )

    async def materialize(self, message: Any) -> ReplyTarget | None:
        return message


class TelegramHistorySource:
    def __init__(self, client: Any):
        self._client = client

    async def fetch_recent(
        self,
        trigger: ReplyTarget,
        *,
        before: ReplyTarget,
        limit: int,
    ) -> tuple[ReplyTarget, ...]:
        if trigger.chat_id is None or before.chat_id != trigger.chat_id:
            return ()
        kwargs: dict[str, Any] = {
            "limit": limit,
            "max_id": before.id,
        }
        reply_header = getattr(trigger, "reply_to", None)
        if bool(getattr(reply_header, "forum_topic", False)):
            topic_id = getattr(reply_header, "reply_to_top_id", None) or getattr(
                reply_header,
                "reply_to_msg_id",
                None,
            )
            if isinstance(topic_id, int) and topic_id > 0:
                kwargs["reply_to"] = topic_id
        messages = [
            message
            async for message in self._client.iter_messages(
                trigger.chat_id,
                **kwargs,
            )
        ]
        messages.reverse()
        return tuple(messages)

    async def fetch_window(
        self,
        chat_id: int,
        *,
        since: datetime,
        until: datetime,
        limit: int,
    ) -> tuple[ReplyTarget, ...]:
        messages: list[ReplyTarget] = []
        async for message in self._client.iter_messages(
            chat_id,
            offset_date=until,
            limit=limit,
        ):
            occurred_at = _message_datetime(message)
            if occurred_at < since:
                break
            if occurred_at <= until:
                messages.append(message)
        messages.reverse()
        return tuple(messages)

    async def fetch_message(
        self,
        chat_id: int,
        message_id: int,
    ) -> ReplyTarget | None:
        message = await self._client.get_messages(chat_id, ids=message_id)
        if isinstance(message, list):
            return message[0] if message else None
        return message

    async def fetch_after(
        self,
        chat_id: int,
        *,
        after_message_id: int,
        until: datetime,
        limit: int,
    ) -> tuple[ReplyTarget, ...]:
        messages: list[ReplyTarget] = []
        async for message in self._client.iter_messages(
            chat_id,
            min_id=after_message_id,
            reverse=True,
            limit=limit,
        ):
            if _message_datetime(message) > until:
                break
            messages.append(message)
        return tuple(messages)


def telegram_source_retry_delay(exc: Exception) -> float | None:
    if isinstance(exc, FloodWaitError):
        return max(0.0, float(exc.seconds))
    return None


def telegram_channel_album_document_id(
    chat_id: int,
    message: ReplyTarget,
) -> str | None:
    if not bool(getattr(message, "post", False)):
        return None
    grouped_id = getattr(message, "grouped_id", None)
    if isinstance(grouped_id, int):
        return f"telegram:channel-album:{chat_id}:{grouped_id}"
    return None
