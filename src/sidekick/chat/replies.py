from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sidekick.chat.identity import ExternalId


@dataclass(frozen=True, slots=True)
class ExternalReply:
    """What is known about a message replied to from another chat."""

    origin: str | None = None
    text: str | None = None


def is_cross_chat_reply(message: Any) -> bool:
    """Return whether a message replies to a message stored in another chat.

    Telegram "external replies" carry the source chat in the reply header
    (``reply_to_peer_id``) or attribute the original sender (``reply_from``);
    their ``reply_to_msg_id`` is an ID in that other chat, not in this one.
    """
    header = getattr(message, "reply_to", None)
    if header is None or getattr(message, "reply_to_msg_id", None) is None:
        return False
    if getattr(header, "reply_from", None) is not None:
        return True
    peer = getattr(header, "reply_to_peer_id", None)
    if peer is None:
        return False
    peer_id = _marked_peer_id(peer)
    return peer_id is None or peer_id != getattr(message, "chat_id", None)


def same_chat_reply_id(message: Any) -> ExternalId | None:
    """Return the replied-to message ID only when it belongs to this chat."""
    if is_cross_chat_reply(message):
        return None
    return getattr(message, "reply_to_msg_id", None)


def _marked_peer_id(peer: Any) -> int | None:
    # Mirrors Telegram's marked IDs, which is what Telethon exposes as chat_id.
    channel_id = getattr(peer, "channel_id", None)
    if isinstance(channel_id, int):
        return -(10**12 + channel_id)
    chat_id = getattr(peer, "chat_id", None)
    if isinstance(chat_id, int):
        return -chat_id
    user_id = getattr(peer, "user_id", None)
    if isinstance(user_id, int):
        return user_id
    return None
