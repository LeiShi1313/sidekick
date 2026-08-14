from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class WeChatMessage:
    connector_key: str
    account_id: str
    memory_cursor: int | str
    id: str
    chat_id: str
    raw_text: str
    content_redacted: bool
    sender_id: str
    reply_to_msg_id: str | None
    date: datetime
    out: bool
    self_id: str
    message_type: str
    chat_type: str
    sender_display_name: str | None
    scope_display_name: str | None
    source: str | None
    sequence: str | None
    media_id: str | None = None
    file: None = None
    post: bool = False
    grouped_id: None = None

    @classmethod
    def from_row(cls, values: Any) -> WeChatMessage:
        return cls(
            connector_key=str(values["connector_key"]),
            account_id=str(values["account_id"]),
            memory_cursor=int(values["memory_order"]),
            id=str(values["message_id"]),
            chat_id=str(values["chat_id"]),
            raw_text=str(values["content"]),
            content_redacted=bool(values["content_redacted"]),
            sender_id=str(values["sender_id"]),
            reply_to_msg_id=(
                str(values["reply_to_message_id"])
                if values["reply_to_message_id"] is not None
                else None
            ),
            date=datetime.fromtimestamp(int(values["timestamp"]), UTC),
            out=str(values["direction"]) == "out",
            self_id=str(values["self_id"]),
            message_type=str(values["message_type"]),
            chat_type=str(values["chat_type"]),
            sender_display_name=str(values["sender_display_name"]),
            scope_display_name=(
                str(values["chat_display_name"])
                if values["chat_display_name"] is not None
                else None
            ),
            source=(str(values["source"]) if values["source"] is not None else None),
            sequence=(
                str(values["sequence"]) if values["sequence"] is not None else None
            ),
            media_id=(
                str(values["media_id"]) if values["media_id"] is not None else None
            ),
        )

    @property
    def is_outgoing(self) -> bool:
        return self.out

    @property
    def entities(self) -> tuple[object, ...]:
        return ()
