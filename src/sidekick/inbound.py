from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

from sidekick.ai import ReplyTarget
from sidekick.chat.commands import is_ai_candidate_text
from sidekick.chat.identity import ExternalId
from sidekick.chat.provenance import MessageOrigin


InboundWorkKind = Literal["message", "message_remove"]
InboundCompletion = Literal["completed", "ignored", "recalled", "failed"]
InboundExecutionStart = Literal["started", "duplicate", "stale"]
InboundDeferral = Literal["pending", "unavailable", "stale"]
InboundSourceState = Literal["present", "recalled"]


class InboundWork(Protocol):
    chat_id: ExternalId
    message_id: ExternalId
    kind: InboundWorkKind
    attempt_count: int
    last_error_code: str | None
    attested_origin: MessageOrigin | None
    lease_id: str | None


_SourcePayload = TypeVar("_SourcePayload")


@dataclass(frozen=True, slots=True)
class InboundSourceRevision(Generic[_SourcePayload]):
    version: str
    state: InboundSourceState
    payload: _SourcePayload | None = None
    attested_origin: MessageOrigin | None = None

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("Inbound source revision cannot be empty")
        if self.state == "present" and self.payload is None:
            raise ValueError("Present inbound source revision requires a payload")
        if self.state == "recalled" and self.payload is not None:
            raise ValueError("Recalled inbound source revision cannot have a payload")
        if self.attested_origin is not None and not isinstance(
            self.attested_origin,
            MessageOrigin,
        ):
            raise ValueError("Inbound source origin is invalid")


class InboundMessageSource(Protocol[_SourcePayload]):
    async def fetch(
        self,
        work: InboundWork,
    ) -> InboundSourceRevision[_SourcePayload]: ...

    async def materialize(self, payload: _SourcePayload) -> ReplyTarget | None: ...


class InboundSourceUnavailable(Exception):
    def __init__(
        self,
        code: str,
        *,
        max_attempts: int | None,
        retry_after_seconds: float | None = None,
    ) -> None:
        if not code:
            raise ValueError("Inbound source error code cannot be empty")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("Inbound source attempts must be positive")
        if retry_after_seconds is not None and retry_after_seconds < 0:
            raise ValueError("Inbound source retry delay cannot be negative")
        super().__init__(code)
        self.code = code
        self.max_attempts = max_attempts
        self.retry_after_seconds = retry_after_seconds


def is_ai_candidate(message: ReplyTarget) -> bool:
    return is_ai_candidate_text(message.raw_text) or message.reply_to_msg_id is not None
