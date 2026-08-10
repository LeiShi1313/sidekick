from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import time
from types import TracebackType
from typing import Any

from sidekick.chat.identity import ExternalId


class MessageOrigin(Enum):
    """Transport-attested origin; MANUAL_OUTGOING authorizes local controls."""

    INCOMING = "incoming"
    MANUAL_OUTGOING = "manual-outgoing"
    SIDEKICK_GENERATED = "sidekick-generated"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class MessageFingerprint:
    digest: bytes


def message_fingerprint(
    *,
    text: str | None,
    reply_to_message_id: ExternalId | None,
    has_attachment: bool,
) -> MessageFingerprint:
    if text is not None and not isinstance(text, str):
        raise ValueError("Generated-message text must be a string or None")
    if reply_to_message_id is not None:
        _validate_external_id(reply_to_message_id, "reply message ID")
    if not isinstance(has_attachment, bool):
        raise ValueError("Generated-message attachment flag must be boolean")
    reply = (
        None
        if reply_to_message_id is None
        else {
            "type": type(reply_to_message_id).__name__,
            "value": str(reply_to_message_id),
        }
    )
    encoded = json.dumps(
        {
            "attachment": has_attachment,
            "reply": reply,
            "text": (text or "").strip(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return MessageFingerprint(sha256(encoded).digest())


def observed_message_fingerprint(message: Any) -> MessageFingerprint:
    text = getattr(message, "raw_text", None)
    if text is None:
        text = getattr(message, "text", None)
    message_type = getattr(message, "message_type", None)
    has_attachment = (
        getattr(message, "file", None) is not None
        or getattr(message, "media", None) is not None
        or getattr(message, "media_id", None) is not None
        or message_type
        in {"audio", "document", "file", "image", "sticker", "video", "voice"}
    )
    return message_fingerprint(
        text=text if isinstance(text, str) else None,
        reply_to_message_id=getattr(message, "reply_to_msg_id", None),
        has_attachment=has_attachment,
    )


class GeneratedMessageReservation:
    def __init__(
        self,
        tracker: GeneratedMessageTracker,
        chat_id: ExternalId,
        fingerprint: MessageFingerprint,
    ) -> None:
        self._tracker = tracker
        self.chat_id = chat_id
        self.fingerprint = fingerprint
        self._settled = False
        self._settlement: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )

    def __enter__(self) -> GeneratedMessageReservation:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if not self._settled:
            self.uncertain()

    def confirm(
        self,
        message_id: ExternalId,
        *,
        echo_expected: bool = True,
    ) -> None:
        _validate_external_id(message_id, "message ID")
        self._tracker._settle(
            self,
            message_id=message_id,
            uncertain=False,
            observed=not echo_expected,
        )

    def failed(self) -> None:
        self._tracker._settle(
            self,
            message_id=None,
            uncertain=False,
            observed=False,
        )

    def complete_without_message_id(self) -> None:
        """Settle a successful send whose possible echo cannot trigger work.

        This is safe only when the transport action cannot emit command-bearing
        text or a reply-triggering event. Without a native message ID, a later
        echo cannot be attributed to this reservation.
        """
        self._tracker._settle(
            self,
            message_id=None,
            uncertain=False,
            observed=False,
        )

    def uncertain(self) -> None:
        self._tracker._settle(
            self,
            message_id=None,
            uncertain=True,
            observed=False,
        )


class GeneratedMessageTracker:
    """Classify outbound echoes by exact native IDs across send/event races.

    Unsettled and unknown sends never expire open. Exact IDs move to a bounded
    replay cache only after the native event has been consumed, or when the
    adapter guarantees that its SDK consumes the send update itself.
    """

    def __init__(
        self,
        *,
        max_pending: int = 256,
        max_confirmed: int = 4_096,
        max_uncertain: int = 4_096,
        max_observed: int = 4_096,
        observed_retention: float = 3_600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_pending < 1:
            raise ValueError("Pending generated-message capacity must be positive")
        if max_confirmed < 1:
            raise ValueError("Confirmed generated-message capacity must be positive")
        if max_uncertain < 1:
            raise ValueError("Uncertain generated-message capacity must be positive")
        if max_observed < 1:
            raise ValueError("Observed generated-message capacity must be positive")
        if observed_retention <= 0:
            raise ValueError("Observed generated-message retention must be positive")
        self._max_pending = max_pending
        self._max_confirmed = max_confirmed
        self._max_uncertain = max_uncertain
        self._max_observed = max_observed
        self._observed_retention = observed_retention
        self._clock = clock
        self._pending: dict[ExternalId, set[GeneratedMessageReservation]] = {}
        self._pending_count = 0
        self._confirmed: OrderedDict[tuple[ExternalId, ExternalId], float] = (
            OrderedDict()
        )
        self._observed: OrderedDict[tuple[ExternalId, ExternalId], float] = (
            OrderedDict()
        )
        self._uncertain: OrderedDict[
            tuple[ExternalId, MessageFingerprint], float
        ] = OrderedDict()

    @property
    def indeterminate_count(self) -> int:
        return len(self._uncertain)

    def reserve(
        self,
        chat_id: ExternalId,
        fingerprint: MessageFingerprint,
    ) -> GeneratedMessageReservation:
        _validate_external_id(chat_id, "chat ID")
        if not isinstance(fingerprint, MessageFingerprint):
            raise ValueError("Generated-message fingerprint is invalid")
        self._prune_observed()
        if self._pending_count >= self._max_pending:
            raise RuntimeError("Generated-message pending capacity reached")
        if len(self._confirmed) + self._pending_count >= self._max_confirmed:
            raise RuntimeError("Generated-message confirmed capacity reached")
        candidate = (chat_id, fingerprint)
        pending_candidates = {
            (pending_chat_id, reservation.fingerprint)
            for pending_chat_id, reservations in self._pending.items()
            for reservation in reservations
        }
        if (
            candidate not in self._uncertain
            and candidate not in pending_candidates
            and len(set(self._uncertain).union(pending_candidates))
            >= self._max_uncertain
        ):
            raise RuntimeError("Generated-message uncertain capacity reached")
        reservation = GeneratedMessageReservation(self, chat_id, fingerprint)
        self._pending.setdefault(chat_id, set()).add(reservation)
        self._pending_count += 1
        return reservation

    async def classify(
        self,
        *,
        chat_id: ExternalId,
        message_id: ExternalId,
        outgoing: bool,
        fingerprint: MessageFingerprint,
    ) -> MessageOrigin:
        if not outgoing:
            return MessageOrigin.INCOMING
        _validate_external_id(chat_id, "chat ID")
        _validate_external_id(message_id, "message ID")
        if not isinstance(fingerprint, MessageFingerprint):
            raise ValueError("Generated-message fingerprint is invalid")
        self._prune_observed()
        key = (chat_id, message_id)
        if key in self._observed:
            self._observed.move_to_end(key)
            return MessageOrigin.SIDEKICK_GENERATED
        if self._consume_confirmation(key):
            return MessageOrigin.SIDEKICK_GENERATED

        if self._chat_is_uncertain(chat_id):
            return MessageOrigin.INDETERMINATE
        reservations = tuple(
            reservation
            for reservation in self._pending.get(chat_id, ())
        )
        if not reservations:
            return MessageOrigin.MANUAL_OUTGOING
        pending = {reservation._settlement for reservation in reservations}
        while pending:
            _, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if key in self._observed:
                self._observed.move_to_end(key)
                return MessageOrigin.SIDEKICK_GENERATED
            if self._consume_confirmation(key):
                return MessageOrigin.SIDEKICK_GENERATED
            if self._chat_is_uncertain(chat_id):
                return MessageOrigin.INDETERMINATE
        return MessageOrigin.MANUAL_OUTGOING

    def clear_uncertain(
        self,
        chat_id: ExternalId,
        fingerprint: MessageFingerprint | None = None,
    ) -> None:
        _validate_external_id(chat_id, "chat ID")
        for key in tuple(self._uncertain):
            if key[0] == chat_id and (
                fingerprint is None or key[1] == fingerprint
            ):
                self._uncertain.pop(key)

    def _consume_confirmation(
        self,
        key: tuple[ExternalId, ExternalId],
    ) -> bool:
        if self._confirmed.pop(key, None) is None:
            return False
        self._remember_observed(key)
        return True

    def _chat_is_uncertain(self, chat_id: ExternalId) -> bool:
        return any(
            candidate_chat_id == chat_id
            for candidate_chat_id, _ in self._uncertain
        )

    def _remember_observed(self, key: tuple[ExternalId, ExternalId]) -> None:
        self._observed[key] = self._clock() + self._observed_retention
        self._observed.move_to_end(key)
        while len(self._observed) > self._max_observed:
            self._observed.popitem(last=False)

    def _prune_observed(self) -> None:
        now = self._clock()
        for key, expires_at in tuple(self._observed.items()):
            if expires_at <= now:
                self._observed.pop(key)

    def _settle(
        self,
        reservation: GeneratedMessageReservation,
        *,
        message_id: ExternalId | None,
        uncertain: bool,
        observed: bool,
    ) -> None:
        if reservation._tracker is not self:
            raise RuntimeError("Generated-message reservation belongs to another tracker")
        if reservation._settled:
            raise RuntimeError("Generated-message reservation is already settled")
        if message_id is not None:
            _validate_external_id(message_id, "message ID")
            key = (reservation.chat_id, message_id)
            if observed:
                self._remember_observed(key)
            else:
                self._confirmed[key] = self._clock()
                self._confirmed.move_to_end(key)
        elif uncertain:
            candidate = (reservation.chat_id, reservation.fingerprint)
            self._uncertain[candidate] = self._clock()
            self._uncertain.move_to_end(candidate)

        reservations = self._pending[reservation.chat_id]
        reservations.remove(reservation)
        if not reservations:
            self._pending.pop(reservation.chat_id)
        self._pending_count -= 1
        reservation._settled = True
        reservation._settlement.set_result(None)


def _validate_external_id(value: ExternalId, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Generated-message {label} is invalid")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Generated-message {label} is invalid")
