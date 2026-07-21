from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sidekick.ai_memory import MemoryEpisode, MemoryEvent


class SegmentEvent(Protocol):
    occurred_at: datetime
    text: str


@dataclass(frozen=True, slots=True)
class MemorySegmentationSettings:
    idle_gap: timedelta = timedelta(minutes=15)
    max_span: timedelta = timedelta(hours=1)
    max_events: int = 30
    max_chars: int = 4_000

    def __post_init__(self) -> None:
        if self.idle_gap <= timedelta(0) or self.max_span <= timedelta(0):
            raise ValueError("Memory segmentation time limits must be positive")
        if self.max_events < 1 or self.max_chars < 1:
            raise ValueError("Memory segmentation size limits must be positive")

    @classmethod
    def from_env(cls) -> MemorySegmentationSettings:
        return cls(
            idle_gap=timedelta(
                seconds=float(
                    os.environ.get(
                        "SIDEKICK_MEMORY_DREAM_SESSION_IDLE_SECONDS",
                        "900",
                    )
                )
            ),
            max_span=timedelta(
                seconds=float(
                    os.environ.get(
                        "SIDEKICK_MEMORY_DREAM_SESSION_MAX_SPAN_SECONDS",
                        "3600",
                    )
                )
            ),
            max_events=int(
                os.environ.get("SIDEKICK_MEMORY_DREAM_SESSION_MAX_EVENTS", "30")
            ),
            max_chars=int(
                os.environ.get("SIDEKICK_MEMORY_DREAM_SESSION_MAX_CHARS", "4000")
            ),
        )


@dataclass(frozen=True, slots=True)
class MemoryIngestionSettings:
    settlement_delay: timedelta = timedelta(seconds=30)
    max_messages: int = 500
    max_thread_messages: int = 100
    segmentation: MemorySegmentationSettings = MemorySegmentationSettings()
    retain_concurrency: int = 4
    preprocess_concurrency: int = 12
    lease_seconds: float = 3_600
    retry_attempts: int = 3
    max_retry_delay: float = 30

    def __post_init__(self) -> None:
        if self.settlement_delay < timedelta(0):
            raise ValueError("Memory settlement delay cannot be negative")
        if (
            self.max_messages < 1
            or self.max_thread_messages < 1
            or self.retain_concurrency < 1
            or self.preprocess_concurrency < 1
        ):
            raise ValueError("Memory ingestion limits must be positive")
        if self.lease_seconds <= 0:
            raise ValueError("Memory ingestion lease duration must be positive")
        if self.retry_attempts < 1 or self.max_retry_delay < 0:
            raise ValueError("Memory ingestion retry settings are invalid")

    @classmethod
    def from_env(cls) -> MemoryIngestionSettings:
        return cls(
            settlement_delay=timedelta(
                seconds=float(
                    os.environ.get(
                        "SIDEKICK_MEMORY_DREAM_SETTLEMENT_SECONDS",
                        "30",
                    )
                )
            ),
            max_messages=int(
                os.environ.get("SIDEKICK_MEMORY_DREAM_MAX_MESSAGES", "500")
            ),
            max_thread_messages=int(
                os.environ.get(
                    "SIDEKICK_MEMORY_DREAM_MAX_THREAD_MESSAGES",
                    "100",
                )
            ),
            segmentation=MemorySegmentationSettings.from_env(),
            retain_concurrency=int(
                os.environ.get("SIDEKICK_MEMORY_DREAM_RETAIN_CONCURRENCY", "4")
            ),
            preprocess_concurrency=int(
                os.environ.get("SIDEKICK_MEMORY_DREAM_PREPROCESS_CONCURRENCY", "12")
            ),
            lease_seconds=float(
                os.environ.get("SIDEKICK_MEMORY_DREAM_LEASE_SECONDS", "3600")
            ),
            retry_attempts=int(
                os.environ.get("SIDEKICK_MEMORY_DREAM_RETRY_ATTEMPTS", "3")
            ),
            max_retry_delay=float(
                os.environ.get("SIDEKICK_MEMORY_DREAM_MAX_RETRY_DELAY", "30")
            ),
        )


@dataclass(frozen=True, slots=True)
class PendingMemoryDocument:
    episode: MemoryEpisode
    staged_source_ids: tuple[str, ...]
    sealed: bool = False

    def __post_init__(self) -> None:
        if len(self.staged_source_ids) != len(set(self.staged_source_ids)):
            raise ValueError("Pending memory source IDs must be unique")

    @property
    def last_event_at(self) -> datetime:
        return max(event.occurred_at for event in self.episode.events)


def segment_accepts(
    current: tuple[SegmentEvent, ...],
    candidate: tuple[SegmentEvent, ...],
    settings: MemorySegmentationSettings,
) -> bool:
    if not current:
        return True
    current_start = min(event.occurred_at for event in current)
    current_end = max(event.occurred_at for event in current)
    candidate_start = min(event.occurred_at for event in candidate)
    candidate_end = max(event.occurred_at for event in candidate)
    if candidate_start - current_end > settings.idle_gap:
        return False
    if (
        max(current_end, candidate_end) - min(current_start, candidate_start)
        > settings.max_span
    ):
        return False
    if len(current) + len(candidate) > settings.max_events:
        return False
    return sum(len(event.text) for event in (*current, *candidate)) <= (
        settings.max_chars
    )


def merge_pending_document(
    current: PendingMemoryDocument,
    candidate: MemoryEpisode,
    staged_source_ids: tuple[str, ...],
) -> PendingMemoryDocument:
    if current.episode.scope_id != candidate.scope_id:
        raise ValueError("Cannot merge pending documents across memory scopes")
    events_by_source = {
        event.source_id: event
        for event in current.episode.events
        if event.source_id is not None
    }
    unkeyed_events = [
        event for event in current.episode.events if event.source_id is None
    ]
    for event in candidate.events:
        if event.source_id is None:
            unkeyed_events.append(event)
        else:
            events_by_source[event.source_id] = event
    events = tuple(
        sorted(
            (*events_by_source.values(), *unkeyed_events),
            key=lambda event: (
                _as_utc(event.occurred_at),
                event.source_id or "",
            ),
        )
    )
    merged_source_ids = tuple(
        dict.fromkeys((*current.staged_source_ids, *staged_source_ids))
    )
    return PendingMemoryDocument(
        episode=MemoryEpisode(
            scope_id=current.episode.scope_id,
            document_id=current.episode.document_id,
            events=events,
            scope_display_name=(
                candidate.scope_display_name or current.episode.scope_display_name
            ),
            source=current.episode.source,
        ),
        staged_source_ids=merged_source_ids,
        sealed=False,
    )


def pending_document_accepts(
    current: PendingMemoryDocument,
    candidate: MemoryEpisode,
    settings: MemorySegmentationSettings,
) -> bool:
    current_source_ids = {
        event.source_id
        for event in current.episode.events
        if event.source_id is not None
    }
    new_events = tuple(
        event
        for event in candidate.events
        if event.source_id is None or event.source_id not in current_source_ids
    )
    if not new_events:
        return True
    return segment_accepts(current.episode.events, new_events, settings)


def decode_memory_episode(
    *,
    document_id: str,
    source: str,
    content: str,
) -> MemoryEpisode:
    try:
        payload = json.loads(content)
        if payload.get("schema") != "sidekick.memory.episode.v1":
            raise ValueError("Unsupported pending memory schema")
        scope = payload["scope"]
        scope_id = _required_string(scope, "id")
        display_name = scope.get("display_name")
        if display_name is not None and not isinstance(display_name, str):
            raise ValueError("Malformed pending memory scope")
        events = tuple(_decode_memory_event(item) for item in payload["events"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed pending memory document") from exc
    return MemoryEpisode(
        scope_id=scope_id,
        document_id=document_id,
        events=events,
        scope_display_name=display_name,
        source=source,
    )


def _decode_memory_event(payload: Any) -> MemoryEvent:
    if not isinstance(payload, dict):
        raise ValueError("Malformed pending memory event")
    actor = payload.get("actor")
    if not isinstance(actor, dict):
        raise ValueError("Malformed pending memory actor")
    actor_display_name = actor.get("display_name")
    if actor_display_name is not None and not isinstance(actor_display_name, str):
        raise ValueError("Malformed pending memory actor")
    source_id = payload.get("source_id")
    reply_to_source_id = payload.get("reply_to_source_id")
    if source_id is not None and not isinstance(source_id, str):
        raise ValueError("Malformed pending memory source")
    if reply_to_source_id is not None and not isinstance(reply_to_source_id, str):
        raise ValueError("Malformed pending memory reply source")
    mentioned_at = payload.get("mentioned_at")
    metadata = payload.get("metadata")
    mentioned_actors = payload.get("mentioned_actors")
    if not isinstance(metadata, dict) or not isinstance(mentioned_actors, list):
        raise ValueError("Malformed pending memory metadata")
    decoded_mentions: list[tuple[str, str | None]] = []
    for mention in mentioned_actors:
        if not isinstance(mention, dict):
            raise ValueError("Malformed pending memory mention")
        display_name = mention.get("display_name")
        if display_name is not None and not isinstance(display_name, str):
            raise ValueError("Malformed pending memory mention")
        decoded_mentions.append((_required_string(mention, "id"), display_name))
    return MemoryEvent(
        source_id=source_id,
        actor_id=_required_string(actor, "id"),
        actor_display_name=actor_display_name,
        occurred_at=_decode_datetime(payload.get("occurred_at")),
        mentioned_at=(
            _decode_datetime(mentioned_at) if mentioned_at is not None else None
        ),
        reply_to_source_id=reply_to_source_id,
        mentioned_actors=tuple(decoded_mentions),
        metadata=metadata,
        text=_required_string(payload, "text"),
    )


def _required_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError("Malformed pending memory document")
    return value


def _decode_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Malformed pending memory timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Malformed pending memory timestamp") from exc
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
