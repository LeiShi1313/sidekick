from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta


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
