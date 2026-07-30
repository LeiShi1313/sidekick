from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote, unquote


ExternalId = int | str


class IdentityCodec(Protocol):
    source: str

    def actor_id(self, actor_id: ExternalId) -> str: ...

    def scope_id(self, scope_id: ExternalId) -> str: ...

    def parse_scope_id(self, scope_id: str) -> ExternalId | None: ...

    def message_source_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str: ...

    def parse_message_source_id(
        self,
        source_id: str,
    ) -> tuple[ExternalId, ExternalId] | None: ...

    def thread_document_id(
        self,
        scope_id: ExternalId,
        root_message_id: ExternalId,
    ) -> str: ...

    def revision_document_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class NamespacedIdentityCodec:
    source: str
    actor_kind: str
    scope_kind: str

    def __post_init__(self) -> None:
        for value in (self.source, self.actor_kind, self.scope_kind):
            if not value or any(character in value for character in ": \t\r\n"):
                raise ValueError(
                    "Identity namespace components must be non-empty tokens"
                )

    def actor_id(self, actor_id: ExternalId) -> str:
        return f"{self.source}:{self.actor_kind}:{_component(actor_id)}"

    def scope_id(self, scope_id: ExternalId) -> str:
        return f"{self.source}:{self.scope_kind}:{_component(scope_id)}"

    def parse_scope_id(self, scope_id: str) -> ExternalId | None:
        prefix = f"{self.source}:{self.scope_kind}:"
        if not scope_id.startswith(prefix):
            return None
        return _parse_component(scope_id.removeprefix(prefix))

    def message_source_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str:
        return f"{self.source}:message:{_component(scope_id)}:{_component(message_id)}"

    def parse_message_source_id(
        self,
        source_id: str,
    ) -> tuple[ExternalId, ExternalId] | None:
        prefix = f"{self.source}:message:"
        if not source_id.startswith(prefix):
            return None
        parts = source_id.removeprefix(prefix).split(":")
        if len(parts) != 2:
            return None
        scope_id = _parse_component(parts[0])
        message_id = _parse_component(parts[1])
        if scope_id is None or message_id is None:
            return None
        return scope_id, message_id

    def thread_document_id(
        self,
        scope_id: ExternalId,
        root_message_id: ExternalId,
    ) -> str:
        return (
            f"{self.source}:thread:{_component(scope_id)}:{_component(root_message_id)}"
        )

    def revision_document_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str:
        return f"{self.source}:revision:{_component(scope_id)}:{_component(message_id)}"


def _component(value: ExternalId) -> str:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid external IDs")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("External IDs cannot be empty")
    return quote(normalized, safe="-_.~")


def _parse_component(value: str) -> ExternalId | None:
    decoded = unquote(value)
    if not decoded or quote(decoded, safe="-_.~") != value:
        return None
    try:
        return int(decoded)
    except ValueError:
        return decoded
