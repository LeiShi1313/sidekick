from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import re
from typing import Any, Literal, Protocol

from PIL import Image


AttachmentKind = Literal[
    "image",
    "audio",
    "video",
    "text",
    "file",
    "sticker",
    "other",
]
MAX_MODEL_IMAGE_BYTES = 2 * 1024 * 1024
MAX_MODEL_IMAGE_DIMENSION = 1_600
OutboundAttachmentDisplay = Literal["image", "file"]
MAX_OUTBOUND_ATTACHMENT_BYTES = 5 * 1024 * 1024
_OUTBOUND_IMAGE_SUFFIXES = {
    "image/jpeg": frozenset({".jpg", ".jpeg"}),
    "image/png": frozenset({".png"}),
}
_OUTBOUND_IMAGE_MIME_TYPES = frozenset(_OUTBOUND_IMAGE_SUFFIXES)
_MIME_TYPE_RE = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*"
)


@dataclass(frozen=True, slots=True)
class AttachmentReference:
    """Metadata and an opaque adapter key, never attachment payload bytes."""

    key: str
    kind: AttachmentKind
    mime_type: str
    filename: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Attachment reference key cannot be empty")
        if not self.mime_type.strip():
            raise ValueError("Attachment MIME type cannot be empty")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("Attachment size cannot be negative")


@dataclass(frozen=True, slots=True)
class ModelInputImage:
    """One normalized image kept only long enough for the current model turn."""

    mime_type: Literal["image/jpeg"]
    data: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.mime_type != "image/jpeg"
            or type(self.data) is not bytes
            or not self.data.startswith(b"\xff\xd8\xff")
        ):
            raise ValueError("Model input image must be a normalized JPEG")
        if len(self.data) > MAX_MODEL_IMAGE_BYTES:
            raise ValueError("Model input image exceeds the byte limit")
        try:
            source = Image.open(BytesIO(self.data))
        except Exception as exc:
            raise ValueError("Model input image must be a decodable JPEG") from exc
        with source:
            if source.format != "JPEG":
                raise ValueError("Model input image must be a decodable JPEG")
            if max(source.size) > MAX_MODEL_IMAGE_DIMENSION:
                raise ValueError("Model input image exceeds the dimension limit")
            try:
                source.load()
            except Exception as exc:
                raise ValueError("Model input image must be a decodable JPEG") from exc


@dataclass(frozen=True, slots=True)
class AttachmentDescription:
    context_text: str
    memory_text: str
    model_image: ModelInputImage | None = None


@dataclass(frozen=True, slots=True)
class OutboundAttachment:
    """One in-memory attachment ready for a chat transport to send."""

    data: bytes = field(repr=False)
    filename: str
    mime_type: str
    display_as: OutboundAttachmentDisplay

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("Outbound attachment data cannot be empty")
        if len(self.data) > MAX_OUTBOUND_ATTACHMENT_BYTES:
            raise ValueError("Outbound attachment exceeds the byte limit")
        if (
            not isinstance(self.filename, str)
            or not self.filename
            or self.filename != self.filename.strip()
            or self.filename in {".", ".."}
            or "/" in self.filename
            or "\\" in self.filename
            or "\x00" in self.filename
            or len(self.filename.encode("utf-8")) > 255
        ):
            raise ValueError("Outbound attachment filename is invalid")
        if (
            not isinstance(self.mime_type, str)
            or _MIME_TYPE_RE.fullmatch(self.mime_type) is None
        ):
            raise ValueError("Outbound attachment MIME type is invalid")
        if self.display_as not in {"image", "file"}:
            raise ValueError("Outbound attachment display must be image or file")
        if (
            self.display_as == "image"
            and self.mime_type not in _OUTBOUND_IMAGE_MIME_TYPES
        ):
            raise ValueError("Outbound images must be PNG or JPEG")
        if self.display_as == "image":
            suffix = (
                f".{self.filename.rsplit('.', 1)[-1].lower()}"
                if "." in self.filename
                else ""
            )
            if suffix not in _OUTBOUND_IMAGE_SUFFIXES[self.mime_type]:
                raise ValueError(
                    "Outbound image filename extension must match its MIME type"
                )


class AttachmentDescriber(Protocol):
    def has_attachment(self, message: Any) -> bool: ...

    async def describe(self, message: Any) -> AttachmentDescription | None: ...
