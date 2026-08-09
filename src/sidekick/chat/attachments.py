from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
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


class AttachmentDescriber(Protocol):
    def has_attachment(self, message: Any) -> bool: ...

    async def describe(self, message: Any) -> AttachmentDescription | None: ...
