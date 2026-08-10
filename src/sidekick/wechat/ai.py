from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import time
from typing import Any, Callable, Protocol
from urllib.parse import quote, unquote

from sidekick.ai import MemoryScopeTarget, MessageIdentity, MentionedUser, ReplyTarget
from sidekick.ai_attachments import (
    AttachmentAnalysisGateway,
    ChatAttachmentDescriber,
)
from sidekick.chat.attachments import AttachmentDescription, OutboundAttachment
from sidekick.chat.formatting import markdown_to_plain_text
from sidekick.chat.identity import ExternalId, IdentityCodec
from sidekick.chat.provenance import (
    GeneratedMessageTracker,
    MessageFingerprint,
    MessageOrigin,
    message_fingerprint,
    observed_message_fingerprint,
)
from sidekick.chat.transport import ChatPresentation, SentMessage
from sidekick.wechat.api import (
    MAX_MEDIA_BYTES,
    MAX_TEXT_BYTES,
    WeChatAPIError,
    WeChatDownloadedImage,
    WeChatSendFailed,
    WeChatSendOperation,
    WeChatSendOutcomeUnknown,
)
from sidekick.wechat.message import WeChatMessage
from sidekick.wechat.store import (
    WeChatGeneratedSendReservation,
    WeChatStateRepository,
)


_SAFE_PRE_ACTIVATION_CODES = frozenset(
    {"REPLY_UNSUPPORTED", "SEND_NOT_READY", "SEND_UNAVAILABLE"}
)


@dataclass(frozen=True, slots=True)
class WeChatIdentityCodec:
    source: str = "wechat"
    account_id: str | None = None

    def __post_init__(self) -> None:
        if self.source != "wechat":
            raise ValueError("WeChat identity source must be 'wechat'")
        if self.account_id is not None:
            _component(self.account_id)

    def actor_id(self, actor_id: ExternalId) -> str:
        return f"{self._identity_prefix()}user:{_component(actor_id)}"

    def scope_id(self, scope_id: ExternalId) -> str:
        return f"{self._identity_prefix()}chat:{_component(scope_id)}"

    def parse_scope_id(self, scope_id: str) -> ExternalId | None:
        prefix = f"{self._identity_prefix()}chat:"
        if not scope_id.startswith(prefix):
            return None
        encoded = scope_id.removeprefix(prefix)
        if not encoded:
            return None
        decoded = unquote(encoded)
        if not decoded or quote(decoded, safe="-_.~") != encoded:
            return None
        return decoded

    def message_source_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str:
        return (
            f"{self._identity_prefix()}message:{_component(scope_id)}:"
            f"{_component(message_id)}"
        )

    def parse_message_source_id(
        self,
        source_id: str,
    ) -> tuple[ExternalId, ExternalId] | None:
        prefix = f"{self._identity_prefix()}message:"
        if not source_id.startswith(prefix):
            return None
        parts = source_id.removeprefix(prefix).split(":")
        if len(parts) != 2:
            return None
        scope_id = _decoded_component(parts[0])
        message_id = _decoded_component(parts[1])
        if scope_id is None or message_id is None:
            return None
        return scope_id, message_id

    def thread_document_id(
        self,
        scope_id: ExternalId,
        root_message_id: ExternalId,
    ) -> str:
        return (
            f"{self._identity_prefix()}thread:{_component(scope_id)}:"
            f"{_component(root_message_id)}"
        )

    def revision_document_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str:
        return (
            f"{self._identity_prefix()}revision:{_component(scope_id)}:"
            f"{_component(message_id)}"
        )

    def _identity_prefix(self) -> str:
        if self.account_id is None:
            return "wechat:"
        return f"wechat:account:{_component(self.account_id)}:"


WECHAT_IDENTITY_CODEC: IdentityCodec = WeChatIdentityCodec()


class WeChatMessageSender(Protocol):
    async def send_text_and_wait(
        self,
        *,
        request_id: str,
        to: str,
        content: str,
        reply_to_message_id: str | None,
    ) -> WeChatSendOperation: ...

    async def send_attachment_and_wait(
        self,
        *,
        request_id: str,
        to: str,
        attachment: OutboundAttachment,
    ) -> WeChatSendOperation: ...

    async def reconcile_send_and_wait(
        self,
        *,
        request_id: str,
        to: str,
    ) -> WeChatSendOperation | None: ...


class WeChatImageDownloader(Protocol):
    async def download_original_image(
        self,
        *,
        request_id: str,
        chat_id: str,
        message_id: str,
        media_id: str,
    ) -> WeChatDownloadedImage: ...

    async def download_image_preview(
        self,
        *,
        media_id: str,
    ) -> WeChatDownloadedImage: ...


class WeChatQuotedImageDescriber:
    def __init__(
        self,
        client: WeChatImageDownloader,
        gateway: AttachmentAnalysisGateway,
        *,
        request_original: bool,
        download_preview: bool,
        logger: Any | None = None,
    ):
        self._client = client
        self._request_original = request_original
        self._download_preview = download_preview
        self._logger = logger
        self._content_describer = ChatAttachmentDescriber(
            gateway,
            max_file_bytes=MAX_MEDIA_BYTES,
            logger=logger,
        )

    def has_attachment(self, message: Any) -> bool:
        return (
            isinstance(message, WeChatMessage)
            and message.message_type == "image"
            and message.media_id is not None
        )

    async def describe(self, message: Any) -> AttachmentDescription | None:
        if not self.has_attachment(message):
            return None
        assert isinstance(message, WeChatMessage)
        assert message.media_id is not None

        fallback: AttachmentDescription | None = None
        if self._request_original:
            try:
                downloaded = await self._client.download_original_image(
                    request_id=_request_id(message, "original"),
                    chat_id=message.chat_id,
                    message_id=message.id,
                    media_id=message.media_id,
                )
                described = await self._content_describer.describe_image_bytes(
                    downloaded.data,
                    mime_type=downloaded.mime_type,
                )
                if described.model_image is not None:
                    return described
                fallback = described
            except Exception as exc:
                self._log_unavailable("original", exc)
        if self._download_preview:
            try:
                downloaded = await self._client.download_image_preview(
                    media_id=message.media_id,
                )
                described = await self._content_describer.describe_image_bytes(
                    downloaded.data,
                    mime_type=downloaded.mime_type,
                )
                if described.model_image is not None:
                    return described
                fallback = described
            except Exception as exc:
                self._log_unavailable("preview", exc)
        if fallback is None:
            return AttachmentDescription(
                context_text=(
                    "Quoted image content is unavailable; neither the original "
                    "nor low-resolution preview could be downloaded."
                ),
                memory_text=(
                    "The subject shared an image, but its original and "
                    "low-resolution preview were unavailable for analysis."
                ),
            )
        return fallback

    def _log_unavailable(self, variant: str, exc: Exception) -> None:
        if self._logger is not None:
            self._logger.debug(
                "WeChat quoted %s image unavailable (%s)",
                variant,
                type(exc).__name__,
            )


@dataclass(slots=True)
class WeChatSentMessage:
    id: str
    text: str | None
    trigger: WeChatMessage
    request_id: str
    sent: bool = False
    failed: bool = False
    uncertain: bool = False


class WeChatChatTransport:
    RECONCILIATION_CONCURRENCY = 8
    RECONCILIATION_BATCH_SIZE = 32
    RECONCILIATION_RETRY_BASE_SECONDS = 2.0
    RECONCILIATION_RETRY_MAX_SECONDS = 300.0
    RECONCILIATION_LOG_INTERVAL_SECONDS = 60.0

    def __init__(
        self,
        client: WeChatMessageSender,
        store: WeChatStateRepository,
        connector_key: str,
        *,
        native_reply_ready: bool,
        logger: Any | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self._client = client
        self._store = store
        self._connector_key = connector_key
        self._native_reply_ready = native_reply_ready
        self._logger = logger
        self._clock = clock
        self._generated_messages = GeneratedMessageTracker()
        self._durable_indeterminate_count: int | None = None
        self._reconciliation_accounts: set[str] = set()
        self._reconciliation_initialization_lock = asyncio.Lock()
        self._deferred_reconciliation_errors: Counter[str] = Counter()
        self._next_reconciliation_log_at = 0.0

    @property
    def indeterminate_outbound_count(self) -> int | None:
        in_memory_count = self._generated_messages.indeterminate_count
        if self._durable_indeterminate_count is None:
            return in_memory_count or None
        return max(
            in_memory_count,
            self._durable_indeterminate_count,
        )

    async def draft_reply(self, message: Any) -> WeChatSentMessage:
        trigger = self._trigger(message)
        request_id = _request_id(trigger, "answer")
        return WeChatSentMessage(
            id=f"draft:{request_id}",
            text=None,
            trigger=trigger,
            request_id=request_id,
        )

    async def get_reply(self, message: Any) -> WeChatMessage | None:
        if not isinstance(message, WeChatMessage) or message.reply_to_msg_id is None:
            return None
        try:
            return await self._store.get_reply_message(
                self._connector_key,
                message.chat_id,
                message.reply_to_msg_id,
            )
        except Exception as exc:
            if self._logger is not None:
                self._logger.debug(
                    "WeChat reply lookup failed (%s)",
                    type(exc).__name__,
                )
            return None

    async def reply(
        self,
        message: Any,
        text: str,
        *,
        presentation: ChatPresentation,
    ) -> SentMessage:
        trigger = self._trigger(message)
        rendered = self._render(text, presentation)
        sent = WeChatSentMessage(
            id=f"draft:{_request_id(trigger, 'reply')}",
            text=None,
            trigger=trigger,
            request_id=_request_id(trigger, "reply"),
        )
        await self._send(sent, rendered)
        return sent

    async def reply_attachment(
        self,
        message: Any,
        attachment: OutboundAttachment,
    ) -> SentMessage | None:
        trigger = self._trigger(message)
        payload_fingerprint = sha256(
            b"\0".join(
                (
                    attachment.display_as.encode("ascii"),
                    attachment.filename.encode("utf-8"),
                    attachment.mime_type.encode("ascii"),
                    attachment.data,
                )
            )
        ).hexdigest()[:32]
        attempt = await self._store.get_attachment_send_attempt(
            self._connector_key,
            trigger.account_id,
            trigger.chat_id,
            trigger.id,
            payload_fingerprint,
        )
        purpose = f"attachment.{payload_fingerprint}"
        if attempt:
            purpose = f"{purpose}.{attempt}"
        request_id = _request_id(trigger, purpose)
        fingerprint = message_fingerprint(
            text=None,
            reply_to_message_id=None,
            has_attachment=True,
        )
        with self._generated_messages.reserve(
            trigger.chat_id,
            fingerprint,
        ) as reservation:
            try:
                lease_id = await self._store.reserve_generated_send(
                    self._connector_key,
                    trigger.account_id,
                    trigger.chat_id,
                    request_id,
                    fingerprint.digest,
                )
            except Exception:
                reservation.failed()
                raise
            try:
                operation = await self._client.send_attachment_and_wait(
                    request_id=request_id,
                    to=trigger.chat_id,
                    attachment=attachment,
                )
            except asyncio.CancelledError:
                await self._defer_generated_send_after_cancellation(
                    trigger.account_id,
                    request_id,
                    lease_id,
                )
                raise
            except WeChatSendOutcomeUnknown:
                await self._store.defer_generated_send(
                    self._connector_key,
                    trigger.account_id,
                    request_id,
                    lease_id,
                )
                raise
            except WeChatSendFailed:
                reservation.failed()
                await self._store.fail_generated_send(
                    self._connector_key,
                    trigger.account_id,
                    request_id,
                    lease_id,
                )
                await self._store.advance_attachment_send_attempt(
                    self._connector_key,
                    trigger.account_id,
                    trigger.chat_id,
                    trigger.id,
                    payload_fingerprint,
                    expected_attempt=attempt,
                )
                raise
            except WeChatAPIError as exc:
                if exc.code in _SAFE_PRE_ACTIVATION_CODES:
                    reservation.failed()
                    await self._store.fail_generated_send(
                        self._connector_key,
                        trigger.account_id,
                        request_id,
                        lease_id,
                    )
                else:
                    await self._store.defer_generated_send(
                        self._connector_key,
                        trigger.account_id,
                        request_id,
                        lease_id,
                    )
                raise
            except Exception:
                await self._store.defer_generated_send(
                    self._connector_key,
                    trigger.account_id,
                    request_id,
                    lease_id,
                )
                raise
            if operation.message_id is None:
                await self._store.defer_generated_send(
                    self._connector_key,
                    trigger.account_id,
                    request_id,
                    lease_id,
                )
                raise RuntimeError("WeChat attachment send returned no message ID")
            await self._store.confirm_generated_send(
                self._connector_key,
                trigger.account_id,
                trigger.chat_id,
                request_id,
                operation.message_id,
            )
            reservation.confirm(operation.message_id, echo_expected=False)
        return WeChatSentMessage(
            id=operation.message_id,
            text=None,
            trigger=trigger,
            request_id=request_id,
            sent=True,
        )

    async def update(
        self,
        message: SentMessage,
        text: str,
        *,
        presentation: ChatPresentation,
        wait: bool,
    ) -> bool:
        if not isinstance(message, WeChatSentMessage):
            raise RuntimeError("WeChat transport requires a WeChat sent message")
        if not wait:
            return False
        rendered = self._render(text, presentation)
        if message.uncertain:
            message.text = rendered
            return True
        if message.sent:
            if message.text == rendered:
                return True
            content_fingerprint = sha256(rendered.encode("utf-8")).hexdigest()[:16]
            message.request_id = _request_id(
                message.trigger,
                f"update.{content_fingerprint}",
            )
            message.sent = False
        if message.failed:
            message.request_id = _request_id(message.trigger, "failure")
            message.failed = False
        await self._send(message, rendered)
        return True

    async def delete(self, _message: Any) -> None:
        # Deletion is deliberately not mapped to WeChat Recall. Recall has a
        # narrower capability and uncertainty contract than local cleanup.
        return None

    async def classify_origin(self, message: Any) -> MessageOrigin:
        origin = await self._generated_messages.classify(
            chat_id=message.chat_id,
            message_id=message.id,
            outgoing=bool(
                getattr(message, "is_outgoing", getattr(message, "out", False))
            )
            and getattr(message, "sender_id", None)
            == getattr(message, "self_id", None),
            fingerprint=observed_message_fingerprint(message),
        )
        if origin in {MessageOrigin.INCOMING, MessageOrigin.SIDEKICK_GENERATED}:
            return origin
        if not isinstance(message, WeChatMessage):
            return origin
        durable_provenance = await self._store.generated_message_provenance(
            message,
        )
        if durable_provenance == "confirmed":
            return MessageOrigin.SIDEKICK_GENERATED
        if durable_provenance == "candidate":
            # A single background owner reconciles durable unknown sends. Event
            # handlers fail closed immediately so an unavailable connector cannot
            # consume every ingress slot. Ambiguous manual controls must be retried
            # after reconciliation rather than executed later out of context.
            return MessageOrigin.INDETERMINATE
        self._generated_messages.clear_uncertain(message.chat_id)
        return (
            MessageOrigin.MANUAL_OUTGOING
            if origin is MessageOrigin.INDETERMINATE
            else origin
        )

    def is_group(self, message: Any) -> bool:
        return getattr(message, "chat_type", None) == "group"

    async def _send(self, message: WeChatSentMessage, text: str) -> None:
        reply_to_message_id = self._native_reply_target(message.trigger)
        plain_fallback_available = reply_to_message_id is not None
        while True:
            fingerprint = message_fingerprint(
                text=text,
                reply_to_message_id=reply_to_message_id,
                has_attachment=False,
            )
            with self._generated_messages.reserve(
                message.trigger.chat_id,
                fingerprint,
            ) as reservation:
                try:
                    lease_id = await self._store.reserve_generated_send(
                        self._connector_key,
                        message.trigger.account_id,
                        message.trigger.chat_id,
                        message.request_id,
                        fingerprint.digest,
                    )
                except Exception:
                    reservation.failed()
                    raise
                try:
                    operation = await self._client.send_text_and_wait(
                        request_id=message.request_id,
                        to=message.trigger.chat_id,
                        content=text,
                        reply_to_message_id=reply_to_message_id,
                    )
                except asyncio.CancelledError:
                    message.uncertain = True
                    message.text = text
                    await self._defer_generated_send_after_cancellation(
                        message.trigger.account_id,
                        message.request_id,
                        lease_id,
                    )
                    raise
                except WeChatSendOutcomeUnknown:
                    message.uncertain = True
                    message.text = text
                    await self._store.defer_generated_send(
                        self._connector_key,
                        message.trigger.account_id,
                        message.request_id,
                        lease_id,
                    )
                    raise
                except WeChatSendFailed:
                    reservation.failed()
                    await self._store.fail_generated_send(
                        self._connector_key,
                        message.trigger.account_id,
                        message.request_id,
                        lease_id,
                    )
                    # The connector reserves `unknown` for every path where Quote
                    # or Send may have taken effect. A terminal `failed` reply can
                    # therefore be replaced by one ordinary text operation.
                    if plain_fallback_available:
                        plain_fallback_available = False
                        reply_to_message_id = None
                        message.request_id = f"{message.request_id}.plain"
                        continue
                    message.failed = True
                    message.text = text
                    raise
                except WeChatAPIError as exc:
                    # These synchronous reply errors are contractually rejected
                    # before operation creation or activation.
                    if exc.code in _SAFE_PRE_ACTIVATION_CODES:
                        reservation.failed()
                        await self._store.fail_generated_send(
                            self._connector_key,
                            message.trigger.account_id,
                            message.request_id,
                            lease_id,
                        )
                        if plain_fallback_available:
                            plain_fallback_available = False
                            reply_to_message_id = None
                            message.request_id = f"{message.request_id}.plain"
                            continue
                        message.failed = True
                        message.text = text
                        raise
                    message.uncertain = True
                    message.text = text
                    await self._store.defer_generated_send(
                        self._connector_key,
                        message.trigger.account_id,
                        message.request_id,
                        lease_id,
                    )
                    raise
                except Exception:
                    # A transport failure can happen after the connector accepted
                    # the request. Keep the original ID/payload reserved and never
                    # replace it with an error message under that ID.
                    message.uncertain = True
                    message.text = text
                    await self._store.defer_generated_send(
                        self._connector_key,
                        message.trigger.account_id,
                        message.request_id,
                        lease_id,
                    )
                    raise
                if operation.message_id is None:
                    await self._store.defer_generated_send(
                        self._connector_key,
                        message.trigger.account_id,
                        message.request_id,
                        lease_id,
                    )
                    raise RuntimeError("WeChat text send returned no message ID")
                await self._store.confirm_generated_send(
                    self._connector_key,
                    message.trigger.account_id,
                    message.trigger.chat_id,
                    message.request_id,
                    operation.message_id,
                )
                reservation.confirm(operation.message_id, echo_expected=False)
                break
        if operation.message_id is None:
            raise RuntimeError("WeChat text send returned no message ID")
        message.id = operation.message_id
        message.text = text
        message.sent = True

    async def reconcile_pending(self, account_id: str) -> int:
        self._durable_indeterminate_count = None
        await self._initialize_reconciliation(account_id)
        reservations = await self._store.list_due_generated_send_reservations(
            self._connector_key,
            account_id,
            now=self._clock(),
            limit=self.RECONCILIATION_BATCH_SIZE,
        )
        deferred = await self._reconcile_reservations(account_id, reservations)
        if deferred:
            self._log_reconciliation_deferred(deferred)
        self._durable_indeterminate_count = (
            await self._store.count_generated_send_reservations(
                self._connector_key,
                account_id,
            )
        )
        return self._durable_indeterminate_count

    async def _initialize_reconciliation(self, account_id: str) -> None:
        if account_id in self._reconciliation_accounts:
            return
        async with self._reconciliation_initialization_lock:
            if account_id in self._reconciliation_accounts:
                return
            await self._store.recover_generated_send_leases(
                self._connector_key,
                account_id,
            )
            self._reconciliation_accounts.add(account_id)

    async def _reconcile_reservations(
        self,
        account_id: str,
        reservations: tuple[WeChatGeneratedSendReservation, ...],
    ) -> tuple[str, ...]:
        deferred: list[str] = []
        for offset in range(0, len(reservations), self.RECONCILIATION_CONCURRENCY):
            results = await asyncio.gather(
                *(
                    self._reconcile_reservation(account_id, reservation)
                    for reservation in reservations[
                        offset : offset + self.RECONCILIATION_CONCURRENCY
                    ]
                )
            )
            deferred.extend(result for result in results if result is not None)
        return tuple(deferred)

    async def _reconcile_reservation(
        self,
        account_id: str,
        reservation: WeChatGeneratedSendReservation,
    ) -> str | None:
        reconciliation_lease = (
            await self._store.claim_generated_send_reconciliation(
                self._connector_key,
                account_id,
                reservation.request_id,
            )
        )
        if reconciliation_lease is None:
            return None
        try:
            operation = await self._client.reconcile_send_and_wait(
                request_id=reservation.request_id,
                to=reservation.chat_id,
            )
            if operation is None:
                await self._store.fail_generated_send(
                    self._connector_key,
                    account_id,
                    reservation.request_id,
                )
                self._generated_messages.clear_uncertain(
                    reservation.chat_id,
                    MessageFingerprint(reservation.fingerprint),
                )
                return None
            if operation.message_id is None:
                raise RuntimeError(
                    "WeChat generated-send reconciliation returned no message ID"
                )
            await self._store.confirm_generated_send(
                self._connector_key,
                account_id,
                reservation.chat_id,
                reservation.request_id,
                operation.message_id,
            )
            self._generated_messages.clear_uncertain(
                reservation.chat_id,
                MessageFingerprint(reservation.fingerprint),
            )
            return None
        except asyncio.CancelledError:
            await self._defer_generated_send_after_cancellation(
                account_id,
                reservation.request_id,
                reconciliation_lease,
            )
            raise
        except WeChatSendFailed:
            try:
                await self._store.fail_generated_send(
                    self._connector_key,
                    account_id,
                    reservation.request_id,
                )
                self._generated_messages.clear_uncertain(
                    reservation.chat_id,
                    MessageFingerprint(reservation.fingerprint),
                )
            except Exception as exc:
                return await self._defer_reconciliation(
                    account_id,
                    reservation,
                    reconciliation_lease,
                    exc,
                )
            return None
        except Exception as exc:
            return await self._defer_reconciliation(
                account_id,
                reservation,
                reconciliation_lease,
                exc,
            )

    async def _defer_reconciliation(
        self,
        account_id: str,
        reservation: WeChatGeneratedSendReservation,
        reconciliation_lease: str,
        exc: Exception,
    ) -> str:
        next_attempt_at = self._clock() + self._reconciliation_delay(reservation)
        try:
            await self._store.defer_generated_send_reconciliation(
                self._connector_key,
                account_id,
                reservation.request_id,
                reconciliation_lease,
                next_attempt_at=next_attempt_at,
            )
        except Exception as state_error:
            return f"{type(exc).__name__}/{type(state_error).__name__}"
        return type(exc).__name__

    async def _defer_generated_send_after_cancellation(
        self,
        account_id: str,
        request_id: str,
        lease_id: str,
    ) -> None:
        try:
            await self._store.defer_generated_send(
                self._connector_key,
                account_id,
                request_id,
                lease_id,
            )
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(
                    "WeChat cancelled-send lease cleanup deferred (%s)",
                    type(exc).__name__,
                )

    def _reconciliation_delay(
        self,
        reservation: WeChatGeneratedSendReservation,
    ) -> float:
        jitter_bucket = int.from_bytes(
            sha256(reservation.request_id.encode("utf-8")).digest()[:2],
            "big",
        )
        jitter = (jitter_bucket / 65_535) * 0.25
        delay = self.RECONCILIATION_RETRY_BASE_SECONDS * (
            2 ** min(reservation.reconciliation_attempts, 16)
        )
        return min(
            self.RECONCILIATION_RETRY_MAX_SECONDS,
            delay * (1 + jitter),
        )

    def _log_reconciliation_deferred(self, errors: tuple[str, ...]) -> None:
        if self._logger is None:
            return
        self._deferred_reconciliation_errors.update(errors)
        now = self._clock()
        if now < self._next_reconciliation_log_at:
            return
        summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                self._deferred_reconciliation_errors.items()
            )
        )
        self._logger.warning(
            "WeChat generated-send reconciliation deferred (%d; %s)",
            self._deferred_reconciliation_errors.total(),
            summary,
        )
        self._deferred_reconciliation_errors.clear()
        self._next_reconciliation_log_at = (
            now + self.RECONCILIATION_LOG_INTERVAL_SECONDS
        )

    def _native_reply_target(self, trigger: WeChatMessage) -> str | None:
        if not self._native_reply_ready:
            return None
        if trigger.message_type != "text" or trigger.content_redacted:
            return None
        if (
            not trigger.raw_text.strip()
            or "\n" in trigger.raw_text
            or "\r" in trigger.raw_text
        ):
            return None
        return trigger.id

    @staticmethod
    def _trigger(message: Any) -> WeChatMessage:
        if not isinstance(message, WeChatMessage):
            raise RuntimeError("WeChat transport requires a WeChat message")
        return message

    @staticmethod
    def _render(text: str, presentation: ChatPresentation) -> str:
        rendered = markdown_to_plain_text(text) if presentation == "agent" else text
        return _truncate_utf8(rendered, MAX_TEXT_BYTES)


class WeChatMessageIdentityResolver:
    def __init__(self, identity_codec: IdentityCodec = WECHAT_IDENTITY_CODEC):
        self._identity_codec = identity_codec

    async def resolve(self, message: ReplyTarget) -> MessageIdentity:
        return MessageIdentity(
            subject_id=(
                self._identity_codec.actor_id(message.sender_id)
                if message.sender_id is not None
                else None
            ),
            subject_display_name=getattr(message, "sender_display_name", None),
            scope_display_name=getattr(message, "scope_display_name", None),
            is_human=message.sender_id is not None,
        )


class WeChatMessageMentionResolver:
    async def resolve(self, _message: ReplyTarget) -> tuple[MentionedUser, ...]:
        return ()


class WeChatHistorySource:
    def __init__(self, store: WeChatStateRepository, connector_key: str):
        self._store = store
        self._connector_key = connector_key

    async def fetch_recent(
        self,
        trigger: ReplyTarget,
        *,
        before: ReplyTarget,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        if (
            not isinstance(trigger.chat_id, str)
            or before.chat_id != trigger.chat_id
            or not isinstance(before.id, str)
        ):
            return ()
        return await self._store.fetch_recent(
            self._connector_key,
            trigger.chat_id,
            before.id,
            limit,
        )

    async def fetch_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> WeChatMessage | None:
        return await self._store.get_message(
            self._connector_key,
            chat_id,
            message_id,
        )

    async def fetch_window(
        self,
        chat_id: str,
        *,
        since: datetime,
        until: datetime,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        return await self._store.fetch_memory_window(
            self._connector_key,
            chat_id,
            since=since,
            until=until,
            limit=limit,
        )

    async def fetch_after(
        self,
        chat_id: str,
        *,
        after_message_id: ExternalId,
        until: datetime,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        if isinstance(after_message_id, bool) or not isinstance(
            after_message_id,
            int,
        ):
            raise ValueError("WeChat continuous memory cursor is invalid")
        return await self._store.fetch_memory_after(
            self._connector_key,
            chat_id,
            after_memory_order=after_message_id,
            until=until,
            limit=limit,
        )


class WeChatMemoryScopeTargetResolver:
    def __init__(self, store: WeChatStateRepository, connector_key: str):
        self._store = store
        self._connector_key = connector_key

    async def resolve(
        self,
        target: str,
        *,
        include_latest_message: bool = False,
    ) -> MemoryScopeTarget:
        chat_id = target.strip()
        if not chat_id or chat_id != target:
            raise ValueError("WeChat target must be an exact stored chat ID")
        chat = await self._store.get_chat(self._connector_key, chat_id)
        if chat is None:
            raise ValueError("WeChat chat is not present in the local projection")
        latest_memory_cursor = (
            await self._store.get_latest_memory_cursor(
                self._connector_key,
                chat_id,
            )
            if include_latest_message
            else 0
        )
        return MemoryScopeTarget(
            chat_id=chat.id,
            display_name=chat.display_name,
            latest_message_id=latest_memory_cursor,
        )


def _component(value: ExternalId) -> str:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid WeChat IDs")
    normalized = str(value)
    if not normalized or normalized != normalized.strip():
        raise ValueError("WeChat IDs cannot be empty or padded")
    return quote(normalized, safe="-_.~")


def _decoded_component(value: str) -> str | None:
    decoded = unquote(value)
    if not decoded or quote(decoded, safe="-_.~") != value:
        return None
    return decoded


def _request_id(trigger: WeChatMessage, purpose: str) -> str:
    fingerprint = "\0".join(
        (trigger.account_id, trigger.chat_id, trigger.id, purpose)
    ).encode("utf-8")
    return f"sidekick.wechat.{purpose}.{sha256(fingerprint).hexdigest()[:40]}"


def _truncate_utf8(text: str, maximum: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum:
        return text
    prefix = encoded[: maximum - 3].decode("utf-8", errors="ignore")
    return f"{prefix}..."
