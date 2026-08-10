from __future__ import annotations

import asyncio

import pytest

from sidekick.chat.provenance import (
    GeneratedMessageTracker,
    MessageOrigin,
    message_fingerprint,
)


GENERATED = message_fingerprint(
    text="generated",
    reply_to_message_id=1,
    has_attachment=False,
)
MANUAL = message_fingerprint(
    text="/ai manual",
    reply_to_message_id=None,
    has_attachment=False,
)


@pytest.mark.asyncio
async def test_incoming_message_never_waits_for_pending_sends() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)

    assert (
        await tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=False,
            fingerprint=GENERATED,
        )
        is MessageOrigin.INCOMING
    )

    reservation.uncertain()


@pytest.mark.asyncio
async def test_confirmed_generated_message_is_classified_by_exact_native_id() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)
    reservation.confirm(101)

    assert (
        await tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
        is MessageOrigin.SIDEKICK_GENERATED
    )
    assert (
        await tracker.classify(
            chat_id=7,
            message_id=102,
            outgoing=True,
            fingerprint=MANUAL,
        )
        is MessageOrigin.MANUAL_OUTGOING
    )


@pytest.mark.asyncio
async def test_event_before_send_response_waits_for_exact_native_id() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)
    classification = asyncio.create_task(
        tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
    )
    await asyncio.sleep(0)

    reservation.confirm(101)

    assert await classification is MessageOrigin.SIDEKICK_GENERATED


@pytest.mark.asyncio
async def test_manual_outgoing_message_during_send_is_preserved_by_its_id() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)

    classification = asyncio.create_task(
        tracker.classify(
            chat_id=7,
            message_id=202,
            outgoing=True,
            fingerprint=MANUAL,
        )
    )
    await asyncio.sleep(0)
    assert not classification.done()

    reservation.confirm(101)
    assert await classification is MessageOrigin.MANUAL_OUTGOING


@pytest.mark.asyncio
async def test_unresolved_or_uncertain_send_is_fail_closed() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)

    classification = asyncio.create_task(
        tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
    )
    await asyncio.sleep(0)
    assert not classification.done()

    reservation.uncertain()

    assert await classification is MessageOrigin.INDETERMINATE

    assert (
        await tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
        is MessageOrigin.INDETERMINATE
    )
    assert (
        await tracker.classify(
            chat_id=7,
            message_id=202,
            outgoing=True,
            fingerprint=MANUAL,
        )
        is MessageOrigin.INDETERMINATE
    )


@pytest.mark.asyncio
async def test_definitively_failed_send_does_not_hide_manual_message() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)
    classification = asyncio.create_task(
        tracker.classify(
            chat_id=7,
            message_id=202,
            outgoing=True,
            fingerprint=GENERATED,
        )
    )
    await asyncio.sleep(0)

    reservation.failed()

    assert await classification is MessageOrigin.MANUAL_OUTGOING


@pytest.mark.asyncio
async def test_success_without_message_id_does_not_hide_manual_message() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)
    classification = asyncio.create_task(
        tracker.classify(
            chat_id=7,
            message_id=202,
            outgoing=True,
            fingerprint=MANUAL,
        )
    )
    await asyncio.sleep(0)

    reservation.complete_without_message_id()

    assert await classification is MessageOrigin.MANUAL_OUTGOING


@pytest.mark.asyncio
async def test_pending_capacity_is_rejected_before_an_untracked_send() -> None:
    tracker = GeneratedMessageTracker(max_pending=1)
    reservation = tracker.reserve(7, GENERATED)

    with pytest.raises(RuntimeError, match="capacity"):
        tracker.reserve(8, GENERATED)

    reservation.failed()


@pytest.mark.asyncio
async def test_confirmation_rejects_a_missing_native_message_id() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)

    with pytest.raises(ValueError, match="message ID"):
        reservation.confirm(None)  # type: ignore[arg-type]

    reservation.uncertain()


@pytest.mark.asyncio
async def test_matching_confirmation_returns_without_waiting_for_other_send() -> None:
    tracker = GeneratedMessageTracker()
    matching = tracker.reserve(7, GENERATED)
    unrelated = tracker.reserve(7, MANUAL)
    classification = asyncio.create_task(
        tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
    )
    await asyncio.sleep(0)

    matching.confirm(101)

    assert await asyncio.wait_for(classification, timeout=0.1) is (
        MessageOrigin.SIDEKICK_GENERATED
    )
    unrelated.failed()


@pytest.mark.asyncio
async def test_confirmed_capacity_rejects_before_evicting_provenance() -> None:
    tracker = GeneratedMessageTracker(max_confirmed=1)
    first = tracker.reserve(7, GENERATED)
    first.confirm(101)

    with pytest.raises(RuntimeError, match="capacity"):
        tracker.reserve(8, GENERATED)

    assert (
        await tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
        is MessageOrigin.SIDEKICK_GENERATED
    )

    second = tracker.reserve(8, GENERATED)
    second.failed()

    assert (
        await tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
        is MessageOrigin.SIDEKICK_GENERATED
    )


@pytest.mark.asyncio
async def test_unobserved_and_uncertain_provenance_never_expires_open() -> None:
    now = 0.0
    tracker = GeneratedMessageTracker(clock=lambda: now)
    confirmed = tracker.reserve(7, GENERATED)
    confirmed.confirm(101)
    uncertain = tracker.reserve(8, MANUAL)
    uncertain.uncertain()

    now = 10_000_000.0

    assert (
        await tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
        is MessageOrigin.SIDEKICK_GENERATED
    )
    assert (
        await tracker.classify(
            chat_id=8,
            message_id=202,
            outgoing=True,
            fingerprint=MANUAL,
        )
        is MessageOrigin.INDETERMINATE
    )


@pytest.mark.asyncio
async def test_pending_send_waits_for_exact_id_despite_fingerprint_change() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)
    classification = asyncio.create_task(
        tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=MANUAL,
        )
    )
    await asyncio.sleep(0)

    assert not classification.done()
    reservation.confirm(101)
    assert await classification is MessageOrigin.SIDEKICK_GENERATED


@pytest.mark.asyncio
async def test_observed_confirmation_remains_idempotently_generated() -> None:
    tracker = GeneratedMessageTracker()
    reservation = tracker.reserve(7, GENERATED)
    reservation.confirm(101)

    for _ in range(2):
        assert (
            await tracker.classify(
                chat_id=7,
                message_id=101,
                outgoing=True,
                fingerprint=GENERATED,
            )
            is MessageOrigin.SIDEKICK_GENERATED
        )


@pytest.mark.asyncio
async def test_confirmation_without_expected_echo_releases_send_capacity() -> None:
    tracker = GeneratedMessageTracker(max_confirmed=1)
    reservation = tracker.reserve(7, GENERATED)
    reservation.confirm(101, echo_expected=False)

    next_reservation = tracker.reserve(8, MANUAL)
    next_reservation.failed()
    assert (
        await tracker.classify(
            chat_id=7,
            message_id=101,
            outgoing=True,
            fingerprint=GENERATED,
        )
        is MessageOrigin.SIDEKICK_GENERATED
    )
