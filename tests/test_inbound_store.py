from __future__ import annotations

import pytest

from sidekick.chat.provenance import MessageOrigin
from sidekick.inbound_store import SQLiteInboundWorkStore


SOURCE_ID = "qq-test"
CHAT_ID = 700
MESSAGE_ID = 101


async def open_store(path) -> SQLiteInboundWorkStore:
    store = await SQLiteInboundWorkStore(path).connect()
    await store.initialize_source(
        SOURCE_ID,
        epoch="account-99",
        initial_cursor=0,
    )
    return store


@pytest.mark.asyncio
async def test_accept_commits_reference_origin_and_cursor_atomically(tmp_path) -> None:
    store = await open_store(tmp_path / "ai.db")
    try:
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=11,
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )

        pending = await store.get_pending_ai_work(
            SOURCE_ID,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.trigger_cursor == 11
        assert pending.attested_origin is MessageOrigin.INCOMING
        assert await store.get_cursor(SOURCE_ID) == 11
        columns = {
            str(row["name"])
            async for row in await store._require_connection().execute(
                "PRAGMA table_info(ai_inbound_work)"
            )
        }
        assert not columns.intersection(
            {"content", "message_payload", "media", "reply_chain"}
        )

        connection = store._require_connection()
        await connection.execute(
            """
            CREATE TRIGGER reject_inbound_cursor_update
            BEFORE UPDATE OF cursor ON ai_inbound_sources
            BEGIN
                SELECT RAISE(ABORT, 'cursor rejected');
            END
            """
        )
        await connection.commit()
        with pytest.raises(Exception, match="cursor rejected"):
            await store.accept_pending_ai_event(
                SOURCE_ID,
                cursor=12,
                chat_id=CHAT_ID,
                message_id=102,
                kind="message",
                attested_origin=MessageOrigin.MANUAL_OUTGOING,
            )
        assert await store.get_pending_ai_work(SOURCE_ID, CHAT_ID, 102) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_newer_source_cursor_survives_stale_worker_completion(tmp_path) -> None:
    store = await open_store(tmp_path / "ai.db")
    try:
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=11,
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        claimed = await store.claim_pending_ai_work(SOURCE_ID, now=100)
        assert claimed is not None
        assert await store.begin_pending_ai_execution(
            claimed,
            version="qq:v1:101",
            now=100,
        ) == "started"

        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=12,
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message_remove",
            attested_origin=None,
        )

        assert await store.complete_pending_ai_work(
            claimed,
            version="qq:v1:101",
            outcome="completed",
            now=101,
        ) is False
        pending = await store.get_pending_ai_work(
            SOURCE_ID,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.trigger_cursor == 12
        assert pending.kind == "message_remove"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_restart_quarantines_execution_with_unknown_side_effects(
    tmp_path,
) -> None:
    path = tmp_path / "ai.db"
    store = await open_store(path)
    await store.accept_pending_ai_event(
        SOURCE_ID,
        cursor=11,
        chat_id=CHAT_ID,
        message_id=MESSAGE_ID,
        kind="message",
        attested_origin=MessageOrigin.INCOMING,
    )
    claimed = await store.claim_pending_ai_work(SOURCE_ID, now=100)
    assert claimed is not None
    assert await store.begin_pending_ai_execution(
        claimed,
        version="qq:v1:101",
        now=100,
    ) == "started"
    await store.close()

    restarted = await SQLiteInboundWorkStore(path).connect()
    try:
        await restarted.recover_pending_ai_work(SOURCE_ID, now=200)
        pending = await restarted.get_pending_ai_work(
            SOURCE_ID,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.status == "failed_unknown"
        assert await restarted.claim_pending_ai_work(SOURCE_ID, now=201) is None
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_new_source_epoch_discards_old_account_work(tmp_path) -> None:
    store = await open_store(tmp_path / "ai.db")
    try:
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=11,
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )

        cursor = await store.initialize_source(
            SOURCE_ID,
            epoch="account-100",
            initial_cursor=90,
        )

        assert cursor == 90
        assert await store.get_pending_ai_work(
            SOURCE_ID,
            CHAT_ID,
            MESSAGE_ID,
        ) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_same_source_epoch_preserves_durable_cursor(tmp_path) -> None:
    store = await open_store(tmp_path / "ai.db")
    try:
        await store.acknowledge_event(SOURCE_ID, 41)

        cursor = await store.initialize_source(
            SOURCE_ID,
            epoch="account-99",
            initial_cursor=99,
        )

        assert cursor == 41
        assert await store.get_cursor(SOURCE_ID) == 41
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_processed_revision_is_deduplicated_by_semantic_version(
    tmp_path,
) -> None:
    store = await open_store(tmp_path / "ai.db")
    try:
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=11,
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
            attested_origin=None,
        )
        first = await store.claim_pending_ai_work(SOURCE_ID, now=100)
        assert first is not None
        assert await store.begin_pending_ai_execution(
            first,
            version="same-revision",
            now=100,
        ) == "started"
        assert await store.complete_pending_ai_work(
            first,
            version="same-revision",
            outcome="completed",
            now=101,
        ) is True

        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=12,
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
            attested_origin=None,
        )
        replay = await store.claim_pending_ai_work(SOURCE_ID, now=102)
        assert replay is not None
        assert await store.begin_pending_ai_execution(
            replay,
            version="same-revision",
            now=102,
        ) == "duplicate"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_retry_backoff_becomes_diagnosably_unavailable(tmp_path) -> None:
    store = await open_store(tmp_path / "ai.db")
    try:
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=11,
            chat_id=CHAT_ID,
            message_id=MESSAGE_ID,
            kind="message",
            attested_origin=None,
        )
        first = await store.claim_pending_ai_work(SOURCE_ID, now=100)
        assert first is not None
        assert await store.defer_pending_ai_work(
            first,
            error_code="SOURCE_NOT_OBSERVED",
            retry_at=102,
            max_attempts=2,
            now=100,
        ) == "pending"
        assert await store.claim_pending_ai_work(SOURCE_ID, now=101) is None

        second = await store.claim_pending_ai_work(SOURCE_ID, now=102)
        assert second is not None
        assert await store.defer_pending_ai_work(
            second,
            error_code="SOURCE_NOT_OBSERVED",
            retry_at=106,
            max_attempts=2,
            now=102,
        ) == "unavailable"

        pending = await store.get_pending_ai_work(
            SOURCE_ID,
            CHAT_ID,
            MESSAGE_ID,
        )
        assert pending is not None
        assert pending.status == "unavailable"
        assert pending.attempt_count == 2
        assert pending.last_error_code == "SOURCE_NOT_OBSERVED"
        assert await store.claim_pending_ai_work(SOURCE_ID, now=1_000) is None
    finally:
        await store.close()
