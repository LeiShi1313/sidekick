from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sqlite3
import time

import pytest

from sidekick.inbound_store import SQLiteInboundWorkStore


SOURCE_ID = "schema-upgrade-source"


def create_origin_main_inbound_database(path: Path) -> None:
    """Create the inbound schema exactly as defined on origin/main."""
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ai_inbound_sources (
                source_id TEXT PRIMARY KEY,
                epoch TEXT NOT NULL,
                cursor BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ai_inbound_work (
                source_id TEXT NOT NULL,
                chat_id BLOB NOT NULL,
                message_id BLOB NOT NULL,
                trigger_cursor BLOB NOT NULL,
                kind TEXT NOT NULL
                    CHECK (kind IN ('message', 'message_remove')),
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'unavailable', 'failed_unknown'
                    )),
                attempt_count INTEGER NOT NULL DEFAULT 0
                    CHECK (attempt_count >= 0),
                next_attempt_at REAL NOT NULL DEFAULT 0,
                last_error_code TEXT,
                attested_origin TEXT CHECK (
                    attested_origin IS NULL OR attested_origin IN (
                        'incoming', 'manual-outgoing'
                    )
                ),
                lease_id TEXT,
                lease_trigger_cursor BLOB,
                current_version TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (source_id, chat_id, message_id),
                FOREIGN KEY (source_id)
                    REFERENCES ai_inbound_sources(source_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ai_inbound_work_due
            ON ai_inbound_work (
                source_id, status, next_attempt_at, updated_at
            );
            CREATE TABLE IF NOT EXISTS ai_inbound_revisions (
                source_id TEXT NOT NULL,
                chat_id BLOB NOT NULL,
                message_id BLOB NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN (
                    'running', 'completed', 'ignored', 'recalled', 'failed',
                    'failed_unknown'
                )),
                started_at REAL NOT NULL,
                finished_at REAL,
                PRIMARY KEY (source_id, chat_id, message_id, version),
                FOREIGN KEY (source_id)
                    REFERENCES ai_inbound_sources(source_id)
                    ON DELETE CASCADE
            );
            """
        )
        connection.execute(
            """
            INSERT INTO ai_inbound_sources (
                source_id, epoch, cursor, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (SOURCE_ID, "legacy-epoch", "legacy-cursor", 1.0),
        )
        connection.execute(
            """
            INSERT INTO ai_inbound_work (
                source_id, chat_id, message_id, trigger_cursor, kind,
                updated_at
            ) VALUES (?, ?, ?, ?, 'message', ?)
            """,
            (
                SOURCE_ID,
                "legacy-chat",
                "legacy-message",
                "legacy-event",
                9_999_999_999.0,
            ),
        )


async def promote_generation(
    store: SQLiteInboundWorkStore,
    *,
    cursor: str,
    message_id: str,
    now: float,
) -> None:
    await store.accept_pending_ai_event(
        SOURCE_ID,
        cursor=cursor,
        chat_id="chat-1",
        message_id=message_id,
        kind="message",
        attested_origin=None,
    )
    work = await store.claim_pending_ai_work(SOURCE_ID, now=now)
    assert work is not None
    assert work.message_id == message_id
    promotion = await store.promote_pending_ai_generation(
        work,
        version=f"version:{message_id}",
        principal_actor_id="actor-1",
        scope_id="scope-1",
        is_owner=True,
        eligible_at=now,
        now=now,
    )
    assert promotion == "queued"


@pytest.mark.asyncio
async def test_origin_main_schema_upgrades_acceptance_sequence_counter_and_index(
    tmp_path,
) -> None:
    database = tmp_path / "ai.db"
    create_origin_main_inbound_database(database)

    store = await SQLiteInboundWorkStore(database).connect()
    try:
        assert (
            await store.initialize_source(
                SOURCE_ID,
                epoch="legacy-epoch",
                initial_cursor="ignored-for-existing-epoch",
            )
            == "legacy-cursor"
        )
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor="new-event",
            chat_id="new-chat",
            message_id="new-message",
            kind="message",
            attested_origin=None,
        )

        first = await store.claim_pending_ai_work(SOURCE_ID, now=time.time())
        assert first is not None
        assert first.message_id == "legacy-message"
        assert first.acceptance_sequence == 1
        await store.release_pending_ai_work(first)
    finally:
        await store.close()

    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ai_inbound_work)")
        }
        counter = connection.execute(
            """
            SELECT last_sequence
            FROM ai_inbound_acceptance_counters
            WHERE source_id = ?
            """,
            (SOURCE_ID,),
        ).fetchone()
        work_sequences = dict(
            connection.execute(
                "SELECT message_id, acceptance_sequence FROM ai_inbound_work"
            )
        )
        due_index_columns = [
            row[2]
            for row in connection.execute("PRAGMA index_info(ai_inbound_work_due)")
        ]

    assert "acceptance_sequence" in columns
    assert counter == (2,)
    assert work_sequences == {"legacy-message": 1, "new-message": 2}
    assert due_index_columns == [
        "source_id",
        "status",
        "next_attempt_at",
        "acceptance_sequence",
    ]


@pytest.mark.asyncio
async def test_legacy_fifo_backfill_uses_time_before_lexical_identity(
    tmp_path,
) -> None:
    database = tmp_path / "ai.db"
    create_origin_main_inbound_database(database)
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO ai_inbound_work (
                source_id, chat_id, message_id, trigger_cursor, kind,
                updated_at
            ) VALUES (?, ?, ?, ?, 'message', ?)
            """,
            (
                (SOURCE_ID, "z-chat", "z-first", "event-first", 10.0),
                (SOURCE_ID, "a-chat", "a-second", "event-second", 20.0),
            ),
        )

    store = await SQLiteInboundWorkStore(database).connect()
    try:
        await store.initialize_source(
            SOURCE_ID,
            epoch="legacy-epoch",
            initial_cursor="ignored-for-existing-epoch",
        )
    finally:
        await store.close()

    with sqlite3.connect(database) as connection:
        ordered = list(
            connection.execute(
                """
                SELECT message_id, acceptance_sequence
                FROM ai_inbound_work
                ORDER BY acceptance_sequence
                """
            )
        )
        counter = connection.execute(
            """
            SELECT last_sequence
            FROM ai_inbound_acceptance_counters
            WHERE source_id = ?
            """,
            (SOURCE_ID,),
        ).fetchone()

    assert ordered == [
        ("z-first", 1),
        ("a-second", 2),
        ("legacy-message", 3),
    ]
    assert counter == (3,)


@pytest.mark.asyncio
async def test_terminal_generation_jobs_survive_reconnect_then_age_out(
    tmp_path,
) -> None:
    database = tmp_path / "ai.db"
    now = time.time()
    store = await SQLiteInboundWorkStore(database).connect()
    await store.initialize_source(
        SOURCE_ID,
        epoch="current-epoch",
        initial_cursor="initial-cursor",
    )
    try:
        await promote_generation(
            store,
            cursor="event-1",
            message_id="completed-message",
            now=now,
        )
        completed = await store.claim_pending_ai_generation(SOURCE_ID, now=now)
        assert completed is not None
        assert await store.begin_ai_generation(completed, now=now) == "started"
        assert await store.complete_ai_generation(
            completed,
            outcome="completed",
            now=now,
        )

        await promote_generation(
            store,
            cursor="event-2",
            message_id="unknown-message",
            now=now,
        )
        unknown = await store.claim_pending_ai_generation(SOURCE_ID, now=now)
        assert unknown is not None
        assert await store.begin_ai_generation(unknown, now=now) == "started"
        assert await store.mark_ai_generation_unknown(unknown, now=now)
    finally:
        await store.close()

    reopened = await SQLiteInboundWorkStore(database).connect()
    try:
        retained = await reopened.get_ai_generation_job(completed.job_id)
        assert retained is not None
        assert retained.status == "completed"
        retained_unknown = await reopened.get_ai_generation_job(unknown.job_id)
        assert retained_unknown is not None
        assert retained_unknown.status == "failed_unknown"
    finally:
        await reopened.close()

    old = now - 31 * 24 * 60 * 60
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE ai_generation_jobs SET updated_at = ?",
            (old,),
        )

    pruned = await SQLiteInboundWorkStore(database).connect()
    try:
        assert await pruned.get_ai_generation_job(completed.job_id) is None
        assert await pruned.get_ai_generation_job(unknown.job_id) is None
    finally:
        await pruned.close()


@pytest.mark.asyncio
async def test_generation_queue_snapshot_is_aggregate_and_content_free(
    tmp_path,
) -> None:
    database = tmp_path / "ai.db"
    now = time.time()
    store = await SQLiteInboundWorkStore(database).connect()
    await store.initialize_source(
        SOURCE_ID,
        epoch="current-epoch",
        initial_cursor="initial-cursor",
    )
    try:
        await promote_generation(
            store,
            cursor="event-1",
            message_id="message-1",
            now=now,
        )
        await promote_generation(
            store,
            cursor="event-2",
            message_id="message-2",
            now=now,
        )
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor="event-3",
            chat_id="chat-1",
            message_id="message-3",
            kind="message",
            attested_origin=None,
        )
        pending = await store.get_pending_ai_work(
            SOURCE_ID,
            "chat-1",
            "message-3",
        )
        assert pending is not None

        snapshot = await store.get_ai_generation_queue_snapshot(SOURCE_ID)
    finally:
        await store.close()

    assert asdict(snapshot) == {
        "pending_intake": 1,
        "queued": 2,
        "active": 0,
        "failed_unknown": 0,
        "oldest_pending_intake_at": pytest.approx(pending.updated_at),
        "oldest_queued_at": pytest.approx(now),
    }
    assert set(asdict(snapshot)) == {
        "pending_intake",
        "queued",
        "active",
        "failed_unknown",
        "oldest_pending_intake_at",
        "oldest_queued_at",
    }

    with sqlite3.connect(database) as connection:
        queue_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(ai_generation_jobs)")
        }
    assert queue_columns.isdisjoint(
        {"content", "payload", "raw_text", "message_text", "chat_history"}
    )
