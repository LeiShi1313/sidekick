from datetime import UTC, datetime
import json
import sqlite3

import pytest

from sidekick.ai import AIStateRepository
from sidekick.ai_memory import MemoryEpisode, MemoryEvent
from sidekick.ai_memory_segments import PendingMemoryDocument


@pytest.mark.asyncio
async def test_memory_outbox_document_and_cursor_survive_repository_restart(tmp_path):
    state_path = tmp_path / "ai.db"
    scope_id = "telegram:chat:-1001"
    event = MemoryEvent(
        source_id="telegram:message:-1001:41",
        actor_id="telegram:user:20",
        actor_display_name="Alice",
        occurred_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        mentioned_at=datetime(2026, 7, 13, 12, 1, tzinfo=UTC),
        reply_to_source_id="telegram:message:-1001:40",
        mentioned_actors=(("telegram:user:30", "Bob"),),
        metadata={"quotation": {"text": "Earlier context"}},
        text="Ship the memory buffer",
    )
    pending = PendingMemoryDocument(
        episode=MemoryEpisode(
            scope_id=scope_id,
            document_id="telegram:memory-session:-1001:20260713T120000Z:41",
            events=(event,),
            scope_display_name="Engineering",
            source="telegram",
        ),
        staged_source_ids=(event.source_id,),
        sealed=True,
    )

    first = await AIStateRepository(state_path).connect()
    await first.set_continuous_memory_enabled(
        scope_id,
        True,
        cursor_message_id=40,
    )
    staged_at = datetime(2026, 7, 13, 12, 2, tzinfo=UTC).timestamp()
    await first.stage_continuous_memory_scan(
        scope_id,
        (pending,),
        cursor_message_id=41,
        scanned_until_at=staged_at,
        succeeded_at=staged_at,
    )
    await first.close()

    second = await AIStateRepository(state_path).connect()
    try:
        outbox = await second.list_memory_outbox_documents(scope_id)
        assert tuple(item.document for item in outbox) == (pending,)
        scope = await second.get_memory_scope_state(scope_id)
        assert scope.continuous_cursor_message_id == 41
        assert scope.continuous_scanned_until_at == staged_at

        await second.complete_memory_outbox_documents(
            scope_id,
            ((pending.episode.document_id, pending.last_event_at.timestamp()),),
            retained_at=staged_at + 1,
        )
        assert await second.list_memory_outbox_documents(scope_id) == ()
        scope = await second.get_memory_scope_state(scope_id)
        assert scope.last_retained_source_at == pending.last_event_at.timestamp()
        assert scope.last_retained_at == staged_at + 1
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_disabling_continuous_memory_seals_queued_documents(tmp_path):
    scope_id = "telegram:chat:-1001"
    event = MemoryEvent(
        source_id="telegram:message:-1001:41",
        actor_id="telegram:user:20",
        actor_display_name="Alice",
        occurred_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        text="Do not retain this after memory is disabled",
    )
    pending = PendingMemoryDocument(
        episode=MemoryEpisode(
            scope_id=scope_id,
            document_id="telegram:memory-session:-1001:20260713T120000Z:41",
            events=(event,),
            source="telegram",
        ),
        staged_source_ids=(event.source_id,),
    )
    store = await AIStateRepository(tmp_path / "ai.db").connect()
    try:
        await store.set_continuous_memory_enabled(
            scope_id,
            True,
            cursor_message_id=40,
        )
        staged_at = datetime(2026, 7, 13, 12, 1, tzinfo=UTC).timestamp()
        await store.stage_continuous_memory_scan(
            scope_id,
            (pending,),
            cursor_message_id=41,
            scanned_until_at=staged_at,
            succeeded_at=staged_at,
        )

        await store.set_continuous_memory_enabled(scope_id, False)

        outbox = await store.list_memory_outbox_documents(scope_id)
        assert len(outbox) == 1
        assert outbox[0].document.sealed is True
        assert outbox[0].dead_lettered_at is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_legacy_pending_documents_migrate_to_the_delivery_outbox(tmp_path):
    path = tmp_path / "ai.db"
    scope_id = "telegram:chat:-1001"
    event = MemoryEvent(
        source_id="telegram:message:-1001:41",
        actor_id="telegram:user:20",
        actor_display_name=None,
        occurred_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        text="Migrate this queued memory",
    )
    episode = MemoryEpisode(
        scope_id=scope_id,
        document_id="telegram:memory-session:-1001:20260713T120000Z:41",
        events=(event,),
        source="telegram",
    )
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE ai_memory_pending_documents (
            scope_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            staged_source_ids TEXT NOT NULL,
            sealed INTEGER NOT NULL DEFAULT 0,
            first_event_at REAL NOT NULL,
            last_event_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (scope_id, document_id)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO ai_memory_pending_documents VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
        """,
        (
            scope_id,
            episode.document_id,
            episode.source,
            episode.content,
            json.dumps((event.source_id,)),
            event.occurred_at.timestamp(),
            event.occurred_at.timestamp(),
            event.occurred_at.timestamp(),
        ),
    )
    connection.commit()
    connection.close()

    store = await AIStateRepository(path).connect()
    try:
        outbox = await store.list_memory_outbox_documents(scope_id)
        assert len(outbox) == 1
        assert outbox[0].pipeline == "continuous"
        assert outbox[0].document.episode == episode
        assert outbox[0].attempt_count == 0
        assert outbox[0].dead_lettered_at is None
    finally:
        await store.close()

    connection = sqlite3.connect(path)
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'ai_memory_pending_documents'"
        ).fetchone()
        assert table is None
    finally:
        connection.close()
