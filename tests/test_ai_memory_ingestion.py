from datetime import UTC, datetime

import pytest

from sidekick.ai import AIStateRepository
from sidekick.ai_memory import MemoryEpisode, MemoryEvent
from sidekick.ai_memory_ingestion import PendingMemoryDocument


@pytest.mark.asyncio
async def test_pending_memory_document_and_cursor_survive_repository_restart(tmp_path):
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
    await first.stage_continuous_memory_documents(
        scope_id,
        (pending,),
        cursor_message_id=41,
        succeeded_at=datetime(2026, 7, 13, 12, 2, tzinfo=UTC).timestamp(),
    )
    await first.close()

    second = await AIStateRepository(state_path).connect()
    try:
        assert await second.list_pending_memory_documents(scope_id) == (pending,)
        assert (
            await second.get_memory_scope_state(scope_id)
        ).continuous_cursor_message_id == 41

        await second.delete_pending_memory_documents(
            scope_id,
            (pending.episode.document_id,),
        )
        assert await second.list_pending_memory_documents(scope_id) == ()
    finally:
        await second.close()
