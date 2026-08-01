from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sidekick.ai import AIStateRepository
from sidekick.ai_memory import MemoryEpisode, MemoryEvent
from sidekick.ai_memory_segments import PendingMemoryDocument


@pytest.mark.asyncio
async def test_operational_state_aggregates_config_ingestion_and_active_runs(
    tmp_path,
) -> None:
    scope_id = "qq:group:700"
    store = await AIStateRepository(tmp_path / "ai.db").connect()
    try:
        await store.set_chat_access_open(scope_id, True)
        await store.set_model_override(scope_id, "openai/gpt-5")
        await store.set_continuous_memory_enabled(
            scope_id,
            True,
            display_name="Example group",
        )
        event = MemoryEvent(
            source_id="qq:message:group:700:1",
            actor_id="qq:user:42",
            actor_display_name="Alice",
            occurred_at=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
            text="Keep this pending",
        )
        pending = PendingMemoryDocument(
            episode=MemoryEpisode(
                scope_id=scope_id,
                document_id="qq:thread:group:700:1",
                events=(event,),
            ),
            staged_source_ids=(event.source_id,),
        )
        await store.stage_continuous_memory_documents(
            scope_id,
            (pending,),
            cursor_message_id=1,
            succeeded_at=1_800_000_000,
        )
        await store.save_memory_document_receipt(
            scope_id,
            "qq:thread:group:700:retained",
            "hash",
            (("qq:message:group:700:2", "event-hash"),),
        )
        await store.start_ai_run(
            run_id="run-1",
            scope_id=scope_id,
            actor_id="qq:user:42",
            adapter_instance_id="qq-default",
            started_at=1_800_000_010,
        )
        await store.mark_ai_run_running("run-1", updated_at=1_800_000_011)

        states = await store.list_channel_operational_states()

        assert len(states) == 1
        state = states[0]
        assert state.scope_id == scope_id
        assert state.display_name == "Example group"
        assert state.access_open is True
        assert state.model_override == "openai/gpt-5"
        assert state.continuous_enabled is True
        assert state.retained_document_count == 1
        assert state.pending_count == 1
        assert state.last_ingested_at is not None
        assert [run.status for run in state.active_runs] == ["RUNNING"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_repository_reconciles_orphaned_active_runs_on_startup(tmp_path) -> None:
    path = tmp_path / "ai.db"
    first = await AIStateRepository(path).connect()
    await first.start_ai_run(
        run_id="run-orphaned",
        scope_id="telegram:chat:-1001",
        actor_id="telegram:user:42",
        adapter_instance_id="telegram-default",
        started_at=1_800_000_000,
    )
    await first.close()

    second = await AIStateRepository(path).connect()
    try:
        state = (await second.list_channel_operational_states())[0]
        assert state.active_runs == ()
        assert state.last_run_error == "ADAPTER_RESTARTED"
        assert state.last_run_id == "run-orphaned"
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_repository_persists_only_allowlisted_run_error_codes(tmp_path) -> None:
    store = await AIStateRepository(tmp_path / "ai.db").connect()
    try:
        await store.start_ai_run(
            run_id="run-unsafe",
            scope_id="qq:private:42",
            actor_id="qq:user:42",
            adapter_instance_id="qq-default",
            started_at=1_800_000_000,
        )
        await store.finish_ai_run(
            "run-unsafe",
            status="FAILED",
            updated_at=1_800_000_001,
            error_code="prompt=secret token=secret",
        )

        state = (await store.list_channel_operational_states())[0]

        assert state.last_run_error == "HANDLER_ERROR"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_later_success_clears_the_channel_run_error(tmp_path) -> None:
    store = await AIStateRepository(tmp_path / "ai.db").connect()
    try:
        for run_id, started_at, status, error_code in (
            ("run-failed", 1_800_000_000, "FAILED", "AGENT_ERROR"),
            ("run-succeeded", 1_800_000_010, "COMPLETED", None),
        ):
            await store.start_ai_run(
                run_id=run_id,
                scope_id="telegram:chat:-1001",
                actor_id="telegram:user:42",
                adapter_instance_id="telegram-default",
                started_at=started_at,
            )
            await store.finish_ai_run(
                run_id,
                status=status,
                updated_at=started_at + 1,
                error_code=error_code,
            )

        state = (await store.list_channel_operational_states())[0]

        assert state.last_run_error is None
        assert state.last_run_error_at is None
        assert state.last_run_id is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_active_run_does_not_clear_the_previous_channel_error(tmp_path) -> None:
    store = await AIStateRepository(tmp_path / "ai.db").connect()
    try:
        await store.start_ai_run(
            run_id="run-failed",
            scope_id="telegram:chat:-1001",
            actor_id="telegram:user:42",
            adapter_instance_id="telegram-default",
            started_at=1_800_000_000,
        )
        await store.finish_ai_run(
            "run-failed",
            status="FAILED",
            updated_at=1_800_000_001,
            error_code="AGENT_ERROR",
        )
        await store.start_ai_run(
            run_id="run-active",
            scope_id="telegram:chat:-1001",
            actor_id="telegram:user:42",
            adapter_instance_id="telegram-default",
            started_at=1_800_000_010,
        )

        state = (await store.list_channel_operational_states())[0]

        assert [run.run_id for run in state.active_runs] == ["run-active"]
        assert state.last_run_error == "AGENT_ERROR"
        assert state.last_run_id == "run-failed"
    finally:
        await store.close()
