from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from sidekick.ai import AIStateRepository
from sidekick.inbound_store import SQLiteInboundWorkStore


@pytest.mark.asyncio
async def test_operations_do_not_upgrade_a_stale_wal_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "ai.db"
    scope_id = "wechat:account:wxid_self:chat:filehelper"
    store = await AIStateRepository(path).connect()
    inbox = await SQLiteInboundWorkStore(path).connect()
    await inbox.initialize_source(
        "wechat-peer",
        epoch="wxid_self",
        initial_cursor="0",
    )
    for index in range(130):
        await store.save_memory_document_receipt(
            scope_id,
            f"document-{index:03d}",
            f"hash-{index:03d}",
            (),
        )

    original_fetchmany = aiosqlite.Cursor.fetchmany
    read_started = asyncio.Event()
    release_read = asyncio.Event()

    async def pause_document_read(
        cursor: aiosqlite.Cursor,
        size: int | None = None,
    ):
        rows = await original_fetchmany(cursor, size)
        columns = tuple(column[0] for column in (cursor.description or ()))
        if (
            rows
            and columns == ("document_id", "content_hash", "event_versions")
            and not read_started.is_set()
        ):
            read_started.set()
            await release_read.wait()
        return rows

    monkeypatch.setattr(aiosqlite.Cursor, "fetchmany", pause_document_read)
    document_ids = tuple(f"document-{index:03d}" for index in range(130))
    read_task = asyncio.create_task(
        store.get_memory_document_receipts(scope_id, document_ids)
    )
    write_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(read_started.wait(), timeout=1)
        await inbox.acknowledge_event("wechat-peer", "1")
        write_task = asyncio.create_task(
            store.start_ai_run(
                run_id="run-after-inbox-commit",
                scope_id=scope_id,
                actor_id="wechat:account:wxid_self:user:wxid_self",
                adapter_instance_id="wechat-peer",
                started_at=1,
            )
        )

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(write_task), timeout=0.5)

        release_read.set()
        receipts = await read_task
        await write_task
        await store.mark_ai_run_running(
            "run-after-inbox-commit",
            updated_at=2,
        )

        assert len(receipts) == 130
    finally:
        release_read.set()
        await asyncio.gather(
            read_task,
            *(task for task in (write_task,) if task is not None),
            return_exceptions=True,
        )
        await inbox.close()
        await store.close()


@pytest.mark.asyncio
async def test_database_error_rolls_back_before_the_next_operation(tmp_path) -> None:
    path = tmp_path / "ai.db"
    scope_id = "wechat:account:wxid_self:chat:filehelper"
    store = await AIStateRepository(path).connect()
    inbox = await SQLiteInboundWorkStore(path).connect()
    await inbox.initialize_source(
        "wechat-peer",
        epoch="wxid_self",
        initial_cursor="0",
    )
    connection = store._require_connection()
    await connection.executemany(
        "INSERT INTO ai_chat_access (scope_id, opened_at) VALUES (?, 1)",
        ((scope_id,), ("wechat:account:wxid_self:chat:second",)),
    )
    await connection.commit()

    held_read = await connection.execute(
        "SELECT scope_id FROM ai_chat_access ORDER BY scope_id"
    )
    await held_read.fetchone()
    try:
        await inbox.acknowledge_event("wechat-peer", "1")
        with pytest.raises(aiosqlite.OperationalError) as error:
            await store.start_ai_run(
                run_id="stale-run",
                scope_id=scope_id,
                actor_id="wechat:account:wxid_self:user:wxid_self",
                adapter_instance_id="wechat-peer",
                started_at=1,
            )

        assert error.value.sqlite_errorname == "SQLITE_BUSY_SNAPSHOT"
        await held_read.close()
        await store.start_ai_run(
            run_id="recovered-run",
            scope_id=scope_id,
            actor_id="wechat:account:wxid_self:user:wxid_self",
            adapter_instance_id="wechat-peer",
            started_at=2,
        )
    finally:
        await held_read.close()
        await inbox.close()
        await store.close()
