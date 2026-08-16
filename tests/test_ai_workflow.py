from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import aiosqlite
import pytest

from sidekick.ai import AIMessageClassification, AIWorkflowCancellation
from sidekick.ai_workflow import AIWorkflow
from sidekick.chat.provenance import MessageOrigin
from sidekick.inbound import InboundSourceRevision, InboundSourceUnavailable
import sidekick.inbound_store as inbound_store_module
from sidekick.inbound_store import SQLiteInboundWorkStore, StoredGenerationJob


SOURCE_ID = "workflow-test"
CHAT_ID = 700


@dataclass(slots=True)
class _Clock:
    value: float = 100.0

    def __call__(self) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class _Message:
    id: int
    chat_id: int
    principal_actor_id: str
    version: str
    kind: Literal["generation", "cancel", "immediate"] = "generation"
    cooldown_after: float | None = None
    is_owner: bool = False


class _Source:
    def __init__(self) -> None:
        self.messages: dict[int, _Message] = {}
        self.block_first_generation_fetch_for: int | None = None
        self.generation_fetch_started = asyncio.Event()
        self.release_generation_fetch = asyncio.Event()
        self._generation_fetch_blocked = False
        self.generation_errors: dict[int, list[InboundSourceUnavailable]] = {}

    def put(self, message: _Message) -> None:
        self.messages[message.id] = message

    async def fetch(self, work: Any) -> InboundSourceRevision[_Message]:
        message = self.messages[int(work.message_id)]
        captured = message
        if (
            isinstance(work, StoredGenerationJob)
            and work.message_id == self.block_first_generation_fetch_for
            and not self._generation_fetch_blocked
        ):
            self._generation_fetch_blocked = True
            self.generation_fetch_started.set()
            await self.release_generation_fetch.wait()
        if isinstance(work, StoredGenerationJob):
            errors = self.generation_errors.get(int(work.message_id))
            if errors:
                raise errors.pop(0)
        return InboundSourceRevision(
            version=f"revision:{captured.version}",
            state="present",
            payload=captured,
            attested_origin=work.attested_origin,
        )

    async def materialize(self, payload: _Message) -> _Message:
        return payload


class _Handler:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.control: Any | None = None
        self.started: dict[int, asyncio.Event] = {}
        self.release: dict[int, asyncio.Event] = {}
        self.version_started: dict[tuple[int, str], asyncio.Event] = {}
        self.version_release: dict[tuple[int, str], asyncio.Event] = {}
        self.handled: list[tuple[int, str]] = []
        self.cooldown_until: dict[str, float] = {}
        self.cancel_results: list[AIWorkflowCancellation] = []
        self.cancel_handled = asyncio.Event()
        self.cancel_release: asyncio.Event | None = None
        self.notices: list[tuple[int, str]] = []
        self.notice_events: dict[int, asyncio.Event] = {}
        self.immediate_handled: list[tuple[int, str]] = []
        self.immediate_events: dict[int, asyncio.Event] = {}

    def gate(self, message_id: int) -> asyncio.Event:
        gate = self.release.setdefault(message_id, asyncio.Event())
        self.started.setdefault(message_id, asyncio.Event())
        return gate

    def started_event(self, message_id: int) -> asyncio.Event:
        return self.started.setdefault(message_id, asyncio.Event())

    def gate_version(self, message_id: int, version: str) -> asyncio.Event:
        key = (message_id, version)
        gate = self.version_release.setdefault(key, asyncio.Event())
        self.version_started.setdefault(key, asyncio.Event())
        return gate

    def started_version_event(
        self,
        message_id: int,
        version: str,
    ) -> asyncio.Event:
        return self.version_started.setdefault(
            (message_id, version),
            asyncio.Event(),
        )

    def notice_event(self, message_id: int) -> asyncio.Event:
        return self.notice_events.setdefault(message_id, asyncio.Event())

    def immediate_event(self, message_id: int) -> asyncio.Event:
        return self.immediate_events.setdefault(message_id, asyncio.Event())

    def bind_workflow_control(self, control: Any) -> None:
        assert self.control is None
        self.control = control

    def unbind_workflow_control(self, control: Any) -> None:
        if self.control is control:
            self.control = None

    async def classify(
        self,
        message: _Message,
        *,
        attested_origin: MessageOrigin | None = None,
    ) -> AIMessageClassification:
        del attested_origin
        if message.kind != "generation":
            return AIMessageClassification("immediate")
        return AIMessageClassification(
            "generation",
            principal_actor_id=message.principal_actor_id,
            scope_id=f"scope:{message.chat_id}",
            is_owner=message.is_owner,
        )

    async def generation_eligible_at(
        self,
        classification: AIMessageClassification,
    ) -> float:
        assert classification.principal_actor_id is not None
        return max(
            self.clock(),
            self.cooldown_until.get(
                classification.principal_actor_id,
                self.clock(),
            ),
        )

    async def reply_workflow_notice(
        self,
        message: _Message,
        notice: Literal["queued", "queue_full"],
    ) -> None:
        self.notices.append((message.id, notice))
        self.notice_event(message.id).set()

    async def handle(
        self,
        message: _Message,
        *,
        attested_origin: MessageOrigin | None = None,
        workflow_admitted: bool = False,
    ) -> bool:
        del attested_origin
        if message.kind == "cancel":
            assert not workflow_admitted
            assert self.control is not None
            self.cancel_results.append(
                await self.control.cancel_generations(
                    message.principal_actor_id,
                    interrupt_running=True,
                )
            )
            self.cancel_handled.set()
            if self.cancel_release is not None:
                await self.cancel_release.wait()
            return True

        if message.kind == "immediate":
            assert not workflow_admitted
            self.immediate_handled.append((message.id, message.version))
            self.immediate_event(message.id).set()
            return True

        assert workflow_admitted
        self.handled.append((message.id, message.version))
        self.started_event(message.id).set()
        self.started_version_event(message.id, message.version).set()
        gate = self.version_release.get(
            (message.id, message.version),
            self.release.get(message.id),
        )
        if gate is not None:
            await gate.wait()
        if message.cooldown_after is not None:
            self.cooldown_until[message.principal_actor_id] = message.cooldown_after
        return True


async def _open_store(path: Path) -> SQLiteInboundWorkStore:
    store = await SQLiteInboundWorkStore(path).connect()
    await store.initialize_source(
        SOURCE_ID,
        epoch="test-account",
        initial_cursor=0,
    )
    return store


def _workflow(
    source: _Source,
    store: SQLiteInboundWorkStore,
    handler: _Handler,
    *,
    generation_concurrency: int,
    clock: _Clock,
) -> AIWorkflow[_Message]:
    return AIWorkflow(
        source,
        store,
        SOURCE_ID,
        cast(Any, handler),
        generation_concurrency=generation_concurrency,
        clock=clock,
    )


async def _accept(
    workflow: AIWorkflow[_Message],
    source: _Source,
    message: _Message,
    *,
    cursor: int,
) -> None:
    source.put(message)
    await workflow.accept(
        cursor=cursor,
        chat_id=message.chat_id,
        message_id=message.id,
        kind="message",
        attested_origin=MessageOrigin.INCOMING,
    )


async def _job_rows(path: Path, message_id: int | None = None) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path) as connection:
        connection.row_factory = aiosqlite.Row
        if message_id is None:
            cursor = await connection.execute(
                "SELECT * FROM ai_generation_jobs ORDER BY queue_sequence"
            )
        else:
            cursor = await connection.execute(
                """
                SELECT * FROM ai_generation_jobs
                WHERE source_id = ? AND chat_id = ? AND message_id = ?
                ORDER BY queue_sequence
                """,
                (SOURCE_ID, CHAT_ID, message_id),
            )
        return [dict(row) for row in await cursor.fetchall()]


async def _wait_for_job(
    path: Path,
    message_id: int,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = 2.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    last_rows: list[dict[str, Any]] = []
    while asyncio.get_running_loop().time() < deadline:
        last_rows = await _job_rows(path, message_id)
        if last_rows and predicate(last_rows[0]):
            return last_rows[0]
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"generation job {message_id} did not reach expected state: {last_rows}"
    )


async def _wait_for_job_rows(
    path: Path,
    message_id: int,
    predicate: Callable[[list[dict[str, Any]]], bool],
    *,
    timeout: float = 2.0,
) -> list[dict[str, Any]]:
    deadline = asyncio.get_running_loop().time() + timeout
    last_rows: list[dict[str, Any]] = []
    while asyncio.get_running_loop().time() < deadline:
        last_rows = await _job_rows(path, message_id)
        if predicate(last_rows):
            return last_rows
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"generation jobs {message_id} did not reach expected state: {last_rows}"
    )


async def _promote_direct_generation(
    store: SQLiteInboundWorkStore,
    *,
    cursor: int,
    message_id: int,
    version: str,
    now: float = 100,
) -> str:
    await store.accept_pending_ai_event(
        SOURCE_ID,
        cursor=cursor,
        chat_id=CHAT_ID,
        message_id=message_id,
        kind="message",
        attested_origin=MessageOrigin.INCOMING,
    )
    work = await store.claim_pending_ai_work(SOURCE_ID, now=now)
    assert work is not None
    assert work.message_id == message_id
    return await store.promote_pending_ai_generation(
        work,
        version=version,
        principal_actor_id="principal:alice",
        scope_id=f"scope:{CHAT_ID}",
        is_owner=False,
        eligible_at=now,
        now=now,
    )


async def _claim_direct_head_with_tail(
    store: SQLiteInboundWorkStore,
) -> StoredGenerationJob:
    assert (
        await _promote_direct_generation(
            store,
            cursor=1,
            message_id=1,
            version="revision:v1",
        )
        == "queued"
    )
    head = await store.claim_pending_ai_generation(SOURCE_ID, now=100)
    assert head is not None
    assert head.message_id == 1
    assert (
        await _promote_direct_generation(
            store,
            cursor=2,
            message_id=2,
            version="revision:tail",
        )
        == "waiting"
    )
    return head


@pytest.mark.asyncio
async def test_non_owner_rapid_requests_admit_head_and_fifo_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    second_release = handler.gate(2)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await _accept(
            workflow,
            source,
            _Message(2, CHAT_ID, "principal:alice", "v1"),
            cursor=2,
        )

        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)
        await _wait_for_job(path, 2, lambda row: row["status"] == "queued")
        await asyncio.wait_for(handler.notice_event(2).wait(), timeout=2)
        await asyncio.sleep(0.05)
        assert not handler.started_event(2).is_set()
        assert handler.notices == [(2, "queued")]

        first_release.set()
        await asyncio.wait_for(handler.started_event(2).wait(), timeout=2)
        assert handler.handled == [(1, "v1"), (2, "v1")]
        second_release.set()

        await _wait_for_job(path, 1, lambda row: row["status"] == "completed")
        await _wait_for_job(path, 2, lambda row: row["status"] == "completed")
    finally:
        first_release.set()
        second_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_non_owner_active_plus_tail_rejects_overflow_with_notice(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    second_release = handler.gate(2)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)

        await _accept(
            workflow,
            source,
            _Message(2, CHAT_ID, "principal:alice", "v1"),
            cursor=2,
        )
        await _wait_for_job(path, 2, lambda row: row["status"] == "queued")
        await asyncio.wait_for(handler.notice_event(2).wait(), timeout=2)

        await _accept(
            workflow,
            source,
            _Message(3, CHAT_ID, "principal:alice", "v1"),
            cursor=3,
        )
        await asyncio.wait_for(handler.notice_event(3).wait(), timeout=2)
        assert handler.notices == [(2, "queued"), (3, "queue_full")]
        assert await _job_rows(path, 3) == []

        first_release.set()
        await asyncio.wait_for(handler.started_event(2).wait(), timeout=2)
        assert not handler.started_event(3).is_set()
        second_release.set()
        await _wait_for_job(path, 1, lambda row: row["status"] == "completed")
        await _wait_for_job(path, 2, lambda row: row["status"] == "completed")
    finally:
        first_release.set()
        second_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_owner_distinct_requests_execute_concurrently(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    second_release = handler.gate(2)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:owner", "v1", is_owner=True),
            cursor=1,
        )
        await _accept(
            workflow,
            source,
            _Message(2, CHAT_ID, "principal:owner", "v1", is_owner=True),
            cursor=2,
        )

        await asyncio.wait_for(
            asyncio.gather(
                handler.started_event(1).wait(),
                handler.started_event(2).wait(),
            ),
            timeout=2,
        )
        assert {message_id for message_id, _version in handler.handled} == {1, 2}

        first_release.set()
        second_release.set()
        await _wait_for_job(path, 1, lambda row: row["status"] == "completed")
        await _wait_for_job(path, 2, lambda row: row["status"] == "completed")
    finally:
        first_release.set()
        second_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_owner_revision_remains_queued_after_active_message_is_unknown(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:owner", "v1", is_owner=True),
            cursor=1,
        )
        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)

        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:owner", "v2", is_owner=True),
            cursor=2,
        )
        active_rows = await _wait_for_job_rows(
            path,
            1,
            lambda rows: len(rows) == 2
            and rows[0]["status"] == "running"
            and rows[1]["status"] == "queued",
        )
        assert [row["expected_version"] for row in active_rows] == [
            "revision:v1",
            "revision:v2",
        ]
        await asyncio.sleep(0.05)
        assert handler.handled == [(1, "v1")]

        await workflow.close()
        unknown_rows = await _wait_for_job_rows(
            path,
            1,
            lambda rows: len(rows) == 2
            and rows[0]["status"] == "failed_unknown"
            and rows[1]["status"] == "queued",
        )
        assert unknown_rows[0]["last_error_code"] == "ADAPTER_RESTARTED"
        assert unknown_rows[1]["last_error_code"] is None

        # Unknown work is terminal and never replayed, but it does not fence a
        # later revision of the same stable message ID.
        await store.recover_pending_ai_work(SOURCE_ID, now=200)
        next_job = await store.claim_pending_ai_generation(SOURCE_ID, now=201)
        assert next_job is not None
        assert next_job.expected_version == "revision:v2"
    finally:
        first_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_different_principals_execute_concurrently(tmp_path: Path) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    second_release = handler.gate(2)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await _accept(
            workflow,
            source,
            _Message(2, CHAT_ID, "principal:bob", "v1"),
            cursor=2,
        )

        await asyncio.wait_for(
            asyncio.gather(
                handler.started_event(1).wait(),
                handler.started_event(2).wait(),
            ),
            timeout=2,
        )
        assert {message_id for message_id, _version in handler.handled} == {1, 2}
        first_release.set()
        second_release.set()
        await _wait_for_job(path, 1, lambda row: row["status"] == "completed")
        await _wait_for_job(path, 2, lambda row: row["status"] == "completed")
    finally:
        first_release.set()
        second_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_cooldown_defers_queued_job_instead_of_rejecting_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock(100)
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(
                1,
                CHAT_ID,
                "principal:alice",
                "v1",
                cooldown_after=130,
            ),
            cursor=1,
        )
        await _accept(
            workflow,
            source,
            _Message(2, CHAT_ID, "principal:alice", "v1"),
            cursor=2,
        )
        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)
        await _wait_for_job(path, 2, lambda row: row["status"] == "queued")

        first_release.set()
        deferred = await _wait_for_job(
            path,
            2,
            lambda row: row["status"] == "queued"
            and row["last_error_code"] == "COOLDOWN",
        )
        assert deferred["eligible_at"] == 130
        assert deferred["attempt_count"] == 1
        assert not handler.started_event(2).is_set()

        clock.value = 130
        workflow.notify()
        await asyncio.wait_for(handler.started_event(2).wait(), timeout=2)
        await _wait_for_job(path, 2, lambda row: row["status"] == "completed")
    finally:
        first_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_new_finite_source_error_resets_cooldown_attempt_accounting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock(100)
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(
                1,
                CHAT_ID,
                "principal:alice",
                "v1",
                cooldown_after=130,
            ),
            cursor=1,
        )
        await _accept(
            workflow,
            source,
            _Message(2, CHAT_ID, "principal:alice", "v1"),
            cursor=2,
        )
        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)
        first_release.set()
        await _wait_for_job(
            path,
            2,
            lambda row: row["status"] == "queued"
            and row["last_error_code"] == "COOLDOWN"
            and row["attempt_count"] == 1,
        )

        source.generation_errors[2] = [
            InboundSourceUnavailable("FINITE_SOURCE", max_attempts=2),
            InboundSourceUnavailable("FINITE_SOURCE", max_attempts=2),
        ]
        clock.value = 130
        workflow.notify()
        first_source_failure = await _wait_for_job(
            path,
            2,
            lambda row: row["status"] == "queued"
            and row["last_error_code"] == "FINITE_SOURCE",
        )
        assert first_source_failure["attempt_count"] == 1
        assert first_source_failure["eligible_at"] == 132

        clock.value = 132
        workflow.notify()
        exhausted = await _wait_for_job(
            path,
            2,
            lambda row: row["status"] == "source_unavailable",
        )
        assert exhausted["last_error_code"] == "FINITE_SOURCE"
        assert not handler.started_event(2).is_set()
    finally:
        first_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_lowered_cooldown_reschedule_wakes_deferred_job(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock(100)
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(
                1,
                CHAT_ID,
                "principal:alice",
                "v1",
                cooldown_after=200,
            ),
            cursor=1,
        )
        await _accept(
            workflow,
            source,
            _Message(2, CHAT_ID, "principal:alice", "v1"),
            cursor=2,
        )
        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)
        first_release.set()
        await _wait_for_job(
            path,
            2,
            lambda row: row["status"] == "queued"
            and row["last_error_code"] == "COOLDOWN"
            and row["eligible_at"] == 200,
        )

        handler.cooldown_until["principal:alice"] = 100
        assert await workflow.reschedule_scope(f"scope:{CHAT_ID}") == 1

        await asyncio.wait_for(handler.started_event(2).wait(), timeout=2)
        await _wait_for_job(path, 2, lambda row: row["status"] == "completed")
        assert clock.value == 100
    finally:
        first_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_cancel_control_bypasses_running_generation_and_cancels_queue(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate(1)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=1,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await _accept(
            workflow,
            source,
            _Message(2, CHAT_ID, "principal:alice", "v1"),
            cursor=2,
        )
        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)
        await _wait_for_job(path, 2, lambda row: row["status"] == "queued")

        await _accept(
            workflow,
            source,
            _Message(
                3,
                CHAT_ID,
                "principal:alice",
                "v1",
                kind="cancel",
            ),
            cursor=3,
        )

        await asyncio.wait_for(handler.cancel_handled.wait(), timeout=2)
        assert handler.cancel_results == [AIWorkflowCancellation(queued=1, running=1)]
        await _wait_for_job(path, 2, lambda row: row["status"] == "cancelled")
        interrupted = await _wait_for_job(
            path,
            1,
            lambda row: row["status"] == "failed_unknown",
        )
        assert interrupted["last_error_code"] == "USER_CANCELLED_OUTCOME_UNKNOWN"

        # Cancelling a generation child must not kill its lane or permanently
        # fence the same Principal after the accepted unknown outcome.
        await _accept(
            workflow,
            source,
            _Message(4, CHAT_ID, "principal:alice", "v1"),
            cursor=4,
        )
        await asyncio.wait_for(handler.started_event(4).wait(), timeout=2)
        await _wait_for_job(path, 4, lambda row: row["status"] == "completed")
    finally:
        first_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_cancel_terminalizes_deferred_edit_of_known_generation(
    tmp_path: Path,
) -> None:
    store = await _open_store(tmp_path / "ai.db")
    try:
        assert (
            await _promote_direct_generation(
                store,
                cursor=1,
                message_id=1,
                version="revision:v1",
            )
            == "queued"
        )
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=2,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        edit = await store.claim_pending_ai_work(SOURCE_ID, now=101)
        assert edit is not None
        assert (
            await store.defer_pending_ai_work(
                edit,
                error_code="SOURCE_NOT_READY",
                retry_at=200,
                max_attempts=None,
                now=101,
            )
            == "pending"
        )

        assert await store.request_ai_generation_cancellation(
            SOURCE_ID,
            "principal:alice",
            now=102,
        ) == (1, 0)
        pending = await store.get_pending_ai_work(SOURCE_ID, CHAT_ID, 1)
        assert pending is not None
        assert pending.status == "unavailable"
        assert pending.last_error_code == "USER_CANCELLED"
        assert await store.claim_pending_ai_work(SOURCE_ID, now=300) is None
        rows = await _job_rows(store.path, 1)
        assert rows[0]["status"] == "cancelled"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancel_interrupts_claimed_pre_execution_work_and_frees_lane(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    source.block_first_generation_fetch_for = 1
    handler = _Handler(clock)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=1,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await asyncio.wait_for(source.generation_fetch_started.wait(), timeout=2)

        await _accept(
            workflow,
            source,
            _Message(
                2,
                CHAT_ID,
                "principal:alice",
                "v1",
                kind="cancel",
            ),
            cursor=2,
        )
        await asyncio.wait_for(handler.cancel_handled.wait(), timeout=2)
        assert handler.cancel_results == [AIWorkflowCancellation(queued=1, running=0)]
        await _wait_for_job(path, 1, lambda row: row["status"] == "cancelled")

        # The cancelled task is still inside source preparation. It must be
        # interrupted so one slow fetch cannot occupy the only lane forever.
        await _accept(
            workflow,
            source,
            _Message(3, CHAT_ID, "principal:bob", "v1"),
            cursor=3,
        )
        await asyncio.wait_for(handler.started_event(3).wait(), timeout=2)
        await _wait_for_job(path, 3, lambda row: row["status"] == "completed")
        assert not handler.started_event(1).is_set()
    finally:
        source.release_generation_fetch.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_recall_cancellation_racing_workflow_close_does_not_hang(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    generation_release = handler.gate(1)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=1,
        clock=clock,
    )
    workflow.start()
    close_tasks: list[asyncio.Task[None]] = []
    cancellation_started = asyncio.Event()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)

        cancel_message = workflow.cancel_message

        def cancel_then_close(chat_id: int, message_id: int) -> None:
            cancel_message(chat_id, message_id)
            close_tasks.append(asyncio.create_task(workflow.close()))
            cancellation_started.set()

        workflow.cancel_message = cast(Any, cancel_then_close)
        await asyncio.wait_for(
            workflow.accept(
                cursor=2,
                chat_id=CHAT_ID,
                message_id=1,
                kind="message_remove",
                attested_origin=MessageOrigin.INCOMING,
            ),
            timeout=2,
        )
        await asyncio.wait_for(cancellation_started.wait(), timeout=2)
        assert len(close_tasks) == 1
        await asyncio.wait_for(close_tasks[0], timeout=2)

        interrupted = await _wait_for_job(
            path,
            1,
            lambda row: row["status"] == "failed_unknown",
        )
        assert interrupted["last_error_code"] in {
            "SOURCE_RECALLED",
            "ADAPTER_RESTARTED",
        }
        assert workflow._tasks == ()
    finally:
        generation_release.set()
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_cancel_cannot_overtake_earlier_generation_with_same_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inbound_store_module,
        "time",
        SimpleNamespace(time=lambda: 123.0),
    )
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    handler.cancel_release = asyncio.Event()
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=1,
        clock=clock,
    )
    try:
        # The later control deliberately has the lower message ID. Intake order
        # must follow acceptance, not a wall-clock tie-break by native identity.
        await _accept(
            workflow,
            source,
            _Message(20, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await _accept(
            workflow,
            source,
            _Message(
                10,
                CHAT_ID,
                "principal:alice",
                "v1",
                kind="cancel",
            ),
            cursor=2,
        )
        workflow.start()

        await asyncio.wait_for(handler.cancel_handled.wait(), timeout=2)
        # Holding the control handler also holds the single intake lane. The
        # earlier generation can exist only if its promotion happened first.
        assert await _job_rows(path, 20)
    finally:
        assert handler.cancel_release is not None
        handler.cancel_release.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_edit_while_old_revision_is_leased_executes_only_new_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    source.block_first_generation_fetch_for = 1
    handler = _Handler(clock)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=1,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await asyncio.wait_for(source.generation_fetch_started.wait(), timeout=2)

        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v2"),
            cursor=2,
        )
        original_job = await _wait_for_job(
            path,
            1,
            lambda row: row["expected_version"] == "revision:v2"
            and row["status"] == "queued"
            and row["lease_id"] is None,
        )

        source.release_generation_fetch.set()
        await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)
        await _wait_for_job(path, 1, lambda row: row["status"] == "completed")

        rows = await _job_rows(path, 1)
        assert len(rows) == 1
        assert rows[0]["job_id"] == original_job["job_id"]
        assert handler.handled == [(1, "v2")]
    finally:
        source.release_generation_fetch.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_newer_accepted_revision_prevents_claimed_old_job_from_starting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    store = await _open_store(path)
    try:
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=1,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        old_work = await store.claim_pending_ai_work(SOURCE_ID, now=100)
        assert old_work is not None
        assert (
            await store.promote_pending_ai_generation(
                old_work,
                version="revision:v1",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=100,
                now=100,
            )
            == "queued"
        )
        old_job = await store.claim_pending_ai_generation(SOURCE_ID, now=100)
        assert old_job is not None

        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=2,
            chat_id=CHAT_ID,
            message_id=2,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        tail_work = await store.claim_pending_ai_work(SOURCE_ID, now=100)
        assert tail_work is not None
        assert (
            await store.promote_pending_ai_generation(
                tail_work,
                version="revision:tail",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=100,
                now=100,
            )
            == "waiting"
        )

        # Pause intake after the newer event commits but before it can promote.
        # The old generation claim must observe that durable event and lose,
        # without releasing its FIFO slot to the already-queued tail.
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=3,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        assert await store.begin_ai_generation(old_job, now=101) == "stale"
        assert await store.claim_pending_ai_generation(SOURCE_ID, now=101) is None

        newer_work = await store.claim_pending_ai_work(SOURCE_ID, now=101)
        assert newer_work is not None
        assert newer_work.message_id == 1
        assert (
            await store.promote_pending_ai_generation(
                newer_work,
                version="revision:v2",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=101,
                now=101,
            )
            == "updated"
        )

        rows = await _job_rows(path, 1)
        assert len(rows) == 1
        assert rows[0]["job_id"] == old_job.job_id
        assert rows[0]["queue_sequence"] == old_job.queue_sequence
        assert rows[0]["status"] == "queued"
        assert rows[0]["expected_version"] == "revision:v2"
        assert rows[0]["started_at"] is None

        edited = await store.claim_pending_ai_generation(SOURCE_ID, now=101)
        assert edited is not None
        assert edited.message_id == 1
        assert edited.expected_version == "revision:v2"
        assert await store.begin_ai_generation(edited, now=102) == "started"
        assert await store.complete_ai_generation(
            edited,
            outcome="completed",
            now=102,
        )
        tail = await store.claim_pending_ai_generation(SOURCE_ID, now=102)
        assert tail is not None
        assert tail.message_id == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_duplicate_revision_releases_retained_fifo_slot(tmp_path: Path) -> None:
    store = await _open_store(tmp_path / "ai.db")
    try:
        old_job = await _claim_direct_head_with_tail(store)
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=3,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        assert await store.begin_ai_generation(old_job, now=101) == "stale"

        duplicate = await store.claim_pending_ai_work(SOURCE_ID, now=101)
        assert duplicate is not None
        assert duplicate.message_id == 1
        assert (
            await store.promote_pending_ai_generation(
                duplicate,
                version="revision:v1",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=101,
                now=101,
            )
            == "duplicate"
        )

        released = await store.claim_pending_ai_generation(SOURCE_ID, now=101)
        assert released is not None
        assert released.job_id == old_job.job_id
        assert released.queue_sequence == old_job.queue_sequence
        assert released.trigger_cursor == 3
        assert await store.begin_ai_generation(released, now=102) == "started"
        assert await store.complete_ai_generation(
            released,
            outcome="completed",
            now=102,
        )
        tail = await store.claim_pending_ai_generation(SOURCE_ID, now=102)
        assert tail is not None
        assert tail.message_id == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_duplicate_reversion_supersedes_retained_fifo_slot(
    tmp_path: Path,
) -> None:
    store = await _open_store(tmp_path / "ai.db")
    try:
        assert (
            await _promote_direct_generation(
                store,
                cursor=1,
                message_id=1,
                version="revision:v1",
            )
            == "queued"
        )
        first = await store.claim_pending_ai_generation(SOURCE_ID, now=100)
        assert first is not None
        assert await store.begin_ai_generation(first, now=100) == "started"
        assert await store.complete_ai_generation(
            first,
            outcome="completed",
            now=100,
        )

        assert (
            await _promote_direct_generation(
                store,
                cursor=2,
                message_id=1,
                version="revision:v2",
                now=101,
            )
            == "queued"
        )
        second = await store.claim_pending_ai_generation(SOURCE_ID, now=101)
        assert second is not None
        assert second.expected_version == "revision:v2"
        assert (
            await _promote_direct_generation(
                store,
                cursor=3,
                message_id=2,
                version="revision:tail",
                now=101,
            )
            == "waiting"
        )

        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=4,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        assert await store.begin_ai_generation(second, now=102) == "stale"
        reversion = await store.claim_pending_ai_work(SOURCE_ID, now=102)
        assert reversion is not None
        assert (
            await store.promote_pending_ai_generation(
                reversion,
                version="revision:v1",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=102,
                now=102,
            )
            == "duplicate"
        )

        rows = await _job_rows(store.path, 1)
        assert rows[-1]["status"] == "superseded"
        tail = await store.claim_pending_ai_generation(SOURCE_ID, now=102)
        assert tail is not None
        assert tail.message_id == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_stale_immediate_edit_cannot_release_older_fifo_slot(
    tmp_path: Path,
) -> None:
    store = await _open_store(tmp_path / "ai.db")
    try:
        old_job = await _claim_direct_head_with_tail(store)
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=3,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        stale_immediate = await store.claim_pending_ai_work(SOURCE_ID, now=101)
        assert stale_immediate is not None
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=4,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )

        assert (
            await store.begin_pending_ai_execution(
                stale_immediate,
                version="revision:v2",
                supersede_queued_generation=True,
                now=102,
            )
            == "stale"
        )
        rows = await _job_rows(store.path, 1)
        assert rows[0]["job_id"] == old_job.job_id
        assert rows[0]["status"] == "queued"
        assert rows[0]["lease_id"] == old_job.lease_id
        assert await store.claim_pending_ai_generation(SOURCE_ID, now=102) is None

        current = await store.claim_pending_ai_work(SOURCE_ID, now=102)
        assert current is not None
        assert (
            await store.promote_pending_ai_generation(
                current,
                version="revision:v3",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=102,
                now=102,
            )
            == "updated"
        )
        updated = await store.claim_pending_ai_generation(SOURCE_ID, now=102)
        assert updated is not None
        assert updated.job_id == old_job.job_id
        assert updated.expected_version == "revision:v3"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_changed_source_revision_waits_for_pending_edit_fifo_reuse(
    tmp_path: Path,
) -> None:
    store = await _open_store(tmp_path / "ai.db")
    try:
        old_job = await _claim_direct_head_with_tail(store)
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=3,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )

        # Generation refetch has already observed v2, but intake still owns
        # the accepted edit. The old job must retain its FIFO position.
        assert not await store.complete_ai_generation(
            old_job,
            outcome="superseded",
            error_code="SOURCE_REVISION_CHANGED",
            require_source_current=True,
            now=101,
        )
        assert await store.claim_pending_ai_generation(SOURCE_ID, now=101) is None

        edit = await store.claim_pending_ai_work(SOURCE_ID, now=101)
        assert edit is not None
        assert (
            await store.promote_pending_ai_generation(
                edit,
                version="revision:v2",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=101,
                now=101,
            )
            == "updated"
        )
        updated = await store.claim_pending_ai_generation(SOURCE_ID, now=101)
        assert updated is not None
        assert updated.job_id == old_job.job_id
        assert updated.queue_sequence == old_job.queue_sequence
        assert updated.expected_version == "revision:v2"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_recovery_releases_fifo_after_newer_unknown_intake(
    tmp_path: Path,
) -> None:
    store = await _open_store(tmp_path / "ai.db")
    try:
        old_job = await _claim_direct_head_with_tail(store)
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=3,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        assert await store.begin_ai_generation(old_job, now=101) == "stale"
        edit = await store.claim_pending_ai_work(SOURCE_ID, now=101)
        assert edit is not None
        assert (
            await store.begin_pending_ai_execution(
                edit,
                version="revision:v2",
                now=101,
            )
            == "started"
        )
        assert await store.mark_pending_ai_execution_unknown(
            edit,
            version="revision:v2",
            now=102,
        )

        await store.recover_pending_ai_work(SOURCE_ID, now=103)
        rows = await _job_rows(store.path, 1)
        assert rows[0]["status"] == "superseded"
        tail = await store.claim_pending_ai_generation(SOURCE_ID, now=103)
        assert tail is not None
        assert tail.message_id == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_editing_queued_generation_to_immediate_supersedes_job(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock(100)
    source = _Source()
    handler = _Handler(clock)
    handler.cooldown_until["principal:alice"] = 200
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=1,
        clock=clock,
    )
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(1, CHAT_ID, "principal:alice", "v1"),
            cursor=1,
        )
        await _wait_for_job(
            path,
            1,
            lambda row: row["status"] == "queued"
            and row["expected_version"] == "revision:v1"
            and row["eligible_at"] == 200,
        )

        await _accept(
            workflow,
            source,
            _Message(
                1,
                CHAT_ID,
                "principal:alice",
                "v2",
                kind="immediate",
            ),
            cursor=2,
        )
        await asyncio.wait_for(handler.immediate_event(1).wait(), timeout=2)

        superseded = await _wait_for_job(
            path,
            1,
            lambda row: row["status"] == "superseded",
        )
        assert superseded["last_error_code"] == "SOURCE_REVISION_CHANGED"
        assert superseded["expected_version"] == "revision:v1"
        assert handler.immediate_handled == [(1, "v2")]
        assert not handler.started_event(1).is_set()
    finally:
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_stale_claimed_removal_cannot_cancel_newer_generation_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    store = await _open_store(path)
    try:
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=1,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        first = await store.claim_pending_ai_work(SOURCE_ID, now=100)
        assert first is not None
        assert (
            await store.promote_pending_ai_generation(
                first,
                version="revision:v1",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=100,
                now=100,
            )
            == "queued"
        )

        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=2,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message_remove",
            attested_origin=MessageOrigin.INCOMING,
        )
        stale_removal = await store.claim_pending_ai_work(SOURCE_ID, now=101)
        assert stale_removal is not None
        assert stale_removal.kind == "message_remove"

        # A later source observation invalidates the removal lease and updates
        # the same queued job to the new authoritative revision.
        await store.accept_pending_ai_event(
            SOURCE_ID,
            cursor=3,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message",
            attested_origin=MessageOrigin.INCOMING,
        )
        newer = await store.claim_pending_ai_work(SOURCE_ID, now=102)
        assert newer is not None
        assert (
            await store.promote_pending_ai_generation(
                newer,
                version="revision:v2",
                principal_actor_id="principal:alice",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=False,
                eligible_at=102,
                now=102,
            )
            == "updated"
        )

        assert not await store.resolve_pending_ai_removal(stale_removal)
        rows = await _job_rows(path, 1)
        assert len(rows) == 1
        assert rows[0]["status"] == "queued"
        assert rows[0]["expected_version"] == "revision:v2"
        assert rows[0]["trigger_cursor"] == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_workflow_stale_removal_does_not_cancel_newer_active_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    clock = _Clock()
    source = _Source()
    handler = _Handler(clock)
    first_release = handler.gate_version(1, "v1")
    second_release = handler.gate_version(1, "v2")
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=1,
        clock=clock,
    )
    original_resolve = store.resolve_pending_ai_removal
    resolve_started = asyncio.Event()
    release_resolve = asyncio.Event()
    resolve_finished = asyncio.Event()
    resolve_results: list[bool] = []

    async def delayed_resolve(work: Any) -> bool:
        resolve_started.set()
        await release_resolve.wait()
        resolved = await original_resolve(work)
        resolve_results.append(resolved)
        resolve_finished.set()
        return resolved

    store.resolve_pending_ai_removal = cast(Any, delayed_resolve)
    workflow.start()
    try:
        await _accept(
            workflow,
            source,
            _Message(
                1,
                CHAT_ID,
                "principal:owner",
                "v1",
                is_owner=True,
            ),
            cursor=1,
        )
        await asyncio.wait_for(
            handler.started_version_event(1, "v1").wait(),
            timeout=2,
        )

        await workflow.accept(
            cursor=2,
            chat_id=CHAT_ID,
            message_id=1,
            kind="message_remove",
            attested_origin=MessageOrigin.INCOMING,
        )
        await asyncio.wait_for(resolve_started.wait(), timeout=2)

        # The newer event invalidates the claimed removal. Promote it directly
        # while the single ordered intake lane is paused in the stale CAS.
        await _accept(
            workflow,
            source,
            _Message(
                1,
                CHAT_ID,
                "principal:owner",
                "v2",
                is_owner=True,
            ),
            cursor=3,
        )
        newer = await store.claim_pending_ai_work(SOURCE_ID, now=clock())
        assert newer is not None
        assert (
            await store.promote_pending_ai_generation(
                newer,
                version="revision:v2",
                principal_actor_id="principal:owner",
                scope_id=f"scope:{CHAT_ID}",
                is_owner=True,
                eligible_at=clock(),
                now=clock(),
            )
            == "queued"
        )

        first_release.set()
        await asyncio.wait_for(
            handler.started_version_event(1, "v2").wait(),
            timeout=2,
        )
        await _wait_for_job_rows(
            path,
            1,
            lambda rows: len(rows) == 2
            and rows[0]["status"] == "completed"
            and rows[1]["status"] == "running",
        )

        release_resolve.set()
        await asyncio.wait_for(resolve_finished.wait(), timeout=2)
        assert resolve_results == [False]
        await asyncio.sleep(0.05)
        still_active = await _job_rows(path, 1)
        assert still_active[1]["status"] == "running"

        second_release.set()
        await _wait_for_job_rows(
            path,
            1,
            lambda rows: len(rows) == 2 and rows[1]["status"] == "completed",
        )
        assert handler.handled == [(1, "v1"), (1, "v2")]
    finally:
        first_release.set()
        second_release.set()
        release_resolve.set()
        await workflow.close()
        await store.close()


@pytest.mark.asyncio
async def test_restart_keeps_unknown_terminal_and_releases_later_fifo_work(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.db"
    # Keep durable timestamps within retention when the store is reopened.
    clock = _Clock(2_000_000_000)
    source = _Source()
    handler = _Handler(clock)
    handler.gate(1)
    store = await _open_store(path)
    workflow = _workflow(
        source,
        store,
        handler,
        generation_concurrency=2,
        clock=clock,
    )
    workflow.start()

    await _accept(
        workflow,
        source,
        _Message(1, CHAT_ID, "principal:alice", "v1"),
        cursor=1,
    )
    await _accept(
        workflow,
        source,
        _Message(2, CHAT_ID, "principal:alice", "v1"),
        cursor=2,
    )
    await asyncio.wait_for(handler.started_event(1).wait(), timeout=2)
    await _wait_for_job(path, 2, lambda row: row["status"] == "queued")

    await workflow.close()
    unknown = await _wait_for_job(
        path,
        1,
        lambda row: row["status"] == "failed_unknown",
    )
    assert unknown["last_error_code"] == "ADAPTER_RESTARTED"
    await store.close()

    restarted = await _open_store(path)
    try:
        await restarted.recover_pending_ai_work(SOURCE_ID, now=2_000_000_100)
        next_job = await restarted.claim_pending_ai_generation(
            SOURCE_ID,
            now=2_000_000_101,
        )
        assert next_job is not None
        assert next_job.message_id == 2
        quarantined = await _job_rows(path, 1)
        assert quarantined[0]["status"] == "failed_unknown"
        assert quarantined[0]["lease_id"] is None
        queued = await _job_rows(path, 2)
        assert queued[0]["status"] == "queued"
        assert queued[0]["last_error_code"] is None
        assert queued[0]["lease_id"] == next_job.lease_id
    finally:
        await restarted.close()


class _LostWakeupStore:
    def __init__(self) -> None:
        self.work_count = 0
        self.empty_scan_started = asyncio.Event()
        self.return_empty_scan = asyncio.Event()
        self._captured_empty_scan = False

    async def next_pending_ai_generation_at(self, source_id: str) -> float | None:
        del source_id
        task = asyncio.current_task()
        if (
            task is not None
            and task.get_name() == "sleeping-generation-lane"
            and not self._captured_empty_scan
        ):
            self._captured_empty_scan = True
            snapshot = 0.0 if self.work_count else None
            self.empty_scan_started.set()
            await self.return_empty_scan.wait()
            return snapshot
        return 0.0 if self.work_count else None


@pytest.mark.asyncio
async def test_queue_transition_fields_are_visible_in_plain_production_log(
    tmp_path: Path,
) -> None:
    store = await _open_store(tmp_path / "ai.db")
    stream = io.StringIO()
    logger = logging.Logger("ai-workflow-observability", level=logging.INFO)
    log_handler = logging.StreamHandler(stream)
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(log_handler)
    workflow = AIWorkflow(
        cast(Any, object()),
        store,
        SOURCE_ID,
        cast(Any, object()),
        generation_concurrency=1,
        clock=_Clock(100),
        logger=logger,
    )
    try:
        await workflow._log_queue_event(
            "ai_workflow_test_transition",
            message_id=42,
            principal_actor_id="principal:test",
            error_code="TEST_ERROR",
            cancelled_queued=1,
            cancelled_running=2,
        )
    finally:
        await store.close()

    output = stream.getvalue()
    assert "event=ai_workflow_test_transition" in output
    assert "source=workflow-test" in output
    assert "message=42" in output
    assert "queued=0" in output
    assert "active=0" in output
    assert "failed_unknown=0" in output
    assert "cancelled_queued=1" in output
    assert "cancelled_running=2" in output


@pytest.mark.asyncio
async def test_notification_is_not_lost_between_empty_scan_and_wait() -> None:
    store = _LostWakeupStore()
    clock = _Clock(0)
    workflow = AIWorkflow(
        cast(Any, object()),
        cast(Any, store),
        SOURCE_ID,
        cast(Any, object()),
        generation_concurrency=2,
        clock=clock,
    )
    sleeping_lane_processed = asyncio.Event()
    running_lane_started = asyncio.Event()
    release_running_lane = asyncio.Event()

    async def process_one() -> Literal["idle", "completed"]:
        if store.work_count == 0:
            return "idle"
        store.work_count -= 1
        task = asyncio.current_task()
        assert task is not None
        if task.get_name() == "running-generation-lane":
            running_lane_started.set()
            await release_running_lane.wait()
        else:
            sleeping_lane_processed.set()
        return "completed"

    sleeping_lane = asyncio.create_task(
        workflow._run_lane(
            process_one,
            available=workflow._generation_available[0],
            intake=False,
        ),
        name="sleeping-generation-lane",
    )
    running_lane: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(store.empty_scan_started.wait(), timeout=1)
        store.work_count = 2
        workflow.notify()

        running_lane = asyncio.create_task(
            workflow._run_lane(
                process_one,
                available=workflow._generation_available[1],
                intake=False,
            ),
            name="running-generation-lane",
        )
        await asyncio.wait_for(running_lane_started.wait(), timeout=1)

        store.return_empty_scan.set()
        await asyncio.wait_for(sleeping_lane_processed.wait(), timeout=0.2)
    finally:
        release_running_lane.set()
        sleeping_lane.cancel()
        if running_lane is not None:
            running_lane.cancel()
            await asyncio.gather(
                sleeping_lane,
                running_lane,
                return_exceptions=True,
            )
        else:
            await asyncio.gather(sleeping_lane, return_exceptions=True)
