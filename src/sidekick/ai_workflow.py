from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
import time
from typing import Any, Generic, Literal, TypeVar

from sidekick.ai import (
    AIConversationHandler,
    AIMessageClassification,
    AIWorkflowCancellation,
)
from sidekick.chat.identity import ExternalId
from sidekick.chat.provenance import MessageOrigin
from sidekick.inbound import (
    InboundMessageSource,
    InboundSourceUnavailable,
    InboundSourceRevision,
)
from sidekick.inbound_store import (
    SQLiteInboundWorkStore,
    StoredGenerationJob,
    StoredInboundWork,
)


_SourcePayload = TypeVar("_SourcePayload")
_LaneResult = Literal[
    "idle",
    "completed",
    "ignored",
    "duplicate",
    "recalled",
    "cancelled",
    "queued",
    "deferred",
    "unavailable",
    "stale",
    "failed",
    "failed_unknown",
]


class AIWorkflow(Generic[_SourcePayload]):
    """Own durable intake and generation scheduling for one channel source."""

    RETRY_BASE_SECONDS = 2.0
    RETRY_MAX_SECONDS = 300.0

    def __init__(
        self,
        source: InboundMessageSource[_SourcePayload],
        store: SQLiteInboundWorkStore,
        source_id: str,
        handler: AIConversationHandler,
        *,
        generation_concurrency: int,
        clock: Callable[[], float] = time.time,
        logger: Any | None = None,
    ) -> None:
        if not source_id:
            raise ValueError("AI workflow source ID cannot be empty")
        if generation_concurrency < 1:
            raise ValueError("AI generation concurrency must be positive")
        self._source = source
        self._store = store
        self._source_id = source_id
        self._handler = handler
        self._generation_concurrency = generation_concurrency
        # One ordered intake lane prevents a due later control event from
        # overtaking promotion of earlier due intake. Source-deferred work is
        # skipped so one unavailable native message cannot poison the lane.
        # Before classification it has no proven Principal and is therefore
        # outside Principal-targeted generation cancellation.
        self._clock = clock
        self._logger = logger
        self._intake_available = asyncio.Event()
        self._generation_available = tuple(
            asyncio.Event() for _ in range(self._generation_concurrency)
        )
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._active_message_tasks: dict[
            tuple[ExternalId, ExternalId], set[asyncio.Task[Any]]
        ] = {}
        self._active_intake_tasks: dict[
            tuple[ExternalId, ExternalId], set[asyncio.Task[Any]]
        ] = {}
        self._active_principal_tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self._recall_cancellations: set[asyncio.Task[Any]] = set()
        self._user_cancellations: set[asyncio.Task[Any]] = set()

    def start(self) -> None:
        if self._tasks:
            raise RuntimeError("AI workflow is already running")
        self._handler.bind_workflow_control(self)
        intake_task = asyncio.create_task(
            self._run_lane(
                self._process_intake_one,
                available=self._intake_available,
                intake=True,
            ),
            name=f"ai-intake-{self._source_id}",
        )
        generation_tasks = tuple(
            asyncio.create_task(
                self._run_lane(
                    self._process_generation_one,
                    available=self._generation_available[index],
                    intake=False,
                ),
                name=f"ai-generation-{self._source_id}-{index}",
            )
            for index in range(self._generation_concurrency)
        )
        self._tasks = (intake_task, *generation_tasks)
        self.notify()

    async def close(self) -> None:
        tasks, self._tasks = self._tasks, ()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._handler.unbind_workflow_control(self)

    async def accept(
        self,
        *,
        cursor: ExternalId,
        chat_id: ExternalId,
        message_id: ExternalId,
        kind: Literal["message", "message_remove"],
        attested_origin: MessageOrigin | None,
    ) -> None:
        await self._store.accept_pending_ai_event(
            self._source_id,
            cursor=cursor,
            chat_id=chat_id,
            message_id=message_id,
            kind=kind,
            attested_origin=attested_origin,
        )
        if kind == "message_remove":
            # A long-running immediate handler occupies the ordered intake
            # lane, so it cannot wait for that same lane to resolve recall.
            # Generation runs use a separate lane and are interrupted only
            # after the exact removal wins the store CAS below.
            self._cancel_tasks(
                self._active_intake_tasks.get((chat_id, message_id), ()),
                reason="SOURCE_RECALLED",
                tracked=self._recall_cancellations,
            )
        self.notify()

    def notify(self) -> None:
        self._intake_available.set()
        for available in self._generation_available:
            available.set()

    def cancel_message(
        self,
        chat_id: ExternalId,
        message_id: ExternalId,
    ) -> None:
        self._cancel_tasks(
            self._active_message_tasks.get((chat_id, message_id), ()),
            reason="SOURCE_RECALLED",
            tracked=self._recall_cancellations,
        )

    async def cancel_generations(
        self,
        principal_actor_id: str,
        *,
        interrupt_running: bool,
    ) -> AIWorkflowCancellation:
        queued, running = await self._store.request_ai_generation_cancellation(
            self._source_id,
            principal_actor_id,
            now=self._clock(),
        )
        if interrupt_running and (queued or running):
            for task in tuple(self._active_principal_tasks.get(principal_actor_id, ())):
                if task.done():
                    continue
                self._user_cancellations.add(task)
                task.cancel("USER_CANCELLED_OUTCOME_UNKNOWN")
        if queued or running:
            self._notify_generation()
        await self._log_queue_event(
            "ai_workflow_cancel_requested",
            principal_actor_id=principal_actor_id,
            cancelled_queued=queued,
            cancelled_running=running,
        )
        return AIWorkflowCancellation(queued=queued, running=running)

    async def reschedule_scope(self, scope_id: str) -> int:
        updated = await self._store.reschedule_ai_generation_scope(
            self._source_id,
            scope_id,
            now=self._clock(),
        )
        if updated:
            self._notify_generation()
            await self._log_queue_event(
                "ai_workflow_scope_rescheduled",
            )
        return updated

    async def _run_lane(
        self,
        process_one: Callable[[], Awaitable[_LaneResult]],
        *,
        available: asyncio.Event,
        intake: bool,
    ) -> None:
        while True:
            available.clear()
            lane = asyncio.current_task()
            item_task = asyncio.create_task(
                process_one(),
                name=(lane.get_name() if lane is not None else "ai-workflow-item"),
            )
            try:
                # Keep per-message cancellation separate from lane lifetime.
                # Recall/user cancellation targets the child; adapter shutdown
                # targets the lane and then explicitly tears down the child.
                result = await asyncio.shield(item_task)
            except asyncio.CancelledError:
                item_task.cancel()
                await asyncio.gather(item_task, return_exceptions=True)
                raise
            except Exception as exc:
                self._log_failure("intake" if intake else "generation", exc)
                result = "idle"
            if result != "idle":
                continue
            try:
                next_at = (
                    await self._store.next_pending_ai_work_at(self._source_id)
                    if intake
                    else await self._store.next_pending_ai_generation_at(
                        self._source_id
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_failure(
                    "intake schedule" if intake else "generation schedule",
                    exc,
                )
                try:
                    await asyncio.wait_for(
                        available.wait(),
                        timeout=self.RETRY_BASE_SECONDS,
                    )
                except TimeoutError:
                    pass
                continue
            timeout = max(0.0, next_at - self._clock()) if next_at is not None else None
            if timeout == 0:
                continue
            try:
                await asyncio.wait_for(available.wait(), timeout=timeout)
            except TimeoutError:
                pass

    async def _process_intake_one(self) -> _LaneResult:
        work = await self._store.claim_pending_ai_work(
            self._source_id,
            now=self._clock(),
        )
        if work is None:
            return "idle"
        if work.kind == "message_remove":
            resolved = await self._store.resolve_pending_ai_removal(work)
            # Only an exact, current removal may interrupt execution. A stale
            # removal for an older source cursor must not cancel a newer edit
            # of the same stable message ID.
            if resolved:
                self.cancel_message(work.chat_id, work.message_id)
            self._notify_generation()
            return "recalled" if resolved else "stale"

        task = self._register_active(
            work.chat_id,
            work.message_id,
            intake=True,
        )
        execution_version: str | None = None
        revision: InboundSourceRevision[_SourcePayload] | None = None
        try:
            try:
                revision = await self._source.fetch(work)
            except InboundSourceUnavailable as exc:
                return await self._defer_inbound(work, exc)
            if revision.state == "recalled":
                return await self._finish_inbound(
                    work,
                    revision,
                    outcome="recalled",
                )

            assert revision.payload is not None
            try:
                message = await self._source.materialize(revision.payload)
            except Exception as exc:
                self._log_message_failure("materialization", exc, work.message_id)
                return await self._finish_inbound(
                    work,
                    revision,
                    outcome="failed",
                )
            if message is None:
                return await self._finish_inbound(
                    work,
                    revision,
                    outcome="ignored",
                )

            try:
                classification = await self._handler.classify(
                    message,
                    attested_origin=revision.attested_origin,
                )
            except Exception as exc:
                self._log_message_failure("classification", exc, work.message_id)
                return await self._finish_inbound(
                    work,
                    revision,
                    outcome="failed",
                )

            if classification.disposition == "generation":
                eligible_at = await self._handler.generation_eligible_at(classification)
                assert classification.principal_actor_id is not None
                assert classification.scope_id is not None
                promotion = await self._store.promote_pending_ai_generation(
                    work,
                    version=revision.version,
                    principal_actor_id=classification.principal_actor_id,
                    scope_id=classification.scope_id,
                    is_owner=classification.is_owner,
                    eligible_at=eligible_at,
                    now=self._clock(),
                )
                await self._log_queue_event(
                    f"ai_workflow_generation_{promotion}",
                    message_id=work.message_id,
                    principal_actor_id=classification.principal_actor_id,
                    warning=promotion == "principal_queue_full",
                )
                if promotion == "stale":
                    return "stale"
                if promotion == "duplicate":
                    self._notify_generation()
                    return "duplicate"
                if promotion in {"waiting", "principal_queue_full"}:
                    notice: Literal["queued", "queue_full"] = {
                        "waiting": "queued",
                        "principal_queue_full": "queue_full",
                    }[promotion]
                    try:
                        await self._handler.reply_workflow_notice(message, notice)
                    except Exception as exc:
                        self._log_message_failure(
                            "workflow notice",
                            exc,
                            work.message_id,
                        )
                self._notify_generation()
                if promotion == "principal_queue_full":
                    return "failed"
                return "queued"

            begin = await self._store.begin_pending_ai_execution(
                work,
                version=revision.version,
                supersede_queued_generation=True,
                now=self._clock(),
            )
            if begin == "stale":
                return "stale"
            self._notify_generation()
            if begin == "duplicate":
                await self._store.complete_pending_ai_work(
                    work,
                    version=revision.version,
                    outcome="ignored",
                    now=self._clock(),
                )
                return "duplicate"
            execution_version = revision.version
            try:
                handled = await self._handler.handle(
                    message,
                    attested_origin=revision.attested_origin,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_message_failure("control", exc, work.message_id)
                await self._store.complete_pending_ai_work(
                    work,
                    version=revision.version,
                    outcome="failed",
                    now=self._clock(),
                )
                return "failed"
            outcome: Literal["completed", "ignored"] = (
                "completed" if handled else "ignored"
            )
            await self._store.complete_pending_ai_work(
                work,
                version=revision.version,
                outcome=outcome,
                now=self._clock(),
            )
            return outcome
        except asyncio.CancelledError:
            cleanup = (
                self._store.release_pending_ai_work(work)
                if execution_version is None
                else self._store.mark_pending_ai_execution_unknown(
                    work,
                    version=execution_version,
                    now=self._clock(),
                )
            )
            await asyncio.shield(cleanup)
            if task in self._recall_cancellations:
                return "stale"
            raise
        except Exception as exc:
            self._log_message_failure("intake", exc, work.message_id)
            if execution_version is not None:
                await asyncio.shield(
                    self._store.mark_pending_ai_execution_unknown(
                        work,
                        version=execution_version,
                        now=self._clock(),
                    )
                )
                return "failed_unknown"
            if revision is not None:
                return await asyncio.shield(
                    self._finish_inbound(work, revision, outcome="failed")
                )
            now = self._clock()
            status = await asyncio.shield(
                self._store.defer_pending_ai_work(
                    work,
                    error_code="SOURCE_FETCH_FAILED",
                    retry_at=now,
                    max_attempts=1,
                    supersede_queued_generation_on_unavailable=True,
                    now=now,
                )
            )
            if status == "unavailable":
                self._notify_generation()
            return "unavailable" if status == "unavailable" else "stale"
        finally:
            self._unregister_active(
                work.chat_id,
                work.message_id,
                task,
                intake=True,
            )

    async def _process_generation_one(self) -> _LaneResult:
        job = await self._store.claim_pending_ai_generation(
            self._source_id,
            now=self._clock(),
        )
        if job is None:
            return "idle"

        task = self._register_active(job.chat_id, job.message_id)
        self._active_principal_tasks.setdefault(
            job.principal_actor_id,
            set(),
        ).add(task)
        execution_started = False
        try:
            try:
                revision = await self._source.fetch(job)
            except InboundSourceUnavailable as exc:
                return await self._defer_generation_source(job, exc)
            if revision.state == "recalled":
                completed = await self._store.complete_ai_generation(
                    job,
                    outcome="cancelled",
                    error_code="SOURCE_RECALLED",
                    require_source_current=True,
                    now=self._clock(),
                )
                return "recalled" if completed else "stale"
            if revision.version != job.expected_version:
                await self._store.complete_ai_generation(
                    job,
                    outcome="superseded",
                    error_code="SOURCE_REVISION_CHANGED",
                    require_source_current=True,
                    now=self._clock(),
                )
                return "stale"

            assert revision.payload is not None
            try:
                message = await self._source.materialize(revision.payload)
            except Exception as exc:
                self._log_message_failure("generation context", exc, job.message_id)
                completed = await self._store.complete_ai_generation(
                    job,
                    outcome="failed",
                    error_code="MESSAGE_MATERIALIZATION_FAILED",
                    require_source_current=True,
                    now=self._clock(),
                )
                return "failed" if completed else "stale"
            if message is None:
                completed = await self._store.complete_ai_generation(
                    job,
                    outcome="ignored",
                    require_source_current=True,
                    now=self._clock(),
                )
                return "ignored" if completed else "stale"

            classification = await self._handler.classify(
                message,
                attested_origin=revision.attested_origin,
            )
            if not self._same_generation_identity(job, classification):
                await self._store.complete_ai_generation(
                    job,
                    outcome="superseded",
                    error_code="GENERATION_IDENTITY_CHANGED",
                    require_source_current=True,
                    now=self._clock(),
                )
                return "stale"
            eligible_at = await self._handler.generation_eligible_at(classification)
            now = self._clock()
            if eligible_at > now:
                deferred = await self._store.defer_ai_generation(
                    job,
                    error_code="COOLDOWN",
                    eligible_at=eligible_at,
                    require_source_current=True,
                    now=now,
                )
                if not deferred:
                    return "stale"
                await self._log_queue_event(
                    "ai_workflow_generation_deferred",
                    message_id=job.message_id,
                    principal_actor_id=job.principal_actor_id,
                    error_code="COOLDOWN",
                )
                return "deferred"

            begin = await self._store.begin_ai_generation(job, now=now)
            if begin == "stale":
                # A newer accepted revision may still be waiting in intake.
                # Leave this queued lease in place so its FIFO slot cannot be
                # overtaken; intake will update or terminalize that same slot.
                return "stale"
            execution_started = True
            try:
                handled = await self._handler.handle(
                    message,
                    attested_origin=revision.attested_origin,
                    workflow_admitted=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_message_failure("generation", exc, job.message_id)
                await self._store.mark_ai_generation_unknown(
                    job,
                    error_code="HANDLER_OUTCOME_UNKNOWN",
                    now=self._clock(),
                )
                await self._log_queue_event(
                    "ai_workflow_generation_failed_unknown",
                    message_id=job.message_id,
                    principal_actor_id=job.principal_actor_id,
                    error_code="HANDLER_OUTCOME_UNKNOWN",
                    warning=True,
                )
                return "failed_unknown"
            await self._store.complete_ai_generation(
                job,
                outcome="completed" if handled else "ignored",
                now=self._clock(),
            )
            return "completed" if handled else "ignored"
        except asyncio.CancelledError:
            cleanup = (
                self._store.mark_ai_generation_unknown(
                    job,
                    error_code=(
                        "SOURCE_RECALLED"
                        if task in self._recall_cancellations
                        else (
                            "USER_CANCELLED_OUTCOME_UNKNOWN"
                            if task in self._user_cancellations
                            else "ADAPTER_RESTARTED"
                        )
                    ),
                    now=self._clock(),
                )
                if execution_started
                else self._store.release_ai_generation(job)
            )
            await asyncio.shield(cleanup)
            if execution_started:
                await asyncio.shield(
                    self._log_queue_event(
                        "ai_workflow_generation_failed_unknown",
                        message_id=job.message_id,
                        principal_actor_id=job.principal_actor_id,
                        error_code=(
                            "SOURCE_RECALLED"
                            if task in self._recall_cancellations
                            else (
                                "USER_CANCELLED_OUTCOME_UNKNOWN"
                                if task in self._user_cancellations
                                else "ADAPTER_RESTARTED"
                            )
                        ),
                        warning=True,
                    )
                )
            if task in self._recall_cancellations:
                return "stale"
            if task in self._user_cancellations:
                return "failed_unknown" if execution_started else "cancelled"
            raise
        except Exception as exc:
            self._log_message_failure("generation preparation", exc, job.message_id)
            if execution_started:
                await asyncio.shield(
                    self._store.mark_ai_generation_unknown(
                        job,
                        error_code="WORKFLOW_OUTCOME_UNKNOWN",
                        now=self._clock(),
                    )
                )
                await asyncio.shield(
                    self._log_queue_event(
                        "ai_workflow_generation_failed_unknown",
                        message_id=job.message_id,
                        principal_actor_id=job.principal_actor_id,
                        error_code="WORKFLOW_OUTCOME_UNKNOWN",
                        warning=True,
                    )
                )
                return "failed_unknown"
            completed = await asyncio.shield(
                self._store.complete_ai_generation(
                    job,
                    outcome="failed",
                    error_code="GENERATION_PREPARATION_FAILED",
                    require_source_current=True,
                    now=self._clock(),
                )
            )
            return "failed" if completed else "stale"
        finally:
            principal_tasks = self._active_principal_tasks.get(job.principal_actor_id)
            if principal_tasks is not None:
                principal_tasks.discard(task)
                if not principal_tasks:
                    self._active_principal_tasks.pop(
                        job.principal_actor_id,
                        None,
                    )
            self._user_cancellations.discard(task)
            self._unregister_active(job.chat_id, job.message_id, task)

    async def _finish_inbound(
        self,
        work: StoredInboundWork,
        revision: InboundSourceRevision[_SourcePayload],
        *,
        outcome: Literal["ignored", "recalled", "failed"],
    ) -> _LaneResult:
        begin = await self._store.begin_pending_ai_execution(
            work,
            version=revision.version,
            supersede_queued_generation=True,
            now=self._clock(),
        )
        if begin == "stale":
            return "stale"
        self._notify_generation()
        if begin == "duplicate":
            await self._store.complete_pending_ai_work(
                work,
                version=revision.version,
                outcome="ignored",
                now=self._clock(),
            )
            return "duplicate"
        await self._store.complete_pending_ai_work(
            work,
            version=revision.version,
            outcome=outcome,
            now=self._clock(),
        )
        return outcome

    async def _defer_inbound(
        self,
        work: StoredInboundWork,
        error: InboundSourceUnavailable,
    ) -> _LaneResult:
        now = self._clock()
        prior_attempts = work.attempt_count if work.last_error_code == error.code else 0
        status = await self._store.defer_pending_ai_work(
            work,
            error_code=error.code,
            retry_at=now
            + max(
                self._retry_delay(prior_attempts),
                error.retry_after_seconds or 0,
            ),
            max_attempts=error.max_attempts,
            supersede_queued_generation_on_unavailable=True,
            now=now,
        )
        if status == "stale":
            return "stale"
        if status == "unavailable":
            self._notify_generation()
        return "unavailable" if status == "unavailable" else "deferred"

    async def _defer_generation_source(
        self,
        job: StoredGenerationJob,
        error: InboundSourceUnavailable,
    ) -> _LaneResult:
        prior_attempts = job.attempt_count if job.last_error_code == error.code else 0
        attempts = prior_attempts + 1
        if error.max_attempts is not None and attempts >= error.max_attempts:
            completed = await self._store.complete_ai_generation(
                job,
                outcome="source_unavailable",
                error_code=error.code,
                require_source_current=True,
                now=self._clock(),
            )
            if not completed:
                return "stale"
            await self._log_queue_event(
                "ai_workflow_generation_source_unavailable",
                message_id=job.message_id,
                principal_actor_id=job.principal_actor_id,
                error_code=error.code,
                warning=True,
            )
            return "unavailable"
        now = self._clock()
        deferred = await self._store.defer_ai_generation(
            job,
            error_code=error.code,
            eligible_at=now
            + max(
                self._retry_delay(prior_attempts),
                error.retry_after_seconds or 0,
            ),
            require_source_current=True,
            now=now,
        )
        if not deferred:
            return "stale"
        await self._log_queue_event(
            "ai_workflow_generation_source_deferred",
            message_id=job.message_id,
            principal_actor_id=job.principal_actor_id,
            error_code=error.code,
        )
        return "deferred"

    @staticmethod
    def _same_generation_identity(
        job: StoredGenerationJob,
        classification: AIMessageClassification,
    ) -> bool:
        return (
            classification.disposition == "generation"
            and classification.principal_actor_id == job.principal_actor_id
            and classification.scope_id == job.scope_id
            and classification.is_owner == job.is_owner
        )

    @classmethod
    def _retry_delay(cls, attempt_count: int) -> float:
        return min(
            cls.RETRY_MAX_SECONDS,
            cls.RETRY_BASE_SECONDS * (2 ** min(attempt_count, 16)),
        )

    def _register_active(
        self,
        chat_id: ExternalId,
        message_id: ExternalId,
        *,
        intake: bool = False,
    ) -> asyncio.Task[Any]:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("AI workflow work requires an asyncio task")
        key = (chat_id, message_id)
        self._active_message_tasks.setdefault(key, set()).add(task)
        if intake:
            self._active_intake_tasks.setdefault(key, set()).add(task)
        return task

    def _unregister_active(
        self,
        chat_id: ExternalId,
        message_id: ExternalId,
        task: asyncio.Task[Any],
        *,
        intake: bool = False,
    ) -> None:
        key = (chat_id, message_id)
        tasks = self._active_message_tasks.get(key)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                self._active_message_tasks.pop(key, None)
        if intake:
            intake_tasks = self._active_intake_tasks.get(key)
            if intake_tasks is not None:
                intake_tasks.discard(task)
                if not intake_tasks:
                    self._active_intake_tasks.pop(key, None)
        self._recall_cancellations.discard(task)

    @staticmethod
    def _cancel_tasks(
        tasks: Iterable[asyncio.Task[Any]],
        *,
        reason: str,
        tracked: set[asyncio.Task[Any]],
    ) -> None:
        for task in tuple(tasks):
            if task.done():
                continue
            tracked.add(task)
            task.cancel(reason)

    def _log_failure(self, lane: str, exc: Exception) -> None:
        if self._logger is not None:
            self._logger.error(
                "AI workflow %s lane failed (%s; source=%s)",
                lane,
                type(exc).__name__,
                self._source_id,
            )

    def _notify_generation(self) -> None:
        for available in self._generation_available:
            available.set()

    def _log_message_failure(
        self,
        stage: str,
        exc: Exception,
        message_id: ExternalId,
    ) -> None:
        if self._logger is not None:
            self._logger.error(
                "AI workflow %s failed (%s; source=%s; message=%s)",
                stage,
                type(exc).__name__,
                self._source_id,
                message_id,
            )

    async def _log_queue_event(
        self,
        event: str,
        *,
        message_id: ExternalId | None = None,
        principal_actor_id: str | None = None,
        error_code: str | None = None,
        cancelled_queued: int | None = None,
        cancelled_running: int | None = None,
        warning: bool = False,
    ) -> None:
        if self._logger is None:
            return
        try:
            snapshot = await self._store.get_ai_generation_queue_snapshot(
                self._source_id
            )
        except Exception as exc:
            self._log_failure("queue snapshot", exc)
            return
        now = self._clock()
        oldest_age = (
            max(0.0, now - snapshot.oldest_queued_at)
            if snapshot.oldest_queued_at is not None
            else None
        )
        log = self._logger.warning if warning else self._logger.info
        log(
            "AI workflow queue transition "
            "(event=%s; source=%s; message=%s; principal=%s; error=%s; "
            "queued=%s; active=%s; failed_unknown=%s; oldest_age=%s; "
            "cancelled_queued=%s; cancelled_running=%s)",
            event,
            self._source_id,
            message_id,
            principal_actor_id,
            error_code,
            snapshot.queued,
            snapshot.active,
            snapshot.failed_unknown,
            oldest_age,
            cancelled_queued,
            cancelled_running,
        )
