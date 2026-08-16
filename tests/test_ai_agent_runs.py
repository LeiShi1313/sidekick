from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from telethon.errors import MessageNotModifiedError

from sidekick.ai import (
    AIAnswerMarker,
    AIConversationHandler,
    AIResponder,
    AIWorkflowCancellation,
    AgentEvent,
    AgentIdentityAnchor,
    AgentRequestIdentity,
    AgentRunOrigin,
    AgentRunRequest,
    PromptBuilder,
)
from sidekick.chat.output_policy import MainlandMessagingOutputPolicy
from sidekick.telegram.ai_identity import TELEGRAM_IDENTITY_CODEC
from sidekick.telegram.ai_transport import TelegramChatTransport


def make_telegram_responder(gateway, **kwargs):
    return AIResponder(
        gateway,
        transport=TelegramChatTransport(edit_cadence=0),
        **kwargs,
    )


class FakeAnswer:
    next_id = 100

    def __init__(self, text: str, chat_id: int, reply_to_msg_id: int):
        self.id = self.__class__.next_id
        self.__class__.next_id += 1
        self.text = text
        self.chat_id = chat_id
        self.reply_to_msg_id = reply_to_msg_id
        self.edits: list[str] = []

    async def edit(self, text: str, **kwargs):
        self.text = text
        self.edits.append(text)
        return self


class FakeMessage:
    next_id = 1

    def __init__(
        self,
        text: str,
        *,
        sender_id: int = 10,
        chat_id: int = -1001,
        reply_to=None,
    ):
        self.id = self.__class__.next_id
        self.__class__.next_id += 1
        self.raw_text = text
        self.sender_id = sender_id
        self.chat_id = chat_id
        self.reply_to_msg_id = reply_to.id if reply_to else None
        self._reply_to = reply_to
        self.replies: list[FakeAnswer] = []

    async def get_reply_message(self):
        return self._reply_to

    async def reply(self, text: str, **kwargs):
        answer = FakeAnswer(text, self.chat_id, self.id)
        self.replies.append(answer)
        return answer


class FakeAgentGateway:
    def __init__(self, answers: list[str] | None = None):
        self.answers = iter(answers or [])
        self.requests: list[AgentRunRequest] = []
        self.cancelled: list[str] = []

    async def run(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        self.requests.append(request)
        answer = next(self.answers)
        session = request.session_id or f"session-{len(self.requests)}"
        yield AgentEvent(
            type="run_started",
            run_id=request.run_id,
            session_id=session,
        )
        yield AgentEvent(type="text_delta", delta=answer, reset=True)
        yield AgentEvent(
            type="run_completed",
            session_id=session,
            entry_id=f"entry-{len(self.requests)}",
            answer=answer,
        )

    async def cancel(self, run_id: str) -> bool:
        self.cancelled.append(run_id)
        return True


class BlockingAgentGateway(FakeAgentGateway):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled_event = asyncio.Event()

    async def run(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        self.requests.append(request)
        self.started.set()
        yield AgentEvent(
            type="run_started",
            run_id=request.run_id,
            session_id="session-blocking",
        )
        await self.cancelled_event.wait()
        yield AgentEvent(
            type="run_failed",
            code="CANCELLED",
            message="Agent run cancelled",
        )

    async def cancel(self, run_id: str) -> bool:
        self.cancelled.append(run_id)
        self.cancelled_event.set()
        return True


class FakeStore:
    def __init__(self, allowed: set[str] | None = None):
        self.allowed = allowed or set()
        self.markers: dict[tuple[str, int], AIAnswerMarker] = {}
        self.ai_command_prefixes: dict[str, str] = {}

    async def get_answer(self, scope_id, answer_message_id):
        return self.markers.get((scope_id, answer_message_id))

    async def get_turn_for_message(self, scope_id, message_id):
        return next(
            (
                marker
                for marker in reversed(tuple(self.markers.values()))
                if marker.scope_id == scope_id
                and message_id
                in {marker.answer_message_id, marker.trigger_message_id}
            ),
            None,
        )

    async def save_answer(self, marker):
        self.markers[(marker.scope_id, marker.answer_message_id)] = marker

    async def get_ai_trigger_command_prefixes(self, scope_id, message_ids):
        return {
            marker.trigger_message_id: marker.command_prefix
            for marker in self.markers.values()
            if marker.scope_id == scope_id and marker.trigger_message_id in message_ids
        }

    async def get_model_override(self, scope_id):
        return None

    async def set_model_override(self, scope_id, model):
        return None

    async def get_ai_command_prefix(self, scope_id):
        return self.ai_command_prefixes.get(scope_id)

    async def set_ai_command_prefix(self, scope_id, prefix):
        if prefix is None:
            self.ai_command_prefixes.pop(scope_id, None)
        else:
            self.ai_command_prefixes[scope_id] = prefix

    async def get_ai_cooldown_override(self, scope_id):
        return None

    async def set_ai_cooldown_override(self, scope_id, cooldown_seconds):
        return None

    async def is_allowed(self, actor_id):
        return actor_id in self.allowed

    async def get_last_request_at(self, scope_id, actor_id):
        return None

    async def set_last_request_at(self, scope_id, actor_id, timestamp):
        return None

    async def allow_user(self, actor_id):
        self.allowed.add(actor_id)

    async def deny_user(self, actor_id):
        self.allowed.discard(actor_id)


@pytest.fixture(autouse=True)
def reset_ids():
    FakeAnswer.next_id = 100
    FakeMessage.next_id = 1


@pytest.mark.asyncio
async def test_tool_snapshot_is_replaced_by_the_streamed_final_answer():
    class SnapshotGateway(FakeAgentGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            yield AgentEvent(
                type="tool_snapshot",
                phase="started",
                tool="web_search",
                summary="Searching web: current release",
            )
            yield AgentEvent(type="text_delta", delta="**Final", reset=True)
            yield AgentEvent(type="text_delta", delta=" answer**", reset=False)
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer="**Final answer**",
            )

    responder = make_telegram_responder(SnapshotGateway())
    trigger = FakeMessage("/ai search")
    request = AgentRunRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        session_id=None,
        parent_entry_id=None,
        prompt="search",
        context=(),
        system_prompt=PromptBuilder().system_prompt,
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
            anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
        ),
        origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
    )

    result = await responder.answer(trigger, request)

    answer = trigger.replies[0]
    assert any("Searching web" in edit for edit in answer.edits)
    assert answer.text == "Final answer"
    assert "Searching web" not in answer.text
    assert result.session_id == "session-1"
    assert result.entry_id == "entry-1"


@pytest.mark.asyncio
async def test_repeated_tool_snapshot_does_not_fail_the_agent_run():
    class TelegramLikeAnswer(FakeAnswer):
        async def edit(self, text: str, **kwargs):
            if text == self.text:
                raise MessageNotModifiedError(request=None)
            return await super().edit(text, **kwargs)

    class TelegramLikeMessage(FakeMessage):
        async def reply(self, text: str, **kwargs):
            answer = TelegramLikeAnswer(text, self.chat_id, self.id)
            self.replies.append(answer)
            return answer

    class ParallelSearchGateway(FakeAgentGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            for _ in range(2):
                yield AgentEvent(
                    type="tool_snapshot",
                    phase="completed",
                    tool="web_search",
                    summary="Web search completed",
                )
            yield AgentEvent(type="text_delta", delta="Final answer", reset=True)
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer="Final answer",
            )

    responder = make_telegram_responder(ParallelSearchGateway())
    trigger = TelegramLikeMessage("/ai search")
    request = AgentRunRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        session_id=None,
        parent_entry_id=None,
        prompt="search",
        context=(),
        system_prompt=PromptBuilder().system_prompt,
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
            anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
        ),
        origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
    )

    result = await responder.answer(trigger, request)

    assert result.succeeded is True
    assert trigger.replies[0].text == "Final answer"


@pytest.mark.asyncio
async def test_provider_rate_limit_gets_an_explicit_telegram_message():
    class RateLimitedGateway(FakeAgentGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            yield AgentEvent(
                type="run_failed",
                code="RATE_LIMITED",
                message="Agent provider is temporarily rate limited",
            )

    responder = make_telegram_responder(RateLimitedGateway())
    trigger = FakeMessage("/ai hello")
    request = AgentRunRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        session_id=None,
        parent_entry_id=None,
        prompt="hello",
        context=(),
        system_prompt=PromptBuilder().system_prompt,
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
            anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
        ),
        origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
    )

    result = await responder.answer(trigger, request)

    assert result.succeeded is False
    assert trigger.replies[0].text == (
        "AI provider is temporarily rate limited. Try again later."
    )


@pytest.mark.asyncio
async def test_unavailable_session_gets_a_short_explicit_telegram_message():
    class UnavailableSessionGateway(FakeAgentGateway):
        async def run(self, request):
            self.requests.append(request)
            yield AgentEvent(
                type="run_failed",
                code="SESSION_UNAVAILABLE",
                message="Agent session is unavailable",
            )

    gateway = UnavailableSessionGateway()
    responder = make_telegram_responder(gateway)
    trigger = FakeMessage("/ai continue")
    request = AgentRunRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        session_id="legacy-session",
        parent_entry_id="legacy-entry",
        prompt="continue",
        context=(),
        system_prompt=PromptBuilder().system_prompt,
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
            anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
        ),
        origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
    )

    result = await responder.answer(trigger, request)

    assert result.succeeded is False
    assert result.failure_code == "SESSION_UNAVAILABLE"
    assert trigger.replies[0].text == "AI thread unavailable. Start over."


@pytest.mark.asyncio
async def test_provider_timeout_gets_an_explicit_telegram_message():
    class TimeoutGateway(FakeAgentGateway):
        async def run(self, request):
            self.requests.append(request)
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-timeout",
            )
            raise TimeoutError

    gateway = TimeoutGateway()
    responder = make_telegram_responder(gateway)
    trigger = FakeMessage("/ai hello")
    request = AgentRunRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        session_id=None,
        parent_entry_id=None,
        prompt="hello",
        context=(),
        system_prompt=PromptBuilder().system_prompt,
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
            anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
        ),
        origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
    )

    result = await responder.answer(trigger, request)

    assert result.succeeded is False
    assert result.failure_code == "TIMEOUT"
    assert trigger.replies[0].text == "AI request timed out. Try again later."


@pytest.mark.asyncio
async def test_provider_timeout_preserves_a_streamed_telegram_answer():
    class PartialTimeoutGateway(FakeAgentGateway):
        async def run(self, request):
            self.requests.append(request)
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-timeout",
            )
            yield AgentEvent(
                type="text_delta",
                delta="I found Alice and Bob, but I could not verify Carol.",
                reset=True,
            )
            raise TimeoutError

    gateway = PartialTimeoutGateway()
    responder = make_telegram_responder(gateway)
    trigger = FakeMessage("/ai hello")
    request = AgentRunRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        session_id=None,
        parent_entry_id=None,
        prompt="hello",
        context=(),
        system_prompt=PromptBuilder().system_prompt,
        tool_policy="delegated",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
            anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
        ),
        origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
    )

    result = await responder.answer(trigger, request)

    assert result.succeeded is False
    assert result.failure_code == "TIMEOUT"
    assert result.text == (
        "I found Alice and Bob, but I could not verify Carol.\n\n"
        "AI time limit reached; this answer may be incomplete."
    )
    assert trigger.replies[0].text == result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_settled", [False, True])
async def test_provider_timeout_does_not_publish_pre_tool_narration(tool_settled):
    class ToolTimeoutGateway(FakeAgentGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-timeout",
            )
            yield AgentEvent(
                type="text_delta",
                delta="I will update that now.",
                reset=True,
            )
            yield AgentEvent(
                type="tool_snapshot",
                phase="started",
                tool="requester_memory",
                summary="Updating requester customization",
            )
            if tool_settled:
                yield AgentEvent(
                    type="tool_snapshot",
                    phase="completed",
                    tool="requester_memory",
                    summary="Requester customization saved",
                )
            raise TimeoutError

    responder = make_telegram_responder(ToolTimeoutGateway())
    trigger = FakeMessage("/ai update it")

    result = await responder.answer(
        trigger,
        AgentRunRequest(
            run_id="11111111-1111-4111-8111-111111111111",
            session_id=None,
            parent_entry_id=None,
            prompt="update it",
            context=(),
            system_prompt=PromptBuilder().system_prompt,
            tool_policy="delegated",
            identity=AgentRequestIdentity(
                requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
                anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
            ),
            origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
        ),
    )

    assert result.succeeded is False
    assert result.failure_code == "TOOL_OUTCOME_UNCONFIRMED"
    assert trigger.replies[0].text == (
        "AI returned no final response after using a tool. "
        "The action may already have completed; verify before retrying."
    )


@pytest.mark.asyncio
async def test_provider_timeout_preserves_post_tool_answer_text():
    class PostToolTimeoutGateway(FakeAgentGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-timeout",
            )
            yield AgentEvent(
                type="text_delta",
                delta="I will check that.",
                reset=True,
            )
            yield AgentEvent(
                type="tool_snapshot",
                phase="started",
                tool="web_search",
                summary="Searching web",
            )
            yield AgentEvent(
                type="tool_snapshot",
                phase="completed",
                tool="web_search",
                summary="Web search completed",
            )
            yield AgentEvent(
                type="text_delta",
                delta="The verified result is 42.",
                reset=True,
            )
            raise TimeoutError

    responder = make_telegram_responder(PostToolTimeoutGateway())
    trigger = FakeMessage("/ai check")

    result = await responder.answer(
        trigger,
        AgentRunRequest(
            run_id="11111111-1111-4111-8111-111111111111",
            session_id=None,
            parent_entry_id=None,
            prompt="check",
            context=(),
            system_prompt=PromptBuilder().system_prompt,
            tool_policy="delegated",
            identity=AgentRequestIdentity(
                requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
                anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
            ),
            origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
        ),
    )

    assert result.succeeded is False
    assert result.failure_code == "TIMEOUT"
    assert result.text == (
        "The verified result is 42.\n\n"
        "AI time limit reached; this answer may be incomplete."
    )


@pytest.mark.asyncio
async def test_provider_timeout_does_not_bypass_output_policy():
    class PartialTimeoutGateway(FakeAgentGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-timeout",
            )
            yield AgentEvent(type="text_delta", delta="partial", reset=True)
            raise TimeoutError

    responder = make_telegram_responder(
        PartialTimeoutGateway(),
        output_policy=MainlandMessagingOutputPolicy(),
    )
    trigger = FakeMessage("/ai hello")

    result = await responder.answer(
        trigger,
        AgentRunRequest(
            run_id="11111111-1111-4111-8111-111111111111",
            session_id=None,
            parent_entry_id=None,
            prompt="hello",
            context=(),
            system_prompt=PromptBuilder().system_prompt,
            tool_policy="delegated",
            identity=AgentRequestIdentity(
                requester=AgentIdentityAnchor("telegram:user:10", "Tester"),
                anchors=(AgentIdentityAnchor("telegram:user:10", "Tester"),),
            ),
            origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
        ),
    )

    assert result.succeeded is False
    assert result.failure_code == "TIMEOUT"
    assert trigger.replies[0].text == "AI request timed out. Try again later."


@pytest.mark.asyncio
async def test_handler_maps_answers_to_pi_sessions_and_forks_by_entry():
    gateway = FakeAgentGateway(["root answer", "child answer", "fork answer"])
    store = FakeStore(allowed={TELEGRAM_IDENTITY_CODEC.actor_id(20)})
    handler = AIConversationHandler(
        owner_id=10,
        responder=make_telegram_responder(gateway),
        store=store,
        prompt_builder=PromptBuilder(identity_codec=TELEGRAM_IDENTITY_CODEC),
        identity_codec=TELEGRAM_IDENTITY_CODEC,
    )
    root = FakeMessage("/ai root prompt", sender_id=20)
    await handler.handle(root)
    root_answer = root.replies[0]
    root_marker = store.markers[
        (TELEGRAM_IDENTITY_CODEC.scope_id(root.chat_id), root_answer.id)
    ]

    child = FakeMessage("child prompt", sender_id=20, reply_to=root_answer)
    await handler.handle(child)
    fork = FakeMessage("fork prompt", sender_id=20, reply_to=root_answer)
    await handler.handle(fork)

    assert gateway.requests[0].tool_policy == "delegated"
    assert gateway.requests[0].session_id is None
    assert gateway.requests[1].session_id == root_marker.agent_session_id
    assert gateway.requests[1].parent_entry_id == root_marker.agent_entry_id
    assert gateway.requests[2].session_id == root_marker.agent_session_id
    assert gateway.requests[2].parent_entry_id == root_marker.agent_entry_id
    assert all(request.prompt != "child prompt" for request in [gateway.requests[2]])


@pytest.mark.asyncio
async def test_ai_cancel_aborts_only_the_requesters_active_run():
    gateway = BlockingAgentGateway()
    store = FakeStore()
    handler = AIConversationHandler(
        owner_id=10,
        responder=make_telegram_responder(gateway),
        store=store,
        prompt_builder=PromptBuilder(identity_codec=TELEGRAM_IDENTITY_CODEC),
        identity_codec=TELEGRAM_IDENTITY_CODEC,
    )
    trigger = FakeMessage("/ai wait")
    running = asyncio.create_task(handler.handle(trigger))
    await gateway.started.wait()
    run_id = gateway.requests[0].run_id

    cancel = FakeMessage("/ai_cancel")
    assert await handler.handle(cancel) is True
    await running

    assert gateway.cancelled == [run_id]
    assert cancel.replies[0].text == "AI request cancellation requested."
    assert trigger.replies[0].text == "AI request cancelled."


@pytest.mark.asyncio
async def test_ai_cancel_persists_queue_intent_before_fallible_pi_cancel():
    events: list[str] = []

    class FailingCancelGateway(FakeAgentGateway):
        async def cancel(self, run_id: str) -> bool:
            del run_id
            events.append("pi-cancel")
            raise RuntimeError("cancel transport failed")

    class WorkflowControl:
        async def cancel_generations(
            self,
            principal_actor_id: str,
            *,
            interrupt_running: bool,
        ) -> AIWorkflowCancellation:
            assert principal_actor_id == TELEGRAM_IDENTITY_CODEC.actor_id(10)
            assert interrupt_running
            events.append("durable-cancel")
            return AIWorkflowCancellation(queued=1, running=1)

        async def reschedule_scope(self, _scope_id: str) -> int:
            return 0

    handler = AIConversationHandler(
        owner_id=10,
        responder=make_telegram_responder(FailingCancelGateway()),
        store=FakeStore(),
        prompt_builder=PromptBuilder(identity_codec=TELEGRAM_IDENTITY_CODEC),
        identity_codec=TELEGRAM_IDENTITY_CODEC,
    )
    control = WorkflowControl()
    handler.bind_workflow_control(control)
    actor_id = TELEGRAM_IDENTITY_CODEC.actor_id(10)
    handler._active_runs[actor_id] = {"run-with-failed-cancel"}

    cancel = FakeMessage("/ai_cancel")
    assert await handler.handle(cancel) is True

    assert events == ["durable-cancel", "pi-cancel"]
    assert cancel.replies[0].text == (
        "AI request cancellation requested; cancelled 1 queued request."
    )


@pytest.mark.asyncio
async def test_reply_to_active_request_reports_that_ai_is_still_working():
    gateway = BlockingAgentGateway()
    handler = AIConversationHandler(
        owner_id=10,
        responder=make_telegram_responder(gateway),
        store=FakeStore(),
        prompt_builder=PromptBuilder(identity_codec=TELEGRAM_IDENTITY_CODEC),
        identity_codec=TELEGRAM_IDENTITY_CODEC,
    )
    trigger = FakeMessage("/ai wait")
    running = asyncio.create_task(handler.handle(trigger))
    await gateway.started.wait()

    try:
        follow_up = FakeMessage("Are you still working?", reply_to=trigger)
        assert await handler.handle(follow_up) is True
        assert follow_up.replies[0].text == (
            "AI is still working. Please wait for the answer."
        )
        assert len(gateway.requests) == 1

        outgoing_echo = FakeMessage("Generated answer", reply_to=trigger)
        outgoing_echo.id = trigger.replies[0].id
        outgoing_echo.out = True
        assert await handler.handle(outgoing_echo) is False
        assert outgoing_echo.replies == []
        assert len(gateway.requests) == 1

        other_chat = FakeMessage(
            "Are you still working?",
            chat_id=-1002,
            reply_to=trigger,
        )
        assert await handler.handle(other_chat) is False
        assert other_chat.replies == []
    finally:
        gateway.cancelled_event.set()
        await running

    late_follow_up = FakeMessage("Still working?", reply_to=trigger)
    assert await handler.handle(late_follow_up) is False
    assert late_follow_up.replies == []


@pytest.mark.asyncio
async def test_generated_command_echo_is_rejected_but_manual_outgoing_is_preserved():
    gateway = FakeAgentGateway(["/ai must not run", "manual answer"])
    transport = TelegramChatTransport(edit_cadence=0)
    handler = AIConversationHandler(
        owner_id=10,
        responder=AIResponder(gateway, transport=transport),
        store=FakeStore(),
        prompt_builder=PromptBuilder(
            transport=transport,
            identity_codec=TELEGRAM_IDENTITY_CODEC,
        ),
        transport=transport,
        identity_codec=TELEGRAM_IDENTITY_CODEC,
    )
    trigger = FakeMessage("/ai first")

    assert await handler.handle(trigger) is True
    generated_answer = trigger.replies[0]
    echoed = FakeMessage("/ai must not run")
    echoed.id = generated_answer.id
    echoed.out = True

    assert await handler.handle(echoed) is False
    assert len(gateway.requests) == 1

    manual = FakeMessage("/ai manual request")
    manual.out = True

    assert await handler.handle(manual) is True
    assert len(gateway.requests) == 2


@pytest.mark.asyncio
async def test_generated_owner_control_cannot_change_channel_configuration():
    gateway = FakeAgentGateway(["/ai_prefix /ask"])
    store = FakeStore()
    transport = TelegramChatTransport(edit_cadence=0)
    handler = AIConversationHandler(
        owner_id=10,
        responder=AIResponder(gateway, transport=transport),
        store=store,
        prompt_builder=PromptBuilder(
            transport=transport,
            identity_codec=TELEGRAM_IDENTITY_CODEC,
        ),
        transport=transport,
        identity_codec=TELEGRAM_IDENTITY_CODEC,
    )
    trigger = FakeMessage("/ai produce a command")
    assert await handler.handle(trigger) is True
    generated_answer = trigger.replies[0]
    echoed = FakeMessage("/ai_prefix /ask")
    echoed.id = generated_answer.id
    echoed.out = True
    echoed.is_group = True

    assert await handler.handle(echoed) is False
    scope_id = TELEGRAM_IDENTITY_CODEC.scope_id(echoed.chat_id)
    assert await store.get_ai_command_prefix(scope_id) is None

    manual = FakeMessage("/ai_prefix /ask")
    manual.out = True
    manual.is_group = True
    assert await handler.handle(manual) is True
    assert await store.get_ai_command_prefix(scope_id) == "/ask"
