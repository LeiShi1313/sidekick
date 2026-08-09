import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from telethon.errors import FloodWaitError
from telethon.tl import functions as telegram_functions
from telethon.tl import types as telegram_types

from sidekick.ai import (
    AIConversationHandler,
    AIResponder,
    AISettings,
    AgentEvent,
    AgentIdentityAnchor,
    AgentModelCatalog,
    AgentRequestIdentity,
    AgentRunOrigin,
    AgentRunRequest,
    PromptBuilder,
)
from sidekick.chat.commands import (
    AIAskCommand,
    BankGrantCommand,
    DirectoryPublishCommand,
    InvalidCommand,
    MemoryBackfillCommand,
    MemoryRememberCommand,
    parse_chat_command,
)
from sidekick.plugins.base import command_registry
from sidekick.telegram.ai_identity import TELEGRAM_IDENTITY_CODEC
from sidekick.telegram.ai_transport import (
    TelegramChatTransport,
    select_telegram_response_format,
)
import sidekick.plugins.ai  # noqa: F401


class FakeAnswer:
    def __init__(self, text: str):
        self.id = 100
        self.initial_text = text
        self.text = text
        self.edits: list[str] = []
        self.edit_calls: list[tuple[str, dict]] = []

    async def edit(self, text: str, **kwargs):
        self.text = text
        self.edits.append(text)
        self.edit_calls.append((text, kwargs))
        return self


class FakeMessage:
    def __init__(self, text: str, sender_id: int = 10, chat_id: int = -1001):
        self.id = 1
        self.chat_id = chat_id
        self.raw_text = text
        self.sender_id = sender_id
        self.reply_to_msg_id = None
        self.replies: list[FakeAnswer] = []

    async def get_reply_message(self):
        return None

    async def reply(self, text: str, **kwargs):
        answer = FakeAnswer(text)
        self.replies.append(answer)
        return answer


class FakeTelegramClient:
    def __init__(self):
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)


class FakeRichAnswer(FakeAnswer):
    def __init__(self, text):
        super().__init__(text)
        self.client = FakeTelegramClient()

    async def get_input_chat(self):
        return telegram_types.InputPeerSelf()


class FakeRichMessage(FakeMessage):
    async def reply(self, text: str, **kwargs):
        answer = FakeRichAnswer(text)
        self.replies.append(answer)
        return answer


class FakeGateway:
    def __init__(
        self,
        chunks=(),
        error: Exception | None = None,
        catalog: AgentModelCatalog | None = None,
        catalog_error: Exception | None = None,
    ):
        self.chunks = chunks
        self.error = error
        self.catalog = catalog or AgentModelCatalog(
            default_model="gpt-5.6-sol",
            models=("claude-sonnet-4-6", "gpt-5.4-mini", "gpt-5.6-sol"),
        )
        self.catalog_error = catalog_error
        self.requests: list[AgentRunRequest] = []

    async def list_models(self) -> AgentModelCatalog:
        if self.catalog_error is not None:
            raise self.catalog_error
        return self.catalog

    async def run(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        self.requests.append(request)
        if self.error:
            raise self.error
        yield AgentEvent(
            type="run_started",
            run_id=request.run_id,
            session_id="session-1",
        )
        text = ""
        for index, chunk in enumerate(self.chunks):
            text += chunk
            yield AgentEvent(type="text_delta", delta=chunk, reset=index == 0)
        yield AgentEvent(
            type="run_completed",
            session_id="session-1",
            entry_id="entry-1",
            answer=text,
        )

    async def cancel(self, run_id: str) -> bool:
        return True


def make_telegram_responder(
    gateway,
    *,
    edit_cadence=0,
    clock=None,
    sleep=None,
    initial_status="🤔 Thinking...",
    response_format="regular_entities",
    **kwargs,
):
    transport_kwargs = {"edit_cadence": edit_cadence}
    if clock is not None:
        transport_kwargs["clock"] = clock
    if sleep is not None:
        transport_kwargs["sleep"] = sleep
    return AIResponder(
        gateway,
        initial_status=initial_status,
        transport=TelegramChatTransport(
            response_format=response_format,
            **transport_kwargs,
        ),
        **kwargs,
    )


def test_telegram_transport_distinguishes_group_messages():
    transport = TelegramChatTransport()

    assert transport.is_group(SimpleNamespace(is_group=True)) is True
    assert transport.is_group(SimpleNamespace(is_group=False)) is False


async def wait_for_edit_count(answer: FakeAnswer, count: int) -> None:
    async with asyncio.timeout(1):
        while len(answer.edits) < count:
            await asyncio.sleep(0)


class FakeStore:
    def __init__(self):
        self.saved = []
        self.model_overrides: dict[str, str] = {}
        self.ai_command_prefixes: dict[str, str] = {}
        self.memory_excluded: set[tuple[str, int, str]] = set()

    async def get_answer(self, scope_id, answer_message_id):
        return None

    async def get_turn_for_message(self, scope_id, message_id):
        return next(
            (
                marker
                for marker in reversed(self.saved)
                if marker.scope_id == scope_id
                and message_id in {marker.answer_message_id, marker.trigger_message_id}
            ),
            None,
        )

    async def save_answer(self, marker):
        self.saved.append(marker)

    async def get_ai_trigger_command_prefixes(self, scope_id, message_ids):
        return {
            marker.trigger_message_id: marker.command_prefix
            for marker in self.saved
            if marker.scope_id == scope_id and marker.trigger_message_id in message_ids
        }

    async def get_model_override(self, scope_id):
        return self.model_overrides.get(scope_id)

    async def set_model_override(self, scope_id, model):
        if model is None:
            self.model_overrides.pop(scope_id, None)
        else:
            self.model_overrides[scope_id] = model

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

    async def mark_memory_excluded_message(self, scope_id, message_id, kind):
        self.memory_excluded.add((scope_id, message_id, kind))

    async def is_memory_excluded_message(self, scope_id, message_id):
        return any(
            item[:2] == (scope_id, message_id) for item in self.memory_excluded
        )

    async def is_allowed(self, actor_id):
        return False

    async def get_last_request_at(self, scope_id, actor_id):
        return None

    async def set_last_request_at(self, scope_id, actor_id, timestamp):
        return None

    async def allow_user(self, actor_id):
        return None

    async def deny_user(self, actor_id):
        return None


def make_handler(owner_id, responder, *, store=None):
    return AIConversationHandler(
        owner_id=owner_id,
        responder=responder,
        store=store or FakeStore(),
        prompt_builder=PromptBuilder(identity_codec=TELEGRAM_IDENTITY_CODEC),
        identity_codec=TELEGRAM_IDENTITY_CODEC,
    )


def make_request(prompt: str) -> AgentRunRequest:
    return AgentRunRequest(
        run_id="11111111-1111-4111-8111-111111111111",
        session_id=None,
        parent_entry_id=None,
        prompt=prompt,
        context=(),
        system_prompt=PromptBuilder().system_prompt,
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("telegram:user:10", "Alice"),
            anchors=(AgentIdentityAnchor("telegram:user:10", "Alice"),),
        ),
        origin=AgentRunOrigin("telegram:chat:-1001", "telegram-test"),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/ai hello", AIAskCommand(prompt="hello")),
        ("/ai\nhello", AIAskCommand(prompt="hello")),
        ("/ai", AIAskCommand(prompt="")),
        ("/ai10 hello", AIAskCommand(prompt="hello", recent_messages=10)),
        (
            "/ai10@SidekickBot summarize this",
            AIAskCommand(prompt="summarize this", recent_messages=10),
        ),
        ("/ai0 invalid", AIAskCommand(prompt="invalid", recent_messages=0)),
        (" /ai hello", None),
        ("/air hello", None),
        ("/ai10x hello", None),
        (
            "/ai_memory hello",
            MemoryRememberCommand(instruction="hello"),
        ),
        ("hello /ai", None),
    ],
)
def test_parse_ai_trigger_has_an_exact_command_boundary(text, expected):
    assert parse_chat_command(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "/ai_memory remember this",
            MemoryRememberCommand(instruction="remember this"),
        ),
        (
            "/ai_memory\nforget that",
            MemoryRememberCommand(instruction="forget that"),
        ),
        ("/ai_memory", MemoryRememberCommand(instruction="")),
        ("/ai_memoryx no", None),
    ],
)
def test_parse_memory_revision_has_an_exact_command_boundary(text, expected):
    assert parse_chat_command(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/ai_directory", DirectoryPublishCommand(arguments="")),
        (
            "/ai_directory @Seele_Leaks 原神爆料频道",
            DirectoryPublishCommand(arguments="@Seele_Leaks 原神爆料频道"),
        ),
        (
            "/ai_directory\ntelegram:chat:-100123 Coder OT",
            DirectoryPublishCommand(arguments="telegram:chat:-100123 Coder OT"),
        ),
        ("/ai_directoryx", None),
        ("/ai_bank_allow", BankGrantCommand(allowed=True, source="")),
        (
            "/ai_bank_allow qq:group:686743769",
            BankGrantCommand(allowed=True, source="qq:group:686743769"),
        ),
        (
            "/ai_bank_deny @Seele_Leaks",
            BankGrantCommand(allowed=False, source="@Seele_Leaks"),
        ),
        ("/ai_bank_allowx", None),
    ],
)
def test_parse_directory_commands_preserves_adapter_arguments(text, expected):
    assert parse_chat_command(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "/ai_memory_backfill days 7",
            MemoryBackfillCommand(mode="days", value=7),
        ),
        (
            "/ai_memory_backfill\nmessages\t500",
            MemoryBackfillCommand(mode="messages", value=500),
        ),
        (
            "/ai_memory_backfill days 0",
            InvalidCommand(name="/ai_memory_backfill"),
        ),
        (
            "/ai_memory_backfill days 31",
            InvalidCommand(name="/ai_memory_backfill"),
        ),
        (
            "/ai_memory_backfill messages 5001",
            InvalidCommand(name="/ai_memory_backfill"),
        ),
        (
            "/ai_memory_backfill weeks 2",
            InvalidCommand(name="/ai_memory_backfill"),
        ),
        (
            "/ai_memory_backfill days 7 extra",
            InvalidCommand(name="/ai_memory_backfill"),
        ),
        ("/ai_memory_backfillx days 7", None),
    ],
)
def test_parse_memory_backfill_has_bounded_exact_syntax(text, expected):
    assert parse_chat_command(text) == expected


def test_ai_settings_are_loaded_without_provider_specific_assumptions(monkeypatch):
    values = {
        "SIDEKICK_PI_URL": "http://agent.test:8790/",
        "SIDEKICK_PI_TOKEN": "test-agent-token-that-is-long-enough",
        "SIDEKICK_AI_MAX_OUTPUT_CHARS": "1234",
        "SIDEKICK_AI_EDIT_CADENCE": "0.25",
        "SIDEKICK_MEMORY_COMMAND_DELETE_DELAY": "2.5",
        "SIDEKICK_PI_RUN_TIMEOUT": "12",
        "SIDEKICK_HINDSIGHT_TOKEN": "memory-api-token-that-is-long-enough",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = AISettings.from_env()

    assert settings.agent_url == "http://agent.test:8790"
    assert settings.agent_token == "test-agent-token-that-is-long-enough"
    assert settings.max_output_chars == 1234
    assert settings.edit_cadence == 0.25
    assert settings.memory_command_delete_delay == 2.5
    assert settings.request_timeout == 12
    assert settings.hindsight_timeout == 90
    assert settings.hindsight_token == "memory-api-token-that-is-long-enough"


def test_ai_settings_fail_closed_when_memory_has_no_credential() -> None:
    with pytest.raises(ValueError, match="Memory API token"):
        AISettings(
            agent_url="http://agent.test:8790",
            agent_token="test-agent-token-that-is-long-enough",
            hindsight_url="http://memory.test:8888",
        )


def test_ai_command_is_registered_under_telegram():
    assert command_registry.as_fire_commands()["telegram"]["ai"]


@pytest.mark.asyncio
async def test_telegram_answer_starts_with_thinking_status():
    responder = make_telegram_responder(
        FakeGateway(["answer"]),
        initial_status=sidekick.plugins.ai.TelegramAI.THINKING_REPLY,
    )
    trigger = FakeMessage("/ai answer")

    await responder.answer(trigger, make_request("answer"))

    assert trigger.replies[0].initial_text == "🤔 Thinking..."


@pytest.mark.asyncio
async def test_owner_can_inspect_the_current_chat_model_and_available_choices():
    gateway = FakeGateway()
    handler = make_handler(10, make_telegram_responder(gateway))
    command = FakeMessage("/ai_model")

    assert await handler.handle(command) is True

    assert command.replies[0].text == (
        "AI model for this chat: gpt-5.6-sol (server default).\n\n"
        "Available models:\n"
        "- claude-sonnet-4-6\n"
        "- gpt-5.4-mini\n"
        "- gpt-5.6-sol\n\n"
        "Use /ai_model <model-id> to switch, or /ai_model default to reset."
    )
    assert gateway.requests == []


@pytest.mark.asyncio
async def test_chat_model_override_applies_only_to_that_chat():
    gateway = FakeGateway(["answer"])
    store = FakeStore()
    handler = make_handler(
        10,
        make_telegram_responder(gateway),
        store=store,
    )

    selected = FakeMessage("/ai_model gpt-5.4-mini", chat_id=-1001)
    assert await handler.handle(selected) is True
    assert selected.replies[0].text == (
        "AI model for this chat set to gpt-5.4-mini."
    )
    assert store.memory_excluded == {
        ("telegram:chat:-1001", selected.id, "ai-control"),
        ("telegram:chat:-1001", selected.replies[0].id, "ai-control"),
    }

    assert await handler.handle(FakeMessage("/ai first", chat_id=-1001)) is True
    assert await handler.handle(FakeMessage("/ai second", chat_id=-1002)) is True

    assert [request.model for request in gateway.requests] == [
        "gpt-5.4-mini",
        None,
    ]


@pytest.mark.asyncio
async def test_unknown_or_nonowner_model_selection_does_not_change_the_chat():
    gateway = FakeGateway()
    store = FakeStore()
    handler = make_handler(
        10,
        make_telegram_responder(gateway),
        store=store,
    )

    unknown = FakeMessage("/ai_model missing-model")
    assert await handler.handle(unknown) is True
    assert unknown.replies[0].text == (
        "Unknown AI model: missing-model. Use /ai_model to list available models."
    )

    nonowner = FakeMessage("/ai_model gpt-5.4-mini", sender_id=20)
    assert await handler.handle(nonowner) is False
    assert nonowner.replies == []
    assert store.model_overrides == {}


@pytest.mark.asyncio
async def test_invalid_model_command_is_excluded_from_continuous_memory():
    store = FakeStore()
    handler = make_handler(
        10,
        make_telegram_responder(FakeGateway()),
        store=store,
    )
    command = FakeMessage("/ai_model two models")

    assert await handler.handle(command) is True

    assert command.replies[0].text == "Usage: /ai_model [model-id|default]"
    assert store.memory_excluded == {
        ("telegram:chat:-1001", command.id, "ai-control"),
        ("telegram:chat:-1001", command.replies[0].id, "ai-control"),
    }


@pytest.mark.asyncio
async def test_model_reset_succeeds_even_when_the_catalog_is_unavailable():
    gateway = FakeGateway(catalog_error=RuntimeError("provider unavailable"))
    store = FakeStore()
    store.model_overrides["telegram:chat:-1001"] = "gpt-5.4-mini"
    handler = make_handler(
        10,
        make_telegram_responder(gateway),
        store=store,
    )
    command = FakeMessage("/ai_model default")

    assert await handler.handle(command) is True

    assert command.replies[0].text == (
        "AI model for this chat reset to the server default."
    )
    assert store.model_overrides == {}

    unavailable = FakeMessage("/ai_model")
    assert await handler.handle(unavailable) is True
    assert unavailable.replies[0].text == (
        "AI model catalog is unavailable. Try again shortly."
    )


@pytest.mark.asyncio
async def test_first_stream_edit_waits_for_meaningful_accumulated_text():
    chunks = ["A", " useful", " first", " update", " " + "x" * 80, " tail"]
    responder = make_telegram_responder(FakeGateway(chunks))
    trigger = FakeMessage("/ai explain")

    await responder.answer(trigger, make_request("explain"))

    first_update = "".join(chunks[:-1])
    assert len(first_update) >= 100
    assert trigger.replies[0].edits[0] == first_update


@pytest.mark.asyncio
async def test_first_stream_gate_counts_rendered_text_instead_of_markup_source():
    long_link = "[A](https://example.com/" + "x" * 150 + ")"
    delta_consumed = asyncio.Event()
    finish = asyncio.Event()

    class PausingGateway(FakeGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            yield AgentEvent(type="text_delta", delta=long_link, reset=True)
            delta_consumed.set()
            await finish.wait()
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer=long_link,
            )

    responder = make_telegram_responder(PausingGateway())
    trigger = FakeMessage("/ai link")

    answering = asyncio.create_task(responder.answer(trigger, make_request("link")))
    await delta_consumed.wait()
    await asyncio.sleep(0)

    assert trigger.replies[0].edits == []

    finish.set()
    await answering
    assert trigger.replies[0].text == "A"


@pytest.mark.asyncio
async def test_first_stream_gate_ignores_rich_markdown_link_targets():
    long_link = "[A](https://example.com/" + "x" * 150 + ")"
    delta_consumed = asyncio.Event()
    finish = asyncio.Event()

    class PausingGateway(FakeGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            yield AgentEvent(type="text_delta", delta=long_link, reset=True)
            delta_consumed.set()
            await finish.wait()
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer=long_link,
            )

    responder = make_telegram_responder(
        PausingGateway(),
        response_format="rich_markdown",
    )
    trigger = FakeRichMessage("/ai link")

    answering = asyncio.create_task(responder.answer(trigger, make_request("link")))
    await delta_consumed.wait()
    await asyncio.sleep(0)

    assert trigger.replies[0].client.requests == []

    finish.set()
    await answering
    assert trigger.replies[0].client.requests[-1].rich_message.markdown == long_link


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("opening", "completion", "expected"),
    [
        ("**", f'{"A" * 49}.**', f'{"A" * 49}.'),
        (
            "[",
            f'{"A" * 49}.](https://example.com)',
            f'{"A" * 49}.',
        ),
        ("- ", f'{"A" * 49}.', f'- {"A" * 49}.'),
    ],
)
async def test_first_stream_gate_ignores_syntax_only_openers(
    opening,
    completion,
    expected,
):
    opening_consumed = asyncio.Event()
    send_content = asyncio.Event()
    content_consumed = asyncio.Event()
    finish = asyncio.Event()
    timer_started = asyncio.Event()

    async def blocked_sleep(_seconds):
        timer_started.set()
        await asyncio.Future()

    class PausingGateway(FakeGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            yield AgentEvent(type="text_delta", delta=opening, reset=True)
            opening_consumed.set()
            await send_content.wait()
            yield AgentEvent(
                type="text_delta",
                delta=completion,
                reset=False,
            )
            content_consumed.set()
            await finish.wait()
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer=opening + completion,
            )

    responder = make_telegram_responder(PausingGateway(), sleep=blocked_sleep)
    trigger = FakeMessage("/ai format")

    answering = asyncio.create_task(responder.answer(trigger, make_request("format")))
    await opening_consumed.wait()
    await asyncio.sleep(0)
    assert timer_started.is_set() is False
    assert trigger.replies[0].edits == []

    send_content.set()
    await content_consumed.wait()
    await wait_for_edit_count(trigger.replies[0], 1)
    assert trigger.replies[0].text == expected

    finish.set()
    await answering


@pytest.mark.asyncio
async def test_first_stream_edit_is_released_by_timer_without_another_delta():
    delta_consumed = asyncio.Event()
    finish = asyncio.Event()
    timer_started = asyncio.Event()
    release_timer = asyncio.Event()

    async def controlled_sleep(seconds):
        assert seconds == 1.0
        timer_started.set()
        await release_timer.wait()

    class PausingGateway(FakeGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            yield AgentEvent(type="text_delta", delta="A", reset=True)
            delta_consumed.set()
            await finish.wait()
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer="A",
            )

    responder = make_telegram_responder(PausingGateway(), sleep=controlled_sleep)
    trigger = FakeMessage("/ai explain")

    answering = asyncio.create_task(responder.answer(trigger, make_request("explain")))
    await delta_consumed.wait()
    await timer_started.wait()
    assert trigger.replies[0].edits == []

    release_timer.set()
    await wait_for_edit_count(trigger.replies[0], 1)
    assert trigger.replies[0].edits == ["A"]

    finish.set()
    await answering


@pytest.mark.asyncio
async def test_first_stream_gate_restarts_after_a_tool_status():
    first_turn = "A" * 100

    class ToolGateway(FakeGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            yield AgentEvent(type="text_delta", delta=first_turn, reset=True)
            yield AgentEvent(
                type="tool_snapshot",
                phase="started",
                tool="web_search",
                summary="Searching web",
            )
            yield AgentEvent(type="text_delta", delta="B", reset=True)
            yield AgentEvent(type="text_delta", delta=" short", reset=False)
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer="B short",
            )

    responder = make_telegram_responder(ToolGateway())
    trigger = FakeMessage("/ai search")

    await responder.answer(trigger, make_request("search"))

    assert trigger.replies[0].edits == [first_turn, "Searching web", "B short"]


@pytest.mark.asyncio
async def test_stream_updates_flush_latest_snapshot_without_another_delta():
    now = [0.0]
    first_update = "A" * 100
    deltas_consumed = asyncio.Event()
    finish = asyncio.Event()
    cooldown_started = asyncio.Event()
    release_cooldown = asyncio.Event()

    async def controlled_sleep(seconds):
        assert seconds == 4.0
        cooldown_started.set()
        await release_cooldown.wait()
        now[0] += seconds

    class PausingGateway(FakeGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            text = first_update
            yield AgentEvent(type="text_delta", delta=text, reset=True)
            for chunk in [" one", " two"]:
                text += chunk
                yield AgentEvent(
                    type="text_delta",
                    delta=chunk,
                    reset=False,
                )
            deltas_consumed.set()
            await finish.wait()
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer=text,
            )

    responder = make_telegram_responder(
        PausingGateway(),
        edit_cadence=4,
        clock=lambda: now[0],
        sleep=controlled_sleep,
    )
    trigger = FakeMessage("/ai explain")

    answering = asyncio.create_task(responder.answer(trigger, make_request("explain")))
    await deltas_consumed.wait()
    await asyncio.wait_for(cooldown_started.wait(), timeout=1)
    assert trigger.replies[0].edits == [first_update]

    release_cooldown.set()
    await wait_for_edit_count(trigger.replies[0], 2)
    assert trigger.replies[0].edits == [
        first_update,
        f"{first_update} one two",
    ]

    finish.set()
    await answering


@pytest.mark.asyncio
async def test_final_answer_supersedes_a_queued_tool_status_and_closes_state():
    now = [0.0]
    final_answer = "A" * 100
    cooldown_started = asyncio.Event()
    release_cooldown = asyncio.Event()

    async def controlled_sleep(seconds):
        assert seconds == 4.0
        cooldown_started.set()
        await release_cooldown.wait()
        now[0] += seconds

    class ToolGateway(FakeGateway):
        async def run(self, request):
            yield AgentEvent(
                type="run_started",
                run_id=request.run_id,
                session_id="session-1",
            )
            yield AgentEvent(type="text_delta", delta=final_answer, reset=True)
            yield AgentEvent(
                type="tool_snapshot",
                phase="started",
                tool="web_search",
                summary="Searching web",
            )
            yield AgentEvent(
                type="run_completed",
                session_id="session-1",
                entry_id="entry-1",
                answer=final_answer,
            )

    responder = make_telegram_responder(
        ToolGateway(),
        edit_cadence=4,
        clock=lambda: now[0],
        sleep=controlled_sleep,
    )
    trigger = FakeMessage("/ai search")

    answering = asyncio.create_task(responder.answer(trigger, make_request("search")))
    await asyncio.wait_for(cooldown_started.wait(), timeout=1)
    finished_before_release = answering.done()
    release_cooldown.set()
    result = await answering
    await asyncio.sleep(0)

    assert finished_before_release is False
    assert result.succeeded is True
    assert trigger.replies[0].text == final_answer
    assert "Searching web" not in trigger.replies[0].edits
    assert responder.transport._update_states == {}


@pytest.mark.asyncio
async def test_owner_gets_one_answer_without_a_tiny_initial_edit():
    gateway = FakeGateway(["Hello", " ", "world"])
    times = iter([0.0, 1.0, 2.0, 3.0])
    responder = make_telegram_responder(
        gateway,
        edit_cadence=0.5,
        clock=lambda: next(times),
    )
    handler = make_handler(owner_id=10, responder=responder)
    trigger = FakeMessage("/ai greet me")

    handled = await handler.handle(trigger)

    assert handled is True
    assert len(trigger.replies) == 1
    assert trigger.replies[0].text == "Hello world"
    assert trigger.replies[0].edits == ["Hello world"]
    assert len(gateway.requests) == 1
    assert gateway.requests[0].prompt == "greet me"
    assert gateway.requests[0].system_prompt == PromptBuilder().system_prompt
    assert gateway.requests[0].tool_policy == "owner"


@pytest.mark.asyncio
async def test_edit_cadence_is_shared_across_answers(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(
        "sidekick.telegram.ai_transport.asyncio.sleep",
        fake_sleep,
    )
    gateway = FakeGateway(["answer"])
    responder = make_telegram_responder(
        gateway,
        edit_cadence=4,
        clock=lambda: 0.0,
    )
    first = FakeMessage("/ai first")
    second = FakeMessage("/ai second")

    await responder.answer(first, make_request("first"))
    await responder.answer(second, make_request("second"))

    assert first.replies[0].text == "answer"
    assert second.replies[0].text == "answer"
    assert sleeps == [4]


@pytest.mark.asyncio
async def test_flood_wait_delays_final_edit_without_replacing_the_answer(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    class FloodOnceAnswer(FakeAnswer):
        def __init__(self, text):
            super().__init__(text)
            self.attempts = 0

        async def edit(self, text: str, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise FloodWaitError(request=None, capture=7)
            return await super().edit(text, **kwargs)

    class FloodOnceMessage(FakeMessage):
        async def reply(self, text: str, **kwargs):
            answer = FloodOnceAnswer(text)
            self.replies.append(answer)
            return answer

    monkeypatch.setattr(
        "sidekick.telegram.ai_transport.asyncio.sleep",
        fake_sleep,
    )
    responder = make_telegram_responder(
        FakeGateway(["final answer"]),
        edit_cadence=0,
        clock=lambda: 0.0,
    )
    trigger = FloodOnceMessage("/ai answer")

    result = await responder.answer(trigger, make_request("answer"))

    assert result.succeeded is True
    assert result.text == "final answer"
    assert trigger.replies[0].text == "final answer"
    assert sleeps == [7]


def test_response_format_switches_only_for_a_bot_rich_transport():
    assert (
        select_telegram_response_format(
            is_bot_account=False,
            rich_messages_available=True,
        )
        == "regular_entities"
    )
    assert (
        select_telegram_response_format(
            is_bot_account=True,
            rich_messages_available=False,
        )
        == "regular_entities"
    )
    assert (
        select_telegram_response_format(
            is_bot_account=True,
            rich_messages_available=True,
        )
        == "rich_markdown"
    )

@pytest.mark.asyncio
async def test_bot_response_uses_telegram_rich_markdown_edit():
    formatted = (
        "**Result**\n\n"
        "- Mode: Rich\n"
        "- [Docs](https://example.com/docs)"
    )
    gateway = FakeGateway([formatted])
    responder = make_telegram_responder(
        gateway,
        edit_cadence=0,
        response_format="rich_markdown",
    )
    trigger = FakeRichMessage("/ai format this")

    result = await responder.answer(trigger, make_request("format this"))

    answer = trigger.replies[0]
    assert result.text == formatted
    assert answer.edit_calls == []
    request = answer.client.requests[-1]
    assert isinstance(request, telegram_functions.messages.EditMessageRequest)
    assert isinstance(request.rich_message, telegram_types.InputRichMessageMarkdown)
    assert request.rich_message.markdown == formatted


@pytest.mark.asyncio
async def test_streamed_markdown_is_sent_as_native_telegram_entities():
    formatted = (
        "**Result**\n"
        "*Estimate*\n"
        "~~Obsolete~~\n"
        "Use `x < y` and [the docs](https://example.com/docs).\n"
        "```\nTeam     Score\nNorway   1\nEngland  2\n```"
    )
    gateway = FakeGateway([formatted])
    responder = make_telegram_responder(gateway)
    trigger = FakeMessage("/ai format this")

    result = await responder.answer(trigger, make_request("format this"))

    answer = trigger.replies[0]
    assert result.text == formatted
    assert answer.text == (
        "Result\nEstimate\nObsolete\nUse x < y and the docs.\n\n"
        "Team     Score\nNorway   1\nEngland  2"
    )
    _, kwargs = answer.edit_calls[-1]
    assert kwargs["parse_mode"] is None
    assert {type(entity).__name__ for entity in kwargs["formatting_entities"]} == {
        "MessageEntityBold",
        "MessageEntityItalic",
        "MessageEntityStrike",
        "MessageEntityCode",
        "MessageEntityTextUrl",
        "MessageEntityPre",
    }
    link = next(
        entity
        for entity in kwargs["formatting_entities"]
        if isinstance(entity, telegram_types.MessageEntityTextUrl)
    )
    assert link.url == "https://example.com/docs"


@pytest.mark.asyncio
async def test_short_telegram_answer_is_not_collapsed():
    responder = make_telegram_responder(FakeGateway(["A short answer."]))
    trigger = FakeMessage("/ai answer briefly")

    await responder.answer(trigger, make_request("answer briefly"))

    entities = trigger.replies[0].edit_calls[-1][1]["formatting_entities"]
    assert not any(
        isinstance(entity, telegram_types.MessageEntityBlockquote)
        for entity in entities
    )


@pytest.mark.asyncio
async def test_long_telegram_answer_is_collapsed_with_existing_formatting():
    formatted = "**🙂 Result**\n" + ("Detailed explanation. " * 40)
    responder = make_telegram_responder(FakeGateway([formatted]))
    trigger = FakeMessage("/ai explain this")

    await responder.answer(trigger, make_request("explain this"))

    answer = trigger.replies[0]
    entities = answer.edit_calls[-1][1]["formatting_entities"]
    quote = next(
        entity
        for entity in entities
        if isinstance(entity, telegram_types.MessageEntityBlockquote)
    )
    assert quote.collapsed is True
    assert quote.offset == 0
    assert quote.length == len(answer.text.encode("utf-16-le")) // 2
    assert any(
        isinstance(entity, telegram_types.MessageEntityBold)
        for entity in entities
    )


@pytest.mark.asyncio
async def test_multiline_telegram_answer_is_collapsed_before_character_limit():
    formatted = "\n".join(f"Line {index}" for index in range(11))
    responder = make_telegram_responder(FakeGateway([formatted]))
    trigger = FakeMessage("/ai list this")

    await responder.answer(trigger, make_request("list this"))

    entities = trigger.replies[0].edit_calls[-1][1]["formatting_entities"]
    assert any(
        isinstance(entity, telegram_types.MessageEntityBlockquote)
        and entity.collapsed is True
        for entity in entities
    )


@pytest.mark.asyncio
async def test_streaming_waits_for_visible_text_when_markdown_is_split():
    gateway = FakeGateway(["**", "Result", "**"])
    responder = make_telegram_responder(gateway)
    trigger = FakeMessage("/ai format this")

    result = await responder.answer(trigger, make_request("format this"))

    answer = trigger.replies[0]
    assert result.succeeded is True
    assert all(text for text, _ in answer.edit_calls)
    assert answer.text == "Result"
    assert {
        type(entity).__name__
        for entity in answer.edit_calls[-1][1]["formatting_entities"]
    } == {"MessageEntityBold"}


@pytest.mark.asyncio
@pytest.mark.parametrize("formatted", ["`+`", "```\n[]\n```"])
@pytest.mark.parametrize("response_format", ["regular_entities", "rich_markdown"])
async def test_symbol_only_code_is_still_delivered_as_the_final_answer(
    formatted,
    response_format,
):
    responder = make_telegram_responder(
        FakeGateway([formatted]),
        response_format=response_format,
    )
    trigger = (
        FakeRichMessage("/ai format this")
        if response_format == "rich_markdown"
        else FakeMessage("/ai format this")
    )

    result = await responder.answer(trigger, make_request("format this"))

    assert result.text == formatted
    if response_format == "rich_markdown":
        request = trigger.replies[0].client.requests[-1]
        assert request.rich_message.markdown == formatted
    else:
        expected = "+" if formatted == "`+`" else "[]"
        assert trigger.replies[0].text == expected


@pytest.mark.asyncio
async def test_regular_telegram_treats_unexpected_html_as_plain_text():
    formatted = "<strong>Result</strong>"
    responder = make_telegram_responder(FakeGateway([formatted]))
    trigger = FakeMessage("/ai format this")

    result = await responder.answer(trigger, make_request("format this"))

    answer = trigger.replies[0]
    assert result.text == formatted
    assert answer.text == formatted
    assert answer.edit_calls[-1][1]["formatting_entities"] == []


@pytest.mark.asyncio
async def test_regular_telegram_leaves_non_https_links_literal():
    formatted = (
        "[legacy](http://example.com) "
        "[unsafe](javascript:alert(1))"
    )
    responder = make_telegram_responder(FakeGateway([formatted]))
    trigger = FakeMessage("/ai format this")

    await responder.answer(trigger, make_request("format this"))

    answer = trigger.replies[0]
    assert answer.text == formatted
    assert answer.edit_calls[-1][1]["formatting_entities"] == []


@pytest.mark.asyncio
async def test_rich_telegram_escapes_html_and_non_https_links():
    formatted = (
        "<strong>Result</strong> and `x < y` "
        "[unsafe](javascript:alert(1))"
    )
    responder = make_telegram_responder(
        FakeGateway([formatted]),
        response_format="rich_markdown",
    )
    trigger = FakeRichMessage("/ai format this")

    await responder.answer(trigger, make_request("format this"))

    request = trigger.replies[0].client.requests[-1]
    assert request.rich_message.markdown == (
        "&lt;strong&gt;Result&lt;/strong&gt; and `x < y` "
        "&#91;unsafe](javascript:alert(1))"
    )


@pytest.mark.asyncio
async def test_unauthorized_trigger_is_silent_and_does_not_call_provider():
    gateway = FakeGateway(["must not be used"])
    handler = make_handler(
        owner_id=10,
        responder=make_telegram_responder(gateway),
    )
    trigger = FakeMessage("/ai secret", sender_id=11)

    assert await handler.handle(trigger) is False
    assert trigger.replies == []
    assert gateway.requests == []


@pytest.mark.asyncio
async def test_owner_ai_requests_are_not_blocked_by_chat_scope():
    gateway = FakeGateway(["answer"])
    handler = AIConversationHandler(
        owner_id=10,
        responder=make_telegram_responder(gateway),
        store=FakeStore(),
        prompt_builder=PromptBuilder(identity_codec=TELEGRAM_IDENTITY_CODEC),
        identity_codec=TELEGRAM_IDENTITY_CODEC,
    )
    trigger = FakeMessage("/ai secret")
    trigger.chat_id = -1002

    assert await handler.handle(trigger) is True
    assert trigger.replies[0].text == "answer"
    assert len(gateway.requests) == 1


@pytest.mark.asyncio
async def test_empty_prompt_finishes_with_usage_without_calling_provider():
    gateway = FakeGateway(["must not be used"])
    handler = make_handler(
        owner_id=10,
        responder=make_telegram_responder(gateway),
    )
    trigger = FakeMessage("/ai")

    assert await handler.handle(trigger) is True
    assert len(trigger.replies) == 1
    assert trigger.replies[0].text == "Usage: /ai <question>"
    assert gateway.requests == []


@pytest.mark.asyncio
async def test_provider_failure_replaces_loading_message():
    gateway = FakeGateway(error=RuntimeError("provider secret detail"))
    handler = make_handler(
        owner_id=10,
        responder=make_telegram_responder(gateway),
    )
    trigger = FakeMessage("/ai hello")

    assert await handler.handle(trigger) is True
    assert len(trigger.replies) == 1
    assert trigger.replies[0].text == "AI request failed. Try again later."
    assert gateway.requests


@pytest.mark.asyncio
async def test_provider_failure_uses_standard_logging_format(caplog):
    import logging

    gateway = FakeGateway(error=RuntimeError("provider secret detail"))
    responder = make_telegram_responder(
        gateway,
        logger=logging.getLogger("sidekick-ai-test"),
    )
    trigger = FakeMessage("/ai hello")

    await responder.answer(trigger, make_request("hello"))

    assert "AI agent request failed (RuntimeError)" in caplog.text
    assert "provider secret detail" not in caplog.text


@pytest.mark.asyncio
async def test_output_is_bounded_and_finalized():
    gateway = FakeGateway(["abcdefghijk"])
    responder = make_telegram_responder(gateway, max_output_chars=10)
    trigger = FakeMessage("/ai long")

    await responder.answer(trigger, make_request("long"))

    assert trigger.replies[0].text == "abcdefg..."
