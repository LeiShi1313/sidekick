from __future__ import annotations

import ast
from collections.abc import AsyncIterator
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
import sqlite3

import pytest

from sidekick.ai import (
    AIAnswerMarker,
    AIConversationHandler,
    AIResponder,
    AIStateRepository,
    AgentEvent,
    AgentIdentityAnchor,
    AgentRequestIdentity,
    AgentRunOrigin,
    AgentRunRequest,
    MemoryScopeState,
    PromptBuilder,
)
from sidekick.chat.attachments import AttachmentDescription, AttachmentReference
from sidekick.chat.commands import (
    AIAskCommand,
    AICancelCommand,
    AILimitCommand,
    AIModelCommand,
    AccessCommand,
    ChatAccessCommand,
    InvalidCommand,
    MemoryBackfillCommand,
    MemoryModeCommand,
    MemoryRememberCommand,
    MemoryStatusCommand,
    parse_chat_command,
)
from sidekick.chat.identity import NamespacedIdentityCodec
from sidekick.chat.transport import ChatPresentation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/ai hello", AIAskCommand(prompt="hello")),
        ("/ai10 hello", AIAskCommand(prompt="hello", recent_messages=10)),
        (
            "/ai10@SidekickBot summarize",
            AIAskCommand(prompt="summarize", recent_messages=10),
        ),
        ("/ai_cancel", AICancelCommand()),
        ("/ai_limit", AILimitCommand(action="show")),
        (
            "/ai_limit 60",
            AILimitCommand(action="set", cooldown_seconds=60),
        ),
        ("/ai_limit 0", AILimitCommand(action="set", cooldown_seconds=0)),
        (
            "/ai_limit 86400",
            AILimitCommand(action="set", cooldown_seconds=86_400),
        ),
        ("/ai_limit default", AILimitCommand(action="reset")),
        ("/ai_limit -1", InvalidCommand(name="/ai_limit")),
        ("/ai_limit 86401", InvalidCommand(name="/ai_limit")),
        ("/ai_limit 1.5", InvalidCommand(name="/ai_limit")),
        ("/ai_limit 10 seconds", InvalidCommand(name="/ai_limit")),
        ("/ai_model", AIModelCommand(action="show")),
        (
            "/ai_model gpt-5.4-mini",
            AIModelCommand(action="set", model="gpt-5.4-mini"),
        ),
        ("/ai_model default", AIModelCommand(action="reset")),
        ("/ai_model two models", InvalidCommand(name="/ai_model")),
        ("/ai_allow", AccessCommand(allowed=True)),
        ("/ai_deny", AccessCommand(allowed=False)),
        ("/ai_access open", ChatAccessCommand(action="open")),
        ("/ai_access restricted", ChatAccessCommand(action="restricted")),
        ("/ai_access status", ChatAccessCommand(action="status")),
        ("/ai_access", InvalidCommand(name="/ai_access")),
        ("/ai_access everyone", InvalidCommand(name="/ai_access")),
        (
            "/ai_memory remember this",
            MemoryRememberCommand(instruction="remember this"),
        ),
        (
            "/ai_memory_backfill messages 500",
            MemoryBackfillCommand(mode="messages", value=500),
        ),
        (
            "/ai_memory_enable qq-group-alias",
            MemoryModeCommand(
                mode="continuous",
                enabled=True,
                target="qq-group-alias",
            ),
        ),
        ("/ai_memory_status", MemoryStatusCommand()),
        (
            "/ai_memory_backfill days 31",
            InvalidCommand(name="/ai_memory_backfill"),
        ),
    ],
)
def test_chat_commands_are_parsed_without_transport_assumptions(text, expected):
    assert parse_chat_command(text) == expected


def test_identity_codec_keeps_network_identities_disjoint():
    telegram = NamespacedIdentityCodec(
        source="telegram",
        actor_kind="user",
        scope_kind="chat",
    )
    qq = NamespacedIdentityCodec(
        source="qq",
        actor_kind="user",
        scope_kind="group",
    )

    assert telegram.actor_id(42) == "telegram:user:42"
    assert qq.actor_id(42) == "qq:user:42"
    assert telegram.scope_id(7) == "telegram:chat:7"
    assert qq.scope_id(7) == "qq:group:7"
    assert qq.message_source_id(7, 9) == "qq:message:7:9"
    assert qq.thread_document_id(7, 9) == "qq:thread:7:9"
    assert qq.revision_document_id(7, 9) == "qq:revision:7:9"
    assert telegram.parse_scope_id("telegram:chat:7") == 7
    assert qq.parse_scope_id("qq:group:7") == 7
    assert qq.parse_scope_id("telegram:chat:7") is None


def test_identity_codec_round_trips_message_source_components():
    codec = NamespacedIdentityCodec(
        source="example",
        actor_kind="user",
        scope_kind="chat",
    )
    source_id = codec.message_source_id("room:one", "message:two")

    assert codec.parse_message_source_id(source_id) == (
        "room:one",
        "message:two",
    )


def test_attachment_reference_contains_metadata_but_no_binary_payload():
    reference = AttachmentReference(
        key="onebot11:self-1:message-2:image-0",
        kind="image",
        mime_type="image/jpeg",
        filename="photo.jpg",
        size_bytes=1234,
    )

    assert reference.size_bytes == 1234
    assert "data" not in {item.name for item in fields(reference)}
    assert "content" not in {item.name for item in fields(reference)}
    assert not any(
        isinstance(getattr(reference, item.name), bytes) for item in fields(reference)
    )


@pytest.mark.asyncio
async def test_chat_model_override_can_be_replaced_reset_and_reloaded(tmp_path):
    path = tmp_path / "ai.db"
    scope_id = "telegram:chat:-1001"
    store = await AIStateRepository(path).connect()
    try:
        assert await store.get_model_override(scope_id) is None

        await store.set_model_override(scope_id, "gpt-5.4-mini")
        assert await store.get_model_override(scope_id) == "gpt-5.4-mini"

        await store.set_model_override(scope_id, "claude-sonnet-4-6")
    finally:
        await store.close()

    restarted = await AIStateRepository(path).connect()
    try:
        assert await restarted.get_model_override(scope_id) == "claude-sonnet-4-6"

        await restarted.set_model_override(scope_id, None)
        assert await restarted.get_model_override(scope_id) is None
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_chat_ai_limit_override_is_scoped_persistent_and_resettable(tmp_path):
    path = tmp_path / "ai.db"
    first_scope = "telegram:chat:-1001"
    second_scope = "telegram:chat:-1002"
    store = await AIStateRepository(path).connect()
    try:
        assert await store.get_ai_cooldown_override(first_scope) is None

        await store.set_ai_cooldown_override(first_scope, 60)

        assert await store.get_ai_cooldown_override(first_scope) == 60
        assert await store.get_ai_cooldown_override(second_scope) is None
    finally:
        await store.close()

    restarted = await AIStateRepository(path).connect()
    try:
        assert await restarted.get_ai_cooldown_override(first_scope) == 60

        await restarted.set_ai_cooldown_override(first_scope, None)

        assert await restarted.get_ai_cooldown_override(first_scope) is None
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_ai_usage_migration_discards_unscoped_cooldown(tmp_path):
    path = tmp_path / "ai.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE ai_usage (
            actor_id TEXT PRIMARY KEY,
            last_request_at REAL NOT NULL
        );
        INSERT INTO ai_usage VALUES ('telegram:user:20', 123);
        """
    )
    connection.commit()
    connection.close()

    first_scope = "telegram:chat:-1001"
    second_scope = "telegram:chat:-1002"
    actor = "telegram:user:20"
    store = await AIStateRepository(path).connect()
    try:
        assert await store.get_last_request_at(first_scope, actor) is None

        await store.set_last_request_at(first_scope, actor, 456)

        assert await store.get_last_request_at(first_scope, actor) == 456
        assert await store.get_last_request_at(second_scope, actor) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_chat_access_mode_is_restricted_by_default_and_persists(tmp_path):
    path = tmp_path / "ai.db"
    telegram_scope = "telegram:chat:-1001"
    qq_scope = "qq:group:1001"
    store = await AIStateRepository(path).connect()
    try:
        assert await store.is_chat_access_open(telegram_scope) is False

        await store.set_chat_access_open(telegram_scope, True)
        assert await store.is_chat_access_open(telegram_scope) is True
        assert await store.is_chat_access_open(qq_scope) is False
    finally:
        await store.close()

    restarted = await AIStateRepository(path).connect()
    try:
        assert await restarted.is_chat_access_open(telegram_scope) is True

        await restarted.set_chat_access_open(telegram_scope, False)
        assert await restarted.is_chat_access_open(telegram_scope) is False
    finally:
        await restarted.close()


class FakeGateway:
    def __init__(self):
        self.requests: list[AgentRunRequest] = []

    async def run(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        self.requests.append(request)
        yield AgentEvent(
            type="run_started",
            run_id=request.run_id,
            session_id="session-1",
        )
        yield AgentEvent(type="text_delta", delta="partial", reset=True)
        yield AgentEvent(
            type="run_completed",
            session_id="session-1",
            entry_id="entry-1",
            answer="final",
        )

    async def cancel(self, run_id: str) -> bool:
        return True


class FakeSentMessage:
    def __init__(self):
        self.id = 100
        self.text = "Thinking..."


class FakeTransport:
    def __init__(self):
        self.sent = FakeSentMessage()
        self.updates: list[tuple[str, ChatPresentation, bool]] = []
        self.replies: list[tuple[object, str, ChatPresentation]] = []
        self.reply_targets: dict[object, object] = {}
        self.deleted: list[object] = []

    async def get_reply(self, message):
        return self.reply_targets.get(message)

    async def reply(self, message, text, *, presentation):
        self.replies.append((message, text, presentation))
        self.sent.text = text
        return self.sent

    async def update(self, message, text, *, presentation, wait):
        self.updates.append((text, presentation, wait))
        self.sent.text = text
        return True

    async def delete(self, message):
        self.deleted.append(message)

    def is_outgoing(self, message):
        return False


@pytest.mark.asyncio
async def test_responder_streams_through_transport_not_message_sdk_methods():
    transport = FakeTransport()
    responder = AIResponder(FakeGateway(), transport=transport)
    trigger = object()
    request = AgentRunRequest(
        run_id="run-1",
        session_id=None,
        parent_entry_id=None,
        prompt="question",
        context=(),
        system_prompt="system",
        tool_policy="owner",
        identity=AgentRequestIdentity(
            requester=AgentIdentityAnchor("test:user:1", "Tester"),
            anchors=(AgentIdentityAnchor("test:user:1", "Tester"),),
        ),
        origin=AgentRunOrigin("test:chat:1", "test-adapter"),
    )

    result = await responder.answer(trigger, request)

    assert result.succeeded is True
    assert result.message is transport.sent
    assert transport.updates[-1] == ("final", "agent", True)


class MinimalMessage:
    def __init__(
        self,
        text: str,
        *,
        message_id: int = 1,
        chat_id: int = 7,
        sender_id: int = 42,
        reply_to_message_id: int | None = None,
    ):
        self.id = message_id
        self.chat_id = chat_id
        self.sender_id = sender_id
        self.raw_text = text
        self.reply_to_msg_id = reply_to_message_id
        self.date = None


class FakeStore:
    def __init__(self):
        self.saved = []

    async def get_answer(self, scope_id, answer_message_id):
        return None

    async def get_turn_for_message(self, scope_id, message_id):
        return None

    async def save_answer(self, marker):
        self.saved.append(marker)

    async def get_model_override(self, scope_id):
        return None

    async def set_model_override(self, scope_id, model):
        return None

    async def get_ai_cooldown_override(self, scope_id):
        return None

    async def set_ai_cooldown_override(self, scope_id, cooldown_seconds):
        return None

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

    async def mark_memory_excluded_message(self, scope_id, message_id, kind):
        return None

    async def get_memory_scope_state(self, scope_id):
        return MemoryScopeState(
            scope_id=scope_id,
            continuous_enabled=True,
        )


@pytest.mark.asyncio
async def test_handler_uses_transport_for_sdk_operations():
    transport = FakeTransport()
    gateway = FakeGateway()
    handler = AIConversationHandler(
        owner_id=42,
        responder=AIResponder(gateway, transport=transport),
        store=FakeStore(),
        prompt_builder=PromptBuilder(transport=transport),
        transport=transport,
    )
    message = MinimalMessage("/ai question")

    handled = await handler.handle(message)

    assert handled is True
    assert transport.replies[0] == (message, "Thinking...", "plain")
    assert transport.updates[-1] == ("final", "agent", True)


class OpaqueAttachmentDescriber:
    def has_attachment(self, message):
        return True

    async def describe(self, message):
        return AttachmentDescription(
            context_text="Generated image description",
            memory_text="The subject shared an image.",
        )


class RecordingQuotedAttachmentDescriber(OpaqueAttachmentDescriber):
    def __init__(self):
        self.described: list[object] = []

    async def describe(self, message):
        self.described.append(message.id)
        return await super().describe(message)


@pytest.mark.asyncio
async def test_attachment_detection_does_not_require_telegram_file_attributes():
    transport = FakeTransport()
    gateway = FakeGateway()
    handler = AIConversationHandler(
        owner_id=42,
        responder=AIResponder(gateway, transport=transport),
        store=FakeStore(),
        prompt_builder=PromptBuilder(
            transport=transport,
            attachment_describer=OpaqueAttachmentDescriber(),
        ),
        transport=transport,
    )
    message = MinimalMessage("/ai")

    handled = await handler.handle(message)

    assert handled is True
    assert transport.updates[-1] == ("final", "agent", True)


@pytest.mark.asyncio
async def test_quote_attachment_strategy_only_describes_the_direct_reply() -> None:
    transport = FakeTransport()
    quoted = MinimalMessage("", message_id=2, reply_to_message_id=None)
    ambient = MinimalMessage("", message_id=3, reply_to_message_id=None)
    trigger = MinimalMessage(
        "/ai2 explain this",
        message_id=4,
        reply_to_message_id=quoted.id,
    )
    transport.reply_targets[trigger] = quoted

    class History:
        async def fetch_recent(self, _trigger, *, before, limit):
            assert before is quoted
            assert limit == 1
            return (ambient,)

    describer = RecordingQuotedAttachmentDescriber()
    builder = PromptBuilder(
        transport=transport,
        history_source=History(),
        quoted_attachment_describer=describer,
    )

    context = await builder.load_chat_context(trigger, recent_messages=2)
    background_description = await builder.describe_attachment(quoted)

    assert describer.described == [quoted.id]
    assert background_description is None
    assert [message.message_id for message in context.messages] == [quoted.id]


@pytest.mark.asyncio
async def test_recent_history_does_not_assume_transport_message_ids_are_ordered():
    class History:
        async def fetch_recent(self, trigger, *, before, limit):
            messages = (
                MinimalMessage(
                    "first",
                    message_id=2_000_000_000,
                    chat_id=trigger.chat_id,
                    sender_id=10,
                ),
                MinimalMessage(
                    "second",
                    message_id=3,
                    chat_id=trigger.chat_id,
                    sender_id=20,
                ),
            )
            occurred_at = datetime(2026, 7, 16, 12, tzinfo=UTC)
            for message in messages:
                message.date = occurred_at
            return messages

    builder = PromptBuilder(
        history_source=History(),
        transport=FakeTransport(),
    )
    trigger = MinimalMessage(
        "/ai2 summarize",
        message_id=100,
        chat_id=7,
        sender_id=42,
    )

    context = await builder.load_chat_context(trigger, recent_messages=2)

    assert [message.content for message in context.messages] == ["first", "second"]


def test_shared_ai_module_has_no_telegram_adapter_imports():
    import sidekick.ai as ai_module

    tree = ast.parse(Path(ai_module.__file__).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(
        name == "telethon"
        or name.startswith("telethon.")
        or name == "sidekick.telegram"
        or name.startswith("sidekick.telegram.")
        for name in imported
    )


@pytest.mark.asyncio
async def test_memory_coordinates_follow_the_injected_chat_identity_codec():
    transport = FakeTransport()
    gateway = FakeGateway()
    qq = NamespacedIdentityCodec(
        source="qq",
        actor_kind="user",
        scope_kind="group",
    )
    prompt_builder = PromptBuilder(
        transport=transport,
        identity_codec=qq,
    )
    store = FakeStore()
    handler = AIConversationHandler(
        owner_id=42,
        responder=AIResponder(gateway, transport=transport),
        store=store,
        prompt_builder=prompt_builder,
        transport=transport,
        memory=object(),
        identity_codec=qq,
    )
    message = MinimalMessage("/ai who am I?")

    assert await handler.handle(message) is True

    memory = gateway.requests[0].memory
    assert memory is not None
    assert memory.primary_bank_id == "qq:group:7"
    assert [
        anchor.identity for anchor in gateway.requests[0].identity.anchors
    ] == ["qq:user:42"]
    assert store.saved[0].scope_id == "qq:group:7"
    assert store.saved[0].requester_id == "qq:user:42"


@pytest.mark.asyncio
async def test_request_identity_is_present_when_memory_is_disabled():
    transport = FakeTransport()
    gateway = FakeGateway()
    codec = NamespacedIdentityCodec(
        source="qq",
        actor_kind="user",
        scope_kind="group",
    )
    handler = AIConversationHandler(
        owner_id=42,
        responder=AIResponder(gateway, transport=transport),
        store=FakeStore(),
        prompt_builder=PromptBuilder(transport=transport, identity_codec=codec),
        transport=transport,
        identity_codec=codec,
    )

    assert await handler.handle(MinimalMessage("/ai who am I?")) is True

    request = gateway.requests[0]
    assert request.memory is None
    assert request.identity.requester.identity == "qq:user:42"
    assert [anchor.identity for anchor in request.identity.anchors] == [
        "qq:user:42"
    ]


@pytest.mark.asyncio
async def test_state_repository_migrates_legacy_telegram_identity_columns(tmp_path):
    path = tmp_path / "ai.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE ai_answers (
            chat_id INTEGER NOT NULL,
            answer_message_id INTEGER NOT NULL,
            trigger_message_id INTEGER NOT NULL,
            requester_id INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            parent_answer_message_id INTEGER,
            reference_context TEXT NOT NULL,
            agent_session_id TEXT,
            agent_entry_id TEXT,
            PRIMARY KEY (chat_id, answer_message_id)
        );
        INSERT INTO ai_answers VALUES (
            -1001, 100, 1, 20, 'question', 'answer', NULL, '', 's1', 'e1'
        );
        CREATE TABLE ai_whitelist (
            user_id INTEGER PRIMARY KEY,
            allowed_at REAL NOT NULL
        );
        INSERT INTO ai_whitelist VALUES (20, 1);
        CREATE TABLE ai_usage (
            user_id INTEGER PRIMARY KEY,
            last_request_at REAL NOT NULL
        );
        INSERT INTO ai_usage VALUES (20, 2);
        CREATE TABLE ai_memory_excluded_messages (
            chat_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (chat_id, message_id)
        );
        INSERT INTO ai_memory_excluded_messages VALUES (-1001, 101, 'ai-answer', 3);
        """
    )
    connection.commit()
    connection.close()

    store = await AIStateRepository(path).connect()
    try:
        marker = await store.get_answer("telegram:chat:-1001", 100)

        assert marker is not None
        assert marker.scope_id == "telegram:chat:-1001"
        assert marker.requester_id == "telegram:user:20"
        assert await store.is_allowed("telegram:user:20") is True
        assert await store.is_allowed("qq:user:20") is False
        assert (
            await store.get_last_request_at(
                "telegram:chat:-1001",
                "telegram:user:20",
            )
            is None
        )
        assert await store.is_memory_excluded_message(
            "telegram:chat:-1001",
            101,
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_state_repository_preserves_opaque_decimal_string_message_ids(tmp_path):
    path = tmp_path / "ai.db"
    scope_id = "wechat:chat:56825427596%40chatroom"
    answer_id = "7158246912028861544"
    trigger_id = "4159667620982040828"
    store = await AIStateRepository(path).connect()
    try:
        await store.save_answer(
            AIAnswerMarker(
                scope_id=scope_id,
                answer_message_id=answer_id,
                trigger_message_id=trigger_id,
                requester_id="wechat:user:wxid_example",
                parent_answer_message_id=None,
                agent_session_id="session-1",
                agent_entry_id="entry-1",
            )
        )
        await store.mark_memory_excluded_message(
            scope_id,
            answer_id,
            "ai-answer",
        )

        marker = await store.get_answer(scope_id, answer_id)
        answer_ids = await store.get_ai_answer_message_ids(
            scope_id,
            (answer_id,),
        )
        excluded_ids = await store.get_memory_excluded_message_ids(
            scope_id,
            (answer_id,),
        )
    finally:
        await store.close()

    assert marker is not None
    assert marker.answer_message_id == answer_id
    assert marker.trigger_message_id == trigger_id
    assert answer_ids == frozenset({answer_id})
    assert excluded_ids == frozenset({answer_id})

    with sqlite3.connect(path) as connection:
        answer_types = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info(ai_answers)")
        }
        exclusion_types = {
            row[1]: row[2]
            for row in connection.execute(
                "PRAGMA table_info(ai_memory_excluded_messages)"
            )
        }

    assert answer_types["answer_message_id"] == "BLOB"
    assert answer_types["trigger_message_id"] == "BLOB"
    assert answer_types["parent_answer_message_id"] == "BLOB"
    assert exclusion_types["message_id"] == "BLOB"


@pytest.mark.asyncio
async def test_state_repository_preserves_opaque_memory_cursors(tmp_path):
    path = tmp_path / "ai.db"
    scope_id = "wechat:chat:56825427596%40chatroom"
    cursor = "4159667620982040828"
    store = await AIStateRepository(path).connect()
    try:
        await store.set_continuous_memory_enabled(
            scope_id,
            True,
            cursor_message_id=cursor,
        )
        await store.record_memory_dream_success(
            scope_id,
            cursor_message_id=cursor,
            scanned_until_at=1_783_772_734,
            succeeded_at=1_783_772_735,
        )

        scope = await store.get_memory_scope_state(scope_id)
        dream = await store.get_memory_dream_state(scope_id)
    finally:
        await store.close()

    assert scope.continuous_cursor_message_id == cursor
    assert dream.cursor_message_id == cursor

    with sqlite3.connect(path) as connection:
        scope_types = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info(ai_memory_scopes)")
        }
        dream_types = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info(ai_memory_dream_state)")
        }

    assert scope_types["continuous_cursor_message_id"] == "BLOB"
    assert dream_types["cursor_message_id"] == "BLOB"


@pytest.mark.asyncio
async def test_state_repository_migrates_integer_memory_cursors(tmp_path):
    path = tmp_path / "ai.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE ai_memory_scopes (
                scope_id TEXT PRIMARY KEY,
                continuous_enabled INTEGER NOT NULL DEFAULT 0,
                dream_enabled INTEGER NOT NULL DEFAULT 0,
                display_name TEXT,
                continuous_cursor_message_id INTEGER,
                continuous_last_attempt_at REAL,
                continuous_last_success_at REAL,
                continuous_last_error TEXT,
                updated_at REAL NOT NULL
            );
            INSERT INTO ai_memory_scopes VALUES (
                'telegram:chat:-1001', 1, 0, 'Example', 42,
                NULL, NULL, NULL, 1
            );
            CREATE TABLE ai_memory_dream_state (
                scope_id TEXT PRIMARY KEY,
                cursor_message_id INTEGER,
                scanned_until_at REAL,
                last_attempt_at REAL,
                last_success_at REAL,
                last_error TEXT,
                lease_owner TEXT,
                lease_expires_at REAL
            );
            INSERT INTO ai_memory_dream_state VALUES (
                'telegram:chat:-1001', 41, 1, 2, 3, NULL, NULL, NULL
            );
            """
        )

    store = await AIStateRepository(path).connect()
    try:
        scope = await store.get_memory_scope_state("telegram:chat:-1001")
        dream = await store.get_memory_dream_state("telegram:chat:-1001")
    finally:
        await store.close()

    assert scope.continuous_cursor_message_id == 42
    assert dream.cursor_message_id == 41

    with sqlite3.connect(path) as connection:
        scope_type = connection.execute(
            "SELECT type FROM pragma_table_info('ai_memory_scopes') "
            "WHERE name = 'continuous_cursor_message_id'"
        ).fetchone()[0]
        dream_type = connection.execute(
            "SELECT type FROM pragma_table_info('ai_memory_dream_state') "
            "WHERE name = 'cursor_message_id'"
        ).fetchone()[0]

    assert scope_type == "BLOB"
    assert dream_type == "BLOB"


@pytest.mark.asyncio
async def test_state_repository_migrates_namespaced_integer_message_id_columns(
    tmp_path,
):
    path = tmp_path / "ai.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE ai_answers (
                scope_id TEXT NOT NULL,
                answer_message_id INTEGER NOT NULL,
                trigger_message_id INTEGER NOT NULL,
                requester_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                parent_answer_message_id INTEGER,
                reference_context TEXT NOT NULL,
                agent_session_id TEXT,
                agent_entry_id TEXT,
                PRIMARY KEY (scope_id, answer_message_id)
            );
            INSERT INTO ai_answers VALUES (
                'telegram:chat:-1001', 100, 1, 'telegram:user:20',
                'question', 'answer', NULL, '', 's1', 'e1'
            );
            CREATE TABLE ai_memory_excluded_messages (
                scope_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (scope_id, message_id)
            );
            INSERT INTO ai_memory_excluded_messages VALUES (
                'telegram:chat:-1001', 101, 'ai-answer', 3
            );
            """
        )

    store = await AIStateRepository(path).connect()
    try:
        marker = await store.get_answer("telegram:chat:-1001", 100)
        excluded = await store.is_memory_excluded_message(
            "telegram:chat:-1001",
            101,
        )
    finally:
        await store.close()

    assert marker is not None
    assert marker.answer_message_id == 100
    assert excluded is True

    with sqlite3.connect(path) as connection:
        answer_types = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info(ai_answers)")
        }
        exclusion_types = {
            row[1]: row[2]
            for row in connection.execute(
                "PRAGMA table_info(ai_memory_excluded_messages)"
            )
        }

    assert answer_types["answer_message_id"] == "BLOB"
    assert exclusion_types["message_id"] == "BLOB"
