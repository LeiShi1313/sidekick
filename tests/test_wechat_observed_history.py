from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from sidekick.ai import PromptBuilder
from sidekick.wechat.ai import (
    WeChatChatTransport,
    WeChatHistorySource,
    WeChatMemoryScopeTargetResolver,
)
from sidekick.wechat.api import (
    WeChatAPIError,
    WeChatChat,
    WeChatChatList,
    WeChatChatSnapshot,
    WeChatObservationCoverage,
    WeChatObservedMessage,
    WeChatObservedMessagePage,
    WeChatObservedPageInfo,
    WeChatSession,
)
from sidekick.wechat.store import WeChatStateRepository


CONNECTOR_KEY = "http://127.0.0.1:18188"
ACCOUNT_ID = "wxid_self"
CHAT_ID = "56825427596@chatroom"


def observed(
    message_id: str,
    *,
    state: str = "present",
    message_type: str = "text",
    content: str = "hello",
    sender_display_name: str | None = "Alice Global",
    sender_group_alias: str | None = None,
    reply_to_message_id: str | None = None,
) -> WeChatObservedMessage:
    common = {
        "id": message_id,
        "chat_id": CHAT_ID,
        "state": state,
        "version": f"mv1:{message_id}",
        "order_timestamp": int(message_id),
        "observed_at": int(message_id) + 100,
        "source": "wechat+localdb",
    }
    if state == "recalled":
        return WeChatObservedMessage(**common)
    return WeChatObservedMessage(
        **common,
        direction="in",
        message_type=message_type,
        content=content,
        sender_id="wxid_alice",
        sender_display_name=sender_display_name,
        sender_group_alias=sender_group_alias,
        timestamp=int(message_id),
        reply_to_message_id=reply_to_message_id,
        media_id=("0123456789abcdef0123456789abcdef" if message_type == "image" else None),
    )


def page(
    *messages: WeChatObservedMessage,
    has_more_before: bool = False,
    has_more_after: bool = False,
) -> WeChatObservedMessagePage:
    return WeChatObservedMessagePage(
        messages=messages,
        page=WeChatObservedPageInfo(
            before=messages[0].id if messages else None,
            after=messages[-1].id if messages else None,
            has_more_before=has_more_before,
            has_more_after=has_more_after,
        ),
        coverage=WeChatObservationCoverage(
            kind="partial",
            oldest_available=None,
            newest_available=None,
            may_have_unobserved_messages=True,
            observation_gaps=(),
            gaps_are_exhaustive=False,
            sources=("localdb",),
        ),
    )


class ObservedClient:
    def __init__(
        self,
        pages: tuple[WeChatObservedMessagePage, ...] = (),
        exact: dict[str, WeChatObservedMessage | Exception] | None = None,
    ):
        self.pages = list(pages)
        self.exact = exact or {}
        self.page_calls: list[dict[str, object]] = []

    async def get_observed_messages(self, chat_id: str, **kwargs):
        assert chat_id == CHAT_ID
        self.page_calls.append(kwargs)
        if not self.pages:
            raise AssertionError("No observed message page prepared")
        return self.pages.pop(0)

    async def get_observed_message(self, chat_id: str, message_id: str):
        assert chat_id == CHAT_ID
        value = self.exact.get(
            message_id,
            WeChatAPIError(404, "MESSAGE_NOT_OBSERVED", "not retained"),
        )
        if isinstance(value, Exception):
            raise value
        return value


async def open_store(path) -> WeChatStateRepository:
    store = await WeChatStateRepository(path).connect()
    await store.bootstrap(
        connector_key=CONNECTOR_KEY,
        session=WeChatSession(
            status="logged_in",
            self_id=ACCOUNT_ID,
            display_name="Sidekick",
            hook_connected=True,
            connection_generation=41,
            content_redacted=False,
            cursor="10",
        ),
        chats=WeChatChatList(
            chats=(
                WeChatChat(
                    id=CHAT_ID,
                    type="group",
                    display_name="Example group",
                ),
            ),
            snapshot=WeChatChatSnapshot(
                id="snapshot-41",
                complete=True,
                current=True,
                count=1,
                cursor="10",
                connection_generation=41,
            ),
            cursor="10",
        ),
    )
    return store


@pytest.mark.asyncio
async def test_recent_history_pages_connector_catalog_without_local_projection(
    tmp_path,
) -> None:
    client = ObservedClient(
        (
            page(
                observed("103", message_type="image"),
                observed("104", sender_display_name="Bob"),
                has_more_before=True,
            ),
            page(
                observed("101", sender_group_alias="项目阿丽"),
                observed("102", state="recalled"),
            ),
        )
    )
    store = await open_store(tmp_path / "wechat.db")
    source = WeChatHistorySource(client, store, CONNECTOR_KEY)
    trigger = SimpleNamespace(chat_id=CHAT_ID, id="105")
    try:
        messages = await source.fetch_recent(trigger, before=trigger, limit=2)

        assert [message.id for message in messages] == ["101", "104"]
        assert [message.sender_display_name for message in messages] == [
            "项目阿丽",
            "Bob",
        ]
        assert [call["before"] for call in client.page_calls] == ["105", "103"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_exact_history_treats_unobserved_and_recalled_as_absent(tmp_path) -> None:
    client = ObservedClient(
        exact={
            "101": observed("101", state="recalled"),
            "102": WeChatAPIError(
                404,
                "MESSAGE_NOT_OBSERVED",
                "not retained",
            ),
        }
    )
    store = await open_store(tmp_path / "wechat.db")
    source = WeChatHistorySource(client, store, CONNECTOR_KEY)
    try:
        assert await source.fetch_message(CHAT_ID, "101") is None
        assert await source.fetch_message(CHAT_ID, "102") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_continuous_history_requires_connector_message_id_cursor(
    tmp_path,
) -> None:
    store = await open_store(tmp_path / "wechat.db")
    source = WeChatHistorySource(ObservedClient(), store, CONNECTOR_KEY)
    try:
        with pytest.raises(ValueError, match="explicit migration"):
            await source.fetch_after(
                CHAT_ID,
                after_message_id=42,
                until=datetime.now(UTC),
                limit=10,
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_scope_target_uses_latest_connector_observation_as_cursor(
    tmp_path,
) -> None:
    client = ObservedClient((page(observed("105", state="recalled")),))
    store = await open_store(tmp_path / "wechat.db")
    resolver = WeChatMemoryScopeTargetResolver(client, store, CONNECTOR_KEY)
    try:
        target = await resolver.resolve(CHAT_ID, include_latest_message=True)

        assert target.chat_id == CHAT_ID
        assert target.display_name == "Example group"
        assert target.latest_message_id == "105"
        assert client.page_calls == [{"limit": 1}]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_reply_context_stops_at_connector_gaps_and_cycles(tmp_path) -> None:
    cycle_a = observed("101", reply_to_message_id="102")
    cycle_b = observed("102", reply_to_message_id="101")
    gap = observed("103", reply_to_message_id="999")
    client = ObservedClient(
        exact={
            cycle_a.id: cycle_a,
            cycle_b.id: cycle_b,
            gap.id: gap,
        }
    )
    store = await open_store(tmp_path / "wechat.db")
    transport = WeChatChatTransport(
        client,
        store,
        CONNECTOR_KEY,
        native_reply_ready=False,
    )
    prompt_builder = PromptBuilder(
        transport=transport,
        max_context_messages=5,
    )
    source = WeChatHistorySource(client, store, CONNECTOR_KEY)
    try:
        cycle_context = await prompt_builder.load_reply_chain(
            await source.fetch_message(CHAT_ID, cycle_a.id)
        )
        gap_context = await prompt_builder.load_reply_chain(
            await source.fetch_message(CHAT_ID, gap.id)
        )

        assert {message.message_id for message in cycle_context.messages} == {
            "101",
            "102",
        }
        assert [message.message_id for message in gap_context.messages] == ["103"]
    finally:
        await store.close()
