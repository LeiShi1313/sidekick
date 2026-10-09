from __future__ import annotations

from types import SimpleNamespace

import pytest
from telethon.tl import types as telegram_types

from sidekick.ai import PromptBuilder
from sidekick.chat.replies import is_cross_chat_reply, same_chat_reply_id
from sidekick.telegram.ai_transport import TelegramChatTransport


def _message(*, chat_id=-1001627702549, reply_to_msg_id=5708, **header):
    return SimpleNamespace(
        chat_id=chat_id,
        reply_to_msg_id=reply_to_msg_id,
        reply_to=telegram_types.MessageReplyHeader(
            reply_to_msg_id=reply_to_msg_id,
            **header,
        ),
    )


def test_same_chat_replies_keep_their_id() -> None:
    message = _message()
    assert not is_cross_chat_reply(message)
    assert same_chat_reply_id(message) == 5708


def test_reply_header_naming_the_same_chat_is_not_cross_chat() -> None:
    message = _message(
        reply_to_peer_id=telegram_types.PeerChannel(channel_id=1627702549),
    )
    assert same_chat_reply_id(message) == 5708


def test_replies_into_another_chat_have_no_same_chat_id() -> None:
    message = _message(
        reply_to_peer_id=telegram_types.PeerChannel(channel_id=1234567890),
    )
    assert is_cross_chat_reply(message)
    assert same_chat_reply_id(message) is None


def test_external_reply_attribution_marks_a_cross_chat_reply() -> None:
    message = _message(
        reply_from=telegram_types.MessageFwdHeader(
            date=None,
            from_name="纯水精灵",
        ),
    )
    assert is_cross_chat_reply(message)
    assert same_chat_reply_id(message) is None


def test_messages_without_reply_headers_use_their_reply_id() -> None:
    plain = SimpleNamespace(chat_id="wxid", reply_to_msg_id="m1")
    assert same_chat_reply_id(plain) == "m1"
    assert same_chat_reply_id(SimpleNamespace(chat_id=1)) is None


class _ExternalClient:
    def __init__(self, text=None, error=None):
        self.calls = []
        self.text = text
        self.error = error

    async def get_messages(self, peer, ids):
        self.calls.append((peer, ids))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(raw_text=self.text)


def _external_trigger(*, quote_text=None, client=None):
    trigger = SimpleNamespace(
        id=1016612,
        chat_id=-1001627702549,
        raw_text="/ai 这条频道消息说的是什么",
        sender_id=465536798,
        date=None,
        reply_to_msg_id=5708,
        reply_to=telegram_types.MessageReplyHeader(
            reply_to_msg_id=5708,
            reply_to_peer_id=telegram_types.PeerChannel(channel_id=987654321),
            quote_text=quote_text,
        ),
        client=client,
        local_reply_lookups=0,
    )

    async def get_reply_message():
        trigger.local_reply_lookups += 1
        return SimpleNamespace(id=5708, chat_id=trigger.chat_id, raw_text="2022")

    trigger.get_reply_message = get_reply_message
    return trigger


@pytest.mark.asyncio
async def test_telegram_never_resolves_a_cross_chat_reply_in_this_chat() -> None:
    transport = TelegramChatTransport(edit_cadence=0)
    trigger = _external_trigger()

    assert await transport.get_reply(trigger) is None
    assert trigger.local_reply_lookups == 0


@pytest.mark.asyncio
async def test_external_reply_quote_reaches_the_model_context() -> None:
    client = _ExternalClient(text="should not be fetched")
    builder = PromptBuilder(transport=TelegramChatTransport(edit_cadence=0))
    trigger = _external_trigger(quote_text="频道里的原话", client=client)

    context = await builder.load_chat_context(trigger)
    rendered = builder.render_chat_context(context)

    assert "another chat" in rendered
    assert "频道里的原话" in rendered
    assert "2022" not in rendered
    assert client.calls == []


@pytest.mark.asyncio
async def test_external_reply_without_quote_fetches_the_source_post() -> None:
    client = _ExternalClient(text="纯水精灵频道的帖子正文")
    builder = PromptBuilder(transport=TelegramChatTransport(edit_cadence=0))
    trigger = _external_trigger(client=client)

    rendered = builder.render_chat_context(await builder.load_chat_context(trigger))

    assert "纯水精灵频道的帖子正文" in rendered
    assert client.calls == [(telegram_types.PeerChannel(channel_id=987654321), 5708)]


@pytest.mark.asyncio
async def test_unreadable_external_source_still_marks_the_reply() -> None:
    builder = PromptBuilder(transport=TelegramChatTransport(edit_cadence=0))
    trigger = _external_trigger(client=_ExternalClient(error=ValueError("private")))

    rendered = builder.render_chat_context(await builder.load_chat_context(trigger))

    assert "another chat" in rendered
