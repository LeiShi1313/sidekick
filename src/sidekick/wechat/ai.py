from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, unquote

from sidekick.ai import ReplyTarget
from sidekick.chat.identity import ExternalId, IdentityCodec
from sidekick.wechat.message import WeChatMessage
from sidekick.wechat.store import WeChatStateRepository


@dataclass(frozen=True, slots=True)
class WeChatIdentityCodec:
    source: str = "wechat"

    def actor_id(self, actor_id: ExternalId) -> str:
        return f"wechat:user:{_component(actor_id)}"

    def scope_id(self, scope_id: ExternalId) -> str:
        return f"wechat:chat:{_component(scope_id)}"

    def parse_scope_id(self, scope_id: str) -> ExternalId | None:
        prefix = "wechat:chat:"
        if not scope_id.startswith(prefix):
            return None
        encoded = scope_id.removeprefix(prefix)
        if not encoded:
            return None
        decoded = unquote(encoded)
        if not decoded or quote(decoded, safe="-_.~") != encoded:
            return None
        return decoded

    def message_source_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str:
        return f"wechat:message:{_component(scope_id)}:{_component(message_id)}"

    def thread_document_id(
        self,
        scope_id: ExternalId,
        root_message_id: ExternalId,
    ) -> str:
        return f"wechat:thread:{_component(scope_id)}:{_component(root_message_id)}"

    def revision_document_id(
        self,
        scope_id: ExternalId,
        message_id: ExternalId,
    ) -> str:
        return f"wechat:revision:{_component(scope_id)}:{_component(message_id)}"


WECHAT_IDENTITY_CODEC: IdentityCodec = WeChatIdentityCodec()


class WeChatHistorySource:
    def __init__(self, store: WeChatStateRepository, connector_key: str):
        self._store = store
        self._connector_key = connector_key

    async def fetch_recent(
        self,
        trigger: ReplyTarget,
        *,
        before: ReplyTarget,
        limit: int,
    ) -> tuple[WeChatMessage, ...]:
        if (
            not isinstance(trigger.chat_id, str)
            or before.chat_id != trigger.chat_id
            or not isinstance(before.id, str)
        ):
            return ()
        return await self._store.fetch_recent(
            self._connector_key,
            trigger.chat_id,
            before.id,
            limit,
        )

    async def fetch_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> WeChatMessage | None:
        return await self._store.get_message(
            self._connector_key,
            chat_id,
            message_id,
        )


def _component(value: ExternalId) -> str:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid WeChat IDs")
    normalized = str(value)
    if not normalized or normalized != normalized.strip():
        raise ValueError("WeChat IDs cannot be empty or padded")
    return quote(normalized, safe="-_.~")
