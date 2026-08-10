from __future__ import annotations

import asyncio
import json
import os
import signal
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from sidekick.ai import (
    AIConversationHandler,
    AIRateLimiter,
    AIResponder,
    AISettings,
    AIStateRepository,
    PiAgentGateway,
    PromptBuilder,
)
from sidekick.ai_attachments import ChatAttachmentDescriber
from sidekick.ai_continuous_memory import (
    ContinuousMemoryScheduler,
    ContinuousMemorySchedulerSettings,
)
from sidekick.ai_dream import (
    DreamScheduler,
    DreamSchedulerSettings,
    DreamSettings,
)
from sidekick.ai_memory import HindsightMemoryClient
from sidekick.ai_memory_ingestion import (
    ChatMemoryIngestor,
    MemoryIngestionSettings,
)
from sidekick.ai_memory_outbox import (
    MemoryOutboxScheduler,
    MemoryOutboxSchedulerSettings,
)
from sidekick.chat.formatting import agent_system_prompt
from sidekick.chat.output_policy import MainlandMessagingOutputPolicy
from sidekick.channel_status import (
    AdapterRuntimeState,
    ChannelOpsServer,
    ChannelOpsSettings,
    ChannelSnapshotService,
)
from sidekick.memory_admin import MemoryAdminService
from sidekick.onebot.ai import (
    QQ_IDENTITY_CODEC,
    OneBotChatTransport,
    OneBotDirectory,
    OneBotDirectorySourceResolver,
    OneBotHistorySource,
    OneBotMemoryScopeTargetResolver,
    OneBotMessageIdentityResolver,
    OneBotMessageMentionResolver,
    onebot_memory_event_metadata,
    onebot_source_retry_delay,
)
from sidekick.onebot.client import OneBotReverseWebSocket
from sidekick.onebot.memory_admin import (
    OneBotMemoryAdminClient,
    mount_onebot_memory_admin,
)
from sidekick.onebot.message import OneBotMessage, OneBotMessageError
from sidekick.plugins.base import PluginMount
from sidekick.runtime import build_logger


@dataclass(frozen=True, slots=True)
class OneBotRuntimeSettings:
    host: str
    port: int
    token: str
    self_id: int

    @classmethod
    def from_env(cls) -> OneBotRuntimeSettings:
        token = os.environ.get("SIDEKICK_ONEBOT_TOKEN", "").strip()
        self_id = _positive_int(os.environ.get("SIDEKICK_ONEBOT_SELF_ID", ""))
        if not token or self_id is None:
            raise ValueError(
                "Missing OneBot configuration: SIDEKICK_ONEBOT_TOKEN and "
                "SIDEKICK_ONEBOT_SELF_ID are required"
            )
        port = _positive_int(os.environ.get("SIDEKICK_ONEBOT_PORT", "8766"))
        if port is None or port > 65_535:
            raise ValueError("SIDEKICK_ONEBOT_PORT must be between 1 and 65535")
        return cls(
            host=os.environ.get("SIDEKICK_ONEBOT_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=port,
            token=token,
            self_id=self_id,
        )


class OneBotAI(metaclass=PluginMount):
    command_group = "onebot"
    command_name = "ai"

    def __init__(self, log_level: str = "info"):
        self._runtime = OneBotRuntimeSettings.from_env()
        self._settings = AISettings.from_env()
        self._ops_settings = ChannelOpsSettings.from_env(
            default_instance_id="qq-default",
        )
        self.logger = build_logger(__name__, log_level=log_level)
        self._gateway = PiAgentGateway(
            self._settings.agent_url,
            token=self._settings.agent_token,
            timeout=self._settings.request_timeout,
        )
        self._store = AIStateRepository(self._settings.state_path)
        self._memory = (
            HindsightMemoryClient(
                self._settings.hindsight_url,
                token=self._settings.hindsight_token or "",
                timeout=self._settings.hindsight_timeout,
            )
            if self._settings.hindsight_url
            else None
        )
        self._bridge = OneBotReverseWebSocket(
            token=self._runtime.token,
            self_id=self._runtime.self_id,
            logger=self.logger,
        )
        self._directory = OneBotDirectory()
        self._handler: AIConversationHandler | None = None
        self._dream_scheduler: DreamScheduler | None = None
        self._continuous_scheduler: ContinuousMemoryScheduler | None = None
        self._memory_outbox_scheduler: MemoryOutboxScheduler | None = None
        self._seen_messages: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._adapter_status = AdapterRuntimeState(
            id=self._ops_settings.instance_id,
            platform="qq",
            account_id=str(self._runtime.self_id),
            connected_probe=lambda: self._bridge.connected,
        )
        self._ops_server = ChannelOpsServer(
            snapshot_service=ChannelSnapshotService(
                state_reader=self._store,
                inventory_loader=self._directory.list_channels,
                adapter=self._adapter_status,
                memory_available=self._memory is not None,
                logger=self.logger,
            ),
            settings=self._ops_settings,
            logger=self.logger,
        )

    def __call__(self) -> None:
        """Run the Pi-powered OneBot 11/NapCat userbot."""
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            pass

    async def _run(self) -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(stop_signal, stop.set)
            except NotImplementedError:
                pass
        try:
            await self._setup()
            await self._bridge.start(self._runtime.host, self._runtime.port)
            await self._ops_server.start()
            self.logger.info("OneBot AI listening")
            await self._bridge.wait_connected()
            await self._verify_account()
            self._adapter_status.update(connected=True)
            await self._refresh_directory()
            if self._continuous_scheduler is not None:
                self._continuous_scheduler.start()
            if self._dream_scheduler is not None:
                self._dream_scheduler.start()
            if self._memory_outbox_scheduler is not None:
                self._memory_outbox_scheduler.start()
            self.logger.info("OneBot AI connected")
            await stop.wait()
        finally:
            self._adapter_status.update(connected=False)
            await self._ops_server.close()
            if self._continuous_scheduler is not None:
                await self._continuous_scheduler.close()
            if self._dream_scheduler is not None:
                await self._dream_scheduler.close()
            if self._memory_outbox_scheduler is not None:
                await self._memory_outbox_scheduler.close()
            await self._bridge.close()
            if self._memory is not None:
                await self._memory.close()
            await self._gateway.close()
            await self._store.close()

    async def _setup(self) -> None:
        output_policy = MainlandMessagingOutputPolicy.from_env()
        transport = OneBotChatTransport(self._bridge, logger=self.logger)
        responder = AIResponder(
            self._gateway,
            max_output_chars=self._settings.max_output_chars,
            transport=transport,
            output_policy=output_policy,
            logger=self.logger,
        )
        await self._store.connect()
        history_source = OneBotHistorySource(
            self._bridge,
            directory=self._directory,
        )
        prompt_builder = PromptBuilder(
            system_prompt=output_policy.apply_to_system_prompt(
                agent_system_prompt(self._settings.system_prompt)
            ),
            max_context_messages=self._settings.max_context_messages,
            max_context_chars=self._settings.max_context_chars,
            attachment_describer=ChatAttachmentDescriber(
                self._gateway,
                allow_unknown_size=True,
                logger=self.logger,
            ),
            identity_resolver=OneBotMessageIdentityResolver(),
            mention_resolver=OneBotMessageMentionResolver(self._directory),
            history_source=history_source,
            transport=transport,
            identity_codec=QQ_IDENTITY_CODEC,
            metadata_resolver=onebot_memory_event_metadata,
        )
        memory_ingestor = (
            ChatMemoryIngestor(
                source=history_source,
                store=self._store,
                memory=self._memory,
                prompt_builder=prompt_builder,
                dream_settings=DreamSettings.from_env(),
                ingestion_settings=MemoryIngestionSettings.from_env(),
                identity_codec=QQ_IDENTITY_CODEC,
                source_retry_delay=onebot_source_retry_delay,
                logger=self.logger,
            )
            if self._memory is not None
            else None
        )
        if memory_ingestor is not None:
            mount_onebot_memory_admin(
                self._bridge,
                MemoryAdminService(
                    store=self._store,
                    dream_runner=memory_ingestor,
                    identity_codec=QQ_IDENTITY_CODEC,
                ),
                display_name_resolver=self._directory.scope_name,
            )
            self._dream_scheduler = DreamScheduler(
                scanner=memory_ingestor,
                store=self._store,
                identity_codec=QQ_IDENTITY_CODEC,
                settings=DreamSchedulerSettings.from_env(),
                logger=self.logger,
            )
            self._continuous_scheduler = ContinuousMemoryScheduler(
                runner=memory_ingestor,
                store=self._store,
                identity_codec=QQ_IDENTITY_CODEC,
                settings=ContinuousMemorySchedulerSettings.from_env(),
                logger=self.logger,
            )
            self._memory_outbox_scheduler = MemoryOutboxScheduler(
                runner=memory_ingestor,
                store=self._store,
                settings=MemoryOutboxSchedulerSettings.from_env(),
                logger=self.logger,
            )
        self._handler = AIConversationHandler(
            owner_id=self._runtime.self_id,
            responder=responder,
            store=self._store,
            prompt_builder=prompt_builder,
            rate_limiter=AIRateLimiter(
                self._store,
                cooldown_seconds=self._settings.delegated_cooldown,
            ),
            memory=self._memory,
            dream_runner=memory_ingestor,
            memory_scope_resolver=OneBotMemoryScopeTargetResolver(self._bridge),
            directory_source_resolver=OneBotDirectorySourceResolver(self._bridge),
            memory_command_delete_delay=(self._settings.memory_command_delete_delay),
            transport=transport,
            identity_codec=QQ_IDENTITY_CODEC,
            run_store=self._store,
            adapter_instance_id=self._ops_settings.instance_id,
            logger=self.logger,
        )
        self._bridge.set_event_handler(self._on_event)

    async def _verify_account(self) -> None:
        info = await self._bridge.call("get_login_info", {}, timeout=30)
        user_id = _positive_int(info.get("user_id")) if isinstance(info, dict) else None
        if user_id != self._runtime.self_id:
            raise RuntimeError("NapCat connected with an unexpected QQ account")

    async def _refresh_directory(self) -> None:
        try:
            await self._directory.refresh(self._bridge)
        except Exception as exc:
            self.logger.warning(
                "OneBot directory refresh failed (%s)",
                type(exc).__name__,
            )

    async def _on_event(self, payload: dict[str, Any]) -> None:
        if payload.get("post_type") not in {"message", "message_sent"}:
            return
        try:
            message = OneBotMessage.from_payload(
                payload,
                action_client=self._bridge,
            )
        except OneBotMessageError:
            self.logger.warning("Ignoring malformed OneBot message")
            return
        if message.self_id != self._runtime.self_id:
            self.logger.warning("Ignoring OneBot message for another account")
            return
        if message.scope_display_name is None:
            message.scope_display_name = self._directory.scope_name(message.chat_id)
        key = (message.chat_id, message.id)
        if key in self._seen_messages:
            return
        self._seen_messages[key] = None
        self._seen_messages.move_to_end(key)
        while len(self._seen_messages) > 4_096:
            self._seen_messages.popitem(last=False)
        if self._handler is None:
            return
        try:
            await self._handler.handle(message)
        except Exception as exc:
            self.logger.error(
                "OneBot AI message handling failed (%s)",
                type(exc).__name__,
            )


class _OneBotMemoryAdminCommand:
    def __init__(self, admin_url: str = "", timeout: float = 900) -> None:
        self._admin_url = admin_url
        self._timeout = timeout

    def _client(self) -> OneBotMemoryAdminClient:
        return _onebot_memory_admin_client(
            admin_url=self._admin_url,
            timeout=self._timeout,
        )

    @staticmethod
    def _print(result: dict[str, Any]) -> None:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


class OneBotMemoryDreamEnable(
    _OneBotMemoryAdminCommand,
    metaclass=PluginMount,
):
    command_group = "onebot.memory"
    command_name = "dream-enable"

    def __call__(self, group_id: int, display_name: str = "") -> None:
        """Quietly enable hourly Dream memory for a QQ group."""
        self._print(
            self._client().set_dream(
                group_id,
                enabled=True,
                display_name=display_name.strip() or None,
            )
        )


class OneBotMemoryDreamDisable(
    _OneBotMemoryAdminCommand,
    metaclass=PluginMount,
):
    command_group = "onebot.memory"
    command_name = "dream-disable"

    def __call__(self, group_id: int) -> None:
        """Quietly disable hourly Dream memory for a QQ group."""
        self._print(
            self._client().set_dream(
                group_id,
                enabled=False,
            )
        )


class OneBotMemoryStatus(
    _OneBotMemoryAdminCommand,
    metaclass=PluginMount,
):
    command_group = "onebot.memory"
    command_name = "status"

    def __call__(self, group_id: int) -> None:
        """Show the quiet memory state for a QQ group."""
        self._print(self._client().status(group_id))


class OneBotMemoryBackfill(
    _OneBotMemoryAdminCommand,
    metaclass=PluginMount,
):
    command_group = "onebot.memory"
    command_name = "backfill"

    def __call__(
        self,
        group_id: int,
        value: int,
        mode: str = "messages",
    ) -> None:
        """Quietly backfill a bounded message count or day window."""
        self._print(
            self._client().backfill(
                group_id,
                mode=mode,
                value=value,
            )
        )


def _onebot_memory_admin_client(
    *,
    admin_url: str = "",
    timeout: float = 900,
) -> OneBotMemoryAdminClient:
    runtime = OneBotRuntimeSettings.from_env()
    resolved_url = admin_url.strip() or os.environ.get(
        "SIDEKICK_ONEBOT_ADMIN_URL",
        "",
    ).strip()
    if not resolved_url:
        publish_port = _positive_int(
            os.environ.get("SIDEKICK_ONEBOT_PUBLISH_PORT", "18867")
        )
        if publish_port is None or publish_port > 65_535:
            raise ValueError(
                "SIDEKICK_ONEBOT_PUBLISH_PORT must be between 1 and 65535"
            )
        publish_host = os.environ.get(
            "SIDEKICK_ONEBOT_PUBLISH_HOST",
            "127.0.0.1",
        ).strip() or "127.0.0.1"
        if publish_host == "0.0.0.0":
            publish_host = "127.0.0.1"
        resolved_url = f"http://{publish_host}:{publish_port}"
    return OneBotMemoryAdminClient(
        resolved_url,
        token=runtime.token,
        self_id=runtime.self_id,
        timeout=timeout,
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None
