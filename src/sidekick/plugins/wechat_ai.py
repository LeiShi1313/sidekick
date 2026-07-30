from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import signal

from sidekick.ai import (
    AIConversationHandler,
    AIRateLimiter,
    AIResponder,
    AISettings,
    AIStateRepository,
    PiAgentGateway,
    PromptBuilder,
)
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
from sidekick.chat.identity import IdentityCodec
from sidekick.chat.formatting import agent_system_prompt
from sidekick.plugins.base import PluginMount
from sidekick.runtime import build_logger
from sidekick.wechat.ai import (
    WeChatChatTransport,
    WeChatHistorySource,
    WeChatIdentityCodec,
    WeChatMemoryScopeTargetResolver,
    WeChatMessageIdentityResolver,
    WeChatMessageMentionResolver,
)
from sidekick.wechat.api import WeChatConnectorClient
from sidekick.wechat.service import (
    WeChatBootstrap,
    WeChatEventPump,
    bootstrap_wechat_channel,
)
from sidekick.wechat.store import WeChatStateRepository


@dataclass(frozen=True, slots=True)
class WeChatRuntimeSettings:
    connector_url: str
    token: str
    state_path: Path
    reconnect_delay: float

    @classmethod
    def from_env(cls) -> WeChatRuntimeSettings:
        connector_url = (
            os.environ.get(
                "SIDEKICK_WECHAT_URL",
                "http://127.0.0.1:18188",
            )
            .strip()
            .rstrip("/")
        )
        if not connector_url:
            raise ValueError("SIDEKICK_WECHAT_URL cannot be empty")
        try:
            reconnect_delay = float(
                os.environ.get("SIDEKICK_WECHAT_RECONNECT_DELAY", "2")
            )
        except ValueError as exc:
            raise ValueError(
                "SIDEKICK_WECHAT_RECONNECT_DELAY must be a number"
            ) from exc
        if reconnect_delay < 0.1:
            raise ValueError(
                "SIDEKICK_WECHAT_RECONNECT_DELAY must be at least 0.1 seconds"
            )
        return cls(
            connector_url=connector_url,
            token=os.environ.get("SIDEKICK_WECHAT_TOKEN", "").strip(),
            state_path=Path(
                os.environ.get(
                    "SIDEKICK_WECHAT_STATE_PATH",
                    Path.home() / ".sidekick" / "wechat.db",
                )
            ).expanduser(),
            reconnect_delay=reconnect_delay,
        )


@dataclass(frozen=True, slots=True)
class _WeChatChannelRuntime:
    handler: AIConversationHandler
    identity_codec: IdentityCodec
    memory_ingestor: ChatMemoryIngestor | None
    dream_scheduler: DreamScheduler | None
    continuous_scheduler: ContinuousMemoryScheduler | None


class WeChatAI(metaclass=PluginMount):
    command_group = "wechat"
    command_name = "ai"

    def __init__(self, log_level: str = "info"):
        self._runtime = WeChatRuntimeSettings.from_env()
        self._settings = AISettings.from_env()
        self.logger = build_logger(__name__, log_level=log_level)
        self._client = WeChatConnectorClient(
            self._runtime.connector_url,
            token=self._runtime.token,
        )
        self._wechat_store = WeChatStateRepository(self._runtime.state_path)
        self._ai_store = AIStateRepository(self._settings.state_path)
        self._gateway = PiAgentGateway(
            self._settings.agent_url,
            token=self._settings.agent_token,
            timeout=self._settings.request_timeout,
        )
        self._memory = (
            HindsightMemoryClient(
                self._settings.hindsight_url,
                timeout=self._settings.hindsight_timeout,
            )
            if self._settings.hindsight_url
            else None
        )
        self._channel_runtime: _WeChatChannelRuntime | None = None

    def __call__(self) -> None:
        """Run the Pi-powered WeChat Linux connector adapter."""
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
        await self._wechat_store.connect()
        await self._ai_store.connect()
        try:
            while not stop.is_set():
                try:
                    # Stop account-scoped ingestion before bootstrap can switch the
                    # active connector account in the local projection.
                    await self._close_channel_runtime()
                    bootstrap = await bootstrap_wechat_channel(
                        self._client,
                        self._wechat_store,
                        self._client.base_url,
                    )
                    handler = await self._activate_channel_runtime(bootstrap)
                    self.logger.info(
                        "WeChat AI connected (self_id=%s, generation=%s)",
                        bootstrap.session.self_id,
                        bootstrap.session.connection_generation,
                    )
                    result = await WeChatEventPump(
                        self._client,
                        self._wechat_store,
                        self._client.base_url,
                        bootstrap,
                    ).run(handler, stop)
                    if result == "stopped":
                        break
                    self.logger.warning(
                        "WeChat event stream ended (%s); reconnecting",
                        result,
                    )
                except Exception as exc:
                    self.logger.exception(
                        "WeChat adapter cycle failed (%s): %s",
                        type(exc).__name__,
                        exc,
                    )
                await _wait_or_stop(stop, self._runtime.reconnect_delay)
        finally:
            await self._close_channel_runtime()
            await self._client.close()
            if self._memory is not None:
                await self._memory.close()
            await self._gateway.close()
            await self._ai_store.close()
            await self._wechat_store.close()

    def _build_channel_runtime(
        self,
        bootstrap: WeChatBootstrap,
    ) -> _WeChatChannelRuntime:
        identity_codec = WeChatIdentityCodec(
            account_id=bootstrap.session.self_id,
        )
        transport = WeChatChatTransport(
            self._client,
            self._wechat_store,
            self._client.base_url,
            logger=self.logger,
        )
        history = WeChatHistorySource(
            self._wechat_store,
            self._client.base_url,
        )
        responder = AIResponder(
            self._gateway,
            max_output_chars=self._settings.max_output_chars,
            initial_status=None,
            transport=transport,
            logger=self.logger,
        )
        prompt_builder = PromptBuilder(
            system_prompt=agent_system_prompt(self._settings.system_prompt),
            max_context_messages=self._settings.max_context_messages,
            max_context_chars=self._settings.max_context_chars,
            identity_resolver=WeChatMessageIdentityResolver(identity_codec),
            mention_resolver=WeChatMessageMentionResolver(),
            history_source=history,
            transport=transport,
            identity_codec=identity_codec,
        )
        memory_ingestor = (
            ChatMemoryIngestor(
                source=history,
                store=self._ai_store,
                memory=self._memory,
                prompt_builder=prompt_builder,
                dream_settings=DreamSettings.from_env(),
                ingestion_settings=MemoryIngestionSettings.from_env(),
                identity_codec=identity_codec,
                logger=self.logger,
            )
            if self._memory is not None
            else None
        )
        dream_scheduler = (
            DreamScheduler(
                scanner=memory_ingestor,
                store=self._ai_store,
                identity_codec=identity_codec,
                settings=DreamSchedulerSettings.from_env(),
                logger=self.logger,
            )
            if memory_ingestor is not None
            else None
        )
        continuous_scheduler = (
            ContinuousMemoryScheduler(
                runner=memory_ingestor,
                store=self._ai_store,
                identity_codec=identity_codec,
                settings=ContinuousMemorySchedulerSettings.from_env(),
                logger=self.logger,
            )
            if memory_ingestor is not None
            else None
        )
        handler = AIConversationHandler(
            owner_id=bootstrap.session.self_id,
            responder=responder,
            store=self._ai_store,
            prompt_builder=prompt_builder,
            rate_limiter=AIRateLimiter(
                self._ai_store,
                cooldown_seconds=self._settings.delegated_cooldown,
            ),
            memory=self._memory,
            dream_runner=memory_ingestor,
            memory_scope_resolver=WeChatMemoryScopeTargetResolver(
                self._wechat_store,
                self._client.base_url,
            ),
            directory_source_resolver=None,
            memory_backfill_caveat=(
                "WeChat backfill covers only messages already observed and stored "
                "by Sidekick; WeChat does not expose complete chat history here."
            ),
            memory_command_delete_delay=(self._settings.memory_command_delete_delay),
            transport=transport,
            identity_codec=identity_codec,
            logger=self.logger,
        )
        return _WeChatChannelRuntime(
            handler=handler,
            identity_codec=identity_codec,
            memory_ingestor=memory_ingestor,
            dream_scheduler=dream_scheduler,
            continuous_scheduler=continuous_scheduler,
        )

    async def _activate_channel_runtime(
        self,
        bootstrap: WeChatBootstrap,
    ) -> AIConversationHandler:
        await self._close_channel_runtime()
        runtime = self._build_channel_runtime(bootstrap)
        self._channel_runtime = runtime
        if runtime.continuous_scheduler is not None:
            runtime.continuous_scheduler.start()
        if runtime.dream_scheduler is not None:
            runtime.dream_scheduler.start()
        return runtime.handler

    async def _close_channel_runtime(self) -> None:
        runtime = self._channel_runtime
        self._channel_runtime = None
        if runtime is None:
            return
        if runtime.continuous_scheduler is not None:
            await runtime.continuous_scheduler.close()
        if runtime.dream_scheduler is not None:
            await runtime.dream_scheduler.close()


async def _wait_or_stop(stop: asyncio.Event, delay: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except TimeoutError:
        pass
