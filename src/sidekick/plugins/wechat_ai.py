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
from sidekick.chat.formatting import agent_system_prompt
from sidekick.plugins.base import PluginMount
from sidekick.runtime import build_logger
from sidekick.wechat.ai import (
    WECHAT_IDENTITY_CODEC,
    WeChatChatTransport,
    WeChatHistorySource,
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
        connector_url = os.environ.get(
            "SIDEKICK_WECHAT_URL",
            "http://127.0.0.1:18188",
        ).strip().rstrip("/")
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
                    bootstrap = await bootstrap_wechat_channel(
                        self._client,
                        self._wechat_store,
                        self._client.base_url,
                    )
                    handler = self._build_handler(bootstrap)
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
            await self._client.close()
            await self._gateway.close()
            await self._ai_store.close()
            await self._wechat_store.close()

    def _build_handler(
        self,
        bootstrap: WeChatBootstrap,
    ) -> AIConversationHandler:
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
            identity_resolver=WeChatMessageIdentityResolver(),
            mention_resolver=WeChatMessageMentionResolver(),
            history_source=history,
            transport=transport,
            identity_codec=WECHAT_IDENTITY_CODEC,
        )
        return AIConversationHandler(
            owner_id=bootstrap.session.self_id,
            responder=responder,
            store=self._ai_store,
            prompt_builder=prompt_builder,
            rate_limiter=AIRateLimiter(
                self._ai_store,
                cooldown_seconds=self._settings.delegated_cooldown,
            ),
            memory=None,
            dream_runner=None,
            memory_scope_resolver=None,
            directory_source_resolver=None,
            memory_command_delete_delay=(
                self._settings.memory_command_delete_delay
            ),
            transport=transport,
            identity_codec=WECHAT_IDENTITY_CODEC,
            logger=self.logger,
        )


async def _wait_or_stop(stop: asyncio.Event, delay: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except TimeoutError:
        pass
