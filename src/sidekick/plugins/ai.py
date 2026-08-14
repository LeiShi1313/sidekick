import asyncio
from datetime import UTC, datetime

from telethon import events
from telethon import utils as telegram_utils
from telethon.errors import PremiumAccountRequiredError
from telethon.tl import functions as telegram_functions
from telethon.tl import types as telegram_types

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
from sidekick.chat.provenance import MessageOrigin
from sidekick.channel_status import (
    AdapterRuntimeState,
    CachedChannelInventory,
    ChannelInventoryItem,
    ChannelOpsServer,
    ChannelOpsSettings,
    ChannelSnapshotService,
)
from sidekick.inbound import (
    DurableInboundPool,
    DurableInboundWorker,
    is_ai_candidate,
)
from sidekick.inbound_store import SQLiteInboundWorkStore
from sidekick.plugins.base import PluginMount
from sidekick.telegram import TelegramCommand
from sidekick.telegram.ai_transport import (
    TelegramChatTransport,
    TelegramEditLimiter,
    select_telegram_response_format,
)
from sidekick.telegram.ai_identity import (
    TELEGRAM_IDENTITY_CODEC,
    TelegramDirectorySourceResolver,
    TelegramMatrixBridgeResolver,
    TelegramMemoryScopeTargetResolver,
    TelegramMessageIdentityResolver,
    TelegramMessageMentionResolver,
    telegram_memory_event_metadata,
)
from sidekick.telegram.ai_history import (
    TelegramHistorySource,
    TelegramInboundMessageSource,
    telegram_channel_album_document_id,
    telegram_source_retry_delay,
)
from sidekick.telegram.message_links import (
    TelegramMessageLink as _TelegramMessageLink,
    parse_telegram_message_link as _parse_telegram_message_link,
)


class _SavedMemoryForwardSourceUnavailable(RuntimeError):
    pass


class _SavedMemoryLinkUnavailable(RuntimeError):
    pass


class TelegramAI(TelegramCommand, metaclass=PluginMount):
    command_name = "ai"
    INBOUND_CONCURRENCY = 8
    THINKING_REPLY = "🤔 Thinking..."
    MEMORY_STORED_REACTION = "✍"
    MEMORY_FAILED_REACTION = "👎"
    MEMORY_PROCESSING_REPLY = "Remembering..."
    MEMORY_STORED_REPLY = "Remembered."
    MEMORY_FAILED_REPLY = "Memory update failed. Forward the message again to retry."
    MEMORY_LINK_FAILED_REPLY = (
        "Memory update failed. Send the message link again to retry."
    )
    MEMORY_SOURCE_UNAVAILABLE_REPLY = (
        "Telegram hid the original source. Paste the original message link in "
        "Saved Messages to remember its reply chain."
    )
    MEMORY_LINK_UNAVAILABLE_REPLY = (
        "Memory update unavailable: the linked message could not be fetched. "
        "Make sure this account can open it and the message still exists."
    )

    def __init__(
        self,
        account: str = "default",
        session: str | None = None,
        log_level: str = "info",
    ):
        super().__init__(account=account, session=session, log_level=log_level)
        settings = AISettings.from_env()
        self._settings = settings
        self._ops_settings = ChannelOpsSettings.from_env(
            default_instance_id=f"telegram-{account}",
        )
        self._edit_limiter = TelegramEditLimiter(
            settings.edit_cadence,
            logger=self.logger,
        )
        self._gateway = PiAgentGateway(
            settings.agent_url,
            token=settings.agent_token,
            timeout=settings.request_timeout,
        )
        self._responder: AIResponder | None = None
        self._transport: TelegramChatTransport | None = None
        self._store = AIStateRepository(settings.state_path)
        self._inbound_store = SQLiteInboundWorkStore(settings.state_path)
        self._memory = (
            HindsightMemoryClient(
                settings.hindsight_url,
                token=settings.hindsight_token or "",
                timeout=settings.hindsight_timeout,
            )
            if settings.hindsight_url
            else None
        )
        self._handler: AIConversationHandler | None = None
        self._inbound_pool: DurableInboundPool | None = None
        self._dream_scheduler: DreamScheduler | None = None
        self._continuous_memory_scheduler: ContinuousMemoryScheduler | None = None
        self._memory_outbox_scheduler: MemoryOutboxScheduler | None = None
        self._owner_id: int | None = None
        self._saved_memory_lock = asyncio.Lock()
        self._adapter_status = AdapterRuntimeState(
            id=self._ops_settings.instance_id,
            platform="telegram",
            connected_probe=self.client.is_connected,
            indeterminate_outbound_probe=lambda: (
                self._transport.indeterminate_outbound_count
                if self._transport is not None
                else None
            ),
        )
        self._telegram_inventory = CachedChannelInventory(
            self._load_channel_inventory,
            ttl_seconds=30.0,
        )
        self._ops_server = ChannelOpsServer(
            snapshot_service=ChannelSnapshotService(
                state_reader=self._store,
                inventory_loader=self._telegram_inventory.list_channels,
                adapter=self._adapter_status,
                memory_available=self._memory is not None,
                logger=self.logger,
            ),
            settings=self._ops_settings,
            logger=self.logger,
        )

    def __call__(self) -> None:
        """Run the reply-based Pi-powered Telegram userbot."""
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            pass

    async def _run(self) -> None:
        await self.service.connect()
        try:
            await self._setup()
            await self._ops_server.start()
            await self.service.wait_until_disconnected()
        finally:
            self._adapter_status.update(connected=False)
            if self._inbound_pool is not None:
                await self._inbound_pool.close()
            await self._ops_server.close()
            if self._continuous_memory_scheduler is not None:
                await self._continuous_memory_scheduler.close()
            if self._dream_scheduler is not None:
                await self._dream_scheduler.close()
            if self._memory_outbox_scheduler is not None:
                await self._memory_outbox_scheduler.close()
            if self._memory is not None:
                await self._memory.close()
            await self._gateway.close()
            await self._inbound_store.close()
            await self._store.close()
            await self.service.close()

    async def _setup(self) -> None:
        owner = await self.client.get_me()
        self._owner_id = owner.id
        self._adapter_status.update(
            account_id=str(owner.id),
            connected=True,
        )
        response_format = select_telegram_response_format(
            is_bot_account=bool(getattr(owner, "bot", False)),
            rich_messages_available=True,
        )
        transport = TelegramChatTransport(
            response_format=response_format,
            edit_limiter=self._edit_limiter,
            logger=self.logger,
        )
        self._transport = transport
        responder = AIResponder(
            self._gateway,
            max_output_chars=self._settings.max_output_chars,
            initial_status=self.THINKING_REPLY,
            transport=transport,
            logger=self.logger,
        )
        self._responder = responder
        await self._store.connect()
        await self._inbound_store.connect()
        await self._inbound_store.initialize_source(
            self._ops_settings.instance_id,
            epoch=str(owner.id),
            initial_cursor=0,
        )
        await self._inbound_store.recover_pending_ai_work(
            self._ops_settings.instance_id
        )
        history_source = TelegramHistorySource(self.client)
        matrix_bridge_resolver = TelegramMatrixBridgeResolver(
            self.service.config.matrix_bridge_bot_ids
        )
        prompt_builder = PromptBuilder(
            system_prompt=agent_system_prompt(self._settings.system_prompt),
            max_context_messages=self._settings.max_context_messages,
            max_context_chars=self._settings.max_context_chars,
            attachment_describer=ChatAttachmentDescriber(
                self._gateway,
                logger=self.logger,
            ),
            identity_resolver=TelegramMessageIdentityResolver(
                bridge_resolver=matrix_bridge_resolver,
                logger=self.logger,
            ),
            mention_resolver=TelegramMessageMentionResolver(
                self.client,
                logger=self.logger,
            ),
            history_source=history_source,
            transport=transport,
            attribution_resolver=matrix_bridge_resolver,
            identity_codec=TELEGRAM_IDENTITY_CODEC,
            metadata_resolver=telegram_memory_event_metadata,
        )
        memory_ingestor = (
            ChatMemoryIngestor(
                source=history_source,
                store=self._store,
                memory=self._memory,
                prompt_builder=prompt_builder,
                dream_settings=DreamSettings.from_env(),
                ingestion_settings=MemoryIngestionSettings.from_env(),
                identity_codec=TELEGRAM_IDENTITY_CODEC,
                source_retry_delay=telegram_source_retry_delay,
                album_document_id=telegram_channel_album_document_id,
                logger=self.logger,
            )
            if self._memory is not None
            else None
        )
        if memory_ingestor is not None:
            self._dream_scheduler = DreamScheduler(
                scanner=memory_ingestor,
                store=self._store,
                identity_codec=TELEGRAM_IDENTITY_CODEC,
                settings=DreamSchedulerSettings.from_env(),
                logger=self.logger,
            )
            self._continuous_memory_scheduler = ContinuousMemoryScheduler(
                runner=memory_ingestor,
                store=self._store,
                identity_codec=TELEGRAM_IDENTITY_CODEC,
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
            owner_id=owner.id,
            responder=responder,
            store=self._store,
            prompt_builder=prompt_builder,
            rate_limiter=AIRateLimiter(
                self._store,
                cooldown_seconds=self._settings.delegated_cooldown,
            ),
            memory=self._memory,
            dream_runner=memory_ingestor,
            memory_scope_resolver=TelegramMemoryScopeTargetResolver(
                self.client,
                logger=self.logger,
            ),
            directory_source_resolver=TelegramDirectorySourceResolver(
                self.client,
                logger=self.logger,
            ),
            memory_command_delete_delay=(self._settings.memory_command_delete_delay),
            transport=transport,
            identity_codec=TELEGRAM_IDENTITY_CODEC,
            run_store=self._store,
            adapter_instance_id=self._ops_settings.instance_id,
            logger=self.logger,
        )
        inbound_source = TelegramInboundMessageSource(self.client)
        inbound_worker = DurableInboundWorker(
            inbound_source,
            self._inbound_store,
            self._ops_settings.instance_id,
            logger=self.logger,
        )
        self._inbound_pool = DurableInboundPool(
            inbound_worker,
            self._inbound_store,
            self._ops_settings.instance_id,
            self._handler,
            concurrency=self.INBOUND_CONCURRENCY,
            logger=self.logger,
        )
        self._inbound_pool.start()
        self.client.add_event_handler(self._on_message, events.NewMessage())
        if self._continuous_memory_scheduler is not None:
            self._continuous_memory_scheduler.start()
        if self._dream_scheduler is not None:
            self._dream_scheduler.start()
        if self._memory_outbox_scheduler is not None:
            self._memory_outbox_scheduler.start()
        self.logger.info("Telegram AI userbot started")

    async def _load_channel_inventory(self) -> tuple[ChannelInventoryItem, ...]:
        items: list[ChannelInventoryItem] = []
        async for dialog in self.client.iter_dialogs(limit=None):
            chat_id = getattr(dialog, "id", None)
            if not isinstance(chat_id, int):
                continue
            if bool(getattr(dialog, "is_group", False)):
                chat_kind = "GROUP"
            elif bool(getattr(dialog, "is_channel", False)):
                chat_kind = "CHANNEL"
            else:
                chat_kind = "DIRECT"
            name = getattr(dialog, "name", None)
            observed = getattr(dialog, "date", None)
            if isinstance(observed, datetime):
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=UTC)
                observed_at = observed.timestamp()
            else:
                observed_at = None
            items.append(
                ChannelInventoryItem(
                    scope_id=TELEGRAM_IDENTITY_CODEC.scope_id(chat_id),
                    display_name=name if isinstance(name, str) else None,
                    chat_kind=chat_kind,
                    last_observed_at=observed_at,
                )
            )
        return tuple(items)

    async def _on_message(self, event) -> None:
        if (
            self._handler is not None
            and self._transport is not None
            and self._inbound_pool is not None
        ):
            try:
                origin = await self._transport.classify_origin(event.message)
                if origin in {
                    MessageOrigin.SIDEKICK_GENERATED,
                    MessageOrigin.INDETERMINATE,
                }:
                    return
                if await self._handle_saved_memory(event.message):
                    return
                if not is_ai_candidate(event.message):
                    return
                await self._inbound_store.accept_pending_ai_event(
                    self._ops_settings.instance_id,
                    cursor=event.message.id,
                    chat_id=event.message.chat_id,
                    message_id=event.message.id,
                    kind="message",
                    attested_origin=origin,
                )
                self._inbound_pool.notify()
            except Exception as exc:
                self.logger.error(
                    "Telegram AI message handling failed (%s)",
                    type(exc).__name__,
                )

    async def _handle_saved_memory(self, message) -> bool:
        if not self._is_saved_messages_message(message):
            return False
        forward = getattr(message, "fwd_from", None)
        link = (
            None
            if forward is not None
            else _parse_telegram_message_link(getattr(message, "raw_text", "") or "")
        )
        if forward is None and link is None:
            return False

        assert self._owner_id is not None
        async with self._saved_memory_lock:
            status_message = None
            try:
                if await self._store.is_memory_forward_processed(
                    owner_id=self._owner_id,
                    saved_message_id=message.id,
                ):
                    await self._set_saved_memory_reaction(
                        message,
                        self.MEMORY_STORED_REACTION,
                    )
                    return True

                status_message = await self._start_saved_memory_status(message)
                if link is not None:
                    source, source_chat_id = await self._resolve_memory_link(link)
                    source_message_id = link.message_id
                else:
                    source_peer = getattr(forward, "saved_from_peer", None)
                    source_message_id = getattr(forward, "saved_from_msg_id", None)
                    if source_peer is None or not isinstance(source_message_id, int):
                        raise _SavedMemoryForwardSourceUnavailable(
                            "forward has no Saved Messages source pointer"
                        )

                    source = await self.client.get_messages(
                        source_peer,
                        ids=source_message_id,
                    )
                    source_chat_id = getattr(source, "chat_id", None)
                    if source is None or not isinstance(source_chat_id, int):
                        raise _SavedMemoryForwardSourceUnavailable(
                            "original Telegram message is unavailable"
                        )
                if not await self._handler.remember_reply_chain(source):
                    raise RuntimeError("reply chain has no ingestible human content")

                await self._store.record_memory_forward(
                    owner_id=self._owner_id,
                    saved_message_id=message.id,
                    source_chat_id=source_chat_id,
                    source_message_id=source_message_id,
                )
            except Exception as exc:
                self.logger.warning(
                    "Saved Messages memory ingest failed (error=%s)",
                    type(exc).__name__,
                )
                await self._set_saved_memory_reaction(
                    message,
                    self.MEMORY_FAILED_REACTION,
                )
                if isinstance(exc, _SavedMemoryForwardSourceUnavailable):
                    reply = self.MEMORY_SOURCE_UNAVAILABLE_REPLY
                elif isinstance(exc, _SavedMemoryLinkUnavailable):
                    reply = self.MEMORY_LINK_UNAVAILABLE_REPLY
                elif link is not None:
                    reply = self.MEMORY_LINK_FAILED_REPLY
                else:
                    reply = self.MEMORY_FAILED_REPLY
                await self._finish_saved_memory_status(
                    message,
                    status_message,
                    reply,
                )
                return True

            self.logger.info("Saved Messages memory ingested")
            await self._set_saved_memory_reaction(
                message,
                self.MEMORY_STORED_REACTION,
            )
            await self._finish_saved_memory_status(
                message,
                status_message,
                self.MEMORY_STORED_REPLY,
            )
            return True

    async def _resolve_memory_link(
        self,
        link: _TelegramMessageLink,
    ) -> tuple[object, int]:
        try:
            source_peer = await self._resolve_memory_link_peer(link)
            expected_chat_id = telegram_utils.get_peer_id(source_peer)
            _, peer_type = telegram_utils.resolve_id(expected_chat_id)
            if peer_type is not telegram_types.PeerChannel:
                raise ValueError("message link did not resolve to a channel")
            source = await self.client.get_messages(
                source_peer,
                ids=link.message_id,
            )
        except Exception as exc:
            raise _SavedMemoryLinkUnavailable(
                "linked Telegram message is unavailable"
            ) from exc

        source_chat_id = getattr(source, "chat_id", None)
        source_message_id = getattr(source, "id", None)
        if (
            source is None
            or source_chat_id != expected_chat_id
            or source_message_id != link.message_id
        ):
            raise _SavedMemoryLinkUnavailable("linked Telegram message is unavailable")
        return source, source_chat_id

    async def _resolve_memory_link_peer(self, link: _TelegramMessageLink):
        if link.username is not None:
            return await self.client.get_input_entity(link.username)

        assert link.channel_id is not None
        peer = telegram_types.PeerChannel(link.channel_id)
        try:
            return await self.client.get_input_entity(peer)
        except ValueError as cache_error:
            self.logger.info(
                "Private message-link peer missing from entity cache; "
                "searching dialogs",
            )
            expected_chat_id = telegram_utils.get_peer_id(peer)
            async for dialog in self.client.iter_dialogs():
                if dialog.id == expected_chat_id:
                    return dialog.input_entity
            raise cache_error

    def _is_saved_messages_message(self, message) -> bool:
        peer = getattr(message, "peer_id", None)
        destination_id = (
            telegram_utils.get_peer_id(peer)
            if peer is not None
            else getattr(message, "chat_id", None)
        )
        return bool(self._owner_id is not None and destination_id == self._owner_id)

    async def _set_saved_memory_reaction(self, message, reaction: str) -> bool:
        try:
            peer = await message.get_input_chat()
            await self.client(
                telegram_functions.messages.SendReactionRequest(
                    peer=peer,
                    msg_id=message.id,
                    reaction=[telegram_types.ReactionEmoji(emoticon=reaction)],
                )
            )
            return True
        except PremiumAccountRequiredError:
            self.logger.info(
                "Saved Messages memory marker unavailable for non-Premium account",
            )
        except Exception as exc:
            self.logger.warning(
                "Saved Messages memory marker failed (error=%s)",
                type(exc).__name__,
            )
        return False

    async def _start_saved_memory_status(self, message):
        try:
            assert self._transport is not None
            return await self._transport.reply(
                message,
                self.MEMORY_PROCESSING_REPLY,
                presentation="plain",
            )
        except Exception as exc:
            self.logger.warning(
                "Saved Messages memory processing reply failed (error=%s)",
                type(exc).__name__,
            )
            return None

    async def _finish_saved_memory_status(
        self,
        message,
        status_message,
        reply: str,
    ) -> None:
        if status_message is not None:
            try:
                if await self._edit_limiter.run(
                    lambda: status_message.edit(reply, parse_mode=None),
                    wait=True,
                ):
                    return
            except Exception as exc:
                self.logger.warning(
                    "Saved Messages memory status edit failed (error=%s)",
                    type(exc).__name__,
                )
        try:
            assert self._transport is not None
            await self._transport.reply(
                message,
                reply,
                presentation="plain",
            )
        except Exception as exc:
            self.logger.warning(
                "Saved Messages memory final reply failed (error=%s)",
                type(exc).__name__,
            )
