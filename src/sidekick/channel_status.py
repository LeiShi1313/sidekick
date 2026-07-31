from __future__ import annotations

import asyncio
import hmac
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from urllib.parse import quote

from aiohttp import web


AccessMode = Literal["OPEN", "RESTRICTED"]
ChatKind = Literal["DIRECT", "GROUP", "CHANNEL", "UNKNOWN"]
MemoryMode = Literal["OFF", "CONTINUOUS", "DREAM", "UNAVAILABLE"]
AIRunStatus = Literal[
    "STARTING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
]
ACTIVE_AI_RUN_STATUSES = frozenset({"STARTING", "RUNNING"})


@dataclass(frozen=True, slots=True)
class AgentRunOrigin:
    scope_id: str
    adapter_instance_id: str

    def __post_init__(self) -> None:
        if not self.scope_id or len(self.scope_id) > 512:
            raise ValueError("Agent run origin scope ID is invalid")
        if not self.adapter_instance_id or len(self.adapter_instance_id) > 128:
            raise ValueError("Agent run origin adapter instance ID is invalid")


@dataclass(frozen=True, slots=True)
class ActiveAIRun:
    run_id: str
    status: AIRunStatus
    started_at: float
    updated_at: float
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class StoredChannelState:
    scope_id: str
    display_name: str | None = None
    access_open: bool = False
    model_override: str | None = None
    continuous_enabled: bool = False
    dream_enabled: bool = False
    continuous_last_attempt_at: float | None = None
    continuous_last_success_at: float | None = None
    continuous_last_error: str | None = None
    dream_last_attempt_at: float | None = None
    dream_last_success_at: float | None = None
    dream_last_error: str | None = None
    retained_document_count: int = 0
    pending_count: int = 0
    last_ingested_at: float | None = None
    active_runs: tuple[ActiveAIRun, ...] = ()
    last_run_error: str | None = None
    last_run_error_at: float | None = None
    last_run_id: str | None = None
    updated_at: float | None = None


class ChannelStateReader(Protocol):
    async def list_channel_operational_states(
        self,
    ) -> tuple[StoredChannelState, ...]: ...


class AIRunStateWriter(Protocol):
    async def start_ai_run(
        self,
        *,
        run_id: str,
        scope_id: str,
        actor_id: str,
        adapter_instance_id: str,
        started_at: float,
    ) -> None: ...

    async def mark_ai_run_running(self, run_id: str, *, updated_at: float) -> None: ...

    async def finish_ai_run(
        self,
        run_id: str,
        *,
        status: AIRunStatus,
        updated_at: float,
        session_id: str | None = None,
        error_code: str | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ChannelInventoryItem:
    scope_id: str
    display_name: str | None
    chat_kind: ChatKind
    last_observed_at: float | None = None


ChannelInventoryLoader = Callable[
    [], Awaitable[tuple[ChannelInventoryItem, ...]]
]


class CachedChannelInventory:
    def __init__(
        self,
        loader: ChannelInventoryLoader,
        *,
        ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Channel inventory cache TTL must be positive")
        self._loader = loader
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: tuple[ChannelInventoryItem, ...] | None = None
        self._expires_at = 0.0
        self._refresh_failed = False
        self._lock = asyncio.Lock()

    async def list_channels(self) -> tuple[ChannelInventoryItem, ...]:
        now = self._clock()
        if now < self._expires_at and self._refresh_failed:
            raise RuntimeError("Channel inventory refresh is in backoff")
        if now < self._expires_at and self._items is not None:
            return self._items
        async with self._lock:
            now = self._clock()
            if now < self._expires_at and self._refresh_failed:
                raise RuntimeError("Channel inventory refresh is in backoff")
            if now < self._expires_at and self._items is not None:
                return self._items
            try:
                items = await self._loader()
            except Exception:
                self._refresh_failed = True
                self._expires_at = now + self._ttl_seconds
                raise
            self._items = items
            self._refresh_failed = False
            self._expires_at = now + self._ttl_seconds
            return items


@dataclass(frozen=True, slots=True)
class ChannelOpsSettings:
    instance_id: str
    host: str = "127.0.0.1"
    port: int = 8781

    def __post_init__(self) -> None:
        if not self.instance_id or len(self.instance_id) > 128:
            raise ValueError("Channel adapter instance ID is invalid")
        if not self.host:
            raise ValueError("Channel ops host cannot be empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("Channel ops port must be between 1 and 65535")

    @classmethod
    def from_env(
        cls,
        *,
        default_instance_id: str,
        environ: dict[str, str] | None = None,
    ) -> ChannelOpsSettings:
        import os

        values = os.environ if environ is None else environ
        instance_id = values.get(
            "SIDEKICK_ADAPTER_INSTANCE_ID",
            default_instance_id,
        ).strip()
        host = values.get("SIDEKICK_OPS_HOST", "127.0.0.1").strip()
        raw_port = values.get("SIDEKICK_OPS_PORT", "8781").strip()
        if not instance_id or len(instance_id) > 128:
            raise ValueError(
                "SIDEKICK_ADAPTER_INSTANCE_ID must contain 1 to 128 characters"
            )
        if not host:
            raise ValueError("SIDEKICK_OPS_HOST cannot be empty")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError("SIDEKICK_OPS_PORT must be an integer") from exc
        if not 1 <= port <= 65_535:
            raise ValueError("SIDEKICK_OPS_PORT must be between 1 and 65535")
        return cls(instance_id=instance_id, host=host, port=port)


@dataclass(slots=True)
class AdapterRuntimeState:
    id: str
    platform: str
    account_id: str | None = None
    connected: bool = False
    observed_at: float | None = None
    connected_probe: Callable[[], bool] | None = None

    def update(
        self,
        *,
        account_id: str | None = None,
        connected: bool | None = None,
        observed_at: float | None = None,
    ) -> None:
        if account_id is not None:
            self.account_id = account_id
        if connected is not None:
            self.connected = connected
        self.observed_at = time.time() if observed_at is None else observed_at

    def is_connected(self) -> bool:
        if self.connected_probe is None:
            return self.connected
        try:
            return bool(self.connected_probe())
        except Exception:
            return self.connected


class ChannelSnapshotService:
    def __init__(
        self,
        *,
        state_reader: ChannelStateReader,
        inventory_loader: ChannelInventoryLoader,
        adapter: AdapterRuntimeState,
        memory_available: bool,
        logger: Any | None = None,
        inventory_timeout: float = 15.0,
    ) -> None:
        if inventory_timeout <= 0:
            raise ValueError("Channel inventory timeout must be positive")
        self._state_reader = state_reader
        self._inventory_loader = inventory_loader
        self._adapter = adapter
        self._memory_available = memory_available
        self._logger = logger
        self._inventory_timeout = inventory_timeout
        self._last_inventory: tuple[ChannelInventoryItem, ...] = ()

    async def snapshot(self) -> dict[str, Any]:
        inventory_error: str | None = None
        try:
            inventory = await asyncio.wait_for(
                self._inventory_loader(),
                timeout=self._inventory_timeout,
            )
            self._last_inventory = inventory
        except Exception as exc:
            inventory = self._last_inventory
            inventory_error = "Channel inventory is temporarily unavailable."
            self._log_inventory_error(exc)
        states = await self._state_reader.list_channel_operational_states()
        items = self._merge(inventory, states)
        self._adapter.update(observed_at=time.time())
        adapter = {
            "id": self._adapter.id,
            "platform": self._adapter.platform,
            "accountId": self._adapter.account_id,
            "connected": self._adapter.is_connected(),
            "observedAt": _timestamp(self._adapter.observed_at),
        }
        if inventory_error is not None:
            adapter["error"] = inventory_error
        return {"adapter": adapter, "items": items}

    def _merge(
        self,
        inventory: tuple[ChannelInventoryItem, ...],
        states: tuple[StoredChannelState, ...],
    ) -> list[dict[str, Any]]:
        inventory_by_scope = {
            item.scope_id: item
            for item in inventory
            if _scope_belongs_to_adapter(
                item.scope_id,
                platform=self._adapter.platform,
                account_id=self._adapter.account_id,
            )
        }
        states_by_scope = {
            state.scope_id: state
            for state in states
            if _scope_belongs_to_adapter(
                state.scope_id,
                platform=self._adapter.platform,
                account_id=self._adapter.account_id,
            )
        }
        rows = [
            self._row(
                scope_id,
                inventory_by_scope.get(scope_id),
                states_by_scope.get(scope_id),
            )
            for scope_id in inventory_by_scope.keys() | states_by_scope.keys()
        ]
        return sorted(
            rows,
            key=lambda row: (
                str(row["displayName"]).casefold(),
                str(row["scopeId"]),
            ),
        )

    def _row(
        self,
        scope_id: str,
        inventory: ChannelInventoryItem | None,
        state: StoredChannelState | None,
    ) -> dict[str, Any]:
        continuous_enabled = bool(state and state.continuous_enabled)
        dream_enabled = bool(state and state.dream_enabled)
        if not self._memory_available and (continuous_enabled or dream_enabled):
            memory_mode: MemoryMode = "UNAVAILABLE"
        elif continuous_enabled:
            memory_mode = "CONTINUOUS"
        elif dream_enabled:
            memory_mode = "DREAM"
        else:
            memory_mode = "OFF"
        errors = _channel_errors(state)
        updated_at = _latest_timestamp(
            inventory.last_observed_at if inventory is not None else None,
            state.updated_at if state is not None else None,
            state.last_ingested_at if state is not None else None,
            state.continuous_last_attempt_at if state is not None else None,
            state.continuous_last_success_at if state is not None else None,
            state.dream_last_attempt_at if state is not None else None,
            state.dream_last_success_at if state is not None else None,
            *(
                (run.updated_at for run in state.active_runs)
                if state is not None
                else ()
            ),
        )
        return {
            "scopeId": scope_id,
            "platform": self._adapter.platform,
            "adapterInstanceId": self._adapter.id,
            "accountId": self._adapter.account_id,
            "displayName": _display_name(scope_id, inventory, state),
            "chatKind": (
                inventory.chat_kind if inventory is not None else _chat_kind(scope_id)
            ),
            "accessMode": "OPEN" if state and state.access_open else "RESTRICTED",
            "modelOverride": state.model_override if state is not None else None,
            "memory": {
                "continuousEnabled": continuous_enabled,
                "dreamEnabled": dream_enabled,
                "effectiveMode": memory_mode,
                "continuousLastAttemptAt": _timestamp(
                    state.continuous_last_attempt_at if state is not None else None
                ),
                "continuousLastSuccessAt": _timestamp(
                    state.continuous_last_success_at if state is not None else None
                ),
                "continuousLastError": _safe_error(
                    state.continuous_last_error if state is not None else None
                ),
                "dreamLastAttemptAt": _timestamp(
                    state.dream_last_attempt_at if state is not None else None
                ),
                "dreamLastSuccessAt": _timestamp(
                    state.dream_last_success_at if state is not None else None
                ),
                "dreamLastError": _safe_error(
                    state.dream_last_error if state is not None else None
                ),
                "pendingDocumentCount": (
                    state.pending_count if state is not None else 0
                ),
                "lastIngestedAt": _timestamp(
                    state.last_ingested_at if state is not None else None
                ),
                "hindsightBankId": scope_id,
                # Hindsight fact cardinality is not available from the adapter DB.
                # The dashboard backend enriches channels from Hindsight's bank list.
                "factCount": None,
                "retainedDocumentCount": (
                    state.retained_document_count if state is not None else 0
                ),
            },
            "lastObservedAt": _timestamp(
                inventory.last_observed_at if inventory is not None else None
            ),
            "activeRuns": [
                {
                    "runId": run.run_id,
                    "status": run.status,
                    "sessionId": run.session_id,
                    "scopeId": scope_id,
                    "adapterInstanceId": self._adapter.id,
                    "startedAt": _timestamp(run.started_at),
                    "updatedAt": _timestamp(run.updated_at),
                }
                for run in (state.active_runs if state is not None else ())
            ],
            "errors": errors,
            "updatedAt": _timestamp(updated_at),
        }

    def _log_inventory_error(self, exc: Exception) -> None:
        if self._logger is None:
            return
        log = getattr(self._logger, "warning", None)
        if callable(log):
            log(
                "Channel inventory refresh failed (%s): %s",
                type(exc).__name__,
                exc,
            )


class ChannelOpsServer:
    def __init__(
        self,
        *,
        snapshot_service: ChannelSnapshotService,
        token: str,
        settings: ChannelOpsSettings,
        logger: Any | None = None,
    ) -> None:
        if not token:
            raise ValueError("Channel ops server requires a bearer token")
        self._snapshot_service = snapshot_service
        self._token = token
        self._settings = settings
        self._logger = logger
        self._runner: web.AppRunner | None = None
        self.application = web.Application(client_max_size=16 * 1024)
        self.application.router.add_get("/health", self._health)
        self.application.router.add_get("/v1/channels", self._channels)

    async def start(self) -> None:
        if self._runner is not None:
            raise RuntimeError("Channel ops server is already running")
        runner = web.AppRunner(self.application, access_log=None)
        await runner.setup()
        try:
            await web.TCPSite(
                runner,
                host=self._settings.host,
                port=self._settings.port,
            ).start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _health(self, request: web.Request) -> web.Response:
        if not self._authenticated(request):
            return _unauthorized()
        adapter = self._snapshot_service._adapter
        return _json_response(
            {
                "ok": True,
                "adapter": {
                    "id": adapter.id,
                    "platform": adapter.platform,
                    "accountId": adapter.account_id,
                    "connected": adapter.is_connected(),
                    "observedAt": _timestamp(adapter.observed_at),
                },
            }
        )

    async def _channels(self, request: web.Request) -> web.Response:
        if not self._authenticated(request):
            return _unauthorized()
        try:
            payload = await self._snapshot_service.snapshot()
        except Exception as exc:
            self._log_snapshot_error(exc)
            return _json_response(
                {
                    "error": {
                        "code": "SNAPSHOT_UNAVAILABLE",
                        "message": "Channel status is temporarily unavailable.",
                    }
                },
                status=503,
            )
        return _json_response(payload)

    def _authenticated(self, request: web.Request) -> bool:
        authorization = request.headers.get("Authorization", "")
        supplied = authorization.removeprefix("Bearer ")
        return authorization.startswith("Bearer ") and hmac.compare_digest(
            supplied,
            self._token,
        )

    def _log_snapshot_error(self, exc: Exception) -> None:
        if self._logger is None:
            return
        log = getattr(self._logger, "exception", None)
        if callable(log):
            log(
                "Channel status snapshot failed (%s): %s",
                type(exc).__name__,
                exc,
            )


def _unauthorized() -> web.Response:
    return _json_response(
        {
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Authentication required.",
            }
        },
        status=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _json_response(
    payload: dict[str, Any],
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> web.Response:
    response = web.json_response(payload, status=status, headers=headers)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=UTC).isoformat().replace("+00:00", "Z")


def _display_name(
    scope_id: str,
    inventory: ChannelInventoryItem | None,
    state: StoredChannelState | None,
) -> str:
    for value in (
        inventory.display_name if inventory is not None else None,
        state.display_name if state is not None else None,
    ):
        if value and value.strip():
            return value.strip()[:256]
    return scope_id


def _scope_belongs_to_adapter(
    scope_id: str,
    *,
    platform: str,
    account_id: str | None,
) -> bool:
    if platform == "telegram":
        return scope_id.startswith("telegram:chat:")
    if platform == "qq":
        return scope_id.startswith(("qq:group:", "qq:private:"))
    if platform == "wechat":
        prefix = "wechat:account:"
        if not scope_id.startswith(prefix):
            return False
        if account_id is None:
            # The adapter envelope may be disconnected before its first
            # bootstrap, but channel rows require a stable account identity.
            return False
        encoded_account_id = quote(account_id, safe="-_.~")
        return scope_id.startswith(f"{prefix}{encoded_account_id}:chat:")
    return False


def _chat_kind(scope_id: str) -> ChatKind:
    if scope_id.startswith("qq:group:"):
        return "GROUP"
    if scope_id.startswith("qq:private:"):
        return "DIRECT"
    return "UNKNOWN"


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(token|authorization|api[-_ ]?key|secret|password)\b\s*[:=]\s*\S+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+\S+")


def _safe_error(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    normalized = _SECRET_ASSIGNMENT_RE.sub(r"\1=[redacted]", normalized)
    normalized = _BEARER_RE.sub("Bearer [redacted]", normalized)
    return normalized[:300] or None


def _channel_errors(state: StoredChannelState | None) -> list[dict[str, Any]]:
    if state is None:
        return []
    candidates = (
        (
            "CONTINUOUS_MEMORY",
            "MEMORY_INGESTION_ERROR",
            state.continuous_last_error,
            state.continuous_last_attempt_at,
            None,
        ),
        (
            "DREAM_MEMORY",
            "MEMORY_INGESTION_ERROR",
            state.dream_last_error,
            state.dream_last_attempt_at,
            None,
        ),
        (
            "AI_RUN",
            state.last_run_error,
            _run_error_message(state.last_run_error),
            state.last_run_error_at,
            state.last_run_id,
        ),
    )
    return [
        {
            "component": component,
            "code": code,
            "message": safe,
            "occurredAt": _timestamp(timestamp),
            "runId": run_id,
        }
        for component, code, message, timestamp, run_id in candidates
        if (safe := _safe_error(message)) is not None
    ]


def _latest_timestamp(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _run_error_message(code: str | None) -> str | None:
    if code is None:
        return None
    messages = {
        "ADAPTER_RESTARTED": "The adapter restarted before the AI run completed.",
        "CANCELLED": "The AI run was cancelled.",
        "RATE_LIMITED": "The AI provider rate limited the run.",
        "DELIVERY_FAILED": "The AI response could not be delivered.",
        "EMPTY_RESPONSE": "The AI run returned no usable response.",
        "AGENT_ERROR": "The AI agent run failed.",
        "PREPARATION_FAILED": "The AI request could not be prepared.",
        "HANDLER_ERROR": "The adapter failed while handling the AI run.",
    }
    return messages.get(code, "The AI run failed.")
