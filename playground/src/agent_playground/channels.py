from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import time
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import aiohttp


DEFAULT_CHANNEL_ADAPTER_URLS: tuple[tuple[str, str], ...] = (
    ("telegram", "http://ai:8781/v1/channels"),
    ("onebot", "http://onebot-ai:8781/v1/channels"),
    ("wechat-host", "http://wechat-host-ai:8781/v1/channels"),
    ("wechat-peer", "http://wechat-peer-ai:8781/v1/channels"),
)

_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_CHANNELS = 20_000
_MAX_RUNS = 10_000
_MAX_ERRORS = 100
_MODEL_CACHE_TTL_SECONDS = 60.0
_BANK_STATS_CONCURRENCY = 8
RESERVED_CHANNEL_SOURCE_IDS = frozenset(
    {"pi-models", "pi-runs", "hindsight", "hindsight-stats"}
)


def parse_adapter_urls(value: str | None) -> tuple[tuple[str, str], ...]:
    """Parse adapter snapshot endpoints from JSON or id=url comma pairs."""
    if value is None or not value.strip():
        return DEFAULT_CHANNEL_ADAPTER_URLS
    raw = value.strip()
    if raw.startswith("{"):
        try:
            supplied = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("PLAYGROUND_CHANNEL_ADAPTER_URLS is invalid JSON") from exc
        if not isinstance(supplied, dict):
            raise ValueError("PLAYGROUND_CHANNEL_ADAPTER_URLS must be a JSON object")
        pairs = list(supplied.items())
    else:
        pairs = []
        for entry in raw.split(","):
            source_id, separator, url = entry.strip().partition("=")
            if not separator:
                raise ValueError(
                    "PLAYGROUND_CHANNEL_ADAPTER_URLS entries must use id=url"
                )
            pairs.append((source_id, url))
    if not pairs or len(pairs) > 32:
        raise ValueError("PLAYGROUND_CHANNEL_ADAPTER_URLS has an invalid source count")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source_id, url in pairs:
        if not isinstance(source_id, str) or not _SOURCE_ID_RE.fullmatch(source_id):
            raise ValueError("Adapter source IDs must be simple identifiers")
        if source_id in RESERVED_CHANNEL_SOURCE_IDS:
            raise ValueError(f"Adapter source ID {source_id!r} is reserved")
        if source_id in seen:
            raise ValueError("Adapter source IDs must be unique")
        if not isinstance(url, str):
            raise ValueError("Adapter source URLs must be strings")
        normalized = url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("Adapter source URLs must use http or https")
        if len(normalized) > 2_048:
            raise ValueError("Adapter source URL is too long")
        seen.add(source_id)
        result.append((source_id, normalized))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ChannelDashboardConfig:
    adapter_urls: tuple[tuple[str, str], ...]
    pi_url: str
    memory_url: str
    pi_token: str
    channel_token: str
    memory_token: str
    poll_interval: float = 2.0
    source_timeout: float = 5.0
    bank_cache_ttl: float = 15.0


@dataclass(slots=True)
class _CachedSource:
    value: Any
    succeeded_at: str


class ChannelDashboard:
    """Maintains one resilient, fan-out snapshot for the Channels view."""

    def __init__(self, config: ChannelDashboardConfig):
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._refresh_lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._cache: dict[str, _CachedSource] = {}
        self._bank_cache_deadline = 0.0
        self._bank_stats_cache: dict[str, _CachedSource] = {}
        self._bank_stats_deadlines: dict[str, float] = {}
        self._bank_stats_errors: dict[str, str] = {}
        self._model_cache_deadline = 0.0
        self._model_cache_status: dict[str, Any] | None = None
        self._fingerprint: str | None = None
        self._generation = 0
        self._stream_id = str(uuid4())
        self._snapshot: dict[str, Any] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._poll_forever(), name="channel-dashboard-poller"
            )

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._subscribers.clear()

    async def snapshot(self) -> dict[str, Any]:
        if self._snapshot is None:
            await self.refresh()
        assert self._snapshot is not None
        return self._snapshot

    async def refresh(self) -> dict[str, Any]:
        async with self._refresh_lock:
            source_tasks: dict[str, asyncio.Task[tuple[Any, dict[str, Any]]]] = {}
            for source_id, url in self._config.adapter_urls:
                source_tasks[source_id] = asyncio.create_task(
                    self._load_source(
                        source_id,
                        "adapter",
                        url,
                        _parse_adapter_snapshot,
                        token=self._config.channel_token,
                    )
                )
            model_cache_valid = (
                self._model_cache_status is not None
                and time.monotonic() < self._model_cache_deadline
            )
            if not model_cache_valid:
                source_tasks["pi-models"] = asyncio.create_task(
                    self._load_source(
                        "pi-models",
                        "pi",
                        f"{self._config.pi_url}/v1/models",
                        _parse_models,
                        token=self._config.pi_token,
                    )
                )
            source_tasks["pi-runs"] = asyncio.create_task(
                self._load_source(
                    "pi-runs",
                    "pi",
                    f"{self._config.pi_url}/v1/runs?status=active",
                    _parse_active_runs,
                    token=self._config.pi_token,
                )
            )
            bank_cache_valid = (
                "hindsight" in self._cache
                and time.monotonic() < self._bank_cache_deadline
            )
            if not bank_cache_valid:
                source_tasks["hindsight"] = asyncio.create_task(
                    self._load_source(
                        "hindsight",
                        "memory",
                        f"{self._config.memory_url}/v1/default/banks",
                        _parse_banks,
                        token=self._config.memory_token,
                    )
                )

            source_ids = list(source_tasks)
            loaded = await asyncio.gather(
                *(source_tasks[source_id] for source_id in source_ids)
            )
            results = dict(zip(source_ids, loaded, strict=True))
            if model_cache_valid:
                cached_models = self._cache.get("pi-models")
                results["pi-models"] = (
                    cached_models.value if cached_models is not None else None,
                    self._model_cache_status,
                )
            else:
                self._model_cache_status = results["pi-models"][1]
                self._model_cache_deadline = time.monotonic() + _MODEL_CACHE_TTL_SECONDS
            for source_id, _url in self._config.adapter_urls:
                value, source = results[source_id]
                adapter_error = value and value["adapter"].get("error")
                if value is not None:
                    source = {**source, "adapter": value["adapter"]}
                if source["status"] == "ok" and adapter_error:
                    source = {
                        **source,
                        "status": "degraded",
                        "error": adapter_error,
                    }
                elif (
                    source["status"] == "ok"
                    and value is not None
                    and not value["adapter"]["connected"]
                ):
                    source = {
                        **source,
                        "status": "degraded",
                        "error": "Adapter is disconnected",
                    }
                results[source_id] = (value, source)
            if bank_cache_valid:
                cached = self._cache["hindsight"]
                results["hindsight"] = (
                    cached.value,
                    _source_status(
                        "hindsight",
                        "memory",
                        "ok",
                        cached.succeeded_at,
                    ),
                )
            elif results["hindsight"][1]["status"] == "ok":
                self._bank_cache_deadline = (
                    time.monotonic() + self._config.bank_cache_ttl
                )

            banks = results["hindsight"][0]
            channel_bank_ids = {
                item["scopeId"]
                for source_id, _url in self._config.adapter_urls
                if results[source_id][0] is not None
                for item in results[source_id][0]["items"]
            }
            stats_banks = (
                [bank for bank in banks if bank["bankId"] in channel_bank_ids]
                if banks is not None
                else None
            )
            bank_stats, bank_stats_status = await self._load_bank_stats(stats_banks)
            if banks is not None:
                banks = [
                    {
                        **bank,
                        **_empty_bank_stats(),
                        **bank_stats.get(bank["bankId"], {}),
                    }
                    for bank in banks
                ]

            adapters = [
                results[source_id][0]
                for source_id, _url in self._config.adapter_urls
                if results[source_id][0] is not None
            ]
            models = results["pi-models"][0]
            runs = results["pi-runs"][0]
            statuses = [
                results[source_id][1] for source_id, _url in self._config.adapter_urls
            ] + [
                results["pi-models"][1],
                results["pi-runs"][1],
                results["hindsight"][1],
                bank_stats_status,
            ]
            content = _build_snapshot_content(
                adapters=adapters,
                models=models,
                runs=runs,
                banks=banks,
                sources=statuses,
            )
            fingerprint = hashlib.sha256(
                json.dumps(
                    _stable_snapshot_content(content),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if self._snapshot is None or fingerprint != self._fingerprint:
                self._generation += 1
                self._fingerprint = fingerprint
                self._snapshot = {
                    "streamId": self._stream_id,
                    "generation": self._generation,
                    "generatedAt": _now(),
                    **content,
                }
                self._publish(self._snapshot)
            return self._snapshot

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def _poll_forever(self) -> None:
        while True:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Each source is isolated already; this is a final guard that keeps
                # the next polling cycle alive after an unexpected parser bug.
                pass
            await asyncio.sleep(self._config.poll_interval)

    async def _load_source(
        self,
        source_id: str,
        kind: str,
        url: str,
        parser: Any,
        *,
        token: str,
    ) -> tuple[Any, dict[str, Any]]:
        try:
            payload = await self._json(url, token=token)
            value = parser(payload, source_id)
            succeeded_at = _now()
            self._cache[source_id] = _CachedSource(value, succeeded_at)
            return value, _source_status(source_id, kind, "ok", succeeded_at)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            cached = self._cache.get(source_id)
            message = _safe_error(exc)
            if cached is not None:
                return cached.value, _source_status(
                    source_id,
                    kind,
                    "stale",
                    cached.succeeded_at,
                    message,
                )
            return None, _source_status(source_id, kind, "unavailable", None, message)

    async def _load_bank_stats(
        self,
        banks: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        source_id = "hindsight-stats"
        if banks is None:
            return {}, _source_status(
                source_id,
                "memory",
                "unavailable",
                None,
                "Hindsight bank list is unavailable",
            )
        bank_ids = tuple(dict.fromkeys(bank["bankId"] for bank in banks))
        active_ids = set(bank_ids)
        for bank_id in set(self._bank_stats_cache) - active_ids:
            self._bank_stats_cache.pop(bank_id, None)
            self._bank_stats_deadlines.pop(bank_id, None)
            self._bank_stats_errors.pop(bank_id, None)
        if not bank_ids:
            succeeded_at = _now()
            return {}, _source_status(
                source_id,
                "memory",
                "ok",
                succeeded_at,
            )
        semaphore = asyncio.Semaphore(_BANK_STATS_CONCURRENCY)

        async def load(
            bank_id: str,
        ) -> tuple[str, dict[str, Any] | None, str, str | None]:
            now = time.monotonic()
            cached = self._bank_stats_cache.get(bank_id)
            if now < self._bank_stats_deadlines.get(bank_id, 0):
                error = self._bank_stats_errors.get(bank_id)
                if cached is not None:
                    return (
                        bank_id,
                        cached.value,
                        "stale" if error else "ok",
                        cached.succeeded_at,
                    )
                if error:
                    return bank_id, None, "unavailable", None
            async with semaphore:
                try:
                    payload = await self._json(
                        f"{self._config.memory_url}/v1/default/banks/"
                        f"{quote(bank_id, safe='')}/stats",
                        token=self._config.memory_token,
                    )
                    value = _parse_bank_stats(payload, bank_id)
                    succeeded_at = _now()
                    self._bank_stats_cache[bank_id] = _CachedSource(
                        value,
                        succeeded_at,
                    )
                    self._bank_stats_errors.pop(bank_id, None)
                    status = "ok"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._bank_stats_errors[bank_id] = _safe_error(exc)
                    cached = self._bank_stats_cache.get(bank_id)
                    value = cached.value if cached is not None else None
                    succeeded_at = cached.succeeded_at if cached is not None else None
                    status = "stale" if cached is not None else "unavailable"
                self._bank_stats_deadlines[bank_id] = (
                    time.monotonic() + self._config.bank_cache_ttl
                )
                return bank_id, value, status, succeeded_at

        loaded = await asyncio.gather(*(load(bank_id) for bank_id in bank_ids))
        values = {
            bank_id: value
            for bank_id, value, _status, _succeeded_at in loaded
            if value is not None
        }
        unavailable = sum(status == "unavailable" for _, _, status, _ in loaded)
        stale = sum(status == "stale" for _, _, status, _ in loaded)
        succeeded = [
            succeeded_at
            for _, _, _, succeeded_at in loaded
            if succeeded_at is not None
        ]
        if unavailable:
            status = "degraded" if values else "unavailable"
            error = f"{unavailable} of {len(bank_ids)} bank stats unavailable"
        elif stale:
            status = "stale"
            error = f"Using cached stats for {stale} of {len(bank_ids)} banks"
        else:
            status = "ok"
            error = None
        return values, _source_status(
            source_id,
            "memory",
            status,
            max(succeeded) if succeeded else None,
            error,
        )

    async def _json(self, url: str, *, token: str) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with self._get_session().get(url, headers=headers) as response:
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(f"HTTP {response.status}")
                content_length = getattr(response, "content_length", None)
                if content_length is not None and content_length > _MAX_SOURCE_BYTES:
                    raise ValueError("response is too large")
                body = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if len(body) + len(chunk) > _MAX_SOURCE_BYTES:
                        raise ValueError("response is too large")
                    body.extend(chunk)
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RuntimeError("request failed") from exc
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("response is not an object")
        return payload

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._config.source_timeout)
            )
        return self._session

    def _publish(self, snapshot: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass


def page_snapshot(
    snapshot: Mapping[str, Any],
    *,
    query: str,
    platform: str | None,
    status: str,
    cursor: int,
    limit: int,
) -> dict[str, Any]:
    needle = query.casefold()
    filtered: list[dict[str, Any]] = []
    for item in snapshot["items"]:
        if platform is not None and item["platform"] != platform:
            continue
        item_status = item["status"]
        if status == "attention":
            if item_status not in {"error", "disconnected", "stale"}:
                continue
        elif status == "active":
            if not item.get("activeRuns"):
                continue
        elif status != "all" and item_status != status:
            continue
        if needle:
            haystack = "\n".join(
                str(item.get(key) or "")
                for key in (
                    "displayName",
                    "scopeId",
                    "accountId",
                    "adapterInstanceId",
                    "platform",
                )
            ).casefold()
            if needle not in haystack:
                continue
        filtered.append(item)
    end = min(cursor + limit, len(filtered))
    return {
        "streamId": snapshot["streamId"],
        "generation": snapshot["generation"],
        "generatedAt": snapshot["generatedAt"],
        "degraded": snapshot["degraded"],
        "stale": snapshot["stale"],
        "sources": snapshot["sources"],
        "platforms": snapshot["platforms"],
        "items": filtered[cursor:end],
        "total": len(filtered),
        "nextCursor": str(end) if end < len(filtered) else None,
    }


def _parse_adapter_snapshot(payload: dict[str, Any], source_id: str) -> dict[str, Any]:
    adapter = payload.get("adapter")
    supplied_items = payload.get("items")
    if not isinstance(adapter, dict) or not isinstance(supplied_items, list):
        raise ValueError("adapter snapshot is malformed")
    if len(supplied_items) > _MAX_CHANNELS:
        raise ValueError("adapter snapshot has too many channels")
    normalized_adapter = {
        "id": _text(adapter, "id", 128),
        "platform": _text(adapter, "platform", 64),
        "accountId": _optional_text(adapter, "accountId", 256),
        "connected": _boolean(adapter, "connected"),
        "observedAt": _timestamp(adapter, "observedAt"),
        "error": _optional_text(adapter, "error", 2_000),
        "sourceId": source_id,
    }
    items = [
        _parse_channel(item, normalized_adapter, source_id) for item in supplied_items
    ]
    return {"adapter": normalized_adapter, "items": items}


def _parse_channel(
    value: Any, adapter: dict[str, Any], source_id: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("channel entry is malformed")
    memory = value.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("channel memory state is malformed")
    return {
        "scopeId": _text(value, "scopeId", 512),
        "platform": _text(value, "platform", 64),
        "adapterInstanceId": _text(value, "adapterInstanceId", 128),
        "accountId": _optional_text(value, "accountId", 256),
        "displayName": _text(value, "displayName", 512),
        "chatKind": _text(value, "chatKind", 64),
        "accessMode": _text(value, "accessMode", 64),
        "modelOverride": _optional_text(value, "modelOverride", 256),
        "memory": {
            "continuousEnabled": _boolean(memory, "continuousEnabled"),
            "dreamEnabled": _boolean(memory, "dreamEnabled"),
            "effectiveMode": _text(memory, "effectiveMode", 64),
            "continuousLastAttemptAt": _optional_timestamp(
                memory, "continuousLastAttemptAt"
            ),
            "continuousLastSuccessAt": _optional_timestamp(
                memory, "continuousLastSuccessAt"
            ),
            "continuousLastError": _optional_text(memory, "continuousLastError", 2_000),
            "dreamLastAttemptAt": _optional_timestamp(memory, "dreamLastAttemptAt"),
            "dreamLastSuccessAt": _optional_timestamp(memory, "dreamLastSuccessAt"),
            "dreamLastError": _optional_text(memory, "dreamLastError", 2_000),
            "pendingDocumentCount": _nonnegative_int(
                memory, "pendingDocumentCount", maximum=1_000_000
            ),
            "retryingDocumentCount": _optional_nonnegative_int(
                memory,
                "retryingDocumentCount",
                maximum=1_000_000,
            ),
            "deadLetterDocumentCount": _optional_nonnegative_int(
                memory,
                "deadLetterDocumentCount",
                maximum=1_000_000,
            ),
            "nextRetryAt": _optional_timestamp(memory, "nextRetryAt"),
            "scanCursor": _optional_external_id(memory, "scanCursor"),
            "scanWatermarkAt": _optional_timestamp(memory, "scanWatermarkAt"),
            "retainWatermarkAt": _optional_timestamp(
                memory,
                "retainWatermarkAt",
            ),
            "retainedSourceAt": _optional_timestamp(memory, "retainedSourceAt"),
            "lastIngestedAt": _optional_timestamp(memory, "lastIngestedAt"),
        },
        "lastObservedAt": _optional_timestamp(value, "lastObservedAt"),
        "activeRuns": _parse_run_list(value.get("activeRuns")),
        "errors": _parse_errors(value.get("errors")),
        "updatedAt": _optional_timestamp(value, "updatedAt"),
        "adapter": dict(adapter),
        "adapterSourceId": source_id,
    }


def _parse_models(payload: dict[str, Any], _source_id: str) -> dict[str, Any]:
    default_model = payload.get("defaultModel")
    models = payload.get("models")
    if (
        not isinstance(default_model, str)
        or not 1 <= len(default_model) <= 256
        or not isinstance(models, list)
        or len(models) > 10_000
        or not all(isinstance(item, str) and 1 <= len(item) <= 256 for item in models)
    ):
        raise ValueError("model catalog is malformed")
    return {"defaultModel": default_model, "models": list(models)}


def _parse_active_runs(
    payload: dict[str, Any], _source_id: str
) -> list[dict[str, Any]]:
    return _parse_run_list(payload.get("items"))


def _parse_run_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > _MAX_RUNS:
        raise ValueError("active runs are malformed")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("active run is malformed")
        run_id = _text(item, "runId", 128)
        result.append(
            {
                "runId": run_id,
                "sessionId": _optional_text(item, "sessionId", 128),
                "scopeId": _optional_text(item, "scopeId", 512),
                "adapterInstanceId": _optional_text(item, "adapterInstanceId", 128),
                "modelId": _optional_text(item, "modelId", 256)
                or _optional_text(item, "model", 256),
                "phase": _optional_text(item, "phase", 64)
                or _optional_text(item, "status", 64)
                or "active",
                "currentTool": _optional_text(item, "currentTool", 256),
                "startedAt": _optional_timestamp(item, "startedAt"),
                "updatedAt": _optional_timestamp(item, "updatedAt"),
            }
        )
    return result


def _parse_errors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > _MAX_ERRORS:
        raise ValueError("channel errors are malformed")
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str) and 1 <= len(item) <= 2_000:
            result.append(
                {
                    "code": None,
                    "message": item,
                    "component": None,
                    "occurredAt": None,
                    "runId": None,
                }
            )
            continue
        if not isinstance(item, dict):
            raise ValueError("channel error is malformed")
        component = _optional_text(item, "component", 128)
        if component is None:
            component = _optional_text(item, "source", 128)
        occurred_at = _optional_timestamp(item, "occurredAt")
        if occurred_at is None:
            occurred_at = _optional_timestamp(item, "at")
        result.append(
            {
                "code": _optional_text(item, "code", 128),
                "message": _text(item, "message", 2_000),
                "component": component,
                "occurredAt": occurred_at,
                "runId": _optional_text(item, "runId", 128),
            }
        )
    return result


def _parse_banks(payload: dict[str, Any], _source_id: str) -> list[dict[str, Any]]:
    supplied = payload.get("banks")
    if not isinstance(supplied, list) or len(supplied) > _MAX_CHANNELS:
        raise ValueError("bank list is malformed")
    result = []
    for item in supplied:
        if not isinstance(item, dict):
            raise ValueError("bank is malformed")
        bank_id = item.get("bank_id")
        if not isinstance(bank_id, str) or not 1 <= len(bank_id) <= 512:
            raise ValueError("bank is malformed")
        result.append(
            {
                "bankId": bank_id,
                "name": _optional_text(item, "name", 512),
                "factCount": _nested_count(
                    item,
                    "fact_count",
                    "facts_count",
                    "factCount",
                    "memory_count",
                ),
                "observationCount": _nested_count(
                    item, "observation_count", "observations_count", "observationCount"
                ),
                "lastDocumentAt": _first_optional_timestamp(
                    item, "last_document_at", "lastDocumentAt"
                ),
            }
        )
    return result


def _empty_bank_stats() -> dict[str, Any]:
    return {
        "lastConsolidatedAt": None,
        "pendingConsolidationCount": None,
        "failedConsolidationCount": None,
        "pendingOperationCount": None,
        "failedOperationCount": None,
    }


def _parse_bank_stats(payload: dict[str, Any], bank_id: str) -> dict[str, Any]:
    supplied_bank_id = payload.get("bank_id", payload.get("bankId"))
    if supplied_bank_id != bank_id:
        raise ValueError("bank stats are malformed")
    return {
        "lastConsolidatedAt": _first_optional_timestamp(
            payload,
            "last_consolidated_at",
            "lastConsolidatedAt",
        ),
        "pendingConsolidationCount": _nested_count(
            payload,
            "pending_consolidation",
            "pendingConsolidation",
        ),
        "failedConsolidationCount": _nested_count(
            payload,
            "failed_consolidation",
            "failedConsolidation",
        ),
        "pendingOperationCount": _nested_count(
            payload,
            "pending_operations",
            "pendingOperations",
        ),
        "failedOperationCount": _nested_count(
            payload,
            "failed_operations",
            "failedOperations",
        ),
    }


def _build_snapshot_content(
    *,
    adapters: list[dict[str, Any]],
    models: dict[str, Any] | None,
    runs: list[dict[str, Any]] | None,
    banks: list[dict[str, Any]] | None,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_id = {source["id"]: source for source in sources}
    runs_by_channel: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs or []:
        scope_id = run.get("scopeId")
        adapter_instance_id = run.get("adapterInstanceId")
        if scope_id and adapter_instance_id:
            runs_by_channel.setdefault((adapter_instance_id, scope_id), []).append(run)
    bank_by_id = {bank["bankId"]: bank for bank in banks or []}
    banks_available = banks is not None
    default_model = models.get("defaultModel") if models else None
    items: list[dict[str, Any]] = []
    for adapter_snapshot in adapters:
        for supplied in adapter_snapshot["items"]:
            item = dict(supplied)
            joined_runs = _deduplicate_runs(
                [
                    *item["activeRuns"],
                    *runs_by_channel.get(
                        (item["adapterInstanceId"], item["scopeId"]), []
                    ),
                ]
            )
            item["activeRuns"] = joined_runs
            item["model"] = item["modelOverride"] or default_model
            item["modelSource"] = (
                "override"
                if item["modelOverride"]
                else "default"
                if default_model
                else "unavailable"
            )
            bank = bank_by_id.get(item["scopeId"])
            item["bank"] = (
                {"status": "PRESENT", **bank}
                if bank
                else {
                    "status": "MISSING" if banks_available else "UNAVAILABLE",
                    "bankId": item["scopeId"],
                    "name": None,
                    "factCount": None,
                    "observationCount": None,
                    "lastDocumentAt": None,
                    **_empty_bank_stats(),
                }
            )
            source = source_by_id.get(item["adapterSourceId"], {})
            stale = source.get("status") == "stale"
            item["stale"] = stale
            if item["errors"] or _memory_errors(item["memory"]):
                item["status"] = "error"
            elif not item["adapter"]["connected"]:
                item["status"] = "disconnected"
            elif stale:
                item["status"] = "stale"
            elif joined_runs:
                item["status"] = "active"
            else:
                item["status"] = "healthy"
            items.append(item)
    items.sort(
        key=lambda item: (
            item["platform"].casefold(),
            item["displayName"].casefold(),
            item["adapterInstanceId"].casefold(),
            item["scopeId"],
        )
    )
    degraded = any(source["status"] != "ok" for source in sources)
    stale = any(source["status"] == "stale" for source in sources)
    return {
        "degraded": degraded,
        "stale": stale,
        "sources": sources,
        "platforms": sorted({item["platform"] for item in items}),
        "items": items,
        "total": len(items),
    }


def _stable_snapshot_content(content: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for source in content["sources"]:
        stable = {key: value for key, value in source.items() if key != "lastSuccessAt"}
        if isinstance(stable.get("adapter"), dict):
            stable["adapter"] = {
                key: value
                for key, value in stable["adapter"].items()
                if key != "observedAt"
            }
        sources.append(stable)
    items = []
    for item in content["items"]:
        adapter = {
            key: value for key, value in item["adapter"].items() if key != "observedAt"
        }
        items.append({**item, "adapter": adapter})
    return {**content, "sources": sources, "items": items}


def _source_status(
    source_id: str,
    kind: str,
    status: str,
    succeeded_at: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "kind": kind,
        "status": status,
        "lastSuccessAt": succeeded_at,
        "error": error,
    }


def _deduplicate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for run in runs:
        by_id[run["runId"]] = run
    return sorted(
        by_id.values(), key=lambda run: run.get("startedAt") or "", reverse=True
    )


def _memory_errors(memory: Mapping[str, Any]) -> list[str]:
    return [
        value
        for value in (
            memory.get("continuousLastError"),
            memory.get("dreamLastError"),
        )
        if value
    ]


def _nested_count(value: dict[str, Any], *keys: str) -> int | None:
    candidates: list[Any] = [value.get(key) for key in keys]
    stats = value.get("stats")
    if isinstance(stats, dict):
        candidates.extend(stats.get(key) for key in keys)
    for candidate in candidates:
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate >= 0
        ):
            return candidate
    return None


def _first_optional_timestamp(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if value.get(key) is not None:
            return _optional_timestamp(value, key)
    return None


def _text(value: Mapping[str, Any], key: str, maximum: int) -> str:
    supplied = value.get(key)
    if not isinstance(supplied, str) or not 1 <= len(supplied) <= maximum:
        raise ValueError(f"{key} is malformed")
    return supplied


def _optional_text(value: Mapping[str, Any], key: str, maximum: int) -> str | None:
    supplied = value.get(key)
    if supplied is None:
        return None
    if not isinstance(supplied, str) or not 1 <= len(supplied) <= maximum:
        raise ValueError(f"{key} is malformed")
    return supplied


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    supplied = value.get(key)
    if not isinstance(supplied, bool):
        raise ValueError(f"{key} is malformed")
    return supplied


def _nonnegative_int(value: Mapping[str, Any], key: str, *, maximum: int) -> int:
    supplied = value.get(key)
    if (
        not isinstance(supplied, int)
        or isinstance(supplied, bool)
        or not 0 <= supplied <= maximum
    ):
        raise ValueError(f"{key} is malformed")
    return supplied


def _optional_nonnegative_int(
    value: Mapping[str, Any],
    key: str,
    *,
    maximum: int,
) -> int:
    if value.get(key) is None:
        return 0
    return _nonnegative_int(value, key, maximum=maximum)


def _optional_external_id(
    value: Mapping[str, Any],
    key: str,
) -> str | int | None:
    supplied = value.get(key)
    if supplied is None:
        return None
    if isinstance(supplied, bool) or not isinstance(supplied, (str, int)):
        raise ValueError(f"{key} is malformed")
    if isinstance(supplied, str) and not 1 <= len(supplied) <= 512:
        raise ValueError(f"{key} is malformed")
    return supplied


def _timestamp(value: Mapping[str, Any], key: str) -> str:
    supplied = _text(value, key, 64)
    _parse_timestamp(supplied)
    return supplied


def _optional_timestamp(value: Mapping[str, Any], key: str) -> str | None:
    supplied = value.get(key)
    if supplied is None:
        return None
    if not isinstance(supplied, str) or len(supplied) > 64:
        raise ValueError(f"{key} is malformed")
    _parse_timestamp(supplied)
    return supplied


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _safe_error(exc: Exception) -> str:
    return f"Upstream request failed ({exc.__class__.__name__})"


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
