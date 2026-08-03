from __future__ import annotations

import asyncio
import json

import aiohttp
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
import pytest

from agent_playground.app import (
    CHANNELS_KEY,
    PlaygroundSettings,
    _channel_events,
    create_app,
)
from agent_playground.channels import (
    ChannelDashboard,
    ChannelDashboardConfig,
    page_snapshot,
    parse_adapter_urls,
)


async def _start(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _channel_dependencies():
    state = {
        "name": "Operations room",
        "updated_at": "2026-07-31T12:00:00Z",
        "memory_available": True,
        "continuous_error": None,
        "model_requests": 0,
        "stats_requests": 0,
    }
    adapter = web.Application()

    async def channels(request: web.Request) -> web.Response:
        assert request.headers["Authorization"] == "Bearer private-pi-token"
        return web.json_response(
            {
                "adapter": {
                    "id": "onebot-main",
                    "platform": "qq",
                    "accountId": None,
                    "connected": True,
                    "observedAt": "2026-07-31T12:00:00Z",
                },
                "items": [
                    {
                        "scopeId": "qq:group:42",
                        "platform": "qq",
                        "adapterInstanceId": "onebot-main",
                        "accountId": "10001",
                        "displayName": state["name"],
                        "chatKind": "GROUP",
                        "accessMode": "OPEN",
                        "modelOverride": None,
                        "memory": {
                            "continuousEnabled": True,
                            "dreamEnabled": False,
                            "effectiveMode": "CONTINUOUS",
                            "continuousLastAttemptAt": "2026-07-31T11:59:00Z",
                            "continuousLastSuccessAt": "2026-07-31T11:59:01Z",
                            "continuousLastError": state["continuous_error"],
                            "dreamLastAttemptAt": None,
                            "dreamLastSuccessAt": None,
                            "dreamLastError": None,
                            "pendingDocumentCount": 2,
                            "retryingDocumentCount": 1,
                            "deadLetterDocumentCount": 0,
                            "nextRetryAt": "2026-07-31T12:00:30Z",
                            "scanCursor": 101,
                            "scanWatermarkAt": "2026-07-31T11:59:30Z",
                            "retainWatermarkAt": "2026-07-31T11:59:01Z",
                            "retainedSourceAt": "2026-07-31T11:58:58Z",
                            "lastIngestedAt": "2026-07-31T11:59:01Z",
                        },
                        "lastObservedAt": None,
                        "activeRuns": [],
                        "errors": [],
                        "updatedAt": state["updated_at"],
                    }
                ],
            }
        )

    async def broken(_request: web.Request) -> web.Response:
        return web.json_response({"error": "offline"}, status=503)

    adapter.router.add_get("/v1/channels", channels)
    adapter.router.add_get("/broken", broken)
    adapter_runner, adapter_url = await _start(adapter)

    pi = web.Application()

    async def models(request: web.Request) -> web.Response:
        assert request.headers["Authorization"] == "Bearer private-pi-token"
        state["model_requests"] += 1
        return web.json_response(
            {"defaultModel": "gpt-5.6-sol", "models": ["gpt-5.6-sol"]}
        )

    async def runs(request: web.Request) -> web.Response:
        assert request.headers["Authorization"] == "Bearer private-pi-token"
        assert dict(request.query) == {"status": "active"}
        return web.json_response(
            {
                "items": [
                    {
                        "runId": "11111111-1111-4111-8111-111111111111",
                        "sessionId": "session-live",
                        "scopeId": "qq:group:42",
                        "adapterInstanceId": "onebot-main",
                        "modelId": "gpt-5.6-sol",
                        "phase": "model_running",
                        "currentTool": None,
                        "startedAt": "2026-07-31T11:59:59Z",
                        "updatedAt": "2026-07-31T12:00:00Z",
                    },
                    {
                        "runId": "22222222-2222-4222-8222-222222222222",
                        "sessionId": "wrong-instance",
                        "scopeId": "qq:group:42",
                        "adapterInstanceId": "onebot-other",
                        "modelId": "gpt-5.6-sol",
                        "phase": "queued",
                        "currentTool": None,
                        "startedAt": "2026-07-31T11:59:58Z",
                        "updatedAt": "2026-07-31T11:59:58Z",
                    },
                ],
                "total": 2,
            }
        )

    pi.router.add_get("/v1/models", models)
    pi.router.add_get("/v1/runs", runs)
    pi_runner, pi_url = await _start(pi)

    memory = web.Application()

    async def banks(_request: web.Request) -> web.Response:
        if not state["memory_available"]:
            return web.json_response({"error": "offline"}, status=503)
        return web.json_response(
            {
                "banks": [
                    {
                        "bank_id": "qq:group:42",
                        "name": "Operations memory",
                        "fact_count": 37,
                        "observation_count": 9,
                        "last_document_at": "2026-07-31T11:59:01Z",
                    }
                ]
            }
        )

    async def stats(request: web.Request) -> web.Response:
        assert request.match_info["bank_id"] == "qq:group:42"
        state["stats_requests"] += 1
        return web.json_response(
            {
                "bank_id": "qq:group:42",
                "last_consolidated_at": "2026-07-31T11:59:04Z",
                "pending_consolidation": 2,
                "failed_consolidation": 0,
                "pending_operations": 1,
                "failed_operations": 0,
            }
        )

    memory.router.add_get("/v1/default/banks", banks)
    memory.router.add_get("/v1/default/banks/{bank_id}/stats", stats)
    memory_runner, memory_url = await _start(memory)
    return (
        [adapter_runner, pi_runner, memory_runner],
        adapter_url,
        pi_url,
        memory_url,
        state,
    )


def _settings(adapter_url: str, pi_url: str, memory_url: str) -> PlaygroundSettings:
    return PlaygroundSettings(
        memory_url=memory_url,
        pi_url=pi_url,
        pi_token="private-pi-token",
        channel_adapter_urls=(
            ("onebot", f"{adapter_url}/v1/channels"),
            ("offline", f"{adapter_url}/broken"),
        ),
        channel_poll_interval=60,
        channel_source_timeout=1,
        channel_bank_cache_ttl=15,
    )


def test_adapter_url_configuration_accepts_json_and_mapping():
    expected = (
        ("onebot", "http://onebot:8781/v1/channels"),
        ("wechat-peer", "http://wechat:8781/v1/channels"),
    )
    assert (
        parse_adapter_urls(
            '{"onebot":"http://onebot:8781/v1/channels",'
            '"wechat-peer":"http://wechat:8781/v1/channels"}'
        )
        == expected
    )
    assert (
        parse_adapter_urls(
            "onebot=http://onebot:8781/v1/channels,"
            "wechat-peer=http://wechat:8781/v1/channels"
        )
        == expected
    )
    with pytest.raises(ValueError, match="id=url"):
        parse_adapter_urls("http://onebot:8781/v1/channels")


@pytest.mark.parametrize(
    "source_id",
    ["pi-models", "pi-runs", "hindsight", "hindsight-stats"],
)
def test_adapter_url_configuration_rejects_internal_source_ids(source_id: str):
    with pytest.raises(ValueError, match="reserved"):
        parse_adapter_urls(f"{source_id}=http://adapter:8781/v1/channels")
    with pytest.raises(ValueError, match="reserved"):
        PlaygroundSettings(
            pi_token="token",
            channel_adapter_urls=((source_id, "http://adapter:8781/v1/channels"),),
        )


def test_active_filter_includes_unhealthy_channel_with_active_run():
    snapshot = {
        "streamId": "stream-a",
        "generation": 1,
        "generatedAt": "2026-07-31T12:00:00Z",
        "degraded": False,
        "stale": False,
        "sources": [],
        "platforms": ["qq"],
        "items": [
            {
                "scopeId": "qq:group:42",
                "platform": "qq",
                "adapterInstanceId": "onebot-main",
                "accountId": "10001",
                "displayName": "Operations room",
                "status": "error",
                "activeRuns": [{"runId": "live-run"}],
            }
        ],
    }

    page = page_snapshot(
        snapshot,
        query="",
        platform=None,
        status="active",
        cursor=0,
        limit=100,
    )

    assert page["total"] == 1
    assert page["items"][0]["status"] == "error"


class _ChunkedContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.consumed = 0

    async def iter_chunked(self, _size: int):
        for chunk in self._chunks:
            self.consumed += 1
            yield chunk


class _ChunkedResponse:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, content: _ChunkedContent):
        self.content = content

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _ChunkedSession:
    closed = False

    def __init__(self, response: _ChunkedResponse):
        self._response = response

    def get(self, *_args, **_kwargs):
        return self._response


@pytest.mark.asyncio
async def test_source_response_stops_reading_at_size_limit(monkeypatch):
    content = _ChunkedContent([b"123", b"456", b"must-not-be-read"])
    dashboard = ChannelDashboard(
        ChannelDashboardConfig(
            adapter_urls=(("adapter", "http://adapter/v1/channels"),),
            pi_url="http://pi",
            memory_url="http://memory",
            token="token",
        )
    )
    monkeypatch.setattr("agent_playground.channels._MAX_SOURCE_BYTES", 5)
    monkeypatch.setattr(
        dashboard,
        "_get_session",
        lambda: _ChunkedSession(_ChunkedResponse(content)),
    )

    with pytest.raises(ValueError, match="too large"):
        await dashboard._json("http://adapter/v1/channels", authenticated=True)

    assert content.consumed == 2


@pytest.mark.asyncio
async def test_hindsight_bank_stats_keep_last_good_value_after_refresh_failure(
    monkeypatch,
):
    dashboard = ChannelDashboard(
        ChannelDashboardConfig(
            adapter_urls=(("adapter", "http://adapter/v1/channels"),),
            pi_url="http://pi",
            memory_url="http://memory",
            token="token",
            bank_cache_ttl=15,
        )
    )
    calls = []

    async def load(url, *, authenticated):
        calls.append((url, authenticated))
        if len(calls) > 1:
            raise RuntimeError("stats unavailable")
        return {
            "bank_id": "wechat:account:wx/id:chat:group",
            "last_consolidated_at": "2026-07-31T11:59:04Z",
            "pending_consolidation": 0,
            "failed_consolidation": 0,
            "pending_operations": 0,
            "failed_operations": 0,
        }

    monkeypatch.setattr(dashboard, "_json", load)
    banks = [{"bankId": "wechat:account:wx/id:chat:group"}]

    first, first_status = await dashboard._load_bank_stats(banks)
    dashboard._bank_stats_deadlines[banks[0]["bankId"]] = 0
    second, second_status = await dashboard._load_bank_stats(banks)

    assert calls[0] == (
        "http://memory/v1/default/banks/"
        "wechat%3Aaccount%3Awx%2Fid%3Achat%3Agroup/stats",
        False,
    )
    assert second == first
    assert first_status["status"] == "ok"
    assert second_status["status"] == "stale"


class _BlockingDashboard:
    def __init__(self):
        self.subscribers: set[asyncio.Queue[dict[str, object]]] = set()
        self.snapshot_started = asyncio.Event()

    def subscribe(self):
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        self.subscribers.discard(queue)

    async def snapshot(self):
        self.snapshot_started.set()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_channel_event_subscription_is_cleaned_up_when_initial_snapshot_cancels():
    dashboard = _BlockingDashboard()
    app = web.Application()
    app[CHANNELS_KEY] = dashboard
    request = make_mocked_request("GET", "/api/channel-events", app=app)

    task = asyncio.create_task(_channel_events(request))
    await dashboard.snapshot_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert dashboard.subscribers == set()


@pytest.mark.asyncio
async def test_channels_aggregate_sources_filter_and_stream_changes():
    runners, adapter_url, pi_url, memory_url, state = await _channel_dependencies()
    app = create_app(_settings(adapter_url, pi_url, memory_url))
    playground_runner, playground_url = await _start(app)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{playground_url}/api/channels?status=active"
            ) as response:
                payload = await response.json()
            assert response.status == 200
            assert payload["degraded"] is True
            assert payload["stale"] is False
            assert payload["total"] == 1
            assert isinstance(payload["streamId"], str)
            assert payload["streamId"]
            item = payload["items"][0]
            assert item["status"] == "active"
            assert item["lastObservedAt"] is None
            assert item["adapter"]["accountId"] is None
            assert item["model"] == "gpt-5.6-sol"
            assert item["bank"] == {
                "status": "PRESENT",
                "bankId": "qq:group:42",
                "name": "Operations memory",
                "factCount": 37,
                "observationCount": 9,
                "lastDocumentAt": "2026-07-31T11:59:01Z",
                "lastConsolidatedAt": "2026-07-31T11:59:04Z",
                "pendingConsolidationCount": 2,
                "failedConsolidationCount": 0,
                "pendingOperationCount": 1,
                "failedOperationCount": 0,
            }
            assert item["memory"]["scanCursor"] == 101
            assert item["memory"]["scanWatermarkAt"] == "2026-07-31T11:59:30Z"
            assert item["memory"]["retryingDocumentCount"] == 1
            assert [run["sessionId"] for run in item["activeRuns"]] == ["session-live"]
            sources = {source["id"]: source for source in payload["sources"]}
            assert sources["onebot"]["status"] == "ok"
            assert sources["offline"]["status"] == "unavailable"
            assert sources["hindsight-stats"]["status"] == "ok"

            async with session.get(
                f"{playground_url}/api/channels?platform=qq&q=operations&limit=1&cursor=0"
            ) as response:
                filtered = await response.json()
            assert response.status == 200
            assert filtered["total"] == 1
            assert filtered["nextCursor"] is None

            async with session.get(
                f"{playground_url}/api/channels?status=not-a-status"
            ) as response:
                invalid = await response.json()
            assert response.status == 400
            assert invalid["error"]["code"] == "INVALID_REQUEST"

            async with session.get(f"{playground_url}/api/channel-events") as response:
                assert response.status == 200
                assert response.headers["Content-Type"].startswith("text/event-stream")
                assert response.headers["X-Accel-Buffering"] == "no"
                retry = await asyncio.wait_for(
                    response.content.readuntil(b"\n\n"), timeout=2
                )
                initial = await asyncio.wait_for(
                    response.content.readuntil(b"\n\n"), timeout=2
                )
                assert retry == b"retry: 3000\n\n"
                assert b"event: snapshot\n" in initial

                state["name"] = "Operations room updated"
                state["updated_at"] = "2026-07-31T12:01:00Z"
                await app[CHANNELS_KEY].refresh()
                changed = await asyncio.wait_for(
                    response.content.readuntil(b"\n\n"), timeout=2
                )
                data_line = next(
                    line for line in changed.splitlines() if line.startswith(b"data: ")
                )
                streamed = json.loads(data_line.removeprefix(b"data: "))
                assert streamed["streamId"] == payload["streamId"]
                assert streamed["generation"] > payload["generation"]
                assert streamed["items"][0]["displayName"] == state["name"]
                assert state["model_requests"] == 1
                assert state["stats_requests"] == 1

            async with session.get(f"{playground_url}/") as response:
                markup = await response.text()
            assert 'id="channels-view"' in markup
            assert "Hindsight bank / facts" in markup
            assert "Pipeline progress" in markup
            async with session.get(f"{playground_url}/app.js") as response:
                script = await response.text()
            assert "new EventSource" in script
            assert "streamId" in script
            assert "innerHTML" not in script
    finally:
        await playground_runner.cleanup()
        for runner in runners:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_channel_bank_is_unavailable_without_hindsight_cache():
    runners, adapter_url, pi_url, memory_url, state = await _channel_dependencies()
    state["memory_available"] = False
    state["updated_at"] = None
    playground_runner, playground_url = await _start(
        create_app(_settings(adapter_url, pi_url, memory_url))
    )
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{playground_url}/api/channels") as response,
        ):
            payload = await response.json()
        assert response.status == 200
        assert payload["items"][0]["updatedAt"] is None
        assert payload["items"][0]["bank"]["status"] == "UNAVAILABLE"
        hindsight = next(
            source for source in payload["sources"] if source["id"] == "hindsight"
        )
        assert hindsight["status"] == "unavailable"
        stats = next(
            source
            for source in payload["sources"]
            if source["id"] == "hindsight-stats"
        )
        assert stats["status"] == "unavailable"
    finally:
        await playground_runner.cleanup()
        for runner in runners:
            await runner.cleanup()
