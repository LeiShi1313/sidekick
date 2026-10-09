from __future__ import annotations

import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from sidekick.memory_egress_gateway import (
    MemoryEgressGateway,
    MemoryEgressGatewaySettings,
)


INTERNAL_TOKEN = "memory-egress-token-that-is-long-enough"


async def _start(application: web.Application) -> TestServer:
    server = TestServer(application)
    await server.start_server()
    return server


def _settings(llm_url: str, embedding_url: str) -> MemoryEgressGatewaySettings:
    return MemoryEgressGatewaySettings(
        llm_upstream_url=llm_url,
        llm_api_key="real-llm-provider-key",
        embedding_upstream_url=embedding_url,
        embedding_api_key="real-embedding-provider-key",
        internal_token=INTERNAL_TOKEN,
    )


def test_settings_require_fixed_upstreams_and_a_strong_internal_token() -> None:
    with pytest.raises(ValueError, match="at least 24"):
        MemoryEgressGatewaySettings(
            llm_upstream_url="https://provider.example/v1",
            llm_api_key="provider-key",
            embedding_upstream_url="http://ollama:11434/v1",
            embedding_api_key="embedding-key",
            internal_token="short",
        )
    with pytest.raises(ValueError, match="http or https"):
        _settings("file:///tmp/provider", "http://ollama:11434/v1")
    with pytest.raises(ValueError, match="credentials or query"):
        _settings(
            "https://user:pass@provider.example/v1?private=yes",
            "http://ollama:11434/v1",
        )


@pytest.mark.asyncio
async def test_only_fixed_authenticated_llm_and_embedding_routes_are_forwarded() -> None:
    received: list[dict[str, object]] = []

    async def capture(request: web.Request) -> web.Response:
        received.append(
            {
                "path": request.path,
                "authorization": request.headers.get("Authorization"),
                "cookie": request.headers.get("Cookie"),
                "forwarded": request.headers.get("Forwarded"),
                "body": await request.json(),
            }
        )
        return web.json_response({"ok": True})

    llm = web.Application()
    llm.router.add_post("/provider/v1/chat/completions", capture)
    embedding = web.Application()
    embedding.router.add_post("/ollama/v1/embeddings", capture)
    llm_server = await _start(llm)
    embedding_server = await _start(embedding)
    gateway_server = await _start(
        MemoryEgressGateway(
            _settings(
                str(llm_server.make_url("/provider/v1")),
                str(embedding_server.make_url("/ollama/v1")),
            )
        ).application
    )
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {INTERNAL_TOKEN}",
                "Cookie": "private=session",
                "Forwarded": "for=127.0.0.1",
                "X-Private": "must-not-cross",
            }
            llm_response = await session.post(
                gateway_server.make_url("/llm/v1/chat/completions"),
                json={"messages": [{"role": "user", "content": "hello"}]},
                headers=headers,
            )
            embedding_response = await session.post(
                gateway_server.make_url("/embeddings/v1/embeddings"),
                json={"input": "hello"},
                headers=headers,
            )
            assert llm_response.status == 200
            assert embedding_response.status == 200

        assert received == [
            {
                "path": "/provider/v1/chat/completions",
                "authorization": "Bearer real-llm-provider-key",
                "cookie": None,
                "forwarded": None,
                "body": {
                    "messages": [{"role": "user", "content": "hello"}]
                },
            },
            {
                "path": "/ollama/v1/embeddings",
                "authorization": "Bearer real-embedding-provider-key",
                "cookie": None,
                "forwarded": None,
                "body": {"input": "hello"},
            },
        ]
    finally:
        await gateway_server.close()
        await embedding_server.close()
        await llm_server.close()


@pytest.mark.asyncio
async def test_gateway_rejects_wrong_tokens_methods_and_unlisted_paths() -> None:
    upstream = web.Application()
    upstream_server = await _start(upstream)
    gateway_server = await _start(
        MemoryEgressGateway(
            _settings(
                str(upstream_server.make_url("/v1")),
                str(upstream_server.make_url("/v1")),
            )
        ).application
    )
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                gateway_server.make_url("/llm/v1/chat/completions"),
                json={},
            )
            assert response.status == 401
            response = await session.get(
                gateway_server.make_url("/llm/v1/chat/completions"),
                headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
            )
            assert response.status == 405
            response = await session.post(
                gateway_server.make_url("/llm/v1/models"),
                json={},
                headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
            )
            assert response.status == 404
            response = await session.get(gateway_server.make_url("/health"))
            assert response.status == 200
            assert await response.json() == {"status": "ok"}
            assert response.headers["Cache-Control"] == "no-store"
    finally:
        await gateway_server.close()
        await upstream_server.close()


@pytest.mark.asyncio
async def test_upstream_errors_and_headers_do_not_cross_the_boundary() -> None:
    async def fail(_request: web.Request) -> web.Response:
        return web.json_response(
            {"private": "provider details"},
            status=429,
            headers={"Retry-After": "19", "X-Upstream-Secret": "hidden"},
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", fail)
    upstream_server = await _start(upstream)
    gateway_server = await _start(
        MemoryEgressGateway(
            _settings(
                str(upstream_server.make_url("/v1")),
                str(upstream_server.make_url("/v1")),
            )
        ).application
    )
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                gateway_server.make_url("/llm/v1/chat/completions"),
                json={},
                headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
            )
            assert response.status == 429
            assert await response.json() == {
                "error": {
                    "code": "UPSTREAM_ERROR",
                    "message": "Provider request failed",
                }
            }
            assert response.headers["Retry-After"] == "19"
            assert "X-Upstream-Secret" not in response.headers
    finally:
        await gateway_server.close()
        await upstream_server.close()


def _hedged_settings(
    llm_url: str,
    embedding_url: str,
    *,
    delay: float,
) -> MemoryEgressGatewaySettings:
    return MemoryEgressGatewaySettings(
        llm_upstream_url=llm_url,
        llm_api_key="real-llm-provider-key",
        embedding_upstream_url=embedding_url,
        embedding_api_key="real-embedding-provider-key",
        internal_token=INTERNAL_TOKEN,
        embedding_hedge_delay=delay,
    )


def test_embedding_hedge_delay_is_disabled_by_default_and_validated() -> None:
    assert _settings("https://llm.example/v1", "http://e:1/v1").embedding_hedge_delay == 0
    with pytest.raises(ValueError, match="hedge delay"):
        _hedged_settings("https://llm.example/v1", "http://e:1/v1", delay=-1)
    settings = MemoryEgressGatewaySettings.from_env(
        {
            "MEMORY_LLM_UPSTREAM_URL": "https://llm.example/v1",
            "MEMORY_LLM_UPSTREAM_API_KEY": "llm-key",
            "MEMORY_EGRESS_TOKEN": INTERNAL_TOKEN,
            "MEMORY_EMBEDDING_HEDGE_DELAY": "0.8",
        }
    )
    assert settings.embedding_hedge_delay == 0.8
    with pytest.raises(ValueError, match="MEMORY_EMBEDDING_HEDGE_DELAY"):
        MemoryEgressGatewaySettings.from_env(
            {
                "MEMORY_LLM_UPSTREAM_URL": "https://llm.example/v1",
                "MEMORY_LLM_UPSTREAM_API_KEY": "llm-key",
                "MEMORY_EGRESS_TOKEN": INTERNAL_TOKEN,
                "MEMORY_EMBEDDING_HEDGE_DELAY": "soon",
            }
        )


async def _hedge_scenario(
    attempts: list[tuple[float, int]],
    *,
    route: str = "/embeddings/v1/embeddings",
    delay: float = 0.05,
) -> tuple[int, dict[str, object], list[dict[str, object]], float]:
    """Run one gateway request whose Nth upstream attempt sleeps then replies."""
    received: list[dict[str, object]] = []

    async def upstream(request: web.Request) -> web.Response:
        index = len(received)
        received.append(await request.json())
        sleep_for, status = attempts[index]
        await asyncio.sleep(sleep_for)
        return web.json_response({"attempt": index}, status=status)

    application = web.Application()
    application.router.add_post("/v1/embeddings", upstream)
    application.router.add_post("/v1/chat/completions", upstream)
    upstream_server = await _start(application)
    gateway_server = await _start(
        MemoryEgressGateway(
            _hedged_settings(
                str(upstream_server.make_url("/v1")),
                str(upstream_server.make_url("/v1")),
                delay=delay,
            )
        ).application
    )
    try:
        async with aiohttp.ClientSession() as session:
            started = asyncio.get_running_loop().time()
            response = await session.post(
                gateway_server.make_url(route),
                json={"input": ["probe"]},
                headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
            )
            payload = await response.json()
            elapsed = asyncio.get_running_loop().time() - started
        return response.status, payload, received, elapsed
    finally:
        await gateway_server.close()
        await upstream_server.close()


@pytest.mark.asyncio
async def test_slow_embedding_attempt_is_hedged_and_first_success_wins() -> None:
    status, payload, received, elapsed = await _hedge_scenario(
        [(2.0, 200), (0.0, 200)]
    )

    assert status == 200
    assert payload == {"attempt": 1}
    assert received == [{"input": ["probe"]}, {"input": ["probe"]}]
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_fast_embedding_attempt_is_not_hedged() -> None:
    status, payload, received, _elapsed = await _hedge_scenario(
        [(0.0, 200)],
        delay=0.5,
    )

    assert status == 200
    assert payload == {"attempt": 0}
    assert len(received) == 1


@pytest.mark.asyncio
async def test_failed_hedge_does_not_replace_a_slower_success() -> None:
    status, payload, received, _elapsed = await _hedge_scenario(
        [(0.3, 200), (0.0, 500)]
    )

    assert status == 200
    assert payload == {"attempt": 0}
    assert len(received) == 2


@pytest.mark.asyncio
async def test_failed_hedged_attempts_return_one_sanitized_failure() -> None:
    status, payload, received, _elapsed = await _hedge_scenario(
        [(0.2, 503), (0.0, 500)]
    )

    assert status in {500, 503}
    assert payload["error"]["code"] == "UPSTREAM_ERROR"
    assert len(received) == 2


@pytest.mark.asyncio
async def test_llm_requests_are_never_hedged() -> None:
    status, payload, received, _elapsed = await _hedge_scenario(
        [(0.3, 200), (0.0, 200)],
        route="/llm/v1/chat/completions",
    )

    assert status == 200
    assert payload == {"attempt": 0}
    assert len(received) == 1


@pytest.mark.asyncio
async def test_cancelled_hedge_race_cancels_every_upstream_attempt() -> None:
    from sidekick.memory_egress_gateway import _first_success

    started: list[asyncio.Task[object]] = []

    async def attempt():
        started.append(asyncio.current_task())
        await asyncio.sleep(10)

    race = asyncio.create_task(_first_success(attempt, 0.01))
    await asyncio.sleep(0.05)
    race.cancel()
    with pytest.raises(asyncio.CancelledError):
        await race

    assert len(started) == 2
    assert all(task.cancelled() for task in started)
