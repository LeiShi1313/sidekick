from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from yarl import URL

from sidekick.memory_gateway import MemoryGateway, MemoryGatewaySettings


TOKEN = "memory-gateway-token-that-is-long-enough"


async def _start(application: web.Application) -> TestServer:
    server = TestServer(application)
    await server.start_server()
    return server


def _settings(upstream_url: str) -> MemoryGatewaySettings:
    return MemoryGatewaySettings(upstream_url=upstream_url, token=TOKEN)


def test_settings_require_a_strong_token_and_fixed_http_upstream() -> None:
    with pytest.raises(ValueError, match="at least 24"):
        MemoryGatewaySettings(upstream_url="http://memory-api:8888", token="short")
    with pytest.raises(ValueError, match="http or https"):
        MemoryGatewaySettings(upstream_url="file:///tmp/memory", token=TOKEN)
    with pytest.raises(ValueError, match="origin"):
        MemoryGatewaySettings(
            upstream_url="http://user:pass@memory-api:8888/base?query=yes",
            token=TOKEN,
        )


@pytest.mark.asyncio
async def test_v1_requires_exact_bearer_token() -> None:
    upstream = web.Application()

    async def banks(_request: web.Request) -> web.Response:
        return web.json_response({})

    upstream.router.add_get("/v1/default/banks", banks)
    upstream_server = await _start(upstream)
    gateway_server = await _start(
        MemoryGateway(_settings(str(upstream_server.make_url("/")))).application
    )
    try:
        async with aiohttp.ClientSession() as session:
            for headers in (
                {},
                {"Authorization": "Bearer wrong-memory-token-that-is-long-enough"},
                {"Authorization": f"Basic {TOKEN}"},
            ):
                response = await session.get(
                    gateway_server.make_url("/v1/default/banks"), headers=headers
                )
                assert response.status == 401
                assert await response.json() == {
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Authentication required",
                    }
                }
                assert response.headers["Cache-Control"] == "no-store"
                assert response.headers["WWW-Authenticate"] == "Bearer"
    finally:
        await gateway_server.close()
        await upstream_server.close()


@pytest.mark.asyncio
async def test_proxy_forwards_body_query_and_status_but_not_credentials() -> None:
    received: dict[str, object] = {}

    async def upstream_handler(request: web.Request) -> web.Response:
        received.update(
            method=request.method,
            path=request.path,
            query=list(request.query.items()),
            body=await request.json(),
            authorization=request.headers.get("Authorization"),
            cookie=request.headers.get("Cookie"),
            forwarded=request.headers.get("Forwarded"),
            content_type=request.headers.get("Content-Type"),
            accept=request.headers.get("Accept"),
        )
        return web.json_response(
            {"error": "busy"},
            status=429,
            headers={
                "Retry-After": "17",
                "X-Upstream-Secret": "must-not-cross",
            },
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/default/banks/{bank}/memories", upstream_handler)
    upstream_server = await _start(upstream)
    gateway_server = await _start(
        MemoryGateway(_settings(str(upstream_server.make_url("/")))).application
    )
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                gateway_server.make_url(
                    "/v1/default/banks/chat%3Aengineering/memories?limit=2&tag=a"
                ),
                json={"items": [{"content": "hello"}]},
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Cookie": "private=session",
                    "Forwarded": "for=10.0.0.1",
                    "Accept": "application/json",
                    "X-Private": "must-not-cross",
                },
            )
            assert response.status == 429
            assert await response.json() == {
                "error": {
                    "code": "UPSTREAM_ERROR",
                    "message": "Memory request failed",
                }
            }
            assert response.headers["Retry-After"] == "17"
            assert "X-Upstream-Secret" not in response.headers
            assert response.headers["Cache-Control"] == "no-store"
        assert received == {
            "method": "POST",
            "path": "/v1/default/banks/chat:engineering/memories",
            "query": [("limit", "2"), ("tag", "a")],
            "body": {"items": [{"content": "hello"}]},
            "authorization": None,
            "cookie": None,
            "forwarded": None,
            "content_type": "application/json",
            "accept": "application/json",
        }
    finally:
        await gateway_server.close()
        await upstream_server.close()


@pytest.mark.asyncio
async def test_health_is_public_minimal_and_does_not_proxy_upstream_details() -> None:
    upstream = web.Application()

    async def health(_request: web.Request) -> web.Response:
        return web.json_response(
            {"status": "healthy", "database": "postgres.internal", "version": "1"}
        )

    upstream.router.add_get("/health", health)
    upstream_server = await _start(upstream)
    gateway_server = await _start(
        MemoryGateway(_settings(str(upstream_server.make_url("/")))).application
    )
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(gateway_server.make_url("/health")) as response,
        ):
            assert response.status == 200
            assert await response.json() == {"status": "ok"}
            assert response.headers["Cache-Control"] == "no-store"
    finally:
        await gateway_server.close()
        await upstream_server.close()


@pytest.mark.asyncio
async def test_oversized_upstream_response_fails_without_partial_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sidekick.memory_gateway as gateway_module

    monkeypatch.setattr(gateway_module, "_MAX_BODY_BYTES", 32)
    upstream = web.Application()

    async def banks(_request: web.Request) -> web.Response:
        return web.Response(body=b"x" * 33)

    upstream.router.add_get("/v1/default/banks", banks)
    upstream_server = await _start(upstream)
    gateway_server = await _start(
        MemoryGateway(_settings(str(upstream_server.make_url("/")))).application
    )
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                gateway_server.make_url("/v1/default/banks"),
                headers={"Authorization": f"Bearer {TOKEN}"},
            ) as response,
        ):
            assert response.status == 502
            assert await response.json() == {
                "error": {
                    "code": "UPSTREAM_UNAVAILABLE",
                    "message": "Memory service unavailable",
                }
            }
    finally:
        await gateway_server.close()
        await upstream_server.close()


@pytest.mark.asyncio
async def test_gateway_rejects_non_v1_paths_and_unsupported_methods() -> None:
    upstream = web.Application()

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"private": "upstream health"})

    upstream.router.add_get("/health", health)
    upstream_server = await _start(upstream)
    gateway_server = await _start(
        MemoryGateway(_settings(str(upstream_server.make_url("/")))).application
    )
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.get(gateway_server.make_url("/internal/config"))
            assert response.status == 404
            traversal = URL(
                f"{gateway_server.make_url('/')}v1/%2e%2e/health",
                encoded=True,
            )
            response = await session.get(
                traversal,
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
            assert response.status == 404
            response = await session.options(
                gateway_server.make_url("/v1/default/banks"),
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
            assert response.status == 405
    finally:
        await gateway_server.close()
        await upstream_server.close()
