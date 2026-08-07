from __future__ import annotations

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
