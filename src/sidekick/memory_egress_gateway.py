from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import hmac
import os
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web


_MAX_BODY_BYTES = 64 * 1024 * 1024
_PRIVATE_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def _fixed_upstream(value: str, label: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} upstream must use http or https")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} upstream cannot contain credentials or query data")
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class MemoryEgressGatewaySettings:
    llm_upstream_url: str
    llm_api_key: str
    embedding_upstream_url: str
    embedding_api_key: str
    internal_token: str
    host: str = "127.0.0.1"
    port: int = 8080
    timeout: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "llm_upstream_url",
            _fixed_upstream(self.llm_upstream_url, "LLM"),
        )
        object.__setattr__(
            self,
            "embedding_upstream_url",
            _fixed_upstream(self.embedding_upstream_url, "Embedding"),
        )
        if not self.llm_api_key:
            raise ValueError("LLM upstream API key cannot be empty")
        if not self.embedding_api_key:
            raise ValueError("Embedding upstream API key cannot be empty")
        if len(self.internal_token) < 24:
            raise ValueError("Memory egress token must contain at least 24 characters")
        if not self.host:
            raise ValueError("Memory egress host cannot be empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("Memory egress port must be between 1 and 65535")
        if self.timeout <= 0:
            raise ValueError("Memory egress timeout must be positive")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> MemoryEgressGatewaySettings:
        values = os.environ if environ is None else environ
        try:
            port = int(values.get("MEMORY_EGRESS_PORT", "8080").strip())
        except ValueError as exc:
            raise ValueError("MEMORY_EGRESS_PORT must be an integer") from exc
        try:
            timeout = float(values.get("MEMORY_EGRESS_TIMEOUT", "300").strip())
        except ValueError as exc:
            raise ValueError("MEMORY_EGRESS_TIMEOUT must be numeric") from exc
        return cls(
            llm_upstream_url=values.get("MEMORY_LLM_UPSTREAM_URL", "").strip(),
            llm_api_key=values.get("MEMORY_LLM_UPSTREAM_API_KEY", "").strip(),
            embedding_upstream_url=values.get(
                "MEMORY_EMBEDDING_UPSTREAM_URL",
                "http://ollama-embedding-ollama-1:11434/v1",
            ).strip(),
            embedding_api_key=values.get(
                "MEMORY_EMBEDDING_UPSTREAM_API_KEY", "ollama"
            ).strip(),
            internal_token=values.get("MEMORY_EGRESS_TOKEN", "").strip(),
            host=values.get("MEMORY_EGRESS_HOST", "127.0.0.1").strip(),
            port=port,
            timeout=timeout,
        )


class MemoryEgressGateway:
    """Authenticated, fixed-route egress for Hindsight provider calls."""

    def __init__(self, settings: MemoryEgressGatewaySettings):
        self._settings = settings
        self._expected_authorization = f"Bearer {settings.internal_token}"
        self._session: aiohttp.ClientSession | None = None
        self.application = web.Application(
            client_max_size=_MAX_BODY_BYTES,
            middlewares=[self._response_boundary],
        )
        self.application.cleanup_ctx.append(self._client_context)
        self.application.router.add_get("/health", self._health)
        self.application.router.add_post(
            "/llm/v1/chat/completions",
            self._proxy_llm,
        )
        self.application.router.add_post(
            "/embeddings/v1/embeddings",
            self._proxy_embeddings,
        )

    async def _client_context(
        self, _application: web.Application
    ) -> AsyncIterator[None]:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._settings.timeout),
        )
        try:
            yield
        finally:
            await self._session.close()
            self._session = None

    @web.middleware
    async def _response_boundary(
        self,
        request: web.Request,
        handler: web.RequestHandler,
    ) -> web.StreamResponse:
        try:
            response = await handler(request)
        except web.HTTPRequestEntityTooLarge:
            response = _error_response(
                413,
                "REQUEST_TOO_LARGE",
                "Provider request is too large",
            )
        except web.HTTPNotFound:
            response = _error_response(404, "NOT_FOUND", "Resource not found")
        except web.HTTPMethodNotAllowed as exc:
            response = _error_response(
                405,
                "METHOD_NOT_ALLOWED",
                "Method not allowed",
                headers={"Allow": ", ".join(sorted(exc.allowed_methods))},
            )
        except web.HTTPException as exc:
            response = _error_response(
                exc.status,
                "REQUEST_FAILED",
                "Provider gateway request failed",
            )
        except Exception:
            response = _error_response(
                500,
                "INTERNAL_ERROR",
                "Provider gateway request failed",
            )
        response.headers.update(_PRIVATE_HEADERS)
        return response

    async def _health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _proxy_llm(self, request: web.Request) -> web.Response:
        return await self._proxy(
            request,
            upstream_url=f"{self._settings.llm_upstream_url}/chat/completions",
            upstream_api_key=self._settings.llm_api_key,
        )

    async def _proxy_embeddings(self, request: web.Request) -> web.Response:
        return await self._proxy(
            request,
            upstream_url=f"{self._settings.embedding_upstream_url}/embeddings",
            upstream_api_key=self._settings.embedding_api_key,
        )

    async def _proxy(
        self,
        request: web.Request,
        *,
        upstream_url: str,
        upstream_api_key: str,
    ) -> web.Response:
        authorization = request.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization, self._expected_authorization):
            return _error_response(
                401,
                "UNAUTHORIZED",
                "Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if request.query_string:
            raise web.HTTPNotFound
        body = await request.read()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {upstream_api_key}",
            "Content-Type": request.headers.get(
                "Content-Type", "application/json"
            ),
        }
        assert self._session is not None
        try:
            async with self._session.post(
                upstream_url,
                data=body,
                headers=headers,
                allow_redirects=False,
            ) as upstream:
                if not 200 <= upstream.status < 300:
                    return _error_response(
                        upstream.status,
                        "UPSTREAM_ERROR",
                        "Provider request failed",
                        headers={
                            name: upstream.headers[name]
                            for name in ("Retry-After",)
                            if name in upstream.headers
                        },
                    )
                payload = await _bounded_body(upstream)
                response_headers = {
                    name: upstream.headers[name]
                    for name in ("Content-Type",)
                    if name in upstream.headers
                }
                return web.Response(
                    body=payload,
                    status=upstream.status,
                    headers=response_headers,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError, _ResponseTooLarge):
            return _error_response(
                502,
                "UPSTREAM_UNAVAILABLE",
                "Provider service unavailable",
            )


class _ResponseTooLarge(RuntimeError):
    pass


async def _bounded_body(response: aiohttp.ClientResponse) -> bytes:
    if response.content_length is not None and response.content_length > _MAX_BODY_BYTES:
        raise _ResponseTooLarge
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        size += len(chunk)
        if size > _MAX_BODY_BYTES:
            raise _ResponseTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def _error_response(
    status: int,
    code: str,
    message: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message}},
        status=status,
        headers=headers,
    )


def main() -> None:
    settings = MemoryEgressGatewaySettings.from_env()
    gateway = MemoryEgressGateway(settings)
    web.run_app(
        gateway.application,
        host=settings.host,
        port=settings.port,
        access_log=None,
        print=None,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
