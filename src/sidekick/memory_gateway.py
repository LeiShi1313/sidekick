from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import hmac
import os
from urllib.parse import unquote, urlsplit

import aiohttp
from aiohttp import web


_MAX_BODY_BYTES = 64 * 1024 * 1024
_SUPPORTED_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
_PRIVATE_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


@dataclass(frozen=True, slots=True)
class MemoryGatewaySettings:
    upstream_url: str
    token: str
    host: str = "127.0.0.1"
    port: int = 8888
    timeout: float = 300.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.upstream_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Memory gateway upstream must use http or https")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Memory gateway upstream must be a fixed origin")
        if len(self.token) < 24:
            raise ValueError("Memory API token must contain at least 24 characters")
        if not self.host:
            raise ValueError("Memory gateway host cannot be empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("Memory gateway port must be between 1 and 65535")
        if self.timeout <= 0:
            raise ValueError("Memory gateway timeout must be positive")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> MemoryGatewaySettings:
        values = os.environ if environ is None else environ
        raw_port = values.get("MEMORY_GATEWAY_PORT", "8888").strip()
        raw_timeout = values.get("MEMORY_GATEWAY_TIMEOUT", "300").strip()
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError("MEMORY_GATEWAY_PORT must be an integer") from exc
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError("MEMORY_GATEWAY_TIMEOUT must be numeric") from exc
        return cls(
            upstream_url=values.get(
                "MEMORY_GATEWAY_UPSTREAM_URL", "http://memory-api:8888"
            )
            .strip()
            .rstrip("/"),
            token=values.get("MEMORY_API_TOKEN", "").strip(),
            host=values.get("MEMORY_GATEWAY_HOST", "127.0.0.1").strip(),
            port=port,
            timeout=timeout,
        )


class MemoryGateway:
    """A fixed-upstream, authenticated boundary around the raw memory service."""

    def __init__(self, settings: MemoryGatewaySettings):
        self._settings = settings
        self._upstream_url = settings.upstream_url.rstrip("/")
        self._expected_authorization = f"Bearer {settings.token}"
        self._session: aiohttp.ClientSession | None = None
        self.application = web.Application(
            client_max_size=_MAX_BODY_BYTES,
            middlewares=[self._response_boundary],
        )
        self.application.cleanup_ctx.append(self._client_context)
        self.application.router.add_get("/health", self._health)
        self.application.router.add_route(
            "*", "/v1/{tail:.*}", self._proxy
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
                "Memory request is too large",
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
                "Memory gateway request failed",
            )
        except Exception:
            response = _error_response(
                500,
                "INTERNAL_ERROR",
                "Memory gateway request failed",
            )
        response.headers.update(_PRIVATE_HEADERS)
        return response

    async def _health(self, _request: web.Request) -> web.Response:
        assert self._session is not None
        try:
            async with self._session.get(
                f"{self._upstream_url}/health", allow_redirects=False
            ) as response:
                if not 200 <= response.status < 300:
                    return _error_response(
                        503,
                        "UPSTREAM_UNAVAILABLE",
                        "Memory service unavailable",
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return _error_response(
                503,
                "UPSTREAM_UNAVAILABLE",
                "Memory service unavailable",
            )
        return web.json_response({"status": "ok"})

    async def _proxy(self, request: web.Request) -> web.Response:
        if _contains_dot_path_segment(request.match_info.get("tail", "")):
            raise web.HTTPNotFound
        if request.method not in _SUPPORTED_METHODS:
            raise web.HTTPMethodNotAllowed(request.method, _SUPPORTED_METHODS)
        authorization = request.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization, self._expected_authorization):
            return _error_response(
                401,
                "UNAUTHORIZED",
                "Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        body = await request.read()
        headers = {
            name: request.headers[name]
            for name in ("Accept", "Content-Type")
            if name in request.headers
        }
        raw_path = request.rel_url.raw_path
        query = request.rel_url.query_string
        upstream_url = f"{self._upstream_url}{raw_path}"
        if query:
            upstream_url = f"{upstream_url}?{query}"
        assert self._session is not None
        try:
            async with self._session.request(
                request.method,
                upstream_url,
                data=body if body else None,
                headers=headers,
                allow_redirects=False,
            ) as upstream:
                if not 200 <= upstream.status < 300:
                    return _error_response(
                        upstream.status,
                        "UPSTREAM_ERROR",
                        "Memory request failed",
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
                "Memory service unavailable",
            )


class _ResponseTooLarge(RuntimeError):
    pass


def _contains_dot_path_segment(path: str) -> bool:
    current = path
    for _ in range(8):
        if any(
            segment in {".", ".."}
            for segment in current.replace("\\", "/").split("/")
        ):
            return True
        decoded = unquote(current)
        if decoded == current:
            return False
        current = decoded
    return True


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
    settings = MemoryGatewaySettings.from_env()
    gateway = MemoryGateway(settings)
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
