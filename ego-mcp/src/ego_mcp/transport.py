"""Transport selection and the Streamable HTTP app for ego-mcp.

stdio stays the default. ``EGO_MCP_TRANSPORT=streamable-http`` serves the same
MCP server over HTTP so it can run as a long-lived service (e.g. in Docker),
guarded by a mandatory bearer token.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "streamable-http"
HEALTH_PATH = "/healthz"


@dataclass(frozen=True)
class TransportSettings:
    """Transport configuration loaded from environment variables.

    Environment variables:
        EGO_MCP_TRANSPORT: "stdio" or "streamable-http" (default: "stdio")
        EGO_MCP_HTTP_HOST: Bind address (default: "127.0.0.1")
        EGO_MCP_HTTP_PORT: Bind port (default: 8765)
        EGO_MCP_HTTP_PATH: MCP endpoint path (default: "/mcp")
        EGO_MCP_HTTP_TOKEN: Bearer token clients must send (required for HTTP)
    """

    transport: str = TRANSPORT_STDIO
    host: str = "127.0.0.1"
    port: int = 8765
    path: str = "/mcp"
    token: str | None = None

    @classmethod
    def from_env(cls) -> TransportSettings:
        transport = os.environ.get("EGO_MCP_TRANSPORT", TRANSPORT_STDIO).strip().lower()
        if transport not in (TRANSPORT_STDIO, TRANSPORT_HTTP):
            raise ValueError(
                f"Invalid EGO_MCP_TRANSPORT '{transport}'. "
                f"Must be '{TRANSPORT_STDIO}' or '{TRANSPORT_HTTP}'."
            )
        if transport == TRANSPORT_STDIO:
            return cls()

        host = os.environ.get("EGO_MCP_HTTP_HOST", "127.0.0.1").strip() or "127.0.0.1"

        port_raw = os.environ.get("EGO_MCP_HTTP_PORT", "8765").strip()
        try:
            port = int(port_raw)
        except ValueError:
            port = -1
        if not 1 <= port <= 65535:
            raise ValueError(
                f"Invalid EGO_MCP_HTTP_PORT '{port_raw}'. Must be 1-65535."
            )

        path = "/" + os.environ.get("EGO_MCP_HTTP_PATH", "/mcp").strip().strip("/")
        if path == HEALTH_PATH:
            raise ValueError(f"EGO_MCP_HTTP_PATH cannot be {HEALTH_PATH}.")

        token = os.environ.get("EGO_MCP_HTTP_TOKEN", "").strip()
        if not token:
            raise ValueError(
                "EGO_MCP_HTTP_TOKEN is not set. The HTTP transport requires a "
                "bearer token (e.g. `openssl rand -hex 32`)."
            )

        return cls(transport=transport, host=host, port=port, path=path, token=token)


class BearerTokenMiddleware:
    """Reject HTTP requests whose ``Authorization: Bearer`` token does not match.

    Pure ASGI (not BaseHTTPMiddleware) so SSE streams pass through untouched.
    """

    def __init__(
        self, app: ASGIApp, token: str, exempt_paths: frozenset[str] = frozenset()
    ) -> None:
        self._app = app
        self._token = token.encode()
        self._exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or (
            scope["type"] == "http" and scope["path"] in self._exempt_paths
        ):
            await self._app(scope, receive, send)
            return

        scheme, _, credential = Headers(scope=scope).get("authorization", "").partition(
            " "
        )
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            credential.strip().encode(), self._token
        ):
            response = JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="ego-mcp"'},
            )
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)


class _SessionManagerEndpoint:
    """ASGI adapter so Starlette's Route hands requests to the session manager."""

    def __init__(self, manager: StreamableHTTPSessionManager) -> None:
        self._manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._manager.handle_request(scope, receive, send)


async def _healthz(request: Request) -> Response:
    return PlainTextResponse("ok")


def build_http_app(server: Server[object, object], settings: TransportSettings) -> Starlette:
    """Build the Starlette app serving ``server`` over Streamable HTTP."""
    if not settings.token:
        raise ValueError("HTTP transport requires a bearer token.")

    manager = StreamableHTTPSessionManager(app=server)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(
        routes=[
            Route(HEALTH_PATH, _healthz, methods=["GET"]),
            Route(settings.path, _SessionManagerEndpoint(manager)),
        ],
        middleware=[
            Middleware(
                BearerTokenMiddleware,
                token=settings.token,
                exempt_paths=frozenset({HEALTH_PATH}),
            )
        ],
        lifespan=lifespan,
    )


async def serve_http(server: Server[object, object], settings: TransportSettings) -> None:
    """Run the Streamable HTTP app with uvicorn until shutdown."""
    import uvicorn

    app = build_http_app(server, settings)
    config = uvicorn.Config(
        app, host=settings.host, port=settings.port, log_config=None, lifespan="on"
    )
    await uvicorn.Server(config).serve()
