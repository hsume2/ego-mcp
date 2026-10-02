"""Tests for transport selection and the Streamable HTTP app."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp.types import TextContent
from starlette.testclient import TestClient

import ego_mcp.server as server_module
from ego_mcp.transport import (
    TRANSPORT_HTTP,
    TRANSPORT_STDIO,
    TransportSettings,
    build_http_app,
)

TOKEN = "test-token-abc"
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "EGO_MCP_TRANSPORT",
        "EGO_MCP_HTTP_HOST",
        "EGO_MCP_HTTP_PORT",
        "EGO_MCP_HTTP_PATH",
        "EGO_MCP_HTTP_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)


def _http_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGO_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("EGO_MCP_HTTP_TOKEN", TOKEN)


def _sse_payloads(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data:") :])
        for line in body.splitlines()
        if line.startswith("data:")
    ]


class TestTransportSettings:
    def test_defaults_to_stdio(self) -> None:
        settings = TransportSettings.from_env()

        assert settings.transport == TRANSPORT_STDIO
        assert settings.token is None

    def test_stdio_ignores_http_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EGO_MCP_HTTP_PORT", "not-a-port")

        assert TransportSettings.from_env().transport == TRANSPORT_STDIO

    def test_invalid_transport_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EGO_MCP_TRANSPORT", "sse")
        with pytest.raises(ValueError, match="Invalid EGO_MCP_TRANSPORT"):
            TransportSettings.from_env()

    def test_http_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _http_env(monkeypatch)
        settings = TransportSettings.from_env()

        assert settings.transport == TRANSPORT_HTTP
        assert settings.host == "127.0.0.1"
        assert settings.port == 8765
        assert settings.path == "/mcp"
        assert settings.token == TOKEN

    def test_http_custom_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _http_env(monkeypatch)
        monkeypatch.setenv("EGO_MCP_TRANSPORT", "Streamable-HTTP")
        monkeypatch.setenv("EGO_MCP_HTTP_HOST", "0.0.0.0")
        monkeypatch.setenv("EGO_MCP_HTTP_PORT", "9000")
        monkeypatch.setenv("EGO_MCP_HTTP_PATH", "ego/")
        settings = TransportSettings.from_env()

        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.path == "/ego"

    def test_http_requires_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EGO_MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("EGO_MCP_HTTP_TOKEN", "   ")
        with pytest.raises(ValueError, match="EGO_MCP_HTTP_TOKEN is not set"):
            TransportSettings.from_env()

    @pytest.mark.parametrize("port", ["0", "70000", "abc"])
    def test_invalid_port_raises(
        self, monkeypatch: pytest.MonkeyPatch, port: str
    ) -> None:
        _http_env(monkeypatch)
        monkeypatch.setenv("EGO_MCP_HTTP_PORT", port)
        with pytest.raises(ValueError, match="Invalid EGO_MCP_HTTP_PORT"):
            TransportSettings.from_env()

    def test_path_cannot_shadow_health(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _http_env(monkeypatch)
        monkeypatch.setenv("EGO_MCP_HTTP_PATH", "/healthz")
        with pytest.raises(ValueError, match="cannot be /healthz"):
            TransportSettings.from_env()


class TestHttpApp:
    @pytest.fixture
    def client(self) -> Any:
        settings = TransportSettings(transport=TRANSPORT_HTTP, token=TOKEN)
        app = build_http_app(server_module.server, settings)
        with TestClient(app) as client:
            yield client

    def test_build_requires_token(self) -> None:
        with pytest.raises(ValueError, match="bearer token"):
            build_http_app(server_module.server, TransportSettings())

    def test_healthz_is_unauthenticated(self, client: TestClient) -> None:
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.text == "ok"

    def test_missing_token_is_rejected(self, client: TestClient) -> None:
        response = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)

        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")

    def test_wrong_token_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/mcp",
            json=INITIALIZE,
            headers={**MCP_HEADERS, "Authorization": "Bearer nope"},
        )

        assert response.status_code == 401

    def test_non_bearer_scheme_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/mcp",
            json=INITIALIZE,
            headers={**MCP_HEADERS, "Authorization": f"Basic {TOKEN}"},
        )

        assert response.status_code == 401

    def test_initialize_and_list_tools(self, client: TestClient) -> None:
        auth = {**MCP_HEADERS, "Authorization": f"Bearer {TOKEN}"}

        init = client.post("/mcp", json=INITIALIZE, headers=auth)
        assert init.status_code == 200
        [init_result] = _sse_payloads(init.text)
        assert init_result["result"]["serverInfo"]["name"] == "ego-mcp"

        session = {**auth, "Mcp-Session-Id": init.headers["mcp-session-id"]}
        client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=session,
        )
        tools = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers=session,
        )

        assert tools.status_code == 200
        [tools_result] = _sse_payloads(tools.text)
        names = {tool["name"] for tool in tools_result["result"]["tools"]}
        assert {"wake_up", "remember", "recall"} <= names


class TestToolCallSerialization:
    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_interleave(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []

        async def fake_call(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            events.append(f"start:{name}")
            await asyncio.sleep(0.01)
            events.append(f"end:{name}")
            return [TextContent(type="text", text=name)]

        monkeypatch.setattr(server_module, "_call_tool_unlocked", fake_call)
        monkeypatch.setattr(server_module, "_tool_call_lock", asyncio.Lock())

        await asyncio.gather(
            server_module.call_tool("a", {}), server_module.call_tool("b", {})
        )

        assert events == ["start:a", "end:a", "start:b", "end:b"]
