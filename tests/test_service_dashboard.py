from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


class DashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture: str | None = None
        self._href: str | None = None
        self.title = ""
        self.heading = ""
        self.links: dict[str, str] = {}
        self._text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in {"title", "h1", "a"}:
            self._capture = tag
            self._text = []
        if tag == "a":
            self._href = dict(attrs).get("href")

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self._capture:
            return
        text = " ".join("".join(self._text).split())
        if tag == "title":
            self.title = text
        elif tag == "h1":
            self.heading = text
        elif tag == "a" and self._href is not None:
            self.links[text] = self._href
        self._capture = None
        self._href = None
        self._text = []


def test_dashboard_links_only_managed_human_facing_surfaces() -> None:
    parser = DashboardParser()
    parser.feed((ROOT / "proxy" / "dashboard" / "index.html").read_text())

    assert parser.title == "Sidekick"
    assert parser.heading == "Sidekick"
    assert parser.links == {
        "Agent Playground": "http://playground.sidekick.localhost:18865/",
    }
    assert "100.99." not in (ROOT / "proxy" / "dashboard" / "index.html").read_text()
    assert "benchmark" not in " ".join(parser.links.values()).lower()


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is required")
def test_compose_routes_dashboard_and_playground_by_name() -> None:
    proxy = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(ROOT / "proxy" / "compose.yml"),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    proxy_config = json.loads(proxy.stdout)
    dashboard_proxy = proxy_config["services"]["dashboard-proxy"]
    assert set(dashboard_proxy["networks"]) == {
        "agent-platform"
    }
    assert dashboard_proxy["image"] == "nginx:1.30.4-alpine"
    assert dashboard_proxy["user"] == "101:101"
    assert all(
        volume.get("source") != "/var/run/docker.sock"
        for volume in dashboard_proxy.get("volumes", [])
    )
    nginx_config = (ROOT / "proxy" / "nginx.conf").read_text()
    assert "sidekick.localhost" in nginx_config
    assert "playground.sidekick.localhost" in nginx_config
    assert "dashboard:8080" in nginx_config
    assert "agent-playground:8780" in nginx_config
    assert "access_log off" in nginx_config
    dashboard = proxy_config["services"]["dashboard"]
    assert "environment" not in dashboard
    assert dashboard["build"]["context"] == str(ROOT / "proxy" / "dashboard")
    assert "volumes" not in dashboard

    agent = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(ROOT / "agent" / ".env.example"),
            "-f",
            str(ROOT / "agent" / "compose.yml"),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    agent_config = json.loads(agent.stdout)
    pi_agent = agent_config["services"]["pi-agent"]
    assert pi_agent["environment"]["MEMORY_API_URL"] == (
        "http://memory-gateway:8888"
    )
    assert pi_agent["environment"]["MEMORY_API_TOKEN"]
    assert pi_agent["environment"]["PI_AGENT_WECHAT_HOST_SCOPE_PREFIX"].startswith(
        "wechat:account:"
    )
    assert pi_agent["environment"]["PI_AGENT_WECHAT_PEER_SCOPE_PREFIX"].startswith(
        "wechat:account:"
    )
    playground = agent_config["services"]["agent-playground"]
    assert "VIRTUAL_HOST" not in playground["environment"]
    assert "VIRTUAL_PORT" not in playground["environment"]
    assert playground["environment"]["MEMORY_API_URL"] == (
        "http://memory-gateway:8888"
    )
    assert playground["environment"]["MEMORY_API_TOKEN"]
    assert playground["environment"]["PLAYGROUND_CHANNEL_TOKEN"]
