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


def test_dashboard_links_every_human_facing_sidekick_surface() -> None:
    parser = DashboardParser()
    parser.feed((ROOT / "proxy" / "dashboard" / "index.html").read_text())

    assert parser.title == "Sidekick"
    assert parser.heading == "Sidekick"
    assert parser.links == {
        "Agent Playground": "http://playground.sidekick.localhost:18865/",
        "Hindsight Memory": "http://hindsight.sidekick.localhost:18865/",
        "All Explainers": "http://100.99.247.60:18081/",
        "Agent Design Atlas": (
            "http://100.99.247.60:18081/agent-design-atlas.html"
        ),
        "Hindsight Memory Flow": (
            "http://100.99.247.60:18081/hindsight-memory-system-flow-zh.html"
        ),
        "Multi-user Prompt Flow": (
            "http://100.99.247.60:18081/multi-user-memory-prompt-flow.html"
        ),
        "Reply-thread Retrieval": (
            "http://100.99.247.60:18081/reply-thread-memory-retrieval-zh.html"
        ),
    }
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
    dashboard = proxy_config["services"]["dashboard"]
    assert dashboard["environment"]["VIRTUAL_HOST"] == "sidekick.localhost"
    assert dashboard["environment"]["VIRTUAL_PORT"] == "8080"

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
    playground = agent_config["services"]["agent-playground"]
    assert playground["environment"]["VIRTUAL_HOST"] == (
        "playground.sidekick.localhost"
    )
