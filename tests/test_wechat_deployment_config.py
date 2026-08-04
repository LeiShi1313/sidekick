from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPOSITORY_ROOT / "docker-compose.yml"


def docker_compose_available() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    return (
        subprocess.run(
            [docker, "compose", "version"],
            capture_output=True,
            check=False,
            text=True,
        ).returncode
        == 0
    )


def render_compose() -> dict[str, object]:
    environment = {
        **os.environ,
        "TELEGRAM_API_ID": "test-id",
        "TELEGRAM_API_HASH": "test-hash",
        "SIDEKICK_ONEBOT_TOKEN": "test-onebot-token",
        "SIDEKICK_ONEBOT_SELF_ID": "123456789",
        "SIDEKICK_PI_TOKEN": "test-pi-token",
    }
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            "/dev/null",
            "--project-directory",
            str(REPOSITORY_ROOT),
            "--file",
            str(COMPOSE_FILE),
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.skipif(not docker_compose_available(), reason="Docker Compose is required")
def test_compose_declares_isolated_workers_for_both_wechat_bridges() -> None:
    rendered = render_compose()
    services = rendered["services"]

    expected = {
        "wechat-host-ai": {
            "network": "wechat-host",
            "volume": "sidekick-wechat-host-runtime",
            "wechat_state": "/sidekick-data/.sidekick/wechat-host.db",
            "ai_state": "/sidekick-data/.sidekick/wechat-host-ai.db",
        },
        "wechat-peer-ai": {
            "network": "wechat-peer",
            "volume": "sidekick-wechat-peer-runtime",
            "wechat_state": "/sidekick-data/.sidekick/wechat-peer.db",
            "ai_state": "/sidekick-data/.sidekick/wechat-peer-ai.db",
        },
    }
    for service_name, settings in expected.items():
        service = services[service_name]
        environment = service["environment"]
        assert service["command"] == [
            "python",
            "-m",
            "sidekick.cli",
            "wechat",
            "ai",
        ]
        assert environment["SIDEKICK_WECHAT_URL"] == "http://wechat:18088"
        assert environment["SIDEKICK_WECHAT_STATE_PATH"] == settings["wechat_state"]
        assert environment["SIDEKICK_AI_STATE_PATH"] == settings["ai_state"]
        assert set(service["networks"]) == {
            "agent-platform",
            "memory-platform",
            settings["network"],
        }
        assert service["read_only"] is True
        assert service["restart"] == "unless-stopped"
        assert service["healthcheck"]["test"][0] == "CMD"
        assert "client.get_messages(limit=1_000)" in service["healthcheck"]["test"][3]
        assert service["volumes"] == [
            {
                "type": "volume",
                "source": settings["volume"],
                "target": "/sidekick-data",
                "volume": {},
            }
        ]

    networks = rendered["networks"]
    assert networks["wechat-host"]["name"] == "wechat-host_default"
    assert networks["wechat-peer"]["name"] == "wechat-peer_default"


@pytest.mark.skipif(not docker_compose_available(), reason="Docker Compose is required")
def test_compose_passes_memory_outbox_settings_to_every_ai_worker() -> None:
    services = render_compose()["services"]
    expected = {
        "SIDEKICK_MEMORY_OUTBOX_MAX_ATTEMPTS": "5",
        "SIDEKICK_MEMORY_OUTBOX_RETRY_BASE_SECONDS": "30",
        "SIDEKICK_MEMORY_OUTBOX_RETRY_MAX_SECONDS": "3600",
        "SIDEKICK_MEMORY_OUTBOX_CYCLE_DOCUMENTS": "100",
        "SIDEKICK_MEMORY_OUTBOX_POLL_SECONDS": "10",
        "SIDEKICK_MEMORY_OUTBOX_CONCURRENCY": "2",
        "SIDEKICK_MEMORY_OUTBOX_SCOPE_BATCH_SIZE": "20",
    }

    for service_name in ("ai", "onebot-ai", "wechat-host-ai", "wechat-peer-ai"):
        environment = services[service_name]["environment"]
        assert {key: environment[key] for key in expected} == expected
