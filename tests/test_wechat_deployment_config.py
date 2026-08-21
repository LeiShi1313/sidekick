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
        "SIDEKICK_TELEGRAM_PI_TOKEN": "telegram-pi-token-that-is-long-enough",
        "SIDEKICK_TELEGRAM_MATRIX_BRIDGE_BOT_IDS": "6332621450,7000000000",
        "SIDEKICK_TELEGRAM_BLOCKED_USER_IDS": "123456789",
        "SIDEKICK_ONEBOT_PI_TOKEN": "onebot-pi-token-that-is-long-enough",
        "SIDEKICK_WECHAT_HOST_PI_TOKEN": "wechat-host-pi-token-that-is-long-enough",
        "SIDEKICK_WECHAT_PEER_PI_TOKEN": "wechat-peer-pi-token-that-is-long-enough",
        "SIDEKICK_OPS_TOKEN": "channel-ops-token-that-is-long-enough",
        "MEMORY_API_TOKEN": "memory-api-token-that-is-long-enough",
        "SIDEKICK_MAINLAND_BLOCKED_TERMS": '["restricted-example"]',
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
def test_telegram_worker_receives_matrix_bridge_bot_ids() -> None:
    rendered = render_compose()

    assert (
        rendered["services"]["ai"]["environment"][
            "SIDEKICK_TELEGRAM_MATRIX_BRIDGE_BOT_IDS"
        ]
        == "6332621450,7000000000"
    )


@pytest.mark.skipif(not docker_compose_available(), reason="Docker Compose is required")
def test_telegram_worker_receives_blocked_user_ids() -> None:
    rendered = render_compose()

    assert (
        rendered["services"]["ai"]["environment"][
            "SIDEKICK_TELEGRAM_BLOCKED_USER_IDS"
        ]
        == "123456789"
    )


@pytest.mark.skipif(not docker_compose_available(), reason="Docker Compose is required")
def test_compose_scopes_mainland_blocked_terms_to_wechat_and_qq() -> None:
    services = render_compose()["services"]

    assert services["onebot-ai"]["environment"][
        "SIDEKICK_MAINLAND_BLOCKED_TERMS"
    ] == '["restricted-example"]'
    for service_name in ("wechat-host-ai", "wechat-peer-ai"):
        assert services[service_name]["environment"][
            "SIDEKICK_MAINLAND_BLOCKED_TERMS"
        ] == '["restricted-example"]'
    assert "SIDEKICK_MAINLAND_BLOCKED_TERMS" not in services["ai"]["environment"]


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
        assert environment["SIDEKICK_HINDSIGHT_URL"] == (
            "http://memory-gateway:8888"
        )
        assert environment["SIDEKICK_HINDSIGHT_TOKEN"] == (
            "memory-api-token-that-is-long-enough"
        )
        assert environment["SIDEKICK_OPS_TOKEN"] == (
            "channel-ops-token-that-is-long-enough"
        )
        assert set(service["networks"]) == {
            "agent-platform",
            "memory-platform",
            settings["network"],
        }
        assert service["read_only"] is True
        assert service["restart"] == "unless-stopped"
        assert service["healthcheck"]["test"][0] == "CMD"
        assert "client.get_messages(" not in service["healthcheck"]["test"][3]
        assert "capabilities.require_ai_channel()" in service["healthcheck"]["test"][3]
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
