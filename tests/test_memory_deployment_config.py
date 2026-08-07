from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MEMORY_COMPOSE_FILE = REPOSITORY_ROOT / "memory" / "compose.yml"


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


def render_memory_service_environment(
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    return render_memory_compose(overrides)["services"]["memory-api"]["environment"]


def render_memory_compose(
    overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    environment = {
        **os.environ,
        "MEMORY_LLM_API_KEY": "test-key",
        "MEMORY_LLM_BASE_URL": "https://provider.example/v1",
        "MEMORY_LLM_MODEL": "global-model",
        "MEMORY_LLM_REASONING_EFFORT": "low",
        "MEMORY_EMBEDDING_API_KEY": "test-embedding-key",
        "MEMORY_API_TOKEN": "memory-api-token-that-is-long-enough",
        "MEMORY_EGRESS_TOKEN": "memory-egress-token-that-is-long-enough",
    }
    for key in (
        "MEMORY_RETAIN_LLM_MODEL",
        "MEMORY_RETAIN_LLM_REASONING_EFFORT",
        "MEMORY_CONSOLIDATION_LLM_MODEL",
        "MEMORY_CONSOLIDATION_LLM_REASONING_EFFORT",
        "MEMORY_REFLECT_LLM_MODEL",
        "MEMORY_REFLECT_LLM_REASONING_EFFORT",
    ):
        environment.pop(key, None)
    environment.update(overrides or {})
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            "/dev/null",
            "--project-directory",
            str(MEMORY_COMPOSE_FILE.parent),
            "--file",
            str(MEMORY_COMPOSE_FILE),
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
def test_memory_compose_renders_per_operation_models_and_effort():
    service_environment = render_memory_service_environment(
        {
            "MEMORY_RETAIN_LLM_MODEL": "retain-model",
            "MEMORY_RETAIN_LLM_REASONING_EFFORT": "xhigh",
            "MEMORY_CONSOLIDATION_LLM_MODEL": "consolidation-model",
            "MEMORY_CONSOLIDATION_LLM_REASONING_EFFORT": "medium",
            "MEMORY_REFLECT_LLM_MODEL": "reflect-model",
            "MEMORY_REFLECT_LLM_REASONING_EFFORT": "high",
        }
    )

    assert service_environment["HINDSIGHT_API_LLM_MODEL"] == "global-model"
    assert service_environment["HINDSIGHT_API_RETAIN_LLM_MODEL"] == "retain-model"
    assert (
        service_environment["HINDSIGHT_API_CONSOLIDATION_LLM_MODEL"]
        == "consolidation-model"
    )
    assert service_environment["HINDSIGHT_API_REFLECT_LLM_MODEL"] == "reflect-model"
    assert service_environment["HINDSIGHT_API_LLM_REASONING_EFFORT"] == "low"
    assert (
        service_environment["HINDSIGHT_API_RETAIN_LLM_REASONING_EFFORT"]
        == "xhigh"
    )
    assert (
        service_environment["HINDSIGHT_API_CONSOLIDATION_LLM_REASONING_EFFORT"]
        == "medium"
    )
    assert (
        service_environment["HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT"]
        == "high"
    )


@pytest.mark.skipif(not docker_compose_available(), reason="Docker Compose is required")
def test_memory_operation_models_fall_back_to_the_global_model():
    service_environment = render_memory_service_environment()

    assert service_environment["HINDSIGHT_API_RETAIN_LLM_MODEL"] == "global-model"
    assert (
        service_environment["HINDSIGHT_API_CONSOLIDATION_LLM_MODEL"]
        == "global-model"
    )
    assert service_environment["HINDSIGHT_API_REFLECT_LLM_MODEL"] == "global-model"
    assert (
        service_environment["HINDSIGHT_API_RETAIN_LLM_REASONING_EFFORT"] == "low"
    )
    assert (
        service_environment["HINDSIGHT_API_CONSOLIDATION_LLM_REASONING_EFFORT"]
        == "low"
    )
    assert (
        service_environment["HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT"] == "low"
    )


@pytest.mark.skipif(not docker_compose_available(), reason="Docker Compose is required")
def test_raw_memory_service_is_reachable_only_through_the_authenticated_gateway():
    rendered = render_memory_compose()
    raw = rendered["services"]["memory-api"]
    gateway = rendered["services"]["memory-gateway"]
    egress = rendered["services"]["memory-egress-gateway"]

    assert "ports" not in raw
    assert set(raw["networks"]) == {"memory-backend"}
    assert raw["environment"]["HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL"] == (
        "http://memory-egress-gateway:8080/embeddings/v1"
    )
    assert raw["environment"]["HINDSIGHT_API_LLM_BASE_URL"] == (
        "http://memory-egress-gateway:8080/llm/v1"
    )
    assert raw["environment"]["HINDSIGHT_API_LLM_API_KEY"] == (
        "memory-egress-token-that-is-long-enough"
    )
    assert raw["environment"]["HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"] == (
        "memory-egress-token-that-is-long-enough"
    )
    assert "VIRTUAL_HOST" not in raw["environment"]
    assert set(gateway["networks"]) == {"default", "memory-backend"}
    assert gateway["environment"]["MEMORY_GATEWAY_UPSTREAM_URL"] == (
        "http://memory-api:8888"
    )
    assert gateway["environment"]["MEMORY_API_TOKEN"] == (
        "memory-api-token-that-is-long-enough"
    )
    assert gateway["ports"] == [
        {
            "mode": "ingress",
            "host_ip": "127.0.0.1",
            "target": 8888,
            "published": "18888",
            "protocol": "tcp",
        }
    ]
    assert rendered["networks"]["memory-backend"]["internal"] is True

    assert "ports" not in egress
    assert set(egress["networks"]) == {
        "memory-backend",
        "memory-egress",
        "ollama-embedding",
    }
    assert egress["environment"]["MEMORY_EGRESS_TOKEN"] == (
        "memory-egress-token-that-is-long-enough"
    )
    assert egress["environment"]["MEMORY_LLM_UPSTREAM_URL"] == (
        "https://provider.example/v1"
    )
    assert egress["environment"]["MEMORY_LLM_UPSTREAM_API_KEY"] == "test-key"
    assert egress["environment"]["MEMORY_EMBEDDING_UPSTREAM_URL"] == (
        "http://ollama-embedding-ollama-1:11434/v1"
    )
    assert egress["environment"]["MEMORY_EMBEDDING_UPSTREAM_API_KEY"] == (
        "test-embedding-key"
    )
    assert egress["read_only"] is True
    assert egress["cap_drop"] == ["ALL"]
    assert egress["security_opt"] == ["no-new-privileges:true"]
    assert rendered["networks"]["memory-egress"].get("external", False) is False
