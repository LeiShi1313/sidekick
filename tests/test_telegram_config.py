from __future__ import annotations

import pytest

import sidekick.telegram.config as telegram_config
from sidekick.telegram.config import TelegramRuntimeConfig


def test_matrix_bridge_bot_ids_are_global_across_telegram_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIDEKICK_TELEGRAM_MATRIX_BRIDGE_BOT_IDS", raising=False)
    monkeypatch.setattr(
        telegram_config,
        "read_config_file",
        lambda: {
            "telegram": {
                "api_id": 12345,
                "api_hash": "test-hash",
                "matrix_bridge_bot_ids": [6332621450, 7000000000],
                "work": {"session_name": "work-session"},
            }
        },
    )

    runtime = TelegramRuntimeConfig.from_account(account="work")

    assert runtime.matrix_bridge_bot_ids == frozenset({6332621450, 7000000000})


def test_matrix_bridge_bot_ids_environment_override_accepts_comma_separated_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SIDEKICK_TELEGRAM_MATRIX_BRIDGE_BOT_IDS",
        "6332621450, 7000000000",
    )
    monkeypatch.setattr(
        telegram_config,
        "read_config_file",
        lambda: {
            "telegram": {
                "api_id": 12345,
                "api_hash": "test-hash",
                "matrix_bridge_bot_ids": [1],
            }
        },
    )

    runtime = TelegramRuntimeConfig.from_account()

    assert runtime.matrix_bridge_bot_ids == frozenset({6332621450, 7000000000})


@pytest.mark.parametrize(
    "configured_ids",
    ([0], [-1], [True], ["not-a-number"], "6332621450,invalid"),
)
def test_matrix_bridge_bot_ids_reject_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    configured_ids: object,
) -> None:
    monkeypatch.delenv("SIDEKICK_TELEGRAM_MATRIX_BRIDGE_BOT_IDS", raising=False)
    monkeypatch.setattr(
        telegram_config,
        "read_config_file",
        lambda: {
            "telegram": {
                "api_id": 12345,
                "api_hash": "test-hash",
                "matrix_bridge_bot_ids": configured_ids,
            }
        },
    )

    with pytest.raises(ValueError, match="Matrix bridge bot IDs"):
        TelegramRuntimeConfig.from_account()
