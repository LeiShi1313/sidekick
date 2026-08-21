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


def test_blocked_user_ids_environment_accepts_comma_separated_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SIDEKICK_TELEGRAM_BLOCKED_USER_IDS",
        "123456789, 7000000000",
    )
    monkeypatch.setattr(
        telegram_config,
        "read_config_file",
        lambda: {
            "telegram": {
                "api_id": 12345,
                "api_hash": "test-hash",
            }
        },
    )

    runtime = TelegramRuntimeConfig.from_account()

    assert runtime.blocked_user_ids == frozenset({123456789, 7000000000})


def test_blank_blocked_user_ids_environment_uses_global_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIDEKICK_TELEGRAM_BLOCKED_USER_IDS", "")
    monkeypatch.setattr(
        telegram_config,
        "read_config_file",
        lambda: {
            "telegram": {
                "api_id": 12345,
                "api_hash": "test-hash",
                "blocked_user_ids": [123456789],
            }
        },
    )

    runtime = TelegramRuntimeConfig.from_account()

    assert runtime.blocked_user_ids == frozenset({123456789})


@pytest.mark.parametrize(
    "configured_ids",
    ("0", "-1", "true", "123456789,invalid"),
)
def test_blocked_user_ids_reject_invalid_environment_configuration(
    monkeypatch: pytest.MonkeyPatch,
    configured_ids: str,
) -> None:
    monkeypatch.setenv("SIDEKICK_TELEGRAM_BLOCKED_USER_IDS", configured_ids)
    monkeypatch.setattr(
        telegram_config,
        "read_config_file",
        lambda: {
            "telegram": {
                "api_id": 12345,
                "api_hash": "test-hash",
            }
        },
    )

    with pytest.raises(ValueError, match="Blocked Telegram user IDs"):
        TelegramRuntimeConfig.from_account()


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
