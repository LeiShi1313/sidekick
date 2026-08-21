import os
from dataclasses import dataclass
from pathlib import Path

from sidekick.config import read_config_file
from sidekick.constants import DEFAULT_SESSION_NAME


@dataclass(slots=True)
class TelegramRuntimeConfig:
    account: str
    api_id: int
    api_hash: str
    session_name: str = DEFAULT_SESSION_NAME
    store_dir: Path = Path.home() / ".sidekick" / "telegram"
    matrix_bridge_bot_ids: frozenset[int] = frozenset()
    blocked_user_ids: frozenset[int] = frozenset()

    @classmethod
    def from_account(
        cls,
        account: str | None = None,
        session: str | None = None,
    ) -> "TelegramRuntimeConfig":
        config = read_config_file()
        telegram = config.get("telegram", {})

        api_id = (os.environ.get("TELEGRAM_API_ID") or str(telegram.get("api_id", ""))).strip()
        api_hash = (os.environ.get("TELEGRAM_API_HASH") or telegram.get("api_hash", "")).strip()
        if not api_id or not api_hash:
            raise ValueError(
                "Set TELEGRAM_API_ID and TELEGRAM_API_HASH, or configure "
                "[telegram] in ~/.sidekick/config.toml"
            )

        selected_account = (
            account or os.environ.get("TELEGRAM_ACCOUNT") or "default"
        ).strip() or "default"

        # Default account reads from [telegram] directly;
        # named accounts read from [telegram.<name>] sub-tables.
        if selected_account == "default":
            account_config = telegram
        else:
            account_config = telegram.get(selected_account)
            if not isinstance(account_config, dict):
                account_config = {}

        session_name = (
            session
            or os.environ.get("TELEGRAM_SESSION_NAME")
            or account_config.get("session_name")
            or (DEFAULT_SESSION_NAME if selected_account == "default" else selected_account)
        ).strip()
        store_dir = Path(
            os.environ.get("TELEGRAM_STORE_DIR")
            or telegram.get("store_dir", Path.home() / ".sidekick" / "telegram")
        )
        configured_bridge_ids = os.environ.get(
            "SIDEKICK_TELEGRAM_MATRIX_BRIDGE_BOT_IDS"
        )
        if configured_bridge_ids is None:
            configured_bridge_ids = telegram.get("matrix_bridge_bot_ids", ())
        configured_blocked_ids = os.environ.get(
            "SIDEKICK_TELEGRAM_BLOCKED_USER_IDS"
        )
        if configured_blocked_ids is None or not configured_blocked_ids.strip():
            configured_blocked_ids = telegram.get("blocked_user_ids", ())
        return cls(
            account=selected_account,
            api_id=int(api_id),
            api_hash=api_hash,
            session_name=session_name,
            store_dir=store_dir,
            matrix_bridge_bot_ids=_parse_positive_integer_ids(
                configured_bridge_ids,
                label="Matrix bridge bot IDs",
            ),
            blocked_user_ids=_parse_positive_integer_ids(
                configured_blocked_ids,
                label="Blocked Telegram user IDs",
            ),
        )

    @classmethod
    def from_env(cls, session: str | None = None) -> "TelegramRuntimeConfig":
        return cls.from_account(session=session)


def _parse_positive_integer_ids(value: object, *, label: str) -> frozenset[int]:
    candidates: object
    if isinstance(value, str):
        candidates = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        candidates = value
    if not isinstance(candidates, (list, tuple, set, frozenset)):
        raise ValueError(f"{label} must be a list of positive integers")

    parsed: set[int] = set()
    for candidate in candidates:
        if isinstance(candidate, bool):
            raise ValueError(f"{label} must be positive integers")
        try:
            item_id = int(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be positive integers") from exc
        if item_id <= 0 or str(item_id) != str(candidate).strip():
            raise ValueError(f"{label} must be positive integers")
        parsed.add(item_id)
    return frozenset(parsed)
