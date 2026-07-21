"""
Configuration management for sidekick.
Loads from ~/.sidekick/config.toml with environment variable overrides.
"""
import os
from pathlib import Path

import tomllib

CONFIG_DIR = Path.home() / ".sidekick"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def read_config_file() -> dict:
    """Read ~/.sidekick/config.toml and return raw nested config."""
    config: dict = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            config = tomllib.load(f)
    return config


def load_config() -> dict:
    """Load startup env overrides from ~/.sidekick/config.toml for CLI bootstrap."""
    config = read_config_file()

    telegram = config.get("telegram", {})

    return {
        "TELEGRAM_API_ID": os.environ.get("TELEGRAM_API_ID") or str(telegram.get("api_id", "")),
        "TELEGRAM_API_HASH": os.environ.get("TELEGRAM_API_HASH") or telegram.get("api_hash", ""),
    }


def apply_config() -> None:
    """Load CLI startup env.

    Priority: existing process env > .env file > config.toml API settings.
    Telegram account settings are resolved later by the adapter runtime.
    """
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and value and not os.environ.get(key):
                    os.environ[key] = value

    for key, value in load_config().items():
        if value and not os.environ.get(key):
            os.environ[key] = value
