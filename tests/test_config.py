from sidekick import config


def test_apply_config_uses_sidekick_file_and_preserves_precedence(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[telegram]\napi_id = 100\napi_hash = "from-config"\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "TELEGRAM_API_ID=200\nTELEGRAM_API_HASH=from-dotenv\nSIDEKICK_TEST_VALUE=loaded\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    monkeypatch.setenv("TELEGRAM_API_ID", "300")
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
    monkeypatch.delenv("SIDEKICK_TEST_VALUE", raising=False)

    config.apply_config()

    assert config.CONFIG_DIR.name == ".sidekick"
    assert config.CONFIG_FILE.name == "config.toml"
    assert config.os.environ["TELEGRAM_API_ID"] == "300"
    assert config.os.environ["TELEGRAM_API_HASH"] == "from-dotenv"
    assert config.os.environ["SIDEKICK_TEST_VALUE"] == "loaded"


def test_apply_config_falls_back_to_config_toml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[telegram]\napi_id = 100\napi_hash = "from-config"\n',
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)

    config.apply_config()

    assert config.os.environ["TELEGRAM_API_ID"] == "100"
    assert config.os.environ["TELEGRAM_API_HASH"] == "from-config"
