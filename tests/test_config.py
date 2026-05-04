from __future__ import annotations

import pytest

from stock_manager.config.loader import ConfigError, load_config, require_keys


def test_load_config_expands_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOT", "/tmp/example")
    path = tmp_path / "config.yaml"
    path.write_text("paths:\n  raw: ${ROOT}/raw\n", encoding="utf-8")

    config = load_config(path)

    assert config["paths"]["raw"] == "/tmp/example/raw"


def test_require_keys_reports_missing_nested_key():
    with pytest.raises(ConfigError, match="data.start"):
        require_keys({"data": {}}, ["data.start"])

