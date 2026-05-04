from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project config is missing required fields or is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand environment variables in string values."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    return _expand_env(data)


def require_keys(config: dict[str, Any], keys: list[str], *, context: str = "config") -> None:
    """Require dotted keys to exist in a nested config mapping."""
    missing = []
    for key in keys:
        cursor: Any = config
        for part in key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                missing.append(key)
                break
            cursor = cursor[part]
    if missing:
        raise ConfigError(f"Missing required {context} keys: {', '.join(missing)}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value

