"""Configuration loading and project path helpers."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, MutableMapping

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.example.yaml"


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def _replace_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_env_vars(item) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_expr = value[2:-1]
        default_value = ""
        if ":" in env_expr:
            env_expr, default_value = env_expr.split(":", 1)
        return os.getenv(env_expr, default_value)
    return value


def _resolve_config_file(config_path: str | Path | None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH

    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / path).resolve()


def _resolve_project_path(value: Any) -> Any:
    if value in (None, ""):
        return value

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _set_nested(config: MutableMapping[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, MutableMapping):
            return
        current = next_value
    if keys[-1] in current:
        current[keys[-1]] = value


def resolve_runtime_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve known relative runtime paths against the repository root."""
    resolved = deepcopy(config)
    path_keys = (
        ("models", "yolo", "model_path"),
        ("models", "embedding", "onnx_path"),
        ("database", "chroma", "persist_directory"),
        ("registry", "persist_path"),
    )

    for keys in path_keys:
        current: Any = resolved
        for key in keys:
            if not isinstance(current, MutableMapping) or key not in current:
                current = None
                break
            current = current[key]
        if current not in (None, ""):
            _set_nested(resolved, keys, _resolve_project_path(current))

    return resolved


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, expand env vars, and resolve runtime paths."""
    path = _resolve_config_file(config_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return resolve_runtime_paths(_replace_env_vars(config))
