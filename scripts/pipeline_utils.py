#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_config_arg(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config file")
    return parser.parse_args()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_minimal_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"{path}:{line_no}: expected 'key: value' YAML mapping")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_no}: empty YAML key")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        raw_value = raw_value.strip()
        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore
    except ImportError:
        config = _load_minimal_yaml(config_path)
    else:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    print(f"Loaded config: {config_path}")
    return config


def resolve_path(value: str | Path | None) -> Path:
    if value is None:
        raise ValueError("Expected path value, got null")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def optional_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(value)


def require_file(value: str | Path | None, label: str) -> Path:
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_dir(value: str | Path | None, label: str) -> Path:
    path = resolve_path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_keys(config: dict[str, Any], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in config or config[key] in (None, "")]
    if missing:
        raise ValueError(f"Missing required config key(s): {', '.join(missing)}")


@contextmanager
def patched_argv(argv: Sequence[str]) -> Iterator[None]:
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


@contextmanager
def patched_environ(values: dict[str, Any]) -> Iterator[None]:
    old_values: dict[str, str | None] = {}
    normalized = {key: str(value) for key, value in values.items() if value not in (None, "")}
    for key, value in normalized.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def run_script_main(script_path: str | Path, argv: Sequence[str], env: dict[str, Any] | None = None) -> None:
    path = resolve_path(script_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing runnable script: {path}")

    module_name = "_pipeline_stage_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import stage script: {path}")

    module = importlib.util.module_from_spec(spec)
    with patched_environ(env or {}), patched_argv([str(path), *[str(arg) for arg in argv]]):
        spec.loader.exec_module(module)
        main = getattr(module, "main", None)
        if not callable(main):
            raise RuntimeError(f"Script does not expose callable main(): {path}")
        main()


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

