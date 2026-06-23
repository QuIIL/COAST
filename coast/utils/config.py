from pathlib import Path
from typing import Any

import yaml


class ConfigNamespace:
    def __init__(self, values: dict[str, Any]):
        for key, value in values.items():
            if isinstance(value, dict):
                value = ConfigNamespace(value)
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)


def recursive_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict):
            base[key] = recursive_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


def auto_cast(value):
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped) if "." in stripped else int(stripped)
        except ValueError:
            return value
    if isinstance(value, dict):
        return {key: auto_cast(item) for key, item in value.items()}
    if isinstance(value, list):
        return [auto_cast(item) for item in value]
    return value


def load_config(yaml_paths: list[str | Path | None]) -> ConfigNamespace:
    merged = {}
    for yaml_path in yaml_paths:
        if yaml_path is None:
            continue
        with Path(yaml_path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        merged = recursive_update(merged, auto_cast(values))
    return ConfigNamespace(merged)
