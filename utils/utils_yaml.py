from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
