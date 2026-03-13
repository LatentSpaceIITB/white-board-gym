from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    return data


def load_experiment_config(config_dir: str | Path, profile: str) -> dict[str, Any]:
    config_dir = Path(config_dir)
    common = load_yaml(config_dir / "common.yaml")
    mvp = load_yaml(config_dir / "mvp.yaml")
    profile_cfg = load_yaml(config_dir / "profiles" / f"{profile}.yaml")
    return deep_merge(deep_merge(common, mvp), profile_cfg)


def build_run_dir(config: Mapping[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    output_root = Path(str(experiment.get("output_root", "runs")))
    run_name = experiment.get("run_name")

    if run_name:
        run_dir = output_root / str(run_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
