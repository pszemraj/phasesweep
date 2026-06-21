#!/usr/bin/env python3
"""Adapt PhaseSweep JSON overrides to decoder-pytorch-template YAML configs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_ROOT = Path(__file__).resolve().parent / "vendor" / "decoder-pytorch-template"


def _resolve_repo_path(path: str | Path) -> Path:
    """Resolve a possibly repo-relative path."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``override`` values."""
    merged = dict(base)
    for key, value in override.items():
        old_value = merged.get(key)
        if isinstance(old_value, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(old_value, value)
        else:
            merged[key] = value
    return merged


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file and require a top-level mapping."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON file and require a top-level mapping."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _last_metric(metrics_path: Path) -> dict[str, Any]:
    """Read the last JSONL metric record emitted by the trainer."""
    last: dict[str, Any] | None = None
    with metrics_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{metrics_path} contains a non-object JSONL record")
                last = record
    if last is None:
        raise ValueError(f"{metrics_path} did not contain any metric records")
    return last


def _write_result(
    trial_dir: Path, config: Mapping[str, Any], metric_record: Mapping[str, Any]
) -> None:
    """Write PhaseSweep's scalar result artifact."""
    val_loss = metric_record.get("val_loss")
    if not isinstance(val_loss, int | float):
        raise ValueError(f"Last metric record is missing numeric val_loss: {metric_record!r}")

    result = {
        "val_loss": float(val_loss),
        "step": metric_record.get("step"),
        "depth": config.get("depth"),
        "dim": config.get("dim"),
        "grad_clip_norm": config.get("grad_clip_norm"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
    }
    (trial_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _run_template(template_root: Path, config_path: Path) -> None:
    """Run the decoder-pytorch-template trainer with the composed YAML config."""
    train_py = template_root / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(
            f"decoder-pytorch-template checkout not found at {template_root}; expected {train_py}"
        )

    subprocess.run(
        [sys.executable, str(train_py), "--config", str(config_path)],
        cwd=template_root,
        env=os.environ.copy(),
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-root",
        default=os.environ.get("DECODER_TEMPLATE_ROOT", str(DEFAULT_TEMPLATE_ROOT)),
        help="Path to a decoder-pytorch-template checkout.",
    )
    parser.add_argument("--base-config", required=True, help="Base YAML config for the trainer.")
    parser.add_argument(
        "--overrides-path",
        required=True,
        help="PhaseSweep overrides JSON generated with override_format: json_file.",
    )
    parser.add_argument("--trial-dir", required=True, help="PhaseSweep trial directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Compose a trial config, run the trainer, and write ``result.json``."""
    args = build_parser().parse_args(argv)

    template_root = _resolve_repo_path(args.template_root)
    base_config = _read_yaml_mapping(_resolve_repo_path(args.base_config))
    overrides = _read_json_mapping(Path(args.overrides_path))
    trial_dir = Path(args.trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)

    trainer_run_dir = trial_dir / "trainer"
    composed = _deep_merge(base_config, overrides)
    composed["run_dir"] = str(trainer_run_dir)

    generated_config = trial_dir / "decoder_config.yaml"
    generated_config.write_text(yaml.safe_dump(composed, sort_keys=True))

    _run_template(template_root, generated_config)
    metric_record = _last_metric(trainer_run_dir / "metrics.jsonl")
    _write_result(trial_dir, composed, metric_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
