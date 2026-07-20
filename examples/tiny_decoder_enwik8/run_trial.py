#!/usr/bin/env python3
"""Adapt PhaseSweep JSON overrides to the upstream decoder trainer's YAML configs."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from phasesweep.runtime.files import atomic_write_text

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_ROOT = Path(__file__).resolve().parent / "upstream"


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


def _read_json_mapping(path: Path) -> tuple[dict[str, Any], str]:
    """Read a JSON mapping and hash the exact bytes supplied by PhaseSweep."""
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data, hashlib.sha256(raw).hexdigest()


def _load_upstream_trainer(template_root: Path) -> Any:
    """Load the pinned trainer module so final evaluation reuses its data utilities."""
    train_py = template_root / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(
            f"decoder-pytorch-template checkout not found at {template_root}; expected {train_py}"
        )
    spec = importlib.util.spec_from_file_location("_phasesweep_decoder_trainer", train_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load decoder trainer from {train_py}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(template_root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(template_root))
    return module


def _evaluate_final_checkpoint(template_root: Path, trainer_run_dir: Path) -> dict[str, Any]:
    """Evaluate the saved final checkpoint with the trainer's validation semantics."""
    trainer = _load_upstream_trainer(template_root)
    torch = trainer.torch
    checkpoint_path = trainer_run_dir / "final.pt"
    device, device_type, amp_dtype = trainer.get_optimal_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{checkpoint_path} is missing its trainer config")

    flash_attn = config.get("flash_attn")
    supports_flash_attn = device_type in ("cuda", "mps")
    if flash_attn is None:
        flash_attn = supports_flash_attn
    elif flash_attn and not supports_flash_attn:
        flash_attn = False

    model = trainer.Llama(
        num_tokens=config.get("num_tokens", 256),
        dim=config.get("dim", 512),
        depth=config.get("depth", 16),
        heads=config.get("heads", 8),
        dim_head=config.get("dim_head", 64),
        tied_embedding=config.get("tied_embedding", True),
        ffn_dim_multiplier=config.get("ffn_dim_multiplier"),
        flash_attn=bool(flash_attn),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if config.get("seed"):
        torch.manual_seed(config["seed"])
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = template_root / data_path
    _, val_data = trainer.load_data(str(data_path))
    val_dataset = trainer.SequenceDataset(val_data, config["seq_len"])
    val_loader = trainer.cycle(
        trainer.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    )

    use_autocast = bool(config.get("use_autocast", True))

    def autocast_context() -> Any:
        if use_autocast:
            return torch.autocast(device_type=device_type, dtype=amp_dtype)
        return contextlib.nullcontext()

    val_loss_sum = 0.0
    val_tokens = 0
    for _ in range(config.get("val_batches", 50)):
        data = next(val_loader).to(device)
        inputs = data[:, :-1]
        targets = data[:, 1:]
        with torch.no_grad(), autocast_context():
            logits = model(inputs, return_loss=False)
            loss_unreduced = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            val_loss_sum += loss_unreduced.sum().item()
            val_tokens += targets.numel()

    return {
        "checkpoint": checkpoint_path.name,
        "policy": "final_checkpoint",
        "step": checkpoint.get("step"),
        "val_loss": val_loss_sum / val_tokens,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one JSON artifact in its destination directory."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_result(
    trial_dir: Path,
    overrides_sha256: str,
    metric_record: Mapping[str, Any],
) -> None:
    """Write an attempt-scoped final-checkpoint result artifact."""
    val_loss = metric_record.get("val_loss")
    if (
        isinstance(val_loss, bool)
        or not isinstance(val_loss, int | float)
        or not math.isfinite(val_loss)
    ):
        raise ValueError(f"Final evaluation is missing finite numeric val_loss: {metric_record!r}")
    step = metric_record.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError(f"Final evaluation is missing its integer step: {metric_record!r}")
    if (
        metric_record.get("policy") != "final_checkpoint"
        or metric_record.get("checkpoint") != "final.pt"
    ):
        raise ValueError(f"Final evaluation has unexpected checkpoint metadata: {metric_record!r}")

    generation_id = os.environ.get("PHASESWEEP_GENERATION_ID")
    attempt_id = os.environ.get("PHASESWEEP_ATTEMPT_ID")
    expected_overrides_sha256 = os.environ.get("PHASESWEEP_OVERRIDES_SHA256")
    if not generation_id or not attempt_id:
        raise ValueError("PhaseSweep generation and attempt IDs are required for result evidence")
    if not expected_overrides_sha256 or overrides_sha256 != expected_overrides_sha256:
        raise ValueError("Resolved overrides do not match this PhaseSweep attempt")

    result = {
        "attempt_id": attempt_id,
        "evaluation": {
            "checkpoint": "final.pt",
            "policy": "final_checkpoint",
            "step": step,
        },
        "generation_id": generation_id,
        "objective": {
            "name": "val_loss",
            "split": "validation",
            "value": float(val_loss),
        },
        "overrides_sha256": overrides_sha256,
        "schema_version": 1,
        "status": "complete",
    }
    _atomic_write_json(trial_dir / "result.json", result)


def _run_template(template_root: Path, config_path: Path) -> None:
    """Run the upstream decoder trainer with the composed YAML config."""
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
    overrides, overrides_sha256 = _read_json_mapping(Path(args.overrides_path))
    trial_dir = Path(args.trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)

    trainer_run_dir = trial_dir / "trainer"
    composed = _deep_merge(base_config, overrides)
    composed["run_dir"] = str(trainer_run_dir)

    generated_config = trial_dir / "decoder_config.yaml"
    generated_config.write_text(yaml.safe_dump(composed, sort_keys=True))

    _run_template(template_root, generated_config)
    metric_record = _evaluate_final_checkpoint(template_root, trainer_run_dir)
    _write_result(trial_dir, overrides_sha256, metric_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
