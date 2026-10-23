"""Raw experiment payloads and YAML files used by configuration tests."""

from __future__ import annotations

from pathlib import Path

import yaml


def json_metric(name: str = "x", **extractor_fields: object) -> dict[str, object]:
    """Build a JSON-envelope metric, replacing whole extractor fields."""
    return {
        "extractor": {
            "type": "json_envelope",
            "objective_name": name,
            "split": "test",
            "policy": "test",
            **extractor_fields,
        }
    }


def phase_payload(**fields: object) -> dict[str, object]:
    """Build one seeded random search phase, replacing whole fields."""
    return {
        "name": "p",
        "n_trials": 1,
        "sampler": {"type": "random", "seed": 0},
        "search_space": {"x": {"type": "int", "low": 0, "high": 10}},
        **fields,
    }


def experiment_yaml(**fields: object) -> str:
    """Serialize an unvalidated experiment, replacing whole top-level fields."""
    payload = {
        "experiment": "t",
        "trial_command": "echo {overrides}",
        "override_format": "argparse",
        "metric": json_metric(),
        "phases": [phase_payload()],
        **fields,
    }
    if payload.get("storage") is not None:
        payload.setdefault("provenance", {"revision": "test-fixture-v1"})
    return yaml.safe_dump(payload, sort_keys=False)


def write_experiment(path: Path, **fields: object) -> Path:
    """Write an unvalidated experiment to an existing parent directory."""
    path.write_text(experiment_yaml(**fields), encoding="utf-8")
    return path


def anchored_phase(high: int = 3) -> dict[str, object]:
    """Build the search phase used by anchored-path and fingerprint tests."""
    return phase_payload(search_space={"x": {"type": "int", "low": 0, "high": high}})


def write_anchored_config(path: Path, **fields: object) -> Path:
    """Write an anchored experiment config whose trial echoes ``x=0.5``, with ``fields`` overriding its keys, and return its path."""
    payload = {
        "experiment": "anchored",
        "trial_command": "echo x=0.5 {overrides}",
        "metric": {"extractor": {"type": "log_regex", "pattern": r"x=(?P<value>[0-9.eE+-]+)"}},
        "phases": [anchored_phase()],
        **fields,
    }
    return write_experiment(path, **payload)


def project_directory(tmp_path: Path) -> Path:
    """Create and return the resolved directory that holds a config file."""
    project = tmp_path / "project"
    project.mkdir()
    return project.resolve()
