"""engine.read: permissive status/winner reads that never raise on a partial file."""

from __future__ import annotations

from pathlib import Path

import optuna
import yaml

from phasesweep.config import Experiment, load_config
from phasesweep.engine import read_status, read_winner, read_winners
from phasesweep.engine.state import _winner_path


def _experiment(tmp_path: Path, *, storage: str | None = None) -> Experiment:
    config = tmp_path / "exp.yaml"
    storage_line = f"storage: {storage}\n" if storage is not None else ""
    config.write_text(
        """\
experiment: read_t
{storage_line}\
workdir: {wd}
trial_command: "python x.py {{overrides}}"
override_format: argparse
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json, path: r.json, key: loss }}
phases:
  - name: p
    n_trials: 1
    search_space:
      lr: {{ type: float, low: 1.0e-5, high: 1.0e-2, log: true }}
""".format(storage_line=storage_line, wd=tmp_path / "wd")
    )
    parsed = load_config(config)
    assert isinstance(parsed, Experiment)
    return parsed


def test_read_winner_parses_a_valid_file(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "phase": "p",
                "trial_number": 3,
                "metric": {"loss": 0.123, "goal": "minimize"},
                "params": {"lr": 0.001},
                "effective_overrides": {"lr": 0.001},
                "gates": [{"name": "g", "passed": True}],
                "completion": {"incomplete": False},
            }
        )
    )
    view = read_winner(exp, "p")
    assert view is not None
    assert view.trial_number == 3
    assert view.metric == 0.123
    assert view.gates_passed is True
    assert view.incomplete is False


def test_read_winner_tolerates_torn_or_malformed_file(tmp_path: Path) -> None:
    # Status reads stay permissive for legacy, hand-edited, or externally corrupted files.
    # Both a truncated file (invalid YAML) and a valid-YAML file missing required keys must
    # read as "no winner yet", never raise.
    exp = _experiment(tmp_path)
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text('{"trial_number": 0, "metric": {"loss":')  # truncated -> YAMLError
    assert read_winner(exp, "p") is None
    assert read_winners(exp) == []

    path.write_text("phase: p\n")  # valid YAML, missing trial_number/metric -> KeyError
    assert read_winner(exp, "p") is None
    assert read_winners(exp) == []


def test_read_status_does_not_create_missing_sqlite_storage(tmp_path: Path) -> None:
    db = tmp_path / "missing.db"
    exp = _experiment(tmp_path, storage=f"sqlite:///{db}")

    status = read_status(exp)

    assert not db.exists()
    assert status["phases"][0]["trials"] == {}


def test_read_status_tolerates_uninitialized_sqlite_file(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    db.touch()
    exp = _experiment(tmp_path, storage=f"sqlite:///{db}")

    status = read_status(exp)

    assert status["phases"][0]["trials"] == {}


def test_read_status_counts_existing_sqlite_trials_without_optuna_loader(
    tmp_path: Path,
) -> None:
    db = tmp_path / "phases.db"
    storage = f"sqlite:///{db}"
    optuna.create_study(study_name="read_t::p", storage=storage).optimize(
        lambda trial: 1.0, n_trials=1
    )
    exp = _experiment(tmp_path, storage=storage)

    status = read_status(exp)

    assert status["phases"][0]["trials"] == {"COMPLETE": 1}


def test_read_status_counts_sqlite_trials_with_url_options(tmp_path: Path) -> None:
    db = tmp_path / "phases.db"
    storage = f"sqlite:///{db}"
    optuna.create_study(study_name="read_t::p", storage=storage).optimize(
        lambda trial: 1.0, n_trials=1
    )
    exp = _experiment(tmp_path, storage=f"{storage}?timeout=30")

    status = read_status(exp)

    assert status["phases"][0]["trials"] == {"COMPLETE": 1}


def test_read_status_counts_sqlite_trials_with_uri_filename(tmp_path: Path) -> None:
    db = tmp_path / "uri.db"
    storage = f"sqlite:///file:{db}?mode=rwc&uri=true"
    optuna.create_study(study_name="read_t::p", storage=storage).optimize(
        lambda trial: 1.0, n_trials=1
    )
    exp = _experiment(tmp_path, storage=storage)

    status = read_status(exp)

    assert status["phases"][0]["trials"] == {"COMPLETE": 1}
