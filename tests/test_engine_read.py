"""engine.read: permissive status/winner reads that never raise on a partial file."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
import yaml

import phasesweep.engine.optuna as engine_optuna
from phasesweep.config import Experiment, FloatParam, LogRegexExtractor, Metric, Phase
from phasesweep.engine import read_status, read_winner, read_winners
from phasesweep.engine.state import _winner_path
from tests.conftest import make_experiment


def _experiment(tmp_path: Path, *, storage: str | None = None) -> Experiment:
    return make_experiment(
        experiment="read_t",
        workdir=tmp_path / "wd",
        storage=storage,
        trial_command="python x.py {overrides}",
        metric=Metric(
            name="loss",
            goal="minimize",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        ),
        phases=[
            Phase(
                name="p",
                n_trials=1,
                search_space={"lr": FloatParam(type="float", low=1.0e-5, high=1.0e-2, log=True)},
            )
        ],
    )


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
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": "p",
                    "trial_number": 3,
                    "generation_id": "generation-test",
                    "attempt_id": "attempt-test",
                    "study": None,
                },
            }
        )
    )
    view = read_winner(exp, "p")
    assert view is not None
    assert view.trial_number == 3
    assert view.metric == 0.123
    assert view.gates_passed is True
    assert view.incomplete is False
    assert view.source is not None
    assert view.source.phase == "p"
    assert view.source.trial_number == 3


@pytest.mark.parametrize(
    "body",
    [
        '{"trial_number": 0, "metric": {"loss":',
        "phase: p\n",
        "- not\n- a\n- mapping\n",
        """\
phase: p
trial_number: 3
metric: {loss: 0.123, goal: minimize}
params: {lr: 0.001}
effective_overrides: {lr: 0.001}
completion: [not, a, mapping]
""",
    ],
    ids=["truncated", "missing_keys", "non_mapping", "bad_completion"],
)
def test_read_winner_tolerates_torn_or_malformed_file(tmp_path: Path, body: str) -> None:
    # Status reads stay permissive for legacy, hand-edited, or externally corrupted files.
    exp = _experiment(tmp_path)
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(body)
    assert read_winner(exp, "p") is None
    assert read_winners(exp) == []


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
def test_read_status_does_not_create_missing_storage(tmp_path: Path, backend: str) -> None:
    path = tmp_path / f"missing.{backend}"
    exp = _experiment(tmp_path, storage=f"{backend}:///{path}")

    status = read_status(exp)

    assert not path.exists()
    assert status["phases"][0]["trials"] == {}
    assert status["phases"][0]["trial_data_available"] is False


def test_read_status_tolerates_uninitialized_sqlite_file(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    db.touch()
    exp = _experiment(tmp_path, storage=f"sqlite:///{db}")

    status = read_status(exp)

    assert status["phases"][0]["trials"] == {}
    assert status["phases"][0]["trial_data_available"] is False


def test_read_status_uses_one_sqlite_snapshot_per_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "phases.db"
    storage = f"sqlite:///{db}"
    optuna.create_study(study_name="read_t::p", storage=storage).optimize(
        lambda trial: 1.0, n_trials=1
    )
    exp = _experiment(tmp_path, storage=storage)
    real_connect = engine_optuna.sqlite3.connect
    connections = 0

    def counting_connect(*args: object, **kwargs: object):
        nonlocal connections
        connections += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(engine_optuna.sqlite3, "connect", counting_connect)

    status = read_status(exp)

    assert connections == 1
    assert status["phases"][0]["trials"] == {"COMPLETE": 1}
    assert status["phases"][0]["trial_data_available"] is True


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
