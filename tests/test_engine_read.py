"""engine.read: permissive status/winner reads that never raise on a partial file."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
import yaml

import phasesweep.engine.optuna as engine_optuna
from phasesweep.config import (
    Experiment,
    FloatParam,
    JsonEnvelopeExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    WandbExtractor,
)
from phasesweep.engine import read_status, read_winner, read_winners
from phasesweep.engine.state import (
    _generation_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_path,
    _winner_path,
)
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
    assert status["metric"]["objective_evidence"] == {
        "kind": "log_regex",
        "attempt_location_scoped": True,
        "attempt_identity_bound": False,
        "source_identity_keyed": False,
        "objective_name_bound": False,
        "split_bound": False,
        "evaluation_policy_bound": False,
        "checkpoint_declared": False,
        "checkpoint_value_bound": False,
        "expected_step_declared": False,
        "expected_step_value_bound": False,
    }


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


def _mark_generation_published(exp: Experiment, generation_id: str, phase_name: str) -> None:
    """Publish an immutable generation (summary + record + phase winner) on disk.

    Pointer validation (:func:`phasesweep.engine.state._last_successful_generation_id`)
    now reads back the generation's own immutable *summary*, not the
    lifecycle record's state, so a summary naming this owner and id is
    required for the pointer to resolve as published (review v0.5.15 /
    blocker 3). The record is written too, purely as the informational,
    post-commit artifact real publications also produce.
    """
    _last_successful_generation_path(exp).parent.mkdir(parents=True, exist_ok=True)
    _last_successful_generation_path(exp).write_text(
        yaml.safe_dump({"experiment": exp.experiment, "generation_id": generation_id})
    )
    summary_path = _generation_summary_path(exp, generation_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        yaml.safe_dump({"experiment": exp.experiment, "generation_id": generation_id})
    )
    record_path = _generation_record_path(exp, generation_id)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        yaml.safe_dump(
            {"experiment": exp.experiment, "generation_id": generation_id, "state": "published"}
        )
    )
    winner_path = _generation_winner_path(exp, generation_id, phase_name)
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_text(
        yaml.safe_dump(
            {
                "phase": phase_name,
                "trial_number": 0,
                "metric": {"loss": 0.1, "goal": "minimize"},
                "params": {"lr": 0.001},
                "effective_overrides": {"lr": 0.001},
                "completion": {"incomplete": False},
                "generation_id": generation_id,
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": phase_name,
                    "trial_number": 0,
                    "generation_id": generation_id,
                    "attempt_id": None,
                    "study": None,
                },
            }
        )
    )


def test_read_status_distinguishes_current_from_published_generation(tmp_path: Path) -> None:
    """A failed rerun's counts must never be mistaken for the older published winner's."""
    exp = _experiment(tmp_path)
    _mark_generation_published(exp, "generation-good", "p")
    # A newer generation has since started (and, in this scenario, failed) without
    # ever publishing - the mutable current-generation pointer moved on.
    _generation_path(exp).parent.mkdir(parents=True, exist_ok=True)
    _generation_path(exp).write_text(yaml.safe_dump({"generation_id": "generation-failed"}))

    status = read_status(exp)

    assert status["current_generation_id"] == "generation-failed"
    assert status["published_generation_id"] == "generation-good"
    assert status["current_generation_id"] != status["published_generation_id"]
    # In default (non-pinned) mode, represented_generation_id is the captured
    # published id, and winner/summary facts scope to it -- never the
    # failed/in-progress current one.
    assert status["represented_generation_id"] == "generation-good"
    assert status["is_published"] is True
    assert status["phases"][0]["winner_present"] is True
    assert status["summary_present"] is True


def test_read_status_explicit_generation_id_is_self_scoped(tmp_path: Path) -> None:
    """An explicit generation_id pins the represented identity to that one generation.

    ``current_generation_id`` and ``published_generation_id`` always report
    the *actual* pointers -- never forced to the pinned id (review v0.5.15 /
    blocker 3, defect 2: "pinned reads lie"). Here no current-pointer file
    exists at all, so ``current_generation_id`` is genuinely ``None`` even
    though the pinned generation itself is fully published.
    """
    exp = _experiment(tmp_path)
    _mark_generation_published(exp, "generation-good", "p")

    status = read_status(exp, generation_id="generation-good")

    assert status["current_generation_id"] is None
    assert status["published_generation_id"] == "generation-good"
    assert status["represented_generation_id"] == "generation-good"
    assert status["is_published"] is True
    assert status["phases"][0]["winner_present"] is True


def test_read_status_pinned_read_of_unpublished_generation_is_not_marked_published(
    tmp_path: Path,
) -> None:
    """Pinning a generation that never published reports is_published=False, not a lie."""
    exp = _experiment(tmp_path)
    _mark_generation_published(exp, "generation-good", "p")
    # A second, never-published generation has its own (unpublished) winner.
    other_winner = _generation_winner_path(exp, "generation-orphan", "p")
    other_winner.parent.mkdir(parents=True, exist_ok=True)
    other_winner.write_text(
        yaml.safe_dump(
            {
                "phase": "p",
                "trial_number": 1,
                "metric": {"loss": 0.2, "goal": "minimize"},
                "params": {"lr": 0.002},
                "effective_overrides": {"lr": 0.002},
                "completion": {"incomplete": False},
                "generation_id": "generation-orphan",
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": "p",
                    "trial_number": 1,
                    "generation_id": "generation-orphan",
                    "attempt_id": None,
                    "study": None,
                },
            }
        )
    )

    status = read_status(exp, generation_id="generation-orphan")

    assert status["published_generation_id"] == "generation-good"
    assert status["represented_generation_id"] == "generation-orphan"
    assert status["is_published"] is False
    # The orphaned generation's own winner is still readable pinned.
    assert status["phases"][0]["winner_present"] is True


def test_objective_evidence_assurance_json_envelope_without_declared_checkpoint(
    tmp_path: Path,
) -> None:
    """An undeclared checkpoint/expected_step must never be reported as bound."""
    exp = _experiment(tmp_path)
    exp = exp.model_copy(
        update={
            "metric": Metric(
                name="loss",
                goal="minimize",
                extractor=JsonEnvelopeExtractor(
                    type="json_envelope",
                    path="result.json",
                    objective_name="loss",
                    split="test",
                    policy="test",
                ),
            )
        }
    )

    status = read_status(exp)

    assert status["metric"]["objective_evidence"] == {
        "kind": "json_envelope",
        "attempt_location_scoped": True,
        "attempt_identity_bound": True,
        "source_identity_keyed": False,
        "objective_name_bound": True,
        "split_bound": True,
        "evaluation_policy_bound": True,
        "checkpoint_declared": False,
        "checkpoint_value_bound": False,
        "expected_step_declared": False,
        "expected_step_value_bound": False,
    }


def test_objective_evidence_assurance_json_envelope_with_declared_checkpoint(
    tmp_path: Path,
) -> None:
    """A declared checkpoint/expected_step is reported as genuinely value-bound."""
    exp = _experiment(tmp_path)
    exp = exp.model_copy(
        update={
            "metric": Metric(
                name="loss",
                goal="minimize",
                extractor=JsonEnvelopeExtractor(
                    type="json_envelope",
                    path="result.json",
                    objective_name="loss",
                    split="test",
                    policy="test",
                    checkpoint="ckpt-42",
                    expected_step=1000,
                ),
            )
        }
    )

    status = read_status(exp)

    assert status["metric"]["objective_evidence"] == {
        "kind": "json_envelope",
        "attempt_location_scoped": True,
        "attempt_identity_bound": True,
        "source_identity_keyed": False,
        "objective_name_bound": True,
        "split_bound": True,
        "evaluation_policy_bound": True,
        "checkpoint_declared": True,
        "checkpoint_value_bound": True,
        "expected_step_declared": True,
        "expected_step_value_bound": True,
    }


@pytest.mark.parametrize(
    ("extractor", "expected_triple"),
    [
        pytest.param(
            JsonEnvelopeExtractor(
                type="json_envelope",
                path="result.json",
                objective_name="loss",
                split="test",
                policy="test",
            ),
            (True, True, False),
            id="json_envelope",
        ),
        pytest.param(
            LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
            (True, False, False),
            id="log_regex",
        ),
        pytest.param(
            WandbExtractor(type="wandb", entity="acme", project="proj", metric_key="eval/loss"),
            (True, False, True),
            id="wandb",
        ),
    ],
)
def test_objective_evidence_assurance_attempt_triple_by_kind(
    tmp_path: Path,
    extractor: JsonEnvelopeExtractor | LogRegexExtractor | WandbExtractor,
    expected_triple: tuple[bool, bool, bool],
) -> None:
    """Each extractor kind reports its own (location, identity, source-key) triple.

    ``json_envelope`` structurally echoes and cross-checks the attempt
    identity; ``wandb`` is keyed by an immutable run id that IS the attempt
    id; ``log_regex`` is merely read from an attempt-scoped location with
    nothing in its contents tying it to that attempt (review v0.5.15 / item C).
    """
    exp = _experiment(tmp_path)
    exp = exp.model_copy(
        update={"metric": Metric(name="loss", goal="minimize", extractor=extractor)}
    )

    status = read_status(exp)
    evidence = status["metric"]["objective_evidence"]

    assert (
        evidence["attempt_location_scoped"],
        evidence["attempt_identity_bound"],
        evidence["source_identity_keyed"],
    ) == expected_triple
