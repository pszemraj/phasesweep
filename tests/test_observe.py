"""Report-only observations: recorded per trial, published with the winner, never ranked."""

from __future__ import annotations

import csv
import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import optuna
import pytest
import yaml
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import Observation, Phase
from phasesweep.config.search import Repeats, Sampler
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _generation_winner_path, _phase_dir
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.read import _trial_table, _winner_view, read_result
from phasesweep.engine.repeats import repeat_axis
from phasesweep.engine.selection import select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    constraint_attr,
    observed_attr,
)
from phasesweep.engine.trial import TrialResult, record_trial_result
from phasesweep.evidence.evaluation import GateResult
from phasesweep.evidence.models import JsonEnvelopeExtractor, JsonExtractor, LogRegexExtractor
from phasesweep.tracking.publication import _accepted_projection
from tests.conftest import (
    log_regex_provenance,
    make_experiment,
    phase_trials,
    write_experiment_config,
    write_trainer,
)
from tests.ledger_fixtures import golden_publication, published_generation_id, reseal_winner


def _log_observation(name: str, pattern: str) -> Observation:
    """Return a ``log_regex`` observation reading ``pattern`` from stdout."""
    return Observation(name=name, extractor=LogRegexExtractor(type="log_regex", pattern=pattern))


def test_retry_replaces_optional_trial_results(tmp_path: Path) -> None:
    """A retained extraction retry replaces optional evidence across all result readers."""
    storage = f"journal:///{tmp_path / 'retry.journal'}"
    study = optuna.create_study(study_name="retry", storage=_resolve_storage(storage))
    trial = study.ask()
    for key, value in {
        ATTEMPT_ID_ATTR: "attempt",
        GENERATION_ID_ATTR: "generation",
        TRAINER_ENV_ATTR: {},
        TRAINER_ENV_DIGEST_ATTR: "a" * 64,
        TRAINER_INPUT_ATTR: {"schema_version": 1},
    }.items():
        trial.set_user_attr(key, value)
    original = TrialResult(
        metric=1.0,
        constraints={"old": 2.0},
        return_code=0,
        duration_seconds=1.0,
        feasible=True,
        gate_results=[GateResult("required_file", True, "present")],
        objective_provenance=log_regex_provenance(),
        observed={"old": 3.0, "kept": 4.0},
    )
    record_trial_result(trial, original, overrides_json="{}")

    study = optuna.load_study(study_name="retry", storage=_resolve_storage(storage))
    retained = study.get_trials()[0]
    retried = optuna.Trial(study, retained._trial_id)
    record_trial_result(
        retried,
        replace(original, metric=0.5, constraints={}, gate_results=None, observed={"kept": 5.0}),
        overrides_json="{}",
    )
    study.tell(retried, 0.5)
    completed = study.get_trials()[0]
    assert completed.user_attrs[constraint_attr("old")] is None
    assert completed.user_attrs[observed_attr("old")] is None
    assert completed.user_attrs[GATES_ATTR] is None

    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    assert select_winner(study, experiment, emit_advisories=False).observed == {"kept": 5.0}
    table = _trial_table(experiment, experiment.phases[0], [completed], None)
    assert table.rows[0][observed_attr("old")] is None
    assert table.rows[0][observed_attr("kept")] == 5.0
    projection = _accepted_projection(completed, {})
    assert projection["summary"]["observed"] == {"kept": 5.0}
    assert projection["summary"]["constraints"] == {}
    assert "phasesweep.observed.old" not in projection["metrics"]
    assert "phasesweep.constraint.old" not in projection["metrics"]

    failed = study.ask()
    rejected = TrialResult(
        metric=None,
        constraints={},
        return_code=0,
        duration_seconds=1.0,
        feasible=False,
        failure_reason="missing evidence",
    )
    record_trial_result(failed, original, overrides_json="{}")
    record_trial_result(failed, rejected, overrides_json="{}")
    assert failed.user_attrs[OBJECTIVE_PROVENANCE_ATTR] is None
    assert failed.user_attrs[constraint_attr("old")] is None
    assert failed.user_attrs[observed_attr("old")] is None

    fresh = study.ask()
    record_trial_result(fresh, rejected, overrides_json="{}")
    assert OBJECTIVE_PROVENANCE_ATTR not in fresh.user_attrs
    assert GATES_ATTR not in fresh.user_attrs
    assert constraint_attr("old") not in fresh.user_attrs
    assert observed_attr("old") not in fresh.user_attrs


def test_observe_failure_records_nothing_and_keeps_trial_complete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed or non-finite read warns and records nothing; the trial and winner are unaffected."""
    caplog.set_level(logging.WARNING, logger="phasesweep")
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os
        trial = os.environ["PHASESWEEP_TRIAL_ID"]
        print("x=" + {"0": "0.5", "1": "1.0", "2": "2.0"}[trial])
        if trial != "1":
            print("mem=" + {"0": "3.0", "2": "5.0"}[trial])
        print("bad=inf")
        """,
    )
    storage = f"journal:///{tmp_path / 'observe.journal'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trainer=trainer,
        n_trials=3,
        observe=[
            _log_observation("mem", r"mem=(?P<value>[0-9.]+)"),
            Observation(
                name="absent",
                extractor=JsonExtractor(type="json", path="never.json", key="v"),
            ),
            _log_observation("unbounded", r"bad=(?P<value>\S+)"),
        ],
    )

    winners = run_experiment(experiment)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(storage))
    trials = study.get_trials(deepcopy=False)
    assert [t.state for t in trials] == [optuna.trial.TrialState.COMPLETE] * 3
    assert [t.user_attrs.get(observed_attr("mem")) for t in trials] == [3.0, None, 5.0]
    assert not any(
        observed_attr(name) in t.user_attrs for t in trials for name in ("absent", "unbounded")
    )
    messages = [record.getMessage() for record in caplog.records]
    for name in ("absent", "unbounded"):
        assert sum(f"observation {name} not recorded" in m for m in messages) == 3
    assert sum("observation mem not recorded" in m for m in messages) == 1

    # The best metric wins whatever was observed, and its own values are published.
    assert winners["p"].trial_number == 0
    assert winners["p"].observed == {"mem": 3.0}
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    assert yaml.safe_load(_generation_winner_path(experiment, generation_id, "p").read_text())[
        "observed"
    ] == {"mem": 3.0}
    _, (published,) = read_result(experiment, phase_names=["p"])
    assert published is not None
    assert published.observed == {"mem": 3.0}

    with (_phase_dir(experiment, "p") / "trials.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["user_attr:observed:mem"] for row in rows] == ["3.0", "", "5.0"]
    assert "user_attr:observed:absent" not in rows[0]


def test_validate_and_dry_run_name_each_observation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Both previews print one line per observation, so a mistyped key shows before a run."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        n_trials=1,
        observe=[
            Observation(
                name="heldout_f05",
                extractor=JsonExtractor(type="json", path="result.json", key="heldout_f05"),
            ),
            _log_observation("mem", r"mem=(?P<value>[0-9.]+)"),
            Observation(
                name="envelope",
                extractor=JsonEnvelopeExtractor(
                    type="json_envelope", objective_name="loss", split="test", policy="final"
                ),
            ),
        ],
    )
    config = tmp_path / "experiment.yaml"
    write_experiment_config(config, experiment)
    expected = [
        "observe heldout_f05: json result.json key heldout_f05 (report-only)",
        "observe mem: log_regex stdout.log pattern 'mem=(?P<value>[0-9.]+)' select last "
        "(report-only)",
        "observe envelope: json_envelope result.json objective loss split test policy final "
        "(report-only)",
    ]

    result = CliRunner().invoke(cli, ["validate", str(config)])
    assert result.exit_code == 0, result.output
    assert all(f"  {line}\n" in result.output for line in expected)

    with caplog.at_level(logging.INFO, logger="phasesweep"):
        run_experiment(experiment, dry_run=True)
    assert all(f"DRY RUN {line}" in caplog.text for line in expected)
    assert not (tmp_path / "runs").exists()


def test_winner_without_observations_publishes_an_empty_map(tmp_path: Path) -> None:
    """An experiment that declares no observations still publishes the field."""
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_trainer(tmp_path / "t.py", 'print("x=1.0")')
    )
    winners = run_experiment(experiment)

    assert winners["p"].observed == {}
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    assert (
        yaml.safe_load(_generation_winner_path(experiment, generation_id, "p").read_text())[
            "observed"
        ]
        == {}
    )


@pytest.mark.parametrize(
    ("observed", "reason"),
    [
        pytest.param(None, "observed must be a mapping", id="absent"),
        pytest.param([1.0], "observed must be a mapping", id="not-a-mapping"),
        pytest.param({"": 1.0}, "must be a nonempty string", id="blank-name"),
        pytest.param({"peak": "7"}, "must be a number", id="not-a-number"),
        pytest.param({"peak": True}, "must be a number", id="boolean"),
        pytest.param({"peak": float("nan")}, "must be finite", id="not-finite"),
        pytest.param({"peak": 10**400}, "must be finite", id="overflow"),
    ],
)
def test_manifest_rejects_a_malformed_observed_block(
    tmp_path: Path, observed: object, reason: str
) -> None:
    """The published winner's observed field is validated, and absence is refused."""
    experiment, generation_id = golden_publication(tmp_path)
    winner = yaml.safe_load(_generation_winner_path(experiment, generation_id, "p").read_text())
    assert winner["observed"] == {}
    if observed is None:
        del winner["observed"]
    else:
        winner["observed"] = observed
    reseal_winner(experiment, generation_id, winner)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert "has no valid observed block" in pointer.error
    assert reason in pointer.error


def test_winner_view_projects_observed_values(tmp_path: Path) -> None:
    """The permissive reader carries observed values through and drops a malformed winner."""
    experiment, generation_id = golden_publication(tmp_path)
    winner = yaml.safe_load(_generation_winner_path(experiment, generation_id, "p").read_text())
    bare = _winner_view(winner, "p", persistent=True)
    assert bare is not None
    assert bare.observed == {}

    winner["observed"] = {"peak": 7.5}
    reported = _winner_view(winner, "p", persistent=True)
    assert reported is not None
    assert reported.observed == {"peak": 7.5}

    for invalid in (float("inf"), 10**400):
        winner["observed"] = {"peak": invalid}
        assert _winner_view(winner, "p", persistent=True) is None
    del winner["observed"]
    assert _winner_view(winner, "p", persistent=True) is None


def test_manifest_accepts_a_well_formed_observed_block(tmp_path: Path) -> None:
    """A resealed winner with finite values validates and reads back."""
    experiment, generation_id = golden_publication(tmp_path)
    winner = yaml.safe_load(_generation_winner_path(experiment, generation_id, "p").read_text())
    winner["observed"] = {"peak": 7.5, "steps": 3}
    reseal_winner(experiment, generation_id, winner)

    assert _resolve_publication_pointer(experiment).state == "ok"
    _, (view,) = read_result(experiment, phase_names=["p"])
    assert view is not None
    assert view.observed == {"peak": 7.5, "steps": 3.0}


@pytest.mark.parametrize(
    ("phase_shape", "subject"),
    [
        pytest.param({"n_trials": 1}, "Winning trial 0", id="single-trial"),
        pytest.param(
            {
                "phases": [
                    Phase(
                        name="p",
                        n_trials=3,
                        sampler=Sampler(type="grid"),
                        repeats=Repeats(key="seed", values=[0, 1, 2]),
                    )
                ]
            },
            "Trial 0 of the winning repeat group",
            id="repeat-group-member",
        ),
    ],
)
def test_malformed_observed_value_names_the_trial_it_came_from(
    tmp_path: Path, phase_shape: dict[str, object], subject: str
) -> None:
    """A repeat member's bad value is not reported as the winning trial's."""
    trainer = write_trainer(tmp_path / "t.py", 'print("x=0.5")\nprint("mem=3.0")')
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        observe=[_log_observation("mem", r"mem=(?P<value>[0-9.]+)")],
        **phase_shape,
    )
    run_experiment(experiment)
    trials = [deepcopy(trial) for trial in phase_trials(experiment)]
    trials[0].user_attrs[observed_attr("mem")] = "3.0"
    study = optuna.create_study(direction="minimize")
    study.add_trials(trials)

    with pytest.raises(TrialEvidenceMissingError) as raised:
        select_winner(study, experiment, phase_name="p", repeats=repeat_axis(experiment.phases[0]))

    assert str(raised.value).startswith(f"{subject} has a malformed 'observed:' value")
