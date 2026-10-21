"""Report-only observations: recorded per trial, published with the winner, never ranked."""

from __future__ import annotations

import csv
import logging
from copy import deepcopy
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
from phasesweep.engine.paths import _generation_winner_path, _winner_path
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.read import _winner_view, read_result
from phasesweep.engine.repeats import repeat_axis
from phasesweep.engine.selection import select_winner
from phasesweep.engine.state import observed_attr
from phasesweep.evidence.models import JsonEnvelopeExtractor, JsonExtractor, LogRegexExtractor
from tests.conftest import _phase_trials, make_experiment, write_trainer
from tests.ledger_fixtures import golden_publication, reseal_winner


def _log_observation(name: str, pattern: str) -> Observation:
    """Return a ``log_regex`` observation reading ``pattern`` from stdout."""
    return Observation(name=name, extractor=LogRegexExtractor(type="log_regex", pattern=pattern))


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
    assert yaml.safe_load(_winner_path(experiment, "p").read_text())["observed"] == {"mem": 3.0}
    _, (published,) = read_result(experiment, phase_names=["p"])
    assert published is not None
    assert published.observed == {"mem": 3.0}

    with (_winner_path(experiment, "p").parent / "trials.csv").open(newline="") as handle:
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
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))
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
    assert yaml.safe_load(_winner_path(experiment, "p").read_text())["observed"] == {}


@pytest.mark.parametrize(
    ("observed", "reason"),
    [
        pytest.param(None, "observed must be a mapping", id="absent"),
        pytest.param([1.0], "observed must be a mapping", id="not-a-mapping"),
        pytest.param({"": 1.0}, "must be a nonempty string", id="blank-name"),
        pytest.param({"peak": "7"}, "must be a number", id="not-a-number"),
        pytest.param({"peak": True}, "must be a number", id="boolean"),
        pytest.param({"peak": float("nan")}, "must be finite", id="not-finite"),
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
    bare = _winner_view(winner, "p")
    assert bare is not None
    assert bare.observed == {}

    winner["observed"] = {"peak": 7.5}
    reported = _winner_view(winner, "p")
    assert reported is not None
    assert reported.observed == {"peak": 7.5}

    winner["observed"] = {"peak": float("inf")}
    assert _winner_view(winner, "p") is None
    del winner["observed"]
    assert _winner_view(winner, "p") is None


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
    trials = [deepcopy(trial) for trial in _phase_trials(experiment)]
    trials[0].user_attrs[observed_attr("mem")] = "3.0"
    study = optuna.create_study(direction="minimize")
    study.add_trials(trials)

    with pytest.raises(TrialEvidenceMissingError) as raised:
        select_winner(study, experiment, phase_name="p", repeats=repeat_axis(experiment.phases[0]))

    assert str(raised.value).startswith(f"{subject} has a malformed 'observed:' value")
