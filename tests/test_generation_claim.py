"""A generation exists only for an invocation that launches a trial or publishes changed winners."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import phasesweep.engine.generation as generation_ops
from phasesweep import load_experiment, run_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.errors import (
    PublicationCommitError,
    StudyFingerprintMismatchError,
    TrialTargetRefusedError,
)
from phasesweep.engine.paths import (
    _generation_dir,
    _generation_path,
    _generation_record_path,
    _generations_dir,
    _last_successful_generation_path,
)
from phasesweep.engine.read import read_status
from phasesweep.engine.run import TerminalReport
from phasesweep.mcp.redaction import status_payload
from phasesweep.mcp.runs import RunStore
from tests.conftest import invoke_cli_boundary, make_experiment
from tests.experiments import write_experiment
from tests.ledger_fixtures import published_generation_id
from tests.mcp_helpers import run_claimed_runner

pytestmark = pytest.mark.integration

_ECHO = "echo x=0.5 {overrides}"


def _tree(experiment: Experiment) -> tuple[list[str], bytes | None, bytes | None]:
    """Return the generation directories and the bytes of both generation pointers.

    :param Experiment experiment: Experiment whose artifact tree is read.
    :return tuple[list[str], bytes | None, bytes | None]: Sorted generation ids, then the
        current pointer and the last-success pointer, each ``None`` when absent.
    """
    root = _generations_dir(experiment)
    names = sorted(path.name for path in root.iterdir()) if root.is_dir() else []
    current, published = (
        path.read_bytes() if path.exists() else None
        for path in (_generation_path(experiment), _last_successful_generation_path(experiment))
    )
    return names, current, published


def _two_phases(tmp_path: Path) -> Experiment:
    """Build a persistent two-phase experiment whose echo trainer always measures 0.5.

    :param Path tmp_path: Directory holding the ledger and the workdir.
    :return Experiment: The experiment.
    """
    sampler = Sampler(type="random", seed=0)
    return make_experiment(
        persistent=tmp_path,
        trial_command=_ECHO,
        phases=[
            Phase(
                name="a",
                n_trials=2,
                sampler=sampler,
                search_space={"x": IntParam(type="int", low=0, high=10)},
            ),
            Phase(
                name="b",
                inherits=["a"],
                n_trials=2,
                sampler=sampler,
                search_space={"y": IntParam(type="int", low=0, high=10)},
            ),
        ],
    )


def test_a_replay_of_the_published_winners_writes_no_generation(tmp_path: Path) -> None:
    """A run, or a resume, that launches nothing and matches the publication is a no-op."""
    experiment = _two_phases(tmp_path)
    reports: list[TerminalReport] = []
    winners = run_experiment(experiment, terminal_callback=reports.append)
    published = reports[0].generation_id
    assert reports[0].replayed_generation_id is None
    before = _tree(experiment)
    assert before[0] == [published]

    for from_phase in (None, "b"):
        assert (
            run_experiment(experiment, from_phase=from_phase, terminal_callback=reports.append)
            == winners
        )
        assert reports[-1].primary_error is None
        assert reports[-1].winners == winners
        assert reports[-1].replayed_generation_id == published
        assert not _generation_dir(experiment, reports[-1].generation_id).exists()
        assert _tree(experiment) == before

    assert generation_ops._publication_holds(experiment, winners) == published
    changed = {**winners, "b": replace(winners["b"], metric=0.25)}
    assert generation_ops._publication_holds(experiment, changed) is None
    assert generation_ops._publication_holds(experiment, {"a": winners["a"]}) is None


def test_a_replay_publishes_a_generation_when_nothing_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishing is work: a replay over a failed publication commits a new generation."""
    experiment = _two_phases(tmp_path)
    reports: list[TerminalReport] = []

    def refuse(*_args: object) -> None:
        raise PublicationCommitError("summary could not be read back")

    with monkeypatch.context() as patch:
        patch.setattr(generation_ops, "_validate_generation_publishable", refuse)
        with pytest.raises(PublicationCommitError):
            run_experiment(experiment, terminal_callback=reports.append)
    failed = reports[0].generation_id
    record = yaml.safe_load(_generation_record_path(experiment, failed).read_text())
    assert record["state"] == "publication_failed"
    assert not _last_successful_generation_path(experiment).exists()

    winners = run_experiment(experiment, terminal_callback=reports.append)

    replayed = reports[1].generation_id
    assert _tree(experiment)[0] == sorted([failed, replayed])
    assert published_generation_id(experiment) == replayed
    assert read_status(experiment)["current_generation_id"] == replayed
    assert generation_ops._publication_holds(experiment, winners) == replayed


@pytest.mark.parametrize(
    ("change", "error"),
    [
        pytest.param(
            {"provenance": {"revision": "rev-2"}},
            StudyFingerprintMismatchError,
            id="fingerprint",
        ),
        pytest.param({"n_trials": 1}, TrialTargetRefusedError, id="lowered-target"),
    ],
)
def test_a_refused_run_leaves_no_generation_and_both_pointers_as_they_were(
    tmp_path: Path, change: dict[str, object], error: type[Exception]
) -> None:
    """A refusal before any trial launches leaves the tree exactly as the last real run left it."""
    experiment = _two_phases(tmp_path)
    run_experiment(experiment)
    before = _tree(experiment)
    if "n_trials" in change:
        phases = [phase.model_copy(update=change) for phase in experiment.phases]
        refused = experiment.model_copy(update={"phases": phases})
    else:
        refused = experiment.model_copy(update=change)
    reports: list[TerminalReport] = []

    with pytest.raises(error):
        run_experiment(refused, terminal_callback=reports.append)

    assert reports[0].primary_error is not None
    assert not _generation_dir(experiment, reports[0].generation_id).exists()
    assert _tree(experiment) == before


def test_an_aborted_run_keeps_its_generation_and_the_refused_reruns_add_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Trials launched before an abort keep their generation; later refusals still print."""
    config = write_experiment(
        tmp_path / "exp.yaml",
        trial_command=_ECHO,
        metric={
            "extractor": {"type": "log_regex", "pattern": r"x=(?P<value>[0-9.eE+-]+)"},
            "max": 0.0,
        },
        storage="auto",
        workdir=str(tmp_path / "runs"),
        provenance={"revision": "rev-1"},
        phases=[
            {
                "name": "p",
                "n_trials": 4,
                "max_consecutive_failures": 2,
                "sampler": {"type": "grid"},
                "search_space": {"x": {"type": "categorical", "choices": [2, 4, 6, 8]}},
            }
        ],
    )
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)

    assert invoke_cli_boundary(["run", str(config)], monkeypatch) != 0
    assert "aborted" in capsys.readouterr().err
    before = _tree(experiment)
    [aborted] = before[0]
    record = yaml.safe_load(_generation_record_path(experiment, aborted).read_text())
    assert record["state"] == "failed"
    assert read_status(experiment)["current_generation_id"] == aborted

    for refused_by, argv in (
        ("previously aborted", []),
        ("different phase config", ["--set", "provenance.revision=rev-2"]),
    ):
        assert invoke_cli_boundary(["run", str(config), *argv], monkeypatch) != 0
        assert refused_by in capsys.readouterr().err
        assert _tree(experiment) == before
        assert read_status(experiment)["current_generation_id"] == aborted


def test_an_mcp_replay_succeeds_without_a_generation_and_a_refusal_still_reports(
    tmp_path: Path,
) -> None:
    """A detached replay represents the generation it matched; a refused launch reports its refusal."""
    config = write_experiment(
        tmp_path / "srv.yaml",
        experiment="srv",
        trial_command=_ECHO,
        metric={"extractor": {"type": "log_regex", "pattern": r"x=(?P<value>[0-9.eE+-]+)"}},
        storage="auto",
        workdir=str(tmp_path / "runs"),
        provenance={"revision": "rev-1"},
        phases=[
            {
                "name": "p",
                "n_trials": 2,
                "sampler": {"type": "random", "seed": 0},
                "search_space": {"x": {"type": "int", "low": 0, "high": 10}},
            }
        ],
    )
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    store = RunStore(tmp_path / "state")
    assert run_claimed_runner(store, "srv-first", config) == 0
    before = _tree(experiment)

    assert run_claimed_runner(store, "srv-replay", config) == 0

    assert _tree(experiment) == before
    first = json.loads(store.status_path("srv-first").read_text())
    replay = json.loads(store.status_path("srv-replay").read_text())
    assert replay["returncode"] == 0
    assert replay["failure"] is None
    assert replay["result_snapshot_state"] == "complete"
    assert "result_publication_state" not in replay
    assert replay["result_snapshot"]["winners"] == first["result_snapshot"]["winners"]
    status = replay["result_snapshot"]["status"]
    assert status["represented_generation_id"] == "srv-first"
    assert status["published_generation_id"] == "srv-first"
    assert (status["is_published"], status["summary_present"]) == (True, True)
    assert [phase["winner_present"] for phase in status["phases"]] == [True]
    assert not any(status["phases"][0]["generation_trials"].values())
    payload = status_payload(
        "srv",
        status,
        {"run_id": "srv-replay"},
        result_source="frozen_run_snapshot",
        elapsed_seconds=None,
    )
    assert payload["run_wrote_generation"] is False
    (phase,) = payload["phases"]
    assert (phase["attempts_this_run"], phase["finished_trials_this_run"]) == (0, 0)
    assert phase["finished_trials_before_run"] == 2
    assert phase["target_already_satisfied"] is True
    first_payload = status_payload(
        "srv",
        first["result_snapshot"]["status"],
        {"run_id": "srv-first"},
        result_source="frozen_run_snapshot",
        elapsed_seconds=None,
    )
    assert first_payload["run_wrote_generation"] is True

    lowered = config.with_name("lowered.yaml")
    lowered.write_text(config.read_text().replace("n_trials: 2", "n_trials: 1"))
    with pytest.raises(TrialTargetRefusedError):
        run_claimed_runner(store, "srv-lowered", lowered)

    assert _tree(experiment) == before
    refused = json.loads(store.status_path("srv-lowered").read_text())
    assert refused["failure"]["code"] == "trial_target_refused"
    assert refused["result_snapshot"]["winners"] == []
    # The refused run's ID names no generation, so it represents none.
    refused_payload = status_payload(
        "srv",
        refused["result_snapshot"]["status"],
        {"run_id": "srv-lowered"},
        result_source="frozen_run_snapshot",
        elapsed_seconds=None,
    )
    assert refused_payload["represented_generation_id"] is None
    assert refused_payload["published_generation_id"] == "srv-first"
    assert (
        refused_payload["is_published"],
        refused_payload["summary_present"],
        refused_payload["run_wrote_generation"],
    ) == (False, False, False)
    (refused_phase,) = refused_payload["phases"]
    assert refused_phase["attempts_this_run"] == 0
    assert refused_phase["finished_trials_before_run"] == 2
