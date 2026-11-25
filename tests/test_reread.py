"""A changed reading setting re-reads the kept evidence of finished trials and retrains nothing."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Constraint, Experiment, Metric, Observation, Phase
from phasesweep.config.search import CategoricalParam, FloatParam, IntParam, Repeats, Sampler
from phasesweep.engine import reread as reread_ops
from phasesweep.engine import trial as trial_ops
from phasesweep.engine.errors import (
    LaunchEnvironmentError,
    PhaseTimeoutError,
    PublishedStudyMissingError,
    StudyFingerprintMismatchError,
    TrialTargetRefusedError,
    WinnerIntegrityError,
)
from phasesweep.engine.ledger import _resolve_storage, validate_ledger
from phasesweep.engine.optuna import _PositionCmaEsSampler
from phasesweep.engine.paths import _generation_config_snapshot_path
from phasesweep.engine.preview import validate_trial_targets
from phasesweep.engine.read import inspect_status, read_status, read_trial_tables
from phasesweep.engine.selection import NoFeasibleTrialError, PhaseAbortRefusedError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FAILURE_REASON_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    PHASE_DECISION_ATTR,
    PHASE_IDENTITY_ATTR,
    PHASE_RECOVERY_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
    _trial_attempt_dir,
)
from phasesweep.engine.study_policy import _record_reruns
from phasesweep.engine.trial_dirs import producing_attempts
from phasesweep.evidence.baseline import EVIDENCE_BASELINE_FILE
from phasesweep.evidence.evaluation import TrialContext
from phasesweep.evidence.models import (
    JsonEnvelopeExtractor,
    JsonExtractor,
    LogRegexExtractor,
    RequiredFileGate,
    WandbExtractor,
)
from phasesweep.evidence.wandb import WandbRunMissingError
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import (
    addressed_phase_study,
    invoke_cli_boundary,
    make_experiment,
    record_accepted_partial_timeout,
    study_name,
    write_experiment_config,
    write_trainer,
)
from tests.ledger_fixtures import tree_changes, tree_snapshot


def _counting_trainer(tmp_path: Path) -> tuple[Path, Path]:
    """Drop a trainer that appends one line per launch and writes loss, acc and mem for ``--x``."""
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--x", type=float, default=1.0)
        ap.add_argument("--y", type=float, default=0.0)
        args, _ = ap.parse_known_args()
        with open({str(launches)!r}, "a") as handle:
            handle.write("launch\\n")
        loss = (args.x - 3) ** 2 + args.y
        Path(args.out).write_text(json.dumps({{"loss": loss, "acc": args.x, "mem": 10 * args.x}}))
        print(f"loss={{loss}}")
        """,
    )
    return trainer, launches


def _launches(path: Path) -> int:
    """Count the trainer's launches so far."""
    return len(path.read_text().splitlines()) if path.exists() else 0


def _metric(key: str, goal: str = "minimize") -> Metric:
    """Read ``key`` from the trainer's ``r.json``."""
    return Metric(
        name="loss", goal=goal, extractor=JsonExtractor(type="json", path="r.json", key=key)
    )


def _phases(*, child: bool = False) -> list[Phase]:
    """A three-cell grid over x, and optionally a child that inherits it and searches y."""
    phases = [
        Phase(
            name="p",
            n_trials=3,
            gpu_policy="none",
            max_consecutive_failures=5,
            sampler=Sampler(type="grid"),
            search_space={"x": CategoricalParam(type="categorical", choices=[1, 2, 4])},
        )
    ]
    if child:
        phases.append(
            Phase(
                name="c",
                inherits=["p"],
                n_trials=2,
                gpu_policy="none",
                max_consecutive_failures=5,
                sampler=Sampler(type="grid"),
                search_space={"y": CategoricalParam(type="categorical", choices=[0, 1])},
            )
        )
    return phases


def _experiment(tmp_path: Path, trainer: Path, metric: Metric, **extra: object) -> Experiment:
    """The experiment under ``tmp_path`` with the given reading settings."""
    phases = extra.pop("phases", None) or _phases()
    return make_experiment(
        persistent=tmp_path, trainer=trainer, metric=metric, phases=phases, **extra
    )


def test_a_misspelled_key_is_corrected_by_re_reading_every_finished_trial(tmp_path: Path) -> None:
    """Every trial failed at extraction under a typo; the corrected key completes them all unlaunched."""
    trainer, launches = _counting_trainer(tmp_path)
    typo = _experiment(tmp_path, trainer, _metric("los"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(typo)
    assert _launches(launches) == 3

    fixed = _experiment(tmp_path, trainer, _metric("loss"))
    winners = run_experiment(fixed)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 4}
    trials = addressed_phase_study(fixed).get_trials()
    assert [trial.state for trial in trials] == [optuna.trial.TrialState.COMPLETE] * 3
    assert sorted(trial.value for trial in trials) == [1.0, 1.0, 4.0]
    assert [t.number for t in trials] == [
        t.number for t in addressed_phase_study(typo).get_trials()
    ]


def test_reread_uses_recorded_trainer_exit_after_evidence_worker_timeout(tmp_path: Path) -> None:
    """The process slot's last worker cannot override the committed trainer exit."""
    trainer, launches = _counting_trainer(tmp_path)
    phase = Phase(name="p", n_trials=1, gpu_policy="none")
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase])
    assert run_experiment(original)["p"].metric == 4
    [measured] = addressed_phase_study(original).trials
    attempt_dir = _trial_attempt_dir(measured)
    assert attempt_dir is not None
    write_attempt_lifecycle(
        attempt_dir,
        attempt_id=measured.user_attrs[ATTEMPT_ID_ATTR],
        state="exited",
        ended_by="timed_out",
        cleanup_confirmed=True,
    )

    corrected = _experiment(tmp_path, trainer, _metric("acc"), phases=[phase])
    assert run_experiment(corrected)["p"].metric == 1
    assert _launches(launches) == 1


def test_reread_keeps_a_repeat_measurement_before_an_unsuccessful_retry(tmp_path: Path) -> None:
    """Repeat selection and status keep a measured source when its retry did not measure."""
    counter = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        from pathlib import Path
        launches = Path({str(counter)!r})
        prior = launches.read_text().splitlines() if launches.exists() else []
        with launches.open("a") as log:
            log.write("launch\\n")
        if len(prior) < 2:
            print(f"loss={{len(prior) + 1.0}}")
        """,
    )
    phase = Phase(
        name="p",
        n_trials=2,
        gpu_policy="none",
        max_consecutive_failures=5,
        sampler=Sampler(type="grid"),
        repeats=Repeats(key="seed", values=[1, 2]),
    )
    retry = phase.model_copy(update={"n_trials": 3})
    for planned in (phase, retry):
        with pytest.raises(NoFeasibleTrialError, match="No complete repeat group"):
            run_experiment(_experiment(tmp_path, trainer, _metric("typo"), phases=[planned]))
    fixed = _experiment(
        tmp_path,
        trainer,
        Metric(extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.]+)")),
        phases=[retry],
    )

    winner = run_experiment(fixed)["p"]

    assert winner.metric == 1.5
    assert _launches(counter) == 3
    assert [trial.state.name for trial in addressed_phase_study(fixed).trials] == [
        "COMPLETE",
        "COMPLETE",
        "FAIL",
    ]
    [group] = read_trial_tables(fixed).phases[0].groups
    assert group.n == 2 and group.aggregate == 1.5 and group.winner


def test_reread_keeps_its_own_attempt_identity_when_its_source_is_removed(tmp_path: Path) -> None:
    """A new draw in an enlarged reading study still belongs to its launch generation."""
    trainer, launches = _counting_trainer(tmp_path)
    one_cell = Phase(
        name="p",
        n_trials=1,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1])},
    )
    source = _experiment(tmp_path, trainer, _metric("loss"), phases=[one_cell])
    run_experiment(source)
    original_generation = read_status(source)["current_generation_id"]
    assert isinstance(original_generation, str)
    enlarged = one_cell.model_copy(
        update={
            "n_trials": 2,
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2])},
        }
    )
    corrected = _experiment(tmp_path, trainer, _metric("acc"), phases=[enlarged])
    run_experiment(corrected)
    assert _launches(launches) == 2
    optuna.delete_study(
        study_name=study_name(source, "p"),
        storage=_resolve_storage(source.resolved_storage),
    )

    current = read_status(corrected)["phases"][0]
    original = read_status(corrected, generation_id=original_generation)["phases"][0]
    assert current["trials"] == {"COMPLETE": 2}
    assert current["generation_trials"] == {"COMPLETE": 1}
    assert original["generation_trials"] == {}
    original_reading = addressed_phase_study(corrected).trials
    reversed_goal = _experiment(tmp_path, trainer, _metric("acc", "maximize"), phases=[enlarged])
    run_experiment(reversed_goal)
    new_reading = addressed_phase_study(reversed_goal).trials
    assert [(row.user_attrs[ATTEMPT_ID_ATTR], row.value) for row in new_reading] == [
        (row.user_attrs[ATTEMPT_ID_ATTR], row.value) for row in original_reading
    ]
    assert _launches(launches) == 2


def test_reread_releases_retry_budget_for_cells_that_now_have_a_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An old retry budget can exceed what the reread grid is able to measure."""
    trainer, launches = _counting_trainer(tmp_path)
    typo = _experiment(tmp_path, trainer, _metric("los"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(typo)
    phase = _phases()[0]
    retry = phase.model_copy(update={"n_trials": 6})
    with pytest.raises(NoFeasibleTrialError, match="5 consecutive failed/infeasible trials"):
        run_experiment(_experiment(tmp_path, trainer, _metric("los"), phases=[retry]))
    source = addressed_phase_study(typo).get_trials()
    assert source[-1].state == optuna.trial.TrialState.WAITING
    assert _launches(launches) == 5

    fixed = _experiment(tmp_path, trainer, _metric("loss"), phases=[retry])
    phase = phase.model_copy(update={"n_trials": 5})
    corrected = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase])
    caplog.set_level(logging.INFO)
    before = tree_snapshot(tmp_path)
    run_experiment(fixed, dry_run=True)
    assert "Its launch count depends on how much retry capacity remains" in caplog.text
    assert "DRY RUN phase=p would launch" not in caplog.text
    assert "Launch counts are unresolved for 1 phase" in caplog.text
    caplog.clear()
    run_experiment(corrected, dry_run=True)
    assert "would select from the re-read evidence and launch no trials" in caplog.text
    assert "DRY RUN total: would launch 0 trials across 0 phases" in caplog.text
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    assert _launches(launches) == 5

    with pytest.raises(TrialTargetRefusedError, match="allow at most 5") as refused:
        run_experiment(fixed)
    assert "reduce the accepted trial target from 6 to 5" in caplog.text
    # 5 is also the accepted floor, so the step names the one target that runs.
    assert refused.value.remediation == "Set phases[p].n_trials to 5."

    assert run_experiment(corrected)["p"].metric == 1.0
    assert _launches(launches) == 5
    copied = addressed_phase_study(corrected).get_trials()
    assert [(t.number, t.user_attrs[ATTEMPT_ID_ATTR]) for t in copied] == [
        (t.number, t.user_attrs[ATTEMPT_ID_ATTR]) for t in source[:-1]
    ]
    assert addressed_phase_study(typo).get_trials()[-1].state == optuna.trial.TrialState.WAITING


def test_reread_at_retry_ceiling_can_acknowledge_an_inherited_failure_streak(
    tmp_path: Path,
) -> None:
    """A lowered target can replay after the new failure limit acknowledges an earlier abort."""
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, sys
        from pathlib import Path
        launches = Path({str(launches)!r})
        with launches.open("a") as handle:
            handle.write("launch\\n")
        if len(launches.read_text().splitlines()) > 3:
            sys.exit(1)
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.write_text(json.dumps({{"loss": 1}}))
        print("loss=1")
        """,
    )
    typo = _experiment(tmp_path, trainer, _metric("los"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(typo)
    phase = _phases()[0]
    retry = phase.model_copy(update={"n_trials": 6, "max_consecutive_failures": 2})
    with pytest.raises(NoFeasibleTrialError, match="2 consecutive failed/infeasible trials"):
        run_experiment(_experiment(tmp_path, trainer, _metric("los"), phases=[retry]))
    source = addressed_phase_study(typo)
    assert source.user_attrs[PHASE_RECOVERY_ATTR]["trial_target"] == 6
    metric = Metric(
        name="loss",
        extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9]+)"),
    )
    corrected = retry.model_copy(update={"n_trials": 5})
    fixed = _experiment(tmp_path, trainer, metric, phases=[corrected])
    with pytest.raises(PhaseAbortRefusedError, match="accepted n_trials=6") as refusal:
        run_experiment(fixed)
    assert "raise phases[p].max_consecutive_failures above 2" in refusal.value.remediation
    assert addressed_phase_study(fixed).user_attrs[PHASE_RECOVERY_ATTR]["trial_target"] == 5

    acknowledged = corrected.model_copy(update={"max_consecutive_failures": 10})
    result = run_experiment(_experiment(tmp_path, trainer, metric, phases=[acknowledged]))
    assert result["p"].metric == 1
    assert _launches(launches) == 5


def test_a_flipped_goal_re_reads_into_a_study_of_the_other_direction(tmp_path: Path) -> None:
    """Maximizing the same evidence flips the winner; the trials keep their numbers and directories."""
    trainer, launches = _counting_trainer(tmp_path)
    minimize = _experiment(tmp_path, trainer, _metric("loss"))
    assert run_experiment(minimize)["p"].effective_overrides == {"x": 4}
    before = {
        t.number: t.user_attrs[TRIAL_DIR_ATTR] for t in addressed_phase_study(minimize).get_trials()
    }

    maximize = _experiment(tmp_path, trainer, _metric("loss", goal="maximize"))
    winners = run_experiment(maximize)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 1}
    study = addressed_phase_study(maximize)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert {t.number: t.user_attrs[TRIAL_DIR_ATTR] for t in study.get_trials()} == before


@pytest.mark.integration
def test_cmaes_topup_relearns_from_the_reread_values(tmp_path: Path) -> None:
    """A direction change replays all retained CMA-ES batches before the next draw."""
    trainer, launches = _counting_trainer(tmp_path)
    phase = Phase(
        name="p",
        n_trials=19,
        gpu_policy="none",
        sampler=Sampler(type="cmaes", seed=5),
        search_space={key: FloatParam(type="float", low=-1, high=1) for key in ("x", "y")},
    )
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase])
    run_experiment(original)
    before = addressed_phase_study(original).trials
    sampler = _PositionCmaEsSampler(seed=5)
    assert sampler._restore_optimizer(before) is not None

    top_up = phase.model_copy(update={"n_trials": 20})
    changed = _experiment(tmp_path, trainer, _metric("loss", goal="maximize"), phases=[top_up])
    assert run_experiment(changed)["p"].metric >= max(
        t.value for t in before if t.value is not None
    )
    after = addressed_phase_study(changed).trials
    retained = after[:-1]
    assert _launches(launches) == len(after) == 20
    assert [trial.params for trial in retained] == [trial.params for trial in before]
    assert [trial.system_attrs.get(sampler._attr_key_generation) for trial in retained] == [
        trial.system_attrs.get(sampler._attr_key_generation) for trial in before
    ]
    assert sampler._restore_optimizer(retained) is None
    assert after[-1].system_attrs[sampler._attr_key_generation] == 3
    optimizer = sampler._restore_optimizer(after)
    assert optimizer is not None
    assert optimizer.generation == 3


def test_managed_envelope_path_change_reuses_the_producing_envelope(tmp_path: Path) -> None:
    """Existing attempts keep their output path and new attempts use the changed path."""
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os
        from pathlib import Path
        from phasesweep import report_objective
        path = Path(os.environ["PHASESWEEP_OBJECTIVE_PATH"])
        with Path({str(launches)!r}).open("a") as log:
            log.write(path.name + "\\n")
        report_objective(name="loss", split="test", value=0.8, policy="final", checkpoint="last", step=10)
        constraint = json.loads(path.read_text())
        constraint["objective"]["value"] = 1.5
        path.with_name("constraint.json").write_text(json.dumps(constraint))
        """,
    )
    phase = Phase(
        name="p",
        n_trials=1,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        allow_partial_grid=True,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )

    def config(path: str, trials: int) -> Experiment:
        metric = Metric(
            extractor=JsonEnvelopeExtractor(
                type="json_envelope", path=path, objective_name="loss", split="test", policy="final"
            )
        )
        constraint = Constraint(
            name="resource",
            max=2,
            extractor=metric.extractor.model_copy(update={"path": "constraint.json"}),
        )
        return _experiment(
            tmp_path,
            trainer,
            metric,
            phases=[phase.model_copy(update={"n_trials": trials})],
            constraints=[constraint],
        )

    original = config("original.json", 1)
    assert run_experiment(original)["p"].metric == 0.8
    original_id = addressed_phase_study(original).trials[0].user_attrs[ATTEMPT_ID_ATTR]
    changed = config("next.json", 2)
    winner = run_experiment(changed)["p"]
    assert winner.metric == 0.8 and winner.constraints == {"resource": 1.5}
    old, new = addressed_phase_study(changed).trials
    assert old.value == new.value == 0.8
    assert old.user_attrs[ATTEMPT_ID_ATTR] == original_id
    assert all(
        json.loads(record.user_attrs[OBJECTIVE_PROVENANCE_ATTR])["source"]["path"] == path
        for record, path in [(old, "original.json"), (new, "next.json")]
    )
    assert run_experiment(original)["p"].attempt_id == original_id
    assert launches.read_text().splitlines() == ["original.json", "next.json"]


def test_wandb_project_change_keeps_prior_trials_and_routes_new_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W&B lookup and offline winner verification use the launch target of each result."""
    launches = tmp_path / "launches.jsonl"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os
        from pathlib import Path
        with Path({str(launches)!r}).open("a") as log:
            log.write(json.dumps([os.environ["WANDB_PROJECT"], os.environ["WANDB_RUN_ID"]]) + "\\n")
        """,
    )
    phase = Phase(
        name="p",
        n_trials=1,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        allow_partial_grid=True,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )

    def config(project: str, trials: int) -> Experiment:
        metric = Metric(
            extractor=WandbExtractor(
                type="wandb",
                entity="owner",
                project=project,
                metric_key="score",
            )
        )
        return _experiment(
            tmp_path,
            trainer,
            metric,
            phases=[phase.model_copy(update={"n_trials": trials})],
            env={"WANDB_PROJECT": project},
        )

    def launched_runs() -> list[list[str]]:
        return [json.loads(line) for line in launches.read_text().splitlines()]

    summary_paths: list[list[str]] = []
    position_paths: list[list[str]] = []

    def summary(*, project: str, run_id: str, **_: object) -> dict[str, object]:
        path = [project, run_id]
        summary_paths.append(path)
        if path not in launched_runs():
            raise WandbRunMissingError(run_id)
        return {
            "values": {"score": 0.25 if project == "original" else 0.5},
            "present_keys": [],
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        }

    def position(*, project: str, run_id: str, **_: object) -> int:
        position_paths.append([project, run_id])
        return 10

    monkeypatch.setattr("phasesweep.evidence.wandb.require_wandb_sdk", lambda: None)
    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", summary)
    monkeypatch.setattr("phasesweep.evidence.evaluation.locate_wandb_position", position)
    original = config("original", 1)
    original_id = run_experiment(original)["p"].attempt_id
    changed = config("new-project", 2)
    winner = run_experiment(changed)["p"]
    old, new = addressed_phase_study(changed).trials
    new_id = new.user_attrs[ATTEMPT_ID_ATTR]
    assert [old.value, new.value] == [0.25, 0.5]
    assert winner.attempt_id == original_id and winner.metric == 0.25
    assert launched_runs() == [["original", original_id], ["new-project", new_id]]
    assert (
        summary_paths
        == position_paths
        == [
            ["original", original_id],
            ["original", original_id],
            ["new-project", new_id],
        ]
    )
    assert run_experiment(original)["p"].attempt_id == original_id
    assert len(launched_runs()) == 2


def test_reread_never_queries_an_unbound_wandb_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local-only launch supplies no managed run when a later extractor asks for W&B."""
    trainer, launches = _counting_trainer(tmp_path)
    phase = Phase(name="p", n_trials=1, gpu_policy="none", sampler=Sampler(type="grid"))
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase])
    assert run_experiment(original)["p"].metric == 4

    def reject_lookup(**_: object) -> dict[str, object]:
        raise AssertionError("a local-only trial cannot own a managed W&B run")

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", reject_lookup)
    metric = Metric(
        extractor=WandbExtractor(type="wandb", entity="owner", project="p", metric_key="score")
    )
    changed = _experiment(tmp_path, trainer, metric, phases=[phase])
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(changed)
    (copied,) = addressed_phase_study(changed).trials
    assert "no managed W&B destination at launch" in copied.user_attrs[FAILURE_REASON_ATTR]
    assert _launches(launches) == 1


def test_wandb_retry_keeps_the_checkpoint_runs_bound_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attempt reusing a checkpoint continues the original W&B run after a project change."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import json, os
        from pathlib import Path
        path = Path(os.environ["PHASESWEEP_TRIAL_DIR"]) / "runs.jsonl"
        first = not path.exists()
        with path.open("a") as log:
            log.write(json.dumps([os.environ["WANDB_PROJECT"], os.environ["WANDB_RUN_ID"],
                                  os.environ["WANDB_RESUME"]]) + "\\n")
        raise SystemExit(1 if first else 0)
        """,
    )
    phase = Phase(name="p", n_trials=1, gpu_policy="none", sampler=Sampler(type="grid"))

    def config(project: str, trials: int) -> Experiment:
        metric = Metric(
            extractor=WandbExtractor(
                type="wandb",
                entity="owner",
                project=project,
                metric_key="score",
            )
        )
        return _experiment(
            tmp_path, trainer, metric, phases=[phase.model_copy(update={"n_trials": trials})]
        )

    paths: list[tuple[str, str]] = []

    def summary(*, project: str, run_id: str, **_: object) -> dict[str, object]:
        paths.append((project, run_id))
        return {
            "values": {"score": 0.25},
            "present_keys": [],
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr("phasesweep.evidence.wandb.require_wandb_sdk", lambda: None)
    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", summary)
    monkeypatch.setattr("phasesweep.evidence.evaluation.locate_wandb_position", lambda **_: None)
    original = config("original", 1)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    changed = config("new-project", 2)
    assert run_experiment(changed)["p"].metric == 0.25
    opener, retry = addressed_phase_study(changed).trials
    opener_id = opener.user_attrs[ATTEMPT_ID_ATTR]
    assert opener.user_attrs[TRIAL_DIR_ATTR] == retry.user_attrs[TRIAL_DIR_ATTR]
    launches = Path(retry.user_attrs[TRIAL_DIR_ATTR], "runs.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in launches] == [
        ["original", opener_id, "never"],
        ["original", opener_id, "allow"],
    ]
    assert paths == [("original", opener_id)]


def test_published_reread_history_belongs_to_the_selecting_study(tmp_path: Path) -> None:
    """An older sibling can neither block replay nor replace the missing publisher."""
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse
        import json
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path)
        args, _ = parser.parse_known_args()
        launches = Path({str(launches)!r})
        with launches.open("a") as handle:
            print("launch", file=handle)
        count = len(launches.read_text().splitlines())
        args.out.write_text(json.dumps({{"loss": count, "acc": count, "mem": count}}))
        """,
    )
    phase = Phase(
        name="p",
        n_trials=1,
        gpu_policy="none",
        allow_partial_grid=True,
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2])},
    )
    initial = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase])
    topped_up = phase.model_copy(update={"n_trials": 2})
    reread = _experiment(tmp_path, trainer, _metric("acc"), phases=[topped_up])
    latest = _experiment(tmp_path, trainer, _metric("mem"), phases=[topped_up])

    assert run_experiment(initial)["p"].trial_number == 0
    assert run_experiment(reread)["p"].trial_number == 0
    assert read_status(initial)["phases"][0]["published_study_unavailable"] is False
    run_experiment(reread, dry_run=True)
    assert run_experiment(reread)["p"].trial_number == 0
    assert run_experiment(latest)["p"].trial_number == 0
    assert _launches(launches) == 2

    optuna.delete_study(
        study_name=study_name(latest, "p"),
        storage=_resolve_storage(latest.resolved_storage),
    )
    assert read_status(reread)["phases"][0]["published_study_unavailable"] is True
    with pytest.raises(PublishedStudyMissingError, match="its persistent study is missing"):
        run_experiment(reread)


@pytest.mark.parametrize("fixed", [False, True], ids=["grid", "fixed"])
@pytest.mark.parametrize("private", [False, True], ids=["shared-json", "private-stdout"])
def test_reread_attributes_results_to_their_attempt_and_keeps_the_retry_budget(
    tmp_path: Path, fixed: bool, private: bool
) -> None:
    """A directory retry preserves its own result and rereading never schedules a new one.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    :param bool fixed: Retry a fixed configuration instead of a grid cell.
    :param bool private: Read the attempt's private captured stream instead of shared JSON.
    """
    trainer, launches = _counting_trainer(tmp_path)
    phase = Phase(name="p", n_trials=1, gpu_policy="none") if fixed else _phases()[0]
    target = phase.n_trials + 1
    retried = phase.model_copy(update={"n_trials": target})
    for planned in (phase, retried):
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(_experiment(tmp_path, trainer, _metric("typo"), phases=[planned]))

    metric = (
        Metric(extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.]+)"))
        if private
        else _metric("loss")
    )
    corrected = _experiment(tmp_path, trainer, metric, phases=[retried])
    run_experiment(corrected)
    trials = addressed_phase_study(corrected).get_trials()
    first, last = trials[0], trials[-1]
    assert last.user_attrs[RERUN_OF_ATTR] == first.number
    assert last.user_attrs[TRIAL_DIR_ATTR] == first.user_attrs[TRIAL_DIR_ATTR]
    assert last.state == optuna.trial.TrialState.COMPLETE
    if private:
        assert first.state == optuna.trial.TrialState.COMPLETE
    else:
        assert first.state == optuna.trial.TrialState.FAIL
        assert "changed after this attempt finished" in first.user_attrs[FAILURE_REASON_ATTR]

    validate_trial_targets(corrected)
    run_experiment(corrected, dry_run=True)
    run_experiment(corrected)
    too_many = retried.model_copy(update={"n_trials": target + 1})
    with pytest.raises(TrialTargetRefusedError):
        validate_trial_targets(_experiment(tmp_path, trainer, metric, phases=[too_many]))
    assert _launches(launches) == target


def test_reread_does_not_score_a_file_overwritten_from_another_reading_study(
    tmp_path: Path,
) -> None:
    """A later retry's shared file is unavailable to a reread from a sibling study.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    """
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        args = argparse.ArgumentParser()
        args.add_argument("--out", type=Path, required=True)
        out = args.parse_known_args()[0].out
        launches = Path({str(launches)!r})
        with launches.open("a") as handle:
            handle.write("launch\\n")
        count = len(launches.read_text().splitlines())
        out.write_text(json.dumps({{"loss": count}}))
        if count == 4:
            raise SystemExit(3)
        """,
    )
    phase = _phases()[0]
    older = _experiment(tmp_path, trainer, _metric("typo"))
    sibling = _experiment(tmp_path, trainer, _metric("another-typo"))
    retried = _experiment(
        tmp_path, trainer, _metric("typo"), phases=[phase.model_copy(update={"n_trials": 4})]
    )
    for planned in (older, sibling, retried):
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(planned)

    corrected = _experiment(tmp_path, trainer, _metric("loss"))
    assert run_experiment(corrected)["p"].metric == 2
    trials = addressed_phase_study(corrected).get_trials()
    assert len(trials) == len(addressed_phase_study(sibling).get_trials()) == 3
    assert trials[0].state == optuna.trial.TrialState.FAIL
    assert "changed after this attempt finished" in trials[0].user_attrs[FAILURE_REASON_ATTR]
    assert _launches(launches) == 4


@pytest.mark.parametrize("interrupt_fill", [False, True])
def test_retry_keeps_the_measured_evidence_of_a_sibling_reading_study(
    tmp_path: Path, interrupt_fill: bool
) -> None:
    """A retry can overwrite shared score files without erasing earlier measurements."""
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        args = argparse.ArgumentParser()
        args.add_argument("--out", type=Path, required=True)
        out = args.parse_args().out
        launches = Path({str(launches)!r})
        with launches.open("a") as handle:
            handle.write("launch\\n")
        count = len(launches.read_text().splitlines())
        out.write_text(json.dumps({{"loss": count, "alternate": count}}))
        out.with_name("resource.json").write_text(json.dumps({{"mem": 10 + count, "peak": 20 + count}}))
        out.with_name("unmeasured.json").write_text(json.dumps({{"peaky": 30 + count}}))
        """,
    )
    phase = Phase(name="p", n_trials=1, gpu_policy="none", sampler=Sampler(type="grid"))
    retry = phase.model_copy(update={"n_trials": 2})

    def config(
        key: str,
        phase: Phase,
        *,
        constraint_key: str = "mem",
        constraint_path: str = "resource.json",
    ) -> Experiment:
        constraint = Constraint(
            name="memory",
            min=0,
            extractor=JsonExtractor(type="json", path=constraint_path, key=constraint_key),
        )
        return _experiment(
            tmp_path, trainer, _metric(key), phases=[phase], constraints=[constraint]
        )

    original = config("typo", phase)
    corrected = config("loss", phase)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    if interrupt_fill:
        with (
            patch.object(reread_ops, "_stamp_identity", side_effect=RuntimeError("stop fill")),
            pytest.raises(RuntimeError, match="stop fill"),
        ):
            run_experiment(corrected)
    else:
        assert run_experiment(corrected)["p"].metric == 1

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(config("typo", retry))
    first, last = addressed_phase_study(original).get_trials()
    assert first.user_attrs[TRIAL_DIR_ATTR] == last.user_attrs[TRIAL_DIR_ATTR]
    assert _launches(launches) == 2
    assert (
        Path(first.user_attrs[TRIAL_DIR_ATTR], "r.json").read_text()
        == '{"loss": 2, "alternate": 2}'
    )

    # An unfinished fill can import the new attempt. A published study replays its own target.
    corrected_phase = retry if interrupt_fill else phase
    winner = run_experiment(config("loss", corrected_phase))["p"]
    assert winner.metric == 1
    assert winner.constraints == {"memory": 11}
    unmeasured = config(
        "alternate", retry, constraint_key="peaky", constraint_path="unmeasured.json"
    )
    assert run_experiment(unmeasured)["p"].metric == 2
    assert [trial.value for trial in addressed_phase_study(unmeasured).get_trials()] == [None, 2]

    alternate = config("alternate", retry, constraint_key="peak")
    winner = run_experiment(alternate)["p"]
    assert winner.metric == 1
    assert winner.constraints == {"memory": 21}
    assert [trial.value for trial in addressed_phase_study(alternate).get_trials()] == [1, 2]

    assert _launches(launches) == 2


def test_reread_unrecorded_path_refuses_a_predecessors_file(tmp_path: Path) -> None:
    """A path that was not checked before a retry needs launch-time evidence of a write."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import sys
        from pathlib import Path
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        alternate = out.with_name("alternate.json")
        if not alternate.exists():
            alternate.write_text('{"loss": 0.01}')
            raise SystemExit(1)
        out.write_text('{"loss": 0.9}')
        """,
    )
    phase = Phase(name="p", n_trials=1, gpu_policy="none", sampler=Sampler(type="grid"))
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase])
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)

    retry = phase.model_copy(update={"n_trials": 2})
    retried = _experiment(tmp_path, trainer, _metric("loss"), phases=[retry])
    assert run_experiment(retried)["p"].metric == 0.9
    first, second = addressed_phase_study(retried).trials
    assert first.user_attrs[TRIAL_DIR_ATTR] == second.user_attrs[TRIAL_DIR_ATTR]

    corrected_metric = Metric(
        extractor=JsonExtractor(type="json", path="alternate.json", key="loss")
    )
    corrected = _experiment(tmp_path, trainer, corrected_metric, phases=[retry])
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(corrected)
    copied = addressed_phase_study(corrected).trials[1]
    assert copied.value is None
    assert "trainer exited without writing it" in copied.user_attrs[FAILURE_REASON_ATTR]


def test_reread_refuses_an_older_file_reached_through_a_retargeted_link(tmp_path: Path) -> None:
    """A redirected result path gives no score even when its new target predates the retry.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os
        from pathlib import Path
        root = Path(os.environ["PHASESWEEP_TRIAL_DIR"])
        latest = root / "latest"
        if latest.is_symlink():
            latest.unlink()
            latest.symlink_to("next")
        else:
            for checkpoint, value in (("prior", 0.1), ("next", 0.2)):
                directory = root / checkpoint
                directory.mkdir()
                (directory / "loss.json").write_text(f'{{"loss": {value}}}')
            latest.symlink_to("prior")
            (root / "r.json").symlink_to("latest/loss.json")
        """,
    )
    first = Phase(name="p", n_trials=1, gpu_policy="none")
    retried = first.model_copy(update={"n_trials": 2})
    for phase in (first, retried):
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(_experiment(tmp_path, trainer, _metric("typo"), phases=[phase]))

    corrected = _experiment(tmp_path, trainer, _metric("loss"), phases=[retried])
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(corrected)
    older, retry = addressed_phase_study(corrected).get_trials()
    assert "changed after this attempt finished" in older.user_attrs[FAILURE_REASON_ATTR]
    assert "last changed before this attempt started" in retry.user_attrs[FAILURE_REASON_ATTR]


@pytest.mark.integration
@pytest.mark.skipif(not hasattr(time, "tzset"), reason="the host cannot change process time zones")
def test_reread_uses_the_next_launch_instead_of_local_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The middle of three attempts owns the shared file it rewrote, across time zones.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    :param pytest.MonkeyPatch monkeypatch: Controller's local time zone for each run.
    """
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        args = argparse.ArgumentParser()
        args.add_argument("--out", type=Path, required=True)
        out = args.parse_known_args()[0].out
        launches = Path({str(launches)!r})
        with launches.open("a") as handle:
            handle.write("launch\\n")
        count = len(launches.read_text().splitlines())
        if count < 3:
            out.write_text(json.dumps({{"loss": count}}))
        """,
    )
    phase = Phase(name="p", n_trials=1, gpu_policy="none")
    env = {"TZ": "UTC0"}
    try:
        with monkeypatch.context() as controller:
            for target, zone in enumerate(("UTC-10", "UTC0", "UTC+10"), 1):
                controller.setenv("TZ", zone)
                time.tzset()
                phase = phase.model_copy(update={"n_trials": target})
                with pytest.raises(NoFeasibleTrialError):
                    run_experiment(
                        _experiment(tmp_path, trainer, _metric("typo"), phases=[phase], env=env)
                    )

            corrected = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase], env=env)
            assert run_experiment(corrected)["p"].metric == 2
    finally:
        time.tzset()

    first, middle, last = addressed_phase_study(corrected).get_trials()
    assert len({trial.user_attrs[TRIAL_DIR_ATTR] for trial in (first, middle, last)}) == 1
    assert [trial.value for trial in (first, middle, last)] == [None, 2, None]
    assert "changed after this attempt finished" in first.user_attrs[FAILURE_REASON_ATTR]
    assert "no fresh write since launch" in last.user_attrs[FAILURE_REASON_ATTR]
    assert _launches(launches) == 3


@pytest.mark.integration
def test_reread_refuses_undated_reuse_only_for_shared_evidence(tmp_path: Path) -> None:
    """A retry's missing baseline prevents dating its shared JSON but leaves private logs usable.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    """
    trainer, launches = _counting_trainer(tmp_path)
    phase = Phase(name="p", n_trials=1, gpu_policy="none")
    retried = phase.model_copy(update={"n_trials": 2})
    original = _experiment(tmp_path, trainer, _metric("typo"), phases=[retried])
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_experiment(tmp_path, trainer, _metric("typo"), phases=[phase]))
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    _, retry = addressed_phase_study(original).get_trials()
    records = Path(retry.user_attrs[TRIAL_DIR_ATTR]).parent / "attempts"
    (records / retry.user_attrs[ATTEMPT_ID_ATTR] / EVIDENCE_BASELINE_FILE).unlink()

    shared = _experiment(tmp_path, trainer, _metric("loss"), phases=[retried])
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(shared)
    opener, _ = addressed_phase_study(shared).get_trials()
    reason = opener.user_attrs[FAILURE_REASON_ATTR]
    assert "cannot be attributed to this attempt" in reason
    assert f"launch time of attempt {retry.user_attrs[ATTEMPT_ID_ATTR]!r}" in reason

    log = Metric(extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.]+)"))
    private = _experiment(tmp_path, trainer, log, phases=[retried])
    assert run_experiment(private)["p"].metric == 4
    assert [trial.value for trial in addressed_phase_study(private).get_trials()] == [4, None]
    assert _launches(launches) == 2


def test_an_added_constraint_is_measured_for_every_finished_trial(tmp_path: Path) -> None:
    """The new constraint reads each kept result, so the trial it rules out loses the phase."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss")))
    mem = Constraint(
        name="mem", extractor=JsonExtractor(type="json", path="r.json", key="mem"), max=25
    )

    constrained = _experiment(tmp_path, trainer, _metric("loss"), constraints=[mem])
    winners = run_experiment(constrained)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 2}
    measured = {
        trial.params["x"]: trial.user_attrs["constraint:mem"]
        for trial in addressed_phase_study(constrained).get_trials()
    }
    assert measured == {1: 10.0, 2: 20.0, 4: 40.0}


def test_evidence_deleted_since_fails_only_its_trial_with_the_readers_reason(
    tmp_path: Path,
) -> None:
    """A trial whose result file is gone fails with the extractor's reason; the others re-read."""
    trainer, launches = _counting_trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _metric("loss"))
    run_experiment(original)
    lost = next(t for t in addressed_phase_study(original).get_trials() if t.params["x"] == 2)
    (Path(lost.user_attrs[TRIAL_DIR_ATTR]) / "r.json").unlink()

    changed = _experiment(tmp_path, trainer, _metric("acc"))
    winners = run_experiment(changed)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 1}
    by_x = {t.params["x"]: t for t in addressed_phase_study(changed).get_trials()}
    assert by_x[2].state == optuna.trial.TrialState.FAIL
    assert "JSON file not found" in by_x[2].user_attrs[FAILURE_REASON_ATTR]
    assert (by_x[1].value, by_x[4].value) == (1.0, 4.0)


def test_a_child_follows_its_re_read_parent_from_its_own_study(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With the parent's winner unchanged, the child re-reads its own trials: nothing launches."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss"), phases=_phases(child=True)))
    assert _launches(launches) == 5
    regex = Metric(
        name="loss",
        extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.eE+-]+)"),
    )

    changed = _experiment(tmp_path, trainer, regex, phases=_phases(child=True))
    before = tree_snapshot(tmp_path)
    caplog.set_level(logging.INFO)
    run_experiment(changed, dry_run=True)
    assert "phase=c study and launch count depend on the inherited values selected by 'p' (x)" in (
        caplog.text
    )
    assert "phase=c would launch" not in caplog.text
    assert "placeholders until 'p' publishes a winner: x" in caplog.text
    assert (
        "DRY RUN total: no launches for phases with known counts. "
        "Launch counts are unresolved for 1 phase"
    ) in caplog.text

    caplog.clear()
    # A child with no studies will need its full budget under every parent winner.
    new_child = changed.phases[1].model_copy(update={"name": "new_child"})
    run_experiment(
        changed.model_copy(update={"phases": [*changed.phases, new_child]}), dry_run=True
    )
    assert "DRY RUN phase=new_child would launch 2 trials" in caplog.text
    assert (
        "DRY RUN total: would launch 2 trials across 1 phase. "
        "Launch counts are unresolved for 1 phase"
    ) in caplog.text
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    assert _launches(launches) == 5
    # Status agrees with the dry run: the child's re-read holds only if the parent keeps x.
    child = inspect_status(changed, validate_ledger(changed)).refusals["c"]
    assert child.remediation == (
        "Run the same configuration. If 'p' keeps its winner after re-reading, this phase "
        "re-reads 2 finished trials from kept evidence without retraining them; a new winner "
        "starts a new study here."
    )

    winners = run_experiment(changed)

    assert _launches(launches) == 5
    assert winners["p"].effective_overrides == {"x": 4}
    assert winners["c"].effective_overrides == {"x": 4, "y": 0}


@pytest.mark.parametrize("measured", [0, 1])
def test_reread_keeps_the_partial_timeout_boundary_with_the_new_measurement_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, measured: int
) -> None:
    """A timed-out fill replays its boundary and selects only measurements the new reader has."""
    trainer, launches = _counting_trainer(tmp_path)
    short = _phases()[0].model_copy(update={"n_trials": 2, "allow_partial_grid": True})
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[short])
    run_experiment(original)
    source = addressed_phase_study(original)
    for trial in source.get_trials()[measured:]:
        (Path(trial.user_attrs[TRIAL_DIR_ATTR]) / "r.json").write_text('{"loss": 1}')
    record_accepted_partial_timeout(source, 3)
    decision = source.user_attrs[PHASE_DECISION_ATTR]
    accepting = short.model_copy(update={"n_trials": 3, "allow_incomplete_on_timeout": True})
    changed = _experiment(tmp_path, trainer, _metric("acc"), phases=[accepting])

    caplog.set_level(logging.INFO)
    before = tree_snapshot(tmp_path)
    run_experiment(changed, dry_run=True)
    assert (
        "accepted a partial result at 2 of 3 trials: would select from the re-read evidence "
        "and launch no trials"
    ) in caplog.text
    refusing = accepting.model_copy(update={"allow_incomplete_on_timeout": False})
    with pytest.raises(WinnerIntegrityError, match="accepted an incomplete result at 2/3"):
        run_experiment(
            _experiment(tmp_path, trainer, _metric("acc"), phases=[refusing]), dry_run=True
        )
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}

    for _ in range(2):
        if measured:
            winner = run_experiment(changed)["p"]
            assert winner.completion["incomplete"]
            assert winner.completion["measured_trials"] == measured
        else:
            with pytest.raises(PhaseTimeoutError, match="no winner can be selected"):
                run_experiment(changed)

    assert _launches(launches) == 2
    assert addressed_phase_study(changed).user_attrs[PHASE_DECISION_ATTR] == {
        **decision,
        "measured_trials": measured,
    }
    assert source.user_attrs[PHASE_DECISION_ATTR]["measured_trials"] == 2


def test_a_changed_trainer_beside_a_changed_reader_is_refused_not_retrained(
    tmp_path: Path,
) -> None:
    """A study with the same inherited values but another trainer config is refused by name."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss")))

    both = _experiment(tmp_path, trainer, _metric("acc"), trainer_config={"y": 5})
    with pytest.raises(StudyFingerprintMismatchError, match="trainer_config"):
        run_experiment(both)
    assert _launches(launches) == 3


def test_status_names_the_run_that_fills_a_phase_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before the fill, status says the phase has not started under the current settings."""
    trainer, _launches_path = _counting_trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _metric("loss"))
    run_experiment(original)
    changed = _experiment(tmp_path, trainer, _metric("acc"))

    refusals = inspect_status(changed, validate_ledger(changed)).refusals

    notice = refusals["p"]
    assert str(notice) == (
        "Phase 'p' has not started under the current reading settings "
        '(metric.extractor.key: "loss" -> "acc").'
    )
    assert notice.remediation == (
        "Run the same configuration to re-read 3 finished trials from kept evidence without "
        "retraining those trials."
    )
    config = write_experiment_config(tmp_path / "sweep config.yaml", original)
    overrides = ["--set", 'metric.extractor.key="acc"']
    for options in ([], ["--trials"]):
        assert invoke_cli_boundary(["status", str(config), *overrides, *options], monkeypatch) == 0
        status = capsys.readouterr().out
        (command,) = [line.lstrip("# ") for line in status.splitlines() if "phasesweep run" in line]
        assert "fix: Run the same configuration to re-read" in status
        assert shlex.split(command) == ["phasesweep", "run", str(config), *overrides]


def test_a_dry_run_says_it_would_re_read_and_writes_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The preview names the re-read and launches no trainer; the ledger and runs stay as they were."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss")))
    changed = _experiment(tmp_path, trainer, _metric("acc"))
    before = tree_snapshot(tmp_path)
    caplog.set_level(logging.INFO)

    run_experiment(changed, dry_run=True)

    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    assert _launches(launches) == 3
    assert (
        'DRY RUN phase=p reading settings changed (metric.extractor.key: "loss" -> "acc"): '
        "would re-read 3 finished trials from kept evidence; nothing would retrain"
    ) in [record.getMessage() for record in caplog.records]
    assert "would select from the re-read evidence and launch no trials" in caplog.text
    assert "DRY RUN total: would launch 0 trials across 0 phases" in caplog.text
    assert "phase replays" not in caplog.text


def test_validate_counts_kept_grid_trials_when_the_reading_settings_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The read-only ceiling includes available attempts without reading their evidence."""
    trainer, launches = _counting_trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _metric("missing"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(original)
    phase = _phases()[0]
    retry = _experiment(
        tmp_path,
        trainer,
        _metric("still_missing"),
        phases=[phase.model_copy(update={"n_trials": 4})],
    )
    retry_path = write_experiment_config(tmp_path / "retry.yaml", retry)
    oversized = _experiment(
        tmp_path,
        trainer,
        _metric("still_missing"),
        phases=[phase.model_copy(update={"n_trials": 7})],
    )
    oversized_path = write_experiment_config(tmp_path / "oversized.yaml", oversized)
    before = tree_snapshot(tmp_path)

    assert invoke_cli_boundary(["validate", str(retry_path)], monkeypatch) == 0
    assert invoke_cli_boundary(["validate", str(oversized_path)], monkeypatch) == 1
    assert "allow at most 6" in capsys.readouterr().err
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    assert _launches(launches) == 3


@pytest.mark.integration
def test_fill_source_keeps_a_failed_draw_missing_from_a_shorter_copy(tmp_path: Path) -> None:
    """Changing the metric keeps a spent failure when two sources offer the same measurements."""
    calls = tmp_path / "calls"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path)
        args, _ = parser.parse_known_args()
        calls = Path({str(calls)!r})
        with calls.open("a") as receipt:
            receipt.write("launch\\n")
        if len(calls.read_text().splitlines()) == 2:
            raise SystemExit(2)
        args.out.write_text('{{"loss": 1, "acc": 2, "mem": 3}}')
        """,
    )

    def config(key: str, n: int) -> Experiment:
        phase = _phases()[0].model_copy(update={"allow_partial_grid": True, "n_trials": n})
        return _experiment(tmp_path, trainer, _metric(key), phases=[phase])

    run_experiment(config("loss", 1))
    run_experiment(config("acc", 1))
    original = config("loss", 2)
    run_experiment(original)
    failed = addressed_phase_study(original).trials[1]
    assert failed.state == optuna.trial.TrialState.FAIL

    changed = config("mem", 2)
    run_experiment(changed)
    copied = addressed_phase_study(changed).trials[1]
    assert copied.state == failed.state
    assert copied.user_attrs[ATTEMPT_ID_ATTR] == failed.user_attrs[ATTEMPT_ID_ATTR]
    assert _launches(calls) == 2


@pytest.mark.integration
def test_a_fill_killed_part_way_is_finished_by_the_next_run(tmp_path: Path) -> None:
    """A runner killed after one copied trial leaves a fill the next run completes in place."""
    trainer, launches = _counting_trainer(tmp_path)
    typo = _experiment(tmp_path, trainer, _metric("los"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(typo)
    fixed = _experiment(tmp_path, trainer, _metric("loss"))
    config = write_experiment_config(tmp_path / "fixed.yaml", fixed)
    killer = textwrap.dedent(
        """
        import os, sys
        import optuna
        from phasesweep.cli import main
        real = optuna.Study.add_trial
        def add_then_die(self, trial):
            real(self, trial)
            os._exit(9)
        optuna.Study.add_trial = add_then_die
        sys.argv = ["phasesweep", "run", sys.argv[1]]
        main()
        """
    )
    killed = subprocess.run(
        [sys.executable, "-c", killer, str(config)],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert killed.returncode == 9, killed.stderr
    assert len(addressed_phase_study(fixed).get_trials()) == 1

    winners = run_experiment(fixed)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 4}
    filled = addressed_phase_study(fixed).get_trials()
    source = addressed_phase_study(typo).get_trials()
    assert [(t.number, t.params, t.user_attrs[TRIAL_DIR_ATTR]) for t in filled] == [
        (t.number, t.params, t.user_attrs[TRIAL_DIR_ATTR]) for t in source
    ]
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in filled)


@pytest.mark.integration
def test_an_interrupted_fill_reuses_new_source_trials_when_studies_grow(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A resumed fill appends new trials from its pinned source even when another study grows."""
    trainer, launches = _counting_trainer(tmp_path)

    def config(key: str, target: int, *, goal: str = "minimize") -> Experiment:
        phase = Phase(
            name="p",
            n_trials=target,
            gpu_policy="none",
            sampler=Sampler(type="random", seed=0),
            search_space={"x": IntParam(type="int", low=1, high=100_000)},
        )
        return _experiment(tmp_path, trainer, _metric(key, goal=goal), phases=[phase])

    run_experiment(config("loss", 1))
    run_experiment(config("acc", 2))
    original = config("loss", 3)
    run_experiment(original)
    interrupted = config("mem", 4, goal="maximize")
    add_trial = optuna.Study.add_trial

    def stop_after_two(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        add_trial(study, trial)
        if len(study.get_trials(deepcopy=False)) == 2:
            raise RuntimeError("stop after the second copied trial")

    with (
        patch.object(optuna.Study, "add_trial", stop_after_two),
        pytest.raises(RuntimeError, match="stop after the second copied trial"),
    ):
        run_experiment(interrupted)
    grown = config("loss", 4)
    run_experiment(grown, dry_run=True)
    run_experiment(grown)
    source_trials = addressed_phase_study(grown).get_trials()
    run_experiment(config("acc", 5))
    before = _launches(launches)

    notice = inspect_status(interrupted, validate_ledger(interrupted)).refusals["p"]
    assert "re-read 2 finished trials" in notice.remediation
    caplog.set_level(logging.INFO)
    run_experiment(interrupted, dry_run=True)
    assert "would re-read 2 finished trials from kept evidence" in caplog.text
    assert "target met (4 of 4 trials recorded)" in caplog.text

    run_experiment(interrupted)
    assert _launches(launches) == before
    filled = addressed_phase_study(interrupted).get_trials()
    assert [t.user_attrs[ATTEMPT_ID_ATTR] for t in filled] == [
        t.user_attrs[ATTEMPT_ID_ATTR] for t in source_trials
    ]


def test_reread_keeps_rows_committed_before_its_run_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expired read stays pending, and the next invocation skips the committed row."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss")))
    changed = _experiment(tmp_path, trainer, _metric("acc"))
    bounded = changed.model_copy(update={"timeout_seconds_per_run": 60.0})
    real_time = time.monotonic
    late_by = 0.0
    reads: list[int] = []

    def observe(
        ctx: TrialContext, observations: list[Observation], *, deadline: float | None
    ) -> dict[str, float]:
        nonlocal late_by
        reads.append(ctx.trial_id)
        if len(reads) == 2 and deadline is not None:
            late_by = deadline - real_time() + 1
        return {}

    with monkeypatch.context() as clock:
        clock.setattr(time, "monotonic", lambda: real_time() + late_by)
        clock.setattr(trial_ops, "_observe", observe)
        with pytest.raises(PhaseTimeoutError, match="run deadline while re-reading kept evidence"):
            run_experiment(bounded)
        assert reads == [0, 1]

    study = addressed_phase_study(changed)
    assert [trial.number for trial in study.get_trials()] == [0]
    assert PHASE_IDENTITY_ATTR not in study.user_attrs
    reads.clear()
    monkeypatch.setattr(trial_ops, "_observe", observe)
    assert run_experiment(changed)["p"].metric == 1
    assert reads == [1, 2]
    assert _launches(launches) == 3


def test_preview_checks_the_executable_before_launching_after_a_reread(tmp_path: Path) -> None:
    """A pending root fill with a certain top-up checks its trainer executable."""
    trainer, _ = _counting_trainer(tmp_path)
    trainer.write_text(f"#!{sys.executable}\n{trainer.read_text()}")
    trainer.chmod(0o700)
    phase = _phases()[0].model_copy(update={"n_trials": 1, "allow_partial_grid": True})
    config = make_experiment(
        persistent=tmp_path,
        trial_command=f"{shlex.quote(str(trainer))} --out {{trial_dir}}/r.json {{overrides}}",
        metric=_metric("loss"),
        phases=[phase],
    )
    run_experiment(config)
    trainer.chmod(0o600)
    changed = config.model_copy(
        update={"metric": _metric("acc"), "phases": [phase.model_copy(update={"n_trials": 2})]}
    )

    with pytest.raises(LaunchEnvironmentError, match="which is not an executable file"):
        run_experiment(changed, dry_run=True)


def test_preview_checks_the_inherited_fills_trainer_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that keeps its parent's values preflights against its donor's cohort."""
    trainer, _ = _counting_trainer(tmp_path)
    parent, child = _phases(child=True)
    short = child.model_copy(update={"n_trials": 1, "allow_partial_grid": True})
    monkeypatch.setenv("TZ", "UTC")
    run_experiment(_experiment(tmp_path, trainer, _metric("loss"), phases=[parent, short]))
    gated = child.model_copy(
        update={"gates": [RequiredFileGate(type="required_file", path="r.json")]}
    )
    changed = _experiment(tmp_path, trainer, _metric("loss"), phases=[parent, gated])
    monkeypatch.setenv("TZ", "US/Pacific")

    with pytest.raises(StudyFingerprintMismatchError, match='TZ "UTC" -> "US/Pacific"'):
        run_experiment(changed, dry_run=True)


def test_preview_refuses_a_reread_target_below_its_accepted_floor(tmp_path: Path) -> None:
    """A preview cannot lower the target below the attempts the fill already keeps."""
    trainer, _ = _counting_trainer(tmp_path)
    phase = _phases()[0].model_copy(update={"allow_partial_grid": True})
    run_experiment(_experiment(tmp_path, trainer, _metric("loss"), phases=[phase]))
    lower = phase.model_copy(update={"n_trials": 2})

    with pytest.raises(TrialTargetRefusedError, match="already accepted a target of 3 finished"):
        run_experiment(_experiment(tmp_path, trainer, _metric("acc"), phases=[lower]), dry_run=True)


def test_preview_reserves_undrawn_cells_after_a_reread(tmp_path: Path) -> None:
    """A grown full grid must reserve its new cell even when a retry filled the old target."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys
        from pathlib import Path
        launched = Path({str(tmp_path / "launched")!r})
        if not launched.exists():
            launched.touch()
            raise SystemExit(1)
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text('{{"loss": 1, "acc": 2}}')
        """,
    )
    phase = _phases()[0]
    smaller = phase.model_copy(
        update={
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2])},
            "n_trials": 2,
        }
    )
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[smaller])
    run_experiment(original)
    retry = smaller.model_copy(update={"n_trials": 3})
    run_experiment(_experiment(tmp_path, trainer, _metric("loss"), phases=[retry]))
    grown = _experiment(tmp_path, trainer, _metric("acc"), phases=[phase])

    with pytest.raises(TrialTargetRefusedError, match="too few trials for the enlarged grid"):
        run_experiment(grown, dry_run=True)


def test_preview_replays_the_partial_boundary_of_a_fully_read_fill(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When all rows already committed, preview uses the known retargeted partial result."""
    trainer, launches = _counting_trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _metric("absent"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(original)
    source = addressed_phase_study(original)
    directory = Path(source.get_trials()[0].user_attrs[TRIAL_DIR_ATTR])
    (directory / "r.json").write_text('{"loss": 1}')
    record_accepted_partial_timeout(source, 5)
    phase = original.phases[0].model_copy(
        update={"n_trials": 4, "allow_incomplete_on_timeout": True}
    )
    changed = _experiment(tmp_path, trainer, _metric("acc"), phases=[phase])
    with (
        patch.object(reread_ops, "_stamp_identity", side_effect=RuntimeError("pause after rows")),
        pytest.raises(RuntimeError, match="pause after rows"),
    ):
        run_experiment(changed)

    caplog.set_level(logging.INFO)
    run_experiment(changed, dry_run=True)
    assert "partial result at 3 of 4 trials" in caplog.text
    assert "would select from the re-read evidence and launch no trials" in caplog.text
    assert run_experiment(changed)["p"].completion["incomplete"]
    assert _launches(launches) == 3


def test_preview_checks_a_failure_streak_when_no_read_can_change_it(tmp_path: Path) -> None:
    """A fill with only trainer failures still refuses the original standing abort."""
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(9)")
    phase = _phases()[0].model_copy(update={"max_consecutive_failures": 1})
    with pytest.raises(NoFeasibleTrialError, match="1 consecutive failed/infeasible"):
        run_experiment(_experiment(tmp_path, trainer, _metric("loss"), phases=[phase]))
    changed = _experiment(tmp_path, trainer, _metric("acc"), phases=[phase])

    with pytest.raises(PhaseAbortRefusedError, match="1 consecutive failed/infeasible"):
        run_experiment(changed, dry_run=True)


@pytest.mark.integration
def test_a_finished_sibling_retry_replays_without_the_trainer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A settled sibling retry fills the pending target even when no trainer can start."""
    launches = tmp_path / "launches.txt"
    pause = tmp_path / "pause"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os, sys, time
        from pathlib import Path
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        started = out.with_name("checkpoint")
        continued = started.exists()
        started.touch()
        with Path({str(launches)!r}).open("a") as log:
            log.write(os.environ["PHASESWEEP_ATTEMPT_ID"] + "\\n")
        pause = Path({str(pause)!r})
        if pause.exists():
            pause.unlink()
            time.sleep(60)
        out.write_text(json.dumps({{"loss": 0.1 if continued else 0.9, "acc": 2 if continued else 3}}))
        """,
    )
    phase = Phase(
        name="p",
        n_trials=1,
        gpu_policy="none",
        sampler=Sampler(type="random", seed=5),
        search_space={"x": IntParam(type="int", low=1, high=100)},
    )
    command = f"{shlex.quote(str(trainer))} --out {{trial_dir}}/r.json {{overrides}}"

    def config(key: str, phase: Phase) -> Experiment:
        return make_experiment(
            persistent=tmp_path, trial_command=command, phases=[phase], metric=_metric(key)
        )

    original = config("loss", phase)
    assert run_experiment(original)["p"].metric == 0.9
    raised = phase.model_copy(update={"n_trials": 2})
    interrupted = config("loss", raised.model_copy(update={"timeout_seconds_per_phase": 2.0}))
    pause.touch()
    with pytest.raises(PhaseTimeoutError, match="timed out via phase guard"):
        run_experiment(interrupted)
    sibling = config("acc", raised)
    result = run_experiment(sibling)["p"]
    assert result.metric == 2
    parent_attempt, sibling_retry = addressed_phase_study(sibling).trials[-2:]
    assert sibling_retry.user_attrs[RERUN_OF_ATTR] == parent_attempt.number
    assert sibling_retry.user_attrs[TRIAL_DIR_ATTR] == parent_attempt.user_attrs[TRIAL_DIR_ATTR]
    assert _launches(launches) == 3
    trainer.unlink()

    restored = config("loss", raised)
    before = tree_snapshot(tmp_path)
    caplog.set_level(logging.INFO)
    run_experiment(restored, dry_run=True)
    assert "would re-read 1 finished trial from a sibling study" in caplog.text
    assert "would launch no trials" in caplog.text
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    needs_launch = config("loss", raised.model_copy(update={"n_trials": 3}))
    with pytest.raises(LaunchEnvironmentError, match="which is not an executable file"):
        run_experiment(needs_launch, dry_run=True)
    with pytest.raises(LaunchEnvironmentError, match="which is not an executable file"):
        run_experiment(needs_launch)
    returned = run_experiment(restored)["p"]
    assert returned.metric == 0.1
    assert returned.attempt_id == result.attempt_id
    assert _launches(launches) == 3


@pytest.mark.integration
def test_preview_keeps_a_budgeted_sibling_retry_conditional(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A sibling needs the requested score to settle a paid retry, even after interruption."""
    retry = tmp_path / "retry"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, time
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--x", type=int, required=True)
        args = ap.parse_args()
        if args.x == 2:
            print("first=0.9")
            print("second=0.9")
        elif (stage := Path({str(retry)!r})).exists():
            if stage.read_text() == "pause":
                stage.write_text("resume")
                time.sleep(30)
            if stage.read_text() == "resume":
                print("first=0.2")
            print("second=0.1")
        """,
    )
    phase = Phase(
        name="p",
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2])},
    )

    def config(key: str, target: int) -> Experiment:
        return make_experiment(
            persistent=tmp_path,
            trial_command=f"{shlex.quote(str(trainer))} {{overrides}}",
            metric=Metric(
                extractor=LogRegexExtractor(type="log_regex", pattern=key + r"=(?P<value>[0-9.]+)")
            ),
            phases=[phase.model_copy(update={"n_trials": target})],
        )

    original = config("first", 2)
    assert run_experiment(original)["p"].metric == 0.9
    retry.touch()
    assert run_experiment(config("second", 3))["p"].metric == 0.1
    before = addressed_phase_study(original).trials
    offline = trainer.rename(tmp_path / "offline")

    targeted_retry = config("first", 3)
    caplog.set_level(logging.INFO)
    artifacts = tree_snapshot(tmp_path)
    run_experiment(targeted_retry, dry_run=True)
    assert "would re-read 1 finished trial from a sibling study" in caplog.text
    assert "may need to launch up to 1 trial depending on the re-read results" in caplog.text
    assert "would launch no trials" not in caplog.text
    assert tree_changes(artifacts, tree_snapshot(tmp_path)) == {}
    with pytest.raises(LaunchEnvironmentError, match="which is not an executable file"):
        run_experiment(targeted_retry)
    assert addressed_phase_study(original).trials == before

    offline.rename(trainer)
    retry.write_text("pause")
    with pytest.raises(PhaseTimeoutError, match="run guard"):
        run_experiment(targeted_retry.model_copy(update={"timeout_seconds_per_run": 3.0}))
    assert retry.read_text() == "resume"
    assert run_experiment(targeted_retry)["p"].metric == 0.2


@pytest.mark.integration
def test_returning_to_an_initialized_study_adopts_a_finished_sibling_rerun(
    tmp_path: Path,
) -> None:
    """A completed draw in a sibling's reopened directory fills a local queued rerun."""
    launches, pause = tmp_path / "launches", tmp_path / "pause"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, time
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        parser.add_argument("--x", type=float, required=True)
        args = parser.parse_args()
        marker = Path({str(pause)!r})
        with Path({str(launches)!r}).open("a") as handle:
            handle.write("launch\\n")
        if marker.exists():
            marker.unlink()
            time.sleep(30)
        Path(args.out).write_text(json.dumps({{"loss": args.x, "acc": 100 + args.x}}))
        """,
    )
    phase = Phase(
        name="p",
        n_trials=1,
        gpu_policy="none",
        sampler=Sampler(type="random", seed=11),
        search_space={"x": FloatParam(type="float", low=0.1, high=3.0)},
    )
    raised = phase.model_copy(update={"n_trials": 2})
    timed = raised.model_copy(update={"timeout_seconds_per_phase": 2})
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase])
    run_experiment(original)
    pause.touch()
    with pytest.raises(PhaseTimeoutError, match="timed out via phase guard"):
        run_experiment(_experiment(tmp_path, trainer, _metric("loss"), phases=[timed]))
    recorded = addressed_phase_study(original)
    interrupted_dir = Path(recorded.trials[-1].user_attrs[TRIAL_DIR_ATTR])
    interrupted_dir.rename(tmp_path / "pruned-original")
    for _ in range(2):
        pause.touch()
        with pytest.raises(PhaseTimeoutError, match="timed out via phase guard"):
            run_experiment(_experiment(tmp_path, trainer, _metric("acc"), phases=[timed]))
    sibling = _experiment(tmp_path, trainer, _metric("acc"), phases=[raised])
    run_experiment(sibling)
    donor_study = addressed_phase_study(sibling)
    donor = donor_study.trials[-1]
    donor_path = Path(donor.user_attrs[TRIAL_DIR_ATTR])
    assert donor_path != interrupted_dir
    _record_reruns(recorded, raised)
    queued = recorded.trials[-1]
    assert queued.state == optuna.trial.TrialState.WAITING
    launches_before = _launches(launches)

    run_experiment(_experiment(tmp_path, trainer, _metric("loss"), phases=[raised]))

    adopted = recorded.trials[-1]
    assert _launches(launches) == launches_before
    assert adopted.number != donor.number
    assert adopted.user_attrs[ATTEMPT_ID_ATTR] == donor.user_attrs[ATTEMPT_ID_ATTR]
    assert adopted.user_attrs[RERUN_OF_ATTR] == queued.number
    assert adopted.value == adopted.params["x"]
    assert not recorded.get_trials(states=(optuna.trial.TrialState.WAITING,))
    attempt_id = donor.user_attrs[ATTEMPT_ID_ATTR]
    owner = producing_attempts((recorded, donor_study))[attempt_id]
    assert owner.study.study_name == study_name(sibling, "p")
    assert owner.trial.number == donor.number
    assert attempt_id not in producing_attempts((recorded,))
    generation = donor.user_attrs[GENERATION_ID_ATTR]
    snapshot = _generation_config_snapshot_path(donor_path.parent.parent, generation)
    snapshot.rename(tmp_path / "removed-generation-config")
    assert attempt_id not in producing_attempts((recorded,))
    assert producing_attempts((recorded, donor_study))[attempt_id] == owner


@pytest.mark.integration
def test_budgeted_sibling_retry_can_use_an_older_measurement(tmp_path: Path) -> None:
    """A newer sibling without the target metric leaves a usable older attempt."""
    stage, launches = tmp_path / "stage", tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--x", type=int)
        args, _ = parser.parse_known_args()
        with Path({str(launches)!r}).open("a") as out:
            out.write("launch\\n")
        stage = Path({str(stage)!r}).read_text()
        if stage == "first" and args.x == 2:
            print("first=0.9 older=0.9 newer=0.9")
        if stage == "older" and args.x == 1:
            print("first=0.2 older=0.1")
        if stage == "newer" and args.x == 1:
            print("newer=0.1")
        """,
    )
    grid = _phases()[0].model_copy(
        update={"search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2])}}
    )

    def config(key: str, target: int) -> Experiment:
        metric = Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=key + r"=(?P<value>[0-9.]+)")
        )
        return _experiment(
            tmp_path, trainer, metric, phases=[grid.model_copy(update={"n_trials": target})]
        )

    stage.write_text("first")
    assert run_experiment(config("first", 2))["p"].metric == 0.9
    stage.write_text("older")
    older = run_experiment(config("older", 3))["p"]
    stage.write_text("newer")
    newer = run_experiment(config("newer", 4))["p"]
    assert older.metric == newer.metric == 0.1
    assert older.attempt_id != newer.attempt_id
    assert _launches(launches) == 4

    returned = run_experiment(config("first", 3))["p"]

    assert returned.attempt_id == older.attempt_id
    assert returned.metric == 0.2
    assert _launches(launches) == 4


@pytest.mark.integration
def test_preview_counts_an_interrupted_checkpoint_that_the_fill_will_rerun(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A known interrupted draw still counts as a rerun while another draw awaits a reread."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys, time
        from pathlib import Path
        launched = Path({str(tmp_path / "launched")!r})
        if launched.exists():
            time.sleep(60)
        launched.touch()
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text('{{"loss": 1, "acc": 2}}')
        """,
    )
    phase = _phases()[0].model_copy(update={"n_trials": 2, "allow_partial_grid": True})
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase])
    bounded = original.model_copy(update={"timeout_seconds_per_run": 3.0})
    with pytest.raises(PhaseTimeoutError, match="timed out via run guard"):
        run_experiment(bounded)
    changed = _experiment(tmp_path, trainer, _metric("acc"), phases=[phase])

    caplog.set_level(logging.INFO)
    run_experiment(changed, dry_run=True)
    assert "would launch 1 trial, 1 of them a rerun" in caplog.text
