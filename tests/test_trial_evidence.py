"""Selection may not rank or republish a trial whose evidence left the tree.

Winner selection reads Optuna alone - state, value, feasibility, execution ids,
constraint readings - so before this guard an experiment whose winning trial
directory had been deleted happily reselected that trial and republished its
number, metric, and provenance from a tree holding nothing behind them, with
``publication_integrity: ok`` (PR #5 review / reviewer 2, blocker 7). These
tests cover the three places that now fail closed: the launch preflight over
every selection-eligible candidate, the digest check on the trial that actually
becomes the published winner, and the manifest rule that a winner carried from
an earlier generation must find that generation in this same tree.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import optuna
import pytest
import yaml

import phasesweep.engine.evidence as evidence_ops
from phasesweep import run_experiment
from phasesweep.config import (
    Constraint,
    ExecutionContext,
    Experiment,
    JsonExtractor,
    Metric,
    Phase,
    Sampler,
    WandbExtractor,
    WandbSummaryRequiredGate,
)
from phasesweep.engine import (
    NoFeasibleTrialError,
    TrialEvidenceMissingError,
    read_status,
)
from phasesweep.engine.optuna import _phase_study_name
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_config_snapshot_path,
    _generation_path,
    _generation_reproducibility_path,
    _generation_summary_path,
    _generation_winner_path,
    _generations_dir,
    _last_successful_generation_path,
    _phase_dir,
    _trial_dir_for,
)
from phasesweep.engine.publication import (
    _last_successful_generation_id,
    _resolve_publication_pointer,
)
from phasesweep.engine.selection import select_winner
from phasesweep.engine.state import (
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_INPUT_ATTR,
)
from tests.conftest import (
    assert_published_winner_evidence_local,
    make_experiment,
    write_constant_trainer,
    write_trainer,
)

# Every test here drives a real sweep through external trainer processes before
# it can check what selection published, so the whole module is integration tier.
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mutation", ["changed", "deleted"])
def test_json_primary_external_trainer_inherits_replays_and_verifies_source(tmp_path, mutation):
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        parser.add_argument("--x", type=int)
        parser.add_argument("--y", type=int, default=0)
        args = parser.parse_args()
        Path(args.out).write_text(json.dumps({
            "eval": {"loss": (args.x - 2)**2 + args.y},
            "parameters": {"x": args.x, "y": args.y},
        }))
    """,
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="eval.loss")),
        phases=[
            Phase(
                name="first",
                n_trials=2,
                sampler=Sampler(type="grid"),
                search_space={"x": {"type": "categorical", "choices": [1, 2]}},
            ),
            Phase(
                name="next",
                n_trials=2,
                inherits=["first"],
                sampler=Sampler(type="grid"),
                search_space={"y": {"type": "categorical", "choices": [0, 1]}},
            ),
        ],
    )
    winners = run_experiment(experiment)
    assert winners["first"].metric == winners["next"].metric == 0
    assert winners["next"].effective_overrides == {"x": 2, "y": 0}
    for output in _phase_dir(experiment, "next").glob("trial_*/r.json"):
        assert json.loads(output.read_text())["parameters"]["x"] == 2
    replay = run_experiment(experiment, from_phase="next")
    assert replay["first"].attempt_id == winners["first"].attempt_id
    assert replay["next"].attempt_id == winners["next"].attempt_id
    assert len(list((tmp_path / "runs" / "t").glob("*/trial_*"))) == 4
    winner = winners["first"]
    source = (
        _trial_dir_for(
            experiment,
            "first",
            winner.trial_number,
            generation_id=winner.generation_id,
            attempt_id=winner.attempt_id,
        )
        / "r.json"
    )
    if mutation == "changed":
        source.write_text(source.read_text().replace('"loss": 0', '"loss": 9'))
    else:
        source.unlink()
    with pytest.raises(TrialEvidenceMissingError):
        run_experiment(experiment, from_phase="next")


@pytest.mark.parametrize("consumers", ["primary", "constraint", "gate", "combined"])
def test_wandb_supervised_capture_publication_inheritance_and_offline_replay(
    tmp_path,
    monkeypatch,
    wandb_worker_sdk,
    consumers,
):
    import sys

    from phasesweep.engine import read_winners

    calls_path = tmp_path / "remote-reads"
    wandb_worker_sdk(f"""
        from pathlib import Path
        class Api:
            def __init__(self, **kwargs): pass
            def run(self, path):
                with Path({str(calls_path)!r}).open("a") as stream:
                    stream.write(path + "\\n")
                return type("Run", (), {{"state": "finished", "summary_metrics":
                    {{"eval/loss": 0.25, "memory": 3.0, "complete": True, "secret": "never persisted"}}}})()
    """)
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json, os
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--x", type=int)
        parser.add_argument("--receipt", required=True)
        args = parser.parse_args()
        Path(args.receipt).write_text(json.dumps({
            "x": args.x,
            "identity": {name: os.environ.get(name) for name in
                ["WANDB_RUN_ID", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_RESUME", "WANDB_BASE_URL"]},
        }))
        print("x=0.5")
    """,
    )
    remote = WandbExtractor(
        type="wandb",
        base_url="https://example.test",
        entity="e",
        project="p",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=5,
    )
    constraints = (
        [
            Constraint(
                name="memory", extractor=remote.model_copy(update={"metric_key": "memory"}), max=5
            )
        ]
        if consumers in {"constraint", "combined"}
        else []
    )
    gates = (
        [
            WandbSummaryRequiredGate(
                type="wandb_summary_required",
                base_url="https://example.test",
                entity="e",
                project="p",
                poll_seconds=0.01,
                timeout_seconds=5,
                keys=["complete"],
            )
        ]
        if consumers in {"gate", "combined"}
        else []
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        trial_command=f"{sys.executable} {trainer} --receipt {{trial_dir}}/receipt.json {{overrides}}",
        metric=Metric(extractor=remote) if consumers in {"primary", "combined"} else None,
        constraints=constraints,
        execution=ExecutionContext(inherit_env="none"),
        phases=[
            Phase(
                name="first",
                n_trials=1,
                fixed_overrides={"x": 7},
                gates=gates,
                sampler=Sampler(type="random", seed=0),
            ),
            Phase(
                name="next",
                n_trials=1,
                inherits=["first"],
                gates=gates,
                sampler=Sampler(type="random", seed=0),
            ),
        ],
    )
    # Ambient managed identity is normalized; no SDK/account authentication is needed by fixtures.
    monkeypatch.setenv("WANDB_RUN_ID", "ambient-old-run")
    monkeypatch.setenv("WANDB_PROJECT", "ambient-old-project")
    winners = run_experiment(experiment)
    assert winners["next"].effective_overrides == {"x": 7}
    assert len(calls_path.read_text().splitlines()) == 2

    for phase in experiment.phases:
        winner = winners[phase.name]
        capture = winner.objective_provenance["remote_capture"]
        assert capture["run_id"] == winner.attempt_id
        assert capture["run_state"] == "finished"
        assert "secret" not in json.dumps(capture)
        trial_dir = next(_phase_dir(experiment, phase.name).glob("trial_*"))
        receipt = json.loads((trial_dir / "receipt.json").read_text())
        assert receipt["x"] == 7
        assert receipt["identity"] == {
            "WANDB_RUN_ID": winner.attempt_id,
            "WANDB_ENTITY": "e",
            "WANDB_PROJECT": "p",
            "WANDB_RESUME": "never",
            "WANDB_BASE_URL": "https://example.test",
        }
        if consumers == "primary" and phase.name == "first":
            from dataclasses import replace

            for fields, replacement in [
                (("source", "metric_key"), "wrong"),
                (("remote_capture", "run_id"), "another-attempt"),
                (("remote_capture", "project"), "another-project"),
                (("remote_capture", "values", "eval/loss"), 9.0),
                (("remote_capture", "constraint_keys"), {"undeclared": "eval/loss"}),
            ]:
                altered = deepcopy(winner.objective_provenance)
                target = altered
                for field in fields[:-1]:
                    target = target[field]
                target[fields[-1]] = replacement
                with pytest.raises(TrialEvidenceMissingError):
                    evidence_ops._verify_winner_objective_evidence(
                        experiment, phase.name, replace(winner, objective_provenance=altered)
                    )
    # Remote deletion and unavailable credentials/SDK cannot affect frozen reads/replay.
    wandb_worker_sdk("raise AssertionError('remote access after publication')")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "wandb.apis.public", None)
    assert len(read_winners(experiment)) == 2
    with monkeypatch.context() as replay_env:
        replay_env.setenv("WANDB_MODE", "offline")
        assert run_experiment(experiment)["next"].attempt_id == winners["next"].attempt_id
    assert len(calls_path.read_text().splitlines()) == 2
    if consumers == "primary":
        from phasesweep.engine.state import TRIAL_TARGET_ATTR
        from phasesweep.errors import PhaseSweepError

        increased = experiment.model_copy(
            update={
                "phases": [
                    experiment.phases[0],
                    experiment.phases[1].model_copy(update={"n_trials": 2}),
                ]
            }
        )
        with pytest.raises(PhaseSweepError, match="optional SDK"):
            run_experiment(increased)
        study = optuna.load_study(study_name="t::next", storage=experiment.resolved_storage)
        assert study.user_attrs[TRIAL_TARGET_ATTR] == 1
        assert len(study.trials) == 1


# The constant objective of ``write_constant_trainer``, but trial 1 exits
# non-zero until a recovery marker file appears. Used to produce a *genuine*
# generation that crashed between a trial completing and anything being
# published (see the carve-out control).
_FLAKY_TRAINER = """
import argparse, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--out")
parser.add_argument("--x", type=int, default=0)
parser.add_argument("--recovered", default="")
args, _ = parser.parse_known_args()
if "trial_00001" in Path(args.out).parent.name and not Path(args.recovered).exists():
    sys.exit(3)
print("x=0.5")
"""


def _evidence_experiment(
    tmp_path: Path,
    *,
    n_trials: int = 1,
    trainer_body: str | None = None,
    max_consecutive_failures: int = 3,
    fixed_overrides: dict[str, object] | None = None,
    override_format: str = "argparse",
) -> Experiment:
    """Build a one-phase experiment on sqlite storage with a seeded sampler.

    Persistent storage is what makes a top-up reselect an *existing* trial
    rather than starting over, which is the whole subject here.
    """
    # Every trial of the default constant trainer logs the same objective, so
    # the tie break (lowest trial number) makes trial 0 the winner however many
    # top-ups run, whatever the sampler draws.
    if trainer_body is None:
        trainer = write_constant_trainer(tmp_path)
    else:
        trainer = write_trainer(tmp_path / "trainer.py", trainer_body)
    trainer_config = None
    if override_format == "yaml_file":
        trainer_input = "--config {config_path}"
        trainer_config = {"model": {"depth": 4}, "output_dir": "{trial_dir}/outputs"}
    elif override_format == "json_file":
        trainer_input = "--input {overrides_path}"
    else:
        trainer_input = "{overrides}"
    return make_experiment(
        persistent=tmp_path,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {trainer_input}",
        override_format=override_format,
        trainer_config=trainer_config,
        n_trials=n_trials,
        n_jobs=1,
        max_consecutive_failures=max_consecutive_failures,
        fixed_overrides=fixed_overrides or {},
    )


def _sole_trial_dir(experiment: Experiment) -> Path:
    """Return the single existing trial directory of phase ``p``."""
    directories = sorted(
        entry
        for entry in _phase_dir(experiment, "p").iterdir()
        if entry.is_dir() and entry.name.startswith("trial_")
    )
    assert len(directories) == 1, f"expected exactly one trial directory, got {directories}"
    return directories[0]


def _trial_count(experiment: Experiment) -> int:
    """Return how many trials the persistent phase study holds."""
    study = optuna.load_study(
        study_name=_phase_study_name(experiment, experiment.phases[0]),
        storage=experiment.storage,
    )
    return len(study.get_trials(deepcopy=False))


def _phase_trial_count(experiment: Experiment, phase_name: str) -> int:
    """Return the durable trial count for one named phase."""
    phase = next(phase for phase in experiment.phases if phase.name == phase_name)
    study = optuna.load_study(
        study_name=_phase_study_name(experiment, phase),
        storage=experiment.storage,
    )
    return len(study.get_trials(deepcopy=False))


_MARKED_TRAINER = """
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--out")
parser.add_argument("--phase_marker", default="")
args, _ = parser.parse_known_args()
if args.phase_marker:
    Path(args.phase_marker).write_text("launched\\n")
print("x=0.5")
"""


def _from_phase_evidence_experiment(
    tmp_path: Path,
    *,
    resumed_trials: int = 1,
) -> tuple[Experiment, Path]:
    """Build a two-phase experiment whose resumed trainer leaves a marker."""
    marker = tmp_path / "resumed-trainer-ran"
    trainer = write_trainer(tmp_path / "from_phase_trainer.py", _MARKED_TRAINER)
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        phases=[
            Phase(name="p", n_trials=1, sampler=Sampler(type="random", seed=0)),
            Phase(
                name="q",
                n_trials=resumed_trials,
                sampler=Sampler(type="random", seed=1),
                fixed_overrides={"phase_marker": str(marker)},
            ),
        ],
    )
    return experiment, marker


def _pointer_bytes(experiment: Experiment) -> bytes:
    """Return the raw last-success pointer, the single publication event."""
    return _last_successful_generation_path(experiment).read_bytes()


@pytest.mark.parametrize(
    ("override_format", "filename"),
    [
        ("yaml_file", "trainer_config.yaml"),
        ("argparse", "overrides_resolved.json"),
        ("hydra", "overrides_resolved.json"),
        ("json_file", "overrides.json"),
    ],
)
def test_trial_records_the_exact_generated_trainer_input(
    tmp_path: Path,
    override_format: str,
    filename: str,
) -> None:
    """Every retained input mode persists a historical trainer-input record."""
    experiment = _evidence_experiment(tmp_path, override_format=override_format)
    run_experiment(experiment)
    input_path = _sole_trial_dir(experiment) / filename
    study = optuna.load_study(
        study_name=_phase_study_name(experiment, experiment.phases[0]),
        storage=experiment.storage,
    )
    record = study.get_trials(deepcopy=False)[0].user_attrs[TRAINER_INPUT_ATTR]

    assert record == {
        "schema_version": 1,
        "format": override_format,
        "filename": filename,
        "size_bytes": input_path.stat().st_size,
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }


def _published_winner_payload(experiment: Experiment, generation_id: str) -> dict:
    """Read one generation's published winner record for phase ``p``."""
    path = _generations_dir(experiment) / generation_id / "phases" / "p" / "winner.yaml"
    return yaml.safe_load(path.read_text())


# --------------------------------------------------------------------------
# Launch preflight: a candidate whose evidence is gone stops the run before
# any new trial launches and before the publication pointer can advance.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        pytest.param("directory", "has no evidence directory", id="whole-trial-directory"),
        pytest.param(
            "overrides",
            "missing its 'overrides_resolved.json' audit artifact",
            id="resolved-overrides-only",
        ),
        pytest.param(
            "objective_source",
            "objective evidence in 'stdout.log', which is missing",
            id="objective-source-only",
        ),
        pytest.param(
            "emptied",
            "has no attempt lifecycle record",
            id="same-named-empty-directory",
        ),
    ],
)
def test_topup_refuses_a_candidate_whose_evidence_left_the_tree(
    tmp_path: Path,
    damage: str,
    expected: str,
) -> None:
    """Missing candidate evidence fails the next top-up closed, before it launches.

    Each variant removes a different piece of the one completed trial's
    evidence: the whole directory, the resolved-overrides audit artifact, the
    objective source the metric was extracted from, and the directory replaced
    by an empty one of the same name (which passes a bare existence check but
    lacks the current-format lifecycle record).
    """
    experiment = _evidence_experiment(tmp_path)
    run_experiment(experiment)
    trial_dir = _sole_trial_dir(experiment)
    pointer_before = _pointer_bytes(experiment)

    if damage == "directory":
        shutil.rmtree(trial_dir)
    elif damage == "overrides":
        (trial_dir / "overrides_resolved.json").unlink()
    elif damage == "objective_source":
        (trial_dir / "stdout.log").unlink()
    else:
        shutil.rmtree(trial_dir)
        trial_dir.mkdir()

    topup = _evidence_experiment(tmp_path, n_trials=2)
    with pytest.raises(TrialEvidenceMissingError) as excinfo:
        run_experiment(topup)

    message = str(excinfo.value)
    assert expected in message
    assert "trial 0" in message
    # Fails before any new trial work: the study still holds only the trial
    # whose evidence was broken, and nothing published over the damaged tree.
    assert _trial_count(topup) == 1
    assert _pointer_bytes(topup) == pointer_before


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "file_without_path",
        "file_without_digest",
        "file_with_parent_path",
        "file_with_absolute_path",
    ],
)
def test_present_incomplete_objective_provenance_blocks_selection(
    tmp_path: Path, damage: str
) -> None:
    """Partial current provenance records cannot publish."""
    experiment = _evidence_experiment(tmp_path)
    run_experiment(experiment)
    study = optuna.load_study(
        study_name=_phase_study_name(experiment, experiment.phases[0]),
        storage=experiment.storage,
    )
    trial = deepcopy(study.get_trials(deepcopy=False)[0])
    record = json.loads(trial.user_attrs[OBJECTIVE_PROVENANCE_ATTR])
    if damage == "empty":
        record = {}
    elif damage == "file_without_path":
        record["source"].pop("path")
    elif damage == "file_without_digest":
        record["source"].pop("sha256")
    elif damage == "file_with_parent_path":
        record["source"]["path"] = "../not-a-trial-file"
    elif damage == "file_with_absolute_path":
        record["source"]["path"] = "/not-a-trial-file"
    trial.user_attrs[OBJECTIVE_PROVENANCE_ATTR] = json.dumps(record)
    study = optuna.create_study(direction="minimize")
    study.add_trial(trial)

    with pytest.raises(TrialEvidenceMissingError, match="objective_provenance"):
        evidence_ops._validate_selection_evidence(experiment, {"p": study})
    with pytest.raises(TrialEvidenceMissingError, match="objective_provenance"):
        selected = select_winner(study, experiment, phase_name="p")
        evidence_ops._verify_winner_objective_evidence(experiment, "p", selected)


def test_untouched_tree_still_publishes_a_clean_topup(tmp_path: Path) -> None:
    """Positive control: the guard costs an intact tree nothing."""
    experiment = _evidence_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)

    topup = _evidence_experiment(tmp_path, n_trials=2)
    winners = run_experiment(topup)

    assert _trial_count(topup) == 2
    second_generation = _last_successful_generation_id(topup)
    assert second_generation is not None
    assert second_generation != first_generation
    # Constant objective: trial 0 still wins on the tie break, so this is also
    # the ordinary carry-forward path the manifest rule below governs.
    assert winners["p"].trial_number == 0
    assert read_status(topup)["publication_integrity"] == "ok"
    assert_published_winner_evidence_local(_experiment_dir(topup))


def test_log_regex_evaluation_revision_change_blocks_study_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phasesweep.evidence.evaluation as evaluation_ops

    experiment = _evidence_experiment(tmp_path)
    run_experiment(experiment)
    pointer_before = _pointer_bytes(experiment)
    monkeypatch.setattr(
        evaluation_ops,
        "LOG_REGEX_EVALUATION_REVISION",
        evaluation_ops.LOG_REGEX_EVALUATION_REVISION + 1,
    )

    with pytest.raises(
        TrialEvidenceMissingError,
        match="extractor evaluation contract.*evaluator revision.*start a new experiment name",
    ):
        run_experiment(_evidence_experiment(tmp_path, n_trials=2))

    assert _trial_count(experiment) == 1
    assert _pointer_bytes(experiment) == pointer_before


@pytest.mark.parametrize(
    ("override_format", "filename", "damage"),
    [
        pytest.param("yaml_file", "trainer_config.yaml", "delete", id="yaml-delete"),
        pytest.param("yaml_file", "trainer_config.yaml", "alter", id="yaml-alter"),
        pytest.param("json_file", "overrides.json", "delete", id="json-delete"),
        pytest.param("json_file", "overrides.json", "alter", id="json-alter"),
    ],
)
def test_topup_refuses_damaged_generated_trainer_input(
    tmp_path: Path,
    override_format: str,
    filename: str,
    damage: str,
) -> None:
    """Every candidate retains the exact historical file its trainer consumed."""
    experiment = _evidence_experiment(tmp_path, override_format=override_format)
    run_experiment(experiment)
    pointer_before = _pointer_bytes(experiment)
    input_path = _sole_trial_dir(experiment) / filename

    if damage == "delete":
        input_path.unlink()
    else:
        original = input_path.read_bytes()
        input_path.write_bytes(b"!" + original[1:])
        assert input_path.stat().st_size == len(original)

    topup = _evidence_experiment(
        tmp_path,
        n_trials=2,
        override_format=override_format,
    )
    with pytest.raises(TrialEvidenceMissingError, match="trainer input"):
        run_experiment(topup)

    assert _trial_count(topup) == 1
    assert _pointer_bytes(topup) == pointer_before


def test_carried_winner_requires_its_original_generated_input(tmp_path: Path) -> None:
    """A later generation cannot carry a winner whose trainer input disappeared."""
    experiment = _evidence_experiment(tmp_path, override_format="yaml_file")
    run_experiment(experiment)
    run_experiment(experiment)
    carrying_generation = _last_successful_generation_id(experiment)
    assert carrying_generation is not None
    pointer_before = _pointer_bytes(experiment)
    (_sole_trial_dir(experiment) / "trainer_config.yaml").unlink()

    with pytest.raises(TrialEvidenceMissingError, match="trainer_config.yaml"):
        run_experiment(experiment)

    assert _last_successful_generation_id(experiment) == carrying_generation
    assert _pointer_bytes(experiment) == pointer_before


# --------------------------------------------------------------------------
# Selection-time digest: only the trial that actually wins is re-hashed, so a
# same-length edit of its objective source is caught where it matters.
# --------------------------------------------------------------------------


def test_republishing_refuses_a_winner_whose_objective_bytes_changed(tmp_path: Path) -> None:
    """A length-preserving edit of the winner's source is caught at selection.

    Existence and recorded size still match, so nothing before the winner-only
    SHA-256 comparison can see this: the guard has to be the digest.
    """
    experiment = _evidence_experiment(tmp_path)
    run_experiment(experiment)
    pointer_before = _pointer_bytes(experiment)
    source = _sole_trial_dir(experiment) / "stdout.log"
    original = source.read_bytes()
    source.write_bytes(b"y" + original[1:])
    assert source.stat().st_size == len(original)

    with pytest.raises(TrialEvidenceMissingError, match="bytes behind the published metric"):
        run_experiment(_evidence_experiment(tmp_path))

    assert _pointer_bytes(experiment) == pointer_before


# --------------------------------------------------------------------------
# --from-phase preflight: skipped winners are verified against their source
# trial before downstream work can begin, without needing the prior ledger.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        pytest.param("changed", "bytes behind the published metric", id="same-size-edit"),
        pytest.param(
            "missing", "objective evidence in 'stdout.log', which is missing", id="missing"
        ),
    ],
)
def test_from_phase_refuses_damaged_skipped_winner_before_downstream_launch(
    tmp_path: Path,
    damage: str,
    expected: str,
) -> None:
    """A carried winner's source must verify before the resumed phase can run."""
    original, marker = _from_phase_evidence_experiment(tmp_path)
    run_experiment(original)
    marker.unlink()
    source = _sole_trial_dir(original)
    pointer_before = _pointer_bytes(original)
    if damage == "changed":
        original_bytes = (source / "stdout.log").read_bytes()
        (source / "stdout.log").write_bytes(b"y" + original_bytes[1:])
        assert (source / "stdout.log").stat().st_size == len(original_bytes)
    else:
        (source / "stdout.log").unlink()

    resumed, _ = _from_phase_evidence_experiment(tmp_path, resumed_trials=2)
    with pytest.raises(TrialEvidenceMissingError, match=expected):
        run_experiment(resumed, from_phase="q")

    assert _phase_trial_count(resumed, "q") == 1
    assert not marker.exists()
    assert _pointer_bytes(resumed) == pointer_before


def test_from_phase_keeps_a_skipped_winner_when_its_ledger_is_unavailable(
    tmp_path: Path,
) -> None:
    """Winner serialization supplies enough evidence identity to skip a missing study."""
    experiment, _marker = _from_phase_evidence_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    payload = yaml.safe_load(
        (_generations_dir(experiment) / generation_id / "phases" / "p" / "winner.yaml").read_text()
    )
    p_study = optuna.load_study(
        study_name=_phase_study_name(experiment, experiment.phases[0]),
        storage=experiment.storage,
    )
    assert (
        payload["trainer_input"]
        == p_study.get_trials(deepcopy=False)[0].user_attrs[TRAINER_INPUT_ATTR]
    )
    optuna.delete_study(
        study_name=_phase_study_name(experiment, experiment.phases[0]),
        storage=experiment.storage,
    )

    resumed = run_experiment(experiment, from_phase="q")

    assert resumed["p"].trial_number == 0


# --------------------------------------------------------------------------
# Manifest rule: a winner carried from an earlier generation must find that
# generation here. Fires pre-commit on the publish path and on every read.
# --------------------------------------------------------------------------


def _tree_with_a_carried_winner(tmp_path: Path) -> tuple[Experiment, str, str]:
    """Publish twice so the second generation's winner cites the first.

    :return tuple[Experiment, str, str]: The experiment plus the source and
        carrying generation ids.
    """
    experiment = _evidence_experiment(tmp_path)
    run_experiment(experiment)
    source_generation = _last_successful_generation_id(experiment)
    assert source_generation is not None
    run_experiment(experiment)
    carrying_generation = _last_successful_generation_id(experiment)
    assert carrying_generation is not None
    assert carrying_generation != source_generation
    payload = _published_winner_payload(experiment, carrying_generation)
    assert payload["winner_source"]["generation_id"] == source_generation
    return experiment, source_generation, carrying_generation


def test_deleting_a_carried_winners_source_generation_fails_read_and_write(
    tmp_path: Path,
) -> None:
    """The cited generation holds the evidence, so its absence is corruption.

    This is the tree ``assert_published_winner_evidence_local`` rejects
    (re-review v0.5.19 / blocker B1) and that the manifest used to report as
    ``ok``.
    """
    experiment, source_generation, _carrying = _tree_with_a_carried_winner(tmp_path)
    shutil.rmtree(_generations_dir(experiment) / source_generation)

    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert source_generation in str(status["publication_error"])
    assert "does not exist in this tree" in str(status["publication_error"])
    with pytest.raises(AssertionError):
        assert_published_winner_evidence_local(_experiment_dir(experiment))

    pointer_before = _pointer_bytes(experiment)
    with pytest.raises(RuntimeError, match="does not exist in this tree"):
        run_experiment(_evidence_experiment(tmp_path))
    assert _pointer_bytes(experiment) == pointer_before
    assert _last_successful_generation_id(experiment) is None


@pytest.mark.parametrize("damage", ["missing", "edited", "coherently_changed"])
def test_published_source_generation_damage_fails_integrity(tmp_path: Path, damage: str) -> None:
    """A carried result cannot outlive or contradict its published source."""
    experiment, source_generation, _ = _tree_with_a_carried_winner(tmp_path)
    source_dir = _generations_dir(experiment) / source_generation
    winner_path = source_dir / "phases" / "p" / "winner.yaml"
    if damage == "missing":
        winner_path.unlink()
        diagnostic = "published but holds no winner record"
    else:
        winner = yaml.safe_load(winner_path.read_text())
        winner["metric"][experiment.metric.name] = 0.9
        winner["params"]["x"] = -1
        winner_path.write_text(yaml.safe_dump(winner, sort_keys=False))
        if damage == "coherently_changed":
            summary_path = source_dir / "summary.yaml"
            summary = yaml.safe_load(summary_path.read_text())
            summary["phases"][0]["metric"] = 0.9
            summary["phases"][0]["params"] = {"x": -1}
            artifact = next(
                item
                for item in summary["artifacts"]
                if item.get("kind") == "winner" and item.get("phase") == "p"
            )
            artifact["sha256"] = hashlib.sha256(winner_path.read_bytes()).hexdigest()
            summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
            diagnostic = "disagrees with the result recorded"
        else:
            diagnostic = "source generation"
    assert (source_dir / "summary.yaml").is_file()

    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert diagnostic in str(status["publication_error"])

    with pytest.raises(RuntimeError, match=diagnostic):
        run_experiment(_evidence_experiment(tmp_path))


@pytest.mark.parametrize("identity_field", ["experiment", "generation_id"])
def test_carried_winner_rejects_reidentified_source_summary(
    tmp_path: Path,
    identity_field: str,
) -> None:
    """A carried winner's source must retain its experiment and generation identity."""
    experiment, source_generation, carrying_generation = _tree_with_a_carried_winner(tmp_path)
    carried = yaml.safe_load(
        _generation_winner_path(experiment, carrying_generation, "p").read_text()
    )
    assert carried["generation_id"] == source_generation

    summary_path = _generation_summary_path(experiment, source_generation)
    summary = yaml.safe_load(summary_path.read_text())
    snapshot_path = _generation_config_snapshot_path(experiment, source_generation)
    snapshot = yaml.safe_load(snapshot_path.read_text())
    record_path = _generation_reproducibility_path(experiment, source_generation)
    record = json.loads(record_path.read_text())
    forged = f"forged-{identity_field}"

    summary[identity_field] = forged
    record[identity_field] = forged
    if identity_field == "experiment":
        snapshot["experiment"] = forged
        snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
        record["config_snapshot"]["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record))
    for entry in summary["artifacts"]:
        if entry.get("path") == "config.snapshot.yaml":
            entry["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        elif entry.get("path") == "reproducibility.json":
            entry["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert f"names a different {identity_field.removesuffix('_id')}" in pointer.error


def test_winner_carried_from_a_generation_that_crashed_before_publication_publishes(
    tmp_path: Path,
) -> None:
    """Control for the carve-out: an unpublished source namespace is legitimate.

    Built entirely from the real flow, no fabrication. The first invocation's
    trial 0 completes and trial 1 exits non-zero, tripping the consecutive
    failure breaker; the phase raises before any winner is saved, so that
    generation's namespace holds its claim-time provenance and *no* summary and
    *no* winner record - exactly the shape a crash between trial completion and
    publication leaves behind. The recovery run raises the trial target (which
    starts a new recovery streak for a random sampler), completes, and
    legitimately publishes trial 0, citing the crashed generation.

    Requiring a winner record in every cited source generation - the reviewer's
    blanket rule - would make this ordinary recovery permanently unpublishable.
    """
    recovery_marker = tmp_path / "recovered.txt"
    crashing = _evidence_experiment(
        tmp_path,
        n_trials=2,
        trainer_body=_FLAKY_TRAINER,
        max_consecutive_failures=1,
        fixed_overrides={"recovered": str(recovery_marker)},
    )
    with pytest.raises(NoFeasibleTrialError, match="consecutive failures"):
        run_experiment(crashing)

    crashed_generation = yaml.safe_load(_generation_path(crashing).read_text())["generation_id"]
    crashed_dir = _generations_dir(crashing) / crashed_generation
    assert crashed_dir.is_dir()
    assert not (crashed_dir / "summary.yaml").exists()
    assert not (crashed_dir / "phases").exists()
    assert _last_successful_generation_id(crashing) is None

    recovery_marker.write_text("ok\n")
    recovering = _evidence_experiment(
        tmp_path,
        n_trials=3,
        trainer_body=_FLAKY_TRAINER,
        max_consecutive_failures=1,
        fixed_overrides={"recovered": str(recovery_marker)},
    )
    winners = run_experiment(recovering)

    assert winners["p"].trial_number == 0
    assert winners["p"].generation_id == crashed_generation
    published = _last_successful_generation_id(recovering)
    assert published is not None
    payload = _published_winner_payload(recovering, published)
    assert payload["winner_source"]["generation_id"] == crashed_generation
    assert read_status(recovering)["publication_integrity"] == "ok"
    assert_published_winner_evidence_local(_experiment_dir(recovering))
