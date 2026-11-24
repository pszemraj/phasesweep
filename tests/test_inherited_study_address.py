"""A child phase's study is addressed by the values it inherits."""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, IntParam, Sampler
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.engine.fingerprints import phase_study_name
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.evidence.models import JsonExtractor
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import make_experiment, write_trainer
from tests.recovery_helpers import fabricate_stale_trial


def _trainer(tmp_path: Path) -> tuple[Path, Path]:
    """Write a trainer that reports ``x`` and logs each call's ``x`` and ``y``.

    :param Path tmp_path: Directory holding the trainer and its call log.
    :return tuple[Path, Path]: The trainer and its call log.
    """
    calls = tmp_path / "calls.log"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out")
        ap.add_argument("--x", type=int, default=0)
        ap.add_argument("--y", type=int, default=0)
        args, _ = ap.parse_known_args()
        with Path({str(calls)!r}).open("a") as log:
            log.write(f"{{args.x}},{{args.y}}\\n")
        print(f"x={{args.x + args.y}}")
        """,
    )
    return trainer, calls


def _plan(tmp_path: Path, trainer: Path, choices: list[int], parent: str = "parent") -> Experiment:
    """Build a parent grid over ``x`` and a child that inherits its winner and searches ``y``.

    :param Path tmp_path: Directory holding the ledger and workdir.
    :param Path trainer: The trainer script.
    :param list[int] choices: The parent's ``x`` choices; the lowest one wins.
    :param str parent: The parent phase's name, defaults to ``parent``.
    :return Experiment: The experiment.
    """
    return make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        phases=[
            Phase(
                name=parent,
                n_trials=len(choices),
                gpu_policy="none",
                sampler=Sampler(type="grid", seed=0),
                search_space={"x": CategoricalParam(type="categorical", choices=choices)},
            ),
            Phase(
                name="child",
                n_trials=2,
                gpu_policy="none",
                inherits=[parent],
                sampler=Sampler(type="grid", seed=0),
                search_space={"y": IntParam(type="int", low=0, high=1)},
            ),
        ],
    )


def _child_studies(experiment: Experiment) -> dict[str, list[optuna.trial.FrozenTrial]]:
    """Read every study of phase ``child`` with its trials.

    :param Experiment experiment: The experiment.
    :return dict[str, list[optuna.trial.FrozenTrial]]: Trials by study name.
    """
    storage = _resolve_storage(experiment.resolved_storage)
    return {
        summary.study_name: optuna.load_study(
            study_name=summary.study_name, storage=storage
        ).get_trials(deepcopy=False)
        for summary in optuna.get_all_study_summaries(storage)
        if summary.study_name.startswith(f"{experiment.experiment}::child::")
    }


def _calls(calls: Path) -> list[str]:
    """Read every trainer call so far as ``x,y``.

    :param Path calls: The trainer's call log.
    :return list[str]: One entry per call, in call order.
    """
    return calls.read_text().split() if calls.exists() else []


@pytest.mark.integration
def test_inherited_cli_order_addresses_only_trials_that_ran_in_that_order(tmp_path: Path) -> None:
    """Reordering inherited alias arguments changes the selected study, including on a reread."""
    calls = tmp_path / "cli_calls.log"
    trainer = write_trainer(
        tmp_path / "cli_trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out")
        parser.add_argument("--lr", "--learning-rate", type=float)
        args = parser.parse_args()
        with Path({str(calls)!r}).open("a") as handle:
            handle.write(f"{{args.lr}}\\n")
        Path(args.out).write_text(json.dumps({{"loss": args.lr, "score": 1 - args.lr}}))
        """,
    )
    left = Phase(name="left", n_trials=1, gpu_policy="none", fixed_overrides={"lr": 0.3})
    right = Phase(
        name="right", n_trials=1, gpu_policy="none", fixed_overrides={"learning-rate": 0.2}
    )
    child = Phase(
        name="child",
        n_trials=1,
        gpu_policy="none",
        inherits=["left", "right"],
        fixed_overrides={"lr": 0.1},
    )
    reordered = child.model_copy(update={"inherits": ["right", "left"]})
    metric = Metric(
        name="loss",
        goal="minimize",
        extractor=JsonExtractor(type="json", path="r.json", key="loss"),
    )
    original = make_experiment(
        persistent=tmp_path, trainer=trainer, phases=[left, right, child], metric=metric
    )
    changed = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        phases=[left, right, reordered],
        metric=Metric(
            name="loss",
            goal="minimize",
            extractor=JsonExtractor(type="json", path="r.json", key="score"),
        ),
    )

    assert run_experiment(original)["child"].metric == 0.2
    assert _calls(calls) == ["0.3", "0.2", "0.2"]
    assert run_experiment(changed)["child"].metric == 0.9
    assert _calls(calls) == ["0.3", "0.2", "0.2", "0.1"]
    assert run_experiment(original)["child"].metric == 0.2

    changed_order = make_experiment(
        persistent=tmp_path, trainer=trainer, phases=[left, right, reordered], metric=metric
    )
    assert run_experiment(changed_order, from_phase="child")["child"].metric == 0.1
    assert _calls(calls) == ["0.3", "0.2", "0.2", "0.1"]


@pytest.mark.parametrize(
    ("fmt", "command", "base"),
    [
        ("yaml_file", "train {config_path}", {}),
        ("json_file", "train {overrides_path}", {}),
        ("argparse", "train {overrides}", {"trainer": {"lr": 0.0}, "learning-rate": 0.0}),
        ("hydra", "train {overrides}", {"trainer": {"lr": 0.0}, "learning-rate": 0.0}),
    ],
)
def test_inheritance_order_reuses_studies_with_equivalent_trainer_inputs(
    fmt: str, command: str, base: dict[str, float | dict[str, float]]
) -> None:
    """File input and existing CLI base positions make the two parent orders equivalent."""
    phase = Phase(name="p", n_trials=1, gpu_policy="none")
    experiment = make_experiment(
        phases=[phase], override_format=fmt, trial_command=command, trainer_config=base
    )
    forward = {"trainer.lr": 0.1, "learning-rate": 0.2}
    reverse = dict(reversed(forward.items()))
    assert phase_study_name(experiment, phase, forward) == phase_study_name(
        experiment, phase, reverse
    )


@pytest.mark.integration
def test_reordering_fixed_cli_arguments_refuses_existing_results(tmp_path: Path) -> None:
    """A changed local order changes the trainer and requires a new phase or study."""
    trainer, calls = _trainer(tmp_path)
    phase = Phase(name="p", n_trials=1, gpu_policy="none", fixed_overrides={"x": 1, "y": 2})
    original = make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])
    run_experiment(original)
    reordered = phase.model_copy(update={"fixed_overrides": {"y": 2, "x": 1}})
    changed = make_experiment(persistent=tmp_path, trainer=trainer, phases=[reordered])
    with pytest.raises(StudyFingerprintMismatchError, match="cli_override_order"):
        run_experiment(changed)
    assert _calls(calls) == ["1,2"]


@pytest.mark.integration
def test_a_changed_parent_winner_runs_the_child_in_a_new_study_and_a_reverted_one_reuses_the_old(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A grown parent with a new winner runs the child again in a second study without refusing, keeps the first study's trials, and a renamed parent whose winner is the first one again launches nothing for the child."""
    trainer, calls = _trainer(tmp_path)
    run_experiment(_plan(tmp_path, trainer, [3, 5]))
    ((first_name, first),) = _child_studies(_plan(tmp_path, trainer, [3, 5])).items()
    assert len(first) == 2

    grown = _plan(tmp_path, trainer, [3, 5, 1])
    before = len(_calls(calls))
    with caplog.at_level(logging.INFO, logger="phasesweep"):
        winners = run_experiment(grown)

    assert winners["child"].effective_overrides["x"] == 1
    assert sorted(_calls(calls)[before:]) == ["1,0", "1,0", "1,1"]
    assert (
        "child  inherits x: 3 -> 1, so it starts a new study; "
        "the earlier study keeps its 2 trials." in caplog.text
    )
    studies = _child_studies(grown)
    assert len(studies) == 2
    assert [(t.state, t.params) for t in studies[first_name]] == [
        (t.state, t.params) for t in first
    ]
    assert sorted(len(trials) for trials in studies.values()) == [2, 2]

    reverted = _plan(tmp_path, trainer, [3, 5], parent="narrow")
    before = len(_calls(calls))
    assert run_experiment(reverted)["child"].effective_overrides["x"] == 3
    assert sorted(_calls(calls)[before:]) == ["3,0", "5,0"]
    assert sorted(len(trials) for trials in _child_studies(reverted).values()) == [2, 2]


@pytest.mark.integration
def test_a_running_trial_in_a_study_the_config_no_longer_addresses_is_settled(
    tmp_path: Path,
) -> None:
    """A trial a crashed run left RUNNING in the child's earlier study is marked FAIL by the next run's recovery, even though that run's parent winner addresses a different child study."""
    trainer, _calls_log = _trainer(tmp_path)
    original = _plan(tmp_path, trainer, [3, 5])
    run_experiment(original)
    (study_name,) = _child_studies(original)
    study = optuna.load_study(
        study_name=study_name, storage=_resolve_storage(original.resolved_storage)
    )
    stale = fabricate_stale_trial(original, "child", attempt_id="abandoned", study=study)
    write_attempt_lifecycle(stale.attempt_dir, attempt_id="abandoned", state="allocated")

    grown = _plan(tmp_path, trainer, [3, 5, 1])
    run_experiment(grown)

    old = _child_studies(grown)[study_name]
    assert old[stale.number].state is optuna.trial.TrialState.FAIL
    assert [t.state for t in old if t.number != stale.number] == [
        optuna.trial.TrialState.COMPLETE
    ] * 2
