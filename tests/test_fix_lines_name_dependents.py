"""Every fix line that renames a phase names the phases that inherit it, whichever command prints it."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import CategoricalParam, IntParam, Sampler
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    write_constant_trainer,
    write_experiment_config,
    write_trainer,
)

_KEYS = {"a": "x", "b": "y", "c": "z"}


def _grid_chain(tmp_path: Path, trainer: Path, *, a_trials: int) -> Experiment:
    """Build a grid chain ``a`` -> ``b`` -> ``c`` of two cells per phase, ``a`` asking for ``a_trials``."""

    def phase(name: str, trials: int, parent: str | None) -> Phase:
        return Phase(
            name=name,
            n_trials=trials,
            inherits=[] if parent is None else [parent],
            sampler=Sampler(type="grid"),
            gpu_policy="none",
            search_space={_KEYS[name]: CategoricalParam(type="categorical", choices=[1, 2])},
        )

    return make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        phases=[phase("a", a_trials, None), phase("b", 2, "a"), phase("c", 2, "b")],
    )


@pytest.fixture(scope="module")
def raised_grid_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the grid chain to completion, then return a config that asks ``a`` for a third trial.

    ``a`` has two cells and both are measured, so the raised target is refused at every entry
    point that reads the recorded study.
    """
    root = tmp_path_factory.mktemp("grid-chain")
    trainer = write_constant_trainer(root)
    run_experiment(_grid_chain(root, trainer, a_trials=2))
    return write_experiment_config(root / "raised.yaml", _grid_chain(root, trainer, a_trials=3))


def _printed(text: str) -> str:
    """Join a terminal's wrapped lines, so a sentence is found however the width broke it."""
    return " ".join(text.split())


@pytest.mark.integration
@pytest.mark.parametrize(
    "command", [["run"], ["validate"], ["run", "--dry-run"]], ids=["run", "validate", "dry-run"]
)
def test_a_refused_grid_target_names_the_phases_that_inherit_the_phase(
    raised_grid_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: list[str],
) -> None:
    """The step to enlarge the grid names `b` and `c` when run, validate or the dry run refuses the target."""
    capsys.readouterr()

    status = invoke_cli_boundary([command[0], str(raised_grid_config), *command[1:]], monkeypatch)

    assert status == 1
    assert (
        "change the search space and give the phase, and each phase that inherits it "
        "(`b`, `c`), a new `name`, updating every `inherits` to match"
    ) in _printed(capsys.readouterr().err)


def _timeout_chain(tmp_path: Path, *, slow_key: str) -> Path:
    """Write a random chain ``a`` -> ``b`` -> ``c`` whose trials time out where they search ``slow_key``.

    The trainer sleeps past the per-trial timeout of the phase that searches ``slow_key``, and
    that phase aborts after two failures, so its trials leave a recorded timeout.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, sys, time
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        if any(arg.startswith("--{slow_key}") for arg in sys.argv):
            time.sleep(2.0)
        with open(args.out, "w") as f:
            json.dump({{"x": 0.5}}, f)
        print("x=0.5")
        """,
    )

    def phase(name: str, parent: str | None) -> Phase:
        timeout = {"timeout_seconds_per_trial": 0.5} if _KEYS[name] == slow_key else {}
        return Phase(
            name=name,
            n_trials=2,
            inherits=[] if parent is None else [parent],
            max_consecutive_failures=2,
            **timeout,
            sampler=Sampler(type="random", seed=0),
            gpu_policy="none",
            search_space={_KEYS[name]: IntParam(type="int", low=0, high=9)},
        )

    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        phases=[phase("a", None), phase("b", "a"), phase("c", "b")],
    )
    return write_experiment_config(tmp_path / "timeouts.yaml", experiment)


@pytest.mark.integration
def test_the_abort_fix_line_names_the_phases_that_inherit_the_timed_out_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A phase whose trials time out aborts with a step that renames it and `b` and `c`."""
    config = _timeout_chain(tmp_path, slow_key="x")

    status = invoke_cli_boundary(["run", str(config)], monkeypatch)

    assert status == 1
    assert (
        "Raise it and give the phase, and each phase that inherits it (`b`, `c`), a new `name`"
    ) in _printed(capsys.readouterr().err)


@pytest.mark.integration
def test_status_names_the_phases_that_inherit_the_phase_that_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With `a` finished and `b` aborted on timeouts, the status fix names `c`, the one phase that inherits `b`."""
    config = _timeout_chain(tmp_path, slow_key="y")
    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 1
    capsys.readouterr()

    assert invoke_cli_boundary(["status", str(config)], monkeypatch) == 0

    fix = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("# fix: A trial timed out")
    )
    assert "Raise it and give the phase, and each phase that inherits it (`c`), a new `name`" in fix
