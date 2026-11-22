"""Every fix line that renames a phase names the phases that inherit it, whichever command prints it."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.config.models import Phase
from phasesweep.config.search import IntParam, Sampler
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    write_experiment_config,
    write_trainer,
)

_KEYS = {"a": "x", "b": "y", "c": "z"}


def _printed(text: str) -> str:
    """Join a terminal's wrapped lines, so a sentence is found however the width broke it."""
    return " ".join(text.split())


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
        "raise phases[a].timeout_seconds_per_trial and give `a`, `b` and `c` new names and "
        "update the `inherits` that name them."
    ) in _printed(capsys.readouterr().err)


@pytest.mark.integration
def test_status_names_the_phases_that_inherit_the_phase_that_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With `a` finished and `b` aborted on timeouts, the status fix names `b` and `c`, the one phase that inherits it."""
    config = _timeout_chain(tmp_path, slow_key="y")
    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 1
    capsys.readouterr()

    assert invoke_cli_boundary(["status", str(config)], monkeypatch) == 0

    fix = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("# fix: For the timed-out trials")
    )
    assert "give `b` and `c` new names and update the `inherits` that name them." in fix
