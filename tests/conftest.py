"""Shared test fixtures and helpers for phasesweep tests.

One experiment factory to replace the 5+ near-identical _minimal_experiment /
_exp / _make_exp helpers scattered across test files. Tests that need
specialized construction (e.g. template validation, selector constraints) keep
their local helpers — this covers the >80% common case.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from phasesweep.config import (
    Constraint,
    Experiment,
    IntParam,
    JsonExtractor,
    Metric,
    Phase,
)

# Repository root, derived from the conftest location. Tests that copy/edit
# the example experiment.yaml read this so they don't hard-code paths.
REPO = Path(__file__).resolve().parent.parent


def make_experiment(
    *,
    workdir: str | Path | None = None,
    storage: str | None = None,
    trial_command: str = "echo {overrides}",
    override_format: str = "hydra",
    constraints: list[Constraint] | None = None,
    phases: list[Phase] | None = None,
    env: dict[str, str] | None = None,
    **phase_overrides: Any,
) -> Experiment:
    """Build a minimal valid Experiment for testing.

    If ``phases`` is not given, a single phase named ``"p"`` is created with
    ``search_space={"x": IntParam(0..10)}``. Extra ``**phase_overrides`` are
    forwarded to that default phase (e.g. ``n_trials=4``, ``gpu_ids=[0]``).
    """
    if phases is None:
        base: dict[str, Any] = dict(
            name="p",
            n_trials=2,
            search_space={"x": IntParam(type="int", low=0, high=10)},
        )
        base.update(phase_overrides)
        phases = [Phase(**base)]  # type: ignore[arg-type]

    kwargs: dict[str, Any] = dict(
        experiment="t",
        trial_command=trial_command,
        override_format=override_format,
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        phases=phases,
    )
    if workdir is not None:
        kwargs["workdir"] = str(workdir)
    if storage is not None:
        kwargs["storage"] = storage
    if constraints is not None:
        kwargs["constraints"] = constraints
    if env is not None:
        kwargs["env"] = env

    return Experiment(**kwargs)


def write_yaml(tmp_path: Path, body: str) -> Path:
    """Write a YAML body verbatim to ``tmp_path/exp.yaml``; return the path.

    Body is passed through ``textwrap.dedent`` so callers can use indented
    triple-quoted strings naturally. No ``.format()`` magic — if a test
    needs ``{tmp}`` substituted, it does so at the call site via
    ``body.format(tmp=tmp_path)`` before calling this helper.
    """
    p = tmp_path / "exp.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def write_trainer(path: Path, body: str) -> Path:
    """Write an executable Python trainer script to ``path``.

    ``path`` may be a directory (the trainer is placed at ``path/trainer.py``)
    or a file path. Returns the resolved file path. Body is dedented and
    given a ``#!/usr/bin/env python3`` shebang.
    """
    if path.is_dir():
        path = path / "trainer.py"
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return path


def write_constant_trainer(tmp_path: Path) -> Path:
    """Drop a minimal trainer that writes ``{"x": 0.5}`` to ``--out``.

    Cheap enough for tests that need a real subprocess run before mutating
    the parent config and re-running with ``--from-phase``.
    """
    return write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"x": 0.5}))
        """,
    )
