"""Shared test fixtures and helpers for phasesweep tests.

One experiment factory to replace the 5+ near-identical _minimal_experiment /
_exp / _make_exp helpers scattered across test files. Tests that need specialized construction can pass explicit phases or phase keyword overrides.
"""

from __future__ import annotations

import shutil
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from phasesweep.config import (
    Constraint,
    ExecutionContext,
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
)
from phasesweep.evidence import TrialContext
from phasesweep.runtime.process import _read_proc_stat

# Repository root, derived from the conftest location. Tests that copy/edit
# the example experiment.yaml read this so they don't hard-code paths.
REPO = Path(__file__).resolve().parent.parent


def is_pid_zombie(pid: int) -> bool:
    """Return whether a test subprocess has exited but awaits parent reaping."""
    stat = _read_proc_stat(Path("/proc") / str(pid))
    return stat is not None and stat.state == "Z"


@pytest.fixture(autouse=True)
def isolate_phasesweep_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host-wide test locks inside each test's temp directory."""
    lock_dir = tmp_path / "phasesweep-locks"
    lock_dir.mkdir(mode=0o700)
    lock_dir.chmod(0o700)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(lock_dir))


@pytest.fixture(autouse=True)
def isolate_signal_ownership_tokens() -> Iterator[None]:
    """Snapshot and restore the process-lifetime signal-ownership tokens per test.

    ``phasesweep.runtime.process._process_lifetime_owner`` and ``_scope_depth``
    are plain module globals (review v0.5.15 / blocker 2B), deliberately not
    re-derived from OS ground truth the way the actual signal handlers are.
    A test that calls ``install_signal_handlers()`` (or drives a CLI/MCP main
    in-process) would otherwise permanently flip ``_process_lifetime_owner``
    for the rest of the pytest process, silently turning every later
    ``signal_handler_scope()`` call into a no-op regardless of what OS-level
    handlers that later test itself restores. This is a cheap attribute
    save/restore, matching the existing convention in this suite of tests
    manually restoring real signal handlers around themselves.
    """
    import phasesweep.runtime.process as process

    prior_owner = process._process_lifetime_owner
    prior_depth = process._scope_depth
    try:
        yield
    finally:
        process._process_lifetime_owner = prior_owner
        process._scope_depth = prior_depth


def copy_fake_train(tmp_path: Path) -> Path:
    trainer = tmp_path / "examples" / "fake_train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "src" / "phasesweep" / "examples" / "fake_train.py", trainer)
    return trainer


def make_experiment(
    *,
    experiment: str = "t",
    workdir: str | Path | None = None,
    storage: str | None = None,
    trial_command: str = "echo {overrides}",
    override_format: str = "argparse",
    metric: Metric | None = None,
    constraints: list[Constraint] | None = None,
    phases: list[Phase] | None = None,
    env: dict[str, str] | None = None,
    execution: ExecutionContext | None = None,
    provenance: dict[str, str] | None = None,
    **phase_overrides: Any,
) -> Experiment:
    """Build a minimal valid Experiment for testing.

    If ``phases`` is not given, a single phase named ``"p"`` is created with
    ``search_space={"x": IntParam(0..10)}``. Extra ``**phase_overrides`` are
    forwarded to that default phase (e.g. ``n_trials=4``, ``gpu_ids=[0]``).
    With persistent ``storage`` and no caller-supplied sampler, that default
    phase uses ``Sampler(type="random", seed=0)`` so it satisfies the
    persistent-storage sampler policy.
    """
    if phases is None:
        base: dict[str, Any] = dict(
            name="p",
            n_trials=2,
            search_space={"x": IntParam(type="int", low=0, high=10)},
        )
        base.update(phase_overrides)
        if storage is not None and "sampler" not in base:
            # Persistent storage rejects an unseeded stochastic sampler and
            # requires acknowledge_nonresumable for tpe/cmaes, so the default
            # tpe sampler cannot be the fixture default here. Seeded random is
            # stochastic, reproducible, and resumable: it satisfies the policy
            # without asking every caller to acknowledge anything.
            base["sampler"] = Sampler(type="random", seed=0)
        phases = [Phase(**base)]  # type: ignore[arg-type]

    kwargs: dict[str, Any] = dict(
        experiment=experiment,
        trial_command=trial_command,
        override_format=override_format,
        metric=metric
        or Metric(
            extractor=LogRegexExtractor(
                type="log_regex",
                pattern=r"x=(?P<value>[0-9.eE+-]+)",
            )
        ),
        phases=phases,
    )
    if workdir is not None:
        kwargs["workdir"] = str(workdir)
    if storage is not None:
        kwargs["storage"] = storage
        kwargs["provenance"] = provenance or {"revision": "test-fixture-v1"}
    elif provenance is not None:
        kwargs["provenance"] = provenance
    if constraints is not None:
        kwargs["constraints"] = constraints
    if env is not None:
        kwargs["env"] = env
    if execution is not None:
        kwargs["execution"] = execution

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
    """Drop a minimal trainer that writes and logs a constant objective.

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
        print("x=0.5")
        """,
    )


def assert_published_winner_evidence_local(experiment_dir: Path) -> None:
    """Assert every published winner's source generation exists in this same tree.

    The invariant a second, divergent artifact tree breaks: a published
    winner's ``winner_source.generation_id`` names the immutable generation
    namespace holding the evidence behind that number, so it must resolve
    *inside the tree the reader is reading*. When one study is allowed to back
    two roots, both report ``publication_integrity: ok`` while the winner in
    one of them cites a generation that only exists in the other (re-review
    v0.5.19 / blocker B1).

    :param Path experiment_dir: Experiment artifact namespace to check, i.e.
        ``<workdir>/<experiment>``.
    """
    pointer = experiment_dir / "last_successful_generation.yaml"
    assert pointer.is_file(), f"{experiment_dir} has never published"
    published = yaml.safe_load(pointer.read_text())["generation_id"]
    generations = experiment_dir / "generations"
    assert (generations / published).is_dir(), (
        f"published generation {published!r} is missing from {generations}"
    )
    winners = sorted((generations / published / "phases").glob("*/winner.yaml"))
    assert winners, f"published generation {published!r} exposes no winner"
    for winner_path in winners:
        source = yaml.safe_load(winner_path.read_text())["winner_source"]
        source_generation = source["generation_id"]
        assert (generations / source_generation).is_dir(), (
            f"{winner_path} cites generation {source_generation!r}, which does not exist "
            f"under {generations}: its evidence lives in another artifact tree"
        )


def make_trial_context(
    tmp_path: Path,
    *,
    experiment: str = "t",
    phase: str = "p",
    trial_id: int = 0,
    run_name: str | None = None,
) -> TrialContext:
    """Build a minimal extractor trial context for unit tests."""
    return TrialContext(
        experiment=experiment,
        phase=phase,
        trial_id=trial_id,
        generation_id="generation-test",
        attempt_id="attempt-test",
        overrides_sha256="a" * 64,
        trial_dir=tmp_path,
        run_name=run_name or f"{experiment}-{phase}-{trial_id}-attempt-test",
        return_code=0,
        duration_seconds=0.0,
    )
