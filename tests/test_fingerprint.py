"""Phase fingerprinting and --from-phase verification. Run-control fields are excluded so top-ups stay compatible; semantic fields are hashed in so changes invalidate the study."""

from __future__ import annotations

import shutil
from pathlib import Path

import optuna
import pytest
import yaml

from phasesweep import __version__, load_experiment, run_experiment
from phasesweep.config import (
    CategoricalParam,
    Experiment,
    FloatParam,
    IntParam,
    JsonExtractor,
    Metric,
    Phase,
)
from phasesweep.engine.guards import _phase_fingerprint
from phasesweep.engine.state import Winner, _phase_dir, _winner_path
from tests.conftest import REPO, make_experiment, write_constant_trainer, write_trainer, write_yaml


def _two_phase_experiment(
    *,
    workdir: Path,
    trainer: Path,
    arch_low: int = 1,
    arch_high: int = 4,
    arch_n_trials: int = 1,
    arch_fixed_overrides: dict | None = None,
    storage: str | None = None,
) -> Experiment:
    """Build an arch -> lr two-phase experiment for from-phase tests.

    Each phase has a search space, so v0.5.7 trial_command validation needs
    ``{overrides}``. The trainer writes a constant ``r.json`` so trials
    actually complete and the experiment produces a saved winner.
    """
    phases = [
        Phase(
            name="arch",
            n_trials=arch_n_trials,
            search_space={"depth": IntParam(type="int", low=arch_low, high=arch_high)},
            fixed_overrides=arch_fixed_overrides or {},
        ),
        Phase(
            name="lr",
            inherits=["arch"],
            n_trials=1,
            search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
        ),
    ]
    return make_experiment(
        workdir=str(workdir),
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        phases=phases,
    )


def test_fingerprint_mismatch_raises(tmp_path):
    """Changing phase config and re-running should fail, not silently mix results."""
    examples_dst = tmp_path / "examples"
    examples_dst.mkdir(parents=True)
    shutil.copy(REPO / "examples" / "fake_train.py", examples_dst / "fake_train.py")

    db_path = tmp_path / "phases.db"
    base_yaml = f"""
experiment: fp_test
storage: sqlite:///{db_path}
workdir: {tmp_path / "runs"}
trial_command: "python {examples_dst / "fake_train.py"} --out {{trial_dir}}/result.json {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
phases:
  - name: a
    n_trials: 2
    allow_partial_grid: true
    sampler: {{ type: grid }}
    search_space:
      n_layers: {{ type: categorical, choices: [4, 8] }}
"""
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(base_yaml)
    exp = load_experiment(yaml_path)
    run_experiment(exp)

    # Now change the search space and re-run — should fail.
    changed_yaml = base_yaml.replace("choices: [4, 8]", "choices: [4, 8, 12]")
    yaml_path.write_text(changed_yaml)
    exp2 = load_experiment(yaml_path)
    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(exp2)


def test_fingerprint_changes_when_parent_winner_changes():
    """A child's fingerprint must change if a parent winner changes, even if child config is identical."""
    exp = Experiment(
        experiment="t",
        trial_command="echo {overrides}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        phases=[
            Phase(
                name="arch",
                n_trials=1,
                search_space={"n_layers": CategoricalParam(type="categorical", choices=[4, 8])},
            ),
            Phase(
                name="lr",
                inherits=["arch"],
                n_trials=1,
                search_space={"lr": IntParam(type="int", low=1, high=10)},
            ),
        ],
    )
    child = exp.phases[1]

    parent_a = Winner(
        trial_number=0, params={"n_layers": 4}, effective_overrides={"n_layers": 4}, metric=0.5
    )
    parent_b = Winner(
        trial_number=1, params={"n_layers": 8}, effective_overrides={"n_layers": 8}, metric=0.4
    )

    fp_a = _phase_fingerprint(exp, child, {"arch": parent_a})
    fp_b = _phase_fingerprint(exp, child, {"arch": parent_b})
    assert fp_a != fp_b, (
        "Fingerprint must encode inherited winner; otherwise --from-phase replay "
        "with a different parent winner silently mixes incompatible trials."
    )


def test_from_phase_dry_run_placeholder_includes_inherited(tmp_path):
    """--from-phase dry-run with missing winner.yaml falls back to placeholder.

    The placeholder must still compose inherited overrides for its descendants.
    """
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json, path: r.json, key: x }}
        phases:
          - name: arch
            fixed_overrides:
              model_family: llama
            n_trials: 1
            search_space:
              n_layers: {{ type: categorical, choices: [4, 8] }}
          - name: lr
            inherits: [arch]
            n_trials: 1
            search_space:
              lr: {{ type: float, low: 1e-5, high: 1e-3, log: true }}
        """,
    )
    exp = load_experiment(p)
    # No winner files on disk; dry-run from phase 'lr' must synthesize a placeholder
    # for arch that still carries its fixed_overrides.
    winners = run_experiment(exp, from_phase="lr", dry_run=True)
    assert winners["arch"].effective_overrides.get("model_family") == "llama"
    assert "n_layers" in winners["arch"].effective_overrides


def test_fingerprint_unchanged_when_n_trials_increased() -> None:
    """Bumping n_trials must not invalidate the study (top-up workflow)."""
    a = make_experiment(n_trials=4)
    b = make_experiment(n_trials=64)
    fp_a = _phase_fingerprint(a, a.phases[0], {})
    fp_b = _phase_fingerprint(b, b.phases[0], {})
    assert fp_a == fp_b, (
        "Fingerprint must ignore n_trials so users can top up a study; "
        "v0.5.1 hashed the whole phase model_dump and broke this."
    )


def test_fingerprint_unchanged_when_run_control_changes() -> None:
    """Throughput and timeout-acceptance knobs must not invalidate the study."""
    a = make_experiment(n_jobs=1)
    b = make_experiment(
        n_jobs=4,
        allow_no_gpu_isolation=True,
        allow_incomplete_on_timeout=True,
    )
    fp_a = _phase_fingerprint(a, a.phases[0], {})
    fp_b = _phase_fingerprint(b, b.phases[0], {})
    assert fp_a == fp_b


def test_fingerprint_changes_when_env_changes() -> None:
    """experiment.env affects training behavior; fingerprint must reflect it."""
    base_phase = Phase(
        name="p", n_trials=4, search_space={"x": IntParam(type="int", low=0, high=10)}
    )
    e1 = Experiment(
        experiment="t",
        trial_command="echo {overrides}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        phases=[base_phase],
        env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )
    e2 = Experiment(
        experiment="t",
        trial_command="echo {overrides}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        phases=[base_phase],
        env={"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
    )
    fp1 = _phase_fingerprint(e1, e1.phases[0], {})
    fp2 = _phase_fingerprint(e2, e2.phases[0], {})
    assert fp1 != fp2, "Changing env vars must invalidate fingerprint"


def test_fingerprint_changes_when_search_space_changes() -> None:
    """Sanity: things that DO change trial meaning still flip the fingerprint."""
    a = make_experiment(search_space={"x": IntParam(type="int", low=0, high=10)})
    b = make_experiment(search_space={"x": IntParam(type="int", low=0, high=20)})
    fp_a = _phase_fingerprint(a, a.phases[0], {})
    fp_b = _phase_fingerprint(b, b.phases[0], {})
    assert fp_a != fp_b


def test_n_trials_top_up_preserves_existing_trials(tmp_path: Path) -> None:
    """End-to-end: run with n_trials=2, then n_trials=4 -> 4 total trials in same study."""
    trainer = write_trainer(
        tmp_path,
        """
        import json, argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('--out', required=True)
        args, _ = ap.parse_known_args()
        with open(args.out, 'w') as f: json.dump({'eval_loss': 0.5}, f)
        """,
    )
    db = tmp_path / "phases.db"
    yaml_text = f"""
experiment: topup
storage: sqlite:///{db}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json, path: result.json, key: eval_loss }}
phases:
  - name: a
    n_trials: 2
    sampler: {{ type: random, seed: 0 }}
    search_space: {{ x: {{ type: int, low: 0, high: 10 }} }}
"""
    p = tmp_path / "exp.yaml"
    p.write_text(yaml_text)
    run_experiment(load_experiment(p))

    # Bump n_trials and re-run; this must not error on fingerprint.
    p.write_text(yaml_text.replace("n_trials: 2", "n_trials: 4"))
    run_experiment(load_experiment(p))

    study = optuna.load_study(study_name="topup::a", storage=f"sqlite:///{db}")
    finished = [t for t in study.get_trials() if t.state.is_finished()]
    assert len(finished) == 4, f"expected 4 trials after top-up, got {len(finished)}"


def test_version_sources_agree() -> None:
    """``phasesweep.__version__`` matches the installed package metadata.

    Source of truth is the static ``version`` field in ``pyproject.toml``;
    both ``__version__`` and ``importlib.metadata.version`` read from that
    same metadata, so this guards against a future refactor that drifts
    them apart.
    """
    from importlib.metadata import version as pkg_version

    assert __version__ == pkg_version("phasesweep")


def test_fingerprint_uses_package_version() -> None:
    """Fingerprint payload's `phasesweep_version` equals the package version."""
    from phasesweep.engine.guards import _phase_semantic_payload

    exp = make_experiment()
    payload = _phase_semantic_payload(exp, exp.phases[0], {})
    assert payload["phasesweep_version"] == __version__


def test_fingerprint_is_full_sha256(tmp_path: Path) -> None:
    """Full 64-hex SHA-256, no truncation (review v0.5.3)."""
    exp = make_experiment(workdir=tmp_path / "wd")
    fp = _phase_fingerprint(exp, exp.phases[0], {})
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_winner_yaml_contains_phase_fingerprint(tmp_path: Path) -> None:
    """Every saved winner carries the SHA-256 fingerprint of its producing
    phase config. ``--from-phase`` reuses winners only if this matches the
    re-computed fingerprint of the current YAML.
    """
    trainer = write_constant_trainer(tmp_path)
    exp = make_experiment(
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
    )
    run_experiment(exp)

    data = yaml.safe_load(_winner_path(exp, "p").read_text())
    assert "phase_fingerprint" in data
    assert isinstance(data["phase_fingerprint"], str)
    assert len(data["phase_fingerprint"]) == 64  # SHA-256 hex digest


def test_from_phase_rejects_stale_parent_winner_after_search_space_change(
    tmp_path: Path,
) -> None:
    """The reviewer's primary scenario: run fully, edit the parent's search
    space between runs, then ``--from-phase`` the child. The old winner is
    incompatible with the new parent config and must be refused.
    """
    trainer = write_constant_trainer(tmp_path)
    workdir = tmp_path / "runs"
    exp_v1 = _two_phase_experiment(workdir=workdir, trainer=trainer, arch_low=1, arch_high=4)
    run_experiment(exp_v1)

    exp_v2 = _two_phase_experiment(workdir=workdir, trainer=trainer, arch_low=12, arch_high=16)
    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(exp_v2, from_phase="lr")


def test_from_phase_rejects_stale_parent_winner_after_fixed_override_change(
    tmp_path: Path,
) -> None:
    """Same idea, different mutation: editing parent ``fixed_overrides``
    between runs must invalidate the skipped winner.
    """
    trainer = write_constant_trainer(tmp_path)
    workdir = tmp_path / "runs"
    exp_v1 = _two_phase_experiment(
        workdir=workdir, trainer=trainer, arch_fixed_overrides={"width": 64}
    )
    run_experiment(exp_v1)

    exp_v2 = _two_phase_experiment(
        workdir=workdir, trainer=trainer, arch_fixed_overrides={"width": 128}
    )
    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(exp_v2, from_phase="lr")


def test_from_phase_accepts_skipped_winner_when_only_n_trials_changed(
    tmp_path: Path,
) -> None:
    """``n_trials`` is a run-control field, excluded from the fingerprint
    so users can top up a study. A bumped ``n_trials`` on a *parent*
    phase must therefore not invalidate that phase's skipped winner.
    """
    trainer = write_constant_trainer(tmp_path)
    workdir = tmp_path / "runs"
    exp_v1 = _two_phase_experiment(workdir=workdir, trainer=trainer, arch_n_trials=1)
    winners1 = run_experiment(exp_v1)

    exp_v2 = _two_phase_experiment(workdir=workdir, trainer=trainer, arch_n_trials=5)
    winners2 = run_experiment(exp_v2, from_phase="lr")

    assert winners2["arch"].params == winners1["arch"].params
    assert "lr" in winners2


def test_from_phase_refuses_winner_yaml_with_no_fingerprint(tmp_path: Path) -> None:
    """A pre-v0.5.7 ``winner.yaml`` (or one a user hand-edited) has no
    fingerprint field. The reviewer asked for *loud* refusal here rather
    than silent reuse, because phasesweep cannot prove the file matches
    the current config.
    """
    trainer = write_constant_trainer(tmp_path)
    workdir = tmp_path / "runs"
    exp = _two_phase_experiment(workdir=workdir, trainer=trainer)
    run_experiment(exp)

    # Strip the fingerprint as if the file were produced by an older version.
    arch_winner_path = _winner_path(exp, "arch")
    data = yaml.safe_load(arch_winner_path.read_text())
    del data["phase_fingerprint"]
    arch_winner_path.write_text(yaml.safe_dump(data, sort_keys=False))

    # Drop the lr phase artifacts so we go through _run_phase rather than
    # short-circuiting on a current-version lr winner.
    shutil.rmtree(_phase_dir(exp, "lr"))

    with pytest.raises(RuntimeError, match="no phase_fingerprint"):
        run_experiment(exp, from_phase="lr")


def test_from_phase_refuses_winner_yaml_with_tampered_fingerprint(
    tmp_path: Path,
) -> None:
    """A hand-tampered fingerprint is also refused — the stored value is
    re-checked against the live recomputed fingerprint, not just inspected
    for presence.
    """
    trainer = write_constant_trainer(tmp_path)
    workdir = tmp_path / "runs"
    exp = _two_phase_experiment(workdir=workdir, trainer=trainer)
    run_experiment(exp)

    arch_winner_path = _winner_path(exp, "arch")
    data = yaml.safe_load(arch_winner_path.read_text())
    data["phase_fingerprint"] = "0" * 64  # plausible-looking but wrong
    arch_winner_path.write_text(yaml.safe_dump(data, sort_keys=False))

    shutil.rmtree(_phase_dir(exp, "lr"))

    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(exp, from_phase="lr")


def test_phase_comment_schema_and_fingerprint(tmp_path: Path) -> None:
    """``comment`` is optional, defaults to None, and is excluded from the
    fingerprint so editing it never invalidates a study.

    Pre-v0.5.9 this was three separate tests (accepted / default / excluded);
    fingerprint exclusion is the only contract that matters at runtime, the
    other two are surface-level Pydantic checks.
    """
    p_with = Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        comment="Why this phase exists.",
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )
    p_without = Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )
    assert p_with.comment == "Why this phase exists."
    assert p_without.comment is None

    def build(comment: str | None) -> Experiment:
        return Experiment(
            experiment="t",
            workdir=str(tmp_path / "wd"),
            trial_command="echo {overrides}",
            metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
            phases=[
                Phase(  # type: ignore[arg-type]
                    name="p",
                    n_trials=4,
                    comment=comment,
                    search_space={"x": IntParam(type="int", low=0, high=10)},
                )
            ],
        )

    fp_a = _phase_fingerprint(*(lambda e: (e, e.phases[0], {}))(build("First version")))
    fp_b = _phase_fingerprint(*(lambda e: (e, e.phases[0], {}))(build("Reworded later")))
    fp_c = _phase_fingerprint(*(lambda e: (e, e.phases[0], {}))(build(None)))
    assert fp_a == fp_b == fp_c


def test_timeout_seconds_per_trial_invalidates_fingerprint() -> None:
    """A different per-trial timeout means a different observation budget,
    therefore a different fingerprint.

    The sibling sanity checks (``n_trials`` and ``n_jobs`` do *not* invalidate
    the fingerprint) live earlier in this file — they pin the run-control
    exclusion. This test only exists to guard the contract that timeout is
    *semantic*, not throughput: a 60s vs 3600s budget changes which trials
    FAIL vs COMPLETE, which changes the observation distribution.
    """
    exp_short = make_experiment(timeout_seconds_per_trial=60.0)
    exp_long = make_experiment(timeout_seconds_per_trial=3600.0)

    fp_short = _phase_fingerprint(exp_short, exp_short.phases[0], {})
    fp_long = _phase_fingerprint(exp_long, exp_long.phases[0], {})

    assert fp_short != fp_long
