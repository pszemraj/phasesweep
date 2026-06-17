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
from phasesweep.engine.state import Winner, _load_winner, _phase_dir, _save_winner, _winner_path
from tests.conftest import make_experiment, write_constant_trainer, write_trainer, write_yaml


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
    from tests.conftest import copy_fake_train

    trainer = copy_fake_train(tmp_path)

    db_path = tmp_path / "phases.db"
    base_yaml = f"""
experiment: fp_test
storage: sqlite:///{db_path}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
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


def test_fingerprint_includes_semantic_fields_but_ignores_run_control() -> None:
    """Top-up and throughput knobs are ignored; trainer semantics still hash in."""

    def env_pair() -> tuple[Experiment, Experiment]:
        base_phase = Phase(
            name="p", n_trials=4, search_space={"x": IntParam(type="int", low=0, high=10)}
        )
        return (
            Experiment(
                experiment="t",
                trial_command="echo {overrides}",
                metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
                phases=[base_phase],
                env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            ),
            Experiment(
                experiment="t",
                trial_command="echo {overrides}",
                metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
                phases=[base_phase],
                env={"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
            ),
        )

    cases = [
        (
            "n_trials_top_up",
            lambda: (make_experiment(n_trials=4), make_experiment(n_trials=64)),
            True,
        ),
        (
            "throughput_run_control",
            lambda: (
                make_experiment(n_jobs=1),
                make_experiment(
                    n_jobs=4,
                    allow_no_gpu_isolation=True,
                    allow_incomplete_on_timeout=True,
                ),
            ),
            True,
        ),
        ("env", env_pair, False),
        (
            "search_space",
            lambda: (
                make_experiment(search_space={"x": IntParam(type="int", low=0, high=10)}),
                make_experiment(search_space={"x": IntParam(type="int", low=0, high=20)}),
            ),
            False,
        ),
        (
            "timeout_seconds_per_trial",
            lambda: (
                make_experiment(timeout_seconds_per_trial=60.0),
                make_experiment(timeout_seconds_per_trial=3600.0),
            ),
            False,
        ),
    ]

    for case, build_pair, expected_equal in cases:
        exp_a, exp_b = build_pair()
        fp_a = _phase_fingerprint(exp_a, exp_a.phases[0], {})
        fp_b = _phase_fingerprint(exp_b, exp_b.phases[0], {})
        if expected_equal:
            assert fp_a == fp_b, case
        else:
            assert fp_a != fp_b, case


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


def test_version_sources_and_fingerprint_payload_agree() -> None:
    """``phasesweep.__version__`` matches the installed package metadata.

    Source of truth is package metadata generated by setuptools-scm from SCM
    state. ``__version__`` reads that metadata rather than a generated source
    file.
    """
    from importlib.metadata import version as pkg_version

    from phasesweep.engine.guards import _phase_semantic_payload

    assert __version__ == pkg_version("phasesweep")

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


def test_from_phase_refuses_winner_yaml_with_invalid_fingerprint(tmp_path: Path) -> None:
    """Skipped winners need a matching fingerprint, not just plausible YAML."""

    def strip_fingerprint(data: dict) -> str:
        del data["phase_fingerprint"]
        return "no phase_fingerprint"

    def tamper_fingerprint(data: dict) -> str:
        data["phase_fingerprint"] = "0" * 64
        return "different phase config"

    for case, mutate in (("missing", strip_fingerprint), ("tampered", tamper_fingerprint)):
        case_dir = tmp_path / case
        case_dir.mkdir()
        trainer = write_constant_trainer(case_dir)
        exp = _two_phase_experiment(workdir=case_dir / "runs", trainer=trainer)
        run_experiment(exp)

        arch_winner_path = _winner_path(exp, "arch")
        data = yaml.safe_load(arch_winner_path.read_text())
        match = mutate(data)
        arch_winner_path.write_text(yaml.safe_dump(data, sort_keys=False))

        shutil.rmtree(_phase_dir(exp, "lr"))

        with pytest.raises(RuntimeError, match=match):
            run_experiment(exp, from_phase="lr")


def test_load_winner_normalizes_malformed_yaml_error(tmp_path: Path) -> None:
    exp = make_experiment(workdir=tmp_path / "runs")
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"trial_number": 0, "metric": {"objective":')

    with pytest.raises(RuntimeError, match="invalid or incomplete"):
        _load_winner(exp, exp.phases[0], {})


def test_load_winner_normalizes_incomplete_mapping_error(tmp_path: Path) -> None:
    exp = make_experiment(workdir=tmp_path / "runs")
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "phase": "p",
                "phase_fingerprint": _phase_fingerprint(exp, exp.phases[0], {}),
                "completion": {"incomplete": False},
            },
            sort_keys=False,
        )
    )

    with pytest.raises(RuntimeError, match="invalid or incomplete"):
        _load_winner(exp, exp.phases[0], {})


def test_save_winner_replace_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp = make_experiment(workdir=tmp_path / "runs")
    phase = exp.phases[0]
    fingerprint = _phase_fingerprint(exp, phase, {})
    original = Winner(
        trial_number=0,
        params={"x": 0},
        effective_overrides={"x": 0},
        metric=1.0,
        phase_fingerprint=fingerprint,
    )
    replacement = Winner(
        trial_number=1,
        params={"x": 1},
        effective_overrides={"x": 1},
        metric=0.5,
        phase_fingerprint=fingerprint,
    )

    _save_winner(exp, phase.name, original)
    path = _winner_path(exp, phase.name)
    before = path.read_text()

    def fail_replace(src: Path | str, dst: Path | str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("phasesweep.engine.state.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _save_winner(exp, phase.name, replacement)

    assert path.read_text() == before
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


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
