"""Phase fingerprinting and --from-phase verification. Run-control fields are excluded so top-ups stay compatible; semantic fields are hashed in so changes invalidate the study."""

from __future__ import annotations

import importlib
import json
import logging
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import yaml

from phasesweep import __version__, load_experiment, run_experiment
from phasesweep.config import (
    CategoricalParam,
    ExecutionContext,
    Experiment,
    FloatParam,
    IntParam,
    JsonEqualsGate,
    LogRegexExtractor,
    Metric,
    Phase,
    Promotion,
    Sampler,
    StudySpec,
    Suite,
    SuiteDefaults,
    WandbExtractor,
)
from phasesweep.engine import (
    ArtifactRootConflictError,
    ArtifactRootRebindError,
    LegacyArtifactRootMigrationRequiredError,
    NoFeasibleTrialError,
    PublishedStudyMissingError,
    RunRequestError,
    SamplerContinuationUnsupportedError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRegressionError,
    WinnerIntegrityError,
    read_status,
    read_winner,
    read_winners,
)
from phasesweep.engine.artifact_roots import _validate_artifact_root_binding
from phasesweep.engine.artifacts import _load_winner, _save_winner
from phasesweep.engine.attempts import _register_active_attempt
from phasesweep.engine.fingerprints import (
    FINGERPRINT_SCHEMA_VERSION,
    _phase_fingerprint,
)
from phasesweep.engine.optuna import _sqlite_study_exists
from phasesweep.engine.paths import (
    _artifact_root_binding_path,
    _attempts_dir,
    _experiment_dir,
    _generation_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_path,
    _phase_dir,
    _summary_path,
    _winner_path,
)
from phasesweep.engine.publication import (
    _last_successful_generation_id,
    _published_winner_path,
)
from phasesweep.engine.relocation import (
    _artifact_root_rebind_entries,
    _ArtifactRootRebindPlan,
    _validate_artifact_root_binding_for_rebind,
    _validate_artifact_root_destination,
)
from phasesweep.engine.resume import _reject_bound_descendant_topups
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    ATTEMPT_ID_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_TARGET_ATTR,
    Winner,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError, _environment_identity
from phasesweep.runtime.files import (
    atomic_text_writer,
    file_url_path,
    sqlite_database_path,
    storage_backend,
)
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import (
    assert_published_winner_evidence_local,
    drop_artifact_root_binding,
    make_experiment,
    patch_path_method_failure,
    write_constant_trainer,
    write_trainer,
    write_yaml,
)


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
            # Seeded random keeps this helper valid with persistent storage,
            # which rejects an unseeded stochastic sampler.
            sampler=Sampler(type="random", seed=0),
            search_space={"depth": IntParam(type="int", low=arch_low, high=arch_high)},
            fixed_overrides=arch_fixed_overrides or {},
        ),
        Phase(
            name="lr",
            inherits=["arch"],
            n_trials=1,
            sampler=Sampler(type="random", seed=1),
            search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
        ),
    ]
    return make_experiment(
        workdir=str(workdir),
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        phases=phases,
    )


def _promotion_chain_experiment(
    *,
    workdir: Path,
    trainer: Path,
    storage: str,
    on_fail: str,
    arch_n_trials: int = 1,
) -> Experiment:
    """Build an arch --promotion baseline--> mid experiment with no inheritance edge.

    ``mid`` depends on ``arch`` only through ``promotion.min_delta_vs``, which
    config validation requires to name a *prior* phase rather than an inherited
    one. The constant trainer makes the delta exactly zero, so the first run
    promotes (``improvement >= min_delta``) and both phases publish a winner.

    :param Path workdir: Artifact root for the experiment.
    :param Path trainer: Trainer script invoked by every trial.
    :param str storage: Optuna storage URL; persistent so studies outlive a run.
    :param str on_fail: Promotion ``on_fail`` policy for ``mid``.
    :param int arch_n_trials: Trial target for ``arch``; raise it to request a top-up.
    :return Experiment: Two-phase experiment linked by a promotion baseline.
    """
    return make_experiment(
        workdir=str(workdir),
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        phases=[
            Phase(
                name="arch",
                n_trials=arch_n_trials,
                sampler=Sampler(type="random", seed=0),
                search_space={"depth": IntParam(type="int", low=1, high=4)},
            ),
            Phase(
                name="mid",
                n_trials=1,
                sampler=Sampler(type="random", seed=1),
                search_space={},
                fixed_overrides={"width": 8},
                promotion=Promotion(
                    min_delta_vs="arch",
                    min_delta=0.0,
                    requires_gates=False,
                    on_fail=on_fail,  # type: ignore[arg-type]
                ),
            ),
        ],
    )


def _fake_top_up_study(*, completed: int = 0, fingerprint: str | None = None) -> SimpleNamespace:
    """Build a stand-in study exposing only what the top-up guard reads.

    :param int completed: Number of terminal trials the study reports.
    :param str | None fingerprint: Bound phase fingerprint, or ``None`` for a
        study that has not been bound to a published winner yet.
    :return SimpleNamespace: Object with the ``user_attrs``/``get_trials``
        surface :func:`_reject_bound_descendant_topups` uses.
    """
    trials = [SimpleNamespace(state=optuna.trial.TrialState.COMPLETE) for _ in range(completed)]
    return SimpleNamespace(
        user_attrs={} if fingerprint is None else {"phasesweep_fingerprint": fingerprint},
        get_trials=lambda *, deepcopy: trials,
    )


def test_fingerprint_mismatch_raises(tmp_path):
    """Changing phase config and re-running should fail, not silently mix results."""
    from tests.conftest import copy_fake_train

    trainer = copy_fake_train(tmp_path)

    db_path = tmp_path / "phases.db"
    base_yaml = f"""
experiment: fp_test
storage: sqlite:///{db_path}
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
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


def test_persistent_storage_requires_declared_provenance(tmp_path: Path) -> None:
    payload = make_experiment().model_dump()
    payload["storage"] = f"sqlite:///{tmp_path / 'studies.db'}"

    with pytest.raises(ValueError, match="Persistent storage requires.*provenance"):
        Experiment.model_validate(payload)


@pytest.mark.parametrize("storage", [":memory:", "sqlite://", "sqlite:///:memory:"])
def test_in_memory_storage_does_not_require_provenance(storage: str) -> None:
    payload = make_experiment().model_dump()
    payload["storage"] = storage

    experiment = Experiment.model_validate(payload)

    assert experiment.provenance == {}


def test_late_child_fingerprint_failure_preserves_last_successful_results(
    tmp_path: Path,
) -> None:
    """A child mismatch after phase one cannot publish a mixed result view."""
    trainer = write_constant_trainer(tmp_path)
    experiment = _two_phase_experiment(
        workdir=tmp_path / "runs",
        trainer=trainer,
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
    )
    run_experiment(experiment)
    protected_paths = [
        _summary_path(experiment),
        _last_successful_generation_path(experiment),
        *(_winner_path(experiment, phase.name) for phase in experiment.phases),
    ]
    before = {path: path.read_bytes() for path in protected_paths}
    changed_child = experiment.phases[1].model_copy(
        update={"search_space": {"lr": IntParam(type="int", low=1, high=5)}}
    )
    changed = experiment.model_copy(update={"phases": [experiment.phases[0], changed_child]})

    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(changed)

    assert {path: path.read_bytes() for path in protected_paths} == before
    current = yaml.safe_load(_generation_path(experiment).read_text())
    assert current["state"] == "failed"
    assert yaml.safe_load(
        _last_successful_generation_path(experiment).read_text()
    ) == yaml.safe_load(before[_last_successful_generation_path(experiment)])


def test_interrupted_first_publication_still_publishes_and_reads_resolve_correctly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A projection failure during the very first publication is diagnostic-only.

    Legacy compatibility projections (root ``winner.yaml`` etc.) are a
    best-effort, post-commit cache (review v0.5.15 / blocker 3): a failure
    partway through them must not fail the run or block the pointer commit,
    even on a generation's first-ever publication. Reads must still resolve
    correctly via the generation-scoped artifacts regardless of which legacy
    copies did or didn't complete -- there is no "partial projection" for a
    read to be exposed to.
    """
    trainer = write_constant_trainer(tmp_path)
    experiment = _two_phase_experiment(workdir=tmp_path / "runs", trainer=trainer)
    generation_ops = importlib.import_module("phasesweep.engine.generation")
    real_copy = generation_ops._copy_yaml_projection
    copied = 0

    def interrupt_after_first_projection(source: Path, destination: Path) -> None:
        nonlocal copied
        if copied == 1:
            raise OSError("publication interrupted")
        real_copy(source, destination)
        copied += 1

    monkeypatch.setattr(generation_ops, "_copy_yaml_projection", interrupt_after_first_projection)

    winners = run_experiment(experiment)

    assert set(winners) == {"arch", "lr"}
    assert _last_successful_generation_path(experiment).is_file()
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    # The first (arch) legacy copy completed before the injected failure; the
    # second one (lr, or the summary) did not. Either way, reads resolve via
    # the generation-scoped artifact once any generation has published.
    assert {view.phase for view in read_winners(experiment)} == {"arch", "lr"}
    assert _load_winner(experiment, experiment.phases[0], {}) is not None


def test_fingerprint_changes_when_parent_winner_changes():
    """A child's fingerprint must change if a parent winner changes, even if child config is identical."""
    exp = Experiment(
        experiment="t",
        trial_command="echo {overrides}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
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
        override_format: argparse
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json_envelope, path: r.json, objective_name: x, split: test, policy: test }}
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
    lr_eff = winners["lr"].effective_overrides
    assert lr_eff["model_family"] == "llama"
    assert "n_layers" in lr_eff
    assert "lr" in lr_eff


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
                override_format="argparse",
                metric=Metric(
                    extractor=LogRegexExtractor(
                        type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"
                    )
                ),
                phases=[base_phase],
                env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            ),
            Experiment(
                experiment="t",
                trial_command="echo {overrides}",
                override_format="argparse",
                metric=Metric(
                    extractor=LogRegexExtractor(
                        type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"
                    )
                ),
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
                    gpu_devices=["GPU-deadbeef"],
                    allow_no_gpu_isolation=True,
                    allow_incomplete_on_timeout=True,
                ),
            ),
            True,
        ),
        ("env", env_pair, False),
        (
            "provenance",
            lambda: (
                make_experiment(provenance={"revision": "trainer-v1"}),
                make_experiment(provenance={"revision": "trainer-v2"}),
            ),
            False,
        ),
        (
            "trainer_config",
            lambda: (
                make_experiment(
                    trial_command="echo {config_path}",
                    override_format="yaml_file",
                    trainer_config={"model": {"depth": 4}},
                ),
                make_experiment(
                    trial_command="echo {config_path}",
                    override_format="yaml_file",
                    trainer_config={"model": {"depth": 8}},
                ),
            ),
            False,
        ),
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
        (
            "gpu_policy",
            lambda: (
                make_experiment(gpu_policy="single_per_trial"),
                make_experiment(gpu_policy="whole_node", gpu_ids=[0]),
            ),
            False,
        ),
        # Under whole_node the device-set SIZE is the trainer's world size —
        # semantic — while its spelling stays run-control, and under
        # single_per_trial the whole list stays run-control (review v0.5.17
        # gap hunt).
        (
            "whole_node_device_count",
            lambda: (
                make_experiment(gpu_policy="whole_node", gpu_ids=[0]),
                make_experiment(gpu_policy="whole_node", gpu_ids=[0, 1, 2, 3]),
            ),
            False,
        ),
        (
            "whole_node_device_spelling",
            lambda: (
                make_experiment(gpu_policy="whole_node", gpu_ids=[0, 1]),
                make_experiment(gpu_policy="whole_node", gpu_devices=["GPU-a", "GPU-b"]),
            ),
            True,
        ),
        (
            "single_per_trial_gpu_ids_are_run_control",
            lambda: (
                make_experiment(gpu_ids=[0]),
                make_experiment(gpu_ids=[0, 1, 2, 3]),
            ),
            True,
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


def test_execution_context_is_semantic_in_experiment_and_phase_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trainer's cwd and env-inheritance contract are semantic inputs.

    Both were unbound implicit inputs before review v0.5.17 / blocker 4: one
    study could mix trials launched from different directories or with
    different credentials visible. Both fingerprints must move when either
    changes, and must not move when the operator merely spells out the
    default block.
    """
    from phasesweep.engine.fingerprints import (
        _experiment_semantic_fingerprint,
        _phase_semantic_payload,
    )

    invocation_a = tmp_path / "invocation-a"
    invocation_b = tmp_path / "invocation-b"
    invocation_a.mkdir()
    invocation_b.mkdir()
    monkeypatch.chdir(invocation_a)

    variants = {
        "unbound": make_experiment(),
        "explicit_default": make_experiment(execution=ExecutionContext()),
        "cwd": make_experiment(execution=ExecutionContext(cwd=str(tmp_path))),
        "inherit_none": make_experiment(execution=ExecutionContext(inherit_env="none")),
        "inherit_names": make_experiment(execution=ExecutionContext(inherit_env=["HF_TOKEN"])),
    }
    experiment_fps = {name: _experiment_semantic_fingerprint(exp) for name, exp in variants.items()}
    phase_fps = {name: _phase_fingerprint(exp, exp.phases[0], {}) for name, exp in variants.items()}

    # Writing the default block out longhand is not a semantic edit.
    assert experiment_fps["unbound"] == experiment_fps["explicit_default"]
    assert phase_fps["unbound"] == phase_fps["explicit_default"]

    monkeypatch.chdir(invocation_b)
    second_invocation = make_experiment()
    assert _experiment_semantic_fingerprint(second_invocation) != experiment_fps["unbound"]
    assert (
        _phase_fingerprint(second_invocation, second_invocation.phases[0], {})
        != phase_fps["unbound"]
    )

    explicit_first = make_experiment(execution=ExecutionContext(cwd=str(invocation_a)))
    assert _experiment_semantic_fingerprint(explicit_first) == experiment_fps["unbound"]
    assert _phase_fingerprint(explicit_first, explicit_first.phases[0], {}) == phase_fps["unbound"]

    distinct = ["unbound", "cwd", "inherit_none", "inherit_names"]
    assert len({experiment_fps[name] for name in distinct}) == len(distinct)
    assert len({phase_fps[name] for name in distinct}) == len(distinct)

    # The phase payload carries the same resolved contract as the experiment one.
    cwd_exp = variants["cwd"]
    payload = _phase_semantic_payload(cwd_exp, cwd_exp.phases[0], {})
    assert payload["execution"] == {"cwd": str(tmp_path.resolve()), "inherit_env": "all"}

    # Names are a set, not a sequence: reordering the list is not an edit.
    reordered = make_experiment(
        execution=ExecutionContext(inherit_env=["HF_TOKEN", "WANDB_API_KEY"])
    )
    ordered = make_experiment(execution=ExecutionContext(inherit_env=["WANDB_API_KEY", "HF_TOKEN"]))
    assert _experiment_semantic_fingerprint(reordered) == _experiment_semantic_fingerprint(ordered)
    assert _phase_fingerprint(reordered, reordered.phases[0], {}) == _phase_fingerprint(
        ordered, ordered.phases[0], {}
    )

    passthrough_a = make_experiment(
        execution=ExecutionContext(inherit_env="none", passthrough_env=["HF_TOKEN"])
    )
    passthrough_b = make_experiment(
        execution=ExecutionContext(inherit_env="none", passthrough_env=["WANDB_API_KEY"])
    )
    assert _phase_fingerprint(passthrough_a, passthrough_a.phases[0], {}) != _phase_fingerprint(
        passthrough_b, passthrough_b.phases[0], {}
    )


def _with_trial_target(experiment: Experiment, n_trials: int) -> Experiment:
    phase = experiment.phases[0].model_copy(update={"n_trials": n_trials})
    return experiment.model_copy(update={"phases": [phase]})


def _environment_cohort_experiment(
    tmp_path: Path,
    trainer: Path,
    execution: ExecutionContext,
) -> Experiment:
    return make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        execution=execution,
        n_trials=1,
    )


def test_changed_semantic_inherited_value_rejects_topup_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = _environment_cohort_experiment(
        tmp_path,
        trainer,
        ExecutionContext(inherit_env=["PHASESWEEP_DATASET_REV"]),
    )
    monkeypatch.setenv("PHASESWEEP_DATASET_REV", "revision-a")
    run_experiment(experiment)

    monkeypatch.setenv("PHASESWEEP_DATASET_REV", "revision-b")
    with pytest.raises(StudyFingerprintMismatchError, match="No trial was allocated"):
        run_experiment(_with_trial_target(experiment, 2))

    study = optuna.load_study(study_name="t::p", storage=experiment.storage)
    assert [trial.number for trial in study.get_trials(deepcopy=False)] == [0]


def test_identical_semantic_environment_resumes_persistent_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = _environment_cohort_experiment(
        tmp_path,
        trainer,
        ExecutionContext(inherit_env=["PHASESWEEP_DATASET_REV"]),
    )
    monkeypatch.setenv("PHASESWEEP_DATASET_REV", "revision-a")

    run_experiment(experiment)
    run_experiment(_with_trial_target(experiment, 2))

    study = optuna.load_study(study_name="t::p", storage=experiment.storage)
    assert [trial.number for trial in study.get_trials(deepcopy=False)] == [0, 1]


def test_passthrough_token_rotation_resumes_persistent_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = _environment_cohort_experiment(
        tmp_path,
        trainer,
        ExecutionContext(inherit_env="none", passthrough_env=["WANDB_API_KEY"]),
    )
    monkeypatch.setenv("WANDB_API_KEY", "first-secret")
    run_experiment(experiment)

    monkeypatch.setenv("WANDB_API_KEY", "rotated-secret")
    run_experiment(_with_trial_target(experiment, 2))

    study = optuna.load_study(study_name="t::p", storage=experiment.storage)
    assert [trial.number for trial in study.get_trials(deepcopy=False)] == [0, 1]


def test_populated_study_without_environment_identity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = _environment_cohort_experiment(
        tmp_path,
        trainer,
        ExecutionContext(inherit_env=["PHASESWEEP_DATASET_REV"]),
    )
    monkeypatch.setenv("PHASESWEEP_DATASET_REV", "revision-a")
    run_experiment(experiment)
    assert experiment.storage is not None
    with sqlite3.connect(experiment.storage.removeprefix("sqlite:///")) as connection:
        connection.execute(
            "DELETE FROM trial_user_attributes WHERE key = ?",
            (TRAINER_ENV_DIGEST_ATTR,),
        )

    with pytest.raises(StudySchemaMismatchError, match="cannot guess.*environment cohort"):
        run_experiment(_with_trial_target(experiment, 2))

    study = optuna.load_study(study_name="t::p", storage=experiment.storage)
    assert [trial.number for trial in study.get_trials(deepcopy=False)] == [0]


def test_wandb_base_endpoint_change_rejects_topup_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def poll_summary(**kwargs: object) -> dict[str, float]:
        calls.append(str(kwargs["base_url"]))
        return {"eval/loss": 0.5}

    monkeypatch.setattr("phasesweep.evidence.evaluation.poll_wandb_summary", poll_summary)
    trainer = write_trainer(tmp_path, "print('trainer complete')")

    def experiment_for(base_url: str, n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{tmp_path / 'studies.db'}",
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
            execution=ExecutionContext(inherit_env="none", passthrough_env=["WANDB_API_KEY"]),
            n_trials=n_trials,
            metric=Metric(
                extractor=WandbExtractor(
                    type="wandb",
                    base_url=base_url,
                    entity="team",
                    project="project",
                    metric_key="eval/loss",
                )
            ),
        )

    first = experiment_for("https://wandb-a.example.test", 1)
    run_experiment(first)
    assert calls == ["https://wandb-a.example.test"]

    second = experiment_for("https://wandb-b.example.test", 2)
    with pytest.raises(StudyFingerprintMismatchError, match="different phase config"):
        run_experiment(second)

    study = optuna.load_study(study_name="t::p", storage=first.storage)
    assert [trial.number for trial in study.get_trials(deepcopy=False)] == [0]
    assert calls == ["https://wandb-a.example.test"]


def test_suite_fingerprint_includes_effective_invocation_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A suite cannot compile the same relative trainer command from two code trees."""
    base = make_experiment()
    suite = Suite(
        suite="identity",
        defaults=SuiteDefaults(
            workdir=base.workdir,
            trial_command=base.trial_command,
            override_format=base.override_format,
            metric=base.metric,
        ),
        studies=[StudySpec(name="study", phases=base.phases)],
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    from phasesweep.engine.fingerprints import _suite_fingerprint

    monkeypatch.chdir(first)
    first_fingerprint = _suite_fingerprint(suite)
    monkeypatch.chdir(second)

    assert _suite_fingerprint(suite) != first_fingerprint


def test_acknowledge_nonresumable_is_run_control_not_semantics() -> None:
    """Toggling the acknowledgement must not move the phase fingerprint.

    The flag never changes what a trial samples or means — on persistent
    storage its legal value is fully determined by ``sampler.type`` — so
    acknowledging an existing study's sampler (review v0.5.18 / finding F7)
    must not invalidate that study.
    """
    from phasesweep.engine.fingerprints import _phase_semantic_payload

    plain = make_experiment(sampler=Sampler(type="tpe", seed=0))
    acknowledged = make_experiment(
        sampler=Sampler(type="tpe", seed=0, acknowledge_nonresumable=True)
    )
    assert _phase_fingerprint(plain, plain.phases[0], {}) == _phase_fingerprint(
        acknowledged, acknowledged.phases[0], {}
    )
    payload = _phase_semantic_payload(acknowledged, acknowledged.phases[0], {})
    assert "acknowledge_nonresumable" not in payload["phase"]["sampler"]
    # The semantic sampler fields still move the fingerprint.
    reseeded = make_experiment(sampler=Sampler(type="tpe", seed=1, acknowledge_nonresumable=True))
    assert _phase_fingerprint(acknowledged, acknowledged.phases[0], {}) != _phase_fingerprint(
        reseeded, reseeded.phases[0], {}
    )


def test_json_equals_gate_scalar_types_move_the_phase_fingerprint() -> None:
    """A json_equals gate value's *type* is semantic, so it must be in the digest.

    ``_json_equals`` compares with ``type(actual) is type(gate.value)``, so
    ``1``, ``"1"``, ``true``, and ``1.0`` accept four disjoint JSON documents.
    The fingerprint hashes ``model_dump(mode="json")`` through
    ``json.dumps(..., default=str)``, which preserves that distinction only for
    values strict JSON can hold — which is why ``JsonEqualsGate.value`` is
    restricted to a strict JSON-scalar union (PR #5 review / reviewer 2,
    blocker 4). Anything it now rejects (a YAML date, an int-keyed mapping)
    would have collapsed onto a colliding twin here.
    """

    def fingerprint_for(value: bool | int | float | str | None) -> str:
        experiment = make_experiment(
            gates=[JsonEqualsGate(type="json_equals", path="result.json", key="k", value=value)]
        )
        return _phase_fingerprint(experiment, experiment.phases[0], {})

    fingerprints = {
        "int": fingerprint_for(1),
        "str": fingerprint_for("1"),
        "bool": fingerprint_for(True),
        "float": fingerprint_for(1.0),
        "null": fingerprint_for(None),
    }

    assert len(set(fingerprints.values())) == len(fingerprints), fingerprints
    # The gate participates at all: dropping it is also an edit.
    ungated = make_experiment()
    assert _phase_fingerprint(ungated, ungated.phases[0], {}) not in set(fingerprints.values())


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
        print('eval_loss=0.5')
        """,
    )
    db = tmp_path / "phases.db"
    yaml_text = f"""
experiment: topup
storage: sqlite:///{db}
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: log_regex, pattern: 'eval_loss=(?P<value>[0-9.eE+-]+)' }}
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


@pytest.mark.parametrize(
    ("sampler", "search_space"),
    [
        # These cases exist to exercise the stateful samplers on persistent
        # storage, which requires the config-level non-resumable acknowledgement;
        # the acknowledgement does not weaken the runtime continuation guard.
        pytest.param(
            Sampler(type="tpe", seed=0, n_startup_trials=10, acknowledge_nonresumable=True),
            {"x": IntParam(type="int", low=0, high=10)},
            id="tpe",
        ),
        pytest.param(
            Sampler(type="cmaes", seed=0, acknowledge_nonresumable=True),
            {
                "x": FloatParam(type="float", low=0.0, high=1.0),
                "y": FloatParam(type="float", low=0.0, high=1.0),
            },
            id="cmaes",
        ),
    ],
)
def test_stateful_sampler_rejects_interrupted_resume_and_top_up(
    tmp_path: Path,
    sampler: Sampler,
    search_space: dict,
) -> None:
    """A partially complete TPE/CMA-ES study can be neither resumed nor topped up.

    Optuna storage does not persist process-local sampler state, so a fresh
    process would restart the seeded suggestion stream and could re-evaluate
    identical startup suggestions (review v0.5.14 / blocker 3).
    """
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)")
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    phase = Phase(
        name="p",
        n_trials=3,
        max_consecutive_failures=1,
        sampler=sampler,
        search_space=search_space,
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        phases=[phase],
    )
    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(experiment)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 3
    assert len(study.trials) == 1

    write_constant_trainer(tmp_path)
    with pytest.raises(SamplerContinuationUnsupportedError, match="interrupted at 1/3"):
        run_experiment(experiment)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert len(study.trials) == 1

    top_up = experiment.model_copy(update={"phases": [phase.model_copy(update={"n_trials": 4})]})
    with pytest.raises(
        SamplerContinuationUnsupportedError,
        match="process-local continuation state.*new experiment name",
    ):
        run_experiment(top_up)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert len(study.trials) == 1


@pytest.mark.parametrize(
    ("sampler", "search_space"),
    [
        # These cases exist to exercise the stateful samplers on persistent
        # storage, which requires the config-level non-resumable acknowledgement;
        # the acknowledgement does not weaken the runtime continuation guard.
        pytest.param(
            Sampler(type="tpe", seed=0, n_startup_trials=10, acknowledge_nonresumable=True),
            {"x": IntParam(type="int", low=0, high=10)},
            id="tpe",
        ),
        pytest.param(
            Sampler(type="cmaes", seed=0, acknowledge_nonresumable=True),
            {
                "x": FloatParam(type="float", low=0.0, high=1.0),
                "y": FloatParam(type="float", low=0.0, high=1.0),
            },
            id="cmaes",
        ),
    ],
)
def test_stateful_sampler_completed_target_reruns_as_noop(
    tmp_path: Path,
    sampler: Sampler,
    search_space: dict,
) -> None:
    """A stateful study that reached its accepted target republishes without new trials."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    phase = Phase(name="p", n_trials=2, sampler=sampler, search_space=search_space)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        phases=[phase],
    )
    first = run_experiment(experiment)
    rerun = run_experiment(experiment)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert len(study.trials) == 2
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 2
    assert rerun["p"].trial_number == first["p"].trial_number
    assert rerun["p"].metric == first["p"].metric


def test_persistent_trial_target_cannot_move_backward(tmp_path: Path) -> None:
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    phase = Phase(
        name="p",
        n_trials=3,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        phases=[phase],
    )
    winners = run_experiment(experiment)
    winner_before = _winner_path(experiment, "p").read_bytes()

    lowered = experiment.model_copy(update={"phases": [phase.model_copy(update={"n_trials": 1})]})
    with pytest.raises(TrialTargetRegressionError, match="accepted a target of 3"):
        run_experiment(lowered)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 3
    assert len(study.trials) == 3
    assert winners["p"].completion["finished_trials"] == 3
    assert _winner_path(experiment, "p").read_bytes() == winner_before


def test_upstream_top_up_is_rejected_before_bound_chain_mutation(tmp_path: Path) -> None:
    """A bound child makes an ancestor top-up a non-destructive preflight refusal."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = _two_phase_experiment(
        workdir=tmp_path / "runs",
        trainer=trainer,
        storage=storage,
        arch_n_trials=1,
    )
    run_experiment(experiment)
    protected_paths = [
        _summary_path(experiment),
        _last_successful_generation_path(experiment),
        *(_winner_path(experiment, phase.name) for phase in experiment.phases),
    ]
    before = {path: path.read_bytes() for path in protected_paths}
    parent_before = optuna.load_study(study_name="t::arch", storage=storage).trials
    topped_up_parent = experiment.phases[0].model_copy(update={"n_trials": 2})
    topped_up = experiment.model_copy(update={"phases": [topped_up_parent, experiment.phases[1]]})

    with pytest.raises(RuntimeError, match="dependent phase study.*new experiment name"):
        run_experiment(topped_up)

    parent_after = optuna.load_study(study_name="t::arch", storage=storage).trials
    assert len(parent_after) == len(parent_before) == 1
    assert {path: path.read_bytes() for path in protected_paths} == before


@pytest.mark.parametrize("on_fail", ["skip", "stop", "continue_baseline"])
def test_promotion_baseline_top_up_is_rejected_before_bound_chain_mutation(
    tmp_path: Path, on_fail: str
) -> None:
    """A bound promotion dependent makes an upstream top-up a preflight refusal.

    Unguarded, every ``on_fail`` policy mutated ``arch`` first and then failed
    differently against the new baseline winner: ``stop`` raised mid-run,
    ``continue_baseline`` republished the baseline's overrides under ``mid``
    (breaking any bound study inheriting it), and ``skip`` - the worst - still
    *succeeded*, publishing a generation containing only ``arch`` and advancing
    the last-success pointer past the previously published ``mid`` winner
    (PR #5 review / reviewer 2 pass 2, blocker 2). The refusal must land before
    any of that, and it is unconditional - the guard refuses the top-up rather
    than predicting whether the new baseline would flip the decision - so a
    constant trainer pins it.
    """
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = _promotion_chain_experiment(
        workdir=tmp_path / "runs",
        trainer=trainer,
        storage=storage,
        on_fail=on_fail,
    )
    winners = run_experiment(experiment)
    assert set(winners) == {"arch", "mid"}
    published = _last_successful_generation_id(experiment)
    assert published is not None
    protected_paths = [
        _summary_path(experiment),
        _last_successful_generation_path(experiment),
        _generation_summary_path(experiment, published),
        _generation_record_path(experiment, published),
        *(_winner_path(experiment, phase.name) for phase in experiment.phases),
        *(
            _generation_winner_path(experiment, published, phase.name)
            for phase in experiment.phases
        ),
    ]
    before = {path: path.read_bytes() for path in protected_paths}
    parent_before = optuna.load_study(study_name="t::arch", storage=storage).trials

    topped_up = _promotion_chain_experiment(
        workdir=tmp_path / "runs",
        trainer=trainer,
        storage=storage,
        on_fail=on_fail,
        arch_n_trials=4,
    )
    with pytest.raises(RuntimeError, match="promotion baseline.*new experiment name"):
        run_experiment(topped_up)

    parent_after = optuna.load_study(study_name="t::arch", storage=storage).trials
    assert len(parent_after) == len(parent_before) == 1
    assert _last_successful_generation_id(topped_up) == published
    assert {path: path.read_bytes() for path in protected_paths} == before


def test_upstream_top_up_detects_transitively_bound_descendant() -> None:
    """A grandchild study binds its ancestor even when the middle study is absent."""
    experiment = make_experiment(
        phases=[
            Phase(
                name="arch",
                n_trials=2,
                search_space={"depth": IntParam(type="int", low=1, high=2)},
            ),
            Phase(name="schedule", inherits=["arch"], n_trials=1, search_space={}),
            Phase(name="final", inherits=["schedule"], n_trials=1, search_space={}),
        ]
    )
    parent_study = SimpleNamespace(
        user_attrs={},
        get_trials=lambda *, deepcopy: [SimpleNamespace(state=optuna.trial.TrialState.COMPLETE)],
    )
    grandchild_study = SimpleNamespace(user_attrs={"phasesweep_fingerprint": "bound-grandchild"})

    with pytest.raises(RuntimeError, match=r"dependent phase study/studies \['final'\]"):
        _reject_bound_descendant_topups(
            experiment,
            from_phase=None,
            existing_studies={"arch": parent_study, "final": grandchild_study},
        )


def test_upstream_top_up_detects_a_bound_promotion_baseline_dependent() -> None:
    """A promotion baseline binds its dependent study with no inheritance edge.

    ``promotion.min_delta_vs`` only has to name a prior phase, so this graph is
    legal and the dependency is invisible to an inherits-only closure
    (PR #5 review / reviewer 2 pass 2, blocker 2).
    """
    experiment = make_experiment(
        phases=[
            Phase(
                name="arch",
                n_trials=2,
                search_space={"depth": IntParam(type="int", low=1, high=2)},
            ),
            Phase(
                name="mid",
                n_trials=1,
                search_space={},
                fixed_overrides={"width": 8},
                promotion=Promotion(min_delta_vs="arch", min_delta=0.0, on_fail="skip"),
            ),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match=r"study/studies \['mid'\] are already bound to its published winner "
        r"via a promotion baseline\.",
    ):
        _reject_bound_descendant_topups(
            experiment,
            from_phase=None,
            existing_studies={
                "arch": _fake_top_up_study(completed=1),
                "mid": _fake_top_up_study(fingerprint="bound-mid"),
            },
        )


def test_upstream_top_up_reaches_a_bound_grandchild_through_a_promotion_baseline() -> None:
    """Mixed promotion/inheritance edges compose, even with no study in between."""
    experiment = make_experiment(
        phases=[
            Phase(
                name="arch",
                n_trials=2,
                search_space={"depth": IntParam(type="int", low=1, high=2)},
            ),
            Phase(
                name="mid",
                n_trials=1,
                search_space={},
                fixed_overrides={"width": 8},
                promotion=Promotion(
                    min_delta_vs="arch", min_delta=0.0, on_fail="continue_baseline"
                ),
            ),
            Phase(name="final", inherits=["mid"], n_trials=1, search_space={}),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match=r"study/studies \['final'\] are already bound to its published winner "
        r"via inheritance and a promotion baseline\.",
    ):
        _reject_bound_descendant_topups(
            experiment,
            from_phase=None,
            existing_studies={
                "arch": _fake_top_up_study(completed=1),
                "final": _fake_top_up_study(fingerprint="bound-final"),
            },
        )


def test_promotion_baseline_dependent_does_not_block_a_completed_upstream_phase() -> None:
    """The guard still only fires for a phase that actually has top-up trials left."""
    experiment = make_experiment(
        phases=[
            Phase(
                name="arch",
                n_trials=2,
                search_space={"depth": IntParam(type="int", low=1, high=2)},
            ),
            Phase(
                name="mid",
                n_trials=1,
                search_space={},
                fixed_overrides={"width": 8},
                promotion=Promotion(min_delta_vs="arch", min_delta=0.0, on_fail="stop"),
            ),
        ]
    )

    _reject_bound_descendant_topups(
        experiment,
        from_phase=None,
        existing_studies={
            "arch": _fake_top_up_study(completed=2),
            "mid": _fake_top_up_study(fingerprint="bound-mid"),
        },
    )


def _artifact_tree_bytes(root: Path) -> dict[str, bytes]:
    """Snapshot every file under an experiment namespace for byte-identity checks.

    :param Path root: Experiment artifact namespace to snapshot.
    :return dict[str, bytes]: Relative path to file content for every regular file.
    """
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_second_workdir_is_rejected_and_leaves_the_bound_root_untouched(tmp_path: Path) -> None:
    """One persistent study cannot back two publication roots (review v0.5.19 / finding F5)."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    bound = _two_phase_experiment(workdir=tmp_path / "runs_a", trainer=trainer, storage=storage)
    run_experiment(bound)
    bound_root = _experiment_dir(bound)
    before = _artifact_tree_bytes(bound_root)
    assert before

    for phase in bound.phases:
        study = optuna.load_study(study_name=f"t::{phase.name}", storage=storage)
        assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(bound_root)

    moved = bound.model_copy(update={"workdir": str(tmp_path / "runs_b")})
    with pytest.raises(ArtifactRootConflictError) as excinfo:
        run_experiment(moved)

    message = str(excinfo.value)
    assert str(bound_root) in message
    assert str(_experiment_dir(moved)) in message
    assert "rebind-workdir" in message
    assert _artifact_tree_bytes(bound_root) == before
    for phase in bound.phases:
        study = optuna.load_study(study_name=f"t::{phase.name}", storage=storage)
        assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(bound_root)


def test_same_workdir_top_up_keeps_the_artifact_root_binding(tmp_path: Path) -> None:
    """An equal binding is a no-op: ordinary resume and top-up still work."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment)
    topped_up = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
    )
    run_experiment(topped_up)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(experiment))
    assert len([trial for trial in study.trials if trial.state.is_finished()]) == 2


def test_binding_claim_ignores_its_atomic_staging_file(tmp_path: Path) -> None:
    """A concurrent binding reader must not misclassify the writer's temp file."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
    )
    binding_path = _artifact_root_binding_path(experiment)
    with atomic_text_writer(binding_path) as staged:
        staging = list(binding_path.parent.glob(f".{binding_path.name}.*.tmp"))
        assert len(staging) == 1

        _validate_artifact_root_binding(experiment, claim_fresh=True)
        staged.write(binding_path.read_text(encoding="utf-8"))

    assert binding_path.is_file()
    assert not list(binding_path.parent.glob(f".{binding_path.name}.*.tmp"))


def test_fresh_binding_ignores_unrelated_operator_files(tmp_path: Path) -> None:
    """A note or OS metadata file does not turn a fresh root into a legacy tree."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
    )
    root = _experiment_dir(experiment)
    root.mkdir(parents=True)
    (root / ".DS_Store").write_bytes(b"metadata")
    (root / "notes.md").write_text("operator notes\n", encoding="utf-8")

    _validate_artifact_root_binding(experiment, claim_fresh=True)

    assert _artifact_root_binding_path(experiment).is_file()


@pytest.mark.parametrize("entry", ["generations", "attempts", "Attempts", "P"])
def test_unbound_known_phasesweep_state_names_the_blocking_entry(
    tmp_path: Path, entry: str
) -> None:
    """Known engine state still requires explicit adoption and is identifiable."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
    )
    (_experiment_dir(experiment) / entry).mkdir(parents=True)

    with pytest.raises(LegacyArtifactRootMigrationRequiredError, match=repr(entry)):
        _validate_artifact_root_binding(experiment, claim_fresh=True)

    assert not _artifact_root_binding_path(experiment).exists()


def test_artifact_tree_rejects_a_second_storage_ledger(tmp_path: Path) -> None:
    """One tree cannot mix publication files from one DB with counts from another."""
    trainer = write_constant_trainer(tmp_path)
    workdir = tmp_path / "runs"

    def _experiment(database: str) -> Experiment:
        return make_experiment(
            workdir=workdir,
            storage=f"sqlite:///{tmp_path / database}",
            trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
            override_format="argparse",
            n_trials=1,
        )

    owner = _experiment("owner.db")
    run_experiment(owner)
    published = _last_successful_generation_id(owner)
    assert published is not None
    generation_dir = _experiment_dir(owner) / "generations"
    generations_before = {path.name for path in generation_dir.iterdir()}
    binding = json.loads(_artifact_root_binding_path(owner).read_text())
    assert set(binding) == {"schema_version", "experiment", "artifact_root", "storage_key"}
    assert len(binding["storage_key"]) == 64
    assert all(character in "0123456789abcdef" for character in binding["storage_key"])

    foreign = _experiment("foreign.db")
    with pytest.raises(ArtifactRootConflictError, match="different storage ledger"):
        run_experiment(foreign)
    with pytest.raises(ArtifactRootConflictError, match="different storage ledger"):
        read_status(foreign)
    with pytest.raises(ArtifactRootConflictError, match="different storage ledger"):
        read_winner(foreign, "p")
    with pytest.raises(ArtifactRootConflictError, match="different storage ledger"):
        read_winner(foreign, "p", generation_id=published)
    with pytest.raises(ArtifactRootConflictError, match="different storage ledger"):
        read_winners(foreign)

    assert not (tmp_path / "foreign.db").exists()
    assert _last_successful_generation_id(owner) == published
    assert {path.name for path in generation_dir.iterdir()} == generations_before
    owner_status = read_status(owner)
    assert owner_status["publication_integrity"] == "ok"
    assert owner_status["phases"][0]["trials"]["COMPLETE"] == 1


def test_unreadable_artifact_root_binding_never_recommends_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permission denial is not evidence of a foreign ledger or authority to rebind."""
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment)
    binding_path = _artifact_root_binding_path(experiment)
    binding_before = binding_path.read_bytes()
    original_read_text = Path.read_text
    patch_path_method_failure(
        monkeypatch,
        binding_path,
        "read_text",
        PermissionError("permission denied"),
    )

    with pytest.raises(ArtifactRootConflictError) as read_info:
        read_status(experiment)
    with pytest.raises(ArtifactRootRebindError) as rebind_info:
        _validate_artifact_root_binding_for_rebind(
            _ArtifactRootRebindPlan(
                experiment=experiment,
                destination=str(_experiment_dir(experiment).resolve()),
                entries=(),
            )
        )

    read_message = str(read_info.value)
    assert "current user" in read_message
    assert "permission denied" in read_message
    assert "different storage ledger" not in read_message
    assert "do not run rebind-workdir" in read_message
    assert "refusing to rebind" in str(rebind_info.value)
    assert binding_path.read_bytes() == binding_before

    monkeypatch.setattr(Path, "read_text", original_read_text)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    snapshot = _generation_summary_path(experiment, generation_id).parent / "config.snapshot.yaml"

    patch_path_method_failure(
        monkeypatch,
        snapshot,
        "read_bytes",
        PermissionError("permission denied"),
    )
    with pytest.raises(ArtifactRootRebindError) as publication_info:
        _validate_artifact_root_destination(
            experiment,
            _artifact_root_rebind_entries(experiment),
        )
    publication_message = str(publication_info.value)
    assert "cannot be validated as the current user" in publication_message
    assert "refusing to rebind" in publication_message
    assert "Move the complete artifact tree" not in publication_message


@pytest.mark.parametrize(
    ("owner_storage", "offered_storage", "secrets"),
    [
        (
            "postgresql://user@host/db?password=FIRSTSECRET",
            "postgresql://user@host/db?password=SECONDSECRET",
            ("FIRSTSECRET", "SECONDSECRET"),
        ),
        (
            "postgresql://user@host/db?access_token=FIRST-TOKEN",
            "postgresql://user@host/db?access_token=SECOND-TOKEN",
            ("FIRST-TOKEN", "SECOND-TOKEN"),
        ),
        (
            "postgresql://user@host/db?sslpassword=FIRST-SECRET",
            "postgresql://user@host/db?sslpassword=SECOND-SECRET",
            ("FIRST-SECRET", "SECOND-SECRET"),
        ),
        (
            "postgresql://user@host/db?client_secret=FIRST-SECRET",
            "postgresql://user@host/db?client_secret=SECOND-SECRET",
            ("FIRST-SECRET", "SECOND-SECRET"),
        ),
        (
            "mssql+pyodbc:///?odbc_connect=DRIVER%3DODBC%3BPWD%3DFIRST-PWD%3BUID%3Duser",
            "mssql+pyodbc:///?odbc_connect=DRIVER%3DODBC%3BPWD%3DSECOND-PWD%3BUID%3Duser",
            ("FIRST-PWD", "SECOND-PWD"),
        ),
        (
            "mssql+pyodbc:///?odbc_connect="
            "DRIVER%3D%7BODBC%20Driver%2017%20for%20SQL%20Server%7D%3B"
            "SERVER%3Ddb.internal%3BDATABASE%3Dstudies%3BEncrypt%3Dyes%3B"
            "ClientSecret%3DFIRST-SECRET",
            "mssql+pyodbc:///?odbc_connect="
            "database%3Dstudies%3Bserver%3Ddb.internal%3B"
            "DRIVER%3D%7BODBC%20Driver%2018%20for%20SQL%20Server%7D%3B"
            "Encrypt%3Dno%3Bclient_secret%3DSECOND-SECRET",
            ("FIRST-SECRET", "SECOND-SECRET"),
        ),
    ],
    ids=[
        "password",
        "access-token",
        "sslpassword",
        "client-secret",
        "nested-odbc-connect",
        "equivalent-nested-odbc-spelling",
    ],
)
def test_artifact_root_binding_survives_rdb_query_credential_rotation(
    tmp_path: Path,
    owner_storage: str,
    offered_storage: str,
    secrets: tuple[str, str],
) -> None:
    """Credential rotation retains root ownership without publishing secrets."""

    def external_experiment(storage: str) -> Experiment:
        placeholder = make_experiment(
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{tmp_path / 'placeholder.db'}",
        )
        return placeholder.model_copy(
            update={
                "storage": storage,
                "allow_external_rdb_single_host": True,
            }
        )

    owner = external_experiment(owner_storage)
    offered = external_experiment(offered_storage)
    _validate_artifact_root_binding(owner, claim_fresh=True)

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _experiment_dir(owner).rglob("*")
        if path.is_file()
    )
    for secret in secrets:
        assert secret not in artifact_text

    binding_path = _artifact_root_binding_path(owner)
    binding_before = binding_path.read_bytes()
    _validate_artifact_root_binding(offered, claim_fresh=False)
    _validate_artifact_root_binding_for_rebind(
        _ArtifactRootRebindPlan(
            experiment=offered,
            destination=str(_experiment_dir(offered).resolve()),
            entries=(),
        )
    )
    assert binding_path.read_bytes() == binding_before


def test_relative_storage_identity_is_bound_to_the_invocation_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same relative URL from another cwd cannot reinterpret one tree."""
    trainer = write_constant_trainer(tmp_path)
    registration_cwd = tmp_path / "registration"
    foreign_cwd = tmp_path / "foreign"
    registration_cwd.mkdir()
    foreign_cwd.mkdir()
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage="sqlite:///ledger.db",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    monkeypatch.chdir(registration_cwd)
    run_experiment(experiment)
    binding_path = _artifact_root_binding_path(experiment)
    binding_before = binding_path.read_bytes()
    generation_before = _last_successful_generation_id(experiment)

    monkeypatch.chdir(foreign_cwd)
    with pytest.raises(ArtifactRootConflictError, match="different storage ledger"):
        run_experiment(experiment)
    with pytest.raises(ArtifactRootConflictError, match="different storage ledger"):
        read_status(experiment)

    assert not (foreign_cwd / "ledger.db").exists()
    assert binding_path.read_bytes() == binding_before
    assert _last_successful_generation_id(experiment) == generation_before


def test_retargeted_experiment_symlink_is_rejected_before_claim(
    tmp_path: Path,
) -> None:
    """A stable configured spelling does not hide a different physical root."""
    trainer = write_constant_trainer(tmp_path)
    original_parent = tmp_path / "original"
    workdir = tmp_path / "runs"
    original_root = original_parent / "t"
    retargeted_root = tmp_path / "physical-b"
    original_parent.mkdir()
    workdir.mkdir()
    retargeted_root.mkdir()
    owner = make_experiment(
        workdir=original_parent,
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(owner)
    original_binding = _artifact_root_binding_path(owner).read_bytes()

    experiment_link = workdir / "t"
    experiment_link.symlink_to(original_root, target_is_directory=True)
    experiment_link.unlink()
    experiment_link.symlink_to(retargeted_root, target_is_directory=True)
    retargeted = owner.model_copy(update={"workdir": str(workdir)})
    with pytest.raises(ArtifactRootConflictError, match="publishes into artifact root"):
        run_experiment(retargeted)

    assert list(retargeted_root.iterdir()) == []
    assert (original_root / "artifact_root_binding.json").read_bytes() == original_binding


def test_preexisting_empty_study_is_adopted_on_first_contact(tmp_path: Path) -> None:
    """An empty study is claimed by the first run that sees it: no evidence can be stranded."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    optuna.create_study(study_name="t::p", storage=storage, direction="minimize")
    assert (
        ARTIFACT_ROOT_ATTR not in optuna.load_study(study_name="t::p", storage=storage).user_attrs
    )

    run_experiment(experiment)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(experiment))


def test_preexisting_empty_study_with_wrong_direction_is_rejected(tmp_path: Path) -> None:
    """An empty Optuna namespace cannot silently override the configured goal."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    experiment = experiment.model_copy(
        update={"metric": experiment.metric.model_copy(update={"goal": "maximize"})}
    )
    study = optuna.create_study(study_name="t::p", storage=storage, direction="minimize")

    with pytest.raises(StudySchemaMismatchError, match="config requires maximize"):
        run_experiment(experiment)

    assert study.get_trials(deepcopy=False) == []


def test_populated_unbound_study_refuses_the_run_instead_of_adopting_it(tmp_path: Path) -> None:
    """A pre-binding study with results is migrated explicitly, never adopted.

    Adopting it would bind whichever workdir happened to run first, so a
    second workdir could publish the same study's winner into a second tree
    that both report as intact (re-review v0.5.19 / blocker B1).
    """
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment)
    root = _experiment_dir(experiment)
    assert_published_winner_evidence_local(root)
    drop_artifact_root_binding(storage, "t::p")

    with pytest.raises(LegacyArtifactRootMigrationRequiredError) as excinfo:
        run_experiment(experiment)

    message = str(excinfo.value)
    assert "rebind-workdir" in message
    assert "nothing was published" in message.lower()
    assert (
        ARTIFACT_ROOT_ATTR not in optuna.load_study(study_name="t::p", storage=storage).user_attrs
    )
    # The refusal is not allowed to cost the tree its existing publication.
    assert _last_successful_generation_id(experiment) is not None
    assert_published_winner_evidence_local(root)


def _unbound_two_phase_studies(
    experiment: Experiment,
    storage: str,
    *,
    bound_root: str,
    arch_trials: int,
) -> None:
    """Create the ``arch``/``lr`` studies with a deliberate mixed binding state.

    ``arch`` is left unbound (optionally populated) and ``lr`` is bound to
    ``bound_root``, so an invocation offering a third root finds one claimable
    study before the one that refuses it.

    :param Experiment experiment: Two-phase experiment naming the studies.
    :param str storage: Persistent storage URL both studies live in.
    :param str bound_root: Artifact root recorded on the ``lr`` study.
    :param int arch_trials: COMPLETE trials to seed into the ``arch`` study.
    """
    arch = optuna.create_study(
        study_name=f"{experiment.experiment}::arch", storage=storage, direction="minimize"
    )
    for _ in range(arch_trials):
        arch.add_trial(optuna.trial.create_trial(value=0.5, state=optuna.trial.TrialState.COMPLETE))
    lr = optuna.create_study(
        study_name=f"{experiment.experiment}::lr", storage=storage, direction="minimize"
    )
    lr.set_user_attr(ARTIFACT_ROOT_ATTR, bound_root)


@pytest.mark.parametrize(
    ("arch_trials", "expected"),
    [
        (0, ArtifactRootConflictError),
        (1, LegacyArtifactRootMigrationRequiredError),
    ],
)
def test_refused_multi_phase_binding_claims_nothing(
    tmp_path: Path,
    arch_trials: int,
    expected: type[Exception],
) -> None:
    """A refused run must not leave an earlier phase bound to the rejected root.

    Claiming inside the check loop bound every phase examined before the
    conflicting one, so the tree the operator was told nothing happened to
    already had a binding pointing at it (re-review v0.5.19 / blocker B3).
    """
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    bound = _two_phase_experiment(workdir=tmp_path / "runs_a", trainer=trainer, storage=storage)
    offered = _two_phase_experiment(workdir=tmp_path / "runs_b", trainer=trainer, storage=storage)
    _unbound_two_phase_studies(
        bound, storage, bound_root=str(_experiment_dir(bound)), arch_trials=arch_trials
    )

    with pytest.raises(expected):
        run_experiment(offered)

    arch = optuna.load_study(study_name="t::arch", storage=storage)
    lr = optuna.load_study(study_name="t::lr", storage=storage)
    assert ARTIFACT_ROOT_ATTR not in arch.user_attrs
    assert lr.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(bound))


def test_unreadable_study_blocks_binding_for_its_siblings_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that aborts in preflight must not leave a binding behind.

    The unreadable study aborts the run either way - as unconfirmed cleanup,
    with the storage failure chained beneath it - so claiming the readable
    studies would invent a binding for an invocation that never ran a trial
    (re-review v0.5.19 / blocker B3).
    """
    import phasesweep.engine.artifact_roots as artifact_roots

    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = _two_phase_experiment(workdir=tmp_path / "runs", trainer=trainer, storage=storage)
    for phase in experiment.phases:
        optuna.create_study(study_name=f"t::{phase.name}", storage=storage, direction="minimize")
    real_loader = artifact_roots._load_existing_phase_study

    def _fail_for_lr(exp: Experiment, phase: Phase) -> optuna.Study | None:
        if phase.name == "lr":
            raise RuntimeError("storage went away")
        return real_loader(exp, phase)

    monkeypatch.setattr(artifact_roots, "_load_existing_phase_study", _fail_for_lr)
    monkeypatch.setattr("phasesweep.engine.relocation._load_existing_phase_study", _fail_for_lr)

    with pytest.raises(ProcessCleanupUncertainError) as excinfo:
        run_experiment(experiment)

    assert isinstance(excinfo.value.__cause__, StudyStorageUnavailableError)

    for phase in experiment.phases:
        study = optuna.load_study(study_name=f"t::{phase.name}", storage=storage)
        assert ARTIFACT_ROOT_ATTR not in study.user_attrs


@pytest.mark.parametrize("backend", ["sqlite", "journal", "auto"])
def test_published_phase_rejects_a_missing_storage_ledger(
    tmp_path: Path,
    backend: str,
) -> None:
    """A lost published ledger cannot be mistaken for a never-run phase."""
    trainer = write_constant_trainer(tmp_path)
    suffix = "db" if backend == "sqlite" else "journal"
    configured_storage = (
        "auto" if backend == "auto" else f"{backend}:///{tmp_path / f'studies.{suffix}'}"
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=configured_storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    assert experiment.resolved_storage is not None
    if storage_backend(experiment.resolved_storage) == "sqlite":
        ledger = sqlite_database_path(experiment.resolved_storage)
        assert ledger is not None
    else:
        ledger = Path(file_url_path(experiment.resolved_storage))
    run_experiment(experiment)
    published = _last_successful_generation_id(experiment)
    assert published is not None
    generation_before = _generation_path(experiment).read_bytes()
    generation_dirs_before = {
        path.name for path in (_experiment_dir(experiment) / "generations").iterdir()
    }

    ledger.unlink()

    with pytest.raises(PublishedStudyMissingError) as excinfo:
        run_experiment(experiment)

    assert "includes a winner for phase 'p'" in str(excinfo.value)
    assert "persistent study is missing" in str(excinfo.value)
    assert "Cleanup state is therefore unknown" not in str(excinfo.value)
    assert "Restore the original complete storage ledger and study" in str(excinfo.value)
    assert _generation_path(experiment).read_bytes() == generation_before
    assert _last_successful_generation_id(experiment) == published
    assert {
        path.name for path in (_experiment_dir(experiment) / "generations").iterdir()
    } == generation_dirs_before
    assert not ledger.exists()


@pytest.mark.parametrize("replacement", ["absent", "empty"])
def test_published_phase_rejects_a_missing_or_empty_named_study(
    tmp_path: Path,
    replacement: str,
) -> None:
    """Deleting or replacing one published study cannot restart trial zero."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment)
    published = _last_successful_generation_id(experiment)
    assert published is not None
    generation_before = _generation_path(experiment).read_bytes()
    generation_dirs_before = {
        path.name for path in (_experiment_dir(experiment) / "generations").iterdir()
    }

    optuna.delete_study(study_name="t::p", storage=storage)
    if replacement == "empty":
        optuna.create_study(study_name="t::p", storage=storage, direction="minimize")

    with pytest.raises(PublishedStudyMissingError) as excinfo:
        run_experiment(experiment)

    expected = "is missing" if replacement == "absent" else "contains no trials"
    assert expected in str(excinfo.value)
    assert "continuing could reuse incomplete or unrelated trials" in str(excinfo.value)
    assert "Cleanup state is therefore unknown" not in str(excinfo.value)
    assert _generation_path(experiment).read_bytes() == generation_before
    assert _last_successful_generation_id(experiment) == published
    assert {
        path.name for path in (_experiment_dir(experiment) / "generations").iterdir()
    } == generation_dirs_before


def test_published_phase_trial_read_failure_preserves_cleanup_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published-phase existence read keeps strict storage-error semantics."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment)
    published = _last_successful_generation_id(experiment)
    assert published is not None
    generation_before = _generation_path(experiment).read_bytes()
    generation_dirs_before = {
        path.name for path in (_experiment_dir(experiment) / "generations").iterdir()
    }

    def fail_trial_read(*_args: object, **_kwargs: object) -> list[optuna.trial.FrozenTrial]:
        raise RuntimeError("storage went away during trial read")

    monkeypatch.setattr(optuna.Study, "get_trials", fail_trial_read)

    with pytest.raises(ProcessCleanupUncertainError) as excinfo:
        run_experiment(experiment)

    cause = excinfo.value.__cause__
    assert isinstance(cause, StudyStorageUnavailableError)
    assert "published phase 'p'" in str(cause)
    assert isinstance(cause.__cause__, RuntimeError)
    assert _generation_path(experiment).read_bytes() == generation_before
    assert {
        path.name for path in (_experiment_dir(experiment) / "generations").iterdir()
    } == generation_dirs_before


@pytest.mark.parametrize("replacement", ["absent", "empty"])
@pytest.mark.parametrize("missing_phase", ["arch", "lr"])
def test_published_study_requirement_starts_at_from_phase(
    tmp_path: Path, replacement: str, missing_phase: str
) -> None:
    """Skipped winners survive ledger loss; phases that execute still need history."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = _two_phase_experiment(workdir=tmp_path / "runs", trainer=trainer, storage=storage)
    original = run_experiment(experiment)
    published = _last_successful_generation_id(experiment)
    generation_before = _generation_path(experiment).read_bytes()
    optuna.delete_study(study_name=f"t::{missing_phase}", storage=storage)
    if replacement == "empty":
        optuna.create_study(study_name=f"t::{missing_phase}", storage=storage, direction="minimize")

    if missing_phase == "lr":
        with pytest.raises(PublishedStudyMissingError, match="phase 'lr'"):
            run_experiment(experiment, from_phase="lr")
        assert _generation_path(experiment).read_bytes() == generation_before
        assert _last_successful_generation_id(experiment) == published
    else:
        winners = run_experiment(experiment, from_phase="lr")
        assert winners["arch"].params == original["arch"].params
        assert winners["arch"].trial_number == original["arch"].trial_number
        assert _last_successful_generation_id(experiment) != published
        if replacement == "absent":
            assert "t::arch" not in optuna.get_all_study_names(storage=storage)
        else:
            assert not optuna.load_study(study_name="t::arch", storage=storage).trials
        assert len(optuna.load_study(study_name="t::lr", storage=storage).trials) == 1


@pytest.mark.parametrize("tree_state", ["fresh", "published", "missing-ledger"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_invalid_from_phase_is_rejected_before_state_writes(
    tmp_path: Path, tree_state: str, dry_run: bool
) -> None:
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage="auto",
        n_trials=1,
        trial_command="echo x=0.5 {overrides}",
    )
    if tree_state != "fresh":
        run_experiment(experiment)
        if tree_state == "missing-ledger":
            (_experiment_dir(experiment) / "study.db").unlink()
    pointer = _generation_path(experiment)
    pointer_before = pointer.read_bytes() if pointer.exists() else None
    generations = _experiment_dir(experiment) / "generations"
    generations_before = set(generations.iterdir()) if generations.exists() else set()

    with pytest.raises(ValueError, match="Unknown --from-phase value 'bogus'"):
        run_experiment(experiment, from_phase="bogus", dry_run=dry_run)

    assert (pointer.read_bytes() if pointer.exists() else None) == pointer_before
    assert (set(generations.iterdir()) if generations.exists() else set()) == generations_before
    if tree_state == "fresh":
        assert not Path(experiment.workdir).exists()
    elif tree_state == "missing-ledger":
        assert not (_experiment_dir(experiment) / "study.db").exists()


def test_ledger_loss_during_execution_preserves_cleanup_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing study is only a static refusal when execution has not started."""
    import phasesweep.engine.run as engine_run

    trainer = write_constant_trainer(tmp_path)
    ledger = tmp_path / "studies.db"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{ledger}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment)
    published = _last_successful_generation_id(experiment)

    def lose_ledger(*_args: object, **_kwargs: object) -> None:
        ledger.unlink()
        raise RuntimeError("execution failed after the ledger disappeared")

    monkeypatch.setattr(engine_run, "_run_experiment_inner", lose_ledger)
    reports = []
    with pytest.raises(ProcessCleanupUncertainError) as excinfo:
        run_experiment(experiment, terminal_callback=reports.append)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert reports[0].cleanup_confirmed is False
    assert _last_successful_generation_id(experiment) == published


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Collect an exception plus everything reachable through its chain.

    :param BaseException error: Exception raised by the code under test.
    :return list[BaseException]: ``error`` and every distinct exception
        reachable through ``__cause__``/``__context__``.
    """
    collected: list[BaseException] = []
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if any(current is seen for seen in collected):
            continue
        collected.append(current)
        pending.extend(
            linked for linked in (current.__cause__, current.__context__) if linked is not None
        )
    return collected


def _fabricate_interrupted_attempt(
    experiment: Experiment,
    storage: str,
    *,
    attempt_id: str,
) -> tuple[str, int, Path]:
    """Leave one phase study holding a stale RUNNING trial with a registry entry.

    Reproduces an orchestrator that died between allocating an attempt and
    launching its trainer: the trial stays RUNNING, its durable lifecycle says
    ``allocated``, and the experiment-level registry still lists it. That is
    exactly the state recovery is allowed to repair - and must refuse to touch
    from a wrong-root invocation.

    :param Experiment experiment: Experiment owning the artifact tree and study.
    :param str storage: Persistent storage URL backing the phase study.
    :param str attempt_id: Immutable attempt identity to persist everywhere.
    :return tuple[str, int, Path]: Study name, stale trial number, registry entry path.
    """
    phase_name = experiment.phases[0].name
    study_name = f"{experiment.experiment}::{phase_name}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    trial = study.ask()
    trial_dir = _phase_dir(experiment, phase_name) / f"trial_{trial.number:05d}__stale"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    write_attempt_lifecycle(trial_dir, attempt_id=attempt_id, state="allocated")
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase_name,
        study_name=study_name,
        trial_number=trial.number,
        trial_dir=trial_dir,
        generation_id="g-stale",
    )
    return study_name, trial.number, _attempts_dir(experiment) / f"{attempt_id}.json"


def test_transient_study_read_failure_aborts_before_any_recovery(tmp_path: Path) -> None:
    """A read that fails once must abort the run, not be retried into recovery.

    Reviewer repro (PR #5 review / reviewer 2, issue 1): discovery swallowed
    the storage failure, the main preflight loop re-read the study, the second
    read succeeded, and stale-trial reaping then mutated a study whose
    artifact-root binding had never been checked - a wrong-root invocation
    silently failing the bound tree's RUNNING trial. Discovery is now the only
    read, so the failure escalates and nothing in either tree is touched. The
    unpatched second invocation pins the same refusal for the ordinary
    wrong-root case: it conflicts before the registry scan can reap anything.
    """
    import phasesweep.engine.artifact_roots as artifact_roots

    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment_a = make_experiment(
        workdir=tmp_path / "runs_a",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment_a)
    root_a = str(_experiment_dir(experiment_a))
    study_name, stale_number, entry_path = _fabricate_interrupted_attempt(
        experiment_a, storage, attempt_id="stale-attempt-1"
    )
    assert entry_path.is_file()

    experiment_b = experiment_a.model_copy(update={"workdir": str(tmp_path / "runs_b")})
    real_loader = artifact_roots._load_existing_phase_study
    calls = {"count": 0}

    def _fail_first_read(exp: Experiment, phase: Phase) -> optuna.Study | None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient storage failure")
        return real_loader(exp, phase)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(artifact_roots, "_load_existing_phase_study", _fail_first_read)
        patched.setattr("phasesweep.engine.relocation._load_existing_phase_study", _fail_first_read)
        with pytest.raises(ProcessCleanupUncertainError) as excinfo:
            run_experiment(experiment_b)

    assert isinstance(excinfo.value.__cause__, StudyStorageUnavailableError)
    chain = _exception_chain(excinfo.value)
    assert any(isinstance(error, StudyStorageUnavailableError) for error in chain)
    # The storage failure must stay the reported cause: a later, luckier read
    # producing a root conflict would mean discovery ran twice.
    assert not any(isinstance(error, ArtifactRootConflictError) for error in chain)

    after_transient = optuna.load_study(study_name=study_name, storage=storage)
    assert (
        after_transient.get_trials(deepcopy=False)[stale_number].state
        == optuna.trial.TrialState.RUNNING
    )
    assert entry_path.is_file()
    assert after_transient.user_attrs[ARTIFACT_ROOT_ATTR] == root_a

    # Same wrong-root invocation, no injected failure: the conflict is raised
    # before the registry scan, so the stale attempt survives untouched here too.
    with pytest.raises(ArtifactRootConflictError):
        run_experiment(experiment_b)

    after_conflict = optuna.load_study(study_name=study_name, storage=storage)
    assert (
        after_conflict.get_trials(deepcopy=False)[stale_number].state
        == optuna.trial.TrialState.RUNNING
    )
    assert entry_path.is_file()
    assert after_conflict.user_attrs[ARTIFACT_ROOT_ATTR] == root_a


def test_sqlite_study_probe_raises_while_the_database_is_locked(tmp_path: Path) -> None:
    """An unreadable database is never reported as study absence.

    A briefly locked file collapsing into "no such study" would skip the
    artifact-root check for a study that becomes readable one call later
    (PR #5 review / reviewer 2, issue 1).
    """
    trainer = write_constant_trainer(tmp_path)
    db_path = tmp_path / "studies.db"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{db_path}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment)
    phase = experiment.phases[0]
    assert _sqlite_study_exists(experiment, phase) is True

    # BEGIN EXCLUSIVE holds the write lock until this connection closes, and
    # the probe connects with timeout=0.1, so the refusal is deterministic.
    locker = sqlite3.connect(db_path, isolation_level=None, timeout=10)
    try:
        locker.execute("BEGIN EXCLUSIVE")
        with pytest.raises(StudyStorageUnavailableError):
            _sqlite_study_exists(experiment, phase)
    finally:
        locker.close()

    assert _sqlite_study_exists(experiment, phase) is True


def test_sqlite_study_probe_reports_absence_only_for_genuine_absence(tmp_path: Path) -> None:
    """Missing file and schema-less file are the only two absence verdicts."""
    trainer = write_constant_trainer(tmp_path)
    trial_command = f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}"

    never_created = make_experiment(
        workdir=tmp_path / "runs_missing",
        storage=f"sqlite:///{tmp_path / 'never-created.db'}",
        trial_command=trial_command,
        n_trials=1,
    )
    assert not (tmp_path / "never-created.db").exists()
    assert _sqlite_study_exists(never_created, never_created.phases[0]) is False

    schemaless_path = tmp_path / "schemaless.db"
    sqlite3.connect(schemaless_path).close()
    assert schemaless_path.exists()
    schemaless = make_experiment(
        workdir=tmp_path / "runs_schemaless",
        storage=f"sqlite:///{schemaless_path}",
        trial_command=trial_command,
        n_trials=1,
    )
    assert _sqlite_study_exists(schemaless, schemaless.phases[0]) is False


@pytest.mark.parametrize("storage", [None, "sqlite:///:memory:"])
def test_fresh_and_repeated_in_memory_roots_never_bind_or_conflict(
    tmp_path: Path, storage: str | None
) -> None:
    """In-memory runs may repeat or move because they create no durable binding."""
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs_a",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(experiment)
    run_experiment(experiment)
    moved = experiment.model_copy(update={"workdir": str(tmp_path / "runs_b")})
    run_experiment(moved)
    run_experiment(moved)

    assert _last_successful_generation_id(experiment) is not None
    assert _last_successful_generation_id(moved) is not None
    assert not _artifact_root_binding_path(experiment).exists()
    assert not _artifact_root_binding_path(moved).exists()


@pytest.mark.parametrize("in_memory_storage", [None, "sqlite:///:memory:"])
def test_persistent_bound_root_rejects_an_in_memory_configuration(
    tmp_path: Path, in_memory_storage: str | None
) -> None:
    """A durable publication tree cannot be reused with an ephemeral ledger."""
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    owner = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    run_experiment(owner)
    pointer_before = _generation_path(owner).read_bytes()
    ledger_before = sqlite_database_path(storage)
    assert ledger_before is not None
    ledger_bytes_before = ledger_before.read_bytes()
    status_before = read_status(owner)

    offered = owner.model_copy(update={"storage": in_memory_storage})
    with pytest.raises(
        ArtifactRootConflictError, match="in-memory configuration cannot reuse"
    ) as excinfo:
        run_experiment(offered)
    assert str(excinfo.value).endswith("Nothing was written.")
    with pytest.raises(
        ArtifactRootConflictError, match="in-memory configuration cannot reuse"
    ) as excinfo:
        read_status(offered)
    assert str(excinfo.value).endswith("Nothing was written.")

    assert _generation_path(owner).read_bytes() == pointer_before
    assert ledger_before.read_bytes() == ledger_bytes_before
    assert read_status(owner) == status_before


def test_generation_id_reuse_is_rejected_without_overwriting_history(tmp_path: Path) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )
    generation_id = "fixed-generation"
    run_experiment(experiment, generation_id=generation_id)
    protected = [
        _generation_record_path(experiment, generation_id),
        _generation_summary_path(experiment, generation_id),
        _generation_winner_path(experiment, generation_id, "p"),
    ]
    before = {path: path.read_bytes() for path in protected}

    with pytest.raises(RunRequestError, match="already exists; refusing to overwrite history"):
        run_experiment(experiment, generation_id=generation_id)

    assert {path: path.read_bytes() for path in protected} == before
    study = optuna.load_study(study_name="t::p", storage=experiment.storage)
    assert len(study.trials) == 1


def test_version_audit_metadata_is_separate_from_fingerprint_schema() -> None:
    """Package version is audit metadata, not semantic experiment identity.

    Source of truth is package metadata generated by setuptools-scm from SCM
    state. ``__version__`` reads that metadata rather than a generated source
    file.
    """
    from importlib.metadata import version as pkg_version

    from phasesweep.engine.fingerprints import _phase_semantic_payload

    assert __version__ == pkg_version("phasesweep")

    exp = make_experiment()
    payload = _phase_semantic_payload(exp, exp.phases[0], {})
    assert payload["fingerprint_schema_version"] == FINGERPRINT_SCHEMA_VERSION
    assert "phasesweep_version" not in payload


def test_winner_yaml_contains_phase_fingerprint(tmp_path: Path) -> None:
    """Every saved winner carries the SHA-256 fingerprint of its producing
    phase config. ``--from-phase`` reuses winners only if this matches the
    re-computed fingerprint of the current YAML.
    """
    trainer = write_constant_trainer(tmp_path)
    exp = make_experiment(
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
    )
    run_experiment(exp)

    data = yaml.safe_load(_winner_path(exp, "p").read_text())
    assert "phase_fingerprint" in data
    assert isinstance(data["phase_fingerprint"], str)
    assert len(data["phase_fingerprint"]) == 64  # SHA-256 hex digest


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            {"arch_low": 1, "arch_high": 4},
            {"arch_low": 12, "arch_high": 16},
            id="search-space",
        ),
        pytest.param(
            {"arch_fixed_overrides": {"width": 64}},
            {"arch_fixed_overrides": {"width": 128}},
            id="fixed-overrides",
        ),
    ],
)
def test_from_phase_rejects_stale_parent_winner(
    tmp_path: Path,
    before: dict[str, object],
    after: dict[str, object],
) -> None:
    """Changing a skipped parent's semantics must invalidate its winner."""
    trainer = write_constant_trainer(tmp_path)
    workdir = tmp_path / "runs"
    exp_v1 = _two_phase_experiment(workdir=workdir, trainer=trainer, **before)
    run_experiment(exp_v1)

    exp_v2 = _two_phase_experiment(workdir=workdir, trainer=trainer, **after)
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


def test_from_phase_ignores_lower_trial_target_on_skipped_phase(tmp_path: Path) -> None:
    """A skipped phase's run-control budget cannot block an unrelated resume."""
    trainer = write_constant_trainer(tmp_path)
    workdir = tmp_path / "runs"
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    exp_v1 = _two_phase_experiment(
        workdir=workdir,
        trainer=trainer,
        arch_n_trials=3,
        storage=storage,
    )
    winners1 = run_experiment(exp_v1)

    exp_v2 = _two_phase_experiment(
        workdir=workdir,
        trainer=trainer,
        arch_n_trials=1,
        storage=storage,
    )
    winners2 = run_experiment(exp_v2, from_phase="lr")

    assert winners2["arch"].trial_number == winners1["arch"].trial_number
    assert winners2["arch"].metric == winners1["arch"].metric
    study = optuna.load_study(study_name="t::arch", storage=storage)
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 3
    assert len(study.trials) == 3


def test_from_phase_preflight_consumes_run_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = write_constant_trainer(tmp_path)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    experiment = _two_phase_experiment(
        workdir=tmp_path / "runs",
        trainer=trainer,
        storage=storage,
    )
    run_experiment(experiment)
    resumed = experiment.model_copy(update={"timeout_seconds_per_run": 1.0})

    run_module = importlib.import_module("phasesweep.engine.run")
    resume_ops = importlib.import_module("phasesweep.engine.resume")
    artifact_io = importlib.import_module("phasesweep.engine.artifacts")
    clock = {"now": 100.0}
    original_load_winner = artifact_io._load_winner

    def delayed_load_winner(*args: object, **kwargs: object) -> Winner:
        winner = original_load_winner(*args, **kwargs)
        clock["now"] += 2.0
        return winner

    monkeypatch.setattr(
        run_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"]),
    )
    monkeypatch.setattr(resume_ops, "time", run_module.time)
    monkeypatch.setattr(artifact_io, "_load_winner", delayed_load_winner)

    with pytest.raises(TimeoutError, match="before phase 'lr' could start"):
        run_experiment(resumed, from_phase="lr")


def test_fresh_run_preflight_consumes_run_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs").model_copy(
        update={"timeout_seconds_per_run": 1.0}
    )
    run_module = importlib.import_module("phasesweep.engine.run")
    clock = {"now": 100.0}

    def delayed_preflight(
        _experiment: Experiment,
        *,
        cleanup_report: object,
        from_phase: str | None,
        preloaded_studies: object,
    ) -> dict[str, optuna.Study]:
        del cleanup_report, from_phase, preloaded_studies
        clock["now"] += 2.0
        return {}

    monkeypatch.setattr(
        run_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"]),
    )
    monkeypatch.setattr("phasesweep.engine.guards._preflight_existing_studies", delayed_preflight)

    with pytest.raises(TimeoutError, match="before phase 'p' could start"):
        run_experiment(experiment)


def test_load_winner_refuses_incompatible_legacy_winner_yaml(tmp_path: Path) -> None:
    """Legacy skipped winners still need matching fingerprints and provenance."""

    def strip_fingerprint(data: dict) -> str:
        del data["phase_fingerprint"]
        return "no phase_fingerprint"

    def tamper_fingerprint(data: dict) -> str:
        data["phase_fingerprint"] = "0" * 64
        return "different phase config"

    def strip_generation_id(data: dict) -> str:
        del data["generation_id"]
        return "no valid generation_id"

    def strip_attempt_id(data: dict) -> str:
        del data["attempt_id"]
        return "no valid attempt_id"

    def strip_winner_source(data: dict) -> str:
        del data["winner_source"]
        return "no valid winner_source"

    cases = (
        ("missing-fingerprint", strip_fingerprint),
        ("tampered-fingerprint", tamper_fingerprint),
        ("legacy-generation", strip_generation_id),
        ("legacy-attempt", strip_attempt_id),
        ("legacy-source", strip_winner_source),
    )
    for case, mutate in cases:
        case_dir = tmp_path / case
        case_dir.mkdir()
        trainer = write_constant_trainer(case_dir)
        exp = _two_phase_experiment(workdir=case_dir / "runs", trainer=trainer)
        run_experiment(exp)

        generation_id = _last_successful_generation_id(exp)
        assert generation_id is not None

        # Exercise the pre-generation compatibility reader directly. Modern
        # pointer-backed runs authenticate the complete summary and winner
        # manifest before this record-level validation is reached.
        _last_successful_generation_path(exp).unlink()
        _generation_path(exp).unlink()

        arch_winner_path = _published_winner_path(exp, "arch")
        assert arch_winner_path is not None
        data = yaml.safe_load(arch_winner_path.read_text())
        match = mutate(data)
        arch_winner_path.write_text(yaml.safe_dump(data, sort_keys=False))

        with pytest.raises(RuntimeError, match=match):
            _load_winner(exp, exp.phases[0], {})


def test_from_phase_reports_published_winner_manifest_failure(tmp_path: Path) -> None:
    """Resume names the corrupt artifact instead of claiming no run completed."""
    trainer = write_constant_trainer(tmp_path)
    exp = _two_phase_experiment(workdir=tmp_path / "runs", trainer=trainer)
    run_experiment(exp)

    arch_winner_path = _published_winner_path(exp, "arch")
    assert arch_winner_path is not None
    data = yaml.safe_load(arch_winner_path.read_text())
    data["phase_fingerprint"] = "0" * 64
    arch_winner_path.write_text(yaml.safe_dump(data, sort_keys=False))
    shutil.rmtree(_phase_dir(exp, "lr"))

    with pytest.raises(
        RuntimeError,
        match="manifest validation failed: winner artifact for phase 'arch' "
        "does not match its recorded hash",
    ):
        run_experiment(exp, from_phase="lr")


def test_load_winner_normalizes_malformed_yaml_error(tmp_path: Path) -> None:
    exp = make_experiment(workdir=tmp_path / "runs")
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"trial_number": 0, "metric": {"objective":')

    with pytest.raises(WinnerIntegrityError, match="invalid or incomplete"):
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
                "generation_id": "generation-test",
                "attempt_id": "attempt-test",
                "winner_source": {
                    "kind": "phase_trial",
                    "phase": "p",
                    "trial_number": 0,
                    "generation_id": "generation-test",
                    "attempt_id": "attempt-test",
                    "study": None,
                },
            },
            sort_keys=False,
        )
    )

    with pytest.raises(WinnerIntegrityError, match="invalid or incomplete"):
        _load_winner(exp, exp.phases[0], {})


def test_winner_and_trial_attrs_record_trainer_environment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every trial records which environment produced it; the winner keeps the digest.

    Names are diagnostic and persisted; values are secrets and never leave the
    process (review v0.5.18 / finding F3).
    """
    monkeypatch.setenv("PHASESWEEP_TEST_TOKEN", "ambient-secret")
    trainer = write_constant_trainer(tmp_path)
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        env={"PHASESWEEP_TEST_SECRET": "config-secret"},
        execution=ExecutionContext(inherit_env=["PHASESWEEP_TEST_TOKEN"]),
        n_trials=1,
    )

    run_experiment(exp)

    identity = _environment_identity(exp)
    data = yaml.safe_load(_winner_path(exp, "p").read_text())
    assert data["trainer_env_digest"] == identity.digest
    assert data["trainer_inherit_env"] == ["PHASESWEEP_TEST_TOKEN"]

    study = optuna.load_study(study_name="t::p", storage=exp.storage)
    attrs = study.get_trials(deepcopy=False)[0].user_attrs
    names = attrs[TRAINER_ENV_NAMES_ATTR]
    assert attrs[TRAINER_ENV_DIGEST_ATTR] == identity.digest
    assert names == sorted(names) == list(identity.names)
    assert {"PHASESWEEP_TEST_SECRET", "PHASESWEEP_TEST_TOKEN"} <= set(names)
    serialized = json.dumps(attrs, default=str)
    assert "config-secret" not in serialized
    assert "ambient-secret" not in serialized


def _environment_free_winner_payload(exp: Experiment) -> dict:
    """Build a valid winner payload that predates environment identity fields."""
    return {
        "phase": "p",
        "metric": {exp.metric.name: 0.5, "goal": exp.metric.goal},
        "trial_number": 0,
        "params": {"x": 1},
        "effective_overrides": {"x": 1},
        "constraints": {},
        "gates": [],
        "completion": {"incomplete": False},
        "generation_id": "generation-test",
        "attempt_id": "attempt-test",
        "winner_source": {
            "kind": "phase_trial",
            "phase": "p",
            "trial_number": 0,
            "generation_id": "generation-test",
            "attempt_id": "attempt-test",
            "study": None,
        },
        "phase_fingerprint": _phase_fingerprint(exp, exp.phases[0], {}),
    }


def test_load_winner_accepts_records_without_environment_identity(tmp_path: Path) -> None:
    """Pre-existing artifact trees stay loadable; the new fields default to None."""
    exp = make_experiment(workdir=tmp_path / "runs")
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_environment_free_winner_payload(exp), sort_keys=False))

    winner = _load_winner(exp, exp.phases[0], {})

    assert winner.trainer_env_digest is None
    assert winner.trainer_inherit_env is None


@pytest.mark.parametrize("diverged", [True, False])
def test_load_winner_warns_once_on_inherited_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    diverged: bool,
) -> None:
    """Inheriting a winner produced under a different environment is reported once."""
    monkeypatch.setattr("phasesweep.engine.artifacts._ENVIRONMENT_DRIFT_WARNED", set())
    exp = make_experiment(workdir=tmp_path / "runs")
    payload = _environment_free_winner_payload(exp)
    payload["trainer_env_digest"] = "0" * 64 if diverged else _environment_identity(exp).digest
    payload["trainer_inherit_env"] = "all"
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.artifacts"):
        for _ in range(2):
            _load_winner(exp, exp.phases[0], {})

    warnings = [r for r in caplog.records if "environment" in r.getMessage()]
    assert len(warnings) == (1 if diverged else 0)
    if warnings:
        assert warnings[0].name == "phasesweep.engine.artifacts"


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
        generation_id="generation-original",
        attempt_id="attempt-original",
    )
    replacement = Winner(
        trial_number=1,
        params={"x": 1},
        effective_overrides={"x": 1},
        metric=0.5,
        phase_fingerprint=fingerprint,
        generation_id="generation-replacement",
        attempt_id="attempt-replacement",
    )

    _save_winner(exp, phase.name, original, generation_id="generation-original")
    path = _generation_winner_path(exp, "generation-original", phase.name)
    before = path.read_text()

    def fail_replace(_src: Path | str, _dst: Path | str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("phasesweep.runtime.files.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _save_winner(exp, phase.name, replacement, generation_id="generation-original")

    assert path.read_text() == before
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_phase_comment_schema_and_fingerprint(tmp_path: Path) -> None:
    """Editing an optional phase comment must not invalidate its fingerprint."""

    def build(comment: str | None) -> Experiment:
        return Experiment(
            experiment="t",
            workdir=str(tmp_path / "wd"),
            trial_command="echo {overrides}",
            override_format="argparse",
            metric=Metric(
                extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
            ),
            phases=[
                Phase(  # type: ignore[arg-type]
                    name="p",
                    n_trials=4,
                    comment=comment,
                    search_space={"x": IntParam(type="int", low=0, high=10)},
                )
            ],
        )

    def fingerprint(comment: str | None) -> str:
        experiment = build(comment)
        return _phase_fingerprint(experiment, experiment.phases[0], {})

    assert fingerprint("First version") == fingerprint("Reworded later") == fingerprint(None)


def test_zero_trial_preflight_failure_does_not_poison_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource failure before any trial must not bind an empty study forever.

    Reviewer repro (review v0.5.17 / finding A): the fingerprint is stamped
    before GPU preflight; a config that failed before launching any trial used
    to leave a zero-trial study bound to that fingerprint, so the *corrected*
    config was rejected with StudyFingerprintMismatchError even though nothing
    ever ran.
    """
    from tests.conftest import make_experiment, write_trainer

    trainer = write_trainer(tmp_path, "print('x=1.0')")
    db = tmp_path / "fp.db"

    def _exp(command: str) -> Experiment:
        return make_experiment(
            experiment="fp_poison",
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{db}",
            trial_command=command,
            n_trials=1,
        )

    def _gpu_preflight_fails(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("no GPUs detected")

    with monkeypatch.context() as ctx:
        ctx.setattr("phasesweep.engine.phase.GpuPool.create", _gpu_preflight_fails)
        with pytest.raises(RuntimeError, match="no GPUs detected"):
            run_experiment(_exp("false {overrides}"))

    # Corrected trainer command = different semantic fingerprint. The empty
    # study must rebind instead of rejecting the fix.
    winners = run_experiment(_exp(f"python {trainer} {{overrides}}"))
    assert winners["p"].metric == pytest.approx(1.0)


def test_zero_trial_crash_after_target_record_does_not_poison_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A target attr without any trial is not evidence that the old config ran."""
    import phasesweep.engine.phase as phase_module

    trainer = write_trainer(tmp_path, "print('x=1.0')")
    db = tmp_path / "fp-target.db"

    def _exp(command: str) -> Experiment:
        return make_experiment(
            experiment="fp_target_poison",
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{db}",
            trial_command=command,
            n_trials=1,
            gpu_policy="none",
        )

    real_record = phase_module._record_trial_target

    def record_then_crash(study: optuna.Study, phase: Phase) -> None:
        real_record(study, phase)
        raise RuntimeError("simulated crash after target record")

    with monkeypatch.context() as ctx:
        ctx.setattr(phase_module, "_record_trial_target", record_then_crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_experiment(_exp("false {overrides}"))

    study = optuna.load_study(
        study_name="fp_target_poison::p",
        storage=f"sqlite:///{db}",
    )
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 1
    assert study.trials == []

    winners = run_experiment(_exp(f"python {trainer} {{overrides}}"))
    assert winners["p"].metric == pytest.approx(1.0)


def test_fingerprint_mismatch_still_raises_once_a_trial_exists(tmp_path: Path) -> None:
    """The empty-study rebind must not weaken identity once results exist."""
    from tests.conftest import make_experiment, write_trainer

    trainer = write_trainer(tmp_path, "print('x=1.0')")
    db = tmp_path / "fp2.db"

    def _exp(command: str) -> Experiment:
        return make_experiment(
            experiment="fp_guard",
            workdir=tmp_path / "runs",
            storage=f"sqlite:///{db}",
            trial_command=command,
            n_trials=1,
        )

    run_experiment(_exp(f"python {trainer} {{overrides}}"))
    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(_exp(f"python {trainer} --changed {{overrides}}"))
