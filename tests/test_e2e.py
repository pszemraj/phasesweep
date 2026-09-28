"""End-to-end: run all three phases against the fake training script.

Verifies:
  - Phase chaining: depth winner is fixed during lr phase, etc.
  - Constraint enforcement: 16-layer trial violates 16 MiB budget and is excluded.
  - Winner persistence: winner.yaml + summary.yaml written.
  - Replay: --from-phase loads prior winners.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import optuna
import pytest
import yaml

from phasesweep import load_experiment, run_experiment
from phasesweep.config import IntParam, Phase, Sampler
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.phase import GpuPool
from phasesweep.engine.state import PHASE_FINGERPRINT_ATTR
from tests.conftest import copy_fake_train, make_experiment, write_constant_trainer

# A full three-phase sweep against the real fake trainer: every test here spawns
# trainer subprocesses and runs for seconds, not milliseconds.
pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_YAML = REPO / "examples" / "experiment.yaml"


def _prep(tmp_path: Path) -> Path:
    """Copy example yaml into tmp, rewrite paths to be tmp-local, return new yaml path."""
    trainer = copy_fake_train(tmp_path)

    text = EXAMPLE_YAML.read_text()
    runs_dir = tmp_path / "runs"
    for old, new in (
        ("./runs/phases.journal", str(runs_dir / "phases.journal")),
        ("./runs", str(runs_dir)),
        ("python -m phasesweep.examples.fake_train", f"python {trainer.resolve()}"),
    ):
        # A replacement that silently matches nothing would run the sweep in the
        # repository's own ./runs while every assertion below still passes.
        assert old in text, f"{EXAMPLE_YAML} no longer contains {old!r}; update _prep"
        text = text.replace(old, new)
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(text)
    return yaml_path


def test_full_sweep_and_replay(tmp_path):
    yaml_path = _prep(tmp_path)
    exp = load_experiment(yaml_path)

    winners = run_experiment(exp)

    # All three phases produced winners.
    assert set(winners) == {"depth", "lr", "regularization"}

    # Constraint should have excluded n_layers=16 (param_bytes = 17.6 MB > 16 MiB).
    depth_winner = winners["depth"]
    assert depth_winner.params["model.n_layers"] in {4, 8, 12}
    # The synthetic objective minimizes at n_layers=8.
    assert depth_winner.params["model.n_layers"] == 8

    # lr winner should be near 3e-4 (the synthetic optimum) within a decade.
    lr = winners["lr"].params["optimizer.lr"]
    assert 1e-5 < lr < 1e-2

    # regularization winner should be near (0.05, 0.10).
    reg = winners["regularization"].params
    assert 0.0 <= reg["optimizer.weight_decay"] <= 0.3
    assert 0.0 <= reg["model.dropout"] <= 0.3

    # Persistence on disk. v0.5.7: outputs are namespaced as
    # <workdir>/<experiment>/<phase>/, summary at <workdir>/<experiment>/summary.yaml.
    runs_dir = tmp_path / "runs"
    exp_dir = runs_dir / exp.experiment
    summary = yaml.safe_load((exp_dir / "summary.yaml").read_text())
    assert {p["name"] for p in summary["phases"]} == {"depth", "lr", "regularization"}

    # Frozen objective evidence provenance (review v0.5.17 / finding F): the
    # winner records the digest of the exact evidence file its scalar came from.
    provenance = depth_winner.objective_provenance
    assert provenance is not None
    assert provenance["extractor"]["kind"] == "json_envelope"
    winner_trial_dirs = [
        d
        for d in (exp_dir / "depth").glob("trial_*")
        if d.name.endswith(f"attempt_{depth_winner.attempt_id}")
    ]
    assert len(winner_trial_dirs) == 1
    evidence = winner_trial_dirs[0] / provenance["source"]["path"]
    assert provenance["source"]["sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    stored = yaml.safe_load((exp_dir / "depth" / "winner.yaml").read_text())
    assert stored["objective_provenance"] == provenance

    # Replay from regularization with its durable study intact. The depth and
    # lr winners are re-loaded from the prior publication without re-running.
    winners2 = run_experiment(exp, from_phase="regularization")
    assert winners2["depth"].params == winners["depth"].params  # loaded from disk
    assert winners2["lr"].params == winners["lr"].params  # loaded from disk
    assert winners2["regularization"].params == winners["regularization"].params
    # Reloaded winners preserve the frozen provenance record verbatim.
    assert winners2["depth"].objective_provenance == provenance


def test_bound_topup_survives_child_preflight_failure_with_empty_fingerprinted_study(
    tmp_path: Path,
) -> None:
    """An upstream top-up is not blocked by a fingerprinted child with no trials.

    ``_resume_phase`` stamps a child study's fingerprint before
    ``GpuPool.create``, so a child that fails GPU preflight leaves a
    fingerprinted study with zero trials. No child result exists whose meaning
    could change, so the parent top-up completes and the child binds to the
    new inherited configuration.
    """
    trainer = write_constant_trainer(tmp_path)
    parent = Phase(
        name="parent",
        n_trials=1,
        sampler=Sampler(type="random", seed=0),
        gpu_policy="none",
        search_space={"x": IntParam(type="int", low=0, high=9)},
    )
    child = Phase(
        name="child",
        inherits=["parent"],
        n_trials=1,
        sampler=Sampler(type="random", seed=0),
        gpu_policy="none",
        search_space={},
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, phases=[parent, child])
    storage = _resolve_storage(experiment.resolved_storage)
    real_create = GpuPool.create
    calls = 0

    def fail_child_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected child GPU preflight failure")
        return real_create(**kwargs)

    with (
        patch("phasesweep.engine.phase.GpuPool.create", side_effect=fail_child_once),
        pytest.raises(RuntimeError, match="injected child GPU preflight failure"),
    ):
        run_experiment(experiment)

    child_study = optuna.load_study(study_name="t::child", storage=storage)
    assert child_study.get_trials(deepcopy=False) == []
    assert isinstance(child_study.user_attrs.get(PHASE_FINGERPRINT_ATTR), str)

    # No GpuPool patch here: the parent top-up must succeed outright now.
    topped_up = experiment.model_copy(
        update={"phases": [parent.model_copy(update={"n_trials": 2}), child]}
    )
    winners = run_experiment(topped_up)

    parent_study = optuna.load_study(study_name="t::parent", storage=storage)
    child_study_after = optuna.load_study(study_name="t::child", storage=storage)
    assert len(parent_study.trials) == 2
    assert child_study_after.get_trials(deepcopy=False)
    expected_fingerprint = _phase_fingerprint(topped_up, child, {"parent": winners["parent"]})
    assert child_study_after.user_attrs[PHASE_FINGERPRINT_ATTR] == expected_fingerprint
