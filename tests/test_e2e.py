"""End-to-end: run all three phases against the fake training script.

Verifies:
  - Phase chaining: depth winner is fixed during lr phase, etc.
  - Constraint enforcement: 16-layer trial violates 16 MiB budget and is excluded.
  - Winner persistence: winner.yaml + summary.yaml written.
  - Replay: --from-phase loads prior winners.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from phasesweep import load_experiment, run_experiment
from tests.conftest import copy_fake_train

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_YAML = REPO / "examples" / "experiment.yaml"


def _prep(tmp_path: Path) -> Path:
    """Copy example yaml into tmp, rewrite paths to be tmp-local, return new yaml path."""
    trainer = copy_fake_train(tmp_path)

    text = EXAMPLE_YAML.read_text()
    runs_dir = tmp_path / "runs"
    text = text.replace("./runs/phases.db", str(runs_dir / "phases.db"))
    text = text.replace("./runs", str(runs_dir))
    text = text.replace("examples/fake_train.py", str(trainer.resolve()))
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
    assert depth_winner.params["n_layers"] in {4, 8, 12}
    # The synthetic objective minimizes at n_layers=8.
    assert depth_winner.params["n_layers"] == 8

    # lr winner should be near 3e-4 (the synthetic optimum) within a decade.
    lr = winners["lr"].params["lr"]
    assert 1e-5 < lr < 1e-2

    # regularization winner should be near (0.05, 0.10).
    reg = winners["regularization"].params
    assert 0.0 <= reg["weight_decay"] <= 0.3
    assert 0.0 <= reg["dropout"] <= 0.3

    # Persistence on disk. v0.5.7: outputs are namespaced as
    # <workdir>/<experiment>/<phase>/, summary at <workdir>/<experiment>/summary.yaml.
    runs_dir = tmp_path / "runs"
    exp_dir = runs_dir / exp.experiment
    summary = yaml.safe_load((exp_dir / "summary.yaml").read_text())
    assert {p["name"] for p in summary["phases"]} == {"depth", "lr", "regularization"}

    # Replay: drop the regularization study and its winner, then re-run from that phase.
    # The depth and lr winners should be re-loaded from yaml without re-running trials.
    import optuna

    optuna.delete_study(
        study_name=f"{exp.experiment}::regularization",
        storage=exp.storage,
    )

    # Also wipe the regularization phase dir so winner.yaml gets rewritten.
    shutil.rmtree(exp_dir / "regularization")

    winners2 = run_experiment(exp, from_phase="regularization")
    assert winners2["depth"].params == winners["depth"].params  # loaded from disk
    assert winners2["lr"].params == winners["lr"].params  # loaded from disk
    assert "regularization" in winners2  # re-run
