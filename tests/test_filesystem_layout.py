"""On-disk layout: <workdir>/<experiment>/<phase>/ namespacing, summary.yaml placement, experiment-name validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.config import (
    Experiment,
    IntParam,
    JsonExtractor,
    Metric,
    Phase,
)
from phasesweep.orchestrator import (
    _experiment_dir,
    _phase_dir,
    _summary_path,
    run_experiment,
)
from tests.conftest import make_experiment, write_constant_trainer


def test_experiment_dir_is_namespaced_by_experiment_name(tmp_path: Path) -> None:
    """``<workdir>/<experiment>/...`` is the v0.5.7 output layout."""
    exp = make_experiment(workdir=str(tmp_path / "runs"))
    assert _experiment_dir(exp) == (tmp_path / "runs" / exp.experiment).resolve()


def test_phase_dir_lives_under_experiment_dir(tmp_path: Path) -> None:
    exp = make_experiment(workdir=str(tmp_path / "runs"))
    assert _phase_dir(exp, "p") == _experiment_dir(exp) / "p"


def test_summary_path_lives_under_experiment_dir(tmp_path: Path) -> None:
    exp = make_experiment(workdir=str(tmp_path / "runs"))
    assert _summary_path(exp) == _experiment_dir(exp) / "summary.yaml"


def test_two_experiments_sharing_workdir_have_disjoint_output_trees(
    tmp_path: Path,
) -> None:
    """Two configs with the same ``workdir`` but different experiment names
    must not share any output paths — pre-v0.5.7 they did, which let one run
    silently overwrite the other's ``trial_*/``, ``winner.yaml``, and
    ``summary.yaml``.
    """
    exp_a = make_experiment(workdir=str(tmp_path / "runs"))
    exp_b = make_experiment(workdir=str(tmp_path / "runs"))
    exp_b = exp_b.model_copy(update={"experiment": "other"})

    a_dir = _experiment_dir(exp_a)
    b_dir = _experiment_dir(exp_b)
    assert a_dir != b_dir
    # Neither path is a prefix of the other.
    assert not str(a_dir).startswith(str(b_dir) + "/")
    assert not str(b_dir).startswith(str(a_dir) + "/")


def test_run_experiment_writes_summary_at_namespaced_path(tmp_path: Path) -> None:
    """End-to-end: a real run must write ``summary.yaml`` under the
    ``<workdir>/<experiment>/`` tree, not directly under ``<workdir>``.
    """
    trainer = write_constant_trainer(tmp_path)
    exp = make_experiment(
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
    )
    run_experiment(exp)
    assert (_summary_path(exp)).is_file()
    # The pre-v0.5.7 location must NOT be created.
    assert not (Path(exp.workdir).resolve() / "summary.yaml").exists()


@pytest.mark.parametrize(
    "bad_name",
    [
        "../../etc/evil",  # path separators escape the workdir
        "my experiment",  # whitespace breaks lock-file paths
    ],
)
def test_experiment_name_rejected(bad_name: str) -> None:
    """Experiment name is used in lock-file paths; unsafe characters break that."""
    with pytest.raises(ValueError, match="Experiment name"):
        Experiment(
            experiment=bad_name,
            trial_command="echo {overrides}",
            metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
            phases=[
                Phase(  # type: ignore[arg-type]
                    name="p",
                    n_trials=1,
                    search_space={"x": IntParam(type="int", low=0, high=1)},
                )
            ],
        )


def test_experiment_name_accepts_valid() -> None:
    """Alphanumeric, underscore, and hyphen are safe."""
    Experiment(
        experiment="tiny_lm-16mb",
        trial_command="echo {overrides}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        phases=[
            Phase(  # type: ignore[arg-type]
                name="p",
                n_trials=1,
                search_space={"x": IntParam(type="int", low=0, high=1)},
            )
        ],
    )
