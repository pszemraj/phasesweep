"""Public shape contract for ``config_status`` / ``experiment_status``.

``config_status`` is a documented package-root API (docs/development.md) and
``phasesweep status`` renders its payload verbatim, so its key set is a
contract for downstream consumers. The suite branch embeds
``experiment_status`` under every study, which means *any* change to the
experiment payload silently changes the suite payload too. These tests pin
both key sets — experiment and suite — so that coupling can never drift
unnoticed again (review finding 11).
"""

from __future__ import annotations

from pathlib import Path

from phasesweep import config_status, load_config, run_experiment
from phasesweep.config import Suite
from phasesweep.engine.run import experiment_status
from tests.conftest import write_trainer, write_yaml

EXPERIMENT_STATUS_KEYS = [
    "kind",
    "experiment",
    "workdir",
    "current_generation_id",
    "published_generation_id",
    "represented_generation_id",
    "is_published",
    "phases",
]
"""Exact ordered key set of an experiment status payload."""

PHASE_STATUS_KEYS = {
    "trials",
    "running",
    "n_trials",
    "completed",
    "generation_trials",
    "name",
    "winner",
}
"""Exact key set of one phase payload inside an experiment status payload."""

# read_status computes these for the MCP read view; experiment_status
# deliberately does not republish them under the path-bearing CLI contract.
READ_STATUS_ONLY_KEYS = {
    "metric",
    "result_context",
    "published_config_matches_current",
    "summary_present",
}


def _write_suite(tmp_path: Path) -> Path:
    """Write a two-study suite whose trainer reports a constant objective."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
        args, _ = ap.parse_known_args()
        open(args.out, 'w').write(json.dumps({'x': 1.0}))
        print('x=1.0')
        """,
    )
    return write_yaml(
        tmp_path,
        f"""
        suite: shape_suite
        defaults:
          workdir: {tmp_path}/runs
          storage: sqlite:///{tmp_path}/shape.db
          provenance: {{revision: test-fixture-v1}}
          trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
          metric:
            name: x
            goal: minimize
            extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
        studies:
          - name: ran
            phases:
              - name: p
                n_trials: 1
                search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
          - name: untouched
            phases:
              - name: p
                n_trials: 1
                search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
        """,
    )


def test_experiment_config_status_shape_is_pinned(tmp_path: Path) -> None:
    """A standalone experiment payload carries generation identity plus phases."""
    config = load_config(_write_suite(tmp_path))
    assert isinstance(config, Suite)
    experiment = config.experiment_for_study(config.studies[0])
    run_experiment(experiment)

    payload = config_status(experiment)

    assert list(payload) == EXPERIMENT_STATUS_KEYS
    assert not READ_STATUS_ONLY_KEYS & set(payload)
    assert payload["kind"] == "experiment"
    assert payload["is_published"] is True
    assert payload["published_generation_id"] == payload["current_generation_id"]
    assert payload["represented_generation_id"] == payload["published_generation_id"]

    (phase,) = payload["phases"]
    assert set(phase) == PHASE_STATUS_KEYS
    assert phase["name"] == "p"
    assert phase["winner"] is not None
    assert phase["completed"] == 1


def test_suite_config_status_embeds_the_full_experiment_status(tmp_path: Path) -> None:
    """Every suite study embeds the same payload a standalone experiment reports.

    Including the four generation-identity keys and current-generation trial
    counts: a suite study is not a reduced view of an experiment.
    """
    config = load_config(_write_suite(tmp_path))
    assert isinstance(config, Suite)
    ran_study, untouched_study = config.studies
    ran = config.experiment_for_study(ran_study)
    run_experiment(ran)

    payload = config_status(config)

    assert list(payload) == ["kind", "suite", "workdir", "studies"]
    assert payload["kind"] == "suite"
    assert payload["suite"] == "shape_suite"

    for study_payload in payload["studies"]:
        assert list(study_payload) == ["name", "depends_on", "status"]
        status = study_payload["status"]
        assert list(status) == EXPERIMENT_STATUS_KEYS
        assert not READ_STATUS_ONLY_KEYS & set(status)
        for phase in status["phases"]:
            assert set(phase) == PHASE_STATUS_KEYS

    ran_status, untouched_status = (study["status"] for study in payload["studies"])

    # The embedded payload is exactly what the standalone call returns.
    assert ran_status == experiment_status(ran)
    assert untouched_status == experiment_status(config.experiment_for_study(untouched_study))

    # The study that ran reports its own generation identity and this
    # generation's trial counts, not an empty placeholder.
    assert ran_status["current_generation_id"] is not None
    assert ran_status["is_published"] is True
    assert ran_status["phases"][0]["generation_trials"] == {"COMPLETE": 1}
    assert ran_status["phases"][0]["winner"] is not None

    # A study that never ran reports null identity and no winner, without
    # borrowing the sibling study's generation.
    assert untouched_status["current_generation_id"] is None
    assert untouched_status["published_generation_id"] is None
    assert untouched_status["represented_generation_id"] is None
    assert untouched_status["is_published"] is False
    assert untouched_status["phases"][0]["generation_trials"] == {}
    assert untouched_status["phases"][0]["winner"] is None
