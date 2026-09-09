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
from phasesweep.engine import read_status
from phasesweep.engine.run import experiment_status
from phasesweep.engine.state import _generation_winner_path, _last_successful_generation_id
from tests.conftest import write_trainer, write_yaml

EXPERIMENT_STATUS_KEYS = [
    "kind",
    "experiment",
    "workdir",
    "current_generation_id",
    "published_generation_id",
    "represented_generation_id",
    "is_published",
    "publication_integrity",
    "phases",
]
"""Exact ordered key set of an experiment status payload."""

FAILED_PUBLICATION_STATUS_KEYS = [
    "kind",
    "experiment",
    "workdir",
    "current_generation_id",
    "published_generation_id",
    "represented_generation_id",
    "is_published",
    "publication_integrity",
    "publication_error",
    "phases",
]
"""Ordered key set when the recorded publication no longer validates.

``publication_error`` is the one conditional key: it appears only alongside
``publication_integrity: "failed"`` or ``"permission_denied"`` so a healthy
payload carries no empty error field (review v0.5.18 / finding F4).
"""

SUITE_STATUS_KEYS = [
    "kind",
    "suite",
    "workdir",
    "published_suite_generation_id",
    "publication_integrity",
    "studies",
]
"""Exact ordered key set of a suite status envelope.

The envelope reports the *suite* last-success pointer's own four-state verdict
beside the per-study payloads (re-review v0.5.19 / observation N2), with
``publication_error`` as the same one conditional key the experiment payload
carries.
"""

PHASE_STATUS_KEYS = {
    "trials",
    "running",
    "n_trials",
    "completed",
    "generation_trials",
    "published_study_unavailable",
    "trial_data_available",
    "name",
    "winner",
}
"""Exact key set of one phase payload inside an experiment status payload."""

# read_status computes these for the MCP read view; experiment_status
# deliberately does not republish them under the path-bearing CLI contract.
# ``result_phase_plan`` is the represented generation's own phase plan, which
# the CLI reads straight from the generation summary in ``show-winners``
# instead (review v0.5.16 / blocker 4).
READ_STATUS_ONLY_KEYS = {
    "metric",
    "result_context",
    "published_config_matches_current",
    "result_phase_plan",
    "summary_present",
}

READ_STATUS_ONLY_PHASE_KEYS = {
    "phase",
    "winner_present",
    "running_attempts",
}
"""Phase keys that exist only in the path-free read_status view.

``running_attempts`` is the identity of each RUNNING row in the same storage
snapshot the counts came from. The MCP terminal snapshot consumes it so it
never has to reread a study to reconcile RUNNING trials (PR #5 review /
reviewer 2, blocker 6); the CLI phase payload keeps exactly
``PHASE_STATUS_KEYS`` and republishes none of it.
"""


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
          override_format: argparse
          metric:
            name: x
            goal: minimize
            extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
        studies:
          - name: ran
            phases:
              - name: p
                n_trials: 1
                sampler: {{ type: random, seed: 0 }}
                search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
          - name: untouched
            phases:
              - name: p
                n_trials: 1
                sampler: {{ type: random, seed: 0 }}
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
    assert payload["publication_integrity"] == "ok"
    assert payload["published_generation_id"] == payload["current_generation_id"]
    assert payload["represented_generation_id"] == payload["published_generation_id"]

    (phase,) = payload["phases"]
    assert set(phase) == PHASE_STATUS_KEYS
    assert not READ_STATUS_ONLY_PHASE_KEYS & set(phase)
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

    assert list(payload) == SUITE_STATUS_KEYS
    assert payload["kind"] == "suite"
    assert payload["suite"] == "shape_suite"
    # Only a component study ran, so the suite itself has published nothing --
    # reported as its own verdict rather than left unstated.
    assert payload["publication_integrity"] == "absent"
    assert payload["published_suite_generation_id"] is None
    assert "publication_error" not in payload

    for study_payload in payload["studies"]:
        assert list(study_payload) == ["name", "depends_on", "status"]
        status = study_payload["status"]
        assert list(status) == EXPERIMENT_STATUS_KEYS
        assert not READ_STATUS_ONLY_KEYS & set(status)
        for phase in status["phases"]:
            assert set(phase) == PHASE_STATUS_KEYS
            assert not READ_STATUS_ONLY_PHASE_KEYS & set(phase)

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
    assert untouched_status["publication_integrity"] == "absent"
    assert untouched_status["phases"][0]["generation_trials"] == {}
    assert untouched_status["phases"][0]["winner"] is None


def test_read_status_only_phase_keys_never_reach_the_cli_contract(tmp_path: Path) -> None:
    """The path-free phase view carries running attempts; the CLI view does not.

    Both directions matter. The MCP terminal snapshot depends on
    ``running_attempts`` arriving from the same tolerant read as the counts
    (PR #5 review / reviewer 2, blocker 6), and ``experiment_status`` is a
    pinned public contract that must not grow a key because of it.
    """
    config = load_config(_write_suite(tmp_path))
    assert isinstance(config, Suite)
    experiment = config.experiment_for_study(config.studies[0])
    run_experiment(experiment)

    (read_phase,) = read_status(experiment)["phases"]
    (cli_phase,) = experiment_status(experiment)["phases"]

    assert set(read_phase) >= READ_STATUS_ONLY_PHASE_KEYS
    assert read_phase["trial_data_available"] is True
    assert read_phase["running_attempts"] == []
    assert set(cli_phase) == PHASE_STATUS_KEYS


def test_status_shape_reports_a_corrupt_publication_without_fabricating_results(
    tmp_path: Path,
) -> None:
    """A publication that no longer validates adds two keys and invents nothing.

    Review v0.5.18 / finding F4: the payload for a corrupt publication used to
    be byte-identical to a tree that had never published, so the operator's
    natural next move was to re-run over the evidence.
    """
    config = load_config(_write_suite(tmp_path))
    assert isinstance(config, Suite)
    experiment = config.experiment_for_study(config.studies[0])
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    payload = config_status(experiment)

    assert list(payload) == FAILED_PUBLICATION_STATUS_KEYS
    assert not READ_STATUS_ONLY_KEYS & set(payload)
    assert payload["publication_integrity"] == "failed"
    assert "does not match its recorded hash" in payload["publication_error"]
    assert payload["published_generation_id"] is None
    assert payload["represented_generation_id"] is None
    assert payload["is_published"] is False
    assert payload["phases"][0]["winner"] is None
