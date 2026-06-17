"""Public engine entrypoints for experiments, suites, and status."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from phasesweep.config import Config, Experiment, Suite
from phasesweep.engine.guards import _experiment_lock, _reap_skipped_phase, _suite_lock
from phasesweep.engine.phase import _placeholder_winner, _run_phase
from phasesweep.engine.read import _phase_status_payloads
from phasesweep.engine.selection import (
    _apply_promotion,
    _apply_study_promotion,
    _winner_summary_item,
)
from phasesweep.engine.state import (
    Winner,
    _experiment_dir,
    _file_log_handler,
    _load_winner,
    _run_log_handler,
    _save_promotion_decision,
    _save_winner,
    _suite_dir,
    _suite_log_path,
    _suite_summary_path,
    _summary_path,
    _write_yaml_atomic,
    _winner_path,
)
from phasesweep.runtime.files import require_posix_runtime
from phasesweep.runtime.process import install_signal_handlers


def run_config(
    config: Config,
    *,
    from_phase: str | None = None,
    dry_run: bool = False,
) -> dict[str, Winner] | dict[str, dict[str, Winner]]:
    """Run an experiment or suite config.

    :param Config config: Parsed experiment or suite config.
    :param str | None from_phase: Optional phase name to resume from for experiment configs.
    :param bool dry_run: If ``True``, preview commands without launching subprocesses.
    :return dict[str, Winner] | dict[str, dict[str, Winner]]: Experiment winners, or suite
        study winners keyed by study name.
    """
    if isinstance(config, Suite):
        if from_phase is not None:
            raise RuntimeError("--from-phase is only supported for single experiment configs.")
        return run_suite(config, dry_run=dry_run)
    return run_experiment(config, from_phase=from_phase, dry_run=dry_run)


def config_status(config: Config) -> dict[str, Any]:
    """Collect read-only status for an experiment or suite config.

    :param Config config: Parsed experiment or suite config to inspect.
    :return dict[str, Any]: Read-only status payload for the config.
    """
    if isinstance(config, Suite):
        return {
            "kind": "suite",
            "suite": config.suite,
            "workdir": str(_suite_dir(config)),
            "studies": [
                {
                    "name": study.name,
                    "depends_on": study.depends_on,
                    "status": experiment_status(config.experiment_for_study(study)),
                }
                for study in config.studies
            ],
        }
    return experiment_status(config)


log = logging.getLogger("phasesweep.engine.run")


def run_experiment(
    experiment: Experiment,
    *,
    from_phase: str | None = None,
    dry_run: bool = False,
) -> dict[str, Winner]:
    """Run all phases in order, returning a map of phase name to Winner.

    If ``from_phase`` is given, prior phases are loaded from disk.
    If ``dry_run`` is True, example commands are logged but nothing launches.

    For non-dry-run invocations this acquires an :func:`_experiment_lock` for
    the entire phase sequence (review v0.5.6 / blocker 1). Phase-level state
    on disk (``winner.yaml``, fingerprints, ``summary.yaml``) is consistent
    only at run granularity, so two same-experiment processes must not
    interleave across phases.

    Signal handlers are installed here (review v0.5.7 / blocker 3) so library
    callers using the public API get the same cleanup guarantees as CLI
    callers. ``install_signal_handlers()`` is idempotent and a no-op when
    invoked from a non-main thread, so re-installation by the CLI is safe.
    Skipped on dry-run because no children will launch.

    Args:
        experiment: Parsed experiment config (result of
            :func:`phasesweep.load_experiment`).
        from_phase: Name of a phase to resume from; earlier phases are loaded
            from their persisted ``winner.yaml`` files (with fingerprint
            verification). ``None`` runs every phase from scratch.
        dry_run: If ``True``, render and log one example trial command per
            phase but launch no subprocesses; no summary is written.

    Returns:
        Mapping from phase name (in declaration order) to that phase's
        :class:`Winner`. For dry runs the winners are midpoint placeholders.

    Raises:
        NoFeasibleTrialError: A phase exhausted ``max_consecutive_failures``
            with no feasible trial.
        UnsafeProcessCleanupError: A phase hard-aborted because a trial's
            process group could not be confirmed dead (review v0.5.11).
        RuntimeError: Lock contention (another orchestrator running),
            fingerprint mismatch on ``--from-phase`` resume, or stale-reaper
            uncertainty (review v0.5.7 / blocker 2).
        FileNotFoundError: ``--from-phase`` requested but a prior phase has
            no persisted ``winner.yaml``.

    """
    if not dry_run:
        require_posix_runtime()
        install_signal_handlers()

    if dry_run:
        return _run_experiment_inner(experiment, from_phase=from_phase, dry_run=True)

    _experiment_dir(experiment).mkdir(parents=True, exist_ok=True)
    with _run_log_handler(experiment), _experiment_lock(experiment):
        return _run_experiment_inner(experiment, from_phase=from_phase, dry_run=False)


def _run_experiment_inner(
    experiment: Experiment,
    *,
    from_phase: str | None,
    dry_run: bool,
) -> dict[str, Winner]:
    """Sequential phase loop assuming locks/signal handlers are already set up.

    Args:
        experiment: Parsed experiment config.
        from_phase: Optional name of the phase to resume from; earlier phases
            are loaded from disk.
        dry_run: If ``True``, no subprocesses launch and no ``summary.yaml`` is written.

    Returns:
        Same as :func:`run_experiment`: a phase-name to :class:`Winner` mapping.

    """
    skip_until = from_phase is not None
    winners: dict[str, Winner] = {}
    promotion_decisions: dict[str, dict[str, Any]] = {}
    run_deadline = (
        None
        if dry_run or experiment.timeout_seconds_per_run is None
        else time.monotonic() + experiment.timeout_seconds_per_run
    )

    for phase in experiment.phases:
        if run_deadline is not None and time.monotonic() >= run_deadline:
            raise TimeoutError(
                f"Run wallclock deadline reached before phase {phase.name!r} could start."
            )
        # Inherited winners must be resolved before either the skip-path winner
        # load (so we can verify its fingerprint against the *current* parent
        # context) or the actual run path. Keeping the construction in one
        # place makes the two paths symmetric.
        inherited = {p: winners[p] for p in phase.inherits}

        if skip_until and phase.name != from_phase:
            try:
                if not dry_run:
                    _reap_skipped_phase(experiment, phase)
                winners[phase.name] = _load_winner(experiment, phase, inherited)
                log.info("phase=%s SKIPPED (loaded compatible winner from disk)", phase.name)
            except FileNotFoundError:
                if not dry_run:
                    raise
                winners[phase.name] = _placeholder_winner(experiment, phase, inherited)
                log.info("phase=%s SKIPPED (DRY RUN placeholder)", phase.name)
            continue
        skip_until = False

        winner = _run_phase(
            experiment,
            phase,
            inherited,
            dry_run=dry_run,
            run_deadline=run_deadline,
        )
        if not dry_run:
            promoted, promotion_decision = _apply_promotion(experiment, phase, winner, winners)
            if promotion_decision is not None:
                promotion_decisions[phase.name] = promotion_decision
                _save_promotion_decision(experiment, phase.name, promotion_decision)
            if promoted is None:
                with contextlib.suppress(FileNotFoundError):
                    _winner_path(experiment, phase.name).unlink()
                if promotion_decision is not None and promotion_decision["action"] == "stop":
                    raise RuntimeError(str(promotion_decision["message"]))
                break
            winner = promoted
            _save_winner(experiment, phase.name, winner)
            log.info(
                "phase=%s WINNER trial=%d metric=%g params=%s",
                phase.name,
                winner.trial_number,
                winner.metric,
                winner.params,
            )
        winners[phase.name] = winner

    if dry_run:
        log.info("DRY RUN complete. No trials launched, no summary written.")
        return winners

    summary_path = _summary_path(experiment)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": experiment.experiment,
        "metric": {"name": experiment.metric.name, "goal": experiment.metric.goal},
        "promotion_decisions": list(promotion_decisions.values()),
        "phases": [_winner_summary_item(pname, w) for pname, w in winners.items()],
    }
    _write_yaml_atomic(summary_path, summary)
    log.info("Wrote %s", summary_path)

    return winners


def experiment_status(experiment: Experiment) -> dict[str, Any]:
    """Collect read-only status for one experiment config.

    :param Experiment experiment: Parsed experiment config to inspect.
    :return dict[str, Any]: Status payload including phase winner paths and trial counts.
    """
    return {
        "kind": "experiment",
        "experiment": experiment.experiment,
        "workdir": str(_experiment_dir(experiment)),
        "phases": _phase_status_payloads(experiment, include_winner_path=True),
    }


def run_suite(suite: Suite, *, dry_run: bool = False) -> dict[str, dict[str, Winner]]:
    """Run every study in a suite in dependency order.

    :param Suite suite: Parsed suite config.
    :param bool dry_run: If ``True``, preview each study without launching subprocesses.
    :return dict[str, dict[str, Winner]]: Winners keyed by study name, then phase name.
    """
    results: dict[str, dict[str, Winner]] = {}
    promotion_decisions: dict[str, dict[str, Any]] = {}
    if dry_run:
        for study_spec in suite.studies:
            experiment = suite.experiment_for_study(study_spec)
            results[study_spec.name] = run_experiment(experiment, dry_run=True)
        return results

    require_posix_runtime()
    _suite_dir(suite).mkdir(parents=True, exist_ok=True)
    with _suite_lock(suite), _file_log_handler(_suite_log_path(suite)):
        for study_spec in suite.studies:
            for dep in study_spec.depends_on:
                if dep not in results:
                    raise RuntimeError(
                        f"Study {study_spec.name!r} dependency {dep!r} did not complete."
                    )
            experiment = suite.experiment_for_study(study_spec)
            log.info("suite=%s study=%s START", suite.suite, study_spec.name)
            study_winners = run_experiment(experiment, dry_run=False)
            exposed_winners, decision = _apply_study_promotion(
                suite=suite,
                study_name=study_spec.name,
                experiment=experiment,
                study_winners=study_winners,
                prior_results=results,
            )
            if decision is not None:
                promotion_decisions[study_spec.name] = decision
            if exposed_winners is not None:
                results[study_spec.name] = exposed_winners
            log.info("suite=%s study=%s COMPLETE", suite.suite, study_spec.name)

        summary = {
            "suite": suite.suite,
            "promotion_decisions": list(promotion_decisions.values()),
            "studies": [
                {
                    "name": study_name,
                    "promotion": promotion_decisions.get(study_name),
                    "phases": [
                        _winner_summary_item(phase_name, winner)
                        for phase_name, winner in study_winners.items()
                    ],
                }
                for study_name, study_winners in results.items()
            ],
        }
        _write_yaml_atomic(_suite_summary_path(suite), summary)
    return results
