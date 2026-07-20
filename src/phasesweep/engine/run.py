"""Public engine entrypoints for experiments, suites, and status."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from phasesweep._metadata import __version__
from phasesweep.config import Config, Experiment, Suite
from phasesweep.engine.guards import (
    _experiment_lock,
    _preflight_existing_studies,
    _suite_lock,
    _verify_fingerprint,
)
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
    _generation_path,
    _generation_promotion_decision_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_path,
    _load_winner,
    _promotion_decision_path,
    _run_log_path,
    _save_promotion_decision,
    _save_winner,
    _suite_dir,
    _suite_log_path,
    _suite_summary_path,
    _summary_path,
    _winner_path,
    _write_yaml_atomic,
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
    terminal_callback: Callable[[str, BaseException | None], None] | None = None,
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
    callers. Skipped on dry-run because no children will launch.

    Args:
        experiment: Parsed experiment config (result of
            :func:`phasesweep.load_experiment`).
        from_phase: Name of a phase to resume from; earlier phases are loaded
            from their persisted ``winner.yaml`` files (with fingerprint
            verification). ``None`` runs every phase from scratch.
        dry_run: If ``True``, render and log one example trial command per
            phase but launch no subprocesses; no summary is written.
        terminal_callback: Optional synchronous callback invoked with the
            generation id and terminal exception, if any, while the experiment
            lock is still held.

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
        return _run_experiment_inner(
            experiment,
            from_phase=from_phase,
            dry_run=True,
            generation_id=None,
        )

    _experiment_dir(experiment).mkdir(parents=True, exist_ok=True)
    generation_id = uuid4().hex
    with _file_log_handler(_run_log_path(experiment)), _experiment_lock(experiment):
        terminal_error: BaseException | None = None
        existing_studies: dict[str, Any] = {}
        generation_prepared = False
        try:
            _write_generation_state(
                experiment,
                generation_id=generation_id,
                state="preflighting",
                from_phase=from_phase,
                publish_current=False,
            )
            run_deadline = (
                time.monotonic() + experiment.timeout_seconds_per_run
                if from_phase is not None and experiment.timeout_seconds_per_run is not None
                else None
            )
            existing_studies = _preflight_existing_studies(experiment)
            preloaded_winners = _preflight_skipped_winners(
                experiment,
                from_phase=from_phase,
                run_deadline=run_deadline,
            )
            _preflight_reached_fingerprint(
                experiment,
                from_phase=from_phase,
                preloaded_winners=preloaded_winners,
                existing_studies=existing_studies,
            )
            _prepare_generation(experiment, from_phase=from_phase, generation_id=generation_id)
            generation_prepared = True
            return _run_experiment_inner(
                experiment,
                from_phase=from_phase,
                dry_run=False,
                generation_id=generation_id,
                preloaded_winners=preloaded_winners,
                existing_studies=existing_studies,
                run_deadline=run_deadline,
            )
        except BaseException as exc:
            terminal_error = exc
            if generation_prepared:
                try:
                    _preflight_existing_studies(experiment)
                except BaseException:
                    log.exception("failed to reconcile all existing studies after run termination")
            with contextlib.suppress(Exception):
                _write_generation_state(
                    experiment,
                    generation_id=generation_id,
                    state="failed",
                    from_phase=from_phase,
                    publish_current=generation_prepared,
                    error_class=type(exc).__name__,
                )
            raise
        finally:
            if terminal_callback is not None:
                try:
                    terminal_callback(generation_id, terminal_error)
                except BaseException:
                    if terminal_error is None:
                        raise
                    log.exception(
                        "terminal callback failed while preserving the engine's original error"
                    )


def _run_experiment_inner(
    experiment: Experiment,
    *,
    from_phase: str | None,
    dry_run: bool,
    generation_id: str | None,
    preloaded_winners: dict[str, Winner] | None = None,
    existing_studies: dict[str, Any] | None = None,
    run_deadline: float | None = None,
) -> dict[str, Winner]:
    """Sequential phase loop assuming locks/signal handlers are already set up.

    Args:
        experiment: Parsed experiment config.
        from_phase: Optional name of the phase to resume from; earlier phases
            are loaded from disk.
        dry_run: If ``True``, no subprocesses launch and no ``summary.yaml`` is written.
        generation_id: Current invocation identity, or ``None`` for dry-run.
        preloaded_winners: Strictly validated skipped-phase winners loaded before
            the current generation was committed.
        existing_studies: Existing studies validated and reaped before launch.
        run_deadline: Optional precomputed whole-run monotonic deadline. Resume
            preflight passes this through so skipped-phase validation consumes
            the same budget as it did before preflight was separated.

    Returns:
        Same as :func:`run_experiment`: a phase-name to :class:`Winner` mapping.

    """
    skip_until = from_phase is not None
    winners: dict[str, Winner] = {}
    promotion_decisions: dict[str, dict[str, Any]] = {}
    if run_deadline is None and not dry_run and experiment.timeout_seconds_per_run is not None:
        run_deadline = time.monotonic() + experiment.timeout_seconds_per_run

    for phase in experiment.phases:
        using_preloaded_winner = (
            skip_until and phase.name != from_phase and preloaded_winners is not None
        )
        if (
            not using_preloaded_winner
            and run_deadline is not None
            and time.monotonic() >= run_deadline
        ):
            raise TimeoutError(
                f"Run wallclock deadline reached before phase {phase.name!r} could start."
            )
        # Inherited winners must be resolved before either the skip-path winner
        # load (so we can verify its fingerprint against the *current* parent
        # context) or the actual run path. Keeping the construction in one
        # place makes the two paths symmetric.
        inherited = {p: winners[p] for p in phase.inherits}

        if skip_until and phase.name != from_phase:
            if preloaded_winners is not None:
                winners[phase.name] = preloaded_winners[phase.name]
                if not dry_run:
                    assert generation_id is not None
                    _save_winner(
                        experiment,
                        phase.name,
                        winners[phase.name],
                        generation_id=generation_id,
                    )
                    prior_promotion = _promotion_decision_path(experiment, phase.name)
                    if prior_promotion.is_file():
                        _copy_yaml_projection(
                            prior_promotion,
                            _generation_promotion_decision_path(
                                experiment, generation_id, phase.name
                            ),
                        )
                log.info("phase=%s SKIPPED (using preflight-validated winner)", phase.name)
            else:
                try:
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
            generation_id=generation_id,
            dry_run=dry_run,
            run_deadline=run_deadline,
            existing_study=(existing_studies or {}).get(phase.name),
        )
        if not dry_run:
            promoted, promotion_decision = _apply_promotion(experiment, phase, winner, winners)
            if promotion_decision is not None:
                promotion_decision["generation_id"] = generation_id
                promotion_decisions[phase.name] = promotion_decision
                _save_promotion_decision(
                    experiment,
                    phase.name,
                    promotion_decision,
                    generation_id=generation_id,
                )
            if promoted is None:
                if promotion_decision is not None and promotion_decision["action"] == "stop":
                    raise RuntimeError(str(promotion_decision["message"]))
                break
            winner = promoted
            assert generation_id is not None
            _save_winner(
                experiment,
                phase.name,
                winner,
                generation_id=generation_id,
            )
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

    assert generation_id is not None
    summary_path = _generation_summary_path(experiment, generation_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "metric": {"name": experiment.metric.name, "goal": experiment.metric.goal},
        "promotion_decisions": list(promotion_decisions.values()),
        "phases": [_winner_summary_item(pname, w) for pname, w in winners.items()],
    }
    _write_yaml_atomic(summary_path, summary)
    _publish_generation(experiment, generation_id, from_phase=from_phase)
    log.info("Wrote %s", summary_path)

    return winners


def _preflight_skipped_winners(
    experiment: Experiment,
    *,
    from_phase: str | None,
    run_deadline: float | None,
) -> dict[str, Winner]:
    """Validate skipped winners before committing a new generation.

    :param Experiment experiment: Experiment whose resume prefix is checked.
    :param str | None from_phase: First phase that the new generation will execute.
    :param float | None run_deadline: Whole-run monotonic deadline, when configured.
    :return dict[str, Winner]: Compatible skipped winners in declaration order.
    """
    if from_phase is None:
        return {}

    winners: dict[str, Winner] = {}
    for phase in experiment.phases:
        if run_deadline is not None and time.monotonic() >= run_deadline:
            raise TimeoutError(
                f"Run wallclock deadline reached before phase {phase.name!r} could start."
            )
        if phase.name == from_phase:
            return winners
        inherited = {parent: winners[parent] for parent in phase.inherits}
        winners[phase.name] = _load_winner(experiment, phase, inherited)

    raise ValueError(f"Unknown --from-phase value {from_phase!r}.")


def _preflight_reached_fingerprint(
    experiment: Experiment,
    *,
    from_phase: str | None,
    preloaded_winners: dict[str, Winner],
    existing_studies: dict[str, Any],
) -> None:
    """Verify the first reached study before publishing the new generation."""
    phase = experiment.phases[0]
    if from_phase is not None:
        phase = next(item for item in experiment.phases if item.name == from_phase)
    study = existing_studies.get(phase.name)
    if study is None:
        return
    inherited = {name: preloaded_winners[name] for name in phase.inherits}
    _verify_fingerprint(study, experiment, phase, inherited)


def _prepare_generation(
    experiment: Experiment,
    *,
    from_phase: str | None,
    generation_id: str,
) -> None:
    """Publish a generation as current after recovery and initial preflight.

    :param Experiment experiment: Experiment whose current generation pointer is advanced.
    :param str | None from_phase: Optional resume point; prior winners remain available.
    :param str generation_id: Fresh identity for this engine invocation.
    """
    _write_generation_state(
        experiment,
        generation_id=generation_id,
        state="running",
        from_phase=from_phase,
        publish_current=True,
    )


def _write_generation_state(
    experiment: Experiment,
    *,
    generation_id: str,
    state: str,
    from_phase: str | None,
    publish_current: bool,
    error_class: str | None = None,
) -> None:
    """Write one generation lifecycle record and optionally its current pointer."""
    payload = {
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "state": state,
        "from_phase": from_phase,
        "error_class": error_class,
        "phasesweep_version": __version__,
    }
    _write_yaml_atomic(_generation_record_path(experiment, generation_id), payload)
    if publish_current:
        _write_yaml_atomic(_generation_path(experiment), payload)


def _copy_yaml_projection(source: Path, destination: Path) -> None:
    """Atomically project one immutable YAML artifact to its compatibility path."""
    payload = yaml.safe_load(source.read_text())
    _write_yaml_atomic(destination, payload)


def _publish_generation(
    experiment: Experiment,
    generation_id: str,
    *,
    from_phase: str | None,
) -> None:
    """Publish a complete immutable generation as the last successful result."""
    for phase in experiment.phases:
        source_winner = _generation_winner_path(experiment, generation_id, phase.name)
        projected_winner = _winner_path(experiment, phase.name)
        if source_winner.is_file():
            _copy_yaml_projection(source_winner, projected_winner)
        else:
            projected_winner.unlink(missing_ok=True)

        source_promotion = _generation_promotion_decision_path(
            experiment, generation_id, phase.name
        )
        projected_promotion = _promotion_decision_path(experiment, phase.name)
        if source_promotion.is_file():
            _copy_yaml_projection(source_promotion, projected_promotion)
        else:
            projected_promotion.unlink(missing_ok=True)

    _copy_yaml_projection(
        _generation_summary_path(experiment, generation_id),
        _summary_path(experiment),
    )
    _write_generation_state(
        experiment,
        generation_id=generation_id,
        state="complete",
        from_phase=from_phase,
        publish_current=False,
    )
    _write_yaml_atomic(
        _last_successful_generation_path(experiment),
        {"experiment": experiment.experiment, "generation_id": generation_id},
    )
    _write_generation_state(
        experiment,
        generation_id=generation_id,
        state="complete",
        from_phase=from_phase,
        publish_current=True,
    )


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
        _suite_summary_path(suite).unlink(missing_ok=True)
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
