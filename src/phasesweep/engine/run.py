"""Public engine entrypoints for experiments, suites, and status."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from phasesweep._metadata import __version__
from phasesweep.config import Config, Experiment, Suite
from phasesweep.config.common import _validate_safe_name
from phasesweep.engine.errors import StudyContextConflictError
from phasesweep.engine.guards import (
    _experiment_lock,
    _preflight_existing_studies,
    _PreflightCleanupReport,
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
    _generation_dir,
    _generation_path,
    _generation_promotion_decision_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _generations_dir,
    _last_successful_generation_path,
    _load_winner,
    _promotion_decision_path,
    _published_promotion_decision_path,
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
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.files import require_posix_runtime
from phasesweep.runtime.process import PhaseSweepShutdown, install_signal_handlers


@dataclass(frozen=True)
class TerminalReport:
    """One engine invocation's primary outcome and independent cleanup evidence."""

    generation_id: str
    primary_error: BaseException | None
    cleanup_confirmed: bool
    recovered_attempt_ids: frozenset[str]
    uncertain_attempt_ids: frozenset[str]
    cleanup_error: BaseException | None = None
    failure_stage: str | None = None


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
    terminal_callback: Callable[[TerminalReport], None] | None = None,
    generation_id: str | None = None,
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
            structured terminal report while the experiment lock is still held.
        generation_id: Optional caller-owned invocation identity. Detached MCP
            runs use their run id; direct callers receive a generated identity.
            Supplied values must contain only alphanumerics, underscores, and dashes.

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
        ValueError: A caller-supplied generation id is not a safe filesystem name.
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

    requested_generation_id = (
        None if generation_id is None else _validate_safe_name("generation", generation_id)
    )
    _experiment_dir(experiment).mkdir(parents=True, exist_ok=True)
    with _file_log_handler(_run_log_path(experiment)), _experiment_lock(experiment):
        generation_id = _claim_generation(experiment, requested_generation_id)
        terminal_error: BaseException | None = None
        terminal_report: TerminalReport | None = None
        cleanup = _PreflightCleanupReport()
        existing_studies: dict[str, Any] = {}
        generation_prepared = False
        try:
            run_deadline = (
                time.monotonic() + experiment.timeout_seconds_per_run
                if experiment.timeout_seconds_per_run is not None
                else None
            )
            _write_generation_state(
                experiment,
                generation_id=generation_id,
                state="preflighting",
                from_phase=from_phase,
                publish_current=False,
            )
            existing_studies = _preflight_existing_studies(
                experiment,
                cleanup_report=cleanup,
            )
            _reject_bound_descendant_topups(
                experiment,
                from_phase=from_phase,
                existing_studies=existing_studies,
            )
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
            result = _run_experiment_inner(
                experiment,
                from_phase=from_phase,
                dry_run=False,
                generation_id=generation_id,
                preloaded_winners=preloaded_winners,
                run_deadline=run_deadline,
            )
            terminal_report = TerminalReport(
                generation_id=generation_id,
                primary_error=None,
                cleanup_confirmed=True,
                recovered_attempt_ids=frozenset(cleanup.recovered_attempt_ids),
                uncertain_attempt_ids=frozenset(),
                failure_stage=None,
            )
            return result
        except BaseException as exc:
            terminal_error = exc
            control_error: BaseException | None = None
            if generation_prepared:
                reconciliation = _PreflightCleanupReport()
                try:
                    _preflight_existing_studies(
                        experiment,
                        cleanup_report=reconciliation,
                    )
                except (KeyboardInterrupt, SystemExit, GeneratorExit) as cleanup_exc:
                    control_error = cleanup_exc
                    terminal_error = cleanup_exc
                except Exception as cleanup_exc:
                    if isinstance(cleanup_exc, ProcessCleanupUncertainError):
                        reconciliation.mark_uncertain(cleanup_exc)
                    log.exception("failed to reconcile all existing studies after run termination")
                cleanup.recovered_attempt_ids.update(reconciliation.recovered_attempt_ids)
                cleanup.uncertain_attempt_ids.update(reconciliation.uncertain_attempt_ids)
                cleanup.cleanup_confirmed = reconciliation.cleanup_confirmed
                cleanup.error = reconciliation.error
            primary_error = control_error or exc
            shutdown_cleanup_uncertain = (
                isinstance(primary_error, PhaseSweepShutdown)
                and not primary_error.report.cleanup_confirmed
            )
            if shutdown_cleanup_uncertain:
                cleanup.mark_uncertain(primary_error)
            if isinstance(primary_error, ProcessCleanupUncertainError):
                cleanup.mark_uncertain(primary_error)
            with contextlib.suppress(Exception):
                _write_generation_state(
                    experiment,
                    generation_id=generation_id,
                    state="failed",
                    from_phase=from_phase,
                    publish_current=generation_prepared,
                    error_class=type(primary_error).__name__,
                )
            terminal_report = TerminalReport(
                generation_id=generation_id,
                primary_error=primary_error,
                cleanup_confirmed=cleanup.cleanup_confirmed,
                recovered_attempt_ids=frozenset(cleanup.recovered_attempt_ids),
                uncertain_attempt_ids=frozenset(cleanup.uncertain_attempt_ids),
                cleanup_error=cleanup.error,
                failure_stage="execution" if generation_prepared else "preflight",
            )
            if (
                not cleanup.cleanup_confirmed
                and not isinstance(primary_error, ProcessCleanupUncertainError)
                and not shutdown_cleanup_uncertain
            ):
                raise ProcessCleanupUncertainError(
                    "The run failed and subsequent process cleanup could not be confirmed."
                ) from primary_error
            if control_error is not None:
                raise control_error from exc
            raise
        finally:
            if terminal_callback is not None:
                try:
                    if terminal_report is None:
                        terminal_report = TerminalReport(
                            generation_id=generation_id,
                            primary_error=terminal_error,
                            cleanup_confirmed=cleanup.cleanup_confirmed,
                            recovered_attempt_ids=frozenset(cleanup.recovered_attempt_ids),
                            uncertain_attempt_ids=frozenset(cleanup.uncertain_attempt_ids),
                            cleanup_error=cleanup.error,
                            failure_stage=("execution" if generation_prepared else "preflight"),
                        )
                    terminal_callback(terminal_report)
                except Exception:
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
        run_deadline: Optional precomputed whole-run monotonic deadline. Preflight
            passes this through so validation and stale cleanup consume the same
            invocation budget as trial execution.

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
                    prior_promotion = _published_promotion_decision_path(experiment, phase.name)
                    if prior_promotion is not None and prior_promotion.is_file():
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


def _reject_bound_descendant_topups(
    experiment: Experiment,
    *,
    from_phase: str | None,
    existing_studies: dict[str, Any],
) -> None:
    """Reject upstream top-ups that could invalidate a bound descendant study."""
    reached = from_phase is None
    for index, phase in enumerate(experiment.phases):
        if phase.name == from_phase:
            reached = True
        if not reached:
            continue
        study = existing_studies.get(phase.name)
        if study is None:
            continue
        terminal = sum(1 for trial in study.get_trials(deepcopy=False) if trial.state.is_finished())
        if terminal >= phase.n_trials:
            continue

        descendants: set[str] = set()
        ancestry = {phase.name}
        for candidate in experiment.phases[index + 1 :]:
            if ancestry.intersection(candidate.inherits):
                descendants.add(candidate.name)
                ancestry.add(candidate.name)
        bound = [
            name
            for name in descendants
            if (dependent := existing_studies.get(name)) is not None
            and isinstance(dependent.user_attrs.get("phasesweep_fingerprint"), str)
        ]
        if bound:
            raise StudyContextConflictError(
                f"Phase {phase.name!r} has {phase.n_trials - terminal} top-up trial(s) "
                f"remaining, but dependent phase study/studies {bound} are already bound "
                "to its published winner. Use a new experiment name to run the larger "
                "upstream budget without mutating this completed phase chain."
            )


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


def _claim_generation(experiment: Experiment, requested_id: str | None) -> str:
    """Create one exclusively owned generation namespace under the experiment lock."""
    root = _generations_dir(experiment)
    root.mkdir(parents=True, exist_ok=True)
    if requested_id is not None:
        try:
            _generation_dir(experiment, requested_id).mkdir()
        except FileExistsError as exc:
            raise RuntimeError(
                f"Generation id {requested_id!r} already exists; refusing to overwrite history."
            ) from exc
        return requested_id

    for _ in range(10):
        candidate = uuid4().hex
        try:
            _generation_dir(experiment, candidate).mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not mint an unused generation id after 10 attempts.")


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
