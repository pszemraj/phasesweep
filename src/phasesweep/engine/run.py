"""Public engine entrypoints for experiments, suites, and status."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

import optuna
import yaml

from phasesweep._metadata import __version__
from phasesweep.config import Config, Experiment, Phase, Suite
from phasesweep.config.common import _validate_safe_name
from phasesweep.config.models import _metric_semantics_payload
from phasesweep.config.search import sampler_capability_line
from phasesweep.engine.errors import (
    PromotionError,
    PublicationAccessError,
    PublicationCommitError,
    PublicationIntegrityError,
    RunRequestError,
    StudyContextConflictError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.guards import (
    _experiment_lock,
    _experiment_semantic_fingerprint,
    _load_accepted_partial_decision,
    _load_and_check_artifact_roots,
    _preflight_existing_studies,
    _PreflightCleanupReport,
    _suite_fingerprint,
    _suite_lock,
    _validate_sampler_continuation,
    _validate_selection_evidence,
    _verify_fingerprint,
)
from phasesweep.engine.phase import _placeholder_winner, _run_phase
from phasesweep.engine.read import read_status
from phasesweep.engine.selection import (
    _apply_promotion,
    _apply_study_promotion,
    _winner_summary_item,
)
from phasesweep.engine.state import (
    GENERATION_SUMMARY_SCHEMA_VERSION,
    PHASE_FINGERPRINT_ATTR,
    PUBLICATION_POINTER_SCHEMA_VERSION,
    SUITE_SUMMARY_SCHEMA_VERSION,
    Winner,
    _experiment_dir,
    _file_log_handler,
    _generation_artifact_manifest,
    _generation_dir,
    _generation_path,
    _generation_promotion_decision_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _generations_dir,
    _last_successful_generation_path,
    _last_successful_suite_generation_path,
    _load_winner,
    _promotion_decision_path,
    _published_promotion_decision_path,
    _read_pointer_target_summary,
    _resolve_suite_publication_pointer,
    _run_log_path,
    _save_promotion_decision,
    _save_winner,
    _suite_dir,
    _suite_generation_dir,
    _suite_generation_path,
    _suite_generation_record_path,
    _suite_generation_summary_path,
    _suite_generations_dir,
    _suite_log_path,
    _suite_summary_path,
    _summary_path,
    _validate_generation_manifest,
    _validate_suite_summary_integrity,
    _winner_path,
    _write_generation_provenance,
    _write_yaml_atomic,
    _write_yaml_exclusive,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.files import ensure_workdir, file_sha256, require_posix_runtime
from phasesweep.runtime.process import (
    PhaseSweepShutdown,
    absorb_shutdown_signals,
    service_pending_shutdown,
    signal_handler_scope,
)
from phasesweep.runtime.time import utc_now_iso


@dataclass(frozen=True)
class TerminalReport:
    """One engine invocation's primary outcome and independent cleanup evidence.

    ``winners`` is the engine's own authoritative winner mapping, populated on
    the success path while the experiment lock is still held (review v0.5.16 /
    blocker 2): a terminal-result consumer (the MCP runner's snapshot capture)
    must be able to freeze the published outcome from the exact in-memory
    state the engine just returned, instead of re-deriving it from mutable
    storage or optional diagnostic files whose absence would then contradict
    an engine-defined success. ``None`` on every failure path.
    """

    generation_id: str
    primary_error: BaseException | None
    cleanup_confirmed: bool
    recovered_attempt_ids: frozenset[str]
    uncertain_attempt_ids: frozenset[str]
    recovered_attempt_generations: Mapping[str, str] = field(default_factory=dict)
    cleanup_error: BaseException | None = None
    failure_stage: str | None = None
    winners: Mapping[str, Winner] | None = None


class PublicationHook(Protocol):
    """Required sidecar transaction around an experiment publication commit.

    Detached MCP runs use this boundary to persist their exact frozen result
    before the experiment's last-success pointer can advance. Ordinary engine
    callers omit the hook and retain the normal single-filesystem transaction.
    """

    def prepare(
        self,
        *,
        experiment: Experiment,
        generation_id: str,
        winners: Mapping[str, Winner],
    ) -> None:
        """Durably prepare sidecar state before the publication pointer advances.

        :param Experiment experiment: Exact experiment configuration being published.
        :param str generation_id: Generation whose validated result is ready to commit.
        :param Mapping[str, Winner] winners: Engine-selected winners for that generation.
        """
        ...

    def committed(self, *, generation_id: str) -> None:
        """Record that the publication pointer advanced to the prepared generation.

        :param str generation_id: Generation whose pointer commit completed.
        """
        ...


def _terminal_report_from_cleanup(
    generation_id: str,
    cleanup: _PreflightCleanupReport,
    *,
    primary_error: BaseException | None,
    failure_stage: str | None,
    winners: Mapping[str, Winner] | None = None,
    cleanup_confirmed: bool | None = None,
) -> TerminalReport:
    """Freeze accumulated cleanup evidence into one terminal report.

    :param str generation_id: Generation whose invocation ended.
    :param _PreflightCleanupReport cleanup: Mutable cleanup evidence to freeze.
    :param BaseException | None primary_error: Invocation's authoritative error.
    :param str | None failure_stage: Stage where the invocation failed.
    :param Mapping[str, Winner] | None winners: Successful published winners.
    :param bool | None cleanup_confirmed: Optional explicit cleanup verdict.
    :return TerminalReport: Immutable terminal outcome snapshot.
    """
    return TerminalReport(
        generation_id=generation_id,
        primary_error=primary_error,
        cleanup_confirmed=(
            cleanup.cleanup_confirmed if cleanup_confirmed is None else cleanup_confirmed
        ),
        recovered_attempt_ids=frozenset(cleanup.recovered_attempt_ids),
        recovered_attempt_generations=MappingProxyType(dict(cleanup.recovered_attempt_generations)),
        uncertain_attempt_ids=frozenset(cleanup.uncertain_attempt_ids),
        cleanup_error=cleanup.error,
        failure_stage=failure_stage,
        winners=MappingProxyType(dict(winners)) if winners is not None else None,
    )


@dataclass(frozen=True)
class ExperimentRunOutcome:
    """One published experiment invocation bound to its generation identity.

    The winners and the generation id that produced them are materialized
    together while the experiment lock is still held, so consumers (suites,
    provenance manifests) can record exact lineage without re-reading mutable
    pointers after the lock is released — an interleaving external top-up
    would otherwise let a manifest name generation B while carrying winners
    from generation A (review v0.5.14 / blocker 2).
    """

    generation_id: str
    winners: Mapping[str, Winner]
    phase_fingerprints: Mapping[str, str | None]


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
    :raises RunRequestError: ``from_phase`` was given for a suite config, which
        has no single phase sequence to resume.
    """
    if isinstance(config, Suite):
        if from_phase is not None:
            raise RunRequestError("--from-phase is only supported for single experiment configs.")
        return run_suite(config, dry_run=dry_run)
    return run_experiment(config, from_phase=from_phase, dry_run=dry_run)


def config_status(config: Config) -> dict[str, Any]:
    """Collect read-only status for an experiment or suite config.

    For an :class:`~phasesweep.config.Experiment` this returns
    :func:`experiment_status` verbatim. For a :class:`~phasesweep.config.Suite`
    it returns ``{kind: "suite", suite, workdir, published_suite_generation_id,
    publication_integrity, studies}`` — plus ``publication_error`` as the one
    conditional key, exactly as the experiment payload carries it — where each
    study is ``{name, depends_on, status}`` and ``status`` is that study's
    compiled experiment status: the same payload, generation identity included,
    that a standalone experiment reports.

    The suite envelope reports the *suite* last-success pointer's own four-state
    verdict (re-review v0.5.19 / observation N2). Reporting only the component
    studies let a suite whose published summary no longer validated print
    ``publication_integrity: "ok"`` for every component and exit 0, while
    ``show-winners`` on the same tree correctly reported corruption — the
    opposite of finding F4's contract that both surfaces escalate. As in the
    experiment payload, ``published_suite_generation_id`` is populated only for
    an ``"ok"`` verdict: a suite generation that failed validation is named in
    the error, never presented as published.

    :param Config config: Parsed experiment or suite config to inspect.
    :return dict[str, Any]: Read-only status payload for the config.
    """
    if isinstance(config, Suite):
        publication = _resolve_suite_publication_pointer(config)
        return {
            "kind": "suite",
            "suite": config.suite,
            "workdir": str(_suite_dir(config)),
            "published_suite_generation_id": (
                publication.generation_id if publication.state == "ok" else None
            ),
            "publication_integrity": publication.state,
            **(
                {"publication_error": publication.error}
                if publication.state in {"failed", "permission_denied"}
                else {}
            ),
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
    publication_hook: PublicationHook | None = None,
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
        publication_hook: Optional required precommit/postcommit sidecar used by
            detached runners to make their frozen result durable with publication.
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
        TrialEvidenceMissingError: A trial eligible to win no longer has the
            on-disk evidence its study records, or the selected winner's
            objective source no longer matches its frozen provenance.
        RunRequestError: A caller-supplied generation id already names an
            immutable generation in this experiment.
        PhaseSweepError: An expected preflight, promotion, storage,
            publication, or recovery refusal requires operator action.
        RuntimeError: An internal engine invariant fails.
        ValueError: A caller-supplied generation id is not a safe filesystem name.
        FileNotFoundError: ``--from-phase`` requested but a prior phase has
            no persisted ``winner.yaml``.

    """
    if dry_run:
        return _run_experiment_inner(
            experiment,
            from_phase=from_phase,
            dry_run=True,
            generation_id=None,
        )
    outcome = _run_experiment_outcome(
        experiment,
        from_phase=from_phase,
        terminal_callback=terminal_callback,
        publication_hook=publication_hook,
        generation_id=generation_id,
    )
    return dict(outcome.winners)


def _run_experiment_outcome(
    experiment: Experiment,
    *,
    from_phase: str | None = None,
    terminal_callback: Callable[[TerminalReport], None] | None = None,
    publication_hook: PublicationHook | None = None,
    generation_id: str | None = None,
) -> ExperimentRunOutcome:
    """Run all phases and bind the winners to their published generation identity.

    This is the non-dry-run engine core behind :func:`run_experiment`. The
    returned :class:`ExperimentRunOutcome` is constructed while the experiment
    lock is still held, so its ``generation_id`` is exactly the generation that
    produced (and published) ``winners`` — never a later external top-up's.

    :param Experiment experiment: Parsed experiment config.
    :param str | None from_phase: Optional resume point; see :func:`run_experiment`.
    :param Callable[[TerminalReport], None] | None terminal_callback: Optional
        diagnostic callback; see :func:`run_experiment`.
    :param PublicationHook | None publication_hook: Optional required
        publication sidecar; see :func:`run_experiment`.
    :param str | None generation_id: Optional caller-owned invocation identity.
    :return ExperimentRunOutcome: Winners bound to the publishing generation id.
    :raises TrialEvidenceMissingError: Launch preflight found a trial eligible
        to win whose recorded evidence is no longer in this tree, or selection
        found the winner's objective source altered; raised before any trial
        work in the first case, before publication in the second.
    :raises ProcessCleanupUncertainError: The run failed and post-failure
        reconciliation could not confirm process cleanup; the original failure
        is chained as the cause but is unsafe to handle on its own.
    :raises BaseException: The run's own failure, re-raised unchanged after the
        generation is durably recorded as failed — or, when reconciliation was
        interrupted by ``KeyboardInterrupt``, ``SystemExit``, or
        ``GeneratorExit``, that control-flow exception chained from it.
    """
    require_posix_runtime()

    requested_generation_id = (
        None if generation_id is None else _validate_safe_name("generation", generation_id)
    )
    ensure_workdir(_experiment_dir(experiment).parent)
    _experiment_dir(experiment).mkdir(parents=True, exist_ok=True)
    # signal_handler_scope() is the outermost context manager so shutdown-signal
    # ownership is scoped to this call tree and restored on every exit path,
    # not left installed on the host process forever (review v0.5.14 /
    # blocker 6). A run_suite caller has already entered its own scope, so
    # this one is a reentrant no-op that installs and restores nothing.
    with (
        signal_handler_scope(),
        _experiment_lock(experiment),
        contextlib.ExitStack() as run_stack,
    ):
        # Resolve both ownership directions before the generation claim. The
        # returned objects are passed into preflight so storage is not reread
        # after this strict discovery-and-claim boundary.
        try:
            existing_studies = _load_and_check_artifact_roots(experiment, from_phase=from_phase)
        except StudyStorageUnavailableError as exc:
            # The ledger may contain attempts from an earlier orchestrator. A
            # failed ownership read cannot prove those processes are resolved,
            # even though this invocation has not claimed a generation or
            # launched anything of its own.
            raise ProcessCleanupUncertainError(
                "Artifact ownership could not be checked because required persistent "
                f"study state is unavailable: {exc} Cleanup state is therefore unknown."
            ) from exc
        run_stack.enter_context(_file_log_handler(_run_log_path(experiment)))
        generation_id = _claim_generation(experiment, requested_generation_id)
        terminal_error: BaseException | None = None
        terminal_report: TerminalReport | None = None
        cleanup = _PreflightCleanupReport()
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
                publish_current=True,
            )
            existing_studies = _preflight_existing_studies(
                experiment,
                cleanup_report=cleanup,
                from_phase=from_phase,
                preloaded_studies=existing_studies,
            )
            # Selection reads Optuna alone, so a tree whose candidate evidence
            # was deleted would silently reselect and republish those trials
            # (PR #5 review / reviewer 2, blocker 7). Refuse here - after
            # recovery has resolved every stale attempt, before the generation
            # is marked running and before any trial launches - so nothing is
            # added to a tree that cannot honestly be ranked. Deliberately not
            # repeated on the post-failure reconciliation call below: an
            # evidence gap found while cleaning up must never displace the
            # primary error, and publication on that path is covered by the
            # selection-time winner check.
            _validate_selection_evidence(experiment, existing_studies)
            _reject_bound_descendant_topups(
                experiment,
                from_phase=from_phase,
                existing_studies=existing_studies,
            )
            _reject_unsupported_sampler_topups(
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
            _write_generation_state(
                experiment,
                generation_id=generation_id,
                state="running",
                from_phase=from_phase,
                publish_current=True,
            )
            generation_prepared = True
            result = _run_experiment_inner(
                experiment,
                from_phase=from_phase,
                dry_run=False,
                generation_id=generation_id,
                preloaded_winners=preloaded_winners,
                run_deadline=run_deadline,
                publication_hook=publication_hook,
            )
            terminal_report = _terminal_report_from_cleanup(
                generation_id,
                cleanup,
                primary_error=None,
                failure_stage=None,
                winners=result,
                cleanup_confirmed=True,
            )
            return ExperimentRunOutcome(
                generation_id=generation_id,
                winners=MappingProxyType(dict(result)),
                phase_fingerprints=MappingProxyType(
                    {name: winner.phase_fingerprint for name, winner in result.items()}
                ),
            )
        except BaseException as exc:
            terminal_error = exc
            control_error: BaseException | None = None
            if generation_prepared:
                reconciliation = _PreflightCleanupReport()
                try:
                    _preflight_existing_studies(
                        experiment,
                        cleanup_report=reconciliation,
                        from_phase=from_phase,
                    )
                except (KeyboardInterrupt, SystemExit, GeneratorExit) as cleanup_exc:
                    control_error = cleanup_exc
                    terminal_error = cleanup_exc
                except Exception as cleanup_exc:
                    if isinstance(cleanup_exc, ProcessCleanupUncertainError):
                        reconciliation.mark_uncertain(cleanup_exc)
                    log.exception("failed to reconcile all existing studies after run termination")
                cleanup.recovered_attempt_ids.update(reconciliation.recovered_attempt_ids)
                cleanup.recovered_attempt_generations.update(
                    reconciliation.recovered_attempt_generations
                )
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
            failed_generation_id = generation_id
            failed_error_class = type(primary_error).__name__
            # If the publication transaction already wrote a terminal record for
            # this generation (state "publication_failed"), it also already drove
            # the current pointer to that more specific state; a second generic
            # "failed" write here would be refused for the record (write-once)
            # but would silently clobber the current pointer, which has no
            # monotonic guard within one invocation (review v0.5.15 / blocker 3).
            _persist_failed_state_unless_recorded(
                _generation_record_path(experiment, failed_generation_id),
                lambda: _write_generation_state(
                    experiment,
                    generation_id=failed_generation_id,
                    state="failed",
                    from_phase=from_phase,
                    publish_current=True,
                    error_class=failed_error_class,
                ),
            )
            terminal_report = _terminal_report_from_cleanup(
                generation_id,
                cleanup,
                primary_error=primary_error,
                failure_stage="execution" if generation_prepared else "preflight",
            )
            if (
                not cleanup.cleanup_confirmed
                and not isinstance(
                    primary_error,
                    (ProcessCleanupUncertainError, StudyStorageUnavailableError),
                )
                and not shutdown_cleanup_uncertain
            ):
                # Cleanup uncertainty intentionally becomes the actionable error;
                # the original failure remains chained for diagnosis but is unsafe to handle alone.
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
                        terminal_report = _terminal_report_from_cleanup(
                            generation_id,
                            cleanup,
                            primary_error=terminal_error,
                            failure_stage=("execution" if generation_prepared else "preflight"),
                        )
                    terminal_callback(terminal_report)
                except BaseException:  # noqa: BLE001 - see comment: outcome authority
                    # A reporting callback is a diagnostic consumer of the
                    # outcome, never an authority over it. On the success path
                    # the generation is already published as the last
                    # successful result, so raising here would present a
                    # committed success as a caller-visible failure; on the
                    # failure path it would replace the engine's original
                    # error (review v0.5.14 / item C). BaseException on
                    # purpose (review v0.5.16 / blocker 1): a
                    # KeyboardInterrupt/SystemExit escaping the callback must
                    # not rewrite the engine outcome either — real shutdown
                    # signals are honored separately by the signal handler,
                    # not by exceptions leaking out of a diagnostic consumer.
                    log.exception("terminal callback failed; the engine outcome is unchanged")


def _run_experiment_inner(
    experiment: Experiment,
    *,
    from_phase: str | None,
    dry_run: bool,
    generation_id: str | None,
    preloaded_winners: dict[str, Winner] | None = None,
    run_deadline: float | None = None,
    publication_hook: PublicationHook | None = None,
) -> dict[str, Winner]:
    """Sequential phase loop assuming locks/signal handlers are already set up.

    Args:
        experiment: Parsed experiment config.
        from_phase: Optional name of the phase to resume from; earlier phases
            are loaded from disk.
        dry_run: If ``True``, no subprocesses launch and no ``summary.yaml`` is
            written; each phase's sampler capability line is logged up front.
        generation_id: Current invocation identity, or ``None`` for dry-run.
        preloaded_winners: Strictly validated skipped-phase winners loaded before
            the current generation was committed.
        run_deadline: Optional precomputed whole-run monotonic deadline. Preflight
            passes this through so validation and stale cleanup consume the same
            invocation budget as trial execution.
        publication_hook: Optional required sidecar around publication commit.

    Returns:
        Same as :func:`run_experiment`: a phase-name to :class:`Winner` mapping.

    Raises:
        TimeoutError: The whole-run wallclock deadline expired before a phase
            could start.
        PromotionError: A phase's promotion decision was ``stop``.
        FileNotFoundError: A skipped phase has no persisted ``winner.yaml``
            (re-raised on non-dry-run; dry runs substitute a placeholder).

    """
    skip_until = from_phase is not None
    winners: dict[str, Winner] = {}
    promotion_decisions: dict[str, dict[str, Any]] = {}
    if run_deadline is None and not dry_run and experiment.timeout_seconds_per_run is not None:
        run_deadline = time.monotonic() + experiment.timeout_seconds_per_run

    if dry_run:
        # Capability disclosure (review v0.5.18 / finding F7): restate the same
        # per-phase resume/reproduce contract `phasesweep validate` prints, in
        # the preview an operator reads immediately before committing to a run.
        for previewed in experiment.phases:
            log.info("DRY RUN %s", sampler_capability_line(previewed))

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
                    # Safe to re-resolve the published pointer per phase here, unlike
                    # status reads: this runs under _experiment_lock, and this
                    # experiment's pointer only advances at this run's final publish.
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
                assert generation_id is not None
                _save_promotion_decision(
                    experiment,
                    phase.name,
                    promotion_decision,
                    generation_id=generation_id,
                )
            if promoted is None:
                if promotion_decision is not None and promotion_decision["action"] == "stop":
                    raise PromotionError(str(promotion_decision["message"]))
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
    # The summary is the generation's versioned result manifest (review
    # v0.5.16 / blockers 3+4): it lists every artifact with its content hash
    # so publication can validate the complete result graph, and it freezes
    # the config semantics (metric, phase plan, config fingerprint) that give
    # those numbers meaning so later reads never reinterpret them through
    # whatever config happens to be loaded.
    summary = {
        "schema_version": GENERATION_SUMMARY_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "phasesweep_version": __version__,
        "config_fingerprint": _experiment_semantic_fingerprint(experiment),
        "metric": _metric_semantics_payload(experiment.metric),
        "phase_plan": [
            {"name": phase.name, "comment": phase.comment} for phase in experiment.phases
        ],
        "promotion_decisions": list(promotion_decisions.values()),
        "phases": [_winner_summary_item(pname, w) for pname, w in winners.items()],
        "artifacts": _generation_artifact_manifest(experiment, generation_id),
    }
    _write_yaml_atomic(summary_path, summary)
    _publish_generation(
        experiment,
        generation_id,
        from_phase=from_phase,
        winners=winners,
        publication_hook=publication_hook,
    )
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
    :raises TimeoutError: The whole-run wallclock deadline expired before a
        skipped phase could be validated.
    :raises ValueError: ``from_phase`` names no phase in the experiment.
    :raises FileNotFoundError: A skipped phase has no persisted ``winner.yaml``.
    :raises WinnerIntegrityError: A skipped phase's persisted winner is invalid
        or incomplete.
    :raises StudyFingerprintMismatchError: A skipped phase's winner fingerprint
        disagrees with the current config.
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
        # Safe to re-resolve the published pointer per phase here, unlike status
        # reads: this preflight runs under _experiment_lock, and this
        # experiment's pointer only advances at this run's final publish.
        winners[phase.name] = _load_winner(experiment, phase, inherited)

    raise ValueError(f"Unknown --from-phase value {from_phase!r}.")


def _phases_from(experiment: Experiment, from_phase: str | None) -> Iterator[tuple[int, Phase]]:
    """Yield ``(index, phase)`` pairs from ``from_phase`` (or the start) onward.

    Shared reached-from-phase iteration prologue for
    :func:`_reject_bound_descendant_topups` and
    :func:`_reject_unsupported_sampler_topups`.

    :param Experiment experiment: Parsed experiment whose phase chain is scanned.
    :param str | None from_phase: Optional resume point; phases before it are
        skipped. ``None`` reaches every phase from the start.

    Yields:
        ``(index, phase)``: Each reached phase paired with its declaration index.

    """
    reached = from_phase is None
    for index, phase in enumerate(experiment.phases):
        if phase.name == from_phase:
            reached = True
        if not reached:
            continue
        yield index, phase


def _reject_bound_descendant_topups(
    experiment: Experiment,
    *,
    from_phase: str | None,
    existing_studies: dict[str, optuna.Study],
) -> None:
    """Reject upstream top-ups that could invalidate a bound descendant study.

    Reachability follows both semantic edges a phase may declare on a prior
    phase, not ``inherits`` alone (PR #5 review / reviewer 2 pass 2, blocker 2).
    ``promotion.min_delta_vs`` is validated only to name a *prior* phase, never
    an inherited one, so ``A --promotion--> B --inherits--> C`` with ``B`` not
    inheriting ``A`` is a legal graph. A promotion baseline is a semantic
    dependency even though it is absent from the promoted phase's own
    fingerprint: a new baseline winner can flip the promotion decision, and
    ``on_fail: continue_baseline`` then exposes a clone of the baseline winner -
    effective overrides included - in the promoted phase's slot, which anything
    inheriting that phase has already hashed into its bound study. ``stop`` and
    ``skip`` need the same protection with no inheriting descendant at all: the
    first turns the top-up into a mid-run failure after the upstream study was
    already mutated, and the second publishes a new generation that silently
    omits the promoted phase and everything after it, advancing the last-success
    pointer past the previously published winners. A phase reached through
    either edge therefore both extends the reachable set (its exposed winner can
    change) and joins the bound-study check set.

    Suite-level promotion needs no handling here: each suite study compiles to
    an independent :class:`Experiment` with its own studies and fingerprints,
    ``depends_on`` only orders execution, and no fingerprint binds across
    compiled experiments - so there is no cross-study binding to invalidate.

    :param Experiment experiment: Parsed experiment whose phase chain is scanned
        from ``from_phase`` (or the start) onward.
    :param str | None from_phase: Optional resume point; phases before it are skipped.
    :param dict[str, optuna.Study] existing_studies: Existing Optuna studies keyed by
        phase name, as returned by :func:`_preflight_existing_studies`.
    :raises StudyContextConflictError: An upstream phase still has unfinished
        top-up trials remaining while a phase depending on it - by inheritance,
        by promotion baseline, or transitively through either - already has a
        study bound to a published winner fingerprint.
    """
    inheritance_kind = "inheritance"
    promotion_kind = "a promotion baseline"
    for index, phase in _phases_from(experiment, from_phase):
        study = existing_studies.get(phase.name)
        if study is None:
            continue
        terminal = sum(1 for trial in study.get_trials(deepcopy=False) if trial.state.is_finished())
        if terminal >= phase.n_trials:
            continue
        partial_decision = _load_accepted_partial_decision(study)
        if partial_decision is not None and phase.n_trials == partial_decision.trial_target:
            # The phase has already committed a terminal accepted-timeout
            # decision at this exact target. Identical replay performs only
            # deterministic selection; its unused slots are not a top-up.
            continue

        # Reachable phase name -> every dependency-edge kind traversed to reach
        # it, so the refusal can name what actually binds each dependent study.
        # The phase itself is reachable through no edge at all.
        reached: dict[str, frozenset[str]] = {phase.name: frozenset()}
        for candidate in experiment.phases[index + 1 :]:
            kinds: set[str] = set()
            for parent in candidate.inherits:
                if parent in reached:
                    kinds.add(inheritance_kind)
                    kinds |= reached[parent]
            baseline = None if candidate.promotion is None else candidate.promotion.min_delta_vs
            if baseline is not None and baseline in reached:
                kinds.add(promotion_kind)
                kinds |= reached[baseline]
            if kinds:
                reached[candidate.name] = frozenset(kinds)
        bound = [
            name
            for name, kinds_reached in reached.items()
            if kinds_reached
            and (dependent := existing_studies.get(name)) is not None
            and isinstance(dependent.user_attrs.get(PHASE_FINGERPRINT_ATTR), str)
        ]
        if bound:
            bound_kinds = frozenset().union(*(reached[name] for name in bound))
            dependency_text = " and ".join(
                kind for kind in (inheritance_kind, promotion_kind) if kind in bound_kinds
            )
            promotion_note = (
                " A new baseline winner can flip that promotion decision, which either "
                "republishes the baseline's own effective overrides in the promoted "
                "phase's slot or drops the promoted phase and its successors from the "
                "published result."
                if promotion_kind in bound_kinds
                else ""
            )
            raise StudyContextConflictError(
                f"Phase {phase.name!r} has {phase.n_trials - terminal} top-up trial(s) "
                f"remaining, but dependent phase study/studies {bound} are already bound "
                f"to its published winner via {dependency_text}.{promotion_note} "
                "Use a new experiment name to run the larger upstream budget without "
                "mutating this completed phase chain."
            )


def _reject_unsupported_sampler_topups(
    experiment: Experiment,
    *,
    from_phase: str | None,
    existing_studies: dict[str, optuna.Study],
) -> None:
    """Reject stateful sampler continuation after bound-descendant checks.

    :param Experiment experiment: Parsed experiment whose phase chain is scanned
        from ``from_phase`` (or the start) onward.
    :param str | None from_phase: Optional resume point; phases before it are skipped.
    :param dict[str, optuna.Study] existing_studies: Existing Optuna studies keyed by
        phase name, as returned by :func:`_preflight_existing_studies`.
    :raises SamplerContinuationUnsupportedError: A reached phase's study cannot
        safely continue with its configured stateful sampler; delegated to
        :func:`_validate_sampler_continuation`.
    """
    for _index, phase in _phases_from(experiment, from_phase):
        study = existing_studies.get(phase.name)
        if study is not None:
            _validate_sampler_continuation(study, phase)


def _preflight_reached_fingerprint(
    experiment: Experiment,
    *,
    from_phase: str | None,
    preloaded_winners: dict[str, Winner],
    existing_studies: dict[str, optuna.Study],
) -> None:
    """Verify the first reached study before publishing the new generation.

    :param Experiment experiment: Parsed experiment whose first reached phase
        (``from_phase``, or the first declared phase) is checked.
    :param str | None from_phase: Optional resume point identifying the first
        phase that will actually execute.
    :param dict[str, Winner] preloaded_winners: Validated skipped-phase winners,
        used to resolve the reached phase's inherited context.
    :param dict[str, optuna.Study] existing_studies: Existing Optuna studies keyed by
        phase name; a no-op if the reached phase has none yet.
    :raises StudyFingerprintMismatchError: The reached study's stored
        fingerprint does not match the current config.
    """
    phase = experiment.phases[0]
    if from_phase is not None:
        phase = next(item for item in experiment.phases if item.name == from_phase)
    study = existing_studies.get(phase.name)
    if study is None:
        return
    inherited = {name: preloaded_winners[name] for name in phase.inherits}
    _verify_fingerprint(study, experiment, phase, inherited)


def _claim_generation(experiment: Experiment, requested_id: str | None) -> str:
    """Create one exclusively owned generation namespace under the experiment lock.

    The claim is only complete once the namespace records the configuration
    that is about to run: :func:`phasesweep.engine.state._write_generation_provenance`
    freezes ``config.snapshot.yaml`` and ``reproducibility.json`` before this
    returns (review v0.5.18 / finding F6), so a generation that later fails
    preflight, execution, or publication still says what search spaces, fixed
    overrides, contracts, env, and trial command produced it. A failure
    writing them fails the claim rather than starting a run whose
    configuration would be unrecoverable. The record also freezes whether the
    id was caller-supplied, so a generation launched under an external
    authority grant (a detached MCP run) stays recognizable as such even if
    the launcher's own state is later lost (PR #5 review / P2 missing-handle
    authority).

    :param Experiment experiment: Experiment whose generations root is created if missing.
    :param str | None requested_id: Caller-supplied generation id to claim, or
        ``None`` to mint a fresh random id.
    :return str: The claimed generation id (``requested_id`` if supplied and
        free, otherwise a freshly minted UUID4 hex string).
    :raises RunRequestError: ``requested_id`` already exists.
    :raises RuntimeError: No unused random id could be minted after 10 attempts,
        which indicates broken UUID generation rather than an operator conflict.
    :raises OSError: The generation namespace or its provenance files could not
        be created.
    """
    root = _generations_dir(experiment)
    root.mkdir(parents=True, exist_ok=True)
    if requested_id is not None:
        try:
            _generation_dir(experiment, requested_id).mkdir()
        except FileExistsError as exc:
            raise RunRequestError(
                f"Generation id {requested_id!r} already exists; refusing to overwrite history."
            ) from exc
        _write_generation_provenance(experiment, requested_id, caller_owned_id=True)
        return requested_id

    for _ in range(10):
        candidate = uuid4().hex
        try:
            _generation_dir(experiment, candidate).mkdir()
        except FileExistsError:
            continue
        _write_generation_provenance(experiment, candidate, caller_owned_id=False)
        return candidate
    raise RuntimeError("Could not mint an unused generation id after 10 attempts.")


_TERMINAL_GENERATION_STATES = frozenset({"published", "publication_failed", "failed"})


def _persist_terminal_failure(write_state: Callable[[], None]) -> None:
    """Persist failed-state bookkeeping without replacing the active exception.

    Bookkeeping is never more authoritative than the failure it records: the
    original exception carries safety-critical semantics (signal exit codes,
    cleanup uncertainty, trainer failure identity) that a secondary filesystem
    error must not overwrite (review v0.5.14 / blocker 5). Catching
    ``BaseException`` is intentional in this narrow helper — even a second
    control-flow exception (another SIGINT/SIGTERM, a nested ``SystemExit``)
    delivered during persistence must not supersede the primary failure that
    is already propagating.

    :param Callable[[], None] write_state: Zero-argument persistence action.
    """
    try:
        write_state()
    except BaseException:  # noqa: BLE001 - see docstring: primary exception must survive
        log.exception("failed to persist terminal failure state; preserving the original error")


def _persist_failed_state_unless_recorded(
    record_path: Path, write_failed: Callable[[], None]
) -> None:
    """Write the generic "failed" state, unless a more specific record already exists.

    If a ``publication_failed`` record was already written for this
    generation, the generic ``failed`` state write must not run: the current
    pointer was already driven terminal with the more specific outcome, and
    the write-once record must not be re-attempted on top of it.

    :param Path record_path: Immutable per-generation record path to check.
    :param Callable[[], None] write_failed: Zero-argument "failed" state
        persistence action, run via :func:`_persist_terminal_failure`.
    """
    if not record_path.is_file():
        _persist_terminal_failure(write_failed)


def _log_on_failure(action: Callable[[], None], message: str) -> None:
    """Run a post-commit best-effort action, logging instead of raising on failure.

    Exists for the post-commit best-effort steps of the publication
    transaction (record write, pointer refresh, compatibility projection):
    once the authoritative pointer has committed, nothing after it may fail
    the run, so each such step must log its own failure and never propagate
    it. Catching ``BaseException`` is intentional (review v0.5.16 / blocker
    1): a ``KeyboardInterrupt``/``SystemExit`` escaping one of these steps
    would propagate into the caller's terminal-failure handler and reclassify
    an already-committed publication as failed. Real shutdown *signals* are
    additionally kept out of these steps entirely by the enclosing
    :func:`phasesweep.runtime.process.absorb_shutdown_signals` window.

    :param Callable[[], None] action: Zero-argument best-effort action.
    :param str message: Message logged (with exception info) on failure.
    """
    try:
        action()
    except BaseException:  # noqa: BLE001 - see docstring: commit outcome must survive
        log.exception(message)


def _recorded_generation_state(record_path: Path) -> str | None:
    """Read one lifecycle record's state label, or ``None`` when unreadable.

    :param Path record_path: Immutable lifecycle record YAML path.
    :return str | None: The recorded ``state`` string, or ``None`` when the
        record is missing, unreadable, or malformed.
    """
    try:
        payload = yaml.safe_load(record_path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    state = payload.get("state")
    return state if isinstance(state, str) else None


def _write_generation_record_once(
    *,
    record_path: Path,
    generation_id: str,
    state: str,
    payload: dict[str, Any],
    label: str,
) -> None:
    """Create one generation's immutable terminal record exactly once.

    Shared write-once core for :func:`_write_generation_state` and
    :func:`_write_suite_generation_state` (review v0.5.15 / blocker 3, item
    B): the per-generation record is written only for a terminal state
    (``published`` / ``publication_failed`` / ``failed``), and only ever
    once. A second attempt for any generation -- even a same-state rewrite --
    is refused and logged; the first content is never touched. Progress
    states (``preflighting`` / ``running``) never reach this function at all;
    see the ``state in _TERMINAL_GENERATION_STATES`` guard in the callers.

    :param Path record_path: Immutable per-generation lifecycle record path.
    :param str generation_id: Immutable generation (or suite generation)
        namespace being recorded; used only for the refusal log message.
    :param str state: Terminal lifecycle state label being written.
    :param dict[str, Any] payload: Full record payload to persist.
    :param str label: Human label identifying the record kind in the refusal
        log message (e.g. ``"generation"``, ``"suite generation"``).
    """
    if _write_yaml_exclusive(record_path, payload):
        return
    existing_state = _recorded_generation_state(record_path)
    log.warning(
        "Refusing to rewrite terminal %s %s record (existing state %r, attempted %r)",
        label,
        generation_id,
        existing_state,
        state,
    )


def _write_generation_state(
    experiment: Experiment,
    *,
    generation_id: str,
    state: str,
    from_phase: str | None,
    publish_current: bool,
    error_class: str | None = None,
    write_record: bool = True,
) -> None:
    """Write one generation's current-pointer projection and/or immutable record.

    The current pointer (``generation.yaml``) is a mutable progress record --
    every state (``preflighting``, ``running``, ``published``,
    ``publication_failed``, ``failed``) may be written to it, with no
    monotonic guard: a new invocation legitimately overwrites it starting from
    ``preflighting`` (review v0.5.15 / blocker 3). The per-generation record
    under ``generations/<id>/generation.yaml`` is truly immutable: it is only
    ever written for a terminal state, and only once (see
    :func:`_write_generation_record_once`); progress states never touch it.

    :param Experiment experiment: Experiment whose generation state is written.
    :param str generation_id: Immutable generation namespace being recorded.
    :param str state: Lifecycle state label (``"preflighting"``, ``"running"``,
        ``"published"``, ``"publication_failed"``, or ``"failed"``).
    :param str | None from_phase: Resume point for this invocation, or ``None``.
    :param bool publish_current: If ``True``, also overwrite the experiment's
        current-generation pointer with this state.
    :param str | None error_class: Optional exception class name to record for
        a failed or publication_failed state.
    :param bool write_record: If ``False``, skip the write-once per-generation
        record even for a terminal state. Used by the post-commit
        current-pointer refresh, whose record was already written moments
        earlier -- re-attempting it would trip the write-once refusal warning
        on every successful publication.
    """
    payload = {
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "state": state,
        "from_phase": from_phase,
        "error_class": error_class,
        "phasesweep_version": __version__,
    }
    if write_record and state in _TERMINAL_GENERATION_STATES:
        _write_generation_record_once(
            record_path=_generation_record_path(experiment, generation_id),
            generation_id=generation_id,
            state=state,
            payload=payload,
            label="generation",
        )
    if publish_current:
        _write_yaml_atomic(_generation_path(experiment), payload)


def _copy_yaml_projection(source: Path, destination: Path) -> None:
    """Atomically project one immutable YAML artifact to its compatibility path.

    :param Path source: Immutable generation-scoped YAML file to read.
    :param Path destination: Legacy compatibility path to atomically overwrite.
    """
    payload = yaml.safe_load(source.read_text())
    _write_yaml_atomic(destination, payload)


def _validate_publishable_summary(
    *,
    summary_path: Path,
    owner_key: str,
    owner_value: str,
    id_key: str,
    id_value: str,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    """Parse back one generation's own immutable summary before its publication commit.

    Shared pre-commit validation core (item A, review v0.5.15) for
    :func:`_validate_generation_publishable` and
    :func:`_validate_suite_generation_publishable`. This runs *before* the
    last-success pointer commits and before the per-generation lifecycle
    record is ever written for this generation, so it cannot check that
    record (it does not exist yet); it instead confirms the immutable summary
    itself parses as a mapping naming the expected owner and id, and returns
    it so the caller can validate the complete manifest without re-reading.

    :param Path summary_path: Immutable summary YAML path to read back.
    :param str owner_key: Summary key naming the owning experiment or suite.
    :param str owner_value: Expected owner name the summary must carry.
    :param str id_key: Summary key holding the generation (or suite generation) id.
    :param str id_value: Expected id the summary must name.
    :param str label: Human label for error text (e.g. ``"Generation"``,
        ``"Suite generation"``).
    :raises PublicationCommitError: The summary cannot be read back as a correctly
        named mapping; the last-success pointer must not advance to it.
    :return tuple[dict[str, Any], bytes]: Parsed summary and the exact bytes validated.
    """
    try:
        summary_bytes = summary_path.read_bytes()
        summary = yaml.safe_load(summary_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise PublicationCommitError(
            f"{label} {id_value!r} summary could not be read back; "
            "refusing to advance the last-success pointer."
        ) from exc
    if (
        not isinstance(summary, dict)
        or summary.get(owner_key) != owner_value
        or summary.get(id_key) != id_value
    ):
        raise PublicationCommitError(
            f"{label} {id_value!r} summary failed publication validation; "
            "refusing to advance the last-success pointer."
        )
    return summary, summary_bytes


def _validate_generation_publishable(experiment: Experiment, generation_id: str) -> bytes:
    """Validate a generation's complete result manifest before its publication commit.

    Checks the generation's immutable summary names this exact experiment and
    generation id, then validates the versioned artifact manifest end to end
    (:func:`phasesweep.engine.state._validate_generation_manifest`, review
    v0.5.16 / blocker 3): every winner and promotion artifact the summary
    claims must exist, hash to its recorded content, parse, and cross-check
    against the summary's own winner facts (trial number, metric name, value,
    goal, source identity fields, completion metadata) — and the namespace
    must hold nothing the manifest does not list. A winner's own
    ``generation_id`` is deliberately *not* required to equal
    ``generation_id``: a winner is legitimately carried forward from
    whichever generation actually produced the best trial, so the field is
    checked for well-formedness only.

    :param Experiment experiment: Experiment whose generation is being published.
    :param str generation_id: Immutable generation namespace to validate.
    :raises PublicationCommitError: The generation summary cannot be read back
        or does not name this generation.
    :raises PublicationAccessError: A manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: A manifest-listed artifact fails validation.
    :return bytes: Exact summary bytes whose manifest was validated.
    """
    summary, summary_bytes = _validate_publishable_summary(
        summary_path=_generation_summary_path(experiment, generation_id),
        owner_key="experiment",
        owner_value=experiment.experiment,
        id_key="generation_id",
        id_value=generation_id,
        label="Generation",
    )
    _validate_generation_manifest(
        _generation_dir(experiment, generation_id),
        generation_id,
        summary,
    )
    return summary_bytes


def _publish_generation(
    experiment: Experiment,
    generation_id: str,
    *,
    from_phase: str | None,
    winners: Mapping[str, Winner],
    publication_hook: PublicationHook | None,
) -> None:
    """Publish a generation as the experiment's last successful result.

    This is the publication transaction (review v0.5.15 / blocker 3), in
    strict order:

    1. Immutable generation-scoped artifacts (winners/promotions/summary) are
       already written by the time this runs.
    2. Pre-commit validation (:func:`_validate_generation_publishable`, item
       A): parse this generation's own summary and winner files back and
       confirm they name themselves correctly.
    3. When supplied, require the detached runner's sidecar hook to durably
       prepare its frozen result.
    4. Commit ``last_successful_generation.yaml`` atomically -- the single
       authoritative publication event. Failures in either preceding step
       leave the prior pointer untouched.

    If step 2 or 3 raises, the immutable per-generation record is written
    once with state ``"publication_failed"`` (+ ``error_class``), the current
    pointer is driven to ``"publication_failed"``, the prior last-success
    pointer is left untouched (still authoritative), and the original
    exception re-raises. Persisting that bookkeeping is itself best-effort
    (:func:`_persist_terminal_failure`): a secondary failure while writing it
    is logged, never substituted for the primary error.

    Once step 4 has committed, nothing after it may fail the run:

    5. Best-effort, notify the prepared sidecar that the pointer committed.
    6. Write the immutable per-generation record once, state ``"published"``.
    7. Best-effort, diagnostic-only: drive the current pointer to
       ``"published"``, then refresh the legacy compatibility projections
       (root ``winner.yaml`` / ``promotion.yaml`` / ``summary.yaml``; review
       v0.5.15 / item D). These are post-commit caches for humans and legacy
       tooling only -- no reader re-derives them, and once any generation has
       published, reads resolve the generation-scoped artifacts directly (see
       :func:`phasesweep.engine.state._published_winner_path_for`), so a
       failure projecting them never affects what callers actually see.

    Steps 5 through 7 each independently log and swallow their own failure so one
    cannot prevent the other from running.

    The whole transaction runs inside an
    :func:`phasesweep.runtime.process.absorb_shutdown_signals` window (review
    v0.5.16 / blocker 1): a shutdown signal that arrives after the pointer
    commit must not reclassify the committed publication as failed or
    cancelled. The race has a deterministic winner — a shutdown delivered
    before this window opens cancels the run with nothing published; one
    delivered inside it is absorbed until the publication is durably
    classified and then honored at the next checkpoint (the suite loop, the
    next trial launch, or the MCP runner's terminal status write) before any
    new work starts.

    :param Experiment experiment: Experiment whose generation is being published.
    :param str generation_id: Immutable generation namespace to publish.
    :param str | None from_phase: Resume point recorded on the lifecycle state.
    :param Mapping[str, Winner] winners: Engine-selected winners to hand to a
        required publication sidecar.
    :param PublicationHook | None publication_hook: Optional detached-run sidecar.
    :raises Exception: Whatever steps 2 through 4 raised, after best-effort
        "publication_failed" bookkeeping.
    """
    with absorb_shutdown_signals() as absorbed:
        try:
            summary_bytes = _validate_generation_publishable(experiment, generation_id)
            if publication_hook is not None:
                publication_hook.prepare(
                    experiment=experiment,
                    generation_id=generation_id,
                    winners=MappingProxyType(dict(winners)),
                )
            _write_yaml_atomic(
                _last_successful_generation_path(experiment),
                {
                    "schema_version": PUBLICATION_POINTER_SCHEMA_VERSION,
                    "experiment": experiment.experiment,
                    "generation_id": generation_id,
                    "summary_size_bytes": len(summary_bytes),
                    "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                },
            )
        except BaseException as exc:
            error_class = type(exc).__name__
            _persist_terminal_failure(
                lambda: _write_generation_state(
                    experiment,
                    generation_id=generation_id,
                    state="publication_failed",
                    from_phase=from_phase,
                    publish_current=True,
                    error_class=error_class,
                )
            )
            raise

        if publication_hook is not None:
            _log_on_failure(
                lambda: publication_hook.committed(generation_id=generation_id),
                "failed to record the detached-run publication receipt after the "
                "last-success pointer committed; the prepared snapshot remains recoverable",
            )

        def _write_published_record() -> None:
            """Step 6: write the immutable per-generation record once, state ``published``."""
            _write_generation_state(
                experiment,
                generation_id=generation_id,
                state="published",
                from_phase=from_phase,
                publish_current=False,
            )

        _log_on_failure(
            _write_published_record,
            "failed to write the immutable generation record after publication; "
            "the published result is unaffected",
        )

        def _refresh_published_pointer_and_projections() -> None:
            """Step 7: drive the pointer to ``published``, refresh legacy projections (best-effort)."""
            _write_generation_state(
                experiment,
                generation_id=generation_id,
                state="published",
                from_phase=from_phase,
                publish_current=True,
                write_record=False,
            )
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

        _log_on_failure(
            _refresh_published_pointer_and_projections,
            "failed to refresh the current-generation pointer or compatibility caches "
            "after publication; the published result is unaffected",
        )

    if absorbed.signum is not None:
        log.warning(
            "Shutdown signal %d arrived during the publication transaction for "
            "generation %s; the committed publication wins and the shutdown is "
            "honored at the next checkpoint before any new work starts.",
            absorbed.signum,
            generation_id,
        )


def experiment_status(experiment: Experiment) -> dict[str, Any]:
    """Collect read-only status for one experiment config.

    Built on a single :func:`phasesweep.engine.read.read_status` call, which
    resolves the current pointer and the last-success pointer exactly once each
    and reuses both for every phase's winner-path and trial-count lookup, so
    one status object can never mix generation A's identity with generation B's
    artifacts (review v0.5.15 / blocker 3).

    The returned mapping is the single experiment status snapshot shared by
    every caller: ``phasesweep status <experiment>`` renders it directly, and
    :func:`config_status` embeds it unchanged under ``studies[*].status`` for a
    suite, so a suite study reports the same generation identity a standalone
    experiment does. Its keys are exactly, in order:

    * ``kind``: always ``"experiment"``.
    * ``experiment``: configured experiment name.
    * ``workdir``: experiment artifact root.
    * ``current_generation_id``, ``published_generation_id``,
      ``represented_generation_id``, ``is_published``: the identity split
      defined by :func:`phasesweep.engine.read.read_status` (unpinned mode, so
      the represented generation is the published one). A pre-generation
      legacy workdir has no generation ids to report and leaves all three
      null, but its compatibility ``winner.yaml`` still counts as published,
      so ``is_published`` never contradicts the ``winner`` path shown beside
      it.
    * ``publication_integrity``: ``"ok"`` / ``"absent"`` / ``"failed"`` /
      ``"permission_denied"``, and ``publication_error`` beside either failure
      verdict -- the one
      conditional key in this payload (review v0.5.18 / finding F4). Published
      results that no longer validate are reported as corrupt rather than as
      an experiment that never published. After printing this payload, the CLI
      turns ``"failed"`` into
      :class:`~phasesweep.engine.PublicationIntegrityError` and
      ``"permission_denied"`` into
      :class:`~phasesweep.engine.PublicationAccessError`; both exit non-zero.
    * ``phases``: one payload per phase in declaration order, each with
      ``trials``, ``running``, ``n_trials``, ``completed``,
      ``generation_trials`` (scoped to ``current_generation_id``), ``name``,
      ``winner`` (path string or ``None``), and ``published_study_unavailable``
      (a published phase has no readable trial history, so executing it is blocked).

    ``read_status``'s summary-derived fields (``metric``, ``result_context``,
    ``published_config_matches_current``, ``summary_present``) are deliberately
    *not* part of this payload: they are the MCP read view's contract, and the
    per-study cost of computing them here is one small YAML read plus a config
    fingerprint, negligible beside the per-phase Optuna storage reads this call
    already performs. ``tests/test_engine_status_shape.py`` pins both key sets
    so neither can drift silently again.

    :param Experiment experiment: Parsed experiment config to inspect.
    :return dict[str, Any]: Status payload with generation identity plus per-phase
        winner paths and trial counts, as enumerated above.
    """
    status = read_status(experiment, _include_winner_paths=True)
    integrity = status["publication_integrity"]
    return {
        "kind": "experiment",
        "experiment": status["experiment"],
        "workdir": str(_experiment_dir(experiment)),
        "current_generation_id": status["current_generation_id"],
        "published_generation_id": status["published_generation_id"],
        "represented_generation_id": status["represented_generation_id"],
        "is_published": status["is_published"],
        "publication_integrity": integrity,
        **(
            {"publication_error": status["publication_error"]}
            if integrity in {"failed", "permission_denied"}
            else {}
        ),
        "phases": status["phases"],
    }


def run_suite(suite: Suite, *, dry_run: bool = False) -> dict[str, dict[str, Winner]]:
    """Run every study in declaration order after validating prior dependencies.

    :param Suite suite: Parsed suite config.
    :param bool dry_run: If ``True``, preview each study without launching subprocesses.
    :return dict[str, dict[str, Winner]]: Winners keyed by study name, then phase name.
    :raises PromotionError: A study declares a dependency that exposed no
        winners, so the suite refuses to start it.
    :raises BaseException: Whatever a component run or the publication
        transaction raised, re-raised after the suite generation is durably
        recorded as ``failed`` (or ``publication_failed``).
    """
    results: dict[str, dict[str, Winner]] = {}
    promotion_decisions: dict[str, dict[str, Any]] = {}
    if dry_run:
        for study_spec in suite.studies:
            experiment = suite.experiment_for_study(study_spec)
            results[study_spec.name] = run_experiment(experiment, dry_run=True)
        return results

    require_posix_runtime()
    ensure_workdir(Path(suite.defaults.workdir).expanduser().resolve())
    _suite_dir(suite).mkdir(parents=True, exist_ok=True)
    # See _run_experiment_outcome: signal_handler_scope() is the outermost
    # context manager here too, so a suite installs shutdown handlers once for
    # the whole component sequence and each component's own scope (entered
    # inside _run_experiment_outcome) is a no-op nested inside this one.
    with (
        signal_handler_scope(),
        _suite_lock(suite),
        _file_log_handler(_suite_log_path(suite)),
    ):
        generation_id = _claim_suite_generation(suite)
        started_at = utc_now_iso()
        _write_suite_generation_state(
            suite,
            generation_id=generation_id,
            state="running",
            started_at=started_at,
        )
        component_records: dict[str, dict[str, Any]] = {}
        try:
            for study_spec in suite.studies:
                # A shutdown absorbed by a component's publication transaction
                # (review v0.5.16 / blocker 1) must stop the suite before the
                # next study starts: the committed component publication wins
                # its own race, but no new work may begin afterward.
                service_pending_shutdown()
                for dep in study_spec.depends_on:
                    if dep not in results:
                        raise PromotionError(
                            f"Study {study_spec.name!r} dependency {dep!r} did not complete."
                        )
                experiment = suite.experiment_for_study(study_spec)
                log.info("suite=%s study=%s START", suite.suite, study_spec.name)
                # The outcome binds winners to the generation that published
                # them while the component's experiment lock is held. Reading
                # the mutable last-success pointer here instead would race an
                # interleaving external top-up and record false provenance
                # (review v0.5.14 / blocker 2).
                component = _run_experiment_outcome(experiment)
                study_winners = dict(component.winners)
                # The component summary's path and content hash anchor this
                # study's exposed results for read-side integrity validation
                # (review v0.5.17 gap hunt): the published generation
                # namespace is immutable, so the hash taken here stays valid
                # for the life of the artifact.
                component_summary_path = _generation_summary_path(
                    experiment, component.generation_id
                )
                component_records[study_spec.name] = {
                    "experiment": experiment.experiment,
                    "experiment_generation_id": component.generation_id,
                    "experiment_phase_fingerprints": dict(component.phase_fingerprints),
                    "component_summary_path": str(component_summary_path),
                    "component_summary_sha256": file_sha256(component_summary_path),
                }
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

            ended_at = utc_now_iso()
            summary = _suite_summary_payload(
                suite,
                generation_id=generation_id,
                started_at=started_at,
                ended_at=ended_at,
                results=results,
                promotion_decisions=promotion_decisions,
                component_records=component_records,
            )
            # Immutable artifact (step 1), unchanged regardless of publication outcome.
            immutable_summary = _suite_generation_summary_path(suite, generation_id)
            _write_yaml_atomic(immutable_summary, summary)

            # Same publication transaction shape as _publish_generation (review
            # v0.5.15 / blocker 3): pre-commit validation, then the pointer
            # commit as the single final authoritative event; the record and
            # compatibility cache both move to best-effort, post-commit steps.
            # The same absorb window applies (review v0.5.16 / blocker 1): a
            # shutdown arriving mid-transaction must not reclassify the
            # committed suite publication.
            with absorb_shutdown_signals() as absorbed:
                try:
                    summary_bytes = _validate_suite_generation_publishable(suite, generation_id)
                    _write_yaml_atomic(
                        _last_successful_suite_generation_path(suite),
                        {
                            "schema_version": PUBLICATION_POINTER_SCHEMA_VERSION,
                            "suite": suite.suite,
                            "suite_generation_id": generation_id,
                            "summary_size_bytes": len(summary_bytes),
                            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                        },
                    )
                except BaseException as exc:
                    error_class = type(exc).__name__
                    _persist_terminal_failure(
                        lambda: _write_suite_generation_state(
                            suite,
                            generation_id=generation_id,
                            state="publication_failed",
                            started_at=started_at,
                            ended_at=ended_at,
                            error_class=error_class,
                            publish_current=True,
                        )
                    )
                    raise

                def _write_published_suite_record() -> None:
                    """Step 4: write the immutable suite record once, state ``published``."""
                    _write_suite_generation_state(
                        suite,
                        generation_id=generation_id,
                        state="published",
                        started_at=started_at,
                        ended_at=ended_at,
                        publish_current=False,
                    )

                _log_on_failure(
                    _write_published_suite_record,
                    "failed to write the immutable suite generation record after "
                    "publication; the published result is unaffected",
                )

                def _refresh_published_suite_pointer_and_cache() -> None:
                    """Step 5: drive the pointer to ``published``, refresh the cache (best-effort)."""
                    _write_suite_generation_state(
                        suite,
                        generation_id=generation_id,
                        state="published",
                        started_at=started_at,
                        ended_at=ended_at,
                        write_record=False,
                    )
                    _copy_yaml_projection(immutable_summary, _suite_summary_path(suite))

                _log_on_failure(
                    _refresh_published_suite_pointer_and_cache,
                    "failed to refresh the current suite-generation pointer or compatibility "
                    "cache after publication; the published result is unaffected",
                )

            if absorbed.signum is not None:
                log.warning(
                    "Shutdown signal %d arrived during the suite publication "
                    "transaction for suite generation %s; the committed "
                    "publication wins and the shutdown is honored at the next "
                    "checkpoint.",
                    absorbed.signum,
                    generation_id,
                )
        except BaseException as exc:
            failed_error_class = type(exc).__name__
            # Mirrors the experiment-level guard in _run_experiment_outcome: if
            # the publication transaction already wrote a terminal record
            # (state "publication_failed") for this suite generation, it also
            # already drove the current pointer to that more specific state;
            # skip the generic write so it is not clobbered back to "failed".
            _persist_failed_state_unless_recorded(
                _suite_generation_record_path(suite, generation_id),
                lambda: _write_suite_generation_state(
                    suite,
                    generation_id=generation_id,
                    state="failed",
                    started_at=started_at,
                    ended_at=utc_now_iso(),
                    error_class=failed_error_class,
                    publish_current=True,
                ),
            )
            raise
    return results


def _claim_suite_generation(suite: Suite) -> str:
    """Create one exclusively owned suite-generation namespace.

    :param Suite suite: Suite whose suite-generations root is created if missing.
    :return str: Freshly minted UUID4 hex suite-generation id.
    :raises RuntimeError: No unused suite-generation id could be minted after
        10 attempts.
    """
    root = _suite_generations_dir(suite)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        generation_id = uuid4().hex
        try:
            _suite_generation_dir(suite, generation_id).mkdir()
        except FileExistsError:
            continue
        return generation_id
    raise RuntimeError("Could not mint an unused suite generation id after 10 attempts.")


def _validate_suite_generation_publishable(suite: Suite, generation_id: str) -> bytes:
    """Validate a suite generation's summary and component references before its commit.

    Suite mirror of :func:`_validate_generation_publishable` (review v0.5.16
    / blocker 3). Beyond the suite's own summary identity, every recorded
    component reference is chased: each study in the summary must name a
    study in the compiled plan, its recorded component experiment and
    generation id must resolve to that component's own immutable summary,
    and that component summary's complete artifact manifest must validate.
    This runs at publication time, while the current config is by definition
    the config that just produced the components, so chasing them through
    ``suite.experiment_for_study`` cannot be confused by later config drift.

    :param Suite suite: Suite whose generation is being published.
    :param str generation_id: Immutable suite-generation namespace to validate.
    :raises PublicationCommitError: The suite summary cannot be read back or
        does not name this suite generation.
    :raises PublicationAccessError: A component artifact cannot be read as the current user.
    :raises PublicationIntegrityError: A component manifest fails validation.
    :return bytes: Exact suite-summary bytes whose component graph was validated.
    """
    summary, summary_bytes = _validate_publishable_summary(
        summary_path=_suite_generation_summary_path(suite, generation_id),
        owner_key="suite",
        owner_value=suite.suite,
        id_key="suite_generation_id",
        id_value=generation_id,
        label="Suite generation",
    )

    def _fail(reason: str) -> PublicationCommitError:
        """Build one uniformly labeled suite publication-validation error.

        :param str reason: Specific validation failure being reported.
        :return PublicationCommitError: Error naming the suite generation and reason.
        """
        return PublicationCommitError(
            f"Suite generation {generation_id!r} failed publication validation: {reason}; "
            "refusing to advance the last-success pointer."
        )

    studies_by_name = {study.name: study for study in suite.studies}
    records = summary.get("studies")
    if not isinstance(records, list):
        raise _fail("summary has no study records")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise _fail("summary study record is malformed")
        name = str(record["name"])
        if name in seen or name not in studies_by_name:
            raise _fail(f"summary study record {name!r} is duplicated or unknown")
        seen.add(name)
        component = suite.experiment_for_study(studies_by_name[name])
        if record.get("experiment") != component.experiment:
            raise _fail(f"study {name!r} names a different component experiment")
        component_generation = record.get("experiment_generation_id")
        if not isinstance(component_generation, str) or not component_generation:
            raise _fail(f"study {name!r} has no component generation id")
        _validate_safe_name("generation", component_generation)
        component_summary = _read_pointer_target_summary(
            _generation_summary_path(component, component_generation),
            id_key="generation_id",
            target_id=component_generation,
            owner_key="experiment",
            owner_name=component.experiment,
        )
        if component_summary is None:
            raise _fail(
                f"study {name!r} component generation {component_generation!r} "
                "has no readable summary"
            )
        if "schema_version" not in component_summary:
            # Pre-manifest legacy component summary: identity gate only,
            # mirroring the read-side rule in _last_successful_generation_id.
            continue
        try:
            _validate_generation_manifest(
                _generation_dir(component, component_generation),
                component_generation,
                component_summary,
            )
        except PublicationAccessError:
            raise
        except PublicationIntegrityError as exc:
            raise _fail(f"study {name!r} component manifest is invalid ({exc})") from exc
    missing = set(studies_by_name) - seen
    if missing:
        raise _fail(f"summary is missing study record(s) {sorted(missing)}")
    try:
        # Same integrity contract the read side enforces: the suite summary's
        # own winner facts must be anchored to the hash-covered component
        # summaries before the pointer may advance (review v0.5.17 gap hunt).
        _validate_suite_summary_integrity(generation_id, summary)
    except PublicationAccessError:
        raise
    except PublicationIntegrityError as exc:
        raise _fail(str(exc)) from exc
    return summary_bytes


def _write_suite_generation_state(
    suite: Suite,
    *,
    generation_id: str,
    state: str,
    started_at: str,
    ended_at: str | None = None,
    error_class: str | None = None,
    publish_current: bool = True,
    write_record: bool = True,
) -> None:
    """Write one suite generation's current-pointer projection and/or immutable record.

    Mirrors :func:`_write_generation_state`'s split (review v0.5.15 / blocker
    3): the current suite-generation pointer is mutable progress bookkeeping
    with no monotonic guard, while the per-suite-generation record under
    ``suite_generations/<id>/generation.yaml`` is written only for a terminal
    state (``published`` / ``publication_failed`` / ``failed``), and only once.

    :param Suite suite: Suite whose generation state is written.
    :param str generation_id: Immutable suite-generation namespace being recorded.
    :param str state: Lifecycle state label (``"running"``, ``"published"``,
        ``"publication_failed"``, or ``"failed"``).
    :param str started_at: ISO timestamp when the suite invocation started.
    :param str | None ended_at: ISO timestamp when the suite invocation ended,
        or ``None`` while still running.
    :param str | None error_class: Optional exception class name to record for
        a failed or publication_failed state.
    :param bool publish_current: If ``True``, also overwrite the suite's
        current-generation pointer with this state.
    :param bool write_record: If ``False``, skip the write-once record even
        for a terminal state -- see :func:`_write_generation_state` for why
        the post-commit pointer refresh needs this.
    """
    payload = {
        "schema_version": 1,
        "suite": suite.suite,
        "suite_generation_id": generation_id,
        "suite_fingerprint": _suite_fingerprint(suite),
        "state": state,
        "started_at": started_at,
        "ended_at": ended_at,
        "error_class": error_class,
        "phasesweep_version": __version__,
    }
    if write_record and state in _TERMINAL_GENERATION_STATES:
        _write_generation_record_once(
            record_path=_suite_generation_record_path(suite, generation_id),
            generation_id=generation_id,
            state=state,
            payload=payload,
            label="suite generation",
        )
    if publish_current:
        _write_yaml_atomic(_suite_generation_path(suite), payload)


def _suite_summary_payload(
    suite: Suite,
    *,
    generation_id: str,
    started_at: str,
    ended_at: str,
    results: dict[str, dict[str, Winner]],
    promotion_decisions: dict[str, dict[str, Any]],
    component_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the immutable suite summary from the compiled historical plan.

    :param Suite suite: Compiled suite plan whose studies are iterated in
        declaration order.
    :param str generation_id: Immutable suite-generation namespace this summary belongs to.
    :param str started_at: ISO timestamp when the suite invocation started.
    :param str ended_at: ISO timestamp when the suite invocation ended.
    :param dict[str, dict[str, Winner]] results: Exposed winners keyed by study
        name, then phase name.
    :param dict[str, dict[str, Any]] promotion_decisions: Study-level promotion
        decisions keyed by study name.
    :param dict[str, dict[str, Any]] component_records: Per-study experiment
        provenance (experiment name, generation id, phase fingerprints) keyed
        by study name.
    :return dict[str, Any]: Immutable suite summary payload, including each
        study's exposed phases, promotion rule/decision, and component provenance.
    """
    studies: list[dict[str, Any]] = []
    for study_spec in suite.studies:
        experiment = suite.experiment_for_study(study_spec)
        exposed = results.get(study_spec.name, {})
        phases = []
        for phase in experiment.phases:
            winner = exposed.get(phase.name)
            if winner is None:
                phases.append({"name": phase.name, "comment": phase.comment, "exposed": False})
                continue
            phases.append(
                {
                    **_winner_summary_item(phase.name, winner),
                    "comment": phase.comment,
                    "exposed": True,
                }
            )
        studies.append(
            {
                "name": study_spec.name,
                "depends_on": study_spec.depends_on,
                "promotion_rule": (
                    None
                    if study_spec.promotion is None
                    else study_spec.promotion.model_dump(mode="json")
                ),
                "promotion": promotion_decisions.get(study_spec.name),
                **component_records[study_spec.name],
                "phases": phases,
            }
        )
    fingerprint = _suite_fingerprint(suite)
    return {
        "schema_version": SUITE_SUMMARY_SCHEMA_VERSION,
        "suite": suite.suite,
        "suite_generation_id": generation_id,
        "suite_fingerprint": fingerprint,
        "started_at": started_at,
        "ended_at": ended_at,
        "phasesweep_version": __version__,
        "promotion_decisions": list(promotion_decisions.values()),
        "studies": studies,
    }
