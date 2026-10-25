"""Experiment execution and publication orchestration."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.attempts as attempt_ops
import phasesweep.engine.evidence as evidence_ops
import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.generation as generation_ops
import phasesweep.engine.guards as guard_ops
import phasesweep.engine.ledger as ledger_ops
import phasesweep.engine.locking as locking_ops
import phasesweep.engine.paths as path_ops
import phasesweep.engine.publication_validation as validation_ops
import phasesweep.engine.resume as resume_ops
from phasesweep._metadata import __version__
from phasesweep.config.common import _validate_safe_name
from phasesweep.config.models import (
    Experiment,
    _metric_semantics_payload,
    experiment_disclosures,
)
from phasesweep.engine.errors import (
    IncompleteJournalRecordError,
    StudyStorageUnavailableError,
    phase_timeout,
)
from phasesweep.engine.phase import _run_phase
from phasesweep.engine.preview import (
    PhasePreview,
    RecordedPhase,
    _dry_run_total_line,
    _preview_ledger,
    _preview_phase,
)
from phasesweep.engine.selection import _winner_summary_item
from phasesweep.engine.state import GENERATION_SUMMARY_SCHEMA_VERSION, Winner
from phasesweep.engine.trial import _environment_identity
from phasesweep.errors import OperatorAction, ProcessCleanupUncertainError
from phasesweep.runtime.files import ensure_artifact_dir, require_supported_runtime
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    service_pending_shutdown,
    signal_handler_scope,
)

log = logging.getLogger("phasesweep.engine.run")


def _preflight_missing_reached_phase_environments(
    experiment: Experiment,
    existing_studies: Mapping[str, object],
    *,
    from_phase: str | None,
) -> None:
    """Validate launch environments for reached phases with no durable study.

    A phase absent from an otherwise bound tree has no completed trial to
    replay, so it must be able to launch before this invocation tops up an
    earlier phase. Existing studies remain deferred to recovery, which can
    distinguish new work from an offline-safe no-op.

    :param Experiment experiment: Parsed experiment supplying phase environments.
    :param Mapping[str, object] existing_studies: Durable studies found under the lock.
    :param str | None from_phase: Optional first phase this invocation can execute.
    :raises PhaseSweepError: A missing reached phase has an invalid launch environment.
    """
    for _index, phase in experiment.phases_from(from_phase):
        if phase.name not in existing_studies:
            _environment_identity(experiment, phase.name)


@dataclass(frozen=True)
class TerminalReport:
    """One engine invocation's primary outcome and independent cleanup evidence.

    ``generation_id`` is the invocation's generation id. A generation namespace
    exists under it only when the invocation launched a trial or published.
    ``failure_stage`` is ``"execution"`` for a failure after that claim and
    ``"preflight"`` for one before it, and ``None`` on success.
    ``winners`` holds the winners the engine captured under the experiment lock
    on success, so a terminal-result consumer need not re-read storage. It is
    ``None`` on every failure path. ``replayed_generation_id`` names the published
    generation whose winners a successful invocation replayed without writing a
    generation, and is ``None`` otherwise.
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
    replayed_generation_id: str | None = None


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
        summary: Mapping[str, Any],
    ) -> None:
        """Durably prepare sidecar state before the publication pointer advances.

        :param Experiment experiment: Exact experiment configuration being published.
        :param str generation_id: Generation whose validated result is ready to commit.
        :param Mapping[str, Winner] winners: Engine-selected winners for that generation.
        :param Mapping[str, Any] summary: Validated generation summary awaiting publication.
        """
        ...

    def committed(self, *, generation_id: str) -> None:
        """Record that the publication pointer advanced to the prepared generation.

        :param str generation_id: Generation whose pointer commit completed.
        """
        ...


def _terminal_report_from_cleanup(
    generation_id: str,
    cleanup: attempt_ops._PreflightCleanupReport,
    *,
    primary_error: BaseException | None,
    failure_stage: str | None,
    winners: Mapping[str, Winner] | None = None,
    replayed_generation_id: str | None = None,
) -> TerminalReport:
    """Freeze accumulated cleanup evidence into one terminal report.

    :param str generation_id: Generation whose invocation ended.
    :param attempt_ops._PreflightCleanupReport cleanup: Mutable cleanup evidence to freeze.
    :param BaseException | None primary_error: Invocation's authoritative error.
    :param str | None failure_stage: Stage where the invocation failed.
    :param Mapping[str, Winner] | None winners: Successful published winners.
    :param str | None replayed_generation_id: Published generation whose winners a successful
        invocation replayed without writing a generation.
    :return TerminalReport: Immutable terminal outcome snapshot.
    """
    return TerminalReport(
        generation_id=generation_id,
        primary_error=primary_error,
        cleanup_confirmed=cleanup.cleanup_confirmed,
        recovered_attempt_ids=frozenset(cleanup.recovered_attempt_ids),
        recovered_attempt_generations=MappingProxyType(dict(cleanup.recovered_attempt_generations)),
        uncertain_attempt_ids=frozenset(cleanup.uncertain_attempt_ids),
        cleanup_error=cleanup.error,
        failure_stage=failure_stage,
        winners=MappingProxyType(dict(winners)) if winners is not None else None,
        replayed_generation_id=replayed_generation_id,
    )


def _failure_stage(generation: generation_ops.InvocationGeneration) -> str:
    """Name the stage a failed invocation was in, from whether it claimed its generation.

    :param generation_ops.InvocationGeneration generation: The invocation's generation.
    :return str: ``"execution"`` once the generation is claimed, else ``"preflight"``.
    """
    return "execution" if generation.claimed else "preflight"


def run_experiment(
    experiment: Experiment,
    *,
    from_phase: str | None = None,
    dry_run: bool = False,
    terminal_callback: Callable[[TerminalReport], None] | None = None,
    publication_hook: PublicationHook | None = None,
    generation_id: str | None = None,
) -> dict[str, Winner]:
    """Run the ordered phases and return their winners.

    Live runs hold the experiment lock across all phases and install shutdown
    handlers. Dry runs inspect recorded state and render commands without writes
    or trainer subprocesses; GPU checks may read local inventory through nvidia-smi.
    Dry-run checks omit recovery-dependent checks, trainer
    environments and trial evidence. Fingerprints with placeholder ancestry
    remain unknown until the parent publishes a winner.

    :param Experiment experiment: Parsed experiment config.
    :param str | None from_phase: Resume phase; validate and load earlier published winners.
        None starts at the first phase, reusing recorded trials where applicable.
    :param bool dry_run: Preview remaining trial counts and one command per phase.
    :param Callable[[TerminalReport], None] | None terminal_callback: Synchronous diagnostic
        callback under the experiment lock.
    :param PublicationHook | None publication_hook: Required sidecar around publication
        commit, if supplied.
    :param str | None generation_id: Invocation identity, generated if omitted. Supplied names
        may contain only alphanumerics, underscores, and dashes.
    :return dict[str, Winner]: Winners in declaration order. Dry runs return
        replayed winners where available and placeholders elsewhere.
    :raises NoFeasibleTrialError: A phase exhausted its failure limit or has no feasible trial.
    :raises UnsafeProcessCleanupError: A trial's process group may still be alive.
    :raises TrialEvidenceMissingError: A candidate's evidence or objective source changed.
    :raises RunRequestError: The supplied generation id already exists.
    :raises PhaseSweepError: Preflight, storage, publication, or recovery refused the run.
    :raises RuntimeError: An engine invariant failed.
    :raises ValueError: An unsafe generation id or unknown resume phase was supplied.
    :raises ResumeWinnerMissingError: A skipped phase has no published winner; a fresh
        artifact tree is refused before anything is written.
    """
    # Revalidate copied models while preserving mutable defaults and field-set state.
    experiment = Experiment.model_validate(experiment)
    if from_phase is not None and from_phase not in {phase.name for phase in experiment.phases}:
        raise ValueError(f"Unknown --from-phase value {from_phase!r}.")
    if dry_run:
        return _run_experiment_inner(
            experiment,
            from_phase=from_phase,
            ledger=None,
            generation=None,
        )
    # Fresh-tree launch checks precede writes; existing trees defer them until work remains.
    environments_preflighted = not path_ops._artifact_root_binding_path(experiment).exists()
    if environments_preflighted:
        # A fresh tree holds no publication, so a resume that needs earlier winners cannot
        # succeed; refuse it before the run creates the tree.
        resume_ops._preflight_skipped_winners(experiment, from_phase=from_phase, run_deadline=None)
        _preflight_missing_reached_phase_environments(experiment, {}, from_phase=from_phase)
    winners = _run_experiment_outcome(
        experiment,
        from_phase=from_phase,
        terminal_callback=terminal_callback,
        publication_hook=publication_hook,
        generation_id=generation_id,
        environments_preflighted=environments_preflighted,
    )
    # Deliver shutdown deferred by the publication commit before returning to the caller.
    try:
        service_pending_shutdown()
    except PhaseSweepShutdown as exc:
        exc.published_result_committed = True
        raise
    return dict(winners)


def _run_experiment_outcome(
    experiment: Experiment,
    *,
    from_phase: str | None = None,
    terminal_callback: Callable[[TerminalReport], None] | None = None,
    publication_hook: PublicationHook | None = None,
    generation_id: str | None = None,
    environments_preflighted: bool = False,
) -> Mapping[str, Winner]:
    """Run the phases and publish their winners while holding the experiment lock.

    The invocation's generation is created only when a phase launches a trial or the
    winners differ from the current publication; a refusal or a no-op replay leaves
    no generation and leaves both generation pointers as they were.

    :param Experiment experiment: Parsed experiment config.
    :param str | None from_phase: Optional resume point; see :func:`run_experiment`.
    :param Callable[[TerminalReport], None] | None terminal_callback: Optional diagnostic
        callback.
    :param PublicationHook | None publication_hook: Optional required publication sidecar.
    :param str | None generation_id: Optional caller-owned invocation identity.
    :param bool environments_preflighted: Launch environments were checked before any
        writes to a fresh artifact tree.
    :return Mapping[str, Winner]: This invocation's winners, captured under the lock.
    :raises TrialEvidenceMissingError: Candidate evidence failed preflight or winner validation.
    :raises ProcessCleanupUncertainError: Post-failure cleanup was not confirmed;
        the original failure is chained.
    :raises PhaseSweepShutdown: Shutdown during MLflow mirroring; the exception
        records that the result is already committed.
    :raises BaseException: The run's failure after recording failed state, or a
        control-flow exception that interrupted reconciliation, chained from it.
    """
    require_supported_runtime()

    requested_generation_id = (
        None if generation_id is None else _validate_safe_name("generation", generation_id)
    )
    # Restore signal ownership on every exit, including lock acquisition failures.
    with (
        signal_handler_scope(),
        locking_ops._experiment_lock(experiment),
        contextlib.ExitStack() as run_stack,
    ):
        try:
            claimed = ledger_ops.claim_ledger(
                ledger_ops.validate_ledger(experiment), from_phase=from_phase
            )
        except StudyStorageUnavailableError as exc:
            # Unreadable ownership state cannot rule out earlier live attempts.
            unknown = (
                "Artifact ownership could not be checked because required persistent "
                f"study state is unavailable: {exc} Cleanup state is therefore unknown."
            )
            if isinstance(exc, IncompleteJournalRecordError):
                # A partial final record's repair did not finish, and it names its
                # own remedy; restoring the whole ledger instead would be wrong.
                raise ProcessCleanupUncertainError.rewrap(
                    exc, f"{unknown} For an MCP run, then run phasesweep mcp recover-run."
                ) from exc
            raise ProcessCleanupUncertainError(
                f"{unknown} An MCP run requires phasesweep mcp recover-run after ledger repair.",
                action=OperatorAction.RESTORE_LEDGER,
            ) from exc
        if not environments_preflighted:
            _preflight_missing_reached_phase_environments(
                experiment,
                claimed.studies,
                from_phase=from_phase,
            )
        ensure_artifact_dir(path_ops._experiment_dir(experiment))
        run_stack.enter_context(artifact_io._file_log_handler(path_ops._run_log_path(experiment)))
        generation = generation_ops.InvocationGeneration(
            experiment,
            generation_id=generation_ops._new_generation_id(experiment, requested_generation_id),
            caller_owned_id=requested_generation_id is not None,
            from_phase=from_phase,
        )
        generation_id = generation.generation_id
        terminal_error: BaseException | None = None
        terminal_report: TerminalReport | None = None
        cleanup = attempt_ops._PreflightCleanupReport()
        try:
            run_deadline = (
                time.monotonic() + experiment.timeout_seconds_per_run
                if experiment.timeout_seconds_per_run is not None
                else None
            )
            existing_studies = guard_ops._preflight_existing_studies(
                claimed,
                cleanup_report=cleanup,
                from_phase=from_phase,
            )
            # Validate candidate evidence after recovery and before launching new work.
            evidence_ops._validate_selection_evidence(experiment, existing_studies)
            resume_ops._reject_bound_descendant_topups(
                experiment,
                from_phase=from_phase,
                existing_studies=existing_studies,
            )
            resume_ops._reject_fixed_targets_without_study(
                experiment,
                from_phase=from_phase,
                existing_studies=existing_studies,
            )
            preloaded_winners = resume_ops._preflight_skipped_winners(
                experiment,
                from_phase=from_phase,
                run_deadline=run_deadline,
            )
            resume_ops._preflight_reached_fingerprint(
                experiment,
                from_phase=from_phase,
                preloaded_winners=preloaded_winners,
                existing_studies=existing_studies,
            )
            result = _run_experiment_inner(
                experiment,
                from_phase=from_phase,
                ledger=claimed,
                generation=generation,
                preloaded_winners=preloaded_winners,
                run_deadline=run_deadline,
                publication_hook=publication_hook,
            )
        except BaseException as exc:
            terminal_error = exc
            control_error: BaseException | None = None
            if generation.claimed:
                # Only a claimed generation launched work whose studies need reconciling.
                reconciliation = attempt_ops._PreflightCleanupReport()
                try:
                    guard_ops._reconcile_existing_studies(
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
            original_cleanup_uncertain = isinstance(exc, ProcessCleanupUncertainError) or (
                isinstance(exc, PhaseSweepShutdown) and not exc.report.cleanup_confirmed
            )
            if original_cleanup_uncertain:
                # Preserve prior cleanup uncertainty even if a signal interrupts reconciliation.
                cleanup.mark_uncertain(exc)
            primary_error = control_error or exc
            shutdown_cleanup_uncertain = (
                isinstance(primary_error, PhaseSweepShutdown)
                and not primary_error.report.cleanup_confirmed
            )
            if shutdown_cleanup_uncertain:
                cleanup.mark_uncertain(primary_error)
            failed_error_class = type(primary_error).__name__
            if generation.claimed:
                # Keep a more specific publication_failed record and pointer intact.
                generation_ops._persist_failed_state_unless_recorded(
                    path_ops._generation_record_path(experiment, generation_id),
                    lambda: generation_ops._write_generation_state(
                        experiment,
                        generation_id=generation_id,
                        state="failed",
                        from_phase=from_phase,
                        publish_current=True,
                        error_class=failed_error_class,
                    ),
                )
            if experiment.mlflow is not None and cleanup.cleanup_confirmed:
                from phasesweep.tracking.publication import publish_failed_generation

                try:
                    publish_failed_generation(experiment, generation_id, primary_error, claimed)
                except ProcessCleanupUncertainError as tracking_exc:
                    cleanup.mark_uncertain(tracking_exc)
                    log.exception("MLflow failure delivery cleanup could not be confirmed")
                except BaseException as tracking_exc:  # primary error must survive
                    if (
                        isinstance(tracking_exc, PhaseSweepShutdown)
                        and not tracking_exc.report.cleanup_confirmed
                    ):
                        cleanup.mark_uncertain(tracking_exc)
                    log.exception(
                        "MLflow failure delivery did not complete; preserving the original error"
                    )
            terminal_report = _terminal_report_from_cleanup(
                generation_id,
                cleanup,
                primary_error=primary_error,
                failure_stage=_failure_stage(generation),
            )
            if (
                not cleanup.cleanup_confirmed
                and not isinstance(
                    primary_error,
                    (ProcessCleanupUncertainError, StudyStorageUnavailableError),
                )
                and not shutdown_cleanup_uncertain
                and control_error is None
            ):
                # Cleanup recovery takes precedence; preserve the original failure as the cause.
                if isinstance(cleanup.error, ProcessCleanupUncertainError):
                    raise ProcessCleanupUncertainError.rewrap(
                        cleanup.error,
                        "The run failed and subsequent process cleanup could not be confirmed. "
                        f"{cleanup.error}",
                    ) from primary_error
                raise ProcessCleanupUncertainError(
                    "The run failed and subsequent process cleanup could not be confirmed.",
                    action=OperatorAction.RUN_RECOVER_RUN,
                ) from primary_error
            if control_error is not None:
                raise control_error from exc
            raise
        else:
            # Outside the except clause: the pointer has committed, so nothing
            # the MLflow mirror raises may enter the failure handling above.
            try:
                # Success without a claim published nothing: the replay matched the publication.
                if experiment.mlflow is not None and generation.claimed:
                    _mirror_publication(experiment, generation_id, cleanup)
            finally:
                terminal_report = _terminal_report_from_cleanup(
                    generation_id,
                    cleanup,
                    primary_error=None,
                    failure_stage=None,
                    winners=result,
                    replayed_generation_id=generation.replayed_generation_id,
                )
            return MappingProxyType(dict(result))
        finally:
            if terminal_callback is not None:
                try:
                    if terminal_report is None:
                        terminal_report = _terminal_report_from_cleanup(
                            generation_id,
                            cleanup,
                            primary_error=terminal_error,
                            failure_stage=_failure_stage(generation),
                        )
                    terminal_callback(terminal_report)
                except BaseException:
                    # Callback failures cannot replace the committed result or the engine error.
                    log.exception("terminal callback failed; the engine outcome is unchanged")


def _mirror_publication(
    experiment: Experiment,
    generation_id: str,
    cleanup: attempt_ops._PreflightCleanupReport,
) -> None:
    """Mirror a committed generation to MLflow without changing the run's outcome.

    Once the last-success pointer has committed, nothing may fail the run. An
    ordinary failure only leaves the mirror for ``phasesweep sync-tracking``,
    which rebuilds any record this left unsaved. An upload worker whose cleanup
    is uncertain is recorded as cleanup evidence, and a shutdown still ends the
    run, marked as arriving after the published result.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: Generation whose publication just committed.
    :param attempt_ops._PreflightCleanupReport cleanup: Cleanup evidence for the
        terminal report.
    :raises PhaseSweepShutdown: A shutdown signal arrived during the mirror.
    """
    from phasesweep.tracking.publication import publish_generation

    try:
        publish_generation(experiment, generation_id)
    except ProcessCleanupUncertainError as exc:
        cleanup.mark_uncertain(exc)
        log.exception(
            "An MLflow upload worker for published generation %s may still be running; the "
            "published result is unaffected. Run `phasesweep sync-tracking` with the same "
            "config to retry its cleanup and the uploads.",
            generation_id,
        )
    except PhaseSweepShutdown as exc:
        if not exc.report.cleanup_confirmed:
            cleanup.mark_uncertain(exc)
        exc.published_result_committed = True
        raise
    except Exception:
        log.exception(
            "MLflow delivery records for published generation %s were not all saved; the "
            "published result is unaffected. `phasesweep sync-tracking` rebuilds them from "
            "the published generation.",
            generation_id,
        )


def _run_experiment_inner(
    experiment: Experiment,
    *,
    from_phase: str | None,
    ledger: ledger_ops.ClaimedLedger | None,
    generation: generation_ops.InvocationGeneration | None,
    preloaded_winners: dict[str, Winner] | None = None,
    run_deadline: float | None = None,
    publication_hook: PublicationHook | None = None,
) -> dict[str, Winner]:
    """Execute the phase sequence under the caller's locks and signal handlers.

    A live run that launched no trial claims its generation and publishes only when its
    winners differ from the validated publication, or nothing validates as published.

    :param Experiment experiment: Parsed experiment config.
    :param str | None from_phase: Resume point whose earlier winners are loaded from disk.
    :param ledger_ops.ClaimedLedger | None ledger: Claimed ledger, or None for read-only
        command and trial-count previews.
    :param generation_ops.InvocationGeneration | None generation: The invocation's
        unclaimed generation, or None for a dry run.
    :param dict[str, Winner] | None preloaded_winners: Validated skipped-phase winners captured
        before generation commit.
    :param float | None run_deadline: Monotonic deadline including preflight and stale cleanup.
    :param PublicationHook | None publication_hook: Optional required publication sidecar.
    :return dict[str, Winner]: Phase winners, or dry-run placeholders and replayed winners.
    :raises TimeoutError: The run deadline expired before a phase started.
    :raises PublicationAccessError: A recorded dry-run publication is inaccessible.
    :raises PublicationIntegrityError: A recorded dry-run publication fails validation.
    :raises PhaseSweepError: Dry-run inspection found a launch refusal; see
        :func:`phasesweep.engine.preview._preview_ledger`.
    """
    dry_run = ledger is None
    skip_until = from_phase is not None
    winners: dict[str, Winner] = {}
    recorded: dict[str, RecordedPhase | None] = {}
    previews: list[PhasePreview] = []
    if run_deadline is None and not dry_run and experiment.timeout_seconds_per_run is not None:
        run_deadline = time.monotonic() + experiment.timeout_seconds_per_run

    if dry_run:
        for _phase, lines in experiment_disclosures(experiment):
            for warning, line in lines:
                if warning:
                    log.warning("DRY RUN warning: %s", line)
                else:
                    log.info("DRY RUN %s", line)
        recorded = _preview_ledger(experiment, from_phase=from_phase)
        preloaded_winners = resume_ops._preflight_skipped_winners(
            experiment, from_phase=from_phase, run_deadline=None
        )
    for phase in experiment.phases:
        using_preloaded_winner = (
            skip_until and phase.name != from_phase and preloaded_winners is not None
        )
        if (
            not using_preloaded_winner
            and run_deadline is not None
            and time.monotonic() >= run_deadline
        ):
            raise phase_timeout(
                f"Run wallclock deadline reached before phase {phase.name!r} could start.",
                scope="run",
                persistent=experiment.resolved_storage is not None,
            )
        if (
            using_preloaded_winner
            and preloaded_winners is not None
            and phase.name not in preloaded_winners
        ):
            log.info("phase=%s SKIPPED (no published or required winner)", phase.name)
            continue
        # Resolve current parents before validating skipped winners or running the phase.
        inherited = {p: winners[p] for p in phase.inherits}

        if skip_until and phase.name != from_phase:
            assert preloaded_winners is not None
            winners[phase.name] = preloaded_winners[phase.name]
            if generation is not None:
                generation.save_winner(phase.name, winners[phase.name])
            log.info("phase=%s SKIPPED (using preflight-validated winner)", phase.name)
            continue
        skip_until = False

        if ledger is None:
            previews.append(_preview_phase(experiment, phase, inherited, recorded.get(phase.name)))
            winner = previews[-1].winner
        else:
            assert generation is not None
            winner = _run_phase(
                experiment,
                phase,
                inherited,
                generation=generation,
                ledger=ledger,
                run_deadline=run_deadline,
            )
        if generation is not None:
            generation.save_winner(phase.name, winner)
            group = winner.repeats
            completion = winner.completion
            log.info(
                "phase=%s WINNER trial=%d metric=%g params=%s%s%s",
                phase.name,
                winner.trial_number,
                winner.metric,
                winner.params,
                ""
                if group is None
                else (
                    f" ({group['reduce']} over {group['spread']['n']} repeats of "
                    f"{group['key']!r}: sd={group['spread']['sd']:g}, "
                    f"range {group['spread']['min']:g}..{group['spread']['max']:g}, "
                    f"members {[member['trial_number'] for member in group['members']]})"
                ),
                ""
                if not completion.get("incomplete")
                else (
                    f" INCOMPLETE {completion['finished_trials']}/"
                    f"{completion['requested_trials']} finished "
                    f"timeout_scope={completion['timeout_scope']}"
                ),
            )
        winners[phase.name] = winner

    if generation is None:
        log.info("DRY RUN %s", _dry_run_total_line(previews))
        log.info("DRY RUN complete. No trials launched, no summary written.")
        return winners

    if not generation.claimed:
        published_id = generation_ops._publication_holds(experiment, winners)
        if published_id is not None:
            log.info(
                "No trial launched and generation %s already publishes these winners; "
                "the publication is unchanged and no new generation was written.",
                published_id,
            )
            generation.replayed_generation_id = published_id
            return winners
    generation_id = generation.claim()
    summary_path = path_ops._generation_summary_path(experiment, generation_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Freeze artifact hashes and result labels together for generation-consistent reads.
    summary = {
        "schema_version": GENERATION_SUMMARY_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "phasesweep_version": __version__,
        "config_fingerprint": fingerprint_ops._experiment_semantic_fingerprint(experiment),
        "metric": _metric_semantics_payload(experiment.metric),
        "phase_plan": [
            {"name": phase.name, "comment": phase.comment} for phase in experiment.phases
        ],
        "phases": [_winner_summary_item(pname, w) for pname, w in winners.items()],
        "artifacts": validation_ops._generation_artifact_manifest(experiment, generation_id),
    }
    if experiment.override_format == "argparse":
        summary["trainer_config"] = experiment.trainer_config
    artifact_io._write_yaml_atomic(summary_path, summary)
    generation_ops._publish_generation(
        experiment,
        generation_id,
        from_phase=from_phase,
        winners=winners,
        publication_hook=publication_hook,
    )
    log.info("Wrote %s", summary_path)

    return winners
