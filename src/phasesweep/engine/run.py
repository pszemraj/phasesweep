"""Public engine entrypoints for experiment execution and status."""

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
from phasesweep.engine.confound import _describe_flagged
from phasesweep.engine.errors import IncompleteJournalRecordError, StudyStorageUnavailableError
from phasesweep.engine.phase import _run_phase
from phasesweep.engine.preview import (
    PhasePreview,
    RecordedPhase,
    _dry_run_total_line,
    _preview_ledger,
    _preview_phase,
)
from phasesweep.engine.read import read_status
from phasesweep.engine.selection import _winner_summary_item
from phasesweep.engine.state import GENERATION_SUMMARY_SCHEMA_VERSION, Winner
from phasesweep.engine.trial import _environment_identity
from phasesweep.errors import OperatorAction, ProcessCleanupUncertainError
from phasesweep.runtime.files import ensure_artifact_dir, require_posix_runtime
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    service_pending_shutdown,
    signal_handler_scope,
)

log = logging.getLogger("phasesweep.engine.run")

# read_status phase keys the CLI status view replaces with ``name`` and ``winner``.
_MCP_ONLY_PHASE_KEYS = frozenset({"phase", "winner_present", "running_attempts"})


def _preflight_missing_reached_phase_environments(
    experiment: Experiment,
    existing_studies: Mapping[str, object],
    *,
    from_phase: str | None,
) -> None:
    """Validate launch environments for reached phases with no durable study.

    A phase absent from an otherwise bound tree has no completed trial to
    replay, so it must be able to launch before this invocation claims a
    generation or tops up an earlier phase. Existing studies remain deferred to
    recovery, which can distinguish new work from an offline-safe no-op.

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

    ``winners`` is the engine's own authoritative winner mapping, populated on
    the success path while the experiment lock is still held: a
    terminal-result consumer (the MCP runner's snapshot capture) must be able
    to freeze the published outcome from the exact in-memory state the engine
    just returned, instead of re-deriving it from mutable storage or optional
    diagnostic files whose absence would then contradict an engine-defined
    success. ``None`` on every failure path.
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
) -> TerminalReport:
    """Freeze accumulated cleanup evidence into one terminal report.

    :param str generation_id: Generation whose invocation ended.
    :param attempt_ops._PreflightCleanupReport cleanup: Mutable cleanup evidence to freeze.
    :param BaseException | None primary_error: Invocation's authoritative error.
    :param str | None failure_stage: Stage where the invocation failed.
    :param Mapping[str, Winner] | None winners: Successful published winners.
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
    )


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

    :param experiment: Parsed experiment config.
    :param from_phase: Resume phase; validate and load earlier published winners.
        None starts at the first phase, reusing recorded trials where applicable.
    :param dry_run: Preview remaining trial counts and one command per phase.
    :param terminal_callback: Synchronous diagnostic callback under the experiment lock.
    :param publication_hook: Required sidecar around publication commit, if supplied.
    :param generation_id: Invocation identity, generated if omitted. Supplied names
        may contain only alphanumerics, underscores, and dashes.
    :return Mapping[str, Winner]: Winners in declaration order. Dry runs return
        replayed winners where available and placeholders elsewhere.
    :raises NoFeasibleTrialError: A phase exhausted its failure limit or has no feasible trial.
    :raises UnsafeProcessCleanupError: A trial's process group may still be alive.
    :raises TrialEvidenceMissingError: A candidate's evidence or objective source changed.
    :raises RunRequestError: The supplied generation id already exists.
    :raises PhaseSweepError: Preflight, storage, publication, or recovery refused the run.
    :raises RuntimeError: An engine invariant failed.
    :raises ValueError: An unsafe generation id or unknown resume phase was supplied.
    :raises FileNotFoundError: A skipped phase has no published winner.
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
            generation_id=None,
        )
    # Fresh-tree launch checks precede writes; existing trees defer them until work remains.
    environments_preflighted = not path_ops._artifact_root_binding_path(experiment).exists()
    if environments_preflighted:
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
    """Run and publish one generation while holding the experiment lock.

    :param experiment: Parsed experiment config.
    :param from_phase: Optional resume point; see :func:`run_experiment`.
    :param terminal_callback: Optional diagnostic callback.
    :param publication_hook: Optional required publication sidecar.
    :param generation_id: Optional caller-owned invocation identity.
    :param environments_preflighted: Launch environments were checked before any
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
    require_posix_runtime()

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
        generation_id = generation_ops._claim_generation(experiment, requested_generation_id)
        terminal_error: BaseException | None = None
        terminal_report: TerminalReport | None = None
        cleanup = attempt_ops._PreflightCleanupReport()
        generation_prepared = False
        try:
            run_deadline = (
                time.monotonic() + experiment.timeout_seconds_per_run
                if experiment.timeout_seconds_per_run is not None
                else None
            )
            generation_ops._write_generation_state(
                experiment,
                generation_id=generation_id,
                state="preflighting",
                from_phase=from_phase,
                publish_current=True,
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
            generation_ops._write_generation_state(
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
                ledger=claimed,
                generation_id=generation_id,
                preloaded_winners=preloaded_winners,
                run_deadline=run_deadline,
                publication_hook=publication_hook,
            )
        except BaseException as exc:
            terminal_error = exc
            control_error: BaseException | None = None
            if generation_prepared:
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
            failed_generation_id = generation_id
            failed_error_class = type(primary_error).__name__
            # Keep a more specific publication_failed record and pointer intact.
            generation_ops._persist_failed_state_unless_recorded(
                path_ops._generation_record_path(experiment, failed_generation_id),
                lambda: generation_ops._write_generation_state(
                    experiment,
                    generation_id=failed_generation_id,
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
                except BaseException as tracking_exc:  # noqa: BLE001 - primary error must survive
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
                failure_stage="execution" if generation_prepared else "preflight",
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
                if experiment.mlflow is not None:
                    _mirror_publication(experiment, generation_id, cleanup)
            finally:
                terminal_report = _terminal_report_from_cleanup(
                    generation_id,
                    cleanup,
                    primary_error=None,
                    failure_stage=None,
                    winners=result,
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
                            failure_stage=("execution" if generation_prepared else "preflight"),
                        )
                    terminal_callback(terminal_report)
                except BaseException:  # noqa: BLE001
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
    generation_id: str | None,
    preloaded_winners: dict[str, Winner] | None = None,
    run_deadline: float | None = None,
    publication_hook: PublicationHook | None = None,
) -> dict[str, Winner]:
    """Execute the phase sequence under the caller's locks and signal handlers.

    :param experiment: Parsed experiment config.
    :param from_phase: Resume point whose earlier winners are loaded from disk.
    :param ledger: Claimed ledger, or None for read-only command and trial-count previews.
    :param generation_id: Invocation identity, or None for a dry run.
    :param preloaded_winners: Validated skipped-phase winners captured before generation commit.
    :param run_deadline: Monotonic deadline including preflight and stale cleanup.
    :param publication_hook: Optional required publication sidecar.
    :return Mapping[str, Winner]: Phase winners, or dry-run placeholders and replayed winners.
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
            raise TimeoutError(
                f"Run wallclock deadline reached before phase {phase.name!r} could start."
            )
        if (
            using_preloaded_winner
            and preloaded_winners is not None
            and phase.name not in preloaded_winners
        ):
            log.info("phase=%s SKIPPED (not inherited by a reached phase)", phase.name)
            continue
        # Resolve current parents before validating skipped winners or running the phase.
        inherited = {p: winners[p] for p in phase.inherits}

        if skip_until and phase.name != from_phase:
            assert preloaded_winners is not None
            winners[phase.name] = preloaded_winners[phase.name]
            if not dry_run:
                assert generation_id is not None
                artifact_io._save_winner(
                    experiment,
                    phase.name,
                    winners[phase.name],
                    generation_id=generation_id,
                )
            log.info("phase=%s SKIPPED (using preflight-validated winner)", phase.name)
            continue
        skip_until = False

        _warn_flagged_inheritance(phase.name, inherited)
        if ledger is None:
            previews.append(_preview_phase(experiment, phase, inherited, recorded.get(phase.name)))
            winner = previews[-1].winner
        else:
            assert generation_id is not None
            winner = _run_phase(
                experiment,
                phase,
                inherited,
                generation_id=generation_id,
                ledger=ledger,
                run_deadline=run_deadline,
            )
        if not dry_run:
            assert generation_id is not None
            artifact_io._save_winner(
                experiment,
                phase.name,
                winner,
                generation_id=generation_id,
            )
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

    if dry_run:
        log.info("DRY RUN %s", _dry_run_total_line(previews))
        log.info("DRY RUN complete. No trials launched, no summary written.")
        return winners

    assert generation_id is not None
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


def _warn_flagged_inheritance(phase_name: str, inherited: Mapping[str, Winner]) -> None:
    """Warn when a phase is about to build on a winner whose selection flagged confounds.

    Downstream phases inherit a winner's values, not its file, so without this
    the verdict stays invisible unless someone opens the upstream winner. The
    in-process and ``--from-phase`` paths both resolve ``inherited`` before
    running a phase, so this one call covers both.

    :param str phase_name: Phase about to run.
    :param Mapping[str, Winner] inherited: Winners the phase inherits, by phase.
    """
    for parent, winner in inherited.items():
        if winner.confound is None:  # dry-run placeholder
            continue
        flagged = _describe_flagged(winner.confound)
        if flagged:
            log.warning(
                "[%s] inherits the winner of phase %r, whose selection flagged potential "
                "confounds: %s. Its winner's confound block has the evidence; confirm they do "
                "not undermine that choice before relying on this phase's results.",
                phase_name,
                parent,
                "; ".join(flagged),
            )


def _cli_phase_payloads(
    experiment: Experiment,
    phases: list[dict[str, Any]],
    published_generation_id: str | None,
) -> list[dict[str, Any]]:
    """Rebuild ``read_status``'s path-free phase payloads into the CLI's view.

    Use the unpinned published generation for winner paths. Replace MCP's ``phase``,
    ``winner_present`` and ``running_attempts`` with CLI ``name`` and ``winner`` fields.

    :param Experiment experiment: Experiment config with artifact root details.
    :param list[dict[str, Any]] phases: Path-free phase payloads from ``read_status``.
    :param str | None published_generation_id: Already-resolved last-success generation id.
    :return list[dict[str, Any]]: Path-bearing phase payloads for the CLI status view.
    """
    cli_phases: list[dict[str, Any]] = []
    for phase in phases:
        winner_path = (
            path_ops._generation_winner_path(experiment, published_generation_id, phase["phase"])
            if phase["winner_present"] and published_generation_id is not None
            else None
        )
        cli_phases.append(
            {
                **{k: v for k, v in phase.items() if k not in _MCP_ONLY_PHASE_KEYS},
                "name": phase["phase"],
                "winner": None if winner_path is None else str(winner_path),
            }
        )
    return cli_phases


def experiment_status(experiment: Experiment) -> dict[str, Any]:
    """Collect CLI status from one read-only generation snapshot.

    Return experiment and ledger paths, current/published/represented generation
    ids, publication integrity, and phase counts and winner paths. MLflow configs
    also include delivery counts across generations. Winner paths use the captured
    published generation. Fresh experiments have null generation ids; root-level
    projections do not establish publication authority. Failed trial inspection is
    distinguished from confirmed absence by ``trial_data_available``.

    CLI callers escalate failed or inaccessible publications after rendering the
    payload. Metric and result-context fields remain in the MCP read view.

    :param Experiment experiment: Parsed experiment config to inspect.
    :return dict: Status payload with generation identity and per-phase results.
    """
    status = read_status(experiment)
    integrity = status["publication_integrity"]
    ledger_path = ledger_ops.resolved_ledger_path(experiment)
    payload: dict[str, object] = {
        "kind": "experiment",
        "experiment": status["experiment"],
        "workdir": str(path_ops._experiment_dir(experiment)),
        "ledger_path": None if ledger_path is None else str(ledger_path),
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
        "phases": _cli_phase_payloads(
            experiment, status["phases"], status["published_generation_id"]
        ),
    }
    if experiment.mlflow is not None:
        from phasesweep.tracking.publication import delivery_status

        payload["tracking_delivery"] = delivery_status(experiment)
    return payload
