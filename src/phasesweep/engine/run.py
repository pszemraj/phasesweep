"""Public engine entrypoints for experiment execution and status."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import yaml

import phasesweep.engine.artifact_roots as artifact_root_ops
import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.attempts as attempt_ops
import phasesweep.engine.evidence as evidence_ops
import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.generation as generation_ops
import phasesweep.engine.guards as guard_ops
import phasesweep.engine.locking as locking_ops
import phasesweep.engine.paths as path_ops
import phasesweep.engine.publication as publication_ops
import phasesweep.engine.publication_validation as validation_ops
import phasesweep.engine.resume as resume_ops
from phasesweep._metadata import __version__
from phasesweep.config import Config, Experiment, Suite
from phasesweep.config.common import _validate_safe_name
from phasesweep.config.models import _metric_semantics_payload
from phasesweep.config.search import sampler_capability_line
from phasesweep.engine.errors import (
    PromotionError,
    RunRequestError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.phase import _placeholder_winner, _run_phase
from phasesweep.engine.read import read_status
from phasesweep.engine.selection import _apply_promotion, _winner_summary_item
from phasesweep.engine.state import GENERATION_SUMMARY_SCHEMA_VERSION, Winner
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.files import ensure_artifact_dir, require_posix_runtime
from phasesweep.runtime.process import PhaseSweepShutdown, signal_handler_scope

log = logging.getLogger("phasesweep.engine.run")


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
    cleanup: attempt_ops._PreflightCleanupReport,
    *,
    primary_error: BaseException | None,
    failure_stage: str | None,
    winners: Mapping[str, Winner] | None = None,
    cleanup_confirmed: bool | None = None,
) -> TerminalReport:
    """Freeze accumulated cleanup evidence into one terminal report.

    :param str generation_id: Generation whose invocation ended.
    :param attempt_ops._PreflightCleanupReport cleanup: Mutable cleanup evidence to freeze.
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
        from phasesweep.engine.suite import run_suite

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
        publication = publication_ops._resolve_suite_publication_pointer(config)
        return {
            "kind": "suite",
            "suite": config.suite,
            "workdir": str(path_ops._suite_dir(config)),
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

    For non-dry-run invocations this acquires an
    :func:`phasesweep.engine.locking._experiment_lock` for
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
        ValueError: A caller-supplied generation id is not a safe filesystem name,
            or ``from_phase`` names no declared phase (rejected before state writes).
        FileNotFoundError: ``--from-phase`` requested but a prior phase has
            no persisted ``winner.yaml``.

    """
    if from_phase is not None and from_phase not in {phase.name for phase in experiment.phases}:
        raise ValueError(f"Unknown --from-phase value {from_phase!r}.")
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
    path_ops._experiment_dir(experiment).mkdir(parents=True, exist_ok=True)
    # signal_handler_scope() is the outermost context manager so shutdown-signal
    # ownership is scoped to this call tree and restored on every exit path,
    # not left installed on the host process forever (review v0.5.14 /
    # blocker 6). A run_suite caller has already entered its own scope, so
    # this one is a reentrant no-op that installs and restores nothing.
    with (
        signal_handler_scope(),
        locking_ops._experiment_lock(experiment),
        contextlib.ExitStack() as run_stack,
    ):
        # Resolve both ownership directions before the generation claim. The
        # returned objects are passed into preflight so storage is not reread
        # after this strict discovery-and-claim boundary.
        try:
            existing_studies = artifact_root_ops._load_and_check_artifact_roots(
                experiment, from_phase=from_phase
            )
        except StudyStorageUnavailableError as exc:
            # The ledger may contain attempts from an earlier orchestrator. A
            # failed ownership read cannot prove those processes are resolved,
            # even though this invocation has not claimed a generation or
            # launched anything of its own.
            raise ProcessCleanupUncertainError(
                "Artifact ownership could not be checked because required persistent "
                f"study state is unavailable: {exc} Cleanup state is therefore unknown. "
                "Restore the original complete storage ledger and access to it before "
                "retrying. For an MCP run, then run phasesweep mcp recover-run."
            ) from exc
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
            evidence_ops._validate_selection_evidence(experiment, existing_studies)
            resume_ops._reject_bound_descendant_topups(
                experiment,
                from_phase=from_phase,
                existing_studies=existing_studies,
            )
            resume_ops._reject_unsupported_sampler_topups(
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
                reconciliation = attempt_ops._PreflightCleanupReport()
                try:
                    guard_ops._preflight_existing_studies(
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
                    artifact_io._save_winner(
                        experiment,
                        phase.name,
                        winners[phase.name],
                        generation_id=generation_id,
                    )
                    # Safe to re-resolve the published pointer per phase here, unlike
                    # status reads: this runs under _experiment_lock, and this
                    # experiment's pointer only advances at this run's final publish.
                    prior_promotion = publication_ops._published_promotion_decision_path(
                        experiment, phase.name
                    )
                    if prior_promotion is not None and prior_promotion.is_file():
                        prior_promotion_payload = yaml.safe_load(prior_promotion.read_text())
                        generation_ops._copy_yaml_projection(
                            prior_promotion,
                            path_ops._generation_promotion_decision_path(
                                experiment, generation_id, phase.name
                            ),
                        )
                        if isinstance(prior_promotion_payload, dict):
                            promotion_decisions[phase.name] = prior_promotion_payload
                log.info("phase=%s SKIPPED (using preflight-validated winner)", phase.name)
            else:
                try:
                    winners[phase.name] = artifact_io._load_winner(experiment, phase, inherited)
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
                artifact_io._save_promotion_decision(
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
            artifact_io._save_winner(
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
    summary_path = path_ops._generation_summary_path(experiment, generation_id)
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
        "config_fingerprint": fingerprint_ops._experiment_semantic_fingerprint(experiment),
        "metric": _metric_semantics_payload(experiment.metric),
        "phase_plan": [
            {"name": phase.name, "comment": phase.comment} for phase in experiment.phases
        ],
        "promotion_decisions": list(promotion_decisions.values()),
        "phases": [_winner_summary_item(pname, w) for pname, w in winners.items()],
        "artifacts": validation_ops._generation_artifact_manifest(experiment, generation_id),
    }
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
      ``winner`` (path string or ``None``), ``published_study_unavailable``
      (a published phase's local trial identity could not be matched), and
      ``trial_data_available`` (true for known counts, including confirmed
      absence; false when inspection failed).

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
        "workdir": str(path_ops._experiment_dir(experiment)),
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
