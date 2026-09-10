"""Suite execution and suite-generation publication."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.generation as generation_ops
import phasesweep.engine.locking as locking_ops
import phasesweep.engine.paths as path_ops
import phasesweep.engine.publication_validation as validation_ops
import phasesweep.engine.run as run_engine
from phasesweep._metadata import __version__
from phasesweep.config import Suite
from phasesweep.config.common import _validate_safe_name
from phasesweep.engine.errors import (
    PromotionError,
    PublicationAccessError,
    PublicationCommitError,
    PublicationIntegrityError,
)
from phasesweep.engine.selection import _apply_study_promotion, _winner_summary_item
from phasesweep.engine.state import (
    PUBLICATION_POINTER_SCHEMA_VERSION,
    SUITE_SUMMARY_SCHEMA_VERSION,
    Winner,
)
from phasesweep.runtime.files import ensure_workdir, file_sha256, require_posix_runtime
from phasesweep.runtime.process import (
    absorb_shutdown_signals,
    service_pending_shutdown,
    signal_handler_scope,
)
from phasesweep.runtime.time import utc_now_iso

log = logging.getLogger(__name__)


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
            results[study_spec.name] = run_engine.run_experiment(experiment, dry_run=True)
        return results

    require_posix_runtime()
    ensure_workdir(Path(suite.defaults.workdir).expanduser().resolve())
    path_ops._suite_dir(suite).mkdir(parents=True, exist_ok=True)
    # See _run_experiment_outcome: signal_handler_scope() is the outermost
    # context manager here too, so a suite installs shutdown handlers once for
    # the whole component sequence and each component's own scope (entered
    # inside _run_experiment_outcome) is a no-op nested inside this one.
    with (
        signal_handler_scope(),
        locking_ops._suite_lock(suite),
        artifact_io._file_log_handler(path_ops._suite_log_path(suite)),
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
                component = run_engine._run_experiment_outcome(experiment)
                study_winners = dict(component.winners)
                # The component summary's path and content hash anchor this
                # study's exposed results for read-side integrity validation
                # (review v0.5.17 gap hunt): the published generation
                # namespace is immutable, so the hash taken here stays valid
                # for the life of the artifact.
                component_summary_path = path_ops._generation_summary_path(
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
            immutable_summary = path_ops._suite_generation_summary_path(suite, generation_id)
            artifact_io._write_yaml_atomic(immutable_summary, summary)

            # Same publication transaction shape as :func:`phasesweep.engine.generation._publish_generation` (review
            # v0.5.15 / blocker 3): pre-commit validation, then the pointer
            # commit as the single final authoritative event; the record and
            # compatibility cache both move to best-effort, post-commit steps.
            # The same absorb window applies (review v0.5.16 / blocker 1): a
            # shutdown arriving mid-transaction must not reclassify the
            # committed suite publication.
            with absorb_shutdown_signals() as absorbed:
                try:
                    summary_bytes = _validate_suite_generation_publishable(suite, generation_id)
                    artifact_io._write_yaml_atomic(
                        path_ops._last_successful_suite_generation_path(suite),
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
                    generation_ops._persist_terminal_failure(
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

                generation_ops._log_on_failure(
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
                    generation_ops._copy_yaml_projection(
                        immutable_summary, path_ops._suite_summary_path(suite)
                    )

                generation_ops._log_on_failure(
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
            generation_ops._persist_failed_state_unless_recorded(
                path_ops._suite_generation_record_path(suite, generation_id),
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
    root = path_ops._suite_generations_dir(suite)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        generation_id = uuid4().hex
        try:
            path_ops._suite_generation_dir(suite, generation_id).mkdir()
        except FileExistsError:
            continue
        return generation_id
    raise RuntimeError("Could not mint an unused suite generation id after 10 attempts.")


def _validate_suite_generation_publishable(suite: Suite, generation_id: str) -> bytes:
    """Validate a suite generation's summary and component references before its commit.

    Suite mirror of :func:`phasesweep.engine.generation._validate_generation_publishable` (review v0.5.16
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
    summary, summary_bytes = generation_ops._validate_publishable_summary(
        summary_path=path_ops._suite_generation_summary_path(suite, generation_id),
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
        component_summary = validation_ops._read_pointer_target_summary(
            path_ops._generation_summary_path(component, component_generation),
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
            validation_ops._validate_generation_manifest(
                path_ops._generation_dir(component, component_generation),
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
        validation_ops._validate_suite_summary_integrity(generation_id, summary)
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

    Mirrors :func:`phasesweep.engine.generation._write_generation_state`'s split (review v0.5.15 / blocker
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
        for a terminal state -- see :func:`phasesweep.engine.generation._write_generation_state`
        for why the post-commit pointer refresh needs this.
    """
    payload = {
        "schema_version": 1,
        "suite": suite.suite,
        "suite_generation_id": generation_id,
        "suite_fingerprint": fingerprint_ops._suite_fingerprint(suite),
        "state": state,
        "started_at": started_at,
        "ended_at": ended_at,
        "error_class": error_class,
        "phasesweep_version": __version__,
    }
    if write_record and state in generation_ops._TERMINAL_GENERATION_STATES:
        generation_ops._write_generation_record_once(
            record_path=path_ops._suite_generation_record_path(suite, generation_id),
            generation_id=generation_id,
            state=state,
            payload=payload,
            label="suite generation",
        )
    if publish_current:
        artifact_io._write_yaml_atomic(path_ops._suite_generation_path(suite), payload)


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
    fingerprint = fingerprint_ops._suite_fingerprint(suite)
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
