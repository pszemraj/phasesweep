"""Validated, path-free terminal result snapshots for MCP run handles."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phasesweep.config import Experiment
from phasesweep.engine import PhaseWinnerView, read_status, read_winners
from phasesweep.engine.read import ResultContext
from phasesweep.engine.state import (
    PublicationState,
    Winner,
    WinnerSource,
    WinnerSourceKind,
    _generation_record_path,
    _winner_source_or_default,
)
from phasesweep.evidence.models import _ObjectiveEvidenceFields, objective_evidence_assurance

log = logging.getLogger("phasesweep.mcp.snapshots")

NonNegativeInt = Annotated[int, Field(ge=0)]


class _SnapshotModel(BaseModel):
    """Strict base for persisted result snapshot records."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectiveEvidenceSnapshot(_SnapshotModel, _ObjectiveEvidenceFields):
    """Assurance properties of the configured objective extractor.

    See :func:`phasesweep.evidence.models.objective_evidence_assurance` for
    exactly what each flag means and which runtime checks back it.
    """


class MetricSnapshot(_SnapshotModel):
    """Optimization metric stored with a terminal run snapshot."""

    name: str
    goal: Literal["minimize", "maximize"]
    objective_evidence: ObjectiveEvidenceSnapshot


class RunningAttemptSnapshot(_SnapshotModel):
    """Concrete RUNNING row identity captured with a terminal snapshot."""

    trial_number: NonNegativeInt
    generation_id: str | None = None
    attempt_id: str | None = None


class PhaseStatusSnapshot(_SnapshotModel):
    """One phase's terminal trial counts and winner presence."""

    phase: str
    trials: dict[str, NonNegativeInt]
    running: NonNegativeInt
    n_trials: NonNegativeInt
    completed: NonNegativeInt
    generation_trials: dict[str, NonNegativeInt]
    winner_present: bool
    trial_data_available: bool
    running_attempts: list[RunningAttemptSnapshot] | None = None
    """RUNNING rows the frozen counts describe, or ``None`` when unknown.

    ``None`` means the capture read no trial data for this phase, i.e. it
    pairs with ``trial_data_available: false`` -- an empty list would assert
    there are no RUNNING rows. Snapshots frozen before this field existed also
    parse as ``None``, which is the truthful reading: they recorded no
    identities. A phase with readable trial data always carries a list, empty
    when nothing is RUNNING.
    """


class StatusSnapshot(_SnapshotModel):
    """Path-free terminal status view captured by the detached runner.

    ``current_generation_id`` and ``published_generation_id`` are always the
    actual mutable/last-success pointers and may differ from each other and
    from ``represented_generation_id`` (e.g. a failed rerun, or a pinned
    snapshot of a generation whose own publication failed).
    ``represented_generation_id`` is the generation whose winner/summary facts
    this snapshot shows, and ``is_published`` says whether that generation is
    the actual published one -- a failed-publication generation's snapshot
    correctly reports ``is_published: False`` while remaining fully readable.
    See :func:`phasesweep.engine.read.read_status`.
    """

    current_generation_id: str | None = None
    published_generation_id: str | None = None
    represented_generation_id: str | None = None
    is_published: bool = False
    publication_integrity: PublicationState = "absent"
    """Publication verdict at capture time.

    Defaults to ``"absent"`` so snapshots frozen before this field existed
    still parse, pairing with the ``is_published: False`` default they already
    carry: a legacy snapshot recorded no verdict, and inventing ``"ok"`` for it
    would be the fail-open answer.
    """

    metric: MetricSnapshot
    phases: list[PhaseStatusSnapshot]
    summary_present: bool

    result_context: ResultContext = "current_config"
    """Whether ``metric`` and ``result_phase_plan`` are the represented
    generation's own recorded semantics or the executing config's.

    Defaults to ``"current_config"`` so snapshots frozen before this field
    existed still parse. That is the conservative reading: such a snapshot
    recorded no proof that its labels came from the generation's own summary.
    """

    published_config_matches_current: bool | None = None
    """Whether the represented generation's recorded config fingerprint matched
    the config this run executed, or ``None`` when it recorded none.

    Frozen at capture time and never recomputed: this snapshot is read long
    after the catalog config may have moved on, and re-answering it against a
    later config would silently change what a frozen result claims. Snapshots
    frozen before this field existed parse as ``None`` -- unknown, which is
    exactly what they recorded.
    """

    result_phase_plan: list[str] | None = None
    """Phase plan the represented generation published under, or ``None``.

    ``None`` for snapshots frozen before this field existed; readers fall back
    to this snapshot's own ``phases`` names, which for every runner-captured
    snapshot is the same plan (the capture is pinned to the generation the
    executing config just produced).
    """


class WinnerSourceSnapshot(_SnapshotModel):
    """Concrete source trial for an exposed phase winner."""

    kind: WinnerSourceKind
    phase: str
    trial_number: int
    generation_id: str | None = None
    attempt_id: str | None = None
    study: str | None = None


class WinnerSnapshot(_SnapshotModel):
    """One sampled phase winner captured without effective overrides."""

    phase: str
    trial_number: NonNegativeInt
    metric: float
    params: dict[str, Any]
    gates_passed: bool | None
    incomplete: bool
    generation_id: str | None = None
    attempt_id: str | None = None
    source: WinnerSourceSnapshot
    promotion: dict[str, Any] | None = None


def _winner_source_snapshot(
    winner: PhaseWinnerView | Winner,
    *,
    phase: str,
) -> WinnerSourceSnapshot:
    """Return the concrete source model for a stored or engine-returned winner.

    Falls back to a ``"phase_trial"`` source built from the winner's own
    phase/trial/generation/attempt identity when ``winner.source`` is unset
    (winners persisted before source tracking was added have no ``source``).

    :param PhaseWinnerView | Winner winner: Winner whose source should be captured.
    :param str phase: Phase under which the winner is exposed.
    :return WinnerSourceSnapshot: Concrete, validated source snapshot.
    """
    source = _winner_source_or_default(winner, phase)
    return WinnerSourceSnapshot(
        kind=source.kind,
        phase=source.phase,
        trial_number=source.trial_number,
        generation_id=source.generation_id,
        attempt_id=source.attempt_id,
        study=source.study,
    )


def _winner_snapshot(
    phase: str,
    winner: PhaseWinnerView | Winner,
) -> WinnerSnapshot:
    """Freeze either engine winner representation into one validated snapshot.

    :param str phase: Phase under which the winner is exposed.
    :param PhaseWinnerView | Winner winner: Winner representation to serialize.
    :return WinnerSnapshot: Validated path-free winner snapshot.
    """
    if isinstance(winner, Winner):
        gates_passed = (
            all(bool(gate.get("passed")) for gate in winner.gates) if winner.gates else None
        )
        incomplete = bool(winner.completion.get("incomplete", False))
    else:
        gates_passed = winner.gates_passed
        incomplete = winner.incomplete
    return WinnerSnapshot(
        phase=phase,
        trial_number=winner.trial_number,
        metric=winner.metric,
        params=dict(winner.params),
        gates_passed=gates_passed,
        incomplete=incomplete,
        generation_id=winner.generation_id,
        attempt_id=winner.attempt_id,
        source=_winner_source_snapshot(winner, phase=phase),
        promotion=winner.promotion,
    )


class RunResultSnapshot(_SnapshotModel):
    """Terminal status and winners frozen for one MCP run id."""

    status: StatusSnapshot
    winners: list[WinnerSnapshot]

    def status_payload(self) -> dict[str, Any]:
        """Return the stored status in the engine reader's path-free shape.

        A snapshot frozen before ``result_phase_plan`` existed reports its own
        frozen phase names instead: a runner capture is always pinned to the
        generation its config just produced, so those names *are* that
        generation's plan. The key is always present so payload builders can
        read one shape for live and frozen reads alike.

        :return dict[str, Any]: Status mapping accepted by the MCP payload builder.
        """
        payload = self.status.model_dump(mode="json")
        for phase in payload["phases"]:
            phase.pop("running_attempts", None)
        if payload["result_phase_plan"] is None:
            payload["result_phase_plan"] = [phase.phase for phase in self.status.phases]
        return payload

    def winner_views(self) -> list[PhaseWinnerView]:
        """Return stored winners as the engine view consumed by MCP redaction.

        :return list[PhaseWinnerView]: Winner views with intentionally empty effective overrides.
        """
        return [
            PhaseWinnerView(
                phase=winner.phase,
                trial_number=winner.trial_number,
                metric=winner.metric,
                params=winner.params,
                effective_overrides={},
                gates_passed=winner.gates_passed,
                incomplete=winner.incomplete,
                generation_id=winner.generation_id,
                attempt_id=winner.attempt_id,
                source=WinnerSource(
                    kind=winner.source.kind,
                    phase=winner.source.phase,
                    trial_number=winner.source.trial_number,
                    generation_id=winner.source.generation_id,
                    attempt_id=winner.source.attempt_id,
                    study=winner.source.study,
                ),
                promotion=winner.promotion,
            )
            for winner in self.winners
        ]


def capture_pre_generation_result_snapshot(experiment: Experiment) -> dict[str, Any]:
    """Freeze known config shape without reading mutable study or winner state.

    Used when the engine crashed before ever claiming a generation, so no
    lock-protected read of shared Optuna study/winner state is safe. Builds a
    snapshot purely from the static experiment config: every phase's declared
    ``n_trials`` with all trial-state counts at zero, no winners, and no
    generation id.

    :param Experiment experiment: Exact config snapshot the detached runner
        attempted to execute.
    :return dict[str, Any]: JSON-serializable placeholder result snapshot.
    """
    zero_counts = {state: 0 for state in ("WAITING", "RUNNING", "COMPLETE", "PRUNED", "FAIL")}
    snapshot = RunResultSnapshot(
        status=StatusSnapshot(
            current_generation_id=None,
            published_generation_id=None,
            represented_generation_id=None,
            is_published=False,
            # Nothing was read from disk here, so no publication was resolved:
            # report the same verdict a never-published tree does rather than
            # implying this placeholder inspected one.
            publication_integrity="absent",
            # No summary was read either, so these labels are the config's own
            # and no drift verdict was reached.
            result_context="current_config",
            published_config_matches_current=None,
            result_phase_plan=[phase.name for phase in experiment.phases],
            metric=MetricSnapshot(
                name=experiment.metric.name,
                goal=experiment.metric.goal,
                objective_evidence=ObjectiveEvidenceSnapshot.model_validate(
                    objective_evidence_assurance(experiment.metric.extractor)
                ),
            ),
            phases=[
                PhaseStatusSnapshot(
                    phase=phase.name,
                    trials=dict(zero_counts),
                    running=0,
                    n_trials=phase.n_trials,
                    completed=0,
                    generation_trials={},
                    winner_present=False,
                    trial_data_available=False,
                    # Nothing was read, so nothing is known about RUNNING rows.
                    running_attempts=None,
                )
                for phase in experiment.phases
            ],
            summary_present=False,
        ),
        winners=[],
    )
    return snapshot.model_dump(mode="json")


def capture_result_snapshot(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    engine_winners: Mapping[str, Winner] | None = None,
) -> dict[str, Any]:
    """Capture one experiment's current path-free status and sampled winners.

    When ``generation_id`` is given (the detached runner always pins its own
    generation), the captured status's ``represented_generation_id`` equals
    it and the frozen winners are read from that exact generation, regardless
    of whether it ever became the actual last-success pointer -- so a
    failed-publication generation's terminal snapshot still reports its own
    (unpublished) partial results, correctly flagged ``is_published: False``.

    The engine and this capture must agree about what a valid terminal
    result is (review v0.5.16 / blocker 2): the per-generation lifecycle
    record is an optional post-commit diagnostic to the engine, so its
    absence is logged, never fatal here; storage reads only *enrich* the
    snapshot (a phase whose counts are unreadable is frozen with
    ``trial_data_available: false`` rather than failing the capture); and
    when the engine hands over its own authoritative ``engine_winners``, the
    frozen winners come from that exact in-memory outcome instead of a
    second read of the winner files.

    Exactly one storage read backs that agreement. Every trial fact frozen
    here -- counts, generation counts, and the RUNNING identities later used
    to reconcile cleanup evidence -- comes from the single tolerant
    :func:`phasesweep.engine.read.read_status` call below. This function used
    to reload each phase study afterwards to collect those identities, an
    intolerant read with no retry: one transient "database is locked" in that
    window failed the capture of an otherwise successful multi-hour run, and
    a terminal snapshot that was never captured is unrecoverable by design
    (see docs/mcp.md). Freezing a valid terminal result must not depend on a
    second storage round trip (PR #5 review / reviewer 2, blocker 6).

    :param Experiment experiment: Exact config snapshot the detached runner executed.
    :param str | None generation_id: Engine generation known to own the experiment lock.
    :param Mapping[str, Winner] | None engine_winners: The engine's own
        winner mapping from its terminal report, when the run succeeded.
    :return dict[str, Any]: JSON-serializable terminal result snapshot.
    """
    if generation_id is not None:
        # Best-effort diagnostic cross-check only: the record is written
        # post-commit and may legitimately be absent.
        try:
            lifecycle = yaml.safe_load(
                _generation_record_path(experiment, generation_id).read_text()
            )
        except (OSError, yaml.YAMLError):
            lifecycle = None
        if lifecycle is not None and (
            not isinstance(lifecycle, Mapping) or lifecycle.get("generation_id") != generation_id
        ):
            log.warning(
                "generation %s lifecycle record does not match its identity; "
                "capturing the terminal snapshot from the authoritative artifacts anyway",
                generation_id,
            )
    # One tolerant read supplies every storage fact this snapshot freezes,
    # including the RUNNING identities in each phase's ``running_attempts``
    # (``None`` where ``trial_data_available`` is false). This capture must
    # never reread a study afterwards (PR #5 review / reviewer 2, blocker 6).
    status = read_status(experiment, generation_id=generation_id)
    if engine_winners is not None:
        # The engine's terminal report is the authority on a successful
        # outcome (review v0.5.16 / blocker 2); freeze exactly what it
        # returned instead of re-reading the winner files.
        winner_snapshots = [
            _winner_snapshot(phase_name, winner) for phase_name, winner in engine_winners.items()
        ]
    else:
        # Winners must be scoped to the generation this snapshot *represents*,
        # not the (possibly different) true current pointer: a pinned capture
        # wants exactly its own generation's winners even when a newer
        # generation has since become current (review v0.5.15 / blocker 3).
        # They are enumerated under that generation's own recorded phase plan
        # for the same reason: an unpinned capture can represent an older
        # publication whose phase names this config no longer declares, and
        # reading it through today's names would freeze a snapshot that omits
        # winners which exist (review v0.5.16 / blocker 4).
        winner_snapshots = [
            _winner_snapshot(winner.phase, winner)
            for winner in read_winners(
                experiment,
                generation_id=status["represented_generation_id"],
                phase_names=status["result_phase_plan"],
            )
        ]
    snapshot = RunResultSnapshot(
        status=StatusSnapshot(
            current_generation_id=status["current_generation_id"],
            published_generation_id=status["published_generation_id"],
            represented_generation_id=status["represented_generation_id"],
            is_published=status["is_published"],
            publication_integrity=status["publication_integrity"],
            metric=status["metric"],
            phases=status["phases"],
            summary_present=status["summary_present"],
            result_context=status["result_context"],
            published_config_matches_current=status["published_config_matches_current"],
            result_phase_plan=status["result_phase_plan"],
        ),
        winners=winner_snapshots,
    )
    return snapshot.model_dump(mode="json")


def finalize_result_snapshot(
    snapshot: Mapping[str, object],
    *,
    confirmed_attempt_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Finalize a previously captured snapshot without rereading shared state.

    A phase whose capture recorded no RUNNING identities (``running_attempts``
    is ``None``, which pairs with ``trial_data_available: false``, and is also
    how a snapshot frozen before that field existed parses) is left exactly as
    captured: there is nothing to reconcile the cleanup report against, and its
    counts were never read either.

    :param Mapping[str, object] snapshot: Raw snapshot captured under the experiment lock.
    :param Collection[str] confirmed_attempt_ids: Exact RUNNING attempts reconciled to FAIL.
    :return dict[str, Any]: Validated terminal snapshot with truthful trial states.
    :raises RuntimeError: If the cleanup report names more recovered attempts
        than the snapshot counts as RUNNING, for a phase or for the represented
        generation.
    :raises ValidationError: If ``snapshot`` is not a valid ``RunResultSnapshot``.
    """
    parsed = RunResultSnapshot.model_validate(snapshot)
    confirmed = set(confirmed_attempt_ids)
    for phase in parsed.status.phases:
        if phase.running_attempts is None:
            # The capture read no trial data for this phase, so it recorded no
            # RUNNING identities to reconcile against. Its counts are equally
            # unread, and inventing a reconciliation over them would fabricate
            # states this snapshot never observed.
            continue
        recovered = [
            attempt
            for attempt in phase.running_attempts
            if attempt.attempt_id is not None and attempt.attempt_id in confirmed
        ]
        if not recovered:
            continue
        running = phase.trials.get("RUNNING", 0)
        if len(recovered) > running:
            raise RuntimeError("cleanup report identifies more attempts than the snapshot records")
        phase.trials["RUNNING"] = running - len(recovered)
        phase.trials["FAIL"] = phase.trials.get("FAIL", 0) + len(recovered)
        phase.running = phase.trials["RUNNING"]
        # generation_trials was captured scoped to represented_generation_id
        # (the pinned id for a run snapshot), not current_generation_id, which
        # is now always the actual mutable pointer and may be unrelated to --
        # or absent for -- this snapshot's own generation (review v0.5.15 /
        # blocker 3).
        generation_id = parsed.status.represented_generation_id
        generation_recovered = [
            attempt
            for attempt in recovered
            if generation_id is not None and attempt.generation_id == generation_id
        ]
        if generation_recovered:
            generation_running = phase.generation_trials.get("RUNNING", 0)
            if len(generation_recovered) > generation_running:
                raise RuntimeError(
                    "cleanup report identifies more generation attempts than the snapshot records"
                )
            phase.generation_trials["RUNNING"] = generation_running - len(generation_recovered)
            phase.generation_trials["FAIL"] = phase.generation_trials.get("FAIL", 0) + len(
                generation_recovered
            )
        recovered_ids = {attempt.attempt_id for attempt in recovered}
        phase.running_attempts = [
            attempt for attempt in phase.running_attempts if attempt.attempt_id not in recovered_ids
        ]
    return parsed.model_dump(mode="json")


def parse_result_snapshot(status: Mapping[str, object]) -> RunResultSnapshot | None:
    """Parse a terminal status's result snapshot, returning None when absent or malformed.

    :param Mapping[str, object] status: Validated runner terminal status payload.
    :return RunResultSnapshot | None: Strict snapshot model when usable.
    """
    raw = status.get("result_snapshot")
    if raw is None:
        return None
    try:
        return RunResultSnapshot.model_validate(raw)
    except ValidationError:
        return None
