"""Read-only views of experiment results for status and winner reporting.

This module is the single public surface for *reading* run state without
launching anything or reaching into engine-private path helpers. The MCP
layer is the primary consumer; the CLI consumes a narrow slice of it (see
:func:`config_status`) for its path-bearing status view, so winner and status
shapes have exactly one definition.

Reads here are permissive about partial run state and never raise on a missing
winner. They still fail closed when the artifact root belongs to another
storage ledger: combining one database's trial counts with another database's
publication is not a partial result. They do NOT re-verify phase fingerprints:
that check belongs to the resume path in ``engine.state._load_winner``, not to
a status read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import yaml

from phasesweep.config import Experiment
from phasesweep.config.common import SAFE_NAME_PATTERN, _validate_safe_name
from phasesweep.config.models import _metric_semantics_payload
from phasesweep.engine.guards import (
    _artifact_root_binding_applies,
    _experiment_semantic_fingerprint,
    _validate_artifact_root_binding,
)
from phasesweep.engine.optuna import _phase_trial_stats
from phasesweep.engine.state import (
    PublicationState,
    WinnerSource,
    WinnerSourceKind,
    _generation_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_id,
    _parse_winner_source,
    _published_summary_path_for,
    _published_winner_path,
    _published_winner_path_for,
    _resolve_publication_pointer,
)

ResultContext: TypeAlias = Literal["represented_generation", "current_config"]
"""Which config's semantics a result payload's labels were read under.

``"represented_generation"``: the labels come from the represented
generation's own recorded summary, so they describe the result as it was
produced. ``"current_config"``: the represented result recorded no semantics
of its own (nothing published yet, or a pre-manifest legacy layout), so the
currently loaded config describes it.
"""


@dataclass(frozen=True)
class PhaseWinnerView:
    """A phase winner reduced to the fields a caller may safely see.

    Notably absent: any filesystem path, the trial command, the environment,
    and the storage URL. ``effective_overrides`` is included for trusted engine
    and CLI readers that need the composed hyperparameter set; MCP filters it
    out before returning agent-visible payloads.
    """

    phase: str
    trial_number: int
    metric: float
    params: dict[str, Any]
    effective_overrides: dict[str, Any]
    gates_passed: bool | None  # None when the phase declared no gates
    incomplete: bool  # True when a wallclock timeout produced a partial winner
    generation_id: str | None = None
    attempt_id: str | None = None
    source: WinnerSource | None = None
    promotion: dict[str, Any] | None = None
    # The metric semantics the winner itself recorded (review v0.5.16 /
    # blocker 4): a persisted winner is historical evidence, so its value is
    # reported under the name and goal it was optimized for, never relabeled
    # by whatever metric the current config declares.
    metric_name: str | None = None
    metric_goal: str | None = None


def _phase_status_payloads(
    experiment: Experiment,
    *,
    include_winner_path: bool,
    trial_counts: Mapping[str, dict[str, int]],
    generation_trial_counts: Mapping[str, dict[str, int]],
    trial_data_available: Mapping[str, bool],
    running_attempts: Mapping[str, list[dict[str, Any]] | None],
    published_phases: set[str],
    winner_scope_generation_id: str | None = None,
    pinned: bool = False,
) -> list[dict[str, Any]]:
    """Build per-phase status payloads for CLI and MCP readers.

    ``winner_scope_generation_id`` must already be resolved by the caller
    exactly once (e.g. a single :func:`phasesweep.engine.state._resolve_publication_pointer`
    call, or a caller-pinned id) and is reused for every phase in this one
    call -- this function never re-resolves the last-success pointer itself,
    so one status object spanning several phases can never mix identities
    from two different pointer resolutions (review v0.5.15 / blocker 3).

    :param Experiment experiment: Parsed experiment whose phase study counts and winner files should be inspected.
    :param bool include_winner_path: If true, include the operator-facing winner path; otherwise return only a boolean winner flag.
    :param Mapping[str, dict[str, int]] trial_counts: Pre-read counts keyed by phase name.
    :param Mapping[str, dict[str, int]] generation_trial_counts: Counts for the represented generation, keyed by phase name.
    :param Mapping[str, bool] trial_data_available: Storage-read
        availability keyed by phase name. True includes confirmed absent studies
        with known zero counts; false means counts could not be established.
    :param Mapping[str, list[dict[str, Any]] | None] running_attempts:
        RUNNING trial identities keyed by phase name, from the same
        storage snapshot as ``trial_counts`` -- ``None`` for a phase whose
        trial data was unreadable. Included only in the path-free status view
        consumed by MCP, whose terminal snapshot must not reread studies to
        learn which rows the RUNNING count refers to (PR #5 review /
        reviewer 2, blocker 6).
    :param set[str] published_phases: Persistent phases with a current published winner.
    :param str | None winner_scope_generation_id: Already-resolved generation id
        whose winner files are represented. When ``pinned`` is ``False``
        (default), this is treated as an already-captured last-success id and
        legacy-fallback semantics apply when it is ``None``. When ``pinned``
        is ``True``, this is a caller-known generation id read directly with
        no legacy fallback.
    :param bool pinned: Whether ``winner_scope_generation_id`` names an exact,
        caller-pinned generation rather than an already-captured last-success id.
    :return list[dict[str, Any]]: One status payload per phase in declaration order.
    """
    phases: list[dict[str, Any]] = []
    for phase in experiment.phases:
        winner_path: Path | None
        if pinned:
            assert winner_scope_generation_id is not None
            winner_path = _generation_winner_path(
                experiment, winner_scope_generation_id, phase.name
            )
        else:
            winner_path = _published_winner_path_for(
                experiment, winner_scope_generation_id, phase.name
            )
        winner_present = winner_path is not None and winner_path.is_file()
        counts = trial_counts[phase.name]
        payload: dict[str, Any] = {
            "trials": counts,
            "running": counts.get("RUNNING", 0),
            "n_trials": phase.n_trials,
            "completed": counts.get("COMPLETE", 0),
            "generation_trials": generation_trial_counts[phase.name],
            "trial_data_available": trial_data_available[phase.name],
            "published_study_unavailable": phase.name in published_phases
            and not any(counts.values()),
        }
        if include_winner_path:
            payload.update(
                {"name": phase.name, "winner": str(winner_path) if winner_present else None}
            )
        else:
            payload.update(
                {
                    "phase": phase.name,
                    "winner_present": winner_present,
                    "running_attempts": running_attempts[phase.name],
                }
            )
        phases.append(payload)
    return phases


def _read_winner_path(path: Path | None, phase_name: str) -> PhaseWinnerView | None:
    """Parse one already-resolved winner path using the permissive read contract.

    :param Path | None path: Winner path selected by the caller, or ``None``.
    :param str phase_name: Phase name exposed by the selected winner.
    :return PhaseWinnerView | None: Parsed winner, or ``None`` when absent or malformed.
    """
    if path is None or not path.is_file():
        return None
    try:
        loaded = yaml.safe_load(path.read_text())
        if loaded is None:
            data: Mapping[str, Any] = {}
        elif isinstance(loaded, Mapping):
            data = loaded
        else:
            return None
        # winner.yaml stores metric as {<metric_name>: value, "goal": ...}. The
        # winner is historical evidence: pull the value by the name the file
        # itself recorded, NOT the currently configured metric name — reading
        # a published x/minimize result through a config renamed to y used to
        # silently drop the winner entirely (review v0.5.16 / blocker 4).
        metric_block = data.get("metric") or {}
        if not isinstance(metric_block, Mapping):
            return None
        stored_metric_names = [key for key in metric_block if key != "goal"]
        if len(stored_metric_names) != 1:
            return None
        stored_metric_name = str(stored_metric_names[0])
        stored_goal = metric_block.get("goal")
        gates = [g for g in (data.get("gates") or []) if isinstance(g, dict)]
        completion = data.get("completion") or {}
        if not isinstance(completion, Mapping):
            return None
        params = data.get("params") or {}
        if not isinstance(params, Mapping):
            return None
        effective_overrides = data.get("effective_overrides") or {}
        if not isinstance(effective_overrides, Mapping):
            return None
        source_data = data.get("winner_source")
        if not isinstance(source_data, Mapping):
            return None
        source_kind = source_data.get("kind")
        if source_kind not in ("phase_trial", "promotion_baseline", "suite_baseline"):
            return None
        source = _parse_winner_source(source_data, cast(WinnerSourceKind, source_kind))
        return PhaseWinnerView(
            phase=phase_name,
            trial_number=int(data["trial_number"]),
            metric=float(metric_block[stored_metric_name]),
            metric_name=stored_metric_name,
            metric_goal=str(stored_goal) if isinstance(stored_goal, str) else None,
            params=dict(params),
            effective_overrides=dict(effective_overrides),
            gates_passed=(all(bool(g.get("passed")) for g in gates) if gates else None),
            incomplete=bool(completion.get("incomplete", False)),
            generation_id=(
                str(data["generation_id"])
                if isinstance(data.get("generation_id"), str) and data["generation_id"]
                else None
            ),
            attempt_id=(
                str(data["attempt_id"])
                if isinstance(data.get("attempt_id"), str) and data["attempt_id"]
                else None
            ),
            source=source,
            promotion=(
                dict(data["promotion"]) if isinstance(data.get("promotion"), Mapping) else None
            ),
        )
    except (KeyError, ValueError, TypeError, OSError, yaml.YAMLError):
        # Partially-written or malformed file (or unlinked between the is_file
        # check and the read): report as no-winner-yet rather than raising.
        return None


def read_winner(
    experiment: Experiment,
    phase_name: str,
    *,
    generation_id: str | None = None,
) -> PhaseWinnerView | None:
    """Read a single phase's persisted winner, or ``None`` if not yet written.

    Args:
        experiment: Parsed experiment config; supplies the artifact roots.
            The metric value is extracted under the name the winner file
            itself recorded, so a historical winner is never reinterpreted
            through the currently configured metric (review v0.5.16 /
            blocker 4).
        phase_name: Phase whose ``winner.yaml`` to read.
        generation_id: Optional generation whose immutable winner should be read.

    Returns:
        A :class:`PhaseWinnerView`, or ``None`` when the phase has no usable
        winner on disk: never run, still running, selection failed, or the file
        is malformed. A malformed read is treated as "not yet written" -
            consistent with this module's permissive contract and with
            ``_phase_trial_stats`` swallowing transient backend errors. The
        strict, fingerprint-verifying read used for ``--from-phase`` resume
        lives in ``engine.state._load_winner`` and is intentionally not
        relaxed here.

    """
    _validate_artifact_root_binding(experiment, claim_fresh=False)
    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    path = (
        _published_winner_path(experiment, phase_name)
        if generation_id is None
        else _generation_winner_path(experiment, generation_id, phase_name)
    )
    return _read_winner_path(path, phase_name)


def read_winners(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    phase_names: Sequence[str] | None = None,
) -> list[PhaseWinnerView]:
    """Read every persisted phase winner, in declared (or supplied) phase order.

    Phases without a winner yet are skipped, so the list length tells the
    caller how far the chain has progressed.

    Args:
        experiment: Parsed experiment config whose phases are read in order.
        generation_id: Optional generation whose immutable winners should be read.
        phase_names: Optional explicit phase plan to enumerate instead of the
            currently configured phases, in the order given. A caller reading a
            *historical* generation passes that generation's own recorded plan
            (see :func:`_summary_phase_plan`), so a phase renamed in the config
            since publication neither hides the published winner nor reports
            the new name as missing (review v0.5.16 / blocker 4). Names are
            path components, so each is validated; the default keeps the
            declared-phase behavior every existing caller relies on.

    Returns:
        One :class:`PhaseWinnerView` per named phase that has a winner on disk.

    Raises:
        ValueError: If ``phase_names`` contains a name that is not a safe path
            component. Plans parsed off disk are filtered before they reach
            here, so this only fires on a programming error.

    """
    _validate_artifact_root_binding(experiment, claim_fresh=False)
    published_generation_id: str | None = None
    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    else:
        published_generation_id = _last_successful_generation_id(experiment)
    names = (
        [phase.name for phase in experiment.phases]
        if phase_names is None
        else [_validate_safe_name("phase", name) for name in phase_names]
    )
    paths = (
        (
            _generation_winner_path(experiment, generation_id, name)
            if generation_id is not None
            else _published_winner_path_for(experiment, published_generation_id, name)
        )
        for name in names
    )
    views = (_read_winner_path(path, name) for name, path in zip(names, paths, strict=True))
    return [view for view in views if view is not None]


def _read_summary_payload(summary_path: Path | None) -> Mapping[str, Any] | None:
    """Read a represented generation's summary for historical interpretation.

    :param Path | None summary_path: Already-resolved summary path, or ``None``.
    :return Mapping[str, Any] | None: Parsed summary mapping, or ``None`` when
        absent, unreadable, or not a mapping.
    """
    if summary_path is None:
        return None
    try:
        payload = yaml.safe_load(summary_path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _summary_phase_plan(summary_payload: Mapping[str, Any] | None) -> list[str] | None:
    """Return the phase names a represented generation published under.

    Only the ``name`` of each ``phase_plan`` entry is read. The summary's other
    phase records (``phases``) carry composed hyperparameters that agent-facing
    payloads must never see; the plan is the design-time list the engine froze
    beside the metric semantics for exactly this purpose (engine/run.py's
    generation summary).

    An entry that is not a mapping with a safe name invalidates the whole plan:
    these names become path components, and a partially parsed plan would
    under-report the publication just as badly as the current config does.

    :param Mapping[str, Any] | None summary_payload: Parsed generation summary, or ``None``.
    :return list[str] | None: Recorded phase names in execution order, or
        ``None`` when the summary is absent or records no usable plan (every
        pre-manifest legacy summary, whose caller must fall back to the current
        config).
    """
    if summary_payload is None:
        return None
    stored_plan = summary_payload.get("phase_plan")
    if not isinstance(stored_plan, list) or not stored_plan:
        return None
    names: list[str] = []
    for item in stored_plan:
        if not isinstance(item, Mapping):
            return None
        name = item.get("name")
        if not isinstance(name, str) or not SAFE_NAME_PATTERN.fullmatch(name):
            return None
        names.append(name)
    return names


def _recorded_objective_evidence(
    candidate: object,
    *,
    fallback: dict[str, str | bool],
) -> dict[str, str | bool]:
    """Return a complete recorded assurance payload or the current fallback.

    Result summaries are durable data and may predate or drift from the current
    assurance schema. A partial/extra key set cannot safely be combined with
    current guarantees, so treat the whole record as absent.

    :param object candidate: Summary ``objective_evidence`` value.
    :param dict[str, str | bool] fallback: Assurance derived from the current extractor.
    :return dict[str, str | bool]: Complete recorded flags, or ``fallback``.
    """
    if not isinstance(candidate, Mapping) or set(candidate) != set(fallback):
        return fallback
    if candidate.get("kind") not in {"json_envelope", "log_regex", "wandb"}:
        return fallback
    if any(type(candidate[key]) is not bool for key in candidate if key != "kind"):
        return fallback
    return dict(candidate)


def _current_pointer_generation_id(experiment: Experiment) -> str | None:
    """Read the mutable current-generation pointer's own id, or ``None``.

    Unlike the last-success pointer, the current pointer is not validated
    against any artifact: it is progress bookkeeping, always the most recent
    invocation's own claim, and may legitimately name a failed or
    in-progress generation.

    :param Experiment experiment: Experiment whose current pointer is read.
    :return str | None: The recorded ``generation_id``, or ``None`` when the
        pointer is missing, unreadable, malformed, or unsafely named.
    """
    try:
        generation = yaml.safe_load(_generation_path(experiment).read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(generation, Mapping):
        return None
    raw_generation_id = generation.get("generation_id")
    if isinstance(raw_generation_id, str) and SAFE_NAME_PATTERN.fullmatch(raw_generation_id):
        return raw_generation_id
    return None


def _legacy_publication_present(experiment: Experiment) -> bool:
    """Whether a pre-generation workdir already holds a published winner.

    Layouts written before generation metadata existed have no
    ``generation.yaml`` and so no last-success pointer to validate: their
    per-phase ``winner.yaml`` *is* the publication. The pointer-identity
    comparison that decides ``is_published`` everywhere else has nothing to
    compare there and used to report "never published" beside a real winner
    path, which an upgrading operator reads as data loss. Winner paths are
    resolved through the same helper the per-phase payload uses, so
    ``is_published`` and ``winner_present`` can never disagree.

    :param Experiment experiment: Experiment whose legacy winner files are probed.
    :return bool: ``True`` when the legacy fallback resolves at least one
        existing ``winner.yaml``; ``False`` for an untouched workdir and for
        every workdir that has a ``generation.yaml`` (there the pointer, not
        the compatibility projection, is authoritative, so an unpublished
        generation stays unpublished).
    """
    return any(
        (winner_path := _published_winner_path_for(experiment, None, phase.name)) is not None
        and winner_path.is_file()
        for phase in experiment.phases
    )


def read_status(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    comparison_experiment: Experiment | None = None,
    _include_winner_paths: bool = False,
) -> dict[str, Any]:
    """Per-phase trial counts and winner presence for one experiment.

    The payload is path-free **only while ``_include_winner_paths`` is
    ``False``** (the default): setting it swaps in the operator-facing phase
    payload, whose ``winner`` field is an absolute filesystem path. Every
    MCP-facing caller must therefore leave it ``False`` -- the redaction layer
    in ``phasesweep.mcp.redaction`` takes this function's default-mode output
    as already path-free and does not strip paths itself. The flag exists for
    :func:`phasesweep.engine.run.experiment_status` / ``config_status``, whose
    CLI and suite consumers are trusted with local paths.

    Trial counts come from ``_phase_trial_stats``, which reports empty counts
    for a study that does not exist yet, never creates one as a side effect,
    and swallows transient backend errors (e.g. a momentary SQLite lock while
    the runner writes) by reporting empty counts rather than raising.
    ``trial_data_available`` distinguishes a successful empty read from missing
    or unreadable storage so callers never treat ambiguous zeros as evidence.
    The path-free phase payload also carries ``running_attempts``: the
    identities of the RUNNING trials those same counts describe, from the same
    snapshot, or ``None`` when ``trial_data_available`` is ``False``. It exists
    so a caller that must reconcile RUNNING rows -- the MCP terminal snapshot
    -- never needs a second, intolerant storage read after this one (PR #5
    review / reviewer 2, blocker 6).

    This resolves the current pointer and the last-success pointer *exactly
    once each* and reuses those two captured ids for every downstream fact in
    this call -- no helper reached from here re-resolves either pointer, so
    one status object can never mix identities from two different pointer
    reads (review v0.5.15 / blocker 3). Four identities are always reported,
    and each is truthful independent of ``generation_id``:

    - ``current_generation_id``: always the actual mutable current pointer
      (the most recent invocation's own claim, whether failed, in-progress,
      or a much older publication); *never* forced to equal a pinned
      ``generation_id``.
    - ``published_generation_id``: always the actual validated last-success
      pointer; *never* forced to equal a pinned ``generation_id`` either. A
      pinned read of a generation whose own publication failed correctly
      reports the *older* generation here, not the pinned one.
    - ``represented_generation_id``: the generation whose winner/summary
      facts this payload actually shows -- ``generation_id`` itself when
      pinned, otherwise the captured ``published_generation_id``.
    - ``is_published``: ``True`` when ``represented_generation_id`` is not
      ``None`` and equals ``published_generation_id``. A pinned read of a
      failed-publication generation is ``is_published: False`` while still
      showing that generation's own (unpublished) winners. The one case with
      no generation identity to compare is a pre-generation legacy workdir
      (no ``generation.yaml`` at all): there the compatibility
      ``winner.yaml`` is the publication, so ``is_published`` follows
      :func:`_legacy_publication_present` and stays consistent with the
      ``winner_present`` reported beside it, while all three identity fields
      remain ``None`` because no generation id exists to report. A workdir
      that *does* have ``generation.yaml`` but no validated last-success
      pointer is unpublished as before.
    - ``publication_integrity``: ``"ok"`` / ``"absent"`` / ``"failed"`` /
      ``"permission_denied"`` --
      *why* ``published_generation_id`` is what it is (review v0.5.18 /
      finding F4). ``"absent"`` means nothing was ever published, a healthy
      state for a fresh tree; ``"failed"`` means a last-success pointer exists
      but its target no longer validates; ``"permission_denied"`` means this
      user cannot complete validation without implying corruption. Both are
      accompanied by a short, path-free ``publication_error``. These states
      used to be indistinguishable,
      which invited a re-run over corrupt evidence. A ``"failed"`` payload
      reports exactly the no-publication facts it always did -- nothing is
      fabricated from an unvalidated generation -- so ``published_generation_id``
      is ``None``, ``is_published`` is ``False``, and no winner reads as
      present.

    Two views coexist in this payload and must not be confused. The
    *config-progress* view -- the ``phases`` list, with its trial counts,
    targets, and per-phase winner presence -- describes the phases the current
    config declares, because that is what a further run would execute. The
    *result* view -- the metric descriptor, ``result_phase_plan``, and the
    publication identity fields -- describes the represented generation on its
    own terms. Under a config edited since publication the two legitimately
    disagree, and ``result_context`` / ``published_config_matches_current``
    say so explicitly rather than leaving a reader to conclude the publication
    is empty (review v0.5.16 / blocker 4).

    Winner/summary facts (``winner_present``, top-level ``summary_present``)
    scope to ``represented_generation_id``. ``generation_trials`` scopes to
    ``current_generation_id`` in default mode (live progress of whatever is
    currently running/most recent) but to the *pinned* id in pinned mode --
    a pinned caller wants that specific generation's own trial counts, not
    whatever else may be current by the time the read happens. ``trials``,
    ``running``, ``completed``, and ``trial_data_available`` are cumulative,
    all-time counts for the phase's study and are not generation-scoped.
    ``published_study_unavailable`` reports a current published phase with no
    readable trial history. When ``trial_data_available`` is true, the study
    is confirmed missing or empty; executing it is refused, while earlier
    phases may still load saved winners via ``from_phase``. False means the
    history could not be inspected; a run can report cleanup uncertainty and
    require operator recovery before further MCP launches. Publication
    integrity describes the artifacts separately.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Optional invocation identity to pin the
        read's *represented* generation: ``represented_generation_id`` and the
        ``generation_trials``/winner/summary scope all equal this id, while
        ``current_generation_id`` and ``published_generation_id`` remain the
        actual (possibly different) pointers. Used by callers (e.g. MCP
        per-run reads) that already know which generation they mean and want
        its own view of itself. When omitted (the default),
        ``represented_generation_id`` is the captured
        ``published_generation_id`` and ``generation_trials`` scopes to the
        captured ``current_generation_id``.
    :param Experiment | None comparison_experiment: Optional current config to
        use only for drift comparison and current-extractor evidence fallback.
        Artifact and trial reads still use ``experiment``. This lets a frozen
        run config locate its own tree while a run-scoped MCP read compares the
        represented result with the catalog config a future run would execute.
    :param bool _include_winner_paths: Internal CLI adapter flag selecting the
        path-bearing phase payload used by :func:`config_status`. Leave
        ``False`` for any agent-visible caller: ``True`` puts absolute winner
        paths in the returned mapping.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment. This includes
        the legacy-tree migration subclass.
    :raises ValueError: If ``generation_id`` is not a safe generation name.
    :return dict[str, Any]: A mapping with the experiment name, the four
        identity fields above, ``publication_integrity`` (plus
        ``publication_error`` only when validation failed or was denied), the metric
        descriptor, a per-phase list of trial counts plus winner presence, and
        whether the represented summary has been written -- path-free unless
        ``_include_winner_paths`` is set.
        The metric descriptor is the *represented generation's own* recorded
        metric whenever its summary declares one
        (``result_context: "represented_generation"``), falling back to the
        current config only when no summary semantics exist
        (``result_context: "current_config"``) — a published x/minimize
        result is never relabeled by a config edited to y/maximize (review
        v0.5.16 / blocker 4). A pre-manifest legacy summary that records a
        name and goal but no ``objective_evidence`` still reports
        ``"represented_generation"``, with the evidence assurance falling back
        to the current extractor's: that is the only historical semantics such
        a layout preserved. ``result_phase_plan`` is likewise the represented
        generation's own recorded phase plan, falling back to the current
        config's phase names when the summary records none — it is the plan
        the publication's winners must be enumerated under, and equals the
        ``phases`` list's names whenever the config has not been edited since.
        ``published_config_matches_current`` compares the summary's recorded
        config fingerprint against the current config's semantic fingerprint;
        ``None`` when the represented summary records no fingerprint.
    """
    _validate_artifact_root_binding(experiment, claim_fresh=False)
    current_generation_id = _current_pointer_generation_id(experiment)
    publication = _resolve_publication_pointer(experiment)
    published_generation_id = publication.generation_id if publication.state == "ok" else None

    if generation_id is None:
        represented_generation_id = published_generation_id
        trial_scope_generation_id = current_generation_id
        winner_scope_generation_id = published_generation_id
        pinned = False
    else:
        _validate_safe_name("generation", generation_id)
        represented_generation_id = generation_id
        trial_scope_generation_id = generation_id
        winner_scope_generation_id = generation_id
        pinned = True

    legacy_publication = False
    if represented_generation_id is None:
        # Only an unpinned read reaches here (a pinned read always represents
        # its own id), so this is either a workdir with nothing published --
        # ``False``, unchanged -- or a pre-generation legacy layout whose
        # compatibility ``winner.yaml`` is itself the publication and has no
        # pointer identity to compare against.
        legacy_publication = _legacy_publication_present(experiment)
        is_published = legacy_publication
    else:
        is_published = represented_generation_id == published_generation_id

    phase_stats = {phase.name: _phase_trial_stats(experiment, phase) for phase in experiment.phases}
    summary_path: Path | None
    if pinned:
        assert winner_scope_generation_id is not None
        summary_path = _generation_summary_path(experiment, winner_scope_generation_id)
    else:
        summary_path = _published_summary_path_for(experiment, winner_scope_generation_id)

    # The represented generation's winner/summary facts are historical
    # evidence, so the metric they are reported under must be the one that
    # generation actually optimized — never the metric the config supplied
    # today (review v0.5.16 / blocker 4). The generation's own summary is the
    # source of that interpretation; only when it records none (nothing
    # published yet, or a pre-manifest legacy summary without semantics) does
    # the current config describe the result.
    comparison = comparison_experiment or experiment
    metric_payload = _metric_semantics_payload(comparison.metric)
    current_objective_evidence = metric_payload["objective_evidence"]
    result_phase_plan = [phase.name for phase in experiment.phases]
    result_context: ResultContext = "current_config"
    published_config_matches_current: bool | None = None
    summary_payload: Mapping[str, Any] | None
    if (
        publication.state == "ok"
        and represented_generation_id == publication.generation_id
        and publication.summary is not None
    ):
        summary_payload = publication.summary
    else:
        summary_payload = _read_summary_payload(summary_path)
    if summary_payload is not None:
        stored_plan = _summary_phase_plan(summary_payload)
        if stored_plan is not None:
            result_phase_plan = stored_plan
        stored_metric = summary_payload.get("metric")
        if (
            isinstance(stored_metric, Mapping)
            and isinstance(stored_metric.get("name"), str)
            and stored_metric.get("goal") in ("minimize", "maximize")
        ):
            stored_evidence = stored_metric.get("objective_evidence")
            metric_payload = {
                "name": stored_metric["name"],
                "goal": stored_metric["goal"],
                "objective_evidence": (
                    _recorded_objective_evidence(
                        stored_evidence,
                        fallback=current_objective_evidence,
                    )
                ),
            }
            result_context = "represented_generation"
        stored_fingerprint = summary_payload.get("config_fingerprint")
        if isinstance(stored_fingerprint, str) and stored_fingerprint:
            published_config_matches_current = (
                stored_fingerprint == _experiment_semantic_fingerprint(comparison)
            )

    publication_state: PublicationState = publication.state
    if publication_state == "absent" and legacy_publication:
        publication_state = "ok"

    return {
        "experiment": experiment.experiment,
        "current_generation_id": current_generation_id,
        "published_generation_id": published_generation_id,
        "represented_generation_id": represented_generation_id,
        "is_published": is_published,
        "publication_integrity": publication_state,
        **(
            {"publication_error": publication.error}
            if publication.state in {"failed", "permission_denied"}
            else {}
        ),
        "result_context": result_context,
        "published_config_matches_current": published_config_matches_current,
        "result_phase_plan": result_phase_plan,
        "metric": metric_payload,
        "phases": _phase_status_payloads(
            experiment,
            published_phases={
                item["name"]
                for item in (publication.summary or {}).get("phases", ())
                if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            }
            if _artifact_root_binding_applies(experiment)
            else set(),
            include_winner_path=_include_winner_paths,
            trial_counts={name: stats.counts for name, stats in phase_stats.items()},
            generation_trial_counts={
                name: (
                    stats.generation_counts.get(trial_scope_generation_id, {})
                    if trial_scope_generation_id
                    else {}
                )
                for name, stats in phase_stats.items()
            },
            trial_data_available={name: stats.available for name, stats in phase_stats.items()},
            running_attempts={
                name: (
                    None
                    if stats.running_attempts is None
                    else [
                        {
                            "trial_number": attempt.trial_number,
                            "generation_id": attempt.generation_id,
                            "attempt_id": attempt.attempt_id,
                        }
                        for attempt in stats.running_attempts
                    ]
                )
                for name, stats in phase_stats.items()
            },
            winner_scope_generation_id=winner_scope_generation_id,
            pinned=pinned,
        ),
        "summary_present": summary_path is not None and summary_path.is_file(),
    }
