"""Resume winner and persistent-study top-up preflight."""

from __future__ import annotations

import time
from collections.abc import Iterator

import optuna

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.study_policy as study_policy_ops
from phasesweep.config import Experiment, Phase
from phasesweep.engine.errors import StudyContextConflictError
from phasesweep.engine.state import PHASE_FINGERPRINT_ATTR, Winner


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
        winners[phase.name] = artifact_io._load_winner(experiment, phase, inherited)

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
        phase name, as returned by :func:`phasesweep.engine.guards._preflight_existing_studies`.
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
        partial_decision = study_policy_ops._load_accepted_partial_decision(study)
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
        phase name, as returned by :func:`phasesweep.engine.guards._preflight_existing_studies`.
    :raises SamplerContinuationUnsupportedError: A reached phase's study cannot
        safely continue with its configured stateful sampler; delegated to
        :func:`phasesweep.engine.study_policy._validate_sampler_continuation`.
    """
    for _index, phase in _phases_from(experiment, from_phase):
        study = existing_studies.get(phase.name)
        if study is not None:
            study_policy_ops._validate_sampler_continuation(study, phase)


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
    fingerprint_ops._verify_fingerprint(study, experiment, phase, inherited)
