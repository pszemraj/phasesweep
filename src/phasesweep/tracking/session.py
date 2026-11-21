"""Durable MLflow run bindings and engine-facing supervised operations."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from phasesweep.errors import OperatorAction, PhaseSweepError, ProcessCleanupUncertainError
from phasesweep.runtime.files import (
    UnsafePrivatePathError,
    open_directory_fd,
    private_atomic_write_text,
)
from phasesweep.runtime.json import (
    JsonObject,
    json_object,
    json_text,
    json_text_map,
    strict_json_loads,
)
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE, recorded_process_check
from phasesweep.runtime.shutdown import servicing_shutdown_signal
from phasesweep.tracking.mlflow import (
    OperationOutcomeUnknownError,
    OperationRejectedError,
    OperationUnavailableError,
    identity_tags,
    request_operation,
    verify_remote,
)
from phasesweep.tracking.records import validate_binding, validate_identity

if TYPE_CHECKING:
    from phasesweep.config.models import Experiment
    from phasesweep.engine.errors import TrackingSyncRefusedError
    from phasesweep.engine.state import Winner
    from phasesweep.tracking.mlflow import _SdkClient
    from phasesweep.tracking.records import (
        CreateAction,
        FindAction,
        PendingCreation,
        RunBinding,
        RunIdentity,
        TrackingAction,
    )

log = logging.getLogger(__name__)
BINDING_FILE = "mlflow-binding.json"
GENERATION_BINDING_FILE = "generation-binding.json"
INHERITS_FILE = "mlflow-inherits.json"
_PENDING_SUFFIX = ".pending.json"
PENDING_GENERATION_BINDING_FILE = Path(GENERATION_BINDING_FILE).with_suffix(_PENDING_SUFFIX).name
_PARENT_LOCK = threading.RLock()
# Failures that say nothing about a request's subject: MLflow gave no answer, or the worker's
# answer was lost. The next request would meet the same tracker, so a pass that sees one stops
# sending and leaves the rest pending.
NO_ANSWER: tuple[type[Exception], ...] = (
    TimeoutError,
    OperationUnavailableError,
    OperationOutcomeUnknownError,
)
# The step that sends what a run left pending, as every line a run logs names it. These lines
# also reach run.log and an MCP runner's log, where no command follows them, so they say in words
# what the command needs; the CLI prints the command as typed after a run that leaves work pending.
SYNC_TRACKING_STEP = "`phasesweep sync-tracking` with this run's config file and `--set` overrides"


@dataclass
class DeliveryTally:
    """What one MLflow delivery has met: a run's, recover-run's or sync-tracking's.

    A run keeps one for the pass after its first target check and its closeout, so its closing
    lines count both. Within a pass, once ``sending`` is False the later steps only save their
    records: a later request would meet the same silent tracker, or a shutdown, which refuses
    every worker launch. A closeout sends again whatever the earlier pass met.
    """

    sending: bool = True
    """Whether requests may still be made."""
    unanswered: str | None = None
    """Why MLflow gave no answer, once a request got none."""
    delivered: int = 0
    """Records of ended generations delivered."""
    settled: int = 0
    """Unconfirmed run creations settled: bound to the run a search found, or retired."""
    refused: set[Path] = field(default_factory=set)
    """Records and creations MLflow last answered with a refusal, such as for a deleted run.

    They stay pending. A record is counted once however many passes it refuses, and leaves the
    set once a later request delivers or settles it.
    """


def record_name(path: Path) -> str:
    """Name a delivery record or creation intent by what it is, for the lines that report it.

    :param Path path: A delivery record under ``tracking/<generation>/delivery``, a summary
        run's creation intent under ``tracking/<generation>``, or an attempt run's creation
        intent in its trial directory.
    :return str: For example ``the summary record of generation <id>`` or ``attempt record
        attempt-<id> of generation <id>``.
    """
    if path.name == PENDING_GENERATION_BINDING_FILE:
        return f"the summary-run creation record of generation {path.parent.name}"
    if path.name.endswith(_PENDING_SUFFIX):
        # pending_creation_records finds these only in trial_<n>__generation_<id>__attempt_<id>.
        # The markers fix the parse, as a generation id may hold "__" itself: an MCP run's is
        # named after its catalog id, which may. An attempt id is hex.
        generation, _, attempt = path.parent.name.partition("__generation_")[2].rpartition(
            "__attempt_"
        )
        return f"the run-creation record of attempt {attempt} of generation {generation}"
    generation = path.parent.parent.name
    if path.stem == "generation":
        return f"the summary record of generation {generation}"
    return f"{path.stem.split('-', 1)[0]} record {path.stem} of generation {generation}"


def tracking_root(experiment: Experiment) -> Path:
    """Return mutable tracking state outside scientific generation manifests.

    :param Experiment experiment: Local experiment namespace.
    :return Path: Tracking state root.
    """
    from phasesweep.engine.paths import _experiment_dir

    return _experiment_dir(experiment) / "tracking"


def write_record(path: Path, record: Mapping[str, object]) -> None:
    """Atomically persist private tracking state.

    :param Path path: Record destination.
    :param Mapping[str, object] record: JSON-compatible payload.
    """
    private_atomic_write_text(
        path,
        json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n",
        require_private_dir=False,
    )


def read_record(path: Path) -> JsonObject:
    """Read a current-format tracking record.

    :param Path path: Record to read.
    :raises ValueError: The record does not use the current format.
    :return JsonObject: Decoded current-format record.
    """
    data: object = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported MLflow tracking record format.")
    return json_object(data)


def read_binding(directory: Path, name: str = BINDING_FILE) -> RunBinding:
    """Load a local run binding.

    :param Path directory: Directory holding the binding.
    :param str name: Binding file name; an attempt's by default.
    :return RunBinding: Validated binding.
    """
    return validate_binding(read_record(directory / name))


def bind_mlflow_run(run_id: str) -> None:
    """Bind a trainer-created MLflow run to the current PhaseSweep attempt.

    Call once inside the trainer's active run, before logging evaluations.

    :param str run_id: Real run ID returned by the trainer's MLflow logger.
    :raises ValueError: The run conflicts with the current attempt or experiment.
    """
    from mlflow import MlflowClient

    directory = Path(os.environ["PHASESWEEP_TRIAL_DIR"])
    expected = read_record(directory / "mlflow-expected.json")
    if (
        expected["run_mode"] != "trainer"
        or expected["attempt_id"] != os.environ["PHASESWEEP_ATTEMPT_ID"]
        or expected["generation_id"] != os.environ["PHASESWEEP_GENERATION_ID"]
        or expected["overrides_sha256"] != os.environ["PHASESWEEP_OVERRIDES_SHA256"]
    ):
        raise ValueError("MLflow binding helper requires this trainer-owned attempt.")
    binding = validate_binding({**expected, "run_id": run_id})
    if (directory / BINDING_FILE).exists() and read_binding(directory) != binding:
        raise ValueError("This attempt is already bound to another MLflow run.")
    client: _SdkClient = MlflowClient(tracking_uri=binding["tracking_uri"])
    run = client.get_run(run_id)
    if run.info.experiment_id != expected["experiment_id"] or run.info.lifecycle_stage != "active":
        raise ValueError("Trainer MLflow run belongs to a different experiment.")
    if run.info.status != "RUNNING":
        raise ValueError("Bind the trainer's MLflow run while it is still RUNNING.")
    tags = identity_tags(binding)
    if any(key in run.data.tags and run.data.tags[key] != value for key, value in tags.items()):
        raise ValueError("Trainer MLflow run already belongs to another attempt.")
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    verify_remote(client.get_run(run_id), binding)
    write_record(directory / BINDING_FILE, binding)


def read_inherits(directory: Path) -> dict[str, str]:
    """Load the native run of each parent winner an attempt inherited.

    :param Path directory: Attempt directory.
    :raises ValueError: The record is not a mapping of phase names to run IDs.
    :return dict[str, str]: Run ID by parent phase name.
    """
    inherits = json_text_map(read_record(directory / INHERITS_FILE).get("inherits"))
    if inherits is None:
        raise ValueError("MLflow inherited-run record is malformed.")
    return inherits


def operation_deadline(experiment: Experiment, deadline: float | None = None) -> float:
    """Cap tracking work by its timeout and any remaining execution budget.

    :param Experiment experiment: Experiment with MLflow settings.
    :param float | None deadline: Existing absolute execution deadline.
    :return float: Absolute bounded operation deadline.
    """
    assert experiment.mlflow is not None
    stop = time.monotonic() + experiment.mlflow.timeout_seconds
    return min(stop, deadline) if deadline is not None else stop


@contextmanager
def tracking_operation(experiment: Experiment, generation_id: str) -> Iterator[tuple[str, Path]]:
    """Register a tracking worker and retain ownership until its cleanup is confirmed.

    Unreadable lifecycle records retain the registry entry. Entries are retired
    when the worker never launched or cleanup is confirmed.

    :param Experiment experiment: Registry owner.
    :param str generation_id: Producing generation.
    :raises ProcessCleanupUncertainError: The lifecycle record is unreadable and
        the body raised nothing; an exception the body raised propagates instead.
    :return Iterator[tuple[str, Path]]: Operation identity and process directory.
    """
    from phasesweep.engine.attempts import _register_tracking_operation, _retire_active_attempt
    from phasesweep.engine.paths import _attempts_dir
    from phasesweep.runtime.process import read_attempt_lifecycle, write_attempt_lifecycle

    operation_id = uuid4().hex
    directory = tracking_root(experiment) / generation_id / "operations" / operation_id
    directory.mkdir(parents=True, exist_ok=True)
    write_attempt_lifecycle(directory, attempt_id=operation_id, state="allocated")
    _register_tracking_operation(
        experiment, operation_id=operation_id, generation_id=generation_id, directory=directory
    )
    body_raised = False
    try:
        yield operation_id, directory
    except BaseException:
        body_raised = True
        raise
    finally:
        try:
            lifecycle = read_attempt_lifecycle(directory, expected_attempt_id=operation_id)
        except ValueError as exc:
            # The body's own failure may already say the worker is unresolved, so it wins.
            if not body_raised:
                entry_path = _attempts_dir(experiment) / f"{operation_id}.json"
                raise ProcessCleanupUncertainError(
                    f"Tracking operation {operation_id} has an unreadable lifecycle record in "
                    f"{directory} ({exc}), so its worker may still be running. Its attempt "
                    f"registry entry {entry_path} is kept. "
                    + recorded_process_check(str(directory / PROCESS_IDENTITY_FILE)),
                    action=OperatorAction.RESTORE_TREE,
                ) from exc
            log.warning(
                "Tracking operation %s has an unreadable lifecycle record in %s (%s); "
                "its attempt registry entry is kept.",
                operation_id,
                directory,
                exc,
            )
        else:
            if lifecycle is not None and (
                lifecycle.state == "allocated"
                or (lifecycle.state == "exited" and lifecycle.cleanup_confirmed)
            ):
                _retire_active_attempt(experiment, operation_id)


def call_tracking(
    experiment: Experiment,
    generation_id: str,
    request: TrackingAction,
    environment: Mapping[str, str],
    deadline: float | None = None,
) -> JsonObject:
    """Perform a registered tracking-only operation.

    :param Experiment experiment: Experiment with configured tracking.
    :param str generation_id: Operation's generation.
    :param TrackingAction request: Trusted SDK action.
    :param Mapping[str, str] environment: Composed credentials/transport.
    :param float | None deadline: Optional absolute run deadline.
    :return JsonObject: Successful worker response.
    """
    assert experiment.mlflow is not None
    with tracking_operation(experiment, generation_id) as (operation_id, directory):
        return request_operation(
            {"settings": experiment.mlflow.target(), **request},
            operation_id=operation_id,
            directory=directory,
            environment=environment,
            deadline=operation_deadline(experiment, deadline),
        )


def _create_bound(
    experiment: Experiment,
    identity: RunIdentity,
    path: Path,
    environment: Mapping[str, str],
    *,
    run_name: str,
    deadline: float | None,
) -> RunBinding:
    """Bind or create an identity's run, persisting intent before sending creation.

    A confirmed rejection removes pending intent. Unknown outcomes, timeouts,
    and other failures retain it; subsequent calls search rather than create.
    Only recovery can retire an intent after an empty search.

    :param Experiment experiment: Configured experiment.
    :param RunIdentity identity: Exact creation identity.
    :param Path path: Durable binding destination.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str run_name: Display name for a newly created run.
    :param float | None deadline: Existing execution deadline.
    :raises OperationRejectedError: The worker reported that the request failed.
    :raises RuntimeError: The creation outcome is unresolved.
    :return RunBinding: Bound native run identity.
    """
    if path.exists():
        return read_binding(path.parent, path.name)
    pending = path.with_suffix(_PENDING_SUFFIX)

    def reconcile(cause: BaseException | None = None) -> RunBinding:
        """Look for the run a sent creation request may have created; never create one.

        :param BaseException | None cause: The failure that left the request unconfirmed, if any.
        :raises RuntimeError: No run carries the identity before the deadline.
        :return RunBinding: The run that request created.
        """
        if deadline is None or time.monotonic() < deadline:
            found = _find_bound(
                experiment,
                identity,
                environment,
                timeout_at=operation_deadline(experiment, deadline),
                deadline=deadline,
            )
            if found is not None:
                return found
        raise RuntimeError(
            f"MLflow run creation for attempt {identity['attempt_id']} of generation "
            f"{identity['generation_id']} is unresolved: an earlier request may still create "
            f"the run, so it is not sent again. Run {SYNC_TRACKING_STEP} to reconcile the intent."
        ) from cause

    if pending.exists():
        binding = reconcile()
    else:
        record: PendingCreation = {"schema_version": 1, "identity": identity, "run_name": run_name}
        write_record(pending, record)
        request: CreateAction = {"action": "create", "binding": identity, "run_name": run_name}
        try:
            created = call_tracking(
                experiment, identity["generation_id"], request, environment, deadline
            )
        except OperationRejectedError:
            # A refused or unsent request created nothing, so a later call may send it afresh.
            _retire_creation(pending)
            raise
        except (OperationOutcomeUnknownError, TimeoutError) as exc:
            binding = reconcile(exc)
        else:
            binding = validate_binding(created)
    write_record(path, binding)
    _retire_creation(pending)
    return binding


def _retire_creation(path: Path) -> None:
    """Remove an intent without following a symlinked parent.

    :param Path path: Creation intent to retire.
    """
    with suppress(FileNotFoundError):
        directory_fd = open_directory_fd(path.parent, create=False, private_final=False)
        try:
            os.unlink(path.name, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)


def _find_bound(
    experiment: Experiment,
    identity: RunIdentity,
    environment: Mapping[str, str],
    *,
    timeout_at: float,
    deadline: float | None,
    worker_generation: str | None = None,
) -> RunBinding | None:
    """Search an identity's run read-only, polling until ``timeout_at``.

    :param Experiment experiment: Configured experiment.
    :param RunIdentity identity: Exact creation identity.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param float timeout_at: Absolute monotonic time the search stops polling at.
    :param float | None deadline: Existing execution deadline.
    :param str | None worker_generation: Invocation registering the search, defaulting to the
        identity's own generation.
    :raises ValueError: The lookup response was malformed.
    :return RunBinding | None: The verified binding, or ``None`` when no run carries the identity.
    """
    request: FindAction = {"action": "find", "binding": identity, "timeout_at": timeout_at}
    found = call_tracking(
        experiment, worker_generation or identity["generation_id"], request, environment, deadline
    )
    if found.get("found") is False and "binding" not in found:
        return None
    if found.get("found") is True:
        return validate_binding(found.get("binding"))
    raise ValueError("MLflow lookup response did not say whether it found the run.")


def pending_creation_records(
    experiment: Experiment, generation_id: str | None = None
) -> list[Path]:
    """Find the creation intents that have yet to be reconciled.

    :param Experiment experiment: Local namespace.
    :param str | None generation_id: Generation filter, defaulting to every generation.
    :return list[Path]: Pending summary and attempt creation records, in that order.
    """
    from phasesweep.engine.paths import _experiment_dir

    scope = generation_id or "*"
    attempt_pending = Path(BINDING_FILE).with_suffix(_PENDING_SUFFIX).name
    return sorted(
        tracking_root(experiment).glob(f"{scope}/{PENDING_GENERATION_BINDING_FILE}")
    ) + sorted(
        _experiment_dir(experiment).glob(
            f"*/trial_*__generation_{scope}__attempt_*/{attempt_pending}"
        )
    )


def creation_window_end(experiment: Experiment, pending: Path) -> float:
    """Say when the lost request of a pending creation can no longer create its run.

    Until then a search that finds no run leaves the intent pending; one started after it
    retires the intent.

    :param Experiment experiment: MLflow-backed experiment, whose ``timeout_seconds`` bounds
        the request.
    :param Path pending: Pending creation intent, written as the request was sent.
    :return float: The window's end, as a ``time.time()`` value.
    """
    settings = experiment.mlflow
    assert settings is not None
    return pending.stat().st_mtime + settings.timeout_seconds


def resolve_pending_creations(
    experiment: Experiment,
    environment: Mapping[str, str],
    generation_id: str | None,
    *,
    skip: str | None = None,
    worker_generation: str | None = None,
    errors: list[TrackingSyncRefusedError] | None = None,
    tally: DeliveryTally | None = None,
    deadline: float | None = None,
) -> None:
    """Reconcile pending creations with one read-only search per record.

    The caller holds the experiment lock and has confirmed cleanup of registered workers for
    the generations it reconciles. Persist each found binding before retiring its intent.
    Retire an unmatched intent only after a successful lookup started outside its request window.
    A search MLflow gives no answer to, or one that fails during a shutdown, ends the pass and
    stops ``tally`` sending, leaving the later intents and every upload for the next pass. On the
    main thread a shutdown raises out of the search in flight instead.

    :param Experiment experiment: Configured experiment with a verified tracking target.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None generation_id: Generation to resolve, or ``None`` for every generation.
    :param str | None skip: Generation left out of the scope, a live invocation's own.
    :param str | None worker_generation: Invocation registering the searches, defaulting to
        each record's own generation.
    :param list[TrackingSyncRefusedError] | None errors: Collect local record failures and
        continue, or raise the first one if omitted.
    :param DeliveryTally | None tally: The delivery this pass belongs to; a fresh one when
        omitted.
    :param float | None deadline: Absolute run deadline every search is cut at, defaults to
        none.
    :raises ProcessCleanupUncertainError: A search worker's cleanup is uncertain.
    :raises TrackingSyncRefusedError: A local creation record could not be read or updated.
    """
    from phasesweep.engine.errors import TrackingSyncRefusedError

    skipped = set() if skip is None else set(pending_creation_records(experiment, skip))
    records = [
        pending
        for pending in pending_creation_records(experiment, generation_id)
        if pending not in skipped
    ]
    tally = DeliveryTally() if tally is None else tally
    for pending in records:
        if not tally.sending:
            return
        try:
            _reconcile_creation(
                experiment, pending, environment, worker_generation, tally, deadline
            )
        except (OSError, UnsafePrivatePathError, ValueError) as exc:
            failure = TrackingSyncRefusedError(
                f"Could not reconcile local MLflow creation record {pending}: {exc}",
                action=OperatorAction.RESTORE_TREE,
            )
            if errors is None:
                raise failure from exc
            failure.__cause__ = exc
            errors.append(failure)


def _reconcile_creation(
    experiment: Experiment,
    pending: Path,
    environment: Mapping[str, str],
    worker_generation: str | None,
    tally: DeliveryTally,
    deadline: float | None,
) -> None:
    """Bind a found run or retire its expired intent after one conclusive lookup.

    :param Experiment experiment: Configured experiment with a verified tracking target.
    :param Path pending: Unconfirmed creation intent.
    :param Mapping[str, str] environment: Composed credentials and transport.
    :param str | None worker_generation: Invocation registering the lookup, defaulting to the intent's generation.
    :param DeliveryTally tally: The delivery the search belongs to, which counts the intent
        settled once it is retired, and stops sending when MLflow gives no answer or a shutdown
        is under way.
    :param float | None deadline: Absolute run deadline the search is cut at, if any.
    """
    path = pending.with_suffix("").with_suffix(".json")
    if path.exists():
        _settle_creation(pending, tally)
        return
    identity = validate_identity(read_record(pending).get("identity"))
    # A search started inside the window may see absence even if it returns after the window.
    expired = time.time() >= creation_window_end(experiment, pending)
    try:
        binding = _find_bound(
            experiment,
            identity,
            environment,
            timeout_at=time.monotonic(),
            deadline=deadline,
            worker_generation=worker_generation,
        )
    except PhaseSweepError:
        raise
    except (TimeoutError, ValueError, RuntimeError) as exc:  # every NO_ANSWER error included
        if servicing_shutdown_signal() is not None:
            # A shutdown refuses every worker launch and kills running workers, so this failure
            # says nothing about MLflow; nothing more is sent, and nothing blames it.
            tally.sending = False
            return
        if not isinstance(exc, NO_ANSWER):
            if isinstance(exc, OperationRejectedError):
                tally.refused.add(pending)
            log.warning("MLflow: %s is still unresolved (%s).", record_name(pending), exc)
            return
        # The pass sends nothing after this, so this is its only outage warning.
        log.warning(
            "MLflow: %s is still unresolved (%s). MLflow did not answer, so later searches and "
            "uploads wait too.",
            record_name(pending),
            exc,
        )
        tally.sending = False
        tally.unanswered = unready_target_text(experiment, exc)
        return
    if binding is not None:
        write_record(path, binding)
    elif not expired:
        return
    _settle_creation(pending, tally)


def _settle_creation(pending: Path, tally: DeliveryTally) -> None:
    """Retire a creation intent a search settled, and count it for the delivery's closing line.

    :param Path pending: Creation intent whose run is bound, or which no run carries.
    :param DeliveryTally tally: The delivery that settled it.
    """
    _retire_creation(pending)
    tally.settled += 1
    tally.refused.discard(pending)


def _resolve_source(
    experiment: Experiment,
    *,
    generation_id: str,
    environment: Mapping[str, str],
    deadline: float | None,
) -> str:
    """Resolve the source experiment and hold it to the cohort's first resolution.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: Current invocation identity.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Existing phase/run deadline.
    :raises ValueError: The experiment resolves differently within this cohort.
    :return str: The resolved experiment ID.
    """
    assert experiment.mlflow is not None
    with _PARENT_LOCK:
        resolved = call_tracking(
            experiment, generation_id, {"action": "resolve"}, environment, deadline
        )
        experiment_id = json_text(resolved, "experiment_id")
        source_path = tracking_root(experiment) / "source.json"
        source = {
            "schema_version": 1,
            "source": experiment.mlflow.evidence_identity(),
            "experiment_id": experiment_id,
        }
        if source_path.exists() and read_record(source_path) != source:
            raise ValueError("MLflow experiment resolution changed within this scientific cohort.")
        write_record(source_path, source)
        return experiment_id


def prepare_generation(
    experiment: Experiment,
    *,
    generation_id: str,
    environment: Mapping[str, str],
    deadline: float | None,
) -> RunBinding:
    """Resolve the source experiment and bind this generation's summary run once.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: Current invocation identity.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Existing phase/run deadline.
    :return RunBinding: The generation summary run's binding.
    """
    assert experiment.mlflow is not None
    with _PARENT_LOCK:
        parent_path = tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE
        if parent_path.exists():
            return read_binding(parent_path.parent, parent_path.name)
        experiment_id = _resolve_source(
            experiment, generation_id=generation_id, environment=environment, deadline=deadline
        )
        from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint

        parent_identity: RunIdentity = {
            "schema_version": 1,
            "tracking_uri": experiment.mlflow.tracking_uri,
            "experiment": experiment.experiment,
            "generation_id": generation_id,
            "overrides_sha256": _experiment_semantic_fingerprint(experiment),
            "run_mode": "managed",
            "experiment_id": experiment_id,
            "phase": "generation",
            "attempt_id": generation_id,
        }
        return _create_bound(
            experiment,
            parent_identity,
            parent_path,
            environment,
            run_name=f"{experiment.experiment}-generation-{generation_id}",
            deadline=deadline,
        )


def require_tracking_target(
    experiment: Experiment,
    *,
    generation_id: str,
    environment: Mapping[str, str],
    deadline: float | None,
    resolve_only: bool = False,
) -> bool:
    """Resolve the MLflow target before a phase allocates trials or a run sends to it.

    A generation whose summary run is bound only resolves the target, as does a check for a
    caller that only sends recovered outcomes, so a run that launches nothing binds no summary
    run. Its worker is registered under ``generation_id`` either way.

    :param Experiment experiment: MLflow-backed experiment.
    :param str generation_id: Generation of the run checking the target, whose summary run is
        bound first unless it already is or ``resolve_only`` is set.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Phase/run deadline; reaching it stays a timeout.
    :param bool resolve_only: Only resolve the target, for a caller that launches nothing and
        sends recovered outcomes, defaults to False.
    :raises OSError: A recovery-only check could not access its local tracking records.
    :raises TrackingTargetUnavailableError: The target cannot be resolved or reached; its step
        is a retry when MLflow gave no answer, the creation it left unconfirmed included, and
        fixing the configuration when MLflow answered with a refusal.
    :return bool: Whether this check bound the generation's summary run, which only a run's
        first successful check does.
    """
    from phasesweep.engine.errors import TrackingTargetUnavailableError

    settings = experiment.mlflow
    assert settings is not None
    try:
        if (
            resolve_only
            or (tracking_root(experiment) / generation_id / GENERATION_BINDING_FILE).exists()
        ):
            # An earlier phase bound the summary run, or the caller binds none; the target must
            # still resolve.
            _resolve_source(
                experiment, generation_id=generation_id, environment=environment, deadline=deadline
            )
            return False
        prepare_generation(
            experiment, generation_id=generation_id, environment=environment, deadline=deadline
        )
        return True
    except PhaseSweepError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        if resolve_only and isinstance(exc, OSError) and not isinstance(exc, TimeoutError):
            raise
        if isinstance(exc, TimeoutError) and deadline is not None and time.monotonic() >= deadline:
            raise
        launch = "" if resolve_only else " No trial was started."
        # A server that gave no answer may give one to the same command later, and so may one
        # whose creation answer was lost; a refusal it answered, such as a missing experiment,
        # needs the configuration or MLflow fixed first.
        unanswered = isinstance(exc, NO_ANSWER) or isinstance(exc.__cause__, NO_ANSWER)
        raise TrackingTargetUnavailableError(
            unready_target_text(experiment, exc) + launch,
            action=OperatorAction.RETRY if unanswered else None,
        ) from exc


def unready_target_text(experiment: Experiment, exc: BaseException) -> str:
    """Say that the configured MLflow target gave no usable answer, and why.

    A run's target check and a delivery MLflow stopped answering word the target alike. The
    tracking URI carries no credentials, which config load refuses, so it is printed as is.

    :param Experiment experiment: MLflow-backed experiment.
    :param BaseException exc: The failed request's error; a timeout is worded as the
        ``timeout_seconds`` it outlasted.
    :return str: One sentence naming the experiment selector, the URI and the cause.
    """
    settings = experiment.mlflow
    assert settings is not None
    cause = (
        "it did not answer within timeout_seconds; check that the server is reachable"
        if isinstance(exc, TimeoutError)
        else str(exc).rstrip(".")
    )
    selector = (
        f"experiment_id={settings.experiment_id!r}"
        if settings.experiment_id is not None
        else f"experiment_name={settings.experiment_name!r}"
    )
    return f"MLflow tracking target {selector} at {settings.tracking_uri!r} is not ready: {cause}."


def prepare_attempt(
    experiment: Experiment,
    *,
    phase: str,
    generation_id: str,
    attempt_id: str,
    run_name: str,
    overrides_sha256: str,
    inherited: Mapping[str, Winner],
    directory: Path,
    environment: Mapping[str, str],
    deadline: float | None,
) -> RunIdentity:
    """Bind the source experiment and optionally create the trainer's native run.

    Persist inherited winner run IDs beside the attempt identity for later projection replay.

    :param Experiment experiment: MLflow-backed experiment.
    :param str phase: Current phase name.
    :param str generation_id: Current invocation identity.
    :param str attempt_id: Allocated trainer identity.
    :param str run_name: The trainer's ``PHASESWEEP_RUN_NAME``, reused for a managed run.
    :param str overrides_sha256: Existing prepared-input identity.
    :param Mapping[str, Winner] inherited: Parent-phase winners fixed into this attempt.
    :param Path directory: Registered attempt directory.
    :param Mapping[str, str] environment: Composed trainer environment.
    :param float | None deadline: Existing phase/run deadline.
    :return RunIdentity: Expected identity, a full binding with a run ID in managed mode.
    """
    assert experiment.mlflow is not None
    parent = prepare_generation(
        experiment, generation_id=generation_id, environment=environment, deadline=deadline
    )
    identity: RunIdentity = {
        "schema_version": 1,
        "tracking_uri": experiment.mlflow.tracking_uri,
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "overrides_sha256": overrides_sha256,
        "experiment_id": parent["experiment_id"],
        "phase": phase,
        "attempt_id": attempt_id,
        "run_mode": experiment.mlflow.run_mode,
        "parent_run_id": parent["run_id"],
    }
    write_record(
        directory / INHERITS_FILE,
        {
            "schema_version": 1,
            "inherits": {
                name: validate_binding(winner.objective_provenance["remote_capture"]["binding"])[
                    "run_id"
                ]
                for name, winner in inherited.items()
                if winner.objective_provenance is not None
            },
        },
    )
    write_record(directory / "mlflow-expected.json", identity)
    if experiment.mlflow.run_mode == "managed":
        return _create_bound(
            experiment,
            identity,
            directory / BINDING_FILE,
            environment,
            run_name=run_name,
            deadline=deadline,
        )
    return identity
