"""The objectives a phase runs on worker threads, and the wait that keeps a shutdown from reaping them.

Under ``n_jobs`` above 1 a shutdown signal raises in the main thread while
Optuna's worker threads are still finishing their trials. The main thread waits
in :func:`_await_worker_trials` before the run's reconciliation may reap
anything, so each worker records its own trial's outcome first.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator

import optuna

# How often the main thread re-checks the workers while it waits for them.
_WORKER_POLL_SECONDS = 0.05
# How long a RUNNING trial with no objective running may take to be told, which
# covers Optuna's own work between ``ask`` and the objective and after it.
_WORKER_SETTLE_SECONDS = 2.0


class _WorkerTrials:
    """The objectives running in one phase and the trials they left to stale recovery."""

    def __init__(self) -> None:
        """Start with no objective running and no trial abandoned."""
        self._condition = threading.Condition()
        self._running = 0
        self._abandoned: set[int] = set()

    @contextlib.contextmanager
    def track(self, trial_number: int) -> Iterator[None]:
        """Count one objective as running for as long as the block runs.

        An exception Optuna does not catch (a shutdown, the outcome-write
        abort) leaves the trial ``RUNNING`` without a tell, so the trial is
        marked abandoned: nothing will finish it in this process.

        :param int trial_number: The trial the objective runs.
        :return Iterator[None]: Context for the objective's body.
        """
        with self._condition:
            self._running += 1
        try:
            yield
        except BaseException as exc:
            if not isinstance(exc, (Exception, KeyboardInterrupt)):
                with self._condition:
                    self._abandoned.add(trial_number)
            raise
        finally:
            with self._condition:
                self._running -= 1
                self._condition.notify_all()

    def wait_idle(self) -> frozenset[int]:
        """Block until no objective is running, then return the abandoned trials.

        :return frozenset[int]: Trials whose objective left them ``RUNNING`` for recovery.
        """
        with self._condition:
            while self._running:
                self._condition.wait(timeout=_WORKER_POLL_SECONDS)
            return frozenset(self._abandoned)


def _await_worker_trials(workers: _WorkerTrials, study: optuna.Study) -> None:
    """Wait until no worker thread still owns a trial of ``study``, so reconciliation never reaps one.

    Waits for every running objective to return, then for every ``RUNNING``
    trial to be told, except those an objective abandoned. A trial still
    ``RUNNING`` :data:`_WORKER_SETTLE_SECONDS` after the last objective
    returned (its tell failed) is left to the reconciliation. The shutdown
    handler killed every worker's trainer and no new one may start, so the
    wait lasts as long as their in-flight bookkeeping; a second signal ends it.

    :param _WorkerTrials workers: The phase's objective tracker.
    :param optuna.Study study: The phase study whose trials the workers run.
    """
    settle_deadline: float | None = None
    while True:
        abandoned = workers.wait_idle()
        try:
            running = {
                trial.number
                for trial in study.get_trials(
                    deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
                )
            }
        except Exception:
            return
        if running <= abandoned:
            return
        now = time.monotonic()
        if settle_deadline is None:
            settle_deadline = now + _WORKER_SETTLE_SECONDS
        elif now >= settle_deadline:
            return
        time.sleep(_WORKER_POLL_SECONDS)
