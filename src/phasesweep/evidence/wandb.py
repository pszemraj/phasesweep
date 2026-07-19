"""W&B polling helpers."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True)
class WandbPollTimeout(TimeoutError):
    """Raised when a W&B run summary is not ready before the poll deadline."""

    run_id: str
    timeout_seconds: float
    last_error: Exception | None = None


@dataclass(frozen=True)
class WandbRunTerminalError(RuntimeError):
    """Raised when the attempt's W&B run terminates unsuccessfully."""

    run_id: str
    state: str


def poll_wandb_summary(
    *,
    entity: str,
    project: str,
    run_id: str,
    poll_seconds: float,
    timeout_seconds: float,
    required_keys: Iterable[str] = (),
    wait_for_keys: bool = True,
) -> dict[str, Any]:
    """Poll W&B until a terminal run summary is available.

    :param str entity: W&B entity or team name.
    :param str project: W&B project name.
    :param str run_id: Immutable W&B run id assigned to this trial attempt.
    :param float poll_seconds: Delay between polling attempts.
    :param float timeout_seconds: Maximum time to wait for a ready run.
    :param Iterable[str] required_keys: Summary keys that must be present.
    :param bool wait_for_keys: Whether to wait for all required keys before returning.
    :raises WandbRunTerminalError: If the run crashes, fails, or is killed.
    :raises WandbPollTimeout: If the run summary is not ready before timeout.
    :return dict[str, Any]: Terminal run summary values.
    """
    from wandb.apis.public import Api  # type: ignore[import-not-found]

    # W&B accepts an integer request timeout. The monotonic deadline below
    # remains the overall polling bound; the native timeout avoids leaving a
    # daemon thread behind when an API request stalls.
    api = Api(timeout=max(1, ceil(timeout_seconds)))
    path = f"{entity}/{project}/{run_id}"
    deadline = time.monotonic() + timeout_seconds
    last_err: Exception | None = None
    required = tuple(required_keys)
    while time.monotonic() < deadline:
        try:
            run = api.run(path)
            if run.state in {"crashed", "failed", "killed"}:
                raise WandbRunTerminalError(run_id, run.state)
            if run.state == "finished":
                summary = dict(run.summary)
                if not wait_for_keys or all(key in summary for key in required):
                    return summary
        except WandbRunTerminalError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        remaining = max(0.0, deadline - time.monotonic())
        time.sleep(min(poll_seconds, remaining))
    raise WandbPollTimeout(run_id, timeout_seconds, last_err)
