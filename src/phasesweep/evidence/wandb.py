"""W&B polling helpers."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from phasesweep.runtime.files import call_with_timeout


@dataclass(frozen=True)
class WandbPollTimeout(TimeoutError):
    """Raised when a W&B run summary is not ready before the poll deadline."""

    run_name: str
    timeout_seconds: float
    last_error: Exception | None = None


def render_trial_run_name(template: str, ctx: Any) -> str:
    """Render a trial run-name template from a trial context-like object.

    :param str template: Format string using trial context fields.
    :param Any ctx: Object exposing experiment, phase, trial_id, and run_name.
    :return str: Rendered W&B run display name.
    """
    return template.format(
        experiment=ctx.experiment,
        phase=ctx.phase,
        trial_id=ctx.trial_id,
        run_name=ctx.run_name,
    )


def poll_wandb_summary(
    *,
    entity: str,
    project: str,
    run_name: str,
    poll_seconds: float,
    timeout_seconds: float,
    required_keys: Iterable[str] = (),
    wait_for_keys: bool = True,
) -> dict[str, Any]:
    """Poll W&B until a terminal run summary is available.

    :param str entity: W&B entity or team name.
    :param str project: W&B project name.
    :param str run_name: Display name of the run to find.
    :param float poll_seconds: Delay between polling attempts.
    :param float timeout_seconds: Maximum time to wait for a ready run.
    :param Iterable[str] required_keys: Summary keys that must be present.
    :param bool wait_for_keys: Whether to wait for all required keys before returning.
    :raises WandbPollTimeout: If the run summary is not ready before timeout.
    :return dict[str, Any]: Terminal run summary values.
    """
    from wandb.apis.public import Api  # type: ignore[import-not-found]

    api = Api()
    path = f"{entity}/{project}"
    deadline = time.time() + timeout_seconds
    last_err: Exception | None = None
    required = tuple(required_keys)
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        try:
            runs = call_with_timeout(
                lambda: api.runs(path, filters={"display_name": run_name}),
                timeout=remaining,
            )
            if len(runs) >= 1:
                run = runs[0]
                if run.state in {"finished", "crashed", "failed"}:
                    summary = dict(run.summary)
                    if not wait_for_keys or all(key in summary for key in required):
                        return summary
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(poll_seconds)
    raise WandbPollTimeout(run_name, timeout_seconds, last_err)
