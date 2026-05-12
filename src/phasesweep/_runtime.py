"""Shared runtime helpers for subprocess-adjacent code."""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def call_with_timeout(fn: Callable[[], T], *, timeout: float) -> T:
    """Run a blocking function in a daemon thread and bound caller wait time."""
    q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            q.put((True, fn()))
        except Exception as exc:  # noqa: BLE001
            q.put((False, exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=max(0.0, timeout))
    if thread.is_alive():
        raise TimeoutError(f"call exceeded {timeout:g}s")
    ok, value = q.get_nowait()
    if ok:
        return value
    raise value


def json_path(data: Any, key: str) -> Any:
    """Resolve a dotted key in a JSON-like object."""
    cur = data
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(part)
    return cur


def load_json_file(path: Path) -> Any:
    """Load JSON from ``path``."""
    return json.loads(path.read_text())


def lock_dir() -> Path:
    """Return the shared same-host phasesweep lock directory."""
    path = Path(tempfile.gettempdir()) / "phasesweep-locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class WandbPollTimeout(TimeoutError):
    """Raised when a W&B run summary is not ready before the poll deadline."""

    run_name: str
    timeout_seconds: float
    last_error: Exception | None = None


def render_trial_run_name(template: str, ctx: Any) -> str:
    """Render a trial run-name template from a trial context-like object."""
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
    """Poll W&B until a terminal run summary is available."""
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
                timeout=min(poll_seconds, remaining),
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
