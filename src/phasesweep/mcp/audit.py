"""Structured audit logging for MCP tool calls.

Audit records are operator-facing JSON lines written under ``state_dir``. They
intentionally log identifiers, counts, outcomes, and state transitions instead
of result payloads, paths, commands, env, logs, or sampled parameters.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.files import open_private_text

log = logging.getLogger("phasesweep.mcp.audit")
MAX_AUDIT_STRING_LENGTH = 256


def _compact_value(value: Any) -> Any:
    """Return an audit-safe scalar with bounded string size.

    :param Any value: Candidate audit field value.
    :return Any: ``value`` unchanged, unless it is a string longer than
        ``MAX_AUDIT_STRING_LENGTH``, in which case a truncated copy ending in
        ``"..."`` is returned.
    """
    if isinstance(value, str) and len(value) > MAX_AUDIT_STRING_LENGTH:
        return f"{value[: MAX_AUDIT_STRING_LENGTH - 3]}..."
    return value


def _compact_mapping(values: dict[str, Any] | None) -> dict[str, Any]:
    """Return a shallow copy without ``None`` values.

    :param dict[str, Any] | None values: Optional mapping to compact.
    :return dict[str, Any]: Empty dict for ``None`` or a shallow copy with null-valued entries removed.
    """
    if values is None:
        return {}
    return {key: _compact_value(value) for key, value in values.items() if value is not None}


@dataclass
class AuditLogger:
    """Append-only JSONL audit sink for MCP tool calls."""

    path: Path
    actor: str = "local-stdio"
    session_id: str = field(default_factory=lambda: uuid4().hex)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def _event(
        self,
        *,
        tool: str,
        args: dict[str, Any] | None,
        outcome: str,
        resolved: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        result_counts: dict[str, int] | None = None,
        authorization: dict[str, Any] | None = None,
        error_type: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Build one bounded audit event without writing it."""
        event: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "actor": self.actor,
            "session_id": self.session_id,
            "transport": "stdio",
            "tool": tool,
            "args": _compact_mapping(args),
            "outcome": outcome,
        }
        resolved_values = _compact_mapping(resolved)
        if resolved_values:
            event["resolved"] = resolved_values
        if state_before is not None:
            event["state_before"] = state_before
        if state_after is not None:
            event["state_after"] = state_after
        if result_counts is not None:
            event["result_counts"] = result_counts
        authorization_values = _compact_mapping(authorization)
        if authorization_values:
            event["authorization"] = authorization_values
        if error_type is not None:
            event["error_type"] = error_type
        if error is not None:
            event["error"] = error
        return event

    def _append(self, event: dict[str, Any], *, durable: bool) -> None:
        """Append one event, optionally syncing it before returning."""
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            created = not self.path.exists()
            with open_private_text(self.path, "a") as fh:
                fh.write(line + "\n")
                if durable:
                    fh.flush()
                    os.fsync(fh.fileno())
            if durable and created:
                fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)

    def record_authorization(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        resolved: dict[str, Any],
        state_before: dict[str, Any],
        authorization: dict[str, Any],
    ) -> None:
        """Durably record granted authority before a side effect starts.

        Unlike ordinary outcome logging, this method propagates all write and
        sync failures. Callers must refuse the side effect when it raises.

        :param str tool: Side-effecting MCP tool name.
        :param dict[str, Any] args: Agent-supplied safe arguments.
        :param dict[str, Any] resolved: Server-resolved experiment and run ids.
        :param dict[str, Any] state_before: Safe state summary before the operation.
        :param dict[str, Any] authorization: Granted catalog policy and config identity.
        """
        event = self._event(
            tool=tool,
            args=args,
            outcome="authorized",
            resolved=resolved,
            state_before=state_before,
            authorization=authorization,
        )
        self._append(event, durable=True)

    def record(
        self,
        *,
        tool: str,
        args: dict[str, Any] | None = None,
        outcome: str,
        resolved: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        result_counts: dict[str, int] | None = None,
        error_type: str | None = None,
        error: str | None = None,
    ) -> None:
        """Write one audit event.

        Audit failure is logged to stderr but does not change the tool result:
        a successful side effect should not be reported to the agent as failed
        solely because post-action audit append hit an operator filesystem
        problem.

        :param str tool: MCP tool name being audited.
        :param dict[str, Any] | None args: Agent-supplied safe arguments to record.
        :param str outcome: Outcome label, usually ``success`` or ``error``.
        :param dict[str, Any] | None resolved: Server-resolved ids such as ``experiment_id`` or ``run_id``.
        :param dict[str, Any] | None state_before: Safe state summary before the operation.
        :param dict[str, Any] | None state_after: Safe state summary after the operation.
        :param dict[str, int] | None result_counts: Counts derived from the result without copying result payloads.
        :param str | None error_type: Exception class name for failed tool calls.
        :param str | None error: Redacted error message for failed tool calls.
        """
        try:
            self._append(
                self._event(
                    tool=tool,
                    args=args,
                    outcome=outcome,
                    resolved=resolved,
                    state_before=state_before,
                    state_after=state_after,
                    result_counts=result_counts,
                    error_type=error_type,
                    error=error,
                ),
                durable=False,
            )
        except (OSError, TypeError, ValueError) as exc:
            log.warning("failed to write MCP audit record to %s: %s", self.path, exc)
