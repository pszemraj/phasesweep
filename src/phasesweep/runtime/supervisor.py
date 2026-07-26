"""Block a trial command until the orchestrator commits its process identity.

STDLIB-ONLY, NO PHASESWEEP IMPORTS (review v0.5.15 / blocker 1): this script
runs in its own interpreter BEFORE PhaseSweep's durable process identity
exists and BEFORE the parent has applied the trainer's environment. Anything
importable from that environment — a poisoned or shadowed ``phasesweep``
package via a trainer-composed ``PYTHONPATH``, a ``sitecustomize``/
``usercustomize`` module, or heavy eager imports in ``phasesweep/__init__.py``
— must never get a chance to run here: it could fork and ``setsid``-escape
undetected before any identity record is written. The parent therefore
launches this file directly (never ``-m phasesweep.runtime.supervisor``)
under ``python -I -S`` with a minimal sanitized environment. ``-S`` skips
site initialization (no ``sitecustomize``/``usercustomize``); ``-I``
(isolated mode) implies ``-P`` on Python >= 3.11, so this script's own
directory — which contains the sibling module ``phasesweep/runtime/json.py``
— is never prepended to ``sys.path``, keeping ``import json`` resolved to the
stdlib module every time. Keep imports here limited to the stdlib modules
``os``, ``sys``, ``json``, and ``signal``.
"""

from __future__ import annotations

import json
import os
import signal
import sys

# Single source of truth for the wire frame's header width:
# phasesweep.runtime.process._encode_launch_payload reads this constant
# directly (the parent already imports this module).
_HEADER_LEN = 10


def _read_exact(fd: int, size: int) -> bytes | None:
    """Read exactly ``size`` bytes from ``fd``, looping past partial reads.

    :param int fd: File descriptor to read from.
    :param int size: Exact number of bytes required.
    :return bytes | None: The bytes read, or ``None`` if ``fd`` hit EOF before
        ``size`` bytes were available.
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_launch_payload(ack_fd: int) -> tuple[str, dict[str, str], str | None] | None:
    """Read and parse the length-prefixed launch payload from the ack pipe.

    The wire format is ``_HEADER_LEN`` ASCII decimal bytes giving the byte
    length of a UTF-8 JSON body, immediately followed by that many body
    bytes. The body must decode to ``{"cmd": <str>, "env": {<str>: <str>}}``
    with an optional ``"cwd": <str>`` naming the trainer's working directory
    (review v0.5.17 / blocker 4).

    Encoded on the other end by ``phasesweep.runtime.process._encode_launch_payload``
    (prose reference only — this module is stdlib-only and must never import
    phasesweep); keep both sides in sync if the wire format changes.

    :param int ack_fd: Read end of the acknowledgement pipe.
    :return tuple[str, dict[str, str], str | None] | None: ``(cmd, env, cwd)``
        on a well-formed payload; ``None`` on EOF, a malformed header, a
        truncated body, or a body of the wrong shape.
    """
    header = _read_exact(ack_fd, _HEADER_LEN)
    if header is None:
        return None
    try:
        length = int(header.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return None
    if length < 0:
        return None
    body = _read_exact(ack_fd, length)
    if body is None:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    cmd = payload.get("cmd")
    env = payload.get("env")
    cwd = payload.get("cwd")
    if not isinstance(cmd, str) or not isinstance(env, dict):
        return None
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        return None
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None
    return cmd, env, cwd


def main(argv: list[str] | None = None) -> int:
    """Signal readiness, await the framed launch payload, then exec the trainer.

    :param list[str] | None argv: Optional ``[ready_fd, ack_fd]`` argument
        vector; defaults to ``sys.argv[1:]`` when omitted.
    :return int: ``64`` if ``argv`` does not have exactly two elements; ``75``
        if the acknowledgement pipe does not deliver a well-formed
        ``{"cmd": str, "env": {str: str}}`` payload before EOF or a shape
        violation; ``76`` if the payload's optional working directory cannot
        be entered; on success, ``os.execve`` replaces this process image with
        ``/bin/sh -c cmd`` under the delivered environment and never returns
        here — the trailing ``127`` only guards against ``execve`` returning
        unexpectedly.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        return 64
    ready_fd = int(args[0])
    ack_fd = int(args[1])

    # The parent spawns this process from inside its shutdown-signal deferral
    # window, and the blocked signal mask survives fork AND exec — without
    # this reset, the supervisor and every trainer exec'd from it would run
    # with SIGTERM/SIGINT/SIGHUP permanently blocked, making the documented
    # SIGTERM -> grace -> SIGKILL escalation a dead letter (trainers could
    # never shut down gracefully and every kill burned the full grace before
    # SIGKILL). This is the first code phasesweep controls after exec, so the
    # mask is cleaned here.
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_SETMASK, set())

    try:
        # Checked by phasesweep.runtime.process._spawn_blocked_supervisor's
        # `os.read(ready_read, 1) != b"R"` readiness gate.
        os.write(ready_fd, b"R")
    finally:
        os.close(ready_fd)

    try:
        payload = _read_launch_payload(ack_fd)
    finally:
        os.close(ack_fd)
    if payload is None:
        return 75
    cmd, env, cwd = payload

    if cwd is not None:
        # The execution contract's trainer working directory (review v0.5.17 /
        # blocker 4). A failed chdir must fail loudly BEFORE exec — running
        # the trainer from the wrong directory would silently change what a
        # relative-path command means.
        try:
            os.chdir(cwd)
        except OSError:
            return 76

    os.execve("/bin/sh", ["sh", "-c", cmd], env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
