"""Block a trial command until the orchestrator commits its process identity."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    """Signal readiness, await the parent acknowledgement, then exec the trainer.

    :param list[str] | None argv: Optional ``[ready_fd, ack_fd, command]``
        argument vector; defaults to ``sys.argv[1:]`` when omitted.
    :return int: ``64`` if ``argv`` does not have exactly three elements;
        ``75`` if the acknowledgement pipe does not deliver exactly ``b"A"``;
        on success, ``os.execl`` replaces this process image with ``/bin/sh -c
        command`` and never returns here — the trailing ``127`` only guards
        against ``execl`` returning unexpectedly.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        return 64
    ready_fd = int(args[0])
    ack_fd = int(args[1])
    command = args[2]

    try:
        os.write(ready_fd, b"R")
    finally:
        os.close(ready_fd)
    try:
        acknowledgement = os.read(ack_fd, 1)
    finally:
        os.close(ack_fd)
    if acknowledgement != b"A":
        return 75

    os.execl("/bin/sh", "sh", "-c", command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
