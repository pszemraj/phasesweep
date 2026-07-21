"""Block a trial command until the orchestrator commits its process identity."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    """Signal readiness, await the parent acknowledgement, then exec the trainer."""
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
