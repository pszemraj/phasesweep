"""Stop a commit or push made outside the development environment.

Every git hook runs a tool from ``PATH``. Started from another Python environment, the
later hooks fail on missing imports and missing stubs, which read like defects in the code.
This check runs first and names the cause instead.

Run by the ``environment`` hook at both the commit and the push stage. It passes when the
Python first on ``PATH`` imports ``phasesweep`` from this checkout's ``src/``.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

CHECKOUT_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "phasesweep"

REMEDY = (
    "The git hooks run the tools of the environment this checkout is installed in "
    '(pip install -e ".[dev]"). Activate that environment, or put its bin directory first '
    "on PATH, and retry."
)


def environment_problem() -> str | None:
    """Return why the running Python is not this checkout's development environment.

    :return str | None: The problem in one sentence, or None when the environment is usable.
    """
    try:
        import phasesweep
    except ImportError as exc:
        return f"{sys.executable} cannot import phasesweep: {exc}."
    imported = Path(inspect.getfile(phasesweep)).resolve().parent
    if imported != CHECKOUT_PACKAGE:
        return (
            f"{sys.executable} imports phasesweep from {imported}, "
            f"not from this checkout ({CHECKOUT_PACKAGE})."
        )
    return None


def main() -> int:
    """Report an unusable hook environment.

    :return int: 1 when the hooks would run outside the development environment, else 0.
    """
    problem = environment_problem()
    if problem is None:
        return 0
    print(f"{problem}\n{REMEDY}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
