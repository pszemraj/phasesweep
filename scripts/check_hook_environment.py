"""Stop a commit or push made outside the development environment.

Every git hook runs a tool from ``PATH``. Started from another Python environment, the
later hooks fail on missing imports and missing stubs, which read like defects in the code.
This check runs first and names the cause instead.

Run by the ``environment`` hook at both the commit and the push stage. It passes when the
Python first on ``PATH`` finds ``phasesweep`` in this checkout's ``src/``. The package is
located, not imported, so a commit of source that does not import yet is not blamed on the
environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CHECKOUT_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "phasesweep"

NOT_INSTALLED_REMEDY = (
    "The git hooks run the tools of the environment this checkout is installed in "
    '(pip install -e ".[dev]"). Activate that environment, or put its bin directory first '
    "on PATH, and retry."
)
OTHER_SOURCE_REMEDY = (
    "The hooks would check that other source. Activate the environment this checkout is "
    'installed in, or install this checkout into the active one (pip install -e ".[dev]").'
)


def environment_problem() -> str | None:
    """Return why the running Python is not this checkout's development environment.

    :return str | None: The problem and its remedy, or None when the environment is usable.
    """
    spec = importlib.util.find_spec("phasesweep")
    if spec is None or spec.origin is None:
        return f"{sys.executable} has no phasesweep installed.\n{NOT_INSTALLED_REMEDY}"
    found = Path(spec.origin).resolve().parent
    if found != CHECKOUT_PACKAGE:
        return (
            f"{sys.executable} finds phasesweep in {found}, "
            f"not in this checkout ({CHECKOUT_PACKAGE}).\n{OTHER_SOURCE_REMEDY}"
        )
    return None


def main() -> int:
    """Report an unusable hook environment.

    :return int: 1 when the hooks would run outside the development environment, else 0.
    """
    problem = environment_problem()
    if problem is None:
        return 0
    print(problem, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
