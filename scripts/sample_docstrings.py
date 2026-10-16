#!/usr/bin/env python3
"""Print a random sample of Python docstrings, including attribute docstrings."""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import random
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the source directory and sample count."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "src",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "src",
        help="Python source directory",
    )
    parser.add_argument("-n", "--count", type=int, default=10, help="Maximum docstrings to sample")
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")
    return args


def collect_docstrings(src: Path) -> list[tuple[Path, int, str]]:
    """Collect standalone documentation strings without importing source modules."""
    docstrings: list[tuple[Path, int, str]] = []
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                docstrings.append((path, node.lineno, inspect.cleandoc(node.value.value)))
    return docstrings


def main(argv: list[str] | None = None) -> int:
    """Print a random sample and return its exit status."""
    try:
        args = parse_args(argv)
        if not args.src.is_dir():
            print(f"error: not a directory: {args.src}", file=sys.stderr)
            return 2
        docstrings = collect_docstrings(args.src)
        sample = random.sample(docstrings, min(args.count, len(docstrings)))
        print(f"Sampled {len(sample)} of {len(docstrings)} docstrings", file=sys.stderr)
        for path, line, docstring in sample:
            print(f"--- {path}:{line} ---\n{docstring}\n")
        sys.stdout.flush()
        return 0
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        with contextlib.suppress(BrokenPipeError):
            sys.stdout.close()
        return 1
    except (OSError, UnicodeError, SyntaxError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
