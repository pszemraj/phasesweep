#!/usr/bin/env python3
"""Sample Python docstrings or audit all docstrings and comments for prose bloat.

Audit findings are review candidates, not proof of a defect. Behavior claims
still need checking against source. Audit mode reports every candidate and
never modifies files.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import io
import random
import re
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

PROSE_RULES = (
    (
        "history/review narration",
        r"\b(?:PR #\d+|review(?:er)? (?:v?\d+|feedback|request)|"
        r"previous implementation|formerly|used to (?:be|have)|no longer needed|"
        r"introduced (?:in|by)|the old (?:implementation|version|helper))\b",
    ),
    (
        "emphatic wording (review context)",
        r"\b(?:deliberately|intentionally|on purpose|by design|exactly wrong|"
        r"would be defensible|the split is deliberate)\b",
    ),
    (
        "consolidation narration",
        r"\b(?:single source of truth|the (?:one|only) place|"
        r"(?:this|the) (?:shared|common) (?:helper|implementation) (?:replaces|avoids)|"
        r"(?:duplicated|duplicate) (?:implementation|logic|code))\b",
    ),
    (
        "test/review citation",
        r"\btests/\S+\.py\b|\b(?:see|per|according to) (?:the )?(?:review|audit)\b",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the source directory, sampling options, and audit mode."""
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
    parser.add_argument("--seed", type=int, help="Seed for repeatable docstring sampling")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Scan all docstrings and comments; --count applies only to sampling",
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")
    return args


def collect_prose(src: Path, *, comments: bool = False) -> list[tuple[Path, int, str]]:
    """Collect standalone documentation strings and optionally adjacent comments."""
    docstrings: list[tuple[Path, int, str]] = []
    for path in sorted(src.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                docstrings.append((path, node.lineno, inspect.cleandoc(node.value.value)))
        if comments:
            pending: list[tuple[int, str]] = []
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type != tokenize.COMMENT:
                    continue
                line = token.start[0]
                if pending and line > pending[-1][0] + 1:
                    docstrings.append((path, pending[0][0], "\n".join(t[1] for t in pending)))
                    pending = []
                pending.append((line, token.string.lstrip("# ")))
            if pending:
                docstrings.append((path, pending[0][0], "\n".join(t[1] for t in pending)))
    return docstrings


def audit_prose(blocks: list[tuple[Path, int, str]]) -> list[tuple[Path, int, str, list[str]]]:
    """Rank lexical, repeated-paragraph, and long-prose candidates with evidence."""
    reasons: dict[int, list[str]] = defaultdict(list)
    repeated: dict[str, set[int]] = defaultdict(set)
    for index, (_, _, text) in enumerate(blocks):
        for label, pattern in PROSE_RULES:
            if match := re.search(pattern, text, flags=re.IGNORECASE):
                reasons[index].append(f"{label}: {match.group()!r}")
        prose = re.split(
            r"(?m)^\s*(?::(?:param|return|raises|rtype|type)\b|"
            r"(?:Args|Arguments|Returns|Raises|Attributes|Examples):\s*$)",
            text,
            maxsplit=1,
        )[0]
        words = len(re.findall(r"\b\w+\b", prose))
        if words >= 120:
            reasons[index].append(f"long narrative: {words} words before API sections")
        for paragraph in re.split(r"\n\s*\n", prose):
            normalized = " ".join(paragraph.split()).casefold()
            if len(re.findall(r"\b\w+\b", normalized)) >= 20:
                repeated[normalized].add(index)
    for indexes in repeated.values():
        if len(indexes) > 1:
            for index in sorted(indexes):
                others = ", ".join(
                    f"{blocks[other][0]}:{blocks[other][1]}" for other in sorted(indexes - {index})
                )
                reasons[index].append(f"repeated paragraph also at {others}")
    return [
        (*blocks[index], labels)
        for index, labels in sorted(
            reasons.items(),
            key=lambda item: (-len(item[1]), str(blocks[item[0]][0]), blocks[item[0]][1]),
        )
    ]


def main(argv: list[str] | None = None) -> int:
    """Print a random sample or all audit candidates and return its exit status."""
    try:
        args = parse_args(argv)
        if not args.src.is_dir():
            print(f"error: not a directory: {args.src}", file=sys.stderr)
            return 2
        docstrings = collect_prose(args.src, comments=args.audit)
        if args.audit:
            findings = audit_prose(docstrings)
            print(
                f"Scanned {len(docstrings)} documentation/comment blocks; "
                f"{len(findings)} review candidates (no automatic deletions)",
                file=sys.stderr,
            )
            for path, line, text, reasons in findings:
                print(f"--- {path}:{line} ---\nFlags: {'; '.join(reasons)}\n{text}\n")
        else:
            sample = random.Random(args.seed).sample(docstrings, min(args.count, len(docstrings)))
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
