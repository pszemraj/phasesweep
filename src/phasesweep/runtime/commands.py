"""Format override dictionaries into the shape the trial command expects.

All values are unconditionally shell-quoted via shlex.quote to prevent
injection or misparse from shell metacharacters in paths or categorical values.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any


def _stringify(value: Any) -> str:
    """Render a Python value into the canonical scalar form for trial commands.

    Args:
        value: Any scalar, list, or tuple sampled by Optuna or read from
            ``fixed_overrides``.

    Returns:
        A string representation: ``"true"``/``"false"`` for ``bool``;
        ``"[a,b,c]"`` for list/tuple; ``str(value)`` otherwise.

    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stringify(v) for v in value) + "]"
    return str(value)


def format_hydra(overrides: dict[str, Any]) -> str:
    """Hydra-style: ``key=value``, each token unconditionally quoted.

    Args:
        overrides: Mapping from override key to value.

    Returns:
        A single space-separated string of shell-quoted ``key=value`` tokens.

    """
    parts: list[str] = []
    for k, v in overrides.items():
        parts.append(shlex.quote(f"{k}={_stringify(v)}"))
    return " ".join(parts)


def format_argparse(overrides: dict[str, Any]) -> str:
    """argparse-style: ``--key value``, both unconditionally quoted.

    Args:
        overrides: Mapping from override key to value.

    Returns:
        A single space-separated string of shell-quoted ``--key`` ``value``
        token pairs.

    """
    parts: list[str] = []
    for k, v in overrides.items():
        parts.append(shlex.quote(f"--{k}"))
        parts.append(shlex.quote(_stringify(v)))
    return " ".join(parts)


def write_json_file(overrides: dict[str, Any], trial_dir: Path) -> Path:
    """Write a JSON overrides file with dotted keys expanded into nested dicts.

    Args:
        overrides: Mapping from override key (may contain dots) to value.
        trial_dir: Per-trial directory; the file is written as
            ``<trial_dir>/overrides.json``.

    Returns:
        The path to the written ``overrides.json`` file.

    """
    nested: dict[str, Any] = {}
    for k, v in overrides.items():
        cur = nested
        parts = k.split(".")
        for part in parts[:-1]:
            next_value = cur.setdefault(part, {})
            if not isinstance(next_value, dict):
                raise ValueError(f"Cannot expand override {k!r}: {part!r} is already scalar.")
            cur = next_value
        if parts[-1] in cur and isinstance(cur[parts[-1]], dict):
            raise ValueError(f"Cannot expand override {k!r}: it would replace a nested object.")
        cur[parts[-1]] = v
    path = trial_dir / "overrides.json"
    path.write_text(json.dumps(nested, indent=2, sort_keys=True))
    return path


def render_command(
    template: str,
    overrides: dict[str, Any],
    fmt: str,
    *,
    trial_dir: Path,
    trial_id: int,
    phase: str,
    run_name: str,
    write_files: bool = True,
) -> str:
    """Substitute placeholders in the user's trial_command template.

    Path-like substitutions (``{trial_dir}``, ``{overrides_path}``) are
    shell-quoted.

    Args:
        template: The user's ``trial_command`` template with ``{...}``
            placeholders. Supported keys: ``overrides``, ``overrides_path``,
            ``trial_dir``, ``trial_id``, ``phase``, ``run_name``.
        overrides: The composed overrides for this trial.
        fmt: One of ``"argparse"``, ``"hydra"``, ``"json_file"``.
        trial_dir: Per-trial directory used for ``{trial_dir}`` and for the
            ``overrides.json`` file when ``fmt == "json_file"``.
        trial_id: Numeric trial number, used for ``{trial_id}``.
        phase: Phase name, used for ``{phase}``.
        run_name: Composite ``<experiment>-<phase>-<trial_id>`` identifier used
            for ``{run_name}``.
        write_files: When ``False``, render paths without writing
            ``overrides.json``. Used by dry-run previews so they are
            filesystem-pure.

    Returns:
        The fully rendered, shell-ready command string.

    Raises:
        ValueError: If ``fmt`` is not one of the three supported formats.

    """
    if fmt == "hydra":
        overrides_str = format_hydra(overrides)
        overrides_path = ""
    elif fmt == "argparse":
        overrides_str = format_argparse(overrides)
        overrides_path = ""
    elif fmt == "json_file":
        overrides_str = ""
        overrides_path = str(
            write_json_file(overrides, trial_dir) if write_files else trial_dir / "overrides.json"
        )
    else:
        raise ValueError(f"Unknown override_format: {fmt}")

    return template.format(
        overrides=overrides_str,
        overrides_path=shlex.quote(overrides_path) if overrides_path else "",
        trial_dir=shlex.quote(str(trial_dir)),
        trial_id=str(trial_id),
        phase=shlex.quote(phase),
        run_name=shlex.quote(run_name),
    )
