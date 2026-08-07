"""Format override dictionaries into the shape the trial command expects.

All values are unconditionally shell-quoted via shlex.quote to prevent
injection or misparse from shell metacharacters in paths or categorical values.
"""

from __future__ import annotations

import json
import math
import shlex
from pathlib import Path
from typing import Any


def _stringify(value: Any) -> str:
    """Render a Python value into the canonical scalar form for trial commands.

    The accepted set is exactly the argparse override contract enforced at
    config load by
    :func:`phasesweep.config.models._validate_argparse_override_values`: values
    whose ``str()`` form is faithful to the JSON-mode dump the semantic
    fingerprint hashes. This function is the defense in depth behind that
    validator — it must fail loudly rather than ``str()`` a mapping or a
    ``datetime.date`` into a command line, because that is how two different
    commands end up sharing one study identity (PR #5 review / reviewer 2,
    blocker 3).

    Args:
        value: ``None``, a bool, an int, a finite float, a string, or a
            list/tuple of those — sampled by Optuna or read from
            ``fixed_overrides``.

    Returns:
        A string representation: ``"true"``/``"false"`` for ``bool``;
        ``"[a,b,c]"`` for list/tuple; ``str(value)`` for the remaining
        scalars (``None`` renders as ``"None"``).

    Raises:
        TypeError: The value is outside the argparse contract — a mapping, a
            non-finite float, or any other object (a YAML-native
            ``date``/``datetime``, a set, ...).

    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stringify(v) for v in value) + "]"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    raise TypeError(
        "override_format='argparse' supports null, booleans, integers, finite "
        "floats, strings, and lists of those; got "
        f"{type(value).__name__}: {value!r}. Use override_format='json_file' for "
        "structured values."
    )


def _stringify_hydra(value: Any) -> str:
    """Render a value for Hydra/OmegaConf override grammar.

    :param Any value: Scalar or list-like value to render.
    :raises TypeError: If the value cannot be represented by the supported grammar.
    :return str: Hydra-compatible representation of the value.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stringify_hydra(v) for v in value) + "]"
    raise TypeError(
        "override_format='hydra' supports scalar values and lists only; "
        f"got {type(value).__name__}. Use override_format='json_file' for structured values."
    )


def format_hydra(overrides: dict[str, Any]) -> str:
    """Hydra-style: ``key=value``, each token unconditionally quoted.

    Args:
        overrides: Mapping from override key to value.

    Returns:
        A single space-separated string of shell-quoted ``key=value`` tokens.

    """
    parts: list[str] = []
    for k, v in overrides.items():
        parts.append(shlex.quote(f"{k}={_stringify_hydra(v)}"))
    return " ".join(parts)


def format_argparse(overrides: dict[str, Any]) -> str:
    """argparse-style: ``--key value``, both unconditionally quoted.

    Args:
        overrides: Mapping from override key to value.

    Returns:
        A single space-separated string of shell-quoted ``--key`` ``value``
        token pairs.

    Raises:
        TypeError: A value is outside the argparse override contract (see
            :func:`_stringify`). Statically-known override values are already
            rejected at config load; this covers anything else.

    """
    parts: list[str] = []
    for k, v in overrides.items():
        parts.append(shlex.quote(f"--{k}"))
        parts.append(shlex.quote(_stringify(v)))
    return " ".join(parts)


def dump_overrides_json(payload: Any) -> str:
    """Serialize an override payload with the canonical strict JSON encoder.

    This is the single definition of "representable as a phasesweep override on
    the ``json_file`` wire". :func:`write_json_file` writes exactly this text,
    and config load runs every statically-known override value through the same
    call so a YAML scalar that PyYAML turned into a non-JSON Python object (an
    unquoted ``2024-01-01`` becomes :class:`datetime.date`) is rejected by
    ``phasesweep validate`` instead of by ``json.dumps`` inside the first real
    trial (review v0.5.17 / finding B). Keep the encoder options here and
    nowhere else — a second, laxer serializer is how the audit artifact and the
    wire artifact drift apart.

    :param Any payload: Value or mapping to serialize.
    :raises TypeError: The payload contains something the strict encoder cannot
        represent (``default=`` is deliberately not set).
    :raises ValueError: The payload contains a non-finite float (``inf``,
        ``-inf``, or ``nan``) — ``allow_nan=False`` rejects the values Python's
        ``json.dumps`` would otherwise render as the non-standard
        ``Infinity``/``-Infinity``/``NaN`` tokens, which this repo's own
        :func:`phasesweep.runtime.json.strict_json_loads` refuses to parse.
    :return str: Sorted, two-space-indented JSON text with no trailing newline.
    """
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def write_json_file(overrides: dict[str, Any], trial_dir: Path) -> Path:
    """Write a JSON overrides file with dotted keys expanded into nested dicts.

    Args:
        overrides: Mapping from override key (may contain dots) to value.
        trial_dir: Per-trial directory; the file is written as
            ``<trial_dir>/overrides.json``.

    Returns:
        The path to the written ``overrides.json`` file.

    Raises:
        ValueError: A dotted key collides with an existing scalar or would
            replace a nested object built by another key, or the payload
            contains a non-finite float.
        TypeError: The payload contains a value the strict encoder cannot
            represent.

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
    path.write_text(dump_overrides_json(nested))
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
        run_name: Composite ``<experiment>-<phase>-<trial_id>-<attempt_id>``
            identifier used for ``{run_name}``.
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
