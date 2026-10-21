"""Format override dictionaries into the shape the trial command expects.

All values are unconditionally shell-quoted via shlex.quote to prevent
injection or misparse from shell metacharacters in paths or categorical values.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError

from phasesweep.config.common import _find_prefix_collisions

_CliOverrideFormat = Literal["argparse", "hydra"]
_FILE_CONFIG_VALUE: TypeAdapter[JsonValue] = TypeAdapter(
    JsonValue,
    config=ConfigDict(strict=True, allow_inf_nan=False, hide_input_in_errors=True),
)

# The trainer-input file each ``override_format`` leaves in a trial directory.
# The trial writer and the evidence verifier both read it; the names are part
# of the documented output layout.
TRAINER_INPUT_FILENAMES: dict[str, str] = {
    "yaml_file": "trainer_config.yaml",
    "argparse": "overrides_resolved.json",
    "hydra": "overrides_resolved.json",
    "json_file": "overrides.json",
}


class _OverrideValueError(TypeError):
    """A value outside the scalar/list CLI override contract."""

    def __init__(self, fmt: _CliOverrideFormat, value: object, position: str) -> None:
        """Describe the exact nested value the selected CLI wire cannot render.

        :param _CliOverrideFormat fmt: Selected CLI override grammar.
        :param object value: Unsupported value encountered while rendering.
        :param str position: Nested list position used in the diagnostic.
        """
        where = f" at position {position}" if position else ""
        reason = (
            "Hydra cannot pass literal strings containing '${' interpolation syntax or control characters. "
            if fmt == "hydra" and isinstance(value, str)
            else ""
        )
        if isinstance(value, float):
            reason = 'Use a finite value, or quote it in YAML (e.g. "inf") to send it as text. '
        elif isinstance(value, date):
            reason = "Quote YAML dates or timestamps to send them as strings. "
        super().__init__(
            f"override_format={fmt!r} supports null, booleans, integers, finite "
            f"floats, strings, and lists of those; got{where} "
            f"{type(value).__name__}: {value!r}. {reason}Use the default "
            "override_format='yaml_file' for structured trainer configuration."
        )


def _render_override_value(
    value: Any,
    fmt: _CliOverrideFormat,
    *,
    position: str = "",
    _ancestors: set[int] | None = None,
) -> str:
    """Render one value using the scalar/list CLI override contract.

    :param Any value: Scalar or recursively list-like override value.
    :param _CliOverrideFormat fmt: Target CLI override grammar.
    :param str position: Nested list position used in diagnostics.
    :param set[int] | None _ancestors: Internal recursive-container guard.
    :raises _OverrideValueError: If a value has no faithful CLI wire form.
    :return str: Canonical value text for ``fmt``.
    """
    if value is None:
        return "None" if fmt == "argparse" else "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if fmt == "argparse":
            return value
        # OmegaConf resolves interpolations even inside Hydra quoted strings.
        # Literal interpolation syntax is outside this wire contract.
        if "${" in value or any(ord(char) < 32 for char in value):
            raise _OverrideValueError(fmt, value, position)
        # Hydra only unescapes backslashes preceding a quote (including the
        # closing quote). JSON escaping changes other backslashes and Unicode.
        escaped = re.sub(r'\\+(?="|$)', lambda match: match.group() * 2, value)
        return '"' + escaped.replace('"', '\\"') + '"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return str(value)
        raise _OverrideValueError(fmt, value, position)
    if isinstance(value, (list, tuple)):
        ancestors = set() if _ancestors is None else _ancestors
        if id(value) in ancestors:
            raise _OverrideValueError(fmt, value, position)
        ancestors.add(id(value))
        try:
            rendered = (
                _render_override_value(
                    item,
                    fmt,
                    position=f"{position}[{index}]",
                    _ancestors=ancestors,
                )
                for index, item in enumerate(value)
            )
            return "[" + ",".join(rendered) + "]"
        finally:
            ancestors.remove(id(value))
    raise _OverrideValueError(fmt, value, position)


def format_cli_overrides(overrides: Mapping[str, object], fmt: _CliOverrideFormat) -> str:
    """Render shell-quoted override tokens for argparse or Hydra.

    :param Mapping[str, object] overrides: Scalar/list values keyed by dotted parameter names.
    :param _CliOverrideFormat fmt: ``argparse`` prefixes keys with ``--``; Hydra uses bare keys.
    :return: Space-separated ``key=value`` tokens in insertion order.
    :raises TypeError: A value has no faithful representation in the selected grammar.
    """
    prefix = "--" if fmt == "argparse" else ""
    return " ".join(
        shlex.quote(f"{prefix}{key}={_render_override_value(value, fmt)}")
        for key, value in overrides.items()
    )


def _validate_file_config_value(payload: object, *, position: str) -> None:
    """Report invalid trainer values with dotted keys and list indices."""
    try:
        _FILE_CONFIG_VALUE.validate_python(payload)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        location = error["loc"]
        # JsonValue locations alternate union tags with mapping keys or indices.
        path = location[1::2]
        is_key = error["type"] == "string_type"
        if is_key:
            path = path[:-1]
        for part in path:
            position += f"[{part}]" if isinstance(part, int) else f".{part}"
        value = error["input"]
        if is_key:
            message = (
                f"{position} has non-string mapping key {value!r} "
                f"({type(value).__name__}). Trainer config keys must be strings."
            )
        elif error["type"] == "finite_number":
            message = f"{position} must be finite; got {type(value).__name__}: {value!r}."
        elif error["type"] == "recursion_loop":
            message = f"{position} contains a recursive YAML container."
        else:
            message = (
                f"{position} has unsupported type {type(value).__name__}: {value!r}. "
                "Use strings, booleans, integers, finite floats, null, lists, and "
                "string-keyed mappings; quote YAML dates or timestamps to keep them strings."
            )
        raise ValueError(message) from exc


def dump_overrides_json(payload: Any) -> str:
    """Encode only JSON-compatible values using the trainer's strict wire format.

    :param Any payload: JSON-compatible scalar, list, or string-keyed mapping.
    :return str: Deterministic JSON text.
    :raises ValueError: A value is non-finite, recursive, or outside the strict JSON domain.
    """
    _validate_file_config_value(payload, position="JSON overrides")
    return encode_overrides_json(payload)


def encode_overrides_json(payload: object) -> str:
    """Encode override metadata, including CLI tuples, as deterministic JSON.

    :param object payload: Values accepted by the JSON encoder.
    :return str: Sorted, two-space-indented JSON without a trailing newline.
    :raises TypeError: A value cannot be JSON-encoded.
    :raises ValueError: A value is non-finite or recursive.
    """
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def compose_trainer_config(
    trainer_config: dict[str, JsonValue],
    overrides: dict[str, JsonValue],
    *,
    substitutions: dict[str, str] | None = None,
) -> dict[str, JsonValue]:
    """Apply dotted overrides to a deep copy of the trainer configuration.

    Create missing mappings and replace the exact leaf, including mapping or list
    values. Expand placeholders in base strings before applying literal overrides.

    :param dict[str, JsonValue] trainer_config: Operator-authored base trainer config.
    :param dict[str, JsonValue] overrides: Composed inherited, fixed, and sampled values.
    :param dict[str, str] | None substitutions: Runtime placeholder replacements.
    :raises ValueError: An override path must descend through a non-mapping value.
    :return dict[str, JsonValue]: Deep-copied complete configuration for one trial.
    """
    base: dict[str, JsonValue] = (
        _substitute_trainer_config_placeholders(trainer_config, substitutions)
        if substitutions
        else trainer_config
    )
    composed = deepcopy(base)
    for key, value in overrides.items():
        current = composed
        parts = key.split(".")
        for index, part in enumerate(parts[:-1]):
            existing = current.get(part)
            if existing is None and part not in current:
                child: dict[str, JsonValue] = {}
                current[part] = child
                current = child
                continue
            if not isinstance(existing, dict):
                prefix = ".".join(parts[: index + 1])
                raise ValueError(
                    f"Cannot apply override {key!r}: trainer_config path {prefix!r} "
                    f"is {type(existing).__name__}, not a mapping."
                )
            current = existing
        current[parts[-1]] = deepcopy(value)
    return composed


def _substitute_trainer_config_placeholders(value: Any, substitutions: dict[str, str]) -> Any:
    """Expand PhaseSweep runtime placeholders inside trainer-config strings.

    Each source string is scanned once so placeholder-like text inside a real
    replacement value (for example, a workdir containing ``{phase}``) remains
    literal instead of being substituted again.

    :param Any value: Configuration subtree to copy and expand.
    :param dict[str, str] substitutions: Literal placeholder-to-value mapping.
    :return Any: Expanded copy of ``value``.
    """
    if isinstance(value, str):
        pattern = "|".join(re.escape(placeholder) for placeholder in substitutions)
        if not pattern:
            return value
        return re.sub(pattern, lambda match: substitutions[match.group(0)], value)
    if isinstance(value, list):
        return [_substitute_trainer_config_placeholders(item, substitutions) for item in value]
    if isinstance(value, dict):
        return {
            key: _substitute_trainer_config_placeholders(item, substitutions)
            for key, item in value.items()
        }
    return value


def flatten_trainer_config(mapping: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Flatten nested mappings into dotted keys for a command-line wire.

    Lists and scalars are leaves. An empty mapping stays a leaf so the CLI
    renderer refuses it with its usual message.

    :param Mapping[str, JsonValue] mapping: Trainer configuration, possibly nested.
    :raises TypeError: A mapping key is not a string.
    :raises ValueError: Two paths flatten to one dotted key, as a literal
        ``"a.b"`` key does beside a nested ``a: {b: ...}``.
    :return dict[str, JsonValue]: Dotted key to leaf value, in traversal order.
    """
    flat: dict[str, JsonValue] = {}
    _flatten_into(flat, mapping, "")
    return flat


def _flatten_into(
    flat: dict[str, JsonValue], mapping: Mapping[str, JsonValue], prefix: str
) -> None:
    """Add the leaves of ``mapping`` to ``flat`` under dotted keys.

    :param dict[str, JsonValue] flat: Accumulator of dotted key to leaf value.
    :param Mapping[str, JsonValue] mapping: Subtree to walk.
    :param str prefix: Dotted path of ``mapping``; empty at the root.
    :raises TypeError: A mapping key is not a string.
    :raises ValueError: A dotted key is produced twice.
    """
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(
                f"trainer_config has non-string mapping key {key!r} "
                f"({type(key).__name__}). Trainer config keys must be strings."
            )
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            _flatten_into(flat, value, path)
        elif path in flat:
            raise ValueError(
                f"trainer_config produces the dotted key {path!r} twice; a literal key "
                "containing a dot collides with the nested mapping of the same path."
            )
        else:
            flat[path] = value


def compose_trainer_input(
    trainer_config: dict[str, JsonValue],
    overrides: dict[str, JsonValue],
    fmt: str,
    *,
    substitutions: dict[str, str] | None = None,
) -> str | dict[str, JsonValue]:
    """Compose trainer input without writing files.

    File modes return serialized text; CLI modes return ordered flattened values.
    An empty CLI base preserves override order and supported tuple values.
    Placeholders expand only in base strings, before overrides apply.

    :param dict[str, JsonValue] trainer_config: Base trainer configuration.
    :param dict[str, JsonValue] overrides: Fully composed trial overrides.
    :param str fmt: One of the four supported override formats.
    :param dict[str, str] | None substitutions: Runtime placeholder replacements.
    :raises ValueError: The format is unknown, configuration paths collide, or
        a file value is outside the finite JSON domain.
    :return str | dict[str, JsonValue]: Deterministic file text, or merged CLI values.
    """
    if fmt not in ("yaml_file", "json_file", "argparse", "hydra"):
        raise ValueError(f"Unknown override_format: {fmt}")
    if fmt in ("argparse", "hydra") and not trainer_config:
        return dict(overrides)
    if fmt == "json_file":
        collisions = {child for _, child in _find_prefix_collisions(set(overrides))}
        for key in overrides:
            if key in collisions:
                raise ValueError(
                    f"Cannot expand override {key!r}: a parent key is also overridden."
                )
    composed = compose_trainer_config(trainer_config, overrides, substitutions=substitutions)
    if fmt == "yaml_file":
        _validate_file_config_value(composed, position="trainer_config")
        return yaml.safe_dump(composed, sort_keys=True, allow_unicode=True)
    if fmt == "json_file":
        return dump_overrides_json(composed)
    return flatten_trainer_config(composed)


def trial_substitutions(
    trial_dir: Path, trial_id: int, phase: str, run_name: str
) -> dict[str, str]:
    """Return the literal runtime bindings used in base trainer-config strings.

    :param Path trial_dir: Trial directory, kept as literal text inside replacements.
    :param int trial_id: Optuna trial number.
    :param str phase: Phase name.
    :param str run_name: Attempt-specific trainer run name.
    :return dict[str, str]: The four runtime placeholder bindings.
    """
    return {
        "{trial_dir}": str(trial_dir),
        "{trial_id}": str(trial_id),
        "{phase}": phase,
        "{run_name}": run_name,
    }


def render_command(
    template: str,
    overrides: dict[str, Any],
    fmt: str,
    *,
    trial_dir: Path,
    trial_id: int,
    phase: str,
    run_name: str,
    trainer_config: dict[str, Any] | None = None,
    materialized_input_path: Path | None = None,
) -> str:
    """Render the trial command with shell-quoted path substitutions.

    Never write input files. A supplied ``materialized_input_path`` is reused exactly;
    otherwise compose input for preview or validation without filesystem writes.

    :param str template: The user's ``trial_command`` template with ``{...}`` placeholders.
        Supported keys: ``config_path``, ``overrides``, ``overrides_path``, ``trial_dir``,
        ``trial_id``, ``phase``, ``run_name``.
    :param dict[str, object] overrides: The composed overrides for this trial.
    :param str fmt: One of ``yaml_file``, ``argparse``, ``hydra``, or ``json_file``.
    :param Path trial_dir: Per-trial directory used for ``{trial_dir}``.
    :param int trial_id: Numeric trial number, used for ``{trial_id}``.
    :param str phase: Phase name, used for ``{phase}``.
    :param str run_name: Composite ``<experiment>-<phase>-<trial_id>-<attempt_id>`` identifier
        used for ``{run_name}``.
    :param dict[str, object] | None trainer_config: Base trainer configuration embedded in the
        PhaseSweep YAML. Every format composes the overrides into it; strings in it expand the
        runtime placeholders (``{trial_dir}``, ``{trial_id}``, ``{phase}``, ``{run_name}``).
    :param Path | None materialized_input_path: Exact generated input already written by the
        launch path. When supplied for a file mode, command rendering uses this path without
        serializing or rewriting the input.
    :raises ValueError: ``fmt`` is not one of the four supported formats, or a dotted override
        cannot be composed into ``trainer_config``.
    :raises TypeError: A configured value cannot be represented by the selected command or
        generated-file format.
    :return str: The fully rendered, shell-ready command string.
    """
    config_path = ""
    overrides_path = ""
    base = {} if trainer_config is None else trainer_config
    config_substitutions = trial_substitutions(trial_dir, trial_id, phase, run_name)
    if fmt in ("yaml_file", "json_file"):
        if materialized_input_path is None:
            compose_trainer_input(base, overrides, fmt, substitutions=config_substitutions)
            materialized_input_path = trial_dir / TRAINER_INPUT_FILENAMES[fmt]
        if fmt == "yaml_file":
            config_path = str(materialized_input_path)
        else:
            overrides_path = str(materialized_input_path)
        overrides_str = ""
    else:
        values = compose_trainer_input(base, overrides, fmt, substitutions=config_substitutions)
        assert isinstance(values, dict)
        overrides_str = format_cli_overrides(values, "argparse" if fmt == "argparse" else "hydra")

    return template.format(
        config_path=shlex.quote(config_path) if config_path else "",
        overrides_path=shlex.quote(overrides_path) if overrides_path else "",
        overrides=overrides_str,
        trial_dir=shlex.quote(str(trial_dir)),
        trial_id=str(trial_id),
        phase=shlex.quote(phase),
        run_name=shlex.quote(run_name),
    )
