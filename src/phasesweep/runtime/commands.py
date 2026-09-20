"""Format override dictionaries into the shape the trial command expects.

All values are unconditionally shell-quoted via shlex.quote to prevent
injection or misparse from shell metacharacters in paths or categorical values.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml

_CliOverrideFormat = Literal["argparse", "hydra"]


class _OverrideValueError(TypeError):
    """A value outside the scalar/list CLI override contract."""

    def __init__(self, fmt: _CliOverrideFormat, value: Any, position: str) -> None:
        """Describe the exact nested value the selected CLI wire cannot render.

        :param _CliOverrideFormat fmt: Selected CLI override grammar.
        :param Any value: Unsupported value encountered while rendering.
        :param str position: Nested list position used in the diagnostic.
        """
        self.value = value
        self.position = position
        where = f" at position {position}" if position else ""
        super().__init__(
            f"override_format={fmt!r} supports null, booleans, integers, finite "
            f"floats, strings, and lists of those; got{where} "
            f"{type(value).__name__}: {value!r}. Use the default "
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
        # Literal interpolation syntax is deliberately outside this wire contract.
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


def format_hydra(overrides: dict[str, Any]) -> str:
    """Render native Hydra arguments without importing the trainer's SDK.

    :param dict[str, Any] overrides: Scalar/list values keyed by dotted parameter names.
    :return str: Shell-quoted native ``key=value`` tokens.
    :raises TypeError: A literal has no faithful supported Hydra representation.
    """
    return " ".join(
        shlex.quote(f"{key}={_render_override_value(value, 'hydra')}")
        for key, value in overrides.items()
    )


def dump_overrides_json(payload: Any) -> str:
    """Encode only JSON-compatible values using the trainer's strict wire format.

    :param Any payload: JSON-compatible scalar, list, or string-keyed mapping.
    :return str: Deterministic JSON text.
    :raises TypeError: A value is outside the JSON data model.
    :raises ValueError: A value is non-finite or recursive.
    """
    _validate_trainer_config_value(payload, position="JSON overrides")
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def dump_json_file_overrides(overrides: dict[str, Any]) -> str:
    """Expand dotted keys and serialize the overrides-only trainer input.

    :param dict[str, Any] overrides: Resolved overrides, without a base configuration.
    :return str: Exact text to write to ``overrides.json``.
    :raises ValueError: A key collides with another key's namespace.
    :raises TypeError: A value cannot be represented as JSON.
    """
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        parts = key.split(".")
        if any(".".join(parts[:index]) in overrides for index in range(1, len(parts))):
            raise ValueError(f"Cannot expand override {key!r}: a parent key is also overridden.")
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return dump_overrides_json(nested)


def write_json_file(overrides: dict[str, Any], trial_dir: Path) -> Path:
    """Write the nested overrides-only JSON consumed by the trainer.

    :param dict[str, Any] overrides: Resolved dotted overrides.
    :param Path trial_dir: Attempt directory.
    :return Path: Written trainer input.
    """
    path = trial_dir / "overrides.json"
    path.write_text(dump_json_file_overrides(overrides), encoding="utf-8")
    return path


def format_argparse(overrides: dict[str, Any]) -> str:
    """Render argparse options as shell-quoted ``--key=value`` tokens.

    Args:
        overrides: Mapping from override key to value.

    Returns:
        A single space-separated string of shell-quoted ``--key=value`` tokens.

    Raises:
        TypeError: A value is outside the shared CLI override contract (see
            :func:`_render_override_value`). Statically-known override values
            are already rejected at config load; this covers anything else.

    """
    parts: list[str] = []
    for k, v in overrides.items():
        parts.append(shlex.quote(f"--{k}={_render_override_value(v, 'argparse')}"))
    return " ".join(parts)


def compose_trainer_config(
    trainer_config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Apply dotted PhaseSweep overrides to a complete trainer configuration.

    A dotted key descends through mappings, creating missing mappings as needed.
    The final segment replaces the base value at that exact path, including a
    complete mapping or list. Descending through an existing non-mapping is an
    error because there is no unambiguous YAML result.

    :param dict[str, Any] trainer_config: Operator-authored base trainer config.
    :param dict[str, Any] overrides: Composed inherited, fixed, and sampled values.
    :raises ValueError: An override path must descend through a non-mapping value.
    :return dict[str, Any]: Deep-copied complete configuration for one trial.
    """
    composed = deepcopy(trainer_config)
    for key, value in overrides.items():
        current = composed
        parts = key.split(".")
        for index, part in enumerate(parts[:-1]):
            existing = current.get(part)
            if existing is None and part not in current:
                child: dict[str, Any] = {}
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


def _validate_trainer_config_value(
    value: Any,
    *,
    position: str = "trainer_config",
    _ancestors: set[int] | None = None,
) -> None:
    """Require deterministic, portable values in a generated trainer YAML.

    :param Any value: Value to validate recursively.
    :param str position: Human-readable path used in diagnostics.
    :param set[int] | None _ancestors: Internal recursive-container guard.
    :raises TypeError: A mapping key is not a string or a value is outside the
        normal YAML configuration scalar/list/mapping domain.
    :raises ValueError: A float is non-finite or a container is recursive.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{position} must be finite; got {value!r}.")

    if not isinstance(value, (dict, list)):
        raise TypeError(
            f"{position} has unsupported type {type(value).__name__}: {value!r}. "
            "Use strings, booleans, integers, finite floats, null, lists, and "
            "string-keyed mappings; quote YAML dates or timestamps to keep them strings."
        )

    ancestors = set() if _ancestors is None else _ancestors
    if id(value) in ancestors:
        raise ValueError(f"{position} contains a recursive YAML container.")
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"{position} has non-string mapping key {key!r} "
                        f"({type(key).__name__}). Trainer config keys must be strings."
                    )
                child_position = f"{position}.{key}" if position else key
                _validate_trainer_config_value(
                    item,
                    position=child_position,
                    _ancestors=ancestors,
                )
        else:
            for index, item in enumerate(value):
                _validate_trainer_config_value(
                    item,
                    position=f"{position}[{index}]",
                    _ancestors=ancestors,
                )
    finally:
        ancestors.remove(id(value))


def dump_trainer_config_yaml(payload: dict[str, Any]) -> str:
    """Serialize one complete trainer config to deterministic safe YAML.

    :param dict[str, Any] payload: Fully composed trainer configuration.
    :raises TypeError: The payload contains an unsupported value or mapping key.
    :raises ValueError: The payload contains a non-finite float or recursive container.
    :return str: Canonical YAML text with keys sorted at every mapping level.
    """
    _validate_trainer_config_value(payload)
    return yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)


def dump_trial_trainer_config_yaml(
    trainer_config: dict[str, Any],
    overrides: dict[str, Any],
    *,
    substitutions: dict[str, str] | None = None,
) -> str:
    """Serialize the complete trainer YAML consumed by one trial.

    :param dict[str, Any] trainer_config: Operator-authored base trainer config.
    :param dict[str, Any] overrides: Composed inherited, fixed, and sampled values.
    :param dict[str, str] | None substitutions: Runtime placeholders expanded
        recursively in base-config string values before overrides are applied.
    :raises TypeError: The complete config contains an unsupported YAML value.
    :raises ValueError: Override composition or YAML validation fails.
    :return str: Exact UTF-8 text for ``trainer_config.yaml``.
    """
    base = (
        trainer_config
        if not substitutions
        else _substitute_trainer_config_placeholders(trainer_config, substitutions)
    )
    return dump_trainer_config_yaml(compose_trainer_config(base, overrides))


def write_trainer_config_yaml(
    trainer_config: dict[str, Any],
    overrides: dict[str, Any],
    trial_dir: Path,
    *,
    substitutions: dict[str, str] | None = None,
) -> Path:
    """Materialize the complete trainer YAML consumed by one trial.

    :param dict[str, Any] trainer_config: Operator-authored base trainer config.
    :param dict[str, Any] overrides: Composed inherited, fixed, and sampled values.
    :param Path trial_dir: Per-trial directory receiving ``trainer_config.yaml``.
    :param dict[str, str] | None substitutions: Runtime placeholders expanded
        recursively in base-config string values before overrides are applied.
    :raises TypeError: The complete config contains an unsupported YAML value.
    :raises ValueError: Override composition or YAML validation fails.
    :return Path: Path to the generated complete trainer config.
    """
    path = trial_dir / "trainer_config.yaml"
    path.write_text(
        dump_trial_trainer_config_yaml(
            trainer_config,
            overrides,
            substitutions=substitutions,
        ),
        encoding="utf-8",
    )
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
    trainer_config: dict[str, Any] | None = None,
    write_files: bool = True,
    materialized_input_path: Path | None = None,
) -> str:
    """Substitute placeholders in the user's trial_command template.

    Path substitutions (``{trial_dir}``, ``{config_path}``, ``{overrides_path}``)
    are shell-quoted.

    Args:
        template: The user's ``trial_command`` template with ``{...}``
            placeholders. Supported keys: ``config_path``, ``overrides``, ``overrides_path``,
            ``trial_dir``, ``trial_id``, ``phase``, ``run_name``.
        overrides: The composed overrides for this trial.
        fmt: One of ``yaml_file``, ``argparse``, ``hydra``, or ``json_file``.
        trial_dir: Per-trial directory used for ``{trial_dir}``.
        trial_id: Numeric trial number, used for ``{trial_id}``.
        phase: Phase name, used for ``{phase}``.
        run_name: Composite ``<experiment>-<phase>-<trial_id>-<attempt_id>``
            identifier used for ``{run_name}``.
        trainer_config: Base trainer configuration embedded in the PhaseSweep
            YAML. Used only by ``yaml_file``.
        write_files: When ``False``, render paths without writing
            ``trainer_config.yaml``. Used by dry-run previews so they are
            filesystem-pure.
        materialized_input_path: Exact generated input already written by the
            launch path. When supplied for a file mode, command rendering uses
            this path without serializing or rewriting the input.

    Returns:
        The fully rendered, shell-ready command string.

    Raises:
        ValueError: If ``fmt`` is not one of the four supported formats, or a
            dotted override cannot be composed into ``trainer_config``.
        TypeError: A configured value cannot be represented by the selected
            command or generated-file format.
        OSError: ``write_files`` is true and a generated trainer input cannot
            be written.

    """
    config_path = ""
    overrides_path = ""
    if fmt == "yaml_file":
        base = {} if trainer_config is None else trainer_config
        config_substitutions = {
            "{trial_dir}": str(trial_dir),
            "{trial_id}": str(trial_id),
            "{phase}": phase,
            "{run_name}": run_name,
        }
        if materialized_input_path is not None:
            config_path = str(materialized_input_path)
        elif write_files:
            config_path = str(
                write_trainer_config_yaml(
                    base,
                    overrides,
                    trial_dir,
                    substitutions=config_substitutions,
                )
            )
        else:
            # Dry-run and config validation remain filesystem-pure while still
            # exercising the exact composition and serializer used at launch.
            expanded_base = _substitute_trainer_config_placeholders(base, config_substitutions)
            payload = compose_trainer_config(expanded_base, overrides)
            dump_trainer_config_yaml(payload)
            config_path = str(trial_dir / "trainer_config.yaml")
        overrides_str = ""
    elif fmt == "argparse":
        overrides_str = format_argparse(overrides)
    elif fmt == "hydra":
        overrides_str = format_hydra(overrides)
    elif fmt == "json_file":
        if materialized_input_path is not None:
            overrides_path = str(materialized_input_path)
        elif write_files:
            overrides_path = str(write_json_file(overrides, trial_dir))
        else:
            dump_json_file_overrides(overrides)
            overrides_path = str(trial_dir / "overrides.json")
        overrides_str = ""
    else:
        raise ValueError(f"Unknown override_format: {fmt}")

    return template.format(
        config_path=shlex.quote(config_path) if config_path else "",
        overrides_path=shlex.quote(overrides_path) if overrides_path else "",
        overrides=overrides_str,
        trial_dir=shlex.quote(str(trial_dir)),
        trial_id=str(trial_id),
        phase=shlex.quote(phase),
        run_name=shlex.quote(run_name),
    )
