"""Strict YAML loading for phasesweep configs."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from phasesweep.config.common import CONFIG_DIR_CONTEXT
from phasesweep.config.models import Experiment
from phasesweep.config.reference import loc_to_path

# Hint lines after a validation error; the first errors are the likeliest root causes.
_MAX_HINTS = 3


class ConfigError(ValueError):
    """A source-labelled failure to read or parse experiment configuration."""


class _StrictMappingLoader(yaml.SafeLoader):
    """Load safe YAML while rejecting duplicate keys and non-UTF-8 strings."""


def _construct_mapping_strict(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    """Construct a YAML mapping, rejecting unhashable or duplicate keys.

    :param yaml.SafeLoader loader: Active loader for nested values.
    :param yaml.MappingNode node: Mapping node to construct.
    :param bool deep: Whether to construct nested objects deeply.
    :raises yaml.constructor.ConstructorError: A key is unhashable or duplicated.
    :return dict: Constructed mapping.
    """
    literal_keys: dict[Any, None] = {}
    merge_key_seen = False
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            if merge_key_seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key '<<'",
                    key_node.start_mark,
                )
            merge_key_seen = True
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key: {exc}",
                key_node.start_mark,
            ) from None
        if key in literal_keys:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        literal_keys[key] = None

    return yaml.constructor.SafeConstructor.construct_mapping(loader, node, deep=deep)


_StrictMappingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_strict,
)


def _construct_str_strict(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    """Construct a YAML string that can be encoded as UTF-8.

    :param yaml.SafeLoader loader: Active YAML loader.
    :param yaml.ScalarNode node: String node to construct.
    :raises yaml.constructor.ConstructorError: The string contains a lone surrogate.
    :return str: Constructed UTF-8-compatible string.
    """
    value = yaml.constructor.SafeConstructor.construct_scalar(loader, node)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise yaml.constructor.ConstructorError(
            None,
            None,
            "found a string with a lone surrogate escape, which is not valid UTF-8 text",
            node.start_mark,
        ) from None
    return value


_StrictMappingLoader.add_constructor("tag:yaml.org,2002:str", _construct_str_strict)


def load_yaml_document(text: str) -> object:
    """Parse one safe YAML document with unique keys and UTF-8-compatible strings.

    :param str text: YAML document to parse.
    :raises yaml.YAMLError: Invalid YAML, duplicate keys, or non-UTF-8 strings.
    :return object: Parsed YAML value.
    """
    value: object = yaml.load(text, Loader=_StrictMappingLoader)
    return value


def _load_yaml_mapping_from_text(text: str, source: str | Path) -> dict[str, Any]:
    """Load YAML text as a strict mapping.

    :param str text: YAML text to parse.
    :param str | Path source: Human-readable source label for errors.
    :raises ConfigError: If parsing fails or the top level is not a mapping.
    :return dict[str, Any]: Parsed top-level YAML mapping.
    """
    try:
        data = load_yaml_document(text)
    except yaml.YAMLError as exc:
        if isinstance(exc, yaml.MarkedYAMLError):
            for mark in (exc.context_mark, exc.problem_mark):
                if mark is not None:
                    mark.name = str(source)
        hint = (
            " Quote text values containing ': ' with single or double quotes."
            if isinstance(exc, yaml.MarkedYAMLError)
            and exc.problem == "mapping values are not allowed here"
            else ""
        )
        raise ConfigError(f"{source}: {exc}{hint}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping.")
    return data


def format_validation_error(exc: ValidationError) -> str:
    """Render validation locations and messages without inputs or library help URLs.

    :param ValidationError exc: Model validation failure.
    :return str: One concise line per invalid field or model-level condition.
    """
    lines: list[str] = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"{location}: {message}" if location else message)
    return "\n".join(lines)


def describe_validation_error(exc: ValidationError) -> str:
    """Append up to three configuration-reference hints to a validation error.

    Hints name the deepest documented key for each distinct error location,
    in error order, and are shell-quoted. Undocumented locations add no hint.

    :param ValidationError exc: Validation failure of an :class:`Experiment`.
    :return str: Concise validation messages plus up to three hint lines.
    """
    paths: list[str] = []
    for error in exc.errors(include_input=False):
        path = loc_to_path(error["loc"])
        if path is not None and path not in paths:
            paths.append(path)
    hints = [f"see: phasesweep explain {shlex.quote(path)}" for path in paths[:_MAX_HINTS]]
    return "\n".join([format_validation_error(exc), *hints])


def load_experiment_bytes(
    data: bytes, source: str | Path = "<bytes>", *, config_dir: Path | None = None
) -> Experiment:
    """Parse and validate an already-read UTF-8 YAML snapshot.

    :param bytes data: Exact configuration bytes.
    :param str | Path source: Source label included in errors.
    :param Path | None config_dir: Absolute source directory for resolving relative
        workdir, storage, and execution.cwd paths; None keeps working-directory semantics.
    :raises ValueError: The supplied config directory is relative.
    :raises ConfigError: Invalid UTF-8, malformed YAML, duplicate keys, or a non-mapping root.
    :raises pydantic.ValidationError: Experiment model or cross-phase validation fails.
    :return Experiment: Validated experiment from exactly these bytes.
    """
    if config_dir is not None and not config_dir.is_absolute():
        raise ValueError(f"config_dir must be absolute; got {str(config_dir)!r}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{source}: config must be UTF-8 text: {exc}") from exc
    parsed = _load_yaml_mapping_from_text(text, source)
    context = None if config_dir is None else {CONFIG_DIR_CONTEXT: config_dir}
    return Experiment.model_validate(parsed, context=context)


def load_experiment(path: str | Path) -> Experiment:
    """Read and validate experiment YAML, rejecting duplicate keys.

    Relative workdir, journal storage, and execution.cwd paths resolve against
    the source file's directory after symlink resolution.

    :param str | Path path: Experiment YAML file.
    :raises OSError: The file cannot be read.
    :raises ConfigError: Invalid YAML, duplicate keys, or a non-mapping root.
    :raises pydantic.ValidationError: Model or cross-phase validation fails.
    :return Experiment: Validated experiment.
    """
    path_obj = Path(path)
    return load_experiment_bytes(
        path_obj.read_bytes(), source=path_obj, config_dir=path_obj.resolve().parent
    )


def load_experiment_snapshot(
    path: Path, expected_sha256: str, *, source: str, config_dir: Path
) -> Experiment:
    """Read, verify, and parse one immutable experiment snapshot.

    :param Path path: Snapshot file to read.
    :param str expected_sha256: Digest recorded when the configuration was frozen.
    :param str source: Human-readable parser source label.
    :param Path config_dir: Absolute source directory for resolving relative config paths.
    :raises OSError: The snapshot cannot be read.
    :raises ValueError: Its digest or syntax is invalid, or ``config_dir`` is not absolute.
    :return Experiment: Verified experiment configuration from the hashed bytes.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("run snapshot hash mismatch")
    return load_experiment_bytes(data, source=source, config_dir=config_dir)
