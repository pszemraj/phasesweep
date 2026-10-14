"""Strict YAML loading for phasesweep configs."""

from __future__ import annotations

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
    """Raised when a config file cannot be read or parsed into a mapping.

    The message always names the source, so the CLI reports it as one line and
    exits without a traceback. Subclassing ``ValueError`` keeps every existing
    ``except ValueError`` caller behaving as before.
    """


class _StrictMappingLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` subclass that rejects duplicate mapping keys.

    Default ``yaml.safe_load`` silently keeps the last value for a duplicated
    key. For experiment specs that's a footgun: a YAML like::

        search_space:
          lr: {type: float, low: 1e-5, high: 1e-3, log: true}
          lr: {type: float, low: 1e-4, high: 1e-2, log: true}  # silently wins

    would run a sweep against the *second* range with no warning. We override
    the default mapping constructor to raise on collisions, mirroring how
    ``Phase``'s collision validators behave for cross-phase keys. Strings that
    are not encodable as UTF-8 are rejected too.
    """


def _construct_mapping_strict(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    """Construct a YAML mapping while rejecting duplicate keys.

    Used as :class:`_StrictMappingLoader`'s mapping constructor: replaces the
    default constructor (which silently keeps the last value for duplicate
    keys) with one that raises ``ConstructorError`` so misspelled or
    copy-pasted duplicate keys surface at load time rather than after the
    sweep finishes.

    Args:
        loader: The active YAML loader; used to construct nested objects.
        node: The mapping node to construct.
        deep: Whether to construct nested objects deeply (PyYAML contract).

    Returns:
        The constructed mapping.

    Raises:
        yaml.constructor.ConstructorError: A key is unhashable or duplicated.

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
    r"""Construct a YAML string while rejecting text that is not encodable as UTF-8.

    The config bytes are strict UTF-8, but a double-quoted ``"\ud800"`` escape
    still yields a lone surrogate. Such a string loads, then fails wherever it
    is encoded: an ``env`` value at trainer launch, any field when the config
    snapshot is written.

    Args:
        loader: The active YAML loader.
        node: The scalar node to construct.

    Returns:
        The constructed string.

    Raises:
        yaml.constructor.ConstructorError: The scalar does not construct a string,
            or the string contains a lone surrogate.

    """
    value = yaml.constructor.SafeConstructor.construct_yaml_str(loader, node)
    if not isinstance(value, str):
        raise yaml.constructor.ConstructorError(
            None, None, f"expected a string, found {type(value).__name__}", node.start_mark
        )
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


def _load_yaml_mapping_from_text(text: str, source: str | Path) -> dict[str, Any]:
    """Load YAML text as a strict mapping.

    :param str text: YAML text to parse.
    :param str | Path source: Human-readable source label for errors.
    :raises ConfigError: If parsing fails or the top level is not a mapping.
    :return dict[str, Any]: Parsed top-level YAML mapping.
    """
    try:
        data = yaml.load(text, Loader=_StrictMappingLoader)  # noqa: S506 — strict SafeLoader subclass
    except yaml.YAMLError as exc:
        # Covers the strict loader's duplicate-key ConstructorError *and* the
        # scanner/parser errors PyYAML raises for the two most common YAML
        # mistakes (bad indentation, unclosed bracket). PyYAML labels those
        # marks ``in "<unicode string>"``, so without this wrap the operator
        # never learns which file failed.
        raise ConfigError(f"{source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping.")
    return data


def describe_validation_error(exc: ValidationError) -> str:
    """Render a validation error followed by pointers into the configuration reference.

    Each error location maps to the deepest documented key it points into. The first
    distinct keys, in error order, become ``see: phasesweep explain <key>`` lines, quoted
    for a shell because reference paths contain ``[]`` and ``<>``. Errors without a
    location, such as the experiment-level validators', and locations the reference does
    not document add no line.

    :param ValidationError exc: Validation failure of an :class:`Experiment`.
    :return str: ``str(exc)`` plus up to three hint lines.
    """
    paths: list[str] = []
    for error in exc.errors(include_input=False):
        path = loc_to_path(error["loc"])
        if path is not None and path not in paths:
            paths.append(path)
    hints = [f"see: phasesweep explain {shlex.quote(path)}" for path in paths[:_MAX_HINTS]]
    return "\n".join([str(exc), *hints])


def load_experiment_bytes(
    data: bytes, source: str | Path = "<bytes>", *, config_dir: Path | None = None
) -> Experiment:
    """Parse and validate a config from an already-read byte snapshot.

    Args:
        data: UTF-8 YAML bytes.
        source: Human-readable source label used in validation errors.
        config_dir: Absolute directory of the config file the bytes came from.
            Relative ``workdir``, journal ``storage`` and ``execution.cwd``
            paths resolve against it. ``None`` leaves them relative to the
            process working directory.

    Returns:
        :class:`Experiment` parsed from exactly ``data``.

    Raises:
        ValueError: ``config_dir`` is not absolute.
        ConfigError: ``data`` is not UTF-8 text, the YAML cannot be parsed (including
            duplicate mapping keys rejected by the strict loader), or the top level
            is not a mapping.
        pydantic.ValidationError: The parsed mapping fails :class:`Experiment`
            model validation.

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
    """Parse and validate a single experiment YAML.

    Uses a strict loader that rejects duplicate mapping keys. PyYAML's default
    ``safe_load`` silently keeps the last value for a duplicate key, which can
    cause an experiment to silently use the wrong search range or fixed
    override.

    Relative ``workdir``, journal ``storage`` and ``execution.cwd`` paths
    resolve against the directory of the file after symlinks are resolved, so
    the same file names the same artifact tree from any working directory.

    Args:
        path: Filesystem path to the experiment YAML file.

    Returns:
        A fully-validated :class:`Experiment` instance with all Pydantic and
        cross-phase consistency checks applied.

    Raises:
        OSError: ``path`` cannot be read.
        ConfigError: YAML parse error, top-level is not a mapping, or duplicate
            mapping keys.
        pydantic.ValidationError: Any Pydantic / cross-phase validation failure.

    """
    path_obj = Path(path)
    return load_experiment_bytes(
        path_obj.read_bytes(), source=path_obj, config_dir=path_obj.resolve().parent
    )
