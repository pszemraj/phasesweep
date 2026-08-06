"""Strict YAML loading for phasesweep configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from phasesweep.config.models import Config, Experiment, Suite


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
    ``Phase``'s collision validators behave for cross-phase keys.
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


def load_config_bytes(data: bytes, source: str | Path = "<bytes>") -> Config:
    """Parse and validate a config from an already-read byte snapshot.

    Args:
        data: UTF-8 YAML bytes.
        source: Human-readable source label used in validation errors.

    Returns:
        :class:`Experiment` or :class:`Suite` parsed from exactly ``data``.

    Raises:
        ConfigError: ``data`` is not UTF-8 text, the YAML cannot be parsed (including
            duplicate mapping keys rejected by the strict loader), or the top level
            is not a mapping.
        pydantic.ValidationError: The parsed mapping fails :class:`Suite` or
            :class:`Experiment` model validation.

    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{source}: config must be UTF-8 text: {exc}") from exc
    parsed = _load_yaml_mapping_from_text(text, source)
    if "suite" in parsed:
        return Suite.model_validate(parsed)
    return Experiment.model_validate(parsed)


def load_config(path: str | Path) -> Config:
    """Parse and validate either a single experiment YAML or a suite YAML.

    Args:
        path: Filesystem path to a phasesweep YAML file.

    Returns:
        :class:`Experiment` for legacy/current single-study configs, or
        :class:`Suite` for configs with a top-level ``suite`` key.

    Raises:
        OSError: ``path`` cannot be read.
        ConfigError: The file is not UTF-8 text, the YAML cannot be parsed
            (including duplicate mapping keys), or the top level is not a mapping.
        pydantic.ValidationError: The parsed mapping fails :class:`Suite` or
            :class:`Experiment` model validation.

    """
    path_obj = Path(path)
    return load_config_bytes(path_obj.read_bytes(), source=path_obj)


def load_experiment(path: str | Path) -> Experiment:
    """Parse and validate a single experiment YAML.

    Uses a strict loader that rejects duplicate mapping keys. PyYAML's default
    ``safe_load`` silently keeps the last value for a duplicate key, which can
    cause an experiment to silently use the wrong search range or fixed
    override (review v0.5.6 / non-blocking hardening item).

    Args:
        path: Filesystem path to the experiment YAML file.

    Returns:
        A fully-validated :class:`Experiment` instance with all Pydantic and
        cross-phase consistency checks applied.

    Raises:
        OSError: ``path`` cannot be read.
        ConfigError: YAML parse error, top-level is not a mapping, duplicate
            mapping keys, or the file is a suite config rather than a single
            experiment.
        pydantic.ValidationError: Any Pydantic / cross-phase validation failure.

    """
    config = load_config(path)
    if isinstance(config, Suite):
        raise ConfigError(f"{path}: expected a single experiment config, got a suite config.")
    return config
