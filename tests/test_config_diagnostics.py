"""YAML parse errors identify their source file and explain how to fix them."""

from pathlib import Path

import pytest

from phasesweep import load_experiment
from phasesweep.config.io import ConfigError


def test_yaml_error_names_the_file_and_explains_colon_text(tmp_path: Path) -> None:
    """A malformed free-text comment shows the real source and quoting advice."""
    path = tmp_path / "broken.yaml"
    path.write_text(
        "experiment: demo\nphases:\n  - name: p\n    comment: text: with colon\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError) as caught:
        load_experiment(path)

    message = str(caught.value)
    assert str(path) in message
    assert "<unicode string>" not in message
    assert "line 4" in message
    assert "quote" in message.lower()
