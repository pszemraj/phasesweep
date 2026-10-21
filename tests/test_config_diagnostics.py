"""Configuration errors identify their source and the supported replacement."""

from pathlib import Path

import pytest

from phasesweep import load_experiment
from phasesweep.config.io import ConfigError
from tests.conftest import invoke_cli_boundary
from tests.experiments import phase_payload, write_experiment


def test_removed_seed_search_key_points_to_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The removed option remains refused and names the seed-audit replacement."""
    path = write_experiment(
        tmp_path / "removed.yaml", phases=[phase_payload(allow_seed_search=True)]
    )

    assert invoke_cli_boundary(["validate", str(path)], monkeypatch) == 2
    error = capsys.readouterr().err
    assert "Extra inputs are not permitted" in error
    assert "allow_seed_search" in error
    assert "repeats" in error
    assert "search_space: {}" in error


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
