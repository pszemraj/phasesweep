"""The documented checkpoint continuation preserves the trainer's additional-step contract."""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.conftest import REPO


def test_decoder_checkpoint_recipe_uses_additional_steps_and_upstream_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the recipe while replacing checkpoint loading and the training subprocess."""
    guide = REPO / "examples/tiny_decoder_enwik8/README.md"
    section = re.search(
        r"```bash\npython - [^\n]+ <<'PY'\n(.*?)\nPY\n```",
        guide.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert section is not None, "The example needs an executable checkpoint continuation recipe"
    checkpoint = tmp_path / "source" / "trainer" / "final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint fixture")
    output = tmp_path / "continued"
    original_config = {
        "num_batches": 80,
        "run_dir": str(checkpoint.parent),
        "data_path": "data/enwik8.gz",
    }
    observed: list[list[str]] = []

    def load(path: Path, *, map_location: str, weights_only: bool) -> dict[str, object]:
        assert path == checkpoint
        assert map_location == "cpu"
        assert weights_only is True
        return {"step": 80, "config": original_config}

    def run(command: list[str], *, cwd: Path, check: bool) -> None:
        assert cwd == REPO / "examples/tiny_decoder_enwik8/upstream"
        assert check is True
        observed.append(command)

    monkeypatch.chdir(REPO)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))
    monkeypatch.setattr(sys, "argv", ["-", str(checkpoint), str(output)])
    monkeypatch.setattr(subprocess, "run", run)
    exec(compile(section.group(1), str(guide), "exec"), {"__name__": "__main__"})

    config_path = output / "trainer.yaml"
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert configuration["num_batches"] == 5
    assert 80 + configuration["num_batches"] == 85
    assert configuration["run_dir"] == str(output / "trainer")
    assert configuration["data_path"] == "data/enwik8.gz"
    assert original_config["num_batches"] == 80
    assert observed == [
        [
            sys.executable,
            str(REPO / "examples/tiny_decoder_enwik8/upstream/train.py"),
            "--config",
            str(config_path),
            "--resume",
            str(checkpoint),
        ]
    ]
