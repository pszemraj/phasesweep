"""The packaged fake trainer's optional seed noise."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phasesweep.examples import fake_train

#: The loss the formula gives the default configuration, with no seed.
UNSEEDED_LOSS = 0.30


def _run_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict[str, object], *tokens: str
) -> float:
    """Run the trainer's ``main`` in process and return the loss it reports.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces ``sys.argv`` and the reporter.
    :param Path tmp_path: Directory for the generated config file.
    :param dict[str, object] config: Trainer config written as the ``--config`` YAML.
    :param str tokens: Extra command-line tokens appended after ``--config``.
    :return float: The objective value passed to ``report_objective``.
    """
    config_path = tmp_path / "trainer.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    reported: list[float] = []

    def capture(value: float, **_kwargs: object) -> Path:
        reported.append(value)
        return tmp_path / "result.json"

    monkeypatch.setattr(fake_train, "report_objective", capture)
    monkeypatch.setattr("sys.argv", ["fake_train", "--config", str(config_path), *tokens])
    fake_train.main()
    (value,) = reported
    return value


def test_no_seed_leaves_the_loss_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _run_trainer(monkeypatch, tmp_path, {}) == pytest.approx(UNSEEDED_LOSS)


def test_seed_shifts_the_loss_deterministically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _run_trainer(monkeypatch, tmp_path, {"seed": 1})
    again = _run_trainer(monkeypatch, tmp_path, {"seed": 1})
    other = _run_trainer(monkeypatch, tmp_path, {"seed": 2})

    assert first == again
    assert first != pytest.approx(UNSEEDED_LOSS)
    assert other != first
    assert abs(first - UNSEEDED_LOSS) < 6 * fake_train.SEED_NOISE_SCALE


def test_one_seed_gives_different_configurations_different_noise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    slow = {"optimizer": {"lr": 0.0002}}
    fast = {"optimizer": {"lr": 0.0004}}
    formula_gap = _run_trainer(monkeypatch, tmp_path, slow) - _run_trainer(
        monkeypatch, tmp_path, fast
    )

    seeded_gap = _run_trainer(monkeypatch, tmp_path, {**slow, "seed": 1}) - _run_trainer(
        monkeypatch, tmp_path, {**fast, "seed": 1}
    )

    assert seeded_gap != pytest.approx(formula_gap)


@pytest.mark.parametrize("token", ["--seed=1", "seed=1"])
def test_seed_token_matches_a_seed_in_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token: str
) -> None:
    from_config = _run_trainer(monkeypatch, tmp_path, {"seed": 1})

    assert _run_trainer(monkeypatch, tmp_path, {}, token) == from_config
