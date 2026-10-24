"""Derive a run from one config without editing or copying that config."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import phasesweep.cli as cli_module
from phasesweep import run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.run_overrides import SetPathError, apply_set
from phasesweep.engine.errors import StudyFingerprintMismatchError, StudyStorageUnavailableError
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_config_snapshot_path,
    _generation_dir,
    _generation_path,
    _last_successful_generation_path,
)
from phasesweep.engine.read import experiment_status, read_result
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import Winner
from phasesweep.runtime.files import file_sha256
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    write_experiment_config,
    write_trainer,
)
from tests.experiments import phase_payload, write_experiment
from tests.ledger_fixtures import materialize, published_generation_id


@pytest.mark.parametrize(
    ("setting", "problem"),
    [
        ("trainer_config={lr: 1, lr: 2}", "duplicate key 'lr'"),
        ('trainer_config="\\udcff"', "lone surrogate escape.*not valid UTF-8"),
        ("trainer_config=\udcff", "unacceptable character #xdcff"),
    ],
)
def test_set_values_follow_strict_config_yaml_rules(setting: str, problem: str) -> None:
    """Reject duplicate mapping keys and strings that cannot represent UTF-8 text."""
    with pytest.raises(SetPathError, match=problem) as caught:
        apply_set({}, setting)

    assert "'trainer_config': the value is not valid YAML:" in str(caught.value)


def test_run_overrides_are_typed_validated_and_leave_yaml_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated dotted assignments update phase lists and top-level fields in memory."""
    path = write_experiment(
        tmp_path / "sweep.yaml",
        storage="auto",
        workdir="original",
        execution={"cwd": "."},
        phases=[phase_payload(n_trials=0)],
    )
    original = path.read_bytes()
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)

    experiment = _derived(
        path,
        monkeypatch,
        "workdir=derived",
        "experiment=smoke",
        "phases.0.n_trials=2",
        "phases.0.fixed_overrides={seed: 9, enabled: true}",
        "env.NOTE='a=b'",
    )
    assert experiment.experiment == "smoke"
    assert Path(experiment.workdir) == tmp_path / "derived"
    assert experiment.execution.cwd == str(tmp_path)
    assert experiment.phases[0].n_trials == 2
    assert experiment.phases[0].fixed_overrides == {"seed": 9, "enabled": True}
    assert experiment.env == {"NOTE": "a=b"}
    assert path.read_bytes() == original
    assert not (caller / "derived").exists()


def test_derived_run_writes_its_overrides_to_the_config_snapshot(tmp_path: Path) -> None:
    """A real one-trial run uses the derived namespace and records its effective config."""
    path = write_experiment(
        tmp_path / "sweep.yaml",
        storage="auto",
        workdir="original",
        trial_command="echo x=0.5 {overrides}",
        metric={"extractor": {"type": "log_regex", "pattern": r"x=(?P<value>[0-9.]+)"}},
    )
    original = path.read_bytes()
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(path),
            "--set",
            "workdir=derived",
            "--set",
            "experiment=smoke",
            "--set",
            "phases.0.gpu_policy=none",
        ],
    )

    assert result.exit_code == 0, result.output
    snapshots = list(
        (tmp_path / "derived" / "smoke" / "generations").glob("*/config.snapshot.yaml")
    )
    assert len(snapshots) == 1
    recorded = yaml.safe_load(snapshots[0].read_text(encoding="utf-8"))
    assert recorded["experiment"] == "smoke"
    assert recorded["workdir"] == str(tmp_path / "derived")
    assert path.read_bytes() == original
    assert not (tmp_path / "original").exists()

    for command in ("validate", "status", "show-winners"):
        inspected = CliRunner().invoke(
            cli, [command, str(path), "--set", "workdir=derived", "--set", "experiment=smoke"]
        )
        assert inspected.exit_code == 0, inspected.output
        assert "smoke" in inspected.output or "trial_number: 0" in inspected.output


@pytest.mark.parametrize("empty_directory", [False, True])
@pytest.mark.parametrize("command", ["status", "show-winners", "sync-tracking"])
def test_study_reads_name_the_missing_directory(
    tmp_path: Path,
    command: str,
    empty_directory: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Study reads refuse a missing output tree instead of inventing empty results."""
    path = write_experiment(tmp_path / "sweep.yaml", workdir="absent")
    if empty_directory:
        (tmp_path / "absent" / "t").mkdir(parents=True)
    assert invoke_cli_boundary([command, str(path)], monkeypatch) == 1
    message = capsys.readouterr().err
    assert "phasesweep: No study directory at" in message
    assert "fix:" in message
    assert str(tmp_path / "absent" / "t") in message
    with pytest.raises(StudyStorageUnavailableError, match="No study directory at"):
        experiment_status(cli_module._load_cli_config(path))


@pytest.mark.parametrize(
    ("command", "supports_override"),
    [
        (("run",), True),
        (("validate",), True),
        (("status",), True),
        (("show-winners",), True),
        (("sync-tracking",), True),
        (("init",), False),
        (("explain",), False),
        (("mcp", "serve"), False),
        (("mcp", "init-catalog"), False),
        ((), False),
    ],
)
def test_removed_workdir_option_points_to_config_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: tuple[str, ...],
    supports_override: bool,
) -> None:
    """Only commands accepting configuration overrides suggest the replacement spelling."""
    path = write_experiment(tmp_path / "sweep.yaml")
    arguments = [*command, *([str(path)] if supports_override else []), "--workdir", "runs"]
    assert invoke_cli_boundary(arguments, monkeypatch) == 2
    message = capsys.readouterr().err
    assert ("--set workdir=PATH" in message) is supports_override
    assert ("config file's directory" in message) is supports_override


@pytest.mark.integration
@pytest.mark.parametrize("continuation", ["topup", "append", "prepend_drift"])
def test_published_winner_survives_a_failed_later_generation(
    tmp_path: Path,
    continuation: str,
) -> None:
    """Failed continuations retain each winner's producing metric, annotations and trainer base."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os, sys
        if os.environ['PHASESWEEP_TRIAL_ID'] != '0' or '--fail=1' in sys.argv or '--fail' in sys.argv:
            sys.exit(1)
        print('x=0.5')
    """,
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        n_trials=1,
        trainer_config={"caption": "original-base"},
        comment="original annotation",
    )
    run_experiment(experiment)
    published_id = published_generation_id(experiment)
    assert published_id is not None
    if continuation == "append":
        phases = [
            *experiment.phases,
            Phase(
                name="next",
                n_trials=1,
                gpu_policy="none",
                max_consecutive_failures=1,
                fixed_overrides={"fail": 1},
            ),
        ]
    elif continuation == "prepend_drift":
        phases = [
            Phase(name="lead", n_trials=1, gpu_policy="none"),
            experiment.phases[0].model_copy(update={"comment": "changed annotation"}),
        ]
    else:
        phases = [
            experiment.phases[0].model_copy(
                update={
                    "n_trials": 2,
                    "max_consecutive_failures": 1,
                }
            )
        ]
    changed = experiment.model_copy(update={"phases": phases})
    if continuation == "prepend_drift":
        changed = changed.model_copy(
            update={
                "metric": experiment.metric.model_copy(update={"name": "changed_metric"}),
                "trainer_config": {"caption": "changed-base"},
            }
        )
        with pytest.raises(StudyFingerprintMismatchError, match="different phase config"):
            run_experiment(changed)
    else:
        with pytest.raises(NoFeasibleTrialError, match="aborted"):
            run_experiment(changed)
    path = tmp_path / "sweep.yaml"
    write_experiment_config(path, changed)
    winners = CliRunner().invoke(cli, ["show-winners", str(path)])
    assert winners.exit_code == 0, winners.exception
    assert "trial_number: 0" in winners.output
    status = experiment_status(changed)
    expected_generation = (
        status["current_generation_id"] if continuation == "append" else published_id
    )
    expected_label = "unpublished" if continuation == "append" else "published"
    assert f"# winner generation: {expected_generation} ({expected_label})" in winners.output
    phases_payload = status["phases"]
    assert isinstance(phases_payload, list) and isinstance(phases_payload[0], dict)
    phase_payload = next(phase for phase in phases_payload if phase["name"] == "p")
    assert phase_payload["winner_generation_id"] == expected_generation
    assert phase_payload["winner_is_published"] is (continuation != "append")
    assert str(expected_generation) in str(phase_payload["winner"])
    if continuation == "prepend_drift":
        lead, published = winners.output.split("=== lead ===", 1)[1].split("=== p ===", 1)
        assert "changed_metric: 0.5" in lead
        assert "objective: 0.5" in published and "changed_metric:" not in published
        assert "# original annotation" in published and "changed annotation" not in published
        assert (
            "# Historical published phase result: does not match the current config." in published
        )
        assert "# Historical experiment result: unpublished generation" not in winners.output
        for generation, base in (
            (published_id, "original-base"),
            (status["current_generation_id"], "changed-base"),
        ):
            assert (
                f"# Producing argparse trainer base for generation {generation}, before phase overrides.\n"
                f"trainer_config:\n  caption: {base}"
            ) in winners.output
            assert winners.output.count(f"caption: {base}") == 1
    table = CliRunner().invoke(cli, ["status", str(path), "--trials"])
    assert table.exit_code == 0, table.exception
    assert f"winner generation: {expected_generation} ({expected_label})" in table.output


def test_show_winners_distinguishes_appended_phases_from_historical_results(tmp_path: Path) -> None:
    """Appending a phase keeps finished phase results current and names the pending phase."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    path = materialized.config_path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["phases"].append({"name": "next", "inherits": ["p"], "n_trials": 1})
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = CliRunner().invoke(cli, ["show-winners", str(path)])
    assert result.exit_code == 0, result.output
    assert "appended phases have not run yet" in result.output
    assert "Historical experiment result" not in result.output
    assert "=== next === (no winner yet)" in result.output
    producing = cli_module._load_cli_config(path)
    result_status, (winner,) = read_result(producing)
    assert result_status["published_config_matches_current"] is True
    assert winner.config_matches_current is True
    _last_successful_generation_path(materialized.experiment).unlink()
    status = CliRunner().invoke(cli, ["status", str(path)])
    assert status.exit_code == 0, status.output
    payload = yaml.safe_load(status.output)
    assert payload["is_published"] is False
    assert payload["phases"][0]["winner_config_matches_current"] is True
    assert payload["phases"][0]["winner"].endswith("/phases/p/winner.yaml")
    table = CliRunner().invoke(cli, ["status", str(path), "--trials"])
    assert table.exit_code == 0, table.output
    assert "marks unpublished generation" in table.output


def test_unpublished_winner_discloses_semantic_config_drift(tmp_path: Path) -> None:
    """An unpublished saved winner reports drift in both CLI labels and status facts."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    _last_successful_generation_path(materialized.experiment).unlink()
    path = materialized.config_path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["metric"]["name"] = "changed_metric"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = CliRunner().invoke(cli, ["show-winners", str(path)])
    assert result.exit_code == 0, result.output
    assert "# Historical experiment result: unpublished generation" in result.output
    assert "does not match the current config" in result.output
    assert "objective: 0.5" in result.output
    inspected = CliRunner().invoke(cli, ["status", str(path)])
    assert inspected.exit_code == 0, inspected.output
    status = yaml.safe_load(inspected.output)
    assert status["phases"][0]["winner_config_matches_current"] is False
    comparison = cli_module._load_cli_config(path)
    result_status, (winner,) = read_result(
        comparison, generation_id=status["represented_generation_id"]
    )
    assert result_status["published_config_matches_current"] is False
    assert winner.config_matches_current is False


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "yaml",
        "model",
        "digest",
        "missing-record",
        "invalid-record",
        "missing-digest",
        "path",
    ],
)
@pytest.mark.parametrize("command", ["status", "show-winners"])
def test_unpublished_snapshot_damage_uses_the_artifact_error_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    damage: str,
) -> None:
    """Damaged producing snapshots or their digest anchors yield one restore-tree refusal."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    generation_id = published_generation_id(materialized.experiment)
    assert generation_id is not None
    _last_successful_generation_path(materialized.experiment).unlink()
    snapshot = _generation_config_snapshot_path(
        _experiment_dir(materialized.experiment), generation_id
    )
    reproducibility = snapshot.with_name("reproducibility.json")
    if damage == "missing":
        snapshot.unlink()
    elif damage in {"yaml", "model"}:
        snapshot.write_text(
            "broken: [" if damage == "yaml" else "experiment: invalid\n", encoding="utf-8"
        )
        record = json.loads(reproducibility.read_text())
        record["config_snapshot"]["sha256"] = file_sha256(snapshot)
        reproducibility.write_text(json.dumps(record))
    elif damage == "digest":
        config = yaml.safe_load(snapshot.read_text())
        config["metric"]["name"] = "altered metric"
        snapshot.write_text(yaml.safe_dump(config))
    elif damage == "missing-record":
        reproducibility.unlink()
    elif damage == "invalid-record":
        reproducibility.write_text("[invalid")
    else:
        record = json.loads(reproducibility.read_text())
        if damage == "missing-digest":
            del record["config_snapshot"]["sha256"]
        else:
            record["config_snapshot"]["path"] = "other.yaml"
        reproducibility.write_text(json.dumps(record))

    assert invoke_cli_boundary([command, str(materialized.config_path)], monkeypatch) == 1
    message = capsys.readouterr().err
    assert "Cannot read configuration snapshot for generation" in message
    assert generation_id in message and str(snapshot) in message
    assert message.count("fix:") == 1
    assert "restore the files and permissions identified in the error" in message
    assert "Traceback" not in message
    if damage == "digest":
        assert "snapshot hash mismatch" in message
    elif damage in {"yaml", "model"}:
        assert "hash mismatch" not in message
    elif damage == "missing-digest":
        assert "sha256" in message
    elif damage == "path":
        assert "does not anchor" in message


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        ("phases.0.n_trials=zero", "n_trials"),
        ("phases.9.n_trials=1", "phases.9"),
        ("unknown=1", "Extra inputs are not permitted"),
        ("nonsense", "KEY=VALUE"),
    ],
)
def test_run_override_errors_refuse_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    assignment: str,
    message: str,
) -> None:
    """Malformed or invalid overrides remain ordinary configuration errors."""
    path = write_experiment(tmp_path / "sweep.yaml", workdir="original")
    result = invoke_cli_boundary(["run", str(path), "--dry-run", "--set", assignment], monkeypatch)

    assert result == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "original").exists()


def test_show_winners_reads_unpublished_phase_results(tmp_path: Path) -> None:
    """Finished phase results remain visible before the run publishes."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment, config_path = materialized.experiment, materialized.config_path
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    generation_path = _generation_dir(_experiment_dir(experiment), generation_id)
    _generation_path(experiment).write_text("generation_id: interrupted\n")

    published = CliRunner().invoke(cli, ["show-winners", str(config_path)])

    assert published.exit_code == 0
    assert "trial_number: 0" in published.output
    assert f"# generation: {generation_path}" in published.output
    assert (generation_path / "summary.yaml").is_file()
    assert (generation_path / "phases/p/winner.yaml").is_file()

    _last_successful_generation_path(experiment).unlink()
    _generation_path(experiment).write_text(f"generation_id: {generation_id}\n")
    unpublished = CliRunner().invoke(cli, ["show-winners", str(config_path)])

    assert unpublished.exit_code == 0
    assert "trial_number: 0" in unpublished.output
    assert "(unpublished)" in unpublished.output
    assert "ranked_trials:" not in unpublished.output


def test_unpublished_winners_keep_the_producing_config_metadata(tmp_path: Path) -> None:
    """Config edits cannot relabel an unfinished generation's objective, phases or base."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    _last_successful_generation_path(materialized.experiment).unlink()

    result = CliRunner().invoke(
        cli,
        [
            "show-winners",
            str(materialized.config_path),
            "--set",
            "metric.name=renamed_objective",
            "--set",
            "phases.0.name=renamed_phase",
            "--set",
            "phases.0.comment=Changed annotation",
            "--set",
            "trainer_config={changed_base: 17}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "(unpublished)" in result.output
    assert "metric:\n  objective: 0.5" in result.output
    assert "=== p ===" in result.output
    assert "trainer_config: {}" in result.output
    for edited in ("renamed_objective", "renamed_phase", "Changed annotation", "changed_base"):
        assert edited not in result.output


def _two_phase_config(tmp_path: Path) -> Path:
    """Write a config with phases ``lr`` and ``wd``, a dotted search-space key and a trainer_config."""
    return write_experiment(
        tmp_path / "sweep.yaml",
        workdir="original",
        provenance={"revision": "r1"},
        trainer_config={"optimizer": {"lr": 0.1}},
        phases=[
            phase_payload(
                name="lr",
                search_space={"model.n_layers": {"type": "categorical", "choices": [4, 5]}},
            ),
            phase_payload(name="wd"),
        ],
    )


def _derived(path: Path, monkeypatch: pytest.MonkeyPatch, *settings: str) -> Experiment:
    """Run ``run --dry-run`` with the settings and return the experiment it would run."""
    received: list[Experiment] = []

    def capture(
        experiment: Experiment, *, from_phase: str | None, dry_run: bool
    ) -> dict[str, Winner]:
        assert dry_run
        assert from_phase is None
        received.append(experiment)
        return {}

    monkeypatch.setattr(cli_module, "run_experiment", capture)
    arguments = ["run", str(path), "--dry-run"]
    for setting in settings:
        arguments += ["--set", setting]
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 0, result.output
    (experiment,) = received
    return experiment


def _refused(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    setting: str,
) -> str:
    """Run ``run --dry-run --set SETTING`` through the process boundary and return its stderr."""
    assert invoke_cli_boundary(["run", str(path), "--dry-run", "--set", setting], monkeypatch) == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("key", ["phases.1.n_trials", "phases[1].n_trials", "phases[wd].n_trials"])
def test_set_addresses_a_phase_by_dotted_index_bracket_index_or_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """The three spellings of the second phase change that phase and no other."""
    experiment = _derived(_two_phase_config(tmp_path), monkeypatch, f"{key}=3")

    assert [phase.n_trials for phase in experiment.phases] == [1, 3]
    assert [phase.name for phase in experiment.phases] == ["lr", "wd"]


def test_set_bracket_addresses_a_search_space_key_that_contains_dots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[model.n_layers]`` is one mapping key, not a nested ``model`` mapping."""
    experiment = _derived(
        _two_phase_config(tmp_path),
        monkeypatch,
        "phases[lr].search_space[model.n_layers].choices=[6, 8]",
    )

    space = experiment.phases[0].search_space
    assert list(space) == ["model.n_layers"]
    assert space["model.n_layers"].model_dump()["choices"] == [6, 8]


@pytest.mark.parametrize(
    ("written", "key"),
    [
        ("a.b", "a.b"),
        ("a=b", "a=b"),
        (r"a\]b", "a]b"),
        (r"a\\b", "a\\b"),
        (r"a[0\]", "a[0]"),
        (" spaced ", " spaced "),
    ],
)
def test_set_bracket_addresses_any_mapping_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, written: str, key: str
) -> None:
    """Dots, equals signs, brackets and backslashes in a key are written inside ``[...]``."""
    experiment = _derived(_two_phase_config(tmp_path), monkeypatch, f"provenance[{written}]=v")

    assert experiment.provenance == {"revision": "r1", key: "v"}


def test_set_can_add_keys_to_the_free_form_trainer_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new key, nested or not, is accepted under ``trainer_config`` and keeps existing keys."""
    experiment = _derived(
        _two_phase_config(tmp_path),
        monkeypatch,
        "trainer_config.optimizer.warmup=100",
        "trainer_config.extra.depth=4",
    )

    assert experiment.trainer_config == {
        "optimizer": {"lr": 0.1, "warmup": 100},
        "extra": {"depth": 4},
    }


def test_set_refuses_an_unknown_phase_name_and_lists_the_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown phase name stops at ``--set`` with the argument, segment and syntax named."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, "phases[wd2].n_trials=2")

    assert "'phases[wd2].n_trials'" in message
    assert "'[wd2]'" in message
    assert "lr, wd" in message
    assert "Accepted syntax" in message
    assert "validation error" not in message
    assert not (tmp_path / "original").exists()


@pytest.mark.parametrize(
    ("setting", "created"),
    [
        ("phases[0].n_trial=2", "phases[0].n_trial"),
        ("phases.0.serach_space.x.low=1", "phases.0.serach_space"),
        ("phases[lr].sampler.sed=3", "phases[lr].sampler.sed"),
        ("unknown=1", "unknown"),
    ],
)
def test_set_names_the_argument_that_created_a_key_validation_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    setting: str,
    created: str,
) -> None:
    """A schema typo is reported against the ``--set`` that created the key."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, setting)

    assert "Extra inputs are not permitted" in message
    assert f"--set {setting.partition('=')[0]!r} created {created!r}" in message
    assert not (tmp_path / "original").exists()


@pytest.mark.parametrize(
    ("setting", "key"),
    [
        # A created key whose value validation rejects.
        ("env.ACCESS_TOKEN=123456789", "env.ACCESS_TOKEN"),
        # A path that addresses nothing.
        ("phases[nope].env.ACCESS_TOKEN=123456789", "phases[nope].env.ACCESS_TOKEN"),
        # A value that is not valid YAML, which the parser's own message would quote.
        ("env.ACCESS_TOKEN='123456789", "env.ACCESS_TOKEN"),
        # A '=' inside brackets belongs to the key.
        ("trainer_config[a=b]={c: 1, c: 123456789}", "trainer_config[a=b]"),
    ],
)
def test_set_diagnostics_name_the_key_but_never_the_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    setting: str,
    key: str,
) -> None:
    """Values can be credentials, so refusals quote the key as config errors hide input."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, setting)

    assert repr(key) in message
    assert "123456789" not in message


def test_set_does_not_blame_an_override_for_an_error_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rejected value at an existing key carries no creation note."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, "phases[0].n_trials=zero")

    assert "n_trials" in message
    assert "created" not in message


@pytest.mark.parametrize(
    ("setting", "fragment"),
    [
        ("phases[9].n_trials=1", "index 9 is out of range"),
        ("phases.lr.n_trials=1", "'lr' is not a list index"),
        ("phases[].n_trials=1", "is named ''"),
        ("phases[0.n_trials=1", "no closing ']'"),
        ("phases0].n_trials=1", "unmatched ']'"),
        ("phases[0]x=1", "must follow a '.'"),
        ("phases..n_trials=1", "empty path segment"),
        (".phases=1", "empty path segment"),
        ("phases.=1", "empty path segment"),
        ("=1", "nonempty KEY"),
        ("[0]=1", "no name before it"),
        ("phases[0].n_trials.x=1", "descend into the int value at 'phases[0].n_trials'"),
        ("phases[lr].sampler.type.x=1", "descend into the str value"),
    ],
)
def test_set_path_errors_name_the_argument_the_segment_and_the_syntax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    setting: str,
    fragment: str,
) -> None:
    """Every refused path quotes its argument, says what failed and repeats the syntax."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, setting)

    assert repr(setting.partition("=")[0]) in message
    assert fragment in message
    assert "Accepted syntax" in message
