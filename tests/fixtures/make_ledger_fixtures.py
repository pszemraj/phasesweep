#!/usr/bin/env python3
"""Generate the golden on-disk ledger fixture committed under tests/fixtures/ledgers.

The fixture is one real PhaseSweep run -- two trials of one phase, seeded
random sampler, no GPU -- captured verbatim on the journal backend.
PhaseSweep keeps no backward compatibility, so this generator produces only
the current-format shape.

Run it from the repository root, in the environment under test::

    python -m tests.fixtures.make_ledger_fixtures

The experiment pins ``execution.cwd`` to ``/``, so the phase fingerprints it
stores name no directory of the generating checkout: a fixture read from any
clone path recomputes the fingerprint it was published with.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Experiment name every fixture is generated under.
EXPERIMENT = "t"
#: Single phase name every fixture is generated with.
PHASE = "p"
#: Trials per fixture. Two is the smallest count that proves a populated study.
N_TRIALS = 2
#: Provenance revision recorded in every fixture config.
PROVENANCE_REVISION = "golden-fixture-v1"
#: Trainer working directory. An unset ``execution.cwd`` fingerprints the
#: invocation directory, which would tie every stored phase fingerprint to the
#: checkout path the generator ran in. ``/`` exists on every host, and the
#: inline trainer reads no files.
TRAINER_CWD = "/"
#: Objective token the inline trainer prints. Deliberately not a parameter name,
#: so an override echoed into a trial log can never be mistaken for the metric.
OBJECTIVE_TOKEN = "PSWOBJ"
#: Constant objective the inline trainer reports.
OBJECTIVE_VALUE = "0.5"
#: Inline trainer: no file on disk, so the fingerprinted ``trial_command`` holds
#: no generation-time path and a materialized fixture reproduces it exactly.
#: ``render_command`` runs ``str.format`` over this template, so it must not
#: contain a literal ``{`` or ``}``.
TRIAL_COMMAND = f"python -c 'print(\"{OBJECTIVE_TOKEN}={OBJECTIVE_VALUE}\")' {{overrides}}"

#: PhaseSweep study schema this fixture set is asserted against. Mirrors
#: ``phasesweep.engine.state.STUDY_SCHEMA_VERSION``; kept as a literal so this
#: module names no PhaseSweep import before ``_isolate_environment`` runs.
TARGET_STUDY_SCHEMA_VERSION = 4
#: Artifact-root binding schema this fixture set is asserted against. Mirrors
#: ``phasesweep.engine.artifact_roots.ARTIFACT_ROOT_BINDING_SCHEMA_VERSION``.
TARGET_BINDING_SCHEMA_VERSION = 3
#: Study user attribute holding the PhaseSweep study schema version.
STUDY_SCHEMA_ATTR = "phasesweep_study_schema_version"
#: Optuna journal op code for ``SET_STUDY_USER_ATTR``.
JOURNAL_SET_STUDY_USER_ATTR = 2
#: Artifact-root binding file, relative to ``<workdir>/<experiment>``.
BINDING_FILENAME = "artifact_root_binding.json"
#: Replacement for the ``.gitignore`` PhaseSweep writes into every artifact
#: root. Its ``*`` keeps ordinary run output out of a user's repository; here it
#: would hide the fixture from git, and a deeper ignore file wins over the
#: repository root's negation, so the generator rewrites it in place.
FIXTURE_TREE_GITIGNORE = (
    "# Golden fixture: this artifact tree is committed on purpose.\n"
    "# PhaseSweep writes '*' here; see tests/fixtures/ledgers/README.md.\n"
    "!*\n"
)

#: Ledger file name, relative to a fixture's ``ledger/`` directory. The
#: journal is the only backend this fixture set produces.
LEDGER_FILENAME = "study.journal"


def experiment_payload(*, storage_url: str, workdir: Path) -> dict[str, Any]:
    """Build the one experiment config every golden ledger fixture is produced from.

    Tests re-materialize a fixture by calling this with the copied ledger and
    artifact-root paths, so the only fields that differ between generation and
    materialization are the two paths -- neither of which the experiment's
    semantic fingerprint covers. The trainer cwd is pinned rather than left to
    the invocation directory, which the fingerprint *does* cover.

    :param str storage_url: Resolved ``journal:///`` ledger URL.
    :param Path workdir: Artifact root parent; artifacts land in ``workdir/t``.
    :return dict[str, Any]: Experiment config ready for YAML serialization.
    """
    return {
        "experiment": EXPERIMENT,
        "storage": storage_url,
        "provenance": {"revision": PROVENANCE_REVISION},
        "workdir": str(workdir),
        "trial_command": TRIAL_COMMAND,
        "override_format": "argparse",
        # Without a narrowed contract the trial records every ambient variable
        # NAME in the ledger, which would commit the generating operator's
        # environment shape and make the fixture unreproducible elsewhere.
        "execution": {"cwd": TRAINER_CWD, "inherit_env": "none"},
        "metric": {
            "name": "objective",
            "goal": "minimize",
            "extractor": {
                "type": "log_regex",
                "pattern": rf"{OBJECTIVE_TOKEN}=(?P<value>[0-9.eE+-]+)",
            },
        },
        "phases": [
            {
                "name": PHASE,
                "n_trials": N_TRIALS,
                "sampler": {"type": "random", "seed": 0},
                "search_space": {"a": {"type": "int", "low": 0, "high": 10}},
            }
        ],
    }


def storage_url(backend: str, ledger_dir: Path) -> str:
    """Build the ledger URL for one backend under a ledger directory.

    :param str backend: ``"journal"``, the only backend this fixture set produces.
    :param Path ledger_dir: Directory holding the ledger file.
    :return str: Resolved storage URL.
    :raises ValueError: The backend is not one this fixture set produces.
    """
    if backend != "journal":
        raise ValueError(f"Unsupported fixture backend: {backend!r}")
    return f"{backend}:///{ledger_dir / LEDGER_FILENAME}"


@dataclass(frozen=True)
class FixtureSpec:
    """One golden ledger fixture: how to produce it and what readers owe it."""

    name: str
    group: str
    backend: str
    modes: tuple[str, ...]
    expect: dict[str, str] | None
    derivation: str
    derived_from: str | None = None
    apply: Callable[[FixturePaths], None] | None = None
    notes: str = ""


@dataclass(frozen=True)
class FixturePaths:
    """Resolved on-disk layout of one fixture under construction."""

    root: Path
    ledger_dir: Path
    artifact_root: Path
    backend: str

    @property
    def ledger(self) -> Path:
        """Return the ledger file this fixture's storage URL points at.

        :return Path: Absolute path to ``study.journal``.
        """
        return self.ledger_dir / LEDGER_FILENAME

    @property
    def experiment_dir(self) -> Path:
        """Return the experiment artifact namespace ``<workdir>/<experiment>``.

        :return Path: Absolute artifact namespace path.
        """
        return self.artifact_root / EXPERIMENT

    @property
    def binding(self) -> Path:
        """Return the artifact-root binding file path.

        :return Path: Absolute path to ``artifact_root_binding.json``.
        """
        return self.experiment_dir / BINDING_FILENAME


#: The committed fixture set. ``expect`` is the verdict a read path owes the
#: fixture in each mode; the generator recomputes it from the produced bytes and
#: refuses to write a manifest that disagrees.
FIXTURE_SPECS: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        name="current-journal",
        group="current",
        backend="journal",
        modes=("tree", "ledger-only"),
        expect={"tree": "ok", "ledger-only": "ok"},
        derivation="",
        notes="Unedited output of this release on the journal backend.",
    ),
)


def _spec_by_name(name: str) -> FixtureSpec:
    """Return the spec with a given name.

    :param str name: Fixture name.
    :return FixtureSpec: The matching spec.
    :raises KeyError: No spec carries that name.
    """
    for spec in FIXTURE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def _select(only: str | None) -> list[FixtureSpec]:
    """Select the specs to generate from a comma-separated name/group filter.

    :param str | None only: Comma-separated fixture names or group names.
    :return list[FixtureSpec]: Selected specs in declaration order.
    :raises ValueError: A requested token matches no fixture name or group.
    """
    if not only:
        return list(FIXTURE_SPECS)
    wanted = [token.strip() for token in only.split(",") if token.strip()]
    selected: list[FixtureSpec] = []
    for token in wanted:
        matches = [spec for spec in FIXTURE_SPECS if token in (spec.name, spec.group)]
        if not matches:
            raise ValueError(f"--only {token!r} matches no fixture name or group")
        selected.extend(spec for spec in matches if spec not in selected)
    return [spec for spec in FIXTURE_SPECS if spec in selected]


def _isolate_environment(sandbox: Path) -> None:
    """Point every ambient PhaseSweep/Optuna state path at a throwaway sandbox.

    Called before ``phasesweep`` is first imported so no generation step can
    read or write the invoking operator's real state, cache, or lock namespace.

    :param Path sandbox: Existing directory to host the isolated state.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("XDG_STATE_HOME", "XDG_CACHE_HOME"):
        directory = sandbox / name.lower()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    locks = sandbox / "locks"
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["PHASESWEEP_LOCK_DIR"] = str(locks)


def _git_describe(path: Path) -> str:
    """Describe the git checkout a path belongs to.

    ``--dirty`` marks a checkout with uncommitted changes, whose bytes no
    commit reproduces.

    :param Path path: Any path inside the checkout.
    :return str: ``git describe --tags --always --dirty`` output, or ``"unknown"``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "describe", "--tags", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _journal_schema_versions(ledger: Path) -> list[tuple[str, object, bool]]:
    """Read every PhaseSweep study's schema stamp from a journal ledger.

    :param Path ledger: Journal ledger file.
    :return list[tuple[str, object, bool]]: Study name, decoded stamp, and
        whether the study holds trials.
    """
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    names: dict[int, str] = {}
    stamps: dict[int, object] = {}
    populated: set[int] = set()
    for record in records:
        op_code = record.get("op_code")
        if op_code == 0:
            names[len(names)] = str(record.get("study_name"))
        elif op_code == JOURNAL_SET_STUDY_USER_ATTR and STUDY_SCHEMA_ATTR in record.get(
            "user_attr", {}
        ):
            stamps[int(record["study_id"])] = record["user_attr"][STUDY_SCHEMA_ATTR]
        elif op_code == 4:
            populated.add(int(record.get("study_id", -1)))
    return [
        (name, stamps.get(study_id), study_id in populated)
        for study_id, name in names.items()
        if "::" in name
    ]


def _ledger_verdict(paths: FixturePaths) -> str:
    """Classify a fixture's ledger the way the format scan classifies it.

    :param FixturePaths paths: Produced fixture.
    :return str: ``"ok"`` or ``"schema-mismatch"``.
    """
    versions = _journal_schema_versions(paths.ledger)
    unsupported = [
        (name, stamp)
        for name, stamp, has_trials in versions
        if (type(stamp) is not int or stamp != TARGET_STUDY_SCHEMA_VERSION)
        and (stamp is not None or has_trials)
    ]
    return "schema-mismatch" if unsupported else "ok"


def _binding_schema_version(paths: FixturePaths) -> int | None:
    """Read the produced artifact-root binding's schema version.

    :param FixturePaths paths: Produced fixture.
    :return int | None: Recorded schema version, or ``None`` when unbound.
    """
    if not paths.binding.is_file():
        return None
    payload = json.loads(paths.binding.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    return int(version) if isinstance(version, int) else None


def _computed_expectations(paths: FixturePaths, modes: tuple[str, ...]) -> dict[str, str]:
    """Derive the verdict each mode owes this fixture, from the produced bytes.

    The artifact-root binding is validated before the ledger is scanned, so a
    missing or mismatched binding decides the ``tree`` verdict on its own; the
    ledger decides ``ledger-only`` and any ``tree`` read whose binding is
    current.

    :param FixturePaths paths: Produced fixture.
    :param tuple[str, ...] modes: Modes this fixture supports.
    :return dict[str, str]: Verdict per mode.
    """
    if not modes:
        return {}
    ledger_verdict = _ledger_verdict(paths)
    binding_version = _binding_schema_version(paths)
    has_durable_tree = (paths.experiment_dir / "generations").is_dir()
    if binding_version is None:
        tree_verdict = "root-conflict" if has_durable_tree else ledger_verdict
    elif binding_version != TARGET_BINDING_SCHEMA_VERSION:
        tree_verdict = "root-conflict"
    else:
        tree_verdict = ledger_verdict
    return {mode: tree_verdict if mode == "tree" else ledger_verdict for mode in modes}


def _run_experiment_into(paths: FixturePaths, scratch: Path) -> None:
    """Run the fixture experiment, writing its ledger and artifact tree in place.

    :param FixturePaths paths: Fixture layout to populate.
    :param Path scratch: Directory for the config file, which is not committed.
    """
    import yaml

    from phasesweep import load_experiment, run_experiment

    payload = experiment_payload(
        storage_url=storage_url(paths.backend, paths.ledger_dir),
        workdir=paths.artifact_root,
    )
    config_path = scratch / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    run_experiment(load_experiment(config_path))
    tree_gitignore = paths.experiment_dir / ".gitignore"
    if tree_gitignore.is_file():
        tree_gitignore.write_text(FIXTURE_TREE_GITIGNORE, encoding="utf-8")


def _produced_by(source_root: Path) -> dict[str, str]:
    """Record the toolchain that produced a fixture.

    :param Path source_root: ``src`` directory the generating phasesweep came from.
    :return dict[str, str]: Version identifiers stored in every manifest.
    """
    import optuna

    return {
        "phasesweep_git": _git_describe(source_root),
        "optuna": optuna.__version__,
        "sqlite": sqlite3.sqlite_version,
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd().resolve()),
    }


def _write_manifest(
    spec: FixtureSpec,
    paths: FixturePaths,
    *,
    name: str,
    expect: dict[str, str],
    produced_by: dict[str, str],
    notes: str,
) -> None:
    """Write one fixture's manifest beside its ledger.

    :param FixtureSpec spec: Spec the fixture was produced from.
    :param FixturePaths paths: Produced fixture layout.
    :param str name: Fixture name.
    :param dict[str, str] expect: Verdict per supported mode.
    :param dict[str, str] produced_by: Toolchain identifiers.
    :param str notes: Human-readable description of what this fixture holds.
    """
    manifest = {
        "name": name,
        "backend": spec.backend if paths.ledger.is_file() else None,
        "ledger": str(paths.ledger.relative_to(paths.root)) if paths.ledger.is_file() else None,
        "experiment": EXPERIMENT,
        "phases": [PHASE],
        "n_trials": N_TRIALS,
        "modes": list(spec.modes),
        "binding_schema_version": _binding_schema_version(paths),
        "expect": expect,
        "derived_from": spec.derived_from,
        "derivation": spec.derivation,
        "notes": notes,
        "produced_by": produced_by,
    }
    (paths.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate(
    spec: FixtureSpec,
    out_dir: Path,
    *,
    name: str,
    scratch: Path,
    produced_by: dict[str, str],
    notes: str | None = None,
) -> Path:
    """Produce one golden ledger fixture from scratch.

    :param FixtureSpec spec: Fixture to produce.
    :param Path out_dir: Directory that holds every fixture.
    :param str name: Name for this fixture.
    :param Path scratch: Throwaway directory for the generation-time config.
    :param dict[str, str] produced_by: Toolchain identifiers for the manifest.
    :param str | None notes: Manifest note replacing the spec's own.
    :return Path: The produced fixture directory.
    :raises RuntimeError: The computed verdicts contradict the declared ones.
    """
    root = out_dir / name
    if root.exists():
        shutil.rmtree(root)
    paths = FixturePaths(
        root=root,
        ledger_dir=root / "ledger",
        artifact_root=root / "artifact_root",
        backend=spec.backend,
    )
    paths.ledger_dir.mkdir(parents=True)
    _run_experiment_into(paths, scratch)
    if spec.apply is not None:
        spec.apply(paths)
    expect = _computed_expectations(paths, spec.modes)
    if spec.expect is not None and expect != spec.expect:
        raise RuntimeError(
            f"{name}: produced bytes imply {expect}, but the spec declares {spec.expect}"
        )
    _write_manifest(
        spec,
        paths,
        name=name,
        expect=expect,
        produced_by=produced_by,
        notes=spec.notes if notes is None else notes,
    )
    logger.info("wrote %s (%s)", root, expect or "no read-path modes")
    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the generator's command line.

    :param list[str] | None argv: Arguments after the program name.
    :return argparse.Namespace: Parsed options.
    """

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
    ):
        """Show defaults while preserving the module docstring's line breaks."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "ledgers",
        help="directory that holds every fixture",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated fixture names or groups to regenerate",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="replace the selected fixtures' manifest notes",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (-v, -vv)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate the selected golden ledger fixtures.

    :param list[str] | None argv: Arguments after the program name.
    :return int: ``0`` on success, ``1`` on a generation failure, ``2`` on bad input.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        stream=sys.stderr,
        format="%(levelname)s: %(message)s",
    )
    try:
        specs = _select(args.only)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phasesweep-golden-") as tmp:
        sandbox = Path(tmp)
        _isolate_environment(sandbox)
        import phasesweep

        source_root = Path(phasesweep.__file__).resolve().parent.parent
        produced_by = _produced_by(source_root)
        try:
            for spec in specs:
                generate(
                    spec,
                    out_dir,
                    name=spec.name,
                    scratch=sandbox,
                    produced_by=produced_by,
                    notes=args.notes,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(f"generated {len(specs)} fixture(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
