#!/usr/bin/env python3
"""Generate the golden journal fixture under tests/fixtures/ledgers.

Run ``python -m tests.fixtures.make_ledger_fixtures`` from the repository root.
The fixture contains two seeded random trials, with no GPU, trainer cwd ``/``,
and the pinned ``FIXTURE_ENVIRONMENT``. Its manifest records the toolchain and
verified read-path expectations.
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
from collections.abc import Iterator
from contextlib import contextmanager
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
#: Trainer cwd, independent of the generating checkout.
TRAINER_CWD = "/"
#: Objective token, distinct from override names.
OBJECTIVE_TOKEN = "PSWOBJ"
#: Constant objective the inline trainer reports.
OBJECTIVE_VALUE = "0.5"
#: Shell-builtin trainer ignores overrides and needs no Python interpreter.
#: Literal braces must be escaped for render_command's str.format expansion.
TRIAL_COMMAND = f"printf '{OBJECTIVE_TOKEN}={OBJECTIVE_VALUE}\\n' {{overrides}}"
#: The base environment a fixture's trials are generated under, and the one a test
#: applies before it resumes a fixture study and allocates new trials. A persistent
#: study refuses a resume whose tracked environment differs from the one recorded
#: on its trials, and each trial records its tracked values in plain text, so
#: generating under the shell at hand would commit that shell's ``PATH`` and
#: make every other shell, the commit hook's included, a different cohort. ``None``
#: means unset. Only the base names ``inherit_env: none`` tracks belong here; this
#: module is stdlib only, so the generator can read it before PhaseSweep is imported.
FIXTURE_ENVIRONMENT: dict[str, str | None] = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/phasesweep",
    "USER": "phasesweep",
    "LOGNAME": "phasesweep",
    "LANG": "C.UTF-8",
    "TZ": "UTC",
    "LC_ALL": None,
}

#: PhaseSweep study schema this fixture set is asserted against. Mirrors
#: ``phasesweep.engine.state.STUDY_SCHEMA_VERSION``; kept as a literal so this
#: module names no PhaseSweep import before ``_isolate_environment`` runs.
TARGET_STUDY_SCHEMA_VERSION = 10
#: Artifact-root binding schema this fixture set is asserted against. Mirrors
#: ``phasesweep.engine.artifact_roots.ARTIFACT_ROOT_BINDING_SCHEMA_VERSION``.
TARGET_BINDING_SCHEMA_VERSION = 13
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

#: Ledger file name, relative to the fixture's ``ledger/`` directory.
LEDGER_FILENAME = "study.journal"
FIXTURE_NAME = "current-journal"
FIXTURE_MODES = ("tree", "ledger-only")
FIXTURE_EXPECT = {"tree": "ok", "ledger-only": "ok"}
FIXTURE_NOTES = "Golden journal fixture."


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
        # The contract is declared rather than left to the default, so the
        # fixture does not move when the default does.
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


def storage_url(ledger_dir: Path) -> str:
    """Return the journal URL for the fixture ledger."""
    return f"journal:///{ledger_dir / LEDGER_FILENAME}"


@dataclass(frozen=True)
class FixturePaths:
    """Resolved on-disk layout of the fixture under construction."""

    root: Path
    ledger_dir: Path
    artifact_root: Path

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


@contextmanager
def _pinned_environment() -> Iterator[None]:
    """Apply :data:`FIXTURE_ENVIRONMENT` to this process, then restore what was there.

    :return Iterator[None]: A context in which every pinned variable has its pinned value, or is
        unset when the pin is ``None``.
    """
    saved = {name: os.environ.get(name) for name in FIXTURE_ENVIRONMENT}
    try:
        for name, value in FIXTURE_ENVIRONMENT.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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


def _computed_expectations(paths: FixturePaths) -> dict[str, str]:
    """Derive the verdict each mode owes this fixture, from the produced bytes.

    The artifact-root binding is validated before the ledger is scanned, so a
    mismatched binding decides the ``tree`` verdict on its own, and so does a
    missing one beside a durable tree. Otherwise the ledger decides both
    ``tree`` and ``ledger-only``.

    :param FixturePaths paths: Produced fixture.
    :return dict[str, str]: Verdict per mode.
    """
    ledger_verdict = _ledger_verdict(paths)
    binding_version = _binding_schema_version(paths)
    has_durable_tree = (paths.experiment_dir / "generations").is_dir()
    if binding_version is None:
        tree_verdict = "root-conflict" if has_durable_tree else ledger_verdict
    elif binding_version != TARGET_BINDING_SCHEMA_VERSION:
        tree_verdict = "root-conflict"
    else:
        tree_verdict = ledger_verdict
    return {"tree": tree_verdict, "ledger-only": ledger_verdict}


def _run_experiment_into(paths: FixturePaths, scratch: Path) -> None:
    """Run the fixture experiment, writing its ledger and artifact tree in place.

    :param FixturePaths paths: Fixture layout to populate.
    :param Path scratch: Directory for the config file, which is not committed.
    """
    import yaml

    from phasesweep import load_experiment, run_experiment

    payload = experiment_payload(
        storage_url=storage_url(paths.ledger_dir),
        workdir=paths.artifact_root,
    )
    config_path = scratch / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with _pinned_environment():
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
    paths: FixturePaths,
    *,
    expect: dict[str, str],
    produced_by: dict[str, str],
    notes: str,
) -> None:
    """Write one fixture's manifest beside its ledger.

    :param FixturePaths paths: Produced fixture layout.
    :param dict[str, str] expect: Verdict per supported mode.
    :param dict[str, str] produced_by: Toolchain identifiers.
    :param str notes: Human-readable description of what this fixture holds.
    """
    manifest = {
        "name": FIXTURE_NAME,
        "backend": "journal",
        "ledger": str(paths.ledger.relative_to(paths.root)),
        "experiment": EXPERIMENT,
        "phases": [PHASE],
        "n_trials": N_TRIALS,
        "modes": list(FIXTURE_MODES),
        "binding_schema_version": _binding_schema_version(paths),
        "expect": expect,
        "derived_from": None,
        "derivation": "",
        "notes": notes,
        "produced_by": produced_by,
    }
    (paths.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate(
    out_dir: Path,
    *,
    scratch: Path,
    produced_by: dict[str, str],
    notes: str | None = None,
) -> Path:
    """Produce the golden ledger fixture from scratch.

    :param Path out_dir: Directory that holds every fixture.
    :param Path scratch: Throwaway directory for the generation-time config.
    :param dict[str, str] produced_by: Toolchain identifiers for the manifest.
    :param str | None notes: Optional replacement for the fixture's default note.
    :return Path: The produced fixture directory.
    :raises RuntimeError: The computed verdicts contradict the declared ones.
    """
    root = out_dir / FIXTURE_NAME
    if root.exists():
        shutil.rmtree(root)
    paths = FixturePaths(
        root=root,
        ledger_dir=root / "ledger",
        artifact_root=root / "artifact_root",
    )
    paths.ledger_dir.mkdir(parents=True)
    _run_experiment_into(paths, scratch)
    expect = _computed_expectations(paths)
    if expect != FIXTURE_EXPECT:
        raise RuntimeError(
            f"{FIXTURE_NAME}: produced bytes imply {expect}, but expected {FIXTURE_EXPECT}"
        )
    _write_manifest(
        paths,
        expect=expect,
        produced_by=produced_by,
        notes=FIXTURE_NOTES if notes is None else notes,
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
        help="directory for the generated fixture",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="replace the fixture's manifest notes",
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
    """Generate the golden ledger fixture.

    :param list[str] | None argv: Arguments after the program name.
    :return int: ``0`` on success, ``1`` on a generation failure, ``2`` on bad input.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        stream=sys.stderr,
        format="%(levelname)s: %(message)s",
    )
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phasesweep-golden-") as tmp:
        sandbox = Path(tmp).resolve()
        _isolate_environment(sandbox)
        import phasesweep

        source_root = Path(phasesweep.__file__).resolve().parent.parent
        produced_by = _produced_by(source_root)
        try:
            generate(out_dir, scratch=sandbox, produced_by=produced_by, notes=args.notes)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(f"generated {FIXTURE_NAME} under {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
