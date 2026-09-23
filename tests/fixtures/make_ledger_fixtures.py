#!/usr/bin/env python3
"""Generate the golden on-disk ledger fixtures committed under tests/fixtures/ledgers.

Each fixture is one real PhaseSweep run (two trials of one phase, seeded random
sampler, no GPU) captured verbatim, plus at most one recorded raw edit that
turns the current-format bytes into the pre-cutover shape a read path has to
refuse. The edits are raw SQL or raw JSON/JSONL rewrites -- never PhaseSweep
code -- so the fixture keeps testing the reader rather than the writer that
produced it.

Run it from the repository root, in the environment under test::

    python -m tests.fixtures.make_ledger_fixtures

The experiment pins ``execution.cwd`` to ``/``, so the phase fingerprints it
stores name no directory of the generating checkout: a fixture read from any
clone path recomputes the fingerprint it was published with.

The two ``release-0.3.1-*`` fixtures are produced from a detached worktree of
the ``v0.3.1`` tag instead of the working tree::

    git worktree add --detach "$W" v0.3.1
    PYTHONPATH="$W/src" python -m tests.fixtures.make_ledger_fixtures \\
        --only current --label release-0.3.1 --expect-source "$W/src"
    git worktree remove --force "$W"

This module uses only the public ``phasesweep`` API to *produce* ledgers, so it
keeps working across the releases it has to generate bytes for.
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
#: ``phasesweep.engine.state.STUDY_SCHEMA_VERSION``; kept as a literal because
#: the legacy-release run imports a *different* phasesweep whose constant
#: describes that release, not the reader under test.
TARGET_STUDY_SCHEMA_VERSION = 3
#: Artifact-root binding schema this fixture set is asserted against. Mirrors
#: ``phasesweep.engine.artifact_roots.ARTIFACT_ROOT_BINDING_SCHEMA_VERSION``.
TARGET_BINDING_SCHEMA_VERSION = 3
#: Study user attribute holding the PhaseSweep study schema version.
STUDY_SCHEMA_ATTR = "phasesweep_study_schema_version"
#: Optuna journal op code for ``SET_STUDY_USER_ATTR``.
JOURNAL_SET_STUDY_USER_ATTR = 2
#: Artifact-root binding file, relative to ``<workdir>/<experiment>``.
BINDING_FILENAME = "artifact_root_binding.json"
#: MCP run-store format marker removed by the ``precutover-mcp-state`` fixture.
MCP_FORMAT_MARKER = ".phasesweep-format.json"
#: Replacement for the ``.gitignore`` PhaseSweep writes into every artifact
#: root. Its ``*`` keeps ordinary run output out of a user's repository; here it
#: would hide the fixture from git, and a deeper ignore file wins over the
#: repository root's negation, so the generator rewrites it in place.
FIXTURE_TREE_GITIGNORE = (
    "# Golden fixture: this artifact tree is committed on purpose.\n"
    "# PhaseSweep writes '*' here; see tests/fixtures/ledgers/README.md.\n"
    "!*\n"
)

#: Ledger file name per backend, relative to a fixture's ``ledger/`` directory.
LEDGER_FILENAME = {"sqlite": "study.db", "journal": "study.journal"}


def experiment_payload(*, storage_url: str, workdir: Path) -> dict[str, Any]:
    """Build the one experiment config every golden ledger fixture is produced from.

    Tests re-materialize a fixture by calling this with the copied ledger and
    artifact-root paths, so the only fields that differ between generation and
    materialization are the two paths -- neither of which the experiment's
    semantic fingerprint covers. The trainer cwd is pinned rather than left to
    the invocation directory, which the fingerprint *does* cover.

    :param str storage_url: Resolved ``sqlite:///`` or ``journal:///`` ledger URL.
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

    :param str backend: ``"sqlite"`` or ``"journal"``.
    :param Path ledger_dir: Directory holding the ledger file.
    :return str: Resolved storage URL.
    :raises ValueError: The backend is not one this fixture set produces.
    """
    if backend not in LEDGER_FILENAME:
        raise ValueError(f"Unsupported fixture backend: {backend!r}")
    return f"{backend}:///{ledger_dir / LEDGER_FILENAME[backend]}"


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
    mcp_state: Path
    backend: str

    @property
    def ledger(self) -> Path:
        """Return the ledger file this fixture's storage URL points at.

        :return Path: Absolute path to ``study.db`` or ``study.journal``.
        """
        return self.ledger_dir / LEDGER_FILENAME[self.backend]

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


def _sqlite_edit(paths: FixturePaths, statement: str) -> None:
    """Apply one raw SQL statement to a fixture's SQLite ledger.

    :param FixturePaths paths: Fixture under construction.
    :param str statement: Exact SQL recorded in the fixture manifest.
    """
    conn = sqlite3.connect(paths.ledger)
    try:
        conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def _journal_records(paths: FixturePaths) -> list[dict[str, Any]]:
    """Read a journal ledger as decoded JSONL records.

    :param FixturePaths paths: Fixture under construction.
    :return list[dict[str, Any]]: One decoded record per journal line.
    """
    text = paths.ledger.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _write_journal_records(paths: FixturePaths, records: list[dict[str, Any]]) -> None:
    """Rewrite a journal ledger from decoded records, preserving Optuna's encoding.

    Optuna writes one compact JSON object per line with a trailing newline and
    refuses to replay a journal whose last record is incomplete, so the exact
    line framing matters more than the field order.

    :param FixturePaths paths: Fixture under construction.
    :param list[dict[str, Any]] records: Records to serialize back, in order.
    """
    body = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    paths.ledger.write_text(body, encoding="utf-8")


def _downgrade_journal_schema_attr(paths: FixturePaths) -> None:
    """Rewrite the journal's schema-version record so it reports schema 2."""
    records = _journal_records(paths)
    rewritten = 0
    for record in records:
        if record.get("op_code") == JOURNAL_SET_STUDY_USER_ATTR and STUDY_SCHEMA_ATTR in record.get(
            "user_attr", {}
        ):
            record["user_attr"][STUDY_SCHEMA_ATTR] = TARGET_STUDY_SCHEMA_VERSION - 1
            rewritten += 1
    if rewritten != 1:
        raise RuntimeError(f"expected exactly one schema record in {paths.ledger}, got {rewritten}")
    _write_journal_records(paths, records)


def _drop_journal_schema_attr(paths: FixturePaths) -> None:
    """Drop the journal's schema-version record, leaving a populated unmarked study."""
    records = _journal_records(paths)
    kept = [
        record
        for record in records
        if not (
            record.get("op_code") == JOURNAL_SET_STUDY_USER_ATTR
            and STUDY_SCHEMA_ATTR in record.get("user_attr", {})
        )
    ]
    if len(kept) != len(records) - 1:
        raise RuntimeError(f"expected exactly one schema record in {paths.ledger}")
    _write_journal_records(paths, kept)


def _downgrade_binding(paths: FixturePaths) -> None:
    """Rewrite the artifact-root binding so it declares the pre-cutover schema."""
    payload = json.loads(paths.binding.read_text(encoding="utf-8"))
    payload["schema_version"] = TARGET_BINDING_SCHEMA_VERSION - 1
    paths.binding.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _remove_binding(paths: FixturePaths) -> None:
    """Delete the artifact-root binding, leaving unmarked durable state behind."""
    paths.binding.unlink()
    if not (paths.experiment_dir / "generations").is_dir():
        raise RuntimeError(f"{paths.experiment_dir} has no generations/ to leave unmarked")


#: The committed fixture set. ``expect`` is the verdict a read path owes the
#: fixture in each mode; the generator recomputes it from the produced bytes and
#: refuses to write a manifest that disagrees.
FIXTURE_SPECS: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        name="current-sqlite",
        group="current",
        backend="sqlite",
        modes=("tree", "ledger-only"),
        expect={"tree": "ok", "ledger-only": "ok"},
        derivation="",
        notes="Unedited output of this release; the reference both modes read cleanly.",
    ),
    FixtureSpec(
        name="current-journal",
        group="current",
        backend="journal",
        modes=("tree", "ledger-only"),
        expect={"tree": "ok", "ledger-only": "ok"},
        derivation="",
        notes="Unedited output of this release on the journal backend.",
    ),
    FixtureSpec(
        name="precutover-schema2-sqlite",
        group="precutover",
        backend="sqlite",
        modes=("tree", "ledger-only"),
        expect={"tree": "schema-mismatch", "ledger-only": "schema-mismatch"},
        derived_from="current-sqlite",
        derivation=(
            "UPDATE study_user_attributes SET value_json='2' "
            "WHERE key='phasesweep_study_schema_version'"
        ),
        apply=lambda paths: _sqlite_edit(
            paths,
            "UPDATE study_user_attributes SET value_json='2' "
            "WHERE key='phasesweep_study_schema_version'",
        ),
        notes="A populated study stamped with the pre-cutover schema version.",
    ),
    FixtureSpec(
        name="precutover-schema2-journal",
        group="precutover",
        backend="journal",
        modes=("tree", "ledger-only"),
        expect={"tree": "schema-mismatch", "ledger-only": "schema-mismatch"},
        derived_from="current-journal",
        derivation=(
            "Rewrite the JSONL record with op_code == 2 (SET_STUDY_USER_ATTR) carrying "
            "user_attr['phasesweep_study_schema_version'] so its value is 2."
        ),
        apply=_downgrade_journal_schema_attr,
        notes="A populated study stamped with the pre-cutover schema version.",
    ),
    FixtureSpec(
        name="precutover-unstamped-sqlite",
        group="precutover",
        backend="sqlite",
        modes=("tree", "ledger-only"),
        expect={"tree": "schema-mismatch", "ledger-only": "schema-mismatch"},
        derived_from="current-sqlite",
        derivation=(
            "DELETE FROM study_user_attributes WHERE key='phasesweep_study_schema_version'"
        ),
        apply=lambda paths: _sqlite_edit(
            paths,
            "DELETE FROM study_user_attributes WHERE key='phasesweep_study_schema_version'",
        ),
        notes="Populated but unmarked: the shape a 0.3.1 ledger has before stamping existed.",
    ),
    FixtureSpec(
        name="precutover-unstamped-journal",
        group="precutover",
        backend="journal",
        modes=("tree", "ledger-only"),
        expect={"tree": "schema-mismatch", "ledger-only": "schema-mismatch"},
        derived_from="current-journal",
        derivation=(
            "Drop the JSONL record with op_code == 2 (SET_STUDY_USER_ATTR) carrying "
            "user_attr['phasesweep_study_schema_version']; the trial records remain."
        ),
        apply=_drop_journal_schema_attr,
        notes="Populated but unmarked: the shape a 0.3.1 ledger has before stamping existed.",
    ),
    FixtureSpec(
        name="precutover-binding2-sqlite",
        group="precutover",
        backend="sqlite",
        modes=("tree",),
        expect={"tree": "root-conflict"},
        derived_from="current-sqlite",
        derivation="Set artifact_root_binding.json 'schema_version' to 2.",
        apply=_downgrade_binding,
        notes="Current ledger, pre-cutover artifact tree: the binding is checked first.",
    ),
    FixtureSpec(
        name="precutover-unmarked-tree-sqlite",
        group="precutover",
        backend="sqlite",
        modes=("tree",),
        expect={"tree": "root-conflict"},
        derived_from="current-sqlite",
        derivation="Delete artifact_root_binding.json; generations/ is left in place.",
        apply=_remove_binding,
        notes="Durable artifact state with no format marker at all.",
    ),
    FixtureSpec(
        name="current-sqlite-optuna40-versioninfo",
        group="optuna40",
        backend="sqlite",
        modes=("tree", "ledger-only"),
        expect={"tree": "ok", "ledger-only": "ok"},
        derived_from="current-sqlite",
        derivation="UPDATE version_info SET library_version='4.0.0'",
        apply=lambda paths: _sqlite_edit(paths, "UPDATE version_info SET library_version='4.0.0'"),
        notes=(
            "Optuna's own recorded library version differs while its SQL schema does not. "
            "PhaseSweep's readers must not consult it."
        ),
    ),
    FixtureSpec(
        name="precutover-mcp-state",
        group="mcp-state",
        backend="sqlite",
        modes=(),
        expect={},
        derivation=f"Delete mcp_state/{MCP_FORMAT_MARKER} from a populated run store.",
        notes="Durable MCP run-store state with no format marker; RunStore refuses to open it.",
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

    :param Path path: Any path inside the checkout.
    :return str: ``git describe --tags --always`` output, or ``"unknown"``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _sqlite_schema_versions(ledger: Path) -> list[tuple[str, object, bool]]:
    """Read every PhaseSweep study's schema stamp from a SQLite ledger.

    :param Path ledger: SQLite ledger file.
    :return list[tuple[str, object, bool]]: Study name, decoded stamp, and
        whether the study holds trials.
    """
    conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        studies = [
            (int(study_id), str(name))
            for study_id, name in conn.execute("SELECT study_id, study_name FROM studies")
            if "::" in str(name)
        ]
        stamps = {
            str(name): json.loads(value_json)
            for name, value_json in conn.execute(
                "SELECT studies.study_name, study_user_attributes.value_json "
                "FROM studies JOIN study_user_attributes "
                "ON studies.study_id = study_user_attributes.study_id "
                "WHERE study_user_attributes.key = ?",
                (STUDY_SCHEMA_ATTR,),
            )
        }
        populated = {
            int(study_id) for (study_id,) in conn.execute("SELECT DISTINCT study_id FROM trials")
        }
    finally:
        conn.close()
    return [(name, stamps.get(name), study_id in populated) for study_id, name in studies]


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
    versions = (
        _sqlite_schema_versions(paths.ledger)
        if paths.backend == "sqlite"
        else _journal_schema_versions(paths.ledger)
    )
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
    pre-cutover or missing binding decides the ``tree`` verdict on its own; the
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


def _vacuum(paths: FixturePaths) -> None:
    """Compact a SQLite fixture so a regenerated copy has stable page layout.

    :param FixturePaths paths: Produced fixture.
    """
    if paths.backend != "sqlite" or not paths.ledger.is_file():
        return
    conn = sqlite3.connect(paths.ledger, isolation_level=None)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()


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


def _make_mcp_state(paths: FixturePaths) -> None:
    """Build a populated MCP run store, then strip its format marker.

    :param FixturePaths paths: Fixture layout whose ``mcp_state`` is populated.
    :raises RuntimeError: The store did not produce the expected marker.
    """
    from phasesweep.mcp.runs import RunHandle, RunStore

    store = RunStore(paths.mcp_state)
    handle = RunHandle(
        run_id="golden-precutover-run",
        experiment_id=EXPERIMENT,
        config_sha256="0" * 64,
        pid=999999,
        pgid=999999,
        pid_starttime=111,
        started_at="2024-01-01T00:00:00Z",
    )
    store.create(handle)
    store.config_snapshot_path(handle.run_id).write_text(
        "# pre-cutover run config snapshot\n", encoding="utf-8"
    )
    marker = paths.mcp_state / MCP_FORMAT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"RunStore did not create {marker}")
    marker.unlink()


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
    :param str name: Fixture name, which a ``--label`` run overrides.
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
    verify_expectations: bool,
    notes: str | None = None,
) -> Path:
    """Produce one golden ledger fixture from scratch.

    :param FixtureSpec spec: Fixture to produce.
    :param Path out_dir: Directory that holds every fixture.
    :param str name: Name for this fixture, which a ``--label`` run overrides.
    :param Path scratch: Throwaway directory for the generation-time config.
    :param dict[str, str] produced_by: Toolchain identifiers for the manifest.
    :param str | None notes: Manifest note replacing the spec's own.
    :param bool verify_expectations: Compare the spec's declared verdicts with
        the ones computed from the produced bytes. A legacy-release run skips
        this, because the declared verdicts describe the current release.
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
        mcp_state=root / "mcp_state",
        backend=spec.backend,
    )
    if spec.group == "mcp-state":
        root.mkdir(parents=True)
        _make_mcp_state(paths)
    else:
        paths.ledger_dir.mkdir(parents=True)
        _run_experiment_into(paths, scratch)
        if spec.apply is not None:
            spec.apply(paths)
        _vacuum(paths)
    expect = _computed_expectations(paths, spec.modes)
    if verify_expectations and spec.expect is not None and expect != spec.expect:
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
        "--label",
        default=None,
        help="rename the generated fixtures from <group>-<backend> to <label>-<backend>",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="replace the selected fixtures' manifest notes, e.g. for a --label run",
    )
    parser.add_argument(
        "--expect-source",
        type=Path,
        default=None,
        help="require the imported phasesweep to come from this src directory",
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
    if args.label and len({spec.group for spec in specs}) != 1:
        print("error: --label requires --only to select a single fixture group", file=sys.stderr)
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phasesweep-golden-") as tmp:
        sandbox = Path(tmp)
        _isolate_environment(sandbox)
        import phasesweep

        source_root = Path(phasesweep.__file__).resolve().parent.parent
        if args.expect_source is not None:
            expected = args.expect_source.resolve()
            if source_root != expected:
                print(
                    f"error: imported phasesweep from {source_root}, expected {expected}",
                    file=sys.stderr,
                )
                return 2
        produced_by = _produced_by(source_root)
        try:
            for spec in specs:
                name = f"{args.label}-{spec.backend}" if args.label else spec.name
                generate(
                    spec,
                    out_dir,
                    name=name,
                    scratch=sandbox,
                    produced_by=produced_by,
                    verify_expectations=args.label is None,
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
