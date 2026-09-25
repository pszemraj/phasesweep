# Golden ledger fixtures

Real on-disk PhaseSweep output, committed so the read paths can be tested
against bytes a reader actually has to survive instead of bytes a test invented.
Each directory is one fixture:

```
<name>/manifest.json          inventory record: backend, modes, expected verdicts, provenance
<name>/ledger/study.journal   journal ledger
<name>/artifact_root/t/...    the artifact tree that claimed that ledger
<name>/mcp_state/...          MCP run store        (precutover-mcp-state only)
```

They are consumed by `tests/test_ledger_read_paths.py` through
`tests/ledger_fixtures.py`, which copies a fixture into a temporary directory,
rebuilds the experiment config from `tests/fixtures/make_ledger_fixtures.py`,
and snapshots the bytes before the read.

## Inventory

| Fixture | Backend | Modes | Binding schema | tree | ledger-only | Derived from |
| --- | --- | --- | --- | --- | --- | --- |
| `current-journal` | journal | tree, ledger-only | 3 | ok | ok | — |
| `precutover-schema2-journal` | journal | tree, ledger-only | 3 | schema-mismatch | schema-mismatch | `current-journal` |
| `precutover-unstamped-journal` | journal | tree, ledger-only | 3 | schema-mismatch | schema-mismatch | `current-journal` |
| `precutover-binding2-journal` | journal | tree | 2 | root-conflict | — | `current-journal` |
| `precutover-unmarked-tree-journal` | journal | tree | none | root-conflict | — | `current-journal` |
| `release-0.3.1-journal` | journal | tree, ledger-only | 2 | root-conflict | schema-mismatch | — |
| `precutover-mcp-state` | — | — | — | — | — | — |

`tree` reads the fixture's own artifact root, so the artifact-root binding is
validated first and decides the verdict on its own when it is pre-cutover.
`ledger-only` points the same ledger at a workdir that has never existed, which
is the case no binding can answer and the ledger format scan must.

Each fixture's `manifest.json` records the exact `derivation` applied to it --
a described JSONL rewrite for the journal, or a JSON edit for the
artifact-root binding. Derivations never run PhaseSweep code, so a fixture
keeps testing the reader rather than the writer that produced it.

## Provenance

`current-journal` and `precutover-*` are produced by the working tree's
PhaseSweep. `release-0.3.1-journal` is produced by the preserved 0.3.1 release
itself, from a detached `v0.3.1` worktree, against the same Optuna range -- so
the format boundary is tested against bytes the old release actually wrote,
not a reconstruction of them. Every manifest carries `produced_by` with the
phasesweep `git describe --tags --always --dirty`, Optuna, the interpreter's
stdlib `sqlite3` module version (unrelated to any fixture's ledger backend --
the journal is the only backend these fixtures produce), and Python versions,
plus the directory the generator ran from. A `release-<version>-*` manifest
must record exactly `v<version>`, and every fixture with a ledger must declare
at least one read mode; the fixture-documentation test refuses anything else.

`release-0.3.1-journal` predates the trainer-cwd pin described below, so its
phase and experiment fingerprints embed the directory it was generated from
and match no materialized config. That is harmless: every read path refuses
that fixture first, at `root-conflict` in `tree` mode and `schema-mismatch` in
`ledger-only` mode, so no read ever compares its fingerprints.

## Regeneration

Fixtures are regenerated, never hand-edited. From the repository root:

```sh
python -m tests.fixtures.make_ledger_fixtures                  # everything but release-0.3.1-journal
python -m tests.fixtures.make_ledger_fixtures --only current-journal   # one fixture
```

`release-0.3.1-journal` needs the tagged source on `PYTHONPATH`:

```sh
W=$(mktemp -d)/psw-0.3.1
git worktree add --detach "$W" v0.3.1
PYTHONPATH="$W/src" python -m tests.fixtures.make_ledger_fixtures \
    --only current --label release-0.3.1 --expect-source "$W/src" \
    --notes "Verbatim output of the preserved PhaseSweep 0.3.1 release: ..."
git worktree remove --force "$W"
```

The generator refuses to write a manifest whose declared verdicts disagree with
the ones it computes from the bytes it just produced, so a regeneration that
silently changes meaning fails instead of landing.

Three things the generator changes about a raw run, all for packaging rather
than for the format boundary:

- `artifact_root/t/.gitignore` is rewritten from PhaseSweep's `*` to `!*`. A
  deeper ignore file wins over the repository root's negation, so the original
  would hide the whole fixture from git.
- the experiment declares `execution.inherit_env: none`, because the default
  records every ambient variable *name* in the ledger, which would commit the
  generating operator's environment shape and differ on every machine.
- the experiment declares `execution.cwd: /`. An unset trainer cwd puts the
  invocation directory into every phase and experiment fingerprint, so a
  fixture would match its own config only in the checkout that generated it.
  With the pin, the stored fingerprints are the same from any clone path and
  any generating directory;
  `tests/test_ledger_read_paths.py::test_fixture_fingerprints_do_not_depend_on_the_working_directory`
  holds that.

## Churn

A regeneration is not byte-identical, and is not expected to be. These differ
every time, in the ledger and in the artifact tree:

- generation ids, attempt ids, trial directory names, and Optuna worker ids;
- `datetime_start` / `datetime_complete` / `recorded_at` timestamps, `run.log`,
  and trial durations;
- `phasesweep_version` in `summary.yaml` and `reproducibility.json`, which
  moves with the SCM revision;
- absolute paths, which name the checkout the fixture was generated in, and with
  them `artifact_root_binding.json`'s `artifact_root` and its `storage_key`
  digest (`tests/ledger_fixtures.py` rewrites both for the copy it reads);
- `phasesweep_trainer_env_digest`, which hashes the host's `PATH`, `HOME`, and
  `TMPDIR` values.

None of that is what the fixtures test. What must stay stable is the format-
bearing state: the study schema stamp, the artifact-root binding's
`schema_version`, the presence of the MCP format marker, and Optuna's own
`version_info` / `alembic_version` rows. Regenerate only when the release
changes one of those on purpose, and expect a large, noisy diff when you do.
