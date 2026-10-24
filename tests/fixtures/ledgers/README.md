# Golden ledger fixture

The `current-journal` fixture contains on-disk output produced through PhaseSweep's public API:

```text
current-journal/manifest.json          inventory record: backend, modes, expected verdicts, provenance
current-journal/ledger/study.journal   journal ledger
current-journal/artifact_root/t/...    the artifact tree that claimed that ledger
```

Use `golden_publication` or `materialize` from [`ledger_fixtures.py`](../../ledger_fixtures.py) to copy the fixture into a temporary directory. The helper rebuilds its config, relocates artifact-root and study bindings in `tree` mode, and snapshots bytes before the read. [`test_ledger_read_paths.py`](../../test_ledger_read_paths.py) runs the read-path matrix.

## Inventory

| Fixture | Backend | Modes | Binding schema | tree verdict | ledger-only verdict |
| --- | --- | --- | --- | --- | --- |
| `current-journal` | journal | tree, ledger-only | 13 | ok | ok |

`tree` reads the fixture's own artifact root, so the artifact-root binding is validated first. `ledger-only` points the same ledger at a workdir that has never existed, testing the ledger format scan without a binding. The table reports direct-reader format verdicts; CLI `status`, `status --trials` and `show-winners` instead refuse the missing study directory, which the read-path matrix classifies as `study-absent`.

The manifest records the generating toolchain: PhaseSweep Git revision, Optuna, SQLite library version, Python, and working directory.

## Regeneration

The fixture is regenerated, never hand-edited. From the repository root:

```bash
python -m tests.fixtures.make_ledger_fixtures
```

`--out DIR` writes the fixture to another directory, `--notes TEXT` replaces its manifest notes, and `-v` raises log verbosity. Generation fails if the produced bytes disagree with the expected read verdicts, so after a stamp bump set `TARGET_STUDY_SCHEMA_VERSION` and `TARGET_BINDING_SCHEMA_VERSION` in [`make_ledger_fixtures.py`](../make_ledger_fixtures.py) to the new values before regenerating, and update the `Binding schema` column above.

The generator keeps the fixture tracked and independent of the generating host:

- `artifact_root/t/.gitignore` contains `!*`, allowing the generated tree to be tracked.
- `execution.inherit_env: none` uses the pinned `FIXTURE_ENVIRONMENT` in [`make_ledger_fixtures.py`](../make_ledger_fixtures.py): `PATH=/usr/bin:/bin`, `HOME=/home/phasesweep`, `USER=LOGNAME=phasesweep`, `LANG=C.UTF-8`, `TZ=UTC`, and `LC_ALL` unset. These values are stored in the ledger under the [trainer contract](../../../docs/config.md#trainer-contract). The trainer uses the shell's `printf` builtin.
- `execution.cwd: /` keeps the experiment fingerprints independent of the generating checkout and invocation directory.

A test that resumes the fixture and allocates trials calls `pin_fixture_environment(monkeypatch)` from [`ledger_fixtures.py`](../../ledger_fixtures.py) to match the recorded environment.

Regenerate when the stored format changes. Generated identities, timestamps, paths and SCM metadata vary between regenerations; the pinned trainer environment and its checksum stay fixed.
