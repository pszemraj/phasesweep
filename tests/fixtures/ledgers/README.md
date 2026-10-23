# Golden ledger fixture

The `current-journal` fixture contains on-disk output produced through PhaseSweep's public API:

```text
current-journal/manifest.json          inventory record: backend, modes, expected verdicts, provenance
current-journal/ledger/study.journal   journal ledger
current-journal/artifact_root/t/...    the artifact tree that claimed that ledger
```

It is consumed through `tests/ledger_fixtures.py`: `tests/test_ledger_read_paths.py` runs its read-path matrix over it, and other test modules (publication, error routing, format cutover, CLI, observations, fingerprints, status shape, read views, MLflow delivery) copy it as a ready-made published generation with `golden_publication` or `materialize`. The helper copies the fixture into a temporary directory, rebuilds the experiment config from `tests/fixtures/make_ledger_fixtures.py`, rewrites the copy's artifact-root binding and per-study root records in `tree` mode, and snapshots the bytes before the read.

## Inventory

| Fixture | Backend | Modes | Binding schema | tree verdict | ledger-only verdict |
| --- | --- | --- | --- | --- | --- |
| `current-journal` | journal | tree, ledger-only | 11 | ok | ok |

`tree` reads the fixture's own artifact root, so the artifact-root binding is validated first. `ledger-only` points the same ledger at a workdir that has never existed, which is the case no binding can answer and the ledger format scan must.

The manifest records the fixture's read verdicts and generating toolchain: PhaseSweep Git revision, Optuna, SQLite library version, Python, and working directory.

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

## Churn

A regeneration is not byte-identical, and is not expected to be. These differ every time, in the ledger and in the artifact tree:

- generation ids, attempt ids, trial directory names, and Optuna worker ids;
- `datetime_start` / `datetime_complete` / `recorded_at` timestamps, `run.log`, and trial durations;
- `phasesweep_version` in `summary.yaml` and `reproducibility.json`, which moves with the SCM revision;
- absolute paths, which name the checkout the fixture was generated in, and with them `artifact_root_binding.json`'s `artifact_root` and its `storage_key` digest (`tests/ledger_fixtures.py` rewrites both for the copy it reads).

`phasesweep_trainer_env` and its checksum `phasesweep_trainer_env_digest` are not churn: they come from the pinned environment, so they match across regenerations and carry nothing of the generating shell.

The read-path tests check format-bearing state, including the study schema stamp and the artifact-root binding's `schema_version`. Regenerate when the stored format changes. Timestamps, identities, and paths make the resulting diff larger than the format change itself.
