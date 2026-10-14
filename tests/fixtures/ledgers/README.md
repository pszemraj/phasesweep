# Golden ledger fixture

The `current-journal` fixture contains on-disk output produced through PhaseSweep's public API:

```text
current-journal/manifest.json          inventory record: backend, modes, expected verdicts, provenance
current-journal/ledger/study.journal   journal ledger
current-journal/artifact_root/t/...    the artifact tree that claimed that ledger
```

It is consumed by `tests/test_ledger_read_paths.py` through `tests/ledger_fixtures.py`, which copies the fixture into a temporary directory, rebuilds the experiment config from `tests/fixtures/make_ledger_fixtures.py`, and snapshots the bytes before the read.

## Inventory

| Fixture | Backend | Modes | Binding schema | tree | ledger-only |
| --- | --- | --- | --- | --- | --- |
| `current-journal` | journal | tree, ledger-only | 6 | ok | ok |

`tree` reads the fixture's own artifact root, so the artifact-root binding is validated first. `ledger-only` points the same ledger at a workdir that has never existed, which is the case no binding can answer and the ledger format scan must.

`current-journal` uses the current journal format. Its `manifest.json` records `derivation: ""` because the generator does not transform that format. Its `produced_by` carries the phasesweep `git describe --tags --always --dirty`, Optuna, the interpreter's stdlib `sqlite3` module version (unrelated to the fixture's ledger backend -- the journal is the only backend this fixture produces), and Python versions, plus the directory the generator ran from.

## Regeneration

The fixture is regenerated, never hand-edited. From the repository root:

```sh
python -m tests.fixtures.make_ledger_fixtures
```

The generator refuses to write a manifest whose declared verdicts disagree with the ones it computes from the bytes it just produced, so a regeneration that silently changes meaning fails instead of landing.

The generator keeps the fixture tracked and independent of the generating host:

- `artifact_root/t/.gitignore` is rewritten from PhaseSweep's `*` to `!*`. A deeper ignore file wins over the repository root's negation, so the original would hide the whole fixture from git.
- the experiment declares `execution.inherit_env: none` explicitly, so the fixture does not move when the default does.
- the trials are generated under a pinned base environment, `FIXTURE_ENVIRONMENT` in `tests/fixtures/make_ledger_fixtures.py`: `PATH=/usr/bin:/bin`, `HOME=/home/phasesweep`, `USER=LOGNAME=phasesweep`, `LANG=C.UTF-8`, `TZ=UTC`, and `LC_ALL` unset. Each trial records its tracked environment in plain text in the ledger, as described by the [trainer contract](../../../docs/config.md#trainer-contract), so the generating shell's `PATH` would otherwise be committed, and a test running from any other shell would be a different environment cohort. The trainer is the shell's `printf` builtin, because the pinned `PATH` holds no Python interpreter.
- the experiment declares `execution.cwd: /`. An unset trainer cwd puts the invocation directory into every phase and experiment fingerprint, so a fixture would match its own config only in the checkout that generated it. With the pin, the stored fingerprints are the same from any clone path and any generating directory; `tests/test_ledger_read_paths.py::test_fixture_fingerprints_do_not_depend_on_the_working_directory` holds that.

A test that resumes a fixture study and allocates new trials calls `pin_fixture_environment(monkeypatch)` from `tests/ledger_fixtures.py`, which gives the process the same environment, so the cohort check passes from any shell, the commit hook's included.

## Churn

A regeneration is not byte-identical, and is not expected to be. These differ every time, in the ledger and in the artifact tree:

- generation ids, attempt ids, trial directory names, and Optuna worker ids;
- `datetime_start` / `datetime_complete` / `recorded_at` timestamps, `run.log`, and trial durations;
- `phasesweep_version` in `summary.yaml` and `reproducibility.json`, which moves with the SCM revision;
- absolute paths, which name the checkout the fixture was generated in, and with them `artifact_root_binding.json`'s `artifact_root` and its `storage_key` digest (`tests/ledger_fixtures.py` rewrites both for the copy it reads).

`phasesweep_trainer_env` and its checksum `phasesweep_trainer_env_digest` are not churn: they come from the pinned environment, so they match across regenerations and carry nothing of the generating shell.

The read-path tests check format-bearing state, including the study schema stamp and the artifact-root binding's `schema_version`. Regenerate when the stored format changes. Timestamps, identities, and paths make the resulting diff larger than the format change itself.
