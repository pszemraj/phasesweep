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
- the experiment declares `execution.inherit_env: none` explicitly. `none` records only the fixed base names in the ledger, while `all` would record every ambient variable *name*, committing the generating operator's environment shape and differing on every machine. The pin keeps the fixture independent of the default.
- the experiment declares `execution.cwd: /`. An unset trainer cwd puts the invocation directory into every phase and experiment fingerprint, so a fixture would match its own config only in the checkout that generated it. With the pin, the stored fingerprints are the same from any clone path and any generating directory; `tests/test_ledger_read_paths.py::test_fixture_fingerprints_do_not_depend_on_the_working_directory` holds that.

## Churn

A regeneration is not byte-identical, and is not expected to be. These differ every time, in the ledger and in the artifact tree:

- generation ids, attempt ids, trial directory names, and Optuna worker ids;
- `datetime_start` / `datetime_complete` / `recorded_at` timestamps, `run.log`, and trial durations;
- `phasesweep_version` in `summary.yaml` and `reproducibility.json`, which moves with the SCM revision;
- absolute paths, which name the checkout the fixture was generated in, and with them `artifact_root_binding.json`'s `artifact_root` and its `storage_key` digest (`tests/ledger_fixtures.py` rewrites both for the copy it reads);
- `phasesweep_trainer_env_digest`, which covers the inherited base environment except `TMPDIR` and the `passthrough_env` names, as described by the [trainer contract](../../../docs/config.md#trainer-contract).

The read-path tests check format-bearing state, including the study schema stamp and the artifact-root binding's `schema_version`. Regenerate when the stored format changes. Timestamps, identities, and paths make the resulting diff larger than the format change itself.
