# Development

Repository: <https://github.com/pszemraj/phasesweep.git>

- [Quality Gates](#quality-gates)
- [Package Map](#package-map)
- [Test Map](#test-map)

## Quality Gates

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy src/phasesweep --ignore-missing-imports
```

Expected result: `179 passed`. Optuna `constant_liar` experimental warnings are expected.

## Package Map

![module dependency graph](/docs/images/diagramF_moduledeps.png)
<!-- img is intended to be linked w absolute path from repo root. Do NOT change it. -->

The package is organized by behavior:

- `phasesweep.config`: Pydantic config models and strict YAML loading.
- `phasesweep.engine`: Optuna study orchestration, fingerprints, locks, promotion, persistence, status, and suite execution.
- `phasesweep.evidence`: metric extractors, post-trial evidence gates, and W&B polling.
- `phasesweep.runtime`: subprocess, GPU, lock, storage URL, and override helpers.
- `phasesweep.cli`: Click command surface.

Stable API calls live at the package root: `load_config`, `load_experiment`, `run_config`, `run_experiment`, `run_suite`, and `config_status`. Schema types are exported from `phasesweep.config`. Tests that need internals import direct submodules under `engine`, `evidence`, or `runtime`.

The control flow of a typical run is as follows:

![control flow](/docs/images/diagramC_controlflow.png)
<!-- img is intended to be linked w absolute path from repo root. Do NOT change it. -->

## Test Map

Tests are organized by behavior:

- `tests/test_e2e.py`: full sweep and `--from-phase` replay.
- `tests/test_storage_urls.py`, `tests/test_locking.py`: storage identity, URL parsing, and same-host advisory locks.
- `tests/test_process_supervision.py`, `tests/test_stale_reaper.py`: subprocess cleanup, signal handling, startup/skipped-phase reaping.
- `tests/test_fingerprint.py`: semantic fingerprints, resume verification, run-control exclusions.
- `tests/test_filesystem_layout.py`: output namespace layout and experiment-name validation.
- `tests/test_param_validation.py`: search-space validation, override keys, sampler compatibility, grids, seeds, template placeholders.
- `tests/test_runtime_behavior.py`, `tests/test_protocol.py`: timeout policy, contracts, evidence gates, promotion, and suites.
- `tests/test_config.py`, `tests/test_extractors.py`, `tests/test_overrides.py`, `tests/test_selector.py`, `tests/test_gpu_pool.py`, `tests/test_cli.py`, `tests/test_public_metadata.py`: focused unit surfaces.
