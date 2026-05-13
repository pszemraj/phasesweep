# Development

## Quality Gates

```bash
pytest
ruff check .
ruff format --check .
mypy src/phasesweep --ignore-missing-imports
pathlint src tests
python doc_check.py src --check-lazy
```

Current expected test result: `262 passed` with Optuna `constant_liar` experimental warnings.

## Test Map

Tests are organized by behavior:

- `tests/test_e2e.py`: full sweep and `--from-phase` replay.
- `tests/test_storage_urls.py`: storage backend parsing, dialect folding, absolute and relative path identity.
- `tests/test_locking.py`: run locks, storage locks, output namespace locks, and phase locks.
- `tests/test_process_supervision.py`: subprocess lifecycle, signal handling, cleanup confirmation, launch-window safety.
- `tests/test_stale_reaper.py`: startup reaping, PID-reuse checks, fail-closed cleanup.
- `tests/test_fingerprint.py`: semantic fingerprints, resume verification, run-control exclusions.
- `tests/test_filesystem_layout.py`: output namespace layout and experiment-name validation.
- `tests/test_param_validation.py`: search-space validation, override keys, sampler compatibility, grids, seeds, template placeholders.
- `tests/test_runtime_behavior.py`: extractor NaN/inf behavior, parallel sampler config, failure aborts, timeout completion policy.
- `tests/test_protocol.py`: contracts, evidence gates, promotion, suites.
- `tests/test_config.py`, `tests/test_extractors.py`, `tests/test_overrides.py`, `tests/test_selector.py`, `tests/test_gpu_pool.py`, `tests/test_cli.py`, `tests/test_wandb_extractor.py`: focused unit surfaces.
