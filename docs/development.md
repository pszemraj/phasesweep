# Development

- [Quality Gates](#quality-gates)
- [Package Map](#package-map)
- [Test Map](#test-map)
- [Tracked TODOs](#tracked-todos)

## Quality Gates

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/phasesweep
pytest
python scripts/mcp_workflow_eval.py
```

Run `pytest` by itself, with no concurrent `ruff`, `mypy`, workflow eval, or other validation jobs. Some process-supervision and timeout tests are timing-sensitive and can fail under unrelated validation load. A clean full-suite run should not print a warning summary; investigate and fix new warnings instead of accepting them as background noise. The MCP workflow eval prints JSON and exits nonzero if discovery, read-only safety, or launch-monitor-winners flow fails.

## Package Map

![module dependency graph](images/diagramF_moduledeps.png)

The package is organized by behavior:

- `phasesweep.config`: Pydantic config models and strict YAML loading.
- `phasesweep.engine`: Optuna study orchestration, fingerprints, locks, promotion, persistence, status, and suite execution.
- `phasesweep.evidence`: metric extractors, post-trial evidence gates, and W&B polling.
- `phasesweep.runtime`: subprocess, GPU, lock, storage URL, and override helpers.
- `phasesweep.mcp`: stdio MCP server, catalog registry, detached runner, and run-handle store.
- `phasesweep.cli`: Click command surface.

Common package-root calls are `load_config`, `load_experiment`, `run_config`, `run_experiment`, `run_suite`, and `config_status`. Schema types are exported from `phasesweep.config`. Tests that need internals import direct submodules under `engine`, `evidence`, `runtime`, or `mcp`.

The control flow of a typical run is as follows:

![control flow](images/diagramC_controlflow.png)

## Test Map

Tests are organized by behavior:

- `tests/test_e2e.py`: full sweep and `--from-phase` replay.
- `tests/test_storage_urls.py`, `tests/test_locking.py`: storage identity, URL parsing, and same-host advisory locks.
- `tests/test_process_supervision.py`, `tests/test_stale_reaper.py`: subprocess cleanup, signal handling, startup/skipped-phase reaping.
- `tests/test_fingerprint.py`: semantic fingerprints, resume verification, run-control exclusions.
- `tests/test_filesystem_layout.py`: output namespace layout and experiment-name validation.
- `tests/test_param_validation.py`: search-space validation, override keys, sampler compatibility, grids, seeds, template placeholders.
- `tests/test_runtime_behavior.py`, `tests/test_protocol.py`: timeout policy, contracts, evidence gates, promotion, and suites.
- `tests/test_mcp_*.py`: MCP catalog validation, redaction, run handles, detached runner, server logic, and e2e flow.
- `tests/test_config.py`, `tests/test_extractors.py`, `tests/test_overrides.py`, `tests/test_selector.py`, `tests/test_gpu_pool.py`, `tests/test_cli.py`, `tests/test_public_metadata.py`: focused unit surfaces.

## Tracked TODOs

- TODO(runtime): Consider a trial bootstrap/exec handshake before claiming hard-crash durability for the `Popen` to identity-file window; normal exceptions and shutdown signals are cleaned up, but SIGKILL or host loss in that narrow interval cannot be repaired by the killed parent process.
- TODO(mcp): Split `mcp/server.py` into SDK-free application logic, schemas, launch lifecycle, and FastMCP adapter modules after the MCP alpha surface stabilizes; keep the current schema and request-handler tests as the safety net during that refactor.
- TODO(mcp): Add active-run indexing, archival, or bounded history pagination before treating thousands of historical MCP handles in one `state_dir` as a supported operating mode.
