# Consolidation release notes

This is a breaking consolidation release. It keeps one ordered Experiment,
local execution, local evidence, local storage, and durable MCP run handling;
it removes alternate orchestration and compatibility surfaces. Existing 0.3.1
runs are not converted. Keep using the preserved 0.3.1 environment for those
runs.

## What changed

- A config is now one Experiment. Suites, suite defaults, compiled studies,
  cross-study promotion, and named contracts are gone. Express a former
  component as a separate complete experiment YAML, or copy the necessary
  fixed overrides and gates directly into a phase after reviewing them.
- Phase promotion and baseline substitution are gone. Local gates now remain
  hard trial failures; an infeasible constraint remains completed but cannot
  win. A phase with no eligible winner fails.
- Multi-parent inheritance is direct and deterministic. Same-origin diamonds
  work; distinct inherited origins require an explicit child
  `fixed_overrides` resolution. Inherited values cannot be resampled.
- Trainer inputs are limited to complete `yaml_file` configuration and generic
  `argparse` options. The overrides-only JSON and Hydra adapters are removed.
  JSON objective envelopes, scalar constraints, and JSON gates are retained as
  local evidence formats.
- PhaseSweep no longer extracts objectives or gates from W&B and no longer
  supports external RDB storage. Trainers may still receive generic `WANDB_*`
  environment settings and can report their local objective normally.
- MCP client installers and the `mcp install`, `mcp uninstall`, and
  `mcp check-install` commands are removed. Each client owns its connection
  configuration and instructions.
- Artifact relocation/rebinding and old-layout interpretation are gone. The
  current immutable generation and last-success pointer remain the result
  authority.
- MCP status and results are run-specific. Follow the [tool workflow](mcp.md#tool-workflow)
  for launch, reconnection, reads, waiting, and cancellation.

## Fresh-state requirement

Use new paths for this release. Follow the [runtime cutover](runtime.md#fresh-state-cutover)
for artifact roots and ledgers, plus [MCP fresh state](mcp.md#fresh-mcp-state)
for detached runs.

## Retained workflow

Continue to use local SQLite, Journal, in-memory, or `auto` storage; ordered
phase inheritance; fixed overrides; grid, random, TPE, and CMA-ES samplers;
local constraints and gates; GPU leasing; immutable publication; and supported
`--from-phase` continuation. The optional MCP server still provides catalog
approval, detached launch, durable run IDs, bounded waiting, cancellation,
operator recovery, redaction, and frozen terminal snapshots.
