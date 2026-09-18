# Runtime behavior

PhaseSweep runs one ordered Experiment on a POSIX host. It uses same-host file
locks and process groups to serialize a persistent experiment and clean up
trainer subprocesses. GPUs are optional; the core CLI can run CPU-only work.
The optional MCP server additionally requires Linux `/proc` process identity
information for detached-run cancellation and recovery.

## Fresh-state cutover

This breaking release has one current on-disk format. Start it with all of the
following:

- a fresh artifact root (`<workdir>/<experiment>`);
- a fresh local SQLite or Journal ledger, if the experiment is persistent; and
- a fresh MCP `state_dir`, if the experiment is launched through MCP.

The runtime checks an existing artifact root and local ledger before it creates
or stamps state. It refuses missing, malformed, or unsupported PhaseSweep
format records without modifying that namespace. An unmarked durable MCP state
directory is likewise refused before it is initialized. A new experiment name
or workdir does not make an old populated local ledger fresh.

There is no migration, adoption, relocation, rebind, or repair path in this
release. Operate an existing 0.3.1 output, ledger, or MCP state directory with
the preserved 0.3.1 environment. Choose new paths for the consolidated
release.

## Output layout

For `workdir: ./runs` and `experiment: demo`, PhaseSweep writes under
`./runs/demo/`:

```text
demo/
  .gitignore
  artifact_root_binding.json
  run.log
  generation.yaml
  last_successful_generation.yaml
  summary.yaml
  attempts/
  generations/
    <generation-id>/
      generation.yaml
      config.snapshot.yaml
      reproducibility.json
      summary.yaml
      phases/<phase>/winner.yaml
  <phase>/
    winner.yaml
    trial_00000__generation_<id>__attempt_<id>/
      trainer_config.yaml       # yaml_file only
      overrides_resolved.json
      command.txt
      stdout.log
      stderr.log
      result.json               # when produced by the trainer
```

The root `summary.yaml` and per-phase `winner.yaml` files are convenience
projections. `last_successful_generation.yaml` points to the immutable
generation that authoritatively represents the latest successful result. A
failed invocation does not replace that pointer.

`config.snapshot.yaml` is the executed configuration, including materialized
defaults and the resolved invocation context. It is private because it can
contain command and environment values. `reproducibility.json` is the
shareable generation identity record. Trial directories retain local evidence
and attempt lifecycle information used for supported continuation and winner
validation.

PhaseSweep writes a `.gitignore` containing `*` in a new experiment namespace;
it does not replace an existing ignore file. Keep generated outputs, ledgers,
and MCP state out of commits.

## Storage and locks

`storage: null` is in-memory and ends with the process. Persistent local
storage is one of:

- `sqlite:///...` for sequential phases;
- `journal:///...` for same-host parallel phases; or
- `auto`, which chooses a sibling `study.db` or `study.journal` according to
  whether any phase has `n_jobs > 1`.

The selected artifact root and ledger are bound to each other. A reused root
cannot combine a second ledger's trial counts with the first ledger's
publication. If changing `n_jobs` would make `auto` choose the other backend,
the run is refused; restore the prior setting or start a fresh namespace.

Persistent phases use same-host locks. A concurrent CLI or MCP launch for the
same experiment waits or fails safely according to the operation; it never
merges two active orchestrators. SQLite is sequential at the PhaseSweep
configuration level. Journal storage supports local parallel trials. External
database backends are not part of the runtime.

## Trial execution and evidence

Each trial has an attempt-scoped directory and process group. The runtime
captures `stdout.log` and `stderr.log`, prepares the configured input, and
supervises cleanup. See the [configuration guide](config.md) for trainer input,
local evidence, gate, constraint, and environment definitions.

## GPU isolation

By default, `gpu_policy: single_per_trial` leases one visible CUDA device per
parallel trial. Explicit `gpu_ids` or `gpu_devices` define the available
tokens; otherwise the runtime uses numeric, GPU UUID, or MIG tokens from
`CUDA_VISIBLE_DEVICES`, or falls back to host discovery. The lease controls
visibility and same-host locking, not GPU memory allocation.

`gpu_policy: whole_node` requires `n_jobs: 1` and a unique explicit device
list. `gpu_policy: none` disables PhaseSweep's GPU isolation and cannot be
combined with an explicit device list; parallel use requires the explicit
`allow_no_gpu_isolation: true` acknowledgement. For CPU-only parallel work,
make that acknowledgement deliberately rather than relying on failed device
detection.

## Timeouts, cleanup, and continuation

Trial, phase, and experiment timeouts are cooperative run controls. On a
trial timeout PhaseSweep terminates the supervised process group and records a
terminal failed attempt after cleanup. The `max_consecutive_failures` threshold
also stops a broken phase. `n_trials` counts terminal attempts, so failed and
pruned attempts consume the configured target.

Before a supported continuation, PhaseSweep reconciles stale active attempts,
checks the current artifact/ledger binding, verifies recorded trial evidence,
and applies sampler continuation rules. `--from-phase` is valid only when the
earlier phase winners remain valid; reused winners keep their original source
provenance. TPE and CMA-ES persistent targets cannot resume mid-target, while
grid and seeded random phases can top up their local study.

An interrupted or failed generation leaves its immutable record for inspection
but does not advance `last_successful_generation.yaml`. Signals are absorbed
across the final publication commit so the authority pointer is never left
ambiguous.

## Inspection commands

Use `validate`, `run --dry-run`, `status`, and `show-winners` to review an
experiment without launching work; an ordinary `run` invocation launches
trials. `--from-phase` requires valid earlier winners. These reads inspect only
the current-format local experiment; use the original 0.3.1 environment for
existing 0.3.1 state.

For operator-managed detached runs, see [the MCP operator guide](mcp.md).
Terminal MCP run results are frozen snapshots associated with a run ID; they
are not a replacement for CLI inspection of an arbitrary experiment tree.
