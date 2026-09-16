# PhaseSweep MCP server

`phasesweep-mcp` and `phasesweep mcp serve` expose a human-curated experiment catalog to an AI agent over the [Model Context Protocol](https://modelcontextprotocol.io) using the [supported MCP runtime](runtime.md#platform-support). See [Tools](#tools) for available requests and [Security model](#security-model) for the authority boundary.

For installation and client setup, use [MCP agent setup](mcp_setup.md).

## The catalog

The server starts from a catalog: a fixed allowlist mapping opaque IDs to local config paths plus per-experiment permissions. The agent only ever sends an ID; it cannot enumerate the filesystem, pass a path, or author a config. Author the catalog with the same trust and review process as the experiment YAML. The server never writes it.

Catalog keys:

`init-catalog` pins an absolute `state_dir` outside the project at
`${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>/`. The digest
is the first 12 hexadecimal characters of SHA-256 over the resolved catalog
path. An owner-only `origin` file records that final catalog path. Each catalog
gets a separate namespace, including catalogs in the same directory. The saved
path stays fixed across server restarts and environment changes; existing
catalogs retain their explicit paths. Setting `PHASESWEEP_HOME` when scaffolding
selects `<PHASESWEEP_HOME>/mcp/<catalog-digest>/` instead. That root must already
be absolute, owned by the current user, and mode `0700`, with no symlinked
components. Empty overrides are unset; relative `XDG_STATE_HOME` values use the
home-directory fallback. Each researcher keeps private state; teams coordinating
GPUs on one node use the existing [shared lock directory](runtime.md#concurrency-model).

- `state_dir: path` (required): operator-owned directory for run handles, runner logs, config snapshots, and `audit.jsonl`. A catalog load that reaches state preparation creates missing directories with mode `0700`, validates existing directories without changing them, then probes each directory with a temporary file. Server startup, `mcp check`, `mcp init-catalog`, and MCP integration preflight for `mcp install` all load the catalog; even a dry run or a cancelled install can therefore create the private state layout. An unsafe existing mode fails with its observed owner/mode and a concrete `chmod 700 ...` remediation when ownership is already correct.
- `max_concurrent_runs: int = 1` (minimum `1`): live-sweep cap across all catalog entries; see [concurrency and single-GPU hosts](#concurrency-and-single-gpu-hosts).
- `experiments: list` (required; at least one item): allowlisted experiment entries. Entry IDs must be unique.
- `experiments[].id: str` (required): agent-visible ID matching nonempty `[A-Za-z0-9_-]+`.
- `experiments[].config: path` (required): local single-experiment YAML subject to the [MCP path and storage rules](#paths-and-the-working-directory).
- `experiments[].cwd: path | null = null`: detached-runner working directory; resolution and default behavior are covered under [paths and the working directory](#paths-and-the-working-directory).
- `experiments[].visible_params: "none" | "all" | list[str] = "none"`: sampled winner values exposed to agents. List entries are stripped, must be nonempty, and are deduplicated. Parameter names remain visible; withheld values return `<redacted>`, and each winner carries `params_redacted`.
- `experiments[].description: str = ""` (maximum 500 characters): optional purpose shown by `list_experiments` and the catalog resource.
- `experiments[].allow: object = {}`: side-effect permissions. `launch: bool = false`, `cancel: bool = false`, and `from_phase: bool = false`; omission leaves the entry read-only.

At startup the server validates each experiment with the same loader the CLI uses, computes a content hash, initializes the private `state_dir/runs` and `state_dir/logs` directories, and refuses an unusable state path, invalid configs, suites, configs that violate the [storage and path requirements](#paths-and-the-working-directory), or two entries that resolve to the same experiment output namespace or Optuna study namespace - one engine experiment is governed by exactly one catalog ID. The loaded registry is frozen for the server's lifetime; catalog edits take effect only after restart. On launch, the server verifies that the config still matches its startup hash, records the launch-time policy, and hands the detached runner a per-run config snapshot. That drift check runs only on `inspect_experiment` and `launch_run`; run reads and the policy applied to historical results follow the contracts under [run state and recovery](#run-state-and-recovery) and the [security model](#security-model).

A run freezes cancellation permission and sampled-parameter visibility at launch. Run-scoped winner visibility is the intersection of launch-time and current `visible_params`: later catalog policy may revoke values but never expose values the launch did not authorize. A missing current policy, or missing or unreadable launch authority, exposes none. A winner carried from an earlier generation uses the later generation's launch policy in that intersection. Cancellation likewise requires launch-time and current permission while the catalog entry exists; removing the entry leaves the launch-time cancellation permission in force so a live runner is not stranded.

Give every catalog its own `state_dir`. Run handles carry only the bare entry ID, so two catalogs sharing one state directory share one ID namespace: same-ID entries would read each other's runs, and each catalog's `max_concurrent_runs` would be evaluated against the combined live set.

The server speaks JSON-RPC over stdio; all logging goes to stderr.

### Paths and the working directory

`storage: auto` satisfies the persistent absolute-storage requirement when paired
with MCP's required absolute `workdir`. It selects `study.db` inside the experiment
namespace, or `study.journal` when any phase has parallel jobs. The existing
provenance, sampler-seed, and non-resumable-acknowledgement requirements apply.

`state_dir`, `config:`, and `experiments[].cwd` paths in the catalog resolve against the catalog file when they are relative. MCP experiment configs must use absolute `workdir` values and non-empty absolute SQLite/Journal storage paths so server restarts, wrappers, IDE launches, and desktop clients monitor the same local-node artifacts and Optuna studies. SQLite filenames keep literal `~` components, so a SQLite URL beginning with `~/` is relative and is rejected; Journal paths retain their documented home expansion. External RDB storage is rejected for MCP because the current cleanup, stale-trial reaping, and GPU lock semantics are same-host only. The detached runner always runs with the catalog entry's frozen `cwd`; omission defaults to the registered config file's directory, while `init-catalog` writes `cwd: "."` so a catalog scaffolded at the project root preserves project-relative trainer commands. The runner *process* is spawned in `state_dir` and receives that `cwd` as an argument it enters only after its own process identity is durable, so interpreter startup never reads the experiment's directory (see [security model](#security-model)). Relative paths inside `trial_command` are trainer-owned shell behavior; PhaseSweep does not parse or rewrite commands. When the experiment's `execution.cwd` is unset, the trainer inherits the catalog entry's frozen `cwd`; the registry materializes that effective path for fingerprint and drift comparisons, and the generation config snapshot freezes it. When `execution.cwd` is configured, it must be absolute. Set it when CLI and MCP launches must share one persistent study regardless of launch directory; the catalog entry's `cwd` then affects only the runner process itself.

### Concurrency and single-GPU hosts

`max_concurrent_runs` (catalog top level, default `1`) caps how many sweeps run at once across **all** experiments. The default of `1` suits a single-GPU host: each sweep's trials use the GPU, so a second concurrent sweep would contend for the device and slow both down. A `launch_run` that would exceed the cap returns up to five blocking run IDs; await one directly and retry the refused launch after it becomes terminal, or ask the user before cancelling it. Raise the cap on multi-GPU hosts where independent sweeps can run side by side.

Launch also refuses when a handle is malformed or per-run evidence survives without a readable handle. In that state the server cannot prove whether the missing run still consumes capacity; an operator must inspect and repair the state directory before another launch. Historical read tools remain available for every healthy handle instead of failing the whole catalog.

The cap counts MCP-launched runs recorded in `state_dir`; it does not count a concurrent CLI `phasesweep run` on the same host. CLI and MCP runs are still coordinated by the runtime locks described in [runtime behavior](runtime.md#concurrency-model).

## Tools

| Tool | Inputs | Effect | Returns |
| --- | --- | --- | --- |
| `list_experiments` | optional `limit` (1-100; default 50), `cursor` | read | catalog IDs, description, phase names, metric name, goal, objective-evidence assurance flags, authorized capabilities, `total_count`, and `next_cursor` |
| `inspect_experiment` | `experiment_id` | read | metric and objective-evidence assurance flags, capabilities, and per-phase name, terminal-attempt target (`n_trials`; COMPLETE, FAIL, and PRUNED all count), sampler, inherited phases, and search-space *keys* (not ranges); a changed config is a tool error |
| `get_latest_run` | `experiment_id` | read | the newest durable run, selected by launch timestamp with a stable tie-breaker, or `found: false` |
| `get_run_status` | exactly one of `experiment_id` or `run_id` | read | dense cumulative state counts plus explicit before-run and this-run progress, computed remaining attempts, storage-read availability, winner presence, result provenance, publication integrity, the represented result's own metric semantics and phase plan with a config-drift flag, run state, elapsed time, and a safe actionable failure category when terminal; terminal run ID reads normally use a frozen snapshot and fail closed to an unavailable placeholder if finalization failed |
| `await_run` | `run_id`, optional `timeout_seconds` (5-600; default 20) | read (waits) | the `get_run_status` payload plus `changed` and `reason` (`recovery_required` / `terminal` / `phase_completed` / `timeout`) |
| `get_run_results` | exactly one of `experiment_id` or `run_id` | read | objective metadata as the represented generation recorded it, result provenance, publication integrity, label provenance and a config-drift flag, declared/winner counts and missing phases measured against that generation's own phase plan, all-phases completeness, safe terminal failure context, and per-winner source, generation, promotion context, trial number, metric, policy-filtered sampled params, gate status, and partial-winner status |
| `launch_run` | `experiment_id`, optional `from_phase` | spawn detached | `{run_id, experiment_id, state}` |
| `cancel_run` | `run_id` | signal | `{run_id, state, cleanup_confirmed, recovery_required}` |

MCP annotations mirror the effects in the table. Permissions, closed input schemas, and safety gates are enforced server-side even when a client ignores those hints.

Every structured tool result also carries `next_action`, containing the normal next tool name when another automatic workflow step is safe and `null` when the workflow is complete or user/operator input is required. An experiment-scoped `get_run_status` with no live run returns `get_run_results` when phase winners exist. `inspect_experiment` always returns `null` there: only the user may authorize a launch, so the server never proposes `launch_run` as an automatic next step.

### Reading status responses

Save the top-level `run_id` returned by `launch_run` and use it for `await_run`,
`get_run_status`, and `get_run_results`. In status and await responses, process
state is **`run.state`** and recovery is **`run.recovery_required`**. A successful
`get_latest_run` lookup also nests these fields under `run`; when `found` is
false, `run` is null. Experiment-scoped status can likewise have `run: null`
when there is no tracked run. `launch_run` and `cancel_run` return `state` at
the top level, so their shape should not be reused to parse status.

For example, a terminal `await_run` response contains these fields (abridged):

```json
{
  "run": {
    "run_id": "example-run-id",
    "state": "succeeded",
    "recovery_required": false,
    "failure": null
  },
  "publication_integrity": "ok",
  "changed": true,
  "reason": "terminal",
  "next_action": "get_run_results"
}
```

Follow the [agent workflow](../src/phasesweep/mcp/agent_prompt.md) for launch, monitoring, reconnection, and result retrieval.

### Run state and recovery

A launched sweep runs as a detached background process in its own session, so it survives the agent's tool call and a server restart and can be cancelled as a group. `get_run_status` reports `running` / `succeeded` / `failed` / `cancelled`. `await_run` waits without preventing cancellation or other MCP calls. Its 20-second default fits clients with a 30-second tool-call deadline under ordinary status-read latency; request a longer value only when the client allows it, and repeat the bounded call while the run remains active. It never starts another read predicted to finish after its deadline and waits out any remaining budget before returning, but a filesystem or storage read already running in its worker thread cannot be safely preempted and may itself finish after the requested timeout. `from_phase` requires every preceding winner to exist and satisfy the current fingerprint and completion policy; an incomplete timeout winner is accepted only while that phase still sets `allow_incomplete_on_timeout: true`. `launch_run` performs the authoritative readiness check before spawning.

`cleanup_confirmed` on `cancel_run` is tri-state:

- `true`: cleanup is known safe, either from terminal evidence after the runner group exited or because the handle belongs to an earlier host boot and no process from that boot can survive.
- `false`: cancellation was attempted but could not confirm cleanup.
- `null`: the run was already terminal and no cancellation was attempted.

Use `recovery_required` as the decision field: when true, stop monitoring and report the uncertain launch, cleanup, or interrupted result finalization to the user. `await_run` returns immediately with `reason: recovery_required`; status and latest-run payloads expose the same flag in their run metadata. Status checks refresh the run metadata after their final snapshot probe, so cleanup or snapshot recovery discovered during the read is reported immediately. After successful operator recovery, this flag becomes false while the terminal `failure`, including its original `remediation`, remains frozen as history; do not repeat recovery based on that historical text. A launch handoff observed mid-transition can still settle when the runner finishes registering itself, but cleanup uncertainty and interrupted finalization require operator action.

While confirmed recovery holds a run transition lock, status and launch-capacity checks keep an unresolved dead run counted as running without waiting for evidence reconciliation. They reevaluate cleanup once recovery releases the lock.

Terminal run metadata includes a path-free `failure` object with `code`, `stage`, `retryable`, `actor`, and a canned `remediation`. Cleanup uncertainty is the actionable outer failure and may retain the safe trainer, storage, or cancellation category under `cause`. Only the outer fields control what happens next; the diagnostic cause retains its original actor and retryability and can therefore differ. External experiment-lock contention is `experiment_busy`, a retryable agent-owned preflight outcome. When a prepared run has no last-success pointer, or a valid pointer names another generation, recovery records `publication_not_committed`, a retryable agent-owned execution failure: report that recovery could not confirm this run as the current publication and start a new run only if the user still wants publication. A definitively missing or replaced local trial for a published phase, including loss in a nonempty stale ledger, is `published_study_missing`, an operator-owned, non-retryable preflight refusal: restore the original ledger and study or use a new experiment identity. It claims no generation and requires no process recovery. Genuine storage inspection failures retain cleanup uncertainty because earlier attempts cannot be accounted for. An attempt-registry write failure is `storage_unavailable` but not retryable under the unchanged persistent config: it durably aborts the accepted trial target, so remediation requires restored workdir access plus a higher supported `n_trials` target, or a new experiment name. Raw exception messages remain operator-only because they can contain storage details or filesystem paths. A refusal after generation claim has its own frozen generation snapshot with zero attempts owned by that generation. A refusal before generation claim freezes the declared phase shape with `represented_generation_id: null`, unavailable trial data, and an explicit private terminal reason. If freezing completes during a live status or results read, the response uses the completed run snapshot rather than the shared-study view. Every run-scoped result fact, including publication identity and integrity, stays frozen; only config drift is recomputed against the current catalog.

`get_run_status` reports three generation IDs plus a publication flag, never conflated. For an experiment-scoped read, `current_generation_id` is the actual mutable pointer (the most recent invocation's own claim - may be failed or in progress), and `published_generation_id` is the actual validated [last-success pointer](runtime.md#output-layout). A terminal run-scoped read instead reports both pointers as that run captured them, keeping its historical result independent of later artifact-tree changes. `represented_generation_id` identifies the generation whose `winner_present`, `summary_present`, and this-run trial counts the payload shows: normally the queried `run_id`, or `published_generation_id` when querying by `experiment_id`. `get_run_results` carries the same `represented_generation_id` beside its winners and completeness fields, including while an experiment-scoped query is temporarily pinned to a live run, so `result_source: current_shared_study` is never the only clue to which generation it represents. A config-only unavailable placeholder has no represented generation. `is_published` records whether `represented_generation_id` equaled `published_generation_id` in that same live or frozen view; a run whose own publication failed still returns its own unpublished winners and this-run counts with `is_published: false`. A workdir written before generation metadata existed has no generation IDs to report and leaves all three `null`, yet its compatibility winner file is a published result: there `is_published` is true and agrees with the `winner_present` reported beside it.

`get_run_status` and `get_run_results` carry `publication_integrity`, while status also reports per-phase study availability; [the runtime inspection guide](runtime.md#inspection-commands) defines those states. Experiment-scoped reads validate the live artifact tree; terminal run snapshots preserve the capture-time verdict. The tools omit winner data when publication integrity is `failed`, `permission_denied`, or `unknown`, while detailed validation messages remain operator-only in `phasesweep status` and `phasesweep show-winners`. Follow the run's `failure` and `recovery_required` fields.

Both tools report a published result under the semantics *it* recorded, never under an edited config's. The metric name, goal, and objective-evidence flags come from the represented generation's own summary; `get_run_results` measures `declared_phase_count`, `missing_phases`, and `all_phases_have_winners` against that generation's phase plan, and `get_run_status` reports the same plan as `result_phase_plan` beside the current config's `phases` progress list. `result_context` says which config supplied those labels (`represented_generation`, or `current_config` when nothing has published and for pre-manifest layouts that recorded no semantics; under `represented_generation` the objective-evidence flags still fall back to the current extractor's when the summary recorded none, so treat them as unproven there). `published_config_matches_current` is `true` when the config behind the result still matches the one a further run would execute, `false` after a semantic edit, and `null` when undeterminable - including when a live run's experiment has been removed from the catalog; never read `null` as "no drift". A phase renamed after publication is the case that used to mislead: results enumerated the *new* names, so a healthy publication reported zero winners with the new name listed as missing, and status showed `is_published: true` beside a phase with no winner.

Cleanup confirmation comes from the engine shutdown handler after it terminates active trial process groups through the stale-recovery cleanup path. If a spawned runner disappears without terminal status, cancellation cannot observe status, or shutdown reports unconfirmed trial cleanup, the server writes a marker that keeps the run live and counted against [`max_concurrent_runs`](#concurrency-and-single-gpu-hosts). Timeout, stale sweeping, and retry do not clear it, so a runner killed by SIGKILL or the OOM killer requires operator recovery. A recorded boot ID from an earlier boot is the exception: it proves the runner and descendants are gone, derives a failed run with confirmed cleanup and no recovery requirement, and prevents signals from targeting reused process IDs. Older handles and hosts without a boot ID retain the recovery requirement. Normal shutdown tears down trial groups, and the stale reaper handles uncertain trainer leftovers before later launches.

Cleanup and result-finalization recovery are operator-only. After inspecting the host and restoring any unavailable storage ledger, run `phasesweep mcp recover-run --state-dir <state_dir> --run-id <run_id>` to verify the saved config snapshot hash and report the cleanup or terminal-snapshot actions that recovery would attempt. This preflight is observational: it sends no signals, writes no state, and requires an existing recognizable state directory. Repeat with `--confirm` to acquire the same experiment locks as a normal run before performing any required process-group cleanup, reconciling the experiment-level active-attempt registry (including attempts from renamed phases or unavailable current storage), reaping stale `RUNNING` trials, accepting cleanup evidence only from the requested run's generation or an older attempt that run's terminal report explicitly named, consuming terminal cleanup evidence so it cannot clear a later run, recording cleanup recovery evidence, and finalizing a snapshot that the runner already captured. A diagnosed ownership-storage failure before a generation was claimed has no trials of its own: successful runner cleanup and registry/study inspection resolve that failure without requiring trial-level cleanup evidence from the unstarted generation. The runner saves its `from_phase` resume point in terminal status. Recovery checks restored storage against the published trial identities required from that phase onward, using the same check as ordinary launch; earlier skipped phases may still use their validated saved winners. Deleting the damaged ledger or substituting an empty or unrelated study cannot satisfy that check; an experiment that has never created a study may still recover after its original empty storage is restored. Failures after generation allocation retain the trial-evidence requirement. A dead spawned runner that wrote no status is recorded as failed only after cleanup is confirmed; its missing historical snapshot is never rebuilt. A hard exit around publication instead leaves a durable prepared snapshot: recovery marks an exact authenticated last-success match published, records an absent or authenticated different pointer as a failed run whose publication could not be confirmed, and fails closed when the publication is invalid or unreadable. It never reconstructs either result from current study state. Transactional pre-spawn preparation is recovered automatically on the next launch when its kernel lease is no longer held. It can also be inspected with `recover-run` and removed with `--confirm` under the launch lock, including when the abandoned preparation has a persisted `launching` handle; a held lease or acknowledged runner prevents this removal. Both automatic and confirmed recovery preserve any captured runner log as `state_dir/logs/<run_id>.log.recovered` before removing the launch reservation. This archive does not count as active-run evidence or block later launches; manual preflight and confirmation name the log paths, and automatic recovery records the archive path in the server log. A legacy lone config snapshot with no other run evidence remains removable through `recover-run --confirm`; any additional evidence fails closed. Lock contention aborts recovery before it sends a signal or changes study state. MCP deliberately has no tool for these operator actions.

When a `run_id` is supplied, live status is read through that run's saved config snapshot, so catalog edits after launch cannot redirect monitoring. Terminal finalization is ordered:

1. After validating the generation summary and artifact graph, the engine requires the detached runner to capture and durably persist the exact path-free result as `result_snapshot_state: pending` and `result_publication_state: prepared`. The winners come from the engine's own selected outcome, while counts and running-attempt identities come from one tolerant storage read. A failure here prevents the last-success pointer from advancing.
2. The engine commits the authenticated last-success pointer, then asks the runner to mark that prepared generation `committed`, updating its per-phase study-availability verdicts from the committed summary. A hard exit on either side leaves the same prepared snapshot on disk; `recover-run` decides the outcome from the authenticated pointer state and generation without consulting a later catalog or study.
3. The terminal callback adds the process outcome and cleanup evidence. Failure paths that never reached publication capture their unpublished generation or a config-only pre-generation placeholder under the same experiment lock.
4. The runner finalizes only the already-frozen object and records the snapshot state as `complete` or `failed`. Writes retry transient OS failures briefly; a serialization error is not retried. If an `OSError` prevents the complete transition, the durable pending record remains recoverable rather than being downgraded to failed.

A durable `failed` snapshot does not change an exit-zero engine outcome: the run derives `succeeded`, while result reads fail closed with the finalization state. While finalization is `pending`, the run remains `running` and counts toward the launch concurrency limit. The shared-state read itself already happened under the engine lock; the pending state covers durable serialization of that immutable object. Later resumes cannot rewrite reads backed by a completed snapshot. Operator cleanup repair keeps an already-complete snapshot readable until its replacement is durable; interrupted evidence writes or finalization leave the frozen result intact and keep a same-boot run reserved for retry. A completed result snapshot is self-contained and remains readable if the separate config snapshot is missing or corrupt; live reads still require the intact saved config. Once cleanup evidence and a complete snapshot are settled, a repeated confirmed recovery makes no new writes.

If a runner dies during a live results read and no frozen snapshot is ready, `get_run_results` returns an unavailable, winner-free shape; `get_run_status` or `await_run` reports whether operator recovery is required.

Terminal run reads fail closed if the result snapshot is missing or malformed; they never substitute the experiment's mutable shared-study results. With an intact config snapshot, the tools still return the terminal run state and a structured `result_snapshot_unavailable` failure, with `result_source: terminal_snapshot_unavailable`, `publication_integrity: unknown`, and a config-only phase shape whose zero counts are explicitly marked unavailable. This also applies to a spawned runner lost across a reboot before writing terminal status: it frees capacity without borrowing later study results. Operator recovery can apply confirmed cleanup evidence to an already-captured snapshot, but a missing historical snapshot is unrecoverable because the current study may include later work. If the run's original experiment ID is no longer in the active catalog, winner parameter values are fully redacted; cancellation falls back to the permission recorded at launch so a live runner is never stranded (`recover-run` still refuses while the runner is alive).

## Resource and prompt

Clients that support MCP resources can attach `phasesweep://catalog`. It returns the first catalog page as compact JSON using the same path-free payload as `list_experiments`. Agents should still call `list_experiments` when they need pagination or autonomous discovery.

The server supplies the packaged [agent instructions](../src/phasesweep/mcp/agent_prompt.md) during initialization. Clients that support MCP prompts can also request them as `phasesweep_run_and_monitor`.

The installer can place the same instructions in supported project instruction files; see [what the installer changes](mcp_setup.md#what-the-installer-changes).

## Security model

The server prevents an agent from:

- changing `trial_command`, `env`, `storage`, `workdir`, search spaces, samplers, gates, or safety waivers - no tool accepts a config or these fields;
- referencing a config by path - tools accept catalog experiment IDs or server-minted run IDs according to their scope; an unknown ID is a clean error;
- reading raw trial artifacts, metric histories, trainer output, or rendered commands - no tool returns log text, because commands and trainer output can carry secrets or PII. Operator-visible log locations are listed under [inspecting runs](#inspecting-runs);
- double-launching (rejected by a run-handle check and ultimately the engine's same-host lock), deleting runs, or corrupting state.

Outbound payloads are built only from path-free typed views. Catalog listings are count-paginated with `limit` and `next_cursor`. Metric descriptors label objective-evidence assurance. `get_run_results` reports the concrete `winner_source`, whether it belongs to the represented or a prior generation, and safe promotion context; it returns sampled `params` under the [catalog visibility policy](#the-catalog) and omits composed `effective_overrides`, which can include operator-authored fixed or inherited values such as private dataset IDs, paths, or tokens. Keep secrets, access tokens, private paths, dataset IDs, hostnames, or other sensitive values out of searchable parameter choices unless you deliberately expose them. A backstop converts any unexpected exception into a path-free operator-directed error rather than leaking a traceback; recoverable domain errors are surfaced as MCP tool errors for model self-correction.

### Objective-evidence assurance

Metric descriptors report the configured extractor's evidence guarantees:

| Flag | When true | Meaning |
| --- | --- | --- |
| `attempt_location_scoped` | All extractors | The trial directory or W&B run ID is unique to this attempt. |
| `attempt_identity_bound` | `json_envelope` | Evidence contents identify the attempt and trainer input, as required by the [result envelope](config.md#result-envelope). |
| `source_identity_keyed` | `wandb` | The source is looked up by the injected `WANDB_RUN_ID`, which equals the attempt ID. |
| `objective_name_bound`, `split_bound`, `evaluation_policy_bound` | `json_envelope` | The reported objective name, split, and evaluation policy are checked against the config. |
| `checkpoint_declared`, `expected_step_declared` | The corresponding envelope setting is configured | The config pins that checkpoint or step. |
| `checkpoint_value_bound`, `expected_step_value_bound` | The corresponding envelope setting is configured | The reported value is checked against the pinned value. |

Location scoping alone does not validate a file's contents: a log copied into the wrong trial directory can still match. W&B keys the source by identity; a JSON envelope also validates identity fields inside the evidence.

Between `exec` and the runner's first durable handle write, the server cannot yet name the process it created, so code that runs in that window could leave the recorded process group and make later cleanup confirmation false. The runner is therefore spawned from `state_dir` with `-P` (no cwd or script directory on `sys.path`), `-s` plus `PYTHONNOUSERSITE=1` (no user site directory), and with `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, and `PYTHONEXECUTABLE` removed from its environment. A project-local `phasesweep` shadow package, a project `sitecustomize.py`, or an injected module therefore cannot execute during interpreter startup. Every other environment variable is inherited unchanged.

MCP restricts tool inputs and outputs. Training subprocesses run without a sandbox, with the authority of the human-authored command, just as they do through `phasesweep run`.

## Inspecting runs

Run handles and per-run logs live under `state_dir`:

- `state_dir/audit.jsonl` - launch/cancel side-effect audit records.
- `state_dir/.launch.lock` - persistent lock file whose nonblocking lease serializes the launch-capacity check and spawn across servers sharing this state directory.
- `state_dir/runs/<run_id>.json` - the run handle.
- `state_dir/logs/<run_id>.log` - captured runner stdout/stderr (operator-only).
- `state_dir/logs/<run_id>.log.recovered` - preserved stdout/stderr from an abandoned transactional launch; ignored by active-run inventory (operator-only).
- `state_dir/logs/<run_id>.status.json` - the recorded terminal cause, explicit result-snapshot finalization state, and, when capture succeeds, the path-free status/winner snapshot used for stable run ID reads.
- `state_dir/logs/<run_id>.config.yaml` - the exact config snapshot executed by the runner (operator-only; may contain command, storage, env, and overrides).
- `state_dir/logs/<run_id>.launch.lock` - transient kernel lease held only until the child receipt is acknowledged; a leftover unlocked lease proves abandoned pre-spawn preparation.
- `state_dir/logs/<run_id>.cleanup_uncertain.json` - server-owned marker that keeps a cleanup-uncertain run counted as live.
- `state_dir/logs/<run_id>.cleanup_recovery.json` - operator recovery evidence written by `phasesweep mcp recover-run --confirm`.

The per-run `.log` captures the detached runner's control-process stdout and stderr, not the trainer's output. Each trial's rendered `command.txt`, `stdout.log`, and `stderr.log` live in its trial directory under the experiment workdir; the engine's durable `run.log` lives at that experiment root. See the [runtime output layout](runtime.md#output-layout).

`audit.jsonl` contains best-effort append-only records for launch and cancel side effects: timestamp, local stdio actor, server session ID, tool name, bounded safe arguments, resolved IDs, outcome, safe error details, and state-transition summaries. Read-only catalog, status, await, and result calls are not logged. Audit records do not include tool result payloads, trainer logs, commands, config paths, storage URLs, environment values, sampled winner params, or effective overrides.

Status uses the SQLite and Journal inspection behavior described under [inspection commands](runtime.md#inspection-commands). Journal status still replays a complete captured snapshot, so avoid tight `get_run_status` polling on very large studies and use bounded `await_run` calls for monitoring.

### Pruning terminal run history

Prune only terminal runs between campaigns, never active or recovery-required runs. Treat one run's history as a unit: `state_dir/runs/<run_id>.json` and every matching `state_dir/logs/<run_id>.*` file. Removing the handle makes the run undiscoverable through latest-run and run ID lookups; removing its sidecars makes run-specific status and winner reads unavailable. Archive the complete set when that history still matters.

### Long-running servers

The server is built to stay up across multi-hour sweeps. Exited detached runners are reaped during status and live-run scans, the server does not hold per-run log file descriptors open, and run state is derived from disk on each query rather than kept in memory, so a server restart re-discovers live runs from their handles. A runner that exits just after a scan can remain a zombie until the next scan. Run history accumulates one small file set per launch.

Run IDs contain a full UUID. Under the launch lock, preparation no-replace atomically creates and strictly fsyncs a transient kernel lease, exact config snapshot, and complete `launching` handle; a partial failure rolls back only that owned preparation. The lease stays locked across `Popen`. The child durably changes the handle to `spawned`, writes a receipt byte, and blocks until the server validates that exact persisted identity and acknowledges it. Only then may the child enter the experiment directory or launch training. A server hard exit before `Popen` releases the lease, so the next launch automatically removes the provably abandoned preparation. A hard exit after `Popen` closes the acknowledgement pipe; the child records a failed terminal outcome and exits without entering the experiment. Once acknowledged, PID, PGID, `/proc` start time, and boot ID identify the runner for cleanup. If later launch bookkeeping fails, death of that runner group alone does not prove separately sessioned trial groups have stopped; the run stays reserved until a runner terminal report or operator recovery confirms cleanup. Terminal `status.json` files and legal handle updates use descriptor-relative atomic replacement, so readers do not observe torn replacements and symlinked path components are rejected before any target is truncated or chmodded. Existing private state must remain owned by the current user with directory mode `0700` and file mode `0600`; unsafe reuse fails closed without changing it.
