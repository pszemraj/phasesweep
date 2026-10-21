# Runtime behavior

PhaseSweep runs one ordered Experiment on a Linux host for a trusted researcher or team. It uses same-host file locks and process groups to serialize a persistent experiment and clean up trainer subprocesses. Every trial's process identity records the PID namespace its numbers belong to, so `/proc` must be the procfs of the orchestrator's own PID namespace, with that namespace's PID 1 readable; the trial launcher refuses to start a trial otherwise. GPUs are optional; the core CLI can run CPU-only work. The optional MCP server additionally requires Linux `/proc` process start times for detached-run cancellation and recovery.

## Output layout

For `workdir: ./runs` and `experiment: demo`, PhaseSweep writes under `runs/demo/` beside the config file:

```text
demo/
  .gitignore
  artifact_root_binding.json
  run.log                         # progress stream: phase headers, TRIAL and RERUN lines
  study.journal                   # storage: auto
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
    trials.csv
    trial_00000__generation_<id>__attempt_<id>/
      attempt_lifecycle.json
      process_identity.json
      trainer_config.yaml       # yaml_file only
      overrides.json            # json_file only; nested trainer_config with overrides applied
      overrides_resolved.json
      command.txt
      stdout.log
      stderr.log
      result.json               # when produced by the trainer
```

The root `summary.yaml` and per-phase `winner.yaml` files are convenience projections. Each summary phase entry retains its winner's `observed`, `repeats` and `confound` blocks, so the reduced metric can be read beside its spread, observations and comparison verdicts. `last_successful_generation.yaml` points to the immutable generation that authoritatively represents the latest successful result. A failed invocation does not replace that pointer. Each generation embeds its carried winners; result reads do not reopen older publications. Selection and resume verify retained trial evidence.

`trials.csv` lists every trial with its sampled parameters and user attributes, including the `observed:<name>` values that [`observe`](config.md#objective-evidence-constraints-and-gates) recorded. Its `state` column reads as in `status --trials`, `INTERRUPTED` and `INFEASIBLE` included, and a rerun's `user_attr:phasesweep_rerun_of` names the [interrupted trial](#interrupted-trials-rerun) it reran. A phase with [`repeats`](config.md#repeats) adds a `repeat_group` column. Each `winner.yaml` carries the selected trial's `observed` map, which is empty when the experiment declares none or the trial recorded none, and a `repeats` block that is `null` outside a repeats phase.

`config.snapshot.yaml` is the executed configuration, including materialized defaults and the resolved invocation context. `reproducibility.json` is the shareable generation identity record. Trial directories retain local evidence and attempt lifecycle information used for supported continuation and winner validation.

MLflow-backed experiments additionally retain native run bindings in each trial directory and mutable delivery state and worker cleanup receipts under `tracking/<generation-id>/`. These records support upload retries and recovery and are separate from the committed scientific manifests. Disposable supervised-worker scratch uses `~/.cache/phasesweep/<backend>/workers/`, or the corresponding `XDG_CACHE_HOME/phasesweep` root; the default home comes from `Path.home()`. Each operation removes its temporary directory. Host locks use a [separate location](#storage-and-locks). See [MLflow tracking](mlflow.md) for ownership and offline replay.

### Following a run

`run.log` is the process-free progress stream. `tail -f <workdir>/<experiment>/run.log` follows a run from any shell, with no MCP server, no lock, and no PhaseSweep process of its own. Each phase that runs logs a header line, and every trial that ends adds one `TRIAL` line, in the order trials end:

```text
2026-01-01 12:00:02 I phasesweep.engine.phase phase=lr study=demo::lr finished=2 remaining=4 reruns=1 n_jobs=1
2026-01-01 12:00:02 I phasesweep.engine.phase phase=lr RERUN trial=3 of=2 reason=orchestrator interrupted by signal 2
2026-01-01 12:00:03 I phasesweep.engine.phase phase=lr TRIAL trial=3 state=COMPLETE outcome=success value=0.3 duration=0.4s rerun_of=2 (3/6 finished)
2026-01-01 12:00:04 I phasesweep.engine.phase phase=lr TRIAL trial=4 state=FAIL outcome=failure value=none duration=0.5s (4/6 finished) reason=non-zero exit code 1; stderr: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
```

The header counts the phase's `finished` trials, the trials still `remaining` to its `n_trials`, and the `reruns` it owes for [interrupted trials](#interrupted-trials-rerun); each rerun it schedules gets a `RERUN` line naming the trial it reruns and the first line of that trial's cause. `state` is the trial's Optuna state, except that a COMPLETE trial that violated a constraint or the metric's `min`/`max` bounds reads `INFEASIBLE`, as selection treats it, and an interrupted trial reads `INTERRUPTED`. `outcome` is `success`, `failure`, or `interrupted`, and a trial whose outcome recorded a cause adds `reason=` with its first line. A rerun's line adds `rerun_of=N`, the trial it reran. A trainer that exits nonzero records `non-zero exit code N`, followed by `; stderr: ` and the last non-empty line of the trial's `stderr.log` when it has one, so a crash is named without opening the trial directory. The line is read from the last few KiB of the file, stripped of control characters and cut at 200 characters with a trailing `...`. The same text is the `user_attr:phasesweep_failure_reason` column of `trials.csv` and the `cause` of the trial's recorded outcome. It is whatever the trainer printed, so it stays in these operator-owned records and no MCP tool returns it. `value` is the metric, or `none` when the trial produced none. `(k/n finished)` counts the finished trials the phase's study holds, resumed ones included and interrupted ones left out, against `n_trials`. A phase with [`repeats`](config.md#repeats) adds `repeat=<key>=<value>`, which names the seed or other repeat value that trial ran. The line carries no params; the `WINNER` line closing each phase does. A fatal objective error logs `FATAL OBJECTIVE ERROR` and skips the trial's `TRIAL` line. So does a trial a shutdown signal interrupts under `n_jobs: 1`; the next run's `RERUN` line names it.

### Repeat groups

A phase with [`repeats`](config.md#repeats) ranks complete configurations by the configured reducer over their member metrics. Each member must be COMPLETE, feasible and inside the metric's bounds. Exact ties go to the group with the lowest member [draw position](#interrupted-trials-rerun). The highest member trial number names the group and becomes the winner's `trial_number`; [invariant 27](invariants.md#27-a-repeat-group-is-fixed-once-complete-and-its-last-member-names-it) defines group identity across interruptions and top-ups.

The winner's scalar fields (`trial_number`, `generation_id`, `attempt_id`, `objective_provenance`, `trainer_input`, `constraints`) describe that last member. `metric` is the reduced value, `observed` holds each name's mean over the members that recorded it, `params` and `effective_overrides` omit the repeat key, and `repeats` records `key`, `reduce`, `spread` (`n`, `mean`, sample `sd`, `min`, `max` of the member metrics) and `members`: per member its `trial_number`, `repeat` value, `metric`, generation and attempt IDs, objective provenance, trainer input, constraints and observed values, in trial-number order. Publication validates this block against its members and the generation's config snapshot; selection and `--from-phase` verify every member's trial evidence directly. In `trials.csv`, `repeat_group` is the highest trial number of the group each trial belongs to so far. `status CONFIG --trials` follows the member table with one summary per group: params without the repeat key, the measured count over the expected repeat count, reducer, eligible aggregate, mean, sample `sd`, range, eligibility and published-winner mark. Statistics include completed infeasible measurements, so a bad seed remains visible; failed, missing, interrupted and replaced attempts contribute no value. A partial or ineligible group has no ranked aggregate, and fewer than two measurements have no `sd`. Eligibility uses the current config's bounds, while the winner mark still names the publication.

The group table also reports each observation's mean as `observed:<name>` and its contributor count as `observed_n:<name>`. Each name counts only completed current members that recorded a finite value for it. For example, two held-out readings in a three-member group have `observed_n:heldout = 2`; the group's objective count remains three. Missing observations do not become zero-valued measurements.

A failed member spends its draw. Seeded random moves on to another configuration slot when its target grows; a persistent grid can retry a failed cell when `n_trials` is raised beyond the full matrix, and the replacement can complete its original repeat group. A completed infeasible member is never retried, so its group cannot rank. An interrupted member reruns. A phase cut short by an accepted timeout ranks only the groups it completed; a failure abort ranks nothing and requires a raised target to continue. [Repeat configuration](config.md#repeats) determines which samplers permit a larger target. When no group is complete, the phase fails with `NoFeasibleTrialError`, which counts the incomplete groups and names any metric bound that kept members out.

### Confound verdicts

Each phase's `winner.yaml` carries a `confound` block: advisory verdicts on the population that selection ranked to pick the winner, computed once and frozen at selection time. It never blocks selection or publication and has no YAML switches or threshold settings. The authoritative copy belongs to the published generation; [invariant 26](invariants.md#26-a-confound-verdict-never-joins-a-fingerprint-or-gets-read-as-clean) describes its integrity, fingerprints, top-ups, and inheritance. In a repeats phase the block also lists `ranked_groups`, each complete group's member trial numbers, and its checks compare groups rather than trials.

The block runs three checks, each `ok` (checked, nothing found), `heterogeneous` (checked, found a potential confound, with evidence), or `n/a` (could not be checked - never treat this as clean):

- `evaluation_point`: whether the ranked objectives share a training-progress position. Envelope and W&B objectives use the configured `evaluation_axis`, such as optimizer step or tokens; MLflow objectives use the selected native observation's step. Positions within 1% of the highest count as the same point, and the exact positions are recorded either way. See [W&B evidence](config.md#wb-evidence) for aggregate positions. Distinct positions are expected when the phase searches training length such as epochs or `max_steps`; otherwise the trials were not trained to one budget. `n/a` for `json` and `log_regex` extractors, which report no position (only `json_envelope`, `wandb` and `mlflow` bind one, and `validate` prints which case applies); when a ranked objective has no recorded position or unreadable provenance; or when only one trial was ranked.
- `survivorship`: whether failed trials were at least as numerous as the ranked ones, so that whatever made configurations fail could have decided which ones were ranked. Failed trials include evidence-gate and extraction failures. Interrupted trials are left out, because their reruns stand in for them. Infeasible trials, which violated a constraint or the metric's bounds, are counted too but never flag: the experiment's own constraints and bounds define what may win. In a repeats phase, `ranked` counts the members of ranked groups, and feasible members of groups that never completed are counted as `incomplete_group`, which never flags either. This check always runs.
- `tie`: whether the draw-position tiebreak chose between trials with different params tied on the exact metric value, rather than between repeats of one configuration. The lowest [draw position](#interrupted-trials-rerun) wins a tie, which outside reruns is the lowest trial number. `n/a` with one ranked trial. `within_noise` is `null` without repeats. In a repeats phase `tied_trials` names the groups whose reduced value equals the winner's, and `within_noise` names the groups whose reduced value lies within two combined standard errors of the winner's (sample standard deviation over the square root of the group size, for either reducer), exact ties included; either list flags the check. Groups are named by their last member, a group with the winner's params is skipped, and the check is `n/a` with one ranked group. Two standard errors is a heuristic for "these configurations are not separated by this many seeds", not a significance test; [Noisy objectives and small budgets](config.md#noisy-objectives-and-small-budgets) covers what extra seeds cost and when they are the right tool.

Selection logs one `WARNING` naming every flagged check and its numbers. For repeat groups within noise, the selection warning includes each competing configuration's params, the winner's params, the absolute aggregate `gap` and their `combined_se`; the comparison uses twice that combined standard error. A phase that is about to build on a flagged parent winner logs its own `WARNING` before it runs. `phasesweep show-winners` prints the raw `winner.yaml`, including the full evidence behind each check.

> [!WARNING]
> `config.snapshot.yaml` is private because it can contain command and environment values. PhaseSweep writes a `.gitignore` containing `*` whenever the experiment namespace lacks one, but it never replaces an existing ignore file, so keep generated outputs, ledgers, and MCP state out of commits yourself.

### Trust boundary

Workdir artifacts use ordinary paths and permissions: new files follow umask, and replacements preserve existing modes. Logs, evidence, winners, and resolved overrides remain accessible to the operator and tooling. Process identity and lifecycle records use atomic replacement.

Private records inside the tree have these additional requirements:

| Path | Mode | PhaseSweep refuses to |
| --- | --- | --- |
| `generations/<generation-id>/config.snapshot.yaml` | `0600` | replace it unless the existing file is an owner-only regular file with no other hard links |
| `environment.json` in a trial directory, written only when `execution.record_env` is set | `0600` | replace it under the same condition |
| `attempts/`, the attempt registry | `0700` directory of `0600` entries | read it through a symlink or with shared permissions |

The two files are owner-only while their directories keep ordinary permissions. The same validated, no-follow path handling protects the runtime lock namespace and the MCP `state_dir`.

## Storage and locks

`storage: null`, or omitting `storage`, is in-memory and ends with the process; it cannot be resumed or topped up by a later invocation, and a trial it interrupts never reruns. Persistent local storage is an explicit `journal:///...` URL, or `auto`, which always resolves to `<workdir>/<experiment>/study.journal`, regardless of `n_jobs`. The journal backend supports parallel trials (`n_jobs > 1`) on one host. Any other storage value, including `sqlite:///...`, `:memory:`, and external database URLs, is refused at config load; the error names the supported forms.

PhaseSweep reads only the current artifact and study formats. When `status`, `show-winners`, or `run --dry-run` refuses an unsupported format, inspect the raw artifacts (`summary.yaml`, `winner.yaml`, trial logs, and the journal), or use the exact checkout that produced them for inspection if available. Development records may require an unreleased checkout. To launch a new run with the current checkout, use a fresh artifact root and ledger namespace.

The selected artifact root and ledger are bound to each other in both directions:

```mermaid
flowchart LR
    root["artifact root<br/>workdir/experiment"]
    ledger["local ledger<br/>journal"]
    root -->|"artifact_root_binding.json names this ledger"| ledger
    ledger -->|"each phase study names this root"| root
    other_ledger["a second ledger"] -.->|"refused: the root names another ledger"| root
    other_root["a second workdir"] -.->|"refused: the studies name another root"| ledger
```

A reused root therefore cannot combine a second ledger's trial counts with the first ledger's publication, and a reused ledger cannot publish into a second tree. Both refusals happen before any trial runs and write nothing.

Runs and confirmed MCP recovery check the tree's binding and ledger format before mutation; a failed check leaves both unchanged. Read-only commands, including `status`, `show-winners`, and `run --dry-run`, never create a missing ledger or directory. Readers skip an incomplete final journal record. The next `phasesweep run` or confirmed `phasesweep mcp recover-run` repairs it under the same inode lock every PhaseSweep append uses, after saving `<ledger>.<UTC stamp>.bak` beside it with permissions no broader than the journal's. Repair waits at most 30 seconds for an active writer; if it times out, wait for that writer to finish and retry. Kernel locks release when their owning process exits. Damage before the final record is refused as corruption. See [durability invariants](invariants.md) for the repair ordering.

MLflow delivery retries use the experiment lock and validate their saved target and generation. They can read retained trial snapshots without constructing file-backed storage or mutating the ledger. See [result delivery and recovery](mlflow.md#result-delivery-and-recovery).

Persistent phases use same-host locks. A concurrent CLI or MCP launch for the same experiment waits or fails safely according to the operation; it never merges two active orchestrators. Only PhaseSweep writes the live ledger; external Optuna writers do not participate in its lock. External database backends are not part of the runtime.

Host locks default to `.cache/phasesweep/locks` beneath the effective user's OS account home, independently of `HOME` and `XDG_CACHE_HOME`. `PHASESWEEP_LOCK_DIR` selects an existing absolute directory provisioned by the operator; PhaseSweep validates its path, ownership, and permissions without creating or changing it.

### Inspect the ledger with Optuna

The ledger is an ordinary Optuna journal holding one study per phase, named `<experiment>::<phase>`. `phasesweep status <config>` prints its real path, with symlinks resolved, as `ledger_path` (`null` for `storage: null`). Optuna's CLI reads the journal in place without writing to it, so these commands are safe while a sweep runs:

```bash
optuna studies --storage <ledger_path> --storage-class JournalFileBackend
optuna trials --study-name '<experiment>::<phase>' --storage <ledger_path> --storage-class JournalFileBackend
optuna best-trial --study-name '<experiment>::<phase>' --storage <ledger_path> --storage-class JournalFileBackend
```

Copy `ledger_path` exactly: Optuna creates an empty journal at a path that does not exist yet.

For the web dashboard, install `optuna-dashboard` and serve a copy of the ledger:

```bash
pip install optuna-dashboard
cp <ledger_path> /tmp/<experiment>.journal
optuna-dashboard /tmp/<experiment>.journal
```

> [!WARNING]
> Never point `optuna-dashboard` at the ledger itself. The dashboard can rename, delete, and write to studies, and a changed phase study no longer matches the results PhaseSweep published from it. The copy is a snapshot; copy the ledger again to see newer trials.

## Trial execution and evidence

Each trial has an attempt-scoped directory and process group. The runtime captures `stdout.log` and `stderr.log`, prepares the configured input, and supervises cleanup. YAML input identifies the complete generated configuration; JSON identifies the nested overrides file. Argparse and Hydra retain the resolved-overrides identity together with the command. Historical inputs are verified using their saved format.

W&B polling starts only after successful trainer cleanup and GPU-lease release. It uses the same durable attempt slot and process supervisor, so recovery can find the reader after abrupt parent exit, even if its phase was removed. Uncertain cleanup blocks further work. One absolute polling deadline covers worker startup, SDK construction, requests, retries, and summary visibility; phase/run deadlines can shorten it, while cleanup grace stays separate. Once the trial is accepted, a second reader in the same slot locates the objective in run history within a fresh `timeout_seconds` and the remaining phase/run budget. History gets up to 30 seconds to catch up with the summary; reading it may use the rest. A failed or timed-out history read records no position and keeps the objective, while uncertain reader cleanup stops the phase. Trainer return code and duration remain trainer measurements.

Finished W&B captures freeze only the requested numeric values and gate-presence evidence, with target, attempt, retrieval time, selected key, and consumer bindings. Published results, selection, and no-op replay need no SDK, credentials, or network reads, even after remote edits or deletion. New remote work checks SDK availability after recognized-state recovery and before accepting a larger target. Missing/invalid evidence fails a trial; a measured constraint violation remains complete but infeasible. See the [configuration guide](config.md) for scoring and environment details.

MLflow captures the selected native observation and its step together after trainer cleanup and GPU release, without a second advisory history lookup. Managed run creation happens before GPU acquisition. See [native MLflow tracking](mlflow.md#choose-run-ownership) for ownership and supervised controller operations.

## GPU isolation

By default, `gpu_policy: single_per_trial` leases one visible CUDA device per parallel trial. Discovery uses explicit phase `gpu_ids`/`gpu_devices`, then top-level `env.CUDA_VISIBLE_DEVICES`, then the controller's ambient value, then local inventory. Visibility tokens may be numeric, GPU UUIDs, or MIG tokens. Discovery reads ambient visibility even when `inherit_env` excludes it. The policy rebinds a leased device token, or preserves an empty/`-1` CUDA-disable sentinel, in each trainer. The lease controls visibility and same-host locking, not GPU memory allocation.

`validate` checks GPU inventory for every phase; `run --dry-run` checks only phases that would launch trials. These read-only checks may invoke `nvidia-smi`, refuse explicit device IDs absent from the local inventory, and warn when workers exceed available devices. They acquire no GPU leases.

`gpu_policy: whole_node` requires `n_jobs: 1` and a unique explicit device list. `gpu_policy: none` disables PhaseSweep's GPU isolation and cannot be combined with an explicit device list; parallel use requires the explicit `allow_no_gpu_isolation: true` acknowledgement. For CPU-only parallel work, make that acknowledgement deliberately rather than relying on failed device detection.

`gpu_policy: none` assigns no binding. Include ambient visibility in the trainer inheritance contract or set it in `env` if it must be retained.

## Timeouts, cleanup, and continuation

Trial, phase, and experiment timeouts are cooperative run controls. The experiment budget, `timeout_seconds_per_run`, bounds one invocation: its clock starts afresh at every `phasesweep run`, `--from-phase` resume or MCP launch, so it never counts time an earlier invocation spent. Set the per-trial timeout above the longest trial you expect. `timeout_seconds_per_trial` defaults to 259200 seconds (72 hours), so a hung trial can hold its phase that long unless a phase or run deadline arrives first; an unbounded trial (`null`) needs `allow_unbounded_trials: true`. Its clock covers supervised launch and execution after GPU acquisition, including supervisor startup, but excludes the GPU queue and post-trainer evidence. Phase/run budgets cover those stages too. Cleanup can extend beyond a deadline. Unlike phase/run budgets, the per-trial timeout participates in semantic study identity; changing it cannot reuse a populated phase.

On a trial timeout PhaseSweep terminates the supervised process group and records a terminal failed attempt after cleanup. A phase or run deadline that stops a trial in flight records it as [interrupted](#interrupted-trials-rerun) instead. The `max_consecutive_failures` threshold also stops a broken phase. The abort is recorded with the outcome that trips it, so a trial that succeeds afterwards cannot turn the phase back into a result, and a later run stays refused until `n_trials` is raised. A trial the abort stops before its trainer launched is [interrupted](#interrupted-trials-rerun), so the raised target reruns it first. Changing `max_consecutive_failures` governs new work only: it never aborts a completed phase retroactively, and a resumed phase whose failure streak already meets the new limit launches nothing until `n_trials` or the limit is raised. When a timeout and the threshold both stop the same phase short of its target, the timeout decides the outcome: with `allow_incomplete_on_timeout` the phase publishes the winner its completed trials earned, and without it the run stops as a timeout, so a retry with a larger budget does not replay the failure abort. A later run or `--from-phase` replays an accepted partial target without filling unused slots or rerunning the trials the deadline interrupted, only while the config still sets `allow_incomplete_on_timeout`; raising `n_trials` runs those reruns first. `n_trials` counts finished attempts: failed attempts consume the configured target, and interrupted ones do not.

Persistent trial targets cannot decrease. Raising `n_trials` schedules additional attempts after a durable abort or accepted partial timeout. Failure and timeout controls never create an alternate winner.

Before a supported continuation, PhaseSweep first checks the artifact-root and ledger binding, then reconciles stale active attempts, and only then verifies recorded trial evidence. `--from-phase` requires valid earlier winners only for the ancestors of the phases it reaches; reused winners keep their original source provenance. An independent arm can therefore start after an earlier arm aborted, without a winner for that unrelated arm. The run still stops when a phase fails; starting another arm is an explicit resume. Failed attempts stay spent. Raising `n_trials` beyond a grid's cardinality buys one retry per cell without a completed result; completed cells never run again, and failed-cell retries consume the extra target. Interrupted attempts rerun without consuming it. [Sampler rules](config.md#search-spaces-and-samplers) govern new draws when `n_trials` rises, and [fixed-configuration rules](config.md#phase-composition) govern retries without a search space. A launch and the dry run refuse an invalid larger target before any phase runs. A [repeats](#repeat-groups) resume completes groups by running members an interruption never started and rerunning interrupted ones; a raised grid target can also retry failed members.

A phase's study stores the canonical payload its fingerprint is the digest of, beside the digest. When a later run's config disagrees, the refusal lists up to ten changed keys as dotted paths with their old and new values, truncates long values, and counts the rest. A trainer working directory that changed because `execution.cwd` is unset is called out with the fix. A winner file records only the hash, so a skipped phase whose winner no longer matches reports the two hashes.

Each trial records `phasesweep_trainer_env`, a plain-text mapping of its tracked trainer environment, and `phasesweep_trainer_env_digest`, the SHA-256 checksum the cohort check compares. The [trainer contract](config.md#trainer-contract) defines the tracked names, pass-through exclusions, and remedies when a resume composes a different environment. The mapping stays in the ledger; `trials.csv` carries only the digest as a `user_attr:` column.

An interrupted or failed generation leaves its immutable record for inspection but does not advance `last_successful_generation.yaml`. Signals are absorbed across the final publication commit so the authority pointer is never left ambiguous.

### Interrupted trials rerun

An interrupt of `phasesweep run` is not a trial timeout. On SIGINT (Ctrl-C), SIGTERM or SIGHUP, PhaseSweep terminates the in-flight trials' process groups, records each trial's outcome as `interrupted` with the cause `orchestrator interrupted by signal N`, marks the trial `FAIL` in the ledger and exits with status 128 plus the signal number: 130 for Ctrl-C, 143 for SIGTERM. Once the signal arrives no trainer starts: a trial still waiting for a GPU slot or about to launch records `interrupted` with the same cause, and under `n_jobs` above 1 the run waits for every in-flight trial to record its own outcome before it exits. A phase or run deadline that stops a trial in flight records `interrupted` too, with the deadline as its cause, and so does a phase abort (a failure streak or a fatal error) that stops a queued trial before its trainer launched. A hard kill (SIGKILL, including the out-of-memory killer, or losing the machine) gives PhaseSweep no chance to do any of that: the trial stays `RUNNING` in `status --trials`, and its trainer can keep running, until the next `phasesweep run` recovers it as stale. Recovery terminates any surviving process group, marks the trial `FAIL` and records the outcome `interrupted` with the cause `stale RUNNING trial recovered after its orchestrator stopped`, unless the trial had already recorded a failure or a fatal error, which it keeps.

An interrupted trial reads `INTERRUPTED` in `status --trials`, `trials.csv` and `run.log`. It never counts toward `n_trials` or the `max_consecutive_failures` streak, and selection, its confound checks and repeat groups pass over it. The next run that reaches its phase adds one trial with the interrupted trial's parameters, logs a `RERUN` line for it, and runs every rerun before any new draw. A rerun is an ordinary trial with its own number, attempt and trial directory, and its `rerun_of` names the trial it reran; a rerun that is interrupted in turn gets a rerun of its own. An interruption therefore costs only the wallclock its in-flight trials had used. A phase whose partial result was [accepted at a deadline](#timeouts-cleanup-and-continuation) replays it without rerunning until `n_trials` is raised, and on `storage: null` nothing reruns, because the study ends with the process.

Every trial has a draw position. A trial that is not a rerun takes the next one, 0, 1, 2 and so on, and a rerun takes its source's. Samplers draw by position rather than trial number, so an interruption never repeats or skips a draw; a discrete space can still yield the same configuration twice. Grid and seeded random reproduce an uninterrupted run at any `n_jobs`; seeded TPE and CMA-ES do at `n_jobs: 1`, given the same trial values. [Invariant 29](invariants.md#29-a-seeded-phase-draws-by-position-so-an-interruption-never-repeats-or-skips-a-draw) gives the position contract. Grid shuffles with a fixed seed when none is set, so it reproduces unseeded too. `validate` and the dry run print this per phase as `(resumable, reproducible)` or `(resumable)`; on `storage: null`, where nothing resumes, they print `(reproducible, not resumable: in-memory storage)` or `(not resumable: in-memory storage)`.

## Inspection commands

`validate` and `run --dry-run` also perform the [read-only GPU inventory checks](#gpu-isolation); GPU leases are acquired only when trials run.

Use `validate`, `run --dry-run`, `status`, and `show-winners` to review an experiment without launching work; an ordinary `run` invocation launches trials. A dry run previews each phase from a sample: the command of a phase that inherits another that still has trials to run carries stand-in values for the parent's winner, and its `DRY RUN example command` line names the inherited keys that hold them and the parent that must publish, as in `(placeholders until 'architecture' publishes a winner: model.name)`; a key the phase fixes itself is a real value and goes unnamed. It also reads the trials the ledger already records, the way `status` does, from one capture of the journal and writing nothing, and applies a run's own checks to them in the run's order before it previews any phase: each phase reports its `finished` and `remaining` trials and the `reruns` it owes, and how many trials a run would launch, saying how many of them rerun [interrupted trials](#interrupted-trials-rerun). A phase that has reached its `n_trials`, or whose timed-out partial result was already accepted at that target, says it would replay its existing result and launch no trials, and the phases that inherit it preview with the winner that replay selects from the recorded trials rather than a stand-in. After the phase lines it prints one total, for example `DRY RUN total: would launch 30 trials (2 reruns of interrupted trials) across 3 phases (2 other phases replay)`: the launch counts those lines give, summed over the phases that launch trials, with the reruns among them named when there are any, the replaying phases counted apart and a phase `--from-phase` skips left out. A dry run that refuses prints no total. The dry run refuses what a run refuses over the recorded trials, with the run's own message: a publication whose phase study is missing or cut below the published boundary, or whose pointer is unreadable; a study that optimizes the other direction or whose failure-policy state is malformed, in any phase; a target below the one the study accepted; a fixed configuration asked for more attempts than it may retry; an upstream top-up over a descendant study already bound to the winner; a phase that aborted, or whose failure streak meets a lowered `max_consecutive_failures`, and has not had `n_trials` raised; a partial result the config no longer accepts; and a replay with no feasible trial. Several refused studies come back as one combined refusal, as a run combines them. A phase whose recorded trials cannot be read is previewed as a fresh phase, with a warning, because a run refuses a ledger it cannot read. A phase with a RUNNING attempt is counted with that attempt as a run's recovery would record it, failed when its objective recorded a failure or a fatal error and otherwise interrupted and owed a rerun, and given only a fixed configuration's attempt limit, because a run recovers that attempt first; its accepted target, abort state and partial decision are not checked, and the warning says so. The dry run compares each recorded study's fingerprint whenever its inherited values are known, and reports the same changed-key diff as a run. A phase with placeholder inherited values warns that its fingerprint comparison awaits the parent's result. It does not recover stale attempts, check the trainer environment, check each study's artifact-root claim, or re-prove replayed trials' evidence on disk, so a run can still refuse a phase on those grounds. `--from-phase` validates the required earlier ancestors and their evidence, and refuses a missing winner instead of supplying a placeholder. These reads inspect only the current-format local experiment.

`status CONFIG` prints trial counts and publication identity as YAML. `status CONFIG --trials` prints a header line and one aligned plain-text table per phase instead, one row per trial: `trial`, `state`, `value`, `params`, `duration_s`, `repeat_group` for a [`repeats`](config.md#repeats) phase, `rerun_of` once the phase holds a [rerun](#interrupted-trials-rerun), one `observed:<name>` column per [`observe`](config.md#objective-evidence-constraints-and-gates) name any trial recorded, and `winner`. `state` is the Optuna state, except that a COMPLETE trial that violated a constraint or the metric's `min`/`max` bounds reads `INFEASIBLE` and an interrupted trial reads `INTERRUPTED`. `rerun_of` names the trial a rerun reran. `duration_s` is the trainer subprocess's wall-clock runtime, and it is blank for a trial that never ran one; the `duration` on a trial's `TRIAL` line in `run.log` spans the whole trial, GPU queue and evidence extraction included. `winner` is `*` on the trials the published winner stands on: the one trial, or every member of the winning repeat group. It reflects the published generation named in the header line, so it is stale while a later run has not published, and it is blank when nothing validates as published. A table prints even when the publication is corrupt or unreadable, followed by the same error and exit status as plain `status`. For plots and the dashboard, point Optuna's own tools at the ledger as described in [Inspect the ledger with Optuna](#inspect-the-ledger-with-optuna).

> [!TIP]
> `status` and `show-winners` take no lock and never write to the artifact root or ledger, so they are safe to run while a sweep is active.

For operator-managed detached runs, see [the MCP operator guide](mcp.md). Terminal MCP run results are frozen snapshots associated with a run ID; they are not a replacement for CLI inspection of an arbitrary experiment tree.
