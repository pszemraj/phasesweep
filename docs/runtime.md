# Runtime behavior

PhaseSweep runs one ordered Experiment on a Linux host. It uses same-host file locks and process groups to serialize a persistent experiment and clean up trainer subprocesses. Every trial's process identity records the PID namespace its numbers belong to, so `/proc` must be the procfs of the orchestrator's own PID namespace, with that namespace's PID 1 readable; the trial launcher refuses to start a trial otherwise. GPUs are optional; the core CLI can run CPU-only work. The optional MCP server additionally requires Linux `/proc` process start times for detached-run cancellation and recovery.

## Output layout

For `workdir: ./runs` and `experiment: demo`, PhaseSweep writes under `runs/demo/` beside the config file:

```text
demo/
  .gitignore
  artifact_root_binding.json
  run.log                         # progress stream, one TRIAL line per finished trial
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

The root `summary.yaml` and per-phase `winner.yaml` files are convenience projections. `last_successful_generation.yaml` points to the immutable generation that authoritatively represents the latest successful result. A failed invocation does not replace that pointer.

`trials.csv` lists every trial with its sampled parameters and user attributes, including the `observed:<name>` values that [`observe`](config.md#objective-evidence-constraints-and-gates) recorded. A phase with [`repeats`](config.md#repeats) adds a `repeat_group` column. Each `winner.yaml` carries the selected trial's `observed` map, which is empty when the experiment declares none or the trial recorded none, and a `repeats` block that is `null` outside a repeats phase.

`config.snapshot.yaml` is the executed configuration, including materialized defaults and the resolved invocation context. `reproducibility.json` is the shareable generation identity record. Trial directories retain local evidence and attempt lifecycle information used for supported continuation and winner validation.

MLflow-backed experiments additionally retain native run bindings in each trial directory and mutable delivery state and worker cleanup receipts under `tracking/<generation-id>/`. These records support upload retries and recovery and are separate from the committed scientific manifests. Disposable supervised-worker scratch uses `~/.cache/phasesweep/<backend>/workers/`, or the corresponding `XDG_CACHE_HOME/phasesweep` root; the default home comes from `Path.home()`. Each operation removes its temporary directory. Host lock placement retains its separate OS-account-home contract. See [MLflow tracking](mlflow.md) for ownership and offline replay.

### Following a run

`run.log` is the process-free progress stream. `tail -f <workdir>/<experiment>/run.log` follows a run from any shell, with no MCP server, no lock, and no PhaseSweep process of its own. Every finished trial adds one `TRIAL` line, in the order trials finish:

```text
2026-01-01 12:00:03 I phasesweep.engine.phase phase=depth TRIAL trial=0 state=COMPLETE outcome=success value=0.3 duration=0.4s (1/2 finished)
2026-01-01 12:00:04 I phasesweep.engine.phase phase=lr TRIAL trial=2 state=FAIL outcome=failure value=none duration=0.5s (3/6 finished) reason=non-zero exit code 1; stderr: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
```

`state` is the trial's Optuna state, except that a COMPLETE trial that violated a constraint or the metric's `min`/`max` bounds reads `INFEASIBLE`, as selection treats it. `outcome` is `success`, `failure`, `pruned`, or `cancelled`, and a failure adds `reason=` with the first line of its recorded cause. A trainer that exits nonzero records `non-zero exit code N`, followed by `; stderr: ` and the last non-empty line of the trial's `stderr.log` when it has one, so a crash is named without opening the trial directory. The line is read from the last few KiB of the file, stripped of control characters and cut at 200 characters with a trailing `...`. The same text is the `user_attr:phasesweep_failure_reason` column of `trials.csv` and the `cause` of the trial's recorded outcome. It is whatever the trainer printed, so it stays in these operator-owned records and no MCP tool returns it. `value` is the metric, or `none` when the trial produced none. `(k/n finished)` counts every terminal trial the phase's study holds, resumed ones included, against `n_trials`. A phase with [`repeats`](config.md#repeats) adds `repeat=<key>=<value>`, which names the seed or other repeat value that trial ran. The line carries no params; the `WINNER` line closing each phase does. A fatal objective error logs `FATAL OBJECTIVE ERROR` and skips the trial's `TRIAL` line.

### Repeat groups

A phase with [`repeats`](config.md#repeats) ranks configurations, not trials. Each trial carries the repeat key as a sampled param. Under grid a configuration is the trial's other params; under seeded random it is also the slot `trial_number // R`, so two slots that happen to draw the same params stay separate groups. A group is complete when it holds one trial per repeat value and every one is COMPLETE, feasible and inside the metric's bounds; only complete groups rank, by the configured reducer over their member metrics, with exact ties going to the group with the lowest member trial number. A group is fixed once complete and is named by its highest member trial number, which is also the winner's `trial_number`, so a resumed or enlarged phase never changes which trials a published group holds.

The winner's scalar fields (`trial_number`, `generation_id`, `attempt_id`, `objective_provenance`, `trainer_input`, `constraints`) describe that last member. `metric` is the reduced value, `observed` holds each name's mean over the members that recorded it, `params` and `effective_overrides` omit the repeat key, and `repeats` records `key`, `reduce`, `spread` (`n`, `mean`, sample `sd`, `min`, `max` of the member metrics) and `members`: per member its `trial_number`, `repeat` value, `metric`, generation and attempt IDs, objective provenance, trainer input, constraints and observed values, in trial-number order. Publication re-derives the reduced value, the spread and the observed means from the members, and compares the block with the phase's `repeats` in the generation's config snapshot (the same key and reducer, one member per repeat value, each value once), so an edited block fails the manifest like an edited winner. The manifest never reads trial evidence: every member's recorded evidence is re-verified when the phase selects its winner and when `--from-phase` reuses it, as a single-trial winner's always was. In `trials.csv`, `repeat_group` is the highest trial number of the group each trial belongs to so far.

A failed, pruned or infeasible member is never retried: its trial number is spent, so its group can never complete. Enlarging `n_trials` adds new configurations only under seeded random: a grid repeats phase, and a seed audit over an empty `search_space`, always runs its full matrix. A phase cut short by an accepted timeout ranks only the groups it completed. A failure abort ranks nothing and fails the phase until `n_trials` is raised, which only seeded random over a non-empty `search_space` allows. When no group is complete, the phase fails with `NoFeasibleTrialError`, which counts the incomplete groups and names any metric bound that kept members out.

### Confound verdicts

Each phase's `winner.yaml` carries a `confound` block: advisory verdicts on the population that selection ranked to pick the winner, computed once and frozen at selection time. It never blocks selection or publication and has no YAML switches or threshold settings. The authoritative copy belongs to the published generation; [invariant 26](invariants.md#26-a-confound-verdict-never-joins-a-fingerprint-or-gets-read-as-clean) describes its integrity, fingerprints, top-ups, and inheritance. In a repeats phase the block also lists `ranked_groups`, each complete group's member trial numbers, and its checks compare groups rather than trials.

The block runs three checks, each `ok` (checked, nothing found), `heterogeneous` (checked, found a potential confound, with evidence), or `n/a` (could not be checked - never treat this as clean):

- `evaluation_point`: whether the ranked objectives share a training-progress position. Envelope and W&B objectives use the configured `evaluation_axis`, such as optimizer step or tokens; MLflow objectives use the selected native observation's step. Positions within 1% of the highest count as the same point, and the exact positions are recorded either way. See [W&B evidence](config.md#wb-evidence) for aggregate positions. Distinct positions are expected when the phase searches training length such as epochs or `max_steps`; otherwise the trials were not trained to one budget. `n/a` for `json` and `log_regex` extractors, which report no position (only `json_envelope`, `wandb` and `mlflow` bind one, and `validate` prints which case applies); when a ranked objective has no recorded position or unreadable provenance; or when only one trial was ranked.
- `survivorship`: whether failed and pruned trials were at least as numerous as the ranked ones, so that whatever made configurations fail could have decided which ones were ranked. Failed trials include evidence-gate and extraction failures. Infeasible trials, which violated a constraint or the metric's bounds, are counted too but never flag: the experiment's own constraints and bounds define what may win. In a repeats phase, `ranked` counts the members of ranked groups, and feasible members of groups that never completed are counted as `incomplete_group`, which never flags either. This check always runs.
- `tie`: whether the trial-number tiebreak chose between trials with different params tied on the exact metric value, rather than between repeats of one configuration. The lowest trial number wins a tie. `n/a` with one ranked trial. `within_noise` is `null` without repeats. In a repeats phase `tied_trials` names the groups whose reduced value equals the winner's, and `within_noise` names the groups whose reduced value lies within two combined standard errors of the winner's (sample standard deviation over the square root of the group size, for either reducer), exact ties included; either list flags the check. Groups are named by their last member, a group with the winner's params is skipped, and the check is `n/a` with one ranked group. Two standard errors is a heuristic for "these configurations are not separated by this many seeds", not a significance test; [Noisy objectives and small budgets](config.md#noisy-objectives-and-small-budgets) covers sizing the number of seeds.

Selection logs one `WARNING` naming every flagged check and its numbers, and a phase that is about to build on a flagged parent winner logs its own `WARNING` before it runs. `phasesweep show-winners` prints the raw `winner.yaml`, including the full evidence behind each check.

> [!WARNING]
> `config.snapshot.yaml` is private because it can contain command and environment values. PhaseSweep writes a `.gitignore` containing `*` whenever the experiment namespace lacks one, but it never replaces an existing ignore file, so keep generated outputs, ledgers, and MCP state out of commits yourself.

### Trust boundary

The workdir is operator-trusted. By default its directories and files are not owner-only: new files take the process umask and a rewrite keeps the mode the file already had, so operators and tooling keep ordinary access to logs, evidence, winners, and resolved overrides. Engine records such as trial artifacts, `process_identity.json`, and `attempt_lifecycle.json` use ordinary path handling and are not symlink-hardened; records that need crash-safe replacement, including those two, are written to a temporary file and renamed into place atomically.

Three things inside the tree are hardened:

| Path | Mode | PhaseSweep refuses to |
| --- | --- | --- |
| `generations/<generation-id>/config.snapshot.yaml` | `0600` | replace it unless the existing file is an owner-only regular file with no other hard links |
| `environment.json` in a trial directory, written only when `execution.record_env` is set | `0600` | replace it under the same condition |
| `attempts/`, the attempt registry | `0700` directory of `0600` entries | read it through a symlink or with shared permissions |

The two files are owner-only while their directories keep ordinary permissions. The same validated, no-follow path handling protects the runtime lock namespace and the MCP `state_dir`.

## Storage and locks

`storage: null`, or omitting `storage`, is in-memory and ends with the process; it cannot be resumed or topped up by a later invocation. Persistent local storage is an explicit `journal:///...` URL, or `auto`, which always resolves to `<workdir>/<experiment>/study.journal`, regardless of `n_jobs`. The journal backend supports parallel trials (`n_jobs > 1`) on one host. Any other storage value, including `sqlite:///...`, `:memory:`, and external database URLs, is refused at config load; the error names the supported forms.

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

Runs and confirmed MCP recovery check the tree's binding and then the ledger's format before mutating the ledger. When either check fails, the command stops and leaves both byte-for-byte as they were. PhaseSweep is not yet stable and does not keep backward compatibility with older on-disk formats: a release that changes what the tree or ledger stores refuses an older one at these checks rather than migrating it. Finish that experiment with the release that wrote it, or start again with a fresh `workdir` and storage. Read-only commands, including `status`, `show-winners`, and `run --dry-run`, never create a missing ledger or its directory. A process killed while it appended to the journal can leave a partial final record, which every reader skips. The next `phasesweep run` or confirmed `phasesweep mcp recover-run` cuts it under Optuna's own journal lock, after saving the journal as `<ledger>.<UTC stamp>.bak` beside it, with permissions no broader than the ledger's. The killed process can also leave that lock behind, as `<ledger>.lock`. If the lock is still held after 30 seconds, the command refuses and names it: once no process is writing the ledger, remove the lock and run the command again. A record damaged anywhere before the last is refused as corruption. Contributors will find the ordering rules behind these guarantees, and the tests that hold them, in [durability invariants](invariants.md).

MLflow delivery retries use the experiment lock and validate their saved target and generation, but do not open or mutate the ledger. See [result delivery and recovery](mlflow.md#result-delivery-and-recovery).

Persistent phases use same-host locks. A concurrent CLI or MCP launch for the same experiment waits or fails safely according to the operation; it never merges two active orchestrators. External database backends are not part of the runtime.

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

`gpu_policy: whole_node` requires `n_jobs: 1` and a unique explicit device list. `gpu_policy: none` disables PhaseSweep's GPU isolation and cannot be combined with an explicit device list; parallel use requires the explicit `allow_no_gpu_isolation: true` acknowledgement. For CPU-only parallel work, make that acknowledgement deliberately rather than relying on failed device detection.

`gpu_policy: none` assigns no binding. Include ambient visibility in the trainer inheritance contract or set it in `env` if it must be retained.

## Timeouts, cleanup, and continuation

Trial, phase, and experiment timeouts are cooperative run controls. The experiment budget, `timeout_seconds_per_run`, bounds one invocation: its clock starts afresh at every `phasesweep run`, `--from-phase` resume or MCP launch, so it never counts time an earlier invocation spent. Set the per-trial timeout above the longest trial you expect; an unbounded trial needs `allow_unbounded_trials: true`. Its clock covers supervised launch and execution after GPU acquisition, including supervisor startup, but excludes the GPU queue and post-trainer evidence. Phase/run budgets cover those stages too. Cleanup can extend beyond a deadline. Unlike phase/run budgets, the per-trial timeout participates in semantic study identity; changing it cannot reuse a populated phase.

On a trial timeout PhaseSweep terminates the supervised process group and records a terminal failed attempt after cleanup. The `max_consecutive_failures` threshold also stops a broken phase. The abort is recorded with the outcome that trips it, so a trial that succeeds afterwards cannot turn the phase back into a result, and a rerun stays refused until `n_trials` is raised. Changing `max_consecutive_failures` governs new work only: it never aborts a completed phase retroactively, and a resumed phase whose failure streak already meets the new limit launches nothing until `n_trials` or the limit is raised. When a timeout and the threshold both stop the same phase short of its target, the timeout decides the outcome: with `allow_incomplete_on_timeout` the phase publishes the winner its completed trials earned, and without it the run stops as a timeout, so a retry with a larger budget does not replay the failure abort. A rerun or `--from-phase` replays an accepted partial target without filling unused slots, only while the config still sets `allow_incomplete_on_timeout`. `n_trials` counts terminal attempts, so failed and pruned attempts consume the configured target.

Persistent trial targets cannot decrease. Raising a resumable sampler's `n_trials` schedules additional attempts after a durable abort or accepted partial timeout. Failure and timeout controls never create an alternate winner.

Before a supported continuation, PhaseSweep first checks the artifact-root and ledger binding, then reconciles stale active attempts, and only then verifies recorded trial evidence and applies sampler continuation rules. `--from-phase` is valid only when the earlier phase winners remain valid; reused winners keep their original source provenance. TPE and CMA-ES persistent targets cannot resume mid-target, while grid and seeded random phases can top up their local study (see [what that costs](config.md#noisy-objectives-and-small-budgets) under a small budget). A top-up never reruns a finished attempt, and an attempt that failed, was cancelled, or was recovered as stale is finished. Under `grid` its point stays lost: a full grid cannot raise `n_trials` past its size, and a higher target on a partial grid runs only points never tried. Under seeded `random` a higher target draws new configurations from the same sequence instead, and with `repeats` either sampler loses the interrupted trial's whole group. A fixed-configuration phase (an empty `search_space` without `repeats`, which draws nothing) retries under any sampler: after a failed or interrupted attempt, raising `n_trials` runs one more attempt per failed one, and never after a completed one, feasible or not. A launch refuses a larger target before any phase runs, and so does the dry run, which reads the recorded attempts. A [repeats](#repeat-groups) phase resumes too: a top-up runs the members an interruption never started, so a group cut short between trials completes, while a member that failed, was cancelled, or was recovered as stale stays spent.

A phase's study stores the canonical payload its fingerprint is the digest of, beside the digest. When a later run's config disagrees, the refusal lists up to ten changed keys as dotted paths with their old and new values, truncates long values, and counts the rest. A trainer working directory that changed because `execution.cwd` is unset is called out with the fix. A winner file records only the hash, so a skipped phase whose winner no longer matches reports the two hashes.

An interrupted or failed generation leaves its immutable record for inspection but does not advance `last_successful_generation.yaml`. Signals are absorbed across the final publication commit so the authority pointer is never left ambiguous.

## Inspection commands

Use `validate`, `run --dry-run`, `status`, and `show-winners` to review an experiment without launching work; an ordinary `run` invocation launches trials. A dry run previews each phase from a sample: the command of a phase that inherits another that still has trials to run carries stand-in values for the parent's winner, and its `DRY RUN example command` line says they are placeholders until the parent publishes a winner. It also reads the trials the ledger already records, the way `status` does, from one capture of the journal and writing nothing, and applies a run's own checks to them in the run's order before it previews any phase: each phase reports `completed` and `remaining` trials and how many a run would launch. A phase that has reached its `n_trials`, or whose timed-out partial result was already accepted at that target, says it would replay its existing result and launch no trials, and the phases that inherit it preview with the winner that replay selects from the recorded trials rather than a stand-in. After the phase lines it prints one total, for example `DRY RUN total: would launch 30 trials across 3 phases (2 other phases replay)`: the launch counts those lines give, summed over the phases that launch trials, with the replaying phases counted apart and a phase `--from-phase` skips left out. A dry run that refuses prints no total. The dry run refuses what a run refuses over the recorded trials, with the run's own message: a publication whose phase study is missing or cut below the published boundary, or whose pointer is unreadable; a study that optimizes the other direction or whose failure-policy state is malformed, in any phase; a target below the one the study accepted; a fixed configuration asked for more attempts than it may retry; an upstream top-up over a descendant study already bound to the winner; a `tpe` or `cmaes` phase cut short or raised past its target; a phase that aborted, or whose failure streak meets a lowered `max_consecutive_failures`, and has not had `n_trials` raised; a partial result the config no longer accepts; and a replay with no feasible trial. Several refused studies come back as one combined refusal, as a run combines them. A phase whose recorded trials cannot be read is previewed as a fresh phase, with a warning, because a run refuses a ledger it cannot read. A phase with a RUNNING attempt is counted with that attempt as failed and given only a fixed configuration's attempt limit, because a run recovers that attempt first; its accepted target, sampler continuation, abort state and partial decision are not checked, and the warning says so. The dry run does not recover stale attempts, check the trainer environment, compare each study's stored fingerprint with the config, check each study's artifact-root claim, or re-prove the trials' evidence on disk, so a run can still refuse a phase on those grounds. `--from-phase` requires valid earlier winners. These reads inspect only the current-format local experiment.

`status CONFIG` prints trial counts and publication identity as YAML. `status CONFIG --trials` prints a header line and one aligned plain-text table per phase instead, one row per trial: `trial`, `state`, `value`, `params`, `duration_s`, `repeat_group` for a [`repeats`](config.md#repeats) phase, one `observed:<name>` column per [`observe`](config.md#objective-evidence-constraints-and-gates) name any trial recorded, and `winner`. `state` is the Optuna state, except that a COMPLETE trial that violated a constraint or the metric's `min`/`max` bounds reads `INFEASIBLE`. `duration_s` is the trainer subprocess's wall-clock runtime, and it is blank for a trial that never ran one; the `duration` on a trial's `TRIAL` line in `run.log` spans the whole trial, GPU queue and evidence extraction included. `winner` is `*` on the trials the published winner stands on: the one trial, or every member of the winning repeat group. It reflects the published generation named in the header line, so it is stale while a rerun has not published, and it is blank when nothing validates as published. A table prints even when the publication is corrupt or unreadable, followed by the same error and exit status as plain `status`. For plots and the dashboard, point Optuna's own tools at the ledger as described in [Inspect the ledger with Optuna](#inspect-the-ledger-with-optuna).

> [!TIP]
> `status` and `show-winners` take no lock and never write to the artifact root or ledger, so they are safe to run while a sweep is active.

For operator-managed detached runs, see [the MCP operator guide](mcp.md). Terminal MCP run results are frozen snapshots associated with a run ID; they are not a replacement for CLI inspection of an arbitrary experiment tree.
