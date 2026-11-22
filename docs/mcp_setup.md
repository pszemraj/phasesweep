# MCP agent setup

Requirements: Python 3.11+, the [MCP runtime requirements](mcp.md), a PhaseSweep experiment that passes validation, and a client with local stdio MCP support.

## Setup

### 1. Install the MCP extra

Install PhaseSweep and its optional MCP dependency in the Python environment that provides the server executable:

```bash
pip install "phasesweep[mcp]"
```

Contributor and editable-install setup is in [development](development.md). Install the tracking extra needed by the cataloged experiment in that same environment: `phasesweep[mcp,wandb]` for W&B or `phasesweep[mcp,mlflow]` for MLflow. A run needs the selected SDK and access to its configured account or server. See [native MLflow setup and run ownership](mlflow.md) for MLflow's tracking server and experiment.

### 2. Create and review the catalog

If you need an experiment, `phasesweep init` writes `starter.yaml` without overwriting files; `phasesweep init --list` lists the other [templates](config.md#templates). Use the [configuration guide](config.md) and `phasesweep explain KEY` for authoring; connected agents can use `explain_config` when asked for that help.

Before cataloging a source-checkout trainer, follow the [trainer environment preflight](config.md#source-checkout-preflight) to verify its imported library and supported arguments under the configured environment. Then create the catalog:

```bash
phasesweep mcp init-catalog --from ./starter.yaml
```

The command writes `catalog.yaml`, validates it, provisions a fresh [private state layout](mcp.md#fresh-mcp-state), and prints one line per entry with its `trainer cwd`, the directory its trainer starts in: the experiment's `execution.cwd`, or the entry's `cwd` when that is unset. Add another `--from` for each experiment; use `-o` for another catalog filename. The scaffold never replaces an existing catalog. Each entry takes its `id` from the config file name, sets `visible_params: none`, and leaves `description` and the `allow` block commented out, so it is read-only until you grant permissions (see [the catalog fields and permission model](mcp.md#the-catalog)). A config beneath the catalog's directory is written as a relative path with `cwd: "."` (the catalog's directory, and the trainer's while `execution.cwd` is unset). A config elsewhere is written as an absolute path with no `cwd`, so the entry starts in that config's own directory.

The scaffold records an absolute `state_dir` under `PHASESWEEP_HOME/mcp/<catalog-digest>`, or `${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>` when unset; later environment changes do not move it. A missing absolute `PHASESWEEP_HOME` is created mode 0700; relative paths, symlink components and existing directories with another mode are refused. To place state elsewhere, edit `state_dir`. It must be writable by each participating shell, including sandboxes. Give the server and `phasesweep mcp call` the same catalog, and `recover-run` the same `--state-dir`; see [state ownership and format rules](mcp.md#fresh-mcp-state).

Validate the reviewed catalog with its absolute path:

```bash
phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
```

This checks the catalog and its private state, including unreadable or orphaned run records that a launch cannot clear itself; the command then prints one error naming them and exits 2 in place of the table. It does not list a run that needs recovery: the entries still print `ok`, and the tools report that run with `recovery_required: true` ([run state and recovery](mcp.md#run-state-and-recovery)). For launch-enabled entries it also checks that the trainer directory exists and that a trial program written as a path is an executable file. A program written without a path is found on the `PATH` the server starts with, so this check never judges it: it is judged at launch, and by `phasesweep run CONFIG --dry-run`. A missing trainer can fail this readiness check while the server still serves results and other experiments. Each entry shows `ok` with its permissions and `trainer cwd`, or `FAIL` with the reason and a `fix:` line. The command exits 0 when all checks pass. At startup, the server logs `trainer cwd` for every entry it serves.

### 3. Configure your client

PhaseSweep never reads or writes MCP client configuration files; the operator manages the client-owned server entry.

Configure your client's stdio server with the absolute `phasesweep-mcp` executable and reviewed catalog path, following its documentation for configuration scope and restart behavior:

```text
/absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
```

For a server on another machine, the client's command is an [SSH command](#over-ssh) instead.

To run the stdio server directly instead:

```bash
phasesweep mcp serve --catalog /absolute/path/to/catalog.yaml
```

For Codex CLI, the [official MCP instructions](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) use:

```bash
codex mcp add phasesweep -- /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
codex mcp list
```

To keep a client from connecting the server at all, disable it in that client's own MCP settings.

The client starts the server with the client's environment, not your shell's, and trainers inherit it, so a `trial_command` program resolves on the `PATH` the client starts the server with, unless the experiment's `env` sets `PATH`, which wins. That `PATH` often lacks the Python environment your shell activates. Give `trial_command` an absolute interpreter path, such as `/absolute/path/to/python train.py --config {config_path}`, or set `PATH` in the environment the server is started with: the client entry's `env`, or the SSH command. A study whose trials ran from your shell continues only under the same tracked environment (`PATH`, `HOME`, `LANG` and the other variables a fingerprint mismatch names), so start the server with those values (in the client entry's `env` or the SSH command), or start the MCP runs under a new `experiment` name. A run whose program is missing from that `PATH` is refused before any trial starts: its `failure.remediation` gives this fix, and its runner log, `<run-id>.log` under `state_dir/logs/`, names the program.

#### Over SSH

To serve a catalog that lives on another machine, make the client's stdio command an SSH command that starts the server there, with both paths absolute and no terminal allocated:

```text
ssh -T HOST /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
```

A relative `--catalog` resolves against the server's working directory, and a bare `phasesweep-mcp` runs only if the session's `PATH` finds it; otherwise the client sees only the connection close. A terminal (`ssh -t`) mixes the server's log lines and echoed input into the protocol stream and breaks it.

As above, trainers inherit the server's environment, which here is the SSH session's, not a login shell's: `ssh -T HOST printenv PATH LANG` shows it. Besides an absolute program path in `trial_command`, `PATH` can be set where the server starts: `ssh -T HOST env PATH=/absolute/path/to/env/bin:/usr/bin:/bin /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml`. If the same study will also be run from a shell on that machine, pin `PATH` and `LANG` under the experiment's `env` instead: the trainer environment is then identical whoever starts the study. `env` is part of the study's identity, so put it in the config before the first trial.

To verify, run the check and a dry run through the same SSH form, with the absolute `phasesweep` path rather than the bare command the setup steps print, and the same `env PATH=…` prefix if you use one:

```bash
ssh -T HOST /absolute/path/to/phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
ssh -T HOST /absolute/path/to/phasesweep run /absolute/path/to/experiment.yaml --dry-run
```

For a study with trials left, only the dry run shows a trial program missing from the session's `PATH`, because `mcp check` judges only a program written as a path; the dry run refuses and names the program. Then confirm that nothing is printed before the server starts: `ssh -T HOST true | wc -c` must print `0`, because anything a shell start-up file writes to stdout corrupts the protocol stream.

A dropped connection ends the server, not the run: the detached runner and its trainer keep going, and a server left over from the old connection is harmless beside the new one. After reconnecting, call `get_latest_run`, then `await_run` and `get_run_results` with the `run_id` it returns; never launch a replacement because a connection dropped. If the runner itself was killed, as when a host kills everything a session started, the tools report the run `running` with `recovery_required: true` and name a `phasesweep mcp recover-run …` command ([run state and recovery](mcp.md#run-state-and-recovery)). That command, like the other troubleshooting commands the tools and this page name, runs in a shell on the remote machine as the server's user.

A study continues only under the tracked environment it started with, so one moved between an SSH session and a shell can differ in `PATH` or `LANG`. The run is then refused before anything launches and no result is dropped; the refusal lists the recorded and current values, which through MCP are in the run's log, `<run-id>.log` under `state_dir/logs/`, not in the tool's remediation. Run with the recorded values, or pin them under `env` and start a new `experiment` name ([environment-mismatch remedy](runtime.md#resolve-a-changed-study-or-trainer-environment)). A replay of a finished study under another environment is not refused: it reuses the recorded result and warns `reused winner ran under a different trainer environment` in the CLI output or the run's log, not in the tool's result. Confirm the difference cannot affect the metric, or rerun the phase under a new `experiment` name.

### 4. Verify the connection

Restart or reconnect the client as its documentation requires. Confirm the connection with the read-only `list_experiments` operation; for example, ask the agent to use PhaseSweep to list the available experiments and permitted actions without launching anything. Name PhaseSweep in the request: the server's instructions tell agents to use its tools only when the user asks for it. The response contains catalog-approved IDs, descriptions, phase names, metrics, permitted actions and `catalog_status`; a stale catalog includes its reload refusal and operator remedy.

To check the same operation without a client, run it from a shell:

```bash
phasesweep mcp call list_experiments --catalog /absolute/path/to/catalog.yaml
```

See [shell calls](mcp.md#calling-a-tool-from-a-shell) for arguments, exit codes and monitoring. For an explicitly authorized run, follow the [tool workflow](mcp.md#tool-workflow) and [response shapes](mcp.md#reading-status-responses).

## Maintenance and troubleshooting

Use `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` after a catalog edit or when diagnosing server startup. If you replace the environment that supplies `phasesweep-mcp`, update the client-owned entry (or the SSH command) to the new absolute executable path using that client's documentation.

- `MCP support is not installed`: install the MCP extra in the environment that supplies `phasesweep-mcp`.
- The client cannot start the server: verify its client-owned stdio entry has absolute executable and catalog paths, then run `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml`. Over SSH, also check that the command uses `-T` and that `ssh -T HOST true | wc -c` prints `0` ([over SSH](#over-ssh)).
- `action 'launch' is not permitted`, `action 'cancel' is not permitted` or `action 'from_phase' is not permitted`: check the [catalog permissions](mcp.md#the-catalog). For a refused cancellation, follow the operator step the error gives for that run.
- `catalog_status.stale: true`, or a launch or inspection refused because a catalog entry `failed to load` or the catalog file `does not validate`: run `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` and fix what it reports; the server's stderr log names the file. See [catalog reload behavior](mcp.md#the-catalog), including when a restart is required.
- `concurrency limit reached` or an experiment that already has a running sweep: [run state and recovery](mcp.md#run-state-and-recovery) says which blockers to await and which need the operator.
- `cannot prove launch capacity`, or an unreadable or orphaned record for one run ID: `mcp check` lists the unreadable or orphaned records a launch cannot clear itself, leaving out a pre-spawn orphan, which the next launch removes, and a run that needs recovery, which the tools report. Run the `phasesweep mcp recover-run` command the launch refusal names for each pre-spawn orphan. For each other listed run ID, run `pgrep -af <run-id>` to list its runner and `ps axeww | awk '/[P]HASESWEEP_GENERATION_ID=<run-id>/ {print $1}'` to print its trial PIDs by environment. Run these on the server host as the server's user so `ps` can see the trial environment. Wait for the listed processes to exit or stop them. Move `runs/<run-id>.json` and `logs/<run-id>.*` out of `state_dir` only once both commands print nothing.
- `another launch is in progress`: a launch sharing this `state_dir` holds the launch lock; retry in a moment.
- Lock-directory access failure (`storage_unavailable`): set the same `PHASESWEEP_LOCK_DIR` for every participating shell, MCP server, and sandbox, following the [lock directory rules](runtime.md#storage-and-locks).
- `recovery_required: true`, unresolved launch, or uncertain cleanup: stop agent activity and follow [run state and recovery](mcp.md#run-state-and-recovery), running the command it names in a shell on the machine that holds the catalog.
- Unavailable terminal snapshot: follow `failure.remediation` and inspect run/server diagnostics when it asks. [Historical results](mcp.md#reading-status-responses) cannot be reconstructed from the current ledger.
- `runner_observable: false`, alone or with `recovery_required: true`, or a PID-namespace refusal: follow [PID namespace recovery](mcp.md#pid-namespaces).
- Catalog path, storage, or working-directory rejection: follow [the catalog requirements](mcp.md#the-catalog).
- Trainer environment cohort mismatch: the [launching server or `mcp call` shell](mcp.md#the-catalog) supplies the trainer's base environment, so a study moved between an SSH session and a shell hits it; reconcile it using the [trainer contract](config.md#trainer-contract) and [over SSH](#over-ssh).
- A run refused because its trial program is not on the trainer's `PATH`: the `PATH` the server was started with lacks it; set `PATH` where the server starts ([the client entry or the SSH command](#3-configure-your-client)), pin it under the experiment's `env` before the first trial, or name the program by absolute path in `trial_command`.
- `its runner exited with code`: the runner failed before it started work; its own error is in `<run-id>.log` under `state_dir/logs/`. A server that imports PhaseSweep only through `PYTHONPATH` fails this way, because the runner does not inherit it; install PhaseSweep for the server's interpreter.
