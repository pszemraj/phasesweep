# MCP agent setup

Use `phasesweep-mcp` or `phasesweep mcp` with an MCP client by installing the MCP extra, writing a catalog, and adding a stdio server entry. Tool behavior and security boundaries are covered in [MCP server](mcp.md).

## Install

Install the MCP extra in the Python environment that your MCP client will use:

```bash
python -m pip install "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git"
```

From a local checkout:

```bash
python -m pip install -e ".[mcp]"
```

If your MCP client runs outside your shell environment, prefer an absolute path to the environment's `phasesweep-mcp` executable in the client config.

## Create a catalog

The agent cannot choose config paths or edit run settings. You expose experiments by writing a catalog that maps stable ids to config files:

```yaml
state_dir: /abs/path/to/project/runs/.mcp
max_concurrent_runs: 1
experiments:
  - id: tiny-lm
    config: /abs/path/to/project/examples/experiment.yaml
    description: "16 MB LM: pick depth, then lr, then regularization"
    allow:
      launch: true
      cancel: true
      from_phase: true
```

Omit `allow` or leave a flag false to make that side effect unavailable. By default, catalog entries are read-only: agents can list, validate, inspect status, and read existing winners, but they cannot launch, cancel, or resume with `from_phase`.

## Before connecting an agent

- Start read-only, then enable `allow.launch`, `allow.cancel`, or `allow.from_phase` only for experiments you are comfortable letting an agent operate.
- Keep secrets, private paths, dataset ids, hostnames, target/dependent-variable values, validation labels, prediction dumps, and metric histories out of sampled categorical choices, phase names, descriptions, and chat prompts.
- Use fixed config fields or the trainer environment for private values. MCP does not return `trial_command`, `env`, `storage`, `workdir`, rendered commands, logs, or effective overrides.
- Expect `phasesweep_get_winners` to return each exposed winner's objective metric value and sampled parameters. That is the summary the agent uses to compare sweep outcomes.
- Do not give the same agent unrestricted filesystem access to the run directory if you do not want it reading raw result files, trainer logs, W&B exports, predictions, or labels.

## Test the server

Run this from the same working directory you intend to use in production, because experiment-relative `workdir` and `storage` paths resolve from the server's current directory:

```bash
phasesweep-mcp --catalog /abs/path/to/project/examples/catalog.yaml
```

The server speaks JSON-RPC over stdio, so it will appear to wait for input. Stop it with `Ctrl-C` after confirming it starts without a catalog error.

## MCP client config

Paste this into any stdio MCP-compatible client config, replacing the catalog path:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/project/examples/catalog.yaml"]
    }
  }
}
```

If the client cannot find `phasesweep-mcp`, use absolute paths:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/project/examples/catalog.yaml"]
    }
  }
}
```

For clients that can launch through `uvx`, use the Git install directly:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "uvx",
      "args": ["--from", "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git", "phasesweep-mcp", "--catalog", "/abs/path/to/project/examples/catalog.yaml"]
    }
  }
}
```

You can also use the main CLI or run the module directly:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/phasesweep",
      "args": ["mcp", "--catalog", "/abs/path/to/project/examples/catalog.yaml"]
    }
  }
}
```

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/python",
      "args": ["-m", "phasesweep.mcp.server", "--catalog", "/abs/path/to/project/examples/catalog.yaml"]
    }
  }
}
```

Restart the MCP client after changing the config.

If your client supports MCP prompts, load `phasesweep_run_and_monitor` instead of pasting the text below. If it supports MCP resources, `phasesweep://catalog` exposes the first catalog page; use `phasesweep_list_experiments` for pagination.

## Paste this to your agent

Paste this into the agent's project instructions or chat before asking it to run a sweep:

```text
You have access to a local phasesweep MCP server. Use it to operate only the human-curated experiment catalog exposed by the server.

Start by calling phasesweep_list_experiments, then call phasesweep_validate_config for the experiment id you plan to use. Do not ask me for config paths, storage URLs, workdirs, commands, environment variables, or run-control settings; the catalog is the authority for those.

If I ask you to run a sweep, call phasesweep_launch_sweep with the catalog experiment id. Use from_phase only when I explicitly ask to resume from a phase or when we have confirmed earlier phase winners already exist. After launch, poll phasesweep_get_status by run_id until the run is succeeded, failed, or cancelled.

Use phasesweep_get_winners to summarize completed phase winners. Treat returned metric values as experiment summaries and sampled params as user-visible hyperparameters, not secrets. Do not inspect raw datasets, target/dependent-variable columns, validation labels, predictions, trainer logs, raw result files, W&B dashboards, or per-trial metric histories unless I explicitly ask for that separate work.

When recommending a next manual experiment, base the recommendation on MCP outputs: catalog descriptions, phase shape, status counts, exposed winner metrics, and sampled params. Do not change the objective metric, extractor, trainer command, search space, constraints, gates, storage, workdir, environment, or safety waivers unless I explicitly ask for config-authoring help.

Use phasesweep_cancel_sweep only when I explicitly ask you to stop a run, or when stopping is clearly necessary to prevent an unwanted active sweep.
```

## Useful agent requests

```text
List the available phasesweep experiments and validate the one that looks like the tiny LM example.
```

```text
Launch the tiny-lm sweep, monitor it until completion, then summarize the winning params for each phase.
```

```text
Check whether run <run_id> is still active and show me the phase-level trial counts.
```

```text
Read the current winners for <experiment_id> and explain what the next manual experiment should try using only the phasesweep MCP outputs.
```

## Troubleshooting

- `action 'launch' is not permitted`: set `allow.launch: true` for that catalog entry and restart the MCP client.
- `action 'cancel' is not permitted`: set `allow.cancel: true` for that catalog entry and restart the MCP client.
- `storage must be persistent`: use a persistent Optuna storage URL such as SQLite on disk; in-memory studies cannot be monitored across processes.
- The client cannot find `phasesweep-mcp`: use the absolute path to the virtualenv or conda environment executable in the MCP client config.
- Relative paths behave differently from the CLI: start the MCP server from the same project directory every time, or use absolute paths for `state_dir`, experiment `workdir`, and experiment `storage`.
