# MCP agent setup

Use `phasesweep-mcp` or `phasesweep mcp` with an MCP client by installing the MCP extra, writing a catalog, and adding a stdio server entry. Tool behavior and security boundaries are in [MCP server](mcp.md).

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

Write a catalog that maps stable ids to reviewed experiment configs:

```yaml
state_dir: /abs/path/to/project/runs/.mcp
max_concurrent_runs: 1
experiments:
  - id: tiny-lm
    config: /abs/path/to/project/examples/mcp_experiment.yaml
    description: "16 MB LM: pick depth, then lr, then regularization"
    allow:
      launch: true
      cancel: true
      from_phase: true
```

Catalog validation, path rules, side-effect permissions, and the security model are described in [the catalog](mcp.md#the-catalog) and [security model](mcp.md#security-model). Start entries read-only, expose only reviewed configs, and enable `allow` actions deliberately. The checked-in example is [examples/catalog.yaml](../examples/catalog.yaml), which points at [examples/mcp_experiment.yaml](../examples/mcp_experiment.yaml) and writes scratch artifacts under `/tmp/phasesweep-mcp-tiny-lm`.

## Test the server

Run this from any working directory:

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

If your client supports MCP prompts, load `phasesweep_run_and_monitor` instead of pasting instructions. If it supports MCP resources, `phasesweep://catalog` exposes the first catalog page; use `phasesweep_list_experiments` for pagination. Tool behavior, catalog validation, run state, and exposed fields are described in [MCP server](mcp.md).

## Paste this to your agent

When prompts are unavailable, paste this into the agent's project instructions or chat before asking it to run a sweep:

```text
Use the local phasesweep MCP server only through the exposed catalog.
Start with phasesweep_list_experiments and phasesweep_validate_config.
Launch only by catalog experiment id. After launch, poll phasesweep_get_status by run_id until terminal, then call phasesweep_get_winners with the same run_id.
Do not ask for or infer config paths, storage URLs, workdirs, commands, environment variables, run-control settings, raw datasets, labels, predictions, logs, result files, dashboards, or per-trial histories unless I explicitly ask for separate filesystem or dashboard work.
Base recommendations on catalog descriptions, phase shape, status counts, exposed winner metrics, and sampled params that are not redacted by catalog policy. Treat `<redacted>` sampled param values as intentional, not as missing data. Do not change the objective, extractor, command, search space, constraints, gates, storage, workdir, environment, or safety waivers unless I ask for config-authoring help.
Cancel only when I ask or when stopping is necessary to prevent an unwanted active sweep.
```

## Useful agent requests

- `List the available phasesweep experiments and validate the tiny LM example.`
- `Launch the tiny-lm sweep, monitor it until completion, then summarize each phase winner.`
- `Check whether run <run_id> is still active and show phase-level trial counts.`
- `Read the current winners for <experiment_id> and suggest a next manual experiment using only MCP outputs.`

## Troubleshooting

- `action 'launch' is not permitted`: set `allow.launch: true` for that catalog entry and restart the MCP client.
- `action 'cancel' is not permitted`: set `allow.cancel: true` for that catalog entry and restart the MCP client.
- `concurrency limit reached`: wait for an active MCP run to finish, cancel it, or raise `max_concurrent_runs` for hosts that can safely run multiple sweeps.
- A cancelled run stays `running` with `cleanup_confirmed: false`: inspect the host for leftover runner/trial process groups, then run `phasesweep mcp-recover-run --state-dir <state_dir> --run-id <run_id>` and repeat with `--confirm` only if the command reports confirmed cleanup.
- `storage must be persistent`: use a persistent Optuna storage URL such as SQLite on disk; in-memory studies cannot be monitored across processes.
- The client cannot find `phasesweep-mcp`: use the absolute path to the virtualenv or conda environment executable in the MCP client config.
- Relative path rejected at startup: catalog `state_dir`, `config`, and `cwd` paths may be relative to the catalog file, but experiment `workdir` and file-backed SQLite/Journal storage paths must be absolute for MCP.
- Old MCP runs clutter status or logs: inspect `state_dir/runs` and `state_dir/logs`, then archive or prune terminal run handles between campaigns.
