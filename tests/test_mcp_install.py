"""Installer behavior: file-edit primitives, agent targets, and the CLI flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from phasesweep.cli import main as cli_main
from phasesweep.mcp.install import edits as install_edits
from phasesweep.mcp.install import installer
from phasesweep.mcp.install.edits import (
    merge_json_member,
    remove_json_member,
    remove_marked,
    replace_or_append_marked,
)
from phasesweep.mcp.install.targets import (
    AGENT_IDS,
    MARKDOWN_END,
    MARKDOWN_START,
    agent_targets,
    codex_toml_content,
    mcp_entry,
)
from tests.mcp_helpers import write_mcp_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mcp_experiment.yaml"

ENTRY = {"command": "/venv/bin/phasesweep-mcp", "args": ["--catalog", "/proj/catalog.yaml"]}


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Redirect HOME (and XDG) so targets never touch the real user config."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("PATH", "")
    return home


# --- JSON member edits ---


def test_merge_json_member_creates_missing_file(tmp_path):
    path = tmp_path / "cfg" / "mcp.json"
    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "created"
    data = json.loads(path.read_text())
    assert data == {"mcpServers": {"phasesweep": ENTRY}}
    assert path.read_text().endswith("\n")


def test_merge_json_member_preserves_data_order_and_indent(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        '{\n    "theme": "dark",\n    "mcpServers": {\n        "other": {"command": "x"}\n    }\n}\n'
    )
    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "updated"
    text = path.read_text()
    data = json.loads(text)
    assert list(data) == ["theme", "mcpServers"]
    assert list(data["mcpServers"]) == ["other", "phasesweep"]
    assert data["theme"] == "dark"
    assert data["mcpServers"]["other"] == {"command": "x"}
    assert '    "theme"' in text  # detected 4-space indent
    assert text.endswith("\n")
    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "unchanged"


def test_merge_json_member_skips_commented_config(tmp_path):
    path = tmp_path / "opencode.json"
    original = '{\n  // my settings\n  "mcp": {}\n}\n'
    path.write_text(original)
    assert merge_json_member(path, "mcp", "phasesweep", ENTRY) == "skipped"
    assert path.read_text() == original


def test_merge_json_member_errors_on_non_object_shapes(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("[1, 2]\n")
    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "error"
    path.write_text('{"mcpServers": [1]}\n')
    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "error"


def test_merge_json_member_refuses_differing_unmanaged_member(tmp_path):
    path = tmp_path / "mcp.json"
    original = '{"mcpServers": {"phasesweep": {"command": "custom"}}}\n'
    path.write_text(original)

    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "conflict"
    assert path.read_text() == original


def test_remove_json_member_prunes_and_unlinks(tmp_path):
    path = tmp_path / "mcp.json"
    merge_json_member(path, "mcpServers", "phasesweep", ENTRY)
    assert remove_json_member(path, "mcpServers", "phasesweep") == "removed"
    assert not path.exists()

    path.write_text(json.dumps({"theme": "dark", "mcpServers": {"phasesweep": ENTRY}}, indent=2))
    assert remove_json_member(path, "mcpServers", "phasesweep") == "removed"
    assert json.loads(path.read_text()) == {"theme": "dark"}
    assert remove_json_member(path, "mcpServers", "phasesweep") == "not-found"
    assert remove_json_member(tmp_path / "absent.json", "mcpServers", "phasesweep") == "not-found"


def test_remove_json_member_errors_on_malformed_container(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text('{"mcpServers": []}\n')

    assert remove_json_member(path, "mcpServers", "phasesweep") == "error"


def test_config_edits_refuse_directories_and_symlinks(tmp_path):
    directory = tmp_path / "config"
    directory.mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}\n")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)

    for path in (directory, symlink):
        assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "error"
        assert remove_json_member(path, "mcpServers", "phasesweep") == "error"
        assert (
            replace_or_append_marked(path, "body", start=MARKDOWN_START, end=MARKDOWN_END)
            == "error"
        )
        assert remove_marked(path, start=MARKDOWN_START, end=MARKDOWN_END) == "error"
    assert target.read_text() == "{}\n"


def test_atomic_edit_failure_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    original = '{"mcpServers": {"other": {"command": "x"}}}\n'
    path.write_text(original)

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(install_edits.os, "replace", fail_replace)

    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "error"
    assert path.read_text() == original
    assert list(tmp_path.glob(".mcp.json.*.tmp")) == []


def test_json_edit_dry_run_reports_actions_without_writing(tmp_path):
    path = tmp_path / "mcp.json"

    assert (
        merge_json_member(
            path,
            "mcpServers",
            "phasesweep",
            ENTRY,
            dry_run=True,
        )
        == "created"
    )
    assert not path.exists()

    path.write_text('{"mcpServers": {"phasesweep": {"command": "old"}}}\n')
    original = path.read_text()
    assert (
        merge_json_member(
            path,
            "mcpServers",
            "phasesweep",
            ENTRY,
            managed=lambda _entry: True,
            dry_run=True,
        )
        == "updated"
    )
    assert (
        remove_json_member(
            path,
            "mcpServers",
            "phasesweep",
            managed=lambda _entry: True,
            dry_run=True,
        )
        == "removed"
    )
    assert path.read_text() == original


# --- marker-fenced blocks ---


def test_marked_block_round_trip_is_byte_identical(tmp_path):
    path = tmp_path / "CLAUDE.md"
    original = "# My project\n\nHouse rules.\n"
    path.write_text(original)
    assert (
        replace_or_append_marked(path, "body", start=MARKDOWN_START, end=MARKDOWN_END) == "updated"
    )
    assert path.read_text() == f"{original}\n{MARKDOWN_START}\nbody\n{MARKDOWN_END}\n"
    assert remove_marked(path, start=MARKDOWN_START, end=MARKDOWN_END) == "removed"
    assert path.read_text() == original


def test_marked_block_round_trip_preserves_missing_final_newline(tmp_path):
    path = tmp_path / "CLAUDE.md"
    original = "# My project"
    path.write_text(original)

    assert (
        replace_or_append_marked(path, "body", start=MARKDOWN_START, end=MARKDOWN_END) == "updated"
    )
    assert path.read_text() == f"{original}\n{MARKDOWN_START}\nbody\n{MARKDOWN_END}\n"
    assert remove_marked(path, start=MARKDOWN_START, end=MARKDOWN_END) == "removed"
    assert path.read_text() == original


def test_marked_block_on_fresh_file_unlinks_on_removal(tmp_path):
    path = tmp_path / "AGENTS.md"
    assert (
        replace_or_append_marked(path, "body", start=MARKDOWN_START, end=MARKDOWN_END) == "created"
    )
    assert remove_marked(path, start=MARKDOWN_START, end=MARKDOWN_END) == "removed"
    assert not path.exists()


def test_marked_block_replaces_in_place(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text(f"before\n\n{MARKDOWN_START}\nold\n{MARKDOWN_END}\nafter\n")
    assert (
        replace_or_append_marked(path, "new", start=MARKDOWN_START, end=MARKDOWN_END) == "updated"
    )
    assert path.read_text() == f"before\n\n{MARKDOWN_START}\nnew\n{MARKDOWN_END}\nafter\n"
    assert (
        replace_or_append_marked(path, "new", start=MARKDOWN_START, end=MARKDOWN_END) == "unchanged"
    )
    assert (
        remove_marked(tmp_path / "absent.md", start=MARKDOWN_START, end=MARKDOWN_END) == "not-found"
    )


def test_marked_block_refuses_unmatched_marker(tmp_path):
    path = tmp_path / "AGENTS.md"
    original = f"before\n{MARKDOWN_START}\nunterminated\n"
    path.write_text(original)

    assert replace_or_append_marked(path, "new", start=MARKDOWN_START, end=MARKDOWN_END) == "error"
    assert remove_marked(path, start=MARKDOWN_START, end=MARKDOWN_END) == "error"
    assert path.read_text() == original


def test_marked_edit_dry_run_reports_actions_without_writing(tmp_path):
    path = tmp_path / "AGENTS.md"
    assert (
        replace_or_append_marked(
            path,
            "body",
            start=MARKDOWN_START,
            end=MARKDOWN_END,
            dry_run=True,
        )
        == "created"
    )
    assert not path.exists()

    path.write_text(f"before\n{MARKDOWN_START}\nold\n{MARKDOWN_END}\n")
    original = path.read_text()
    assert (
        replace_or_append_marked(
            path,
            "new",
            start=MARKDOWN_START,
            end=MARKDOWN_END,
            dry_run=True,
        )
        == "updated"
    )
    assert (
        remove_marked(
            path,
            start=MARKDOWN_START,
            end=MARKDOWN_END,
            dry_run=True,
        )
        == "removed"
    )
    assert path.read_text() == original


# --- targets ---


def test_agent_targets_match_public_ids_and_scopes(fake_home, tmp_path):
    project = tmp_path / "proj"
    targets = {target.id: target for target in agent_targets(project)}
    assert tuple(targets) == AGENT_IDS

    assert targets["claude"].mcp.path == project / ".mcp.json"
    assert targets["claude"].mcp.key == "mcpServers"
    assert targets["claude"].instructions_path == project / "CLAUDE.md"
    assert targets["claude-desktop"].mcp.scope == "user"
    assert targets["claude-desktop"].instructions_path is None
    assert targets["codex"].mcp.path == fake_home / ".codex" / "config.toml"
    assert targets["codex"].mcp.format == "toml"
    assert targets["codex"].mcp.notice
    assert targets["cursor"].mcp.path == project / ".cursor" / "mcp.json"
    assert targets["vscode"].mcp.key == "servers"
    assert targets["vscode"].mcp.style == "stdio-typed"
    assert targets["vscode"].instructions_path == project / ".github" / "copilot-instructions.md"
    assert targets["gemini"].mcp.path == project / ".gemini" / "settings.json"
    assert targets["gemini"].instructions_path == project / "GEMINI.md"
    assert targets["opencode"].mcp.key == "mcp"
    assert targets["opencode"].mcp.style == "opencode"
    for agent_id in ("codex", "cursor", "opencode"):
        assert targets[agent_id].instructions_path == project / "AGENTS.md"


def test_detection_uses_binary_or_config_dir(fake_home, tmp_path):
    project = tmp_path / "proj"
    assert [t.id for t in agent_targets(project) if t.is_detected()] == []
    (fake_home / ".cursor").mkdir()
    assert [t.id for t in agent_targets(project) if t.is_detected()] == ["cursor"]


def test_server_command_prefers_running_python_environment(tmp_path, monkeypatch):
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    command = env_bin / "phasesweep-mcp"
    command.write_text("#!/bin/sh\n")
    command.chmod(0o755)
    monkeypatch.setattr(installer.sys, "executable", str(env_bin / "python"))
    monkeypatch.setattr(installer.shutil, "which", lambda _name: "/other/bin/phasesweep-mcp")

    assert installer.resolve_server_command() == str(command.resolve())


def test_server_command_refuses_missing_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.sys, "executable", str(tmp_path / "env" / "bin" / "python"))
    monkeypatch.setattr(installer.shutil, "which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="cannot find an executable"):
        installer.resolve_server_command()


def test_entry_styles_and_codex_toml(tmp_path):
    catalog = Path("/proj/catalog.yaml")
    assert mcp_entry("stdio", "/bin/x", catalog) == {
        "command": "/bin/x",
        "args": ["--catalog", "/proj/catalog.yaml"],
    }
    assert mcp_entry("stdio-typed", "/bin/x", catalog)["type"] == "stdio"
    opencode = mcp_entry("opencode", "/bin/x", catalog)
    assert opencode["command"] == ["/bin/x", "--catalog", "/proj/catalog.yaml"]
    assert opencode["type"] == "local"
    parsed = tomllib.loads(codex_toml_content("/bin/x", catalog))
    assert parsed["mcp_servers"]["phasesweep"]["args"] == ["--catalog", "/proj/catalog.yaml"]

    unicode_catalog = Path("/proj/🧪/catalog.yaml")
    unicode_parsed = tomllib.loads(codex_toml_content("/tools/🚀/phasesweep-mcp", unicode_catalog))
    assert unicode_parsed["mcp_servers"]["phasesweep"] == {
        "command": "/tools/🚀/phasesweep-mcp",
        "args": ["--catalog", str(unicode_catalog)],
    }


# --- installer orchestration ---


def _write_valid_catalog(project: Path) -> Path:
    """Write an installer fixture that references the repo example config."""
    return write_mcp_catalog(project, {"example": EXAMPLE_CONFIG})


def test_installer_round_trip_across_all_targets(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)

    code = installer.run(
        "install",
        project,
        catalog,
        list(AGENT_IDS),
        "all",
        yes=True,
        allow_user_scope=True,
    )
    assert code == 0, capsys.readouterr().out
    entry = json.loads((project / ".mcp.json").read_text())["mcpServers"]["phasesweep"]
    assert entry["args"] == ["--catalog", str(catalog)]
    assert entry["command"].endswith("phasesweep-mcp")
    assert "phasesweep" in json.loads((project / ".vscode" / "mcp.json").read_text())["servers"]
    assert "phasesweep" in json.loads((project / "opencode.json").read_text())["mcp"]
    assert MARKDOWN_START in (project / "CLAUDE.md").read_text()
    assert MARKDOWN_START in (project / "AGENTS.md").read_text()
    codex_config = fake_home / ".codex" / "config.toml"
    assert "[mcp_servers.phasesweep]" in codex_config.read_text()

    # Second install is a no-op.
    capsys.readouterr()
    assert (
        installer.run(
            "install",
            project,
            catalog,
            list(AGENT_IDS),
            "all",
            yes=True,
            allow_user_scope=True,
        )
        == 0
    )
    assert "unchanged" in capsys.readouterr().out

    assert installer.run("uninstall", project, None, list(AGENT_IDS), "all", yes=True) == 0
    leftovers = [p for p in project.rglob("*") if p.is_file() and p.name != "catalog.yaml"]
    assert leftovers == []
    assert not codex_config.exists()


def test_shared_instructions_are_removed_only_after_last_owner(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    instructions = project / "AGENTS.md"
    original = "# Project instructions\n"
    instructions.write_text(original)

    assert (
        installer.run("install", project, None, ["codex", "cursor"], "instructions", yes=True) == 0
    )
    installed = instructions.read_text()
    assert "<!-- PHASESWEEP_OWNERS: codex,cursor -->" in installed
    assert not (project / ".cursor" / "mcp.json").exists()

    capsys.readouterr()
    assert installer.run("uninstall", project, None, ["cursor"], "instructions", yes=True) == 0
    assert "retained for: codex" in capsys.readouterr().out
    assert instructions.read_text() == installed.replace(
        "<!-- PHASESWEEP_OWNERS: codex,cursor -->",
        "<!-- PHASESWEEP_OWNERS: codex -->",
    )

    assert installer.run("uninstall", project, None, ["codex"], "instructions", yes=True) == 0
    assert instructions.read_text() == original


def test_installer_refuses_missing_server_command_before_edits(
    fake_home, tmp_path, capsys, monkeypatch
):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)

    def missing_server_command():
        raise FileNotFoundError("missing executable")

    monkeypatch.setattr(installer, "resolve_server_command", missing_server_command)

    code = installer.run("install", project, catalog, ["claude"], "all", yes=True)

    assert code == 1
    assert "no client config was touched" in capsys.readouterr().err
    assert not (project / ".mcp.json").exists()
    assert not (project / "CLAUDE.md").exists()


def test_installer_flags_commented_config_for_manual_merge(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    (project / "opencode.json").write_text('{\n  // keep\n  "mcp": {}\n}\n')

    code = installer.run("install", project, catalog, ["opencode"], "mcp", yes=True)
    out = capsys.readouterr().out
    assert code == 1
    assert "skipped" in out
    assert '"phasesweep"' in out  # manual snippet printed
    assert "// keep" in (project / "opencode.json").read_text()


def test_installer_skips_unmanaged_codex_table(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    codex_config = fake_home / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    original = '[mcp_servers.phasesweep]\ncommand = "custom"\n'
    codex_config.write_text(original)

    code = installer.run(
        "install", project, catalog, ["codex"], "mcp", yes=True, allow_user_scope=True
    )
    assert code == 1
    assert "skipped" in capsys.readouterr().out
    assert codex_config.read_text() == original
    # Uninstall never touches unmanaged tables either.
    assert installer.run("uninstall", project, None, ["codex"], "mcp", yes=True) == 0
    assert codex_config.read_text() == original


@pytest.mark.parametrize(
    "agent_id",
    ["claude", "claude-desktop", "cursor", "vscode", "gemini", "opencode"],
)
def test_installer_preserves_unmanaged_json_entry(fake_home, tmp_path, capsys, agent_id):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    target = next(item for item in agent_targets(project) if item.id == agent_id)
    assert target.mcp is not None
    target.mcp.path.parent.mkdir(parents=True, exist_ok=True)
    original_data = {target.mcp.key: {"phasesweep": {"command": "custom-server"}}}
    target.mcp.path.write_text(json.dumps(original_data, indent=2) + "\n")

    install_code = installer.run(
        "install",
        project,
        catalog,
        [agent_id],
        "mcp",
        yes=True,
        allow_user_scope=agent_id == "claude-desktop",
    )
    install_output = capsys.readouterr().out
    assert install_code == 1
    assert "unmanaged phasesweep entry" in install_output
    assert json.loads(target.mcp.path.read_text()) == original_data

    uninstall_code = installer.run("uninstall", project, None, [agent_id], "mcp", yes=True)
    uninstall_output = capsys.readouterr().out
    assert uninstall_code == 1
    assert "left untouched" in uninstall_output
    assert json.loads(target.mcp.path.read_text()) == original_data


def test_installer_updates_recognizable_managed_json_entry(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    path = project / ".mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "phasesweep": {
                        "command": "/old/bin/phasesweep-mcp",
                        "args": ["--catalog", "/old/catalog.yaml"],
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )

    code = installer.run("install", project, catalog, ["claude"], "mcp", yes=True)

    assert code == 0, capsys.readouterr().out
    entry = json.loads(path.read_text())["mcpServers"]["phasesweep"]
    assert entry["args"] == ["--catalog", str(catalog)]


def test_installer_refuses_project_config_symlink_escape(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / ".cursor").symlink_to(outside, target_is_directory=True)

    code = installer.run("install", project, catalog, ["cursor"], "mcp", yes=True)

    assert code == 1
    assert "resolves outside the project" in capsys.readouterr().out
    assert not (outside / "mcp.json").exists()


@pytest.mark.parametrize(
    "original",
    [
        '[mcp_servers."phasesweep"]\ncommand = "custom"\n',
        'mcp_servers = { phasesweep = { command = "custom" } }\n',
        'mcp_servers.phasesweep.command = "custom"\n',
    ],
)
def test_installer_detects_unmanaged_codex_table_semantically(
    fake_home, tmp_path, capsys, original
):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    codex_config = fake_home / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text(original)

    code = installer.run(
        "install", project, catalog, ["codex"], "mcp", yes=True, allow_user_scope=True
    )

    assert code == 1
    assert "unmanaged" in capsys.readouterr().out
    assert codex_config.read_text() == original


@pytest.mark.parametrize(
    "original",
    [
        "theme = [\n",
        'mcp_servers = { other = { command = "x" } }\n',
    ],
)
def test_installer_refuses_invalid_codex_toml_merge(fake_home, tmp_path, capsys, original):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    codex_config = fake_home / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text(original)

    code = installer.run(
        "install", project, catalog, ["codex"], "mcp", yes=True, allow_user_scope=True
    )

    assert code == 1
    assert "invalid TOML" in capsys.readouterr().out
    assert codex_config.read_text() == original


def test_installer_writes_unicode_codex_catalog_path(fake_home, tmp_path, capsys):
    project = tmp_path / "proj-🧪"
    project.mkdir()
    catalog = _write_valid_catalog(project)

    code = installer.run(
        "install", project, catalog, ["codex"], "mcp", yes=True, allow_user_scope=True
    )

    assert code == 0, capsys.readouterr().out
    parsed = tomllib.loads((fake_home / ".codex" / "config.toml").read_text())
    assert parsed["mcp_servers"]["phasesweep"]["args"] == ["--catalog", str(catalog)]


# --- CLI flow ---


@pytest.mark.parametrize("agent_id", ["codex", "claude-desktop"])
def test_cli_unattended_user_scope_requires_dedicated_acknowledgement(
    fake_home,
    tmp_path,
    agent_id,
):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    target = next(item for item in agent_targets(project) if item.id == agent_id)
    assert target.mcp is not None
    args = [
        "install",
        "--catalog",
        str(catalog),
        "--project",
        str(project),
        "--agent",
        agent_id,
        "--type",
        "mcp",
    ]
    runner = CliRunner()

    preview = runner.invoke(cli_main, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert not target.mcp.path.exists()

    refused = runner.invoke(cli_main, [*args, "--yes"])
    assert refused.exit_code == 2
    assert "--allow-user-scope" in refused.output
    assert "no client config was touched" in refused.output
    assert not target.mcp.path.exists()

    accepted = runner.invoke(cli_main, [*args, "--yes", "--allow-user-scope"])
    assert accepted.exit_code == 0, accepted.output
    assert target.mcp.path.is_file()


def test_cli_install_uninstall_e2e_round_trip(fake_home, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    preexisting_claude_md = "# House rules\n"
    (project / "CLAUDE.md").write_text(preexisting_claude_md)
    runner = CliRunner()

    scaffold = runner.invoke(cli_main, ["init-catalog", "--from", str(EXAMPLE_CONFIG)])
    assert scaffold.exit_code == 0, scaffold.output

    install = runner.invoke(cli_main, ["install", "--agent", "claude", "--type", "all", "--yes"])
    assert install.exit_code == 0, install.output
    assert "restart your mcp client" in install.output.lower()
    entry = json.loads((project / ".mcp.json").read_text())["mcpServers"]["phasesweep"]
    assert entry["args"] == ["--catalog", str(project / "catalog.yaml")]
    claude_md = (project / "CLAUDE.md").read_text()
    assert claude_md.startswith(preexisting_claude_md)
    assert MARKDOWN_START in claude_md and MARKDOWN_END in claude_md

    uninstall = runner.invoke(cli_main, ["uninstall", "--agent", "claude", "--yes"])
    assert uninstall.exit_code == 0, uninstall.output
    assert not (project / ".mcp.json").exists()
    assert (project / "CLAUDE.md").read_text() == preexisting_claude_md


def test_cli_install_and_uninstall_dry_run_never_mutate_client_files(
    fake_home,
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    runner = CliRunner()
    scaffold = runner.invoke(cli_main, ["init-catalog", "--from", str(EXAMPLE_CONFIG)])
    assert scaffold.exit_code == 0, scaffold.output

    preview_install = runner.invoke(
        cli_main,
        ["install", "--agent", "claude", "--type", "all", "--dry-run"],
    )
    assert preview_install.exit_code == 0, preview_install.output
    assert "would-create" in preview_install.output
    assert "nothing was changed" in preview_install.output
    assert not (project / ".mcp.json").exists()
    assert not (project / "CLAUDE.md").exists()

    install = runner.invoke(
        cli_main,
        ["install", "--agent", "claude", "--type", "all", "--yes"],
    )
    assert install.exit_code == 0, install.output
    mcp_before = (project / ".mcp.json").read_bytes()
    instructions_before = (project / "CLAUDE.md").read_bytes()

    preview_uninstall = runner.invoke(
        cli_main,
        ["uninstall", "--agent", "claude", "--type", "all", "--dry-run"],
    )
    assert preview_uninstall.exit_code == 0, preview_uninstall.output
    assert "would-remove" in preview_uninstall.output
    assert "nothing was changed" in preview_uninstall.output
    assert (project / ".mcp.json").read_bytes() == mcp_before
    assert (project / "CLAUDE.md").read_bytes() == instructions_before


def test_cli_install_dry_run_does_not_offer_to_scaffold_missing_catalog(
    fake_home,
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)

    result = CliRunner().invoke(
        cli_main,
        ["install", "--agent", "claude", "--dry-run"],
    )

    assert result.exit_code == 2
    assert "Scaffold one first" in result.output
    assert "experiment config to scaffold" not in result.output
    assert not (project / "catalog.yaml").exists()


def test_cli_install_requires_catalog_when_unattended(fake_home, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    result = CliRunner().invoke(cli_main, ["install", "--agent", "claude", "--yes"])
    assert result.exit_code == 2
    assert "init-catalog" in result.output
    assert not (project / ".mcp.json").exists()


def test_cli_install_offers_catalog_scaffold_interactively(fake_home, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    # Prompt answers: config path to scaffold from, then per-plan confirmation.
    result = CliRunner().invoke(
        cli_main,
        ["install", "--agent", "claude", "--type", "mcp"],
        input=f"{EXAMPLE_CONFIG}\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert (project / "catalog.yaml").is_file()
    assert "phasesweep" in json.loads((project / ".mcp.json").read_text())["mcpServers"]


def test_cli_install_instructions_only_needs_no_catalog_or_sdk(fake_home, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr("phasesweep.cli.importlib.util.find_spec", lambda _name: None)
    result = CliRunner().invoke(
        cli_main, ["install", "--agent", "claude", "--type", "instructions", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert MARKDOWN_START in (project / "CLAUDE.md").read_text()
    assert not (project / ".mcp.json").exists()


@pytest.mark.parametrize("integration", ["mcp", "all"])
def test_cli_install_requires_mcp_sdk_before_client_edits(
    fake_home, tmp_path, monkeypatch, integration
):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    monkeypatch.setattr("phasesweep.cli.importlib.util.find_spec", lambda _name: None)

    result = CliRunner().invoke(
        cli_main,
        [
            "install",
            "--catalog",
            str(catalog),
            "--agent",
            "claude",
            "--type",
            integration,
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "pip install 'phasesweep[mcp]'" in result.output
    assert "no client config was touched" in result.output
    assert not (project / ".mcp.json").exists()
    assert not (project / "CLAUDE.md").exists()


def test_cli_interactive_selection_among_detected_agents(fake_home, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    (fake_home / ".cursor").mkdir()
    _write_valid_catalog(project)
    # Prompt answers: configure Cursor? yes; proceed? yes.
    result = CliRunner().invoke(cli_main, ["install", "--type", "mcp"], input="y\ny\n")
    assert result.exit_code == 0, result.output
    assert "Cursor" in result.output
    assert (project / ".cursor" / "mcp.json").is_file()


def test_cli_install_rejects_invalid_catalog_before_touching_configs(
    fake_home, tmp_path, monkeypatch
):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    (project / "experiment.yaml").write_text("experiment: broken\n")
    (project / "catalog.yaml").write_text(
        f"state_dir: {project}/state\nexperiments:\n  - id: broken\n    config: ./experiment.yaml\n"
    )
    result = CliRunner().invoke(cli_main, ["install", "--agent", "claude", "--yes"])
    assert result.exit_code == 2
    assert "no client config was touched" in result.output
    assert not (project / ".mcp.json").exists()


def test_install_help_is_operator_readable():
    runner = CliRunner()
    install_help = runner.invoke(cli_main, ["install", "--help"], terminal_width=120)
    assert install_help.exit_code == 0
    for flag in (
        "--catalog",
        "--agent",
        "--type",
        "--project",
        "--yes",
        "--allow-user-scope",
        "--dry-run",
    ):
        assert flag in install_help.output
    assert "claude" in install_help.output and "opencode" in install_help.output
    assert "Args:" not in install_help.output

    uninstall_help = runner.invoke(cli_main, ["uninstall", "--help"], terminal_width=120)
    assert uninstall_help.exit_code == 0
    assert "--agent" in uninstall_help.output
    assert "--dry-run" in uninstall_help.output
    assert "--catalog" not in uninstall_help.output
    assert "generated-shape JSON" in uninstall_help.output
    assert "Unmanaged same-name entries stay untouched" in uninstall_help.output
