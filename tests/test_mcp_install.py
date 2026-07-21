"""Installer behavior: file-edit primitives, agent targets, and the CLI flow."""

from __future__ import annotations

import json
import math
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
from phasesweep.mcp.install.edits import removed_marked_text, updated_marked_text
from phasesweep.mcp.install.targets import (
    MARKDOWN_END,
    MARKDOWN_START,
    TOML_END,
    TOML_START,
    agent_ids,
    agent_targets,
    codex_toml_content,
    is_managed_mcp_entry,
    mcp_entry,
)
from tests.mcp_helpers import write_mcp_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mcp_experiment.yaml"

ENTRY = {"command": "/venv/bin/phasesweep-mcp", "args": ["--catalog", "/proj/catalog.yaml"]}
ALL_AGENT_IDS = agent_ids()


def _is_test_entry(value: object) -> bool:
    """Recognize the managed entry used by low-level edit tests."""
    return value == ENTRY


merge_json_member = partial(install_edits.merge_json_member, managed=_is_test_entry)
remove_json_member = partial(install_edits.remove_json_member, managed=_is_test_entry)


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


def test_config_edits_apply_umask_to_new_paths_and_preserve_existing_mode(tmp_path):
    path = tmp_path / "cfg" / "mcp.json"
    previous_umask = os.umask(0o002)
    try:
        assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "created"
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o775
    assert stat.S_IMODE(path.stat().st_mode) == 0o664

    path.chmod(0o640)
    replacement = {**ENTRY, "env": {"UPDATED": "1"}}
    assert (
        install_edits.merge_json_member(
            path,
            "mcpServers",
            "phasesweep",
            replacement,
            managed=lambda _entry: True,
        )
        == "updated"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


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


def test_merge_json_member_preserves_literal_unicode(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text('{"label":"café","mcpServers":{}}\n')

    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "updated"

    text = path.read_text()
    assert "café" in text
    assert "\\u00e9" not in text


def test_merge_json_member_skips_commented_config(tmp_path):
    path = tmp_path / "opencode.json"
    original = '{\n  // my settings\n  "mcp": {}\n}\n'
    path.write_text(original)
    assert merge_json_member(path, "mcp", "phasesweep", ENTRY) == "skipped"
    assert path.read_text() == original


@pytest.mark.parametrize(
    "original",
    [
        b'{"theme":"first","theme":"second","mcpServers":{}}\n',
        b'{"threshold":1e400,"mcpServers":{}}\n',
        b'{"threshold":NaN,"mcpServers":{}}\n',
    ],
)
def test_merge_json_member_rejects_ambiguous_or_nonfinite_json(tmp_path, original):
    path = tmp_path / "mcp.json"
    path.write_bytes(original)

    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "skipped"
    assert path.read_bytes() == original


def test_merge_json_member_refuses_nonfinite_generated_entry(tmp_path):
    path = tmp_path / "mcp.json"
    entry = {**ENTRY, "threshold": math.inf}

    assert merge_json_member(path, "mcpServers", "phasesweep", entry) == "error"
    assert not path.exists()


def test_json_edits_refuse_invalid_utf8_without_raising(tmp_path):
    path = tmp_path / "mcp.json"
    original = b"\xff\xfe"
    path.write_bytes(original)

    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "error"
    assert remove_json_member(path, "mcpServers", "phasesweep") == "error"
    assert path.read_bytes() == original


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


def test_remove_json_member_retains_unowned_container_and_file(tmp_path):
    path = tmp_path / "mcp.json"
    merge_json_member(path, "mcpServers", "phasesweep", ENTRY)
    assert remove_json_member(path, "mcpServers", "phasesweep") == "removed"
    assert json.loads(path.read_text()) == {"mcpServers": {}}

    path.write_text(json.dumps({"theme": "dark", "mcpServers": {"phasesweep": ENTRY}}, indent=2))
    assert remove_json_member(path, "mcpServers", "phasesweep") == "removed"
    assert json.loads(path.read_text()) == {"theme": "dark", "mcpServers": {}}
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
    assert target.read_text() == "{}\n"


def test_atomic_edit_failure_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    original = '{"mcpServers": {"other": {"command": "x"}}}\n'
    path.write_text(original)

    def fail_replace(_source, _destination, **_kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(install_edits.os, "replace", fail_replace)

    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "error"
    assert path.read_text() == original
    assert list(tmp_path.glob(".mcp.json.*.tmp")) == []


def test_atomic_edit_refuses_external_change_before_replace(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    path.write_text('{"mcpServers": {"other": {"command": "x"}}}\n')
    external = '{"changed_by": "another process"}\n'
    real_new_temporary_fd = install_edits._new_temporary_fd

    def change_after_temp_creation(parent_fd, leaf, mode):
        fd, name = real_new_temporary_fd(parent_fd, leaf, mode)
        path.write_text(external)
        return fd, name

    monkeypatch.setattr(install_edits, "_new_temporary_fd", change_after_temp_creation)

    assert merge_json_member(path, "mcpServers", "phasesweep", ENTRY) == "error"
    assert path.read_text() == external
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


# --- marker-fenced text transforms ---


@pytest.mark.parametrize(
    "original",
    ["# My project\n\nHouse rules.\n", "# My project", "before\r\n", "   \r\n", ""],
)
def test_marked_text_round_trip_is_identical(original):
    updated = updated_marked_text(
        original,
        "body",
        start=MARKDOWN_START,
        end=MARKDOWN_END,
    )
    assert removed_marked_text(updated, start=MARKDOWN_START, end=MARKDOWN_END) == original


def test_marked_text_replaces_in_place_and_preserves_crlf():
    existing = f"before\r\n\r\n{MARKDOWN_START}\r\nold\r\n{MARKDOWN_END}\r\n"
    updated = updated_marked_text(
        existing,
        "new",
        start=MARKDOWN_START,
        end=MARKDOWN_END,
    )
    assert updated == f"before\r\n\r\n{MARKDOWN_START}\r\nnew\r\n{MARKDOWN_END}\r\n"
    assert removed_marked_text(updated, start=MARKDOWN_START, end=MARKDOWN_END) == "before\r\n"


def test_marker_text_must_be_an_exact_standalone_line():
    original = f"mention {MARKDOWN_START} inline and leave it alone\n"
    updated = updated_marked_text(
        original,
        "body",
        start=MARKDOWN_START,
        end=MARKDOWN_END,
    )
    assert updated.startswith(original)
    assert updated.count(MARKDOWN_START) == 2
    assert removed_marked_text(updated, start=MARKDOWN_START, end=MARKDOWN_END) == original


def test_marked_text_refuses_unmatched_marker():
    original = f"before\n{MARKDOWN_START}\nunterminated\n"
    with pytest.raises(ValueError):
        updated_marked_text(original, "new", start=MARKDOWN_START, end=MARKDOWN_END)
    with pytest.raises(ValueError):
        removed_marked_text(original, start=MARKDOWN_START, end=MARKDOWN_END)


# --- targets ---


def test_agent_target_scopes(fake_home, tmp_path):
    project = tmp_path / "proj"
    targets = {target.id: target for target in agent_targets(project)}

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


def test_server_command_preserves_lexical_symlink_name(tmp_path, monkeypatch):
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    target = env_bin / "shared-launcher"
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    command = env_bin / "phasesweep-mcp"
    command.symlink_to(target.name)
    monkeypatch.setattr(installer.sys, "executable", str(env_bin / "python"))
    monkeypatch.setattr(installer.shutil, "which", lambda _name: None)

    resolved = installer.resolve_server_command()

    assert resolved == str(command.absolute())
    assert is_managed_mcp_entry("stdio", mcp_entry("stdio", resolved, Path("/p/catalog.yaml")))


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
        list(ALL_AGENT_IDS),
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
            list(ALL_AGENT_IDS),
            "all",
            yes=True,
            allow_user_scope=True,
        )
        == 0
    )
    assert "unchanged" in capsys.readouterr().out

    assert installer.run("uninstall", project, None, list(ALL_AGENT_IDS), "all", yes=True) == 0
    for target in agent_targets(project):
        if target.mcp.format == "toml":
            assert target.mcp.path.read_bytes() == b""
        else:
            data = json.loads(target.mcp.path.read_text())
            assert "phasesweep" not in data[target.mcp.key]
    instruction_paths = {
        target.instructions_path
        for target in agent_targets(project)
        if target.instructions_path is not None
    }
    assert all(path.read_bytes() == b"" for path in instruction_paths)


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


def test_concurrent_shared_instruction_installs_preserve_both_owners(
    fake_home, tmp_path, monkeypatch
):
    project = tmp_path / "proj"
    project.mkdir()
    targets = {target.id: target for target in agent_targets(project)}
    first_read = threading.Event()
    release_first = threading.Event()
    second_read_while_first_locked = threading.Event()
    second_lock_attempt = threading.Event()
    call_lock = threading.Lock()
    calls = 0
    lock_attempts = 0
    real_read = install_edits._read_editable_text
    real_flock = install_edits.fcntl.flock

    def controlled_read(path):
        nonlocal calls
        with call_lock:
            calls += 1
            call = calls
        if call == 1:
            first_read.set()
            assert release_first.wait(timeout=3)
        elif not release_first.is_set():
            second_read_while_first_locked.set()
        return real_read(path)

    def observed_flock(handle, operation):
        nonlocal lock_attempts
        if operation == install_edits.fcntl.LOCK_EX:
            with call_lock:
                lock_attempts += 1
                if lock_attempts == 2:
                    second_lock_attempt.set()
        return real_flock(handle, operation)

    monkeypatch.setattr(install_edits, "_read_editable_text", controlled_read)
    monkeypatch.setattr(install_edits.fcntl, "flock", observed_flock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            installer._apply_instructions, targets["cursor"], "install", project, False
        )
        assert first_read.wait(timeout=3)
        second = pool.submit(
            installer._apply_instructions,
            targets["opencode"],
            "install",
            project,
            False,
        )
        assert second_lock_attempt.wait(timeout=3)
        assert not second_read_while_first_locked.is_set()
        release_first.set()
        assert first.result(timeout=3).action == "created"
        assert second.result(timeout=3).action == "updated"

    text = (project / "AGENTS.md").read_text()
    assert "<!-- PHASESWEEP_OWNERS: cursor,opencode -->" in text


def test_installer_refuses_invalid_utf8_instructions_without_raising(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    instructions = project / "AGENTS.md"
    original = b"\xff\xfe"
    instructions.write_bytes(original)

    code = installer.run("install", project, None, ["cursor"], "instructions", yes=True)

    assert code == 1
    assert "readable regular UTF-8" in capsys.readouterr().out
    assert instructions.read_bytes() == original


def test_installer_reports_repeated_instruction_markers_separately_from_ownership(
    fake_home, tmp_path, capsys
):
    project = tmp_path / "proj"
    project.mkdir()
    instructions = project / "AGENTS.md"
    original = (
        f"{MARKDOWN_START}\nfirst\n{MARKDOWN_END}\n{MARKDOWN_START}\nsecond\n{MARKDOWN_END}\n"
    )
    instructions.write_text(original)

    code = installer.run("install", project, None, ["cursor"], "instructions", yes=True)

    output = capsys.readouterr().out
    assert code == 1
    assert "incomplete or repeated PhaseSweep markers" in output
    assert "invalid ownership metadata" not in output
    assert instructions.read_text() == original


def test_installer_identifies_symlinked_instruction_file(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    target = project / "instructions-target.md"
    target.write_text("# Keep me\n")
    (project / "AGENTS.md").symlink_to(target.name)

    code = installer.run("install", project, None, ["cursor"], "instructions", yes=True)

    assert code == 1
    assert "instructions path is a symlink" in capsys.readouterr().out
    assert target.read_text() == "# Keep me\n"


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


@pytest.mark.parametrize("path_kind", ["directory", "symlink"])
def test_installer_reports_non_regular_json_config(fake_home, tmp_path, capsys, path_kind):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    config = project / ".mcp.json"
    if path_kind == "directory":
        config.mkdir()
    else:
        target = project / "target.json"
        target.write_text("{}\n")
        config.symlink_to(target)

    code = installer.run("install", project, catalog, ["claude"], "mcp", yes=True)

    assert code == 1
    assert "config path is not a regular file" in capsys.readouterr().out


@pytest.mark.parametrize(
    "agent_id",
    ["claude", "claude-desktop", "cursor", "vscode", "gemini", "opencode"],
)
def test_installer_preserves_unmanaged_json_entry(fake_home, tmp_path, capsys, agent_id):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    target = next(item for item in agent_targets(project) if item.id == agent_id)
    target.mcp.path.parent.mkdir(parents=True, exist_ok=True)
    unmanaged_entry = {"command": "custom-server"}
    if agent_id == "claude":
        unmanaged_entry = {
            "command": "/manual/bin/phasesweep-mcp",
            "args": ["--catalog", "relative.yaml"],
            "env": {"CUSTOM": "1"},
        }
    original_data = {target.mcp.key: {"phasesweep": unmanaged_entry}}
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
    assert entry["command"] == installer.resolve_server_command()
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


def test_installer_refuses_project_parent_symlink_swap(fake_home, tmp_path, capsys, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_config = outside / "mcp.json"
    original = '{"outside": true}\n'
    outside_config.write_text(original)
    config_parent = project / ".cursor"
    real_open_directory_fd = install_edits._open_directory_fd
    swapped = False

    def swap_before_open(path, **kwargs):
        nonlocal swapped
        if path == config_parent and not swapped:
            swapped = True
            config_parent.symlink_to(outside, target_is_directory=True)
        return real_open_directory_fd(path, **kwargs)

    monkeypatch.setattr(install_edits, "_open_directory_fd", swap_before_open)

    code = installer.run("install", project, catalog, ["cursor"], "mcp", yes=True)

    assert code == 1
    assert swapped
    assert "error" in capsys.readouterr().out
    assert outside_config.read_text() == original
    assert list(outside.glob(".mcp.json.*.tmp")) == []


@pytest.mark.parametrize(
    "original",
    [
        '[mcp_servers.phasesweep]\ncommand = "custom"\n',
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
    assert installer.run("uninstall", project, None, ["codex"], "mcp", yes=True) == 0
    assert codex_config.read_text() == original


def test_installer_refuses_toml_markers_inside_string_data(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    codex_config = fake_home / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    original = f'note = """\n{TOML_START}\nold\n{TOML_END}\n"""\n'.encode()
    codex_config.write_bytes(original)

    install_code = installer.run(
        "install", project, catalog, ["codex"], "mcp", yes=True, allow_user_scope=True
    )

    assert install_code == 1
    assert "recognizable managed" in capsys.readouterr().out
    assert codex_config.read_bytes() == original
    assert "mcp_servers" not in tomllib.loads(codex_config.read_text())

    uninstall_code = installer.run("uninstall", project, None, ["codex"], "mcp", yes=True)
    assert uninstall_code == 1
    assert "recognizable managed" in capsys.readouterr().out
    assert codex_config.read_bytes() == original


def test_installer_refuses_invalid_utf8_toml_without_raising(fake_home, tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    codex_config = fake_home / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    original = b"\xff\xfe"
    codex_config.write_bytes(original)

    code = installer.run(
        "install", project, catalog, ["codex"], "mcp", yes=True, allow_user_scope=True
    )

    assert code == 1
    assert "readable regular UTF-8" in capsys.readouterr().out
    assert codex_config.read_bytes() == original


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


def test_cli_catalog_scaffold_honors_umask(tmp_path):
    output = tmp_path / "project" / "catalog.yaml"
    previous_umask = os.umask(0o002)
    try:
        result = CliRunner().invoke(
            cli_main,
            ["mcp", "init-catalog", "--from", str(EXAMPLE_CONFIG), "-o", str(output)],
        )
    finally:
        os.umask(previous_umask)

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(output.stat().st_mode) == 0o664


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
    args = [
        "mcp",
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

    scaffold = runner.invoke(cli_main, ["mcp", "init-catalog", "--from", str(EXAMPLE_CONFIG)])
    assert scaffold.exit_code == 0, scaffold.output

    install = runner.invoke(
        cli_main, ["mcp", "install", "--agent", "claude", "--type", "all", "--yes"]
    )
    assert install.exit_code == 0, install.output
    assert "restart your mcp client" in install.output.lower()
    entry = json.loads((project / ".mcp.json").read_text())["mcpServers"]["phasesweep"]
    assert entry["args"] == ["--catalog", str(project / "catalog.yaml")]
    claude_md = (project / "CLAUDE.md").read_text()
    assert claude_md.startswith(preexisting_claude_md)
    assert MARKDOWN_START in claude_md and MARKDOWN_END in claude_md

    uninstall = runner.invoke(cli_main, ["mcp", "uninstall", "--agent", "claude", "--yes"])
    assert uninstall.exit_code == 0, uninstall.output
    assert json.loads((project / ".mcp.json").read_text()) == {"mcpServers": {}}
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
    scaffold = runner.invoke(cli_main, ["mcp", "init-catalog", "--from", str(EXAMPLE_CONFIG)])
    assert scaffold.exit_code == 0, scaffold.output

    preview_install = runner.invoke(
        cli_main,
        ["mcp", "install", "--agent", "claude", "--type", "all", "--dry-run"],
    )
    assert preview_install.exit_code == 0, preview_install.output
    assert "would-create" in preview_install.output
    assert "no client files were changed" in preview_install.output
    assert not (project / ".mcp.json").exists()
    assert not (project / "CLAUDE.md").exists()

    install = runner.invoke(
        cli_main,
        ["mcp", "install", "--agent", "claude", "--type", "all", "--yes"],
    )
    assert install.exit_code == 0, install.output
    mcp_before = (project / ".mcp.json").read_bytes()
    instructions_before = (project / "CLAUDE.md").read_bytes()

    preview_uninstall = runner.invoke(
        cli_main,
        ["mcp", "uninstall", "--agent", "claude", "--type", "all", "--dry-run"],
    )
    assert preview_uninstall.exit_code == 0, preview_uninstall.output
    assert "would-remove" in preview_uninstall.output
    assert "no client files were changed" in preview_uninstall.output
    assert (project / ".mcp.json").read_bytes() == mcp_before
    assert (project / "CLAUDE.md").read_bytes() == instructions_before


def test_cli_install_provisions_catalog_state_before_client_edits(
    fake_home,
    tmp_path,
):
    project = tmp_path / "proj"
    project.mkdir()
    catalog = _write_valid_catalog(project)
    state_dir = project / "state"
    args = [
        "mcp",
        "install",
        "--catalog",
        str(catalog),
        "--project",
        str(project),
        "--agent",
        "claude",
        "--type",
        "mcp",
    ]
    runner = CliRunner()

    preview = runner.invoke(cli_main, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert "no client files were changed" in preview.output
    assert (state_dir / "runs").is_dir()
    assert (state_dir / "logs").is_dir()

    rejected = runner.invoke(cli_main, args, input="n\n")
    assert rejected.exit_code == 2, rejected.output
    assert "cancelled; no client files were changed" in rejected.output
    assert (state_dir / "runs").is_dir()
    assert (state_dir / "logs").is_dir()

    accepted = runner.invoke(cli_main, args, input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert (state_dir / "runs").is_dir()
    assert (state_dir / "logs").is_dir()


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
        ["mcp", "install", "--agent", "claude", "--dry-run"],
    )

    assert result.exit_code == 2
    assert "Scaffold one first" in result.output
    assert "experiment config to scaffold" not in result.output
    assert not (project / "catalog.yaml").exists()


def test_cli_install_requires_catalog_when_unattended(fake_home, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    result = CliRunner().invoke(cli_main, ["mcp", "install", "--agent", "claude", "--yes"])
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
        ["mcp", "install", "--agent", "claude", "--type", "mcp"],
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
        cli_main,
        ["mcp", "install", "--agent", "claude", "--type", "instructions", "--yes"],
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
            "mcp",
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
    # Prompt answers: accept the detected default selection; proceed.
    result = CliRunner().invoke(cli_main, ["mcp", "install", "--type", "mcp"], input="\ny\n")
    assert result.exit_code == 0, result.output
    assert "Select coding agents" in result.output
    assert "[x] Cursor" in result.output
    assert "[ ] Claude Code" in result.output
    assert "Cursor" in result.output
    assert (project / ".cursor" / "mcp.json").is_file()


def test_cli_interactive_selection_can_choose_undetected_agent(fake_home, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    _write_valid_catalog(project)

    # With no detected clients, Cursor is fourth in the complete supported-client menu.
    result = CliRunner().invoke(
        cli_main,
        ["mcp", "install", "--type", "mcp"],
        input="9\n4\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert "[ ] Cursor" in result.output
    assert "invalid selection '9'" in result.output
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
    result = CliRunner().invoke(cli_main, ["mcp", "install", "--agent", "claude", "--yes"])
    assert result.exit_code == 2
    assert "no client config was touched" in result.output
    assert not (project / ".mcp.json").exists()


def test_install_help_is_operator_readable():
    runner = CliRunner()
    install_help = runner.invoke(cli_main, ["mcp", "install", "--help"], terminal_width=120)
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

    uninstall_help = runner.invoke(cli_main, ["mcp", "uninstall", "--help"], terminal_width=120)
    assert uninstall_help.exit_code == 0
    assert "--agent" in uninstall_help.output
    assert "--dry-run" in uninstall_help.output
    assert "--catalog" not in uninstall_help.output
