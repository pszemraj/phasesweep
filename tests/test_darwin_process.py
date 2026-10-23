"""macOS process identity and conservative signal authorization."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from phasesweep.runtime import darwin, reaper


@pytest.fixture
def darwin_reaper(monkeypatch):
    monkeypatch.setattr(reaper, "sys", SimpleNamespace(platform="darwin"))
    return reaper


def test_darwin_namespace_never_matches_linux(darwin_reaper):
    assert darwin_reaper.read_pid_namespace() == "darwin"
    assert darwin_reaper.is_pid_namespace("darwin")
    assert not darwin_reaper.same_pid_namespace("pid:[4026531836]@1")
    assert darwin_reaper.in_initial_pid_namespace()
    assert "cannot inspect" in darwin_reaper.recorded_process_blocker(
        "pid:[4026531836]@1", pids={123}, pgids={123}
    )


@pytest.mark.parametrize("info", [None, darwin.ProcessInfo(102, 123, "R")])
def test_unreadable_or_reused_darwin_pid_never_authorizes_signals(monkeypatch, darwin_reaper, info):
    monkeypatch.setattr(reaper, "is_pid_alive", lambda _pid: True)
    monkeypatch.setattr(reaper, "read_process_info", lambda _pid: info)
    monkeypatch.setattr(reaper, "_process_group_exists", lambda _pgid: True)
    monkeypatch.setattr(
        reaper.os, "killpg", lambda *_args: pytest.fail("unverified identity must not be signalled")
    )
    assert not reaper.is_same_live_process(123, 101)
    assert not reaper.kill_stale_group(123, 101, pgid=123)


def test_darwin_saved_group_leader_requires_exact_starttime(monkeypatch, darwin_reaper):
    monkeypatch.setattr(reaper, "is_pid_alive", lambda _pid: True)
    for info, expected in [
        (None, None),
        (darwin.ProcessInfo(101, 123, "R"), False),
        (darwin.ProcessInfo(102, 123, "R"), True),
        (darwin.ProcessInfo(102, 456, "R"), False),
    ]:
        monkeypatch.setattr(reaper, "read_process_info", lambda _pid, value=info: value)
        assert reaper._stored_pgid_is_reused_group_leader(123, 101) is expected


def test_darwin_zombie_is_not_a_live_runner(monkeypatch, darwin_reaper):
    monkeypatch.setattr(reaper, "is_pid_alive", lambda _pid: True)
    monkeypatch.setattr(reaper, "read_process_info", lambda _pid: darwin.ProcessInfo(101, 123, "Z"))
    assert not reaper.is_same_live_process(123, 101)


def test_darwin_group_probe_fails_closed_without_procfs(monkeypatch, darwin_reaper):
    monkeypatch.setattr(reaper, "_process_group_exists", lambda _pgid: True)
    assert reaper._process_group_alive(123)
    assert not reaper._group_member_pids(123).complete
    monkeypatch.setattr(reaper, "_process_group_exists", lambda _pgid: False)
    assert not reaper._process_group_alive(123)


@pytest.mark.parametrize(
    "length,pid,seconds,microseconds",
    [
        (0, 123, 1, 1),
        (135, 123, 1, 1),
        (136, 124, 1, 1),
        (136, 123, 0, 1),
        (136, 123, 1, 1_000_000),
    ],
)
def test_native_partial_or_invalid_process_info_is_unknown(
    monkeypatch, length, pid, seconds, microseconds
):
    record = darwin._ProcBsdInfo()
    record.pbi_pid = pid
    record.pbi_start_tvsec = seconds
    record.pbi_start_tvusec = microseconds

    def query(_pid, _flavor, _arg, buffer, _size):
        ctypes.memmove(buffer, ctypes.byref(record), ctypes.sizeof(record))
        return length

    monkeypatch.setattr(darwin, "_libproc", lambda: SimpleNamespace(proc_pidinfo=query))
    assert darwin.read_process_info(123) is None


def test_unavailable_native_api_is_unknown(monkeypatch):
    monkeypatch.setattr(darwin, "_libproc", lambda: None)
    assert darwin.read_process_info(123) is None

    def unavailable(*_args, **_kwargs):
        raise OSError("native identity unavailable")

    monkeypatch.setattr(darwin.ctypes, "CDLL", unavailable)
    assert darwin.read_boot_session_uuid() is None


@pytest.mark.skipif(sys.platform != "darwin", reason="reads the native macOS process and boot APIs")
def test_native_darwin_identity_survives_reader_restart():
    pid = os.getpid()
    info = darwin.read_process_info(pid)
    boot_id = reaper.read_boot_id()
    assert info is not None and info.pgrp == os.getpgrp() and info.state != "Z"
    assert boot_id is not None
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from phasesweep.runtime.reaper import read_proc_starttime, read_boot_id; print(read_proc_starttime({pid})); print(read_boot_id())",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.stdout.splitlines() == [str(info.starttime), boot_id]
    assert reaper.is_same_live_process(pid, info.starttime)
    assert not reaper.is_same_live_process(pid, info.starttime + 1)
