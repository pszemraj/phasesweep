"""Native macOS process and boot identity, without optional dependencies."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import cache
from uuid import UUID


class _ProcBsdInfo(ctypes.Structure):
    """The public ``proc_bsdinfo`` layout from macOS ``sys/proc_info.h``."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]
    pbi_pid: int
    pbi_pgid: int
    pbi_status: int
    pbi_start_tvsec: int
    pbi_start_tvusec: int


@dataclass(frozen=True)
class ProcessInfo:
    """One kernel snapshot of process start time, group and zombie state."""

    starttime: int
    pgrp: int
    state: str


@cache
def _libproc() -> ctypes.CDLL | None:
    """Load the native process API, failing closed when it is unavailable."""
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.proc_pidinfo.restype = ctypes.c_int
    except (OSError, AttributeError):
        return None
    return library


def read_process_info(pid: int) -> ProcessInfo | None:
    """Read the kernel's microsecond start time and group in one snapshot.

    ``PROC_PIDTBSDINFO`` supplies the same start timeval as ``kinfo_proc``
    through the smaller public libproc ABI. Partial or unreadable records
    never supply an identity that could authorize a signal.

    :param int pid: Positive process identifier to inspect.
    :return ProcessInfo | None: Complete process information, or unknown.
    """
    if not 0 < pid < 2**31:
        return None
    library = _libproc()
    if library is None:
        return None
    info = _ProcBsdInfo()
    size = ctypes.sizeof(info)
    result = library.proc_pidinfo(pid, 3, 0, ctypes.byref(info), size)
    if result != size or info.pbi_pid != pid:
        return None
    if info.pbi_start_tvsec <= 0 or not 0 <= info.pbi_start_tvusec < 1_000_000:
        return None
    return ProcessInfo(
        starttime=info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec,
        pgrp=info.pbi_pgid,
        state="Z" if info.pbi_status == 5 else "R",  # SZOMB in sys/proc.h.
    )


def read_boot_session_uuid() -> str | None:
    """Return ``kern.bootsessionuuid``, or unknown on any read or parse failure."""
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        query = library.sysctlbyname
        query.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(64)
        size = ctypes.c_size_t(len(buffer))
        if query(b"kern.bootsessionuuid", buffer, ctypes.byref(size), None, 0) != 0:
            return None
        if size.value != 37:
            return None
        return str(UUID(buffer.value.decode("ascii")))
    except (OSError, AttributeError, UnicodeError, ValueError):
        return None
