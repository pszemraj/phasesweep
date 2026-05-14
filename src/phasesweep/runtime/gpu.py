"""GPU pool: hands out GPU indices to parallel trials, prevents double-booking.

Every trial subprocess gets at most one visible CUDA device when phasesweep can
resolve numeric device IDs. If no GPUs are visible, single-job CPU work remains
a transparent no-op; parallel CPU work requires an explicit opt-in.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from phasesweep.runtime.files import lock_dir, try_lock_file, unlock_file

log = logging.getLogger("phasesweep.runtime.gpu")


@dataclass
class _HostGpuLease:
    """Host-wide flock handle for one CUDA device index."""

    gpu_id: int
    handle: IO[str]


def _gpu_lock_path(gpu_id: int) -> Path:
    """Return the host-wide lock file for a CUDA device index."""
    return lock_dir() / f"gpu_{gpu_id}.lock"


def _try_host_gpu_lease(gpu_id: int) -> _HostGpuLease | None:
    """Try to acquire the per-GPU host lock without blocking."""
    handle = try_lock_file(_gpu_lock_path(gpu_id))
    if handle is None:
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return _HostGpuLease(gpu_id=gpu_id, handle=handle)


def _release_host_gpu_lease(lease: _HostGpuLease | None) -> None:
    """Release a host-wide GPU lock."""
    if lease is None:
        return
    unlock_file(lease.handle)


def _detect_gpu_ids() -> list[int]:
    """Probe ``nvidia-smi`` for visible CUDA device indices.

    Returns:
        Numeric device indices reported by ``nvidia-smi --query-gpu=index``,
        or an empty list if the binary is missing, times out, or returns a
        nonzero exit code (this is normal on CPU-only machines).

    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return []
        return [int(x.strip()) for x in out.stdout.strip().splitlines() if x.strip().isdigit()]
    except FileNotFoundError:
        return []
    except Exception:  # noqa: BLE001
        return []


class GpuPool:
    """Thread-safe pool of GPU indices.

    Usage:
        pool = GpuPool.create(n_jobs=4)
        with pool.acquire() as gpu_id:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else ""
            run_trial(...)
    """

    def __init__(self, gpu_ids: list[int]) -> None:
        """Build a pool from a fixed list of GPU indices.

        Args:
            gpu_ids: The GPU indices to manage. If empty, the pool is a
                transparent no-op that always yields ``None`` (single-job
                or CPU-only mode). Prefer :meth:`create` for the normal
                construction path that handles auto-detection and policy.

        """
        self._gpu_ids = gpu_ids
        self._available: list[int] = []
        self._condition = threading.Condition()

        if gpu_ids:
            self._available = list(gpu_ids)

    @classmethod
    def create(
        cls,
        n_jobs: int,
        explicit_ids: list[int] | None = None,
        allow_no_gpu: bool = False,
    ) -> GpuPool:
        """Build a pool, applying phasesweep's GPU isolation policy.

        Args:
            n_jobs: number of parallel trials.
            explicit_ids: GPU indices from YAML config. If ``None``, auto-detect
                numeric visible devices even for ``n_jobs == 1`` so independent
                single-job phasesweep processes do not double-book cuda:0.
            allow_no_gpu: if ``True``, run without CUDA isolation when no numeric
                GPU IDs can be resolved. Parallel CPU-only sweeps need this opt-in.

        Returns:
            A configured :class:`GpuPool`. The pool is "active" (hands out
            non-``None`` IDs) iff a GPU list is in play.

        Raises:
            RuntimeError: ``explicit_ids`` was provided but empty; or
                ``CUDA_VISIBLE_DEVICES`` is set to non-numeric IDs without
                ``allow_no_gpu``; or no GPUs are visible and ``n_jobs > 1``
                without ``allow_no_gpu``.

        """
        # Explicit IDs always win, even at n_jobs==1.
        if explicit_ids is not None:
            ids = list(dict.fromkeys(explicit_ids))  # dedupe, preserve order
            if not ids:
                raise RuntimeError("gpu_ids was provided but empty.")
            if n_jobs > len(ids):
                log.warning(
                    "n_jobs=%d but only %d GPU(s) configured (%s). "
                    "Excess trials will queue for a GPU.",
                    n_jobs,
                    len(ids),
                    ids,
                )
            else:
                log.info("GPU pool: %d GPUs for %d parallel job(s).", len(ids), n_jobs)
            return cls(gpu_ids=ids)

        user_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if user_cvd is not None:
            raw = [x.strip() for x in user_cvd.split(",") if x.strip()]
            if raw and not all(x.isdigit() for x in raw):
                if allow_no_gpu:
                    log.warning(
                        "CUDA_VISIBLE_DEVICES=%r contains non-numeric device identifiers; "
                        "running without CUDA isolation because allow_no_gpu_isolation=true.",
                        user_cvd,
                    )
                    return cls(gpu_ids=[])
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES contains non-numeric device identifiers "
                    f"({user_cvd!r}). Set phase.gpu_ids explicitly with integer indices, "
                    "or set allow_no_gpu_isolation: true if this run intentionally "
                    "does not need CUDA isolation."
                )
            ids = [int(x) for x in raw]
        else:
            ids = _detect_gpu_ids()
        if not ids:
            if n_jobs <= 1:
                log.info("No GPUs detected; single-job phase will run without CUDA isolation.")
                return cls(gpu_ids=[])
            if allow_no_gpu:
                log.warning(
                    "n_jobs=%d, no GPUs detected — running without CUDA_VISIBLE_DEVICES "
                    "isolation (allow_no_gpu_isolation: true).",
                    n_jobs,
                )
                return cls(gpu_ids=[])
            raise RuntimeError(
                f"n_jobs={n_jobs} but no GPUs detected. Set gpu_ids explicitly in "
                f"the phase config, or set allow_no_gpu_isolation: true if this is "
                f"an intentional CPU-only parallel sweep."
            )
        if n_jobs > len(ids):
            log.warning(
                "n_jobs=%d but only %d GPU(s) available (%s). Excess trials will queue for a GPU.",
                n_jobs,
                len(ids),
                ids,
            )
        else:
            log.info("GPU pool: %d GPUs for %d parallel jobs.", len(ids), n_jobs)
        return cls(gpu_ids=ids)

    def _acquire(self) -> tuple[int, _HostGpuLease] | None:
        """Block until a local slot and host-wide GPU lease are available.

        Returns:
            ``(gpu_id, lease)`` or ``None`` if the pool is inactive (no
            isolation requested; caller's environment is passed through
            unchanged).

        """
        if not self._gpu_ids:
            return None
        while True:
            with self._condition:
                while not self._available:
                    self._condition.wait()
                candidates = list(self._available)
                self._available.clear()

            remaining: list[int] = []
            for index, gpu_id in enumerate(candidates):
                lease = _try_host_gpu_lease(gpu_id)
                if lease is not None:
                    remaining.extend(candidates[index + 1 :])
                    with self._condition:
                        self._available.extend(remaining)
                        self._condition.notify_all()
                    log.debug("GPU %d host lease acquired", gpu_id)
                    return gpu_id, lease
                remaining.append(gpu_id)

            with self._condition:
                self._available.extend(remaining)
                self._condition.notify_all()
            time.sleep(0.2)

    def _release(self, acquired: tuple[int, _HostGpuLease] | None) -> None:
        """Return a previously-acquired GPU index to the pool.

        Args:
            acquired: Pair returned by :meth:`_acquire`. ``None`` is a no-op
                (inactive pool).

        """
        if acquired is None:
            return
        gpu_id, lease = acquired
        _release_host_gpu_lease(lease)
        with self._condition:
            self._available.append(gpu_id)
            self._condition.notify()

    @contextmanager
    def acquire(self) -> Generator[int | None, None, None]:
        """Block until a GPU is available, yield its index, release on exit.

        Yields:
            The GPU index for this critical section, or ``None`` when the
            pool is inactive.

        """
        acquired = self._acquire()
        try:
            yield None if acquired is None else acquired[0]
        finally:
            self._release(acquired)
