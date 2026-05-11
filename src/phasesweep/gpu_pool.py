"""GPU pool: hands out GPU indices to parallel trials, prevents double-booking.

When n_jobs > 1, every trial subprocess would otherwise see the full GPU list and
race for cuda:0. This pool assigns disjoint CUDA_VISIBLE_DEVICES per trial.

If no GPUs are detected (cpu-only box) or n_jobs=1, this is a transparent no-op.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections.abc import Generator
from contextlib import contextmanager

log = logging.getLogger("phasesweep.gpu_pool")


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
        self._semaphore: threading.Semaphore | None = None
        self._available: list[int] = []
        self._lock = threading.Lock()

        if gpu_ids:
            self._semaphore = threading.Semaphore(len(gpu_ids))
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
            explicit_ids: GPU indices from YAML config. If None, auto-detect when n_jobs>1.
                Explicit IDs are honored even for n_jobs==1 — single-job phases on shared
                hardware still need device isolation (review item #1).
            allow_no_gpu: if True, n_jobs > 1 with no GPUs is allowed (CPU-only sweep).
                If False (default), that combination raises RuntimeError.

        Returns:
            A configured :class:`GpuPool`. The pool is "active" (hands out
            non-``None`` IDs) iff a GPU list is in play.

        Raises:
            RuntimeError: ``explicit_ids`` was provided but empty; or
                ``CUDA_VISIBLE_DEVICES`` is set to non-numeric IDs and ``n_jobs > 1``;
                or no GPUs are visible and ``n_jobs > 1`` without ``allow_no_gpu``.

        """
        # Explicit IDs always win, even at n_jobs==1. Without explicit IDs, single-job
        # phases are a transparent no-op and inherit the caller's CUDA_VISIBLE_DEVICES.
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

        if n_jobs <= 1:
            return cls(gpu_ids=[])

        user_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if user_cvd is not None:
            raw = [x.strip() for x in user_cvd.split(",") if x.strip()]
            if raw and not all(x.isdigit() for x in raw):
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES contains non-numeric device identifiers "
                    f"({user_cvd!r}). Set phase.gpu_ids explicitly with integer indices, "
                    "or add string/UUID GPU ID support before using MIG/UUID devices."
                )
            ids = [int(x) for x in raw]
        else:
            ids = _detect_gpu_ids()
        if not ids:
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

    @property
    def active(self) -> bool:
        """Whether the pool has GPUs to hand out.

        Returns:
            ``True`` iff ``acquire()`` yields a real integer device index;
            ``False`` for the no-op (CPU-only or single-job) pool.

        """
        return bool(self._gpu_ids)

    def _acquire(self) -> int | None:
        """Block until a GPU is free, then claim it.

        Returns:
            A GPU index, or ``None`` if the pool is inactive (no isolation
            requested; caller's environment is passed through unchanged).

        """
        if not self._semaphore:
            return None
        self._semaphore.acquire()
        with self._lock:
            return self._available.pop(0)

    def _release(self, gpu_id: int | None) -> None:
        """Return a previously-acquired GPU index to the pool.

        Args:
            gpu_id: The index returned by :meth:`_acquire`. ``None`` is a
                no-op (inactive pool).

        """
        if gpu_id is None or not self._semaphore:
            return
        with self._lock:
            self._available.append(gpu_id)
        self._semaphore.release()

    @contextmanager
    def acquire(self) -> Generator[int | None, None, None]:
        """Block until a GPU is available, yield its index, release on exit.

        Yields:
            The GPU index for this critical section, or ``None`` when the
            pool is inactive.

        """
        gpu_id = self._acquire()
        try:
            yield gpu_id
        finally:
            self._release(gpu_id)
