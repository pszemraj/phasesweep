"""GPU pool: hands out CUDA device tokens and prevents double-booking.

Every trial subprocess gets at most one visible CUDA device when phasesweep can
resolve CUDA device tokens. If no GPUs are visible, single-job CPU work remains
a transparent no-op; parallel CPU work requires an explicit opt-in.

Host locks key on a *canonical physical device identity*, not on the spelling of
the configured token: at pool construction every numeric index is resolved to
the GPU UUID ``nvidia-smi`` reports for it, so ``gpu_ids: [0]`` in one run and
``gpu_devices: ["GPU-<uuid of 0>"]`` in another contend for one lock file
instead of double-booking the card (review v0.5.17 / blocker 6).

MIG limitation: a ``MIG-...`` token locks on the MIG instance itself. phasesweep
does not bind a MIG instance to its parent GPU, so a run holding a MIG instance
and a run holding the whole parent device do not exclude each other.

Scope caveat: "host-wide" throughout this module means "every orchestrator that
resolves the same phasesweep lock directory". The default lock directory is
per-user (``~/.cache/phasesweep/locks``); coordinating across users or launch
surfaces with different ``HOME``s requires a shared ``PHASESWEEP_LOCK_DIR``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from phasesweep.config.models import GpuPolicy
from phasesweep.runtime.files import lock_dir, try_lock_file, unlock_file

log = logging.getLogger("phasesweep.runtime.gpu")

_SAFE_LOCK_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")


class GpuLeaseTimeoutError(TimeoutError):
    """Raised only when a GPU lease wait exhausts its wallclock deadline."""


@dataclass(frozen=True)
class GpuDevice:
    """A CUDA_VISIBLE_DEVICES token with a host-lock-safe file stem.

    ``visible_token`` is what the trainer sees in ``CUDA_VISIBLE_DEVICES``.
    ``lock_token`` is the canonical physical-device identity the host lock keys
    on — normally the GPU UUID resolved from a numeric index at pool
    construction. It defaults to the visible token, so a device built without
    resolution locks exactly as it did before (review v0.5.17 / blocker 6).
    """

    visible_token: str
    lock_token: str | None = None

    @property
    def lock_identity(self) -> str:
        """Return the canonical device identity this device's host lock keys on."""
        return self.lock_token or self.visible_token

    @property
    def lock_name(self) -> str:
        """Return a stable, path-safe lock identifier for this device's physical GPU.

        Derived from :attr:`lock_identity`, never from the visible token, so a
        numeric index and the GPU UUID naming the same card share one lock file
        (review v0.5.17 / blocker 6).
        """
        identity = self.lock_identity
        if identity.isdigit():
            return identity
        normalized = _SAFE_LOCK_TOKEN.sub("_", identity).strip("_") or "device"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"{normalized[:48]}_{digest}"


@dataclass
class _HostGpuLease:
    """Host-wide flock handle for one CUDA device token."""

    device: GpuDevice
    handle: IO[str]


@dataclass
class _GpuAcquisition:
    """One local GPU assignment plus the host locks that back it."""

    devices: list[GpuDevice]
    leases: list[_HostGpuLease]
    visible_devices: str


def _coerce_device(device: GpuDevice | int | str) -> GpuDevice:
    """Normalize a CUDA device token into :class:`GpuDevice`.

    :param GpuDevice | int | str device: CUDA device token to normalize.
    :return GpuDevice: Existing device instance or a device wrapping the stringified token.
    """
    if isinstance(device, GpuDevice):
        return device
    return GpuDevice(str(device))


def _gpu_lock_path(device: GpuDevice | int | str) -> Path:
    """Return the host-wide lock file for a CUDA device token.

    :param GpuDevice | int | str device: CUDA device token.
    :return Path: Host-wide lock file path for ``device``.
    """
    return lock_dir() / f"gpu_{_coerce_device(device).lock_name}.lock"


def _try_host_gpu_lease(device: GpuDevice) -> _HostGpuLease | None:
    """Try to acquire the per-GPU host lock without blocking.

    :param GpuDevice device: CUDA device token to lease.
    :return _HostGpuLease | None: Acquired lease, or ``None`` if already locked.
    """
    handle = try_lock_file(_gpu_lock_path(device))
    if handle is None:
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return _HostGpuLease(device=device, handle=handle)


def _release_host_gpu_lease(lease: _HostGpuLease | None) -> None:
    """Release a host-wide GPU lock."""
    if lease is None:
        return
    unlock_file(lease.handle)


def _detect_gpu_inventory() -> tuple[list[int], dict[str, str]]:
    """Probe ``nvidia-smi`` once for visible indices and usable device UUIDs.

    Returns:
        Numeric indices plus the subset mapped to a ``GPU-...`` or ``MIG-...``
        identity. Both are empty when the probe is unavailable.

    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return [], {}
    except FileNotFoundError:
        return [], {}
    except Exception:  # noqa: BLE001
        return [], {}
    ids: list[int] = []
    uuid_map: dict[str, str] = {}
    for line in out.stdout.strip().splitlines():
        index, _, uuid = line.partition(",")
        index, uuid = index.strip(), uuid.strip()
        if index.isdigit():
            ids.append(int(index))
        # Require a real device-identity shape. Restricted drivers (vGPU,
        # locked-down containers) report "[N/A]"/"[Not Supported]" for every
        # index; accepting those would map ALL indices to one identity and
        # silently collapse a multi-GPU pool to a single lock (review
        # v0.5.17 gap hunt).
        if index.isdigit() and uuid.startswith(("GPU-", "MIG-")):
            uuid_map[index] = uuid
    return ids, uuid_map


def _detect_gpu_uuid_map() -> dict[str, str]:
    """Return the usable index-to-UUID portion of one GPU inventory probe."""
    return _detect_gpu_inventory()[1]


def _nvidia_driver_reports_gpus() -> bool:
    """Return whether the NVIDIA kernel driver lists any GPUs on this host.

    Distinguishes a genuinely CPU-only host from a GPU host whose
    ``nvidia-smi`` probe failed (missing from a minimal PATH, wedged, or
    timing out): ``/proc/driver/nvidia/gpus`` holds one directory per card
    the kernel driver knows about, independent of userspace tooling.

    :return bool: ``True`` when the driver procfs lists at least one GPU.
    """
    try:
        return any(Path("/proc/driver/nvidia/gpus").iterdir())
    except OSError:
        return False


_FULL_GPU_UUID_LENGTH = len("GPU-") + 36


def _is_abbreviated_gpu_uuid(token: str) -> bool:
    """Return whether ``token`` is a shortened ``GPU-`` UUID prefix.

    CUDA accepts unambiguous UUID prefixes in ``CUDA_VISIBLE_DEVICES``, so a
    ``GPU-`` token shorter than a full canonical UUID names the same card as
    its full spelling and needs resolution before locking (review v0.5.17
    gap hunt). Full-length tokens are treated as already canonical.

    :param str token: Configured CUDA device token.
    :return bool: ``True`` for a shortened ``GPU-`` UUID prefix.
    """
    return token.startswith("GPU-") and len(token) < _FULL_GPU_UUID_LENGTH


def _resolve_lock_identities(
    devices: list[GpuDevice],
    *,
    uuid_map: dict[str, str] | None = None,
) -> list[GpuDevice]:
    """Bind each device to the canonical physical GPU its host lock keys on.

    Numeric indices resolve to the UUID ``nvidia-smi`` reports for them, so the
    same card configured as ``0`` in one run and ``GPU-<uuid>`` in another takes
    one host lock instead of two (review v0.5.17 / blocker 6). Abbreviated
    ``GPU-`` UUID prefixes — which CUDA accepts when unambiguous — resolve to
    the full UUID the same way (review v0.5.17 gap hunt). Full opaque tokens
    (``GPU-...`` UUIDs, ``MIG-...`` instances) are already canonical and lock on
    themselves; phasesweep does not bind a MIG instance to its parent device.

    Args:
        devices: Devices built from configured, ambient, or detected tokens.
        uuid_map: Already-probed index-to-UUID map, when available.

    Returns:
        The same devices with ``lock_token`` populated, deduplicated by lock
        identity so one physical device is never leased twice in one pool.

    Raises:
        RuntimeError: ``nvidia-smi`` cannot be read *and* the token set mixes
            numeric indices with opaque tokens (phasesweep cannot tell whether
            they name the same card), or a configured numeric index is absent
            from a readable index-to-UUID map (the device does not exist).

    """
    numeric = [device for device in devices if device.visible_token.isdigit()]
    abbreviated = [device for device in devices if _is_abbreviated_gpu_uuid(device.visible_token)]
    if not numeric and not abbreviated:
        # Full opaque token sets are already canonical: no probe, no ambiguity.
        return _dedupe_by_lock_identity(devices)

    if uuid_map is None:
        uuid_map = _detect_gpu_uuid_map()
    if not uuid_map:
        opaque = [device.visible_token for device in devices if not device.visible_token.isdigit()]
        if numeric and opaque:
            raise RuntimeError(
                "Cannot resolve canonical GPU identity: this device set mixes numeric "
                f"indices ({[device.visible_token for device in numeric]}) with opaque "
                f"tokens ({opaque}), and nvidia-smi could not be read, so phasesweep "
                "cannot tell whether they name the same physical GPU and could "
                "double-book it. Use one convention for every device on this host "
                "(all indices or all UUID/MIG tokens), or fix nvidia-smi."
            )
        if abbreviated:
            log.warning(
                "nvidia-smi UUID resolution is unavailable; abbreviated GPU UUID "
                "prefix(es) %s lock on the prefix spelling and would not exclude a "
                "concurrent run using the full UUID for the same card. Use full "
                "UUIDs, or fix nvidia-smi.",
                [device.visible_token for device in abbreviated],
            )
            return _dedupe_by_lock_identity(devices)
        # No map and one consistent spelling: indices remain their own identity,
        # which still collides with itself across runs on this host — but NOT
        # with a concurrent run whose nvidia-smi probe succeeded and locked the
        # UUID form. Lock-identity resolution must succeed or fail the same way
        # for every orchestrator on a host (watch systemd/cron units with a
        # minimal PATH), so make the degradation loud instead of silent.
        log.warning(
            "nvidia-smi index-to-UUID resolution is unavailable; GPU host locks for %s "
            "fall back to index-form names. A concurrent run that CAN resolve UUIDs "
            "would lock the same cards under different names and could double-book "
            "them. Fix nvidia-smi (PATH, driver) for every orchestrator on this host.",
            [device.visible_token for device in numeric],
        )
        return _dedupe_by_lock_identity(devices)

    resolved: list[GpuDevice] = []
    for device in devices:
        token = device.visible_token
        if not token.isdigit():
            # Canonicalize an abbreviated UUID prefix to the full UUID when
            # the readable map matches exactly one device; ambiguous or
            # unmatched prefixes stay their own identity.
            if _is_abbreviated_gpu_uuid(token):
                matches = [
                    uuid for uuid in uuid_map.values() if uuid.lower().startswith(token.lower())
                ]
                if len(matches) == 1:
                    resolved.append(GpuDevice(visible_token=token, lock_token=matches[0]))
                    continue
            resolved.append(device)
            continue
        uuid = uuid_map.get(token)
        if uuid is None:
            raise RuntimeError(
                f"Configured CUDA device index {token} does not exist on this host, or "
                f"nvidia-smi reported no usable GPU/MIG UUID for it (resolvable indices: "
                f"{sorted(uuid_map, key=int)}). Fix gpu_ids, gpu_devices, or "
                "CUDA_VISIBLE_DEVICES."
            )
        resolved.append(GpuDevice(visible_token=token, lock_token=uuid))
    return _dedupe_by_lock_identity(resolved)


def _dedupe_by_lock_identity(devices: list[GpuDevice]) -> list[GpuDevice]:
    """Drop devices that resolve to a physical GPU an earlier device already claims.

    :param list[GpuDevice] devices: Devices with lock identities already resolved.
    :return list[GpuDevice]: First device per lock identity, in the original order.
    """
    unique: list[GpuDevice] = []
    seen: dict[str, str] = {}
    for device in devices:
        first = seen.get(device.lock_identity)
        if first is not None:
            log.warning(
                "CUDA device token %r names the same physical GPU as %r; leasing it once.",
                device.visible_token,
                first,
            )
            continue
        seen[device.lock_identity] = device.visible_token
        unique.append(device)
    return unique


class GpuPool:
    """Thread-safe pool of CUDA-visible device tokens.

    Usage:
        pool = GpuPool.create(n_jobs=4)
        with pool.acquire() as gpu_device:
            env["CUDA_VISIBLE_DEVICES"] = gpu_device if gpu_device is not None else ""
            run_trial(...)
    """

    def __init__(self, devices: list[GpuDevice], *, whole_node: bool = False) -> None:
        """Build a pool from a fixed list of CUDA device tokens.

        Args:
            devices: The CUDA device tokens to manage. If empty, the pool is a
                transparent no-op that always yields ``None`` (single-job
                or CPU-only mode). Prefer :meth:`create` for the normal
                construction path that handles auto-detection and policy.
            whole_node: Whether each acquisition leases all devices as one
                comma-joined CUDA_VISIBLE_DEVICES assignment.

        """
        self._devices = devices
        self._whole_node = whole_node
        self._whole_node_in_use = False
        self._available: list[GpuDevice] = []
        self._condition = threading.Condition()

        if devices:
            self._available = list(devices)

    @classmethod
    def create(
        cls,
        n_jobs: int,
        explicit_ids: list[int] | None = None,
        explicit_devices: list[str] | None = None,
        allow_no_gpu: bool = False,
        policy: GpuPolicy = "single_per_trial",
        cuda_visible_devices: str | None = None,
    ) -> GpuPool:
        """Build a pool, applying phasesweep's GPU isolation policy.

        Args:
            n_jobs: number of parallel trials.
            explicit_ids: GPU indices from YAML config. If ``None``, auto-detect
                visible devices even for ``n_jobs == 1`` so independent
                single-job phasesweep processes do not double-book cuda:0.
            explicit_devices: Opaque CUDA_VISIBLE_DEVICES tokens from YAML config,
                such as GPU UUIDs or MIG instance IDs. Mutually exclusive with
                ``explicit_ids``.
            allow_no_gpu: if ``True``, run without CUDA isolation when no numeric
                GPU IDs can be resolved. Parallel CPU-only sweeps need this opt-in.
            policy: CUDA visibility policy. ``single_per_trial`` leases one
                token per trial. ``whole_node`` leases all tokens for one trial
                and exposes them comma-joined. ``none`` disables CUDA isolation
                and GPU locks.
            cuda_visible_devices: Configured trainer-environment override for
                ``CUDA_VISIBLE_DEVICES``. When omitted, the ambient value is used.

        Returns:
            A configured :class:`GpuPool`. The pool is "active" (hands out
            non-``None`` IDs) iff a GPU list is in play.

        Raises:
            ValueError: Direct arguments violate the same GPU isolation
                invariants enforced by :class:`phasesweep.config.Phase`.
            RuntimeError: No GPUs are visible and ``n_jobs > 1`` without
                ``allow_no_gpu``, or the configured device tokens cannot be
                resolved to canonical physical devices (see
                :func:`_resolve_lock_identities`).

        """
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError(f"n_jobs must be a positive integer; got {n_jobs!r}.")
        if policy not in ("single_per_trial", "whole_node", "none"):
            raise ValueError(f"Unknown GPU policy {policy!r}.")
        if explicit_ids is not None and explicit_devices is not None:
            raise ValueError("explicit_ids and explicit_devices are mutually exclusive.")
        if explicit_ids is not None:
            if not explicit_ids:
                raise ValueError(
                    "explicit_ids must be omitted or contain at least one CUDA device index."
                )
            bad_ids = [
                value
                for value in explicit_ids
                if isinstance(value, bool) or not isinstance(value, int) or value < 0
            ]
            if bad_ids:
                raise ValueError(
                    f"explicit_ids must contain non-negative CUDA device indices; got {bad_ids}."
                )
        if explicit_devices is not None:
            if not explicit_devices:
                raise ValueError(
                    "explicit_devices must be omitted or contain at least one CUDA device token."
                )
            if any(not isinstance(token, str) for token in explicit_devices):
                raise ValueError("explicit_devices must contain only string device tokens.")
            explicit_devices = [token.strip() for token in explicit_devices]
            bad_devices = [
                token for token in explicit_devices if not token or "," in token or token == "-1"
            ]
            if bad_devices:
                raise ValueError(
                    "explicit_devices entries must be non-empty CUDA_VISIBLE_DEVICES tokens "
                    f"without commas or -1; got {bad_devices}."
                )
        if policy == "whole_node" and n_jobs != 1:
            raise ValueError("policy='whole_node' requires n_jobs=1.")
        if policy == "whole_node" and explicit_ids is None and explicit_devices is None:
            raise ValueError(
                "policy='whole_node' requires an explicit_ids or explicit_devices list."
            )
        if policy == "none" and (explicit_ids is not None or explicit_devices is not None):
            raise ValueError(
                "policy='none' cannot be combined with explicit_ids or explicit_devices."
            )
        if policy == "none" and n_jobs > 1 and not allow_no_gpu:
            raise ValueError("policy='none' with n_jobs > 1 requires allow_no_gpu=True.")

        if policy == "none":
            log.info("GPU isolation disabled by gpu_policy='none'.")
            return cls(devices=[])

        # Explicit IDs always win, even at n_jobs==1.
        if explicit_ids is not None:
            devices = _resolve_lock_identities(_normalize_devices(explicit_ids))
            _log_pool_size(n_jobs, [device.visible_token for device in devices], "configured")
            return cls(devices=devices, whole_node=policy == "whole_node")
        if explicit_devices is not None:
            devices = _resolve_lock_identities(_normalize_devices(explicit_devices))
            _log_pool_size(n_jobs, [device.visible_token for device in devices], "configured")
            return cls(devices=devices, whole_node=policy == "whole_node")

        user_cvd = (
            cuda_visible_devices
            if cuda_visible_devices is not None
            else os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        detected_uuid_map: dict[str, str] | None = None
        if user_cvd is not None:
            devices = _devices_from_cuda_visible_devices(user_cvd)
        else:
            detected_ids, detected_uuid_map = _detect_gpu_inventory()
            devices = _normalize_devices(detected_ids)
        if not devices:
            if n_jobs <= 1:
                if user_cvd is not None:
                    log.info(
                        "CUDA_VISIBLE_DEVICES exposes no devices; single-job phase "
                        "will preserve that visibility without GPU host locks."
                    )
                elif _nvidia_driver_reports_gpus():
                    # The kernel driver knows about GPUs, so the empty probe is
                    # a broken nvidia-smi, not a CPU-only host: this phase will
                    # run with no CUDA_VISIBLE_DEVICES binding and hold zero
                    # GPU host locks while real cards sit on the host
                    # (review v0.5.17 gap hunt).
                    log.warning(
                        "nvidia-smi detected no GPUs but /proc/driver/nvidia/gpus lists "
                        "hardware; running WITHOUT CUDA isolation or GPU host locks. Fix "
                        "nvidia-smi or set gpu_ids/gpu_devices explicitly."
                    )
                else:
                    log.info("No GPUs detected; single-job phase will run without CUDA isolation.")
                return cls(devices=[])
            if allow_no_gpu:
                log.warning(
                    "n_jobs=%d, no GPUs detected — running without CUDA_VISIBLE_DEVICES "
                    "isolation (allow_no_gpu_isolation: true).",
                    n_jobs,
                )
                return cls(devices=[])
            raise RuntimeError(
                f"n_jobs={n_jobs} but no GPUs detected. Set gpu_ids or gpu_devices "
                "explicitly in the phase config, or set allow_no_gpu_isolation: true "
                "if this is an intentional CPU-only parallel sweep."
            )
        devices = _resolve_lock_identities(devices, uuid_map=detected_uuid_map)
        _log_pool_size(n_jobs, [device.visible_token for device in devices], "available")
        return cls(devices=devices, whole_node=policy == "whole_node")

    def _remaining_seconds(self, deadline: float | None) -> float | None:
        """Return seconds until ``deadline``, or raise when it has expired.

        :param float | None deadline: Optional ``time.monotonic()`` deadline.
        :return float | None: Remaining seconds, or ``None`` when no deadline is active.
        """
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise GpuLeaseTimeoutError("Wallclock deadline reached while waiting for a GPU lease.")
        return remaining

    def _acquire_single(self, *, deadline: float | None = None) -> _GpuAcquisition | None:
        """Block until a local slot and host-wide GPU lease are available.

        Args:
            deadline: Optional ``time.monotonic()`` deadline. When set, waiting
                for a local slot or host-wide lock fails with ``TimeoutError``
                instead of extending a phase/run wallclock budget.

        Returns:
            A one-device acquisition, or ``None`` if the pool is inactive.

        Raises:
            TimeoutError: ``deadline`` expired before a GPU could be leased.

        """
        if not self._devices:
            return None
        while True:
            with self._condition:
                while not self._available:
                    wait_seconds = self._remaining_seconds(deadline)
                    self._condition.wait(timeout=wait_seconds)
                candidates = list(self._available)
                self._available.clear()

            remaining_devices: list[GpuDevice] = []
            for index, device in enumerate(candidates):
                lease = _try_host_gpu_lease(device)
                if lease is not None:
                    remaining_devices.extend(candidates[index + 1 :])
                    with self._condition:
                        self._available.extend(remaining_devices)
                        self._condition.notify_all()
                    log.debug("GPU %s host lease acquired", device.visible_token)
                    return _GpuAcquisition(
                        devices=[device],
                        leases=[lease],
                        visible_devices=device.visible_token,
                    )
                remaining_devices.append(device)

            with self._condition:
                self._available.extend(remaining_devices)
                self._condition.notify_all()
            remaining_seconds = self._remaining_seconds(deadline)
            time.sleep(0.2 if remaining_seconds is None else min(0.2, remaining_seconds))

    def _acquire_whole_node(self, *, deadline: float | None = None) -> _GpuAcquisition | None:
        """Acquire every configured device token as one assignment.

        :param float | None deadline: Optional ``time.monotonic()`` deadline for waiting.
        :raises TimeoutError: If the deadline expires before all devices can be leased.
        :return _GpuAcquisition | None: Whole-node assignment, or ``None`` if inactive.
        """
        if not self._devices:
            return None
        while True:
            with self._condition:
                while self._whole_node_in_use:
                    wait_seconds = self._remaining_seconds(deadline)
                    self._condition.wait(timeout=wait_seconds)
                self._whole_node_in_use = True

            leases: list[_HostGpuLease] = []
            all_leased = True
            for device in self._devices:
                lease = _try_host_gpu_lease(device)
                if lease is None:
                    all_leased = False
                    break
                leases.append(lease)
            if not all_leased:
                for lease in leases:
                    _release_host_gpu_lease(lease)
                with self._condition:
                    self._whole_node_in_use = False
                    self._condition.notify_all()
                remaining_seconds = self._remaining_seconds(deadline)
                time.sleep(0.2 if remaining_seconds is None else min(0.2, remaining_seconds))
                continue

            visible_devices = ",".join(device.visible_token for device in self._devices)
            log.debug("Whole-node GPU host leases acquired: %s", visible_devices)
            return _GpuAcquisition(
                devices=list(self._devices),
                leases=leases,
                visible_devices=visible_devices,
            )

    def _acquire(self, *, deadline: float | None = None) -> _GpuAcquisition | None:
        """Acquire a GPU assignment according to the configured policy.

        :param float | None deadline: Optional ``time.monotonic()`` deadline for waiting.
        :raises TimeoutError: If the deadline expires before an assignment can be leased.
        :return _GpuAcquisition | None: GPU assignment, or ``None`` if the pool is inactive.
        """
        if self._whole_node:
            return self._acquire_whole_node(deadline=deadline)
        return self._acquire_single(deadline=deadline)

    def _release(self, acquired: _GpuAcquisition | None) -> None:
        """Return a previously-acquired CUDA device token to the pool.

        Args:
            acquired: Assignment returned by :meth:`_acquire`. ``None`` is a no-op
                (inactive pool).

        """
        if acquired is None:
            return
        for lease in acquired.leases:
            _release_host_gpu_lease(lease)
        if self._whole_node:
            with self._condition:
                self._whole_node_in_use = False
                self._condition.notify_all()
            return
        device = acquired.devices[0]
        with self._condition:
            self._available.append(device)
            self._condition.notify()

    @contextmanager
    def acquire(self, *, deadline: float | None = None) -> Generator[str | None, None, None]:
        """Block until a GPU is available, yield its CUDA-visible token, release on exit.

        Args:
            deadline: Optional ``time.monotonic()`` deadline for the wait.

        Yields:
            The CUDA_VISIBLE_DEVICES token for this critical section, or ``None`` when the
            pool is inactive.

        Raises:
            GpuLeaseTimeoutError: ``deadline`` expired before a GPU could be leased.

        """
        acquired = self._acquire(deadline=deadline)
        try:
            yield None if acquired is None else acquired.visible_devices
        finally:
            self._release(acquired)


def _normalize_devices(tokens: Iterable[int | str]) -> list[GpuDevice]:
    """Deduplicate non-empty CUDA device tokens while preserving order.

    :param Iterable[int | str] tokens: CUDA device tokens to normalize and deduplicate.
    :return list[GpuDevice]: Unique non-empty devices in their original order.
    """
    devices: list[GpuDevice] = []
    seen: set[str] = set()
    for raw in tokens:
        token = str(raw).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        devices.append(GpuDevice(token))
    return devices


def _devices_from_cuda_visible_devices(value: str) -> list[GpuDevice]:
    """Parse CUDA_VISIBLE_DEVICES as opaque tokens.

    :param str value: Comma-separated CUDA visibility value to parse.
    :raises RuntimeError: If ``-1`` is mixed with visible device tokens.
    :return list[GpuDevice]: Parsed, deduplicated devices, or an empty list for ``-1``.
    """
    raw = [token.strip() for token in value.split(",") if token.strip()]
    if raw == ["-1"]:
        return []
    if "-1" in raw:
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1 cannot be mixed with visible device tokens.")
    return _normalize_devices(raw)


def _log_pool_size(n_jobs: int, tokens: list[str], source: str) -> None:
    """Log whether configured/available CUDA devices cover requested parallelism.

    :param int n_jobs: Requested number of parallel jobs.
    :param list[str] tokens: Configured or detected CUDA device tokens.
    :param str source: Description of the token source included in warnings.
    """
    if n_jobs > len(tokens):
        log.warning(
            "n_jobs=%d but only %d GPU device(s) %s (%s). Excess trials will queue for a GPU.",
            n_jobs,
            len(tokens),
            source,
            tokens,
        )
    else:
        log.info("GPU pool: %d GPU device(s) for %d parallel job(s).", len(tokens), n_jobs)
