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
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from phasesweep.config.models import GpuPolicy
from phasesweep.runtime.files import lock_dir, try_lock_file, unlock_file

log = logging.getLogger("phasesweep.runtime.gpu")

_SAFE_LOCK_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")
_MIG_UUID_INVENTORY = re.compile(r"\(UUID:\s*(MIG-[^)]+)\)", re.IGNORECASE)


class GpuLeaseTimeoutError(TimeoutError):
    """Raised only when a GPU lease wait exhausts its wallclock deadline."""


@dataclass(frozen=True)
class GpuDevice:
    """A CUDA_VISIBLE_DEVICES token with a host-lock-safe file stem.

    ``visible_token`` is what the trainer sees in ``CUDA_VISIBLE_DEVICES``.
    GPU/MIG prefixes are replaced with the inventory's canonical spelling;
    numeric indices keep the operator's ordinal.
    ``lock_token`` is the canonical physical-device identity the host lock keys
    on — normally the GPU UUID resolved from a numeric index at pool
    construction. It defaults to the visible token, while pool construction
    resolves every UUID spelling to the inventory's canonical identity.
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


@dataclass(frozen=True)
class GpuAssignment:
    """Trainer visibility plus descriptors transferred to its trusted guardian."""

    visible_devices: str | None
    lease_fds: tuple[int, ...]


@dataclass
class _HostGpuLease:
    """Host-wide flock handle for one CUDA device token."""

    handle: IO[str]


@dataclass
class _GpuAcquisition:
    """One local GPU assignment plus the host locks that back it."""

    devices: list[GpuDevice]
    leases: list[_HostGpuLease]
    visible_devices: str


def _gpu_lock_path(device: GpuDevice) -> Path:
    """Return the host-wide lock file for a CUDA device token.

    :param GpuDevice device: Canonical CUDA device identity.
    :return Path: Host-wide lock file path for ``device``.
    """
    return lock_dir() / f"gpu_{device.lock_name}.lock"


def _try_host_gpu_lease(device: GpuDevice) -> _HostGpuLease | None:
    """Try to acquire the per-GPU host lock without blocking.

    :param GpuDevice device: CUDA device token to lease.
    :return _HostGpuLease | None: Acquired lease, or ``None`` if already locked.
    """
    handle = try_lock_file(_gpu_lock_path(device))
    if handle is None:
        return None
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except BaseException:
        # The flock is already held; failing to stamp the pid must not leak a
        # locked handle that would exclude every other run from this GPU.
        unlock_file(handle)
        raise
    return _HostGpuLease(handle=handle)


def _release_host_gpu_lease(lease: _HostGpuLease | None) -> None:
    """Close this process's lease copy without unlocking guardian copies.

    ``flock(LOCK_UN)`` operates on the shared open-file description and would
    release the supervisor guardian's inherited lock too. Closing only our descriptor
    keeps the lock alive after an orchestrator crash or uncertain cleanup,
    until the guardian exits with the trainer.
    """
    if lease is None:
        return
    with suppress(OSError):
        lease.handle.close()


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


def _detect_mig_uuid_set() -> set[str]:
    """Return MIG instance UUIDs reported by ``nvidia-smi -L``.

    :return set[str]: Canonical ``MIG-...`` identities, empty when the probe
        is unavailable or reports none.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return set()
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return set()
    return {match.group(1) for match in _MIG_UUID_INVENTORY.finditer(out.stdout)}


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


def _validate_device_token_shapes(devices: list[GpuDevice]) -> None:
    """Reject device tokens that cannot syntactically name a CUDA device.

    CUDA silently exposes no device for a token that is neither a numeric
    index nor a ``GPU-``/``MIG-`` identity, so accepting one would hand a
    trial an empty visibility set while the real GPUs sit unlocked — the
    opaque-token equivalent of a nonexistent numeric index, which already
    fails closed in :func:`_resolve_lock_identities`.

    :param list[GpuDevice] devices: Devices built from configured or ambient tokens.
    :raises RuntimeError: A token is neither numeric nor ``GPU-``/``MIG-`` shaped.
    """
    for device in devices:
        token = device.visible_token
        if token.isdigit() or token[:4].upper() in ("GPU-", "MIG-"):
            continue
        raise RuntimeError(
            f"CUDA device token {token!r} is not a numeric index, GPU- UUID, or "
            "MIG- instance ID. CUDA would expose no device for it while phasesweep "
            "holds no lock for any real GPU. Fix gpu_ids, gpu_devices, or "
            "CUDA_VISIBLE_DEVICES."
        )


def _resolve_lock_identities(
    devices: list[GpuDevice],
    *,
    uuid_map: dict[str, str] | None = None,
) -> list[GpuDevice]:
    """Bind each device to its canonical physical GPU and drop alias duplicates.

    Args:
        devices: Devices built from configured, ambient, or detected tokens.
        uuid_map: Already-probed index-to-UUID map, when available.

    Returns:
        The devices from :func:`_bind_lock_identities`, deduplicated by lock
        identity so one physical device is never leased twice in one pool.

    Raises:
        RuntimeError: Propagated from :func:`_bind_lock_identities`.

    """
    return _dedupe_by_lock_identity(_bind_lock_identities(devices, uuid_map=uuid_map))


def _bind_lock_identities(
    devices: list[GpuDevice],
    *,
    uuid_map: dict[str, str] | None = None,
    mig_uuids: set[str] | None = None,
) -> list[GpuDevice]:
    """Bind each device to the canonical physical GPU its host lock keys on.

    Numeric indices resolve to the UUID ``nvidia-smi`` reports for them, so the
    same card configured as ``0`` in one run and ``GPU-<uuid>`` in another takes
    one host lock instead of two (review v0.5.17 / blocker 6). Every ``GPU-``
    token, including a full-length or lowercase spelling, resolves through the
    readable inventory; abbreviated UUID prefixes resolve only when exactly one
    card matches. ``MIG-`` tokens similarly resolve through ``nvidia-smi -L``.
    This both canonicalizes case and rejects a UUID that CUDA would turn into an
    empty visibility set while phasesweep held no real-device lock.

    Binding is kept separate from deduplication so a caller that must fail
    closed on aliasing can see which tokens collapsed onto one card before they
    are silently dropped (:func:`_require_distinct_whole_node_devices`). Every
    token fails closed when its required ``nvidia-smi`` inventory cannot be
    read: accepting an unverified index or UUID could expose no CUDA device or
    split the host lock namespace between index and UUID spellings.

    Args:
        devices: Devices built from configured, ambient, or detected tokens.
        uuid_map: Already-probed index-to-UUID map, when available.
        mig_uuids: Already-probed MIG instance identities, when available.

    Returns:
        The same devices, in order, with ``lock_token`` populated where a
        canonical identity could be resolved. No device is dropped.

    Raises:
        RuntimeError: A token is neither a numeric index nor a
            ``GPU-``/``MIG-`` identity (see :func:`_validate_device_token_shapes`),
            ``nvidia-smi`` cannot validate an opaque token, or a configured
            numeric/UUID identity is absent from the readable inventory.

    """
    _validate_device_token_shapes(devices)
    numeric = [device for device in devices if device.visible_token.isdigit()]
    gpu_tokens = [device for device in devices if device.visible_token[:4].upper() == "GPU-"]
    mig_tokens = [device for device in devices if device.visible_token[:4].upper() == "MIG-"]

    if uuid_map is None and (numeric or gpu_tokens):
        uuid_map = _detect_gpu_uuid_map()
    uuid_map = uuid_map or {}
    if mig_uuids is None and mig_tokens:
        mig_uuids = _detect_mig_uuid_set()
    mig_uuids = mig_uuids or set()
    if mig_tokens and not mig_uuids:
        raise RuntimeError(
            "Cannot validate configured MIG token(s) "
            f"{[device.visible_token for device in mig_tokens]} because nvidia-smi -L "
            "reported no usable MIG inventory. Fix nvidia-smi or the configured "
            "gpu_devices/CUDA_VISIBLE_DEVICES value."
        )
    if not uuid_map:
        if gpu_tokens:
            raise RuntimeError(
                "Cannot validate configured GPU UUID token(s) "
                f"{[device.visible_token for device in gpu_tokens]} because nvidia-smi "
                "reported no usable GPU UUID inventory. Fix nvidia-smi or the configured "
                "gpu_devices/CUDA_VISIBLE_DEVICES value."
            )
        if numeric:
            raise RuntimeError(
                "Cannot validate configured CUDA device index token(s) "
                f"{[device.visible_token for device in numeric]} because nvidia-smi "
                "reported no usable index-to-UUID inventory. Accepting index-form locks "
                "would not exclude a concurrent UUID-form lock for the same card. Fix "
                "nvidia-smi or the configured gpu_ids/gpu_devices/CUDA_VISIBLE_DEVICES value."
            )

    resolved: list[GpuDevice] = []
    for device in devices:
        token = device.visible_token
        if token[:4].upper() == "GPU-":
            matches = [uuid for uuid in uuid_map.values() if uuid.lower().startswith(token.lower())]
            if len(matches) != 1:
                detail = "matches multiple devices" if matches else "matches no device"
                raise RuntimeError(
                    f"Configured CUDA GPU UUID token {token!r} {detail} in the nvidia-smi "
                    "inventory. Fix gpu_devices or CUDA_VISIBLE_DEVICES."
                )
            resolved.append(GpuDevice(visible_token=matches[0], lock_token=matches[0]))
            continue
        if token[:4].upper() == "MIG-":
            matches = [uuid for uuid in mig_uuids if uuid.lower().startswith(token.lower())]
            if len(matches) != 1:
                detail = "matches multiple devices" if matches else "matches no device"
                raise RuntimeError(
                    f"Configured CUDA MIG token {token!r} {detail} in the nvidia-smi "
                    "inventory. Fix gpu_devices or CUDA_VISIBLE_DEVICES."
                )
            resolved.append(GpuDevice(visible_token=matches[0], lock_token=matches[0]))
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
    return resolved


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


def _require_distinct_whole_node_devices(requested: list[str], bound: list[GpuDevice]) -> None:
    """Fail closed when a whole-node device set collapses to fewer physical GPUs.

    Under ``gpu_policy='whole_node'`` the configured token list *declares* the
    trainer's world size, and that declared count is what the phase fingerprint
    records (``whole_node_device_count``). Deduplication is therefore not a
    convenience here: dropping an aliased token would run a 1-GPU world under a
    2-GPU study identity, and the two runs would be indistinguishable afterwards
    (PR #5 review / reviewer 2, blocker 2). ``single_per_trial`` is unaffected —
    there the list is a pool whose size is pure throughput, so dedupe with a
    warning stays correct.

    Only *actual* collapses are reported. A collapse is visible either as a
    token that :func:`_normalize_devices` removed (an exact repeat, or an empty
    token) or as two bound devices sharing one ``lock_identity``. Failed
    inventory resolution raises before this check.

    :param list[str] requested: Stripped device tokens exactly as configured,
        before normalization dropped anything.
    :param list[GpuDevice] bound: Normalized devices with lock identities bound
        but not yet deduplicated.
    :raises RuntimeError: The configured tokens name fewer distinct physical
        GPUs than the declared world size.
    """
    mig_devices = [device for device in bound if device.lock_identity[:4].upper() == "MIG-"]
    if mig_devices and len(bound) > 1:
        raise RuntimeError(
            "gpu_policy='whole_node' cannot declare a multi-device world containing "
            "MIG tokens. CUDA's enumerated device count depends on the driver version "
            "and permits at most one compute instance per GPU instance, while the MIG "
            "UUID inventory does not identify those parent GPU instances here. Configure "
            "exactly one MIG token, or use full-GPU tokens for a multi-device world."
        )

    aliases: list[str] = []
    first_by_identity: dict[str, str] = {}
    for device in bound:
        first = first_by_identity.get(device.lock_identity)
        if first is None:
            first_by_identity[device.lock_identity] = device.visible_token
            continue
        aliases.append(
            f"{device.visible_token!r} names the same physical GPU as {first!r} "
            f"(identity {device.lock_identity!r})"
        )
    repeats = sorted({token for token in requested if requested.count(token) > 1})
    effective = len(first_by_identity)
    if effective == len(requested):
        return
    detail = "; ".join(aliases + ([f"token(s) {repeats} appear more than once"] if repeats else []))
    raise RuntimeError(
        f"gpu_policy='whole_node' declares {len(requested)} CUDA device token(s) "
        f"({requested}) but they resolve to only {effective} distinct physical "
        f"GPU(s): {detail or 'an empty token was dropped'}. The phase fingerprint "
        "records the declared count as the trainer's world size, so leasing fewer "
        "devices would silently run a smaller world under a larger study identity. "
        "List one distinct GPU per world-size slot, or use "
        "gpu_policy='single_per_trial' if this list is a pool rather than a world."
    )


class GpuPool:
    """Thread-safe pool of CUDA-visible device tokens.

    Usage:
        pool = GpuPool.create(n_jobs=4)
        with pool.acquire() as gpu_device:
            if gpu_device is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_device
            run_trial(...)
    """

    def __init__(
        self,
        devices: list[GpuDevice],
        *,
        whole_node: bool = False,
        pinned_visible_devices: str | None = None,
    ) -> None:
        """Build a pool from a fixed list of CUDA device tokens.

        Args:
            devices: The CUDA device tokens to manage. If empty, the pool is a
                transparent no-op that always yields ``None`` (single-job
                or CPU-only mode). Prefer :meth:`create` for the normal
                construction path that handles auto-detection and policy.
            whole_node: Whether each acquisition leases all devices as one
                comma-joined CUDA_VISIBLE_DEVICES assignment.
            pinned_visible_devices: CUDA_VISIBLE_DEVICES value every
                acquisition yields when ``devices`` is empty. Set when the
                pool leases nothing *because* a configured or ambient disable
                sentinel (``""``/``-1``) says CUDA is off: the sentinel must
                reach the trial environment even under a narrowed
                ``inherit_env`` contract, or the trainer would see every host
                GPU while phasesweep holds zero GPU locks.

        """
        self._devices = devices
        self._whole_node = whole_node
        self._whole_node_in_use = False
        self._available: list[GpuDevice] = []
        self._condition = threading.Condition()
        self._pinned_visible_devices = pinned_visible_devices

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
            explicit_devices: Numeric, GPU UUID, or MIG instance tokens from
                YAML config. Mutually exclusive with ``explicit_ids``.
            allow_no_gpu: if ``True``, run without CUDA isolation when no numeric
                GPU IDs can be resolved. Parallel auto-detected CPU-only sweeps
                need this opt-in; an explicit CUDA-disable value does not.
            policy: CUDA visibility policy. ``single_per_trial`` leases one
                token per trial. ``whole_node`` leases all tokens for one trial
                and exposes them comma-joined. ``none`` disables CUDA isolation
                and GPU locks.
            cuda_visible_devices: Configured trainer-environment override for
                ``CUDA_VISIBLE_DEVICES``. When omitted, the ambient value is used.

        Returns:
            A configured :class:`GpuPool`. The pool leases devices iff a GPU
            list is in play; an inactive pool built from a CUDA-disable
            sentinel still yields that sentinel so trials inherit the
            disable (see ``pinned_visible_devices``).

        Raises:
            RuntimeError: No GPUs are visible and ``n_jobs > 1`` without
                ``allow_no_gpu``; the NVIDIA driver reports hardware that
                ``nvidia-smi`` cannot enumerate without ``allow_no_gpu``; the
                configured device tokens cannot be resolved to canonical
                physical devices (see
                :func:`_bind_lock_identities`), or ``policy='whole_node'`` and
                the configured tokens collapse to fewer physical GPUs than the
                declared world size (see
                :func:`_require_distinct_whole_node_devices`).

        """
        if policy == "none":
            log.info("GPU isolation disabled by gpu_policy='none'.")
            return cls(devices=[])

        # Explicit configuration always wins, even at n_jobs==1.
        explicit = explicit_ids if explicit_ids is not None else explicit_devices
        if explicit is not None:
            bound = _bind_lock_identities(_normalize_devices(explicit))
            if policy == "whole_node":
                # Config validation rejects exactly-repeated tokens, but physical
                # aliasing (index vs UUID for one card) can only be seen once
                # nvidia-smi has been probed — which happens here, after the
                # no-op-republish early return in the phase runner. Fail closed
                # rather than lease a smaller world than the fingerprint claims.
                _require_distinct_whole_node_devices(
                    [str(token).strip() for token in explicit], bound
                )
            devices = _dedupe_by_lock_identity(bound)
            _log_pool_size(n_jobs, [device.visible_token for device in devices], "configured")
            return cls(devices=devices, whole_node=policy == "whole_node")

        ambient_cvd = cuda_visible_devices is None and "CUDA_VISIBLE_DEVICES" in os.environ
        user_cvd = (
            cuda_visible_devices
            if cuda_visible_devices is not None
            else os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        detected_uuid_map: dict[str, str] | None = None
        if user_cvd is not None:
            try:
                devices = _devices_from_cuda_visible_devices(user_cvd)
            except RuntimeError:
                if not ambient_cvd:
                    raise
                log.warning(
                    "Ambient CUDA_VISIBLE_DEVICES=%r mixes the '-1' disable sentinel "
                    "with device tokens; treating it as empty visibility and pinning "
                    "CUDA_VISIBLE_DEVICES='' for trial processes.",
                    user_cvd,
                )
                devices = []
                user_cvd = ""
        else:
            detected_ids, detected_uuid_map = _detect_gpu_inventory()
            devices = _normalize_devices(detected_ids)
        if user_cvd is not None and not devices and not _is_cuda_disable_value(user_cvd):
            if not ambient_cvd:
                raise RuntimeError(
                    f"Configured CUDA_VISIBLE_DEVICES value {user_cvd!r} is not the empty "
                    "string, '-1', or a comma-separated device list."
                )
            log.warning(
                "Ambient CUDA_VISIBLE_DEVICES=%r does not name a device or a supported "
                "disable sentinel; treating it as empty visibility and pinning "
                "CUDA_VISIBLE_DEVICES='' for trial processes.",
                user_cvd,
            )
            user_cvd = ""
        if devices:
            try:
                devices = _resolve_lock_identities(devices, uuid_map=detected_uuid_map)
            except RuntimeError:
                if not ambient_cvd:
                    raise
                log.warning(
                    "Ambient CUDA_VISIBLE_DEVICES=%r cannot be resolved against the local "
                    "GPU inventory; treating it as empty visibility and pinning "
                    "CUDA_VISIBLE_DEVICES='' for trial processes.",
                    user_cvd,
                    exc_info=True,
                )
                devices = []
                user_cvd = ""
        if not devices:
            # A configured or ambient sentinel ("" / "-1") is a decision, not
            # an absence: pin it so trial environments actually receive the
            # disable even when a narrowed inherit_env drops the ambient value.
            pinned = user_cvd.strip() if user_cvd is not None else None
            if pinned is not None:
                log.info(
                    "CUDA_VISIBLE_DEVICES exposes no devices; phase will pin "
                    "CUDA_VISIBLE_DEVICES=%r in trial environments without GPU host locks.",
                    pinned,
                )
                return cls(devices=[], pinned_visible_devices=pinned)
            if _nvidia_driver_reports_gpus():
                message = (
                    "nvidia-smi could not enumerate GPUs, but "
                    "/proc/driver/nvidia/gpus reports hardware. PhaseSweep cannot "
                    "derive canonical device locks, so launching with unrestricted "
                    "CUDA visibility could double-book the host. Fix nvidia-smi or "
                    "set gpu_ids/gpu_devices explicitly"
                )
                if not allow_no_gpu:
                    raise RuntimeError(
                        f"{message}; set allow_no_gpu_isolation: true only to accept "
                        "running without GPU host locks."
                    )
                log.warning(
                    "%s; running without CUDA isolation because "
                    "allow_no_gpu_isolation: true is set.",
                    message,
                )
                return cls(devices=[])
            if n_jobs <= 1:
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
        _log_pool_size(n_jobs, [device.visible_token for device in devices], "available")
        return cls(devices=devices, whole_node=policy == "whole_node")

    def _remaining_seconds(self, deadline: float | None) -> float | None:
        """Return seconds until ``deadline``, or raise when it has expired.

        :param float | None deadline: Optional ``time.monotonic()`` deadline.
        :return float | None: Remaining seconds, or ``None`` when no deadline is active.
        :raises GpuLeaseTimeoutError: The deadline has already passed, so
            waiting further would overrun the phase/run wallclock budget.
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

            # This thread owns every candidate until it hands them back. A
            # lock-layer failure that skipped the hand-back would shrink the
            # pool permanently: later acquires would block forever (or report
            # a lease timeout that phase attribution mistakes for wallclock
            # exhaustion), so the hand-back must survive any exception.
            acquired: _GpuAcquisition | None = None
            try:
                for device in candidates:
                    lease = _try_host_gpu_lease(device)
                    if lease is not None:
                        log.debug("GPU %s host lease acquired", device.visible_token)
                        acquired = _GpuAcquisition(
                            devices=[device],
                            leases=[lease],
                            visible_devices=device.visible_token,
                        )
                        break
            finally:
                leased = None if acquired is None else acquired.devices[0]
                give_back = [device for device in candidates if device is not leased]
                with self._condition:
                    self._available.extend(give_back)
                    self._condition.notify_all()
            if acquired is not None:
                return acquired
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

            # ``_whole_node_in_use`` is already claimed: a lock-layer failure
            # that left it set (or leaked partial leases) would deadlock every
            # later acquisition, so failures must roll both back.
            leases: list[_HostGpuLease] = []
            all_leased = True
            try:
                for device in self._devices:
                    lease = _try_host_gpu_lease(device)
                    if lease is None:
                        all_leased = False
                        break
                    leases.append(lease)
            except BaseException:
                for lease in leases:
                    _release_host_gpu_lease(lease)
                with self._condition:
                    self._whole_node_in_use = False
                    self._condition.notify_all()
                raise
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
    def acquire(self, *, deadline: float | None = None) -> Generator[GpuAssignment, None, None]:
        """Block until a GPU is available and yield its visibility plus lease FDs.

        Args:
            deadline: Optional ``time.monotonic()`` deadline for the wait.

        Yields:
            A :class:`GpuAssignment` carrying the CUDA visibility token (the
            pinned disable sentinel for explicit CPU isolation, or ``None``
            for an inactive pool) and the host-lock descriptors the trainer
            must inherit so exclusion survives an orchestrator hard exit.

        Raises:
            GpuLeaseTimeoutError: ``deadline`` expired before a GPU could be leased.

        """
        acquired = self._acquire(deadline=deadline)
        try:
            yield GpuAssignment(
                visible_devices=(
                    acquired.visible_devices
                    if acquired is not None
                    else self._pinned_visible_devices
                ),
                lease_fds=(
                    tuple(lease.handle.fileno() for lease in acquired.leases)
                    if acquired is not None
                    else ()
                ),
            )
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


def _is_cuda_disable_value(value: str) -> bool:
    """Return whether a visibility string is an empty or ``-1`` sentinel.

    :param str value: CUDA visibility string to classify.
    :return bool: Whether ``value`` disables CUDA device visibility.
    """
    if not value.strip():
        return True
    return [token.strip() for token in value.split(",") if token.strip()] == ["-1"]


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
