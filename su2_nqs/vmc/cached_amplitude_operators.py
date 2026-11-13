"""Exact reusable real-amplitude derivatives with bounded device memory.

The original cached solver does not change the SR metric, native parameter
coordinates, sampling measure, or the existing LSQR solver. It evaluates a
full amplitude Jacobian once at a frozen parent, retaining all records and
columns on the host (optionally in a new ``.npy`` memory-mapped file).
Subsequent SR algebra streams parameter blocks instead of differentiating
the complete projected wavefunction at every Krylov product.

Only real float64 amplitudes/parameters are supported here.  Complex-source
problems must use the separate complex-amplitude implementation, not discard
their imaginary derivatives.  The host cache is O(NP), not a compressed or
matrix-free approximation.  Its explicit byte budget is therefore essential.
The separate opt-in node-block solver changes update geometry explicitly;
it does not claim to solve the coupled global Fisher system.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Progress = Callable[[dict[str, Any]], None]


def _require_x64() -> None:
    if not jax.config.x64_enabled:
        raise RuntimeError("cached amplitude derivatives require x64")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _real64(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != ndim or array.dtype != np.dtype("float64")
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite real float64 with ndim={ndim}")
    return array


def _sha256(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def amplitude_cache_identity(theta: Any, rows: Any, psi: Any, density: Any,
                             *, source_identity: str = "") -> dict[str, str]:
    """Hash a frozen parent, physical record ordering, and importance measure.

    ``source_identity`` should identify the amplitude implementation/config;
    the caller must not reuse derivatives after a native-parameter update.
    This records identity, not the scientific adequacy of the sample support.
    """
    return dict(parameter_sha256=_sha256(theta), records_sha256=_sha256(rows),
                amplitude_sha256=_sha256(psi), density_sha256=_sha256(density),
                source_identity=str(source_identity))


def cached_derivative_memory_plan(records: int, parameters: int, *,
                                  row_batch_size: int = 1,
                                  parameter_chunk_size: int = 16384) -> dict[str, int]:
    """Return conservative array-storage bounds, excluding AD/XLA workspace.

    AD row batching and resident algebra-block size are different choices.
    No value returned here certifies model-dependent activation workspace or
    a particular accelerator's measured peak.
    """
    for name, value in (("records", records), ("parameters", parameters),
                        ("row_batch_size", row_batch_size),
                        ("parameter_chunk_size", parameter_chunk_size)):
        _positive_int(value, name)
    row_width = min(records, row_batch_size)
    column_width = min(parameters, parameter_chunk_size)
    cache_bytes = records * parameters * 8
    block_bytes = records * column_width * 8
    return dict(raw_host_cache_bytes=cache_bytes,
                host_build_array_bound_bytes=cache_bytes + 2 * row_width * parameters * 8,
                host_solve_array_bound_bytes=cache_bytes + 2 * parameters * 8 + 2 * block_bytes,
                resident_raw_device_bytes=cache_bytes,
                streamed_solve_device_array_bound_bytes=(4 * block_bytes
                    + 3 * records * records * 8 + 2 * parameters * 8
                    + 16 * records * 8),
                AD_row_batch=row_width, algebra_parameter_chunk=column_width,
                exact_solver_Jacobian_passes=3)


@dataclass(frozen=True)
class CachedAmplitudeJacobian:
    """Immutable parent identity and a real [records, native parameters] cache.

    The ndarray/memmap itself must not be mutated by the caller.  A file cache
    is read-only when returned.  Failed builds leave an explicitly incomplete
    output file, never a silently reusable derivative cache.
    """
    jacobian: np.ndarray
    psi: np.ndarray
    density: np.ndarray
    metadata: dict[str, Any]

    def verify(self, theta: Any, rows: Any, psi: Any, density: Any, *,
               source_identity: str = "") -> None:
        """Reject reuse after a parameter, sample, normalization, or code change."""
        _require_x64()
        found = amplitude_cache_identity(theta, rows, psi, density,
                                         source_identity=source_identity)
        if found != self.metadata["identity"]:
            raise ValueError("derivative cache does not match this frozen parent/records/Q")


def build_cached_amplitude_jacobian(
    batch_amplitude: Callable, theta: Any, rows: Any, psi: Any, density: Any, *,
    row_batch_size: int = 1, maximum_host_bytes: int,
    storage_path: str | Path | None = None, source_identity: str = "",
    progress: Progress | None = None,
) -> CachedAmplitudeJacobian:
    """Differentiate each frozen physical row once, without dividing by psi.

    ``batch_amplitude(theta, rows)`` returns one real scalar per row, including
    exact target nodes.  All native columns are retained.  The first and warm
    block timings are separate; they must not be extrapolated interchangeably.
    ``maximum_host_bytes`` is an explicit array-storage budget, not an RSS or
    GPU certificate.  A memory-mapped output target must not already exist.
    """
    _require_x64()
    theta_host = _real64(theta, "theta", 1)
    psi_host = _real64(psi, "psi", 1)
    density_host = _real64(density, "density", 1)
    row_host = np.asarray(rows)
    if (row_host.ndim != 2 or row_host.dtype.kind not in "iu"
            or len(row_host) < 2 or len(theta_host) < 1
            or psi_host.shape != (len(row_host),) or density_host.shape != psi_host.shape
            or np.any(density_host <= 0)
            or not np.isfinite(np.sum(psi_host ** 2 / density_host))
            or np.sum(psi_host ** 2 / density_host) <= 0):
        raise ValueError("aligned physical integer rows, positive Q and nonzero parent required")
    _positive_int(row_batch_size, "row_batch_size")
    _positive_int(maximum_host_bytes, "maximum_host_bytes")
    plan = cached_derivative_memory_plan(len(row_host), len(theta_host),
                                          row_batch_size=row_batch_size)
    if plan["host_build_array_bound_bytes"] > maximum_host_bytes:
        raise MemoryError("exact raw derivative cache exceeds registered host array budget")
    shape = (len(row_host), len(theta_host))
    path = None if storage_path is None else Path(storage_path)
    if path is not None:
        if path.exists() or path.is_symlink():
            raise FileExistsError("never overwrite an existing derivative-cache target")
        if Path(str(path)+".json").exists():
            raise FileExistsError("never overwrite an existing derivative-cache manifest")
        if not path.parent.is_dir():
            raise FileNotFoundError("derivative-cache parent directory must already exist")
        # Exclusive reservation prevents accidental clobbering before open_memmap.
        with path.open("xb"):
            pass
        jacobian = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
    else:
        jacobian = np.empty(shape, np.float64)
    flat = jnp.asarray(theta_host)

    def row_value(values, row):
        value = batch_amplitude(values, row[None, :])
        if value.shape != (1,) or value.dtype != jnp.float64:
            raise ValueError("batch_amplitude must return real float64 [rows], not complex")
        return value[0]

    evaluate = jax.jit(jax.vmap(jax.value_and_grad(row_value), in_axes=(None, 0)))
    timings = []
    raw_digest = hashlib.sha256()
    observed = np.empty_like(psi_host)
    for first in range(0, len(row_host), row_batch_size):
        final = min(first + row_batch_size, len(row_host))
        tick = perf_counter()
        values, derivatives = evaluate(flat, jnp.asarray(row_host[first:final]))
        derivatives.block_until_ready()
        kernel_seconds = perf_counter() - tick
        tick = perf_counter()
        block = np.asarray(derivatives)
        observed[first:final] = np.asarray(values)
        if not np.isfinite(block).all() or not np.isfinite(observed[first:final]).all():
            raise FloatingPointError("nonfinite amplitude derivative cache; no repaired entries")
        jacobian[first:final] = block
        raw_digest.update(memoryview(np.ascontiguousarray(block)).cast("B"))
        transfer_seconds = perf_counter() - tick
        note = dict(stage="raw_J_block_cached", first_record=first, records=final-first,
                    kernel_seconds=kernel_seconds, host_transfer_seconds=transfer_seconds,
                    cold_or_new_shape=first == 0 or final-first != row_batch_size,
                    retained_records=final, total_records=len(row_host))
        timings.append(note)
        if progress is not None:
            progress(note)
    relative = float(np.linalg.norm(observed-psi_host)
                     / max(np.linalg.norm(psi_host), np.finfo(float).tiny))
    weighted_relative = float(np.sqrt(np.sum((observed-psi_host)**2/density_host)
                                     / np.sum(psi_host**2/density_host)))
    if (not np.isfinite(relative) or relative > 1e-10
            or not np.isfinite(weighted_relative) or weighted_relative > 1e-10):
        raise ArithmeticError("cached derivative values disagree with supplied frozen amplitudes")
    if isinstance(jacobian, np.memmap):
        jacobian.flush()
        del jacobian
        jacobian = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        jacobian.flags.writeable = False
    metadata = dict(identity=amplitude_cache_identity(theta_host, row_host, psi_host,
                        density_host, source_identity=source_identity),
                    complete=True, records=len(row_host), parameters=len(theta_host),
                    dtype="float64", raw_cache_bytes=int(np.prod(shape))*8,
                    storage_path=None if path is None else str(path),
                    amplitude_relative_replay_error=relative, block_timings=timings,
                    amplitude_Q_weighted_relative_replay_error=weighted_relative,
                    raw_content_sha256=raw_digest.hexdigest(),
                    registered_maximum_host_bytes=maximum_host_bytes,
                    active_maximum_host_bytes=maximum_host_bytes,
                    full_native_columns=True, amplitude_not_log_derivative=True,
                    no_AD_recomputed_by_cached_solver=True, memory_plan=plan)
    result = CachedAmplitudeJacobian(jacobian, psi_host.copy(), density_host.copy(), metadata)
    if path is not None:
        manifest = dict(metadata=metadata, psi=psi_host.tolist(), density=density_host.tolist())
        # A manifest is created only after all rows, replay, and flush succeed.
        with Path(str(path)+".json").open("x") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def load_cached_amplitude_jacobian(storage_path: str | Path, *,
                                   maximum_host_bytes: int,
                                   verify_content: bool = True,
                                   hash_row_batch_size: int = 1,
                                   ) -> CachedAmplitudeJacobian:
    """Open a completed cache read-only, with optional full content verification.

    The default checks all raw bytes, not only file existence.  The caller must
    additionally call ``cache.verify(...)`` against the intended frozen parent,
    ordered records and Q.  An interrupted build without its complete manifest
    is not accepted.  Hashing is bounded by row batches but reads the full file
    and must be included in the restart wall-time budget.
    """
    _require_x64()
    _positive_int(maximum_host_bytes, "maximum_host_bytes")
    _positive_int(hash_row_batch_size, "hash_row_batch_size")
    path = Path(storage_path)
    saved = json.loads(Path(str(path)+".json").read_text())
    metadata = saved["metadata"]
    if not metadata.get("complete") or metadata.get("dtype") != "float64":
        raise ValueError("only complete real float64 derivative caches can be restored")
    psi = _real64(np.asarray(saved["psi"], np.float64), "saved psi", 1)
    density = _real64(np.asarray(saved["density"], np.float64), "saved Q", 1)
    if (_sha256(psi) != metadata["identity"]["amplitude_sha256"]
            or _sha256(density) != metadata["identity"]["density_sha256"]):
        raise ValueError("saved amplitude/Q content SHA256 differs from frozen cache identity")
    shape = (int(metadata["records"]), int(metadata["parameters"]))
    plan = cached_derivative_memory_plan(*shape, row_batch_size=hash_row_batch_size)
    if plan["host_build_array_bound_bytes"] > maximum_host_bytes:
        raise MemoryError("restored exact derivative cache exceeds registered host array budget")
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    if raw.shape != shape or raw.dtype != np.dtype("float64"):
        raise ValueError("derivative cache file shape/dtype differs from completed manifest")
    if verify_content:
        digest = hashlib.sha256()
        for first in range(0, shape[0], hash_row_batch_size):
            block = np.ascontiguousarray(raw[first:first+hash_row_batch_size])
            if not np.isfinite(block).all():
                raise ValueError("nonfinite restored derivative cache")
            digest.update(memoryview(block).cast("B"))
        if digest.hexdigest() != metadata["raw_content_sha256"]:
            raise ValueError("restored derivative cache raw-content SHA256 mismatch")
    metadata = dict(metadata, active_maximum_host_bytes=maximum_host_bytes)
    result = CachedAmplitudeJacobian(raw, psi, density, metadata)
    _physical_data(result, psi, density)
    return result


def _physical_data(cache: CachedAmplitudeJacobian, psi: Any, density: Any,
                   hpsi: Any | None = None):
    _require_x64()
    p, q = _real64(psi, "psi", 1), _real64(density, "density", 1)
    if (p.shape != (cache.jacobian.shape[0],) or q.shape != p.shape
            or np.any(q <= 0) or not np.array_equal(p, cache.psi)
            or not np.array_equal(q, cache.density) or not cache.metadata.get("complete")):
        raise ValueError("complete cache with identical frozen psi/Q required")
    norm = np.sum(p*p/q)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("positive finite sampled parent norm required")
    scale = 1/np.sqrt(q*norm)
    u = p*scale
    if hpsi is None:
        return jnp.asarray(scale), jnp.asarray(u)
    hp = _real64(hpsi, "Hpsi", 1)
    if hp.shape != p.shape:
        raise ValueError("Hpsi must match the complete cache record count")
    energy = float(np.sum(p*hp/q)/norm)
    force = 2*(hp-energy*p)*scale
    return jnp.asarray(scale), jnp.asarray(u), jnp.asarray(force), energy, norm


def cached_projected_operators(cache: CachedAmplitudeJacobian, psi: Any, density: Any,
                               *, maximum_device_bytes: int):
    """Return resident exact A/A.T closures compatible with existing LSQR.

    This is appropriate for the old CT12 profile, whose 1024-record real
    derivative cache is about 11.4 GiB.  The all-width28 158-GiB cache must use
    the host-streamed dual solver below rather than this resident backend.
    """
    _require_x64()
    _positive_int(maximum_device_bytes, "maximum_device_bytes")
    if cache.jacobian.nbytes * 2 > maximum_device_bytes:
        raise MemoryError("resident cached operators exceed registered device array budget")
    scale, u = _physical_data(cache, psi, density)
    raw = jnp.asarray(cache.jacobian)
    def forward(vector):
        z = (raw @ vector)*scale
        return z-u*jnp.vdot(u, z).real
    def adjoint(weights):
        z = weights-u*jnp.vdot(u, weights).real
        return raw.T @ (z*scale)
    return forward, adjoint


def solve_cached_projected_sr(
    cache: CachedAmplitudeJacobian, psi: Any, hpsi: Any, density: Any, *,
    damping: float, parameter_chunk_size: int = 16384,
    maximum_device_bytes: int, maximum_host_bytes: int | None = None,
    progress: Progress | None = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Exact host-cached SR, (A.T A+lambda I)d=A.T b, in three J passes.

    Accumulate A A.T by bounded parameter blocks, solve the record-space
    system, and stream A.T times its solution.  A third pass checks the true
    native-coordinate normal residual.  Every parameter column/record is
    retained; the metric is F, not 2F.  Nothing here estimates population
    geometry, validates finite steps, or certifies importance-tail coverage.
    The host solve budget defaults to the active build/restore budget stored
    in the cache; an explicit override is checked before any algebra allocation.
    """
    _require_x64()
    if isinstance(damping, bool) or not isfinite(damping) or damping <= 0:
        raise ValueError("positive finite damping required")
    _positive_int(parameter_chunk_size, "parameter_chunk_size")
    _positive_int(maximum_device_bytes, "maximum_device_bytes")
    count, parameters = cache.jacobian.shape
    width = min(parameters, parameter_chunk_size)
    plan = cached_derivative_memory_plan(count, parameters, parameter_chunk_size=width)
    host_budget = (cache.metadata.get("active_maximum_host_bytes")
                   if maximum_host_bytes is None else maximum_host_bytes)
    _positive_int(host_budget, "maximum_host_bytes")
    if plan["host_solve_array_bound_bytes"] > host_budget:
        raise MemoryError("streamed cached algebra exceeds registered host array budget")
    if plan["streamed_solve_device_array_bound_bytes"] > maximum_device_bytes:
        raise MemoryError("streamed cached algebra exceeds registered device array budget")
    scale, u, force, energy, norm = _physical_data(cache, psi, density, hpsi)
    direction_host, gradient_host = np.empty(parameters), np.empty(parameters)
    gram = jnp.zeros((count, count), jnp.float64)
    product = jnp.zeros(count, jnp.float64)

    @jax.jit
    def horizontal(raw):
        scaled = raw*scale[:, None]
        return scaled-u[:, None]*(u@scaled)[None, :]

    @jax.jit
    def add_gram(current, raw):
        a = horizontal(raw)
        return current+a@a.T

    @jax.jit
    def native_vectors(raw, coefficients):
        a = horizontal(raw)
        d, g = a.T@coefficients, a.T@force
        return d, g, a@d

    @jax.jit
    def residual_squared(raw, product, d, g):
        a = horizontal(raw)
        residual = a.T@product+damping*d-g
        return jnp.vdot(residual, residual).real

    def blocks():
        for first in range(0, parameters, width):
            final = min(first+width, parameters)
            block = np.ascontiguousarray(cache.jacobian[:, first:final])
            if not np.isfinite(block).all():
                raise FloatingPointError("nonfinite raw derivative block; no clipped columns")
            if final-first != width:
                block = np.pad(block, ((0, 0), (0, width-(final-first))))
            yield first, final, jnp.asarray(block)

    timings = []
    def report(stage, started):
        note = dict(stage=stage, seconds=perf_counter()-started, records=count,
                    parameters=parameters, parameter_blocks=(parameters+width-1)//width)
        timings.append(note)
        if progress is not None:
            progress(note)

    tick = perf_counter()
    for _, _, raw in blocks():
        gram = add_gram(gram, raw)
        gram.block_until_ready()  # do not queue all host blocks on the accelerator
    report("cached_J_pass1_dual_Gram", tick)
    tick = perf_counter()
    coefficients = jax.scipy.linalg.solve(gram+damping*jnp.eye(count), force, assume_a="pos")
    coefficients.block_until_ready()
    report("cached_dual_solve", tick)
    tick = perf_counter()
    for first, final, raw in blocks():
        d, g, image = native_vectors(raw, coefficients)
        direction_host[first:final] = np.asarray(d)[:final-first]
        gradient_host[first:final] = np.asarray(g)[:final-first]
        product = product+image
        product.block_until_ready()
    report("cached_J_pass2_native_direction", tick)
    squared = jnp.asarray(0., jnp.float64)
    tick = perf_counter()
    for first, final, raw in blocks():
        d = np.pad(direction_host[first:final], (0, width-(final-first)))
        g = np.pad(gradient_host[first:final], (0, width-(final-first)))
        squared = squared+residual_squared(raw, product, jnp.asarray(d), jnp.asarray(g))
        squared.block_until_ready()
    report("cached_J_pass3_true_normal_residual", tick)
    gradient_norm = float(np.linalg.norm(gradient_host))
    normal_norm = float(jnp.sqrt(squared))
    relative = normal_norm/gradient_norm if gradient_norm else (0. if normal_norm == 0 else float("inf"))
    valid = bool(np.isfinite(direction_host).all() and np.isfinite(gradient_host).all()
                 and np.isfinite(relative))
    if not valid:
        raise FloatingPointError("nonfinite exact cached SR result")
    tangent = np.asarray(product)
    return jnp.asarray(direction_host), jnp.asarray(gradient_host), dict(
        numerically_valid=True, relative_linear_residual=relative,
        normal_residual_norm=normal_norm, gradient_norm=gradient_norm,
        raw_fisher_norm=float(np.linalg.norm(tangent)), sampled_parent_norm=norm/count,
        sampled_energy_total=energy, exact_node_records=int(np.count_nonzero(np.asarray(psi)==0)),
        parameter_blocks=(parameters+width-1)//width, parameter_chunk_size=width,
        retained_raw_jacobian_entries=int(cache.jacobian.size),
        metric_is_F_not_two_F=True, force_is_population_Hermitian_estimator=True,
        no_AD_recomputed_in_solve=True, exact_Jacobian_passes=3,
        chosen_direction_tangent=tangent,
        memory_plan=plan, timings=timings,
        finite_quadrature_is_not_independent_direction_or_finite_step_confirmation=True)


def solve_cached_node_block_sr(
    cache: CachedAmplitudeJacobian, psi: Any, hpsi: Any, density: Any, *,
    blocks: list[tuple[str, int, int]], relative_damping: float = .5,
    ridge_floor_relative: float = 1e-12, parameter_chunk_size: int = 16384,
    maximum_device_bytes: int, maximum_host_bytes: int,
    progress: Progress | None = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """All-column node-block SR with a sample-eigenvalue damping scale.

    Each explicit contiguous block solves ``(F_bb+lambda_b I)d_b=g_b``;
    ``lambda_b=relative_damping*trace(F_bb)/(records-1)`` with a disclosed
    relative numerical floor. This is a block-diagonal *update geometry*,
    not the exact global Fisher solve. All records and native columns remain.
    The combined physical tangent includes cross-node terms and is computed
    using the complete cached Jacobian. No derivatives or H queries are made.
    Real amplitudes imply the phase tangent is identically orthogonal; global
    normalization is projected over the complete panel, never per row chunk.
    """
    _require_x64()
    if cache.jacobian.dtype != np.dtype('float64'):
        raise ValueError('node-block SR requires real float64 cached derivatives')
    count, parameters = cache.jacobian.shape
    if (isinstance(relative_damping, bool) or isinstance(ridge_floor_relative, bool)
            or count < 2 or not blocks or not np.isfinite((relative_damping, ridge_floor_relative)).all()
            or relative_damping <= 0 or ridge_floor_relative <= 0):
        raise ValueError("positive damping/floor and at least two records required")
    cursor = 0; labels = set()
    for label, first, final in blocks:
        if (not isinstance(label, str) or not label or label in labels
                or isinstance(first, bool) or isinstance(final, bool)
                or not isinstance(first, int) or not isinstance(final, int)
                or first != cursor or not first < final <= parameters):
            raise ValueError("named contiguous blocks must partition every native column exactly once")
        labels.add(label); cursor = final
    if cursor != parameters:
        raise ValueError("blocks must retain all native columns")
    _positive_int(parameter_chunk_size, "parameter_chunk_size")
    _positive_int(maximum_host_bytes, "maximum_host_bytes")
    _positive_int(maximum_device_bytes, "maximum_device_bytes")
    width = min(parameters, parameter_chunk_size)
    plan = cached_derivative_memory_plan(count, parameters, parameter_chunk_size=width)
    host_extra = (len(blocks)*count*count+len(blocks)*count)*8
    if plan['host_solve_array_bound_bytes']+host_extra > maximum_host_bytes:
        raise MemoryError("node Gram storage exceeds registered host array budget")
    if plan['streamed_solve_device_array_bound_bytes'] > maximum_device_bytes:
        raise MemoryError("node-block algebra exceeds registered device array budget")
    scale, unit, force, energy, norm = _physical_data(cache, psi, density, hpsi)
    direction, gradient = np.zeros(parameters), np.zeros(parameters)
    node_tangents = np.zeros((len(blocks), count))
    grams = []; notes = []; timings = []

    @jax.jit
    def horizontal(raw):
        z = raw*scale[:, None]
        return z-unit[:, None]*(unit@z)[None, :]

    @jax.jit
    def add_gram(gram, raw):
        a = horizontal(raw)
        return gram+a@a.T

    @jax.jit
    def vectors(raw, coefficients):
        a = horizontal(raw)
        d, g = a.T@coefficients, a.T@force
        return d, g, a@d

    @jax.jit
    def errors(raw, local_tangent, total_tangent, d, g, damping):
        a = horizontal(raw)
        local = a.T@local_tangent+damping*d-g
        coupled = a.T@total_tangent+damping*d-g
        return jnp.vdot(local, local).real, jnp.vdot(coupled, coupled).real

    def chunks(first, final):
        for start in range(first, final, width):
            stop = min(start+width, final)
            raw = np.ascontiguousarray(cache.jacobian[:, start:stop])
            if not np.isfinite(raw).all():
                raise FloatingPointError("nonfinite node derivative block")
            if stop-start != width:
                raw = np.pad(raw, ((0, 0), (0, width-(stop-start))))
            yield start, stop, jnp.asarray(raw)

    def update(stage, tick, **extra):
        note = dict(stage=stage, seconds=perf_counter()-tick, **extra)
        timings.append(note)
        if progress is not None:
            progress(note)

    for label, first, final in blocks:
        tick = perf_counter()
        gram = jnp.zeros((count, count), jnp.float64)
        for _, _, raw in chunks(first, final):
            gram = add_gram(gram, raw); gram.block_until_ready()
        matrix = np.asarray(gram)
        trace = float(np.trace(matrix))
        if not np.isfinite(matrix).all() or not np.isfinite(trace) or trace < 0:
            raise ArithmeticError("nonfinite/nonpositive node Fisher Gram")
        grams.append(matrix)
        notes.append(dict(label=label, first=first, final=final, parameters=final-first,
                          Fisher_trace=trace, sample_eigenvalue_scale=trace/(count-1)))
        update('node_Gram_cached', tick, label=label, completed_nodes=len(grams))
    reference = max(x['sample_eigenvalue_scale'] for x in notes)
    if reference <= 0:
        raise ArithmeticError("all native blocks have zero physical metric")
    floor = ridge_floor_relative*reference
    for index, (label, first, final) in enumerate(blocks):
        tick = perf_counter()
        ridge = max(relative_damping*notes[index]['sample_eigenvalue_scale'], floor)
        matrix = jnp.asarray(grams[index])
        coefficients = jax.scipy.linalg.solve(matrix+ridge*jnp.eye(count), force, assume_a='pos')
        coefficients.block_until_ready()
        tangent = jnp.zeros(count, jnp.float64)
        for start, stop, raw in chunks(first, final):
            d, g, image = vectors(raw, coefficients)
            direction[start:stop] = np.asarray(d)[:stop-start]
            gradient[start:stop] = np.asarray(g)[:stop-start]
            tangent = tangent+image; tangent.block_until_ready()
        node_tangents[index] = np.asarray(tangent)
        notes[index]['ridge'] = ridge
        update('node_native_direction', tick, label=label, completed_nodes=index+1)
    combined = node_tangents.sum(axis=0)
    coupled_squared = 0.; block_squared = 0.
    for index, (label, first, final) in enumerate(blocks):
        tick = perf_counter(); local_squared = jnp.asarray(0., jnp.float64)
        coupled = jnp.asarray(0., jnp.float64)
        for start, stop, raw in chunks(first, final):
            d = np.pad(direction[start:stop], (0, width-(stop-start)))
            g = np.pad(gradient[start:stop], (0, width-(stop-start)))
            a, b = errors(raw, jnp.asarray(node_tangents[index]), jnp.asarray(combined),
                         jnp.asarray(d), jnp.asarray(g), notes[index]['ridge'])
            local_squared += a; coupled += b; local_squared.block_until_ready()
        block_squared += float(local_squared); coupled_squared += float(coupled)
        rhs = float(np.linalg.norm(gradient[first:final]))
        notes[index]['relative_block_linear_residual'] = float(np.sqrt(float(local_squared)))/max(rhs,np.finfo(float).tiny)
        update('true_node_normal_residual', tick, label=label, completed_nodes=index+1)
    rhs = float(np.linalg.norm(gradient))
    residual = np.sqrt(block_squared)/max(rhs,np.finfo(float).tiny)
    if not np.isfinite(direction).all() or not np.isfinite(residual):
        raise FloatingPointError("nonfinite node-block SR result")
    return jnp.asarray(direction), jnp.asarray(gradient), dict(
        numerically_valid=True, relative_linear_residual=float(residual),
        full_global_Fisher_solved=False, update_geometry='explicit_node_block_diagonal_Fisher',
        global_cross_node_normal_residual=float(np.sqrt(coupled_squared)/max(rhs,np.finfo(float).tiny)),
        relative_damping=float(relative_damping), ridge_floor_relative=float(ridge_floor_relative),
        numerical_ridge_floor=float(floor), ridge_scale='trace(F_node)/(records-1), not trace/P',
        all_native_columns_retained=True, all_records_retained=True,
        raw_fisher_norm=float(np.linalg.norm(combined)), chosen_direction_tangent=combined,
        node_tangent_Gram=node_tangents@node_tangents.T, blocks=notes,
        gradient_norm=rhs, sampled_parent_norm=norm/count, sampled_energy_total=energy,
        exact_Jacobian_passes=3, no_AD_recomputed_in_solve=True,
        raw_J_storage_bytes=int(cache.jacobian.nbytes), additional_node_Gram_bytes=host_extra,
        memory_plan=plan, timings=timings,
        direction_and_finite_energy_require_new_independent_confirmation=True)


__all__ = ["CachedAmplitudeJacobian", "amplitude_cache_identity",
           "cached_derivative_memory_plan", "build_cached_amplitude_jacobian",
           "load_cached_amplitude_jacobian", "cached_projected_operators",
           "solve_cached_projected_sr", "solve_cached_node_block_sr"]
