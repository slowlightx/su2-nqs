"""JAX kernels for full SU(2)-invariant physical-spin observables.

The structure factor implemented here is

``S(w) = normalization**-1 * sum_ij w_i conj(w_j) S_i . S_j``.

For reciprocal phases ``w_i = exp(1j * q . r_i)`` this is the usual
``S(q)``.  The default normalization is the literature-standard ``N``.  The
order-parameter-squared convention with normalization ``N**2`` must be
requested explicitly and is labelled ``m2`` in the compiled specification.

Unlike a diagonal ``S^z S^z`` estimator, the action below includes the
transverse ``(S_i^+ S_j^- + S_i^- S_j^+)/2`` exchange.  It is therefore a
local estimator of the full rotationally invariant spin correlation.  Bits
follow the repository convention ``0=down`` and ``1=up``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np


SpinAmplitude = Callable[[object, jax.Array], jax.Array]


@dataclass(frozen=True)
class SpinStructureFactorSpec:
    """Static coefficients for one reciprocal-space spin structure factor.

    Instances are prepared on the host and can then be closed over by a JIT
    compiled sampled-VMC kernel.  ``pair_coefficients`` multiply
    ``S_i . S_j`` for unordered pairs; the two ordered Fourier terms have
    already been combined into ``2 Re(w_i conj(w_j))``.
    """

    n_sites: int
    phase_weights: tuple[complex, ...]
    normalization: float
    normalization_kind: str
    onsite_coefficient: float
    pair_first: tuple[int, ...]
    pair_second: tuple[int, ...]
    pair_coefficients: tuple[float, ...]


def reciprocal_phase_weights(
    coordinates: Sequence[Sequence[float]],
    wavevector: Sequence[float],
) -> np.ndarray:
    """Return ``exp(1j * q . r_i)`` for finite real coordinates.

    Coordinates may have any positive spatial dimension.  Square-lattice
    ``Neel`` weights use Cartesian ``q=(pi,pi)``.  With the triangular phase
    coordinates ``(row,column)`` used by the benchmark layer, the registered
    points are ``K=2*pi*(1/3,1/3)`` and
    ``M=(pi,0),(0,pi),(pi,pi)``.
    """

    points = np.asarray(coordinates, dtype=np.float64)
    momentum = np.asarray(wavevector, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] < 1:
        raise ValueError("coordinates must be a nonempty rank-two array")
    if momentum.ndim != 1 or momentum.shape[0] != points.shape[1]:
        raise ValueError("wavevector dimension must match the coordinates")
    if np.any(~np.isfinite(points)) or np.any(~np.isfinite(momentum)):
        raise ValueError("coordinates and wavevector must be finite")
    weights = np.exp(1.0j * (points @ momentum)).astype(np.complex128)
    weights.setflags(write=False)
    return weights


def prepare_spin_structure_factor(
    phase_weights: Sequence[complex],
    *,
    normalization: float | None = None,
) -> SpinStructureFactorSpec:
    """Validate reciprocal weights and compile unordered-pair coefficients.

    ``normalization=None`` selects the standard structure factor ``1/N``.
    Passing ``N**2`` explicitly selects the order-parameter-squared quantity
    ``m2``.  For a complete discrete reciprocal grid their spin-half sum
    rules are ``sum_q S(q)=3*N/4`` and ``sum_q m2(q)=3/4``, respectively.
    General complex weights are allowed; the compiled observable is the
    Hermitian positive operator ``sum_a A_a A_a^dagger`` with
    ``A_a=sum_i w_i S_i^a``.
    """

    weights = np.asarray(tuple(phase_weights), dtype=np.complex128)
    if weights.ndim != 1 or weights.size < 1:
        raise ValueError("phase_weights must be a nonempty rank-one sequence")
    if np.any(~np.isfinite(weights)):
        raise ValueError("phase_weights must be finite")
    n_sites = int(weights.size)
    if normalization is None:
        scale = float(n_sites)
    else:
        scale = float(normalization)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("normalization must be positive and finite")
    if scale == float(n_sites):
        normalization_kind = "structure_factor"
    elif scale == float(n_sites**2):
        normalization_kind = "m2"
    else:
        normalization_kind = "custom"

    first, second = np.triu_indices(n_sites, k=1)
    coefficients = 2.0 * np.real(weights[first] * np.conj(weights[second]))
    return SpinStructureFactorSpec(
        n_sites=n_sites,
        phase_weights=tuple(complex(value) for value in weights),
        normalization=scale,
        normalization_kind=normalization_kind,
        onsite_coefficient=float(0.75 * np.vdot(weights, weights).real),
        pair_first=tuple(int(value) for value in first),
        pair_second=tuple(int(value) for value in second),
        pair_coefficients=tuple(float(value) for value in coefficients),
    )


def spin_structure_factor_multi_hpsi(
    params: object,
    samples: jax.Array,
    specs: Sequence[SpinStructureFactorSpec],
    *,
    amplitude_fn: SpinAmplitude,
) -> tuple[jax.Array, jax.Array]:
    """Return centers and several ``S(w)|psi>`` actions on one sample batch.

    ``amplitude_fn`` has the narrow signature ``(params, bits) -> amplitude``.
    Model-specific static configuration objects should be supplied by a
    closure.  All modes share the same center and pair-swapped amplitude
    evaluations; only their static Fourier coefficients and normalizations
    differ.  The returned action shape is ``(batch, modes)``.

    The kernel is differentiable in ``params`` and can be JIT compiled;
    ``specs`` is intended to remain a static closed-over tuple.
    """

    compiled = tuple(specs)
    if not compiled:
        raise ValueError("specs must contain at least one structure factor")
    reference = compiled[0]
    for spec in compiled[1:]:
        if spec.n_sites != reference.n_sites:
            raise ValueError("all structure-factor specs must have the same site count")
        if (
            spec.pair_first != reference.pair_first
            or spec.pair_second != reference.pair_second
        ):
            raise ValueError("all structure-factor specs must share pair ordering")
    spins = jnp.asarray(samples, dtype=jnp.int32)
    if spins.ndim != 2 or spins.shape[0] < 1:
        raise ValueError("samples must be a nonempty rank-two array")
    if spins.shape[1] != reference.n_sites:
        raise ValueError(
            f"expected {reference.n_sites} spin bits per sample, got {spins.shape[1]}"
        )
    centers = jax.vmap(lambda row: amplitude_fn(params, row))(spins)
    if centers.ndim != 1:
        raise ValueError("amplitude_fn must return one scalar per configuration")

    real_dtype = jnp.real(centers).dtype
    scales = jnp.asarray(
        tuple(spec.normalization for spec in compiled), dtype=real_dtype
    )
    onsite = jnp.asarray(
        tuple(spec.onsite_coefficient for spec in compiled), dtype=real_dtype
    )
    if not reference.pair_first:
        actions = centers[:, None] * (onsite / scales)[None, :]
        return centers, actions

    first = jnp.asarray(reference.pair_first, dtype=jnp.int32)
    second = jnp.asarray(reference.pair_second, dtype=jnp.int32)
    coefficients = jnp.asarray(
        tuple(spec.pair_coefficients for spec in compiled), dtype=real_dtype
    )
    pair_slots = jnp.arange(first.shape[0], dtype=jnp.int32)

    same = spins[:, first] == spins[:, second]
    diagonal = jnp.where(same, 0.25, -0.25)
    diagonal_coefficients = onsite[None, :] + diagonal @ coefficients.T

    def swapped_pairs(row: jax.Array) -> jax.Array:
        connected = jnp.broadcast_to(row, (first.shape[0], reference.n_sites))
        connected = connected.at[pair_slots, first].set(row[second])
        return connected.at[pair_slots, second].set(row[first])

    connected = jax.vmap(swapped_pairs)(spins)
    flattened = connected.reshape((-1, reference.n_sites))
    connected_amplitudes = jax.vmap(
        lambda row: amplitude_fn(params, row)
    )(flattened).reshape(same.shape)
    active_connected = jnp.where(
        same,
        jnp.zeros((), dtype=connected_amplitudes.dtype),
        connected_amplitudes,
    )
    transverse = 0.5 * active_connected @ coefficients.T
    actions = (
        diagonal_coefficients * centers[:, None] + transverse
    ) / scales[None, :]
    return centers, actions


def spin_structure_factor_hpsi(
    params: object,
    samples: jax.Array,
    spec: SpinStructureFactorSpec,
    *,
    amplitude_fn: SpinAmplitude,
) -> tuple[jax.Array, jax.Array]:
    """Return center amplitudes and one ``S(w)|psi>`` action.

    This compatibility wrapper uses the shared multi-mode kernel, so there is
    only one implementation of the full spin action.
    """

    centers, actions = spin_structure_factor_multi_hpsi(
        params,
        samples,
        (spec,),
        amplitude_fn=amplitude_fn,
    )
    return centers, actions[:, 0]


def spin_structure_factor_multi_local_values(
    params: object,
    samples: jax.Array,
    specs: Sequence[SpinStructureFactorSpec],
    *,
    amplitude_fn: SpinAmplitude,
    zero_tolerance: float = 0.0,
) -> jax.Array:
    """Return local values for several modes with shared amplitude work.

    A center with ``abs(psi) <= zero_tolerance`` yields ``NaN`` rather than a
    finite placeholder.  Such a state has zero Born probability in exact
    arithmetic and must not silently enter a sampled average.  This mirrors
    the fail-closed behavior of the physical local-energy kernels.  The
    returned shape is ``(batch, modes)``.
    """

    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    centers, actions = spin_structure_factor_multi_hpsi(
        params,
        samples,
        specs,
        amplitude_fn=amplitude_fn,
    )
    local = actions / centers[:, None]
    invalid = jnp.full_like(local, jnp.nan)
    return jnp.where(jnp.abs(centers[:, None]) > tolerance, local, invalid)


def spin_structure_factor_local_values(
    params: object,
    samples: jax.Array,
    spec: SpinStructureFactorSpec,
    *,
    amplitude_fn: SpinAmplitude,
    zero_tolerance: float = 0.0,
) -> jax.Array:
    """Return one full-SU(2) structure-factor local-value vector."""

    return spin_structure_factor_multi_local_values(
        params,
        samples,
        (spec,),
        amplitude_fn=amplitude_fn,
        zero_tolerance=zero_tolerance,
    )[:, 0]


__all__ = [
    "SpinAmplitude",
    "SpinStructureFactorSpec",
    "prepare_spin_structure_factor",
    "reciprocal_phase_weights",
    "spin_structure_factor_hpsi",
    "spin_structure_factor_local_values",
    "spin_structure_factor_multi_hpsi",
    "spin_structure_factor_multi_local_values",
]
