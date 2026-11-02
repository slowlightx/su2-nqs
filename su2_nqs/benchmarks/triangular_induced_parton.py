"""Exact three-state induced carrier for the 6x6 triangular torus.

The carrier is the Gutzwiller-projected, spin-independent closed-shell
``pi/0`` flux state.  A convenient gauge has a two-site unit cell and three
gapped boundary conditions, ``PP``, ``AP`` and ``PA``.  Space-group actions
close exactly on this three-dimensional span.  Consequently the fully
projected ``q=0, Gamma=A1`` amplitude is

    (psi_PP - psi_AP - psi_PA) / 3,

instead of a sum over all 432 space-group images.  The fourth boundary
condition, ``AA``, is gapless on 6x6 and is deliberately excluded.

This module contains no trainable parameters.  It is a symmetry-exact carrier
and a benchmark building block, not a claim that the projected parton state
itself identifies the phase of the J1-J2 model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, isqrt, sqrt
from typing import Literal

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.projected_singlet_agp import (
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.models.projected_singlet_agp import (
    ProjectedAGPBatch,
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.lattices.triangular_hnf import (
    C6,
    hnf_site_permutation,
    matrix_power,
)


Boundary = Literal["P", "A"]
BOUNDARY_ORBIT: tuple[tuple[Boundary, Boundary], ...] = (
    ("P", "P"),
    ("A", "P"),
    ("P", "A"),
)

# Columns transform under the convention psi(g sigma) = psi(sigma) @ D_g.
# These integer matrices are fixed by the 2x1 pi/0-flux gauge, rather than
# estimated from Monte Carlo samples.
TX_REPRESENTATION = np.eye(3, dtype=np.float64)
TY_REPRESENTATION = np.eye(3, dtype=np.float64)
C6_REPRESENTATION = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)
MIRROR_REPRESENTATION = np.asarray(
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)

# The factor 1/3 makes this exactly the group projector applied to psi_PP.
A1_COEFFICIENTS = np.asarray([1.0, -1.0, -1.0], dtype=np.float64) / 3.0


def _triangular_gamma_a1_group(linear_size: int) -> np.ndarray:
    return np.asarray(
        triangular_hnf_gamma_a1_group(
            ((int(linear_size), 0), (0, int(linear_size)))
        ),
        dtype=np.int32,
    )


@dataclass(frozen=True)
class PiZeroFluxOrbit:
    """Closed-shell pair kernels spanning the gapped boundary orbit."""

    pairings: np.ndarray
    one_body_gaps: np.ndarray
    excluded_aa_gap: float


@dataclass(frozen=True)
class InducedCarrierBatch:
    """Three component amplitudes and their exact A1 combination."""

    amplitudes: np.ndarray
    components: np.ndarray
    cancellation_ratio: np.ndarray


@dataclass(frozen=True)
class ProjectedScalarBatch:
    """Scalar-state Reynolds projection diagnostics."""

    amplitudes: np.ndarray
    cancellation_ratio: np.ndarray


@dataclass(frozen=True)
class CompressedCollectiveTowerBatch:
    """Exact orbit-reduced collective tower and its distinct image terms."""

    amplitudes: np.ndarray
    components: np.ndarray
    cancellation_ratio: np.ndarray
    semantic_term_count: int


def _validate_boundary(boundary: str) -> Boundary:
    if boundary not in {"P", "A"}:
        raise ValueError("boundary must be 'P' or 'A'")
    return boundary  # type: ignore[return-value]


def _site_index(x: int, y: int, linear_size: int) -> int:
    return (y % linear_size) * linear_size + (x % linear_size)


def _wrap_phase(
    x: int,
    y: int,
    dx: int,
    dy: int,
    linear_size: int,
    boundary_x: Boundary,
    boundary_y: Boundary,
) -> float:
    phase = 1.0
    if boundary_x == "A" and (x + dx < 0 or x + dx >= linear_size):
        phase *= -1.0
    if boundary_y == "A" and (y + dy < 0 or y + dy >= linear_size):
        phase *= -1.0
    return phase


def triangular_pi_zero_flux_hopping(
    linear_size: int = 6,
    *,
    boundary_x: Boundary = "P",
    boundary_y: Boundary = "P",
) -> np.ndarray:
    """Return the real nearest-neighbour hopping matrix in a 2x1 gauge.

    The positive bond directions are ``(1,0)``, ``(0,1)`` and ``(1,1)``.
    Their gauge signs are respectively ``+1``, ``(-1)^x`` and ``(-1)^x``.
    Thus one elementary-triangle orientation has pi flux and the other has
    zero flux.  ``linear_size`` must be positive and even so the gauge closes
    on the torus.
    """

    if linear_size <= 0 or linear_size % 2:
        raise ValueError("linear_size must be a positive even integer")
    boundary_x = _validate_boundary(boundary_x)
    boundary_y = _validate_boundary(boundary_y)
    n_sites = linear_size * linear_size
    hopping = np.zeros((n_sites, n_sites), dtype=np.float64)
    directions = ((1, 0), (0, 1), (1, 1))
    for y in range(linear_size):
        for x in range(linear_size):
            source = _site_index(x, y, linear_size)
            for dx, dy in directions:
                target = _site_index(x + dx, y + dy, linear_size)
                gauge = 1.0 if (dx, dy) == (1, 0) else (-1.0) ** x
                value = gauge * _wrap_phase(
                    x,
                    y,
                    dx,
                    dy,
                    linear_size,
                    boundary_x,
                    boundary_y,
                )
                hopping[source, target] = value
                hopping[target, source] = value
    return hopping


def triangular_elementary_flux_products(
    hopping: np.ndarray,
    linear_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gauge products on the two elementary triangle orientations."""

    hopping = np.asarray(hopping, dtype=np.float64)
    expected = (linear_size * linear_size, linear_size * linear_size)
    if hopping.shape != expected:
        raise ValueError(f"hopping must have shape {expected}")
    down: list[float] = []
    up: list[float] = []
    for y in range(linear_size):
        for x in range(linear_size):
            a = _site_index(x, y, linear_size)
            b = _site_index(x + 1, y, linear_size)
            c = _site_index(x + 1, y + 1, linear_size)
            d = _site_index(x, y + 1, linear_size)
            down.append(hopping[a, b] * hopping[b, c] * hopping[c, a])
            up.append(hopping[a, d] * hopping[d, c] * hopping[c, a])
    return np.asarray(down), np.asarray(up)


def closed_shell_pairing_from_hopping(
    hopping: np.ndarray,
    *,
    gap_tolerance: float = 1.0e-10,
    require_closed_shell: bool = True,
) -> tuple[np.ndarray, float]:
    """Build the singlet pair kernel from the occupied one-body subspace."""

    hopping = np.asarray(hopping, dtype=np.float64)
    if hopping.ndim != 2 or hopping.shape[0] != hopping.shape[1]:
        raise ValueError("hopping must be a square matrix")
    n_sites = hopping.shape[0]
    if n_sites % 2:
        raise ValueError("half filling requires an even number of sites")
    eigenvalues, eigenvectors = np.linalg.eigh(hopping)
    n_occupied = n_sites // 2
    gap = float(eigenvalues[n_occupied] - eigenvalues[n_occupied - 1])
    if require_closed_shell and gap <= gap_tolerance:
        raise ValueError(
            f"one-body shell is open: gap={gap:.6e} <= {gap_tolerance:.6e}"
        )
    occupied = eigenvectors[:, :n_occupied]
    pair_matrix = occupied @ occupied.T
    upper = np.triu_indices(n_sites, 1)
    return np.asarray(pair_matrix[upper], dtype=np.float64), gap


def build_n36_pi_zero_flux_orbit(
    *, gap_tolerance: float = 1.0e-10
) -> PiZeroFluxOrbit:
    """Return the three gapped 6x6 boundary sectors and audit the excluded AA."""

    pairings: list[np.ndarray] = []
    gaps: list[float] = []
    for boundary_x, boundary_y in BOUNDARY_ORBIT:
        hopping = triangular_pi_zero_flux_hopping(
            6, boundary_x=boundary_x, boundary_y=boundary_y
        )
        pairing, gap = closed_shell_pairing_from_hopping(
            hopping, gap_tolerance=gap_tolerance, require_closed_shell=True
        )
        pairings.append(pairing)
        gaps.append(gap)
    aa_hopping = triangular_pi_zero_flux_hopping(
        6, boundary_x="A", boundary_y="A"
    )
    _, aa_gap = closed_shell_pairing_from_hopping(
        aa_hopping, gap_tolerance=gap_tolerance, require_closed_shell=False
    )
    if aa_gap > gap_tolerance:
        raise RuntimeError("AA unexpectedly became a closed-shell boundary sector")
    return PiZeroFluxOrbit(
        pairings=np.stack(pairings, axis=0),
        one_body_gaps=np.asarray(gaps),
        excluded_aa_gap=aa_gap,
    )


def induced_representation_audit() -> dict[str, float | int | np.ndarray]:
    """Audit D6 relations and the unique A1 vector of the three-state span."""

    identity = np.eye(3)
    rotation = C6_REPRESENTATION
    mirror = MIRROR_REPRESENTATION
    matrices = [np.linalg.matrix_power(rotation, power) for power in range(6)]
    matrices += [mirror @ matrix for matrix in matrices]
    projector = sum(matrices) / 12.0
    singular_values = np.linalg.svd(projector, compute_uv=False)
    projected_pp = projector @ np.asarray([1.0, 0.0, 0.0])
    return {
        "tx_identity_error": float(np.linalg.norm(TX_REPRESENTATION - identity)),
        "ty_identity_error": float(np.linalg.norm(TY_REPRESENTATION - identity)),
        "c6_order_error": float(
            np.linalg.norm(np.linalg.matrix_power(rotation, 6) - identity)
        ),
        "mirror_order_error": float(np.linalg.norm(mirror @ mirror - identity)),
        "dihedral_error": float(
            np.linalg.norm(mirror @ rotation @ mirror - np.linalg.inv(rotation))
        ),
        "projector_idempotency_error": float(
            np.linalg.norm(projector @ projector - projector)
        ),
        "projector_rank": int(np.sum(singular_values > 1.0e-12)),
        "a1_vector_error": float(np.linalg.norm(projected_pp - A1_COEFFICIENTS)),
        "projector": projector,
    }


def _identity_group(n_sites: int) -> np.ndarray:
    return np.arange(n_sites, dtype=np.int32)[None, :]


def _group_tuple(group: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(site) for site in image) for image in np.asarray(group))


def projected_pi_zero_flux_a1_amplitudes(
    configurations: np.ndarray,
    *,
    orbit: PiZeroFluxOrbit | None = None,
    microbatch_size: int = 64,
) -> InducedCarrierBatch:
    """Evaluate the exact three-determinant ``q=0, Gamma=A1`` carrier.

    A caller-supplied ``orbit`` must already have been certified to transform
    with the fixed induced representation above.  In particular, an arbitrary
    deformation of all three pairing kernels need not preserve closure; the
    full Reynolds oracle must be used to re-certify it.
    """

    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[1] != 36:
        raise ValueError("configurations must have shape (batch, 36)")
    if orbit is None:
        orbit = build_n36_pi_zero_flux_orbit()
    group = _identity_group(36)
    config = ProjectedSingletAGPConfig(
        n_sites=36, group_images=_group_tuple(group)
    )
    components = []
    for pairing in orbit.pairings:
        batch = projected_singlet_agp_amplitudes(
            {"pairing_real": pairing},
            states,
            config,
            microbatch_size=microbatch_size,
        )
        components.append(np.asarray(batch.amplitudes))
    component_array = np.stack(components, axis=1)
    weighted = component_array * A1_COEFFICIENTS[None, :]
    amplitudes = np.sum(weighted, axis=1)
    denominator = np.sum(np.abs(weighted), axis=1)
    cancellation = np.divide(
        np.abs(amplitudes),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0.0,
    )
    return InducedCarrierBatch(
        amplitudes=np.asarray(amplitudes),
        components=component_array,
        cancellation_ratio=np.asarray(cancellation),
    )


def projected_pi_zero_flux_a1_amplitudes_jax(
    configurations,
    *,
    orbit: PiZeroFluxOrbit | None = None,
):
    """Return the induced A1 carrier as a traceable JAX array.

    Unlike :func:`projected_pi_zero_flux_a1_amplitudes`, this narrow evaluator
    deliberately returns only the amplitudes and performs no host conversion.
    It is therefore suitable as an immutable carrier inside a differentiable
    correlator or Hamiltonian action.  The three pairing kernels remain fixed;
    no parton parameter is exposed to the optimiser.
    """

    states = jnp.asarray(configurations, dtype=jnp.int8)
    if states.ndim != 2 or states.shape[1] != 36:
        raise ValueError("configurations must have shape (batch, 36)")
    if orbit is None:
        orbit = build_n36_pi_zero_flux_orbit()
    config = ProjectedSingletAGPConfig(
        n_sites=36,
        group_images=_group_tuple(_identity_group(36)),
    )
    components = []
    for pairing in orbit.pairings:
        components.append(
            projected_singlet_agp_amplitudes(
                {"pairing_real": jnp.asarray(pairing, dtype=jnp.float64)},
                states,
                config,
            ).amplitudes
        )
    return jnp.stack(components, axis=1) @ jnp.asarray(
        A1_COEFFICIENTS, dtype=jnp.float64
    )


def full_reynolds_pi_zero_flux_reference(
    configurations: np.ndarray,
    *,
    boundary_index: int = 0,
    orbit: PiZeroFluxOrbit | None = None,
    microbatch_size: int = 64,
) -> ProjectedAGPBatch:
    """Evaluate the 432-image Reynolds sum, intended only as a test oracle."""

    if boundary_index not in range(3):
        raise ValueError("boundary_index must be 0, 1, or 2")
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[1] != 36:
        raise ValueError("configurations must have shape (batch, 36)")
    if orbit is None:
        orbit = build_n36_pi_zero_flux_orbit()
    group = _triangular_gamma_a1_group(6)
    config = ProjectedSingletAGPConfig(
        n_sites=36, group_images=_group_tuple(group)
    )
    return projected_singlet_agp_amplitudes(
        {"pairing_real": orbit.pairings[boundary_index]},
        states,
        config,
        microbatch_size=microbatch_size,
    )


def _infer_linear_size(configurations: np.ndarray) -> int:
    n_sites = int(configurations.shape[1])
    linear_size = isqrt(n_sites)
    if linear_size * linear_size != n_sites:
        raise ValueError("configurations must describe a square torus")
    return linear_size


def collective_tower_base_amplitudes(
    configurations: np.ndarray,
    *,
    family: Literal["120", "stripe"],
) -> np.ndarray:
    """Evaluate exact collective-spin singlet tower seeds.

    ``120`` couples three maximally symmetric sublattices through the unique
    ``(S_A,S_B,S_C)=(n/6,n/6,n/6)`` singlet.  ``stripe`` couples the two
    maximally symmetric stripe sublattices to a singlet.  These are analytic
    reference states, not finite-size phase labels.
    """

    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2:
        raise ValueError("configurations must be rank two")
    linear_size = _infer_linear_size(states)
    if family == "120":
        if linear_size % 3:
            raise ValueError("120 tower requires linear_size divisible by 3")
        labels = np.asarray(
            [(x + y) % 3 for y in range(linear_size) for x in range(linear_size)]
        )
        counts = [int(np.sum(labels == label)) for label in range(3)]
        if len(set(counts)) != 1:
            raise ValueError("120 sublattices must have equal size")
        q_collective = counts[0]
        output = np.zeros(states.shape[0], dtype=np.float64)
        for row, state in enumerate(states):
            n_up = [int(np.sum(state[labels == label])) for label in range(3)]
            m = [2 * value - q_collective for value in n_up]
            if sum(m) != 0:
                continue
            coefficient_ab = clebsch_gordan(
                q_collective,
                m[0],
                q_collective,
                m[1],
                q_collective,
                m[0] + m[1],
            )
            coefficient_c = clebsch_gordan(
                q_collective,
                m[0] + m[1],
                q_collective,
                m[2],
                0,
                0,
            )
            normalization = sqrt(
                comb(q_collective, n_up[0])
                * comb(q_collective, n_up[1])
                * comb(q_collective, n_up[2])
            )
            output[row] = coefficient_ab * coefficient_c / normalization
        return output
    if family == "stripe":
        labels = np.asarray(
            [x % 2 for y in range(linear_size) for x in range(linear_size)]
        )
        counts = [int(np.sum(labels == label)) for label in range(2)]
        if counts[0] != counts[1]:
            raise ValueError("stripe sublattices must have equal size")
        q_collective = counts[0]
        output = np.zeros(states.shape[0], dtype=np.float64)
        for row, state in enumerate(states):
            n_up = [int(np.sum(state[labels == label])) for label in range(2)]
            m = [2 * value - q_collective for value in n_up]
            if sum(m) != 0:
                continue
            coefficient = clebsch_gordan(
                q_collective, m[0], q_collective, m[1], 0, 0
            )
            normalization = sqrt(
                comb(q_collective, n_up[0]) * comb(q_collective, n_up[1])
            )
            output[row] = coefficient / normalization
        return output
    raise ValueError("family must be '120' or 'stripe'")


def compressed_collective_tower_a1(
    configurations: np.ndarray,
    *,
    family: Literal["120", "stripe"],
) -> CompressedCollectiveTowerBatch:
    """Evaluate a collective A1 tower using only its distinct spatial images.

    On an even square triangular torus commensurate with the three-sublattice
    pattern, the collective ``120`` singlet is invariant under every physical
    space-group operation and needs one term.  A stripe singlet is invariant
    within one little group and has exactly the three ``M``-star orientations,
    represented here by ``C6**0``, ``C6**1`` and ``C6**2``.  Equal orbit
    multiplicities make their arithmetic mean identical to the full Reynolds
    mean.  :func:`project_collective_tower_a1` remains the independent full
    group oracle.
    """

    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("configurations must be a nonempty rank-two array")
    linear_size = _infer_linear_size(states)
    if linear_size != 6:
        raise ValueError(
            "the compressed collective-tower evaluator is certified only "
            "for the 6x6 triangular torus"
        )
    if family == "120":
        component_array = collective_tower_base_amplitudes(
            states, family="120"
        )[:, None]
    elif family == "stripe":
        supercell = ((linear_size, 0), (0, linear_size))
        images = tuple(
            np.asarray(
                hnf_site_permutation(
                    supercell, operation=matrix_power(C6, power)
                ),
                dtype=np.int32,
            )
            for power in range(3)
        )
        component_array = np.stack(
            tuple(
                collective_tower_base_amplitudes(
                    states[:, image], family="stripe"
                )
                for image in images
            ),
            axis=1,
        )
    else:
        raise ValueError("family must be '120' or 'stripe'")

    summed = np.sum(component_array, axis=1)
    denominator = np.sum(np.abs(component_array), axis=1)
    cancellation = np.divide(
        np.abs(summed),
        denominator,
        out=np.zeros(states.shape[0], dtype=np.float64),
        where=denominator > 0.0,
    )
    return CompressedCollectiveTowerBatch(
        amplitudes=np.asarray(summed / component_array.shape[1]),
        components=np.asarray(component_array),
        cancellation_ratio=np.asarray(cancellation),
        semantic_term_count=int(component_array.shape[1]),
    )


def project_collective_tower_a1(
    configurations: np.ndarray,
    *,
    family: Literal["120", "stripe"],
) -> ProjectedScalarBatch:
    """Apply the full spatial Reynolds projector as a collective-tower oracle."""

    states = np.asarray(configurations, dtype=np.int8)
    linear_size = _infer_linear_size(states)
    group = _triangular_gamma_a1_group(linear_size)
    transformed = np.take(states, group, axis=1)
    flat = transformed.reshape((-1, states.shape[1]))
    terms = collective_tower_base_amplitudes(flat, family=family).reshape(
        (states.shape[0], group.shape[0])
    )
    amplitudes = np.mean(terms, axis=1)
    denominator = np.sum(np.abs(terms), axis=1)
    cancellation = np.divide(
        np.abs(np.sum(terms, axis=1)),
        denominator,
        out=np.zeros(states.shape[0], dtype=np.float64),
        where=denominator > 0.0,
    )
    return ProjectedScalarBatch(
        amplitudes=np.asarray(amplitudes),
        cancellation_ratio=np.asarray(cancellation),
    )
