"""Physics-labelled deformations of an exact-singlet projected-pair carrier.

The carrier is a Gutzwiller-projected singlet AGP.  A full Reynolds average
places every member of the nonlinear family in ``Gamma.A1``.  The deformation
coordinates change the pair kernel *before* the determinant is evaluated and
can therefore change nodes and signs; they are not a linear operator head on a
fixed state.

For the triangular lattice the registered physical coordinates are an
isotropic nearest-neighbour pair field, an isotropic next-nearest-neighbour
pair field, and one real ``E2`` nearest-neighbour component.  The final state
is still ``A1``.  In particular, the ``E2`` coordinate enters the projected
state through nonlinear even-order resonance processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .clebsch_tree_exact import physical_permutation_index_table
from .physical_interpretation import hnf_oriented_bond_families
from .triangular_induced_parton import (
    closed_shell_pairing_from_hopping,
    triangular_pi_zero_flux_hopping,
)
from .xdiag_physical_seed_krylov import PhysicalSeedCase, ordered_gamma_a1_group
from ..lattices.triangular_hnf import J1_POSITIVE, J2_POSITIVE


def _checked_configurations(
    configurations: np.ndarray | Sequence[Sequence[int]],
) -> np.ndarray:
    states = np.asarray(configurations)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] % 2:
        raise ValueError("configurations must have shape (nonzero_batch, even_n)")
    if states.dtype == np.bool_ or np.any((states != 0) & (states != 1)):
        raise ValueError("configurations must contain integer binary labels")
    if np.any(np.sum(states, axis=1) != states.shape[1] // 2):
        raise ValueError("every configuration must be half filled")
    return np.asarray(states, dtype=np.int8)


def batched_singlet_agp_amplitudes(
    configurations: np.ndarray | Sequence[Sequence[int]],
    pairing: np.ndarray | Sequence[Sequence[complex]],
) -> np.ndarray:
    r"""Evaluate a singlet AGP on a fixed-``Sz=0`` batch using batched dets.

    The sign is
    ``(-1)^(I(sigma)+n(n-1)/2) det(F[up,down])`` and matches the repository's
    site-major fermionic convention.
    """

    states = _checked_configurations(configurations)
    matrix = np.asarray(pairing)
    n_sites = states.shape[1]
    if matrix.shape != (n_sites, n_sites):
        raise ValueError("pairing must have shape (n_sites, n_sites)")
    if np.any(~np.isfinite(matrix.real)) or np.any(~np.isfinite(matrix.imag)):
        raise ValueError("pairing must be finite")
    scale = max(float(np.max(np.abs(matrix))), np.finfo(np.float64).tiny)
    if float(np.max(np.abs(matrix - matrix.T))) > 1.0e-12 * scale:
        raise ValueError("spin-singlet pairing must be symmetric")
    if float(np.max(np.abs(np.diag(matrix)))) > 1.0e-12 * scale:
        raise ValueError("onsite pairing must vanish")

    pair_count = n_sites // 2
    site_order = np.argsort(states, axis=1, kind="stable")
    down = site_order[:, :pair_count]
    up = site_order[:, pair_count:]
    blocks = matrix[up[:, :, None], down[:, None, :]]
    left_down = np.cumsum(1 - states, axis=1) - (1 - states)
    inversions = np.sum(states * left_down, axis=1)
    parity = (inversions + pair_count * (pair_count - 1) // 2) % 2
    signs = np.where(parity == 0, 1.0, -1.0)
    return np.asarray(signs * np.linalg.det(blocks))


def _pairing_from_packed(packed: np.ndarray, n_sites: int) -> np.ndarray:
    upper = np.triu_indices(n_sites, 1)
    if np.asarray(packed).shape != (len(upper[0]),):
        raise ValueError("packed pairing has the wrong size")
    pairing = np.zeros((n_sites, n_sites), dtype=np.asarray(packed).dtype)
    pairing[upper] = packed
    pairing[(upper[1], upper[0])] = packed
    return pairing


def _adjacency_from_families(
    n_sites: int,
    families: Sequence[Sequence[tuple[int, int]]],
    weights: Sequence[float],
) -> np.ndarray:
    if len(families) != len(weights):
        raise ValueError("one weight is required per oriented family")
    matrix = np.zeros((n_sites, n_sites), dtype=np.float64)
    for weight, bonds in zip(weights, families, strict=True):
        for first, second in bonds:
            matrix[int(first), int(second)] += float(weight)
            matrix[int(second), int(first)] += float(weight)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _orthogonal_deformations(
    base: np.ndarray, candidates: Sequence[np.ndarray]
) -> tuple[np.ndarray, ...]:
    """Remove the homogeneous scale gauge and equalize Frobenius scales."""

    base_norm = float(np.linalg.norm(base))
    if not np.isfinite(base_norm) or base_norm <= 1.0e-14:
        raise ValueError("base pairing must have nonzero finite norm")
    directions: list[np.ndarray] = []
    anchors = [np.asarray(base, dtype=np.float64) / base_norm]
    for raw in candidates:
        direction = np.asarray(raw, dtype=np.float64).copy()
        for anchor in (*anchors, *[value / base_norm for value in directions]):
            direction -= float(np.vdot(anchor, direction).real) * anchor
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError("pairing deformation bank is linearly dependent")
        directions.append(direction * (base_norm / norm))
    return tuple(directions)


@dataclass(frozen=True)
class AdaptiveProjectedPairCarrier:
    """A low-dimensional nonlinear AGP family with exact ``S=0, Gamma.A1``."""

    configurations: np.ndarray
    permutation_indices: np.ndarray
    base_pairing: np.ndarray
    deformation_pairings: tuple[np.ndarray, ...]
    deformation_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        states = _checked_configurations(self.configurations)
        indices = np.asarray(self.permutation_indices)
        if indices.ndim != 2 or indices.shape[1] != len(states):
            raise ValueError("permutation_indices must have shape (group,batch)")
        if np.any(indices < 0) or np.any(indices >= len(states)):
            raise ValueError("permutation index lies outside the configuration batch")
        n_sites = states.shape[1]
        base = np.asarray(self.base_pairing, dtype=np.float64)
        if base.shape != (n_sites, n_sites):
            raise ValueError("base_pairing has the wrong shape")
        deformations = tuple(
            np.asarray(value, dtype=np.float64)
            for value in self.deformation_pairings
        )
        if not deformations or any(value.shape != base.shape for value in deformations):
            raise ValueError("deformation pairings must be nonempty and match the base")
        if len(deformations) != len(self.deformation_labels):
            raise ValueError("one label is required per deformation")
        object.__setattr__(self, "configurations", states)
        object.__setattr__(self, "permutation_indices", np.asarray(indices, dtype=np.int32))
        object.__setattr__(self, "base_pairing", base)
        object.__setattr__(self, "deformation_pairings", deformations)

    @property
    def parameter_count(self) -> int:
        return len(self.deformation_pairings)

    def pairing(self, parameters: Sequence[float] | np.ndarray) -> np.ndarray:
        coordinates = np.asarray(parameters, dtype=np.float64)
        if coordinates.shape != (self.parameter_count,):
            raise ValueError("parameters have the wrong shape")
        if np.any(~np.isfinite(coordinates)):
            raise ValueError("parameters must be finite")
        result = np.array(self.base_pairing, copy=True)
        for coordinate, direction in zip(
            coordinates, self.deformation_pairings, strict=True
        ):
            result += float(coordinate) * direction
        return result

    def amplitudes(self, parameters: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the exact Reynolds-projected carrier on the stored support."""

        base = batched_singlet_agp_amplitudes(
            self.configurations, self.pairing(parameters)
        )
        return np.mean(base[self.permutation_indices], axis=0)


def triangular_physical_adaptive_pair_carrier(
    configurations: np.ndarray | Sequence[Sequence[int]],
    case: PhysicalSeedCase,
    *,
    boundary_x: str = "A",
    boundary_y: str = "P",
) -> AdaptiveProjectedPairCarrier:
    """Build the three-coordinate physical carrier on a square triangular torus."""

    states = _checked_configurations(configurations)
    linear_size = int(round(case.n_sites**0.5))
    if linear_size * linear_size != case.n_sites or case.supercell != (
        (linear_size, 0),
        (0, linear_size),
    ):
        raise ValueError("physical adaptive carrier requires an LxL HNF torus")
    if states.shape[1] != case.n_sites:
        raise ValueError("configurations and case have different sizes")
    base, directions, labels = triangular_physical_pairing_bank(
        linear_size,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )
    group = ordered_gamma_a1_group(case)
    basis = tuple(tuple(int(bit) for bit in row) for row in states)
    return AdaptiveProjectedPairCarrier(
        configurations=states,
        permutation_indices=physical_permutation_index_table(basis, group),
        base_pairing=base,
        deformation_pairings=directions,
        deformation_labels=labels,
    )


def triangular_physical_pairing_bank(
    linear_size: int,
    *,
    boundary_x: str = "A",
    boundary_y: str = "P",
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[str, ...]]:
    """Return size-transferable pair coordinates without a Hilbert support.

    These are intrinsic pair kernels.  A large-system caller must still apply
    the exact lattice projector: nonlinear deformations are not assumed to
    stay inside the undeformed three-boundary-sector induced orbit.
    """

    if (
        isinstance(linear_size, bool)
        or not isinstance(linear_size, Integral)
        or int(linear_size) < 2
        or int(linear_size) % 2
    ):
        raise ValueError("linear_size must be a positive even integer")
    linear_size = int(linear_size)
    supercell = ((linear_size, 0), (0, linear_size))
    n_sites = linear_size * linear_size
    hopping = triangular_pi_zero_flux_hopping(
        linear_size,
        boundary_x=boundary_x,  # type: ignore[arg-type]
        boundary_y=boundary_y,  # type: ignore[arg-type]
    )
    packed, _gap = closed_shell_pairing_from_hopping(hopping)
    base = _pairing_from_packed(packed, n_sites)
    j1 = hnf_oriented_bond_families(supercell, J1_POSITIVE)
    j2 = hnf_oriented_bond_families(supercell, J2_POSITIVE)
    raw = (
        _adjacency_from_families(n_sites, j1, (1.0, 1.0, 1.0)),
        _adjacency_from_families(n_sites, j2, (1.0, 1.0, 1.0)),
        _adjacency_from_families(n_sites, j1, (2.0, -1.0, -1.0)),
    )
    return (
        base,
        _orthogonal_deformations(base, raw),
        ("A1_J1_pair", "A1_J2_pair", "E2_J1_pair"),
    )


def generic_adaptive_pair_control(
    physical: AdaptiveProjectedPairCarrier,
    *,
    seed: int,
) -> AdaptiveProjectedPairCarrier:
    """Return a Hamiltonian-blind parameter-matched random-pair control."""

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    n_sites = physical.configurations.shape[1]
    candidates = []
    for _ in range(physical.parameter_count):
        raw = rng.normal(size=(n_sites, n_sites))
        raw = 0.5 * (raw + raw.T)
        np.fill_diagonal(raw, 0.0)
        candidates.append(raw)
    return AdaptiveProjectedPairCarrier(
        configurations=physical.configurations,
        permutation_indices=physical.permutation_indices,
        base_pairing=physical.base_pairing,
        deformation_pairings=_orthogonal_deformations(
            physical.base_pairing, candidates
        ),
        deformation_labels=tuple(
            f"generic_pair_{index}" for index in range(physical.parameter_count)
        ),
    )


__all__ = [
    "AdaptiveProjectedPairCarrier",
    "batched_singlet_agp_amplitudes",
    "generic_adaptive_pair_control",
    "triangular_physical_adaptive_pair_carrier",
    "triangular_physical_pairing_bank",
]
