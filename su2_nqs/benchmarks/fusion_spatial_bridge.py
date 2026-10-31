"""Exact-small bridge between computational and left-fusion singlet bases.

This module is deliberately exhaustive.  It provides an independent oracle
for converting a fixed-``M`` computational vector to the multiplicity basis,
and for comparing the physical Reynolds projector with the audited
common-basis fusion-space projector.  It is useful through ``N <= 16`` and is
not a production ``N=36`` transform or sampler.

All spin labels are doubled integers.  Computational bits follow the
repository convention: bit zero is ``m=-1/2``, bit one is ``m=+1/2``, and
site zero is the most-significant bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from numbers import Integral
from typing import Sequence

import numpy as np

from ..lattices.triangular_hnf import IntMatrix, hnf_sites, validate_hnf
from ..su2.clebsch_gordan import clebsch_gordan, spin_half_cg
from ..su2.fusion_path import FusionPath, count_paths, enumerate_paths
from ..su2.quantum_numbers import validate_m, validate_q
from ..su2.sz_basis import enumerate_sz_configs
from .common_basis_gamma_a1 import FixedBasisGammaA1Circuit
from .projected_singlet_agp import ProjectedSingletAGP
from .triangular_induced_parton import (
    closed_shell_pairing_from_hopping,
    triangular_pi_zero_flux_hopping,
)


SpinConfiguration = tuple[int, ...]
SitePermutation = tuple[int, ...]


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _validated_vector(values: Sequence[complex], dimension: int, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.complex128)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},)")
    if np.any(~np.isfinite(vector.real)) or np.any(~np.isfinite(vector.imag)):
        raise ValueError(f"{name} must be finite")
    return vector


def _configuration_integer(configuration: Sequence[int]) -> int:
    value = 0
    for bit in configuration:
        value = (value << 1) | int(bit)
    return value


@dataclass(frozen=True)
class FixedMLeftFusionTransform:
    """Dense exact-small isometry from a left-fusion basis to fixed ``M``.

    ``matrix[configuration, path] = <configuration|path>``.  The matrix is
    real in the repository's Condon--Shortley convention, but conversion
    methods accept complex amplitudes.  Construction is protected by an
    explicit entry cap so callers cannot accidentally request ``N=36``.
    """

    n_sites: int
    target_q: int
    total_m: int
    configurations: tuple[SpinConfiguration, ...]
    paths: tuple[FusionPath, ...]
    matrix: np.ndarray

    @classmethod
    def build(
        cls,
        n_sites: int,
        *,
        target_q: int = 0,
        total_m: int = 0,
        maximum_entries: int = 25_000_000,
    ) -> "FixedMLeftFusionTransform":
        """Construct the exact isometry without allocating ``2**N`` rows."""

        if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
            raise TypeError("n_sites must be an integer")
        n_sites = int(n_sites)
        if n_sites < 1:
            raise ValueError("n_sites must be positive")
        validate_q(target_q, name="target_q")
        validate_m(target_q, total_m, name="total_m")
        target_q, total_m = int(target_q), int(total_m)
        if (n_sites + total_m) % 2:
            raise ValueError("total_m parity is incompatible with n_sites")
        n_up = (n_sites + total_m) // 2
        if not 0 <= n_up <= n_sites:
            raise ValueError("total_m is outside the physical magnetization range")
        if (
            isinstance(maximum_entries, bool)
            or not isinstance(maximum_entries, Integral)
            or int(maximum_entries) < 1
        ):
            raise ValueError("maximum_entries must be a positive integer")

        configurations = enumerate_sz_configs(n_sites, n_up)
        paths = enumerate_paths(n_sites, target_q)
        entry_count = len(configurations) * len(paths)
        if entry_count > int(maximum_entries):
            raise ValueError(
                "exact fixed-M/fusion transform exceeds the entry cap: "
                f"{entry_count} > {int(maximum_entries)}"
            )

        states = np.asarray(configurations, dtype=np.int16)
        path_array = np.asarray(paths, dtype=np.int16)
        prefix_m = np.cumsum(2 * states - 1, axis=1, dtype=np.int16)

        # Offline-sized numeric CG lookup: no symbolic work appears in the
        # vectorized transform construction.
        table = np.zeros(
            (n_sites + 1, 2 * n_sites + 1, 2, n_sites + 1), dtype=np.float64
        )
        for q_left in range(n_sites + 1):
            for m_left in range(-q_left, q_left + 1, 2):
                for bit, m_phys in ((0, -1), (1, 1)):
                    for q_total in (q_left - 1, q_left + 1):
                        if 0 <= q_total <= n_sites:
                            table[q_left, m_left + n_sites, bit, q_total] = (
                                spin_half_cg(q_left, m_left, m_phys, q_total)
                            )

        matrix = np.ones((len(configurations), len(paths)), dtype=np.float64)
        for site in range(1, n_sites):
            q_left = path_array[:, site - 1]
            q_total = path_array[:, site]
            m_left = prefix_m[:, site - 1]
            bit = states[:, site]
            matrix *= table[
                q_left[None, :],
                m_left[:, None] + n_sites,
                bit[:, None],
                q_total[None, :],
            ]
        matrix.setflags(write=False)
        return cls(
            n_sites=n_sites,
            target_q=target_q,
            total_m=total_m,
            configurations=configurations,
            paths=paths,
            matrix=matrix,
        )

    @property
    def fixed_m_dimension(self) -> int:
        return len(self.configurations)

    @property
    def fusion_dimension(self) -> int:
        return len(self.paths)

    @property
    def entry_count(self) -> int:
        return int(self.matrix.size)

    @property
    def storage_bytes(self) -> int:
        return int(self.matrix.nbytes)

    def orthonormality_error(self) -> float:
        gram = self.matrix.T @ self.matrix
        return float(np.max(np.abs(gram - np.eye(self.fusion_dimension)), initial=0.0))

    def fusion_to_computational(self, amplitudes: Sequence[complex]) -> np.ndarray:
        vector = _validated_vector(amplitudes, self.fusion_dimension, name="amplitudes")
        return np.asarray(self.matrix @ vector, dtype=np.complex128)

    def computational_to_fusion(self, amplitudes: Sequence[complex]) -> np.ndarray:
        vector = _validated_vector(
            amplitudes, self.fixed_m_dimension, name="amplitudes"
        )
        return np.asarray(self.matrix.T @ vector, dtype=np.complex128)

    def q_sector_reconstruction(self, amplitudes: Sequence[complex]) -> np.ndarray:
        """Return the orthogonal projection of a fixed-M vector into ``target_q``."""

        return self.fusion_to_computational(self.computational_to_fusion(amplitudes))


def computational_reynolds_project(
    amplitudes: Sequence[complex],
    configurations: Sequence[Sequence[int]],
    group_images: Sequence[Sequence[int]],
) -> np.ndarray:
    """Apply a physical trivial-character projector in a fixed-M basis."""

    states = tuple(tuple(int(bit) for bit in state) for state in configurations)
    if not states:
        raise ValueError("configurations must be nonempty")
    n_sites = len(states[0])
    if any(
        len(state) != n_sites
        or any(bit not in (0, 1) for bit in state)
        for state in states
    ):
        raise ValueError("configurations must be equal-length binary sequences")
    vector = _validated_vector(amplitudes, len(states), name="amplitudes")
    images = tuple(tuple(int(site) for site in image) for image in group_images)
    if not images:
        raise ValueError("group_images must be nonempty")
    expected = tuple(range(n_sites))
    if any(len(image) != n_sites or tuple(sorted(image)) != expected for image in images):
        raise ValueError("every group image must be a site permutation")

    positions = {_configuration_integer(state): index for index, state in enumerate(states)}
    if len(positions) != len(states):
        raise ValueError("configurations must not contain duplicates")
    state_array = np.asarray(states, dtype=np.int8)
    bit_weights = np.asarray(
        [1 << (n_sites - site - 1) for site in range(n_sites)], dtype=np.int64
    )
    result = np.zeros_like(vector)
    for image in images:
        source_integers = np.asarray(state_array[:, image] @ bit_weights, dtype=np.int64)
        try:
            source_rows = np.fromiter(
                (positions[int(value)] for value in source_integers),
                dtype=np.int64,
                count=len(states),
            )
        except KeyError as error:
            raise ValueError("group action leaves the supplied fixed-M basis") from error
        result += vector[source_rows]
    return np.asarray(result / len(images), dtype=np.complex128)


def spin_raising_relative_norm(
    amplitudes: Sequence[complex], configurations: Sequence[Sequence[int]]
) -> float:
    """Return ``||S+ psi|| / ||psi||`` from a complete fixed-M vector."""

    states = tuple(tuple(int(bit) for bit in state) for state in configurations)
    if not states:
        raise ValueError("configurations must be nonempty")
    n_sites = len(states[0])
    if any(len(state) != n_sites or any(bit not in (0, 1) for bit in state) for state in states):
        raise ValueError("configurations must be equal-length binary sequences")
    n_up = sum(states[0])
    if any(sum(state) != n_up for state in states):
        raise ValueError("configurations must share one fixed magnetization")
    expected_size = comb(n_sites, n_up)
    if len(states) != expected_size or len(set(states)) != expected_size:
        raise ValueError("configurations must exhaust the fixed-M basis")
    vector = _validated_vector(amplitudes, len(states), name="amplitudes")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("amplitudes have zero norm")
    if n_up == n_sites:
        return 0.0

    raised_states = enumerate_sz_configs(n_sites, n_up + 1)
    positions = {state: index for index, state in enumerate(raised_states)}
    raised = np.zeros(len(raised_states), dtype=np.complex128)
    for state, amplitude in zip(states, vector, strict=True):
        if amplitude == 0.0:
            continue
        for site, bit in enumerate(state):
            if bit == 0:
                target = state[:site] + (1,) + state[site + 1 :]
                raised[positions[target]] += amplitude
    return float(np.linalg.norm(raised) / norm)


def certify_fusion_gamma_a1(
    amplitudes: Sequence[complex],
    transform: FixedMLeftFusionTransform,
    circuit: FixedBasisGammaA1Circuit,
    group_images: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Certify ``q`` and ``Gamma.A1`` for one arbitrary fusion vector."""

    if (
        circuit.n_sites != transform.n_sites
        or circuit.target_q != transform.target_q
        or circuit.dimension != transform.fusion_dimension
    ):
        raise ValueError("circuit and fixed-M transform sectors disagree")
    fusion = _validated_vector(amplitudes, transform.fusion_dimension, name="amplitudes")
    norm = float(np.linalg.norm(fusion))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("amplitudes have zero norm")
    projected_fusion = np.asarray(circuit.apply(fusion), dtype=np.complex128)
    computational = transform.fusion_to_computational(fusion)
    projected_computational = computational_reynolds_project(
        computational, transform.configurations, group_images
    )
    intertwined = transform.fusion_to_computational(projected_fusion)
    return {
        "input_fusion_norm": norm,
        "input_q_sector_reconstruction_relative_error": _relative_error(
            computational, transform.q_sector_reconstruction(computational)
        ),
        "input_Splus_relative_norm": spin_raising_relative_norm(
            computational, transform.configurations
        ),
        "input_Gamma_A1_relative_residual": _relative_error(fusion, projected_fusion),
        "projector_intertwining_relative_error": _relative_error(
            projected_computational, intertwined
        ),
        "projected_Gamma_A1_relative_residual": _relative_error(
            projected_fusion, np.asarray(circuit.apply(projected_fusion))
        ),
        "projected_Splus_relative_norm": spin_raising_relative_norm(
            intertwined, transform.configurations
        ),
    }


def _collective_singlet_amplitudes(
    configurations: Sequence[Sequence[int]], labels: Sequence[int]
) -> np.ndarray:
    states = np.asarray(configurations, dtype=np.int8)
    partition = np.asarray(labels, dtype=np.int16)
    if states.ndim != 2 or partition.shape != (states.shape[1],):
        raise ValueError("labels must assign one sublattice to every site")
    unique = tuple(int(value) for value in np.unique(partition))
    if unique not in ((0, 1), (0, 1, 2)):
        raise ValueError("collective singlet requires two or three labelled sublattices")
    sizes = tuple(int(np.count_nonzero(partition == label)) for label in unique)
    if len(set(sizes)) != 1:
        raise ValueError("collective sublattices must have equal size")
    q_collective = sizes[0]
    result = np.zeros(states.shape[0], dtype=np.float64)
    for row, state in enumerate(states):
        n_up = tuple(int(np.sum(state[partition == label])) for label in unique)
        m = tuple(2 * value - q_collective for value in n_up)
        normalization = sqrt(
            float(np.prod([comb(q_collective, value) for value in n_up]))
        )
        if len(unique) == 2:
            result[row] = clebsch_gordan(
                q_collective, m[0], q_collective, m[1], 0, 0
            ) / normalization
        elif sum(m) == 0:
            result[row] = (
                clebsch_gordan(
                    q_collective,
                    m[0],
                    q_collective,
                    m[1],
                    q_collective,
                    m[0] + m[1],
                )
                * clebsch_gordan(
                    q_collective,
                    m[0] + m[1],
                    q_collective,
                    m[2],
                    0,
                    0,
                )
                / normalization
            )
    return result


def triangular_hnf_120_teacher(
    configurations: Sequence[Sequence[int]], supercell: IntMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and compact-A1 three-sublattice collective amplitudes."""

    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    states = tuple(tuple(int(bit) for bit in row) for row in configurations)
    if not states or len(sites) != len(states[0]):
        raise ValueError("supercell volume and configuration size disagree")
    labels = tuple((x + y) % 3 for x, y in sites)
    raw = _collective_singlet_amplitudes(states, labels)
    return raw, np.array(raw, copy=True)


def triangular_square_stripe_teacher(
    configurations: Sequence[Sequence[int]], linear_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one stripe orientation, the three-star mean, and components."""

    if (
        isinstance(linear_size, bool)
        or not isinstance(linear_size, Integral)
        or int(linear_size) < 2
        or int(linear_size) % 2
    ):
        raise ValueError("linear_size must be a positive even integer")
    linear_size = int(linear_size)
    states = tuple(tuple(int(bit) for bit in row) for row in configurations)
    if not states or any(len(row) != linear_size**2 for row in states):
        raise ValueError("configurations do not describe the requested square torus")
    coordinates = tuple(
        (x, y) for y in range(linear_size) for x in range(linear_size)
    )
    label_sets = (
        tuple(x % 2 for x, _ in coordinates),
        tuple(y % 2 for _, y in coordinates),
        tuple((x + y) % 2 for x, y in coordinates),
    )
    components = np.column_stack(
        [_collective_singlet_amplitudes(states, labels) for labels in label_sets]
    )
    return components[:, 0], np.mean(components, axis=1), components


def _pairing_matrix_from_hopping(
    hopping: np.ndarray, *, gap_tolerance: float = 1.0e-10
) -> tuple[np.ndarray, float]:
    packed, gap = closed_shell_pairing_from_hopping(
        hopping, gap_tolerance=gap_tolerance, require_closed_shell=True
    )
    n_sites = hopping.shape[0]
    pairing = np.zeros((n_sites, n_sites), dtype=np.float64)
    upper = np.triu_indices(n_sites, 1)
    pairing[upper] = packed
    pairing[(upper[1], upper[0])] = packed
    return pairing, float(gap)


def triangular_square_pi_zero_dsl_teacher(
    configurations: Sequence[Sequence[int]], linear_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the registered compact pi/0-flux DSL teacher on 4x4 or 6x6.

    For 4x4 the closed-shell orbit is ``AP, PA, AA`` with equal coefficients;
    for 6x6 it is ``PP, AP, PA`` with coefficients ``(+,-,-)/3``.  This helper
    only evaluates computational amplitudes; full Reynolds projection remains
    the independent spatial-symmetry oracle.
    """

    if linear_size not in (4, 6):
        raise ValueError("pi/0 DSL compact orbit is registered only for L=4 or L=6")
    states = tuple(tuple(int(bit) for bit in row) for row in configurations)
    if not states or any(len(row) != linear_size**2 for row in states):
        raise ValueError("configurations do not describe the requested square torus")
    if linear_size == 4:
        boundaries = (("A", "P"), ("P", "A"), ("A", "A"))
        coefficients = np.asarray((1.0, 1.0, 1.0), dtype=np.float64) / 3.0
    else:
        boundaries = (("P", "P"), ("A", "P"), ("P", "A"))
        coefficients = np.asarray((1.0, -1.0, -1.0), dtype=np.float64) / 3.0
    identity = (tuple(range(linear_size**2)),)
    components: list[np.ndarray] = []
    gaps: list[float] = []
    for boundary_x, boundary_y in boundaries:
        hopping = triangular_pi_zero_flux_hopping(
            linear_size, boundary_x=boundary_x, boundary_y=boundary_y
        )
        pairing, gap = _pairing_matrix_from_hopping(hopping)
        carrier = ProjectedSingletAGP(pairing=pairing, group_images=identity)
        components.append(
            np.asarray([carrier.base_amplitude(state) for state in states])
        )
        gaps.append(gap)
    component_matrix = np.column_stack(components)
    compact = component_matrix @ coefficients
    return component_matrix[:, 0], compact, component_matrix, np.asarray(gaps)


__all__ = [
    "FixedMLeftFusionTransform",
    "certify_fusion_gamma_a1",
    "computational_reynolds_project",
    "spin_raising_relative_norm",
    "triangular_hnf_120_teacher",
    "triangular_square_pi_zero_dsl_teacher",
    "triangular_square_stripe_teacher",
]
