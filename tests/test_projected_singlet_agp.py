from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import (
    computational_basis,
    heisenberg_sz_matrix,
    spin_squared_matrix,
)
from su2_nqs.benchmarks.common_basis_gamma_a1 import (
    triangular_hnf_gamma_a1_circuit,
)
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.projected_singlet_agp import (
    ProjectedSingletAGP,
    ProjectedSingletAGPCombination,
    exact_small_heisenberg_reference,
    heisenberg_hpsi_amplitude,
    pullback_spin_configuration,
    reynolds_projected_amplitude,
    singlet_agp_determinant_amplitude,
    singlet_agp_pfaffian_reference,
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.square_symmetry import square_site_generators
from su2_nqs.lattices.triangular_hnf import (
    C6,
    MX,
    hnf_site_permutation,
    triangular_hnf_j1j2_bonds,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.su2.fusion_path import enumerate_paths


def _random_symmetric_pairing(n_sites: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_sites, n_sites)) + 1.0j * rng.normal(
        size=(n_sites, n_sites)
    )
    pairing = (raw + raw.T) / (2.0 * np.sqrt(n_sites))
    np.fill_diagonal(pairing, 0.0)
    return pairing


def _integer_bits(state: int, n_sites: int) -> tuple[int, ...]:
    """Independent MSB-first conversion for dense-reference bases."""

    return tuple(
        (int(state) >> (n_sites - site - 1)) & 1
        for site in range(n_sites)
    )


def _relative_norm(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
    )


def test_two_site_condon_shortley_and_msb_bit_convention() -> None:
    pairing = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128)

    # bit 1 is up, bit 0 is down.  A positive symmetric pair therefore gives
    # the Condon-Shortley singlet |up,down> - |down,up>.
    assert singlet_agp_determinant_amplitude((1, 0), pairing) == 1.0
    assert singlet_agp_determinant_amplitude((0, 1), pairing) == -1.0

    # Integer computational order is |down,up>, |up,down> because site zero
    # is the most-significant bit.
    integer_basis = computational_basis(2, n_up=1)
    assert integer_basis == (1, 2)
    configurations = tuple(_integer_bits(state, 2) for state in integer_basis)
    assert configurations == ((0, 1), (1, 0))
    np.testing.assert_array_equal(
        [singlet_agp_determinant_amplitude(config, pairing) for config in configurations],
        [-1.0, 1.0],
    )


@pytest.mark.parametrize("n_sites", (2, 4, 6, 8))
def test_determinant_sign_matches_independent_site_major_pfaffian(
    n_sites: int,
) -> None:
    pairing = _random_symmetric_pairing(n_sites, 2026091310 + n_sites)
    for configuration in enumerate_sz_configs(n_sites, n_sites // 2):
        determinant = singlet_agp_determinant_amplitude(configuration, pairing)
        pfaffian = singlet_agp_pfaffian_reference(configuration, pairing)
        np.testing.assert_allclose(determinant, pfaffian, atol=2.0e-12)


def test_arbitrary_complex_symmetric_agp_is_annihilated_by_total_spin_squared() -> None:
    n_sites = 8
    pairing = _random_symmetric_pairing(n_sites, 2026091318)
    spin_squared, integer_basis = spin_squared_matrix(
        n_sites, n_up=n_sites // 2
    )
    state = np.asarray(
        [
            singlet_agp_determinant_amplitude(
                _integer_bits(integer_state, n_sites), pairing
            )
            for integer_state in integer_basis
        ],
        dtype=np.complex128,
    )
    assert np.linalg.norm(state) > 1.0e-8
    assert np.linalg.norm(spin_squared @ state) / np.linalg.norm(state) < 2.0e-13


def test_active_pullback_matches_an_independent_computational_permutation() -> None:
    n_sites = 4
    image = (1, 3, 0, 2)
    rng = np.random.default_rng(2026091319)
    base = rng.normal(size=1 << n_sites) + 1.0j * rng.normal(size=1 << n_sites)

    # Build U[new,old] independently by moving every old-site bit to its
    # active image.  This catches using image as a target-to-source map.
    matrix = np.zeros((1 << n_sites, 1 << n_sites), dtype=np.float64)
    for old_state in range(1 << n_sites):
        new_state = 0
        for old_site, new_site in enumerate(image):
            bit = (old_state >> (n_sites - old_site - 1)) & 1
            new_state |= bit << (n_sites - new_site - 1)
        matrix[new_state, old_state] = 1.0

    transformed = matrix @ base
    for target_state in range(1 << n_sites):
        target = _integer_bits(target_state, n_sites)
        source = pullback_spin_configuration(target, image)
        source_state = sum(
            bit << (n_sites - site - 1) for site, bit in enumerate(source)
        )
        np.testing.assert_allclose(transformed[target_state], base[source_state])


def test_reynolds_projector_is_hermitian_and_idempotent_on_spin_amplitudes() -> None:
    n_sites = 4
    group = square_gamma_a1_group(2)
    basis = enumerate_sz_configs(n_sites, n_sites // 2)
    positions = {configuration: index for index, configuration in enumerate(basis)}
    projector = np.zeros((len(basis), len(basis)), dtype=np.float64)
    for row, output in enumerate(basis):
        for image in group:
            source = tuple(output[image[old]] for old in range(n_sites))
            projector[row, positions[source]] += 1.0 / len(group)

    np.testing.assert_allclose(projector, projector.T, atol=2.0e-15)
    np.testing.assert_allclose(projector @ projector, projector, atol=2.0e-15)

    rng = np.random.default_rng(2026091320)
    base_vector = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    base = {configuration: base_vector[index] for index, configuration in enumerate(basis)}
    once = {
        configuration: reynolds_projected_amplitude(
            configuration, base.__getitem__, group
        )
        for configuration in basis
    }
    twice = np.asarray(
        [
            reynolds_projected_amplitude(configuration, once.__getitem__, group)
            for configuration in basis
        ]
    )
    np.testing.assert_allclose(
        np.asarray(tuple(once.values())), projector @ base_vector, atol=3.0e-15
    )
    np.testing.assert_allclose(twice, np.asarray(tuple(once.values())), atol=3.0e-15)


@dataclass(frozen=True)
class _TriangularN12State:
    carrier: ProjectedSingletAGP
    configurations: tuple[tuple[int, ...], ...]
    projected: dict[tuple[int, ...], complex]
    base: dict[tuple[int, ...], complex]


@pytest.fixture(scope="module")
def triangular_n12_state() -> _TriangularN12State:
    supercell = ((6, 4), (0, 2))
    group = triangular_hnf_gamma_a1_group(supercell)
    assert len(group) == 144
    carrier = ProjectedSingletAGP(
        _random_symmetric_pairing(12, 2026091321), group
    )
    configurations = enumerate_sz_configs(12, 6)
    projected = {
        configuration: carrier.amplitude(configuration)
        for configuration in configurations
    }
    base = {
        configuration: carrier.base_amplitude(configuration)
        for configuration in configurations
    }
    return _TriangularN12State(carrier, configurations, projected, base)


def test_non_space_symmetric_pairing_projects_to_triangular_gamma_a1(
    triangular_n12_state: _TriangularN12State,
) -> None:
    state = triangular_n12_state
    supercell = ((6, 4), (0, 2))
    generators = {
        "tx": hnf_site_permutation(supercell, translation=(1, 0)),
        "ty": hnf_site_permutation(supercell, translation=(0, 1)),
        "c6": hnf_site_permutation(supercell, operation=C6),
        "mx": hnf_site_permutation(supercell, operation=MX),
    }
    projected = np.asarray([state.projected[config] for config in state.configurations])
    base = np.asarray([state.base[config] for config in state.configurations])
    assert np.linalg.norm(projected) > 1.0e-8

    base_residuals = []
    for image in generators.values():
        transformed_projected = np.asarray(
            [
                state.projected[pullback_spin_configuration(config, image)]
                for config in state.configurations
            ]
        )
        transformed_base = np.asarray(
            [
                state.base[pullback_spin_configuration(config, image)]
                for config in state.configurations
            ]
        )
        assert _relative_norm(transformed_projected, projected) < 2.0e-14
        base_residuals.append(_relative_norm(transformed_base, base))
    assert max(base_residuals) > 0.1

    # Applying the full physical group average a second time changes nothing.
    for configuration in state.configurations[::137]:
        twice = reynolds_projected_amplitude(
            configuration, state.projected.__getitem__, state.carrier.group_images
        )
        np.testing.assert_allclose(twice, state.projected[configuration], atol=2.0e-13)


def test_triangular_n12_projected_state_is_annihilated_by_spin_raising(
    triangular_n12_state: _TriangularN12State,
) -> None:
    state = triangular_n12_state
    target_norm_squared = 0.0
    for up_sites in combinations(range(12), 7):
        target_up = set(up_sites)
        coefficient = 0.0 + 0.0j
        for flipped_site in up_sites:
            source = tuple(
                int(site in target_up and site != flipped_site)
                for site in range(12)
            )
            coefficient += state.projected[source]
        target_norm_squared += abs(coefficient) ** 2
    source_norm = np.linalg.norm(tuple(state.projected.values()))
    assert np.sqrt(target_norm_squared) / source_norm < 3.0e-13


def test_computational_reynolds_average_matches_existing_common_basis_projector(
    triangular_n12_state: _TriangularN12State,
) -> None:
    """Tie the alternative carrier to the audited fixed fusion-basis action."""

    state = triangular_n12_state
    n_sites = 12
    paths = enumerate_paths(n_sites, 0)
    coupled = coupled_basis_matrix(paths, 0, 0)
    circuit = triangular_hnf_gamma_a1_circuit(((6, 4), (0, 2)), target_q=0)

    full_base = np.zeros(1 << n_sites, dtype=np.complex128)
    full_projected = np.zeros_like(full_base)
    for integer_state in computational_basis(n_sites, n_up=n_sites // 2):
        configuration = _integer_bits(integer_state, n_sites)
        full_base[integer_state] = state.base[configuration]
        full_projected[integer_state] = state.projected[configuration]

    base_fusion = coupled.conj().T @ full_base
    common_basis_projected = circuit.apply(base_fusion)
    computational_projected_in_fusion = coupled.conj().T @ full_projected
    assert np.linalg.norm(common_basis_projected) > 1.0e-8
    assert (
        _relative_norm(
            computational_projected_in_fusion, common_basis_projected
        )
        < 2.0e-12
    )
    assert _relative_norm(coupled @ common_basis_projected, full_projected) < 2.0e-12


def test_square_n16_projection_has_tx_ty_c4_and_reflection_invariance() -> None:
    n_sites = 16
    group = square_gamma_a1_group(4)
    assert len(group) == 128
    carrier = ProjectedSingletAGP(
        _random_symmetric_pairing(n_sites, 2026091322), group
    )
    generators = square_site_generators(4)
    rng = np.random.default_rng(2026091323)
    all_configurations = enumerate_sz_configs(n_sites, n_sites // 2)
    selected = tuple(
        all_configurations[index]
        for index in rng.choice(len(all_configurations), size=24, replace=False)
    )
    reference = carrier.amplitudes(selected)
    assert np.linalg.norm(reference) > 1.0e-10
    for name in ("tx", "ty", "c4", "mx"):
        transformed = carrier.amplitudes(
            pullback_spin_configuration(configuration, generators[name])
            for configuration in selected
        )
        assert _relative_norm(transformed, reference) < 8.0e-14


@pytest.mark.parametrize("geometry", ("square", "triangular"))
def test_matrix_free_hpsi_matches_independent_dense_action_and_stays_invariant(
    geometry: str,
    triangular_n12_state: _TriangularN12State,
) -> None:
    if geometry == "square":
        n_sites = 4
        group = square_gamma_a1_group(2)
        generators = square_site_generators(2)
        bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.31, periodic=True)
        tested_generators = (generators["tx"], generators["ty"], generators["c4"], generators["mx"])
        first = ProjectedSingletAGP(
            _random_symmetric_pairing(4, 2026091324), group
        )
        second = ProjectedSingletAGP(
            _random_symmetric_pairing(4, 2026091325), group
        )
        amplitude = ProjectedSingletAGPCombination(
            (first, second), np.asarray((0.7 - 0.2j, -0.3 + 0.4j))
        )
    else:
        # The four-site triangular torus has no nonzero q=0,A1 vector.  The
        # twelve-site D6 torus is the first even exact-small cell used here
        # whose projected singlet carrier is nonzero.
        n_sites = 12
        supercell = ((6, 4), (0, 2))
        bonds = triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.125)
        tested_generators = (
            hnf_site_permutation(supercell, translation=(1, 0)),
            hnf_site_permutation(supercell, translation=(0, 1)),
            hnf_site_permutation(supercell, operation=C6),
            hnf_site_permutation(supercell, operation=MX),
        )
        amplitude = triangular_n12_state.carrier

    reference = exact_small_heisenberg_reference(amplitude, n_sites, bonds)
    assert np.linalg.norm(reference.amplitudes) > 1.0e-10

    if isinstance(amplitude, ProjectedSingletAGPCombination):
        spin_squared, spin_integer_basis = spin_squared_matrix(
            n_sites, n_up=n_sites // 2
        )
        spin_configurations = tuple(
            _integer_bits(state, n_sites) for state in spin_integer_basis
        )
        spin_positions = {
            configuration: index
            for index, configuration in enumerate(spin_configurations)
        }
        spin_reorder = [
            spin_positions[configuration]
            for configuration in reference.configurations
        ]
        reordered_spin_squared = spin_squared[np.ix_(spin_reorder, spin_reorder)]
        assert (
            np.linalg.norm(reordered_spin_squared @ reference.amplitudes)
            / np.linalg.norm(reference.amplitudes)
            < 2.0e-13
        )

    # This dense builder is independent of SzHeisenbergGraph, including its
    # integer/MSB representation and matrix assembly.  Reorder it onto the
    # configuration tuples used by the carrier before comparing H and H psi.
    independent_hamiltonian, independent_integer_basis = heisenberg_sz_matrix(
        n_sites, bonds, n_up=n_sites // 2
    )
    independent_configurations = tuple(
        _integer_bits(state, n_sites) for state in independent_integer_basis
    )
    independent_positions = {
        configuration: index
        for index, configuration in enumerate(independent_configurations)
    }
    reorder = [
        independent_positions[configuration]
        for configuration in reference.configurations
    ]
    independent_hamiltonian = independent_hamiltonian[np.ix_(reorder, reorder)]
    np.testing.assert_allclose(
        reference.hamiltonian, independent_hamiltonian, atol=2.0e-15
    )
    np.testing.assert_allclose(
        reference.hamiltonian_actions,
        independent_hamiltonian @ reference.amplitudes,
        atol=8.0e-13,
    )

    tested_indices = tuple(
        range(len(reference.configurations))
        if n_sites == 4
        else range(0, len(reference.configurations), 113)
    )
    matrix_free = np.asarray(
        [
            heisenberg_hpsi_amplitude(
                amplitude, reference.configurations[index], bonds
            )
            for index in tested_indices
        ]
    )
    np.testing.assert_allclose(
        matrix_free, reference.hamiltonian_actions[list(tested_indices)], atol=8.0e-12
    )
    if isinstance(amplitude, ProjectedSingletAGPCombination):
        for output_index, basis_index in enumerate(tested_indices):
            configuration = reference.configurations[basis_index]
            component_action = sum(
                coefficient
                * heisenberg_hpsi_amplitude(carrier, configuration, bonds)
                for coefficient, carrier in zip(
                    amplitude.coefficients, amplitude.carriers, strict=True
                )
            )
            np.testing.assert_allclose(
                matrix_free[output_index], component_action, atol=3.0e-13
            )

    amplitudes = {
        configuration: reference.amplitudes[index]
        for index, configuration in enumerate(reference.configurations)
    }
    actions = {
        configuration: reference.hamiltonian_actions[index]
        for index, configuration in enumerate(reference.configurations)
    }
    for image in tested_generators:
        np.testing.assert_allclose(
            [amplitudes[pullback_spin_configuration(config, image)] for config in reference.configurations],
            reference.amplitudes,
            atol=4.0e-13,
        )
        np.testing.assert_allclose(
            [actions[pullback_spin_configuration(config, image)] for config in reference.configurations],
            reference.hamiltonian_actions,
            atol=8.0e-13,
        )


def test_pairing_constraints_fail_closed() -> None:
    nonsymmetric = np.asarray(
        (
            (0.0, 1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 0.0),
        )
    )
    with pytest.raises(ValueError, match="symmetric"):
        singlet_agp_determinant_amplitude((1, 0, 1, 0), nonsymmetric)

    onsite = np.eye(4, dtype=np.complex128)
    with pytest.raises(ValueError, match="onsite"):
        singlet_agp_determinant_amplitude((1, 0, 1, 0), onsite)
