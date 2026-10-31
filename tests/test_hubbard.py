import numpy as np
import pytest

from su2_nqs.benchmarks.hubbard_coupled_basis import (
    hubbard_coupled_basis_matrix,
)
from su2_nqs.benchmarks.hubbard_fock import (
    apply_fermion_hop,
    enumerate_fock_basis,
    hubbard_fock_matrix,
)
from su2_nqs.lattices.square import square_bonds
from su2_nqs.operators.hubbard import HubbardHamiltonian, hubbard_fusion_matrix
from su2_nqs.su2.hubbard_fusion import (
    HubbardFusionState,
    decode_hubbard_fusion_state,
    encode_hubbard_fusion_state,
    enumerate_hubbard_fusion_states,
    is_valid_hubbard_fusion_state,
)


def test_fermion_hop_uses_canonical_interleaved_orbital_signs():
    # c^dagger_1up c_0up crosses the occupied 0down orbital.
    mask = (1 << 0) | (1 << 1) | (1 << 3)
    assert apply_fermion_hop(mask, 2, 0) == ((1 << 1) | (1 << 2) | (1 << 3), -1)
    assert apply_fermion_hop(mask, 0, 1) is None


def test_two_site_charge_spin_basis_counts_and_labels():
    singlets = enumerate_hubbard_fusion_states(2, 2, 0)
    triplets = enumerate_hubbard_fusion_states(2, 2, 2)
    assert len(singlets) == 3
    assert len(triplets) == 1
    assert {state.occupations for state in singlets} == {(0, 2), (1, 1), (2, 0)}
    assert all(
        is_valid_hubbard_fusion_state(state, n_electrons=2, target_q=0)
        for state in singlets
    )
    assert not is_valid_hubbard_fusion_state(HubbardFusionState((1, 1), (1, 1)))


def test_hubbard_fusion_flat_labels_round_trip():
    for state in enumerate_hubbard_fusion_states(4, 4, 0):
        assert decode_hubbard_fusion_state(encode_hubbard_fusion_state(state)) == state


@pytest.mark.parametrize(
    ("n_sites", "n_electrons", "target_q"),
    [(2, 2, 0), (2, 2, 2), (4, 2, 0), (4, 2, 2), (4, 4, 0), (4, 4, 2)],
)
def test_hubbard_coupled_basis_is_orthonormal(n_sites, n_electrons, target_q):
    states = enumerate_hubbard_fusion_states(n_sites, n_electrons, target_q)
    transform, basis = hubbard_coupled_basis_matrix(states, total_m=target_q)
    assert basis == enumerate_fock_basis(n_sites, n_electrons)
    np.testing.assert_allclose(
        transform.conj().T @ transform, np.eye(len(states)), atol=2e-14
    )


@pytest.mark.parametrize(
    ("n_sites", "n_electrons", "bonds"),
    [
        (2, 2, ((0, 1),)),
        (4, 2, square_bonds(2, 2)),
        (4, 4, square_bonds(2, 2)),
    ],
)
def test_direct_hubbard_fusion_hamiltonian_matches_fock_projection(
    n_sites, n_electrons, bonds
):
    interaction = 3.7
    fock, fock_basis = hubbard_fock_matrix(
        n_sites, n_electrons, bonds, interaction=interaction
    )
    maximum_q = min(n_electrons, 2 * n_sites - n_electrons)
    for target_q in range(n_electrons % 2, maximum_q + 1, 2):
        direct, states = hubbard_fusion_matrix(
            n_sites,
            n_electrons,
            target_q,
            bonds,
            interaction=interaction,
        )
        transform, basis = hubbard_coupled_basis_matrix(states, total_m=target_q)
        assert basis == fock_basis
        projected = transform.conj().T @ fock @ transform
        np.testing.assert_allclose(direct, projected, atol=3e-14)
        np.testing.assert_allclose(direct, direct.conj().T, atol=3e-14)


def test_two_site_half_filled_spectrum_matches_analytic_result():
    interaction = 8.0
    hopping = 1.0
    singlet, _ = hubbard_fusion_matrix(
        2, 2, 0, ((0, 1, hopping),), interaction=interaction
    )
    triplet, _ = hubbard_fusion_matrix(
        2, 2, 2, ((0, 1, hopping),), interaction=interaction
    )
    root = np.sqrt(interaction**2 + 16.0 * hopping**2)
    expected_singlets = np.sort(
        ((interaction - root) / 2.0, interaction, (interaction + root) / 2.0)
    )
    np.testing.assert_allclose(np.linalg.eigvalsh(singlet), expected_singlets, atol=2e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(triplet), (0.0,), atol=2e-14)


@pytest.mark.parametrize(("n_sites", "n_electrons"), [(2, 2), (4, 2), (4, 4)])
def test_spin_sector_spectra_reconstruct_complete_fock_spectrum(n_sites, n_electrons):
    bonds = ((0, 1),) if n_sites == 2 else square_bonds(2, 2)
    fock, _ = hubbard_fock_matrix(n_sites, n_electrons, bonds, interaction=2.3)
    reconstructed = []
    maximum_q = min(n_electrons, 2 * n_sites - n_electrons)
    for target_q in range(n_electrons % 2, maximum_q + 1, 2):
        sector, _ = hubbard_fusion_matrix(
            n_sites, n_electrons, target_q, bonds, interaction=2.3
        )
        for eigenvalue in np.linalg.eigvalsh(sector):
            reconstructed.extend([eigenvalue] * (target_q + 1))
    np.testing.assert_allclose(
        np.sort(reconstructed), np.linalg.eigvalsh(fock), atol=5e-14
    )


def test_hubbard_operator_wrapper_preserves_new_old_connectivity_orientation():
    operator = HubbardHamiltonian(
        n_sites=4,
        n_electrons=4,
        target_q=0,
        bonds=tuple(square_bonds(2, 2)),
        interaction=4.0,
    )
    matrix = operator.to_dense()
    assert len(operator.basis) == matrix.shape[0]
    for column, labels in enumerate(operator.basis):
        connected, elements = operator.get_conn(labels)
        rows = [operator.basis.index(value) for value in connected]
        np.testing.assert_allclose(elements, matrix[rows, column], atol=2e-14)
