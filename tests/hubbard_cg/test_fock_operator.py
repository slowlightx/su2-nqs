import numpy as np
import pytest

from su2_nqs.benchmarks.hubbard_fock import (
    apply_fermion_hop,
    enumerate_fock_basis,
    hubbard_fock_matrix,
)
from su2_nqs.operators.hubbard_fock import (
    apply_canonical_hop,
    apply_site_permutation,
    enumerate_hubbard_tokens,
    fock_mask_to_tokens,
    hubbard_fock_sector_matrix,
    tokens_to_fock_mask,
)


@pytest.mark.parametrize("n_sites", [1, 2, 3])
def test_physical_tokens_round_trip_every_fock_mask(n_sites):
    for mask in range(1 << (2 * n_sites)):
        assert tokens_to_fock_mask(fock_mask_to_tokens(mask, n_sites)) == mask


def test_interval_parity_hop_matches_independent_sequential_oracle():
    for n_modes in range(2, 13):
        for mask in range(1 << n_modes):
            for destination in range(n_modes):
                for source in range(n_modes):
                    assert apply_canonical_hop(mask, destination, source) == apply_fermion_hop(
                        mask, destination, source
                    )


@pytest.mark.parametrize(
    ("n_sites", "n_electrons", "total_m2"),
    [(2, 2, 0), (3, 2, 0), (3, 3, 1), (4, 4, 0), (6, 2, 0)],
)
def test_fixed_charge_magnetic_enumerator_matches_filtered_fock_basis(
    n_sites, n_electrons, total_m2
):
    actual = tuple(
        tokens_to_fock_mask(tokens)
        for tokens in enumerate_hubbard_tokens(
            n_sites, n_electrons, total_m2=total_m2
        )
    )
    expected = tuple(
        mask
        for mask in enumerate_fock_basis(n_sites, n_electrons)
        if sum((mask >> (2 * site)) & 1 for site in range(n_sites))
        - sum((mask >> (2 * site + 1)) & 1 for site in range(n_sites))
        == total_m2
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("n_sites", "n_electrons", "total_m2", "bonds"),
    [
        (2, 2, 0, ((0, 1),)),
        (3, 2, 0, ((0, 1), (1, 2), (2, 0, -0.4))),
        (4, 4, 0, ((0, 1), (1, 2, 0.7), (2, 3), (3, 0, -0.2))),
        (6, 2, 0, ((0, 5), (1, 4, -0.3), (2, 3, 1.1))),
    ],
)
def test_production_sector_matrix_matches_independent_full_fock_oracle(
    n_sites, n_electrons, total_m2, bonds
):
    interaction = 3.7
    actual, token_basis = hubbard_fock_sector_matrix(
        n_sites,
        n_electrons,
        total_m2,
        bonds,
        interaction=interaction,
    )
    full, full_basis = hubbard_fock_matrix(
        n_sites, n_electrons, bonds, interaction=interaction
    )
    positions = {mask: index for index, mask in enumerate(full_basis)}
    indices = [positions[tokens_to_fock_mask(tokens)] for tokens in token_basis]
    expected = full[np.ix_(indices, indices)]
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(actual, actual.T, atol=0.0, rtol=0.0)


def test_fermionic_site_permutations_form_an_exact_representation():
    n_sites = 4
    translation = (1, 2, 3, 0)
    reflection = (3, 2, 1, 0)
    for mask in enumerate_fock_basis(n_sites, 4):
        tokens = fock_mask_to_tokens(mask, n_sites)
        translated, sign_t = apply_site_permutation(tokens, translation)
        sequential, sign_tt = apply_site_permutation(translated, reflection)
        composition = tuple(reflection[translation[site]] for site in range(n_sites))
        direct, sign_direct = apply_site_permutation(tokens, composition)
        assert sequential == direct
        assert sign_t * sign_tt == sign_direct


def test_ring_hamiltonian_commutes_with_signed_translation_and_reflection():
    n_sites = 4
    basis = enumerate_hubbard_tokens(n_sites, 4, total_m2=0)
    positions = {state: index for index, state in enumerate(basis)}
    bonds = ((0, 1), (1, 2), (2, 3), (3, 0))
    hamiltonian, matrix_basis = hubbard_fock_sector_matrix(
        n_sites, 4, 0, bonds, interaction=5.0
    )
    assert basis == matrix_basis
    for image in ((1, 2, 3, 0), (3, 2, 1, 0)):
        operator = np.zeros_like(hamiltonian)
        for column, state in enumerate(basis):
            transformed, sign = apply_site_permutation(state, image)
            operator[positions[transformed], column] = sign
        np.testing.assert_allclose(operator.T @ hamiltonian @ operator, hamiltonian, atol=0.0, rtol=0.0)
