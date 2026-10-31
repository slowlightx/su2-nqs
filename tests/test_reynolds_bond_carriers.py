from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.physical_interpretation import (
    FixedSzPairActions,
    pair_permutation_indices,
)
from su2_nqs.operators.reynolds_bond_carriers import (
    bond_pair_orbit,
    connected_gram_schmidt,
    hermitian_bond_pair_action,
    reynolds_bond_pair_action,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def _cyclic_group(n_sites: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((site + shift) % n_sites for site in range(n_sites))
        for shift in range(n_sites)
    )


def test_reynolds_pair_is_invariant_and_hamiltonian_blind() -> None:
    basis = tuple(enumerate_sz_configs(4, 2))
    actions = FixedSzPairActions(basis)
    group = _cyclic_group(4)
    raw = np.arange(1, len(basis) + 1, dtype=np.float64)
    indices = np.stack(
        [pair_permutation_indices(basis, 0, shift) if shift else np.arange(len(basis))
         for shift in range(4)]
    )
    # Build an invariant input directly from the physical site action.  The
    # configuration-index action is obtained explicitly to avoid relying on a
    # model or Hamiltonian.
    permutations = []
    positions = {row: index for index, row in enumerate(basis)}
    for site_image in group:
        inverse = np.argsort(np.asarray(site_image))
        permutations.append(
            np.asarray([positions[tuple(np.asarray(row)[inverse])] for row in basis])
        )
    state = np.mean(raw[np.asarray(permutations)], axis=0).astype(np.complex128)
    state /= np.linalg.norm(state)

    image = reynolds_bond_pair_action(
        state, actions.pair, (0, 1), (1, 2), group
    )
    for permutation in permutations:
        np.testing.assert_allclose(image[permutation], image, atol=2.0e-14)
    assert len(bond_pair_orbit((0, 1), (1, 2), group)) == 4
    assert np.isfinite(np.vdot(state, image).real)


def test_hermitian_pair_has_real_expectation_and_whitening_preserves_prefix() -> None:
    basis = tuple(enumerate_sz_configs(4, 2))
    actions = FixedSzPairActions(basis)
    rng = np.random.default_rng(91)
    state = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    state /= np.linalg.norm(state)
    first = hermitian_bond_pair_action(state, actions.pair, (0, 1), (1, 2))
    second = hermitian_bond_pair_action(state, actions.pair, (0, 2), (1, 3))
    assert abs(np.vdot(state, first).imag) < 2.0e-14
    short = connected_gram_schmidt(state, (first, second))
    long = connected_gram_schmidt(state, (first, second, first + 0.2 * second))
    assert len(short) == 2
    np.testing.assert_allclose(np.column_stack(short), np.column_stack(long[:2]))
    overlap = np.column_stack(short).conj().T @ np.column_stack(short)
    np.testing.assert_allclose(overlap, np.eye(2), atol=2.0e-14)
