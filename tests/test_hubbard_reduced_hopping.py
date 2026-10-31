from itertools import product

import numpy as np
import pytest

from su2_nqs.benchmarks.hubbard_coupled_basis import (
    hubbard_coupled_basis_matrix,
)
from su2_nqs.benchmarks.hubbard_fock import (
    apply_fermion_hop,
    hubbard_fock_matrix,
)
from su2_nqs.lattices.square import square_bonds
from su2_nqs.operators.hubbard import (
    HubbardHamiltonian,
    hubbard_fusion_matrix,
    hubbard_reduced_fusion_matrix,
    hubbard_reduced_fusion_sparse,
)
from su2_nqs.su2.hubbard_spinor import (
    Q_OP_FERMION_SPINOR,
    SPINOR_COMPONENTS,
    canonical_hopping_parity,
)


def _allowed_endpoint_mask(
    occupation: int, spinor_component: int, *, creation: bool
) -> int | None:
    orbital = 0 if spinor_component == 1 else 1
    bit = 1 << orbital
    if creation:
        if occupation == 0:
            return 0
        if occupation == 1:
            return 1 << (1 - orbital)
        return None
    if occupation == 1:
        return bit
    if occupation == 2:
        return 3
    return None


def test_qop1_parity_matches_every_intermediate_spin_orientation() -> None:
    assert Q_OP_FERMION_SPINOR == 1
    assert SPINOR_COMPONENTS == (1, -1)
    occupations = (1, 2, 1, 0, 1)
    for destination, source in ((0, 4), (4, 0), (1, 3), (3, 1)):
        for component in SPINOR_COMPONENTS:
            destination_mask = _allowed_endpoint_mask(
                occupations[destination], component, creation=True
            )
            source_mask = _allowed_endpoint_mask(
                occupations[source], component, creation=False
            )
            if destination_mask is None or source_mask is None:
                with pytest.raises(ValueError, match="Pauli forbidden"):
                    canonical_hopping_parity(
                        occupations, destination, source, component
                    )
                continue
            other_single_sites = [
                site
                for site, occupation in enumerate(occupations)
                if occupation == 1 and site not in (destination, source)
            ]
            observed_signs = set()
            for orientations in product((1, 2), repeat=len(other_single_sites)):
                local_masks = []
                assigned = dict(zip(other_single_sites, orientations, strict=True))
                for site, occupation in enumerate(occupations):
                    if site == destination:
                        local_masks.append(destination_mask)
                    elif site == source:
                        local_masks.append(source_mask)
                    elif occupation == 0:
                        local_masks.append(0)
                    elif occupation == 2:
                        local_masks.append(3)
                    else:
                        local_masks.append(assigned[site])
                mask = sum(value << (2 * site) for site, value in enumerate(local_masks))
                source_orbital = 2 * source + (0 if component == 1 else 1)
                destination_orbital = 2 * destination + (
                    0 if component == 1 else 1
                )
                result = apply_fermion_hop(
                    mask, destination_orbital, source_orbital
                )
                assert result is not None
                observed_signs.add(result[1])
            assert observed_signs == {
                canonical_hopping_parity(
                    occupations, destination, source, component
                )
            }


@pytest.mark.parametrize(
    ("n_sites", "n_electrons", "target_q", "bonds"),
    (
        (2, 2, 0, ((0, 1),)),
        (2, 2, 2, ((0, 1),)),
        (4, 2, 0, square_bonds(2, 2)),
        (4, 4, 0, square_bonds(2, 2)),
        (4, 4, 2, square_bonds(2, 2)),
        (6, 4, 0, square_bonds(2, 3)),
        (6, 4, 2, square_bonds(2, 3)),
        (6, 6, 0, square_bonds(2, 3)),
        (6, 6, 2, square_bonds(2, 3)),
        (6, 6, 4, square_bonds(2, 3)),
    ),
)
def test_reduced_spinor_compiler_matches_direct_cg_through_l6(
    n_sites: int,
    n_electrons: int,
    target_q: int,
    bonds,
) -> None:
    direct, direct_states = hubbard_fusion_matrix(
        n_sites,
        n_electrons,
        target_q,
        bonds,
        interaction=3.7,
    )
    reduced, reduced_states = hubbard_reduced_fusion_matrix(
        n_sites,
        n_electrons,
        target_q,
        bonds,
        interaction=3.7,
    )
    assert reduced_states == direct_states
    np.testing.assert_allclose(reduced, direct, atol=2.0e-14)
    np.testing.assert_allclose(reduced, reduced.conj().T, atol=2.0e-14)


def test_l6_reduced_spinor_matrix_matches_independent_fock_projection() -> None:
    bonds = square_bonds(2, 3)
    interaction = -1.3
    reduced, states = hubbard_reduced_fusion_matrix(
        6, 6, 0, bonds, interaction=interaction
    )
    fock, fock_basis = hubbard_fock_matrix(
        6, 6, bonds, interaction=interaction
    )
    transform, basis = hubbard_coupled_basis_matrix(states, total_m=0)
    assert basis == fock_basis
    projected = transform.conj().T @ fock @ transform
    np.testing.assert_allclose(reduced, projected, atol=5.0e-14)


def test_sparse_reduced_compiler_matches_dense_and_has_new_old_orientation() -> None:
    bonds = square_bonds(2, 3)
    sparse, states = hubbard_reduced_fusion_sparse(
        6, 4, 0, bonds, interaction=2.1
    )
    dense, dense_states = hubbard_reduced_fusion_matrix(
        6, 4, 0, bonds, interaction=2.1
    )
    assert states == dense_states
    assert sparse.format == "csr"
    np.testing.assert_allclose(sparse.toarray(), dense, atol=2.0e-14)
    np.testing.assert_allclose(
        sparse.toarray(), sparse.toarray().conj().T, atol=2.0e-14
    )
    for column in (0, len(states) // 2, len(states) - 1):
        rows = sparse[:, column].nonzero()[0]
        np.testing.assert_allclose(
            sparse[rows, column].toarray().reshape(-1), dense[rows, column]
        )


def test_hubbard_operator_exposes_reduced_compiler_connectivity() -> None:
    bonds = tuple(square_bonds(2, 3))
    direct = HubbardHamiltonian(6, 4, 0, bonds, 2.5)
    reduced = HubbardHamiltonian(
        6, 4, 0, bonds, 2.5, compiler="reduced_spinor"
    )
    assert reduced.basis == direct.basis
    np.testing.assert_allclose(reduced.to_dense(), direct.to_dense(), atol=2.0e-14)
    for column in (0, len(reduced.basis) // 2, len(reduced.basis) - 1):
        connected, elements = reduced.get_conn(reduced.basis[column])
        positions = [reduced.basis.index(labels) for labels in connected]
        np.testing.assert_allclose(
            elements, reduced.to_dense()[positions, column], atol=2.0e-14
        )
    with pytest.raises(ValueError, match="compiler"):
        HubbardHamiltonian(2, 2, 0, ((0, 1),), 1.0, compiler="unknown")
