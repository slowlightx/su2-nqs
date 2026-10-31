from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.hubbard_space_group import (
    fermionic_permutation_index_sign_tables,
)
from su2_nqs.lattices.spatial_symmetry import (
    compose_site_permutations,
    rectangular_gamma_a1_group,
    rectangular_translation_group,
)
from su2_nqs.lattices.square import square_bonds
from su2_nqs.operators.hubbard_fock import hubbard_fock_sector_sparse


def _act(vector: np.ndarray, indices: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return signs * vector[indices]


def test_signed_2x4_rectangular_group_is_a_hamiltonian_representation() -> None:
    translations = rectangular_translation_group(2, 4)
    group = rectangular_gamma_a1_group(2, 4)
    assert len(translations) == 8
    assert len(group) == 16
    assert set(translations) < set(group)

    hamiltonian, basis = hubbard_fock_sector_sparse(
        8,
        8,
        0,
        square_bonds(2, 4, periodic=True),
        interaction=4.0,
    )
    indices, signs = fermionic_permutation_index_sign_tables(basis, group)
    positions = {image: index for index, image in enumerate(group)}
    rng = np.random.default_rng(17)
    vector = rng.normal(size=len(basis))

    maximum_representation_error = 0.0
    for left_index, left in enumerate(group):
        for right_index, right in enumerate(group):
            product_index = positions[compose_site_permutations(left, right)]
            successive = _act(
                _act(vector, indices[right_index], signs[right_index]),
                indices[left_index],
                signs[left_index],
            )
            direct = _act(vector, indices[product_index], signs[product_index])
            maximum_representation_error = max(
                maximum_representation_error,
                float(np.max(np.abs(successive - direct))),
            )
    assert maximum_representation_error < 1.0e-13

    maximum_commutator_error = 0.0
    maximum_unsigned_error = 0.0
    h_vector = np.asarray(hamiltonian @ vector)
    for image_indices, image_signs in zip(indices, signs, strict=True):
        transformed = _act(vector, image_indices, image_signs)
        signed_error = np.asarray(hamiltonian @ transformed) - _act(
            h_vector, image_indices, image_signs
        )
        maximum_commutator_error = max(
            maximum_commutator_error, float(np.max(np.abs(signed_error)))
        )
        unsigned_error = np.asarray(hamiltonian @ vector[image_indices]) - h_vector[
            image_indices
        ]
        maximum_unsigned_error = max(
            maximum_unsigned_error, float(np.max(np.abs(unsigned_error)))
        )
    assert maximum_commutator_error < 1.0e-13
    assert maximum_unsigned_error > 1.0


def test_signed_rectangular_projectors_are_nested_and_have_expected_rank() -> None:
    hamiltonian, basis = hubbard_fock_sector_sparse(
        8,
        8,
        0,
        square_bonds(2, 4, periodic=True),
        interaction=4.0,
    )
    del hamiltonian
    rng = np.random.default_rng(23)
    vector = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))

    projectors = {}
    ranks = {}
    for name, group in (
        ("translation", rectangular_translation_group(2, 4)),
        ("rectangular_g16", rectangular_gamma_a1_group(2, 4)),
    ):
        indices, signs = fermionic_permutation_index_sign_tables(basis, group)
        projectors[name] = lambda x, i=indices, s=signs: np.mean(s * x[i], axis=0)
        ranks[name] = int(
            round(
                sum(
                    float(signs[g, n])
                    for g in range(signs.shape[0])
                    for n in range(signs.shape[1])
                    if indices[g, n] == n
                )
                / signs.shape[0]
            )
        )
    translated = projectors["translation"](vector)
    full = projectors["rectangular_g16"](vector)
    assert np.linalg.norm(projectors["translation"](translated) - translated) < 1.0e-12
    assert np.linalg.norm(projectors["rectangular_g16"](full) - full) < 1.0e-12
    assert np.linalg.norm(projectors["translation"](full) - full) < 1.0e-12
    assert np.linalg.norm(projectors["rectangular_g16"](translated) - full) < 1.0e-12
    assert ranks == {"translation": 628, "rectangular_g16": 340}
