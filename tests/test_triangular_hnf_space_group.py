"""Exact small-system gates for triangular HNF space-group actions."""

from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.triangular_hnf_space_group import (
    d6_character_projectors,
    d6_projector_audit,
    fraction_momentum_json,
    hnf_fraction_character_numerator,
    hnf_fraction_momentum_allowed,
    momentum_projector_audit,
    momentum_projector_matrix,
    parse_fraction_momentum,
    reciprocal_little_group,
    reciprocal_star_audit,
    regular_character_projector_audit,
)
from su2_nqs.lattices.triangular_hnf import (
    C6,
    IDENTITY,
    MX,
    hnf_character_numerators,
    hnf_point_group_permutations,
    hnf_site_permutation,
    hnf_sites,
    hnf_translation_permutations,
    matrix_power,
    reduce_hnf,
    triangular_hnf_j1j2_bonds,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.permutation import (
    heisenberg_graph_fusion_matrix,
    site_permutation_fusion_matrix,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[site]] for site in range(len(left)))


def _inverse(image: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(image)
    for old, new in enumerate(image):
        result[new] = old
    return tuple(result)


def _computational_permutation_matrix(image: tuple[int, ...]) -> np.ndarray:
    """Independent active spin movement with site zero as the MSB."""

    n = len(image)
    matrix = np.zeros((1 << n, 1 << n), dtype=np.float64)
    for old_index in range(1 << n):
        new_index = 0
        for old_site, new_site in enumerate(image):
            bit = (old_index >> (n - old_site - 1)) & 1
            new_index |= bit << (n - new_site - 1)
        matrix[new_index, old_index] = 1.0
    return matrix


def test_hnf_reduction_and_rectangular_bonds_match_existing_4x4() -> None:
    supercell = ((4, 0), (0, 4))
    assert len(hnf_sites(supercell)) == 16
    assert reduce_hnf((5, -1), supercell) == (1, 3)
    assert triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.37) == (
        triangular_j1j2_bonds(4, 4, j1=1.0, j2=0.37, periodic=True)
    )


@pytest.mark.parametrize(
    ("supercell", "target_q", "translation", "operation"),
    (
        (((2, 0), (0, 2)), 0, (1, 0), IDENTITY),
        (((2, 0), (0, 2)), 2, (0, 1), C6),
        (((3, 1), (0, 2)), 0, (1, 1), IDENTITY),
        (((3, 0), (0, 3)), 1, (0, 0), MX),
    ),
)
def test_hnf_fusion_action_matches_independent_computational_projection(
    supercell, target_q, translation, operation
) -> None:
    image = hnf_site_permutation(
        supercell, translation=translation, operation=operation
    )
    paths = enumerate_paths(len(image), target_q)
    coupled = coupled_basis_matrix(paths, target_q, target_q)
    expected = (
        coupled.conj().T @ _computational_permutation_matrix(image) @ coupled
    )
    actual = site_permutation_fusion_matrix(
        image, target_q=target_q
    ).toarray()
    np.testing.assert_allclose(actual, expected, atol=6.0e-13)


def test_full_d6_and_translation_site_maps_obey_semidirect_relations() -> None:
    supercell = ((3, 0), (0, 3))
    identity = hnf_site_permutation(supercell)
    c6 = hnf_site_permutation(supercell, operation=C6)
    mx = hnf_site_permutation(supercell, operation=MX)
    tx = hnf_site_permutation(supercell, translation=(1, 0))
    ty = hnf_site_permutation(supercell, translation=(0, 1))
    c6_inverse = hnf_site_permutation(
        supercell, operation=matrix_power(C6, 5)
    )
    assert len(hnf_point_group_permutations(supercell)) == 12
    assert matrix_power(C6, 6) == IDENTITY
    assert _compose(c6, _compose(c6, _compose(c6, _compose(c6, _compose(c6, c6))))) == identity
    assert _compose(mx, mx) == identity
    assert _compose(mx, _compose(c6, mx)) == c6_inverse
    assert _compose(tx, ty) == _compose(ty, tx)
    expected_diagonal = hnf_site_permutation(supercell, translation=(1, 1))
    assert _compose(c6, _compose(tx, _inverse(c6))) == expected_diagonal
    expected_inverse_x = hnf_site_permutation(supercell, translation=(-1, 0))
    assert _compose(c6, _compose(ty, _inverse(c6))) == expected_inverse_x


@pytest.mark.parametrize(
    ("supercell", "target_q"),
    (
        (((2, 0), (0, 2)), 0),
        (((2, 0), (0, 2)), 2),
        (((3, 1), (0, 2)), 0),
        (((3, 0), (0, 3)), 1),
    ),
)
def test_hnf_momentum_projectors_are_complete_orthogonal_and_idempotent(
    supercell, target_q
) -> None:
    audit = momentum_projector_audit(supercell, target_q=target_q)
    assert audit["character_count"] == len(hnf_sites(supercell))
    assert audit["rank_sum"] == audit["dimension"]
    assert audit["completeness_error"] < 2.0e-12
    assert audit["maximum_orthogonality_error"] < 2.0e-12
    assert audit["maximum_hermiticity_error"] < 2.0e-12
    assert audit["maximum_idempotency_error"] < 2.0e-12


@pytest.mark.parametrize("target_q", (0, 2))
def test_d6_character_projectors_resolve_small_fusion_sectors(target_q: int) -> None:
    audit = d6_projector_audit(((2, 0), (0, 2)), target_q=target_q)
    assert audit["rank_sum"] == audit["dimension"]
    assert audit["completeness_error"] < 2.0e-12
    assert audit["maximum_orthogonality_error"] < 2.0e-12
    assert audit["maximum_hermiticity_error"] < 2.0e-12
    assert audit["maximum_idempotency_error"] < 2.0e-12


def test_d6_maps_each_momentum_projector_to_one_allowed_character() -> None:
    supercell = ((3, 0), (0, 3))
    target_q = 1
    characters = hnf_character_numerators(supercell)
    projectors = {
        numerator: momentum_projector_matrix(
            supercell, numerator, target_q=target_q
        )
        for numerator in characters
    }
    point_actions = hnf_point_group_permutations(supercell)
    for _, _, image in point_actions:
        group = site_permutation_fusion_matrix(
            image, target_q=target_q
        ).toarray()
        for projector in projectors.values():
            transformed = group @ projector @ group.T
            assert min(
                np.linalg.norm(transformed - candidate)
                for candidate in projectors.values()
            ) < 3.0e-12


def test_hnf_hamiltonian_commutes_with_retained_space_group() -> None:
    supercell = ((3, 0), (0, 3))
    target_q = 1
    bonds = triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.37)
    hamiltonian = heisenberg_graph_fusion_matrix(
        9, bonds, target_q=target_q
    ).toarray()
    images = [image for _, image in hnf_translation_permutations(supercell)]
    images += [image for _, _, image in hnf_point_group_permutations(supercell)]
    for image in images:
        group = site_permutation_fusion_matrix(
            image, target_q=target_q
        ).toarray()
        np.testing.assert_allclose(
            group @ hamiltonian @ group.T, hamiltonian, atol=2.0e-12
        )


def test_invalid_momentum_and_non_d6_cell_are_rejected() -> None:
    with pytest.raises(ValueError, match="not a character"):
        momentum_projector_matrix(((2, 0), (0, 2)), (1, 0), target_q=0)
    with pytest.raises(ValueError, match="full D6"):
        d6_character_projectors(((3, 1), (0, 2)), target_q=0)


PRIMARY_P9_HNFS = (
    ((6, 0), (0, 6)),
    ((12, 8), (0, 4)),
    ((18, 12), (0, 6)),
    ((12, 0), (0, 12)),
)
X_STAR = tuple(
    parse_fraction_momentum(raw)
    for raw in (
        (1, 6, 1, 6),
        (5, 6, 1, 3),
        (2, 3, 1, 6),
        (5, 6, 5, 6),
        (1, 6, 2, 3),
        (1, 3, 5, 6),
    )
)
K_STAR = tuple(
    parse_fraction_momentum(raw) for raw in ((1, 3, 1, 3), (2, 3, 2, 3))
)
M_STAR = tuple(
    parse_fraction_momentum(raw)
    for raw in ((1, 2, 0, 1), (0, 1, 1, 2), (1, 2, 1, 2))
)


@pytest.mark.parametrize("supercell", PRIMARY_P9_HNFS)
@pytest.mark.parametrize(
    ("star", "stabilizer"),
    (
        (X_STAR, ("C6^0", "Mx*C6^4")),
        (
            K_STAR,
            ("C6^0", "C6^2", "C6^4", "Mx*C6^0", "Mx*C6^2", "Mx*C6^4"),
        ),
        (M_STAR, ("C6^0", "C6^3", "Mx*C6^2", "Mx*C6^5")),
    ),
)
def test_p9_xkm_stars_are_exact_allowed_and_d6_sewn(
    supercell, star, stabilizer
) -> None:
    audit = reciprocal_star_audit(supercell, star[0], star)
    assert audit["all_arms_allowed"]
    assert audit["all_arms_distinct"]
    assert audit["computed_star_matches_expected"]
    assert audit["D6_sewing_closed"]
    assert audit["D6_sewing_bijective"]
    assert tuple(audit["little_group_stabilizer"]) == stabilizer
    for momentum, numerator in zip(
        star, audit["character_numerators_over_N"], strict=True
    ):
        assert hnf_fraction_momentum_allowed(supercell, momentum)
        assert list(hnf_fraction_character_numerator(supercell, momentum)) == numerator
        assert parse_fraction_momentum(fraction_momentum_json(momentum)) == momentum


@pytest.mark.parametrize(
    ("momentum", "characters"),
    (
        (X_STAR[0], {"C6^0": 1, "Mx*C6^4": 1}),
        (
            K_STAR[0],
            {
                "C6^0": 1,
                "C6^2": 1,
                "C6^4": 1,
                "Mx*C6^0": 1,
                "Mx*C6^2": 1,
                "Mx*C6^4": 1,
            },
        ),
    ),
)
def test_p9_dense_regular_projectors_close_without_fusion_space(
    momentum, characters
) -> None:
    audit = regular_character_projector_audit(
        PRIMARY_P9_HNFS[0], momentum, characters, j2=0.125
    )
    assert audit["space"] == "single-particle_regular_site_representation_not_fusion"
    assert audit["projector_rank"] == 1
    assert tuple(audit["little_group_stabilizer"]) == reciprocal_little_group(
        PRIMARY_P9_HNFS[0], momentum
    )
    for key, value in audit.items():
        if key.endswith("_error"):
            assert float(value) < 1e-12


def test_p9_fraction_momentum_boundary_validation() -> None:
    with pytest.raises(ValueError, match="requires"):
        parse_fraction_momentum((1, 2, 0))
    with pytest.raises(ValueError, match="positive"):
        parse_fraction_momentum((1, 0, 0, 1))
    with pytest.raises(ValueError, match="not an allowed"):
        hnf_fraction_character_numerator(
            ((4, 0), (0, 4)), parse_fraction_momentum((1, 3, 1, 3))
        )
    with pytest.raises(ValueError, match="cover the exact little group"):
        regular_character_projector_audit(
            PRIMARY_P9_HNFS[0], X_STAR[0], {"C6^0": 1}
        )
