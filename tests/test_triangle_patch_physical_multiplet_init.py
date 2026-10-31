from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import (
    heisenberg_sz_matrix,
    spin_squared_matrix,
)
from su2_nqs.frontier.triangle_patch_physical_init import (
    init_triangle_patch_cluster_singlet,
    triangle_patch_cluster_reference,
)
from su2_nqs.frontier.triangle_patch_physical_multiplet_init import (
    init_triangle_patch_cluster_multiplets,
    triangle_patch_cluster_multiplet_reference,
)
from su2_nqs.models.triangle_patch_clebsch_tree import (
    triangle_patch_6x6_canary_plan,
    triangle_patch_amplitudes,
    triangle_patch_grid_plan,
    triangle_patch_pair_layout,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.local_schur import cell_schur_spec, cell_schur_tensor
from su2_nqs.su2.quantum_numbers import coupling_range, magnetic_numbers


jax.config.update("jax_enable_x64", True)


def _configurations(n_sites: int) -> np.ndarray:
    return np.asarray(
        [
            [(state >> (n_sites - site - 1)) & 1 for site in range(n_sites)]
            for state in range(1 << n_sites)
        ],
        dtype=np.int32,
    )


def _multiplet_member_from_reduced_coefficients(
    geometry: str,
    total_q: int,
    target_m2: int,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Independently reconstruct one magnetic member in patch-layout order."""

    configurations = _configurations(6)
    local_sites = (
        ((0, 1, 3), (2, 4, 5))
        if geometry == "square"
        else ((0, 3, 4), (1, 2, 5))
    )
    local_indices = []
    for sites in local_sites:
        local_bits = configurations[:, np.asarray(sites)]
        local_indices.append(4 * local_bits[:, 0] + 2 * local_bits[:, 1] + local_bits[:, 2])
    table = cell_schur_tensor("triangular_triangle")
    spec = cell_schur_spec("triangular_triangle")
    result = np.zeros(64, dtype=np.float64)
    offset = 0
    for q_left in spec.q_values:
        for q_right in spec.q_values:
            if total_q not in coupling_range(q_left, q_right):
                continue
            right_m_values = set(magnetic_numbers(q_right))
            for left_a in range(spec.multiplicity(q_left)):
                for right_a in range(spec.multiplicity(q_right)):
                    coefficient = coefficients[offset]
                    offset += 1
                    for left_m2 in magnetic_numbers(q_left):
                        right_m2 = target_m2 - left_m2
                        if right_m2 not in right_m_values:
                            continue
                        result += (
                            coefficient
                            * table[local_indices[0], q_left, left_m2 + 3, left_a]
                            * table[local_indices[1], q_right, right_m2 + 3, right_a]
                            * clebsch_gordan(
                                q_left,
                                left_m2,
                                q_right,
                                right_m2,
                                total_q,
                                target_m2,
                            )
                        )
    assert offset == len(coefficients)
    return result


@pytest.mark.parametrize(
    ("geometry", "ratio", "total_q", "level", "expected_energy", "dimension"),
    (
        ("square", 0.5, 0, 0, -2.6778862533179955, 5),
        ("square", 0.5, 0, 1, -1.7500000000000007, 5),
        ("square", 0.5, 2, 0, -1.9316125508776192, 9),
        ("square", 0.5, 4, 0, -0.3903882032022076, 5),
        ("square", 0.5, 6, 0, 2.25, 1),
        ("triangular", 0.125, 0, 0, -2.656913725802569, 5),
        ("triangular", 0.125, 0, 1, -1.7812500000000004, 5),
        ("triangular", 0.125, 2, 0, -1.9597167876454678, 9),
        ("triangular", 0.125, 4, 0, -0.355121145710702, 5),
        ("triangular", 0.125, 6, 0, 2.34375, 1),
    ),
)
def test_complete_two_triangle_multiplets_are_exact_h_and_s2_eigenstates(
    geometry: str,
    ratio: float,
    total_q: int,
    level: int,
    expected_energy: float,
    dimension: int,
) -> None:
    multiplet = triangle_patch_cluster_multiplet_reference(
        geometry, ratio, total_q, eigen_index=level
    )
    hamiltonian, basis = heisenberg_sz_matrix(6, multiplet.bonds)
    spin_squared, spin_basis = spin_squared_matrix(6)
    assert basis == spin_basis == tuple(range(64))
    np.testing.assert_allclose(multiplet.energy, expected_energy, atol=4.0e-14)
    assert multiplet.multiplicity_dimension == dimension
    assert multiplet.basis_orthonormality_error < 2.0e-14
    assert multiplet.eigen_residual < 4.0e-14
    assert multiplet.spin_squared_residual < 4.0e-14
    np.testing.assert_allclose(
        np.linalg.norm(multiplet.coefficients), 1.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        np.linalg.norm(multiplet.highest_weight_state), 1.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        hamiltonian @ multiplet.highest_weight_state,
        multiplet.energy * multiplet.highest_weight_state,
        atol=5.0e-14,
        rtol=0.0,
    )
    spin_eigenvalue = total_q * (total_q + 2) / 4.0
    np.testing.assert_allclose(
        spin_squared @ multiplet.highest_weight_state,
        spin_eigenvalue * multiplet.highest_weight_state,
        atol=5.0e-14,
        rtol=0.0,
    )
    configurations = _configurations(6)
    supported_up = (6 + total_q) // 2
    outside = configurations.sum(axis=1) != supported_up
    assert np.count_nonzero(
        np.abs(multiplet.highest_weight_state[outside]) > 2.0e-14
    ) == 0


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
@pytest.mark.parametrize(("total_q", "level"), ((0, 1), (2, 0), (4, 0), (6, 0)))
def test_one_reduced_vector_generates_the_same_exact_energy_for_every_magnetic_member(
    geometry: str,
    ratio: float,
    total_q: int,
    level: int,
) -> None:
    """Catch a coefficient-order or CG-orientation error hidden at highest weight."""

    reference = triangle_patch_cluster_multiplet_reference(
        geometry, ratio, total_q, eigen_index=level
    )
    hamiltonian, basis = heisenberg_sz_matrix(6, reference.bonds)
    spin_squared, spin_basis = spin_squared_matrix(6)
    assert basis == spin_basis == tuple(range(64))
    configurations = _configurations(6)
    for target_m2 in magnetic_numbers(total_q):
        member = _multiplet_member_from_reduced_coefficients(
            geometry,
            total_q,
            target_m2,
            reference.coefficients,
        )
        np.testing.assert_allclose(np.linalg.norm(member), 1.0, atol=3.0e-14)
        np.testing.assert_allclose(
            hamiltonian @ member,
            reference.energy * member,
            atol=7.0e-14,
            rtol=0.0,
        )
        spin_eigenvalue = total_q * (total_q + 2) / 4.0
        np.testing.assert_allclose(
            spin_squared @ member,
            spin_eigenvalue * member,
            atol=7.0e-14,
            rtol=0.0,
        )
        supported_up = (6 + target_m2) // 2
        assert np.count_nonzero(
            np.abs(member[configurations.sum(axis=1) != supported_up]) > 2.0e-14
        ) == 0


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_physical_multiplet_initializer_replaces_only_local_support_carrier(
    geometry: str,
    ratio: float,
) -> None:
    plan = triangle_patch_6x6_canary_plan(geometry)
    key = jax.random.PRNGKey(651)
    original = init_triangle_patch_cluster_singlet(
        key, plan, j2_over_j1=ratio, background_scale=0.05
    )
    physical = init_triangle_patch_cluster_multiplets(
        key, plan, j2_over_j1=ratio, background_scale=0.05
    )
    for name in ("reduced", "root", "patch_gates", "reduced_gates"):
        for actual, expected in zip(physical[name], original[name], strict=True):
            np.testing.assert_array_equal(actual, expected)

    layout = triangle_patch_pair_layout(plan)
    installed = {}
    for total_q in (0, 2, 4, 6):
        pieces = [
            np.asarray(block)[1].reshape(-1)
            for descriptor, block in zip(
                layout, physical["patch_pair"], strict=True
            )
            if descriptor.q_parent == total_q
        ]
        installed[total_q] = np.concatenate(pieces)
        level = 1 if total_q == 0 else 0
        reference = triangle_patch_cluster_multiplet_reference(
            geometry, ratio, total_q, eigen_index=level
        )
        np.testing.assert_allclose(
            installed[total_q], reference.coefficients, atol=0.0, rtol=0.0
        )
        np.testing.assert_allclose(
            np.linalg.norm(installed[total_q]), 1.0, atol=2.0e-14
        )

    ground = triangle_patch_cluster_reference(geometry, ratio).coefficients
    np.testing.assert_allclose(np.vdot(ground, installed[0]), 0.0, atol=2.0e-14)
    for descriptor, actual, expected in zip(
        layout, physical["patch_pair"], original["patch_pair"], strict=True
    ):
        if descriptor.q_parent == 0:
            np.testing.assert_array_equal(
                np.asarray(actual)[0], np.asarray(expected)[0]
            )
    assert any(
        not np.array_equal(actual, expected)
        for actual, expected in zip(
            physical["patch_pair"], original["patch_pair"], strict=True
        )
    )


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_n12_physical_multiplet_floor_is_normalized_and_core_orthogonal(
    geometry: str,
    ratio: float,
) -> None:
    plan = triangle_patch_grid_plan(
        geometry, 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    key = jax.random.PRNGKey(652)
    core = init_triangle_patch_cluster_multiplets(
        key, plan, j2_over_j1=ratio, background_scale=0.0
    )
    mixed = init_triangle_patch_cluster_multiplets(
        key, plan, j2_over_j1=ratio, background_scale=0.05
    )
    configurations = _configurations(12)
    configurations = jnp.asarray(configurations[configurations.sum(axis=1) == 6])
    core_state = np.asarray(
        triangle_patch_amplitudes(core, configurations, plan)
    )
    mixed_state = np.asarray(
        triangle_patch_amplitudes(mixed, configurations, plan)
    )
    floor = mixed_state - core_state
    np.testing.assert_allclose(np.linalg.norm(core_state), 1.0, atol=2.0e-13)
    np.testing.assert_allclose(np.linalg.norm(floor), 0.05, atol=2.0e-13)
    np.testing.assert_allclose(np.vdot(core_state, floor), 0.0, atol=5.0e-14)
    np.testing.assert_allclose(
        np.linalg.norm(mixed_state), np.sqrt(1.0 + 0.05**2), atol=2.0e-13
    )
    assert np.count_nonzero(mixed_state == 0.0) == 0


def test_multiplet_reference_rejects_invalid_sector_and_level() -> None:
    with pytest.raises(ValueError, match="total_q"):
        triangle_patch_cluster_multiplet_reference("square", 0.5, 1)
    with pytest.raises(TypeError, match="eigen_index"):
        triangle_patch_cluster_multiplet_reference(
            "square", 0.5, 2, eigen_index=False
        )
    with pytest.raises(ValueError, match="exceeds"):
        triangle_patch_cluster_multiplet_reference(
            "square", 0.5, 6, eigen_index=1
        )
