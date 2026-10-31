import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import (
    coo_matvec,
    fixed_m_spin_basis,
    graph_hamiltonian_coo,
    physical_permutation_index_table,
    reynolds_project_state,
)
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.projected_singlet_agp import (
    pullback_spin_configuration,
    square_gamma_a1_group,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitude,
    spin_clebsch_tree_amplitudes,
    spin_clebsch_tree_layouts,
    spin_clebsch_tree_parameter_count,
)
from su2_nqs.su2.fusion_tree import FusionTree


jax.config.update("jax_enable_x64", True)


def _spin_squared_in_basis(basis: tuple[tuple[int, ...], ...]) -> np.ndarray:
    n_sites = len(basis[0])
    matrix, integer_basis = spin_squared_matrix(
        n_sites, n_up=sum(basis[0])
    )
    positions = {value: index for index, value in enumerate(integer_basis)}
    order = [
        positions[
            sum(bit << (n_sites - site - 1) for site, bit in enumerate(row))
        ]
        for row in basis
    ]
    return matrix[np.ix_(order, order)]


@pytest.mark.parametrize(
    "n_sites,target_q,target_m2,maximum_q",
    ((4, 0, 0, None), (6, 2, 2, None), (8, 0, 0, 2)),
)
def test_random_physical_tree_state_has_exact_total_spin(
    n_sites: int,
    target_q: int,
    target_m2: int,
    maximum_q: int | None,
) -> None:
    config = SpinClebschTreeConfig(
        FusionTree.balanced(n_sites),
        target_q=target_q,
        target_m2=target_m2,
        feature_dim=3,
        maximum_intermediate_q=maximum_q,
    )
    params = init_spin_clebsch_tree(jax.random.PRNGKey(90 + n_sites), config)
    basis = fixed_m_spin_basis(n_sites, target_m2)
    amplitudes = np.asarray(
        spin_clebsch_tree_amplitudes(params, jnp.asarray(basis), config)
    )
    assert np.linalg.norm(amplitudes) > 1.0e-14
    spin_squared = _spin_squared_in_basis(basis)
    spin = target_q / 2.0
    residual = spin_squared @ amplitudes - spin * (spin + 1.0) * amplitudes
    assert np.linalg.norm(residual) / np.linalg.norm(amplitudes) < 2.0e-12


def test_equal_size_merges_share_reduced_parameters() -> None:
    config = SpinClebschTreeConfig(FusionTree.balanced(8), feature_dim=2)
    layouts = spin_clebsch_tree_layouts(config)
    pair_indices = {
        layout.parameter_index
        for layout in layouts
        if (layout.left_size, layout.right_size) == (1, 1)
    }
    assert len(pair_indices) == 1
    params = init_spin_clebsch_tree(jax.random.PRNGKey(0), config)
    assert spin_clebsch_tree_parameter_count(params) < 200


def test_gamma_a1_gather_projection_is_invariant_without_new_parameters() -> None:
    group = square_gamma_a1_group(2)
    basis = fixed_m_spin_basis(4, 0)
    indices = physical_permutation_index_table(basis, group)
    config = SpinClebschTreeConfig(FusionTree.balanced(4), feature_dim=2)
    params = init_spin_clebsch_tree(jax.random.PRNGKey(12), config)
    raw = spin_clebsch_tree_amplitudes(params, jnp.asarray(basis), config)
    projected = np.asarray(reynolds_project_state(raw, indices))
    positions = {row: index for index, row in enumerate(basis)}
    for image in group:
        transformed = np.asarray(
            [
                projected[
                    positions[pullback_spin_configuration(row, image)]
                ]
                for row in basis
            ]
        )
        np.testing.assert_allclose(transformed, projected, atol=2.0e-14)


def test_coo_matvec_and_gradient_match_dense_reference() -> None:
    bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.3, periodic=False)
    rows, columns, values, basis = graph_hamiltonian_coo(4, 0, bonds)
    dense = np.zeros((len(basis), len(basis)))
    np.add.at(dense, (rows, columns), values)
    vector = jnp.arange(1, len(basis) + 1, dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(coo_matvec(rows, columns, values, vector)), dense @ vector
    )

    config = SpinClebschTreeConfig(FusionTree.balanced(4), feature_dim=2)
    params = init_spin_clebsch_tree(jax.random.PRNGKey(4), config)

    def loss(candidate):
        amplitudes = spin_clebsch_tree_amplitudes(
            candidate, jnp.asarray(basis), config
        )
        applied = coo_matvec(rows, columns, values, amplitudes)
        return jnp.vdot(amplitudes, applied).real / jnp.vdot(
            amplitudes, amplitudes
        ).real

    gradients = jax.grad(loss)(params)
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradients)
    )


def test_invalid_configuration_returns_zero_and_invalid_config_fails() -> None:
    config = SpinClebschTreeConfig(FusionTree.balanced(4), feature_dim=2)
    params = init_spin_clebsch_tree(jax.random.PRNGKey(1), config)
    assert float(
        spin_clebsch_tree_amplitude(params, jnp.asarray((0, 0, 0, 0)), config)
    ) == 0.0
    assert float(
        spin_clebsch_tree_amplitude(params, jnp.asarray((0, 1, 2, 0)), config)
    ) == 0.0
    with pytest.raises(ValueError, match="unreachable"):
        SpinClebschTreeConfig(
            FusionTree.balanced(4), target_q=1, target_m2=1
        )
