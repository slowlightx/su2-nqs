import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.energy_tangent_projection import (
    factor_born_complex_tangent,
    local_energy_tangent_response,
    materialize_message_head_log_derivatives,
    project_born_complex_tangent,
)
from su2_nqs.benchmarks.sparse_causal_recurrence import (
    compile_sparse_causal_recurrence_plan,
)
from su2_nqs.benchmarks.sparse_causal_tangent import HEAD_GROUPS
from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig
from su2_nqs.models.sparse_causal_tree import (
    SparseCausalTreeGRUConfig,
    init_sparse_causal_tree_gru,
    sparse_causal_tree_log_psi,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree

jax.config.update("jax_enable_x64", True)


def _n6_exact_problem():
    n = 6
    bonds = tuple((site, site + 1, 1.0) for site in range(n - 1)) + tuple(
        (site, site + 2, 0.47) for site in range(n - 2)
    )
    tagged = tuple(
        (left, right, coupling, int(right - left == 2))
        for left, right, coupling in bonds
    )
    plan = compile_sparse_causal_recurrence_plan(n, tagged)
    base = CompactTreeGRUConfig(
        FusionTree.left_associated(n), 0, embedding_dim=3, hidden_dim=7
    )
    config = SparseCausalTreeGRUConfig(base, plan, message_hidden_dim=3)
    params = init_sparse_causal_tree_gru(jax.random.PRNGKey(9121), config)
    states = jnp.asarray(
        [state.labels for state in config.tree.enumerate_states(0)],
        dtype=jnp.int32,
    )
    logs = np.asarray(
        jax.vmap(
            lambda state: sparse_causal_tree_log_psi(params, state, config)
        )(states)
    )
    wavefunction = np.exp(logs)
    weights = np.abs(wavefunction) ** 2
    hamiltonian = heisenberg_graph_fusion_matrix(
        n, bonds, target_q=0
    ).toarray()
    local_energy = (hamiltonian @ wavefunction) / wavefunction
    return params, config, states, weights, local_energy


def test_n6_exact_head_projection_bounds_and_natural_gradient_identity():
    params, config, states, weights, local_energy = _n6_exact_problem()
    derivative = materialize_message_head_log_derivatives(
        params, config, states, autodiff_block_size=4
    )
    tangent = derivative.group("all")
    projection = project_born_complex_tangent(tangent, weights, local_energy)

    assert tangent.shape[0] == 5
    assert tangent.shape[1] > 0
    assert np.all(np.isfinite(tangent))
    assert 0.0 <= projection.projection_fraction <= 1.0
    assert 0 <= projection.numerical_rank <= 2 * (len(states) - 1)
    np.testing.assert_allclose(
        projection.direct_normal_equation_response,
        projection.projected_norm_squared,
        rtol=2e-10,
        atol=2e-12,
    )

    response = local_energy_tangent_response(tangent, weights, local_energy)
    np.testing.assert_allclose(
        response["predicted_energy_decrease_per_unit_natural_gradient_step"],
        2.0 * response["captured_local_energy_variance"],
        rtol=2e-12,
        atol=2e-12,
    )
    assert response["interpretation"].endswith("not_training_recovery")


def test_n6_exact_svd_projection_agrees_with_direct_least_squares():
    params, config, states, weights, local_energy = _n6_exact_problem()
    derivative = materialize_message_head_log_derivatives(
        params,
        config,
        states,
        head_names=HEAD_GROUPS["vector"],
        autodiff_block_size=5,
    )
    projection = project_born_complex_tangent(
        derivative.group("vector"),
        weights,
        local_energy,
        relative_rank_cutoff=1.0e-10,
    )
    direct, *_ = np.linalg.lstsq(
        projection.design_matrix,
        projection.target_vector,
        rcond=1.0e-10,
    )
    np.testing.assert_allclose(
        projection.projected_target_vector,
        projection.design_matrix @ direct,
        rtol=3e-10,
        atol=3e-11,
    )


def test_zero_tangent_has_zero_rank_and_zero_physical_response():
    weights = np.asarray([0.03, 0.17, 0.22, 0.31, 0.27])
    target = np.asarray([0.2, -0.1j, 0.4 + 0.2j, -0.3, 0.1j])
    tangent = np.zeros((5, 7), dtype=np.complex128)
    projection = project_born_complex_tangent(tangent, weights, target)

    assert projection.numerical_rank == 0
    assert projection.projection_fraction == 0.0
    assert projection.projected_norm_squared == 0.0
    assert projection.direct_normal_equation_response == 0.0
    assert projection.tangent_condition_number is None
    assert projection.fisher_condition_number is None


def test_projection_is_invariant_under_nonsingular_reparameterization():
    rng = np.random.default_rng(822)
    weights = rng.uniform(0.1, 1.0, size=5)
    tangent = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    target = rng.normal(size=5) + 1j * rng.normal(size=5)
    mixing = np.asarray(
        [[1.3, 0.2, -0.1], [0.1, 0.8, 0.3], [-0.2, 0.1, 1.1]]
    )
    assert abs(np.linalg.det(mixing)) > 0.5

    original = project_born_complex_tangent(tangent, weights, target)
    changed = project_born_complex_tangent(tangent @ mixing, weights, target)

    assert original.numerical_rank == changed.numerical_rank
    np.testing.assert_allclose(
        original.projected_target_vector,
        changed.projected_target_vector,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        original.projection_fraction,
        changed.projection_fraction,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        original.projected_norm_squared,
        changed.projected_norm_squared,
        rtol=2e-12,
        atol=2e-12,
    )


def test_damped_natural_gradient_response_matches_direct_fisher_solve():
    rng = np.random.default_rng(510)
    weights = rng.uniform(0.2, 1.0, size=5)
    tangent = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    target = rng.normal(size=5) + 1j * rng.normal(size=5)
    damping = 3.0e-3
    factorization = factor_born_complex_tangent(tangent, weights)
    projection = factorization.project(target)
    response = factorization.damped_natural_gradient_response(
        target, damping
    )
    design = projection.design_matrix
    rhs = design.T @ projection.target_vector
    direct = np.linalg.solve(
        design.T @ design + damping * np.eye(design.shape[1]), rhs
    )

    np.testing.assert_allclose(
        response["predicted_energy_decrease_per_unit_damped_step"],
        rhs @ direct,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        response["parameter_direction_l2_norm"],
        np.linalg.norm(direct),
        rtol=2e-12,
        atol=2e-12,
    )
