from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.qsl_three_arm import (
    triangular_pi_flux_operator_components,
)
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import (
    SMALL_CASES,
    ordered_gamma_a1_group,
    spin_raising_relative_norm,
)
from su2_nqs.benchmarks.clebsch_tree_exact import (
    physical_permutation_index_table,
)
from su2_nqs.models.su2_a1_operator_vit import (
    SU2A1OperatorViTConfig,
    init_su2_a1_operator_vit,
    su2_a1_operator_vit_amplitudes,
    su2_a1_operator_vit_coefficients,
    su2_a1_operator_vit_component_weights,
    su2_a1_operator_vit_parameter_count,
)


def _config() -> SU2A1OperatorViTConfig:
    return SU2A1OperatorViTConfig(
        operator_labels=("bond", "ring"),
        operator_descriptors=((1.0, 0.0), (2.0, 1.0)),
        embedding_dim=4,
        coefficient_scale=0.4,
    )


def test_operator_vit_component_expansion_matches_two_residual_layers() -> None:
    config = _config()
    params = init_su2_a1_operator_vit(jax.random.PRNGKey(7), config)
    coefficients = np.asarray(su2_a1_operator_vit_coefficients(params, config))
    first, second = coefficients
    operators = (
        np.asarray([[0.2, 0.7], [0.3, -0.1]]),
        np.asarray([[-0.4, 0.2], [0.8, 0.5]]),
    )
    carrier = np.asarray([0.6, -0.3])
    first_columns = [operator @ carrier for operator in operators]
    second_columns = [
        outer @ inner
        for outer in operators
        for inner in first_columns
    ]
    components = np.column_stack((carrier, *first_columns, *second_columns))
    model = np.asarray(
        su2_a1_operator_vit_amplitudes(params, jnp.asarray(components), config)
    )
    explicit = carrier.copy()
    explicit += sum(value * column for value, column in zip(first, first_columns))
    intermediate = carrier + sum(
        value * column for value, column in zip(first, first_columns)
    )
    explicit += sum(
        value * (operator @ intermediate)
        for value, operator in zip(second, operators)
    )
    assert np.allclose(model, explicit, atol=2.0e-13)
    weights = np.asarray(su2_a1_operator_vit_component_weights(params, config))
    assert weights.shape == (config.component_count,)
    assert su2_a1_operator_vit_parameter_count(params) > 0


def test_physical_operator_components_remain_singlet_gamma_a1() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    residuals = spin_raising_relative_norm(
        data.component_matrix, data.integer_basis, 16
    )
    assert np.max(residuals) < 3.0e-11
    basis = tuple(tuple(int(bit) for bit in row) for row in data.configurations)
    indices = physical_permutation_index_table(
        basis, ordered_gamma_a1_group(SMALL_CASES["square4"])
    )
    projected = np.mean(data.component_matrix[indices], axis=0)
    relative = np.linalg.norm(projected - data.component_matrix) / np.linalg.norm(
        data.component_matrix
    )
    assert relative < 3.0e-12
    assert data.component_matrix.shape == (12870, 31)
    assert data.scalar_operator_union_nnz > data.hamiltonian_nnz
    assert data.scalar_operator_union_nnz < 2 * data.hamiltonian_nnz
