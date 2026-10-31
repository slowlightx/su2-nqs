import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.hubbard_tree_control import (
    evaluate_physical_tree_exact,
    exact_physical_tree_energy,
    reconstruct_physical_tree_state,
)
from su2_nqs.models.hubbard_abelian_tree import (
    HubbardAbelianTreeConfig,
    hubbard_abelian_parameter_count,
    hubbard_abelian_tree_amplitude,
    init_hubbard_abelian_tree,
)
from su2_nqs.models.hubbard_cg import (
    HubbardCGConfig,
    active_hubbard_cg_amplitude,
    hubbard_cg_parameter_count,
    init_active_hubbard_cg,
)
from su2_nqs.operators.hubbard_fock import hubbard_fock_sector_matrix


def test_abelian_tree_has_exact_charge_magnetization_support_and_gradients():
    config = HubbardAbelianTreeConfig(4, 4, target_m2=0, feature_dim=2)
    params = init_hubbard_abelian_tree(jax.random.PRNGKey(10), config)
    assert hubbard_abelian_tree_amplitude(
        params, jnp.asarray((1, 2, 1, 2)), config
    ) != 0.0
    assert hubbard_abelian_tree_amplitude(
        params, jnp.asarray((1, 1, 2, 0)), config
    ) == 0.0
    bonds = ((0, 1), (1, 2), (2, 3))
    hamiltonian, _ = hubbard_fock_sector_matrix(
        4, 4, 0, bonds, interaction=4.0
    )
    loss = lambda values: exact_physical_tree_energy(
        values,
        config,
        jnp.asarray(hamiltonian),
        n_sites=4,
        n_electrons=4,
        total_m2=0,
        amplitude_fn=hubbard_abelian_tree_amplitude,
    )
    value, gradient = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(value))
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradient)
    )


def test_abelian_control_is_not_accidentally_spin_projected():
    config = HubbardAbelianTreeConfig(4, 4, target_m2=0, feature_dim=2)
    params = init_hubbard_abelian_tree(jax.random.PRNGKey(11), config)
    metrics = evaluate_physical_tree_exact(
        params,
        config,
        ((0, 1), (1, 2), (2, 3)),
        n_sites=4,
        n_electrons=4,
        total_m2=0,
        interaction=4.0,
        amplitude_fn=hubbard_abelian_tree_amplitude,
    )
    assert metrics["spin_squared_expectation"] > 1.0e-4


def test_same_width_cg_uses_fewer_active_parameters_than_abelian_control():
    cg_config = HubbardCGConfig(6, 6, feature_dim=3)
    abelian_config = HubbardAbelianTreeConfig(6, 6, feature_dim=3)
    cg = init_active_hubbard_cg(jax.random.PRNGKey(12), cg_config)
    abelian = init_hubbard_abelian_tree(jax.random.PRNGKey(12), abelian_config)
    assert hubbard_cg_parameter_count(cg) == 2001
    assert hubbard_abelian_parameter_count(abelian) == 3315


def test_physical_tree_reconstruction_uses_common_fock_basis():
    cg_config = HubbardCGConfig(3, 3, target_q=1, target_m2=1, feature_dim=1)
    abelian_config = HubbardAbelianTreeConfig(3, 3, target_m2=1, feature_dim=1)
    cg = init_active_hubbard_cg(jax.random.PRNGKey(13), cg_config)
    abelian = init_hubbard_abelian_tree(jax.random.PRNGKey(13), abelian_config)
    _, cg_basis = reconstruct_physical_tree_state(
        cg,
        cg_config,
        n_sites=3,
        n_electrons=3,
        total_m2=1,
        amplitude_fn=active_hubbard_cg_amplitude,
    )
    _, abelian_basis = reconstruct_physical_tree_state(
        abelian,
        abelian_config,
        n_sites=3,
        n_electrons=3,
        total_m2=1,
        amplitude_fn=hubbard_abelian_tree_amplitude,
    )
    assert cg_basis == abelian_basis
