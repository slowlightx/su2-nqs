import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.hubbard_cg_exact import (
    evaluate_hubbard_cg_exact,
    exact_hubbard_cg_energy,
    hubbard_spin_squared_matrix,
    hubbard_spin_squared_sector_sparse,
    reconstruct_hubbard_cg_state,
)
from su2_nqs.models.hubbard_cg import (
    HubbardCGConfig,
    active_hubbard_cg_amplitude,
    hubbard_cg_amplitude,
    hubbard_cg_parameter_count,
    hubbard_cg_tree_schedule,
    init_hubbard_cg,
    pack_hubbard_cg_parameters,
)
from su2_nqs.operators.hubbard_fock import hubbard_fock_sector_matrix


@pytest.mark.parametrize(
    ("n_sites", "n_electrons", "target_q", "target_m2"),
    [
        (2, 2, 0, 0),
        (3, 3, 1, 1),
        (4, 2, 0, 0),
        (5, 3, 1, -1),
        (6, 4, 2, 0),
    ],
)
def test_random_hubbard_cg_state_has_exact_total_spin(
    n_sites, n_electrons, target_q, target_m2
):
    config = HubbardCGConfig(
        n_sites=n_sites,
        n_electrons=n_electrons,
        target_q=target_q,
        target_m2=target_m2,
        feature_dim=2,
    )
    params = init_hubbard_cg(jax.random.PRNGKey(100 + n_sites), config)
    amplitudes, basis = reconstruct_hubbard_cg_state(params, config)
    spin_squared, spin_basis = hubbard_spin_squared_matrix(
        n_sites, n_electrons, total_m2=target_m2
    )
    assert basis == spin_basis
    vector = np.array(amplitudes)
    vector /= np.linalg.norm(vector)
    spin = 0.5 * target_q
    residual = spin_squared @ vector - spin * (spin + 1.0) * vector
    assert np.linalg.norm(residual) < 1.0e-11


def test_different_magnetic_components_share_norm_and_hubbard_energy():
    base = dict(n_sites=3, n_electrons=3, target_q=1, feature_dim=2)
    positive = HubbardCGConfig(target_m2=1, **base)
    negative = HubbardCGConfig(target_m2=-1, **base)
    params = init_hubbard_cg(jax.random.PRNGKey(14), positive)
    bonds = ((0, 1), (1, 2))
    first = evaluate_hubbard_cg_exact(params, positive, bonds, interaction=4.0)
    second = evaluate_hubbard_cg_exact(params, negative, bonds, interaction=4.0)
    assert first["norm_before_normalization"] == pytest.approx(
        second["norm_before_normalization"], abs=1.0e-13
    )
    assert first["energy"] == pytest.approx(second["energy"], abs=1.0e-12)


def test_amplitude_is_zero_outside_declared_charge_or_magnetic_sector():
    config = HubbardCGConfig(3, 2, target_q=0, target_m2=0, feature_dim=1)
    params = init_hubbard_cg(jax.random.PRNGKey(4), config)
    assert hubbard_cg_amplitude(params, jnp.asarray((1, 2, 0)), config) != 0.0
    assert hubbard_cg_amplitude(params, jnp.asarray((1, 1, 0)), config) == 0.0
    assert hubbard_cg_amplitude(params, jnp.asarray((3, 1, 0)), config) == 0.0


def test_exact_energy_is_differentiable_after_h0_reconstruction():
    config = HubbardCGConfig(2, 2, target_q=0, target_m2=0, feature_dim=1)
    params = init_hubbard_cg(jax.random.PRNGKey(9), config)
    hamiltonian = jnp.asarray(
        [[4.0, -1.0, -1.0, 0.0], [-1.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, -1.0], [0.0, -1.0, -1.0, 4.0]],
        dtype=jnp.float64,
    )
    # Use the actual physical basis dimension rather than relying on this
    # illustrative matrix ordering.
    amplitudes, _ = reconstruct_hubbard_cg_state(params, config)
    hamiltonian = jnp.eye(amplitudes.shape[0], dtype=jnp.float64)
    value, gradients = jax.value_and_grad(exact_hubbard_cg_energy)(
        params, config, hamiltonian
    )
    assert np.isfinite(float(value))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))


def test_balanced_schedule_is_deterministic_for_non_power_of_two_size():
    assert hubbard_cg_tree_schedule(5) == (
        (0, 1, 2, 0),
        (3, 4, 5, 1),
        (2, 3, 5, 2),
        (0, 2, 5, 3),
    )


@pytest.mark.parametrize(
    ("n_sites", "n_electrons", "target_q", "target_m2"),
    [(3, 3, 1, -1), (4, 2, 0, 0), (6, 6, 0, 0)],
)
def test_active_channel_pack_replays_dense_amplitudes_exactly(
    n_sites, n_electrons, target_q, target_m2
):
    config = HubbardCGConfig(
        n_sites,
        n_electrons,
        target_q=target_q,
        target_m2=target_m2,
        feature_dim=2,
    )
    dense = init_hubbard_cg(jax.random.PRNGKey(77 + n_sites), config)
    active = pack_hubbard_cg_parameters(dense, config)
    dense_vector, basis = reconstruct_hubbard_cg_state(dense, config)
    active_vector, active_basis = reconstruct_hubbard_cg_state(
        active, config, amplitude_fn=active_hubbard_cg_amplitude
    )
    assert basis == active_basis
    np.testing.assert_allclose(active_vector, dense_vector, atol=2.0e-15, rtol=2.0e-14)


def test_active_channel_pack_replays_energy_h2_gradients_and_jvp():
    config = HubbardCGConfig(4, 4, target_q=0, target_m2=0, feature_dim=2)
    bonds = ((0, 1), (1, 2), (2, 3))
    hamiltonian, _ = hubbard_fock_sector_matrix(
        4, 4, 0, bonds, interaction=4.0
    )
    hamiltonian = jnp.asarray(hamiltonian)
    dense = init_hubbard_cg(jax.random.PRNGKey(81), config)
    active = pack_hubbard_cg_parameters(dense, config)
    dense_loss = lambda values: exact_hubbard_cg_energy(
        values, config, hamiltonian
    )
    active_loss = lambda values: exact_hubbard_cg_energy(
        values,
        config,
        hamiltonian,
        amplitude_fn=active_hubbard_cg_amplitude,
    )
    dense_energy, dense_gradient = jax.value_and_grad(dense_loss)(dense)
    active_energy, active_gradient = jax.value_and_grad(active_loss)(active)
    packed_gradient = pack_hubbard_cg_parameters(dense_gradient, config)
    np.testing.assert_allclose(active_energy, dense_energy, atol=2.0e-14)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(active_gradient),
        jax.tree_util.tree_leaves(packed_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=2.0e-11)

    basis = reconstruct_hubbard_cg_state(dense, config)[1]
    batch = jnp.asarray(basis, dtype=jnp.int32)
    dense_vector_fn = lambda values: jax.vmap(
        lambda tokens: hubbard_cg_amplitude(values, tokens, config)
    )(batch)
    active_vector_fn = lambda values: jax.vmap(
        lambda tokens: active_hubbard_cg_amplitude(values, tokens, config)
    )(batch)
    dense_tangent = jax.tree_util.tree_map(jnp.ones_like, dense)
    active_tangent = pack_hubbard_cg_parameters(dense_tangent, config)
    dense_vector, dense_jvp = jax.jvp(dense_vector_fn, (dense,), (dense_tangent,))
    active_vector, active_jvp = jax.jvp(
        active_vector_fn, (active,), (active_tangent,)
    )
    np.testing.assert_allclose(active_vector, dense_vector, atol=2.0e-15, rtol=2.0e-14)
    np.testing.assert_allclose(active_jvp, dense_jvp, atol=2.0e-13, rtol=2.0e-12)
    np.testing.assert_allclose(
        hamiltonian @ active_vector,
        hamiltonian @ dense_vector,
        atol=2.0e-14,
        rtol=2.0e-13,
    )


def test_active_channels_remove_dense_unreachable_parameter_overhead():
    config = HubbardCGConfig(6, 6, target_q=0, target_m2=0, feature_dim=3)
    dense = init_hubbard_cg(jax.random.PRNGKey(92), config)
    active = pack_hubbard_cg_parameters(dense, config)
    assert hubbard_cg_parameter_count(active) < 0.2 * hubbard_cg_parameter_count(dense)


@pytest.mark.parametrize(
    ("n_sites", "n_electrons", "total_m2"),
    ((2, 2, 0), (3, 3, 1), (4, 2, 0), (4, 4, -2)),
)
def test_sparse_sector_spin_squared_matches_independent_dense_reference(
    n_sites, n_electrons, total_m2
):
    dense, dense_basis = hubbard_spin_squared_matrix(
        n_sites, n_electrons, total_m2=total_m2
    )
    sparse, sparse_basis = hubbard_spin_squared_sector_sparse(
        n_sites, n_electrons, total_m2=total_m2
    )
    assert sparse_basis == dense_basis
    np.testing.assert_allclose(sparse.toarray(), dense, atol=2.0e-15)
