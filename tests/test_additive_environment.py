"""Independent coordinates/rank/AD checks; no N36 amplitudes or simulation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.additive_environment import (
    ADDITIVE_ENVIRONMENT_PARAMETER_COUNT, additive_environment_basis,
    additive_environment_coordinates, additive_environment_generators,
)
from su2_nqs.models.multicontext_resonance_attention import (
    free_multicontext_generators, interaction_contrast, multicontext_active_filters,
    restrict_multicontext_generators,
)

jax.config.update('jax_enable_x64', True)


def test_basis_is_complete_orthonormal_and_has_no_redundant_coordinate():
    basis = additive_environment_basis()
    assert basis.shape == (11, 6, 6) and not basis.flags.writeable
    gram = basis.reshape(11, 36) @ basis.reshape(11, 36).T
    np.testing.assert_allclose(gram, np.eye(11), atol=5e-16)
    # Independently form the row/column additive incidence map; its only
    # redundancy per active coordinate is the common constant shift.
    incidence = np.concatenate((np.repeat(np.eye(6), 6, axis=0), np.tile(np.eye(6), (6, 1))), axis=1)
    assert np.linalg.matrix_rank(incidence) == 11
    assert ADDITIVE_ENVIRONMENT_PARAMETER_COUNT == 110
    np.testing.assert_allclose(basis.reshape(11, 36).T @ basis.reshape(11, 36) @ incidence,
                               incidence, atol=8e-16)


def test_arbitrary_additive_table_roundtrip_and_frobenius_isometry():
    rng = np.random.default_rng(14001)
    left, right = rng.normal(size=(6, 10)), rng.normal(size=(6, 10))
    target = left[:, None, :] + right[None, :, :]
    coordinates = additive_environment_coordinates(target)
    observed = additive_environment_generators(coordinates)
    error = np.max(abs(np.asarray(observed) - target))
    np.testing.assert_allclose(observed, target, atol=3e-15)
    np.testing.assert_allclose(np.linalg.norm(coordinates), np.linalg.norm(target), atol=8e-15)
    theta = rng.normal(size=110)
    np.testing.assert_allclose(additive_environment_coordinates(additive_environment_generators(theta)).reshape(110), theta, atol=1e-15)
    print('additive110 table roundtrip maximum absolute error', error)


def test_projection_matches_independent_row_column_formula_and_existing_control():
    table = np.random.default_rng(14002).normal(size=(6, 6, 10))
    projected = np.asarray(additive_environment_generators(additive_environment_coordinates(table)))
    expected = table.mean(axis=0, keepdims=True) + table.mean(axis=1, keepdims=True) - table.mean(axis=(0, 1), keepdims=True)
    np.testing.assert_allclose(projected, expected, atol=9e-16)
    np.testing.assert_allclose(projected, restrict_multicontext_generators(table, control='additive'), atol=9e-16)
    np.testing.assert_allclose(interaction_contrast(projected), 0., atol=7e-16)
    discarded = table - projected
    np.testing.assert_allclose(discarded.mean(axis=0), 0., atol=1e-15)
    np.testing.assert_allclose(discarded.mean(axis=1), 0., atol=1e-15)
    np.testing.assert_allclose(np.vdot(discarded, projected), 0., atol=2e-14)


def test_full110_jacobian_rank_and_transpose_gradient():
    theta = jnp.asarray(np.random.default_rng(14003).normal(size=110))
    jacobian = np.asarray(jax.jacfwd(lambda t: additive_environment_generators(t).reshape(360))(theta))
    assert jacobian.shape == (360, 110)
    np.testing.assert_allclose(jacobian.T @ jacobian, np.eye(110), atol=7e-16)
    weight = np.random.default_rng(14004).normal(size=(6, 6, 10))
    gradient = jax.grad(lambda t: jnp.sum(additive_environment_generators(t) * weight))(theta)
    np.testing.assert_allclose(gradient, additive_environment_coordinates(weight).reshape(110), atol=9e-16)
    direction = np.random.default_rng(14005).normal(size=110)
    direction /= np.linalg.norm(direction)
    loss = lambda t: jnp.sum(jnp.sin(additive_environment_generators(t)) * weight)
    derivative = float(jax.jvp(loss, (theta,), (jnp.asarray(direction),))[1])
    epsilon = 1e-5
    finite = float((loss(theta + epsilon * direction) - loss(theta - epsilon * direction)) / (2 * epsilon))
    np.testing.assert_allclose(derivative, finite, atol=5e-10)
    print('additive110 nonlinear directional derivative FD error', abs(derivative - finite))


def test_zero_and_free360_replay_preserve_existing_filter_interface():
    zero = additive_environment_generators(np.zeros(110))
    np.testing.assert_array_equal(zero, np.zeros((6, 6, 10)))
    np.testing.assert_array_equal(multicontext_active_filters(zero), np.broadcast_to(np.eye(16), (6, 6, 16, 16)))
    table = additive_environment_generators(np.random.default_rng(14006).normal(size=110) * .03)
    np.testing.assert_array_equal(free_multicontext_generators(table.reshape(360)), table)
    assert np.linalg.norm(np.asarray(table) - np.asarray(table).mean(axis=0, keepdims=True)) > .1
    assert np.linalg.norm(np.asarray(table) - np.asarray(table).mean(axis=1, keepdims=True)) > .1


@pytest.mark.parametrize('bad', [np.zeros(120), np.zeros((12, 10)), np.zeros(110, complex), np.full(110, np.inf)])
def test_invalid_coordinates_fail_closed(bad):
    with pytest.raises(ValueError):
        additive_environment_generators(bad)


def test_invalid_tables_and_dynamic_nonfinite_fail_closed():
    for bad in (np.zeros((6, 5, 10)), np.zeros((6, 6, 10), complex), np.full((6, 6, 10), np.nan)):
        with pytest.raises(ValueError):
            additive_environment_coordinates(bad)
    assert np.isnan(np.asarray(jax.jit(additive_environment_generators)(jnp.full(110, jnp.nan)))).all()
    assert np.isnan(np.asarray(jax.jit(additive_environment_coordinates)(jnp.full((6, 6, 10), jnp.inf)))).all()
