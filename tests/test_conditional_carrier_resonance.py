"""Algebra/AD tests, not new variational training or sampled physics."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from su2_nqs.models.conditional_carrier_resonance import (
    ACTIVE_DIAGONAL_COORDINATES, conditional_filter_coefficients,
    conditional_filter_matrix, conditional_reduced_generators,
    factorized_to_conditional, diagonal_conditioned_to_conditional,
    local_filter_rows, local_scalar_basis, local_scalar_components, packed_scalar_rows,
)


def test_complete_spectator_projectors_and_active_reduced_basis():
    parts = local_scalar_components()
    assert parts.channels == ((0, 0), (0, 1), (2, 0), (2, 1), (2, 2), (4, 0))
    np.testing.assert_allclose(parts.spectator_projectors.sum(axis=0), np.eye(16), atol=2e-14)
    for z, p in enumerate(parts.spectator_projectors):
        np.testing.assert_allclose(p@p, p, atol=2e-14)
        np.testing.assert_allclose(np.trace(p), parts.channels[z][0]+1, atol=2e-14)
        for w, other in enumerate(parts.spectator_projectors):
            if w != z: np.testing.assert_allclose(p@other, 0., atol=2e-14)
    np.testing.assert_allclose(parts.active_basis[list(ACTIVE_DIAGONAL_COORDINATES)].sum(axis=0), np.eye(16), atol=2e-14)
    gram = sum(np.einsum('rab,sab->rs', b, b) for b in parts.reduced_basis.values())
    np.testing.assert_allclose(gram, np.eye(10), atol=2e-15)


def test_identity_is_exact_in_coefficients_matrix_and_packed_rows():
    zero = jnp.zeros(60)
    np.testing.assert_array_equal(conditional_filter_coefficients(zero), np.zeros(60))
    np.testing.assert_array_equal(conditional_filter_matrix(zero), np.eye(256))
    np.testing.assert_array_equal(conditional_filter_matrix(zero, delta_only=True), np.zeros((256, 256)))
    packed = packed_scalar_rows()
    rows = np.asarray(local_filter_rows(zero))
    np.testing.assert_array_equal(rows.sum(axis=1), np.ones(256))
    assert np.all(rows[~packed.mask] == 0)
    assert np.all(packed.columns[np.arange(256), packed.identity_positions] == np.arange(256))


def test_reduced_exponentials_match_independent_physical_reconstruction():
    rng = np.random.default_rng(2026101001); theta = rng.normal(scale=.12, size=(6, 10))
    parts = local_scalar_components(); generators = conditional_reduced_generators(theta)
    coefficients = np.zeros((6, 10))
    for q, basis in parts.reduced_basis.items():
        for z in range(6): coefficients[z] += np.einsum('rab,ab->r', basis, expm(np.asarray(generators[q][z]))-np.eye(basis.shape[-1]))
    expected = np.eye(256)+np.einsum('r,rij->ij', coefficients.reshape(60), local_scalar_basis())
    actual = np.asarray(conditional_filter_matrix(theta))
    np.testing.assert_allclose(actual, expected, atol=8e-15, rtol=1e-13)
    np.testing.assert_allclose(actual, actual.T, atol=2e-14)
    assert np.linalg.eigvalsh(actual).min() > 0


def test_packed_rows_match_all_dense_basis_entries_and_complex_action():
    packed = packed_scalar_rows(); basis = local_scalar_basis()
    assert packed.columns.shape == (256, 36)
    assert packed.mask.sum(axis=1).max() == 36
    np.testing.assert_allclose(packed.coefficients,
        basis[:, np.arange(256)[:, None], packed.columns]*packed.mask[None], atol=0, rtol=0)
    rng = np.random.default_rng(2026101002)
    theta = rng.normal(scale=.07, size=60)
    carrier = rng.normal(size=256)+1j*rng.normal(size=256)
    observed = np.sum(np.asarray(local_filter_rows(theta))*carrier[packed.columns], axis=1)
    np.testing.assert_allclose(observed, np.asarray(conditional_filter_matrix(theta))@carrier, atol=2e-14)
    for row, inputs in enumerate(packed.columns):
        for col in inputs[packed.mask[row]]:
            assert (int(row)//16).bit_count() == (int(col)//16).bit_count()
            assert (int(row)%16).bit_count() == (int(col)%16).bit_count()


def test_factorized_null_and_diagonal_conditioned_null_are_exactly_nested():
    rng = np.random.default_rng(2026101003); p = rng.normal(scale=.08, size=16)
    theta = factorized_to_conditional(p)
    parts = local_scalar_components()
    active = expm(np.einsum('r,rij->ij', p[:10], parts.active_basis))
    spectator = np.einsum('z,zij->ij', np.exp(p[10:]), parts.spectator_projectors)
    np.testing.assert_allclose(conditional_filter_matrix(theta), np.kron(active, spectator), atol=2e-14)
    diagonal = np.concatenate((p[:10], np.repeat(p[10:], 6)))
    np.testing.assert_array_equal(diagonal_conditioned_to_conditional(diagonal), theta)
    diagonal = rng.normal(scale=.1, size=46)
    embedded = np.asarray(diagonal_conditioned_to_conditional(diagonal))
    expected = np.tile(diagonal[:10], (6, 1))
    expected[:, ACTIVE_DIAGONAL_COORDINATES] += diagonal[10:].reshape(6, 6)
    np.testing.assert_array_equal(embedded, expected)


def test_finite_autodiff_at_zero_and_finite_parameters():
    rng = np.random.default_rng(2026101004)
    direction = jnp.asarray(rng.normal(size=60)); weights = jnp.asarray(rng.normal(size=60))
    objective = jax.jit(lambda theta:jnp.vdot(weights, conditional_filter_coefficients(theta)).real)
    for point in (jnp.zeros(60), jnp.asarray(rng.normal(scale=.1, size=60))):
        gradient = jax.grad(objective)(point)
        assert np.isfinite(np.asarray(gradient)).all()
        derivative = float(jnp.vdot(gradient, direction))
        eps = 1e-6
        finite = float((objective(point+eps*direction)-objective(point-eps*direction))/(2*eps))
        np.testing.assert_allclose(derivative, finite, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(jax.jacfwd(conditional_filter_coefficients)(jnp.zeros(60)), np.eye(60), atol=2e-14)


def test_conditional_offdiagonal_is_outside_factorized_generator_tangent():
    parts = local_scalar_components()
    # P_(q0,a0) on the spectator controlling the q0 active mixing generator.
    conditional = np.kron(parts.active_basis[2], parts.spectator_projectors[0]).reshape(-1)
    basis = np.column_stack([np.kron(a, np.eye(16)).reshape(-1) for a in parts.active_basis]+
        [np.kron(np.eye(16), p).reshape(-1) for p in parts.spectator_projectors])
    coeff, *_ = np.linalg.lstsq(basis, conditional, rcond=1e-12)
    relative = np.linalg.norm(conditional-basis@coeff)/np.linalg.norm(conditional)
    assert relative > .5


@pytest.mark.parametrize('bad', [np.zeros(59), np.zeros((6, 9)), np.zeros(60, dtype=complex), np.full(60, np.nan)])
def test_invalid_conditional_parameters_fail_closed(bad):
    with pytest.raises(ValueError): conditional_filter_coefficients(bad)
