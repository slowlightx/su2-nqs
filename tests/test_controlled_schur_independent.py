"""Independent Pauli/exponential oracles for the local scalar carrier gate.

No energies, ED ground states, or optimization results are used here.  The
small D8 example tests complete spatial projection, not a triangular lattice
benchmark or a claim that the single-filter manifold is unrestricted.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from su2_nqs.models.conditional_carrier_resonance import (
    conditional_filter_coefficients,
    conditional_filter_matrix,
    diagonal_conditioned_to_conditional,
    factorized_to_conditional,
    local_filter_rows,
    local_scalar_basis,
    local_scalar_components,
    packed_scalar_rows,
)


def _spin_operators(n_sites):
    """Direct Pauli sums: zero=down and site zero is the MSB."""
    dim = 1 << n_sites
    raising = np.zeros((dim, dim))
    z = np.zeros(dim)
    for old in range(dim):
        z[old] = old.bit_count() - n_sites / 2
        for site in range(n_sites):
            mask = 1 << (n_sites - 1 - site)
            if not old & mask:
                raising[old | mask, old] = 1.
    return raising, raising.T, np.diag(z)


def _permutation_indices(site_images):
    """Map each old computational index to its permuted index."""
    n_sites = len(site_images)
    indices = np.zeros(1 << n_sites, dtype=int)
    for old in range(len(indices)):
        for site, image in enumerate(site_images):
            bit = (old >> (n_sites - 1 - site)) & 1
            indices[old] |= bit << (n_sites - 1 - image)
    return indices


def _d8_maps():
    return tuple(_permutation_indices([(sign * s + shift) % 8 for s in range(8)])
                 for sign in (1, -1) for shift in range(8))


def _project_group(vector, maps):
    output = np.zeros_like(vector)
    for mapping in maps:
        output[mapping] += vector
    return output / len(maps)


def test_direct_pauli_su2_not_merely_casimir_commutation():
    theta = np.random.default_rng(82001).normal(scale=.12, size=60)
    matrix = np.asarray(conditional_filter_matrix(theta))
    spin = _spin_operators(8)
    errors = [np.max(abs(matrix @ s - s @ matrix)) for s in spin]
    assert max(errors) < 3e-13
    # A complete spectator multiplet projector must also commute with S±;
    # a fixed-m projector could incorrectly pass only the S² or Sz test.
    spectator_errors = [np.max(abs(p @ s - s @ p))
                        for p in local_scalar_components().spectator_projectors
                        for s in _spin_operators(4)]
    assert max(spectator_errors) < 3e-14
    print(f"independent Pauli commutator max={max(errors):.3e}; "
          f"spectator max={max(spectator_errors):.3e}")


def test_full_256_dimensional_exponential_and_finite_operator_span():
    rng = np.random.default_rng(82002)
    theta = rng.normal(scale=.10, size=60)
    basis = local_scalar_basis()
    dense_generator = np.einsum("r,rij->ij", theta, basis)
    expected = expm(dense_generator)
    actual = np.asarray(conditional_filter_matrix(theta))
    error = np.max(abs(actual - expected))
    assert error < 3e-13
    span_replay = np.eye(256) + np.einsum(
        "r,rij->ij", np.asarray(conditional_filter_coefficients(theta)), basis)
    np.testing.assert_allclose(actual, span_replay, atol=2e-14, rtol=2e-14)
    # Exponentiation is a nonlinear parameterization, but cannot escape this
    # fixed operator span.  An arbitrary complex carrier need not share nodes.
    carrier = rng.normal(size=256) + 1j * rng.normal(size=256)
    carrier[::7] = 0.
    packed = packed_scalar_rows()
    action = np.sum(np.asarray(local_filter_rows(theta)) * carrier[packed.columns], axis=1)
    action_error = np.max(abs(action - expected @ carrier))
    assert action_error < 4e-13
    assert np.linalg.norm(action[::7]) > .01
    print(f"full dense exp max={error:.3e}; node-safe action max={action_error:.3e}")


def test_exponential_jvp_against_independent_full_dense_finite_difference():
    rng = np.random.default_rng(82003)
    theta = rng.normal(scale=.08, size=60)
    direction = rng.normal(size=60)
    direction /= np.linalg.norm(direction)
    basis = local_scalar_basis()
    matrix_fn = lambda t: conditional_filter_matrix(t)
    _, derivative = jax.jvp(matrix_fn, (jnp.asarray(theta),), (jnp.asarray(direction),))
    generator = np.einsum("r,rij->ij", theta, basis)
    tangent = np.einsum("r,rij->ij", direction, basis)
    epsilon = 2e-6
    independent = (expm(generator + epsilon * tangent) - expm(generator - epsilon * tangent)) / (2 * epsilon)
    error = np.max(abs(np.asarray(derivative) - independent))
    assert error < 5e-9
    _, zero_derivative = jax.jvp(matrix_fn, (jnp.zeros(60),), (jnp.asarray(direction),))
    zero_error = np.max(abs(np.asarray(zero_derivative) - tangent))
    assert zero_error < 5e-14
    assert np.linalg.norm(zero_derivative) > .1
    print(f"dense-exponential JVP FD max={error:.3e}; identity derivative max={zero_error:.3e}")


def test_null_parameter_gauges_and_nested_generator_dimensions():
    # These ranks are generator-coordinate capacities, NOT physical-state
    # ranks on any chosen carrier, and not the dimensions after normalization.
    factor_map = np.column_stack([
        np.asarray(factorized_to_conditional(e)).reshape(-1) for e in np.eye(16)])
    diagonal_map = np.column_stack([
        np.asarray(diagonal_conditioned_to_conditional(e)).reshape(-1)
        for e in np.eye(46)])
    assert np.linalg.matrix_rank(factor_map, tol=1e-12) == 15
    assert np.linalg.matrix_rank(diagonal_map, tol=1e-12) == 40
    recovered, *_ = np.linalg.lstsq(diagonal_map, factor_map, rcond=1e-12)
    assert np.max(abs(diagonal_map @ recovered - factor_map)) < 1e-13


def test_complete_d8_projection_preserves_singlet_with_nonzero_ray_tangent():
    maps = _d8_maps()
    singlet = np.array([0., -1., 1., 0.]) / np.sqrt(2.)
    carrier = singlet
    for _ in range(3):
        carrier = np.kron(carrier, singlet)
    carrier = _project_group(carrier, maps)
    carrier /= np.linalg.norm(carrier)
    theta = np.random.default_rng(82004).normal(scale=.08, size=60)
    delta = np.asarray(conditional_filter_matrix(theta, delta_only=True)) @ carrier
    state = carrier + _project_group(delta, maps)
    symmetry_errors = [np.max(abs(state[mapping] - state)) for mapping in maps]
    spin_errors = [np.linalg.norm(s @ state) for s in _spin_operators(8)]
    assert max(symmetry_errors) < 3e-14
    assert max(spin_errors) < 3e-13
    direction = np.random.default_rng(82005).normal(size=60)
    direction /= np.linalg.norm(direction)
    _, tangent = jax.jvp(
        lambda t: conditional_filter_matrix(t, delta_only=True) @ jnp.asarray(carrier),
        (jnp.zeros(60),), (jnp.asarray(direction),))
    tangent = _project_group(np.asarray(tangent), maps)
    ray_tangent = tangent - carrier * np.vdot(carrier, tangent)
    assert np.linalg.norm(ray_tangent) > 1e-4
    assert max(np.linalg.norm(s @ ray_tangent) for s in _spin_operators(8)) < 3e-13
    print(f"full D8 invariance max={max(symmetry_errors):.3e}; "
          f"singlet residual max={max(spin_errors):.3e}; "
          f"projective tangent norm={np.linalg.norm(ray_tangent):.6g}")
