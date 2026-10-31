"""Independent derivative, full-H action and physical-coordinate checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_projected_capacity import (
    capacity_coordinate_family, coordinate_bank, make_capacity_amplitude, physical_mode_rotation, tangent_moments,
    raw_pair_direction_responses, _pair_profiles,
    screening_context,
)
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, init_mixed_clebsch_family
from su2_nqs.models.triangular_mixed_clebsch import make_mixed_projected_amplitude
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.operators.heisenberg_graph import heisenberg_graph_matrix
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action

jax.config.update('jax_enable_x64', True)


def test_complex_physical_energy_force_matches_complete_Hdelta_and_finite_difference():
    n = 6
    bonds = ((0, 1, 1.), (1, 2, .9), (2, 3, 1.1), (3, 4, 1.),
             (4, 5, .8), (0, 5, 1.), (0, 2, .125), (1, 4, .125))
    full_h, basis = heisenberg_graph_matrix(n, bonds, n_up=3)
    transform = coupled_basis_matrix(enumerate_paths(n, 0), 0, 0)
    full_coeff = np.array([.7+.2j, .3-.15j, -.4+.1j, .23+.3j, .18-.23j])
    full_delta = np.array([-.2+.3j, .15+.2j, .1-.3j, .4+.1j, -.3+.15j])
    psi_all, delta_all = transform @ full_coeff, transform @ full_delta
    psi = psi_all[np.asarray(basis)]
    delta = delta_all[np.asarray(basis)]
    assert np.min(abs(psi)) > 1e-4
    rows = (np.asarray(basis)[:, None] >> np.arange(n-1, -1, -1)) & 1
    powers = jnp.asarray(1 << np.arange(n-1, -1, -1))
    amplitude = lambda theta, bits: jnp.asarray(psi_all)[bits @ powers]+theta*jnp.asarray(delta_all)[bits @ powers]
    action = make_amplitude_heisenberg_action(amplitude, bonds, n_sites=n)
    (_, hpsi), (_, hdelta) = jax.jvp(lambda x: action(x, jnp.asarray(rows)),
                                    (jnp.asarray(0.),), (jnp.asarray(1.),))
    np.testing.assert_allclose(hpsi, full_h @ psi, atol=4e-15, rtol=0)
    np.testing.assert_allclose(hdelta, full_h @ delta, atol=4e-15, rtol=0)
    probability = abs(psi)**2/np.vdot(psi, psi).real
    metric, force = tangent_moments((delta/psi)[:, None], hpsi/psi, probability)
    energy = np.vdot(psi, full_h @ psi)/np.vdot(psi, psi)
    direct = 2*np.real(np.vdot(delta, full_h @ psi-energy*psi)/np.vdot(psi, psi))
    np.testing.assert_allclose(force[0], direct, atol=4e-15, rtol=0)
    epsilon = 1e-5
    values = []
    for step in (epsilon, -epsilon):
        trial = psi+step*delta
        values.append(np.vdot(trial, full_h @ trial).real/np.vdot(trial, trial).real)
    np.testing.assert_allclose(force[0], (values[0]-values[1])/(2*epsilon), atol=2e-9, rtol=0)
    assert metric[0, 0] > 0
    gauge_metric, gauge_force = tangent_moments(np.full((len(psi), 1), 2+3j), hpsi/psi, probability)
    np.testing.assert_allclose(gauge_metric, 0, atol=1e-27)
    np.testing.assert_allclose(gauge_force, 0, atol=2e-15)


def test_scores_respect_record_permutation_and_fail_closed():
    rng = np.random.default_rng(74120)
    scores = rng.normal(size=(24, 4))+1j*rng.normal(size=(24, 4))
    energies = rng.normal(size=24)+1j*rng.normal(size=24)
    p = rng.permutation(24)
    first = tangent_moments(scores, energies)
    second = tangent_moments(scores[p], energies[p])
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b, atol=8e-16)
    with pytest.raises(ValueError):
        tangent_moments(scores, energies, np.ones(24))


def test_actual_capacity_family_changes_physical_not_virtual_Schur_subspace():
    config = MixedClebschConfig(multiplicity_cap=20, maximum_q=10)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(74123), config)['ternary_free']
    canonical, delta, data = capacity_coordinate_family(bank, config)
    assert len(data['metadata']['tokens']) == 24
    assert data['arrays']['pair_relations'].shape == (24, 24, 4)
    for token in data['metadata']['tokens']:
        edge, q, index = token['edge'], token['q'], token['index']
        qi = config.local_q.index(q)
        w, change = canonical['local'][edge][qi], delta['local'][edge][qi][index]
        assert token['marginal_native_scalars'] == {1: 2802, 3: 4748, 5: 5707}[q]
        assert np.linalg.matrix_rank(change, tol=1e-10) == 1
        if token['kind'] == 'rotate':
            np.testing.assert_allclose(w.T @ change, 0, atol=1e-12)
    theta = np.zeros(24); theta[1] = .01
    trial = coordinate_bank(canonical, delta, theta)
    np.testing.assert_allclose(trial['local'][0][0],
                               canonical['local'][0][0]+.01*delta['local'][0][0][1], atol=1e-15)
    with pytest.raises(ValueError):
        coordinate_bank(canonical, delta, np.zeros(23))


def test_rotation_uses_gauge_invariant_physical_orientation_not_QR_completion():
    rng = np.random.default_rng(74124)
    w, _ = np.linalg.qr(rng.normal(size=(42, 20)))
    gauge, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    v = rng.normal(size=20); v /= np.linalg.norm(v)
    first, complement = physical_mode_rotation(w, v)
    second, changed_complement = physical_mode_rotation(w @ gauge, -gauge.T @ v)
    np.testing.assert_allclose(complement, changed_complement, atol=8e-16)
    np.testing.assert_allclose(np.outer(complement, w @ first),
                               np.outer(changed_complement, w @ gauge @ second), atol=8e-16)
    np.testing.assert_allclose(w.T @ complement, 0, atol=8e-16)


def test_projected_coordinate_derivatives_match_independent_physical_bank_and_Ward_identity():
    config = MixedClebschConfig(multiplicity_cap=16, maximum_q=5)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(74125), config)['ternary_free']
    canonical, deltas, _ = capacity_coordinate_family(bank, config)
    amplitude = make_capacity_amplitude(canonical, deltas, config)
    group = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    legacy = jax.jit(make_mixed_projected_amplitude(config, group, chunk=4))
    rng = np.random.default_rng(74126)
    bits = jnp.asarray(rng.permutation([0]*18+[1]*18))
    theta = np.zeros(24); theta[1] = .3; theta[8] = -.2; theta[21] = .1
    zero = jnp.zeros(24)
    value, derivative = jax.jit(lambda: jax.jvp(lambda x: amplitude(x, bits), (zero,), (jnp.asarray(theta),)))()
    np.testing.assert_allclose(value, legacy(canonical, bits), atol=1e-16, rtol=2e-11)
    epsilon = 1e-4
    plus = legacy(coordinate_bank(canonical, deltas, epsilon*theta), bits)
    minus = legacy(coordinate_bank(canonical, deltas, -epsilon*theta), bits)
    np.testing.assert_allclose(derivative, (plus-minus)/(2*epsilon), atol=2e-13, rtol=3e-7)
    # A literal S+ test of the TANGENT, independent of its doubled-q labels.
    bra = rng.permutation([0]*17+[1]*19)
    lowered = np.repeat(bra[None, :], 19, axis=0)
    lowered[np.arange(19), np.flatnonzero(bra)] = 0
    derivative_rows = jax.jit(lambda rows: jax.lax.map(lambda row: jax.jvp(
        lambda x: amplitude(x, row), (zero,), (jnp.asarray(theta),))[1], rows))(jnp.asarray(lowered))
    assert abs(np.sum(derivative_rows))/np.linalg.norm(derivative_rows) < 2e-11


def test_cheap_raw_pair_direction_feature_includes_all_q_normalization():
    config = MixedClebschConfig(multiplicity_cap=16, maximum_q=5)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(74127), config)['ternary_free']
    canonical, deltas, data = capacity_coordinate_family(bank, config)
    response = raw_pair_direction_responses(canonical, deltas, config)
    for index in (0, 1, 12, 17):
        theta = np.zeros(24); theta[index] = 1.
        eps = 1e-4
        plus = _pair_profiles(coordinate_bank(canonical, deltas, eps*theta), config)
        minus = _pair_profiles(coordinate_bank(canonical, deltas, -eps*theta), config)
        edge = data['metadata']['tokens'][index]['edge']
        np.testing.assert_allclose(response[index], (plus[edge]-minus[edge])/(2*eps), atol=1e-9, rtol=2e-6)
    # A rotation within fixed-q cannot change the region's total S^2.
    rotation = data['arrays']['token_kind'] == 1
    np.testing.assert_allclose(2*response[rotation].sum(axis=1), 0, atol=2e-14)


def test_screening_context_only_reads_fit_bits_and_keeps_physical_matrix_symmetry():
    rng = np.random.default_rng(74130)
    bits = rng.integers(0, 2, size=(32, 6), dtype=np.int32)
    fit = np.arange(32) < 24
    response = rng.normal(size=(4, 3))
    supports, nodes = [[0, 1, 2], [3, 4, 5]], np.asarray([0, 0, 1, 1])
    first = screening_context(bits, fit, supports, nodes, response)
    bits[~fit] = 1-bits[~fit]
    second = screening_context(bits, fit, supports, nodes, response)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(first['screening_context'], first['screening_context'].T, atol=5e-16)
    zero = screening_context(bits, fit, supports, nodes, np.zeros((4, 3)))
    np.testing.assert_array_equal(zero['screening_context'], np.zeros((4, 4)))


def test_exact_three_spin_singlet_pair_plus_mixed_spin_has_correct_vector_correlations():
    # rho=|singlet12><singlet12| tensor I3/2 is rotationally invariant.
    # Its four equally likely Sz configurations suffice to estimate 3<SzSz>.
    bits = np.asarray([[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]], np.int32)
    result = screening_context(bits, np.ones(4, bool), [[0, 1, 2]], np.asarray([0, 0]),
                               np.asarray([[1., 2., 3.], [-1., .5, 2.]]))
    expected = np.diag([.75, .75, .75]); expected[0, 1] = expected[1, 0] = -.75
    np.testing.assert_allclose(result['projected_fit_connected_spin_matrix'], expected, atol=0)
    np.testing.assert_allclose(result['raw_spin_response_profiles'][0], [3, 4, 5], atol=0)
    f = result['raw_spin_response_profiles']
    normalized = f/np.linalg.norm(f, axis=1)[:, None]
    np.testing.assert_allclose(result['screening_context'], normalized @ expected @ normalized.T, atol=1e-16)
