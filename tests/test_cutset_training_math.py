"""Independent small-support quadrature and paired-chain regression tests."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.cutset_training import (
    weighted_diagonal_loss, weighted_state_distance, paired_chain_audit,
)

jax.config.update('jax_enable_x64', True)


def _singlet_problem():
    bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(int)
    selected = np.flatnonzero(bits.sum(axis=1) == 2)
    pair = np.array([[0., -1/np.sqrt(2)], [1/np.sqrt(2), 0.]])
    first = pair[bits[:, 0], bits[:, 1]]*pair[bits[:, 2], bits[:, 3]]
    second = pair[bits[:, 0], bits[:, 2]]*pair[bits[:, 1], bits[:, 3]]
    spin = (np.array([[0, .5], [.5, 0]], complex),
            np.array([[0, .5j], [-.5j, 0]], complex), np.diag([-.5, .5]))
    h = np.zeros((16, 16), complex)
    for i, j, coupling in ((0, 1, 1.), (1, 2, .7), (2, 3, .8), (0, 3, .9), (0, 2, .2)):
        for s in spin:
            term = np.ones((1, 1))
            for site in range(4):
                term = np.kron(term, s if site in (i, j) else np.eye(2))
            h += coupling*term
    return first[selected], second[selected], h[np.ix_(selected, selected)]


def _quadrature():
    q = np.arange(1., 7.)
    # Exact quadrature for Q=q/sum(q), represented as repeated observations.
    indices = np.repeat(np.arange(6), q.astype(int))
    return q, indices


def test_fixed_Q_direct_amplitude_energy_gradient_matches_dense_singlet_even_at_nodes():
    parent, direction, h = _singlet_problem()
    q, indices = _quadrature()
    diagonal = 3*np.diag(h).real
    def vector(theta):
        return jnp.asarray(parent)+(theta[0]+1j*theta[1])*jnp.asarray(direction)
    def objective(theta):
        return weighted_diagonal_loss(vector(theta)[indices], q[indices], diagonal[indices])
    def exact(theta):
        psi = vector(theta)
        return jnp.vdot(psi, jnp.asarray(h)@psi).real/jnp.vdot(psi, psi).real
    for theta in (jnp.zeros(2), jnp.array([.17, -.09])):
        value, grad = jax.jit(jax.value_and_grad(objective))(theta)
        wanted, wanted_grad = jax.value_and_grad(exact)(theta)
        np.testing.assert_allclose(value, wanted, atol=2e-14, rtol=0.)
        np.testing.assert_allclose(grad, wanted_grad, atol=2e-13, rtol=0.)
        scaled = weighted_diagonal_loss(vector(theta)[indices], 7.4*q[indices], diagonal[indices])
        np.testing.assert_allclose(value, scaled, atol=2e-14, rtol=0.)
    assert np.any(parent == 0) and np.any(direction[parent == 0] != 0)
    assert np.linalg.norm(np.asarray(jax.grad(objective)(jnp.zeros(2)))) > .01
    # Proposal quadrature is fixed, not a variational parameter.
    proposal_gradient = jax.grad(lambda density: weighted_diagonal_loss(
        vector(jnp.array([.1, .2]))[indices], density, diagonal[indices]))(jnp.asarray(q[indices]))
    np.testing.assert_array_equal(proposal_gradient, np.zeros(len(indices)))


def test_FS_captures_opened_nodes_and_is_invariant_to_unknown_Q_normalizer():
    parent, direction, _ = _singlet_problem()
    candidate = parent+(.23+.11j)*direction
    q, indices = _quadrature()
    result = weighted_state_distance(parent[indices], candidate[indices], q[indices])
    expected_overlap = abs(np.vdot(parent, candidate))/np.linalg.norm(parent)/np.linalg.norm(candidate)
    assert result['fs'] == pytest.approx(np.arccos(expected_overlap), abs=3e-15)
    assert result['norm_ratio'] == pytest.approx(np.linalg.norm(candidate)**2/np.linalg.norm(parent)**2, abs=2e-15)
    assert result['opened_node_samples'] > 0
    scaled = weighted_state_distance(parent[indices], candidate[indices], 13.9*q[indices])
    for key in ('fs', 'norm_ratio', 'ess_fraction', 'max_weight_fraction'):
        assert result[key] == pytest.approx(scaled[key], abs=3e-15)
    phase = weighted_state_distance(parent[indices], np.exp(.3j)*parent[indices], q[indices])
    assert phase['fs'] < 3e-8


def _audit(amplitudes, actions, q, groups, names=('parent', 'candidate')):
    return paired_chain_audit(amplitudes, actions, q, groups, n_sites=4,
                              bootstrap_replicates=128, seed=991, arm_names=names)


def test_full_H_quadrature_keeps_node_H2_and_reports_only_relative_norms():
    parent, direction, h = _singlet_problem()
    wavefunctions = np.column_stack((parent, parent+(.23+.11j)*direction))
    actions = h@wavefunctions
    q, indices = _quadrature()
    draws = np.tile(indices, 4)
    groups = np.repeat(np.arange(4), len(indices))
    result = _audit(wavefunctions[draws], actions[draws], q[draws], groups)
    scaled = _audit(wavefunctions[draws], actions[draws], 7*q[draws], groups)
    for arm, name in enumerate(('parent', 'candidate')):
        psi, hp = wavefunctions[:, arm], actions[:, arm]
        norm = np.vdot(psi, psi).real
        energy = np.vdot(psi, hp)/norm
        variance = np.vdot(hp, hp).real/norm-abs(energy)**2
        data = result['arms'][name]
        assert data['energy_per_site']['value'] == pytest.approx(energy.real/4, abs=2e-15)
        assert data['variance_per_site']['value'] == pytest.approx(variance/4, abs=3e-15)
        assert data['norm_over_proposal_partition'] == pytest.approx(norm/q.sum(), abs=2e-16)
        for metric in ('energy_per_site', 'variance_per_site', 'norm_ratio_to_parent'):
            assert data[metric]['value'] == pytest.approx(scaled['arms'][name][metric]['value'], abs=3e-15)
    assert result['arms']['parent']['node_H2_contribution_fraction'] > .01
    assert not result['imaginary_energy_forced_zero']
    assert result['ratio_estimates_finite_sample_biased']
    assert not result['equilibration_certified']


def test_joint_chain_resampling_preserves_pairing_and_does_not_count_repeated_draws_as_IID():
    parent, direction, h = _singlet_problem()
    candidate = parent+(.23+.11j)*direction
    waves = np.column_stack((parent, candidate, np.exp(.3j)*candidate))
    actions = h@waves
    q = np.arange(1., 7.)
    rng = np.random.default_rng(990)
    indices = rng.choice(6, size=(8, 15), p=q/q.sum())
    # Every independent chain has a nonzero norm but differing quadrature.
    indices[:, :2] = [1, 2]
    indices = indices.reshape(-1)
    groups = np.repeat(np.arange(8), 15)
    names = ('parent', 'candidate', 'phase_clone')
    result = _audit(waves[indices], actions[indices], q[indices], groups, names)
    repeated = _audit(np.repeat(waves[indices], 3, axis=0), np.repeat(actions[indices], 3, axis=0),
                      np.repeat(q[indices], 3), np.repeat(groups, 3), names)
    for metric in ('energy_per_site', 'variance_per_site'):
        data = result['paired']['phase_clone-minus-candidate'][metric]
        assert abs(data['value']) < 3e-15
        assert data['chain_bootstrap_se'] < 3e-15
        assert data['chain_jackknife_se'] < 3e-15
        ordinary = result['arms']['candidate'][metric]
        assert ordinary['chain_bootstrap_se'] > 1e-5
        assert ordinary['chain_bootstrap_se'] == pytest.approx(
            repeated['arms']['candidate'][metric]['chain_bootstrap_se'], abs=4e-15)
        assert 'iid_delta_se' not in ordinary
    assert abs(result['arms']['candidate']['imaginary_energy_per_site']['value']) > 1e-7


def test_invalid_proposals_and_unequal_chain_design_fail_closed():
    psi = np.array([1., 0., 2.])
    for proposal in (np.array([1., 0., 2.]), np.array([1., -1., 2.]), np.array([1., np.nan, 2.])):
        with pytest.raises(ValueError):
            weighted_state_distance(psi, psi, proposal)
        with pytest.raises(ValueError):
            weighted_diagonal_loss(jnp.asarray(psi), proposal, jnp.ones(3))
    with pytest.raises(ValueError):
        weighted_state_distance(np.zeros(3), psi, np.ones(3))
    with pytest.raises(ValueError):
        _audit(np.ones((9, 2)), np.ones((9, 2)), np.ones(9), np.array([0, 0, 0, 1, 1, 2, 2, 3, 3]))


def test_chunked_centered_bank_force_and_vjp_equal_full_fixed_Q_ratio_gradient():
    parent, direction, h = _singlet_problem()
    q, indices = _quadrature()
    rng = np.random.default_rng(9912)
    matrix = jnp.asarray(rng.normal(size=(6, 3)))
    imaginary = jnp.asarray(rng.normal(size=(6, 3)))
    def generate(theta):
        return dict(real=jnp.sin(theta), imag=.3*jnp.cos(2*theta))
    def residual(bank):
        return matrix@bank['real']+1j*(imaginary@bank['imag'])
    reference = residual(generate(jnp.array([.2, -.1, .3])))
    def amplitudes(bank):
        return jnp.asarray(parent)+residual(bank)-reference
    theta = jnp.array([.23, -.07, .31])
    diagonal = jnp.asarray(3*np.diag(h).real[indices])
    density = jnp.asarray(q[indices])
    full = jax.grad(lambda current: weighted_diagonal_loss(
        amplitudes(generate(current))[indices], density, diagonal))(theta)
    bank, pullback = jax.vjp(generate, theta)
    psi = amplitudes(bank)[indices]
    weights = abs(psi)**2/density
    norm = float(jnp.sum(weights))
    energy = float(jnp.sum(weights*diagonal)/norm)
    # The numerator is a centered force surrogate, not a new objective. E and
    # norm are held fixed from the COMPLETE corpus, including uneven chunks.
    for chunk_size in (1, 4, 8):
        accumulated = jax.tree_util.tree_map(jnp.zeros_like, bank)
        for start in range(0, len(indices), chunk_size):
            sl = slice(start, start+chunk_size)
            def micro_loss(candidate):
                values = amplitudes(candidate)[indices[sl]]
                return jnp.sum(abs(values)**2/density[sl]*(diagonal[sl]-energy))/norm
            gradient = jax.grad(micro_loss)(bank)
            accumulated = jax.tree_util.tree_map(lambda x, y: x+y, accumulated, gradient)
        actual = pullback(accumulated)[0]
        np.testing.assert_allclose(actual, full, atol=3e-14, rtol=3e-14)


def test_unnormalized_exchange_bridge_preserves_partition_function_and_null_swaps():
    from su2_nqs.vmc.exchange_bridge_importance import make_exchange_bridge
    all_bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int32)
    bits = all_bits[all_bits.sum(axis=1) == 2]
    integers = bits @ (1 << np.arange(3, -1, -1))
    base = np.array([0., 2., 0., 5., 8., 3.])+.2
    table = np.zeros(16)
    table[integers] = base
    edges = ((0, 1), (1, 2), (0, 1), (2, 3))
    probabilities = np.array([.2, .25, .3, .25])
    bridge_fraction = .4
    log_base = lambda rows: jnp.log(jnp.asarray(table)[rows@jnp.asarray(1 << np.arange(3, -1, -1))])
    density, _ = make_exchange_bridge(log_base, lambda key, count: None,
        edges, probabilities, n_sites=4, bridge_fraction=bridge_fraction)
    measured = np.exp(np.asarray(density(jnp.asarray(bits))[0]))
    locations = {tuple(row): index for index, row in enumerate(bits)}
    pushed = np.zeros(6)
    null_weight = 0.
    for old, row in enumerate(bits):
        for (first, second), probability in zip(edges, probabilities):
            transformed = row.copy()
            transformed[first], transformed[second] = row[second], row[first]
            pushed[locations[tuple(transformed)]] += probability*base[old]
            if row[first] == row[second]:
                null_weight += probability*base[old]
    expected = (1-bridge_fraction)*base+bridge_fraction*pushed
    np.testing.assert_allclose(measured, expected, atol=3e-15, rtol=2e-15)
    assert measured.sum() == pytest.approx(base.sum(), abs=4e-15)
    assert null_weight > 1.
    assert np.all(measured >= .2)
    # The denominator is unnormalized, but exact forward quadrature still
    # gives every normalized target expectation without knowing Z_Q.
    target = np.array([.3, 0., .8, -.2j, .1, -.4])
    observable = np.array([1., 4., -2., .3, .7, -.4])
    forward_probability = expected/base.sum()
    norm = np.sum(forward_probability*abs(target)**2/measured)
    energy = np.sum(forward_probability*abs(target)**2/measured*observable)/norm
    assert energy == pytest.approx(np.sum(abs(target)**2*observable)/np.sum(abs(target)**2), abs=2e-15)
