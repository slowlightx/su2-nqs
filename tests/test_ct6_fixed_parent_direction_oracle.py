"""Independent finite-Hilbert-space and chain-statistics direction oracles.

These tests do not train a model.  In particular, the four-spin singlet
example deliberately opens exact parent nodes: a score-only energy slope
then misses physical first-order terms even with exhaustive Born quadrature.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as campaign


def _four_spin_dimer(pairs):
    """Explicit product singlets, bit zero down, site zero most significant."""
    amplitude = np.zeros(16, dtype=np.complex128)
    for index in range(16):
        bits = [(index >> (3 - site)) & 1 for site in range(4)]
        value = 1.0
        for left, right in pairs:
            if bits[left] == bits[right]:
                value = 0.0
                break
            value *= (1.0 if bits[left] == 0 else -1.0) / np.sqrt(2.0)
        amplitude[index] = value
    return amplitude


def _four_spin_hamiltonian():
    """Independent computational-basis Heisenberg ring with both diagonals."""
    hamiltonian = np.zeros((16, 16), dtype=np.complex128)
    bonds = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0),
             (0, 2, 0.125), (1, 3, 0.125)]
    for old in range(16):
        bits = [(old >> (3 - site)) & 1 for site in range(4)]
        for left, right, coupling in bonds:
            hamiltonian[old, old] += coupling * (bits[left] - 0.5) * (bits[right] - 0.5)
            if bits[left] != bits[right]:
                new = old ^ (1 << (3 - left)) ^ (1 << (3 - right))
                hamiltonian[new, old] += 0.5 * coupling
    return hamiltonian


def _exact_energy_slope(parent, tangent, hamiltonian):
    norm = np.vdot(parent, parent).real
    energy = np.vdot(parent, hamiltonian @ parent).real / norm
    slope = 2.0 * np.real(np.vdot(parent, hamiltonian @ tangent)
                          - energy * np.vdot(parent, tangent)) / (4.0 * norm)
    return float(slope)


def test_complete_h_tangent_slope_recovers_four_spin_singlet_node_opening():
    hamiltonian = _four_spin_hamiltonian()
    parent = _four_spin_dimer([(0, 1), (2, 3)])
    tangent = -_four_spin_dimer([(0, 2), (1, 3)])
    support = np.flatnonzero(parent)
    assert len(support) == 4
    assert np.count_nonzero(tangent[parent == 0]) == 2
    # Equal parent probabilities, repeated deterministic quadrature in four
    # time-major columns.  This checks arithmetic, not chain equilibration.
    rows = np.repeat(support, 4)
    candidate = parent + 0.002 * tangent
    measured = campaign.direction_check_moments(
        parent[rows], (hamiltonian @ parent)[rows], tangent[rows],
        (hamiltonian @ tangent)[rows], candidate[rows], chains=4, n_sites=4)
    exact = _exact_energy_slope(parent, tangent, hamiltonian)
    assert exact == pytest.approx(-0.328125, abs=2e-15)
    assert measured['energy_slope_per_site'] == pytest.approx(exact, abs=2e-15)
    assert measured['score_covariance_slope_per_site'] == pytest.approx(-0.109375, abs=2e-15)
    assert measured['hermitian_minus_score_slope_per_site'] == pytest.approx(-0.21875, abs=2e-15)
    assert measured['energy_slope_chain_jackknife_SE'] == pytest.approx(0.0, abs=2e-15)
    # Averaging the two Born-support sides is NOT a repair: it recovers only
    # half of the missing node contribution. Hermiticity must supply both.
    naive_two_sides = (exact + measured['score_covariance_slope_per_site']) / 2.0
    assert abs(naive_two_sides - exact) > 0.1
    exact_fs = np.sqrt(1 - abs(np.vdot(parent, candidate))**2
                       / (np.vdot(parent, parent).real * np.vdot(candidate, candidate).real))
    assert exact_fs > 1.5 * measured['sampled_fs_sine']
    assert measured['finite_FS_node_complete'] is False


def test_full_support_complex_tangent_agrees_with_covariance_and_exact_finite_fs():
    hamiltonian = _four_spin_hamiltonian()
    first = _four_spin_dimer([(0, 1), (2, 3)])
    second = _four_spin_dimer([(0, 2), (1, 3)])
    parent = first + second
    tangent = -second + 0.7j * first
    support = np.flatnonzero(parent)
    probabilities = abs(parent[support])**2 / np.vdot(parent, parent).real
    counts = np.rint(probabilities * 12).astype(int)
    np.testing.assert_allclose(counts / 12, probabilities, atol=2e-16, rtol=0)
    rows = np.repeat(np.repeat(support, counts), 4)
    candidate = parent + 0.002 * tangent
    args = [parent[rows], (hamiltonian @ parent)[rows], tangent[rows],
            (hamiltonian @ tangent)[rows], candidate[rows]]
    measured = campaign.direction_check_moments(*args, chains=4, n_sites=4)
    exact = _exact_energy_slope(parent, tangent, hamiltonian)
    assert measured['energy_slope_per_site'] == pytest.approx(exact, abs=2e-15)
    assert measured['score_covariance_slope_per_site'] == pytest.approx(exact, abs=2e-15)
    exact_fs = np.sqrt(1 - abs(np.vdot(parent, candidate))**2
                       / (np.vdot(parent, parent).real * np.vdot(candidate, candidate).real))
    assert measured['sampled_fs_sine'] == pytest.approx(exact_fs, abs=4e-13)
    # A common complex amplitude gauge must not alter any ratio statistic.
    gauged = campaign.direction_check_moments(
        *[value * (0.3 + 1.7j) for value in args], chains=4, n_sites=4)
    for key in ('energy_slope_per_site', 'score_covariance_slope_per_site',
                'sampled_fs_sine', 'reweighting_ess_fraction'):
        assert gauged[key] == pytest.approx(measured[key], abs=4e-13)


def test_chain_jackknife_deletes_whole_time_major_chains_and_recomputes_products():
    rng = np.random.default_rng(86113)
    shape = (7, 5)
    parent = np.exp(rng.normal(scale=0.1, size=shape))
    e = rng.normal(size=shape) + 0.2j * rng.normal(size=shape)
    o = rng.normal(size=shape) + 0.3j * rng.normal(size=shape)
    ho = rng.normal(size=shape) + 0.4j * rng.normal(size=shape)
    candidate = parent * (1.0 + 0.001 * o)
    args = [parent, parent * e, parent * o, parent * ho, candidate]
    measured = campaign.direction_check_moments(
        *[value.ravel() for value in args], chains=5, n_sites=36)
    independent = []
    for omitted in range(5):
        keep = [chain for chain in range(5) if chain != omitted]
        independent.append(2 * np.real(ho[:, keep].mean()
                           - e[:, keep].mean() * o[:, keep].mean()) / 36)
    independent = np.asarray(independent)
    expected_se = np.sqrt(4 / 5 * np.sum((independent - independent.mean())**2))
    np.testing.assert_allclose(measured['leave_one_chain_slopes_per_site'], independent,
                               atol=2e-17, rtol=0)
    assert measured['energy_slope_chain_jackknife_SE'] == pytest.approx(expected_se, abs=2e-17)
    expected_parent_se = e.real.mean(axis=0).std(ddof=1) / np.sqrt(5) / 36
    assert measured['parent_energy_chain_SE'] == pytest.approx(expected_parent_se, abs=2e-17)
    # The half-window drift is paired within each chain, not treated as two
    # independent means. With seven records the middle record is omitted.
    paired_half_drift = (e.real[-3:].mean(axis=0) - e.real[:3].mean(axis=0)) / 36
    assert measured['parent_half_window_drift_per_site'] == pytest.approx(
        paired_half_drift.mean(), abs=2e-17)
    assert measured['parent_half_window_drift_chain_SE'] == pytest.approx(
        paired_half_drift.std(ddof=1) / np.sqrt(5), abs=2e-17)
    # Duplicating every record in time carries no new independent-chain
    # information, so it must not spuriously reduce this uncertainty.
    duplicated = campaign.direction_check_moments(
        *[np.repeat(value, 2, axis=0).ravel() for value in args], chains=5, n_sites=36)
    assert duplicated['energy_slope_chain_jackknife_SE'] == pytest.approx(expected_se, abs=2e-17)


def test_frozen_gate_uses_independent_slope_se_and_only_preregistered_fs_safeguard():
    moments = dict(finite=True, energy_slope_per_site=-0.03,
                   energy_slope_chain_jackknife_SE=0.01,
                   sampled_fs_sine=0.003, reweighting_ess_fraction=0.99)
    settings = dict(slope_standard_errors=2, minimum_fs_sine=1e-12,
                    maximum_fs_sine=0.005, minimum_ess_fraction=0.98)
    original = copy.deepcopy(moments)
    assert campaign.direction_check_gate(moments, settings)['passed']
    assert moments == original
    # The narrow half-to-double fit interval is diagnostic, not a new gate.
    for distance in (0.0001, 0.003, 0.005):
        assert campaign.direction_check_gate(dict(moments, sampled_fs_sine=distance), settings)['passed']
    for changes in (dict(energy_slope_per_site=-0.019), dict(sampled_fs_sine=0.006),
                    dict(sampled_fs_sine=0.0), dict(reweighting_ess_fraction=0.979)):
        assert not campaign.direction_check_gate(dict(moments, **changes), settings)['passed']
    assert not campaign.direction_check_gate(dict(finite=False), settings)['passed']


@pytest.mark.parametrize('chains,n_sites', [(1, 4), (True, 4), (2, 0), (2, 2.5)])
def test_direction_moments_reject_invalid_chain_and_site_counts(chains, n_sites):
    with pytest.raises(ValueError):
        campaign.direction_check_moments(*[np.ones(4)] * 5, chains=chains, n_sites=n_sites)


def test_sampled_parent_nodes_and_nonfinite_inputs_fail_closed():
    arrays = [np.ones(8, dtype=complex) for _ in range(5)]
    arrays[0][0] = 0
    assert not campaign.direction_check_moments(*arrays, chains=4)['finite']
    arrays[0][0] = 1
    arrays[3][0] = np.nan
    assert not campaign.direction_check_moments(*arrays, chains=4)['finite']


def test_predeclared_same_parent_energy_veto_combines_independent_chain_uncertainties():
    moments = dict(finite=True, energy_slope_per_site=-0.03,
                   energy_slope_chain_jackknife_SE=0.01,
                   sampled_fs_sine=0.003, reweighting_ess_fraction=0.99,
                   parent_energy_per_site=-0.4, parent_energy_chain_SE=0.03)
    settings = dict(slope_standard_errors=2, minimum_fs_sine=1e-12,
                    maximum_fs_sine=0.005, minimum_ess_fraction=0.98,
                    parent_energy_consistency_standard_errors=3)
    reference = dict(energy_per_site=-0.4, energy_chain_SE=0.04, audit_sha256='same-parent')
    assert not campaign.direction_check_gate(moments, settings)['passed']
    for delta, passed in [(0.149, True), (0.151, False), (-0.149, True), (-0.151, False)]:
        annotated = campaign.with_parent_energy_reference(
            dict(moments, parent_energy_per_site=reference['energy_per_site'] + delta), reference)
        comparison = annotated['parent_energy_reference_check']
        assert comparison['combined_independent_chain_SE'] == pytest.approx(0.05, abs=1e-16)
        assert comparison['standardized_difference'] == pytest.approx(delta / 0.05, abs=2e-15)
        assert comparison['agreement_does_not_certify_equilibrium'] is True
        assert campaign.direction_check_gate(annotated, settings)['passed'] is passed
    assert 'parent_energy_reference_check' not in moments


def test_energy_reference_reads_only_the_sealed_parent_audit(tmp_path):
    audit_path = tmp_path / 'terminal_audit.json'
    campaign.atomic_json(audit_path, dict(energy_per_site=-0.5149, energy_chain_SE=4e-5))
    digest = campaign.file_sha256(audit_path)
    plan = dict(parent=dict(directory=str(tmp_path), sealed_files={'terminal_audit.json': digest}))
    reference = campaign.frozen_parent_energy_reference(plan)
    assert reference['audit_sha256'] == digest
    assert reference['energy_per_site'] == -0.5149
    campaign.atomic_json(audit_path, dict(energy_per_site=-0.5153, energy_chain_SE=4e-5))
    with pytest.raises(ValueError, match='audit changed'):
        campaign.frozen_parent_energy_reference(plan)


def test_resampled_populations_preserve_parameters_moments_and_restore_every_rng(monkeypatch, tmp_path):
    """More chains are a declared fresh population, not cloned old walkers."""
    params = dict(tensor=jnp.asarray([0.25, -0.125]))
    moments = (jnp.asarray(424), dict(tensor=jnp.asarray([0.02, 0.01])),
               dict(tensor=jnp.asarray([0.003, 0.004])))
    old_train = campaign.initialize_spin_exchange_chains(
        jax.random.PRNGKey(17), n_chains=64, n_sites=36, n_up=18)
    old_cal = campaign.initialize_spin_exchange_chains(
        jax.random.PRNGKey(23), n_chains=64, n_sites=36, n_up=18)
    parent = (params, moments, jax.random.PRNGKey(31), old_train,
              jax.random.PRNGKey(37), old_cal)
    plan = dict(training=dict(chains=256, initial_burn_proposals=1440),
                calibration=dict(chains=128, initial_burn_proposals=1440),
                seeds=dict(training=261026101, calibration=261026102))
    traced_burns = []

    def mock_burn(prepared, key, states, *, amplitude_fn, particle_kind, sweeps, exchange_edges):
        traced_burns.append((tuple(states.shape), int(sweeps), particle_kind))
        # Keep the physical magnetization invariant and consume a key. This
        # mock checks orchestration only; no equilibration claim is made.
        successor, _ = jax.random.split(key)
        after = jnp.roll(states, 1, axis=1)
        return after, jnp.ones(len(states)), successor, jnp.asarray(1.0)

    monkeypatch.setattr(campaign, 'run_exchange_sweeps', mock_burn)
    runtime = dict(prepare=lambda p: p['tensor'], projected=lambda p, x: jnp.ones(x.shape[:-1]),
                   edges=((0, 1),))
    state, record = campaign.initialize_resampled_chains(plan, parent, runtime)
    assert state[0] is params and state[1] is moments
    assert traced_burns == [((256, 36), 1440, 'spin'), ((128, 36), 1440, 'spin')]
    assert record['parameters_and_optimizer_inherited_bitwise']
    assert record['mixing_certified'] is False
    np.testing.assert_array_equal(record['parent_training_rng'], parent[2])
    np.testing.assert_array_equal(record['parent_calibration_rng'], parent[4])
    for section, key_index, states_index in [('training', 2, 3), ('calibration', 4, 5)]:
        key, initializer = jax.random.split(jax.random.PRNGKey(plan['seeds'][section]))
        expected = campaign.initialize_spin_exchange_chains(
            initializer, n_chains=plan[section]['chains'], n_sites=36, n_up=18)
        expected_key, _ = jax.random.split(key)
        np.testing.assert_array_equal(state[key_index], expected_key)
        np.testing.assert_array_equal(state[states_index], np.roll(np.asarray(expected), 1, axis=1))
        np.testing.assert_array_equal(np.asarray(state[states_index]).sum(axis=1), 18)
    assert not np.array_equal(np.asarray(state[3]), np.tile(np.asarray(old_train), (4, 1)))
    assert not np.array_equal(np.asarray(state[5]), np.tile(np.asarray(old_cal), (2, 1)))
    assert record['parent_sampler_rng_sha256'] == campaign.pytree_sha256(parent[2:])
    assert record['new_sampler_rng_sha256'] == campaign.pytree_sha256(state[2:])
    checkpoint = tmp_path / 'declared_population_checkpoint.pkl'
    campaign.shared.save_state(checkpoint, state, dict(registration_sha256='independent-test'))
    restored, _ = campaign.shared.restore_state(checkpoint, 'independent-test')
    for saved, recovered in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(saved, recovered)
