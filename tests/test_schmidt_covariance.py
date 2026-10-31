"""Independent complex-state/node oracles for the matched Renyi control."""
import numpy as np
import pytest

from su2_nqs.benchmarks.schmidt_covariance import (
    renyi2_product_numerator, dense_schmidt_covariances,
    compare_renyi_matched_covariances,
    schmidt_transition_window,
    diagonal_swap_controls,
)


@pytest.mark.parametrize('fixed_m', [False, True])
def test_cached_diagonal_product_independent_complex_node_oracle(fixed_m):
    rng = np.random.default_rng(61017075)
    psi = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    if fixed_m:
        psi[np.array([[a.bit_count()+b.bit_count() != 2 for b in range(4)]
                      for a in range(4)])] = 0
    psi[1, 2] = 0  # additional node; never postselect forbidden SWAPs
    psi /= np.linalg.norm(psi)
    rho = psi@psi.conj().T
    o = np.array([.2+1j, -.7+.3j, 1.1-.8j, -.4-.5j])
    terms = []
    for a, b in np.argwhere(psi != 0):
        for c, d in np.argwhere(psi != 0):
            probability = abs(psi[a, b]*psi[c, d])**2
            w = psi[c, b]*psi[a, d]/(psi[a, b]*psi[c, d])
            terms.append(probability*w*(abs(o[a])**2+abs(o[c])**2)/2)
    from math import fsum
    estimate = complex(fsum(x.real for x in terms), fsum(x.imag for x in terms))
    expected = np.trace(rho@rho@np.diag(abs(o)**2))
    np.testing.assert_allclose(estimate, expected, atol=3e-16, rtol=0)


def test_frozen_prefix18_fourier_intensities_are_invariant_over_all12_orbits():
    rng = np.random.default_rng(61017076)
    bits = rng.integers(0, 2, size=(32, 6, 6))
    coordinates = np.array([[i % 6, i//6] for i in range(18)])
    momenta = np.array([[0, 0], [1/3, 1/3], [-1/3, -1/3], [.5, 0], [0, .5], [.5, .5]])
    phases = np.exp(2j*np.pi*coordinates@momenta.T)
    original = abs((bits[:, :3].reshape(-1, 18)-.5)@phases)**2
    for shift in range(6):
        for flip in (False, True):
            moved = np.roll(bits, shift, axis=2)
            if flip: moved = 1-moved
            actual = abs((moved[:, :3].reshape(-1, 18)-.5)@phases)**2
            np.testing.assert_allclose(actual, original, atol=1e-12, rtol=0)
    # An incomplete row does not have this guarantee.
    incomplete = np.exp(2j*np.pi*coordinates[:-1]@momenta[1])
    a = abs((bits[:, :3].reshape(-1, 18)[:, :-1]-.5)@incomplete)**2
    b = abs((np.roll(bits, 1, axis=2)[:, :3].reshape(-1, 18)[:, :-1]-.5)@incomplete)**2
    assert np.max(abs(a-b)) > 1


def test_diagonal_controls_preserve_shared_purity_and_pair_covariance():
    labels = np.repeat(np.arange(8), 4)
    w = np.linspace(.1, 1, 32)
    ordinary = np.broadcast_to([2., 3.], (32, 2))
    result = diagonal_swap_controls(ordinary, w, w[:, None]*ordinary, labels,
                                   squared_magnitude_orbit_invariant=True)
    np.testing.assert_allclose(result['values'], [[2, 3]]*3, atol=2e-15, rtol=0)
    assert max(result['M_minus_C2_SE']) == 0
    assert max(result['M_over_C2_SE']) == 0
    with pytest.raises(ValueError, match='invariance'):
        diagonal_swap_controls(ordinary, w, w[:, None]*ordinary, labels,
                               squared_magnitude_orbit_invariant=False)
    with pytest.raises(ValueError, match='purity'):
        diagonal_swap_controls(ordinary, -w, w[:, None]*ordinary, labels,
                               squared_magnitude_orbit_invariant=True)


def test_exhaustive_complex_two_copy_estimator_and_swapped_nodes():
    # Two-site A and B; no amplitude/SU2/real-state assumptions.
    rng = np.random.default_rng(671002)
    psi = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    psi[0, 0] = psi[1, 3] = psi[2, 1] = 0
    psi /= np.linalg.norm(psi)
    rho = psi@psi.conj().T
    raw = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    operator = raw+raw.conj().T
    o_psi = operator@psi
    expectation = 0j
    retained_nodal_records = 0
    for a, b in np.argwhere(psi != 0):
        for c, d in np.argwhere(psi != 0):
            original = np.array([psi[a, b], psi[c, d]])
            swapped = np.array([psi[c, b], psi[a, d]])
            action = np.array([o_psi[c, b], o_psi[a, d]])
            record = renyi2_product_numerator(action/original, swapped/original)
            expectation += abs(np.prod(original))**2*record
            retained_nodal_records += int(np.any(swapped == 0) and abs(record) > 1e-14)
    assert retained_nodal_records > 0
    np.testing.assert_allclose(expectation, np.trace(rho@rho@operator), atol=5e-16, rtol=0)


def test_dense_commutator_and_eigenbasis_oracles():
    rng = np.random.default_rng(671003)
    x = rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8))
    rho = x@x.conj().T; rho /= np.trace(rho)
    x = rng.normal(size=(3, 8, 8))+1j*rng.normal(size=(3, 8, 8))
    ops = x+x.conj().transpose(0, 2, 1)
    result = dense_schmidt_covariances(rho, ops)
    np.testing.assert_allclose(result['commutator_metric'], result['independent_commutator_metric'],
                               atol=4e-14, rtol=0)
    values, vectors = np.linalg.eigh(rho)
    transformed = vectors.conj().T@ops@vectors
    oracle = np.einsum('ab,iab,jab->ij', (values[:, None]-values)**2,
                       transformed.conj(), transformed).real/(2*result['purity'])
    np.testing.assert_allclose(result['commutator_metric'], oracle, atol=4e-14, rtol=0)
    assert np.linalg.eigvalsh(result['commutator_metric'])[0] > 0


def test_commuting_operator_zero_is_kinematic_and_not_a_phase_test():
    rho = np.diag([.7, .2, .1])
    result = dense_schmidt_covariances(rho, np.array([np.diag([1, -1, 2])]))
    np.testing.assert_allclose(result['commutator_metric'], 0, atol=2e-16, rtol=0)
    assert abs(result['ordinary'][0, 0]-result['renyi_matched'][0, 0]) > .2
    with pytest.raises(ValueError):
        dense_schmidt_covariances(rho*2, np.array([np.eye(3)]))
    with pytest.raises(ValueError):
        renyi2_product_numerator([1, 2, 3], [1, 2, 3])


def test_paired_jackknife_retains_shared_noise_cancellation():
    labels = np.repeat(np.arange(8), 4)
    rng = np.random.default_rng(671004)
    swap = rng.uniform(.1, 1, 32)
    coefficients = np.array([1., 2., 3., .1, .2, .3])
    numerator = swap[:, None]*coefficients
    result = compare_renyi_matched_covariances(numerator, numerator, swap, labels)
    np.testing.assert_allclose(result['commutator_metric'], 0, atol=0, rtol=0)
    np.testing.assert_allclose(result['commutator_metric_SE'], 0, atol=0, rtol=0)
    assert result['commutator_fraction'] == 0
    assert result['commutator_fraction_SE'] == 0


def test_serial_frozen_pipeline_binds_upstream_seals_without_changing_parent(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from projects.clebsch_tree_nqs.workflows import spin_schmidt_covariance as pipeline
    from projects.clebsch_tree_nqs.workflows import spin_chirality_covariance as ordinary
    from projects.clebsch_tree_nqs.workflows import spin_replica_response as replica
    from su2_nqs.frontier.artifacts import atomic_json
    from su2_nqs.frontier.provenance import file_sha256

    base = tmp_path/'base.json'
    frozen = dict(checkpoint_sha256='immutable-parent', streams={'a':'A','b':'B'},
                  momenta={'Gamma':[0,0]},
                  regions={'prefix18':list(range(18))}, chirality_channels={'prefix18':{'fixed':[[0,1]]}})
    atomic_json(base, frozen)
    plan = dict(base_config=str(base), base_config_sha256=file_sha256(base),
                output=str(tmp_path/'outputs'), stage_wall_seconds=dict(replica=20,ordinary=30,escort=40))
    path=tmp_path/'plan.json';atomic_json(path,plan)
    monkeypatch.setattr(pipeline,'load_frozen_inputs',lambda settings:(None,None,None,None,{'sealed':True}))
    seen=[]
    def fake(stage):
        def execute(config):
            settings=json.loads(config.read_text())
            if stage=='escort':
                settings={**json.loads(Path(settings['base_config']).read_text()),**settings}
            assert settings['checkpoint_sha256']==frozen['checkpoint_sha256']
            assert settings['streams']==frozen['streams']
            assert settings['regions']==frozen['regions']
            assert settings['chirality_channels']==frozen['chirality_channels']
            assert settings['wall_seconds']==plan['stage_wall_seconds'][stage]
            out=Path(settings['output']);out.mkdir()
            if stage=='ordinary':
                ref=Path(settings['replica_reference'])
                assert file_sha256(ref/'registration.json')==settings['replica_registration_file_sha256']
            if stage=='escort':
                ref=Path(settings['ordinary_reference'])
                assert file_sha256(ref/'registration.json')==settings['ordinary_registration_file_sha256']
            atomic_json(out/'registration.json',{'settings':settings})
            atomic_json(out/'results.json',{'stage':stage})
            seen.append(stage)
        return execute
    monkeypatch.setattr(replica,'run',fake('replica'))
    monkeypatch.setattr(ordinary,'run',fake('ordinary'))
    monkeypatch.setattr(pipeline,'run',fake('escort'))
    result=pipeline.run_pipeline(path)
    assert seen==['replica','ordinary','escort']
    assert result['new_Born_samples']==result['new_H_calls']==result['new_training']==0
    with pytest.raises(FileExistsError):pipeline.run_pipeline(path)
    atomic_json(base,dict(frozen,checkpoint_sha256='changed'))
    with pytest.raises(ValueError,match='changed'):pipeline.run_pipeline(path)
    atomic_json(base,dict(frozen,momenta={}))
    atomic_json(path,dict(plan,base_config_sha256=file_sha256(base)))
    with pytest.raises(ValueError,match='nonempty'):pipeline.run_pipeline(path)


def test_actual_fixed_Sz_chirality_product_mask_including_accidental_nodes():
    from su2_nqs.benchmarks.chirality_covariance import chirality_product_connections
    from su2_nqs.benchmarks.replica_spin_response import operator_action_over_sample
    rng = np.random.default_rng(671005)
    bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int8)
    psi = rng.normal(size=16)+1j*rng.normal(size=16)
    psi[bits.sum(1) != 2] = 0
    psi[6] = 0; psi /= np.linalg.norm(psi)
    edge = ((0, 1),)
    t, s, c = chirality_product_connections(bits, edge, edge)
    product = np.zeros((16, 16), complex)
    product[s, t@(1 << np.arange(3, -1, -1))] = c
    # Independent two-spin K^2=diag(0,1/4,1/4,0), not an estimator identity.
    local_operator = np.diag([0, .25, .25, 0])
    np.testing.assert_allclose(product, np.kron(local_operator, np.eye(4)), atol=0, rtol=0)
    rho = psi.reshape(4, 4)@psi.reshape(4, 4).conj().T
    result = 0j; legal_nodes = 0; illegal = 0
    for a in np.flatnonzero(psi):
        for b in np.flatnonzero(psi):
            u, v = (b & 12) | (a & 3), (a & 12) | (b & 3)
            if bits[u].sum() != 2:
                illegal += 1
                assert bits[v].sum() != 2
                continue
            rows = np.array([u, v])
            target, source, coefficients = chirality_product_connections(bits[rows], edge, edge)
            with np.errstate(divide='ignore'):
                action = operator_action_over_sample(
                    np.log(psi[target@(1 << np.arange(3, -1, -1))]), source,
                    coefficients, np.log(psi[[a, b]]))
            swap = psi[rows]/psi[[a, b]]
            result += abs(psi[a]*psi[b])**2*renyi2_product_numerator(action, swap)
            legal_nodes += int(np.any(psi[rows] == 0))
    assert illegal > 0 and legal_nodes > 0
    np.testing.assert_allclose(result, np.trace(rho@rho@local_operator), atol=3e-16, rtol=0)


def test_single_channel_ratio_is_a_weighted_entanglement_level_filter_not_a_gap():
    # Include exact zero Schmidt weights: transitions outside the populated
    # support contribute to C2/D, even though their M contribution is zero.
    rho = np.diag([.63, .27, .10, 0.])
    rng = np.random.default_rng(671006)
    a = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    operator = a+a.conj().T
    r = dense_schmidt_covariances(rho, operator[None])
    lam = np.diag(rho)
    squared = lam[:, None]**2+lam[None, :]**2
    weights = squared*abs(operator)**2/2
    filter_value = np.divide(2*lam[:, None]*lam[None, :], squared,
                             out=np.zeros_like(squared), where=squared != 0)
    populated = (lam[:, None] > 0) & (lam[None, :] > 0)
    positive_a, positive_b = np.broadcast_arrays(lam[:, None], lam[None, :])
    np.testing.assert_allclose(filter_value[populated],
        1/np.cosh(np.log(positive_a[populated])-np.log(positive_b[populated])),
        atol=2e-16, rtol=0)
    expected = np.sum(weights*filter_value)/weights.sum()
    observed = r['inter_schmidt'][0, 0]/r['renyi_matched'][0, 0]
    np.testing.assert_allclose(observed, expected, atol=3e-16, rtol=0)
    shifted = dense_schmidt_covariances(rho, (operator+2.4*np.eye(4))[None])
    np.testing.assert_allclose(shifted['commutator_metric'], r['commutator_metric'],
                               atol=1e-14, rtol=0)


def test_charge_window_factorization_and_degenerate_gauge_invariance():
    rng = np.random.default_rng(671007)
    a, b = np.array([.23, .23, .04]), np.array([.12, .12])
    ops = rng.normal(size=(3, 2, 3))+1j*rng.normal(size=(3, 2, 3))
    purity = .19
    r = schmidt_transition_window(a, b, ops, purity)
    oracle = np.array([[np.trace(np.diag(a)@oi.conj().T@np.diag(b)@oj).real/purity
                        for oj in ops] for oi in ops])
    np.testing.assert_allclose(r['gram'], oracle, atol=2e-15, rtol=0)
    np.testing.assert_allclose(r['coherent_sum_contributions'].sum(), r['gram'].sum(), atol=2e-15)
    u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
    v = np.eye(3, dtype=complex); v[:2, :2] = u
    transformed = u.conj().T@ops@v
    np.testing.assert_allclose(schmidt_transition_window(a, b, transformed, purity)['gram'],
                               r['gram'], atol=2e-15, rtol=0)
    with pytest.raises(ValueError):
        schmidt_transition_window(a*4, b, ops, purity)
