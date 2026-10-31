"""Independent physical oracles for legal noninvariant multiplicity readout."""
import numpy as np
import pytest

from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns
from su2_nqs.benchmarks.su2_physics_protocol import (
    multiplicity_operator_data, multiplicity_operator_distance, regional_su2_protocol,
    saved_amplitude_region_density,
    compact6_resonance_readout,
    rotational_symmetry_defect, rotational_defect_chain_audit,
    regional_reference_chain_audit,
)


def test_cached_six_site_quality_comparison_keeps_both_refinement_sequences(tmp_path, monkeypatch):
    from pathlib import Path
    import json
    from projects.clebsch_tree_nqs.workflows.spin_su2_analysis import analyze, ROOT
    from projects.clebsch_tree_nqs.workflows import spin_spatial_vit_quality as quality
    config = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_spatial_vit_six_site_quality_comparison.json'
    def forbidden(*args, **kwargs):
        raise AssertionError('saved quality analysis queried a model/H/sampler')
    monkeypatch.setattr(quality,'model_log',forbidden)
    monkeypatch.setattr(quality,'make_full_heisenberg_ratios',forbidden)
    monkeypatch.setattr(quality,'make_collector',forbidden)
    result = analyze(config,tmp_path/'replay')
    existing = json.loads((ROOT/json.loads(config.read_text())['output']/'results.json').read_text())
    for cut, row in result['regions'].items():
        assert len(row['dmrg_quality_sequence']) == 17
        for name, stats in row['scalars'].items():
            assert stats == existing['regions'][cut]['scalars'][name]
            assert stats['vit_quality_independent_SE'] == pytest.approx(
                np.hypot(stats['vit_before_SE'],stats['vit_after_SE']),abs=1e-15)
            assert not stats['paired_quality_endpoints']
        assert row['primary_passing_screens'] == []
    assert result['new_samples'] == result['new_H_calls'] == result['new_amplitude_queries'] == 0
    assert not result['phase_claim'] and not result['strict_spectrum_reported']


def test_product_dimer_region_matches_independent_full_partial_trace():
    from projects.clebsch_tree_nqs.workflows.spin_su2_analysis import product_dimer_region_density
    # Oriented singlet factors in a four-site system; two different cuts.
    v = np.kron(np.array([0., -1., 1., 0.])/np.sqrt(2),
                np.array([0., -1., 1., 0.])/np.sqrt(2))
    for sites in ([0, 1], [0, 2], [3, 1, 0]):
        complement=[i for i in range(4) if i not in sites]
        amplitudes=v.reshape([2]*4).transpose(sites+complement).reshape(2**len(sites),-1)
        expected=amplitudes@amplitudes.conj().T
        actual=product_dimer_region_density(sites,[[0,1],[2,3]])
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
    with pytest.raises(ValueError):
        product_dimer_region_density([0,1],[[0,1],[1,2]])


def test_unified_cached_readout_artifact_has_missing_regions_and_chain_covariance():
    import hashlib,json
    from pathlib import Path
    from su2_nqs.benchmarks.spin_partition_readout import singlet_resonance_readout
    root=Path(__file__).resolve().parents[1]
    folder=root/'projects/clebsch_tree_nqs/results/spin_su2_unified_quality_v1'
    if not (folder/'completion.json').exists():
        pytest.skip('optional local quality artifact absent')
    result=json.loads((folder/'results.json').read_text())
    done=json.loads((folder/'completion.json').read_text())
    for name,key in [('results.json','results_sha256'),('readouts.npz','readouts_sha256')]:
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==done[key]
    assert len(result['records'])==47 and len(result['missing'])==19
    assert all(r['method']=='DMRG' and r['cut']=='local_hexagon6' for r in result['missing'])
    with np.load(folder/'readouts.npz') as data:
        for row in result['records']:
            if row['method']!='ViT':continue
            key='ViT__'+row['state_id']+'__'+row['cut']+'__'
            r,block=singlet_resonance_readout(data[key+'rho'],[0,1,2])
            np.testing.assert_allclose(block,data[key+'rows_singlet_block'],atol=2e-14,rtol=0)
            assert row['scalars']['rows_p33_singlet']==pytest.approx(r['p33_singlet'],abs=2e-14)
            jack=data[key+'combined_scalar_jackknife'];center=jack-jack.mean(0)
            covariance=127/128*(center.T@center)
            np.testing.assert_allclose(data[key+'combined_covariance'],covariance,atol=2e-14,rtol=0)
            for name,se in zip(row['scalar_names'],np.sqrt(np.diag(covariance))):
                assert row['scalar_SE'][name]==pytest.approx(se,abs=2e-14)
    assert all(result[k]==0 for k in ('new_training','new_samples','new_H_calls','new_amplitude_queries','new_MPS_contractions'))
    assert not result['phase_claim'] and not result['strict_entanglement_spectrum']


def test_rotational_audit_uses_only_diagnostic_comparator_and_scalar_invariance():
    up = np.eye(4)[3]
    rho = np.outer(up, up)
    result = rotational_symmetry_defect(rho)
    # Independent triplet isotropic mixture: eigenvalues 1/3,1/3,1/3,0.
    assert result['rotational_defect_trace_distance'] == pytest.approx(2/3, abs=2e-14)
    assert result['rotational_defect_frobenius'] == pytest.approx(np.sqrt(2/3), abs=2e-14)
    assert result['maximum_absolute_pair_axis_anisotropy'] == pytest.approx(1/6, abs=2e-14)
    assert not result['strict_su2_eligible'] and not result['strict_entanglement_spectrum_reported']
    assert result['diagnostic_comparator_only'] and not result['input_repaired']
    assert result['continuity_bounds']['entropy_difference_upper_bound_nats'] >= np.log(3)
    for row in result['invariant_operator_checks'].values():
        assert row['total_spin_commutator_frobenius'] < 2e-14
        assert abs(row['comparator_change']) < 2e-14
    rng = np.random.default_rng(2026100802)
    # Full rank avoids assigning a rigorous positivity-based bound to tiny
    # negative roundoff eigenvalues of a deliberately rank-deficient matrix.
    z = rng.normal(size=(64, 65))+1j*rng.normal(size=(64, 65))
    mixed = z@z.conj().T; mixed /= np.trace(mixed)
    report = rotational_symmetry_defect(mixed, compact_triangular_2x3=True)
    for row in report['invariant_operator_checks'].values():
        assert row['total_spin_commutator_frobenius'] < 3e-13
        assert abs(row['comparator_change']) < 3e-14
    for row in report['pair_axis_anisotropy']:
        assert abs(row['szsz_minus_scalar_over_three']) <= row['trace_distance_bound']+2e-14
    assert report['continuity_bounds']['eigenvalue_l1_difference_upper_bound'] >= 0


def test_nonpositive_estimator_gets_no_symmetry_entropy_bound_or_repair():
    report = rotational_symmetry_defect(np.diag([-.1, .5, .3, .3]))
    assert not report['physical_matrix_valid'] and report['continuity_bounds'] is None
    assert not report['strict_su2_eligible'] and not report['input_repaired']
    with pytest.raises(ValueError):
        rotational_symmetry_defect(np.eye(4))


def test_tiny_negative_eigenvalue_is_not_a_positivity_continuity_certificate():
    # An invariant triplet mixture with a genuinely negative singlet weight
    # passes the numerical 1e-9 diagnostic, but is NOT a density matrix.
    singlet = np.array([0., 1., -1., 0.])/np.sqrt(2)
    projector = np.outer(singlet, singlet)
    matrix = (1+1e-10)*(np.eye(4)-projector)/3-1e-10*projector
    report = rotational_symmetry_defect(matrix)
    assert report['physical_matrix_valid']
    assert report['minimum_eigenvalue'] < 0
    assert not report['continuity_bounds_eligible']
    assert report['continuity_bounds'] is None and not report['input_repaired']
    assert report['numerical_positivity_tolerance'] == 1e-9


def test_native_schur_diagonal_readout_retains_draw_covariance_and_exact_cuts():
    """Replay only saved labels, never query a model or infer coherences."""
    import hashlib
    import json
    from pathlib import Path
    from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
    from su2_nqs.su2.fusion_path import count_paths
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_reduced_vit_train_v1'
    path = directory/'native_spin_readout.json'
    if not path.exists():
        pytest.skip('optional immutable native training audit absent')
    result = json.loads(path.read_text())
    seal = result.pop('result_sha256')
    assert seal == hashlib.sha256(json.dumps(result, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    for relative, digest in result['inputs_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest
    for relative, digest in result['native_source_snapshot_sha256'].items():
        assert hashlib.sha256((directory/'source_snapshot'/relative).read_bytes()).hexdigest() == digest
    with np.load(directory/'untouched_audit.npz') as data:
        labels = np.array(data['labels'])
        assert str(data['frozen_params_sha256']) == result['frozen_params_sha256']
    layout = rectangular_four_spin_layout(6, 6)
    decoded = layout.decode(labels)
    for i, row in enumerate(result['pooled_cell_readout']['spin_weight_estimates']):
        per_draw = (decoded.cell_q == row['q']).mean(axis=1)
        assert row['value'] == pytest.approx(per_draw.mean(), abs=2e-15)
        assert row['iid_draw_SE'] == pytest.approx(per_draw.std(ddof=1)/np.sqrt(512), abs=2e-15)
    assert result['cells'][0]['physical_sites'] == [0, 5, 6, 11]
    assert result['cells'][0]['physical_sites'] != [0, 1, 6, 7]
    assert result['exact_global_SU2_singlet'] and not result['full_A1']
    assert not result['multiplicity_coherences_inferred']
    assert not result['strict_entanglement_spectrum_reported'] and not result['phase_claim']
    reference = result['kinematic_uniform_complete_singlet_reference']
    assert reference['complete_singlet_dimension'] == count_paths(36, 0) == 477638700
    for row in reference['spin_weights']:
        count = count_paths(4, row['q'])*count_paths(32, row['q'])
        assert row['joint_multiplicity_count'] == count
        assert row['probability'] == pytest.approx(count/count_paths(36, 0), abs=2e-15)


def test_native_intracell_casimir_H_lower_bound_is_an_operator_identity():
    from itertools import combinations
    from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
    scalar_sum = sum(spin_dot_matrix(4, i, j) for i, j in combinations(range(4), 2))
    missing = spin_dot_matrix(4, 0, 3)
    cell_H = scalar_sum-.875*missing
    lower_operator = scalar_sum-.21875*np.eye(16)
    # S_i.S_j <= 1/4: no assumption about the endpoint's pair correlations.
    assert np.linalg.eigvalsh(cell_H-lower_operator).min() >= -2e-14
    assert np.linalg.eigvalsh(cell_H-lower_operator).max() == pytest.approx(.875, abs=2e-14)


def test_cross_chain_defect_retains_negative_noise_estimates_and_ratio_caveat():
    # An exactly zero mean defect with four symmetric independent-chain
    # observations gives a NEGATIVE finite U statistic. Never clip or certify.
    anisotropy = np.diag([1., -1., -1., 1.])*.1
    records = np.stack([np.eye(4)/4+sign*anisotropy for sign in (-1, 1, -1, 1) for _ in range(2)])
    groups = np.repeat(np.arange(4), 2)
    report = rotational_defect_chain_audit(records, groups)
    assert report['plug_in_squared_frobenius_defect'] < 2e-28
    assert report['cross_chain_squared_defect'] < 0 and report['negative_estimate_retained']
    assert report['whole_chain_jackknife_SE'] >= 0
    assert not report['exact_SU2_certified'] and not report['normalization_ratio_bias_removed']
    polarized = np.stack([np.diag([0., 0., 0., 1.]) for _ in range(8)])
    report = rotational_defect_chain_audit(polarized, groups)
    assert report['cross_chain_squared_defect'] == pytest.approx(2/3, abs=2e-14)
    with pytest.raises(ValueError, match='equal-size'):
        rotational_defect_chain_audit(records, np.array([0, 0, 0, 1, 1, 2, 2, 3]))


def test_reference_discrepancy_uses_distinct_chains_and_preserves_negative_estimates():
    reference = np.eye(4)/4
    noise = np.diag([1., -1., -1., 1.])*.1
    matrices = np.stack([reference+sign*noise for sign in (-1, 1, -1, 1) for _ in range(2)])
    groups = np.repeat(np.arange(4), 2)
    report = regional_reference_chain_audit(matrices, groups, reference)
    means = np.stack([matrices[groups == i].mean(axis=0)-reference for i in range(4)])
    exact = np.mean([np.vdot(means[i], means[j]).real for i in range(4) for j in range(4) if i != j])
    assert report['features']['raw']['cross_chain_squared_frobenius_discrepancy'] == pytest.approx(exact, abs=2e-15)
    assert report['features']['raw']['negative_estimate_retained']
    assert report['decomposition_closure_error'] < 2e-15
    assert not report['physical_state_replaced'] and not report['mixing_certified']
    covariance = np.asarray(report['whole_chain_delete_covariance'])
    for i, feature in enumerate(report['features'].values()):
        assert covariance[i, i] == pytest.approx(feature['whole_chain_delete_jackknife_SE']**2, abs=2e-15)
    with pytest.raises(ValueError, match='unit-trace'):
        regional_reference_chain_audit(matrices*2, groups, reference)


def test_saved_symmetry_defect_quality_seals_and_parent_specific_interpretation():
    """Replay current sealed caches, not new states or a singlet certificate."""
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_symmetry_defect_quality_v1'
    seal = json.loads((directory/'completion.json').read_text())
    for name, key in [('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                      ('actual_rdms.npz', 'actual_rdms_sha256')]:
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == seal[key]
    result = json.loads((directory/'results.json').read_text())
    for relative, digest in result['registration']['inputs'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest
    for relative, digest in result['registration']['sources'].items():
        assert hashlib.sha256((directory/'source_snapshots'/relative).read_bytes()).hexdigest() == digest
    assert len(result['records']) == 90
    assert all(result[key] == 0 for key in ('new_training', 'new_ED', 'new_samples', 'new_MPS_contractions', 'new_H_calls', 'new_amplitude_queries'))
    assert not result['phase_claim'] and not result['input_repaired']
    assert not result['strict_entanglement_spectrum_reported']
    with np.load(directory/'actual_rdms.npz') as data:
        for row in result['records']:
            tag = '__'.join(str(row[key]) for key in ('method', 'problem', 'family', 'state_id', 'cut'))
            replay = rotational_symmetry_defect(data[tag+'_actual_rho'],
                compact_triangular_2x3=row['compact_triangular_2x3'])
            assert replay['rotational_defect_trace_distance'] == pytest.approx(row['rotational_audit']['rotational_defect_trace_distance'], abs=2e-14)
            for measured in replay['invariant_operator_checks'].values():
                assert abs(measured['comparator_change']) < 2e-14
                assert measured['total_spin_commutator_frobenius'] < 3e-13
            assert not replay['strict_entanglement_spectrum_reported']
    terminal = [r for r in result['records'] if r['method'] == 'ordinary_spatial_vit_4032' and r['state_id'] == 'step00942']
    assert len(terminal) == 3 and len({r['params_sha256'] for r in terminal}) == 1
    assert terminal[0]['params_sha256'] == '2b2afb846a3b5501ce5014fd1e0f9832175e0818da872722ddc02a691899aa82'
    assert all(r['chain_noise_audit']['independent_chains'] == 64 for r in terminal)
    assert all(not r['chain_noise_audit']['exact_SU2_certified'] for r in terminal)
    # A noisy norm statistic does NOT negate the directional axis diagnostic
    # or independently measured nonzero global S² of this particular parent.
    compact = next(r for r in terminal if r['cut'] == 'compact6')
    assert compact['global_spin_squared']['value'] > 10*compact['global_spin_squared']['chain_SE']
    linear = compact['chain_noise_audit']['linear_observable_chain_estimates']['axis_anisotropy_pair_0_2']
    assert linear['signal_over_SE'] > 3
    assert abs(compact['chain_noise_audit']['signal_over_SE']) < 2


def test_conditioned_resonance_matches_explicit_sector_projection_not_raw_covariance():
    from itertools import combinations
    from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
    from su2_nqs.benchmarks.singlet_physics_readout import compact6_channel_operators
    rng = np.random.default_rng(2026100801)
    z = rng.normal(size=(64, 64))+1j*rng.normal(size=(64, 64))
    rho = z@z.conj().T; rho /= np.trace(rho)
    report = compact6_resonance_readout(rho)
    # Independent Casimir-derived projectors, not squared chirality.
    support = np.eye(64)
    for triple in ((0, 3, 4), (1, 5, 2)):
        support = support@(.5*np.eye(64)-2/3*sum(spin_dot_matrix(6, *p)
                          for p in combinations(triple, 2)))
    conditioned = support@rho@support/np.trace(rho@support)
    bank = compact6_channel_operators()
    axes = [[bank[f'E2_triangle{side}_component{i}']/np.sqrt(3/8)
             for i in range(2)]+[bank[f'chi_triangle{side}']/(np.sqrt(3)/4)]
            for side in range(2)]
    means = np.array([[np.trace(conditioned@a).real for a in row] for row in axes])
    corr = np.array([[np.trace(conditioned@a@b).real for b in axes[1]] for a in axes[0]])
    assert report['p_doublet_doublet'] == pytest.approx(np.trace(rho@support).real, abs=2e-14)
    np.testing.assert_allclose(report['conditional']['means'], means, atol=2e-14, rtol=0)
    np.testing.assert_allclose(report['conditional']['correlations'], corr, atol=2e-14, rtol=0)
    np.testing.assert_allclose(report['conditional']['connected'], corr-np.outer(*means), atol=2e-14, rtol=0)
    assert report['joint_probability_closure_error'] < 2e-14
    assert not report['strict_entanglement_spectrum']
    polarized = np.zeros(64); polarized[-1] = 1
    empty = compact6_resonance_readout(np.outer(polarized, polarized))
    assert abs(empty['p_doublet_doublet']) < 2e-14 and empty['conditional'] is None


def test_equal_doublet_occupation_does_not_fix_resonance_correlations():
    bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    def dimer(pairs):
        v = np.prod([(bits[:, i]-bits[:, j])/np.sqrt(2) for i, j in pairs], axis=0)
        return v/np.linalg.norm(v)
    a = dimer(((0, 3), (1, 5), (4, 2)))
    b = dimer(((0, 4), (1, 2), (3, 5)))
    resonance = a+b; resonance /= np.linalg.norm(resonance)
    r0 = compact6_resonance_readout(np.outer(a, a))
    r1 = compact6_resonance_readout(np.outer(resonance, resonance))
    assert r0['p_doublet_doublet'] == pytest.approx(1., abs=2e-14)
    assert r1['p_doublet_doublet'] == pytest.approx(1., abs=2e-14)
    assert abs(r0['conditional']['A2_axis_correlation']) < 2e-14
    assert abs(r1['conditional']['A2_axis_correlation']) > .1


def test_saved_resonance_protocol_preserves_quality_sequences_and_raw_states():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/results'
    for relative, count in [('spin_tensor_network_continuation_v2/su2_protocol_v4', 13),
                            ('spin_full_vit_vap_v1/su2_protocol_v2', 12)]:
        directory = root/relative
        if not directory.exists():
            pytest.skip('optional saved resonance analysis absent')
        seal = json.loads((directory/'completion.json').read_text())
        result = json.loads((directory/'results.json').read_text())
        for name, key in [('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                          ('readouts.npz', 'readouts_sha256')]:
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == seal[key]
        assert len(result['records']) == count and not result['phase_claim']
        assert all(result[k] == 0 for k in ('new_training', 'new_ED', 'new_samples', 'new_H_calls', 'new_MPS_contractions'))
        for row in result['records']:
            if row['problem'].startswith('triangular'):
                resonance = row['regions']['compact6']['triangle_resonance']
                assert resonance['joint_probability_closure_error'] < 2e-14
                assert not resonance['strict_entanglement_spectrum']
                assert resonance['triangles'] == [[0, 3, 4], [1, 5, 2]]
                if row['problem'] == 'triangular_n36':
                    assert not row['regions']['compact6']['multiplicity']['physical']['strict_su2_eligible']


def test_magnetic_orientation_is_not_a_multiplicity_or_entanglement_signal():
    # Two up spins and isotropic spin-one mixed state have the same invariant
    # algebra, but different physical spectra. No fake strict spectrum.
    up = np.eye(4)[3]
    triplet0 = np.array([0, 1, 1, 0])/np.sqrt(2)
    iso = (np.outer(np.eye(4)[0], np.eye(4)[0])
           +np.outer(up, up)+np.outer(triplet0, triplet0))/3
    r, data = multiplicity_operator_data(np.outer(up, up))
    assert r['sectors']['2']['p'] == pytest.approx(1.)
    assert not r['physical']['strict_su2_eligible']
    assert 'invariant_readout' not in r['physical']
    assert r['physical_region_entropy_nats'] == pytest.approx(0.)
    assert data[2][0, 0] == pytest.approx(1.)
    distance = multiplicity_operator_distance(np.outer(up, up), iso)
    assert distance['physical_trace_distance'] == pytest.approx(2/3)
    assert distance['invariant_operator_trace_distance'] < 1e-14
    assert not distance['strict_entanglement_spectrum_comparison']


def test_two_orthogonal_dimer_singlets_have_identical_total_q_different_multiplicity():
    # Explicit product dimers and orthogonalized crossed cover; no eigensolve.
    singlet = np.array([0, -1, 1, 0])/np.sqrt(2)
    a = np.kron(singlet, singlet)
    crossed = a.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(-1)
    b = crossed-a*np.vdot(a, crossed)
    b /= np.linalg.norm(b)
    ra, sa = multiplicity_operator_data(np.outer(a, a))
    assert ra['physical']['strict_su2_eligible']
    assert ra['sectors']['0']['conditional_operator_purity'] == pytest.approx(1.)
    assert 'invariant_readout' in ra['physical']
    result = multiplicity_operator_distance(np.outer(a, a), np.outer(b, b))
    assert result['spin_weight_total_variation'] < 1e-14
    assert result['physical_trace_distance'] == pytest.approx(1.)
    assert result['excess_over_spin_weights'] == pytest.approx(1.)
    assert result['by_q']['0']['conditional_trace_distance'] == pytest.approx(1.)
    assert np.trace(sa[0]) == pytest.approx(1.)


def test_reduced_matrices_reproduce_general_invariant_observables_and_contract():
    rng = np.random.default_rng(71)
    z = rng.normal(size=(16, 16))+1j*rng.normal(size=(16, 16))
    rho = z @ z.conj().T
    rho /= np.trace(rho)
    r, data = multiplicity_operator_data(rho)
    u, labels = regional_schur_columns(4)
    operator = np.zeros((16, 16), complex)
    expected = 0j
    for q, sigma in data.items():
        x = rng.normal(size=sigma.shape)+1j*rng.normal(size=sigma.shape)
        t = (x+x.conj().T)/2
        expected += np.trace(sigma @ t)
        for m in range(-q, q+1, 2):
            ids = [i for i, label in enumerate(labels) if label[:2] == (q, m)]
            operator[np.ix_(ids, ids)] = t
    assert np.trace(rho @ u @ operator @ u.conj().T) == pytest.approx(expected, abs=2e-14)
    for _ in range(4):
        v = rng.normal(size=16)+1j*rng.normal(size=16)
        v /= np.linalg.norm(v)
        d = multiplicity_operator_distance(rho, np.outer(v, v.conj()))
        assert d['contractivity_error'] < 2e-14
    assert not r['physical']['strict_su2_eligible']


def test_protocol_six_body_closure_and_illegal_inputs():
    singlet = np.array([0, -1, 1, 0])/np.sqrt(2)
    v = np.kron(np.kron(singlet, singlet), singlet)
    r, _ = regional_su2_protocol(np.outer(v, v), (0, 1, 2), compact_triangular_2x3=True)
    assert r['conventional']['spin_squared_pair_closure_error'] < 2e-14
    assert r['six_body']['independent_three_bond_closure_error'] < 2e-13
    assert r['multiplicity']['physical']['strict_su2_eligible']
    assert r['partition']['diagnostics']['total_spin_marginal_error'] < 2e-14
    for rho in (np.eye(4), np.diag([-.1, .4, .3, .4]), np.ones((3, 3))):
        with pytest.raises(ValueError):
            multiplicity_operator_data(rho)
    with pytest.raises(ValueError):
        multiplicity_operator_distance(np.eye(4)/4, np.eye(8)/8)
    with pytest.raises(ValueError):
        regional_su2_protocol(np.eye(16)/16, (0, 1), compact_triangular_2x3=True)


def test_saved_coherent_amplitude_partial_trace_and_cut_order():
    # The Bell/singlet coherence must survive: frequencies alone give a
    # diagonal mixture and fail this oracle. Bit orientation is explicit.
    basis = np.array([[int(x) for x in f'{i:04b}'] for i in range(16)])
    singlet = np.array([0, -1, 1, 0])/np.sqrt(2)
    v = np.kron(singlet, singlet)
    rho = saved_amplitude_region_density(v, basis, (0, 1))
    np.testing.assert_allclose(rho, np.outer(singlet, singlet), atol=1e-15)
    np.testing.assert_allclose(saved_amplitude_region_density(v, basis, (0, 2)), np.eye(4)/4, atol=1e-15)
    rng = np.random.default_rng(19)
    v = rng.normal(size=16)+1j*rng.normal(size=16)
    v /= np.linalg.norm(v)
    cut = (2, 0, 3)
    permuted = v.reshape((2,)*4).transpose(*cut, 1).reshape(8, 2)
    np.testing.assert_allclose(saved_amplitude_region_density(v, basis, cut), permuted@permuted.conj().T, atol=1e-15)
    for bad_basis, bad_v, bad_cut in ((basis, 2*v, cut), (basis, v, (0, 0)),
                                    (np.vstack([basis, basis[:1]]), np.r_[v, 0], cut),
                                    (basis, v, (True,)), (basis, v, (4,))):
        with pytest.raises(ValueError):
            saved_amplitude_region_density(bad_v, bad_basis, bad_cut)


def test_completed_dmrg_sweeps_keep_same_chi_distinct_and_verify_raw_seals(tmp_path):
    import hashlib
    import json
    from projects.clebsch_tree_nqs.workflows.spin_su2_analysis import _saved_sequence_inputs, _problem_signature
    problem = dict(geometry='triangular', rows=6, columns=6, j2=.125,
        boundary='PBC_both_physical', regions={'compact4': [14, 15, 20, 21]},
        left_sites={'compact4': [0, 1]}, compact_triangular_2x3=[], cyclic_regions=[])
    _, bonds = _problem_signature(problem)
    directory = tmp_path/'continuation'
    (directory/'sweep_readouts').mkdir(parents=True)
    (directory/'registration.json').write_text(json.dumps(dict(config=problem|dict(chi=2048), bonds=bonds)))
    for sweep in (9, 10):
        path = directory/'sweep_readouts'/f'sweep{sweep:04d}.npz'
        np.savez(path, chi2048_compact4=np.eye(16)/16)
        metadata = dict(global_sweep=sweep, chi=2048, energy_per_site=-.51-.0001*sweep,
            symmetry_repair=False, rdms_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(metadata))
    spec = dict(problems={'tri': problem}, artifacts=[dict(kind='dmrg_completed_sweeps',
        path='continuation', problem='tri', family='reference')])
    rows, inputs = _saved_sequence_inputs(spec, tmp_path)
    assert [row['state_id'] for row in rows] == ['chi2048_sweep0009', 'chi2048_sweep0010']
    assert len(inputs) == 5 and all(row['chi'] == 2048 for row in rows)
    # Missing, tampered or cross-H data cannot enter an apparently good plot.
    damaged = directory/'sweep_readouts'/'sweep0010.npz'
    np.savez(damaged, chi2048_compact4=np.eye(16)/8)
    with pytest.raises(ValueError, match='seal mismatch'):
        _saved_sequence_inputs(spec, tmp_path)
    invalid = dict(problem, regions={'compact6': [0, 2, 1, 6, 7, 8]},
                   left_sites={'compact6': [0, 1, 2]}, compact_triangular_2x3=['compact6'])
    with pytest.raises(ValueError, match='actual ordered 2x3'):
        _problem_signature(invalid)


def test_saved_full_vit_protocol_exact_symmetry_and_quality_reproducibility():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_full_vit_vap_v1/su2_protocol_v1'
    seal = json.loads((directory/'completion.json').read_text())
    for name, key in [('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                      ('readouts.npz', 'readouts_sha256')]:
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == seal[key]
    result = json.loads((directory/'results.json').read_text())
    for relative, digest in result['registration']['inputs'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest
    for relative, digest in result['registration']['sources'].items():
        assert hashlib.sha256((directory/'source_snapshots'/relative).read_bytes()).hexdigest() == digest
    assert len(result['records']) == 12
    assert sum(len(row['regions']) for row in result['records']) == 36
    assert not result['phase_claim']
    assert all(result[key] == 0 for key in ('new_training', 'new_ED', 'new_samples', 'new_MPS_contractions', 'new_H_calls'))
    for row in result['records']:
        assert row['quality']['parameter_count'] == 3992
        assert row['variance_measured']
        for cut in row['regions'].values():
            physical = cut['multiplicity']['physical']
            if row['state_id'] == 'raw_parent':
                assert not physical['strict_su2_eligible']
                assert 'invariant_readout' not in physical
            else:
                assert physical['strict_su2_eligible']
                assert physical['su2_factorization_frobenius'] < 4e-14
                assert 'invariant_readout' in physical
            assert cut['conventional']['spin_squared_pair_closure_error'] < 2e-14
        assert row['regions']['compact6']['six_body']['independent_three_bond_closure_error'] < 2e-14
    for replica in result['replica_diagnostics']:
        assert not replica['physically_distinct_families']
        if replica['problem'] == 'triangular_n16':
            assert replica['regions']['compact6']['distance']['invariant_operator_trace_distance'] < 1.4e-5
            assert abs(replica['regions']['compact6']['scalar_changes']['projector_q3_high']) < 1.1e-6


def test_saved_dmrg_protocol_preserves_seals_actual_rho_and_quality_limits():
    """Replay frozen caches, not a new MPS contraction or phase certificate."""
    import hashlib
    import json
    from pathlib import Path
    from su2_nqs.benchmarks.readout_quality import physical_problem_signature
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_tensor_network_reference_audit_v2/su2_protocol_v1'
    completion = json.loads((directory/'completion.json').read_text())
    for name, key in [('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                      ('readouts.npz', 'readouts_sha256')]:
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == completion[key]
    result = json.loads((directory/'results.json').read_text())
    registry = result['registration']
    for path, digest in registry['inputs'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest
    for path, digest in registry['sources'].items():
        assert hashlib.sha256((directory/'source_snapshots'/path).read_bytes()).hexdigest() == digest
    expected = physical_problem_signature(36, triangular_j1j2_bonds(6, 6, j2=.125, periodic=True),
        geometry='triangular120_6x6_row_major', boundary='PBC_both_physical')
    assert registry['physical_problem_signature'] == expected
    assert not result['phase_claim'] and not result['globally_strict_SU2_claim']
    assert all(result[key] == 0 for key in ('new_training', 'new_ED', 'new_samples', 'new_MPS_contractions', 'new_H_calls'))
    assert len(result['records']) == 17
    assert sum(len(row['regions']) for row in result['records']) == 43
    final = next(row for row in result['records'] if row['family'] == 'dimer' and row['chi'] == 2048)
    assert final['global_spin'] is None and not final['variance_measured']
    with np.load(directory/'readouts.npz') as archive:
        for region, readout in final['regions'].items():
            rho = archive[f'dimer_chi2048_{region}_raw_rho']
            replay, sigma = multiplicity_operator_data(rho)
            assert not replay['physical']['strict_su2_eligible']
            assert 'invariant_readout' not in replay['physical']
            for q, matrix in sigma.items():
                np.testing.assert_allclose(matrix, archive[f'dimer_chi2048_{region}_sigma_q{q}'], atol=2e-14)
            assert readout['conventional']['spin_squared_pair_closure_error'] < 3e-14
            if 'six_body' in readout:
                assert readout['six_body']['independent_three_bond_closure_error'] < 2e-14
    change = result['quality_sequences']['dimer'][-1]['regions']['compact4']['distance']
    assert change['spin_weight_total_variation'] == pytest.approx(.006720347245424849, abs=2e-14)
    assert change['invariant_operator_trace_distance'] > .34
    assert all(not item['physically_distinct_families'] for item in result['initialization_diagnostics'])


def test_late_dmrg_protocol_replays_raw_matrices_without_symmetry_repair():
    """Saved fixed-chi sweeps are distinct quality nodes, not new phases."""
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_tensor_network_continuation_v2/su2_protocol_v1'
    seal = json.loads((directory/'completion.json').read_text())
    for name, key in [('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                      ('readouts.npz', 'readouts_sha256')]:
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == seal[key]
    result = json.loads((directory/'results.json').read_text())
    for relative, digest in result['registration']['inputs'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest
    for relative, digest in result['registration']['sources'].items():
        assert hashlib.sha256((directory/'source_snapshots'/relative).read_bytes()).hexdigest() == digest
    assert not result['phase_claim'] and not result['replica_diagnostics']
    assert len(result['records']) == 5
    assert [r['global_sweep'] for r in result['records']] == [8, 9, 10, 11, 12]
    assert [r['chi'] for r in result['records']] == [2048, 2048, 2048, 3072, 3072]
    assert len({r['state_id'] for r in result['records']}) == 5
    assert all(result[k] == 0 for k in ('new_training', 'new_ED', 'new_samples', 'new_MPS_contractions', 'new_H_calls'))
    with np.load(directory/'readouts.npz') as archive:
        for row in result['records']:
            assert not row['variance_measured'] and row['variance_per_site'] is None
            assert row['quality']['canonical_norm_error'] < 1e-12
            for cut, region in row['regions'].items():
                tag = '__'.join((row['problem'], row['family'], row['state_id'], cut))
                rho = archive[tag+'_raw_rho']
                replay, sigma = multiplicity_operator_data(rho)
                assert not replay['physical']['strict_su2_eligible']
                assert 'invariant_readout' not in replay['physical']
                for q, matrix in sigma.items():
                    np.testing.assert_allclose(matrix, archive[tag+f'_sigma_q{q}'], atol=2e-14)
                assert region['conventional']['spin_squared_pair_closure_error'] < 3e-14
                if 'six_body' in region:
                    assert region['six_body']['independent_three_bond_closure_error'] < 2e-14
        rows = result['records']
        transitions = result['quality_sequences']['triangular_n36/dimer_reference']
        for a, b, change in zip(rows[:-1], rows[1:], transitions):
            for cut in a['regions']:
                key_a = '__'.join((a['problem'], a['family'], a['state_id'], cut))
                key_b = '__'.join((b['problem'], b['family'], b['state_id'], cut))
                replay = multiplicity_operator_distance(archive[key_a+'_raw_rho'], archive[key_b+'_raw_rho'])
                for key in ('physical_trace_distance', 'spin_weight_total_variation', 'invariant_operator_trace_distance'):
                    assert replay[key] == pytest.approx(change['regions'][cut]['distance'][key], abs=2e-14)


def test_completed_chi3072_protocol_keeps_quality_drift_and_actual_symmetry():
    """Final saved sweeps are quality nodes, not a converged-phase certificate."""
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_tensor_network_continuation_v2/su2_protocol_v2'
    seal = json.loads((directory/'completion.json').read_text())
    for name, key in [('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                      ('readouts.npz', 'readouts_sha256')]:
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == seal[key]
    result = json.loads((directory/'results.json').read_text())
    for relative, digest in result['registration']['inputs'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest
    for relative, digest in result['registration']['sources'].items():
        assert hashlib.sha256((directory/'source_snapshots'/relative).read_bytes()).hexdigest() == digest
    rows = result['records']
    assert [r['global_sweep'] for r in rows] == list(range(8, 17))
    assert len({r['state_id'] for r in rows}) == 9
    assert sum(len(r['regions']) for r in rows) == 27
    assert not result['phase_claim'] and not result['replica_diagnostics']
    assert all(result[k] == 0 for k in ('new_training', 'new_ED', 'new_samples', 'new_MPS_contractions', 'new_H_calls'))
    with np.load(directory/'readouts.npz') as archive:
        for row in rows:
            assert not row['variance_measured'] and row['variance_per_site'] is None
            for cut, region in row['regions'].items():
                tag = '__'.join((row['problem'], row['family'], row['state_id'], cut))
                rho = archive[tag+'_raw_rho']
                replay, sigma = multiplicity_operator_data(rho)
                assert not replay['physical']['strict_su2_eligible']
                assert 'invariant_readout' not in replay['physical']
                for q, matrix in sigma.items():
                    np.testing.assert_allclose(matrix, archive[tag+f'_sigma_q{q}'], atol=2e-14)
                assert region['conventional']['spin_squared_pair_closure_error'] < 3e-14
                if 'six_body' in region:
                    assert region['six_body']['independent_three_bond_closure_error'] < 2e-14
        changes = result['quality_sequences']['triangular_n36/dimer_reference']
        for before, after, change in zip(rows[:-1], rows[1:], changes):
            for cut in before['regions']:
                key = lambda r: '__'.join((r['problem'], r['family'], r['state_id'], cut))+'_raw_rho'
                replay = multiplicity_operator_distance(archive[key(before)], archive[key(after)])
                for name in ('physical_trace_distance', 'spin_weight_total_variation', 'invariant_operator_trace_distance'):
                    assert replay[name] == pytest.approx(change['regions'][cut]['distance'][name], abs=2e-14)
        # Full-chi sensitivity and fixed-chi drift are different diagnostics.
        key = lambda r: '__'.join((r['problem'], r['family'], r['state_id'], 'compact6'))+'_raw_rho'
        between_chis = multiplicity_operator_distance(archive[key(rows[2])], archive[key(rows[-1])])
        assert between_chis['invariant_operator_trace_distance'] > 10*between_chis['spin_weight_total_variation']
        assert changes[-1]['regions']['compact6']['distance']['invariant_operator_trace_distance'] < between_chis['invariant_operator_trace_distance']


def test_cached_5600_reference_comparison_preserves_current_parent_and_whole_chains():
    """The new 99904-parameter endpoint is never replaced by the old weak ViT."""
    import hashlib
    import json
    from pathlib import Path
    from su2_nqs.benchmarks.su2_physics_protocol import regional_reference_chain_audit
    root = Path(__file__).resolve().parents[1]
    base = root/'projects/clebsch_tree_nqs/results/spin_spatial_vit_5600_physics_v1'
    directory = base/'reference_quality_comparison_v1'
    seal = json.loads((directory/'completion.json').read_text())
    for name,key in [('results.json','results_sha256'),('registration.json','registration_sha256')]:
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==seal[key]
    result=json.loads((directory/'results.json').read_text())
    registration=result['registration']
    for relative,digest in registration['inputs'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
    for relative,digest in registration['sources'].items():
        assert hashlib.sha256((directory/'source_snapshots'/relative).read_bytes()).hexdigest()==digest
    assert registration['candidate_params_sha256']=='4336e48aedee5451654de2e366ec8ee796f948d5d6ff3080e06247489a37f116'
    assert registration['whole_chains']==128
    assert registration['config']['problem']['regions']['compact4']==[14,15,20,21]
    assert [r['global_sweep'] for r in result['records']]==[16,17,18,19,20]
    assert not result['phase_claim'] and not result['candidate_own_quality_sequence_available']
    assert not result['strict_entanglement_spectrum_reported'] and not result['physical_state_replaced']
    assert all(result[key]==0 for key in ('new_training','new_ED','new_samples','new_MPS_contractions','new_H_calls','new_amplitude_queries'))
    with np.load(base/'physics.npz') as saved, np.load(root/'projects/clebsch_tree_nqs/results/spin_tensor_network_continuation_v2/su2_protocol_v4/readouts.npz') as fixed:
        rho=fixed['triangular_n36__dimer_reference__chi4096_sweep0020__compact4_raw_rho']
        replay=regional_reference_chain_audit(saved['per_record_matrices'],saved['chain_ids'],rho)
        last=result['records'][-1]['squared_distance']
        for key in ('raw','invariant_operator','rotation_noninvariant'):
            assert replay['features'][key]==last['features'][key]
        assert last['features']['raw']['negative_estimate_retained']
        assert abs(last['features']['invariant_operator']['signal_over_SE'])<.1
        # Same-chain covariance, not independent S(k) and zz error addition.
        chains=saved['chain_ids']; values=saved['scalar'][:,1]-3*saved['zz'][:,1]
        means=np.array([values[chains==c].real.mean() for c in np.unique(chains)])
        reported=result['structure_factor_paired_diagnostics']['scalar_minus_three_zz']['K']
        assert reported['value']==pytest.approx(means.mean(),abs=3e-15)
        assert reported['whole_chain_SE']==pytest.approx(means.std(ddof=1)/np.sqrt(128),abs=3e-15)
