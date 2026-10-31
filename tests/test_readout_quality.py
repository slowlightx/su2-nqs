"""Independent physical-density oracles for refinement-aware comparisons."""
import numpy as np
import pytest

from su2_nqs.benchmarks.readout_quality import regional_readout_distance, compare_with_refinement


def _singlet_triplet(p):
    v = np.array([0., -1., 1., 0.])/np.sqrt(2)
    singlet = np.outer(v, v)
    return p*singlet+(1-p)*(np.eye(4)-singlet)/3


def test_invariant_distance_agrees_with_analytic_spin_weights_and_spectra():
    result = regional_readout_distance(_singlet_triplet(.8), _singlet_triplet(.3))
    for key in ('physical_trace_distance', 'p_q_total_variation', 'multiplicity_spectrum_total_variation'):
        assert result[key] == pytest.approx(.5, abs=2e-14)
    assert result['measurement_contractivity_error'] < 2e-14


def test_broken_magnetic_states_have_valid_weights_but_no_invariant_spectrum():
    up, down = np.eye(4)[3], np.eye(4)[0]
    result = regional_readout_distance(np.outer(up, up), np.outer(down, down))
    assert result['physical_trace_distance'] == pytest.approx(1.)
    assert result['p_q_total_variation'] == pytest.approx(0., abs=2e-14)
    assert not result['both_strict_su2_eligible']
    assert 'multiplicity_spectrum_total_variation' not in result


def test_global_spin_frame_alignment_preserves_weights_without_twirling():
    # Down/up convention: exp(-i theta Sy)=[[cos,sin],[-sin,cos]].
    up = np.eye(4)[3]; original = np.outer(up,up)
    theta = np.pi; c,s = np.cos(theta/2),np.sin(theta/2)
    one = np.array([[c,s],[-s,c]]); u = np.kron(one,one)
    rotated = u@original@u.T
    np.testing.assert_allclose(rotated,np.diag([1.,0.,0.,0.]),atol=2e-14)
    result = regional_readout_distance(original,rotated)
    assert result['physical_trace_distance']==pytest.approx(1.,abs=2e-14)
    assert result['p_q_total_variation']<2e-14
    assert not result['both_strict_su2_eligible']
    restored = u.T@rotated@u
    assert regional_readout_distance(original,restored)['physical_trace_distance']<2e-14
    assert 'multiplicity_spectrum_total_variation' not in result


def test_refinement_scales_are_endpoint_specific_and_missing_data_never_pass():
    a, b = _singlet_triplet(.8), _singlet_triplet(.3)
    result = compare_with_refinement(a, b, refinements_a=[_singlet_triplet(.6)],
                                    refinements_b=[_singlet_triplet(.4)])
    metric = result['metrics']['p_q_total_variation']
    assert metric['refinement_a'] == pytest.approx(.2, abs=2e-14)
    assert metric['refinement_b'] == pytest.approx(.1, abs=2e-14)
    assert metric['empirical_margin'] == pytest.approx(.2, abs=2e-14)
    assert metric['exceeds_declared_refinement']
    result = compare_with_refinement(a, b, refinements_a=[], refinements_b=[b])
    assert not result['refinement_histories_available']
    assert result['metrics']['physical_trace_distance']['refinement_sum'] is None
    assert not result['metrics']['physical_trace_distance']['exceeds_declared_refinement']


def test_incompatible_or_nonphysical_inputs_are_rejected():
    with pytest.raises(ValueError, match='same physical region'):
        regional_readout_distance(np.eye(4)/4, np.eye(8)/8)
    with pytest.raises(ValueError, match='normalized'):
        regional_readout_distance(np.eye(4), np.eye(4)/4)


def test_frozen_calibration_replays_each_refinement_without_a_phase_or_size_release():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_readout_calibration_v1'
    result = json.loads((directory/'results.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert not result['phase_claim'] and not result['size_evolution_released']
    assert all(result[k]==0 for k in ('new_ED','new_training','new_DMRG','new_samples','new_GPU_jobs'))
    registration = result['registration']
    assert not registration['external_DMRG_required']
    for path,sha in registration['sources'].items():
        assert digest(directory/'source_snapshots'/path)==sha
    for path,sha in {**registration['inputs'],**result['contracted_inputs']}.items():
        source = Path(path)
        if source.exists():
            assert digest(source)==sha
    assert [r['j2'] for r in result['ipeps_records']]==[0.,.05,.075,.1,.125,.15]
    densities = []
    for record in result['ipeps_records']:
        assert not record['strict_su2_eligible']
        artifact = root/record['artifact']
        source = json.loads((artifact/'results.json').read_text())
        assert source['registration']['j2']==record['j2']
        with np.load(artifact/'rdms.npz',allow_pickle=False) as arrays:
            coarse,fine = [arrays[f'chi{chi}_physical_compact4'] for chi in (24,32)]
        replay = regional_readout_distance(fine,coarse)
        for key in ('physical_trace_distance','p_q_total_variation'):
            assert replay[key]==pytest.approx(record['refinement'][key],abs=2e-14)
        assert record['compact4_observables']['spin_squared_pair_closure_error']<2e-13
        for stage in record['stages']:
            assert 'invariant_readout' not in stage['compact4']
            assert stage['last_corner_spectrum_change']<1e-8
        densities.append((coarse,fine))
    for i,record in enumerate(result['adjacent_coupling_calibration']):
        assert not record['same_hamiltonian'] and record['calibration_only']
        replay = compare_with_refinement(densities[i][1],densities[i+1][1],
                  refinements_a=[densities[i][0]],refinements_b=[densities[i+1][0]])
        for key,metric in record['metrics'].items():
            assert metric['refinement_sum']==pytest.approx(replay['metrics'][key]['refinement_sum'],abs=2e-14)
            assert metric['empirical_margin']==pytest.approx(replay['metrics'][key]['empirical_margin'],abs=2e-14)
    assert all(not r['metrics']['physical_trace_distance']['exceeds_declared_refinement']
               for r in result['n16']['two_VAP_seeds'].values())
