"""Physical-problem identity and phase-independent state screening oracles."""
import numpy as np
import pytest

from su2_nqs.benchmarks.readout_quality import physical_problem_signature, physical_state_similarity


def test_problem_identity_combines_edges_but_separates_hamiltonians_and_boundaries():
    kwargs = dict(geometry='triangle',boundary='physical periodic')
    original = physical_problem_signature(3,[(0,1,1.),(1,2,.5)],**kwargs)
    assert original==physical_problem_signature(3,[(2,1,.25),(1,0,1.),(1,2,.25)],**kwargs)
    assert original!=physical_problem_signature(3,[(0,1,1.),(1,2,.6)],**kwargs)
    assert original!=physical_problem_signature(3,[(0,1,1.),(1,2,.5)],geometry='triangle',boundary='open')
    with pytest.raises(ValueError): physical_problem_signature(3,[(0,0,1.)],**kwargs)
    with pytest.raises(ValueError): physical_problem_signature(3,[(0,3,1.)],**kwargs)


def test_similarity_removes_only_global_phase_and_scale():
    a = np.array([1.,1j])/np.sqrt(2)
    result = physical_state_similarity(a,17*np.exp(.7j)*a)
    assert result['fidelity']==pytest.approx(1.,abs=2e-14)
    assert result['phase_aligned_distance']<2e-14
    b = np.array([1.,-1j])/np.sqrt(2)
    result = physical_state_similarity(a,b)
    assert result['fidelity']<2e-28
    assert result['phase_aligned_distance']==pytest.approx(np.sqrt(2),abs=2e-14)
    with pytest.raises(ValueError):physical_state_similarity(a,np.zeros(2))
    with pytest.raises(ValueError):physical_state_similarity(a,np.ones(3))


def test_saved_candidate_screen_replays_physical_H_and_each_endpoint_refinement():
    import hashlib
    import json
    from pathlib import Path
    from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
    from su2_nqs.benchmarks.readout_quality import compare_with_refinement
    from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_candidate_state_screen_v1'
    result = json.loads((directory/'results.json').read_text())
    spec = result['registration']['config']
    assert not result['phase_claim'] and not result['size_evolution_released']
    assert result['resources']['maximum_Ritz_dimension']==5
    assert all(result['resources'][k]==0 for k in ('new_ED','new_neural_training','new_DMRG','new_samples','new_GPU_jobs'))
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for path,sha in result['registration']['inputs'].items():
        assert digest(root/path)==sha
    for path,sha in result['registration']['sources'].items():
        assert digest(directory/'source_snapshots'/path)==sha
    assert digest(root/result['state_bundle']['path'])==result['state_bundle']['sha256']
    problem = result['physical_problem']
    assert physical_problem_signature(16,problem['bonds'],geometry=problem['geometry'],
                boundary=problem['boundary'])==problem['signature']
    with np.load(root/result['state_bundle']['path'],allow_pickle=False) as bundle:
        actions = FixedSzPairActions(bundle['basis'])
        for family,names in result['primary_quality_flows'].items():
            energies = [result['states'][n]['quality']['energy_per_site'] for n in names]
            assert np.all(np.diff(energies)<=1e-12)
            for name in names:
                v = bundle[name+'_state']; hv = actions.weighted_bonds(v,problem['bonds'])
                e = np.vdot(v,hv).real; variance = np.linalg.norm(hv-e*v)**2/16
                assert e/16==pytest.approx(result['states'][name]['quality']['energy_per_site'],abs=2e-13)
                assert variance==pytest.approx(result['states'][name]['quality']['variance_per_site'],abs=2e-13)
                assert result['states'][name]['quality']['full_S0_A1_projection_residual']<1e-10
                assert all(r['strict_su2_eligible'] for r in result['states'][name]['regions'].values())
            v = bundle[names[-1]+'_state']
            replay = reduced_density_matrix(v,bundle['basis'],spec['regions']['strip_6'])
            np.testing.assert_allclose(replay,bundle[names[-1]+'_strip_6_rho'],atol=2e-14)
        for pair in result['primary_pair_comparisons']:
            a,b = pair['family_a'],pair['family_b']
            for region in ('strip_4','strip_8'):
                replay = compare_with_refinement(bundle[f'{a}_k4_{region}_rho'],bundle[f'{b}_k4_{region}_rho'],
                    refinements_a=[bundle[f'{a}_k2_{region}_rho']],refinements_b=[bundle[f'{b}_k2_{region}_rho']])
                recorded = pair['endpoint_against_each_own_refinement'][region]
                for key,metric in recorded['metrics'].items():
                    assert metric['difference']==pytest.approx(replay['metrics'][key]['difference'],abs=2e-14)
                    assert metric['refinement_sum']==pytest.approx(replay['metrics'][key]['refinement_sum'],abs=2e-14)
                    assert metric['exceeds_declared_refinement']==replay['metrics'][key]['exceeds_declared_refinement']
        # Unaware/Sz states are retained without projection or fabricated spectra.
        for name in ('optimized_pi_flux_jastrow_two_lanczos','pi_flux_fixed_sz_gamma_a1_spatial_vit'):
            region = result['states'][name]['regions']['strip_8']
            assert not region['strict_su2_eligible'] and 'invariant_readout' not in region
    assert max(result['validation'].values())<1e-10
