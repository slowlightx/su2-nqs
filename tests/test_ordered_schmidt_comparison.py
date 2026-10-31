"""Same-cut physical readout replay, without new states or SU2 spectrum repair."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.ground_state_structure_analysis import compare_ordered_spatial_controls
from projects.clebsch_tree_nqs.workflows.ground_state_structure_analysis import analyze_schmidt_response
from projects.clebsch_tree_nqs.workflows.dmrg_region_probe import _canonical

ROOT=Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/results/ground_state_structure_20261014'
INPUTS=['tower/ordered_J0_chi2048_spatial_v1/worker_result.json',
        'tower/ordered_J0_chi4096_spatial_v1/worker_result.json',
        'tower/local_chi5120_spatial_v1/worker_result.json',
        'tower/chi8192_spatial_v1/worker_result.json',
        'tower/chi10240_spatial_v1/worker_result.json']


def _copy_inputs(target):
    for name in INPUTS:
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,p)


def test_actual_same_cut_contrast_and_quality_drift_without_phase_claim(tmp_path):
    _copy_inputs(tmp_path)
    r=compare_ordered_spatial_controls(tmp_path)
    ordered=r['cases']['J0'][-1]; frustrated=r['cases']['J0125'][-1]
    np.testing.assert_allclose(ordered['singlet_triplet_xi_separation'],
                               3.0726836186175275-2.1738157059278134,atol=1e-14)
    np.testing.assert_allclose(frustrated['interior_polarization']-ordered['interior_polarization'],
                               -.26792693411811014+.07922564246807041,atol=1e-14)
    for name,rows in r['cases'].items():
        for v in rows:
            np.testing.assert_allclose(sum(v['first_triplet_profile']),1.,atol=1e-8)
            assert v['first_triplet_higher_spin_weight_bound'] < 1e-5
            assert not v['strict_SU2_spectrum_claimed']
        assert all(v['checkpoint_sha256'] for v in rows)
    assert not r['QSL_certified'] and not r['exact_SU2_spectra_claimed']
    assert r['new_ED']==r['new_training']==r['new_Born_samples']==0
    assert r['comparisons']['interior_polarization']['exceeds_sum_of_last_observed_drifts']
    assert r['comparisons']['interior_polarization']['drift_is_not_a_certified_quality_error_bound']
    # The actual ordered control also violates the ideal quadratic-rotor bound;
    # the analysis therefore must not equate that violation with a QSL.
    assert ordered['spin2_to_triplet_gap_ratio'] > 3
    assert (tmp_path/'spatial/ordered_comparison/ordered_vs_frustrated.png').is_file()


def test_changed_worker_and_mismatched_geometry_fail_closed(tmp_path):
    _copy_inputs(tmp_path)
    p=tmp_path/INPUTS[0];d=json.loads(p.read_text())
    d['readout']['cut']=16
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='changed physical worker'):
        compare_ordered_spatial_controls(tmp_path)
    # Re-seal only a synthetic temporary invalid fixture, never the archive.
    d['record_sha256']=_canonical({k:v for k,v in d.items() if k!='record_sha256'})
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='identical physical'):
        compare_ordered_spatial_controls(tmp_path)


def test_frozen_spin_dark_chiral_bright_branch_and_neural_replica_crosscheck(tmp_path):
    source=ROOT.parent/'schmidt_response_20261015'
    inputs=[]
    for name in ('ordered_J0_chi2048','ordered_J0_chi4096','chi8192','chi10240'):
        inputs.extend(f'{kind}/{name}/worker_result.json' for kind in ('mps','chirality'))
    for name in ('tri_j0','tri_j0125'):
        inputs.extend(f'vit_orbit/{name}/{file}' for file in ('results.json','prefix18.npz','compact6.npz'))
    for name in inputs:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,path)
    r=analyze_schmidt_response(tmp_path)
    ordered=r['mps_quality_sequences']['J0'][-1]
    frustrated=r['mps_quality_sequences']['J0125'][-1]
    assert ordered['vector_chirality_mechanism']['chirality_active_level'] == 2
    assert frustrated['vector_chirality_mechanism']['chirality_active_level'] == 0
    # Independent replay of every site contribution, not an inferred q label.
    raw=json.loads((tmp_path/'mps/chi10240/worker_result.json').read_text())['readout']
    matrices=raw['leading_spin_content']['raising_transitions']['0_to_2']['site_matrices']
    norm=sum(v['real'][0][0]**2+v['imag'][0][0]**2 for v in matrices.values())
    np.testing.assert_allclose(norm,frustrated['lowest_triplet_branches'][0]['site_squared_norm'],atol=1e-19)
    assert norm < 2e-6 and ordered['lowest_triplet_branches'][0]['site_squared_norm'] > 1
    assert frustrated['vector_chirality_mechanism']['chiral_strength'] > 9
    assert r['ViT_K_response_coupling_contrast']['signal_over_SE'] > 4
    assert not r['QSL_certified'] and not r['exact_SU2_assumed']
    assert r['new_training'] == r['new_ED'] == r['new_Born_samples'] == 0
    assert (tmp_path/'analysis/magnetic_chiral_reorganization.png').exists()
