from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import cost


def test_physical_sampled_path_has_no_64bit_encoding_and_exact_checks(tmp_path):
    spec=dict(geometry='square',rows=2,columns=2,j2_over_j1=0.5,multiplicity_cap=2,
              maximum_q=2,projection_chunk=2,require_gpu=False,
              seed=70,calibration_seed=71,sampling_seed=72)
    r=cost(spec,tmp_path)
    assert r['status'] == 'passed'
    assert r['validation']['raw_S_plus_relative_max'] < 2e-11
    assert r['validation']['H_relative_error'] < 2e-11
    assert r['validation']['gradient_finite_difference_error'] < 2e-6
    assert r['training_steps'] == 0 and r['energy_accuracy_claim'] is False
