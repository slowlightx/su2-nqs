"""Data roles and fixed state/budget are part of the update rule."""
import json
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_finite_schur_reliability import CONFIG, registration, combined_regions


def test_direction_cannot_refit_from_consumed_audits_or_change_architecture():
    spec=json.loads(CONFIG.read_text());reg=registration(spec)
    assert spec['seeds']['audit'] != spec['seeds']['independent_calibration']
    assert spec['convergence_budget']['maximum_attempts']==8
    assert not any('audit' in p or 'candidate' in p for p in reg['inputs'])
    assert len([p for p in reg['inputs'] if 'spin_finite_schur_curvature_v1' in p and p.endswith('products.npz')])==2
    for changed in ({'maximum_q':2},{'response_safety':.25},
                    {'seeds':{**spec['seeds'],'independent_calibration':2026100274}}):
        with pytest.raises(ValueError):registration({**spec,**changed})


def test_union_distance_includes_all_records_six_folds_and_every_chain():
    old=np.tile(np.arange(8),8);new=np.tile(np.arange(8),16)
    masks=combined_regions(old,new)
    assert len(masks)==23 and masks['full'].all()
    assert masks['fit_full'].sum()==64 and masks['calibration_full'].sum()==128
    for dataset,count in [('fit',8),('calibration',16)]:
        assert np.array_equal(masks[dataset+'_half_0']|masks[dataset+'_half_1'],masks[dataset+'_full'])
        for chain in range(8):assert masks[f'{dataset}_chain_{chain}'].sum()==count
