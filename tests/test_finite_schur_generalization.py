"""Pre-data controls, disjoint data roles and physical distance coverage."""
import json
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_finite_schur_generalization import (
    CONFIG, METHODS, inputs, registration, trust_regions,
)


def test_fixed_ladder_and_geometry_only_data_role():
    spec=json.loads(CONFIG.read_text());reg=registration(spec)
    assert spec['methods']==list(METHODS) and spec['primary_method']=='hermitian'
    assert spec['steps']==1 and len(set(spec['seeds'].values()))==3
    assert not any('audit.npz' in p or 'candidate' in p for p in reg['inputs'])
    for arm in ('network','free'):
        fit,geo,cache,units=inputs(spec,arm)
        assert set(geo)=={'states','carrier','q','groups'}
        assert len(fit['q'])==64 and len(geo['q'])==128
        assert cache['energy_gradients'].shape==cache['residual_gradients'].shape
        assert cache['control_jacobian'].shape[1]==len(cache['direction'])
        assert units['residual_scale']>0
    for changed in ({'maximum_q':2},{'steps':8},{'relative_cutoff':1e-3},
                    {'primary_method':'pool'},{'response_safety':.25},
                    {'seeds':{**spec['seeds'],'audit':spec['seeds']['calibration']}}):
        with pytest.raises(ValueError):registration({**spec,**changed})


def test_trust_union_contains_all_three_datasets_folds_and_24_chains():
    data=[dict(q=np.ones(n),groups=np.repeat(np.arange(8),n//8)) for n in (64,128,128)]
    masks=trust_regions(data)
    assert len(masks)==34 and masks['full'].all()
    assert len(masks['full'])==320
    for name,n in zip(('fit','geometry','calibration'),(64,128,128),strict=True):
        assert masks[name+'_full'].sum()==n
        assert np.array_equal(masks[name+'_half_0']|masks[name+'_half_1'],masks[name+'_full'])
        assert not (masks[name+'_half_0']&masks[name+'_half_1']).any()
        assert sum(masks[f'{name}_chain_{i}'].sum() for i in range(8))==n
