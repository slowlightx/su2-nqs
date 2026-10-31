"""Closed pilot controls and replay persistence without CPU production training."""
from copy import deepcopy
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_active_two_step_pilot import (
    select_step, _state_save, _state_load, _seeds, products, registered_bonds,
    validate_runtime_fixture, validate_parent_coupling, checked_plan,
)


def test_shared_selection_and_independent_guard_never_pick_smaller_step():
    x=np.linspace(-1,1,512); parent=np.ones(512)
    # Actual normalized-ray perturbation with largest fit radius.002.
    phi=np.stack([parent+.0035*f*x for f in (1,.5,.25,.125,0)],axis=1)
    chosen,report=select_step(phi,np.ones(512))
    assert chosen==0 and report['accepted']
    phi[256:,0]=parent[256:]+.03*x[256:]
    chosen,report=select_step(phi,np.ones(512))
    assert chosen==4 and not report['accepted']
    assert not report['guard_used_to_select_smaller_fraction']


def test_trained_parameter_optimizer_and_next_rng_restore(tmp_path):
    rng=np.random.default_rng(2610279910)
    old=deepcopy(rng.bit_generator.state); seeds,following=_seeds(old)
    p={'w':jnp.asarray([1.2,-.7])};m=(jnp.asarray(2),{'w':jnp.zeros(2)},{'w':jnp.ones(2)})
    _state_save(tmp_path/'state.pkl',(p,m,following),completed_updates=2)
    restored=_state_load(tmp_path/'state.pkl')
    assert int(restored[1][0])==2
    np.testing.assert_array_equal(restored[0]['w'],p['w'])
    assert _seeds(restored[2])==_seeds(following)
    assert seeds!=_seeds(restored[2])[0]


def test_complete_H_product_budget_guard_preserves_fixed_record_count(tmp_path):
    def query(bank,rows):
        x=np.asarray(rows).sum(1)
        return x,x*2,x*3,x*4
    rt=dict(H=query,remaining=lambda:2.,warm_H_seconds=1.,audit_reserve_seconds=3.)
    rows=np.zeros((4,144),np.int32)
    with pytest.raises(TimeoutError):products(rt,[{}],rows,tmp_path,'fit',2)
    assert not (tmp_path/'fit_products.npz').exists()
    rt['remaining']=lambda:1000.
    data,warm=products(rt,[{}],rows,tmp_path,'fit',2)
    assert data.shape==(4,1,4) and np.isfinite(warm)


@pytest.mark.parametrize('records,split',[(512,256),(64,32)])
def test_full_transition_fresh_process_fixture_replay_and_rng(monkeypatch,tmp_path,records,split):
    """Mock products test transaction control, not physics or convergence."""
    import projects.clebsch_tree_nqs.workflows.triangle_active_two_step_pilot as module
    def draw(banks,rt,count,seed,bridge):
        rng=np.random.default_rng(seed)
        return rng.integers(0,2,size=(count,144),dtype=np.int32),{'seed':seed}
    monkeypatch.setattr(module,'exact_draw',draw)
    monkeypatch.setattr(module,'canonical_active_sampling_bank',lambda p,c,w:(p,{'original_raw_norm':1.}))
    def raw(p,row):return p['w'][0]+p['w'][1]*(row[0]-.5)
    monkeypatch.setattr(module,'make_feature_lut_amplitude',lambda *args,**kwargs:raw)
    def H(p,rows):
        phi=jax.vmap(lambda x:raw(p,x))(rows)
        return phi,-phi,phi*phi,phi*phi
    def update(p,m,rows,h,q,lr):
        return {'w':p['w']+jnp.asarray([0.,.001])},(m[0]+1,m[1],m[2]),{'accepted':True}
    rt=dict(cfg=None,profile=[],images=np.tile(np.arange(144),(16,1)),
        H=H,update=update,prepare=lambda p:p)
    plan=dict(H_chunk=8,projection_chunk=16,action_chunk=4,maximum_peak_device_bytes=70*1024**3,
        fit_records=records,calibration_split=split)
    p={'w':jnp.asarray([1.,.05])};z={'w':jnp.zeros(2)}
    rng=np.random.default_rng(1234).bit_generator.state
    state=(p,(jnp.asarray(0),z,z),rng)
    first,receipt=module.transition(plan,rt,state,tmp_path/'step1')
    assert receipt['accepted'] and int(first[1][0])==1
    assert receipt['full_active_jacobian_rows']==records
    assert receipt['full_H_centers']==records+2
    assert np.load(tmp_path/'step1/fit_products.npz')['rows'].shape==(records,144)
    assert np.load(tmp_path/'step1/geometry_fixture.npz')['rows'].shape==(2*split,144)
    replay,checked=module.transition(plan,rt,state,tmp_path/'step1',replay=True)
    assert checked['maximum_state_relative_replay_error']==0
    assert checked['selected_index']==receipt['selected_index']
    np.testing.assert_array_equal(replay[0]['w'],first[0]['w'])
    assert _seeds(replay[2])==_seeds(first[2])


def test_J0_contains_only432_true_nearest_neighbor_bonds():
    """Independent minimum-image metric, not a second call to bond builder."""
    zero=registered_bonds(0.)
    assert len(zero)==432 and all(weight==1. for _,_,weight in zero)
    degree=np.zeros(144,int)
    for a,b,_ in zero:
        ra,ca=divmod(a,12); rb,cb=divmod(b,12)
        lengths=[(rb-ra+12*u)**2+(cb-ca+12*v)**2
            -(rb-ra+12*u)*(cb-ca+12*v) for u in (-1,0,1) for v in (-1,0,1)]
        assert min(lengths)==1
        degree[a]+=1;degree[b]+=1
    np.testing.assert_array_equal(degree,np.full(144,6))
    frustrated=registered_bonds(.125)
    assert len(frustrated)==864
    assert sum(w==1. for _,_,w in frustrated)==432
    assert sum(w==.125 for _,_,w in frustrated)==432
    for invalid in (True,None,-.125,.1,np.nan):
        with pytest.raises(ValueError):registered_bonds(invalid)


def test_new_fixture_is_bound_to_actual_J0_parent_and_Hamiltonian():
    plan=dict(j2=0.,parent_sha256='a'*64,initial_replay_fixture_sha256='b'*64)
    rows=np.tile(np.tile([0,1],72),(2,1)).astype(np.int32)
    fixture=dict(rows=rows,products=np.ones((2,4)),j2=np.asarray(0.),n_sites=np.asarray(144),
        physical_H_bonds=np.asarray(432),complete_A1_images=np.asarray(1728),
        parent_checkpoint_sha256=np.asarray('a'*64))
    validate_runtime_fixture(plan,fixture)
    for key,bad in [('j2',.125),('physical_H_bonds',864),('parent_checkpoint_sha256','c'*64)]:
        with pytest.raises(ValueError,match='mismatch'):
            validate_runtime_fixture(plan,dict(fixture,**{key:np.asarray(bad)}))
    validate_parent_coupling(plan,dict(j2=0.))
    with pytest.raises(ValueError,match='actual optimized J2'):
        validate_parent_coupling(plan,dict(j2=.125))
    # The historical identity-free fixture is allowed only at its original
    # coupling and exact original parent SHA, never as a J0 fixture.
    legacy=dict(plan,j2=.125,parent_sha256='56cd517f5bf30805317a08994a4465a7ba6cced58a16a2b33f1fd2e2282896a3',
        initial_replay_fixture_sha256='0f2709f7acf418b1cf74c72fb14a4b283da2777b49874fae6b7d6571e084ad86')
    validate_runtime_fixture(legacy,dict(rows=rows,products=np.ones((2,4))))
    with pytest.raises(ValueError):
        validate_runtime_fixture(dict(legacy,j2=0.),dict(rows=rows,products=np.ones((2,4))))


def test_original_plan_unchanged_and_J0_record_panels_are_explicit():
    import json
    from pathlib import Path
    original=json.loads((Path(__file__).resolve().parents[1]/
        'configs/remote/ct12_active_pilot_20261027/plan.json').read_text())
    checked_plan(original)
    assert (original['j2'],original['fit_records'],original['calibration_records'],original['audit_records'])==(.125,512,512,1024)
    new=dict(original,j2=0.,initial_endpoint='actual_J0_parent_preselected',
        fit_records=64,calibration_records=64,calibration_split=32,audit_records=128)
    checked_plan(new)
    for key,bad in [('j2',.1),('fit_records',True),('fit_records',33),('calibration_records',512),('audit_records',130)]:
        with pytest.raises(ValueError):checked_plan(dict(new,**{key:bad}))


def test_already_submitted_J0125_source_bundle_cannot_be_rebuilt():
    from projects.clebsch_tree_nqs.workflows.triangle_active_pilot_bundle import build
    with pytest.raises(PermissionError,match='immutable'):
        build()
