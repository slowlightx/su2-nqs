"""CPU safety/math tests; no 20-million-column or GPU training allocation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc import budgeted_triangular_born as sampler
from projects.clebsch_tree_nqs.workflows import ct12_wide_reentry_20261101 as flow


def test_closed_plan_preserves_width_all_native_statistics_and_old_seals():
    p=flow.default_plan();flow.validate_plan(p)
    assert flow.canonical_sha256(flow.legacy_plan())==flow.OLD_PLAN_SHA
    assert p['native_parameters']==20748160 and p['maximum_attempts']==8
    with pytest.raises(PermissionError):flow.validate_plan(p,executing=True)
    p['execution_authorized']=True;flow.validate_plan(p,executing=True)
    for key,value in [('fit_records',512),('calibration_records',512),('bilateral_records',16),
                      ('audit_records',1024),('maximum_FS',.03),('sampler_aggregate_bytes',2**40),
                      ('automatically_start_pilot',True),('geometry_factors',[1.,0.]),
                      ('pilot_wall_cap_seconds',172800),('native_parameters',1495080)]:
        altered=deepcopy(p);altered[key]=value
        with pytest.raises(ValueError):flow.validate_plan(altered)
    for key,value in [('fit_records',1024.),('automatically_start_pilot',0),('execution_authorized',1)]:
        altered=deepcopy(p);altered[key]=value
        with pytest.raises(ValueError):flow.validate_plan(altered)
    for key in ('nodelist','heads','low_rank_J'):
        altered=deepcopy(p);altered[key]=1
        with pytest.raises(ValueError):flow.validate_plan(altered)
    p['seeds']['guard']=p['seeds']['selection']
    with pytest.raises(ValueError):flow.validate_plan(p)


def test_actual_wide_host_tensor_shape_bound_real_and_complex():
    cfg=MixedClebschConfig(length=12,multiplicity_cap=28,maximum_q=10)
    profile=tuple({q:28 for q in range(0,11,2)} for _ in range(14))+({0:1},)
    real=sampler.magnetic_cache_storage_plan(cfg,profile,itemsize=8,bank_count=8)
    complex_=sampler.magnetic_cache_storage_plan(cfg,profile,itemsize=16,bank_count=8)
    assert real['per_bank_bytes']/1024**3==pytest.approx(5.395976305007935)
    assert real['all_bank_bytes']/1024**3==pytest.approx(43.16781044006348)
    assert complex_['all_bank_bytes']==2*real['all_bank_bytes']
    assert real['per_bank_bytes']>768*1024**2
    assert real['per_bank_bytes']<8*1024**3 and complex_['per_bank_bytes']>8*1024**3


def _fake_sampler_setup(monkeypatch):
    monkeypatch.setattr(sampler,'magnetic_cache_storage_plan',lambda cfg,profile,**kw:
        dict(per_bank_bytes=10,all_bank_bytes=10*kw['bank_count']))
    calls=[]
    class Fake:
        diagnostics={'unique_magnetic_tensor_bytes':10}
        def draw(self,rng):
            return SimpleNamespace(bits=rng.permutation(np.array([0,0,1,1],np.int32)),
                                   log_probability_replay_error=0.)
    def factory(bank,cfg,**kw):calls.append(kw);return Fake()
    rt=dict(cfg=SimpleNamespace(length=2),profile=(),images=np.array([[0,1,2,3],[3,2,1,0]]),
        inverse=np.array([[0,1,2,3],[3,2,1,0]]),edges=np.array([[0,1],[1,2]]),probability=np.array([.25,.75]),epsilon=.125)
    banks=[dict(top=((np.zeros(1),),)) for _ in range(2)]
    return rt,banks,factory,calls


def test_budgeted_drawer_refuses_aggregate_and_other_live_allocations_before_construct(monkeypatch):
    rt,banks,factory,calls=_fake_sampler_setup(monkeypatch)
    for budgets in [dict(per_bank_bytes=9,aggregate_bytes=30,host_array_bytes=50),
                    dict(per_bank_bytes=10,aggregate_bytes=19,host_array_bytes=50),
                    dict(per_bank_bytes=10,aggregate_bytes=30,host_array_bytes=30,other_live_host_bytes=11)]:
        with pytest.raises(MemoryError):sampler.BudgetedTriangularOrbitDrawer(banks,rt,sampler_factory=factory,**budgets)
    assert not calls


def test_frozen_sampler_reuse_exact_seed_replay_and_independent_child_streams(monkeypatch):
    rt,banks,factory,calls=_fake_sampler_setup(monkeypatch)
    frozen=sampler.BudgetedTriangularOrbitDrawer(banks,rt,per_bank_bytes=10,aggregate_bytes=20,
        host_array_bytes=40,other_live_host_bytes=20,sampler_factory=factory)
    a,first=frozen.draw(256,123,bridge=True)
    b,replay=frozen.draw(256,123,bridge=True)
    c,other=frozen.draw(256,124,bridge=True)
    assert len(calls)==2 and all(x['memory_limit_bytes']==10 for x in calls)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,c)
    assert first['rng_before']==replay['rng_before']
    assert first['rng_before']['stream_identities']!=other['rng_before']['stream_identities']
    assert np.all(a.sum(1)==2) and first['target_wavefunction_changed'] is False
    assert first['empirical_branch_weights_used'] is False
    assert len(first['base_draw']['latents'])==first['base_records']
    assert frozen.diagnostics['measured_all_bank_tensor_bytes']==20
    rt['remaining']=lambda:0
    with pytest.raises(TimeoutError):frozen.draw(4,123,bridge=False)


def test_geometry_large_amplitude_bracket_targets_actual_FS_not_coefficient_norm():
    signs=np.tile([-1.,1.],512)
    phi=np.stack([np.cos(.004*f)+np.sin(.004*f)*signs for f in flow.FACTORS],axis=1)
    chosen,info=flow.select_geometry(phi,np.ones(1024))
    assert chosen==1 and info['accepted'] # factor4 -> .016 finiteFS
    assert info['guard_not_used_for_reselection']
    corrupted=phi.copy()
    for i,f in enumerate(flow.FACTORS):corrupted[512:,i]=np.cos(.015*f)+np.sin(.015*f)*signs[512:]
    chosen,veto=flow.select_geometry(corrupted,np.ones(1024))
    assert chosen==7 and not veto['accepted']


def test_full_cached_fit_is_same_legacy_F_direction_and_not_two_F():
    from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr
    A=jnp.array([[1.,0.],[0.,1.],[1.,1.],[-1.,2.]])
    p={'w':jnp.array([0.,1.])};zeros=jax.tree.map(jnp.zeros_like,p)
    state=(p,(jnp.asarray(45),zeros,zeros),np.random.default_rng(17).bit_generator.state)
    rows=jnp.arange(4)[:,None]
    batch=lambda p,x:(A@p['w'])[x[:,0]]
    rt=dict(prepare=lambda x:x,amplitude=batch)
    psi=np.asarray(batch(p,rows));h=np.array([.7,-.3,.4,.2]);q=np.array([.13,.27,.31,.29])
    proposed,info,cache=flow.cached_fit(flow.default_plan(),rt,state,rows,psi,h,q,identity='toy')
    legacy=make_importance_amplitude_sr(lambda p,x:batch(p,x[None])[0],p,damping=.5,
        score_chunk=1,sr_tangent_fs_radius=.003)
    fitted=jax.block_until_ready(legacy(p,state[1],rows,h,q,.03))
    expected=flow.old.normalize_legacy_proposal(p,fitted)
    np.testing.assert_allclose(proposed[0]['w'],expected[0]['w'],rtol=2e-12,atol=2e-12)
    assert int(proposed[1][0])==46 and proposed[2]==state[2]
    assert info['metric_is_F_not_two_F'] and info['sampled_tangent_FS']==.015
    assert np.linalg.norm(cache.jacobian[0])>0


def test_role_ledger_preserves_next_rng_and_excludes_failed_old_and_future_roles():
    p=flow.default_plan();prior=flow.legacy_plan();rng=np.random.default_rng(98).bit_generator.state
    before=deepcopy(rng);ledger=flow.role_ledger(p,rng,prior)
    assert rng==before and len(ledger['future_native_role_seeds'])==24
    collision=deepcopy(p);collision['seeds']['guard']=prior['seeds']['wide28_guard']
    with pytest.raises(ValueError):flow.role_ledger(collision,rng,prior)
    collision=deepcopy(p);collision['seeds']['guard']=ledger['future_native_role_seeds'][0]
    with pytest.raises(ValueError):flow.role_ledger(collision,rng,prior)
    with pytest.raises(ValueError):flow.role_ledger(p,rng,prior,historical={p['seeds']['guard']})
    with pytest.raises(ValueError):flow.role_ledger(p,rng,prior,historical={ledger['future_native_role_seeds'][0]})


def test_actual_cold_step_forecast_does_not_claim_solve_only_speed_or_48h_fit():
    p=flow.default_plan();timing=dict(H_seconds_per_center=6.358794931991259,
        H_JVP_warm_seconds_per_center=22.30236675,actual1024_amplitude_warm_seconds=18.960603,
        SR_cold_seconds=1935.20591,SR_warm_seconds=275.996742)
    gate=flow.gate_forecast(p,timing);pilot=flow.pilot_forecast(p,timing)
    assert gate['forecast_seconds']<18*3600
    assert pilot['forecast_before_sampling_seconds']>48*3600
    assert pilot['forecast_before_sampling_seconds']<72*3600
    other=flow.pilot_forecast(p,dict(timing,SR_warm_seconds=1e9))
    assert other==pilot and pilot['terminal_H_centers']==8192
    measured=flow.pilot_forecast(p,timing,gate_measurements=dict(
        geometry=dict(full_geometry_wall_seconds=400.),
        bilateral=dict(sampling_and_constructor_seconds=120.),
        confirmation=dict(sampling_and_constructor_seconds=200.)))
    assert measured['full_forecast_seconds']==pytest.approx(pilot['forecast_before_sampling_seconds']+1.5*(8*(400+240)+800))


def test_raw_independent_artifacts_bound_and_no_escape(tmp_path):
    for name in ('geometry/fixture.npz','geometry/draws.pkl','geometry/result.json','geometry/sampler_budget.json'):
        target=tmp_path/name;target.parent.mkdir(exist_ok=True);target.write_bytes(name.encode())
    result=dict(science_artifact_sha256=flow.science_artifact_seals(tmp_path))
    flow.verify_science_artifacts(tmp_path,result,complete=False)
    with pytest.raises(ValueError):flow.verify_science_artifacts(tmp_path,result,complete=True)
    (tmp_path/'geometry/fixture.npz').write_bytes(b'changed physical geometry')
    with pytest.raises(ValueError):flow.verify_science_artifacts(tmp_path,result,complete=False)
    with pytest.raises(ValueError):flow.verify_science_artifacts(tmp_path,dict(science_artifact_sha256={
        '../outside': 'a'*64,'a':'a'*64,'b':'a'*64,'c':'a'*64}),complete=False)


def test_new_launcher_no_node_pin_and_gate_does_not_start_pilot():
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'configs/remote/ct12_wide_reentry_20261101/plan.json').read_text())
    flow.validate_plan(plan,executing=True)
    script=(root/'configs/remote/ct12_wide_reentry_20261101/run.sbatch').read_text()
    assert '#SBATCH --mem=320G' in script and '#SBATCH --time=18:00:00' in script
    assert '--nodelist' not in script and '#SBATCH -w' not in script
    assert 'separate same-certificate root release required' in script
    gate=script.split('if [ "$NQS_CT12_MODE" = gate ]; then',1)[1].split('\nelse\n',1)[0]
    assert '--phase gate' in gate and '--phase replay' in gate and 'pilot_prepare' not in gate
