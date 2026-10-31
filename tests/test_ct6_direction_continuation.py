"""Frozen-parent direction protocol, no local36-site physics training."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as campaign
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import load_checkpoint

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[1]


def plan_fixture():
    plan=json.loads((ROOT/'projects/clebsch_tree_nqs/configs/remote/ct6_relaunch_20261022/j0125_ridge005.json').read_text())
    plan.update(task=campaign.DIRECTION_TASK,optimization_control='frozen_parent_direction_then_same_width_SR',
        continuation_profile='inherit_parameters_optimizer_explicit_resampled_chains',
        reset_optimizer=False,pilot_steps=0,campaign_deadline_utc='2099-01-01T00:00:00+00:00')
    plan['model']['multiplicity_cap']=28;plan['parent']['model']['multiplicity_cap']=28
    plan['training'].update(samples=4096,chains=256,steps=32,initial_burn_proposals=1440,
                            refresh_proposals=36,spacing_proposals=36)
    plan['calibration'].update(samples=2048,chains=128,initial_burn_proposals=1440,
                               refresh_proposals=144,spacing_proposals=36)
    plan['direction_check']=dict(samples=2048,chains=128,refresh_proposals=1440,spacing_proposals=36,
        minimum_fs_sine=1e-12,maximum_fs_sine=.005,minimum_ess_fraction=.98,slope_standard_errors=2.,
        parent_energy_consistency_standard_errors=3.)
    plan['audit'].update(samples=16384,chains=128,refresh_proposals=1440,spacing_proposals=36)
    plan['seeds'].update(training=2610261111,direction=2610262222)
    plan['budget'].update(pilot_seconds=4*3600,replay_seconds=2*3600,train_seconds=17*3600)
    return plan


def test_explicit_same_width_zero_update_pilot_protocol():
    plan=plan_fixture();campaign.validate_plan(plan)
    assert plan['training']['samples']//plan['training']['chains']==16
    assert plan['pilot_steps']==0 and plan['training']['steps']==32
    assert plan['model']==plan['parent']['model']


@pytest.mark.parametrize('change',['width','parent_width','reset','pilot','steps','train_chains',
    'cal_chains','confirm_samples','confirm_FS','confirm_SE','audit','budget','same_seed','old_profile'])
def test_new_protocol_cannot_change_scope_or_masquerade_as_parent_sampler(change):
    plan=plan_fixture()
    if change=='width':plan['model']['multiplicity_cap']=32
    if change=='parent_width':plan['parent']['model']['multiplicity_cap']=20
    if change=='reset':plan['reset_optimizer']=True
    if change=='pilot':plan['pilot_steps']=2
    if change=='steps':plan['training']['steps']=128
    if change=='train_chains':plan['training']['chains']=64
    if change=='cal_chains':plan['calibration']['chains']=64
    if change=='confirm_samples':plan['direction_check']['samples']=4096
    if change=='confirm_FS':plan['direction_check']['maximum_fs_sine']=.01
    if change=='confirm_SE':plan['direction_check']['slope_standard_errors']=0
    if change=='audit':plan['audit']['samples']=65536
    if change=='budget':plan['budget']['train_seconds']=24*3600
    if change=='same_seed':plan['seeds']['direction']=plan['seeds']['audit']
    if change=='old_profile':plan['continuation_profile']='inherit_full_optimizer_sampler_rng'
    with pytest.raises(ValueError):campaign.validate_plan(plan)


def mock_workflow(monkeypatch,tmp_path,*,descent=True):
    plan=plan_fixture();plan.update(require_gpu=False,output=str(tmp_path/'run'))
    parent_dir=tmp_path/'parent';parent_dir.mkdir(exist_ok=True)
    atomic_json(parent_dir/'terminal_audit.json',dict(energy_per_site=0.,energy_chain_SE=.01))
    plan['parent']['directory']=str(parent_dir)
    plan['parent']['sealed_files']['terminal_audit.json']=file_sha256(parent_dir/'terminal_audit.json')
    p={'x':jnp.array([.1,.2])};_,m,v=_manual_adam_init(p)
    moments=(128,jax.tree.map(lambda x:x+.3,m),jax.tree.map(lambda x:x+.7,v))
    parent=(p,moments,jax.random.PRNGKey(11),jnp.zeros((64,36),jnp.int32),
            jax.random.PRNGKey(12),jnp.ones((64,36),jnp.int32))
    lineage=dict(absolute_completed_steps=444,checkpoint_sha256='parent',
                 params_sha256=pytree_sha256(p),optimizer_sha256=pytree_sha256(moments))
    monkeypatch.setattr(campaign,'load_parent',lambda _: (parent,lineage))
    monkeypatch.setattr(campaign,'numerical_runtime_fingerprint',lambda:dict(node=plan['runtime_contract']['node']))
    monkeypatch.setattr(campaign,'transitive_local_source_manifest',lambda *a,**k:{'source':'frozen'})
    calls={'transition':0,'confirmation':0,'audit':0,'branch':0}
    def branch(plan,old,rt):
        calls['branch']+=1
        state=(old[0],old[1],jax.random.PRNGKey(plan['seeds']['training']),
            jnp.zeros((256,36),jnp.int32),jax.random.PRNGKey(plan['seeds']['calibration']),
            jnp.ones((128,36),jnp.int32))
        return state,dict(parent_sampler_rng_sha256=pytree_sha256(old[2:]),
            parameters_and_optimizer_inherited_bitwise=True)
    monkeypatch.setattr(campaign,'initialize_resampled_chains',branch)
    def transition(old):
        calls['transition']+=1
        new=(jax.tree.map(lambda x:x+.001,old[0]),(old[1][0]+1,old[1][1],old[1][2]),
             old[2]+1,old[3],old[4]+1,old[5])
        return new,dict(accepted=True,full_H_calls=4096,diagnostic=.001),dict(H_seconds=.001)
    def confirmation(old,new):
        calls['confirmation']+=1
        alternating=np.tile([-1.,1.],1024)
        values=dict(psi=np.ones(2048),hpsi=np.zeros(2048),dpsi=alternating*.001,
                    dhpsi=np.full(2048,-.002 if descent else .002),candidate_psi=1+alternating*.001,
                    bits=np.zeros((2048,36),np.int32),terminal_rng=np.array([0,7],np.uint32),
                    terminal_states=np.zeros((128,36),np.int32),chain_labels=np.tile(np.arange(128),16))
        moments=campaign.direction_check_moments(*[values[k] for k in ('psi','hpsi','dpsi','dhpsi','candidate_psi')],chains=128)
        return moments,values
    monkeypatch.setattr(campaign,'runtime',lambda *a:dict(transition=transition,direction_confirmation=confirmation))
    monkeypatch.setattr(campaign.shared,'preflight',lambda *a:dict(status='passed',full_H_centers=4))
    def audit(rt,p,spec,seed,out):
        calls['audit']+=1
        return dict(params_sha256=pytree_sha256(p),samples=spec['samples'],chains=spec['chains'],
                    seed=seed,independent_from_training_and_calibration=True)
    monkeypatch.setattr(campaign.shared,'terminal_audit',audit)
    return plan,parent,calls


def test_zero_commit_pilot_replay_then_fixed32_update_endpoint(monkeypatch,tmp_path):
    plan,parent,calls=mock_workflow(monkeypatch,tmp_path)
    digest=canonical_sha256(plan);out=Path(plan['output'])
    pilot=campaign.run(plan,'pilot',digest)
    assert pilot['status']=='passed' and pilot['completed_steps']==0
    assert calls==dict(transition=2,confirmation=1,audit=0,branch=1)
    checkpoint=load_checkpoint(out/'checkpoint.pkl')
    assert pytree_sha256(checkpoint['params'])==pytree_sha256(parent[0])
    assert pytree_sha256(checkpoint['optimizer_state'])==pytree_sha256(parent[1])
    assert checkpoint['metadata']['completed_steps']==0
    assert checkpoint['metadata']['sampler_trajectory_inherited_bitwise'] is False
    seal=file_sha256(out/'checkpoint.pkl')
    assert campaign.run(plan,'replay',digest)['status']=='replay_passed'
    assert file_sha256(out/'checkpoint.pkl')==seal
    assert calls==dict(transition=3,confirmation=2,audit=0,branch=1)
    result=campaign.run(plan,'train',digest)
    assert result['completed_steps']==32 and result['absolute_completed_steps']==476
    assert result['terminal_audit']['samples']==16384 and result['terminal_audit']['chains']==128
    assert calls==dict(transition=35,confirmation=2,audit=1,branch=1)
    checkpoint=load_checkpoint(out/'checkpoint.pkl')
    assert int(checkpoint['optimizer_state'][0])==160
    assert len(checkpoint['metadata']['history'])==32
    assert result['uncommitted_direction_and_replay_training_H_centers']==3*4096
    assert result['direction_and_replay_tangent_H_JVP_centers']==2*2048


def test_failed_confirmation_preserves_parent_and_is_never_redrawn(monkeypatch,tmp_path):
    plan,parent,calls=mock_workflow(monkeypatch,tmp_path,descent=False)
    digest=canonical_sha256(plan);out=Path(plan['output'])
    report=campaign.run(plan,'pilot',digest)
    assert report['status']=='failed' and report['gate']['passed'] is False
    assert 'confirmation' in report['failure_reasons']
    before=file_sha256(out/'checkpoint.pkl');counts=dict(calls)
    assert campaign.run(plan,'pilot',digest)==report
    assert calls==counts and file_sha256(out/'checkpoint.pkl')==before
    for phase in ('replay','train'):
        with pytest.raises(ValueError,match='direction gate'):campaign.run(plan,phase,digest)
    assert calls==counts and pytree_sha256(load_checkpoint(out/'checkpoint.pkl')['params'])==pytree_sha256(parent[0])


def test_train_requires_separate_replay_and_sealed_raw_confirmation(monkeypatch,tmp_path):
    plan,_,_=mock_workflow(monkeypatch,tmp_path);digest=canonical_sha256(plan);out=Path(plan['output'])
    campaign.run(plan,'pilot',digest)
    with pytest.raises(FileNotFoundError):campaign.run(plan,'train',digest)
    campaign.run(plan,'replay',digest)
    raw=json.loads((out/'direction_check.json').read_text())
    raw['moments']['energy_slope_per_site']=-100
    atomic_json(out/'direction_check.json',raw)
    with pytest.raises(ValueError,match='direction gate'):campaign.run(plan,'train',digest)


def test_checkpointed_new_sampler_rng_restores_without_reinitialization(monkeypatch,tmp_path):
    plan,_,calls=mock_workflow(monkeypatch,tmp_path);digest=canonical_sha256(plan);out=Path(plan['output'])
    campaign.run(plan,'pilot',digest);campaign.run(plan,'replay',digest)
    original=campaign.wide_production_time_guard
    def guard(plan,phase,history,*args,**kwargs):
        if phase=='train' and len(history)>=2 and not kwargs.get('audit'):
            return dict(can_start=False,available_seconds=0.,reserved_seconds=1.,measured_step_seconds=.1,action='transition')
        return original(plan,phase,history,*args,**kwargs)
    monkeypatch.setattr(campaign,'wide_production_time_guard',guard)
    assert campaign.run(plan,'train',digest)['status']=='safe_pause'
    checkpoint=load_checkpoint(out/'checkpoint.pkl')
    assert checkpoint['metadata']['completed_steps']==2
    assert calls['audit']==0 and calls['branch']==1
    monkeypatch.setattr(campaign,'wide_production_time_guard',original)
    final=campaign.run(plan,'train',digest)
    assert final['completed_steps']==32 and calls['branch']==1 and calls['confirmation']==2


def test_fresh_process_pilot_replay_and_training_use_the_same_frozen_state(tmp_path):
    driver="""
import importlib.util
from pathlib import Path
import sys
import pytest
spec=importlib.util.spec_from_file_location('direction_test',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
patch=pytest.MonkeyPatch()
plan,parent,calls=module.mock_workflow(patch,Path(sys.argv[2]))
out=module.campaign.run(plan,sys.argv[3],module.canonical_sha256(plan))
assert out['status'] in ('passed','replay_passed','fixed_steps_complete'),out
assert calls['confirmation']==(1 if sys.argv[3] in ('pilot','replay') else 0),calls
assert calls['branch']==(1 if sys.argv[3]=='pilot' else 0),calls
"""
    for phase in ('pilot','replay','train'):
        subprocess.run([sys.executable,'-c',driver,str(Path(__file__).resolve()),str(tmp_path),phase],
                       check=True,capture_output=True,text=True,timeout=60)
    checkpoint=load_checkpoint(tmp_path/'run/checkpoint.pkl')
    assert checkpoint['metadata']['completed_steps']==32
    assert checkpoint['metadata']['absolute_completed_steps']==476
