"""Paper-capacity contract and bounded optimizer reentry; no new large solve."""
import copy
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.square_clebsch_paper_continue import validate_plan
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training import (
    _validate_endpoint_budget_change, _validate_optimizer_condition_change, train,
)
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import cost, components
from su2_nqs.models.literature_clebsch_tree import clebsch_layers
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.frontier.provenance import file_sha256

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'projects/clebsch_tree_nqs/configs/remote/square_paper_continue_20261020'


@pytest.mark.parametrize('case',['j0','j05'])
def test_paper_width_is_per_sector_and_existing_native_capacity_is_unchanged(case):
    plan=json.loads((BASE/f'{case}_plan.json').read_text())
    spec=json.loads((BASE/f'{case}_train.json').read_text())
    validate_plan(plan,spec)
    assert plan['training_config_sha256']==file_sha256(BASE/f'{case}_train.json')
    cfg,_,_,_,images,_=components(spec)
    layers=clebsch_layers(cfg)
    assert len(images)==512
    assert sum(x.parameter_size for x in layers)+layers[-1].widths[0]==227958
    assert max(x.widths[0] for x in layers)==16
    assert all(max(x.widths[5:])<=8 for x in layers)
    assert cfg.maximum_q==8  # jmax4, not jmax8.
    assert spec['steps']==plan['parent']['completed_steps']+128
    assert sum(plan['phase_seconds'].values())==17*3600


@pytest.mark.parametrize('change',['width','qmax','optimizer','sample','zero_guard','budget','audit','parent'])
def test_scope_and_budget_cannot_drift(change):
    plan=json.loads((BASE/'j0_plan.json').read_text());spec=json.loads((BASE/'j0_train.json').read_text())
    if change=='width':spec['multiplicity_cap']=20
    if change=='qmax':spec['maximum_q']=10
    if change=='optimizer':spec['sr_use_adam']=False
    if change=='sample':spec['adam_samples']=1024
    if change=='zero_guard':spec['strict_nonzero_updates']=False
    if change=='budget':plan['phase_seconds']['train']+=1
    if change=='audit':plan['audit']['audit_seeds'][0]=spec['training_seed']
    if change=='parent':plan['parent']['completed_steps']=33
    with pytest.raises(ValueError):validate_plan(plan,spec)


def test_endpoint_budget_declaration_is_exact_and_not_a_physics_override():
    old=dict(steps=32,milestones=[0,32],maximum_training_seconds=200,maximum_q=8)
    new=dict(old,steps=160,milestones=[0,32,34,36,160],maximum_training_seconds=300)
    change=dict(from_completed_steps=32,additional_updates=128,time_extension_seconds=100,
        registered_steps=dict(from_value=32,to_value=160),
        registered_milestones=dict(from_value=old['milestones'],to_value=new['milestones']))
    a,b=old.copy(),new.copy();_validate_endpoint_budget_change(a,b,dict(endpoint_budget_change=change))
    assert a==b==dict(maximum_q=8)
    for key,value in [('additional_updates',True),('additional_updates',1001),
                      ('time_extension_seconds',-1),('from_completed_steps',33)]:
        bad=copy.deepcopy(change);bad[key]=value
        with pytest.raises(ValueError):
            _validate_endpoint_budget_change(old.copy(),new.copy(),dict(endpoint_budget_change=bad))
    with pytest.raises(ValueError):
        _validate_endpoint_budget_change(old.copy(),dict(new,maximum_training_seconds=301),
                                         dict(endpoint_budget_change=change))


def test_actual_optimizer_reentry_split_restart_keeps_params_chains_rng_and_moments(tmp_path):
    spec=dict(geometry='square',rows=2,columns=2,j2_over_j1=.5,multiplicity_cap=2,
        maximum_q=2,projection_chunk=2,require_gpu=False,seed=12,calibration_seed=13,sampling_seed=14)
    gate=cost(spec,tmp_path/'cost')
    spec.update(cost_parent=dict(directory=str(tmp_path/'cost'),checkpoint_sha256=gate['checkpoint_sha256'],
        registration_sha256=gate['registration_sha256']),training_seed=15,chains=4,steps=3,
        sr_warm_steps=1,warm_samples=8,adam_samples=8,burn_proposals=8,proposals_between_snapshots=4,
        score_chunk=1,H_sample_chunk=4,sr_damping=.01,maximum_FS_sine=.05,warm_lr=.005,adam_lr=.001,
        reset_moments_at_phase_switch=True,milestones=[0,1,2,3],maximum_training_seconds=120)
    result=train(spec,tmp_path/'parent',2,120)
    continuation=dict(directory=str(tmp_path/'parent'),checkpoint_sha256=result['checkpoint_sha256'],
        registration_sha256=result['registration_sha256'],source_workspace=str(ROOT),
        same_projected_physical_state_proved=True,reset_optimizer_moments=True,
        optimizer_reset_reason='registered paper-like reentry, not strict old optimizer continuation',
        optimizer_condition_change=dict(sr_warm_steps=dict(from_value=1,to_value=4)),
        endpoint_budget_change=dict(from_completed_steps=2,additional_updates=6,time_extension_seconds=60,
            registered_steps=dict(from_value=3,to_value=8),
            registered_milestones=dict(from_value=[0,1,2,3],to_value=[0,2,4,6,8])))
    child=dict(spec,steps=8,sr_warm_steps=4,milestones=[0,2,4,6,8],maximum_training_seconds=180,
               training_resume_from=continuation)
    train(child,tmp_path/'whole',8,180)
    train(child,tmp_path/'split',4,180);train(child,tmp_path/'split',6,180);train(child,tmp_path/'split',8,180)
    a=load_checkpoint(tmp_path/'whole/checkpoint.pkl');b=load_checkpoint(tmp_path/'split/checkpoint.pkl')
    assert int(b['optimizer_state'][0])==4
    for name in ('params','optimizer_state'):
        for x,y in zip(jax.tree.leaves(a[name]),jax.tree.leaves(b[name]),strict=True):
            np.testing.assert_array_equal(x,y)
    for name in ('rng','sampler_states'):
        np.testing.assert_array_equal(a['metadata'][name],b['metadata'][name])
    assert a['metadata']['optimizer_sha256']==b['metadata']['optimizer_sha256']
    assert a['metadata']['sampler_rng_sha256']==b['metadata']['sampler_rng_sha256']
