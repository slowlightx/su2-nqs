"""J0 reuses the full-support native-W workflow without changing J.125 kernels."""
from copy import deepcopy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct6_importance_amplitude_continuation as w
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from test_ct6_importance_amplitude_continuation import fixture,plan


def j0_plan():
    p=plan();profile=w.protocol_profile(dict(task=w.J0_TASK))
    p.update({key:value for key,value in profile.items() if key!='parent_global_step'})
    p['parent']['global_step']=profile['parent_global_step']
    return p


def test_explicit_joint_j0_profile_and_actual_complete_hamiltonian():
    p=j0_plan();w.validate_plan(p)
    runtime=w.runtime_plan(p)
    assert runtime['arity']=='ternary_joint' and runtime['j2_over_j1']==0
    assert runtime['model']==dict(length=6,multiplicity_cap=28,maximum_q=10)
    assert len(triangular_j1j2_bonds(6,6,j2=runtime['j2_over_j1'],periodic=True))==108
    assert len(triangular_hnf_gamma_a1_group(((6,0),(0,6))))==432
    for field,value in [('arity','binary_truncated'),('j2_over_j1',.125),('updates',3),
                        ('terminal_audit_records',16384),('wall_cap_seconds',14400),
                        ('interim_after_updates',2),('audit_reserve_seconds',0)]:
        bad=deepcopy(p);bad[field]=value
        with pytest.raises(ValueError):w.validate_plan(bad)
    bad=deepcopy(p);bad['parent']['global_step']=476
    with pytest.raises(ValueError):w.validate_plan(bad)


def test_original_j0125_runtime_is_exactly_unchanged():
    p=plan();w.validate_plan(p)
    assert w.runtime_plan(p)==w.diagnostic.runtime_plan(w.kernel_plan(p))
    assert w.protocol_profile(p)['parent_global_step']==476


def test_j0_two_committed_updates_process_restore_and_one_fresh8192_audit(monkeypatch,tmp_path):
    p,parent,out,calls=fixture(monkeypatch,tmp_path)
    profile=w.protocol_profile(dict(task=w.J0_TASK))
    p.update({key:value for key,value in profile.items() if key!='parent_global_step'})
    p['parent']['global_step']=456
    monkeypatch.setattr(w,'checked_parent',lambda _:(parent,dict(native_parameter_count=2)))
    monkeypatch.setattr(w.os,'getpid',lambda:100)
    first=w.run(p,'prepare',out)
    assert first['status']=='prepared_first_transition_uncommitted'
    assert w.load_training_state(out/'checkpoint.pkl')[1]['absolute_completed_steps']==456
    assert first['cost_gate']['maximum_not_guaranteed_updates']==2
    monkeypatch.setattr(w.os,'getpid',lambda:101)
    assert w.run(p,'replay',out)['status']=='replay_passed'
    assert w.load_training_state(out/'checkpoint.pkl')[1]['absolute_completed_steps']==456
    monkeypatch.setattr(w.os,'getpid',lambda:102)
    result=w.run(p,'train',out)
    state,meta=w.load_training_state(out/'checkpoint.pkl')
    assert result['status']=='fixed2_updates_complete_independent_audit'
    assert result['absolute_completed_steps']==458 and result['completed_updates']==2
    assert meta['absolute_completed_steps']==458 and meta['original_parent_global_step']==456
    assert int(state[1][0])==6 and state[2]['next_step']==3
    assert calls['transitions']==[1,1,2]
    assert calls['audits']==[(8192,p['seeds']['audit'])]
    assert 'interim_audit' not in result and not (out/'interim_audit').exists()
    registration=json.loads((out/'registration.json').read_text())
    assert registration['initial_global_step']==456
    assert registration['declared_maximum_endpoint_global_step']==458
    w.run(p,'train',out)
    assert calls['transitions']==[1,1,2]


def make_sealed_parent(tmp_path):
    p=j0_plan();directory=tmp_path/'parent';directory.mkdir()
    p['parent']['directory']=str(directory)
    reg=dict(model=p['model'],arity=p['arity'],j2_over_j1=0.,binary_rank=8)
    reg['sha256']=canonical_sha256(reg);atomic_json(directory/'registration.json',reg)
    params={'x':jnp.array([1.,.25])};mom=(3,{'x':jnp.zeros(2)},{'x':jnp.ones(2)})
    state=(params,mom,jnp.array([1,2],jnp.uint32),jnp.zeros((2,36),jnp.int32),
        jnp.array([3,4],jnp.uint32),jnp.ones((2,36),jnp.int32))
    w.diagnostic.legacy.shared.save_state(directory/'checkpoint.pkl',state,
        dict(registration_sha256=reg['sha256'],absolute_completed_steps=456))
    audit=dict(params_sha256=pytree_sha256(params),independent_from_training_and_calibration=True)
    atomic_json(directory/'terminal_audit.json',audit)
    result=dict(status='fixed_steps_complete',absolute_completed_steps=456,
        params_sha256=pytree_sha256(params),checkpoint_sha256=file_sha256(directory/'checkpoint.pkl'),
        terminal_audit=audit)
    atomic_json(directory/'results.json',result)
    np.savez(directory/'terminal_audit.npz',bits=np.zeros((2,36)),local_energies=np.ones(2))
    p['parent'].update(params_sha256=pytree_sha256(params),registration_self_sha256=reg['sha256'],
        sealed_files={name:file_sha256(directory/name) for name in p['parent']['sealed_files']},
        terminal_products_sha256=file_sha256(directory/'terminal_audit.npz'))
    return p,state


def test_j0_parent_verifier_checks_native_optimizer_rng_and_endpoint(tmp_path):
    p,state=make_sealed_parent(tmp_path)
    saved,report=w.checked_parent(p)
    assert pytree_sha256(saved['params'])==pytree_sha256(state[0])
    assert report['native_parameter_count']==2
    assert report['parent_optimizer_sha256']==pytree_sha256(state[1])
    assert report['parent_sampler_rng_sha256']==pytree_sha256(state[2:])
    for field,value in [('global_step',476),('params_sha256','1'*64),('registration_self_sha256','2'*64)]:
        bad=deepcopy(p);bad['parent'][field]=value
        with pytest.raises(ValueError):w.checked_parent(bad)
    bad=deepcopy(p);bad['arity']='binary_truncated'
    with pytest.raises(ValueError):w.checked_parent(bad)


def test_j0_budget_preserves_fresh_audit_reserve_and_original_parent(monkeypatch,tmp_path):
    p,parent,out,calls=fixture(monkeypatch,tmp_path)
    profile=w.protocol_profile(dict(task=w.J0_TASK))
    p.update({key:value for key,value in profile.items() if key!='parent_global_step'})
    p['parent']['global_step']=456
    monkeypatch.setattr(w,'checked_parent',lambda _:(parent,{}))
    monkeypatch.setattr(w,'remaining_seconds',lambda *a:3601.)
    report=w.run(p,'prepare',out)
    assert report['status']=='cost_gate_original456_preserved'
    assert w.load_training_state(out/'checkpoint.pkl')[1]['absolute_completed_steps']==456
    assert calls['audits']==[]


def test_j0_audit_cannot_silently_reuse_an_interim_count(tmp_path):
    p=j0_plan()
    with pytest.raises(ValueError):w.paired_audit(p,{}, {},({},(),{}),tmp_path/'audit',4096,1,2,lambda:10000.)
