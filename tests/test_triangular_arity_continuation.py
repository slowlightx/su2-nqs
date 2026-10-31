"""Sealed full-state continuation and explicit same-state Hamiltonian transfer."""
from copy import deepcopy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import (
    checked_arity_endpoint,check_arity_pilot,checked_target_H_cost_certificate,
    checked_declared_H_transfer,
)
from su2_nqs.frontier.artifacts import canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import init_arity_pair,arity_clebsch_bank
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank,make_mixed_clebsch_amplitude,
)
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.adam import _manual_adam_init,_manual_adam_update
from su2_nqs.vmc.checkpoint import save_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains,run_exchange_sweeps

jax.config.update('jax_enable_x64',True)


def endpoint_fixture(directory):
    cfg=MixedClebschConfig(multiplicity_cap=9,maximum_q=3)
    parent=init_mixed_clebsch_family(jax.random.PRNGKey(16061),cfg)['ternary_c3']
    family,_=init_arity_pair(mixed_clebsch_bank(parent,cfg,'ternary_c3'),cfg,2)
    p=family['binary_truncated'];opt=_manual_adam_init(p)
    # This oracle checks a NONZERO optimizer/RNG/history, not convergence.
    for _ in range(2):
        p,opt=_manual_adam_update(p,jax.tree.map(lambda x:.001*jnp.cos(x),p),opt,1e-4)
    key=jax.random.PRNGKey(16062)
    states=initialize_spin_exchange_chains(jax.random.PRNGKey(16063),n_chains=2,n_sites=36,n_up=18)
    parent_spec=dict(model=dict(length=6,multiplicity_cap=9,maximum_q=3),arm='binary_truncated',
        arity_comparison=dict(binary_rank=2),j2_over_j1=.125,steps=2,chains=2,
        samples=4,projection_chunk=4,damping=.05,learning_rate=1e-4,
        maximum_FS_sine=.025,spacing_proposals=36)
    reg=dict(config=parent_spec,sources={},cost_registration_sha256='frozen-cost')
    reg['sha256']=canonical_sha256(reg)
    (directory/'registration.json').write_text(json.dumps(reg))
    meta=dict(registration_sha256=reg['sha256'],params_sha256=pytree_sha256(p),completed_steps=2,
        history=[dict(step=1),dict(step=2)],optimizer_step=2,optimizer_sha256=pytree_sha256(opt),
        rng=np.asarray(key),sampler_states=np.asarray(states),sampler_rng_sha256=pytree_sha256((key,states)))
    save_checkpoint(directory/'checkpoint.pkl',p,optimizer_state=opt,metadata=meta)
    audit=dict(registration_sha256=reg['sha256'],params_sha256=pytree_sha256(p),energy_per_site=-.4,
        seed=16064,independent_from_training=True)
    (directory/'terminal_audit.json').write_text(json.dumps(audit))
    req=dict(directory=str(directory),registration_sha256=reg['sha256'],checkpoint_sha256=file_sha256(directory/'checkpoint.pkl'),
        params_sha256=pytree_sha256(p),completed_steps=2,audit_sha256=file_sha256(directory/'terminal_audit.json'))
    spec=dict(parent_spec,arity_endpoint_parent=req,endpoint_protocol='same_H_full_state_continuation',
        additional_steps=40,steps=42)
    return cfg,p,opt,key,states,spec


def test_exact_full_state_continuation_and_rng_replay(tmp_path):
    cfg,p,opt,key,states,spec=endpoint_fixture(tmp_path)
    family,note,endpoint=checked_arity_endpoint(cfg,spec)
    assert not any(note[k] for k in ('optimizer_reset','sampler_reset','rng_reset'))
    assert note['parent_history_preserved'] and not note['fresh_initial_audit_required']
    saved=endpoint['saved'];meta=saved['metadata']
    assert pytree_sha256(family['binary_truncated'])==pytree_sha256(p)
    assert pytree_sha256(saved['optimizer_state'])==pytree_sha256(opt)
    assert meta['completed_steps']==2 and len(meta['history'])==2
    raw=make_mixed_clebsch_amplitude(cfg)
    bank=arity_clebsch_bank(p,cfg,'binary_truncated',2)
    edges=[(a,b) for a,b,_ in triangular_j1j2_bonds(6,6,j2=0.,periodic=True)]
    sample=jax.jit(lambda k,x:run_exchange_sweeps(bank,k,x,amplitude_fn=raw,particle_kind='spin',
        sweeps=2,exchange_edges=edges))
    actual=sample(key,states);replay=sample(jnp.asarray(meta['rng']),jnp.asarray(meta['sampler_states']))
    for a,b in zip(actual,replay,strict=True):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('target_j2', [0., .025])
def test_H_transfer_preserves_both_amplitudes_JVP_and_full_target_H(tmp_path,target_j2):
    cfg,p,_,_,states,spec=endpoint_fixture(tmp_path)
    spec.update(endpoint_protocol='new_H_common_binary_state',j2_over_j1=target_j2,steps=40,
        hamiltonian_transfer=dict(source_j2=.125,target_j2=target_j2))
    family,note,endpoint=checked_arity_endpoint(cfg,spec)
    assert all(note[k] for k in ('optimizer_reset','sampler_reset','rng_reset','fresh_initial_audit_required'))
    assert endpoint['saved']=={} and note['source_H_audit_is_not_target_H_audit']
    raw=make_mixed_clebsch_amplitude(cfg)
    banks={mode:arity_clebsch_bank(z,cfg,mode,2) for mode,z in family.items()}
    evaluate=jax.jit(jax.vmap(raw,in_axes=(None,0)))
    np.testing.assert_allclose(evaluate(banks['binary_truncated'],states),evaluate(banks['ternary_joint'],states),rtol=3e-12,atol=3e-17)
    tangent=jax.tree.map(lambda x:.001*jnp.cos(x),p)
    lift=lambda z:arity_clebsch_bank(z,cfg,'binary_truncated',2)
    bank,dbank=jax.jvp(lift,(p,),(tangent,))
    d1=jax.jvp(lambda z:raw(lift(z),states[0]),(p,),(tangent,))[1]
    d2=jax.jvp(lambda z:raw(z,states[0]),(bank,),(dbank,))[1]
    np.testing.assert_allclose(d1,d2,rtol=3e-12,atol=3e-17)
    bonds=tuple(triangular_j1j2_bonds(6,6,j2=target_j2,periodic=True))
    action=jax.jit(make_amplitude_heisenberg_action(raw,bonds,n_sites=36))
    _,h1=action(banks['binary_truncated'],states);_,h2=action(banks['ternary_joint'],states)
    np.testing.assert_allclose(h1,h2,rtol=3e-12,atol=3e-17)
    columns,elements=SzHeisenbergGraph(36,bonds,n_up=18).get_conn(tuple(np.asarray(states[0])))
    expected=np.dot(elements,np.asarray(evaluate(bank,np.asarray(columns))))
    np.testing.assert_allclose(h1[0],expected,rtol=3e-12,atol=3e-17)


def test_H_transfer_uses_actual_sealed_endpoint_not_original_cost_coupling(tmp_path):
    _, _, _, _, _, spec = endpoint_fixture(tmp_path)
    # A later J0 endpoint is legal even when the original model/cost gate
    # happened to be generated at J=.125. Never relabel a source-H audit.
    path = tmp_path/'registration.json'
    registration = json.loads(path.read_text())
    registration['config']['j2_over_j1'] = 0.
    registration.pop('sha256')
    registration['sha256'] = canonical_sha256(registration)
    path.write_text(json.dumps(registration))
    spec['arity_endpoint_parent']['registration_sha256'] = registration['sha256']
    spec.update(endpoint_protocol='new_H_common_binary_state', j2_over_j1=.025,
                hamiltonian_transfer=dict(source_j2=0., target_j2=.025))
    assert checked_declared_H_transfer(spec)
    for declaration in (dict(source_j2=.125, target_j2=.025),
                        dict(source_j2=0., target_j2=.05)):
        bad = deepcopy(spec)
        bad['hamiltonian_transfer'] = declaration
        with pytest.raises(ValueError, match='actual sealed source and target'):
            checked_declared_H_transfer(bad)
    # A changed source file without a new, explicitly requested seal fails.
    registration['config']['j2_over_j1'] = .125
    path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match='registration changed'):
        checked_declared_H_transfer(spec)


def test_H_transfer_without_declared_new_H_protocol_is_not_authorized(tmp_path):
    _, _, _, _, _, spec = endpoint_fixture(tmp_path)
    assert not checked_declared_H_transfer(spec)
    spec.update(endpoint_protocol='new_H_common_binary_state',
                hamiltonian_transfer=dict(source_j2=.125, target_j2=.125))
    with pytest.raises(ValueError, match='actual sealed source and target'):
        checked_declared_H_transfer(spec)


def test_same_H_common_restart_keeps_parent_bank_but_resets_full_training_state(tmp_path):
    cfg,p,_,_,states,spec=endpoint_fixture(tmp_path)
    spec.update(endpoint_protocol='same_H_common_binary_state',steps=40,arm='ternary_joint')
    family,note,endpoint=checked_arity_endpoint(cfg,spec)
    assert all(note[k] for k in ('optimizer_reset','sampler_reset','rng_reset'))
    assert not note['fresh_initial_audit_required'] and not note['parent_history_preserved']
    assert note['same_physical_initial_state_between_arity_arms'] and not endpoint['saved']
    binary=arity_clebsch_bank(family['binary_truncated'],cfg,'binary_truncated',2)
    joint=arity_clebsch_bank(family['ternary_joint'],cfg,'ternary_joint',2)
    for a,b in zip(jax.tree.leaves(binary),jax.tree.leaves(joint),strict=True):
        np.testing.assert_array_equal(a,b)
    assert note['reference_bank_sha256']==pytree_sha256(binary)
    spec['j2_over_j1']=0.
    with pytest.raises(ValueError,match='same-H common restart'):checked_arity_endpoint(cfg,spec)
    spec['j2_over_j1']=.125;spec['hamiltonian_transfer']={}
    with pytest.raises(ValueError,match='same-H common restart'):checked_arity_endpoint(cfg,spec)


def test_target_H_certificate_has_exact_seal_and_model_and_parent_guards(tmp_path):
    _,_,_,_,_,spec=endpoint_fixture(tmp_path)
    assert not checked_target_H_cost_certificate(spec)
    registration=json.loads((tmp_path/'registration.json').read_text())
    preflight=dict(status='passed',registration_sha256=registration['sha256'],
        target_H_independent_oracle=True,validation=dict(target_H_oracle_relative_error=1e-16))
    path=tmp_path/'preflight.json';path.write_text(json.dumps(preflight))
    spec['target_H_cost_certificate']=dict(directory=str(tmp_path),registration_sha256=registration['sha256'],
        preflight_sha256=file_sha256(path))
    assert checked_target_H_cost_certificate(spec)
    spec['j2_over_j1']=0.
    with pytest.raises(ValueError,match='does not certify'):checked_target_H_cost_certificate(spec)
    spec['j2_over_j1']=.125
    preflight['validation']['target_H_oracle_relative_error']=1e-4
    path.write_text(json.dumps(preflight))
    with pytest.raises(ValueError,match='seal changed'):checked_target_H_cost_certificate(spec)
    spec['target_H_cost_certificate']['preflight_sha256']=file_sha256(path)
    with pytest.raises(ValueError,match='does not certify'):checked_target_H_cost_certificate(spec)


@pytest.mark.parametrize('field',['registration_sha256','checkpoint_sha256','params_sha256','audit_sha256'])
def test_modified_parent_seal_rejected(tmp_path,field):
    cfg,_,_,_,_,spec=endpoint_fixture(tmp_path)
    spec['arity_endpoint_parent'][field]='0'*64
    with pytest.raises(ValueError):checked_arity_endpoint(cfg,spec)


@pytest.mark.parametrize('change',[
    dict(j2_over_j1=0.),dict(arm='ternary_joint'),dict(steps=43),
    dict(arity_comparison=dict(binary_rank=3)),dict(endpoint_protocol='implicit_reset'),
    dict(endpoint_protocol='new_H_common_binary_state',j2_over_j1=0.,steps=40,
         hamiltonian_transfer=dict(source_j2=0.,target_j2=0.)),
])
def test_implicit_H_or_representation_or_budget_changes_rejected(tmp_path,change):
    cfg,_,_,_,_,spec=endpoint_fixture(tmp_path);spec.update(change)
    with pytest.raises(ValueError):checked_arity_endpoint(cfg,spec)


def test_pilot_gate_requires_replay_and_safe_FS_not_a_noisy_energy_decrease(tmp_path):
    result=dict(completed_steps=22,checkpoint_rng_sampler_replay_exact=True,checkpoint_optimizer_exact=True,
        history=[dict(accepted=True,FS_sine=.02,ESS_fraction=.99,energy_before_update_per_site=-.49)]*2)
    (tmp_path/'preflight.json').write_text(json.dumps(dict(status='passed')))
    for field,value in [(None,None),('checkpoint_optimizer_exact',False),('completed_steps',21)]:
        record=deepcopy(result)
        if field is not None:record[field]=value
        (tmp_path/'results.json').write_text(json.dumps(record))
        if field is None:check_arity_pilot(tmp_path,22)
        else:
            with pytest.raises(ValueError):check_arity_pilot(tmp_path,22)
    result['history'][-1]['FS_sine']=.026
    (tmp_path/'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):check_arity_pilot(tmp_path,22)


def schedule_fixture(tmp_path):
    """Synthetic checkpoint bookkeeping only; never production physics data."""
    parent=tmp_path/'parent';parent.mkdir()
    cfg,p,opt,key,states,spec=endpoint_fixture(parent)
    registration=json.loads((parent/'registration.json').read_text())
    audit=json.loads((parent/'terminal_audit.json').read_text())
    result=dict(status='fixed_steps_complete',completed_steps=2,
        registration_sha256=registration['sha256'],terminal_audit=audit,
        checkpoint_optimizer_exact=True,checkpoint_rng_sampler_replay_exact=True)
    (parent/'results.json').write_text(json.dumps(result))
    template=deepcopy(registration['config'])
    plan=dict(parent=dict(directory=str(parent),completed_steps=2,
        registration_sha256=registration['sha256']),template_config=template,
        execution_sources={},stages=[dict(additional_steps=3,samples=8,audit_samples=16,
            audit_chains=4,audit_burn_proposals=72,terminal_audit_seed=17000+i,
            output=str(tmp_path/f'stage{i}'),generated_config=f'config{i}.json') for i in range(2)])
    return plan,cfg,p,opt,key,states


def test_higher_statistics_schedule_keeps_physical_state_optimizer_and_rng(tmp_path):
    from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_schedule import materialize_stage,prepare
    plan,cfg,p,opt,key,states=schedule_fixture(tmp_path)
    spec=materialize_stage(plan,0,root=tmp_path)
    assert spec['steps']==5 and spec['samples']==8
    assert spec['audit_chains']==4 and spec['chains']==2
    family,_,saved=checked_arity_endpoint(cfg,spec)
    assert pytree_sha256(family[spec['arm']])==pytree_sha256(p)
    assert pytree_sha256(saved['saved']['optimizer_state'])==pytree_sha256(opt)
    np.testing.assert_array_equal(saved['saved']['metadata']['rng'],key)
    np.testing.assert_array_equal(saved['saved']['metadata']['sampler_states'],states)
    assert spec['continuation_plan']['sha256']==canonical_sha256(plan)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));digest=file_sha256(path)
    output=prepare(path,digest,0,root=tmp_path)
    assert prepare(path,digest,0,root=tmp_path)==output
    changed=json.loads(output.read_text());changed['samples']=16;output.write_text(json.dumps(changed))
    with pytest.raises(FileExistsError):prepare(path,digest,0,root=tmp_path)
    with pytest.raises(ValueError,match='bytes'):prepare(path,'changed',0,root=tmp_path)


@pytest.mark.parametrize('change',[
    dict(samples=7),dict(audit_samples=15),dict(additional_steps=0),
    dict(terminal_audit_seed=16064),dict(learning_rate=.1),
])
def test_schedule_rejects_unregistered_budget_or_statistics(tmp_path,change):
    from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_schedule import materialize_stage
    plan,*_=schedule_fixture(tmp_path);plan['stages'][0].update(change)
    with pytest.raises(ValueError):materialize_stage(plan,0,root=tmp_path)


def test_schedule_refuses_unfinished_or_changed_parent_and_reused_outputs(tmp_path):
    from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_schedule import materialize_stage,prepare
    plan,*_=schedule_fixture(tmp_path)
    path=Path(plan['parent']['directory'])/'results.json'
    result=json.loads(path.read_text());result['status']='safe_pause';path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='incomplete'):materialize_stage(plan,0,root=tmp_path)
    result['status']='fixed_steps_complete';path.write_text(json.dumps(result))
    bad=deepcopy(plan);bad['stages'][0]['output']=plan['parent']['directory']
    with pytest.raises(ValueError,match='nonoverlapping'):materialize_stage(bad,0,root=tmp_path)
    bad=deepcopy(plan);bad['stages'][1]['terminal_audit_seed']=bad['stages'][0]['terminal_audit_seed']
    with pytest.raises(ValueError,match='distinct'):materialize_stage(bad,0,root=tmp_path)
    bad=deepcopy(plan);bad['execution_sources']={'results.json':'wrong'}
    source=tmp_path/'results.json';source.write_text('{}')
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='source changed'):prepare(plan_path,file_sha256(plan_path),0,root=tmp_path)


def test_schedule_second_stage_requires_exact_completed_first_stage(tmp_path):
    from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_schedule import materialize_stage
    plan,cfg,p,opt,key,states=schedule_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):materialize_stage(plan,1,root=tmp_path)
    spec=materialize_stage(plan,0,root=tmp_path)
    directory=Path(spec['output']);directory.mkdir()
    reg=dict(config=spec,sources={},cost_registration_sha256='fixture only')
    reg['sha256']=canonical_sha256(reg)
    (directory/'registration.json').write_text(json.dumps(reg))
    for _ in range(3):
        p,opt=_manual_adam_update(p,jax.tree.map(lambda x:.001*jnp.cos(x),p),opt,1e-4)
    meta=dict(registration_sha256=reg['sha256'],params_sha256=pytree_sha256(p),completed_steps=5,
        history=[dict(step=i+1) for i in range(5)],optimizer_step=5,optimizer_sha256=pytree_sha256(opt),
        rng=np.asarray(key),sampler_states=np.asarray(states),sampler_rng_sha256=pytree_sha256((key,states)))
    save_checkpoint(directory/'checkpoint.pkl',p,optimizer_state=opt,metadata=meta)
    audit=dict(registration_sha256=reg['sha256'],params_sha256=pytree_sha256(p),
        seed=17000,independent_from_training=True,energy_per_site=-.4)
    (directory/'terminal_audit.json').write_text(json.dumps(audit))
    result=dict(status='fixed_steps_complete',completed_steps=5,terminal_audit=audit,
        registration_sha256=reg['sha256'],checkpoint_optimizer_exact=True,checkpoint_rng_sampler_replay_exact=True)
    (directory/'results.json').write_text(json.dumps(result))
    second=materialize_stage(plan,1,root=tmp_path)
    assert second['steps']==8 and second['arity_endpoint_parent']['completed_steps']==5
    actual=checked_arity_endpoint(cfg,second)[2]['saved']
    assert pytree_sha256(actual['params'])==pytree_sha256(p)
    assert pytree_sha256(actual['optimizer_state'])==pytree_sha256(opt)
    reg['config']['samples']=16;reg.pop('sha256');reg['sha256']=canonical_sha256(reg)
    (directory/'registration.json').write_text(json.dumps(reg))
    with pytest.raises(ValueError,match='preceding frozen stage'):materialize_stage(plan,1,root=tmp_path)
