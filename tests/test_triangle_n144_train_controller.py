"""Independent controller-only fault injection; NOT physical N144 evidence.

The mathematical MH/H/SR transition has separate actual-N6 exact tests. Here
small test doubles isolate run-level registration, atomic CP/results ordering,
fixed endpoints and resume. No GPU allocation or production artifact is used.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import projects.clebsch_tree_nqs.workflows.triangle_n144_train as driver
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.vmc.checkpoint import load_checkpoint

jax.config.update('jax_enable_x64', True)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    params=dict(local=jnp.asarray([.03,-.02]),top=jnp.asarray([.01]))
    pair={arm:params for arm in driver.ARMS}
    source=tmp_path/'source';source.mkdir()
    atomic_json(source/'terminal_audit.json',dict(independent_from_training=True))
    atomic_json(source/'results.json',dict(status='fixed_steps_complete',completed_steps=3))
    monkeypatch.setattr(driver,'checked_training_gate',lambda spec,**kw:
        dict(parameter_counts={a:3 for a in driver.ARMS}))
    monkeypatch.setattr(driver,'checked_scale_parent',lambda *a,**kw:
        dict(params=params,metadata=dict(completed_steps=3,history=[{}, {}, {}])))
    monkeypatch.setattr(driver,'scale_arity_pair',lambda *a,**kw:
        (pair,dict(controller_test_double=True)))
    monkeypatch.setattr(driver,'transitive_local_source_manifest',lambda *a,**kw:{})
    monkeypatch.setattr(driver,'numerical_runtime_identity',lambda :dict(controller_test_double=True))
    monkeypatch.setattr(driver,'arity_clebsch_bank',lambda p,*a,**kw:p)
    monkeypatch.setattr(driver,'prepare_feature_lut_bank',lambda p,*a,**kw:p)
    monkeypatch.setattr(driver,'mixed_parameter_count',lambda p:3)
    monkeypatch.setattr(driver,'triangular_hnf_gamma_a1_group',lambda *a,**kw:
        np.arange(144,dtype=np.int32)[None])
    monkeypatch.setattr(driver,'triangular_j1j2_bonds',lambda *a,**kw:
        ((0,1,1.),(1,2,1.)))

    def amplitude(bank,bits):
        return jnp.exp(bank['local'][0]*(2*bits[0]-1)+
            bank['local'][1]*(2*bits[1]-1)+bank['top'][0]*(2*bits[2]-1))

    def action(bank,rows):
        values=jax.vmap(amplitude,in_axes=(None,0))(bank,rows)
        # Purely a cheap controller fixture, not an SU(2) Hamiltonian.
        return values,values*(2*rows[:,0]-1)

    monkeypatch.setattr(driver,'make_feature_lut_projected_amplitude',lambda *a,**kw:amplitude)
    monkeypatch.setattr(driver,'make_cached_projected_heisenberg_action',lambda *a,**kw:
        jax.jit(action))
    spec=dict(main_agent_training_release=True,j2_over_j1=0.,
        model=dict(length=12,multiplicity_cap=16,maximum_q=10),require_gpu=False,
        parent=dict(checkpoint=str(source/'checkpoint.pkl'),
            model=dict(length=6,multiplicity_cap=16,maximum_q=10),
            arm='binary_truncated',completed_steps=3),binary_rank=8,
        initialization_seed=20310101,projection_chunk=1,action_chunk=2,tolerance=2e-11,
        training=dict(steps=2,chains=2,records_per_chain=2,
            initial_burn_proposals=2,refresh_proposals=1,proposals_between=1),
        audit=dict(chains=2,records_per_chain=2,burn_proposals=2,proposals_between=1),
        sampler=dict(bond_proposal_probability=.75,lazy_probability=.05),
        seeds=dict(training=31,training_initialization=32,initial_audit=33,
            final_audits=dict(binary_truncated=35,ternary_joint=37)),
        sr=dict(damping=.1,step_size=.01,fisher_trust_radius=.01,
            maximum_overlap_distance=.01,minimum_ess_fraction=.8,
            backtracking_steps=8,score_chunk_size=2),
        seconds_cap=60,budget=dict(maximum_A100_hours=1,forecast_seconds=5),
        output=str(tmp_path/'run'))
    path=tmp_path/'config.json';atomic_json(path,spec)
    return path,Path(spec['output'])


def test_run_partial_then_resume_has_exactly_two_updates_and_full_replay(controller):
    path,output=controller
    first=driver.run(path,until_step=1)
    assert first['status']=='stopped_at_atomic_boundary'
    for arm in driver.ARMS:
        saved=load_checkpoint(output/arm/'checkpoint.pkl')
        assert int(saved['optimizer_state']['completed_steps'])==1
        assert len(saved['metadata']['history'])==1
    final=driver.run(path,resume=True)
    assert final['status']=='completed'
    for arm in driver.ARMS:
        assert final['arms'][arm]['completed_steps']==2
        assert len(final['arms'][arm]['history'])==2
        assert final['arms'][arm]['trained_optimizer_sampler_full_transition_replay_bitwise_exact']
    replay=driver.run(path,replay=True)
    assert replay['status']=='passed'
    assert all(a['full_next_sampler_H_SR_bitwise_exact'] for a in replay['arms'].values())


def test_resume_recovers_registration_before_initial_results(controller,monkeypatch):
    path,output=controller
    original=driver.atomic_json

    def interrupt(destination,payload):
        if Path(destination)==output/'results.json':
            raise InterruptedError('inject registration-before-results kill')
        return original(destination,payload)

    monkeypatch.setattr(driver,'atomic_json',interrupt)
    with pytest.raises(InterruptedError):driver.run(path)
    assert (output/'registration.json').exists()
    assert not (output/'results.json').exists()
    monkeypatch.setattr(driver,'atomic_json',original)
    recovered=driver.run(path,resume=True,until_step=0)
    assert recovered['status']=='stopped_at_atomic_boundary'
    assert 'initial_audit' in recovered
    assert all(not (output/arm/'checkpoint.pkl').exists() for arm in driver.ARMS)


def test_resume_uses_atomic_checkpoint_when_result_ledger_lags(controller,monkeypatch):
    path,output=controller
    original_save=driver.save_checkpoint;original_json=driver.atomic_json
    crashed=[False]

    def checkpoint_then_kill(destination,*a,**kw):
        original_save(destination,*a,**kw)
        if Path(destination)==output/'binary_truncated'/'checkpoint.pkl':
            crashed[0]=True
            raise InterruptedError('inject after atomic trained checkpoint')

    def suppress_post_kill_result(destination,payload):
        if crashed[0] and Path(destination)==output/'results.json':
            raise InterruptedError('simulate hard kill: no exception-ledger rewrite')
        return original_json(destination,payload)

    monkeypatch.setattr(driver,'save_checkpoint',checkpoint_then_kill)
    monkeypatch.setattr(driver,'atomic_json',suppress_post_kill_result)
    with pytest.raises(InterruptedError):driver.run(path)
    saved=load_checkpoint(output/'binary_truncated'/'checkpoint.pkl')
    assert int(saved['optimizer_state']['completed_steps'])==1
    monkeypatch.setattr(driver,'save_checkpoint',original_save)
    monkeypatch.setattr(driver,'atomic_json',original_json)
    final=driver.run(path,resume=True)
    assert final['status']=='completed'
    for arm in driver.ARMS:
        saved=load_checkpoint(output/arm/'checkpoint.pkl')
        assert int(saved['optimizer_state']['completed_steps'])==2
        assert [r['step'] for r in saved['metadata']['history']]==[1,2]
