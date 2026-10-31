"""A failed numerical replay cannot silently become a successful old gate."""
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as campaign
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.replay_validation import compare_replay

jax.config.update('jax_enable_x64', True)
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/remote/ct6_width_20261020'


def test_stable_width_plans_keep_fixed_samples_geometry_tolerance_and_branch_lineage():
    plans={p.name:json.loads(p.read_text()) for p in CONFIG.glob('*.json')
           if 'model' in json.loads(p.read_text())}
    assert len(plans)==13
    seeds=[]
    for name,p in plans.items():
        campaign.validate_plan(p)
        assert p['model']['maximum_q']==10 and p['binary_rank']==8
        assert p['training']['samples']==2048 and p['audit']['samples']==16384
        assert p['sr_solver']=='dual_force' and p['tolerance']==2e-11
        seeds+=list(p['seeds'].values())
        if '_w20' in name:
            assert p['parent']['model']['multiplicity_cap']==18
        if '_w24' in name:
            assert p['parent']['model']['multiplicity_cap']==20
        if '_w18_control' in name:
            wider=json.loads(Path(p['matched_wider_arm']).read_text())
            assert p['parent']==wider['parent']
            assert p['training']==wider['training']
            assert p['calibration']==wider['calibration']
            assert p['maximum_fs_sine']==wider['maximum_fs_sine']
            assert p['parent']['model']==p['model']
    assert len(seeds)==len(set(seeds))
    repair=plans['j0125_binary_truncated_w18_repair.json']
    assert repair['training']['steps']==14
    for key,value in [('sr_solver','woodbury_gradient'),('tolerance',3e-11)]:
        bad=copy.deepcopy(repair);bad[key]=value
        with pytest.raises(ValueError):campaign.validate_plan(bad)


def _sealed_failed_parent(tmp_path):
    plan=json.loads((CONFIG/'j0125_binary_truncated_w18_repair.json').read_text())
    req=plan['parent'];req['directory']=str(tmp_path)
    source=json.loads((ROOT/req['plan_path']).read_text())
    source_path=tmp_path/'old_plan.json';source_path.write_text(json.dumps(source))
    req['plan_path']=str(source_path)
    identity=dict(plan_sha256=canonical_sha256(source),**req['expected_identity'])
    seal=canonical_sha256(identity)
    (tmp_path/'registration.json').write_text(json.dumps(dict(identity,sha256=seal)))
    p=dict(x=jnp.array([.1,.2],jnp.float64));_,a,b=_manual_adam_init(p)
    state=(p,(jnp.asarray(2),a,b),jax.random.PRNGKey(1),jnp.zeros((64,36),jnp.int32),
           jax.random.PRNGKey(2),jnp.ones((64,36),jnp.int32))
    meta=dict(registration_sha256=seal,completed_steps=2,history=[{},{}],
              absolute_completed_steps=254,optimizer_base_step=0)
    campaign.shared.save_state(tmp_path/'checkpoint.pkl',state,meta)
    checksum=file_sha256(tmp_path/'checkpoint.pkl')
    expected=(jnp.array([1.,2.]),jnp.array([1,2],jnp.int32))
    actual=(expected[0]+1e-9,expected[1])
    report=compare_replay(expected,actual,tolerance=2e-11)
    assert not report['passed']
    (tmp_path/'pilot.json').write_text(json.dumps(dict(status='passed',
        registration_sha256=seal,checkpoint_sha256=checksum)))
    (tmp_path/'process_replay.json').write_text(json.dumps(dict(status='failed',
        registration_sha256=seal,checkpoint_sha256=checksum,
        optimizer_updates_committed=0,report=report)))
    req['sealed_files']={name:file_sha256(tmp_path/name) for name in
                        ('registration.json','checkpoint.pkl','pilot.json','process_replay.json')}
    return plan,state


def test_rebase_preserves_saved_state_but_explicitly_retains_old_failed_gate(tmp_path):
    plan,state=_sealed_failed_parent(tmp_path)
    restored,lineage=campaign.load_parent(plan)
    for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored),strict=True):
        np.testing.assert_array_equal(a,b)
    assert lineage['old_replay_gate_passed'] is False
    assert lineage['absolute_completed_steps']==254
    assert 'inherit_all_Adam_moments' in lineage['optimizer_policy']
    with (tmp_path/'pilot.json').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='artifact changed'):campaign.load_parent(plan)


def test_rebase_cannot_accept_a_discrete_trajectory_disagreement(tmp_path):
    plan,_=_sealed_failed_parent(tmp_path)
    path=tmp_path/'process_replay.json';replay=json.loads(path.read_text())
    next(row for row in replay['report']['leaves'] if row['discrete'])['bitwise_exact']=False
    path.write_text(json.dumps(replay));plan['parent']['sealed_files'][path.name]=file_sha256(path)
    with pytest.raises(ValueError,match='discrete trajectory'):campaign.load_parent(plan)


def test_same_width_capacity_control_explicitly_resets_Adam_not_physical_parent(monkeypatch,tmp_path):
    from su2_nqs.frontier.artifacts import pytree_sha256
    from su2_nqs.vmc.checkpoint import load_checkpoint
    plan=json.loads((CONFIG/'j0_binary_truncated_w18_control.json').read_text())
    plan['output']=str(tmp_path);plan['require_gpu']=False
    p=dict(x=jnp.array([.1,.2],jnp.float64));_,a,b=_manual_adam_init(p)
    parent=(p,(jnp.asarray(16),jax.tree.map(lambda x:x+.7,a),b),
            jax.random.PRNGKey(1),jnp.zeros((64,36),jnp.int32),
            jax.random.PRNGKey(2),jnp.ones((64,36),jnp.int32))
    lineage=dict(absolute_completed_steps=248,params_sha256=pytree_sha256(p),
                 checkpoint_sha256='fixed',model=plan['parent']['model'])
    monkeypatch.setattr(campaign,'load_parent',lambda _:(parent,lineage))
    monkeypatch.setattr(campaign,'transitive_local_source_manifest',lambda *a,**k:{'test':'frozen'})
    seen=[]
    def transition(state):
        seen.append(state)
        nxt=(jax.tree.map(lambda x:x+.001,state[0]),(state[1][0]+1,state[1][1],state[1][2]),
             state[2]+1,state[3],state[4]+1,state[5])
        return nxt,dict(accepted=True),dict(sampling_seconds=.01)
    monkeypatch.setattr(campaign,'runtime',lambda *a:dict(transition=transition))
    monkeypatch.setattr(campaign.shared,'preflight',lambda *a:dict(status='passed'))
    assert campaign.run(plan,'pilot',canonical_sha256(plan))['status']=='passed'
    for a,b in zip(jax.tree.leaves(seen[0][0]),jax.tree.leaves(parent[0]),strict=True):
        np.testing.assert_array_equal(a,b)
    for a,b in zip(jax.tree.leaves(seen[0][2:]),jax.tree.leaves(parent[2:]),strict=True):
        np.testing.assert_array_equal(a,b)
    assert int(seen[0][1][0])==0
    assert all(np.count_nonzero(x)==0 for x in jax.tree.leaves(seen[0][1][1:]))
    cp=load_checkpoint(tmp_path/'checkpoint.pkl')
    assert cp['metadata']['absolute_completed_steps']==250
    assert cp['metadata']['optimizer_base_step']==0
    assert cp['metadata']['optimizer_policy']=='explicit_reset_matching_wider_capacity_arm'
    assert json.loads((tmp_path/'capacity_control_initialization.json').read_text())['same_parent_parameters_bitwise']
