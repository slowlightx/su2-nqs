"""Adversarial holdout, ancestry identity and cost-ledger validation."""
import json
from pathlib import Path
import jax
import pytest
from fnqs.artifacts import write_json
from fnqs.pilots.engine import init_model,model_signature
from fnqs.pilots.lineage import (seal_protocol,save_node,read_node,verify_holdout,inclusive_cost,
                               cost,check_cost,validate_splits,sha)
from fnqs.training import init_optimizer
from fnqs.variants.size_features import SizeFeatureConfig


def node(root, name, points, parents=(), cfg=None):
    cfg=cfg or SizeFeatureConfig(2)
    p=init_model(jax.random.PRNGKey(len(name)),cfg)
    state=init_optimizer(p)
    schedule=[[g] for g in points]; state['work']=cost(schedule,32); state['step']=len(schedule)
    history=[dict(step=i+1,tasks=row) for i,row in enumerate(schedule)]
    return save_node(root/name,p,state,jax.random.PRNGKey(1),signature=model_signature(cfg,4),
        protocol_path=root/'protocol.json',train_points=points,history=history,parents=list(parents),stage='test')


def setup(tmp_path):
    output=tmp_path/'sealed';seal_protocol(output,dict(kind='unit'))
    return output


def test_transitive_physical_holdout_and_deduplicated_cost(tmp_path):
    root=setup(tmp_path)
    a=node(root,'a',[[1.,0.,0.,1.]])
    b=node(root,'b',[[1.,.1,0.,1.]],[a])
    c=node(root,'c',[[1.,.2,0.,1.]],[a])
    verify_holdout([b,c],'C')
    with pytest.raises(ValueError,match='held physical family'):verify_holdout([b],'A')
    work=inclusive_cost([b,c])
    assert work['unique_checkpoints']==3
    assert work['work']['born_samples']==96
    assert inclusive_cost([b])['work']['born_samples']==64
    assert inclusive_cost([b,b])['work']['born_samples']==64


def test_feature_policy_parent_mismatch(tmp_path):
    root=setup(tmp_path); a=node(root,'a',[[1.,0.,0.,1.]])
    b=node(root,'b',[[1.,.1,0.,1.]],[a],SizeFeatureConfig(2,size_features='legacy'))
    with pytest.raises(ValueError,match='signature mismatch'):read_node(b)


def test_corrupted_ancestor_checkpoint_rejected(tmp_path):
    root=setup(tmp_path); a=node(root,'a',[[1.,0.,0.,1.]])
    b=node(root,'b',[[1.,.1,0.,1.]],[a])
    cp=a.parent/'checkpoint.npz'; cp.write_bytes(cp.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):verify_holdout([b],'C')


def test_missing_ancestry_not_assumed_cold(tmp_path):
    root=setup(tmp_path); a=node(root,'a',[[1.,0.,0.,1.]])
    d=json.loads(a.read_text());d['complete']=False;write_json(a,d)
    with pytest.raises(ValueError,match='incomplete'):read_node(a)


def test_cycle_guard(tmp_path):
    root=setup(tmp_path); a=node(root,'a',[[1.,0.,0.,1.]])
    with pytest.raises(ValueError,match='cycle'):read_node(a,active={a.resolve()})


def test_undeclared_task_history_rejected(tmp_path):
    root=setup(tmp_path);a=node(root,'a',[[1.,0.,0.,1.]])
    # Even changing both checkpoint and manifest cannot turn a B task into A.
    from fnqs.artifacts import load_checkpoint,save_checkpoint
    cp=a.parent/'checkpoint.npz';p,s,k,m=load_checkpoint(cp)
    m['train_points']=[[1.,.1,0.,1.]]
    save_checkpoint(cp,p,s,k,m)
    d=json.loads(a.read_text());d['train_points']=m['train_points'];d['checkpoint_sha256']=sha(cp);write_json(a,d)
    with pytest.raises(ValueError,match='actual task history'):read_node(a)


def test_physical_split_alias_and_exact_cost():
    with pytest.raises(ValueError,match='field leakage'):
        validate_splits([[1,0,0,1]],[[1.,0.,0.,1.]],[])
    expected=cost([[[1,0,0,1]]]*3,64)
    assert expected['born_samples']==192
    with pytest.raises(ValueError,match='work mismatch'):check_cost({**expected,'born_samples':193},expected)


def test_phase2_declaration_has_whole_family_and_query_matched_arms(tmp_path):
    from fnqs.scripts.run_transfer_pilot import prepare
    output=tmp_path/'transfer';prepare(output)
    p=json.loads((output/'protocol.json').read_text())
    assert {f['held_family'] for f in p['folds']}==set('ABC')
    from fnqs.staging import candidate_family
    from fnqs.training import Task
    from fnqs.hamiltonians import IsingCouplings
    for fold in p['folds']:
        assert all(candidate_family(Task(6,IsingCouplings(*g)))!=fold['held_family'] for g in fold['pretrain_points'])
        for arm in fold['arms']:
            assert len(set(map(tuple,arm['train_points'])))==arm['shots']
            assert arm['cost']['born_samples']==256
            validate_splits(arm['train_points'],fold['held_audit_points'],[])
        assert sum(x['cost']['born_samples'] for x in fold['pointwise'])==256
