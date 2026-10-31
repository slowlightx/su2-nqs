"""Filesystem/scheduler safeguards; no training, GPU or phase claims."""
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import save_checkpoint
from projects.clebsch_tree_nqs.workflows import spin_clebsch_tree_migration as migration

jax.config.update('jax_enable_x64',True)


@pytest.fixture
def prepared_inputs(tmp_path):
    old,new,run,proof=(tmp_path/name for name in ('old_source','new_source','old_run','proof'))
    for path in (old,new,run,proof): path.mkdir()
    training='projects/clebsch_tree_nqs/workflows/spin_clebsch_tree_training.py'
    prep='projects/clebsch_tree_nqs/workflows/spin_clebsch_tree_migration.py'
    kernel='su2_nqs/models/clebsch_tree_orbit_dag.py'
    for base in (old,new):
        (base/'math.py').write_text('value=1\n')
        (base/training).parent.mkdir(parents=True,exist_ok=True)
        (base/training).write_text('value="legacy"\n' if base==old else 'value="new factory"\n')
    (new/prep).write_text('value="preparer"\n')
    (new/kernel).parent.mkdir(parents=True)
    kernel_code='def _leaves(x):\n    return x\ndef make_orbit_dag_clebsch(x):\n    return x\n'
    (new/kernel).write_text(kernel_code)
    prototype=tmp_path/'validated.py'
    prototype.write_text(kernel_code)
    cfg=dict(rows=2,columns=2,chains=2,maximum_training_seconds=172800,
             output=str(run),execution=dict(score_chunk=4),sr_warm_steps=100)
    reg=dict(config=cfg,source_manifest={name:file_sha256(old/name) for name in ('math.py',training)})
    reg['sha256']=canonical_sha256(reg)
    atomic_json(run/'registration.json',reg)
    params=dict(weights=jnp.asarray([.2,-.1],dtype=jnp.float64))
    optimizer=(jnp.asarray(5),jax.tree.map(jnp.zeros_like,params),jax.tree.map(jnp.zeros_like,params))
    save_checkpoint(run/'checkpoint.pkl',params,optimizer_state=optimizer,metadata=dict(
        completed_steps=5,history=[dict(step=i+1) for i in range(5)],optimizer_step=5,
        registration_sha256=reg['sha256'],params_sha256=pytree_sha256(params),active_seconds=200.,
        sampler_states=np.asarray([[0,1,0,1],[1,0,1,0]]),rng=np.asarray([123,456],dtype=np.uint32)))
    atomic_json(run/'training.json',dict(status='safe_pause',registration_sha256=reg['sha256'],
        completed_steps=5,checkpoint_sha256=file_sha256(run/'checkpoint.pkl'),params_sha256=pytree_sha256(params)))
    proof_reg=dict(parent_registration_sha256=reg['sha256'],sources={'orbit_dag.py':file_sha256(prototype)})
    proof_reg['sha256']=canonical_sha256(proof_reg)
    atomic_json(proof/'registration.json',proof_reg)
    results=dict(status='passed',exact_replay=dict(samples=True,states=True,key=True),parent_unchanged=True,
        arms={name:dict(cost=dict(samples=4096,full_H_calls=4096,synchronized_complete_step_seconds=seconds))
              for name,seconds in (('baseline',970.),('dag',400.))})
    atomic_json(proof/'results.json',results)
    spec=dict(old_workspace=str(old),new_workspace=str(new),old_training_directory=str(run),
        new_output=str(tmp_path/'new_run'),migration_directory=str(tmp_path/'archive'),
        optimizer_replay_directory=str(proof),optimizer_replay_results_sha256=file_sha256(proof/'results.json'),
        validated_kernel_source=str(prototype),until_step=250,session_seconds=43200,wall_limit='13:00:00',
        allow_submit=False,staged_source_manifest={name:file_sha256(new/name) for name in (kernel,training,prep)})
    return spec


def test_preparer_freezes_latest_full_state_without_submitting_or_changing_protocol(prepared_inputs,monkeypatch):
    spec=prepared_inputs
    monkeypatch.setattr(migration.subprocess,'run',lambda *a,**kw:pytest.fail('dry-run must not call scheduler'))
    before=file_sha256(Path(spec['old_training_directory'])/'checkpoint.pkl')
    result=migration.prepare(spec)
    assert result['status']=='prepared' and result['parent_step']==5
    assert result['submitted_training_job'] is None and result['session_seconds']==43200
    config=migration.json.loads(Path(result['config_path']).read_text())
    assert config['execution']==dict(score_chunk=4,kernel='orbit_dag')
    assert config['sr_warm_steps']==100 and config['maximum_training_seconds']==172800
    assert config['training_resume_from']['checkpoint_sha256']==before
    assert file_sha256(Path(result['snapshot'])/'checkpoint.pkl')==before
    assert file_sha256(Path(spec['old_training_directory'])/'checkpoint.pkl')==before
    assert not Path(spec['new_output']).exists()
    with pytest.raises(FileExistsError,match='immutable'): migration.prepare(spec)


def test_preparer_refuses_to_race_active_old_writer(prepared_inputs,monkeypatch):
    spec=dict(prepared_inputs,writer_jobs_to_verify=[292304],scontrol_binary='scontrol')
    monkeypatch.setattr(migration.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='JobId=292304 JobState=RUNNING'))
    with pytest.raises(ValueError,match='not confirmed terminal'): migration.prepare(spec)
    assert not Path(spec['migration_directory']).exists()


def test_preparer_does_not_automatically_resume_a_failed_primary_writer(prepared_inputs,monkeypatch):
    spec=dict(prepared_inputs,old_writer_job=292304,writer_jobs_to_verify=[292304],scontrol_binary='scontrol')
    monkeypatch.setattr(migration.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='JobId=292304 JobState=FAILED'))
    with pytest.raises(ValueError,match='abnormally'): migration.prepare(spec)
    assert not Path(spec['migration_directory']).exists()


def test_preparer_refuses_source_and_kernel_changes(prepared_inputs):
    spec=prepared_inputs
    new=Path(spec['new_workspace'])
    (new/'math.py').write_text('value=2\n')
    with pytest.raises(ValueError,match='old math'): migration.prepare(spec)
    (new/'math.py').write_text('value=1\n')
    kernel=new/'su2_nqs/models/clebsch_tree_orbit_dag.py'
    kernel.write_text(kernel.read_text().replace('return x','return 2*x'))
    with pytest.raises(ValueError,match='differs from'): migration.prepare(spec)


def test_preparer_rejects_unapproved_submission_and_handles_already_reached_target(prepared_inputs):
    spec=prepared_inputs
    with pytest.raises(ValueError,match='not explicitly authorized'): migration.prepare(spec,submit=True)
    result=migration.prepare(dict(spec,until_step=5))
    assert result['status']=='target_already_reached' and result['submitted_training_job'] is None
    assert not Path(spec['new_output']).exists()
