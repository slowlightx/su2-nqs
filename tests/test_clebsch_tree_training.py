import jax
import numpy as np
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import cost
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training import train,_validate_optimizer_condition_change
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_audit import audit,summarize_chains
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.frontier.provenance import file_sha256
from pathlib import Path
import pytest


ROOT=Path(__file__).resolve().parents[1]


def test_real_training_optimizer_and_sampler_resume_exactly(tmp_path):
    spec=dict(geometry='square',rows=2,columns=2,j2_over_j1=0.5,multiplicity_cap=2,
              maximum_q=2,projection_chunk=2,require_gpu=False,seed=12,calibration_seed=13,sampling_seed=14,
              literature_target_energy_total=-2.,literature_target_total_standard_error=.01)
    gate=cost(spec,tmp_path/'cost')
    spec.update(cost_parent=dict(directory=str(tmp_path/'cost'),checkpoint_sha256=gate['checkpoint_sha256'],registration_sha256=gate['registration_sha256']),
                training_seed=15,chains=4,steps=3,sr_warm_steps=1,warm_samples=8,adam_samples=8,
                burn_proposals=8,proposals_between_snapshots=4,score_chunk=1,H_sample_chunk=4,
                sr_damping=.01,maximum_FS_sine=.05,warm_lr=.005,adam_lr=.001,
                reset_moments_at_phase_switch=True,milestones=[0,1,2,3],maximum_training_seconds=120)
    a=train(spec,tmp_path/'a',3,120)
    train(spec,tmp_path/'b',2,120)
    b=train(spec,tmp_path/'b',3,120)
    assert a['completed_steps'] == b['completed_steps'] == 3
    x=load_checkpoint(tmp_path/'a/checkpoint.pkl'); y=load_checkpoint(tmp_path/'b/checkpoint.pkl')
    for k in ('params','optimizer_state'):
        for p,q in zip(jax.tree.leaves(x[k]),jax.tree.leaves(y[k]),strict=True):np.testing.assert_array_equal(p,q)
    for k in ('rng','sampler_states'):np.testing.assert_array_equal(x['metadata'][k],y['metadata'][k])
    # A root-swap subgroup is an execution optimization, not a prerequisite
    # for the full Reynolds sum (e.g. square6 Morton trees have no such fold).
    readout=dict(training_directory=str(tmp_path/'a'),endpoint_step=3,
        require_gpu=False,streams=2,chains_per_stream=4,snapshots_per_stream=2,
        burn_proposals=8,proposals_between_snapshots=4,audit_seeds=[101,102],
        bootstrap_seed=103,H_sample_chunk=4)
    # This small test has a known total-energy reference; it is not a solve.
    registration=__import__('json').loads((tmp_path/'a/registration.json').read_text())
    assert not registration['config'].get('fold_root_swap',False)
    independent=audit(readout,tmp_path/'unfolded_audit')
    assert independent['statistics']['records'] == 16
    np.testing.assert_allclose(independent['statistics']['energy_per_site'],-.75,atol=2e-12)


def test_folded_projection_continues_existing_physical_training_state(tmp_path):
    spec=dict(geometry='square',rows=2,columns=2,j2_over_j1=0.5,multiplicity_cap=2,
              maximum_q=2,projection_chunk=2,require_gpu=False,seed=12,calibration_seed=13,sampling_seed=14,
              literature_target_energy_total=-2.,literature_target_total_standard_error=.01)
    gate=cost(spec,tmp_path/'cost')
    spec.update(cost_parent=dict(directory=str(tmp_path/'cost'),checkpoint_sha256=gate['checkpoint_sha256'],
                registration_sha256=gate['registration_sha256']),training_seed=15,chains=4,steps=3,
                sr_warm_steps=1,warm_samples=8,adam_samples=8,burn_proposals=8,
                proposals_between_snapshots=4,score_chunk=1,H_sample_chunk=4,sr_damping=.01,
                maximum_FS_sine=.05,warm_lr=.005,adam_lr=.001,
                reset_moments_at_phase_switch=True,milestones=[0,1,2,3],maximum_training_seconds=120)
    old_dir=tmp_path/'old'; old=train(spec,old_dir,2,120)
    old_checkpoint=load_checkpoint(old_dir/'checkpoint.pkl')
    train(spec,old_dir,3,120)
    folded=dict(spec,fold_root_swap=True,output=str(tmp_path/'folded'),
                training_resume_from=dict(directory=str(old_dir),
                    checkpoint_sha256=old['checkpoint_sha256'],
                    registration_sha256=old['registration_sha256'],
                    source_workspace=str(ROOT),same_projected_physical_state_proved=True))
    # Freeze the step-two parent before the ordinary branch advances to step three.
    assert file_sha256(old_dir/'quality/step00002.pkl') == old['checkpoint_sha256']
    folded['training_resume_from']['directory']=str(old_dir/'quality_parent')
    (old_dir/'quality_parent').mkdir()
    (old_dir/'quality_parent'/'checkpoint.pkl').write_bytes((old_dir/'quality/step00002.pkl').read_bytes())
    (old_dir/'quality_parent'/'registration.json').write_bytes((old_dir/'registration.json').read_bytes())
    result=train(folded,tmp_path/'folded',3,120)
    assert result['completed_steps']==3
    new_checkpoint=load_checkpoint(tmp_path/'folded/checkpoint.pkl')
    for name in ('params','optimizer_state'):
        for left,right in zip(jax.tree.leaves(old_checkpoint[name]),
                              jax.tree.leaves(load_checkpoint(old_dir/'quality/step00002.pkl')[name]),strict=True):
            np.testing.assert_array_equal(left,right)
    for name in ('rng','sampler_states'):
        np.testing.assert_array_equal(new_checkpoint['metadata'][name],
                                      load_checkpoint(old_dir/'checkpoint.pkl')['metadata'][name])
    for left,right in zip(jax.tree.leaves(new_checkpoint['params']),
                          jax.tree.leaves(load_checkpoint(old_dir/'checkpoint.pkl')['params']),strict=True):
        np.testing.assert_allclose(left,right,rtol=2e-10,atol=2e-10)

    broken=dict(folded,training_resume_from=dict(folded['training_resume_from'],
                                                  checkpoint_sha256='0'*64))
    with pytest.raises(ValueError,match='parent changed'):
        train(broken,tmp_path/'invalid',3,120)

    wider=dict(folded,output=str(tmp_path/'wider'),maximum_FS_sine=.2)
    with pytest.raises(ValueError,match='changed physical model'):
        train(wider,tmp_path/'unregistered_condition',3,120)
    wider['training_resume_from']=dict(folded['training_resume_from'],
        optimizer_condition_change={'maximum_FS_sine':dict(from_value=.05,to_value=.2)})
    wider_result=train(wider,tmp_path/'wider',3,120)
    assert wider_result['completed_steps']==3
    np.testing.assert_allclose(wider_result['history'][-1]['energy_before_update_per_site'],
                               result['history'][-1]['energy_before_update_per_site'],rtol=0,atol=1e-12)

    # Only execution batches change: exact state, H, optimizer and RNG agree.
    batched=dict(folded,execution=dict(projection_chunk=4,score_chunk=2,H_sample_chunk=4))
    batched_result=train(batched,tmp_path/'batched',3,120)
    checkpoint=load_checkpoint(tmp_path/'batched/checkpoint.pkl')
    reference=load_checkpoint(tmp_path/'folded/checkpoint.pkl')
    for name in ('params','optimizer_state'):
        for left,right in zip(jax.tree.leaves(checkpoint[name]),jax.tree.leaves(reference[name]),strict=True):
            np.testing.assert_allclose(left,right,rtol=3e-10,atol=3e-10)
    for name in ('rng','sampler_states'):
        np.testing.assert_array_equal(checkpoint['metadata'][name],reference['metadata'][name])
    execution=__import__('json').loads((tmp_path/'batched/execution_replay.json').read_text())
    assert max(execution['maximum_relative_errors'].values()) < 2e-12
    assert batched_result['completed_steps']==3
    for bad_execution in ({'learning_rate':1},{'projection_chunk':True},{'score_chunk':0}):
        with pytest.raises(ValueError,match='execution settings'):
            train(dict(folded,execution=bad_execution),tmp_path/'invalid_execution',3,120)

    dag=dict(folded,execution=dict(kernel='orbit_dag',projection_chunk=4,
                                  score_chunk=2,H_sample_chunk=4))
    train(dag,tmp_path/'dag',3,120)

    readout=dict(training_directory=str(tmp_path/'folded'),output=str(tmp_path/'audit'),endpoint_step=3,
                 require_gpu=False,streams=2,chains_per_stream=4,snapshots_per_stream=2,
                 burn_proposals=8,proposals_between_snapshots=4,audit_seeds=[101,102],
                 bootstrap_seed=103,H_sample_chunk=4)
    result=audit(readout,tmp_path/'audit')
    assert result['statistics']['records']==16
    assert np.isfinite(result['statistics']['energy_per_site'])
    with pytest.raises(FileExistsError):audit(readout,tmp_path/'audit')
    dag_readout=dict(readout,training_directory=str(tmp_path/'dag'))
    dag_audit=audit(dag_readout,tmp_path/'dag_audit')
    assert dag_audit['execution_kernel'] == 'orbit_dag'
    assert max(dag_audit['independent_execution_replay'][name] for name in (
        'amplitude_relative_error','H_product_relative_error')) < 2e-12
    np.testing.assert_allclose(dag_audit['statistics']['energy_per_site'],
                               result['statistics']['energy_per_site'],atol=3e-11,rtol=0)

    # A frozen milestone remains auditable after the live endpoint advanced.
    frozen_readout=dict(readout,endpoint_step=2,quality_checkpoint=True)
    frozen=audit(frozen_readout,tmp_path/'quality_audit')
    assert frozen['statistics']['records']==16
    bad_quality=dict(readout,quality_checkpoint='yes')
    with pytest.raises(ValueError,match='must be boolean'):
        audit(bad_quality,tmp_path/'bad_quality')


def test_chain_audit_uses_chain_resampling_and_rejects_nonfinite():
    values=np.array([[[1.,3.],[2.,4.]],[[5.,7.],[6.,8.]]])
    result=summarize_chains(values,4,bootstrap_seed=9)
    np.testing.assert_allclose(result['energy_per_site'],1.125)
    np.testing.assert_allclose(result['variance_per_site'],np.var(values)/4)
    assert result['independent_chains']==4
    with pytest.raises(ValueError):summarize_chains(values*np.nan,4,bootstrap_seed=9)


def test_registered_optimizer_condition_allowlist_is_not_a_model_override():
    old=dict(maximum_FS_sine=.2,warm_samples=256,adam_samples=4096,sr_warm_steps=100,
             geometry='square',maximum_q=8)
    new=dict(old,maximum_FS_sine=.05,warm_samples=1024,adam_samples=1024,sr_warm_steps=274)
    changes={k:dict(from_value=old[k],to_value=new[k]) for k in old if old[k] != new[k]}
    a,b=old.copy(),new.copy()
    _validate_optimizer_condition_change(a,b,dict(optimizer_condition_change=changes,
        reset_optimizer_moments=True,optimizer_reset_reason='same-parent direction-recipe comparison'))
    assert a == b
    with pytest.raises(ValueError,match='unsupported'):
        _validate_optimizer_condition_change(old.copy(),dict(new,maximum_q=10),
            dict(optimizer_condition_change=dict(maximum_q=dict(from_value=8,to_value=10))))
    with pytest.raises(ValueError,match='values disagree'):
        _validate_optimizer_condition_change(old.copy(),new.copy(),
            dict(optimizer_condition_change=dict(maximum_FS_sine=dict(from_value=.2,to_value=.1))))
    with pytest.raises(ValueError,match='needs a registered reason'):
        _validate_optimizer_condition_change(old.copy(),old.copy(),dict(reset_optimizer_moments=True))
    with pytest.raises(ValueError,match='must be boolean'):
        _validate_optimizer_condition_change(old.copy(),old.copy(),dict(reset_optimizer_moments=1))


def test_imported_optimizer_recipe_resets_once_and_split_resume_matches(tmp_path):
    spec=dict(geometry='square',rows=2,columns=2,j2_over_j1=.5,multiplicity_cap=2,
              maximum_q=2,projection_chunk=2,require_gpu=False,seed=12,
              calibration_seed=13,sampling_seed=14)
    gate=cost(spec,tmp_path/'cost')
    spec.update(cost_parent=dict(directory=str(tmp_path/'cost'),checkpoint_sha256=gate['checkpoint_sha256'],
                registration_sha256=gate['registration_sha256']),training_seed=15,chains=4,steps=4,
                sr_warm_steps=1,warm_samples=8,adam_samples=8,burn_proposals=8,
                proposals_between_snapshots=4,score_chunk=1,H_sample_chunk=4,sr_damping=.01,
                maximum_FS_sine=.05,warm_lr=.005,adam_lr=.001,fold_root_swap=True,
                reset_moments_at_phase_switch=True,milestones=[0,1,2,3,4],maximum_training_seconds=120)
    parent=train(spec,tmp_path/'parent',2,120)
    continuation=dict(directory=str(tmp_path/'parent'),checkpoint_sha256=parent['checkpoint_sha256'],
        registration_sha256=parent['registration_sha256'],source_workspace=str(ROOT),
        same_projected_physical_state_proved=True,reset_optimizer_moments=True,
        optimizer_reset_reason='bounded same-parent optimizer-stage comparison',
        optimizer_condition_change=dict(sr_warm_steps=dict(from_value=1,to_value=4)))
    child=dict(spec,sr_warm_steps=4,training_resume_from=continuation)
    train(child,tmp_path/'uninterrupted',4,120)
    train(child,tmp_path/'split',3,120)
    at_mid=load_checkpoint(tmp_path/'split/checkpoint.pkl')
    assert int(at_mid['optimizer_state'][0]) == 1
    train(child,tmp_path/'split',4,120)
    a=load_checkpoint(tmp_path/'uninterrupted/checkpoint.pkl')
    b=load_checkpoint(tmp_path/'split/checkpoint.pkl')
    assert int(b['optimizer_state'][0]) == 2
    for name in ('params','optimizer_state'):
        for x,y in zip(jax.tree.leaves(a[name]),jax.tree.leaves(b[name]),strict=True):
            np.testing.assert_array_equal(x,y)
    for name in ('rng','sampler_states'):
        np.testing.assert_array_equal(a['metadata'][name],b['metadata'][name])
