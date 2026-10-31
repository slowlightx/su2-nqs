import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import cost
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training import train
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_radius_compare import compare,compare_condition_arm
from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_radius_compare import held_direction_statistics
from su2_nqs.frontier.artifacts import canonical_sha256,atomic_json
from su2_nqs.frontier.provenance import file_sha256


ROOT=Path(__file__).resolve().parents[1]


def test_held_real_state_direction_energy_derivative_matches_independent_finite_difference():
    import numpy as np
    # Equal native Born weights at theta=0 for (1,1)/sqrt(2), repeated in
    # independent chain labels. H=diag(1,-1); dlogpsi=(1,-1) gives dE=+2.
    e=np.tile([1.,-1.],8);d=np.tile([1.,-1.],8);labels=np.repeat(np.arange(4),4)
    result=held_direction_statistics(e,d,labels,n_sites=1)
    theta=1e-5
    def energy(t):
        p=np.exp(2*t*np.array([1.,-1.]));return p @ np.array([1.,-1.])/p.sum()
    finite=(energy(theta)-energy(-theta))/(2*theta)
    np.testing.assert_allclose(result['energy_slope_per_site'],finite,atol=4e-10,rtol=0)
    np.testing.assert_allclose(result['tangent_FS_sine_linear'],1.,atol=1e-15)
    assert result['conditional_chain_jackknife_SE']==0 and not result['variance_derivative_claim']
    with pytest.raises(ValueError):held_direction_statistics(e,d,np.zeros_like(labels),n_sites=1)


def test_condition_evidence_gate_rejects_changed_or_non_descending_readout(tmp_path):
    path=tmp_path/'probe.json'
    probe=dict(status='complete',no_optimized_state_created=True,folds=[dict(directions=dict(
        direct_SR=dict(held=dict(energy_slope_per_site=1.),finite_step_overlap=dict(
            held=dict(sampled_fs_sine=.02,reweighting_ess_fraction=.99)))))])
    atomic_json(path,probe)
    plan=dict(pre_release_direction_evidence=dict(path=str(path),sha256='0'*64,
        negative_held_directions=['direct_SR'],maximum_held_FS_sine=.025))
    with pytest.raises(ValueError,match='evidence changed'):
        compare_condition_arm(plan,'not_released',plan_sha256=canonical_sha256(plan))
    plan['pre_release_direction_evidence']['sha256']=file_sha256(path)
    with pytest.raises(ValueError,match='does not support'):
        compare_condition_arm(plan,'not_released',plan_sha256=canonical_sha256(plan))


def test_same_parent_one_step_radius_comparison_has_fresh_audits(tmp_path):
    spec=dict(geometry='square',rows=2,columns=2,j2_over_j1=.5,multiplicity_cap=2,
              maximum_q=2,projection_chunk=2,require_gpu=False,seed=12,
              calibration_seed=13,sampling_seed=14,literature_target_energy_total=-2.,
              literature_target_total_standard_error=.01)
    gate=cost(spec,tmp_path/'cost')
    spec.update(cost_parent=dict(directory=str(tmp_path/'cost'),checkpoint_sha256=gate['checkpoint_sha256'],
                registration_sha256=gate['registration_sha256']),training_seed=15,chains=4,steps=2,
                sr_warm_steps=2,warm_samples=8,adam_samples=8,burn_proposals=8,
                proposals_between_snapshots=4,score_chunk=1,H_sample_chunk=4,sr_damping=.01,
                maximum_FS_sine=.05,warm_lr=.005,adam_lr=.001,fold_root_swap=True,
                reset_moments_at_phase_switch=True,milestones=[0,1,2],maximum_training_seconds=120)
    parent_dir=tmp_path/'parent'; parent=train(spec,parent_dir,1,120)
    registration=json.loads((parent_dir/'registration.json').read_text())
    plan=dict(parent=dict(directory=str(parent_dir),source_workspace=str(ROOT),
                          registration_sha256=registration['sha256'],
                          checkpoint_sha256=parent['checkpoint_sha256'],step=1),
              maximum_training_session_seconds=120,require_gpu=False,
              arms={name:dict(maximum_FS_sine=radius,
                training_output=str(tmp_path/name),audit_output=str(tmp_path/(name+'_audit')),
                generated_training_config=str(tmp_path/(name+'_train.json')),
                generated_audit_config=str(tmp_path/(name+'_audit.json')))
                    for name,radius in (('radius005',.05),('radius020',.2))},
              audit=dict(streams=2,chains_per_stream=4,snapshots_per_stream=2,
                         burn_proposals=8,proposals_between_snapshots=4,
                         audit_seeds=[101,102],bootstrap_seed=103,H_sample_chunk=4),
              comparison_output=str(tmp_path/'comparison.json'))
    with pytest.raises(ValueError,match='plan changed'):
        compare(plan,plan_sha256='0'*64)
    result=compare(plan,plan_sha256=canonical_sha256(plan))
    assert result['same_parent_sample_and_full_H_calls']
    assert all(result['arms'][name]['audit']['records']==16 for name in plan['arms'])
    assert all(result['arms'][name]['training']['completed_steps']==2 for name in plan['arms'])
    assert result['arms']['radius005']['training']['history'][-1]['energy_before_update_per_site'] == result['arms']['radius020']['training']['history'][-1]['energy_before_update_per_site']


def test_condition_arm_fixed_endpoint_resumes_and_reuses_frozen_audit(tmp_path):
    spec=dict(geometry='square',rows=2,columns=2,j2_over_j1=.5,multiplicity_cap=2,
              maximum_q=2,projection_chunk=2,require_gpu=False,seed=12,
              calibration_seed=13,sampling_seed=14,literature_target_energy_total=-2.,
              literature_target_total_standard_error=.01)
    gate=cost(spec,tmp_path/'cost')
    spec.update(cost_parent=dict(directory=str(tmp_path/'cost'),checkpoint_sha256=gate['checkpoint_sha256'],
                registration_sha256=gate['registration_sha256']),training_seed=15,chains=4,steps=4,
                sr_warm_steps=1,warm_samples=8,adam_samples=8,burn_proposals=8,
                proposals_between_snapshots=4,score_chunk=1,H_sample_chunk=4,sr_damping=.01,
                maximum_FS_sine=.05,warm_lr=.005,adam_lr=.001,fold_root_swap=True,
                reset_moments_at_phase_switch=True,milestones=[0,1,2,3,4],maximum_training_seconds=120)
    original=train(spec,tmp_path/'parent',2,120)
    parent=dict(directory=str(tmp_path/'parent'),source_workspace=str(ROOT),
        checkpoint_sha256=original['checkpoint_sha256'],registration_sha256=original['registration_sha256'],
        step=2,active_seconds=original['active_seconds'])
    child=dict(spec,output=str(tmp_path/'condition'),sr_warm_steps=4,
        training_resume_from=dict(parent,same_projected_physical_state_proved=True,
            reset_optimizer_moments=True,optimizer_reset_reason='tested same-parent recipe',
            optimizer_condition_change=dict(sr_warm_steps=dict(from_value=1,to_value=4))))
    audit_spec=dict(training_directory=child['output'],output=str(tmp_path/'condition_audit'),
        endpoint_step=4,require_gpu=False,streams=2,chains_per_stream=4,snapshots_per_stream=2,
        burn_proposals=8,proposals_between_snapshots=4,audit_seeds=[501,502],
        bootstrap_seed=503,H_sample_chunk=4)
    train_config=tmp_path/'train.json';audit_config=tmp_path/'audit.json'
    atomic_json(train_config,child);atomic_json(audit_config,audit_spec)
    plan=dict(parent=parent,updates_per_arm=2,session_seconds_per_half=120,
        arms=dict(test=dict(training_config=str(train_config),training_config_sha256=file_sha256(train_config),
            audit_config=str(audit_config),audit_config_sha256=file_sha256(audit_config),
            midpoint_step=3,endpoint_step=4)))
    result=compare_condition_arm(plan,'test',plan_sha256=canonical_sha256(plan))
    assert result['completed_new_updates'] == result['optimizer_steps'] == 2
    assert result['checkpoint_resume']['midpoint_optimizer_steps'] == 1
    assert result['audit']['records'] == 16
    repeated=compare_condition_arm(plan,'test',plan_sha256=canonical_sha256(plan))
    assert repeated['checkpoint_sha256'] == result['checkpoint_sha256']
    assert repeated['audit'] == result['audit']
    # Preserve the parent's existing Adam state when a separately registered
    # direct-SR continuation is requested. No second import reset is hidden.
    from su2_nqs.vmc.checkpoint import load_checkpoint
    count=int(load_checkpoint(tmp_path/'parent/checkpoint.pkl')['optimizer_state'][0])
    direct=dict(child,output=str(tmp_path/'direct'),sr_use_adam=False,sr_tangent_fs_radius=.02,
        training_resume_from=dict(child['training_resume_from'],reset_optimizer_moments=False,
            optimizer_condition_change=dict(sr_warm_steps=dict(from_value=1,to_value=4),
                sr_use_adam=dict(from_value=True,to_value=False),
                sr_tangent_fs_radius=dict(from_value=None,to_value=.02))))
    direct_audit=dict(audit_spec,training_directory=direct['output'],output=str(tmp_path/'direct_audit'))
    dc=tmp_path/'direct.json';da=tmp_path/'direct_audit.json';atomic_json(dc,direct);atomic_json(da,direct_audit)
    dp=dict(plan,require_import_moment_reset=False,initial_optimizer_step=count,
        arms=dict(test=dict(training_config=str(dc),training_config_sha256=file_sha256(dc),
            audit_config=str(da),audit_config_sha256=file_sha256(da),midpoint_step=3,endpoint_step=4)))
    dr=compare_condition_arm(dp,'test',plan_sha256=canonical_sha256(dp))
    assert dr['optimizer_steps']==count+2
    assert dr['checkpoint_resume']['initial_optimizer_step']==count
    assert not dr['checkpoint_resume']['explicit_once_only_import_reset']
