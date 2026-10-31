import copy
import jax
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct6_native_direction_calibration as workflow


def plan():
    return dict(task=workflow.TASK,model=dict(length=6,multiplicity_cap=28,maximum_q=10),
        j2_over_j1=.125,arity='binary_truncated',binary_rank=8,fractions=list(workflow.GRID),
        fit=dict(samples=4096,chains=256,refresh_proposals=1476,spacing_proposals=36),
        calibration_records=4096,calibration_split=2048,confirmation_records=2048,damping=.5,
        target_FS=.001,maximum_FS=.005,wall_cap_seconds=14400,maximum_peak_device_bytes=70*1024**3,
        automatic_continuation=False,seeds=dict(zip(('fit','selection','guard','confirmation','statistics'),range(701,706))),
        parent_registered_seeds=[801],parent=dict(global_step=476,params_sha256='a'*64,
            terminal_products_sha256='c'*64,
            sealed_files={n:'b'*64 for n in ('checkpoint.pkl','registration.json','results.json','terminal_audit.json')}),
        projection_chunk=4,action_chunk=4,score_chunk=2,confirmation_chunk=4,execution_authorized=False)


def test_fixed_protocol_cannot_silently_increase_score_memory_or_release_run():
    p=plan();workflow.validate_plan(p)
    with pytest.raises(PermissionError):workflow.validate_plan(p,executing=True)
    for field,value in [('damping',.05),('calibration_records',128),('automatic_continuation',True),('fractions',[1.,0.])]:
        q=copy.deepcopy(p);q[field]=value
        with pytest.raises(ValueError):workflow.validate_plan(q)
    q=copy.deepcopy(p);q['fit']['samples']=8192
    with pytest.raises(ValueError):workflow.validate_plan(q)
    q=copy.deepcopy(p);q['seeds']['guard']=q['seeds']['selection']
    with pytest.raises(ValueError):workflow.validate_plan(q)
    q=copy.deepcopy(p);q['seeds']['guard']=801
    with pytest.raises(ValueError):workflow.validate_plan(q)


def test_x64_is_checked_before_any_parent_pickle_load(monkeypatch):
    monkeypatch.setattr(workflow,'load_checkpoint',lambda *_:pytest.fail('must reject BEFORE load'))
    before=jax.config.x64_enabled
    try:
        jax.config.update('jax_enable_x64',False)
        with pytest.raises(RuntimeError,match='BEFORE'):
            workflow.checked_parent(plan())
    finally:
        jax.config.update('jax_enable_x64',before)


def test_sampler_only_canonicalization_does_not_mutate_native_optimizer_coordinates():
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank
    from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
    cfg=MixedClebschConfig(multiplicity_cap=9,maximum_q=3)
    native=init_mixed_clebsch_family(jax.random.PRNGKey(271203),cfg)['ternary_c3']
    before=workflow.pytree_sha256(native)
    rt=dict(cfg=cfg,bank=lambda p:mixed_clebsch_bank(p,cfg,'ternary_c3'))
    bank,report=workflow.normalized_raw_bank(native,rt)
    assert workflow.pytree_sha256(native)==before
    assert triangular_raw_overlap(bank,bank,cfg)==pytest.approx(1.,abs=3e-14)
    assert report['native_optimizer_coordinates_unchanged']
    assert report['raw_cross_overlap_error']<3e-14


def test_actual_raw_IID_draw_and_orbit_query_batch_shapes():
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    cfg=MixedClebschConfig(multiplicity_cap=9,maximum_q=3)
    native=init_mixed_clebsch_family(jax.random.PRNGKey(271207),cfg)['ternary_c3']
    rt=dict(cfg=cfg,bank=lambda p:mixed_clebsch_bank(p,cfg,'ternary_c3'),
        images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32),
        bonds=tuple(triangular_j1j2_bonds(6,6,j2=.125,periodic=True)))
    bank,_=workflow.normalized_raw_bank(native,rt)
    rows,draw=workflow.draw_orbit_rows([bank,bank],rt,2,271208,bridge=True)
    repeat,_=workflow.draw_orbit_rows([bank,bank],rt,2,271208,bridge=True)
    assert np.array_equal(rows,repeat) and rows.shape==(2,36)
    assert np.all(rows.sum(1)==18) and draw['latent'].shape==(2,4)
    assert draw['maximum_Born_replay_error']<3e-12
    prepared=workflow.prepare_feature_lut_bank(bank,cfg)
    orbit=workflow.make_orbit_amplitude_products(workflow.make_feature_lut_amplitude(cfg),rt['images'],projection_chunk=4)
    query=jax.jit(lambda b,x:jax.lax.map(lambda row:workflow.jnp.stack(orbit(b,row[None]))[:,0],x,batch_size=2))
    values=np.asarray(query(prepared,rows))
    assert values.shape==(2,2) and np.isfinite(values).all()
    assert np.all(values[:,1].real>0)
    assert np.all(abs(values[:,0])**2<=values[:,1].real*(1+2e-13))
    # Regression for the actual failed path: this wrapper is NOT itself jitted.
    # The production orbit function dynamically indexes its input during remat.
    from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_projected_amplitude
    replay_rt=dict(prepare=lambda p:workflow.prepare_feature_lut_bank(p,cfg),
        projected=make_feature_lut_projected_amplitude(cfg,rt['images'],chunk=4))
    host_replay=np.asarray(workflow.native_projected_amplitude(replay_rt,bank,np.asarray(rows)))
    device_replay=np.asarray(workflow.native_projected_amplitude(replay_rt,bank,workflow.jnp.asarray(rows)))
    np.testing.assert_allclose(host_replay,values[:,0],rtol=2e-12,atol=1e-20)
    np.testing.assert_array_equal(host_replay,device_replay)


def test_saved_parent_replay_preserves_time_major_record_order(tmp_path):
    rows=np.zeros((16,36),np.int32);rows[:,0]=np.arange(16)%2
    energies=np.arange(16).reshape(4,4)+1.
    path=tmp_path/'audit.npz';np.savez(path,bits=rows,local_energies=energies)
    rt=dict(prepare=lambda p:p,action=lambda p,x:(np.ones(len(x)),np.arange(len(x))+1.))
    result=workflow.audited_parent_replay(rt,{},path)
    assert result['maximum_relative_error']==0 and result['full_H_centers']==2
    rt['action']=lambda p,x:(np.ones(len(x)),np.array([1.,5.]))
    with pytest.raises(ArithmeticError,match='replay'):
        workflow.audited_parent_replay(rt,{},path)


def test_absolute_allocation_deadline_keeps_serialization_margin_and_science_cap():
    budget=workflow.allocation_science_seconds
    assert budget(14400,None,1000)==14400
    assert budget(14400,'50000',1000)==14400
    assert budget(14400,'15000',1000)==13880
    assert budget(14400,'15000',2000)==12880
    assert budget(14400,'1120.5',1000)==.5
    for end in ('1120','1000'):
        with pytest.raises(TimeoutError):budget(14400,end,1000)
    for end in ('nan','inf','not-an-epoch'):
        with pytest.raises(ValueError):budget(14400,end,1000)
    with pytest.raises(ValueError):budget(float('nan'),None,1000)


def test_resume_requires_all_prior_file_seals_and_never_consumed_confirmation(tmp_path):
    p=plan();p.update(execution_authorized=True,authorization_note='unit fixture only')
    archived=tmp_path/'registered_sources';archived.mkdir()
    (archived/'old.py').write_text('old numerical source\n')
    for name in ['frozen_direction.pkl','selection_rows.npz','guard_rows.npz']:
        (tmp_path/name).write_bytes(name.encode())
    for i in range(5):
        (tmp_path/f'finite_native_{i}.pkl').write_bytes(str(i).encode())
        (tmp_path/f'finite_canonical_{i}.pkl').write_bytes(('canonical'+str(i)).encode())
    reg=dict(plan=p,plan_sha256=workflow.canonical_sha256(p),
        sources={'old.py':workflow.file_sha256(archived/'old.py')})
    failed=dict(status='failed_validation',error='TracerArrayConversionError(known boundary)',registration=reg,
        fit_SR={'accepted':True},work=dict(fit_H_centers=4096,calibration_bank_centers=20480,
            native_calibration_replay_amplitude_centers=0,confirmation_scalar_H_centers=0,confirmation_H_JVP_centers=0),
        source_seals_unchanged={'old.py':True},parent_seals_unchanged={'checkpoint.pkl':True})
    grid=dict(registration=reg,direction_sha256=workflow.file_sha256(tmp_path/'frozen_direction.pkl'),
        native_checkpoint_sha256=[workflow.file_sha256(tmp_path/f'finite_native_{i}.pkl') for i in range(5)])
    for name,value in [('registration.json',reg),('results.json',failed),('finite_grid_frozen.json',grid)]:
        workflow.atomic_json(tmp_path/name,value)
    manifest=workflow.resume_source_manifest(tmp_path)
    assert workflow.checked_resume_source(tmp_path,manifest)==(reg,failed,grid)
    (tmp_path/'selection_rows.npz').write_bytes(b'replacement rows')
    with pytest.raises(ValueError,match='seals changed'):workflow.checked_resume_source(tmp_path,manifest)
    (tmp_path/'selection_rows.npz').write_bytes(b'selection_rows.npz')
    workflow.atomic_json(tmp_path/'confirmation_draw.json',{'already_consumed':True})
    with pytest.raises(ValueError,match='boundary'):
        workflow.checked_resume_source(tmp_path,workflow.resume_source_manifest(tmp_path))
