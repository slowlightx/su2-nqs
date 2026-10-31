"""Zero-H coverage guards for fixed complete-A1 endpoint support chains."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_audit import (
    make_pair_support_guide, run_pair_support_stage, support_coverage_diagnostics,
    support_work_limit, support_initial_rng_key, support_runtime_identity, run_pair_support_cost,
    support_projection_chunk, bridge_projection_chunk, run_h_bridge_endpoint_audit,
)


def test_support_guide_is_exact_joint_probability_without_floor_or_underflow():
    jax.config.update('jax_enable_x64',True)
    def amplitude(bank,bits):
        return bank*(1+1j*bits[0])
    guide=jax.jit(make_pair_support_guide(amplitude))
    bits=jnp.asarray([1,0],dtype=jnp.int32)
    for scale in (1.,1e-200):
        actual=float(guide((3*scale,4*scale),bits))
        assert actual>0
        np.testing.assert_allclose(actual/scale,np.sqrt(50),rtol=3e-16)
    assert float(guide((0.,0.),bits))==0


def test_fixed_support_window_does_not_promote_the_old_uniform_floor_platform():
    p=np.full((16,8,2),1e-20,dtype=complex)
    report=support_coverage_diagnostics(p,old_floor=1.)
    assert not report['operational_gate_passed']
    assert not report['checks']['left_old_floor_platform']
    assert report['checks']['endpoint_norm_contribution_spread']
    assert report['finite_burn_equilibrium_certified'] is False
    assert report['effective_contributions_are_not_autocorrelation_corrected_ESS']


def test_exactly_constant_observables_can_pass_necessary_checks_but_not_certify_mixing():
    p=np.full((16,8,2),2.,dtype=complex)
    report=support_coverage_diagnostics(p,old_floor=1.)
    assert report['operational_gate_passed']
    assert not report['finite_burn_equilibrium_certified']
    assert report['all_chains_retained']
    np.testing.assert_allclose(report['final_log_Q0_over_old_floor'],np.log(8))
    np.testing.assert_allclose(report['endpoint_weight_diagnostics'][0]['effective_contributions'],128,rtol=3e-16)


def test_monotonic_search_growth_is_not_mistaken_for_stationary_coverage():
    logq=10+np.arange(16,dtype=float)[:,None]+np.linspace(-.05,.05,8)[None,:]
    p=np.repeat(np.sqrt(np.exp(logq)/2)[...,None],2,axis=-1)
    report=support_coverage_diagnostics(p,old_floor=1.)
    assert report['checks']['left_old_floor_platform']
    assert not report['checks']['log_Q0_between_chain_and_time_stability']
    assert report['diagnostics']['log_Q0']['paired_half_window_drift']>7.9


def test_only_the_fixed_trailing_window_enters_the_operational_gate():
    p=np.concatenate((np.full((16,8,2),1e-100),np.full((16,8,2),2.)),axis=0)
    report=support_coverage_diagnostics(p,old_floor=1.,window_records=16)
    assert report['operational_gate_passed']
    assert report['records_per_chain']==32
    assert report['fixed_window_records']==16
    np.testing.assert_allclose(report['endpoint_weight_diagnostics'][0]['effective_contributions'],128,rtol=3e-16)


def test_all_chains_are_retained_when_one_has_no_endpoint_weight():
    p=np.full((16,8,2),2.)
    p[:,:,0]=0.;p[:,0,0]=4.
    report=support_coverage_diagnostics(p,old_floor=1.)
    assert not report['operational_gate_passed']
    assert not report['checks']['endpoint_norm_contribution_spread']
    assert report['endpoint_weight_diagnostics'][0]['maximum_chain_fraction']==1.
    assert report['chains']==8


@pytest.mark.parametrize('change',[{'old_floor':0.},{'window_records':7},
    {'maximum_split_rhat':np.nan},{'minimum_effective_fraction':1.2},
    {'maximum_chain_fraction':0.}])
def test_support_diagnostic_rejects_invalid_guards(change):
    with pytest.raises(ValueError):
        support_coverage_diagnostics(np.ones((16,8,2)),**(dict(old_floor=1.)|change))


def test_common_nodes_fail_closed_and_short_prefix_never_passes():
    with pytest.raises(FloatingPointError):
        support_coverage_diagnostics(np.zeros((16,8,2)),old_floor=1.)
    report=support_coverage_diagnostics(np.ones((8,8,2)),old_floor=1.)
    assert report['status']=='insufficient_fixed_window'
    assert not report['operational_gate_passed']


@pytest.mark.parametrize('changes',[{'segments':5},{'chains':7},{'segment_records':0},
    {'seconds_cap':np.inf},{'seconds_cap':5401},{'maximum_registered_proposals':2049},
    {'maximum_registered_proposals':True},{'maximum_registered_proposals':0}])
def test_support_stage_work_limits_fail_before_inputs_or_any_physical_queries(tmp_path,changes):
    spec=dict(model=dict(length=12,multiplicity_cap=20,maximum_q=10),
        chains=8,segments=4,segment_burn=8,segment_records=8,segment_between=15,seconds_cap=1200.)
    with pytest.raises(ValueError):run_pair_support_stage(spec|changes,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_explicit_separately_registered_continuation_budget_reaches_input_checks(tmp_path):
    spec=dict(model=dict(length=12,multiplicity_cap=20,maximum_q=10),
        chains=8,segments=12,segment_burn=8,segment_records=8,segment_between=15,
        seconds_cap=3600.,maximum_registered_proposals=1536)
    # Missing checkpoints stop this test before files, JAX, sampling or H.
    with pytest.raises(KeyError,match='control_checkpoint'):
        run_pair_support_stage(spec,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_large_support_budget_requires_explicit_single_gpu_profile():
    with pytest.raises(ValueError):
        support_work_limit({'maximum_registered_proposals': 10240})
    assert support_work_limit(dict(execution_profile='registered_single_gpu',
        maximum_registered_proposals=10240)) == 10240
    assert support_work_limit(dict(execution_profile='registered_single_gpu',
        maximum_registered_proposals=16384)) == 16384
    for changes in ({'maximum_registered_proposals': 16385},
                    {'maximum_registered_proposals': True}, {'execution_profile': 'automatic'}):
        with pytest.raises(ValueError):
            support_work_limit(dict(execution_profile='registered_single_gpu') | changes)


@pytest.mark.parametrize('chunk', [0, -4, True, 4., 5, 1729])
@pytest.mark.parametrize('entry', [run_pair_support_cost, run_pair_support_stage, run_h_bridge_endpoint_audit])
def test_sampler_projection_chunk_fails_before_inputs_or_physical_queries(tmp_path, chunk, entry):
    spec = dict(model=dict(length=12, multiplicity_cap=20, maximum_q=10),
        sampler_projection_chunk=chunk)
    with pytest.raises(ValueError, match='sampler_projection_chunk'):
        entry(spec, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_sampler_projection_chunk_default_and_full_group_divisors():
    assert support_projection_chunk({}) == 4
    for chunk in (4, 16, 32, 64):
        assert support_projection_chunk({'sampler_projection_chunk':chunk}) == chunk


@pytest.mark.parametrize('chunk', [0, -4, True, 4., 5, 1729])
def test_H_projection_chunk_fails_before_inputs_or_physical_queries(tmp_path, chunk):
    spec = dict(model=dict(length=12, multiplicity_cap=20, maximum_q=10),
        H_projection_chunk=chunk)
    with pytest.raises(ValueError, match='H_projection_chunk'):
        run_h_bridge_endpoint_audit(spec, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_H_batching_is_separate_from_sampler_batching_and_legacy_default():
    assert bridge_projection_chunk({'sampler_projection_chunk':64}) == 4
    assert support_projection_chunk({'H_projection_chunk':64}) == 4
    for chunk in (4,16,32,64):
        assert bridge_projection_chunk({'H_projection_chunk':chunk}) == chunk


def test_explicit_saved_rng_is_not_reseeded():
    key=np.asarray(jax.random.PRNGKey(2718))
    np.testing.assert_array_equal(support_initial_rng_key({'initial_rng_key':key.tolist(), 'seed':1}),key)
    np.testing.assert_array_equal(support_initial_rng_key({'seed':2718}),key)
    for key in ([0.,1.], [True,False], [-1,0], [2**32,0], [1], [1,2,3]):
        with pytest.raises(ValueError):support_initial_rng_key({'initial_rng_key':key})


def test_registered_gpu_route_fails_closed_on_cpu_runtime():
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':pytest.skip('CPU-only negative runtime check')
    with pytest.raises(ValueError,match='exactly one visible GPU'):
        support_runtime_identity({'execution_profile':'registered_single_gpu'})
    identity=support_runtime_identity({'execution_profile':'registered_single_gpu'},cpu_reference=True)
    assert identity['backend']=='cpu' and identity['cpu_reference_only'] and identity['x64_enabled']


def _cost_fixture(tmp_path, monkeypatch):
    import hashlib
    from projects.clebsch_tree_nqs.workflows import triangle_hierarchy_audit as audit
    initial=np.tile(np.r_[np.zeros(72,dtype=np.int32),np.ones(72,dtype=np.int32)],(8,1))
    psi=np.tile([1.,2.],(8,1));key=np.asarray(jax.random.PRNGKey(47))
    sample=tmp_path/'samples.npz';np.savez(sample,terminal_states=initial,terminal_psi=psi,rng=key)
    paths=[tmp_path/'old.pkl',tmp_path/'new.pkl']
    for path in paths:path.write_bytes(path.name.encode())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    spec=dict(model=dict(length=12,multiplicity_cap=20,maximum_q=10),j2_over_j1=.125,
        execution_profile='registered_single_gpu',maximum_registered_proposals=10240,
        chains=8,segments=80,segment_burn=8,segment_records=8,segment_between=15,
        initial_rng_key=key.tolist(),initial_samples_path=str(sample),initial_samples_sha256=sha(sample),
        active_top_widths=[{'0':8}]*14+[{'0':1}],cpu_reference_only=True)
    for name,path in zip(('control','candidate'),paths):
        spec[name+'_checkpoint']=str(path);spec[name+'_sha256']=sha(path)
    monkeypatch.setattr(audit,'load_checkpoint',lambda path:{'params':{'c':jnp.asarray(1. if path==paths[0] else 2.)}})
    monkeypatch.setattr(audit,'repack_active_clebsch_bank',lambda bank,*args:bank)
    monkeypatch.setattr(audit,'prepare_feature_lut_bank',lambda bank,*args:bank)
    monkeypatch.setattr(audit,'triangular_hnf_gamma_a1_group',lambda *args:np.tile(np.arange(144),(1728,1)))
    monkeypatch.setattr(audit,'triangular_j1j2_bonds',lambda *args,**kwargs:[(0,1,1.)])
    monkeypatch.setattr(audit,'make_feature_lut_projected_amplitude',lambda *args,**kwargs:lambda bank,bits:bank['c'])
    called=[]
    def make_sampler(guide,*,records,burn_proposals,proposals_between,**kwargs):
        called.append((records,burn_proposals,proposals_between))
        def sample(prepared,key,states):
            return states,key+jnp.uint32(burn_proposals+records*proposals_between),jnp.repeat(states[None],records,0),{}
        return sample
    monkeypatch.setattr(audit,'make_record_sampler',make_sampler)
    return spec,called


def test_CPU_reference_never_runs_a128_proposal_segment(tmp_path,monkeypatch):
    spec,called=_cost_fixture(tmp_path,monkeypatch)
    result=run_pair_support_cost(spec,tmp_path/'reference')
    assert result['status']=='CPU_short_reference_complete_no_long_sampling'
    assert result['actual_full_H_centers']==0 and not result['production_chain_advanced']
    assert result['completed_amplitude_centers']==128
    assert called==[(1,1,1)]
    with np.load(tmp_path/'reference/short_reference.npz') as z:
        np.testing.assert_array_equal(z['initial_rng_key'],spec['initial_rng_key'])
        assert z['short_rows'].shape==(1,8,144)


def test_sampler_chunk_changes_only_forward_factory_and_is_recorded(tmp_path, monkeypatch):
    import json
    from projects.clebsch_tree_nqs.workflows import triangle_hierarchy_audit as audit
    spec, called = _cost_fixture(tmp_path, monkeypatch)
    chunks = []
    def factory(*args, chunk, **kwargs):
        chunks.append(chunk)
        return lambda bank, bits: bank['c']
    monkeypatch.setattr(audit, 'make_feature_lut_projected_amplitude', factory)
    result = run_pair_support_cost(spec|{'sampler_projection_chunk':64}, tmp_path/'cost64')
    registration = json.loads((tmp_path/'cost64/registration.json').read_text())
    assert chunks == [64] and called == [(1,1,1)]
    assert result['sampler_projection_chunk'] == registration['sampler_projection_chunk'] == 64
    assert result['actual_full_H_centers'] == 0
    assert registration['batching_changes_coherent_sum_order_only']


def test_explicit_saved_rng_mismatch_stops_cost_before_outputs(tmp_path,monkeypatch):
    spec,_=_cost_fixture(tmp_path,monkeypatch)
    spec['initial_rng_key']=[0,48]
    with pytest.raises(ValueError,match='saved starting RNG'):
        run_pair_support_cost(spec,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_GPU_cost_reuses_the_same_initial_state_and_key_without_production_advance(tmp_path,monkeypatch):
    import hashlib
    from projects.clebsch_tree_nqs.workflows import triangle_hierarchy_audit as audit
    spec,called=_cost_fixture(tmp_path,monkeypatch)
    run_pair_support_cost(spec,tmp_path/'cpu')
    path=tmp_path/'cpu/short_reference.npz'
    monkeypatch.setattr(audit,'support_runtime_identity',lambda *args,**kwargs:dict(backend='gpu',x64_enabled=True))
    gpu=spec|dict(cpu_reference_only=False,cpu_reference_path=str(path),
        cpu_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=run_pair_support_cost(gpu,tmp_path/'gpu')
    assert result['status']=='GPU_support_cost_complete'
    assert result['CPU_GPU_short_states_and_rng_exact']
    assert result['registered_segment_count']==80 and result['actual_full_H_centers']==0
    assert not result['production_chain_advanced']
    assert called[-2:]==[(1,1,1),(8,8,15)]
    with np.load(tmp_path/'gpu/segment_cost_trace.npz') as z:
        np.testing.assert_array_equal(z['initial_states'],z['terminal_states'])
        np.testing.assert_array_equal(z['rng'],np.asarray(spec['initial_rng_key'],dtype=np.uint32)+128)


def test_actual_GPU_short_trace_mismatch_is_rejected_before_long_cost(tmp_path,monkeypatch):
    import hashlib
    from projects.clebsch_tree_nqs.workflows import triangle_hierarchy_audit as audit
    spec,called=_cost_fixture(tmp_path,monkeypatch)
    run_pair_support_cost(spec,tmp_path/'cpu')
    path=tmp_path/'cpu/short_reference.npz'
    with np.load(path) as z:data={k:z[k].copy() for k in z.files}
    data['short_rng']+=1
    bad=tmp_path/'bad_reference.npz';np.savez(bad,**data)
    monkeypatch.setattr(audit,'support_runtime_identity',lambda *args,**kwargs:dict(backend='gpu',x64_enabled=True))
    gpu=spec|dict(cpu_reference_only=False,cpu_reference_path=str(bad),
        cpu_reference_sha256=hashlib.sha256(bad.read_bytes()).hexdigest())
    with pytest.raises(ArithmeticError,match='short_rng'):
        run_pair_support_cost(gpu,tmp_path/'gpu')
    assert called==[(1,1,1),(1,1,1)]


def test_N36_complete_A1_chunk4_chunk16_amplitudes_and_short_fixed_rng_trace():
    from su2_nqs.models.triangular_mixed_clebsch import (
        MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_bank,
    )
    from su2_nqs.models.triangular_clebsch_feature_lut import (
        prepare_feature_lut_bank, make_feature_lut_projected_amplitude,
    )
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_audit import make_record_sampler
    jax.config.update('jax_enable_x64', True)
    cfg = MixedClebschConfig(length=6, multiplicity_cap=9, maximum_q=3)
    family = init_mixed_clebsch_family(jax.random.PRNGKey(65001), cfg)
    bank = mixed_clebsch_bank(family['ternary_c3'], cfg, 'ternary_c3')
    # Complex root phase tests coherent summation; the all-down row is an exact node.
    bank = dict(bank, top=tuple(bank['top'][:-1]) +
        (tuple((1.+.37j)*w for w in bank['top'][-1]),))
    prepared = prepare_feature_lut_bank(bank, cfg)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))), np.int32)
    rng = np.random.default_rng(65002)
    rows = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(4)] + [[0]*36], np.int32)
    amplitudes = [jax.jit(make_feature_lut_projected_amplitude(cfg, images, chunk=chunk))
                  for chunk in (4,16)]
    values = [np.asarray(jax.jit(jax.vmap(a, in_axes=(None,0)))(prepared,rows)) for a in amplitudes]
    np.testing.assert_allclose(values[0], values[1], rtol=2e-12, atol=2e-16)
    print('N36 chunk4/16 maximum amplitude discrepancy:', float(np.max(abs(values[0]-values[1]))))
    assert values[0][-1] == values[1][-1] == 0
    key = jax.random.PRNGKey(65003)
    traces = [make_record_sampler(make_pair_support_guide(a), records=1,
        burn_proposals=1, proposals_between=1,
        exchange_edges=tuple((i,j) for i,j,_ in triangular_j1j2_bonds(6,6,j2=.125,periodic=True)))
        ((prepared,prepared), key, jnp.asarray(rows[:2]))
        for a in amplitudes]
    for first, second in zip(traces[0][:3], traces[1][:3], strict=True):
        np.testing.assert_array_equal(first, second)

    # Full432-image coherent H and Qpush, with all216 physical J1/J2 edges.
    # Batch size changes neither the sum over images nor any connection term.
    from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_bridge
    bonds = triangular_j1j2_bonds(6,6,j2=.125,periodic=True)
    probability = np.abs([value for _,_,value in bonds]); probability /= probability.sum()
    products = []
    for chunk in (4,16):
        action, metadata = make_cached_projected_heisenberg_bridge(
            cfg, images, bonds, probability, projection_chunk=chunk)
        products.append(tuple(np.asarray(v) for v in action(prepared, rows[:1])))
        assert metadata['physical_unique_edges'] == len(bonds) == 216
    errors = []
    for before, after in zip(products[0], products[1], strict=True):
        relative = float(np.linalg.norm(after-before)/max(np.linalg.norm(before),1e-300))
        errors.append(relative)
        assert relative < 3e-12
    print('N36 chunk4/16 relative errors (psi,Hpsi,Qpush):', errors)
