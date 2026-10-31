"""Independent dense finite-Q and projective-calibration tests; no new ED."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.cached_crossfold import (
    CachedCrossfoldOracle, CrossfoldConfig, directional_moments, whole_chain_folds,
)


def _problem():
    rng = np.random.default_rng(2026101801)
    a = rng.normal(size=(12,12))+1j*rng.normal(size=(12,12))
    h = (a+a.conj().T)/4
    c = rng.normal(size=12)+1j*rng.normal(size=12); c[0] = 0
    f = rng.normal(size=(12,3))+1j*rng.normal(size=(12,3))
    q = np.exp(rng.normal(size=12))
    def state(x):
        p = jnp.asarray(c)+jnp.asarray(f)@jnp.sin(x)
        return p, jnp.asarray(h)@p
    return state, q


def test_directional_energy_and_variance_agree_with_finite_differences_at_nodes():
    state,q = _problem();x=np.zeros(3);d=np.array([.7,-.3,.2])
    (p,hp),(dp,dhp)=jax.jvp(state,(jnp.asarray(x),),(jnp.asarray(d),))
    assert p[0] == 0 and abs(hp[0]) > .1
    exact=directional_moments(p,hp,dp,dhp,q,12)
    def moments(a):
        p,hp=map(np.asarray,state(x+a*d)); z=np.sum(abs(p)**2/q); e=np.sum(p.conj()*hp/q)/z
        return np.array([e.real/12,np.sum(abs(hp-e*p)**2/q)/z/12])
    fd=(moments(1e-6)-moments(-1e-6))/(2e-6)
    errors=abs(fd-np.array([exact['energy_slope_per_site'],exact['variance_slope_per_site']]))
    assert max(errors) < 2e-9
    print('directional energy/variance FD maxerror',max(errors))
    scaled=directional_moments(p*3j,hp*3j,dp*3j,dhp*3j,q*19,12)
    for key in exact:
        if key!='norm_sum':np.testing.assert_allclose(scaled[key],exact[key],atol=2e-14)


def test_tangent_calibration_is_invariant_under_parameter_rescaling_and_same_initial_state():
    q=np.array([.3,.8,1.7,.4]);c=jnp.array([1.,.2,.5,-.6]);v=jnp.array([.3,1.,-.2,.7])
    h=jnp.diag(jnp.array([-.4,.6,-.2,.8]));mask=np.array([True,False,True,False])
    def make(scale):
        def state(x):
            p=c+scale*x[0]*v
            return p,h@p
        return state
    config=CrossfoldConfig(n_sites=4)
    results=[CachedCrossfoldOracle(make(s),q,config).trial(np.zeros(1),mask) for s in (1.,10000.)]
    for record,arrays in results:
        assert record['matched']
        assert record['physical_step']['evaluation_count']==17
        assert abs(record['fit']['initial']['projective_speed']-1)<2e-15
        assert record['fit']['delta_energy_per_site']<0
        assert record['ledger']['new_H_queries']==record['ledger']['new_samples']==0
    np.testing.assert_allclose(results[0][1]['psi_initial'],results[1][1]['psi_initial'],atol=0)
    np.testing.assert_allclose(results[0][1]['psi_endpoint'],results[1][1]['psi_endpoint'],atol=2e-15)
    np.testing.assert_allclose(results[0][1]['hpsi_endpoint'],results[1][1]['hpsi_endpoint'],atol=2e-15)


def test_full_gradient_uses_fit_only_and_both_sides_of_finite_Q_action():
    state,q=_problem();cfg=CrossfoldConfig(n_sites=12);mask=np.arange(12)%4!=0
    oracle=CachedCrossfoldOracle(state,q,cfg);x=np.array([.1,.2,-.3]);record,arrays=oracle.trial(x,mask)
    def direct(x):
        p,hp=map(np.asarray,state(x));return (np.sum(p[mask].conj()*hp[mask]/q[mask])/np.sum(abs(p[mask])**2/q[mask])).real/12
    eye=np.eye(3);fd=np.array([(direct(x+1e-6*d)-direct(x-1e-6*d))/2e-6 for d in eye])
    np.testing.assert_allclose(arrays['gradient'],fd,atol=2e-9)
    np.testing.assert_allclose(record['fit_gradient_dot_direction'],record['fit']['initial']['energy_slope_per_site'],atol=2e-13)
    assert record['matched']
    np.testing.assert_array_equal(x,[.1,.2,-.3])


def test_whole_chain_assignment_and_invalid_or_gauge_cases_fail_closed():
    groups=np.tile(np.arange(16),8);fold=whole_chain_folds(groups)
    for group in range(16):assert np.unique(fold[groups==group]).tolist()==[group%4]
    assert np.bincount(fold).tolist()==[32]*4
    with pytest.raises(ValueError):whole_chain_folds(np.array([1,2]))
    state,q=_problem();oracle=CachedCrossfoldOracle(state,q,CrossfoldConfig(n_sites=12))
    with pytest.raises(ValueError):oracle.trial(np.zeros(3),np.ones(12,dtype=bool))
    with pytest.raises(ValueError):CachedCrossfoldOracle(state,-q,CrossfoldConfig())
    def gauge(x):return (1+x[0])*jnp.ones(4),(1+x[0])*jnp.arange(4.)
    with pytest.raises(FloatingPointError,match='tangent'):
        CachedCrossfoldOracle(gauge,np.ones(4),CrossfoldConfig()).trial(np.zeros(1),np.array([1,1,0,0],dtype=bool))


def test_workflow_protocol_forbids_new_physics_calls_or_posthoc_direction_choices():
    import json
    from projects.clebsch_tree_nqs.workflows.spin_resonance_crossfold import CONFIG, validate_protocol
    spec=json.loads(CONFIG.read_text());validate_protocol(spec)
    for key,value in [('new_H_calls',1),('new_samples',1),('new_ED_calls',1),('starts',[0]),
                      ('target_fs',.002),('wall_seconds_cap',121),('direction','best_energy')]:
        with pytest.raises(ValueError):validate_protocol(dict(spec,**{key:value}))


def test_saved_crossfold_replay_and_source_seals_without_model_or_H_calls():
    import json
    from pathlib import Path
    from su2_nqs.frontier.artifacts import canonical_sha256
    from su2_nqs.frontier.provenance import file_sha256
    root=Path(__file__).resolve().parents[1]
    output=root/'projects/clebsch_tree_nqs/results/spin_resonance_crossfold_v1'
    if not (output/'results.json').exists():pytest.skip('bounded diagnostic artifact not present')
    result=json.loads((output/'results.json').read_text());unsigned=dict(result);seal=unsigned.pop('result_sha256')
    assert canonical_sha256(unsigned)==seal
    assert file_sha256(output/'registration.json')==result['registration_sha256']
    reg=json.loads((output/'registration.json').read_text())
    for path,sha in reg['sources'].items():assert file_sha256(root/path)==sha
    assert result['independent_evidence'] is False and result['variational_win_claimed'] is False
    assert result['new_H_queries']==result['new_samples']==result['new_ED_calls']==0
    assert len(result['records'])==24
    assert file_sha256(output/'trials.npz')==result['artifacts']['trials.npz']
    worst=0.
    with np.load(output/'trials.npz',allow_pickle=False) as data:
        for key,row in result['records'].items():
            fit=data[key+'__fit_mask'];q=data['q']
            assert row['physical_step']['evaluation_count']==17
            assert row['matched']
            for name,mask in [('fit',fit),('held_out',~fit)]:
                p,h,dp,dh=[data[key+'__'+field][mask] for field in ('psi_initial','hpsi_initial','tangent','htangent')]
                initial=directional_moments(p,h,dp,dh,q[mask],36)
                p,h=[data[key+'__'+field][mask] for field in ('psi_endpoint','hpsi_endpoint')]
                end=directional_moments(p,h,np.zeros_like(p),np.zeros_like(h),q[mask],36)
                for field,value in initial.items():worst=max(worst,abs(value-row[name]['initial'][field]))
                for field,value in end.items():worst=max(worst,abs(value-row[name]['endpoint'][field]))
            np.testing.assert_allclose(data[key+'__endpoint'],data[key+'__initial']+
                row['physical_step']['alpha']*data[key+'__direction'],atol=0,rtol=0)
        for start in range(3):
            for fold in range(4):
                for field in ('psi_initial','hpsi_initial'):
                    np.testing.assert_array_equal(data[f'matched_free{start}__fold{fold}__{field}'],
                                                  data[f'attention{start}__fold{fold}__{field}'])
    assert worst < 1e-12
    print('saved crossfold moments replay maxerror',worst)
