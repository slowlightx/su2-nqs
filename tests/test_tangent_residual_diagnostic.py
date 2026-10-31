"""Independent exact finite-vector tangent/FS checks, no eigensolver."""
import numpy as np
import pytest
from su2_nqs.vmc.tangent_residual_diagnostic import project_residual_on_tangents,gradient_conflict_matrix,projector_secant_alignment
from su2_nqs.vmc.matched_state_step import matched_state_step


def test_real_tangent_projection_not_complex_span_and_gauge_invariance():
    p=np.array([1.,0,0],complex);h=np.array([-.3,2.+3j,4j]);t=np.array([[.7j,2.],[1.,0],[0,1j]])
    result=project_residual_on_tangents(p,h,t,n_sites=3)
    np.testing.assert_allclose(result.projected_residual,[0,2,4j],atol=1e-14)
    assert result.metrics['projected_residual_fraction']==pytest.approx(20/29)
    assert result.metrics['unit_FS_descent_slope_per_site']==pytest.approx(-2*np.sqrt(20)/3)
    assert result.metrics['unit_FS_speed']==pytest.approx(1)
    transformed=t*np.array([3.,-.2])+p[:,None]*np.array([9+4j,2-8j])
    other=project_residual_on_tangents(p*1j,h*1j,transformed*1j,n_sites=3)
    np.testing.assert_allclose(other.projected_residual,1j*result.projected_residual,atol=1e-14)
    assert other.metrics['projected_residual_fraction']==pytest.approx(20/29)


def test_declared_span_projection_matches_directional_FD_and_fixed_FS_probe():
    rng=np.random.default_rng(12019);p=rng.normal(size=7)+1j*rng.normal(size=7);p/=np.linalg.norm(p)
    a=rng.normal(size=(7,7))+1j*rng.normal(size=(7,7));h=(a+a.conj().T)/2
    t=rng.normal(size=(7,3))+1j*rng.normal(size=(7,3))
    r=project_residual_on_tangents(p,h@p,t,n_sites=7)
    delta=t@r.descent_coefficients
    def candidate(alpha):
        v=p+alpha*delta;return v/np.linalg.norm(v)
    def energy(alpha):
        v=candidate(alpha);return np.vdot(v,h@v).real/7
    fd=(energy(1e-6)-energy(-1e-6))/2e-6
    assert abs(fd-r.metrics['unit_FS_descent_slope_per_site'])<2e-9
    step=matched_state_step(candidate,p,np.ones(7),.001,.002,16)
    assert step.matched and energy(step.alpha)<energy(0)
    assert r.metrics['projected_residual_fraction']<=1+1e-14
    assert r.metrics['orthogonality_defect']<1e-12


def test_redundant_or_gauge_columns_and_invalid_inputs():
    p=np.array([1,0],complex);h=np.array([0,1],complex)
    r=project_residual_on_tangents(p,h,np.array([[1j,2,0],[0,0,1]]),n_sites=2)
    assert r.metrics['rank']==1 and r.metrics['projected_residual_fraction']==pytest.approx(1)
    null=project_residual_on_tangents(p,h,p[:,None]*1j,n_sites=2)
    assert null.metrics['rank']==0 and null.metrics['unit_FS_speed']==0
    with pytest.raises(ValueError):project_residual_on_tangents(2*p,2*h,p[:,None],n_sites=2)
    with pytest.raises(ValueError):project_residual_on_tangents(p,h*1j+p*1j,p[:,None],n_sites=2)
    c=gradient_conflict_matrix(np.array([[1,0],[-1,0],[0,0]]))
    assert c['cosines'][0][1]==-1 and c['defined'][2][0] is False


def test_projector_secants_match_explicit_density_differences_and_independent_phases():
    rng=np.random.default_rng(14190);v=rng.normal(size=(4,7))+1j*rng.normal(size=(4,7));v/=np.linalg.norm(v,axis=1)[:,None]
    actual=projector_secant_alignment(*v,coupling_span=.3)
    matrices=[np.outer(x,x.conj()) for x in v];a=(matrices[1]-matrices[0])/.3;b=(matrices[3]-matrices[2])/.3
    assert actual['projector_secant_cosine']==pytest.approx(np.vdot(a,b).real/(np.linalg.norm(a)*np.linalg.norm(b)),abs=1e-14)
    assert actual['first_over_second_speed']==pytest.approx(np.linalg.norm(a)/np.linalg.norm(b),abs=1e-14)
    shifted=v*np.array([2j,-3.,.2-.8j,7j])[:,None]
    other=projector_secant_alignment(*shifted,coupling_span=.3)
    assert actual==pytest.approx(other)


@pytest.fixture(scope='module')
def stationarity_artifact():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    folder=root/'projects/clebsch_tree_nqs/results/spin_coupling_stationarity_v1'
    if not (folder/'results.json').exists():pytest.skip('read-only stationarity diagnostic not present')
    result=json.loads((folder/'results.json').read_text())
    with np.load(folder/'directions_and_probes.npz',allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
    source=root/'projects/clebsch_tree_nqs/results/spin_coupling_transfer_v1'
    with np.load(source/'operators.npz',allow_pickle=False) as a:operators={k:a[k].copy() for k in a.files}
    return root,folder,result,data,operators


def test_stationarity_artifact_sources_and_fixed_scope(stationarity_artifact):
    import json
    from su2_nqs.frontier.artifacts import canonical_sha256
    from su2_nqs.frontier.provenance import file_sha256
    root,folder,result,_,_=stationarity_artifact
    unsigned=dict(result);seal=unsigned.pop('result_sha256');assert canonical_sha256(unsigned)==seal
    assert file_sha256(folder/'registration.json')==result['registration_sha256']
    reg=json.loads((folder/'registration.json').read_text());bare=dict(reg);seal=bare.pop('registration_sha256')
    assert canonical_sha256(bare)==seal
    for kind in ('sources','inputs'):
        for file,sha in reg[kind].items():assert file_sha256(root/file)==sha
    for file,sha in result['artifacts'].items():assert file_sha256(folder/file)==sha
    assert len(result['records'])==40 and result['finite_state_probes']==80
    assert result['full_parameter_gradient_objectives']==120 and result['state_tangent_JVPs']==200
    assert result['conditioning_bias_JVPs']==160 and result['coupling_input_JVPs']==10
    assert result['FS_candidate_state_evaluations']==1360
    for key in ('training_updates_committed','new_ED_calls','new_samples','new_H_constructions'):assert result[key]==0
    assert result['finite_direction_lower_bounds_only'] and not result['independent_evidence'] and not result['variational_win_claimed']
    for value in result['gradient_conflicts'].values():
        if value['kind']=='shared':
            assert value['mean_training_objective_product_FS_training_gradient_span']['direction_count']==3


def test_stationarity_saved_projections_probes_and_H_units(stationarity_artifact):
    _,_,result,data,o=stationarity_artifact;worst=0.;projection_error=0.;orthogonality_error=0.;slope_error=0.
    for key,row in result['records'].items():
        p=data[key+'__state'];hp=data[key+'__hstate'];t=data[key+'__tangents'];h=o['h0']+row['coupling']*o['h2']
        np.testing.assert_allclose(hp,h@p,atol=1e-13,rtol=0)
        energy=np.vdot(p,hp).real;residual=hp-energy*p
        horizontal=t-p[:,None]*(p.conj()@t)[None,:]
        for label,indices in [('training_J_only',[0,2,4]),('all_five_J',list(range(5)))]:
            saved=row['subspaces'][label];projected=data[key+'__'+label+'__residual_projection']
            restricted=horizontal[:,indices]
            # Independently form the physical projection with U U.T, avoiding
            # cancellation when mapping an ill-conditioned span back to its
            # parameter coefficients. The sealed implementation uses T @ c;
            # test that representation against its floating-point condition
            # bound, rather than imposing a scale-dependent normal-equation
            # tolerance. The core well-conditioned exact tests remain strict.
            lengths=np.linalg.norm(restricted,axis=0);normalized=restricted/lengths
            design=np.concatenate((normalized.real,normalized.imag));target=np.r_[residual.real,residual.imag]
            u,s,vh=np.linalg.svd(design,full_matrices=False);rank=np.sum(s>1e-10*s[0])
            coordinate=u[:,:rank].T@target;real_projected=u[:,:rank]@coordinate
            stable=real_projected[:len(p)]+1j*real_projected[len(p):]
            coeff=(vh[:rank].T@(coordinate/s[:rank]))/lengths
            condition=s[0]/s[rank-1]
            assert rank==saved['rank'] and condition<1e7
            vector_bound=64*np.finfo(float).eps*(np.linalg.norm(restricted)*np.linalg.norm(coeff)+np.linalg.norm(residual))
            difference=np.linalg.norm(projected-stable)
            assert difference<=vector_bound
            assert np.linalg.norm(np.real(restricted.conj().T@(residual-stable)))<2e-10
            # The initial horizontal subtraction also has roundoff, which
            # near-dependent columns amplify when forming this span.
            assert abs(np.vdot(p,stable))<=vector_bound
            assert np.linalg.norm(stable)<=np.linalg.norm(residual)+1e-12
            # Unit-FS slope of the stored direction, evaluated without any
            # normal-equation assumption, must replay directly.
            direct_slope=-2*np.vdot(projected,residual).real/(16*np.linalg.norm(projected))
            assert abs(direct_slope-saved['unit_FS_descent_slope_per_site'])<=vector_bound
            orthogonality_error=max(orthogonality_error,np.linalg.norm(np.real(restricted.conj().T@(residual-projected))))
            projection_error=max(projection_error,difference)
            slope_error=max(slope_error,abs(-2*np.linalg.norm(stable)/16-saved['unit_FS_descent_slope_per_site']))
            trial=data[key+'__'+label+'__probe_state'];action=h@trial;et=np.vdot(trial,action).real
            var=np.linalg.norm(action-et*trial)**2/16
            worst=max(worst,abs(et/16-saved['probe_energy_per_site']),abs(var-saved['probe_variance_per_site']))
            phase=np.vdot(p,trial);fs=np.arctan2(np.linalg.norm(trial-p*phase),abs(phase))
            worst=max(worst,abs(fs-saved['physical_step']['distance']['fs']))
            assert saved['physical_step']['matched'] and saved['physical_step']['evaluation_count']==17
            if row['representation']=='singlet_vap':assert np.linalg.norm(o['s2']@trial)<2e-10
    assert worst<1e-12
    print('stationarity saved H/projection/FS replay maxerror',worst)
    print('conditioned T@coeff vs stable U projection: vector error',projection_error,
        'raw normal-equation defect',orthogonality_error,'slope error',slope_error)


def test_conditioning_capacity_from_saved_bias_Jacobians(stationarity_artifact):
    _,_,result,data,_=stationarity_artifact;worst=0.;count=0
    for key,row in result['records'].items():
        if 'conditioning_capacity' not in row:continue
        count+=1;p=data[key+'__state'];hp=data[key+'__hstate'];target=hp-np.vdot(p,hp).real*p
        derivative=data[key+'__coupling_derivative'];derivative-=p*np.vdot(p,derivative)
        jac=data[key+'__bias_jacobian'];jac=jac-p[:,None]*(p.conj()@jac)[None,:]
        report=row['conditioning_capacity'];real=np.concatenate((jac.real,jac.imag));r=np.concatenate((target.real,target.imag))
        u,s,_=np.linalg.svd(real,full_matrices=False);rank=np.sum(s>max(1e-12,1e-10*s[0]))
        fraction=np.linalg.norm(u[:,:rank].T@r)**2/np.linalg.norm(r)**2
        learned_fraction=np.vdot(derivative,target).real**2/(np.linalg.norm(derivative)**2*np.linalg.norm(target)**2)
        worst=max(worst,abs(fraction-report['all_patch_bias_tangents']['captured_squared_fraction']),
            abs(learned_fraction-report['learned_coupling_tangent']['captured_squared_fraction']))
        assert rank==report['all_patch_bias_tangents']['real_tangent_rank']<=15
        assert np.linalg.norm(jac@np.ones(16))<1e-11
        assert report['coupling_chain_rule_absolute_error']<2e-13
        assert 'full_backbone_tangents' not in report
    assert count==10 and worst<2e-12
    print('saved conditioning residual coverage replay maxerror',worst)


def test_saved_projector_secant_against_explicit_orbit_density_differences(stationarity_artifact):
    _,_,result,data,_=stationarity_artifact;couplings=result['protocol']['couplings'];worst=0.
    for name,saved in result['saved_state_projector_secants'].items():
        parent,j=name.rsplit('__',1);index=couplings.index(float(j));left,right=couplings[index-1],couplings[index+1]
        vectors=[]
        for shared in (True,False):
            for coupling in (left,right):
                kind='shared' if shared else f'point_{coupling:.2f}'
                vectors.append(data[parent+'__singlet_vap__'+kind+'__'+str(coupling)+'__state'])
        rho=[np.outer(p,p.conj()) for p in vectors];first=(rho[1]-rho[0])/(right-left);second=(rho[3]-rho[2])/(right-left)
        cosine=np.vdot(first,second).real/(np.linalg.norm(first)*np.linalg.norm(second))
        ratio=np.linalg.norm(first)/np.linalg.norm(second)
        worst=max(worst,abs(cosine-saved['projector_secant_cosine']),abs(ratio-saved['first_over_second_speed']))
    assert worst<1e-12
    print('saved explicit-projector secant replay maxerror',worst)
