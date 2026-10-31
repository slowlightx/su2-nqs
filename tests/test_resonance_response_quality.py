"""Independent finite-source checks for a regional, not Hamiltonian, response."""
import numpy as np
import json
import hashlib
import pytest
from scipy.linalg import expm

from projects.clebsch_tree_nqs.workflows.spin_resonance_response_quality import filter_response, singlet_ring_operator
from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split, spin_dot_matrix


def _series_fixture(tmp_path):
    """A hashed synthetic physical RDM, not a fake optimized quantum state."""
    from su2_nqs.frontier.artifacts import canonical_sha256
    from projects.clebsch_tree_nqs.workflows.spin_su2_analysis import _problem_signature, _regional_scalar_data
    problem=dict(geometry='triangular',rows=6,columns=6,j2=.125,
        boundary='PBC_both_physical',regions={'compact6':[12,13,14,18,19,20]},
        left_sites={'compact6':[0,1,2]})
    rho=np.eye(64)/64; raw=tmp_path/'raw.npz'
    np.savez(raw,compact6__rho=rho,compact6__jackknife_rhos=np.stack([rho]*4),chain_ids=np.arange(4))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def sealed(name,row):
        row=dict(row);row['result_sha256']=canonical_sha256(row)
        p=tmp_path/name;p.write_text(json.dumps(row));return row
    common=dict(params_sha256='frozen-parameters',checkpoint_sha256='frozen-checkpoint')
    reg=sealed('registration.json',dict(config=dict(problem=problem),**common))
    sealed('audit.json',dict(step=1,energy=dict(energy_per_site=0.),**common))
    _,_,scalars=_regional_scalar_data(rho,dict(sites=problem['regions']['compact6'],left=[0,1,2]))
    sealed('regional.json',dict(step=1,registration_sha256=reg['result_sha256'],
        physical_problem_signature=_problem_signature(problem)[0],independent_chains=4,
        raw_sha256=sha(raw),regions={'compact6':dict(sites=problem['regions']['compact6'],
            left=[0,1,2],scalars=scalars,scalar_whole_chain_SE={k:0. for k in scalars})},**common))
    spec=dict(output=str(tmp_path/'output'),seconds_cap=60,inputs_sha256={
        str(p):sha(p) for p in tmp_path.iterdir() if p.is_file()},sources=[dict(
            kind='NQS',label='synthetic',step=1,problem=problem,
            audit=str(tmp_path/'audit.json'),registration=str(tmp_path/'registration.json'),
            regional=str(tmp_path/'regional.json'),raw=str(raw))])
    return spec


def test_registered_series_keeps_seals_and_never_reports_strict_spectrum(tmp_path):
    from projects.clebsch_tree_nqs.workflows.spin_resonance_response_quality import registered_quality_series
    spec=_series_fixture(tmp_path);config=tmp_path/'config.json';config.write_text(json.dumps(spec))
    result=registered_quality_series(config)
    assert result['new_H_calls']==result['new_training']==result['new_samples']==0
    assert not result['strict_entanglement_spectrum'] and not result['phase_claim']
    assert result['records'][0]['scalars']['p_q0']==pytest.approx(5/64)
    assert not result['records'][0]['strict_entanglement_spectrum']
    assert registered_quality_series(config)==result
    (tmp_path/'output/results.json').write_text('{}')
    with pytest.raises(ValueError,match='completed series changed'):registered_quality_series(config)


@pytest.mark.parametrize('change', ['hash','seal','parent','duplicate'])
def test_registered_series_rejects_changed_or_misaligned_sources(tmp_path,change):
    from projects.clebsch_tree_nqs.workflows.spin_resonance_response_quality import registered_quality_series
    from su2_nqs.frontier.artifacts import canonical_sha256
    spec=_series_fixture(tmp_path);audit=tmp_path/'audit.json'
    if change=='duplicate':spec['sources']*=2
    else:
        row=json.loads(audit.read_text());row['params_sha256']='another-state'
        if change=='parent':row['result_sha256']=canonical_sha256({k:v for k,v in row.items() if k!='result_sha256'})
        audit.write_text(json.dumps(row))
        if change!='hash':spec['inputs_sha256'][str(audit)]=hashlib.sha256(audit.read_bytes()).hexdigest()
    config=tmp_path/'config.json';config.write_text(json.dumps(spec))
    with pytest.raises(ValueError):registered_quality_series(config)


def test_filter_convention_and_noncommuting_cross_response():
    rng=np.random.default_rng(991)
    v=rng.normal(size=(4,3))+1j*rng.normal(size=(4,3))
    rho=v@v.conj().T;rho/=np.trace(rho)
    op=np.diag([-.7,-.1,.2,.8])
    b=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));b=(b+b.conj().T)/2
    point=filter_response(rho,op,{'b':b})
    h=1e-5
    means=[]
    for eta in (-h,h):
        f=expm(eta*op/2);r=f@rho@f;r/=np.trace(r)
        means.append(np.trace(r@b).real)
    np.testing.assert_allclose((means[1]-means[0])/(2*h),point['b__filter_slope'],atol=2e-10)
    np.testing.assert_allclose(point['source_variance'],np.trace(rho@op@op).real-np.trace(rho@op).real**2,atol=1e-14)


def test_frozen_resonance_operator_is_scalar_and_three_bond_remainder():
    split=joint_spin_operator_split(6,[0,1,2],[1,1,0])
    o=split.high
    sz=np.diag([bin(i).count('1')-3 for i in range(64)])
    raising=np.zeros((64,64))
    for i in range(64):
        for k in range(6):
            if not i&(1<<k):raising[i|(1<<k),i]=1
    assert np.linalg.norm(o@sz-sz@o)<1e-12
    assert np.linalg.norm(o@raising-raising@o)<1e-12
    x=np.zeros((64,64))
    for a in range(3):
        for b in range(3,6):
            left=[i for i in range(3) if i!=a];right=[i for i in range(3,6) if i!=b]
            x+=spin_dot_matrix(6,*left)@spin_dot_matrix(6,*right)@spin_dot_matrix(6,a,b)
    np.testing.assert_allclose(o,-4*x/9,atol=1e-13)


def test_ring_exchange_has_nontrivial_singlet_multiplicity_action():
    from su2_nqs.benchmarks.spin_partition_readout import _joint_projectors
    r, diagnostics=singlet_ring_operator()
    assert diagnostics['cyclic_order_error']<1e-14
    assert diagnostics['singlet_trace_error']<1e-12
    blocks=dict(_joint_projectors(6,(0,1,2)))
    rebuilt=sum(np.trace(r@p)/np.trace(p)*p for p in blocks.values())
    assert np.linalg.norm(r-rebuilt)>0.5
    p0=sum(p for (_,_,qt),p in blocks.items() if qt==0)
    np.testing.assert_allclose(p0@r@p0,r,atol=1e-13)
    raising=np.zeros((64,64))
    for i in range(64):
        for k in range(6):
            if not i&(1<<k):raising[i|(1<<k),i]=1
    assert np.linalg.norm(r@raising-raising@r)<1e-12


def test_ring_channels_resolve_singlet_and_sequential_screening_is_positive():
    from projects.clebsch_tree_nqs.workflows.spin_resonance_response_quality import ring_channel_projectors,ring_channel_readout
    from su2_nqs.benchmarks.spin_partition_readout import _three_plus_three_singlet_columns
    projectors=ring_channel_projectors();columns=_three_plus_three_singlet_columns((0,1,2))
    p0=columns@columns.conj().T
    np.testing.assert_allclose(sum(projectors.values()),p0,atol=1e-13)
    np.testing.assert_allclose(-projectors['minus_one']+.5*projectors['half']+projectors['one'],singlet_ring_operator()[0],atol=1e-13)
    rng=np.random.default_rng(1701);v=rng.normal(size=5)+1j*rng.normal(size=5);v/=np.linalg.norm(v)
    psi=columns@v;rho=np.outer(psi,psi.conj());out=ring_channel_readout(rho)
    assert abs(out['p_regional_q0']-1)<1e-13
    assert abs(sum(out[k+'_given_q0'] for k in projectors)-1)<1e-13
    for k in projectors:
        assert -1e-13<=out[k+'_quartet_sequential_weight']<=out[k+'_weight']+1e-13
        assert -1e-13<=out[k+'_quartet_given_channel']<=1+1e-13


def test_ring_maxent_uncertainty_refits_the_comparator(tmp_path):
    from projects.clebsch_tree_nqs.workflows.spin_resonance_response_quality import ring_maxent_refits
    from su2_nqs.benchmarks.su2_maximum_entropy import invariant_constraint_space,fit_invariant_maximum_entropy
    rng=np.random.default_rng(1702)
    raw=[]
    for _ in range(4):
        v=rng.normal(size=(64,12));a=v@v.T;a/=np.trace(a)
        raw.append(.8*np.eye(64)/64+.2*a)
    rho=np.mean(raw,axis=0);jack=np.stack([np.mean([a for j,a in enumerate(raw) if j!=i],axis=0) for i in range(4)])
    cache=tmp_path/'fits.json';report=ring_maxent_refits(rho,jack,cache)
    space=invariant_constraint_space(6);r=singlet_ring_operator()[0]
    independently=[]
    for a in jack:
        _,model,_=fit_invariant_maximum_entropy(space.data(a),space)
        lift=np.zeros((64,64),complex)
        for q,sl,us in zip(space.qs,space.slices,space.magnetic_columns):
            lift+=sum(u@model[sl,sl]@u.conj().T for u in us)/(q+1)
        independently.append(np.trace(a@r).real-np.trace(lift@r).real)
    independently=np.array(independently)
    expected=np.sqrt(3/4*np.sum((independently-independently.mean())**2))
    np.testing.assert_allclose(report['whole_chain_refit_SE']['excess_mean'],expected,atol=2e-12)
    replay=ring_maxent_refits(rho,jack,cache)
    assert replay['point']==report['point']
    assert report['full_refit_count']==5


def test_center_ratio_reconstruction_and_full_h_filter_tangent():
    rng=np.random.default_rng(710)
    psi=rng.normal(size=64)+1j*rng.normal(size=64);psi/=np.linalg.norm(psi)
    r,_=singlet_ring_operator()
    h=sum(spin_dot_matrix(6,i,(i+1)%6) for i in range(6))
    gram=np.outer(psi,psi.conj())
    ratios=np.array([r[a]@gram[:,a]/gram[a,a] for a in range(64)])
    np.testing.assert_allclose(ratios,r@psi/psi,atol=2e-13)
    energies=h@psi/psi
    energy=np.vdot(psi,h@psi).real
    slope=np.sum(abs(psi)**2*(energies-energy).conj()*ratios).real
    delta=1e-5
    finite=[]
    for eta in (-delta,delta):
        p=expm(eta*r/2)@psi;p/=np.linalg.norm(p)
        finite.append(np.vdot(p,h@p).real)
    np.testing.assert_allclose(slope,(finite[1]-finite[0])/(2*delta),atol=2e-10)


def test_conditional_gram_full_h_response_includes_boundary_bonds():
    """A six-site conditional estimator must retain its seventh-site boundary.

    The full H does not commute with the local ring filter. In particular,
    multiplying a conditional mean of R by the center energy is not valid.
    """
    rng = np.random.default_rng(711)
    psi = rng.normal(size=128) + 1j*rng.normal(size=128)
    psi /= np.linalg.norm(psi)
    local_r, _ = singlet_ring_operator()
    r = np.kron(local_r, np.eye(2))
    # Independent physical-basis oracle: S_i.S_j = P_ij/2 - I/4.
    # The production regional helper deliberately caps its size at six.
    states = np.arange(128)
    identity = np.eye(128)
    h = np.zeros((128, 128))
    for i in range(7):
        j = (i+1) % 7
        different = ((states >> (6-i)) ^ (states >> (6-j))) & 1
        image = states ^ (different*((1 << (6-i)) | (1 << (6-j))))
        h += identity[image]/2 - identity/4
    assert np.linalg.norm(h@r-r@h) > 0.1
    wave = psi.reshape(64, 2)
    ratios = np.empty_like(wave)
    conditional_means = np.empty_like(wave)
    for environment in range(2):
        branch = wave[:, environment]
        gram = np.outer(branch, branch.conj()) / np.vdot(branch, branch).real
        for center in range(64):
            ratios[center, environment] = (
                local_r[center]@gram[:, center] / gram[center, center])
        conditional_means[:, environment] = np.trace(gram@local_r)
    np.testing.assert_allclose(ratios.reshape(-1), r@psi/psi, atol=2e-13)
    energies = h@psi/psi
    energy = np.vdot(psi, h@psi).real
    weighted = abs(psi)**2 * (energies-energy).conj()
    slope = np.sum(weighted*ratios.reshape(-1)).real
    wrong_slope = np.sum(weighted*conditional_means.reshape(-1)).real
    assert abs(slope-wrong_slope) > 1e-3
    delta = 1e-5
    finite = []
    for eta in (-delta, delta):
        p = expm(eta*r/2)@psi
        p /= np.linalg.norm(p)
        finite.append(np.vdot(p, h@p).real)
    np.testing.assert_allclose(slope, (finite[1]-finite[0])/(2*delta), atol=2e-10)


def test_spatial_average_has_same_first_energy_response_but_different_finite_filter():
    rng=np.random.default_rng(714)
    bits=(np.arange(128)[:,None]>>np.arange(6,-1,-1))&1
    image=np.roll(bits,1,axis=1)@(2**np.arange(6,-1,-1))
    t=np.zeros((128,128));t[image,np.arange(128)]=1
    powers=[np.linalg.matrix_power(t,k) for k in range(7)]
    psi=rng.normal(size=128)+1j*rng.normal(size=128)
    psi=sum(u@psi for u in powers);psi/=np.linalg.norm(psi)
    r=np.kron(singlet_ring_operator()[0],np.eye(2))
    rbar=sum(u@r@u.T for u in powers)/7
    identity=np.eye(128);h=np.zeros((128,128));states=np.arange(128)
    for i in range(7):
        j=(i+1)%7;different=((states>>(6-i))^(states>>(6-j)))&1
        swap=states^(different*((1<<(6-i))|(1<<(6-j))))
        h+=identity[swap]/2-identity/4
    e=np.vdot(psi,h@psi).real
    slope=lambda op:np.vdot((h-e*identity)@psi,op@psi).real
    np.testing.assert_allclose(slope(r),slope(rbar),atol=1e-13)
    a,b=expm(.2*r/2)@psi,expm(.2*rbar/2)@psi
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    assert np.linalg.norm(a-b)>1e-3
    np.testing.assert_allclose(t@b,b,atol=1e-13)
