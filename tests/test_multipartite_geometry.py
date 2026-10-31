"""Independent physical and convex-construction checks for regional GME."""
import numpy as np
import pytest

from su2_nqs.benchmarks.multipartite_geometry import (
    partial_trace_sites, induced_geometry, local_schur_channel,
    low_order_observables, lift_invariant_model, biseparable_moment_fit,
    ppt_mixture_moment_fit,
    triplet_screening_certificate,
    spin_measurement_branches, measurement_witness_pullback,
    solve_physical_multipartite_witness,
    covering_region_density,
    biseparable_residual_ball,
    localizable_block_negativity,
)
from su2_nqs.benchmarks.su2_maximum_entropy import invariant_constraint_space
from su2_nqs.benchmarks.spin_rotation_calibration import singlet_covering_state
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix


def test_partial_trace_complex_order_and_batch_against_pure_Schmidt():
    rng=np.random.default_rng(88); v=rng.normal(size=32)+1j*rng.normal(size=32);v/=np.linalg.norm(v)
    rho=np.outer(v,v.conj()); theta=v.reshape((2,)*5).transpose(3,0,4,1,2).reshape(8,4)
    expected=theta@theta.conj().T
    actual=partial_trace_sites(np.array([rho,rho]),(3,0,4))
    np.testing.assert_allclose(actual[0],expected,atol=2e-16)
    np.testing.assert_allclose(actual[1],expected,atol=2e-16)


def test_local_two_spin_charge_channel_matches_swap_pinching():
    rng=np.random.default_rng(31); v=rng.normal(size=16)+1j*rng.normal(size=16);v/=np.linalg.norm(v)
    rho=np.outer(v,v.conj()); pairs=((0,2),(1,3)); expected=rho.copy()
    for i,j in pairs:
        singlet=np.eye(16)/4-spin_dot_matrix(4,i,j); triplet=np.eye(16)-singlet
        expected=singlet@expected@singlet+triplet@expected@triplet
    for mode in ('charge','magnetic'):
        np.testing.assert_allclose(local_schur_channel(rho,pairs,mode),expected,atol=2e-15)


@pytest.mark.parametrize('mode',['charge','magnetic','multiplicity'])
def test_schur_channels_CPTP_idempotence_and_adjoint(mode):
    rng=np.random.default_rng(80); z=rng.normal(size=(64,64))+1j*rng.normal(size=(64,64))
    rho=z@z.conj().T;rho/=np.trace(rho); w=z+z.conj().T
    parties=((0,2,4),(1,3,5)); mapped=local_schur_channel(rho,parties,mode)
    assert abs(np.trace(mapped)-1)<2e-14
    assert np.linalg.eigvalsh(mapped).min()>-2e-14
    np.testing.assert_allclose(local_schur_channel(mapped,parties,mode),mapped,atol=2e-14)
    np.testing.assert_allclose(np.trace(w@mapped),np.trace(local_schur_channel(w,parties,mode)@rho),atol=2e-13)
    theta=.731;single=np.cos(theta/2)*np.eye(2)-1j*np.sin(theta/2)*np.array([[0,1],[1,0]])
    rotation=np.ones((1,1),complex)
    for _ in range(6):rotation=np.kron(rotation,single)
    np.testing.assert_allclose(local_schur_channel(rotation@rho@rotation.conj().T,parties,mode),
                               rotation@mapped@rotation.conj().T,atol=2e-14)


def test_geometry_distinguishes_J2_chords_and_open_chain():
    bonds=[(0,1,1),(1,2,1),(2,3,1),(3,0,1),(0,2,.125)]
    g=induced_geometry((0,1,2,3),bonds,.125)
    assert g['J1']['cycle_rank']==1 and g['active']['cycle_rank']==2
    g=induced_geometry((0,1,2),[(0,1,1),(1,2,1)],0)
    assert g['active']['cycle_rank']==0


def test_low_moment_lift_closes_every_scalar_without_imposing_input_symmetry():
    rng=np.random.default_rng(52);z=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
    rho=z@z.conj().T;rho/=np.trace(rho);space=invariant_constraint_space(4)
    reconstructed=lift_invariant_model(space.data(rho),4);ops=low_order_observables(4)
    np.testing.assert_allclose(np.einsum('kij,ji->k',ops,reconstructed-rho),0,atol=2e-14)


def test_constructive_biseparable_fit_replays_dimer_moments():
    v=singlet_covering_state(4,((0,1),(2,3)));rho=np.outer(v,v.conj());ops=low_order_observables(4)
    report,saved=biseparable_moment_fit(np.einsum('kij,ji->k',ops,rho).real,
        ops,((0,),(1,),(2,),(3,)),iterations=35,tolerance=2e-7)
    assert report['compatible_to_declared_tolerance'],report
    assert report['product_factor_error']<2e-14
    assert np.all(saved['weights']>=0)
    np.testing.assert_allclose(saved['rho'],np.einsum('a,ai,aj->ij',saved['weights'],saved['vectors'],saved['vectors']),atol=1e-15)


def test_PPT_relaxation_does_not_call_a_candidate_biseparable():
    pytest.importorskip('cvxpy');ops=low_order_observables(3);rho=np.eye(8)/8
    report,_=ppt_mixture_moment_fit(np.einsum('kij,ji->k',ops,rho).real,ops,((0,),(1,),(2,)),seconds=5)
    assert report['numerical_PPT_candidate']
    assert not report['biseparable_proved'] and not report['infeasibility_certified']


def test_product_completion_cannot_fake_the_singlet_pair_bound():
    v=singlet_covering_state(2,((0,1),));rho=np.outer(v,v.conj());ops=low_order_observables(2)
    report,_=biseparable_moment_fit(np.einsum('kij,ji->k',ops,rho).real,
                                  ops,((0,),(1,)),iterations=4)
    # A separable pair has singlet probability <= 1/2, independent of solver.
    assert report['moment_residual_max']>=.5-2e-14
    assert not report['compatible_to_declared_tolerance']
    assert report['failed_search_excludes_nothing']


def test_LP_timeout_returns_a_valid_candidate_without_false_compatibility(monkeypatch):
    from types import SimpleNamespace
    import su2_nqs.benchmarks.multipartite_geometry as module
    monkeypatch.setattr(module,'linprog',lambda *a,**k:SimpleNamespace(success=False,message='Time limit'))
    v=singlet_covering_state(2,((0,1),));ops=low_order_observables(2)
    report,saved=biseparable_moment_fit(np.einsum('kij,i,j->k',ops,v,v.conj()).real,
                                      ops,((0,),(1,)),iterations=2)
    assert not report['compatible_to_declared_tolerance']
    assert report['last_LP_status']=='Time limit'
    np.testing.assert_allclose(saved['rho'],np.eye(4)/4,atol=1e-15)


def test_local_pair_singlet_outcome_is_biseparable_without_total_spin_filter():
    rng=np.random.default_rng(841);v=rng.normal(size=64)+1j*rng.normal(size=64);v/=np.linalg.norm(v)
    rho=np.outer(v,v.conj());p=np.eye(64)/4-spin_dot_matrix(6,0,3)
    outcome=p@rho@p
    order=(0,3,1,4,2,5)
    grouped=outcome.reshape((2,)*12).transpose(order+tuple(6+i for i in order)).reshape(4,16,4,16)
    remaining=np.trace(grouped,axis1=0,axis2=2)
    singlet=np.array([0,1,-1,0])/np.sqrt(2)
    expected=np.kron(np.outer(singlet,singlet),remaining)
    np.testing.assert_allclose(grouped.reshape(64,64),expected,atol=2e-16)


def test_analytic_three_triplet_screening_witness_saturates_pure_state_bound():
    cert,ops=triplet_screening_certificate(((0,3),(1,4),(2,5)))
    ev,u=np.linalg.eigh(ops['P222_total0']);v=u[:,-1]
    assert ev[-1]==pytest.approx(1,abs=2e-14)
    for cut in cert.cuts:
        order=cut+tuple(i for i in range(6) if i not in cut)
        s=np.linalg.svd(v.reshape((2,)*6).transpose(order).reshape(2**len(cut),-1),compute_uv=False)**2
        np.testing.assert_allclose(s[:3],[1/3]*3,atol=2e-14)
        assert np.max(s[3:],initial=0)<2e-14
    assert -np.trace(ops['P222_total0']@cert.witness).real==pytest.approx(1,abs=2e-8)
    assert np.linalg.norm(local_schur_channel(cert.witness,((0,3),(1,4),(2,5)),'charge')-cert.witness)<2e-13


@pytest.mark.parametrize('axis',['x','y','z'])
def test_measurement_matches_independent_complex_wavefunction_and_adjoint(axis):
    rng=np.random.default_rng(907);v=rng.normal(size=16)+1j*rng.normal(size=16);v/=np.linalg.norm(v)
    rho=np.outer(v,v.conj());sigma={'x':np.array([[0,1],[1,0]]),
        'y':np.array([[0,1j],[-1j,0]]),'z':np.diag([-1.,1.])}[axis]
    _,vectors=np.linalg.eigh(sigma)
    for site in range(4):
        branches,prob=spin_measurement_branches(np.array([rho,rho]),site,axis)
        order=tuple(i for i in range(4) if i!=site)+(site,)
        theta=v.reshape((2,)*4).transpose(order).reshape(8,2)
        for k,m in enumerate((-1,1)):
            child=theta@vectors[:,k].conj()
            expected=np.outer(child,child.conj())
            np.testing.assert_allclose(branches[0,k],expected,atol=2e-16)
            w=np.diag(np.arange(8))
            pulled=measurement_witness_pullback(w,site,axis,m)
            np.testing.assert_allclose(np.trace(pulled@rho),np.trace(w@expected),atol=2e-15)
        np.testing.assert_allclose(branches.sum(axis=-3),partial_trace_sites(np.array([rho,rho]),order[:-1]),atol=2e-16)
        np.testing.assert_allclose(prob.sum(axis=-1),1,atol=2e-16)


def test_GHZ_measurement_vs_discard_and_all_outcome_probabilities():
    v=np.zeros(16);v[0]=v[-1]=1/np.sqrt(2);rho=np.outer(v,v)
    discard=partial_trace_sites(rho,(1,2,3))
    x,p=spin_measurement_branches(rho,0,'x');z,pz=spin_measurement_branches(rho,0,'z')
    np.testing.assert_allclose(p,[.5,.5],atol=2e-16)
    np.testing.assert_allclose(pz,[.5,.5],atol=2e-16)
    assert abs(x[0,0,7])==pytest.approx(.25)
    assert abs(x[1,0,7])==pytest.approx(.25)
    assert np.max(abs(z[:,0,7]))==0 and discard[0,7]==0


def test_unrestricted_physical_witness_detects_nonSU2_GHZ_child():
    pytest.importorskip('cvxpy')
    v=np.zeros(8);v[0]=v[-1]=1/np.sqrt(2);rho=np.outer(v,v)
    report,cert=solve_physical_multipartite_witness(rho,((0,),(1,),(2,)),seconds=5,real_operators=True)
    assert report['entanglement_lower_bound']==pytest.approx(.5,abs=2e-4)
    assert not report['optimality_certified'] and not report['state_repaired']
    from su2_nqs.benchmarks.su2_multipartite_witness import validate_witness_certificate
    validate_witness_certificate(cert)


def test_measurement_labels_and_sites_reject_invalid_inputs():
    for site,axis in [(True,'z'),(-1,'x'),(2,'z'),(0,'other')]:
        with pytest.raises(ValueError):spin_measurement_branches(np.eye(4)/4,site,axis)
    for m in [0,.5,True]:
        with pytest.raises(ValueError):measurement_witness_pullback(np.eye(4),0,'z',m)


def test_large_covering_RDM_matches_exact_small_parent_with_cut_dimers():
    from su2_nqs.benchmarks.spin_rotation_calibration import pure_state_rdm
    pairs=((0,3),(1,6),(2,5),(4,7));v=singlet_covering_state(8,pairs)
    for sites in [(0,1,2,3),(7,4,0,3),(3,0,1,6,2,4)]:
        np.testing.assert_allclose(covering_region_density(8,pairs,sites),pure_state_rdm(v,sites),atol=2e-16)


def test_full_RDM_residual_ball_closes_explicit_mixture_without_state_repair():
    v=np.zeros(16);v[0]=1;eta=np.outer(v,v);t=1.1
    rho=eta/t+(1-1/t)*np.eye(16)/16;before=rho.copy()
    report,arrays=biseparable_residual_ball(rho,np.array([1.]),v[None,:],np.array([1]),
                                          ((0,),(1,),(2,),(3,)),inflation=t)
    assert report['numerically_certified_biseparable']
    assert report['residual_ball_radius']<1e-13
    np.testing.assert_array_equal(rho,before)
    np.testing.assert_allclose(arrays['tau'],np.eye(16)/16,atol=2e-16)
    report,_=biseparable_residual_ball(rho,np.array([1.]),v[None,:],np.array([1]),
        ((0,),(1,),(2,),(3,)),inflation=t,shared_su2_average=True)
    # The certificate is rechecked against the FULL original rho, not its SU2 average.
    assert not report['numerically_certified_biseparable']


def test_residual_ball_does_not_certify_an_entangled_target_or_entangled_atom():
    v=np.array([0,1,-1,0])/np.sqrt(2);rho=np.outer(v,v)
    report,_=biseparable_residual_ball(rho,np.ones(4)/4,np.eye(4),np.ones(4,dtype=int),
                                      ((0,),(1,)))
    assert not report['numerically_certified_biseparable']
    with pytest.raises(ValueError):
        biseparable_residual_ball(rho,np.array([1.]),v[None,:],np.array([1]),((0,),(1,)))


def test_localizable_GHZ_entanglement_retains_same_charge_coherence():
    v=np.zeros(32);v[0]=v[-1]=1/np.sqrt(2)
    report=localizable_block_negativity(np.outer(v,v),2,(0,1))
    assert report['unassisted_negativity']==pytest.approx(0,abs=1e-15)
    assert report['axes']['x']['assisted_negativity']==pytest.approx(.5,abs=1e-15)
    assert report['axes']['x']['after_local_charge_negativity']==pytest.approx(.5,abs=1e-15)
    assert report['axes']['z']['assisted_negativity']==pytest.approx(0,abs=1e-15)
    assert sum(x['probability'] for x in report['axes']['x']['outcomes'])==pytest.approx(1)


def test_fixed_covering_has_no_assistance_gain_and_probability_closure():
    from su2_nqs.benchmarks.spin_rotation_calibration import pure_state_rdm
    v=singlet_covering_state(6,((0,1),(2,3),(4,5)))
    rho=pure_state_rdm(v,(0,1,2,4,5));report=localizable_block_negativity(rho,1,(0,3))
    for row in report['axes'].values():
        assert row['assistance_gain']==pytest.approx(0,abs=1e-15)
        assert row['charge_coherence_loss']>=-1e-15
        for outcome in row['outcomes']:
            assert sum(x['probability'] for x in outcome['local_q_sectors'].values())==pytest.approx(outcome['probability'])


def test_assistance_axis_independence_on_an_actual_SU2_parent():
    from su2_nqs.benchmarks.spin_rotation_calibration import pure_state_rdm,lieb_mattis_singlet
    rho=pure_state_rdm(lieb_mattis_singlet(12,(0,2,4,6,8,10)),(0,1,2,3,4))
    report=localizable_block_negativity(rho,2,(0,1),('x','y','z'))
    for key in ['assisted_negativity','after_local_charge_negativity','assistance_gain']:
        np.testing.assert_allclose([report['axes'][axis][key] for axis in ('x','y','z')],report['axes']['z'][key],atol=2e-15)
