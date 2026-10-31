"""Exact physical controls for invariant-algebra maximum entropy, not VMC."""
import numpy as np
import pytest

from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.benchmarks.su2_maximum_entropy import (
    FixedWeightExponentialFamily, fit_invariant_maximum_entropy,
    heldout_log_score_operator, invariant_constraint_space, matrix_entropy,
    positive_matrix_log,
    audit_invariant_chains,
)


def test_constraint_rank_and_uniform_gauge():
    for n,rank,missing in [(2,0,0),(4,7,0),(6,58,14)]:
        space=invariant_constraint_space(n)
        assert len(space.operators)==rank
        assert space.diagnostics['unfixed_TR_even_directions']==missing
        a=space.operators
        np.testing.assert_allclose(np.einsum('kij,lji->kl',a,a),np.eye(rank),atol=3e-14)
        for s in space.slices:
            np.testing.assert_allclose(np.trace(a[:,s,s],axis1=1,axis2=2),0,atol=2e-14)


def test_independent_six_body_perturbation_with_identical_low_moments_and_p():
    b=lambda i,j:spin_dot_matrix(6,i,j)
    remainder=(b(0,1)@b(2,3)-b(0,2)@b(1,3))@b(4,5)
    rho=np.eye(64)/64+.01*remainder
    assert np.linalg.eigvalsh(rho).min()>0
    space=invariant_constraint_space(6)
    target,uniform=space.data(rho),space.data(np.eye(64)/64)
    for s in space.slices:
        assert abs(np.trace(target[s,s])-np.trace(uniform[s,s]))<2e-14
    np.testing.assert_allclose(np.einsum('kij,ji->k',space.operators,target-uniform),0,atol=2e-14)
    report,model,_=fit_invariant_maximum_entropy(target,space)
    assert report['certified_moment_fit']
    np.testing.assert_allclose(model,uniform,atol=2e-13)
    assert report['information_TR_even']>1e-5
    assert report['information_TR_even']==pytest.approx(matrix_entropy(uniform)-matrix_entropy(target),abs=2e-13)
    assert abs(report['information_TR_odd'])<2e-14


def test_noncommuting_gibbs_state_reconstructed_without_new_information():
    space=invariant_constraint_space(6)
    h=.9*spin_dot_matrix(6,0,1)-.4*spin_dot_matrix(6,1,2)
    h+=.3*spin_dot_matrix(6,0,3)@spin_dot_matrix(6,2,5)
    ev,u=np.linalg.eigh(h)
    rho=(u*np.exp(-ev))@u.T;rho/=np.trace(rho)
    target=space.data(rho)
    report,model,log_model=fit_invariant_maximum_entropy(target,space)
    assert report['certified_moment_fit']
    assert abs(report['information_total'])<1e-10
    np.testing.assert_allclose(model,target,atol=2e-9)
    assert np.linalg.norm(heldout_log_score_operator(target,log_model))<1e-6


def test_four_site_saturation_and_TR_odd_split():
    rng=np.random.default_rng(190)
    space=invariant_constraint_space(4)
    z=rng.normal(size=(16,20))+1j*rng.normal(size=(16,20))
    rho=z@z.conj().T;rho/=np.trace(rho)
    target=space.data(rho)
    report,model,_=fit_invariant_maximum_entropy(target,space)
    assert report['certified_moment_fit']
    np.testing.assert_allclose(model,target.real,atol=2e-9)
    assert abs(report['information_TR_even'])<1e-10
    assert report['information_TR_odd']>0
    assert report['information_total']==pytest.approx(report['information_TR_odd'],abs=1e-10)


def test_kubo_mori_gradient_hessian_against_finite_differences():
    space=invariant_constraint_space(4)
    family=FixedWeightExponentialFamily(space,space.data(np.eye(16)/16))
    theta=np.random.default_rng(10).normal(size=len(space.operators))*.2
    f,g,h,_,_=family.evaluate(theta)
    delta=2e-5
    numeric_g=[];numeric_h=[]
    for unit in np.eye(len(theta)):
        plus=family.evaluate(theta+delta*unit);minus=family.evaluate(theta-delta*unit)
        numeric_g.append((plus[0]-minus[0])/(2*delta))
        numeric_h.append((plus[1]-minus[1])/(2*delta))
    np.testing.assert_allclose(g,numeric_g,atol=2e-10)
    np.testing.assert_allclose(h,np.array(numeric_h).T,atol=2e-10)


def test_failure_budget_and_signed_heldout_score():
    space=invariant_constraint_space(4)
    family=FixedWeightExponentialFamily(space,space.data(np.eye(16)/16))
    _,_,_,train,log_train=family.evaluate(np.ones(len(space.operators))*1.5)
    report,_,_=fit_invariant_maximum_entropy(train,space,max_iterations=1)
    assert not report['certified_moment_fit']
    uniform=space.data(np.eye(16)/16)
    score=heldout_log_score_operator(train,positive_matrix_log(uniform))
    # Independent Gibbs inequality: fitting noise can make held-out score negative.
    assert np.trace(uniform@score).real<0
    with pytest.raises(ValueError,match='no flooring'):
        positive_matrix_log(np.diag([1.,0.]))
    with pytest.raises(ValueError,match='no repair'):
        space.data(np.eye(16))
    with pytest.raises(ValueError,match='integer'):
        invariant_constraint_space(6.)


def test_moment_refinement_resolves_objective_precision_stop_without_relaxation(monkeypatch):
    from types import SimpleNamespace
    import su2_nqs.benchmarks.su2_maximum_entropy as module
    space=invariant_constraint_space(4)
    family=FixedWeightExponentialFamily(space,space.data(np.eye(16)/16))
    theta=np.linspace(-.3,.4,len(space.operators))
    _,_,_,target,_=family.evaluate(theta)
    # Independent known Gibbs solution; mimic a trust-exact precision stop.
    stopped=theta+np.linspace(1e-7,3e-7,len(theta))
    monkeypatch.setattr(module,'minimize',lambda *a,**k:SimpleNamespace(
        x=stopped,nit=8,success=False,message='objective precision stop'))
    report,model,_=fit_invariant_maximum_entropy(target,space,max_iterations=10)
    assert report['moment_refinement_iterations']>0
    assert report['iterations']<=10 and report['moment_tolerance']==2e-9
    assert report['certified_moment_fit'] and not report['optimizer_success']
    np.testing.assert_allclose(model,target,atol=3e-13,rtol=0)
    assert abs(report['relative_entropy_gap_closure'])<1e-12


def test_refinement_also_checks_entropy_closure_when_moments_already_pass(monkeypatch):
    from types import SimpleNamespace
    import su2_nqs.benchmarks.su2_maximum_entropy as module
    space=invariant_constraint_space(4)
    family=FixedWeightExponentialFamily(space,space.data(np.eye(16)/16))
    theta=np.full(len(space.operators),2.)
    _,_,hessian,target,_=family.evaluate(theta)
    stopped=theta+np.linalg.solve(hessian,np.full(len(theta),1.8e-9))
    target_family=FixedWeightExponentialFamily(space,target)
    _,gradient,_,_,_=target_family.evaluate(stopped)
    assert abs(gradient).max()<2e-9
    assert abs(stopped@gradient)>1e-8
    monkeypatch.setattr(module,'minimize',lambda *a,**k:SimpleNamespace(
        x=stopped,nit=8,success=False,message='objective precision stop'))
    report,model,_=fit_invariant_maximum_entropy(target,space,max_iterations=10)
    assert report['moment_refinement_iterations']>0 and report['certified_moment_fit']
    assert abs(report['relative_entropy_gap_closure'])<1e-12
    np.testing.assert_allclose(model,target,atol=3e-13,rtol=0)


def test_whole_chain_refits_match_independent_stratified_jackknife():
    rng=np.random.default_rng(531)
    space=invariant_constraint_space(6)
    chains=[]
    for _ in range(8):
        z=rng.normal(size=(64,70))
        rho=z@z.T;rho/=np.trace(rho)
        chains.append(space.data(rho))
    chains=np.array(chains);folds=np.repeat([0,1],4)
    audit,arrays=audit_invariant_chains(chains,folds,space)
    def independent_score(left,right):
        scores=[]
        for train,test in [(left,right),(right,left)]:
            mean=train.mean(0)
            _,_,log_me=fit_invariant_maximum_entropy(mean,space)
            op=positive_matrix_log(mean.real)-log_me
            scores.append(np.trace(test.mean(0)@op).real)
        return np.mean(scores)
    point=independent_score(chains[:4],chains[4:])
    deleted=[];variance=0
    for g in (0,1):
        group=[]
        for i in range(4):
            parts=[chains[:4],chains[4:]]
            parts[g]=np.delete(parts[g],i,axis=0)
            group.append(independent_score(*parts))
        group=np.array(group);variance+=3/4*np.sum((group-group.mean())**2)
        deleted.extend(group)
    assert audit['point']['heldout_TR_even']==pytest.approx(point,abs=2e-13)
    assert audit['stratified_whole_chain_refit_SE']['heldout_TR_even']==pytest.approx(np.sqrt(variance),abs=2e-13)
    np.testing.assert_allclose(arrays['jackknife_values'][:,3],deleted,atol=2e-13)
    assert audit['full_refit_count']==27


def test_saved_full_refits_and_independent_stream_scores_replay():
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    folder=root/'projects/clebsch_tree_nqs/results/spin_spatial_vit_gpu_continuation_6600_v1/analysis/invariant_maximum_entropy_v1'
    if not folder.exists():
        pytest.skip('optional completed readout archive not present')
    result=json.loads((folder/'results.json').read_text())
    done=json.loads((folder/'completion.json').read_text())
    assert hashlib.sha256((folder/'results.json').read_bytes()).hexdigest()==done['results_sha256']
    assert hashlib.sha256((folder/'readouts.npz').read_bytes()).hexdigest()==done['readouts_sha256']
    space=invariant_constraint_space(6)
    with np.load(folder/'readouts.npz') as data:
        for row in result['records']:
            if row['method']!='ViT':
                continue
            prefix='ViT__'+row['state_id']+'__'+row['cut']+'__'
            omega,model=data[prefix+'omega'],data[prefix+'maxent']
            residual=np.einsum('kij,ji->k',space.operators,omega-model)
            assert abs(residual).max()<2e-9
            chains,tags=data[prefix+'chain_omegas'],data[prefix+'folds']
            ops=data[prefix+'score_operators']
            scores=[np.einsum('nij,kji->nk',chains[tags!=g],ops[g]).real.mean(0) for g in (0,1)]
            np.testing.assert_allclose(scores,row['audit']['fixed_training_fold_scores'],atol=2e-14)
            assert row['audit']['point']['heldout_TR_even']==pytest.approx(np.mean(scores,axis=0)[0],abs=2e-14)
    assert not result['phase_claim'] and not result['is_entanglement_witness']
