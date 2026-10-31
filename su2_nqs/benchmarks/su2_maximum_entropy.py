"""Maximum entropy on a bounded SU(2) invariant observable algebra.

This is an offline regional readout, NOT a state ansatz or SU(2) repair.
All charges q=2S are integers. The real Condon--Shortley regional CG basis
fixes the time-reversal convention: conjugation on multiplicity data.
Constraints are fixed p(q), every pair scalar, and every symmetrized product
of two pair scalars. They are NOT all observables of support <=4.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, combinations_with_replacement

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .regional_su2_readout import regional_schur_columns
from .spin_operator_information import spin_dot_matrix


@dataclass(frozen=True)
class InvariantConstraintSpace:
    """Block-traceless, orthonormal real constraints; doubled spin labels."""
    n_sites: int
    qs: tuple[int, ...]
    dimensions: tuple[int, ...]
    slices: tuple[slice, ...]
    magnetic_columns: tuple[tuple[np.ndarray, ...], ...]
    operators: np.ndarray
    diagnostics: dict

    def data(self, rho: np.ndarray) -> np.ndarray:
        """Omega=direct_sum_q Tr_m(P_q rho P_q), legal even without SU(2)."""
        rho = np.asarray(rho, np.complex128)
        if (rho.shape != (2**self.n_sites,)*2 or not np.isfinite(rho).all()
                or np.linalg.norm(rho-rho.conj().T) > 1e-9
                or abs(np.trace(rho)-1) > 1e-9
                or np.linalg.eigvalsh(rho).min() < -1e-9):
            raise ValueError('normalized positive Hermitian regional rho required; no repair')
        out = np.zeros((sum(self.dimensions),)*2, np.complex128)
        for sl, columns in zip(self.slices, self.magnetic_columns):
            out[sl, sl] = sum(u.conj().T @ rho @ u for u in columns)
        return out


@lru_cache(maxsize=5)
def invariant_constraint_space(n_sites: int = 6) -> InvariantConstraintSpace:
    """Exact reduced pair/two-bond span on 2..6 spins, without state fitting."""
    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or not 2 <= n_sites <= 6:
        raise ValueError('integer region size 2..6 required')
    u, labels = regional_schur_columns(int(n_sites))
    qs = tuple(range(n_sites % 2, n_sites+1, 2))
    columns = tuple(tuple(u[:, [i for i, x in enumerate(labels) if x[:2] == (q, m)]]
                          for m in range(-q, q+1, 2)) for q in qs)
    dims = tuple(a[0].shape[1] for a in columns)
    offsets = np.cumsum((0,)+dims)
    slices = tuple(slice(int(a), int(b)) for a,b in zip(offsets[:-1], offsets[1:]))
    pairs = [spin_dot_matrix(n_sites, i, j) for i,j in combinations(range(n_sites), 2)]
    full = [np.eye(2**n_sites), *pairs]
    full += [(a@b+b@a)/2 for a,b in combinations_with_replacement(pairs, 2)]
    reduced = np.zeros((len(full), int(offsets[-1]), int(offsets[-1])))
    for sl, us in zip(slices, columns):
        reduced[:, sl, sl] = np.array([us[-1].T @ a @ us[-1] for a in full]).real
    for sl,d in zip(slices,dims):
        reduced[:,sl,sl] -= np.trace(reduced[:,sl,sl],axis1=1,axis2=2)[:,None,None]*np.eye(d)/d
    vectors, singular, _ = np.linalg.svd(reduced.reshape(len(full),-1).T, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-11))
    operators = vectors[:,:rank].T.reshape(rank,int(offsets[-1]),int(offsets[-1]))
    operators.setflags(write=False)
    diag = dict(raw_operator_count=len(full), centered_constraint_rank=rank,
                fixed_sector_weights=len(qs), rank_including_sector_projectors=rank+len(qs),
                real_symmetric_algebra_dimension=sum(d*(d+1)//2 for d in dims),
                hermitian_algebra_dimension=sum(d*d for d in dims),
                unfixed_TR_even_directions=sum(d*(d+1)//2 for d in dims)-rank-len(qs),
                singular_value_cutoff=1e-11, no_state_dependent_feature_selection=True)
    return InvariantConstraintSpace(int(n_sites), qs, dims, slices, columns, operators, diag)


def matrix_entropy(matrix: np.ndarray) -> float:
    """Von Neumann entropy (nats); support zeros are not given artificial mass."""
    ev = np.linalg.eigvalsh(matrix)
    if ev.min() < -1e-9:
        raise ValueError('matrix is not positive')
    positive = ev[ev > 0]
    return float(-positive @ np.log(positive))


def positive_matrix_log(matrix: np.ndarray) -> np.ndarray:
    """Log on a strictly positive matrix; no ridge or eigenvalue flooring."""
    ev, u = np.linalg.eigh(matrix)
    if ev.min() <= 0:
        raise ValueError('logarithmic scoring requires positive training support; no flooring')
    return (u*np.log(ev)) @ u.conj().T


class FixedWeightExponentialFamily:
    """Dual objective with p(q) fixed exactly and all uniform gauges removed."""
    def __init__(self, space: InvariantConstraintSpace, omega: np.ndarray):
        self.space = space
        self.weights = np.array([np.trace(omega[s,s]).real for s in space.slices])
        self.target = np.einsum('kij,ji->k',space.operators,omega).real

    def evaluate(self, theta: np.ndarray) -> tuple[float,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Dual, gradient, exact Kubo--Mori Hessian, model and log(model)."""
        space = self.space
        k = len(space.operators)
        gradient = -self.target.copy()
        hessian = np.zeros((k,k))
        value = -float(theta @ self.target)
        model = np.zeros((sum(space.dimensions),)*2)
        log_model = np.zeros_like(model)
        for weight,sl in zip(self.weights,space.slices):
            if weight <= 0:
                raise ValueError('zero sector weight needs a support-face solver; no repair')
            a = space.operators[:,sl,sl]
            generator = np.einsum('k,kij->ij',theta,a)
            ev, u = np.linalg.eigh(generator)
            log_z = logsumexp(ev)
            probs = np.exp(ev-log_z)
            model[sl,sl] = weight*(u*probs) @ u.T
            log_model[sl,sl] = generator+(np.log(weight)-log_z)*np.eye(len(ev))
            b = np.einsum('ai,kab,bj->kij',u,a,u)
            means = np.einsum('kii,i->k',b,probs)
            gradient += weight*means
            delta = ev[:,None]-ev[None,:]
            numerator = probs[:,None]-probs[None,:]
            factor = np.exp((ev[:,None]+ev[None,:])/2-log_z)
            np.divide(numerator,delta,out=factor,where=np.abs(delta)>1e-8)
            hessian += weight*(np.einsum('kij,lji,ij->kl',b,b,factor)-np.outer(means,means))
            value += weight*log_z
        return value,gradient,hessian,model,log_model


def fit_invariant_maximum_entropy(
    omega: np.ndarray, space: InvariantConstraintSpace, *,
    max_iterations: int = 100, moment_tolerance: float = 2e-9,
) -> tuple[dict,np.ndarray,np.ndarray]:
    """Fit exactly declared moments and report closure, not optimizer status alone.

    Returns diagnostics, Omega_ME and log(Omega_ME). The returned comparator
    never replaces the physical state. TR-even and TR-odd information are
    separated in this fixed real CG basis. Boundary cases are not regularized.
    """
    omega = np.asarray(omega,np.complex128)
    dimension = sum(space.dimensions)
    if (omega.shape!=(dimension,dimension) or not np.isfinite(omega).all()
            or np.linalg.norm(omega-omega.conj().T)>1e-9
            or abs(np.trace(omega)-1)>1e-9 or np.linalg.eigvalsh(omega).min() < -1e-9):
        raise ValueError('normalized positive Hermitian invariant-algebra data required')
    offblock=omega.copy()
    for sl in space.slices:
        offblock[sl,sl]=0
    if np.linalg.norm(offblock)>1e-10:
        raise ValueError('Omega must be block diagonal in doubled total spin')
    if max_iterations < 1 or not np.isfinite(moment_tolerance) or moment_tolerance <= 0:
        raise ValueError('positive iteration budget and moment tolerance required')
    family = FixedWeightExponentialFamily(space,omega)
    cache = [None,None]
    def evaluate(theta):
        if cache[0] is None or not np.array_equal(theta,cache[0]):
            cache[:] = [theta.copy(),family.evaluate(theta)]
        return cache[1]
    if len(space.operators):
        fit = minimize(lambda t:evaluate(t)[0],np.zeros(len(space.operators)),
                       jac=lambda t:evaluate(t)[1],hess=lambda t:evaluate(t)[2],
                       method='trust-exact',options=dict(maxiter=max_iterations,gtol=1e-11))
        theta,iterations,status,message=fit.x,int(fit.nit),bool(fit.success),str(fit.message)
    else:
        theta,iterations,status,message=np.zeros(0),0,True,'all sectors scalar'
    # A trust-region objective difference can round to zero before its moments
    # converge. Refine the SAME unregularized dual equations, within the original
    # iteration budget. Acceptance requires lower residual and no dual increase
    # beyond objective roundoff; neither data nor scientific tolerances change.
    refinement=0
    for _ in range(min(8,max_iterations-iterations)):
        value,gradient,hessian,_,_=evaluate(theta)
        residual=float(np.max(abs(gradient),initial=0))
        # The entropy identity also involves theta.gradient. A small moment
        # residual alone can fail that unchanged closure check at large theta.
        if residual<=moment_tolerance and abs(float(theta@gradient))<=1e-9:break
        if not len(theta) or np.linalg.eigvalsh(hessian).min()<=0:break
        try:direction=np.linalg.solve(hessian,gradient)
        except np.linalg.LinAlgError:break
        if not np.isfinite(direction).all():break
        accepted=False
        for halving in range(12):
            candidate=theta-direction/(2**halving)
            next_value,next_gradient,_,_,_=evaluate(candidate)
            roundoff=64*np.finfo(float).eps*max(1.,abs(value))
            if (next_value<=value+roundoff
                    and float(np.max(abs(next_gradient),initial=0))<residual):
                theta=candidate;refinement+=1;accepted=True;break
        if not accepted:break
    _,gradient,hessian,model,log_model=evaluate(theta)
    s_data,s_even,s_model=map(matrix_entropy,(omega,omega.real,model))
    cross_entropy=-float(np.trace(omega@log_model).real)
    residual=float(np.max(abs(gradient),initial=0))
    identity_error=cross_entropy-s_data-(s_model-s_data)
    report=dict(space=space.diagnostics,qs=list(space.qs),multiplicities=list(space.dimensions),
                sector_weights=family.weights.tolist(),iterations=iterations+refinement,
                trust_region_iterations=iterations,moment_refinement_iterations=refinement,
                optimizer_success=status,optimizer_message=message,
                moment_residual_max=residual,moment_tolerance=moment_tolerance,
                certified_moment_fit=bool(residual<=moment_tolerance and abs(identity_error)<=1e-8),
                entropy_data=s_data,entropy_TR_even=s_even,entropy_model=s_model,
                information_total=cross_entropy-s_data,
                information_TR_even=cross_entropy-s_even,
                information_TR_odd=s_even-s_data,
                entropy_gap=s_model-s_data,relative_entropy_gap_closure=identity_error,
                dual_moment_closure_error=abs(identity_error-float(theta@gradient)),
                parameter_norm=float(np.linalg.norm(theta)),
                hessian_min_eigenvalue=float(np.linalg.eigvalsh(hessian).min()) if len(theta) else None,
                input_min_eigenvalue=float(np.linalg.eigvalsh(omega).min()),
                model_min_eigenvalue=float(np.linalg.eigvalsh(model).min()),
                strict_entanglement_spectrum=False,physical_state_replaced=False,
                is_entanglement_measure=False,TR_even_constraint_set_only=True)
    return report,model,log_model


def heldout_log_score_operator(
    omega_train: np.ndarray, log_maxent_train: np.ndarray, *, tr_even: bool = True,
) -> np.ndarray:
    """Fixed-training log-score difference; evaluate linearly on untouched data.

    Positive means full-block data predicts better than the lower-order model.
    Negative values are retained. This is NOT the nonnegative plug-in entropy
    gap. Fitting uncertainty requires refitting; test-chain SE is conditional.
    """
    data = omega_train.real if tr_even else omega_train
    return positive_matrix_log(data)-log_maxent_train


def audit_invariant_chains(
    chain_omegas: np.ndarray, folds: np.ndarray, space: InvariantConstraintSpace,
    *, max_iterations: int = 100,
) -> tuple[dict,dict[str,np.ndarray]]:
    """Two independent streams, with delete-whole-chain FULL refitting.

    The equal-stream estimator keeps weights 1/2 under stratified deletion.
    The refit SE accounts for both fitted and test chains; per-fold fixed-fit
    SEs are separately labelled conditional. This corrects neither Born
    nonequilibration nor deterministic ansatz error.
    """
    chain_omegas=np.asarray(chain_omegas,np.complex128)
    folds=np.asarray(folds)
    if (chain_omegas.ndim!=3 or folds.shape!=(len(chain_omegas),)
            or set(folds.tolist())!={0,1} or min(np.bincount(folds))<3):
        raise ValueError('two preassigned streams with >=3 whole chains each required')
    numerical=[]
    def fit(omega):
        report,model,log_model=fit_invariant_maximum_entropy(omega,space,max_iterations=max_iterations)
        numerical.append((report['moment_residual_max'],abs(report['relative_entropy_gap_closure'])))
        if not report['certified_moment_fit']:
            raise RuntimeError('maximum-entropy moment fit failed; no information claim')
        return report,model,log_model
    names=['information_TR_even','information_TR_odd','information_total',
           'heldout_TR_even','heldout_TR_odd','heldout_total']
    def evaluate(omegas,tags):
        means=[omegas[tags==g].mean(0) for g in (0,1)]
        full,model,log_model=fit((means[0]+means[1])/2)
        score_operators=[]; fold_scores=[]; fold_SE=[]
        for g in (0,1):
            _,_,log_fit=fit(means[g])
            even=heldout_log_score_operator(means[g],log_fit)
            total=heldout_log_score_operator(means[g],log_fit,tr_even=False)
            ops=np.stack((even,total-even,total))
            scores=np.einsum('nij,kji->nk',omegas[tags!=g],ops).real
            score_operators.append(ops);fold_scores.append(scores.mean(0))
            fold_SE.append(scores.std(0,ddof=1)/np.sqrt(len(scores)))
        values=np.r_[[full[k] for k in names[:3]],np.mean(fold_scores,axis=0)]
        return values,full,model,log_model,np.array(score_operators),np.array(fold_scores),np.array(fold_SE)
    point,full,model,log_model,operators,fold_scores,fold_SE=evaluate(chain_omegas,folds)
    deleted=[]
    for i in range(len(chain_omegas)):
        mask=np.arange(len(chain_omegas))!=i
        deleted.append(evaluate(chain_omegas[mask],folds[mask])[0])
    deleted=np.array(deleted)
    covariance=np.zeros((len(names),len(names)))
    bias=np.zeros(len(names))
    for g in (0,1):
        jack=deleted[folds==g];ng=len(jack)
        center=jack-jack.mean(0)
        covariance+=(ng-1)/ng*(center.T@center)
        bias+=(ng-1)*(jack.mean(0)-point)
    report=dict(point=dict(zip(names,map(float,point))),
        stratified_whole_chain_refit_SE=dict(zip(names,map(float,np.sqrt(np.diag(covariance))))),
        jackknife_bias_estimate=dict(zip(names,map(float,bias))),
        jackknife_bias_corrected=dict(zip(names,map(float,point-bias))),
        fixed_training_fold_scores=fold_scores.tolist(),
        fixed_training_conditional_test_SE=fold_SE.tolist(),
        full_fit=full,chains_per_stream=[int(np.sum(folds==g)) for g in (0,1)],
        full_refit_count=len(numerical),maximum_moment_residual=max(x[0] for x in numerical),
        maximum_entropy_identity_error=max(x[1] for x in numerical),
        negative_heldout_values_retained=True,
        conditional_and_refit_uncertainty_are_distinct=True,
        warning='Nonlinear plug-in information is biased. Signed cross-fitted log-score tests predictive improvement, not entanglement or a nonnegative information estimate. Stratified delete-chain refits are asymptotic, not a certified confidence/bias bound.')
    arrays=dict(omega=(chain_omegas[folds==0].mean(0)+chain_omegas[folds==1].mean(0))/2,
        maxent=model,log_maxent=log_model,chain_omegas=chain_omegas,folds=folds,
        score_operators=operators,jackknife_values=deleted,covariance=covariance)
    return report,arrays
