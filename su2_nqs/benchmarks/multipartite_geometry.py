"""Regional GME geometry, local Schur channels and constructive moment tests.

All operations use physical down/up MSB ordering and doubled q=2S,m=2M.
Channels act locally on the DECLARED parties. No total-spin postselection is
used. SDP PPT mixtures are an outer relaxation, never a separability proof.
"""
from __future__ import annotations

from itertools import combinations
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from .regional_su2_readout import regional_schur_columns
from .su2_multipartite_witness import canonical_bipartitions, partial_transpose
from .su2_multipartite_witness import repair_fully_decomposable_witness, WitnessCertificate
from .su2_maximum_entropy import invariant_constraint_space
from .spin_operator_information import spin_dot_matrix


def partial_trace_sites(matrix: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Trace physical sites, retaining the supplied MSB order and batch axes."""
    x = np.asarray(matrix, complex)
    d = x.shape[-1]; n = d.bit_length()-1
    keep = tuple(keep)
    if (x.shape[-2:] != (d, d) or d != 2**n or not keep or
            len(set(keep)) != len(keep) or
            any(isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k < n for k in keep)):
        raise ValueError('square qubit matrices and distinct retained integer sites required')
    other = tuple(i for i in range(n) if i not in keep)
    batch = x.ndim-2; order = keep+other
    axes = list(range(batch))+[batch+i for i in order]+[batch+n+i for i in order]
    y = x.reshape(x.shape[:-2]+(2,)*(2*n)).transpose(axes)
    a, b = 2**len(keep), 2**len(other)
    return np.trace(y.reshape(x.shape[:-2]+(a,b,a,b)), axis1=-3, axis2=-1)


def spin_measurement_branches(matrix: np.ndarray, site: int, axis: str = 'z'
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalized remaining-spin RDMs after local S_axis measurement.

    Return two branches and their probabilities in doubled outcome order
    m=(-1,+1). Leading batch axes are preserved. Summing the branches equals
    discarding the measured spin. No branch is selected or normalized here.
    """
    x = np.asarray(matrix, np.complex128)
    if x.ndim < 2 or x.shape[-2] != x.shape[-1]:
        raise ValueError('square physical qubit matrices required')
    d = x.shape[-1]; n = d.bit_length()-1
    if (d != 2**n or n < 2 or isinstance(site, bool) or
            not isinstance(site, Integral) or not 0 <= site < n or axis not in ('x','y','z')):
        raise ValueError('valid physical site and x/y/z measurement axis required')
    sigma = {'x': np.array([[0,1],[1,0]]),
             'y': np.array([[0,1j],[-1j,0]]),
             'z': np.diag([-1.,1.])}[axis]
    _, vectors = np.linalg.eigh(sigma)
    other = tuple(i for i in range(n) if i != site); order = other+(int(site),)
    batch = x.ndim-2
    axes = tuple(range(batch))+tuple(batch+i for i in order)+tuple(batch+n+i for i in order)
    grouped = x.reshape(x.shape[:-2]+(2,)*(2*n)).transpose(axes)
    grouped = grouped.reshape(x.shape[:-2]+(d//2,2,d//2,2))
    branches = np.stack([np.einsum('i,...aibj,j->...ab',v.conj(),grouped,v)
                         for v in vectors.T], axis=-3)
    return branches, np.trace(branches, axis1=-2, axis2=-1).real


def measurement_witness_pullback(witness: np.ndarray, site: int, axis: str,
                                  outcome_m: int) -> np.ndarray:
    """Embed a child W as |m_axis><m_axis| at a measured physical site.

    Its expectation on the original parent equals p_m Tr(W rho_child|m).
    Thus independent-chain audits avoid normalization and rare-event ratios.
    outcome_m is the doubled magnetic label -1 or +1, never a half-integer.
    """
    w = np.asarray(witness, np.complex128)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] & (w.shape[0]-1):
        raise ValueError('physical child qubit operator required')
    n = w.shape[0].bit_length()
    if (isinstance(site, bool) or not isinstance(site, Integral) or not 0 <= site < n or
            isinstance(outcome_m, bool) or not isinstance(outcome_m, Integral) or
            outcome_m not in (-1,1) or axis not in ('x','y','z')):
        raise ValueError('valid parent site, axis and doubled outcome m=-1/+1 required')
    sigma = {'x': np.array([[0,1],[1,0]]),
             'y': np.array([[0,1j],[-1j,0]]),
             'z': np.diag([-1.,1.])}[axis]
    projector = (np.eye(2)+outcome_m*sigma)/2
    other = tuple(i for i in range(n) if i != site); order = other+(int(site),)
    inv = tuple(np.argsort(order))
    grouped = np.kron(w, projector).reshape((2,)*(2*n))
    return grouped.transpose(inv+tuple(n+i for i in inv)).reshape(2**n,2**n)


def covering_region_density(n_sites: int, pairs: Sequence[Sequence[int]],
                            region: Sequence[int]) -> np.ndarray:
    """Exact regional RDM of a fixed product of physical singlet dimers.

    Dimers internal to the region remain singlets; a partner outside leaves
    I/2. This constructs controls even for N=36 without a 2**N state vector.
    Averaging different coverings is an explicitly mixed control, not a
    coherent space-group projection or a QSL assignment.
    """
    region = tuple(region); pairs = tuple(tuple(p) for p in pairs)
    if (isinstance(n_sites,bool) or not isinstance(n_sites,Integral) or n_sites<2 or n_sites%2 or
            any(len(p)!=2 for p in pairs) or
            any(isinstance(i,bool) or not isinstance(i,Integral) for p in pairs for i in p) or
            sorted(i for p in pairs for i in p)!=list(range(n_sites)) or
            not 2<=len(region)<=6 or len(set(region))!=len(region) or
            any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<n_sites for i in region)):
        raise ValueError('even physical system, complete covering and distinct regional sites required')
    n=len(region);d=2**n;rho=np.eye(d,dtype=np.complex128)/d
    for i,j in pairs:
        if i in region and j in region:
            rho=rho@(np.eye(d)-4*spin_dot_matrix(n,region.index(i),region.index(j)))
    return rho


def solve_physical_multipartite_witness(rho: np.ndarray,
        parties: Sequence[Sequence[int]], *, seconds: float = 8.,
        real_operators: bool = False) -> tuple[dict, WitnessCertificate]:
    """Fit an unrestricted physical witness, including symmetry-broken children.

    Uses the existing fully-decomposable normalization and conservative repair.
    Real-symmetric operators may be requested as an explicit smaller witness
    family on any state. Imaginary coherences remain in the input; only the
    detector family is restricted, with no SU2 or spin-rotation restriction.
    A repaired feasible candidate is useful even if optimization is incomplete;
    no solver status establishes an optimal GMN or absence of entanglement.
    """
    import cvxpy as cp
    from .su2_multipartite_witness import _matrix, _partial_transpose_indices, evaluate_frozen_witness
    x = _matrix(rho, normalized=True); d = len(x); n = d.bit_length()-1
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite solver time required')
    cuts = canonical_bipartitions(parties,n)
    w = cp.Variable((d,d),symmetric=True) if real_operators else cp.Variable((d,d),hermitian=True)
    auxiliary = []; constraints = []
    for cut in cuts:
        q = cp.Variable((d,d),symmetric=True) if real_operators else cp.Variable((d,d),hermitian=True)
        pt = cp.reshape(cp.reshape(q,(d*d,),order='C')[_partial_transpose_indices(n,cut).reshape(-1)],(d,d),order='C')
        auxiliary.append(q); constraints.extend((q >> 0,np.eye(d)-q >> 0,w-pt >> 0))
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(w@x))),constraints)
    problem.solve(solver='SCS',eps=2e-6,max_iters=15000,time_limit_secs=float(seconds))
    if w.value is None or any(q.value is None for q in auxiliary):
        raise RuntimeError('physical SDP returned no candidate: '+str(problem.status))
    cert = repair_fully_decomposable_witness(w.value,[q.value for q in auxiliary],cuts)
    report = evaluate_frozen_witness(cert,x)
    report.update(solver_status=str(problem.status),solver_solve_seconds=problem.solver_stats.solve_time,
        solver_iterations=problem.solver_stats.num_iters,operator_span='physical_unrestricted',
        real_symmetric_operators=real_operators,optimality_certified=False,state_repaired=False,
        certificate=cert.diagnostics)
    return report,cert


def biseparable_residual_ball(rho: np.ndarray, weights: np.ndarray,
        vectors: np.ndarray, cut_masks: np.ndarray, parties: Sequence[Sequence[int]],
        *, inflation: float = 1.02, shared_su2_average: bool = False
        ) -> tuple[dict, dict[str,np.ndarray]]:
    """Close a full-RDM product-mixture residual with a known separable ball.

    Verify every saved atom is product across an allowed party cut. eta is
    their convex mixture, optionally averaged by a shared global SU2 rotation
    (a mixture of local unitaries, preserving biseparability). Define
    tau=(inflation*rho-eta)/(inflation-1). If ||d*tau-I||_F<1, Gurvits--Barnum
    Theorem1 (quant-ph/0204159) makes tau separable across any bipartition.
    rho=eta/inflation+(1-1/inflation)*tau is then biseparable, closing even
    the imaginary/magnetic residual without changing the physical input.
    Returns a numerical certificate with explicit margins, not interval proof.
    """
    from .su2_multipartite_witness import _matrix
    r=_matrix(rho,normalized=True);d=len(r);n=d.bit_length()-1
    if not np.isfinite(inflation) or inflation<=1:
        raise ValueError('finite inflation greater than one required')
    cuts=canonical_bipartitions(parties,n)
    weights=np.asarray(weights);vectors=np.asarray(vectors,np.complex128);cut_masks=np.asarray(cut_masks)
    if (weights.ndim!=1 or vectors.shape!=(len(weights),d) or cut_masks.shape!=weights.shape or
            not np.isfinite(weights).all() or not np.isfinite(vectors).all() or
            weights.min(initial=0)<0 or abs(weights.sum()-1)>1e-12):
        raise ValueError('explicit normalized positive product mixture required')
    allowed={sum(1<<i for i in cut):cut for cut in cuts};factor=0.
    for v,mask in zip(vectors,cut_masks):
        if mask not in allowed or abs(np.vdot(v,v)-1)>1e-12:
            raise ValueError('atom norm/cut differs from declared party structure')
        cut=allowed[mask];order=cut+tuple(i for i in range(n) if i not in cut)
        s=np.linalg.svd(v.reshape((2,)*n).transpose(order).reshape(2**len(cut),-1),compute_uv=False)
        factor=max(factor,float(np.max(s[1:],initial=0)))
    if factor>1e-12:raise ValueError('saved atom is not a product across its declared cut')
    eta=np.einsum('a,ai,aj->ij',weights,vectors,vectors.conj())
    if shared_su2_average:
        eta=lift_invariant_model(invariant_constraint_space(n).data(eta),n)
    tau=(inflation*r-eta)/(inflation-1)
    radius=float(np.linalg.norm(d*tau-np.eye(d)))
    margin=float(1-radius);trace_error=float(abs(np.trace(tau)-1))
    reconstruction=float(np.linalg.norm(r-(eta+(inflation-1)*tau)/inflation))
    passed=margin>1e-8 and trace_error<1e-9 and reconstruction<1e-12
    return dict(numerically_certified_biseparable=bool(passed),formal_interval_certificate=False,
        theorem='Gurvits-Barnum2002 Theorem1: I+Delta separable if Frobenius(Delta)<=1; bipartite, not full separability.',
        reference='https://arxiv.org/abs/quant-ph/0204159',inflation=inflation,
        residual_ball_radius=radius,residual_ball_margin=margin,tau_trace_error=trace_error,
        exact_decomposition_residual=reconstruction,product_Schmidt_tail_max=factor,
        shared_su2_average_of_candidate_only=shared_su2_average,physical_input_repaired=False,
        failed_certificate_excludes_nothing=True),dict(eta=eta,tau=tau)


def localizable_block_negativity(rho: np.ndarray, center: int,
        left: Sequence[int], axes: Sequence[str] = ('z','x')) -> dict:
    """Entanglement of two pairs assisted by measuring the fifth spin.

    Compare the SAME two remaining parties with the center discarded versus
    all measurement outcomes retained. Homogeneous negativity avoids ratios.
    Local q=0/2 pinching diagnoses singlet--triplet coherence dependence of
    the detected negativity; a PPT result does not exclude bound entanglement.
    Assistance alone is not a GME or QSL witness. All charges are doubled.
    """
    from .su2_multipartite_witness import _matrix
    r=_matrix(rho,normalized=True);left=tuple(left)
    if (r.shape!=(32,32) or isinstance(center,bool) or not isinstance(center,Integral) or
            not 0<=center<5 or len(left)!=2 or len(set(left))!=2 or center in left or
            any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<5 for i in left) or
            not axes or any(a not in ('x','y','z') for a in axes)):
        raise ValueError('five physical spins, center and disjoint two-spin left block required')
    keep=tuple(i for i in range(5) if i!=center);a=tuple(keep.index(i) for i in left)
    b=tuple(i for i in range(4) if i not in a)
    def negativity(x):
        return float((np.abs(np.linalg.eigvalsh(partial_transpose(x,a))).sum()-np.trace(x).real)/2)
    unassisted=negativity(partial_trace_sites(r,keep));out={}
    for axis in axes:
        branches,prob=spin_measurement_branches(r,center,axis);rows=[]
        for m,x,p in zip((-1,1),branches,prob):
            pinched=local_schur_channel(x,(a,b),'charge');sectors={}
            for qa in (0,2):
                pa=np.eye(16)/4-spin_dot_matrix(4,*a)
                if qa==2:pa=np.eye(16)-pa
                for qb in (0,2):
                    pb=np.eye(16)/4-spin_dot_matrix(4,*b)
                    if qb==2:pb=np.eye(16)-pb
                    projector=pa@pb;block=projector@x@projector
                    sectors[str(qa)+','+str(qb)]=dict(probability=float(np.trace(block).real),
                                                     weighted_negativity=negativity(block))
            rows.append(dict(m=m,probability=float(p),weighted_negativity=negativity(x),
                weighted_charge_negativity=negativity(pinched),local_q_sectors=sectors))
        assisted=sum(x['weighted_negativity'] for x in rows)
        charge=sum(x['weighted_charge_negativity'] for x in rows)
        out[axis]=dict(assisted_negativity=assisted,after_local_charge_negativity=charge,
            assistance_gain=assisted-unassisted,charge_coherence_loss=assisted-charge,outcomes=rows)
    return dict(unassisted_negativity=unassisted,axes=out,
        interpretation='Detected bipartite negativity after assistance, not tripartite GME. Charge pinching is local to each remaining pair; neither nonlocal singlet filtering nor input symmetry repair is used.')


def induced_geometry(sites: Sequence[int], bonds: Sequence[Sequence[float]], j2: float) -> dict:
    """Classify the actual supplied Hamiltonian edges; no inferred lattice metric."""
    sites = tuple(sites); edges = {'J1': [], 'J2': []}
    for i,j,value in bonds:
        if i in sites and j in sites:
            kind = 'J1' if value == 1 else 'J2'
            if kind == 'J2' and value != j2:
                raise ValueError('unexpected Hamiltonian coupling')
            edges[kind].append(sorted((sites.index(i), sites.index(j))))
    def topology(pairs):
        parent = list(range(len(sites)))
        def root(i):
            while parent[i] != i: i = parent[i]
            return i
        for i,j in pairs: parent[root(i)] = root(j)
        components = len({root(i) for i in parent})
        return dict(components=components, edges=len(pairs),
                    cycle_rank=len(pairs)-len(sites)+components)
    return dict(sites=list(sites), edges=edges, J1=topology(edges['J1']),
                active=topology(edges['J1']+(edges['J2'] if j2 else [])),
                graph_derived_from_saved_Hamiltonian=True)


def local_schur_channel(matrix: np.ndarray, parties: Sequence[Sequence[int]],
                        keep: str = 'charge') -> np.ndarray:
    """Apply local charge measurement and optional local index erasure.

    'charge' keeps q,magnetic,multiplicity within each block; 'magnetic'
    replaces multiplicity by I/a, 'multiplicity' replaces magnetic by I/(q+1).
    q flags survive as orthogonal blocks. All three maps are self-adjoint,
    unital CPTP maps, so they also pull back a fixed physical witness. Their
    outputs are deliberate operational comparisons, not repaired input states.
    """
    x = np.asarray(matrix, complex).copy(); d = len(x); n = d.bit_length()-1
    canonical_bipartitions(parties, n)
    if x.shape != (d,d) or keep not in ('charge','magnetic','multiplicity'):
        raise ValueError('physical matrix and declared Schur channel required')
    for party in parties:
        party = tuple(party); other = tuple(i for i in range(n) if i not in party)
        order = party+other; inv = np.argsort(order)
        da,db = 2**len(party),2**len(other)
        grouped = x.reshape((2,)*(2*n)).transpose(order+tuple(n+i for i in order)).reshape(da,db,da,db)
        u,labels = regional_schur_columns(len(party))
        z = np.einsum('ia,irjs,jb->arbs',u.conj(),grouped,u,optimize=True)
        out = np.zeros_like(z)
        for q in sorted({a[0] for a in labels}):
            ids = [i for i,label in enumerate(labels) if label[0] == q]
            a = len(ids)//(q+1); m = q+1
            block = z[ids][:,:,ids,:].reshape(m,a,db,m,a,db)
            if keep == 'magnetic':
                reduced = np.trace(block,axis1=1,axis2=4)
                block = np.einsum('irjs,ab->iarjbs',reduced,np.eye(a)/a)
            elif keep == 'multiplicity':
                reduced = np.trace(block,axis1=0,axis2=3)
                block = np.einsum('arbs,ij->iarjbs',reduced,np.eye(m)/m)
            out[np.ix_(ids,np.arange(db),ids,np.arange(db))] = block.reshape(len(ids),db,len(ids),db)
        physical = np.einsum('ia,arbs,jb->irjs',u,out,u.conj(),optimize=True)
        x = physical.reshape((2,)*(2*n)).transpose(tuple(inv)+tuple(n+inv)).reshape(d,d)
    return x


def low_order_observables(n: int) -> np.ndarray:
    """p(q), all pair scalars and symmetrized two-bond constraints, no duplicates."""
    space = invariant_constraint_space(n); bank = []
    for sl,columns in zip(space.slices,space.magnetic_columns):
        bank.append(sum(u@u.conj().T for u in columns))
    for op in space.operators:
        bank.append(sum(u@op[sl,sl]@u.conj().T
                        for sl,columns in zip(space.slices,space.magnetic_columns) for u in columns))
    return np.asarray(bank)


def lift_invariant_model(model: np.ndarray, n: int) -> np.ndarray:
    """Lift a fitted reduced maximum-entropy model, not the measured state."""
    space = invariant_constraint_space(n)
    return sum(u@model[sl,sl]@u.conj().T/(q+1)
               for q,sl,columns in zip(space.qs,space.slices,space.magnetic_columns) for u in columns)


def triplet_screening_certificate(parties: Sequence[Sequence[int]]) -> tuple[WitnessCertificate,dict]:
    """Analytic three-party spin-1 screening witness on six physical spins.

    Each party is a fixed physical pair. P222 selects LOCAL q_i=2; within
    that support the total q=0 projector P0 has rank one and Schmidt weights
    (1/3,1/3,1/3) across every party cut. Thus p222,total0 > p222/3 witnesses
    GME of the three pairs. W=P222/2-3*P0/2 has 0<=W^T_M<=I for each cut,
    yielding an explicit fully-decomposable GMN certificate. A total-singlet
    probability is measured in the original state, not used as a state filter.
    """
    cuts=canonical_bipartitions(parties,6)
    if len(parties)!=3 or any(len(p)!=2 for p in parties):
        raise ValueError('three disjoint two-spin physical parties required')
    p222=np.eye(64)
    for pair in parties: p222=p222@(3*np.eye(64)/4+spin_dot_matrix(6,*pair))
    u,labels=regional_schur_columns(6);singlets=u[:,[i for i,l in enumerate(labels) if l[0]==0]]
    p0=p222@singlets@singlets.conj().T@p222
    if abs(np.trace(p0)-1)>1e-12 or np.linalg.norm(p0@p0-p0)>1e-12:
        raise RuntimeError('three spin-1 total singlet must have rank one')
    w=p222/2-1.5*p0
    cert=repair_fully_decomposable_witness(w,[partial_transpose(w,cut) for cut in cuts],cuts,safety_margin=1e-10)
    return cert,dict(P222=p222,P222_total0=p0,unshifted_witness=w)


def ppt_mixture_moment_fit(target: np.ndarray, operators: np.ndarray,
                            parties: Sequence[Sequence[int]], *, seconds: float = 30.) -> tuple[dict,np.ndarray]:
    """Minimize worst moment residual in the PPT-mixture OUTER relaxation.

    A feasible point does not imply biseparability. Inaccurate/infeasible solver
    status is not a rigorous exclusion certificate. Use a feasible dual witness
    restricted to the same low-order span for any actual exclusion claim.
    """
    import cvxpy as cp
    d = operators.shape[-1]; n = d.bit_length()-1
    cuts = canonical_bipartitions(parties,n)
    components = [cp.Variable((d,d),symmetric=True) for _ in cuts]
    total = sum(components); residual = cp.Variable(nonneg=True)
    constraints = [cp.trace(total)==1]
    ids = np.arange(d*d).reshape(d,d)
    for cut,x in zip(cuts,components):
        axes=list(range(2*n))
        for site in cut: axes[site],axes[n+site]=axes[n+site],axes[site]
        indices = ids.reshape((2,)*(2*n)).transpose(axes).reshape(-1)
        pt = cp.reshape(cp.reshape(x,(d*d,),order='C')[indices],(d,d),order='C')
        constraints += [x >> 0, pt >> 0]
    moments = operators.real.reshape(len(operators),-1) @ cp.reshape(total,(d*d,),order='C')
    constraints += [moments-target <= residual, target-moments <= residual]
    problem = cp.Problem(cp.Minimize(residual),constraints)
    problem.solve(solver='SCS',eps=2e-6,max_iters=10000,time_limit_secs=seconds)
    if any(x.value is None for x in components):
        return dict(status=str(problem.status),biseparable_proved=False,
                    infeasibility_certified=False),np.empty((0,d,d))
    values = np.array([x.value for x in components]); rho = values.sum(0)
    minimum = min(min(np.linalg.eigvalsh(x).min(),np.linalg.eigvalsh(partial_transpose(x,cut)).min())
                  for x,cut in zip(values,cuts))
    error = float(np.max(abs(np.einsum('kij,ji->k',operators,rho).real-target)))
    return dict(status=str(problem.status),minimum_PSD_PT_eigenvalue=float(minimum),
                trace_error=float(abs(np.trace(rho)-1)),moment_residual_max=error,
                solver_seconds=problem.solver_stats.solve_time,
                numerical_PPT_candidate=bool(minimum>=-1e-6 and error<1e-5 and abs(np.trace(rho)-1)<1e-6),
                biseparable_proved=False,infeasibility_certified=False),values


def biseparable_moment_fit(target: np.ndarray, operators: np.ndarray,
                           parties: Sequence[Sequence[int]], *, iterations: int = 30,
                           seed: int = 2026101307, tolerance: float = 1e-7) -> tuple[dict,dict]:
    """Explicit convex mixture of product-across-a-cut pure states.

    Column generation uses bounded alternating product-state minimization.
    A small independently replayed residual supplies a constructive compatible
    state to that tolerance; failure is inconclusive, never proof of GME.
    Real product atoms suffice for a valid inner approximation; no completeness
    claim is made. Keep every atom's actual cut and factor vectors for replay.
    """
    d = operators.shape[-1]; n = d.bit_length()-1; rng = np.random.default_rng(seed)
    cuts = canonical_bipartitions(parties,n); atoms=[]; factors=[]; cut_ids=[]
    def add(a,b,k):
        a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); cut=cuts[k]
        order=cut+tuple(i for i in range(n) if i not in cut)
        v=np.outer(a,b).reshape((2,)*n).transpose(np.argsort(order)).reshape(d)
        atoms.append(v); factors.append((a,b)); cut_ids.append(k)
    # Complete computational product basis is an explicit feasible simplex.
    dl=2**len(cuts[0]); dr=d//dl
    for i in range(dl):
        for j in range(dr): add(np.eye(dl)[i],np.eye(dr)[j],0)
    for k,cut in enumerate(cuts):
        dl=2**len(cut); dr=d//dl
        for _ in range(2): add(rng.normal(size=dl),rng.normal(size=dr),k)
    accepted_weights=np.r_[np.full(d,1/d),np.zeros(len(atoms)-d)]
    last_status='not_started'
    for iteration in range(iterations+1):
        v=np.array(atoms); columns=np.einsum('ai,kij,aj->ka',v,operators,v,optimize=True).real
        count=len(v); objective=np.r_[np.zeros(count),1.]
        aub=np.vstack((np.column_stack((columns,-np.ones(len(target)))),
                       np.column_stack((-columns,-np.ones(len(target))))))
        fit=linprog(objective,A_ub=aub,b_ub=np.r_[target,-target],
                    A_eq=np.array([np.r_[np.ones(count),0.]]),b_eq=[1.],bounds=(0,None),method='highs-ipm',
                    options=dict(time_limit=5.,maxiter=2000,ipm_optimality_tolerance=1e-10))
        last_status=str(fit.message)
        # The previous explicit convex mixture remains valid if a solve stalls.
        # There is no inference of infeasibility from an optimization failure.
        if not fit.success: break
        accepted_weights=np.asarray(fit.x[:-1]).copy()
        if fit.fun <= tolerance or iteration == iterations: break
        dual=fit.ineqlin.marginals[:len(target)]-fit.ineqlin.marginals[len(target):]
        h=np.einsum('k,kij->ij',dual,operators).real
        # Pricing seeks MAX dual dot moment. Each party update is a small eigh.
        for k,cut in enumerate(cuts):
            order=cut+tuple(i for i in range(n) if i not in cut); dl=2**len(cut); dr=d//dl
            hh=h.reshape((2,)*(2*n)).transpose(order+tuple(n+i for i in order)).reshape(dl,dr,dl,dr)
            best=None
            for _ in range(2):
                b=rng.normal(size=dr); b/=np.linalg.norm(b)
                for _ in range(10):
                    a=np.linalg.eigh(np.einsum('j,ijkl,l->ik',b,hh,b))[1][:,-1]
                    b=np.linalg.eigh(np.einsum('i,ijkl,k->jl',a,hh,a))[1][:,-1]
                val=float(np.einsum('i,j,ijkl,k,l',a,b,hh,a,b))
                if best is None or val>best[0]: best=(val,a,b)
            add(best[1],best[2],k)
    weights=np.maximum(accepted_weights,0); weights/=weights.sum()
    selected=np.flatnonzero(weights>0); v=np.array(atoms)[selected]; weights=weights[selected]
    rho=np.einsum('a,ai,aj->ij',weights,v,v)
    error=float(np.max(abs(np.einsum('kij,ji->k',operators,rho).real-target)))
    factor_error=0.
    for idx,vector in zip(selected,v):
        cut=cuts[cut_ids[idx]]; order=cut+tuple(i for i in range(n) if i not in cut)
        factor_error=max(factor_error,float(np.linalg.svd(vector.reshape((2,)*n).transpose(order).reshape(2**len(cut),-1),compute_uv=False)[1:].max(initial=0)))
    return dict(moment_residual_max=error,tolerance=tolerance,
                compatible_to_declared_tolerance=bool(error<=tolerance and factor_error<1e-12),
                exact_moment_identity_claim=False,explicit_biseparable_state=True,
                component_count=len(v),iterations=iteration,product_factor_error=factor_error,
                last_LP_status=last_status,LP_seconds_cap=5.,LP_method='highs-ipm',
                failed_search_excludes_nothing=True),dict(rho=rho,weights=weights,vectors=v,
                    cuts=np.array([sum(1<<i for i in cuts[cut_ids[idx]]) for idx in selected]))
