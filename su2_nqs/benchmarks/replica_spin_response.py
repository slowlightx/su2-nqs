"""Two-copy physical spin response without a dense reduced density matrix.

This observable is valid for any pure state; it does not assume SU(2).
The Born samples must be independently drawn pairs. Fixed-magnetization
forbidden swaps contribute zero, not a postselected conditional average.
"""
from __future__ import annotations

from math import comb
from numbers import Integral
import numpy as np


def row_pi_bonds(row: int, columns: int) -> tuple[tuple[int, int, float], ...]:
    """Periodic row bonds for Q++=sum_c (-1)^c S+_(r,c) S+_(r,c+1).

    This is the doubled tensor component q=m=4 (without a spherical 1/2).
    Even columns are required for winding momentum pi.
    """
    if (isinstance(row, bool) or not isinstance(row, Integral) or row < 0
            or isinstance(columns, bool) or not isinstance(columns, Integral)
            or columns < 4 or columns % 2):
        raise ValueError('nonnegative integer row and even columns >=4 required')
    return tuple((row*columns+c, row*columns+(c+1)%columns, float((-1)**c))
                 for c in range(columns))


def double_spin_connections(
    states: np.ndarray, bonds: tuple[tuple[int, int, float], ...], *, delta_m: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of sum w S+_i S+_j, or its adjoint for doubled delta_m=-4.

    Return targets, source-row indices and O[row,column]. A raising row has
    two UP spins; its connected column has two DOWN spins. This row/column
    distinction matters for swapped configurations outside fixed-Sz support.
    """
    x = np.asarray(states)
    if (x.ndim != 2 or not np.isin(x, (0, 1)).all()
            or isinstance(delta_m, bool) or delta_m not in (-4, 4)):
        raise ValueError('binary rows and doubled charge change +/-4 required')
    combined: dict[tuple[int, int], float] = {}
    for edge in bonds:
        if (len(edge) != 3 or any(isinstance(i, bool) or not isinstance(i, Integral)
                or not 0 <= i < x.shape[1] for i in edge[:2])
                or edge[0] == edge[1] or not np.isreal(edge[2]) or not np.isfinite(edge[2])):
            raise ValueError('legal distinct sites and real finite bond weights required')
        a, b, weight = edge
        key = (min(a, b), max(a, b))
        combined[key] = combined.get(key, 0.) + float(weight)
    targets, sources, coefficients = [], [], []
    value = 1 if delta_m == 4 else 0
    for (a, b), weight in sorted(combined.items()):
        if weight == 0: continue
        ids = np.flatnonzero((x[:, a] == value) & (x[:, b] == value))
        y = x[ids].copy(); y[:, [a, b]] = 1-value
        targets.append(y); sources.append(ids); coefficients.append(np.full(len(ids), weight))
    if not targets:
        return np.empty((0, x.shape[1]), x.dtype), np.empty(0, int), np.empty(0, complex)
    return np.concatenate(targets), np.concatenate(sources), np.concatenate(coefficients)


def double_spin_vector_action(
    vectors: np.ndarray, bonds: tuple[tuple[int, int, float], ...], *, delta_m: int = 4,
) -> np.ndarray:
    """Apply the same physical operator to bounded dense reference vectors.

    Site zero is MSB; this does not construct a dense operator or an RDM.
    """
    v = np.asarray(vectors, complex)
    if v.ndim != 2 or len(v) < 2 or len(v) & (len(v)-1):
        raise ValueError('columns of complete binary-basis vectors required')
    n = len(v).bit_length()-1
    bits = ((np.arange(len(v))[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.int8)
    target, source, weights = double_spin_connections(bits, bonds, delta_m=delta_m)
    packed = target @ (1 << np.arange(n-1, -1, -1))
    result = np.zeros_like(v)
    np.add.at(result, source, weights[:, None]*v[packed])
    return result


def vector_chirality_z_connections(
    states: np.ndarray, bonds: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of K=sum_(i,j) (S_i cross S_j)^z, with bit 0 down/1 up.

    Return connected configurations, source-row indices, and K[row,column].
    Parallel spins give no entry. Repeated oriented bonds are combined before
    returning targets. No amplitude division is used: a zero at the row does
    not imply that (K psi)[row] vanishes. Site zero remains the MSB.
    """
    x = np.asarray(states)
    if x.ndim != 2 or not np.isin(x, (0, 1)).all():
        raise ValueError('binary physical configurations required')
    if (not bonds or any(len(b) != 2 or b[0] == b[1]
            or any(isinstance(i, bool) or not isinstance(i, Integral)
                   or not 0 <= i < x.shape[1] for i in b) for b in bonds)):
        raise ValueError('nonempty legal oriented physical bonds required')
    combined: dict[tuple[int, int], int] = {}
    for a, b in bonds:
        edge = (min(a, b), max(a, b))
        combined[edge] = combined.get(edge, 0)+(1 if a < b else -1)
    targets, sources, coefficients = [], [], []
    for (a, b), weight in sorted(combined.items()):
        if not weight:
            continue
        ids = np.flatnonzero(x[:, a] != x[:, b])
        y = x[ids].copy()
        y[:, a], y[:, b] = x[ids, b], x[ids, a]
        targets.append(y)
        sources.append(ids)
        coefficients.append(.5j*weight*(x[ids, a].astype(float)-x[ids, b]))
    if not targets:
        return np.empty((0, x.shape[1]), x.dtype), np.empty(0, int), np.empty(0, complex)
    return np.concatenate(targets), np.concatenate(sources), np.concatenate(coefficients)


def operator_action_over_sample(
    connected_logs: np.ndarray, sources: np.ndarray, coefficients: np.ndarray,
    sampled_logs: np.ndarray,
) -> np.ndarray:
    """Compute (K psi)[u]/psi[x] using sampled nonzero denominators x.

    The possibly nodal operator-action row u need not equal sample x. This is
    essential for SWAP(K tensor K); dividing by psi[u] would lose nodal terms.
    """
    logs, ids, c, old = (np.asarray(connected_logs, complex), np.asarray(sources),
                        np.asarray(coefficients, complex), np.asarray(sampled_logs, complex))
    if (old.ndim != 1 or not np.isfinite(old).all() or logs.ndim != 1
            or ids.shape != logs.shape or c.shape != logs.shape
            or ids.dtype.kind not in 'iu' or not np.isfinite(c).all()
            or np.any(ids < 0) or np.any(ids >= len(old))
            or np.any(np.isnan(logs)) or np.any(np.isposinf(logs.real))):
        raise ValueError('finite sampled denominators and legal sparse rows required')
    result = np.zeros(len(old), complex)
    terms = c*np.exp(logs-old[ids])
    if not np.isfinite(terms).all():
        raise FloatingPointError('operator tail overflow; do not clip')
    np.add.at(result, ids, terms)
    return result


def hermitian_replica_numerator(actions_over_samples: np.ndarray) -> np.ndarray:
    """Local record for SWAP_A (K_A tensor K_A), K_A Hermitian.

    Input columns are (K psi)[y_A,x_B]/psi[x] and
    (K psi)[x_A,y_B]/psi[y]. Multiply WITHOUT a conjugate. Complex-state
    exhaustive oracles, rather than real-state tests alone, fix this rule.
    Its Born mean is Tr(rho_A K_A rho_A K_A), not a thermal susceptibility.
    """
    a = np.asarray(actions_over_samples, complex)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError('two finite operator actions per independently sampled pair required')
    return np.prod(a, axis=1)


def swap_response_records(log_original: np.ndarray, log_swapped: np.ndarray,
                          spin_fourier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return SWAP and SWAP*O(x_A)^*O(y_A), with O=sum exp(ik.r) Sz.

    Log amplitudes have shape (records,2), Fourier values (records,2,momenta).
    Actual sampled denominators must be nonzero. No clipping of rare ratios.
    """
    old = np.asarray(log_original, complex)
    new = np.asarray(log_swapped, complex)
    o = np.asarray(spin_fourier, complex)
    if (old.ndim != 2 or old.shape[1] != 2 or new.shape != old.shape
            or o.ndim != 3 or o.shape[:2] != old.shape
            or not np.isfinite(old).all() or not np.isfinite(o).all()
            or np.any(np.isnan(new)) or np.any(np.isposinf(new.real))):
        raise ValueError('finite sampled logs and legal zero-support swapped logs required')
    weights = np.exp(np.sum(new-old, axis=1))
    if not np.isfinite(weights).all():
        raise FloatingPointError('swap tail overflow; never silently truncate weights')
    return weights, weights[:, None]*o[:, 0].conj()*o[:, 1]


def summarize_swap_response(weights: np.ndarray, numerators: np.ndarray,
                            independent_pair_labels: np.ndarray) -> dict:
    """Cluster jackknife ratio retaining numerator/denominator covariance.

    A cluster is a pair of independently initialized Markov chains; no chain
    may be reused in another pair. SE is descriptive, not a mixing guarantee.
    """
    w = np.asarray(weights, complex)
    n = np.asarray(numerators, complex)
    labels = np.asarray(independent_pair_labels)
    if (w.ndim != 1 or n.ndim != 2 or n.shape[0] != len(w)
            or labels.shape != w.shape or not np.isfinite(w).all()
            or not np.isfinite(n).all() or labels.dtype.kind not in 'iu'):
        raise ValueError('finite matched records and integer independent pair labels required')
    groups = np.unique(labels)
    if len(groups) < 4:
        raise ValueError('at least four independent chain pairs required')
    counts = np.array([np.count_nonzero(labels == g) for g in groups])
    if not np.all(counts == counts[0]):
        raise ValueError('equal observations per independent pair required')
    wg = np.array([w[labels == g].mean() for g in groups])
    ng = np.array([n[labels == g].mean(axis=0) for g in groups])
    purity = wg.real.mean()
    purity_se = float(wg.real.std(ddof=1)/np.sqrt(len(groups)))
    if purity <= 0:
        raise ValueError('nonpositive sampled purity; estimator unresolved')
    leave_w = (wg.real.sum()-wg.real)/(len(groups)-1)
    leave_n = (ng.real.sum(axis=0)-ng.real)/(len(groups)-1)
    if np.any(leave_w <= 0):
        raise ValueError('leave-one-chain purity unresolved; do not interpret response')
    leave = leave_n/leave_w[:, None]
    error = np.sqrt((len(groups)-1)/len(groups)*np.sum((leave-leave.mean(axis=0))**2, axis=0))
    tail_count = max(1, int(np.ceil(.01*len(w))))
    absolute = np.abs(w)
    return dict(purity=float(purity), purity_chain_pair_SE=purity_se,
        response=(ng.real.mean(axis=0)/purity).tolist(), response_chain_pair_SE=error.tolist(),
        denominator_resolved_at_3SE=bool(purity > 3*purity_se),
        imaginary_purity_mean=float(wg.imag.mean()),
        imaginary_purity_SE=float(wg.imag.std(ddof=1)/np.sqrt(len(groups))),
        imaginary_numerator_means=ng.imag.mean(axis=0).tolist(),
        zero_swap_fraction=float(np.mean(absolute == 0)),
        maximum_absolute_swap=float(absolute.max()),
        top_one_percent_absolute_swap_share=float(np.sort(absolute)[-tail_count:].sum()/max(absolute.sum(),1e-300)),
        independent_chain_pairs=len(groups), records=len(w), mixing_certified=False,
        definition='Tr(rho_A Sz(-k) rho_A Sz(k))/Tr(rho_A^2); unnormalized Fourier sum',
        strict_SU2_assumed=False, phase_classification=False)


def explicit_half_rdm_cost(n_region: int = 18) -> dict:
    """Exact entry counts for an even 2*n region globally at fixed Sz=0."""
    if isinstance(n_region, bool) or not isinstance(n_region, Integral) or n_region < 1:
        raise ValueError('positive integer region size required')
    d = 2**n_region
    entries = comb(2*n_region, n_region)
    multiplicities = {q: comb(n_region,(n_region-q)//2)
        -(comb(n_region,(n_region-q)//2-1) if q<n_region else 0)
        for q in range(n_region%2,n_region+1,2)}
    reduced_entries=sum(m*m for m in multiplicities.values())
    return dict(n_region=int(n_region), dimension=d,
        dense_complex128_GiB=d*d*16/2**30,
        charge_blocked_complex128_GiB=entries*16/2**30,
        charge_blocked_real64_GiB=entries*8/2**30,
        full_fixed_magnetization_amplitude_count=entries,
        largest_block_dimension=comb(n_region,n_region//2),
        strict_SU2_multiplicity_complex128_GiB=reduced_entries*16/2**30,
        strict_SU2_multiplicity_entries=reduced_entries,
        strict_SU2_compression_requires_actual_rotational_invariance=True,
        twirled_RDM_is_not_the_original_noninvariant_RDM=True,
        eigensolver_workspaces_included=False)
