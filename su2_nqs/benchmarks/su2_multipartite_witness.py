"""Bounded SU(2)-invariant fully-decomposable multipartite witnesses.

This offline readout never twirls, projects, or repairs its input state.  For
every nontrivial bipartition of the declared spatial PARTIES it imposes
``W = P_M + partial_transpose(Q_M, M)``, ``P_M >= 0``, ``0 <= Q_M <= I``.
There is deliberately NO upper bound on P_M.  A negative expectation of a
feasible W certifies genuine entanglement of those parties, not necessarily
of all individual spins.  A zero bound does not exclude entanglement.

Only W is restricted to the SU(2)-invariant observable algebra.  The
auxiliary Q matrices must remain unrestricted because partial transpose is
not SU(2) covariant in the same representation.  CVXPY/SCS are optional and
are imported only by the fitting function; frozen linear readout uses NumPy.
Bits are down/up, site zero MSB; Schur labels q=2S and m=2M are integers.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from .regional_su2_readout import regional_schur_columns


def _matrix(value: np.ndarray, *, normalized: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    d = len(matrix) if matrix.ndim == 2 else 0
    if (not 4 <= d <= 64 or d & (d-1) or matrix.shape != (d, d)
            or not np.isfinite(matrix).all()):
        raise ValueError('finite square matrix on 2..6 physical spins required')
    if np.linalg.norm(matrix-matrix.conj().T) > 1e-10:
        raise ValueError('Hermitian matrix required; input is not repaired')
    if normalized:
        if abs(np.trace(matrix)-1) > 1e-10 or np.linalg.eigvalsh(matrix).min() < -1e-12:
            raise ValueError('positive unit-trace RDM required; no input repair')
    return matrix


def _sites(sites: Sequence[int], n: int) -> tuple[int, ...]:
    result = tuple(sites)
    if (not result or len(result) >= n or len(set(result)) != len(result)
            or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, Integral)
                   or not 0 <= s < n for s in result)):
        raise ValueError('nonempty proper subset of distinct integer sites required')
    return tuple(sorted(map(int, result)))


@lru_cache(maxsize=128)
def _partial_transpose_indices(n: int, sites: tuple[int, ...]) -> np.ndarray:
    d = 2**n
    axes = list(range(2*n))
    for site in sites:
        axes[site], axes[n+site] = axes[n+site], axes[site]
    result = np.arange(d*d).reshape((2,)*(2*n)).transpose(axes).reshape(d, d)
    result.setflags(write=False)
    return result


def partial_transpose(matrix: np.ndarray, sites: Sequence[int]) -> np.ndarray:
    """Transpose the declared physical-site factors, in the fixed MSB basis."""
    matrix = _matrix(matrix)
    cut = _sites(sites, len(matrix).bit_length()-1)
    return matrix.reshape(-1)[_partial_transpose_indices(len(matrix).bit_length()-1, cut)]


def canonical_bipartitions(parties: Sequence[Sequence[int]], n_sites: int
                          ) -> tuple[tuple[int, ...], ...]:
    """Return all 2**(k-1)-1 inequivalent cuts of k disjoint covering parties.

    Canonical representatives contain the first declared party.  For three
    spatial parties this gives three cuts, regardless of sites per party.
    """
    if (isinstance(n_sites, (bool, np.bool_)) or not isinstance(n_sites, Integral)
            or not 2 <= n_sites <= 6):
        raise ValueError('integer n_sites between 2 and 6 required')
    blocks = tuple(_sites(party, int(n_sites)) for party in parties)
    if (not 2 <= len(blocks) <= n_sites
            or sorted(site for block in blocks for site in block) != list(range(n_sites))):
        raise ValueError('parties must be disjoint and cover every physical site')
    result = []
    for count in range(len(blocks)-1):
        for selected in combinations(range(1, len(blocks)), count):
            result.append(tuple(sorted(site for i in (0, *selected) for site in blocks[i])))
    return tuple(result)


@lru_cache(maxsize=10)
def su2_invariant_witness_basis(n_sites: int, time_reversal_even: bool = False
                                ) -> np.ndarray:
    """Return an HS-orthonormal real-parameter Hermitian SU(2) operator basis.

    The six-site full algebra has 132 Hermitian directions, or 76 real/TR-even
    directions in the existing real CG convention.  A real basis restriction
    is explicit; it is not silently applied to complex neural states.
    """
    if (isinstance(n_sites, (bool, np.bool_)) or not isinstance(n_sites, Integral)
            or not 2 <= n_sites <= 6 or not isinstance(time_reversal_even, bool)):
        raise ValueError('integer 2..6 n_sites and boolean TR restriction required')
    u, labels = regional_schur_columns(int(n_sites))
    operators = []
    for q in range(n_sites % 2, n_sites+1, 2):
        groups = [[i for i, label in enumerate(labels) if label[:2] == (q, m)]
                  for m in range(-q, q+1, 2)]
        size = len(groups[0])
        matrices = []
        for a in range(size):
            x = np.zeros((size, size), complex); x[a, a] = 1
            matrices.append(x)
        for a, b in combinations(range(size), 2):
            x = np.zeros((size, size), complex)
            x[a, b] = x[b, a] = 1/np.sqrt(2)
            matrices.append(x)
            if not time_reversal_even:
                x = np.zeros((size, size), complex)
                x[a, b], x[b, a] = 1j/np.sqrt(2), -1j/np.sqrt(2)
                matrices.append(x)
        for x in matrices:
            physical = sum(u[:, ids]@x@u[:, ids].conj().T for ids in groups)/np.sqrt(q+1)
            operators.append(physical)
    result = np.stack(operators)
    result.setflags(write=False)
    return result


def _spectral_enclosure(matrix: np.ndarray) -> tuple[float, float, dict[str, float]]:
    """Conservative floating-point eigenvalue enclosure, not interval arithmetic.

    Reconstruction/orthogonality residuals bound the exact spectral error of
    the computed decomposition; an additional roundoff allowance covers the
    matrix operations used in this audit.  The advertised certificate remains
    a numerically verified certificate, not a formal interval-arithmetic proof.
    """
    e, v = np.linalg.eigh(matrix)
    residual = float(np.linalg.norm(matrix-(v*e)@v.conj().T))
    orthogonality = float(np.linalg.norm(v.conj().T@v-np.eye(len(e))))
    guard = 64*np.finfo(float).eps*len(e)*max(1., float(np.linalg.norm(matrix)))
    error = residual+orthogonality*float(np.max(abs(e)))+guard
    return float(e[0]-error), float(e[-1]+error), dict(
        eigenvalue_min=float(e[0]), eigenvalue_max=float(e[-1]),
        conservative_error_allowance=error, decomposition_residual=residual,
        eigenvector_orthogonality=orthogonality)


@dataclass(frozen=True)
class WitnessCertificate:
    """Frozen physical witness and all auxiliary decomposition certificates."""

    witness: np.ndarray
    q_operators: tuple[np.ndarray, ...]
    cuts: tuple[tuple[int, ...], ...]
    diagnostics: dict[str, Any]


def _parties_from_complete_cuts(cuts: tuple[tuple[int, ...], ...], n: int
                               ) -> tuple[tuple[int, ...], ...]:
    """Infer spatial atoms and reject missing or duplicate-complement cuts."""
    universe = frozenset(range(n))
    def equivalent(cut):
        subset = frozenset(cut)
        return min(tuple(sorted(subset)), tuple(sorted(universe-subset)))
    signatures: dict[tuple[bool, ...], list[int]] = {}
    for site in range(n):
        signatures.setdefault(tuple(site in cut for cut in cuts), []).append(site)
    parties = tuple(tuple(sites) for sites in signatures.values())
    actual = {equivalent(cut) for cut in cuts}
    required = {equivalent(cut) for cut in canonical_bipartitions(parties, n)}
    if len(actual) != len(cuts) or actual != required:
        raise ValueError('complete nonduplicated bipartitions of the inferred parties required')
    return parties


def validate_witness_certificate(certificate: WitnessCertificate) -> dict[str, Any]:
    """Recompute PSD feasibility from frozen arrays; do not trust a stored flag."""
    w = _matrix(certificate.witness)
    n = len(w).bit_length()-1
    cuts = tuple(_sites(cut, n) for cut in certificate.cuts)
    if len(cuts) != len(certificate.q_operators) or not cuts:
        raise ValueError('one auxiliary matrix per cut required')
    parties = _parties_from_complete_cuts(cuts, n)
    rows = []
    for cut, value in zip(cuts, certificate.q_operators):
        q = _matrix(value)
        if q.shape != w.shape:
            raise ValueError('auxiliary matrix and witness shapes differ')
        pl, _, _ = _spectral_enclosure(w-partial_transpose(q, cut))
        ql, qu, _ = _spectral_enclosure(q)
        if pl < 0 or ql < 0 or qu > 1:
            raise ValueError('frozen witness lacks a numerically feasible decomposition')
        rows.append(dict(cut=list(cut), P_lower_bound=pl, Q_lower_bound=ql, Q_upper_bound=qu))
    return dict(feasible_to_numerical_margin=True, formal_interval_certificate=False,
                inferred_parties=[list(party) for party in parties], cuts=rows)


def repair_fully_decomposable_witness(
    witness: np.ndarray, q_operators: Sequence[np.ndarray],
    cuts: Sequence[Sequence[int]], *, safety_margin: float = 1e-9,
) -> WitnessCertificate:
    """Conservatively repair the WITNESS, never the physical state.

    Each auxiliary Q is spectrally clipped, given a PSD margin, and scaled to
    have an upper bound below I.  Afterwards one common positive delta*I is
    added to W so every P=W-PT(Q) is positive.  For unit-trace rho the witness
    expectation cost of this final repair is exactly delta.  No P<=I is used.
    """
    w = _matrix(witness).copy()
    n, d = len(w).bit_length()-1, len(w)
    if not np.isfinite(safety_margin) or not 0 < safety_margin < 1e-3:
        raise ValueError('finite positive safety_margin below 1e-3 required')
    canonical = tuple(_sites(cut, n) for cut in cuts)
    if (not canonical or len(canonical) != len(q_operators)
            or len(set(canonical)) != len(canonical)):
        raise ValueError('one auxiliary Q per distinct nonempty cut required')
    parties = _parties_from_complete_cuts(canonical, n)
    repaired = []; changes = []
    for value in q_operators:
        q = _matrix(value)
        if q.shape != w.shape:
            raise ValueError('all auxiliary matrices must match W')
        e, v = np.linalg.eigh(q)
        clipped = (v*np.clip(e, 0., 1.))@v.conj().T
        orth = float(np.linalg.norm(v.conj().T@v-np.eye(d)))
        guard = 64*np.finfo(float).eps*d*max(1., float(np.linalg.norm(clipped)))
        shift = safety_margin+4*guard
        scale = 1+orth+2*shift
        safe = (clipped+shift*np.eye(d))/scale
        # This only removes roundoff antihermiticity in an auxiliary witness.
        safe = (safe+safe.conj().T)/2
        repaired.append(safe)
        changes.append(dict(auxiliary_frobenius_change=float(np.linalg.norm(safe-q)),
                            spectral_shift=shift, spectral_scale=scale,
                            original_minimum_eigenvalue=float(e[0]),
                            original_maximum_eigenvalue=float(e[-1])))
    p_before = [w-partial_transpose(q, cut) for q, cut in zip(repaired, canonical)]
    lower = [_spectral_enclosure(p)[0] for p in p_before]
    delta = max(0., safety_margin-min(lower))
    w += delta*np.eye(d)
    # The audit's norm-dependent roundoff allowance may grow after the
    # identity shift. Re-evaluate the final matrices, never certify using
    # the pre-repair allowance. The extra cost remains one recorded delta.
    for _ in range(8):
        final_lower = min(_spectral_enclosure(w-partial_transpose(q, cut))[0]
                          for q, cut in zip(repaired, canonical))
        if final_lower >= 0:
            break
        extra = safety_margin-final_lower
        w += extra*np.eye(d)
        delta += extra
    rows = []
    feasible = True
    for q, cut, change in zip(repaired, canonical, changes):
        p = w-partial_transpose(q, cut)
        pl, _, pd = _spectral_enclosure(p)
        ql, qu, qd = _spectral_enclosure(q)
        ok = pl >= 0 and ql >= 0 and qu <= 1
        feasible &= ok
        rows.append(dict(cut=list(cut), P=pd, Q=qd, P_lower_bound=pl,
                         Q_lower_bound=ql, Q_upper_bound=qu,
                         feasible_to_numerical_margin=ok, **change))
    if not feasible:
        raise RuntimeError('conservative witness repair did not produce feasible decompositions')
    for value in (w, *repaired):
        value.setflags(write=False)
    return WitnessCertificate(w, tuple(repaired), canonical, dict(
        identity_shift=delta, unit_trace_expectation_repair_cost=delta,
        inferred_parties=[list(party) for party in parties],
        safety_margin=safety_margin, cuts=rows,
        feasible_to_numerical_margin=True, formal_interval_certificate=False,
        P_upper_bound_imposed=False, physical_state_repaired=False))


def evaluate_frozen_witness(certificate: WitnessCertificate, rho: np.ndarray
                           ) -> dict[str, Any]:
    """Evaluate a frozen linear witness on an actual, unrepaired physical RDM."""
    matrix = _matrix(rho, normalized=True)
    if matrix.shape != certificate.witness.shape:
        raise ValueError('RDM and frozen witness shapes differ')
    feasible = validate_witness_certificate(certificate)
    value = np.trace(matrix@certificate.witness)
    return dict(expectation=float(value.real), expectation_imaginary=float(value.imag),
        entanglement_lower_bound=max(0., -float(value.real)),
        strictly_negative_expectation=bool(value.real < 0),
        numerical_feasibility=feasible['feasible_to_numerical_margin'],
        inferred_parties=feasible['inferred_parties'],
        cuts=[list(cut) for cut in certificate.cuts],
        scope='Genuine entanglement of the declared spatial parties; zero does not exclude entanglement. No state repair or SU2 projection.',
        strict_entanglement_spectrum_reported=False)


def frozen_witness_chain_audit(
    certificate: WitnessCertificate, matrices: np.ndarray, chain_ids: np.ndarray,
) -> dict[str, Any]:
    """Linear independent-WHOLE-chain readout; fitting chains must be disjoint.

    Negative bound estimates are retained.  This routine cannot verify chain
    provenance; the workflow must seal the fit and audit identities separately.
    """
    a, groups = np.asarray(matrices, complex), np.asarray(chain_ids)
    feasible = validate_witness_certificate(certificate)
    d = len(certificate.witness)
    if (a.ndim != 3 or a.shape[1:] != (d, d) or groups.shape != (len(a),)
            or not np.isfinite(a).all()):
        raise ValueError('finite matching per-record RDMs and explicit chain IDs required')
    ids, counts = np.unique(groups, return_counts=True)
    if len(ids) < 3 or len(set(counts)) != 1:
        raise ValueError('at least three equally weighted independent whole chains required')
    means = np.stack([a[groups == chain].mean(axis=0) for chain in ids])
    for mean in means:
        _matrix(mean, normalized=True)
    values = np.einsum('cij,ji->c', means, certificate.witness)
    if np.max(abs(values.imag)) > 1e-10:
        raise ValueError('frozen Hermitian witness produced complex expectations')
    value = float(values.real.mean())
    se = float(values.real.std(ddof=1)/np.sqrt(len(ids)))
    return dict(expectation=value, whole_chain_SE=se,
        signed_entanglement_lower_estimate=-value,
        two_SE_lower_bound=max(0., -value-2*se),
        independent_chains=len(ids), records_per_chain=int(counts[0]),
        inferred_parties=feasible['inferred_parties'],
        chain_expectations=values.real.tolist(),
        negative_estimates_retained=True, normal_approximation_not_rigorous_CI=True,
        independent_of_witness_fitting_must_be_verified_by_caller=True,
        scope='Frozen linear witness with whole-chain uncertainty; not a plug-in SDP negativity estimate.')


def solve_su2_multipartite_witness(
    rho: np.ndarray, parties: Sequence[Sequence[int]], *, solver: str = 'SCS',
    time_reversal_even: bool = False, solver_options: dict[str, Any] | None = None,
    safety_margin: float = 1e-9, operator_span: str = 'full',
) -> tuple[dict[str, Any], WitnessCertificate]:
    """Fit a bounded SU2-invariant fully-decomposable witness to one RDM.

    ``max(0,-Tr(rho W_repaired))`` is a feasible lower bound, not a claimed
    optimal GMN.  A solver timeout/inaccurate status can still yield a useful
    repaired witness.  Numerical status is never used as a PSD certificate.
    The caller sets a finite SCS ``time_limit_secs``/``max_iters`` budget.
    ``operator_span='low_order'`` instead restricts W to I, ALL physical pairs
    and symmetrized products of two pairs (61 directions on six sites).  The
    variant ``'low_order_and_spin_weights'`` adds every total-spin projector
    (62 directions on six sites), matching the maximum-entropy constraints.
    comparison then tests the power of this declared LINEAR witness family,
    not every possible nonlinear inference from low-order observations.
    """
    matrix = _matrix(rho, normalized=True)
    try:
        import cvxpy as cp
    except ImportError as error:
        raise ImportError('witness fitting needs optional CVXPY and an SDP solver; frozen audits only need NumPy') from error
    d, n = len(matrix), len(matrix).bit_length()-1
    cuts = canonical_bipartitions(parties, n)
    if operator_span == 'full':
        basis = su2_invariant_witness_basis(n, time_reversal_even)
    elif operator_span in ('low_order', 'low_order_and_spin_weights'):
        from .spin_operator_information import _low_order_span
        orthogonal, _, _ = _low_order_span(n)
        if operator_span == 'low_order_and_spin_weights':
            u, labels = regional_schur_columns(n)
            projectors = []
            for q in range(n % 2, n+1, 2):
                indices = [i for i, label in enumerate(labels) if label[0] == q]
                projectors.append((u[:, indices]@u[:, indices].conj().T).real.reshape(-1))
            bank = np.column_stack((orthogonal, *projectors))
            left, singular, _ = np.linalg.svd(bank, full_matrices=False)
            orthogonal = left[:, singular > 1e-11*singular[0]]
        basis = orthogonal.T.reshape(orthogonal.shape[1], d, d)
    else:
        raise ValueError("operator_span must be 'full', 'low_order', or 'low_order_and_spin_weights'")
    coefficients = cp.Variable(len(basis))
    effective_real = time_reversal_even or operator_span != 'full'
    if effective_real:
        physical_basis = basis.real
    else:
        physical_basis = basis
    witness = cp.reshape(coefficients @ physical_basis.reshape(len(basis), -1), (d, d), order='C')
    auxiliary = []
    constraints = []
    for cut in cuts:
        q = cp.Variable((d, d), symmetric=True) if effective_real else cp.Variable((d, d), hermitian=True)
        auxiliary.append(q)
        transposed = cp.reshape(cp.reshape(q, (d*d,), order='C')[_partial_transpose_indices(n, cut).reshape(-1)], (d, d), order='C')
        constraints.extend((q >> 0, np.eye(d)-q >> 0, witness-transposed >> 0))
    target = np.einsum('kij,ji->k', physical_basis, matrix).real
    problem = cp.Problem(cp.Minimize(target @ coefficients), constraints)
    options = dict(solver_options or {})
    if solver.upper() == 'SCS':
        if 'eps' not in options:
            options.setdefault('eps_abs', 2e-6)
            options.setdefault('eps_rel', 2e-6)
        options.setdefault('max_iters', 10000)
        options.setdefault('time_limit_secs', 120.)
    problem.solve(solver=solver, **options)
    if coefficients.value is None or any(q.value is None for q in auxiliary):
        raise RuntimeError(f'SDP returned no finite candidate witness: {problem.status}')
    raw = np.einsum('k,kij->ij', coefficients.value, physical_basis)
    certificate = repair_fully_decomposable_witness(raw, [q.value for q in auxiliary], cuts,
                                                   safety_margin=safety_margin)
    result = evaluate_frozen_witness(certificate, matrix)
    result.update(solver_status=str(problem.status), solver=solver,
        solver_options=options, solver_solve_seconds=problem.solver_stats.solve_time,
        solver_iterations=problem.solver_stats.num_iters,
        raw_solver_objective=float(problem.value),
        repaired_objective=result['expectation'],
        invariant_parameters=len(basis), time_reversal_even_restriction=effective_real,
        operator_span=operator_span,
        parties=[list(party) for party in parties],
        optimality_certified=False, physical_state_twirl_performed=False,
        certificate=certificate.diagnostics)
    return result, certificate


__all__ = ['WitnessCertificate', 'partial_transpose', 'canonical_bipartitions',
           'su2_invariant_witness_basis', 'repair_fully_decomposable_witness',
           'validate_witness_certificate',
           'evaluate_frozen_witness', 'frozen_witness_chain_audit',
           'solve_su2_multipartite_witness']
