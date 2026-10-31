"""Same-operator ordinary, Renyi-matched and inter-Schmidt correlations.

These are state/ansatz independent readouts, not QSL or entanglement witnesses.
No twirling, eigenvalue clipping or symmetry repair is performed. Magnetic
labels, when provided separately by a Schur analysis, use doubled integers.
"""
from __future__ import annotations

import numpy as np

from .chirality_covariance import cluster_jackknife, jackknife_error, symmetric_matrix


def diagonal_swap_controls(
    ordinary: np.ndarray, swap: np.ndarray, inter_schmidt_numerators: np.ndarray,
    labels: np.ndarray, *, squared_magnitude_orbit_invariant: bool,
) -> dict:
    """Replay diagonal-operator C/C2/M from the same independent pair cache.

    ``ordinary[:,k]`` is (|O_k(x_A)|²+|O_k(y_A)|²)/2 and ``swap`` may
    already be averaged over a relative-replica symmetry orbit. Reusing their
    product for Tr(rho_A² O_k†O_k) is legal only after independently checking
    that |O_k|² is invariant under EVERY averaged transformation. It is not a
    general product-of-orbit-averages identity. No SU(2) assumption is used.

    Whole-pair deletions preserve purity/numerator covariance in all ratios
    and differences. This scalar-channel control does not reconstruct a
    row-resolved matrix, selected Schmidt gap, or an entanglement witness.
    """
    c, w, m, groups = (np.asarray(ordinary), np.asarray(swap, complex),
                        np.asarray(inter_schmidt_numerators, complex), np.asarray(labels))
    if squared_magnitude_orbit_invariant is not True:
        raise ValueError('independently verified squared-magnitude orbit invariance required')
    if (c.ndim != 2 or c.shape != m.shape or w.shape != (len(c),)
            or groups.shape != w.shape or groups.dtype.kind not in 'iu'
            or not np.isrealobj(c) or not np.isfinite(c).all() or np.any(c < 0)
            or not np.isfinite(w).all() or not np.isfinite(m).all()):
        raise ValueError('finite matched nonnegative diagonal products and pair records required')
    mean_c, leave_c = cluster_jackknife(c, groups)
    mean_n, leave_n = cluster_jackknife(np.stack((w[:, None]*c, m), axis=1), groups)
    mean_w, leave_w = cluster_jackknife(w, groups)
    if mean_w.real <= 0 or np.any(leave_w.real <= 0):
        raise ValueError('pair-deleted purity unresolved; do not clip')
    values = np.stack((mean_c, mean_n[0].real/mean_w.real, mean_n[1].real/mean_w.real))
    leaves = np.stack((leave_c, leave_n[:, 0].real/leave_w.real[:, None],
                       leave_n[:, 1].real/leave_w.real[:, None]), axis=1)
    filters = {}
    if np.all(values[1] > 0) and np.all(leaves[:, 1] > 0):
        filters = dict(M_over_C2=(values[2]/values[1]).tolist(),
                       M_over_C2_SE=jackknife_error(leaves[:, 2]/leaves[:, 1]).tolist())
    return dict(order=['C', 'C2', 'M'], values=values.tolist(),
                values_SE=jackknife_error(leaves).tolist(),
                C2_minus_C=(values[1]-values[0]).tolist(),
                C2_minus_C_SE=jackknife_error(leaves[:, 1]-leaves[:, 0]).tolist(),
                M_minus_C2=(values[2]-values[1]).tolist(),
                M_minus_C2_SE=jackknife_error(leaves[:, 2]-leaves[:, 1]).tolist(),
                purity=float(mean_w.real), purity_SE=float(jackknife_error(leave_w.real)),
                imaginary_purity=float(mean_w.imag),
                imaginary_numerator_means=mean_n.imag.tolist(),
                independent_chain_pairs=len(np.unique(groups)), records=len(c),
                strict_SU2_assumed=False, squared_magnitude_orbit_invariance_required=True,
                new_amplitude_queries=0, **filters)


def schmidt_transition_window(
    initial_weights: np.ndarray, final_weights: np.ndarray,
    amplitudes: np.ndarray, purity: float,
) -> dict:
    """Resolve a measured charge-changing window without completing a spectrum.

    ``amplitudes[i,b,a] = <final_b|O_i|initial_a>``. The returned Gram
    matrix is the ONE-DIRECTION contribution with weights lambda_a lambda_b/P.
    For a raising operator it is not the full Hermitian-component M, and no
    factor-three isotropy or total-spin assignment is assumed. The supplied
    weights may cover only a stored window and must NOT be renormalized to it.
    ``purity`` must be that of the original, complete regional RDM.
    """
    a, b = np.asarray(initial_weights, float), np.asarray(final_weights, float)
    ops = np.asarray(amplitudes, complex)
    if (a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b)
            or ops.ndim != 3 or not len(ops) or ops.shape[1:] != (len(b), len(a))
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a < 0) or np.any(b < 0)
            or a.sum() > 1+2e-12 or b.sum() > 1+2e-12
            or not np.isfinite(ops).all() or not np.isfinite(purity) or not 0 < purity <= 1+2e-12):
        raise ValueError('physical unnormalized window weights, matched matrices and full purity required')
    weights = b[:, None]*a[None, :]/purity
    gram = np.einsum('ba,iba,jba->ij', weights, ops.conj(), ops).real
    strengths = np.abs(ops)**2
    total_probe = ops.sum(axis=0)  # coherent sum, never sum of intensities
    return dict(gram=gram, pair_weights=weights,
                per_operator_strengths=strengths,
                coherent_sum_strength=np.abs(total_probe)**2,
                coherent_sum_contributions=weights*np.abs(total_probe)**2,
                full_spectrum=False, strict_SU2_assumed=False)


def renyi2_product_numerator(
    action_over_original: np.ndarray, swapped_over_original: np.ndarray,
) -> np.ndarray:
    """Node-safe local records for Tr(rho_A**2 O_A), with Hermitian O_A.

    ``action[...,r]`` is (O psi)[swapped_r]/psi[original_r], NOT
    (O psi)[swapped_r]/psi[swapped_r]. The other input contains
    psi[swapped_r]/psi[original_r]. Both have a final two-replica axis.
    A symmetric replica average keeps the full complex local record.
    It remains correct when one swapped amplitude is zero but O psi is not.
    """
    action = np.asarray(action_over_original, complex)
    swap = np.asarray(swapped_over_original, complex)
    if (action.shape != swap.shape or action.ndim < 1 or action.shape[-1] != 2
            or not np.isfinite(action).all() or not np.isfinite(swap).all()):
        raise ValueError('matched finite arrays with a final two-replica axis required')
    return (action[..., 0]*swap[..., 1] + swap[..., 0]*action[..., 1])/2


def dense_schmidt_covariances(rho: np.ndarray, operators: np.ndarray) -> dict:
    """Exact bounded-RDM oracle for C, C2, M and D=C2-M.

    C_ij=Re Tr[rho {O_i,O_j}/2], C2 uses rho**2/Tr(rho**2), and
    M_ij=Re Tr[rho O_i rho O_j]/Tr(rho**2). The matrix D is the
    Wigner--Yanase skew-information bilinear form of the ESCORT state
    sigma=rho**2/Tr(rho**2), not of rho itself. These are uncentered moments;
    their use as connected C requires independently established zero means.
    """
    r = np.asarray(rho, complex)
    ops = np.asarray(operators, complex)
    if (r.ndim != 2 or r.shape[0] == 0 or r.shape[0] != r.shape[1] or ops.ndim != 3
            or len(ops) == 0
            or ops.shape[1:] != r.shape or not np.isfinite(r).all()
            or not np.isfinite(ops).all()):
        raise ValueError('finite square density matrix and Hermitian operators required')
    if (not np.allclose(r, r.conj().T, atol=2e-12, rtol=0)
            or abs(np.trace(r)-1) > 2e-12
            or np.linalg.eigvalsh(r)[0] < -2e-12
            or not np.allclose(ops, ops.conj().transpose(0, 2, 1), atol=2e-12, rtol=0)):
        raise ValueError('normalized positive Hermitian rho and Hermitian operators required')
    purity = float(np.trace(r@r).real)
    count = len(ops)
    c, c2, m = (np.empty((count, count)) for _ in range(3))
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            product = (a@b+b@a)/2
            c[i, j] = np.trace(r@product).real
            c2[i, j] = np.trace(r@r@product).real/purity
            m[i, j] = np.trace(r@a@r@b).real/purity
    commutators = [r@a-a@r for a in ops]
    independent = np.array([[np.vdot(a, b).real/(2*purity)
                             for b in commutators] for a in commutators])
    return dict(ordinary=c, renyi_matched=c2, inter_schmidt=m,
                commutator_metric=c2-m, independent_commutator_metric=independent,
                purity=purity, one_point=np.array([np.trace(r@a) for a in ops]),
                strict_SU2_assumed=False, phase_classification=False)


def compare_renyi_matched_covariances(
    product_numerators: np.ndarray, replica_numerators: np.ndarray,
    swap: np.ndarray, labels: np.ndarray,
) -> dict:
    """Common-chain-pair uncertainty of C2, M and D for three frozen operators.

    Both numerator arrays use (00,11,22,01,02,12) order; off-diagonal
    product_numerators use the symmetrized product. No PSD repair is allowed,
    including for a noisy D that should be PSD in the population limit.
    """
    a, b, w = map(np.asarray, (product_numerators, replica_numerators, swap))
    if a.ndim != 2 or a.shape[1] != 6 or b.shape != a.shape or w.shape != (len(a),):
        raise ValueError('two matching six-entry numerator arrays and scalar swap required')
    am, al = cluster_jackknife(a, labels)
    bm, bl = cluster_jackknife(b, labels)
    wm, wl = cluster_jackknife(w, labels)
    if wm.real <= 0 or np.any(wl.real <= 0):
        raise ValueError('purity unresolved; no clipped ratio')
    c2 = symmetric_matrix(am.real/wm.real)
    lc2 = symmetric_matrix(al.real/wl.real[:, None])
    m = symmetric_matrix(bm.real/wm.real)
    lm = symmetric_matrix(bl.real/wl.real[:, None])
    d, ld = c2-m, lc2-lm
    trace = float(np.trace(c2))
    ltrace = np.trace(lc2, axis1=-2, axis2=-1)
    result = dict(renyi_matched_matrix=c2.tolist(),
        renyi_matched_matrix_SE=jackknife_error(lc2).tolist(),
        inter_schmidt_matrix=m.tolist(), inter_schmidt_matrix_SE=jackknife_error(lm).tolist(),
        commutator_metric=d.tolist(), commutator_metric_SE=jackknife_error(ld).tolist(),
        raw_commutator_eigenvalues=np.linalg.eigvalsh(d).tolist(),
        product_numerator_imaginary_means=am.imag.tolist(),
        product_numerator_imaginary_SE=jackknife_error(al.imag).tolist(),
        purity=float(wm.real), purity_SE=float(jackknife_error(wl.real)),
        independent_chain_pairs=len(wl), PSD_clipping=False, mirror_repair=False,
        strict_SU2_assumed=False, phase_classification=False,
        definition='D_ij=-Re Tr([rho,O_i][rho,O_j])/(2 Tr(rho^2))=C2_ij-M_ij',
        skew_information_state='rho^2/Tr(rho^2), NOT rho')
    if trace > 0 and np.all(ltrace > 0):
        result.update(commutator_fraction=float(np.trace(d)/trace),
            commutator_fraction_SE=float(jackknife_error(
                np.trace(ld, axis1=-2, axis2=-1)/ltrace)))
    return result
