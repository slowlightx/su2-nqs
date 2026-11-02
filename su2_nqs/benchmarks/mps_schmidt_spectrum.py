"""Actual magnetic-charge Schmidt spectra of an unchanged finite U(1) MPS.

The output label is ``m=2*S_A^z``. Matching spectra at different ``m`` is a
necessary consequence of SU(2), not a verification of rotation covariance.
No doubled-spin ``q`` labels or entanglement tower are inferred here.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def gram_schmidt_spectrum(
    left_gram: np.ndarray, right_transposed_gram: np.ndarray,
    magnetic_labels: np.ndarray, *, n_region: int, tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compute physical RDM eigenvalues from ``L`` and ``R.T`` Gram metrics.

    For ``psi=sum_a |left_a>|right_a>``, ``L[a,b]=<left_a|left_b>``
    and ``R.T[a,b]=<right_b|right_a>``. The nonzero spectrum is that of
    ``sqrt(L) @ R.T @ sqrt(L) / trace(L @ R.T)``. Both metrics, including
    complex off-diagonal entries, are required for a noncanonical parent.
    Numerical null eigenvalues are reported; clipping is confined to the
    square root of a verified positive-semidefinite Gram matrix.
    """
    left = np.asarray(left_gram, dtype=np.complex128)
    right = np.asarray(right_transposed_gram, dtype=np.complex128)
    m = np.asarray(magnetic_labels)
    if (isinstance(n_region, bool) or not isinstance(n_region, Integral)
            or n_region < 1 or not 0 < tolerance < 1 or left.ndim != 2
            or left.shape[0] != left.shape[1] or right.shape != left.shape
            or m.shape != (left.shape[0],) or m.dtype.kind not in 'iu'
            or np.any(abs(m) > n_region) or np.any((m-n_region) % 2)
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError('finite square Grams and physical doubled magnetic charges required')
    left_scale = max(float(np.linalg.norm(left)), 1e-300)
    right_scale = max(float(np.linalg.norm(right)), 1e-300)
    hermitian_error = max(float(np.linalg.norm(left-left.conj().T))/left_scale,
                          float(np.linalg.norm(right-right.conj().T))/right_scale)
    if hermitian_error > tolerance:
        raise ValueError('Gram metrics must be Hermitian')
    left = .5*(left+left.conj().T)
    right = .5*(right+right.conj().T)
    forbidden = m[:, None] != m[None, :]
    charge_error = max(float(np.linalg.norm(left[forbidden]))/left_scale,
                       float(np.linalg.norm(right[forbidden]))/right_scale)
    if charge_error > tolerance:
        raise ValueError('magnetic charge blocks are not conserved')
    denominator = complex(np.einsum('ij,ji->', left, right))
    if (not np.isfinite(denominator) or denominator.real <= 0
            or abs(denominator.imag) > tolerance*denominator.real):
        raise ValueError('physical Gram contraction must have positive real norm')
    sectors = {}
    all_values = []
    reconstructed_norm = 0.
    most_negative_gram_relative = 0.
    for charge in np.arange(-n_region, n_region+1, 2):
        indices = np.flatnonzero(m == charge)
        values = np.empty(0)
        if len(indices):
            select = np.ix_(indices, indices)
            l, r = left[select], right[select]
            eig_l, vectors = np.linalg.eigh(l)
            eig_r = np.linalg.eigvalsh(r)
            negative = min(float(eig_l.min())/max(float(np.max(abs(eig_l))), 1e-300),
                           float(eig_r.min())/max(float(np.max(abs(eig_r))), 1e-300))
            most_negative_gram_relative = min(most_negative_gram_relative, negative)
            if negative < -tolerance:
                raise ValueError('physical Gram block is not positive semidefinite')
            root_l = (vectors*np.sqrt(np.maximum(eig_l, 0.))) @ vectors.conj().T
            physical = root_l @ r @ root_l
            physical = .5*(physical+physical.conj().T)
            values = np.linalg.eigvalsh(physical)[::-1]/denominator.real
            reconstructed_norm += float(np.trace(l @ r).real)
            if np.any(values < -tolerance):
                raise ValueError('physical Schmidt eigenvalues are negative')
        positive = values > 0
        energies = [float(-np.log(x)) if x > 0 else None for x in values]
        sectors[str(int(charge))] = dict(
            m=int(charge), virtual_dimension=len(indices),
            eigenvalues=values.tolist(), entanglement_energies=energies,
            weight=float(values.sum()), purity=float(values @ values),
            entropy_nats=float(-values[positive] @ np.log(values[positive])))
        all_values.extend(values.tolist())
    values = np.asarray(all_values)
    norm_error = abs(reconstructed_norm-denominator.real)/denominator.real
    spectral_trace_error = abs(float(values.sum())-1.)
    if max(norm_error, spectral_trace_error) > tolerance:
        raise ValueError('physical spectrum failed normalization closure')
    plus_minus_l1 = 0.
    for charge in range(1 if n_region % 2 else 2, n_region+1, 2):
        plus = np.asarray(sectors[str(charge)]['eigenvalues'])
        minus = np.asarray(sectors[str(-charge)]['eigenvalues'])
        size = max(len(plus), len(minus))
        plus_minus_l1 += float(np.sum(abs(np.pad(plus, (0, size-len(plus)))
                                        -np.pad(minus, (0, size-len(minus))))))
    positive = values > 0
    purity = float(values @ values)
    return dict(
        sectors=sectors, magnetic_labels=sorted(set(m.tolist())),
        norm_squared_without_external_mps_norm=float(denominator.real),
        spectral_trace_error=spectral_trace_error, gram_norm_closure_error=norm_error,
        gram_hermiticity_relative_error=hermitian_error,
        magnetic_block_relative_error=charge_error,
        most_negative_gram_relative_eigenvalue=most_negative_gram_relative,
        magnetic_spin_inversion_spectrum_l1=plus_minus_l1,
        von_neumann_entropy_nats=float(-values[positive] @ np.log(values[positive])),
        purity=purity, renyi2_entropy_nats=float(-np.log(purity)),
        actual_schmidt_eigenvalues_measured=True, stored_schmidt_weights_used=False,
        parent_canonicalized=False, strict_su2_verified=False,
        q_labels_assigned=False, entanglement_tower_identified=False,
        warning='m=2Sz spectra are physical. Multiplet coincidences alone do not verify SU(2), identify an entanglement tower, or classify a QSL.')


def finite_mps_prefix_schmidt_spectrum(
    psi: Any, cut: int, *, tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Use genuine finite LP/RP, with actual S factors, without changing psi.

    Virtual charges are gauge shifted. The mapping to physical prefix charge
    is ``m_A=vL(cut)-vL(0)+sum_{i<cut} tensor_qtotal(i)``. TeNPy is optional
    and imported only by this routine. Memory and time caps belong to the
    workflow because full environment preparation and block eigensolves are
    cubic in the virtual dimension.
    """
    if (psi.bc != 'finite' or isinstance(cut, bool) or not isinstance(cut, Integral)
            or not 0 < cut < psi.L or any(form is None for form in psi.form)
            or not 0 < tolerance < 1):
        raise ValueError('finite MPS with defined forms and an internal integer cut required')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPSEnvironment
    names = list(psi.chinfo.names)
    if names.count('2*Sz') != 1:
        raise ValueError('explicit conserved physical 2*Sz charge required')
    index = names.index('2*Sz')
    if int(psi.chinfo.mod[index]) != 1:
        raise ValueError('2*Sz must be U(1)')
    for i, site in enumerate(psi.sites):
        leg = psi.get_B(i, form=None).get_leg('p')
        sz = np.asarray(site.get_op('Sz').to_ndarray())
        flat = leg.to_qflat()[:, index]
        if (leg.qconj != 1 or sz.shape != (2, 2)
                or not np.allclose(sz, np.diag(np.diag(sz)), atol=tolerance, rtol=0)
                or not np.allclose(flat, 2*np.diag(sz), atol=tolerance, rtol=0)
                or set(flat.tolist()) != {-1, 1}):
            raise ValueError('physical legs must carry actual spin-half doubled Sz')
    virtual = psi.get_B(int(cut), form=None).get_leg('vL')
    left = psi.get_B(0, form=None).get_leg('vL')
    right = psi.get_B(psi.L-1, form=None).get_leg('vR')
    if (virtual.qconj != 1 or left.qconj != 1 or right.qconj != -1
            or left.ind_len != 1 or right.ind_len != 1):
        raise ValueError('explicit trivial finite boundary charges required')
    offset = (sum(int(psi.get_B(i, form=None).qtotal[index]) for i in range(cut))
              -int(left.to_qflat()[0, index]))
    env = MPSEnvironment(psi, psi)
    lp, rp = env._full_contraction_LP_RP(int(cut)-1)
    lp.get_leg('vR').test_contractible(virtual)
    rp.get_leg('vL').test_equal(virtual)
    contraction = complex(npc.inner(lp, rp,
        axes=[['vR*', 'vR'], ['vL*', 'vL']], do_conj=False))
    report = gram_schmidt_spectrum(
        lp.transpose(['vR*', 'vR']).to_ndarray(),
        rp.transpose(['vL', 'vL*']).to_ndarray(),
        virtual.to_qflat()[:, index]+offset, n_region=int(cut), tolerance=tolerance)
    closure = abs(report['norm_squared_without_external_mps_norm']-contraction)/abs(contraction)
    if closure > tolerance:
        raise ValueError('Gram orientation disagrees with actual tensor contraction')
    report.update(cut=int(cut), n_sites=int(psi.L), virtual_dimension=int(virtual.ind_len),
                  virtual_charge_to_physical_offset=offset,
                  tensor_contraction_closure_relative_error=float(closure),
                  physical_total_doubled_sz=int(psi.get_total_charge(only_physical_legs=True)[index]),
                  contracted_norm_squared=float(contraction.real*abs(psi.norm)**2))
    return report
