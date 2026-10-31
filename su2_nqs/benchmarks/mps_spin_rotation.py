"""Physical magnetization FCS of an unchanged, finite U(1) spin-half MPS.

The labels here are doubled *magnetic* charges ``m=2*S_A^z``, not SU(2)
irreps ``q=2*S_A``.  Virtual charges can be gauge shifted.  True finite
left/right environments, including their connecting singular-value factors,
measure the physical probability; stored ``_S**2`` is not substituted for it.
No SU(2) twirl, canonicalization, topological inference, or new state is made.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any, Sequence

import numpy as np


def exact_spin_half_charge_distribution(
    state: np.ndarray, sites: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Small dense oracle for ``m=2*S_A^z``; down/up bits, site zero MSB.

    Bounded to 18 spins.  This is a Born-probability measurement, not an
    entanglement spectrum and not a measurement of relative amplitudes.
    """
    state = np.asarray(state, dtype=np.complex128)
    if (state.ndim != 1 or len(state) < 2 or len(state) & (len(state)-1)
            or not np.isfinite(state).all()):
        raise ValueError('a finite spin-half state vector is required')
    n = len(state).bit_length()-1
    if n > 18:
        raise ValueError('dense physical oracle is bounded to 18 spins')
    region = tuple(sites)
    if (not region or len(set(region)) != len(region)
            or any(isinstance(i, bool) or not isinstance(i, Integral)
                   or not 0 <= i < n for i in region)):
        raise ValueError('region needs distinct in-range integer sites')
    norm = float(np.vdot(state, state).real)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('state must have positive finite norm')
    packed = np.arange(len(state), dtype=np.int64)
    count = sum((packed >> (n-1-int(i))) & 1 for i in region)
    weights = np.bincount(count, weights=np.abs(state)**2,
                          minlength=len(region)+1)/norm
    return np.arange(-len(region), len(region)+1, 2, dtype=np.int64), weights


def sampled_prefix_charge_distribution(
    states: np.ndarray, cut: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical prefix magnetic histogram from existing Born configurations.

    ``states[..., site]`` contains down/up 0/1. Leading dimensions are only
    pooled for this point estimate: this helper does NOT assert independent
    samples or provide an error bar. Whole independent chains must be kept
    separate by a statistical caller when estimating uncertainty.
    """
    bits = np.asarray(states)
    if (bits.ndim < 2 or bits.dtype.kind not in 'biu' or bits.size == 0
            or np.any((bits != 0) & (bits != 1))
            or isinstance(cut, bool) or not isinstance(cut, Integral)
            or not 0 < cut <= bits.shape[-1]):
        raise ValueError('nonempty spin-half Born samples and a valid integer prefix required')
    counts = np.sum(bits[..., :cut], axis=-1, dtype=np.int64).reshape(-1)
    weights = np.bincount(counts, minlength=cut+1).astype(np.float64)/len(counts)
    return np.arange(-cut, cut+1, 2, dtype=np.int64), weights


def physical_spin_rotation_summary(
    magnetic_labels: np.ndarray, probabilities: np.ndarray,
    *, n_region: int, angles: np.ndarray, tolerance: float = 1e-10,
) -> dict[str, Any]:
    """FCS ``Z(theta)=<exp(i*theta*S_A^z)>`` and physical cumulants.

    Number entropy is only the entropy of the measured magnetic charge, not
    full entanglement entropy.  Finite curves are analytic; zeros, parity,
    and a spin-half center sign alone are not QSL-specific evidence.
    """
    if (isinstance(n_region, bool) or not isinstance(n_region, Integral)
            or n_region < 1 or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('positive integer region size and tolerance required')
    m = np.asarray(magnetic_labels)
    p = np.asarray(probabilities, dtype=np.float64)
    theta = np.asarray(angles, dtype=np.float64)
    if (m.ndim != 1 or m.dtype.kind not in 'iu' or p.shape != m.shape
            or len(set(m.tolist())) != len(m) or not np.isfinite(p).all()
            or np.any(abs(m) > n_region) or np.any((m-n_region) % 2)
            or theta.ndim != 1 or not np.isfinite(theta).all()):
        raise ValueError('physical doubled magnetic labels and finite arrays required')
    if np.any(p < -tolerance) or abs(float(p.sum())-1.) > tolerance:
        raise ValueError('probabilities must be normalized and nonnegative')
    # No clipping or renormalization repair: retain a valid measured parent.
    z = np.exp(.5j*np.outer(theta, m)) @ p
    spin = m/2
    mean = float(p @ spin)
    variance = float(p @ ((spin-mean)**2))
    fourth = float(p @ ((spin-mean)**4))
    positive = p > 0
    center = complex(np.exp(1j*np.pi*m) @ p)
    return dict(magnetic_labels=m.tolist(), probabilities=p.tolist(),
                angles=theta.tolist(), rotation_real=z.real.tolist(),
                rotation_imaginary=z.imag.tolist(), mean_sz=mean,
                variance_sz=variance, fourth_cumulant_sz=fourth-3*variance**2,
                magnetic_number_entropy_nats=float(-p[positive] @ np.log(p[positive])),
                physical_2pi_center_error=float(abs(center-(-1)**int(n_region))),
                normalization_error=float(abs(p.sum()-1.)),
                minimum_probability=float(p.min()),
                strict_su2_verified=False, schmidt_spectrum_measured=False,
                qsl_specific=False,
                warning='Magnetic FCS is a physical U(1) measurement. No SU(2) irrep weights, Schmidt spectrum, anyon charge, or phase classification are inferred.')


def finite_mps_prefix_charge_distribution(
    psi: Any, cut: int, *, tolerance: float = 1e-10, renyi2: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Measure ``m=2*sum_{i<cut} S_i^z`` on the actual finite MPS.

    Requires TeNPy and a verified conserved ``2*Sz`` U(1) charge on physical
    spin-half legs. For tensor charges t_i and nominal virtual vL charges,
    ``m_A = vL(cut)-vL(0)+sum_{i<cut} t_i``.  Thus neither the nominal
    virtual charge nor tensor qtotal alone is a physical observable.

    Full two-sided environments are contracted with ``store=False`` so the
    O(L*chi^3), charge-blocked preparation need not retain every cut. Once
    prepared, projector contractions cost O(number_of_charges*chi^2). No
    assumption of stored canonicality, and no modification to psi, is used.

    Optional ``renyi2`` adds actual ``Tr(rho_A,m**2)`` per magnetic sector.
    This keeps two dense chi-by-chi environments and contracts block Gram
    products with EXTRA charge-blocked cubic cost. It does not use raw S².
    """
    if (psi.bc != 'finite' or isinstance(cut, bool) or not isinstance(cut, Integral)
            or not 0 < cut < psi.L or any(form is None for form in psi.form)
            or not np.isfinite(tolerance) or tolerance <= 0
            or not isinstance(renyi2, bool)):
        raise ValueError('a finite MPS with defined forms and an internal integer cut is required')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPSEnvironment

    names = list(psi.chinfo.names)
    if names.count('2*Sz') != 1:
        raise ValueError('an explicit conserved physical 2*Sz charge is required')
    index = names.index('2*Sz')
    if int(psi.chinfo.mod[index]) != 1:
        raise ValueError('2*Sz must be U(1), not a modulo charge')
    for i, site in enumerate(psi.sites):
        leg = psi.get_B(i, form=None).get_leg('p')
        sz = np.asarray(site.get_op('Sz').to_ndarray(), dtype=np.complex128)
        flat = leg.to_qflat()[:, index]
        if (leg.qconj != 1 or sz.shape != (2, 2)
                or not np.allclose(sz, np.diag(np.diag(sz)), atol=tolerance, rtol=0)
                or not np.allclose(flat, 2*np.diag(sz), atol=tolerance, rtol=0)
                or set(flat.tolist()) != {-1, 1}):
            raise ValueError('physical legs must carry actual doubled spin-half Sz eigenvalues')
    virtual = psi.get_B(int(cut), form=None).get_leg('vL')
    left = psi.get_B(0, form=None).get_leg('vL')
    right = psi.get_B(psi.L-1, form=None).get_leg('vR')
    if (virtual.qconj != 1 or left.qconj != 1 or right.qconj != -1
            or left.ind_len != 1 or right.ind_len != 1):
        raise ValueError('explicit trivial finite boundary charges required')
    offset = (sum(int(psi.get_B(i, form=None).qtotal[index]) for i in range(cut))
              -int(left.to_qflat()[0, index]))
    physical_total = int(psi.get_total_charge(only_physical_legs=True)[index])
    env = MPSEnvironment(psi, psi)
    # This method inserts precisely the bra/ket S factors also used by
    # MPSEnvironment.full_contraction. Tested against dense and gauge oracles.
    lp, rp = env._full_contraction_LP_RP(int(cut)-1)
    lp.get_leg('vR').test_contractible(virtual)
    rp.get_leg('vL').test_equal(virtual)
    axes = [['vR*', 'vR'], ['vL*', 'vL']]
    denominator = complex(npc.inner(lp, rp, axes=axes, do_conj=False))
    if (not np.isfinite(denominator) or denominator.real <= 0
            or abs(denominator.imag) > tolerance*denominator.real):
        raise ValueError('actual finite contraction must have a positive real norm')
    charges = virtual.to_qflat()[:, index]+offset
    expected = np.arange(-cut, cut+1, 2, dtype=np.int64)
    if not set(charges.tolist()) <= set(expected.tolist()):
        raise ValueError('virtual-gauge mapping gives nonphysical prefix magnetic charges')
    weights = np.zeros(len(expected), dtype=np.complex128)
    for j, charge in enumerate(expected):
        mask = (charges == charge).astype(np.float64)
        if mask.any():
            projected = lp.scale_axis(mask, 'vR')
            weights[j] = npc.inner(projected, rp, axes=axes, do_conj=False)/denominator
    imaginary = float(np.max(abs(weights.imag)))
    if (imaginary > tolerance or np.any(weights.real < -tolerance)
            or abs(weights.sum()-1.) > tolerance):
        raise ValueError('actual finite prefix probabilities failed physicality checks')
    details = dict(cut=int(cut), n_sites=int(psi.L),
                   physical_total_doubled_sz=physical_total,
                   virtual_charge_to_physical_offset=int(offset),
                   virtual_dimension=int(virtual.ind_len),
                   occupied_virtual_magnetic_charges=len(np.unique(charges)),
                   maximum_imaginary_probability=imaginary,
                   contracted_norm_squared=float(denominator.real*abs(psi.norm)**2),
                   stored_schmidt_weights_used=False, parent_canonicalized=False,
                   full_environment_preparation=True, environments_cached=False,
                   measure='actual physical prefix S^z Born distribution',
                   warning='Physical prefix magnetic charges are not SU(2) irreps or a single-boundary anyon charge. Full environments have actual large-chi contraction cost.')
    if renyi2:
        # Contraction axes, not a guessed matrix convention: LP[bra,ket]
        # multiplies RP[ket,bra]. The RP's *natural* axis order is not relied
        # on, and trace(L@R) must equal the actual npc.inner denominator.
        left_gram = lp.transpose(['vR*','vR']).to_ndarray()
        right_transposed_gram = rp.transpose(['vL','vL*']).to_ndarray()
        purity_weights = np.zeros(len(expected), dtype=np.complex128)
        reconstructed_norm = 0j
        for j, charge in enumerate(expected):
            indices = np.flatnonzero(charges == charge)
            if len(indices):
                select = np.ix_(indices,indices)
                gram_product = left_gram[select] @ right_transposed_gram[select]
                reconstructed_norm += np.trace(gram_product)
                purity_weights[j] = np.einsum('ij,ji->',gram_product,gram_product)/denominator**2
        norm_closure = float(abs(reconstructed_norm-denominator)/denominator.real)
        purity = complex(purity_weights.sum())
        if (norm_closure > tolerance or not np.isfinite(purity_weights).all()
                or np.max(abs(purity_weights.imag)) > tolerance
                or np.min(purity_weights.real) < -tolerance
                or abs(purity.imag) > tolerance or not 0 < purity.real <= 1+tolerance):
            raise ValueError('actual Gram contraction failed charged-purity invariants')
        details['renyi2'] = dict(
            magnetic_labels=expected.tolist(),
            sector_purities=purity_weights.real.tolist(),
            purity=float(purity.real),
            normalized_purity_charge_distribution=(purity_weights.real/purity.real).tolist(),
            renyi2_entanglement_entropy_nats=float(-np.log(purity.real)),
            gram_norm_closure_error=norm_closure,
            maximum_imaginary_sector_purity=float(np.max(abs(purity_weights.imag))),
            extra_charge_block_cubic_contractions=True,
            raw_schmidt_values_used=False,schmidt_eigenvalues_computed=False,
            strict_su2_irrep_resolution=False,
            warning='Actual U(1)-charge-resolved second RDM moment, not q=2S, a Schmidt spectrum, or a fractionalization certificate.')
    return expected, weights.real, details


def estimate_prefix_rotation_memory(
    chi: int, checkpoint_bytes: int, *, itemsize: int = 16,
) -> dict[str, Any]:
    """Conservative dense storage budget; charge sparsity is NOT assumed.

    Two checkpoint-sized resident copies plus twelve chi-by-chi complex
    matrices and 512 MiB reserve. This is a refusal heuristic, not a hard
    guarantee; a subprocess RSS/address-space and wallclock cap is mandatory.
    Time still scales cubically with chi before charge-block speedups.
    """
    if (isinstance(chi, bool) or not isinstance(chi, Integral) or chi < 1
            or isinstance(checkpoint_bytes, bool) or not isinstance(checkpoint_bytes, Integral)
            or checkpoint_bytes < 0 or itemsize not in (8, 16)):
        raise ValueError('positive integer chi, nonnegative checkpoint bytes, and 8/16-byte dtype required')
    return dict(estimated_total_bytes=int(2*checkpoint_bytes+12*itemsize*chi**2+512*1024**2),
                chi_upper_bound=int(chi), checkpoint_bytes=int(checkpoint_bytes),
                itemsize=int(itemsize), sparsity_assumed=False,
                estimate_is_hard_guarantee=False)
