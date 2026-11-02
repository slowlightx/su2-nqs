"""Static SU(2)-scalar sum rules, not spectroscopy of an approximate state.

Spin operators are physical spin half; bit zero is down and site zero is
the MSB. For O_alpha=sum_i phase_i S_i^alpha/sqrt(N), the numerator is
one half the sum of three double commutators. No symmetry projection is
performed. Interpreting the ratio as a triplet variational gap requires an
exact singlet ground state, not merely a small variational energy error.
"""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np


def scalar_oscillator_strength(
    correlations: np.ndarray,
    bonds: Sequence[tuple[int, int, float]],
    phases: np.ndarray,
    *,
    unresolved_structure_tolerance: float = 1e-12,
) -> dict:
    """Return static S and the sum-alpha double-commutator weight for H.

    H=sum_(ij) J_ij S_i.S_j and C_ij=<S_i.S_j> on the actual input state.
    S=phase^dagger C phase/N and F=-sum_(ij) J_ij C_ij|phase_i-phase_j|^2/N.
    This identity holds without SU(2) invariance. It does NOT give a gap for
    an approximate/non-singlet input, nor information beyond the supplied
    two-body correlations. Repeated bonds are counted as separate H terms.
    A tolerance marks an unresolved denominator; it never floors one.
    """
    c = np.asarray(correlations)
    phase = np.asarray(phases, dtype=np.complex128)
    if (c.ndim != 2 or not len(c) or c.shape[0] != c.shape[1]
            or not np.isfinite(c).all() or phase.shape != (len(c),)
            or not np.isfinite(phase).all()
            or not np.isfinite(unresolved_structure_tolerance)
            or unresolved_structure_tolerance <= 0):
        raise ValueError('finite square correlations and matching finite phases required')
    if np.max(abs(c.imag)) > 1e-10 or np.max(abs(c-c.T.conj())) > 1e-10:
        raise ValueError('physical commuting pair correlations must be real symmetric; no repair')
    n = len(c)
    if np.max(abs(np.diag(c)-.75)) > 1e-9:
        raise ValueError('spin-half diagonal correlations must be 3/4')
    numerator = 0.
    for i, j, coupling in bonds:
        if (any(isinstance(x, bool) or not isinstance(x, Integral) for x in (i,j))
                or not 0 <= i < n or not 0 <= j < n or i == j
                or not np.isfinite(coupling)):
            raise ValueError('valid distinct integer bond endpoints and finite J required')
        numerator -= float(coupling)*float(c[i,j].real)*abs(phase[i]-phase[j])**2/n
    structure = float(np.vdot(phase, c@phase).real/n)
    resolved = structure > unresolved_structure_tolerance
    return dict(structure_factor=structure, oscillator_strength=float(numerator),
        static_ratio=float(numerator/structure) if resolved else None,
        denominator_resolved=resolved,
        spatially_constant_mode=bool(np.max(abs(phase-phase[0])) <= 1e-12),
        exact_ground_state_gap_upper_bound=False, strict_triplet_label=False,
        new_spectral_information=False, input_repaired=False)


def longitudinal_residual_bound(
    total_H_variance: float,
    mean_A: float,
    variance_A: float,
) -> float:
    """Cauchy bound on parent-residual covariance / <A>, A=O_z^dagger O_z.

    The result bounds the correction between a longitudinal oscillator ratio
    and the SYMMETRIZED O_k/O_-k excited energies relative to the parent.
    For complex Fourier O_k, it does not bound one momentum separately unless
    equality of the two energy numerators is independently established (e.g.
    by inversion symmetry). A Hermitian O has no such qualification.
    It does not bound E_parent-E_ground. Use Var(H), not Var(H)/N. Carrying
    this to scalar/three-axis sums needs invariant input or the corresponding
    off-diagonal fourth moments; longitudinal Born counts cannot supply them.
    """
    numbers = (total_H_variance, mean_A, variance_A)
    if (not all(np.isfinite(x) for x in numbers) or total_H_variance < 0
            or variance_A < 0 or mean_A <= 0):
        raise ValueError('nonnegative finite variances and positive <A> required')
    return float(np.sqrt(total_H_variance*variance_A)/mean_A)
