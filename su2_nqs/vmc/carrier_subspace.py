"""Full-complex sampled Ritz fits in fixed, explicitly selected state spans.

This fits a finite-Q training objective, not an exact physical ground state.
Every requested column must survive the registered rank/condition checks;
there is no ridge, rank truncation, real-coefficient restriction, trust-radius
search or validation-data optimization. Hold the fitted coefficients fixed
when evaluating independent chains, including inside their bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .sampled_multistate import (
    FailClosedGeneralizedRitz,
    NumericalSubspaceRankError,
    SampledSubspaceMoments,
    fail_closed_generalized_ritz,
    sampled_subspace_moments,
)


def _complex_matrix(value: np.ndarray) -> dict[str, list]:
    array = np.asarray(value, dtype=np.complex128)
    return dict(real=array.real.tolist(), imag=array.imag.tolist())


@dataclass(frozen=True)
class CarrierSubspaceFit:
    """Coefficients in the ORIGINAL complete bank, plus scaled-fit diagnostics.

    Unselected entries of ``coefficients`` are exactly zero. ``column_rms``
    follows ``indices``; all moment matrices refer to selected columns after
    this RMS normalization. A common unknown Q normalization affects state
    normalization but not its ray, energies, or variance. ``statistics``
    contains in-sample diagnostics only and is not an independent audit.
    """

    indices: tuple[int, ...]
    coefficients: np.ndarray
    scaled_coefficients: np.ndarray
    column_rms: np.ndarray
    moments: SampledSubspaceMoments
    ritz: FailClosedGeneralizedRitz
    statistics: dict[str, float | int | bool]
    relative_rank_tolerance: float
    maximum_condition_number: float
    n_sites: int

    def as_dict(self) -> dict:
        """JSON-safe replay data; no complex numbers are silently discarded."""
        return dict(
            indices=list(self.indices),
            coefficients_real=self.coefficients.real.tolist(),
            coefficients_imag=self.coefficients.imag.tolist(),
            scaled_coefficients_real=self.scaled_coefficients.real.tolist(),
            scaled_coefficients_imag=self.scaled_coefficients.imag.tolist(),
            column_rms=self.column_rms.tolist(),
            scaled_overlap=_complex_matrix(self.moments.overlap),
            scaled_hamiltonian_raw=_complex_matrix(self.moments.hamiltonian_raw),
            scaled_hamiltonian_hermitized=_complex_matrix(self.moments.hamiltonian),
            scaled_action_gram=_complex_matrix(self.moments.action_gram),
            eigenvalues_total=self.ritz.energies.tolist(),
            eigenvalues_per_site=(self.ritz.energies / self.n_sites).tolist(),
            overlap_eigenvalues=self.ritz.overlap_eigenvalues.tolist(),
            rank=self.ritz.rank,
            condition_number=self.ritz.overlap_condition_number,
            orthonormality_error=self.ritz.orthonormality_error,
            raw_hamiltonian_antihermiticity_max_absolute=self.moments.hamiltonian_hermiticity.maximum_absolute_error,
            raw_hamiltonian_antihermiticity_relative_frobenius=self.moments.hamiltonian_hermiticity.relative_frobenius_error,
            relative_rank_tolerance=self.relative_rank_tolerance,
            maximum_condition_number=self.maximum_condition_number,
            sample_count=self.moments.sample_count, n_sites=self.n_sites,
            statistics=dict(self.statistics),
            complex_coefficients=True, rank_truncated=False, ridge_added=False,
            optimum_scope='global minimum of this finite-Q real Rayleigh quotient in the requested fixed span',
            physical_variational_bound_certified=False,
            coefficients_must_be_frozen_for_independent_audit=True,
        )


def fit_carrier_subspace(
    amplitudes: np.ndarray, h_amplitudes: np.ndarray, q: np.ndarray, *,
    indices: Sequence[int] | None = None,
    observation_weights: np.ndarray | None = None,
    n_sites: int = 1,
    relative_rank_tolerance: float = 1e-10,
    maximum_condition_number: float = 1e8,
) -> CarrierSubspaceFit:
    """Fit the lowest full-complex Ritz root without discarding any column.

    Phi and HPhi have shape (sample, complete-bank-column). Q is a positive
    shared proposal, possibly unnormalized; unknown common normalization
    cancels. The default mean over rows also applies to correlated draws,
    but does NOT make their uncertainty IID. For exhaustive reference sums,
    supply observation_weights proportional to Q. H actions use TOTAL energy
    units; n_sites controls per-site reporting, not the eigenproblem.

    Columns are first normalized by sqrt(sum_x w_x |Phi_xa|^2 / Q_x).
    The existing sampled_subspace_moments / fail_closed_generalized_ritz
    implementation then solves Kc=lambda Sc with K=(B+B†)/2. This preserves
    Re(c†Bc) exactly. Raw B, imaginary energy, and the direct residual variance
    remain available; they are not replaced by their Hermitian surrogates.
    """
    phi = np.asarray(amplitudes, dtype=np.complex128)
    hphi = np.asarray(h_amplitudes, dtype=np.complex128)
    if phi.ndim != 2 or min(phi.shape) < 1 or hphi.shape != phi.shape:
        raise ValueError('amplitudes and h_amplitudes must be matching nonempty matrices')
    if not np.isfinite(phi).all() or not np.isfinite(hphi).all():
        raise ValueError('amplitudes and H actions must be finite')
    count, width = phi.shape
    if np.iscomplexobj(q):
        raise ValueError('q must be real')
    proposal = np.asarray(q, dtype=np.float64)
    if proposal.shape != (count,) or not np.isfinite(proposal).all() or np.any(proposal <= 0):
        raise ValueError('q must contain one finite positive density per sample')
    if isinstance(n_sites, (bool, np.bool_)) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError('n_sites must be a positive integer')
    if (maximum_condition_number is None or not np.isfinite(maximum_condition_number)
            or maximum_condition_number < 1):
        raise ValueError('maximum_condition_number must be finite and at least one')
    selected = tuple(range(width)) if indices is None else tuple(indices)
    if (not selected or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                            or i < 0 or i >= width for i in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('indices must be unique valid integer bank columns')
    selected = tuple(int(i) for i in selected)
    if observation_weights is None:
        weights = np.full(count, 1 / count, dtype=np.float64)
    else:
        if np.iscomplexobj(observation_weights):
            raise ValueError('observation_weights must be real')
        weights = np.asarray(observation_weights, dtype=np.float64)
        if (weights.shape != (count,) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0):
            raise ValueError('observation_weights must be finite nonnegative with positive sum')
        weights = weights / weights.sum()
    chosen = phi[:, selected]
    chosen_h = hphi[:, selected]
    maxima = np.max(abs(chosen), axis=0)
    if np.any(maxima == 0):
        raise NumericalSubspaceRankError('a requested bank column has zero sampled amplitude')
    # Max-rescaling avoids squaring the original possibly very different
    # carrier/decoder amplitude units before RMS normalization.
    relative = chosen / maxima[None, :]
    rms = maxima * np.sqrt(np.sum(weights[:, None] * abs(relative) ** 2 / proposal[:, None], axis=0))
    if not np.isfinite(rms).all():
        raise FloatingPointError('column RMS scaling is nonfinite')
    if np.any(rms <= 0):
        raise NumericalSubspaceRankError('a requested bank column has zero weighted norm')
    scaled, scaled_h = chosen / rms[None, :], chosen_h / rms[None, :]
    moments = sampled_subspace_moments(scaled, scaled_h, proposal,
                                       observation_weights=weights)
    ritz = fail_closed_generalized_ritz(moments,
        relative_rank_tolerance=relative_rank_tolerance,
        maximum_condition_number=maximum_condition_number)
    root = np.array(ritz.coefficients[:, 0], copy=True)
    pivot = int(np.argmax(abs(root)))
    root *= np.exp(-1j * np.angle(root[pivot]))
    coefficients = np.zeros(width, dtype=np.complex128)
    coefficients[list(selected)] = root / rms
    state, action = phi @ coefficients, hphi @ coefficients
    measure = weights / proposal
    norm = float(np.sum(measure * abs(state) ** 2))
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('fitted state has invalid norm after coefficient mapping')
    energy = np.sum(measure * state.conj() * action) / norm
    variance = float(np.sum(measure * abs(action - energy * state) ** 2) / norm)
    objective = np.vdot(root, moments.hamiltonian @ root).real / np.vdot(root, moments.overlap @ root).real
    eigenvalue = float(ritz.energies[0])
    equation_residual = moments.hamiltonian @ root - eigenvalue * (moments.overlap @ root)
    denominator = ((np.linalg.norm(moments.hamiltonian) + abs(eigenvalue) * np.linalg.norm(moments.overlap))
                   * np.linalg.norm(root))
    optimality = float(np.linalg.norm(equation_residual) / max(denominator, np.finfo(float).tiny))
    mass = measure * abs(state) ** 2
    statistics = dict(
        energy_total=float(energy.real), imaginary_energy_total=float(energy.imag),
        energy_per_site=float(energy.real / n_sites),
        imaginary_energy_per_site=float(energy.imag / n_sites),
        variance_total=variance, variance_per_site=variance / n_sites,
        norm_over_unknown_Q_partition=norm,
        maximum_norm_row_fraction=float(mass.max() / norm),
        relative_optimality_residual=optimality,
        real_rayleigh_hermitized_discrepancy=float(abs(energy.real - objective)),
        real_rayleigh_lowest_root_discrepancy=float(abs(energy.real - eigenvalue)),
        in_sample_fit_only=True,
    )
    return CarrierSubspaceFit(selected, coefficients, root, rms, moments, ritz,
                              statistics, float(relative_rank_tolerance),
                              float(maximum_condition_number), int(n_sites))


__all__ = ['CarrierSubspaceFit', 'fit_carrier_subspace']
