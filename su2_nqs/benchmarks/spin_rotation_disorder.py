"""Finite-region spin-rotation moments, NOT a finite-size QSL classifier.

Magnetic labels are doubled integers m=2M; site zero is the MSB and bit zero
is down. Direct axis moments use the actual rho. The character average is
a scalar observable, not a twirled replacement state or a strict SU2 spectrum.
"""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np

from .regional_su2_readout import regional_schur_columns


def rotation_characteristic(m: np.ndarray, probabilities: np.ndarray,
                            angles: np.ndarray) -> np.ndarray:
    """Return sum_m p_m exp(i theta m/2), using doubled integer projections."""
    m, probabilities, angles = np.asarray(m), np.asarray(probabilities), np.asarray(angles)
    if (m.ndim != 1 or not len(m) or m.dtype.kind not in 'iu' or probabilities.shape != m.shape
            or probabilities.dtype.kind not in 'fiu'
            or angles.ndim != 1 or not len(angles) or angles.dtype.kind not in 'fiu'
            or not np.isfinite(angles).all()
            or not np.isfinite(probabilities).all()):
        raise ValueError('finite angle vector and doubled-integer projection distribution required')
    if np.min(probabilities) < -1e-10 or abs(probabilities.sum()-1) > 1e-10:
        raise ValueError('invalid distribution; no clipping or normalization repair')
    return probabilities @ np.exp(.5j*m[:, None]*angles[None, :])


def regional_spin_rotation(rho: np.ndarray, angles: np.ndarray) -> dict:
    """Direct x/z and rotation-averaged scalar moments of an actual 1..8-spin rho.

    Scalar = sum_q p(q) chi_q(theta)/(q+1) is legal also for broken SU2.
    Rotational invariance guarantees equality to the direct axis moment;
    agreement alone is not a sufficient test of invariance.
    No eigenvalue repair, twirl, log floor, or cusp fitting is performed.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if (rho.ndim != 2 or rho.shape[0] != rho.shape[1] or not np.isfinite(rho).all()
            or rho.shape[0] < 2 or rho.shape[0] & (rho.shape[0]-1)):
        raise ValueError('finite power-of-two square density matrix required')
    n = rho.shape[0].bit_length()-1
    if not 1 <= n <= 8:
        raise ValueError('dense readout bounded to 1..8 spins')
    if (abs(np.trace(rho)-1) > 1e-10 or np.linalg.norm(rho-rho.conj().T) > 1e-10
            or np.linalg.eigvalsh(rho).min() < -1e-10):
        raise ValueError('nonphysical rho; no repair')
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    m = (2*bits.sum(axis=1)-n).astype(np.int32)
    x_columns = np.array([[1., 1.], [-1., 1.]])/np.sqrt(2)
    change = np.array([[1.]])
    for _ in range(n):
        change = np.kron(change, x_columns)
    z = rotation_characteristic(m, np.diag(rho).real, angles)
    x = rotation_characteristic(m, np.diag(change.T@rho@change).real, angles)
    u, labels = regional_schur_columns(n)
    schur = u.conj().T@rho@u
    averaged = np.zeros_like(schur)
    scalar = np.zeros(len(angles), dtype=complex)
    weights = {}
    for q in range(n % 2, n+1, 2):
        indices = [[i for i, label in enumerate(labels) if label[:2] == (q, magnetic)]
                   for magnetic in range(-q, q+1, 2)]
        reduced = sum(schur[np.ix_(i, i)] for i in indices)
        weight = float(np.trace(reduced).real)
        weights[str(q)] = weight
        scalar += weight*rotation_characteristic(np.arange(-q, q+1, 2, dtype=np.int32),
                                                  np.full(q+1, 1/(q+1)), angles)
        for i in indices:
            averaged[np.ix_(i, i)] = reduced/(q+1)
    return dict(z=z, x=x, scalar=scalar, p_q=weights,
                factorization_frobenius=float(np.linalg.norm(schur-averaged)),
                axis_scalar_max_difference=float(max(np.max(abs(z-scalar)),
                                                     np.max(abs(x-scalar)))),
                input_repaired=False, strict_SU2_spectrum=False,
                finite_region=True, QSL_claim=False)


def sampled_spin_rotation(states: np.ndarray, chain_ids: np.ndarray,
                          sites: Sequence[int], angles: np.ndarray) -> dict:
    """Bounded diagonal estimator from existing Born records, preserving chains.

    All angles share the same chain means, so differences retain covariance.
    Descriptive whole-chain SE does not certify equilibrium or rare tails.
    No wavefunction/amplitude/H calls or extra samples are made here.
    """
    states, chain_ids, angles = np.asarray(states), np.asarray(chain_ids), np.asarray(angles)
    sites = tuple(sites)
    if (states.ndim != 2 or not np.isin(states, (0, 1)).all()
            or chain_ids.shape != (len(states),) or chain_ids.dtype.kind not in 'iu'
            or not sites or len(set(sites)) != len(sites)
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in sites)
            or min(sites) < 0 or max(sites) >= states.shape[1]):
        raise ValueError('bit records, integer chain IDs and distinct valid region sites required')
    chains = np.unique(chain_ids)
    if len(chains) < 2:
        raise ValueError('at least two independent chain groups required')
    counts = states[:, sites].sum(axis=1).astype(np.int32)
    distributions = np.array([np.bincount(counts[chain_ids == c], minlength=len(sites)+1)
                               /np.sum(chain_ids == c) for c in chains])
    m = np.arange(-len(sites), len(sites)+1, 2, dtype=np.int32)
    curves = np.array([rotation_characteristic(m, p, angles) for p in distributions])
    return dict(value=curves.mean(axis=0), real_chain_SE=curves.real.std(axis=0, ddof=1)/np.sqrt(len(chains)),
                imaginary_chain_SE=curves.imag.std(axis=0, ddof=1)/np.sqrt(len(chains)),
                chain_means=curves, records=len(states), chains=len(chains),
                groups_equally_weighted=True, equilibrium_certified=False,
                new_samples=0, new_H_calls=0, QSL_claim=False)


def sampled_translation_rotation(states: np.ndarray, chain_ids: np.ndarray,
                                 sites: Sequence[int], angles: np.ndarray, *,
                                 rows: int, columns: int,
                                 spin_inversion_average: bool = False) -> dict:
    """Average a region's rotation over torus translations on EACH record.

    This always measures the translation-averaged observable. It has the
    original region's expectation only for a translation-invariant state;
    spin-inversion averaging additionally requires that invariance. Images
    are correlated, not independent samples. Keep the original chain groups.
    The variance benefit is empirical, not a claim of equilibrium or topology.
    """
    if (isinstance(rows, bool) or isinstance(columns, bool)
            or not isinstance(rows, Integral) or not isinstance(columns, Integral)
            or rows < 1 or columns < 1 or np.shape(states)[1:] != (rows*columns,)):
        raise ValueError('positive integer torus dimensions must match bit records')
    direct = sampled_spin_rotation(states, chain_ids, sites, angles)
    states, chain_ids = np.asarray(states), np.asarray(chain_ids)
    regions = [[((i//columns+dr)%rows)*columns+(i%columns+dc)%columns for i in sites]
               for dr in range(rows) for dc in range(columns)]
    counts = np.stack([states[:, region].sum(axis=1) for region in regions],axis=1).astype(int)
    distributions = np.array([np.bincount(counts[chain_ids==c].ravel(),minlength=len(sites)+1)
                               /counts[chain_ids==c].size for c in np.unique(chain_ids)])
    m = np.arange(-len(sites),len(sites)+1,2,dtype=np.int32)
    curves = np.array([rotation_characteristic(m,p,angles) for p in distributions])
    if spin_inversion_average:
        curves = curves.real.astype(complex)
    paired = curves-direct['chain_means']
    n = len(curves)
    return dict(value=curves.mean(axis=0), chain_means=curves,
                real_chain_SE=curves.real.std(axis=0,ddof=1)/np.sqrt(n),
                imaginary_chain_SE=curves.imag.std(axis=0,ddof=1)/np.sqrt(n),
                paired_difference=paired.mean(axis=0),
                paired_real_chain_SE=paired.real.std(axis=0,ddof=1)/np.sqrt(n),
                paired_imaginary_chain_SE=paired.imag.std(axis=0,ddof=1)/np.sqrt(n),
                records=len(states),chains=n,translation_images=rows*columns,
                images_are_independent=False,spin_inversion_average=spin_inversion_average,
                same_expectation_requires_state_symmetry=True,equilibrium_certified=False,
                new_samples=0,new_H_calls=0,QSL_claim=False)
