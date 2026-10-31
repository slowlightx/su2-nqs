"""Exact bounded regional SU(2) multiplicity readouts of physical density matrices.

Separate from the legacy readout module to preserve historical source seals.
This is an offline physical measurement, not a fusion-path variational model.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np

from ..su2.fusion_tree import FusionTree
from ..su2.recoupling import tree_basis_matrix
from .su2_entanglement_spectrum import _entropy


@lru_cache(maxsize=8)
def regional_schur_columns(
    n_sites: int,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    """Complete local CG columns labelled ``(q=2S,m=2M,a)`` for 1..8 spins.

    This is an offline regional readout, not a direct-fusion variational model.
    Multiplicity bases may change with the local tree; their spectra do not.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    if not 1 <= n_sites <= 8:
        raise ValueError("exact regional Schur readout is bounded to 1..8 spins")
    tree = FusionTree.balanced(int(n_sites))
    columns, labels = [], []
    for q in range(n_sites % 2, n_sites + 1, 2):
        for m in range(-q, q + 1, 2):
            matrix, states = tree_basis_matrix(tree, q, m)
            columns.extend(matrix[:, a] for a in range(len(states)))
            labels.extend((q, m, a) for a in range(len(states)))
    unitary = np.column_stack(columns)
    if np.linalg.norm(unitary.conj().T @ unitary - np.eye(2**n_sites)) > 1e-11:
        raise RuntimeError("regional CG columns are incomplete or nonorthogonal")
    unitary.setflags(write=False)
    return unitary, tuple(labels)


def su2_multiplicity_readout(
    rho: np.ndarray, *, tolerance: float = 1e-9,
) -> dict[str, object]:
    """Read physical ``rho`` as ``direct_sum_q I/(q+1) tensor R_q``.

    Return p(q), normalized and unnormalized multiplicity spectra, and the
    physical entanglement levels ``-log(lambda(R_q)/(q+1))``, one per multiplet.
    Reject broken SU(2), positivity or normalization; never twirl/repair rho.
    Magnetic degeneracy is kinematic, not evidence of order or a spin liquid.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or not np.isfinite(rho).all():
        raise ValueError("rho must be a finite square matrix")
    dimension = rho.shape[0]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("rho dimension must be a power of two")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    n_sites = dimension.bit_length() - 1
    unitary, labels = regional_schur_columns(n_sites)
    transformed = unitary.conj().T @ rho @ unitary
    reconstructed = np.zeros_like(transformed)
    sectors = {}
    spin_entropy = magnetic_entropy = multiplicity_entropy = 0.0
    for q in range(n_sites % 2, n_sites + 1, 2):
        mult = sum(1 for label in labels if label[:2] == (q, q))
        indices = [[i for i, label in enumerate(labels) if label[:2] == (q, m)]
                   for m in range(-q, q + 1, 2)]
        reduced = sum(transformed[np.ix_(block, block)] for block in indices)
        p = float(np.trace(reduced).real)
        spectrum = np.linalg.eigvalsh(reduced)[::-1]
        conditional = spectrum / p if p > 1e-14 else np.zeros(mult)
        for block in indices:
            reconstructed[np.ix_(block, block)] = reduced / (q + 1)
        positive = spectrum[spectrum > 1e-14]
        sectors[str(q)] = {
            "q": q, "multiplicity": mult, "p": p,
            "unnormalized_multiplicity_spectrum": spectrum.tolist(),
            "conditional_multiplicity_spectrum": conditional.tolist(),
            "physical_entanglement_levels": (-np.log(positive / (q + 1))).tolist(),
            "conditional_entropy": _entropy(conditional)[0],
        }
        if p > 1e-14:
            spin_entropy -= p * np.log(p)
            magnetic_entropy += p * np.log(q + 1)
            multiplicity_entropy += p * _entropy(conditional)[0]
    raising = np.zeros_like(rho)
    for packed in range(dimension):
        for site in range(n_sites):
            mask = 1 << site
            if not packed & mask:
                raising[packed | mask, packed] += 1
    values = np.linalg.eigvalsh(rho)
    entropy = _entropy(values)[0]
    diagnostics = {
        "trace_error": float(abs(np.trace(rho) - 1)),
        "hermiticity_frobenius": float(np.linalg.norm(rho - rho.conj().T)),
        "raising_commutator_frobenius": float(np.linalg.norm(rho @ raising - raising @ rho)),
        "factorization_frobenius": float(np.linalg.norm(transformed - reconstructed)),
        "unitarity_frobenius": float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(dimension))),
        "minimum_eigenvalue": float(values.min()),
        "entropy_sum_error": float(abs(entropy - spin_entropy - magnetic_entropy - multiplicity_entropy)),
    }
    if values.min() < -tolerance or max(v for k, v in diagnostics.items() if k != "minimum_eigenvalue") > tolerance:
        raise ValueError("rho fails physical SU(2) factorization; no twirl/repair allowed")
    return {
        "region_size": n_sites, "sectors": sectors, "entropy_nats": entropy,
        "spin_weight_entropy": float(spin_entropy), "magnetic_entropy": float(magnetic_entropy),
        "multiplicity_entropy": float(multiplicity_entropy), "diagnostics": diagnostics,
        "spin_squared_mean": float(sum(s["p"] * s["q"] * (s["q"] + 2) / 4 for s in sectors.values())),
    }


def su2_cyclic_entanglement_readout(
    rho: np.ndarray, site_permutation: tuple[int, ...], *, order: int,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Resolve multiplets by q and a region-preserving cyclic spatial symmetry.

    ``site_permutation[old]=new`` is active; bit zero is down and site zero
    is MSB. The region must respect this symmetry before measurement. Return
    k_index=0..order-1 for exp(2pi*i*k_index/order), NOT bulk crystal momentum.
    Do not spatially twirl a nonsymmetric rho to obtain a labelled spectrum.
    """
    base = su2_multiplicity_readout(rho, tolerance=tolerance)
    n = int(base['region_size'])
    if (len(site_permutation) != n or any(isinstance(i, bool) or not isinstance(i, Integral) for i in site_permutation)
            or sorted(site_permutation) != list(range(n))):
        raise ValueError('site_permutation must be an active permutation of region sites')
    if isinstance(order, bool) or not isinstance(order, Integral) or not 1 <= order <= 16:
        raise ValueError('order must be an integer in 1..16')
    rho = np.asarray(rho, dtype=complex)
    dimension = 2**n
    powers = 2**np.arange(n-1,-1,-1)
    bits = ((np.arange(dimension)[:,None] // powers) % 2).astype(int)
    image = np.zeros_like(bits)
    image[:,site_permutation] = bits
    t = np.zeros((dimension,dimension),complex)
    t[image@powers,np.arange(dimension)] = 1
    spatial_residual = float(np.linalg.norm(rho@t-t@rho))
    if np.linalg.norm(np.linalg.matrix_power(t,order)-np.eye(dimension)) > tolerance:
        raise ValueError('permutation does not have the specified cyclic order')
    if spatial_residual > tolerance:
        raise ValueError('rho does not respect the regional cyclic symmetry; no twirl allowed')
    u, labels = regional_schur_columns(n)
    result = {}
    maximum_projector_error = 0.0
    for q_string, sector in base['sectors'].items():
        q = int(q_string)
        ids = [i for i,l in enumerate(labels) if l[:2] == (q,q)]
        t_reduced = u[:,ids].conj().T @ t @ u[:,ids]
        reduced = sum(u[:,[i for i,l in enumerate(labels) if l[:2]==(q,m)]].conj().T @ rho @
                      u[:,[i for i,l in enumerate(labels) if l[:2]==(q,m)]] for m in range(-q,q+1,2))
        identity = np.eye(len(ids))
        t_powers = [np.linalg.matrix_power(t_reduced,j) for j in range(order)]
        by_k = {}
        for k in range(order):
            projector = sum(np.exp(-2j*np.pi*k*j/order)*t_powers[j] for j in range(order))/order
            error = max(np.linalg.norm(projector-projector.conj().T),
                        np.linalg.norm(projector@projector-projector))
            maximum_projector_error = max(maximum_projector_error,float(error))
            spectrum = np.linalg.eigvalsh(projector@reduced@projector)[::-1]
            positive = spectrum[spectrum>1e-14]
            by_k[str(k)] = {
                'q':q, 'k_index':k, 'cyclic_order':int(order),
                'available_multiplicity':int(round(np.trace(projector).real)),
                'p':float(np.trace(projector@reduced).real),
                'physical_entanglement_levels':(-np.log(positive/(q+1))).tolist(),
            }
        if abs(sum(s['p'] for s in by_k.values())-sector['p']) > tolerance:
            raise RuntimeError('cyclic sector weights fail q marginal closure')
        result[q_string] = by_k
    if maximum_projector_error > tolerance:
        raise RuntimeError('regional cyclic projector invariant failed')
    return dict(region_size=n,cyclic_order=int(order),sectors=result,
                spatial_commutator_frobenius=spatial_residual,
                maximum_projector_error=maximum_projector_error,
                warning='Regional entanglement momentum is not a bulk excitation momentum or a phase diagnosis.')


__all__ = ["regional_schur_columns", "su2_multiplicity_readout", "su2_cyclic_entanglement_readout"]
