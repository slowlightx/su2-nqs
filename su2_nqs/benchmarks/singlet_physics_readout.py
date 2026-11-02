"""Physical spin/space channels of saved states, not a variational ansatz.

The cheap diagonal structure estimator REQUIRES an independently established
total-spin singlet: rotational invariance gives <S_i.S_j>=3<S_i^z S_j^z>.
Fixed Sz=0 alone is insufficient. Bits are down/up, site zero MSB. Regional
operators act on the actual coherent RDM, without SU2 twirling or repair.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from .spin_operator_information import spin_dot_matrix


def singlet_structure_factor_records(
    configurations: np.ndarray, coordinates: np.ndarray,
    momenta: Mapping[str, Sequence[float]], *, strict_singlet: bool,
) -> dict[str, np.ndarray]:
    """Return 3/N |sum_i Sz_i exp(2pi i k.r_i)|² for each saved record.

Coordinates/momenta are reciprocal fractional pairs. This is the scalar
static structure factor, NOT susceptibility. A caller must establish that
the momenta are allowed by the finite physical boundary conditions.
"""
    bits = np.asarray(configurations)
    coords = np.asarray(coordinates, float)
    if strict_singlet is not True:
        raise ValueError('3*SzSz requires an independently established strict singlet')
    if (bits.ndim != 2 or len(bits) == 0 or not np.isin(bits, (0, 1)).all()
            or coords.shape != (bits.shape[1], 2) or not np.isfinite(coords).all()):
        raise ValueError('binary records and matching finite two-dimensional coordinates required')
    n = bits.shape[1]
    if n < 2 or n % 2 or np.any(bits.sum(axis=1) != n//2):
        raise ValueError('even-site physical Sz=0 support required for a singlet')
    records = {}
    for label, raw in momenta.items():
        k = np.asarray(raw, float)
        if k.shape != (2,) or not np.isfinite(k).all():
            raise ValueError('finite two-component reciprocal momenta required')
        phase = np.exp(2j*np.pi*(coords@k))
        records[label] = 3*abs((bits-.5)@phase)**2/n
    return records


def triangular_torus_structure_records(
    configurations: np.ndarray, rows: int, columns: int, *, strict_singlet: bool,
) -> dict[str, np.ndarray]:
    """Allowed K/ALL M on the repository NN metric (1,0),(0,1),(1,1).

K requires both periods divisible by three; all M require even periods.
The proxy (S(K)+S(-K))/N includes finite-size fluctuations/self terms and
is NOT an extrapolated thermodynamic ordered moment squared.
"""
    if (any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral)
            or x < 1 or x % 6 for x in (rows, columns))
            or np.asarray(configurations).ndim != 2
            or np.asarray(configurations).shape[1] != rows*columns):
        raise ValueError('matching torus periods divisible by six required for K and ALL M')
    coords = np.array([divmod(i, columns) for i in range(rows*columns)])
    result = singlet_structure_factor_records(configurations, coords,
        {'Gamma': (0., 0.), 'K': (1/3, 1/3), 'minus_K': (-1/3, -1/3),
         'M1': (.5, 0.), 'M2': (0., .5), 'M3': (.5, .5)},
        strict_singlet=strict_singlet)
    result['M_A1_average'] = sum(result[k] for k in ('M1', 'M2', 'M3'))/3
    result['m120_proxy_squared'] = (result['K']+result['minus_K'])/(rows*columns)
    return result


def importance_jackknife_observables(
    record_values: Mapping[str, np.ndarray], psi: np.ndarray,
    proposal_density: np.ndarray, chain_ids: np.ndarray,
) -> dict:
    """Node-safe self-normalized means and delete-WHOLE-chain uncertainty.

    The positive proposal normalization cancels. Returned jackknife values
    permit paired comparisons on THE SAME records. SE is not an equilibrium
    certificate or a bound on omitted rare-tail bias.
    """
    a, q, groups = np.asarray(psi), np.asarray(proposal_density), np.asarray(chain_ids)
    if (a.ndim != 1 or q.shape != a.shape or groups.shape != a.shape
            or not np.isfinite(a).all() or not np.isfinite(q).all() or np.any(q <= 0)):
        raise ValueError('matching finite amplitudes, positive density and chain IDs required')
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('at least two independent chains required')
    weights = abs(a)**2/q
    if weights.sum() <= 0:
        raise ValueError('unresolved physical norm')
    result = {}
    for label, raw in record_values.items():
        values = np.asarray(raw, float)
        if values.shape != a.shape or not np.isfinite(values).all():
            raise ValueError('finite scalar record observable required')
        mean = float(weights@values/weights.sum())
        deletes = []
        for chain in unique:
            keep = groups != chain
            if weights[keep].sum() <= 0:
                raise ValueError('unresolved leave-chain-out norm')
            deletes.append(weights[keep]@values[keep]/weights[keep].sum())
        deletes = np.asarray(deletes)
        se = np.sqrt((len(unique)-1)*np.mean((deletes-deletes.mean())**2))
        result[label] = dict(value=mean, chain_SE=float(se), jackknife_values=deletes)
    return dict(observables=result, independent_chains=len(unique), records=len(a),
        weight_ESS_fraction=float(weights.sum()**2/(weights@weights*len(a))),
        maximum_weight_fraction=float(weights.max()/weights.sum()),
        ratio_bias_not_removed=True, equilibrium_certified=False)


def scalar_chirality_matrix(n: int, sites: Sequence[int]) -> np.ndarray:
    """Return oriented chi_ijk=S_i.(S_j cross S_k)=i[B_ij,B_jk]."""
    triple = tuple(sites)
    if len(triple) != 3 or len(set(triple)) != 3:
        raise ValueError('three distinct oriented sites required')
    i, j, k = triple
    a, b = spin_dot_matrix(n, i, j), spin_dot_matrix(n, j, k)
    return 1j*(a@b-b@a)


@lru_cache(maxsize=1)
def compact6_channel_operators() -> dict[str, np.ndarray]:
    """Fixed 2x3 compact-region scalar E2/chirality/dimer operator bank.

Physical sites (r,c) are (0,0),(0,1),(0,2),(1,0),(1,1),(1,2), with
e_r=(1,0), e_c=(-1/2,sqrt(3)/2). Counterclockwise triangles are
(0,3,4) and (1,5,2). Their e_r,e_c,e_r+e_c bond triplets are
[(0,3),(3,4),(0,4)] and [(2,5),(1,2),(1,5)]. Orthonormal E2
components use (2,-1,-1)/sqrt(6), (0,-1,1)/sqrt(2).
These are LOCAL channels, not a full-lattice irrep susceptibility.
"""
    edges = (((0, 3), (3, 4), (0, 4)), ((2, 5), (1, 2), (1, 5)))
    coefficients = np.array([[2, -1, -1], [0, -1, 1]], float)
    coefficients /= np.array([[np.sqrt(6)], [np.sqrt(2)]])
    dots = [[spin_dot_matrix(6, *bond) for bond in side] for side in edges]
    e2 = [[sum(c*b for c, b in zip(row, side)) for row in coefficients] for side in dots]
    bank = {}
    for side in range(2):
        for component in range(2):
            bank[f'E2_triangle{side}_component{component}'] = e2[side][component]
        bank[f'E2_triangle{side}_norm_squared'] = sum(op@op for op in e2[side])
    bank['E2_intertriangle_dot'] = sum(a@b for a, b in zip(*e2))
    chi = [scalar_chirality_matrix(6, triple) for triple in ((0, 3, 4), (1, 5, 2))]
    for side in range(2):
        bank[f'chi_triangle{side}'] = chi[side]
        bank[f'chi_triangle{side}_squared'] = chi[side]@chi[side]
    bank['chi_intertriangle_product'] = chi[0]@chi[1]
    for direction in range(3):
        for side in range(2):
            bank[f'bond_triangle{side}_direction{direction}'] = dots[side][direction]
        bank[f'dimer_parallel_direction{direction}_product'] = dots[0][direction]@dots[1][direction]
    for op in bank.values():
        op.setflags(write=False)
    return bank


def compact6_channel_readout(rho: np.ndarray) -> dict[str, float]:
    """Measure coherent actual rho; no SU2/spatial averaging or eigenvalue repair."""
    matrix = np.asarray(rho, complex)
    if (matrix.shape != (64, 64) or not np.isfinite(matrix).all()
            or np.linalg.norm(matrix-matrix.conj().T) > 1e-10
            or abs(np.trace(matrix)-1) > 1e-10 or np.linalg.eigvalsh(matrix).min() < -1e-10):
        raise ValueError('normalized positive Hermitian compact6 RDM required')
    result = {}
    for name, op in compact6_channel_operators().items():
        value = np.trace(matrix@op)
        if abs(value.imag) > 1e-10:
            raise ValueError('Hermitian observable has nonreal mean')
        result[name] = float(value.real)
    for side in range(2):
        result[f'E2_triangle{side}_connected_norm_squared'] = (
            result[f'E2_triangle{side}_norm_squared']
            - sum(result[f'E2_triangle{side}_component{a}']**2 for a in range(2)))
    result['E2_intertriangle_connected_dot'] = result['E2_intertriangle_dot']-sum(
        result[f'E2_triangle0_component{a}']*result[f'E2_triangle1_component{a}'] for a in range(2))
    result['chi_intertriangle_connected'] = result['chi_intertriangle_product']-result['chi_triangle0']*result['chi_triangle1']
    for direction in range(3):
        result[f'dimer_parallel_direction{direction}_connected'] = (
            result[f'dimer_parallel_direction{direction}_product']
            - result[f'bond_triangle0_direction{direction}']*result[f'bond_triangle1_direction{direction}'])
    result['dimer_parallel_connected_average'] = sum(
        result[f'dimer_parallel_direction{direction}_connected'] for direction in range(3))/3
    return result


__all__ = ['singlet_structure_factor_records', 'triangular_torus_structure_records',
    'importance_jackknife_observables', 'scalar_chirality_matrix',
    'compact6_channel_operators', 'compact6_channel_readout']
