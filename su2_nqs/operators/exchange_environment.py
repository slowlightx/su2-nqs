"""Spectator-dependent coefficients of complete physical scalar exchanges.

The coefficient is an operator on disjoint spectator cells, not an arbitrary
function of magnetic spin bits. Spin features are diagonal in local q;
dimer correlation acts on spectator multiplicities. Products must be formed
BEFORE spatial projection: P O F P is generally not (P O P)(P F P).
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .cell_spin_channels import PairAction, _cell, cell_casimir_action, weighted_cell_exchange_action
from .cell_spin_transitions import four_site_dimer_action

EnvironmentFeature = Literal["spin_content", "spin_pair", "dimer_correlation"]
ENVIRONMENT_FEATURES: tuple[EnvironmentFeature, ...] = ("spin_content", "spin_pair", "dimer_correlation")


def spectator_environment_action(
    state: np.ndarray, pair_action: PairAction,
    cell_c: Sequence[int], cell_d: Sequence[int], feature: EnvironmentFeature,
) -> np.ndarray:
    """Apply one dimensionless SU(2)-scalar spectator feature.

    Spin content is (Q_C^2+Q_D^2)/(Cmax+Dmax); spin pair is
    Q_C^2 Q_D^2/(Cmax Dmax), where Cmax=n_C(n_C+2)/4. The dimer
    feature is [R0(C)R0(D)+R1(C)R1(D)]/6 on two four-site cells.
    All spin labels implicit in the Casimirs are doubled integers.
    """
    c, d = _cell(cell_c), _cell(cell_d)
    if set(c) & set(d):
        raise ValueError("spectator cells must be disjoint")
    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim not in (1, 2) or not np.all(np.isfinite(vector)):
        raise ValueError("state must be a finite vector or column batch")
    cmax, dmax = len(c)*(len(c)+2)/4, len(d)*(len(d)+2)/4
    if feature == "spin_content":
        return (cell_casimir_action(vector,pair_action,c)+cell_casimir_action(vector,pair_action,d))/(cmax+dmax)
    if feature == "spin_pair":
        return cell_casimir_action(cell_casimir_action(vector,pair_action,d),pair_action,c)/(cmax*dmax)
    if feature == "dimer_correlation":
        return sum(four_site_dimer_action(four_site_dimer_action(vector,pair_action,d,mu),pair_action,c,mu) for mu in (0,1))/6
    raise ValueError("unknown environment feature")


def environment_modulated_exchange_action(
    state: np.ndarray, pair_action: PairAction, cells: Sequence[Sequence[int]],
    weights_a: Sequence[float], weights_b: Sequence[float],
    feature: EnvironmentFeature,
) -> np.ndarray:
    """Apply O_AB F_CD in a common physical-amplitude representation.

    Disjointness guarantees Hermiticity and commuting action order. Each
    factor is a global SU(2) scalar. A subsequent spatial projector belongs
    to the evaluator, not to either factor separately.
    """
    if len(cells) != 4:
        raise ValueError("four cells A,B,C,D are required")
    a,b,c,d = tuple(_cell(sites) for sites in cells)
    sites = a+b+c+d
    if len(set(sites)) != len(sites):
        raise ValueError("all four cells must be disjoint")
    environment = spectator_environment_action(state,pair_action,c,d,feature)
    return weighted_cell_exchange_action(environment,pair_action,a,b,weights_a,weights_b)
