"""Node-faithful third H moment of a diagonal spin-source state.

One immutable amplitude bank evaluates H phi and H^2 phi at the Q1 centers.
Q1 covers H phi, hence <H phi|H^2 phi>. It does not cover ||H^2 phi||^2;
therefore this module deliberately returns no fourth moment or Ritz residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .spin_response_moments import swap_moment_plan


@dataclass(frozen=True)
class TwoActionPlan:
    """Queries and two sparse exact H rows in physical bits, site zero MSB."""

    configurations: np.ndarray
    proposal_indices: np.ndarray
    center_indices: np.ndarray
    first_h_centers: np.ndarray
    first_h_matrix: object
    second_h_matrix: object


def two_action_plan(states: np.ndarray, bonds: Sequence[tuple[int, int, float]]) -> TwoActionPlan:
    """Deduplicate the Q1 density bank and every reachable two-exchange query.

    This physical representation currently supports at most 62 sites. Runtime
    group theory is absent; all spin-dependent gates are physical bit swaps.
    Repeated bonds combine through CSR construction, as in the H convention.
    """
    from scipy.sparse import csr_matrix

    qplan = swap_moment_plan(states)
    x = np.asarray(states)
    n = x.shape[1]
    if n > 62:
        raise ValueError('this query planner requires at most 62 physical spins')
    checked = []
    for i, j, coupling in bonds:
        if (isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, Integral)
                or not isinstance(j, Integral) or not 0 <= i < n or not 0 <= j < n
                or i == j or not np.isfinite(coupling)):
            raise ValueError('valid physical bond sites and finite exchange required')
        checked.append((1 << (n-1-i), 1 << (n-1-j), float(coupling)))
    powers = 1 << np.arange(n-1, -1, -1, dtype=np.int64)

    def row(word: int) -> dict[int, float]:
        coefficients = {word: 0.}
        for first, second, coupling in checked:
            if bool(word & first) == bool(word & second):
                coefficients[word] += .25*coupling
            else:
                coefficients[word] -= .25*coupling
                swapped = word ^ first ^ second
                coefficients[swapped] = coefficients.get(swapped, 0.) + .5*coupling
        return coefficients

    centers = [int(w) for w in x @ powers]
    center_rows = [row(word) for word in centers]
    first_words = sorted({word for action in center_rows for word in action})
    first_rows = [row(word) for word in first_words]
    proposal_words = [int(w) for w in qplan.configurations @ powers]
    query_words = sorted({*proposal_words, *(word for action in first_rows for word in action)})
    queries = {word: i for i, word in enumerate(query_words)}
    first_positions = {word: i for i, word in enumerate(first_words)}

    def matrix(rows, columns):
        rr, cc, vv = [], [], []
        for i, action in enumerate(rows):
            for word, coefficient in action.items():
                if coefficient:
                    rr.append(i); cc.append(columns[word]); vv.append(coefficient)
        return csr_matrix((vv, (rr, cc)), shape=(len(rows), len(columns)))

    query_bits = ((np.asarray(query_words, dtype=np.int64)[:, None] >>
                   np.arange(n-1, -1, -1)) & 1).astype(np.int8)
    proposal_lookup = np.array([queries[word] for word in proposal_words])
    return TwoActionPlan(
        configurations=query_bits, proposal_indices=proposal_lookup[qplan.indices],
        center_indices=np.array([queries[word] for word in centers]),
        first_h_centers=np.array([queries[word] for word in first_words]),
        first_h_matrix=matrix(first_rows, queries),
        second_h_matrix=matrix(center_rows, first_positions),
    )


def third_source_moments(plan: TwoActionPlan, log_amplitudes: np.ndarray,
                         sources: dict[str, np.ndarray], *, energy_shift: float) -> dict[str, np.ndarray]:
    """Return Q1-weighted m0,m1,m2,m3, plus imaginary-action diagnostics.

    D=H-energy_shift. The third moment is Re[(D phi)^* D^2 phi], so centers
    outside the support of D phi contribute exactly zero even if D^2 phi is
    nonzero. A parent node is never divided by the parent amplitude.
    """
    logs = np.asarray(log_amplitudes, dtype=np.complex128)
    n = plan.configurations.shape[1]
    if (logs.shape != (len(plan.configurations),) or np.isnan(logs).any()
            or np.isposinf(logs.real).any() or not np.isfinite(logs.imag).all()
            or not np.isfinite(energy_shift) or not sources or 'parent' in sources):
        raise ValueError('valid finite phases, source bank and energy shift required')
    amplitude = np.exp(logs - np.max(logs.real))
    density = np.mean(abs(amplitude[plan.proposal_indices])**2, axis=1)
    if not np.isfinite(density).all() or np.any(density <= 0):
        raise ValueError('zero or invalid Q1 density')
    output = {}
    for name, phase in [('parent', None), *sources.items()]:
        if phase is not None and (np.asarray(phase).shape != (n,) or not np.isfinite(phase).all()):
            raise ValueError('one finite source coefficient per physical site required')
        diagonal = (np.ones(len(amplitude)) if phase is None else
                    (plan.configurations-.5) @ np.asarray(phase) / np.sqrt(n))
        phi = amplitude*diagonal
        d_first = plan.first_h_matrix @ phi - energy_shift*phi[plan.first_h_centers]
        d_center = plan.second_h_matrix @ phi[plan.first_h_centers] - energy_shift*phi[plan.center_indices]
        d2_center = plan.second_h_matrix @ d_first - energy_shift*d_center
        center = phi[plan.center_indices]
        values = dict(m0=abs(center)**2, m1=(center.conj()*d_center).real,
                      m2=abs(d_center)**2, m3=(d_center.conj()*d2_center).real,
                      m1_imag=(center.conj()*d_center).imag,
                      m3_imag=(d_center.conj()*d2_center).imag)
        output.update({name+'__'+key: value/density for key,value in values.items()})
    return output
