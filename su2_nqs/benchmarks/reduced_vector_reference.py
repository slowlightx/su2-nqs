"""Independent computational-basis references for reduced spin vectors.

The routines here deliberately materialize ``2**N`` states and operators.
They are offline validation tools for ``N<=8`` and must not be called by a
production neural forward pass.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Sequence

import numpy as np

from .coupled_basis import coupled_state_to_sz
from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.fusion_path import FusionPath, enumerate_paths
from ..su2.quantum_numbers import (
    is_admissible_triangle,
    magnetic_numbers,
)
from ..su2.reduced_vector import (
    Q_OP_VECTOR,
    ReducedVectorTable,
    SPHERICAL_MU_Q,
    make_reduced_vector_table,
)


def _weights(values: Sequence[complex]) -> np.ndarray:
    raw = tuple(values)
    if not raw:
        raise ValueError("endpoint_weights must contain at least one site")
    if any(isinstance(value, bool) or not isinstance(value, Number) for value in raw):
        raise TypeError("endpoint_weights must be numeric")
    answer = np.asarray(raw, dtype=np.complex128)
    if np.any(~np.isfinite(answer)):
        raise ValueError("endpoint_weights must be finite")
    return answer


def _paths(n_sites: int) -> tuple[FusionPath, ...]:
    return tuple(
        path
        for q in range(n_sites % 2, n_sites + 1, 2)
        for path in enumerate_paths(n_sites, q)
    )


def _spin_element(mu_q: int, m_new: int, m_old: int) -> float:
    if mu_q == 0 and m_new == m_old:
        return 0.5 * m_old
    if mu_q == 2 and (m_new, m_old) == (1, -1):
        return -1.0 / math.sqrt(2.0)
    if mu_q == -2 and (m_new, m_old) == (-1, 1):
        return 1.0 / math.sqrt(2.0)
    return 0.0


def _operator(
    n_sites: int, endpoint_weights: np.ndarray, mu_q: int
) -> np.ndarray:
    local = np.asarray(
        [
            [_spin_element(mu_q, row, column) for column in (-1, 1)]
            for row in (-1, 1)
        ],
        dtype=np.complex128,
    )
    identity = np.eye(2, dtype=np.complex128)
    answer = np.zeros((1 << n_sites, 1 << n_sites), dtype=np.complex128)
    for site, weight in enumerate(endpoint_weights):
        term = np.asarray([[1.0 + 0.0j]])
        for position in range(n_sites):
            term = np.kron(term, local if position == site else identity)
        answer += weight * term
    return answer


def _project(
    q_new: int,
    q_old: int,
    new_path: FusionPath,
    old_path: FusionPath,
    states: dict[tuple[FusionPath, int], np.ndarray],
    operators: dict[int, np.ndarray],
) -> complex:
    total = 0.0j
    for m_old in magnetic_numbers(q_old):
        for mu_q in SPHERICAL_MU_Q:
            m_new = m_old + mu_q
            if abs(m_new) > q_new or (q_new - m_new) % 2:
                continue
            coefficient = clebsch_gordan(
                q_old, m_old, Q_OP_VECTOR, mu_q, q_new, m_new
            )
            if coefficient:
                total += coefficient * np.vdot(
                    states[(new_path, m_new)],
                    operators[mu_q] @ states[(old_path, m_old)],
                )
    return total / math.sqrt(q_new + 1.0)


def compile_weighted_spin_vector_reference(
    endpoint_weights: Sequence[complex],
    *,
    zero_tolerance: float = 2.0e-14,
) -> ReducedVectorTable:
    """Compile by explicit computational states for independent ``N<=8`` tests."""

    weights = _weights(endpoint_weights)
    n_sites = len(weights)
    paths = _paths(n_sites)
    operators = {
        mu_q: _operator(n_sites, weights, mu_q) for mu_q in SPHERICAL_MU_Q
    }
    states = {
        (path, m): coupled_state_to_sz(path, path[-1], m)
        for path in paths
        for m in magnetic_numbers(path[-1])
    }
    entries: dict[tuple[FusionPath, FusionPath], complex] = {}
    for new_path in paths:
        q_new = new_path[-1]
        for old_path in paths:
            q_old = old_path[-1]
            if not is_admissible_triangle(q_old, Q_OP_VECTOR, q_new):
                continue
            value = _project(
                q_new,
                q_old,
                new_path,
                old_path,
                states,
                operators,
            )
            if abs(value) > zero_tolerance:
                entries[(new_path, old_path)] = value
    return make_reduced_vector_table(
        weights,
        entries,
        compiler="computational_basis_reference",
        zero_tolerance=zero_tolerance,
    )


__all__ = ["compile_weighted_spin_vector_reference"]
