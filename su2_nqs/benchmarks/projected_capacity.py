"""Physical-metric and exact small-budget controls for cached CT tangents.

No Hamiltonian or wavefunction is defined here.  The dictionary must already
have passed the caller's strict SU(2), coherent spatial-projection and
finite-difference validation.  These diagnostics cannot certify a finite
update's energy/variance or an actual reduction in padded tensor cost.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..su2.quantum_numbers import validate_q


def tangent_moments(scores: np.ndarray, local_energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FS metric and Re Cov(O, E_local) for the full Hamiltonian.

    The returned force is HALF the real-parameter energy derivative; callers
    must use ``2*force.dot(direction)`` for the full linear energy slope.
    """
    scores, local_energies = np.asarray(scores), np.asarray(local_energies)
    if scores.ndim != 2 or local_energies.shape != (len(scores),) or len(scores) < 2:
        raise ValueError("at least two aligned scores/energies required")
    if not np.isfinite(scores).all() or not np.isfinite(local_energies).all():
        raise ValueError("nonfinite cached moments")
    centered = scores-scores.mean(axis=0)
    energy = local_energies-local_energies.mean()
    gram = np.real(centered.conj().T @ centered)/len(scores)
    force = np.real(centered.conj().T @ energy)/len(scores)
    return (gram+gram.T)/2, force


def budgeted_multiplet_actions(
    merit: Sequence[float], native_cost: Sequence[int], doubled_q: Sequence[int],
    *, native_budget: int, magnetic_budget: int,
    allowed: Sequence[bool] | None = None,
) -> np.ndarray:
    """Exact 0/1 selection under native-scalar AND full-multiplet budgets.

    An action is indivisible: its magnetic cost is q+1, never a single m copy.
    The supplied native cost must include *both* tensors attached to a changed
    edge. Rotations do not delete a channel: for them this is only a declared
    action-cost proxy, not a parameter-saving certificate. Negative merits
    are never selected; empty selection is legal. Equal-merit ties are broken
    by lower costs and then lexicographic action ids, independently of labels.
    """
    values, costs, qs = np.asarray(merit, float), np.asarray(native_cost), tuple(doubled_q)
    count = len(values)
    if values.ndim != 1 or costs.shape != (count,) or len(qs) != count:
        raise ValueError("one merit, native cost and doubled q per action required")
    if not np.isfinite(values).all() or costs.dtype.kind not in "iu" or np.any(costs <= 0):
        raise ValueError("finite merits and positive integer native costs required")
    qs = tuple(validate_q(q) for q in qs)
    if any(isinstance(b, bool) or not isinstance(b, (int, np.integer)) or b < 0
           for b in (native_budget, magnetic_budget)):
        raise ValueError("nonnegative integer budgets required")
    mask = np.ones(count, bool) if allowed is None else np.asarray(allowed, bool)
    if mask.shape != (count,):
        raise ValueError("one legal-action flag required per token")
    # Sparse dynamic programming retains the best set for each exact cost pair.
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0., ())}
    for index in range(count):
        if not mask[index] or values[index] <= 0:
            continue
        for (native, magnetic), (value, indices) in tuple(states.items()):
            new_cost = native+int(costs[index]), magnetic+qs[index]+1
            if new_cost[0] > native_budget or new_cost[1] > magnetic_budget:
                continue
            proposal = value+float(values[index]), indices+(index,)
            current = states.get(new_cost)
            if current is None or proposal[0] > current[0] or (proposal[0] == current[0] and proposal[1] < current[1]):
                states[new_cost] = proposal
    best = min(states, key=lambda cost: (-states[cost][0], cost[0], cost[1], states[cost][1]))
    return np.asarray(states[best][1], dtype=int)


def restricted_natural_direction(
    gram: np.ndarray, force: np.ndarray, selected: Sequence[int],
    *, ridge: float = .01, fs_radius: float = .001,
) -> np.ndarray:
    """Fit real coefficients ONLY with fit-split moments, at fixed fit FS.

    Diagonal scaling makes ridge dimensionless, and zero-norm directions are
    discarded. This is the free selected-subspace control, not a controller.
    """
    gram, force = np.asarray(gram, float), np.asarray(force, float)
    if (force.ndim != 1 or gram.shape != (len(force), len(force))
            or not np.isfinite(gram).all() or not np.isfinite(force).all()
            or ridge < 0 or fs_radius <= 0):
        raise ValueError("valid physical moments, ridge and positive radius required")
    selected = np.asarray(selected, int)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(force)) or len(np.unique(selected)) != len(selected):
        raise ValueError("selected actions must be distinct valid indices")
    result = np.zeros_like(force)
    if not len(selected):
        return result
    sub = gram[np.ix_(selected, selected)]
    diagonal = np.maximum(np.diag(sub), 0.)
    good = diagonal > max(diagonal.max(initial=0.)*1e-12, 1e-24)
    ids, diagonal = selected[good], diagonal[good]
    if not len(ids):
        return result
    scale = np.sqrt(diagonal)
    normalized = gram[np.ix_(ids, ids)]/scale[:, None]/scale[None, :]
    values, vectors = np.linalg.eigh((normalized+normalized.T)/2)
    inverse = np.where(values > 1e-10, 1/(np.maximum(values, 0)+ridge), 0.)
    local = -((vectors*inverse) @ (vectors.T @ (force[ids]/scale)))/scale
    norm_squared = float(local @ gram[np.ix_(ids, ids)] @ local)
    if norm_squared > 1e-24:
        result[ids] = local*(fs_radius/np.sqrt(norm_squared))
    return result


def conditional_greedy_actions(
    gram: np.ndarray, force: np.ndarray, native_cost: Sequence[int],
    doubled_q: Sequence[int], *, native_budget: int, magnetic_budget: int,
    allowed: Sequence[bool] | None = None, ridge: float = .01,
) -> np.ndarray:
    """Strong fit-moment control: add conditional residual gain per native cost.

    The full fitted Gram removes information already spanned by retained
    actions, unlike independent squared-force scores. This is a greedy
    conditional oracle, NOT the exact combinatorial optimum, and it uses no
    held-out scores or energies. All magnetic and native budget semantics are
    the same as :func:`budgeted_multiplet_actions`.
    """
    gram, force = np.asarray(gram, float), np.asarray(force, float)
    count, costs = len(force), np.asarray(native_cost)
    if gram.shape != (count, count) or ridge < 0:
        raise ValueError("valid Gram, force and nonnegative ridge required")
    # Reuse complete validation without selecting anything.
    budgeted_multiplet_actions(np.zeros(count), costs, doubled_q,
        native_budget=native_budget, magnetic_budget=magnetic_budget, allowed=allowed)
    mask = np.ones(count, bool) if allowed is None else np.asarray(allowed, bool)
    selected: list[int] = []
    native = magnetic = 0

    def explained(ids: list[int]) -> float:
        if not ids:
            return 0.
        sub = gram[np.ix_(ids, ids)]
        diagonal = np.maximum(np.diag(sub), 0.)
        regularized = sub+ridge*np.diag(diagonal)
        return float(force[ids] @ np.linalg.pinv(regularized, rcond=1e-10) @ force[ids])

    current = 0.
    while True:
        candidates = []
        for index in range(count):
            if (not mask[index] or index in selected
                    or native+costs[index] > native_budget
                    or magnetic+doubled_q[index]+1 > magnetic_budget):
                continue
            value = explained(selected+[index])
            gain = value-current
            if gain > 1e-14*max(1., abs(current), abs(value)):
                candidates.append((-gain/costs[index], -gain, index, value))
        if not candidates:
            break
        _, _, index, current = min(candidates)
        selected.append(index)
        native += int(costs[index])
        magnetic += doubled_q[index]+1
    return np.asarray(selected, int)


def heldout_direction_metrics(
    direction: np.ndarray, gram: np.ndarray, force: np.ndarray,
) -> dict[str, float]:
    """Audit tangent slope and physical norm without refitting the direction."""
    direction, gram, force = np.asarray(direction), np.asarray(gram), np.asarray(force)
    if (direction.ndim != 1 or force.shape != direction.shape
            or gram.shape != (direction.size, direction.size)
            or not all(np.isfinite(value).all() for value in (direction, gram, force))):
        raise ValueError("finite aligned real direction, physical metric and force required")
    norm = float(np.sqrt(max(float(direction @ gram @ direction), 0.)))
    slope = float(2*force @ direction)
    return {"fs_tangent": norm, "energy_slope_total": slope,
            "energy_slope_per_fs_total": slope/norm if norm > 0 else 0.}
