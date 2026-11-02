"""Exact SU(2) exchange-gate and connected-commutator references.

For spin half, a bond imaginary-time gate is exactly linear in the exchange
operator because ``h_ij = S_i.S_j`` has only the singlet and triplet
eigenvalues.  Products of gates generate all disconnected histories without
new variational coefficients.  Their leading non-factorizing error is a sum
of commutators of overlapping bonds; disjoint-bond commutators vanish.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ExchangeCommutatorAction:
    """Leading Lie-product error split by physical bond connectivity."""

    connected: np.ndarray
    disconnected: np.ndarray
    total: np.ndarray
    connected_pair_count: int
    disconnected_pair_count: int
    connected_incoherent_norm: float
    disconnected_incoherent_norm: float

    @property
    def disconnected_relative_norm(self) -> float:
        """Norm of disjoint-bond commutators relative to the full action."""

        denominator = max(np.linalg.norm(self.total), np.finfo(float).tiny)
        return float(np.linalg.norm(self.disconnected) / denominator)


@dataclass(frozen=True)
class ExchangeSecondOrderAction:
    """Exact ``H**2 psi`` split into local operator-product sectors.

    ``connected_jordan`` contains ``J_b J_c {h_b,h_c} psi`` for bonds that
    share one site.  ``connected_commutator`` contains the corresponding
    oriented half-commutators and diagnoses schedule dependence; it is not a
    term in ``H**2``.  Disjoint bonds commute, so their ordered-pair sum is
    stored once in ``disconnected``.  Columns retain individual bond or bond-
    pair actions for exact small-sector projection oracles.
    """

    same_bond: np.ndarray
    connected_jordan: np.ndarray
    disconnected: np.ndarray
    total: np.ndarray
    connected_commutator: np.ndarray
    same_bond_columns: np.ndarray
    connected_jordan_columns: np.ndarray
    disconnected_columns: np.ndarray
    connected_commutator_columns: np.ndarray
    connected_pair_count: int
    disconnected_pair_count: int


def exchange_gate_coefficients(step: Real, coupling: Real) -> tuple[float, float]:
    """Return ``a, b`` with ``exp(-step J h) = a I + b h`` for spin half."""

    tau = float(step)
    weight = float(coupling)
    if not np.isfinite(tau) or not np.isfinite(weight):
        raise ValueError("step and coupling must be finite")
    triplet = float(np.exp(-tau * weight / 4.0))
    singlet = float(np.exp(3.0 * tau * weight / 4.0))
    return 0.75 * triplet + 0.25 * singlet, triplet - singlet


def apply_exchange_gate(
    initial: np.ndarray, operator: object, step: Real, coupling: Real
) -> np.ndarray:
    """Apply one exact spin-half bond imaginary-time gate."""

    vector = np.asarray(initial, dtype=np.complex128)
    if vector.ndim != 1 or len(vector) < 1:
        raise ValueError("initial must be a nonempty vector")
    if getattr(operator, "shape", None) != (len(vector), len(vector)):
        raise ValueError("operator has the wrong shape")
    first, second = exchange_gate_coefficients(step, coupling)
    return first * vector + second * np.asarray(operator @ vector)


def _validated_problem(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    bond_endpoints: Sequence[tuple[int, int]],
    couplings: Sequence[Real],
) -> tuple[
    np.ndarray,
    tuple[object, ...],
    tuple[tuple[int, int], ...],
    np.ndarray,
]:
    vector = np.asarray(initial, dtype=np.complex128)
    if vector.ndim != 1 or len(vector) < 1 or not np.all(np.isfinite(vector)):
        raise ValueError("initial must be a nonempty finite vector")
    operators = tuple(local_exchange_operators)
    endpoints = tuple((int(first), int(second)) for first, second in bond_endpoints)
    weights = np.asarray(couplings, dtype=np.float64)
    if not operators:
        raise ValueError("at least one local exchange operator is required")
    if len(endpoints) != len(operators) or weights.shape != (len(operators),):
        raise ValueError("operators, endpoints, and couplings must have equal length")
    if not np.all(np.isfinite(weights)):
        raise ValueError("couplings must be finite")
    for index, (operator, endpoint) in enumerate(zip(operators, endpoints, strict=True)):
        if getattr(operator, "shape", None) != (len(vector), len(vector)):
            raise ValueError(f"local operator {index} has the wrong shape")
        if endpoint[0] == endpoint[1] or min(endpoint) < 0:
            raise ValueError("each bond must join distinct non-negative sites")
    return vector, operators, endpoints, weights


def lie_exchange_flow(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    couplings: Sequence[Real],
    step: Real,
) -> np.ndarray:
    """Apply one ordered first-order product of exact exchange gates."""

    vector = np.asarray(initial, dtype=np.complex128)
    operators = tuple(local_exchange_operators)
    weights = tuple(couplings)
    if len(operators) != len(weights):
        raise ValueError("operators and couplings must have equal length")
    for operator, coupling in zip(operators, weights, strict=True):
        vector = apply_exchange_gate(vector, operator, step, coupling)
    return vector


def strang_exchange_flow(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    couplings: Sequence[Real],
    step: Real,
) -> np.ndarray:
    """Apply a palindromic second-order product of exact exchange gates."""

    vector = np.asarray(initial, dtype=np.complex128)
    operators = tuple(local_exchange_operators)
    weights = tuple(couplings)
    if len(operators) != len(weights):
        raise ValueError("operators and couplings must have equal length")
    for operator, coupling in zip(operators, weights, strict=True):
        vector = apply_exchange_gate(vector, operator, 0.5 * float(step), coupling)
    for operator, coupling in zip(
        reversed(operators), reversed(weights), strict=True
    ):
        vector = apply_exchange_gate(vector, operator, 0.5 * float(step), coupling)
    return vector


def twirled_strang_exchange_flow(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    couplings: Sequence[Real],
    step: Real,
    symmetry_operators: Sequence[object],
) -> np.ndarray:
    """Apply the group twirl of a fixed-order Strang exchange flow.

    For a finite unitary representation ``R_g``, this returns

    ``|G|^-1 sum_g R_g U_Strang R_g^dagger initial``.

    A complete group catalogue makes the resulting linear operator commute
    with every ``R_g`` even though the underlying fixed bond schedule need not.
    No orbit-dependent trainable coefficient is introduced.
    """

    vector = np.asarray(initial, dtype=np.complex128)
    symmetries = tuple(symmetry_operators)
    if vector.ndim != 1 or len(vector) < 1:
        raise ValueError("initial must be a nonempty vector")
    if not symmetries:
        raise ValueError("at least one symmetry operator is required")
    result = np.zeros_like(vector)
    for index, symmetry in enumerate(symmetries):
        if getattr(symmetry, "shape", None) != (len(vector), len(vector)):
            raise ValueError(f"symmetry operator {index} has the wrong shape")
        pulled_back = np.asarray(symmetry.conj().T @ vector, dtype=np.complex128)
        flowed = strang_exchange_flow(
            pulled_back, local_exchange_operators, couplings, step
        )
        result += np.asarray(symmetry @ flowed, dtype=np.complex128)
    return result / len(symmetries)


def exchange_commutator_action(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    bond_endpoints: Sequence[tuple[int, int]],
    couplings: Sequence[Real],
) -> ExchangeCommutatorAction:
    """Return the exact ``tau**2`` coefficient of Lie-product error.

    With gates applied in the supplied order, the coefficient is

    ``1/2 sum_(c>b) J_c J_b [h_c, h_b] psi``.

    Every disjoint term must vanish.  For overlapping spin exchanges the
    commutator is a local three-spin scalar-chirality operator up to its
    orientation and a factor of ``i``.
    """

    vector, operators, endpoints, weights = _validated_problem(
        initial, local_exchange_operators, bond_endpoints, couplings
    )
    first_actions = tuple(
        np.asarray(operator @ vector, dtype=np.complex128)
        for operator in operators
    )
    connected = np.zeros_like(vector)
    disconnected = np.zeros_like(vector)
    connected_squared = 0.0
    disconnected_squared = 0.0
    connected_count = 0
    disconnected_count = 0
    endpoint_sets = tuple(set(pair) for pair in endpoints)
    for first in range(len(operators)):
        for second in range(first + 1, len(operators)):
            bare_term = 0.5 * (
                np.asarray(operators[second] @ first_actions[first])
                - np.asarray(operators[first] @ first_actions[second])
            )
            term = weights[second] * weights[first] * bare_term
            squared = float(np.vdot(term, term).real)
            if endpoint_sets[first] & endpoint_sets[second]:
                connected += term
                connected_squared += squared
                connected_count += 1
            else:
                disconnected += term
                disconnected_squared += squared
                disconnected_count += 1
    return ExchangeCommutatorAction(
        connected=connected,
        disconnected=disconnected,
        total=connected + disconnected,
        connected_pair_count=connected_count,
        disconnected_pair_count=disconnected_count,
        connected_incoherent_norm=float(np.sqrt(connected_squared)),
        disconnected_incoherent_norm=float(np.sqrt(disconnected_squared)),
    )


def exchange_second_order_action(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    bond_endpoints: Sequence[tuple[int, int]],
    couplings: Sequence[Real],
) -> ExchangeSecondOrderAction:
    """Return an exact local-product decomposition of ``H**2 initial``.

    For ``H = sum_b J_b h_b``, the unordered pair ``b<c`` contributes
    ``J_b J_c (h_c h_b + h_b h_c)``.  The function separates pairs whose
    physical bonds overlap from pairs that are disjoint and also returns the
    half-commutator ``J_b J_c [h_c,h_b]/2`` for the same connected catalogue.
    The latter is the leading Lie-schedule error, whereas a palindromic Strang
    flow retains the symmetric Jordan product and cancels that commutator at
    second order.
    """

    vector, operators, endpoints, weights = _validated_problem(
        initial, local_exchange_operators, bond_endpoints, couplings
    )
    first_actions = tuple(
        np.asarray(operator @ vector, dtype=np.complex128)
        for operator in operators
    )
    endpoint_sets = tuple(set(pair) for pair in endpoints)
    same_columns = []
    connected_columns = []
    disconnected_columns = []
    commutator_columns = []
    for bond, operator in enumerate(operators):
        same_columns.append(
            weights[bond] ** 2
            * np.asarray(operator @ first_actions[bond], dtype=np.complex128)
        )
    for first in range(len(operators)):
        for second in range(first + 1, len(operators)):
            forward = np.asarray(
                operators[second] @ first_actions[first], dtype=np.complex128
            )
            reverse = np.asarray(
                operators[first] @ first_actions[second], dtype=np.complex128
            )
            weight = weights[first] * weights[second]
            jordan = weight * (forward + reverse)
            if endpoint_sets[first] & endpoint_sets[second]:
                connected_columns.append(jordan)
                commutator_columns.append(0.5 * weight * (forward - reverse))
            else:
                disconnected_columns.append(jordan)

    def matrix(columns: list[np.ndarray]) -> np.ndarray:
        if not columns:
            return np.zeros((len(vector), 0), dtype=np.complex128)
        return np.column_stack(columns)

    same_matrix = matrix(same_columns)
    connected_matrix = matrix(connected_columns)
    disconnected_matrix = matrix(disconnected_columns)
    commutator_matrix = matrix(commutator_columns)

    def summed(columns: np.ndarray) -> np.ndarray:
        return np.sum(columns, axis=1) if columns.shape[1] else np.zeros_like(vector)

    same = summed(same_matrix)
    connected = summed(connected_matrix)
    disconnected = summed(disconnected_matrix)
    commutator = summed(commutator_matrix)
    return ExchangeSecondOrderAction(
        same_bond=same,
        connected_jordan=connected,
        disconnected=disconnected,
        total=same + connected + disconnected,
        connected_commutator=commutator,
        same_bond_columns=same_matrix,
        connected_jordan_columns=connected_matrix,
        disconnected_columns=disconnected_matrix,
        connected_commutator_columns=commutator_matrix,
        connected_pair_count=connected_matrix.shape[1],
        disconnected_pair_count=disconnected_matrix.shape[1],
    )


def exchange_commutator_motif_diagnostics(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    bond_endpoints: Sequence[tuple[int, int]],
    couplings: Sequence[Real],
    bond_shells: Sequence[str],
) -> dict[str, object]:
    """Resolve local chirality-fluctuation weight by three-site motif.

    An overlapping bond pair is labeled by its two input shells and by the
    shell, if any, connecting its two outer sites.  The latter distinguishes a
    closed triangular exchange loop from an open wedge.  Squared norms are
    accumulated incoherently, making the diagnostic independent of pair
    orientation and of cancellations caused by a Trotter schedule.
    """

    vector, operators, endpoints, weights = _validated_problem(
        initial, local_exchange_operators, bond_endpoints, couplings
    )
    shells = tuple(str(value) for value in bond_shells)
    if len(shells) != len(operators) or any(not value for value in shells):
        raise ValueError("bond_shells must supply one nonempty label per operator")
    closure_shells: dict[frozenset[int], set[str]] = {}
    for endpoint, shell in zip(endpoints, shells, strict=True):
        closure_shells.setdefault(frozenset(endpoint), set()).add(shell)
    first_actions = tuple(
        np.asarray(operator @ vector, dtype=np.complex128)
        for operator in operators
    )
    norm_squared = float(np.vdot(vector, vector).real)
    sectors: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for first in range(len(operators)):
        first_sites = set(endpoints[first])
        for second in range(first + 1, len(operators)):
            second_sites = set(endpoints[second])
            common = first_sites & second_sites
            if not common:
                continue
            outer = first_sites ^ second_sites
            if len(common) != 1 or len(outer) != 2:
                raise ValueError("overlapping catalogue bonds must form a three-site motif")
            closing = closure_shells.get(frozenset(outer), set())
            closure = "+".join(sorted(closing)) if closing else "open"
            shell_pair = tuple(sorted((shells[first], shells[second])))
            key = (shell_pair[0], shell_pair[1], closure)
            bare_term = 0.5 * (
                np.asarray(operators[second] @ first_actions[first])
                - np.asarray(operators[first] @ first_actions[second])
            )
            term = weights[second] * weights[first] * bare_term
            expectation = np.vdot(vector, term) / norm_squared
            perpendicular = term - expectation * vector
            entry = sectors.setdefault(
                key,
                {
                    "pair_count": 0,
                    "incoherent_norm_squared": 0.0,
                    "bare_incoherent_norm_squared": 0.0,
                    "orthogonal_incoherent_norm_squared": 0.0,
                    "maximum_expectation_absolute": 0.0,
                },
            )
            entry["pair_count"] += 1
            entry["incoherent_norm_squared"] += float(np.vdot(term, term).real)
            entry["bare_incoherent_norm_squared"] += float(
                np.vdot(bare_term, bare_term).real
            )
            entry["orthogonal_incoherent_norm_squared"] += float(
                np.vdot(perpendicular, perpendicular).real
            )
            entry["maximum_expectation_absolute"] = max(
                float(entry["maximum_expectation_absolute"]), abs(expectation)
            )
    total = sum(
        float(entry["incoherent_norm_squared"]) for entry in sectors.values()
    )
    bare_total = sum(
        float(entry["bare_incoherent_norm_squared"])
        for entry in sectors.values()
    )
    rows = []
    for (first_shell, second_shell, closure), entry in sorted(sectors.items()):
        rows.append(
            {
                "first_shell": first_shell,
                "second_shell": second_shell,
                "outer_closure_shell": closure,
                **entry,
                "incoherent_norm_squared_fraction": float(
                    entry["incoherent_norm_squared"]
                )
                / max(total, np.finfo(float).tiny),
                "bare_incoherent_norm_squared_fraction": float(
                    entry["bare_incoherent_norm_squared"]
                )
                / max(bare_total, np.finfo(float).tiny),
                "bare_mean_norm_squared_per_pair": float(
                    entry["bare_incoherent_norm_squared"]
                )
                / int(entry["pair_count"]),
            }
        )
    return {
        "connected_pair_count": sum(int(row["pair_count"]) for row in rows),
        "total_incoherent_norm_squared": total,
        "total_bare_incoherent_norm_squared": bare_total,
        "closed_loop_incoherent_norm_squared_fraction": sum(
            float(row["incoherent_norm_squared_fraction"])
            for row in rows
            if row["outer_closure_shell"] != "open"
        ),
        "closed_loop_bare_incoherent_norm_squared_fraction": sum(
            float(row["bare_incoherent_norm_squared_fraction"])
            for row in rows
            if row["outer_closure_shell"] != "open"
        ),
        "maximum_expectation_absolute": max(
            (float(row["maximum_expectation_absolute"]) for row in rows),
            default=0.0,
        ),
        "sectors": rows,
    }


def orthogonal_component(initial: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Remove the component parallel to ``initial``."""

    vector = np.asarray(initial, dtype=np.complex128)
    candidate = np.asarray(value, dtype=np.complex128)
    if vector.ndim != 1 or candidate.shape != vector.shape:
        raise ValueError("initial and value must be equal-length vectors")
    norm_squared = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("initial must have positive finite norm")
    return candidate - vector * (np.vdot(vector, candidate) / norm_squared)


__all__ = [
    "ExchangeCommutatorAction",
    "ExchangeSecondOrderAction",
    "apply_exchange_gate",
    "exchange_commutator_action",
    "exchange_commutator_motif_diagnostics",
    "exchange_gate_coefficients",
    "exchange_second_order_action",
    "lie_exchange_flow",
    "orthogonal_component",
    "strang_exchange_flow",
    "twirled_strang_exchange_flow",
]
