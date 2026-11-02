"""SU(2)-reduced two- and three-spin local-state tomography.

For a global singlet, every local reduced density matrix commutes with the
diagonal SU(2) action.  Two spins therefore require only one pair correlator.
For three spins the complete invariant Hermitian algebra is resolved by the
identity, three pair correlators, and one oriented scalar chirality.  This
module reconstructs the density matrices from those reduced observables and
provides independent computational-basis partial traces for exact-small
validation.  The reconstruction is complete only for a global singlet (or an
explicitly SU(2)-twirled multiplet); it must not be applied directly to one
fixed-M component of a triplet.  Arbitrary input moments need not describe a
positive density matrix, so callers interpreting external/noisy data must
check positivity separately.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


def _local_spin_operators(n_sites: int) -> tuple[tuple[np.ndarray, ...], ...]:
    identity = np.eye(2, dtype=np.complex128)
    pauli = (
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128) / 2.0,
        np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128) / 2.0,
        np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128) / 2.0,
    )
    result: list[tuple[np.ndarray, ...]] = []
    for site in range(n_sites):
        components = []
        for local in pauli:
            value = np.asarray([[1.0]], dtype=np.complex128)
            for index in range(n_sites):
                value = np.kron(value, local if index == site else identity)
            components.append(value)
        result.append(tuple(components))
    return tuple(result)


def _pair_operator(
    spins: tuple[tuple[np.ndarray, ...], ...], first: int, second: int
) -> np.ndarray:
    return sum(
        (spins[first][component] @ spins[second][component] for component in range(3)),
        start=np.zeros_like(spins[0][0]),
    )


def _chirality_operator(
    spins: tuple[tuple[np.ndarray, ...], ...],
    first: int,
    middle: int,
    last: int,
) -> np.ndarray:
    answer = np.zeros_like(spins[0][0])
    for alpha, beta, gamma, sign in (
        (0, 1, 2, 1.0),
        (1, 2, 0, 1.0),
        (2, 0, 1, 1.0),
        (0, 2, 1, -1.0),
        (2, 1, 0, -1.0),
        (1, 0, 2, -1.0),
    ):
        answer += sign * (
            spins[first][alpha] @ spins[middle][beta] @ spins[last][gamma]
        )
    return answer


_TWO_SPINS = _local_spin_operators(2)
_TWO_BASIS = (
    np.eye(4, dtype=np.complex128),
    _pair_operator(_TWO_SPINS, 0, 1),
)
_THREE_SPINS = _local_spin_operators(3)
_THREE_BASIS = (
    np.eye(8, dtype=np.complex128),
    _pair_operator(_THREE_SPINS, 0, 1),
    _pair_operator(_THREE_SPINS, 1, 2),
    _pair_operator(_THREE_SPINS, 0, 2),
    _chirality_operator(_THREE_SPINS, 0, 1, 2),
)


def _reconstruct_from_moments(
    basis: Sequence[np.ndarray], moments: Sequence[complex], *, hermitize: bool = True
) -> np.ndarray:
    operators = tuple(np.asarray(value, dtype=np.complex128) for value in basis)
    values = np.asarray(tuple(moments), dtype=np.complex128)
    if len(operators) != len(values):
        raise ValueError("one moment is required for every invariant operator")
    gram = np.asarray(
        [[np.trace(left @ right) for left in operators] for right in operators],
        dtype=np.complex128,
    )
    coefficients = np.linalg.solve(gram, values)
    result = sum(
        (coefficient * operator for coefficient, operator in zip(coefficients, operators, strict=True)),
        start=np.zeros_like(operators[0]),
    )
    if hermitize:
        result = 0.5 * (result + result.conj().T)
    return np.asarray(result, dtype=np.complex128)


def two_spin_su2_rdm(pair_correlation: float) -> np.ndarray:
    """Reconstruct ``rho=I/4+(4/3)<S0.S1> S0.S1``.

    Consequently the trace distance of two physical invariant two-spin
    states is exactly the absolute difference of their pair correlators.
    """

    value = float(pair_correlation)
    if not np.isfinite(value):
        raise ValueError("pair_correlation must be finite")
    return _reconstruct_from_moments(_TWO_BASIS, (1.0, value))


def three_spin_su2_rdm(
    pair_correlations: Sequence[float], scalar_chirality: float
) -> np.ndarray:
    """Reconstruct a normalized SU(2)-invariant three-spin density matrix.

    Pair correlations are ordered ``(01,12,02)`` and chirality is
    ``S0 . (S1 x S2)``.  Explicitly,
    ``rho=I/8+(2/3)*sum_ij c_ij S_i.S_j+(4/3)*kappa*chi``.

    A zero ``kappa`` for a time-reversal-real nondegenerate singlet does not
    exclude a chiral susceptibility or chiral combinations of a degenerate
    manifold.
    """

    pairs = tuple(float(value) for value in pair_correlations)
    chirality = float(scalar_chirality)
    if len(pairs) != 3 or not all(np.isfinite(value) for value in pairs + (chirality,)):
        raise ValueError("three finite pair correlations and one chirality are required")
    return _reconstruct_from_moments(_THREE_BASIS, (1.0, *pairs, chirality))


def two_spin_su2_transition(
    overlap: complex, pair_transition: complex
) -> np.ndarray:
    """Reconstruct an invariant two-site transition RDM.

    Moments mean ``overlap=<right|left>`` and
    ``pair_transition=<right|S0.S1|left>``.  The result is generally
    non-Hermitian.
    """

    values = (complex(overlap), complex(pair_transition))
    if not all(np.isfinite(value.real) and np.isfinite(value.imag) for value in values):
        raise ValueError("transition moments must be finite")
    return _reconstruct_from_moments(_TWO_BASIS, values, hermitize=False)


def three_spin_su2_transition(
    overlap: complex,
    pair_transitions: Sequence[complex],
    chirality_transition: complex,
) -> np.ndarray:
    """Reconstruct a complete invariant three-site transition RDM."""

    pairs = tuple(complex(value) for value in pair_transitions)
    chirality = complex(chirality_transition)
    values = (complex(overlap), *pairs, chirality)
    if len(pairs) != 3 or not all(
        np.isfinite(value.real) and np.isfinite(value.imag) for value in values
    ):
        raise ValueError("complete finite complex transition moments are required")
    return _reconstruct_from_moments(_THREE_BASIS, values, hermitize=False)


def local_invariant_moments(rdm: np.ndarray) -> dict[str, object]:
    """Return pair and chirality moments from a two- or three-spin RDM."""

    value = np.asarray(rdm, dtype=np.complex128)
    if value.shape == (4, 4):
        return {"pair_correlations": (float(np.trace(value @ _TWO_BASIS[1]).real),)}
    if value.shape == (8, 8):
        moments = [float(np.trace(value @ operator).real) for operator in _THREE_BASIS[1:]]
        return {
            "pair_correlations": tuple(moments[:3]),
            "scalar_chirality": moments[3],
        }
    raise ValueError("rdm must describe exactly two or three spin-half sites")


def pure_state_partial_trace(
    state: np.ndarray, n_sites: int, retained_sites: Sequence[int]
) -> np.ndarray:
    """Trace a computational-basis pure state down to ordered retained sites."""

    vector = np.asarray(state, dtype=np.complex128)
    retained = tuple(int(site) for site in retained_sites)
    if vector.shape != (2**int(n_sites),):
        raise ValueError("state length must be 2**n_sites")
    if len(set(retained)) != len(retained) or any(
        not 0 <= site < int(n_sites) for site in retained
    ):
        raise ValueError("retained_sites must be distinct valid site indices")
    traced = tuple(site for site in range(int(n_sites)) if site not in retained)
    tensor = vector.reshape((2,) * int(n_sites)).transpose(retained + traced)
    matrix = tensor.reshape(2 ** len(retained), -1)
    return np.asarray(matrix @ matrix.conj().T, dtype=np.complex128)


def pure_state_transition_partial_trace(
    left_state: np.ndarray,
    right_state: np.ndarray,
    n_sites: int,
    retained_sites: Sequence[int],
) -> np.ndarray:
    """Return ``Tr_env |left><right|`` in the computational basis."""

    left = np.asarray(left_state, dtype=np.complex128)
    right = np.asarray(right_state, dtype=np.complex128)
    retained = tuple(int(site) for site in retained_sites)
    if left.shape != (2**int(n_sites),) or right.shape != left.shape:
        raise ValueError("both states must have length 2**n_sites")
    if len(set(retained)) != len(retained) or any(
        not 0 <= site < int(n_sites) for site in retained
    ):
        raise ValueError("retained_sites must be distinct valid site indices")
    traced = tuple(site for site in range(int(n_sites)) if site not in retained)
    order = retained + traced
    left_matrix = left.reshape((2,) * int(n_sites)).transpose(order).reshape(
        2 ** len(retained), -1
    )
    right_matrix = right.reshape((2,) * int(n_sites)).transpose(order).reshape(
        2 ** len(retained), -1
    )
    return np.asarray(left_matrix @ right_matrix.conj().T, dtype=np.complex128)


def trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return ``0.5*||first-second||_1`` for Hermitian local states."""

    left = np.asarray(first, dtype=np.complex128)
    right = np.asarray(second, dtype=np.complex128)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("local states must be square matrices of equal shape")
    difference = 0.5 * ((left - right) + (left - right).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def transition_trace_norm(transition: np.ndarray) -> float:
    """Return ``||Tr_env |left><right|||_1`` from its singular values."""

    value = np.asarray(transition, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("transition must be a square matrix")
    return float(np.sum(np.linalg.svd(value, compute_uv=False)))


def _validate_two_root_local_blocks(
    rho00: np.ndarray, rho11: np.ndarray, rho01: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate local blocks for a two-root subspace.

    The convention is ``rho_ab = Tr_env |psi_a><psi_b|``.  The missing
    ``rho10`` block is therefore fixed as ``rho01.conj().T``.
    """

    blocks = tuple(
        np.asarray(value, dtype=np.complex128)
        for value in (rho00, rho11, rho01)
    )
    shape = blocks[0].shape
    if (
        len(shape) != 2
        or shape[0] != shape[1]
        or any(value.shape != shape for value in blocks[1:])
    ):
        raise ValueError("two-root local blocks must be equal-size square matrices")
    if not all(np.all(np.isfinite(value)) for value in blocks):
        raise ValueError("two-root local blocks must be finite")
    if not np.allclose(blocks[0], blocks[0].conj().T, atol=1.0e-11) or not np.allclose(
        blocks[1], blocks[1].conj().T, atol=1.0e-11
    ):
        raise ValueError("rho00 and rho11 must be Hermitian")
    return blocks


def local_code_decoupling_defect(
    rho00: np.ndarray, rho11: np.ndarray, rho01: np.ndarray
) -> float:
    """Return a basis-invariant two-root local-code defect in ``[0, 1]``.

    Define ``J_A=sum_ab |a><b|_R tensor rho_ab`` and
    ``rho_bar=(rho00+rho11)/2``.  This function returns
    ``0.25*||J_A-I_R tensor rho_bar||_1``.  Equivalently, it is the trace
    distance between the normalized Choi state ``J_A/2`` and
    ``(I_R/2) tensor rho_bar``.  It vanishes exactly when every local
    operator acts as a scalar on the two-root subspace.

    This is a maximally-entangled-reference decoupling diagnostic, not a
    diamond norm and not by itself a recovery-error bound.  The roots must be
    orthonormal for the normalized-distance interpretation.
    """

    first, second, transition = _validate_two_root_local_blocks(
        rho00, rho11, rho01
    )
    average = 0.5 * (first + second)
    difference = np.block(
        [
            [first - average, transition],
            [transition.conj().T, second - average],
        ]
    )
    difference = 0.5 * (difference + difference.conj().T)
    return float(0.25 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def local_code_qddm_singular_values(
    rho00: np.ndarray, rho11: np.ndarray, rho01: np.ndarray
) -> np.ndarray:
    """Return the three Hilbert--Schmidt QDDM singular values.

    The input roots span a two-dimensional subspace.  The logical Pauli
    operators ``(X,Y,Z)/sqrt(2)`` form a Hilbert--Schmidt orthonormal basis of
    its traceless operator space.  Their images under
    ``Phi(|a><b|)=rho_ab`` are vectorized as the three columns of the QDDM.
    The resulting singular spectrum is invariant under arbitrary U(2)
    rotations of the two roots and vanishes exactly at the local
    Knill--Laflamme condition.  The values are Hilbert--Schmidt response
    scales rather than trace-distance probabilities.  For physical blocks
    obtained from an orthonormal two-root code, each singular value lies in
    ``[0, 1]``; noisy external estimates should audit that bound rather than
    silently clip it.
    """

    first, second, transition = _validate_two_root_local_blocks(
        rho00, rho11, rho01
    )
    reverse = transition.conj().T
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    images = (
        (transition + reverse) * inv_sqrt_two,
        (-1.0j * transition + 1.0j * reverse) * inv_sqrt_two,
        (first - second) * inv_sqrt_two,
    )
    qddm = np.column_stack([value.reshape(-1) for value in images])
    return np.asarray(np.linalg.svd(qddm, compute_uv=False), dtype=np.float64)


def two_spin_trace_distance_from_correlators(first: float, second: float) -> float:
    """Return the exact invariant two-spin distance ``|c_first-c_second|``."""

    values = (float(first), float(second))
    if not all(np.isfinite(value) for value in values):
        raise ValueError("pair correlators must be finite")
    return abs(values[0] - values[1])


def three_spin_trace_distance_from_moments(
    first_pairs: Sequence[float],
    first_chirality: float,
    second_pairs: Sequence[float],
    second_chirality: float,
) -> float:
    """Return the exact invariant three-spin distance from four scalars.

    With ``delta_c`` the three pair-correlation differences,
    ``delta_C=sum(delta_c)`` and ``delta_kappa`` the chirality difference,
    define ``A=3*sum(delta_c**2)-delta_C**2+6*delta_kappa**2``.  Then
    ``D3=(|delta_C|+max(|delta_C|,sqrt(2*A)))/3``.  No 8x8 density matrix is
    needed, which makes this the production-facing estimator.
    """

    left = tuple(float(value) for value in first_pairs)
    right = tuple(float(value) for value in second_pairs)
    chiralities = (float(first_chirality), float(second_chirality))
    if len(left) != 3 or len(right) != 3 or not all(
        np.isfinite(value) for value in left + right + chiralities
    ):
        raise ValueError("two sets of three finite pairs and two chiralities are required")
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    delta_total = float(np.sum(difference))
    delta_chirality = chiralities[0] - chiralities[1]
    radicand = float(
        3.0 * np.dot(difference, difference)
        - delta_total**2
        + 6.0 * delta_chirality**2
    )
    root = math.sqrt(max(0.0, 2.0 * radicand))
    return (abs(delta_total) + max(abs(delta_total), root)) / 3.0


__all__ = [
    "local_code_decoupling_defect",
    "local_code_qddm_singular_values",
    "local_invariant_moments",
    "pure_state_partial_trace",
    "pure_state_transition_partial_trace",
    "three_spin_su2_rdm",
    "three_spin_su2_transition",
    "three_spin_trace_distance_from_moments",
    "trace_distance",
    "transition_trace_norm",
    "two_spin_su2_rdm",
    "two_spin_su2_transition",
    "two_spin_trace_distance_from_correlators",
]
