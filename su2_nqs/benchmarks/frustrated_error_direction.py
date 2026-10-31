"""Exact-small utilities for symmetry-resolved observable-error benchmarks.

The functions in this module are deliberately model agnostic.  They provide
an exact full-computational-basis Heisenberg action, natural observable scales,
and target-fidelity trajectory interpolation.  The corresponding benchmark
can therefore compare fusion, fixed-``Sz``, and unrestricted neural states
without changing the physical observables or materializing a ``2**N`` dense
Hamiltonian.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .triangular_locality import physical_dimer_product


WeightedBond = tuple[int, int, float]


@dataclass(frozen=True)
class FullGraphAction:
    """Precomputed full-Hilbert action of a Heisenberg bond graph.

    ``diagonal`` has length ``2**n``.  ``swapped_indices`` and
    ``offdiagonal_weights`` have shape ``(number_of_bonds, 2**n)``.  The
    computational integer order uses site zero as the most significant bit.
    """

    n: int
    diagonal: np.ndarray
    swapped_indices: np.ndarray
    offdiagonal_weights: np.ndarray


def four_spin_multiplicity_operators() -> dict[str, np.ndarray]:
    """Return a fixed real basis for the four-spin singlet multiplicity.

    The ordered sites are ``(a,b,c,d)``.  ``horizontal`` is the product of
    singlets ``(a,b)(c,d)`` and ``vertical`` is ``(a,c)(b,d)``.  They overlap
    by ``1/2`` in the repository's Condon--Shortley convention.  ``x`` mixes
    ``horizontal`` with the normalized direction in ``vertical`` orthogonal
    to it, while ``z`` resolves those two orthonormal multiplicity channels.
    Both operators vanish outside the local total-singlet subspace and are
    therefore SU(2) invariant four-spin operators.
    """

    horizontal = physical_dimer_product(4, ((0, 1), (2, 3)))
    vertical = physical_dimer_product(4, ((0, 2), (1, 3)))
    overlap = complex(np.vdot(horizontal, vertical))
    if abs(overlap.imag) > 1.0e-14 or abs(overlap.real - 0.5) > 1.0e-14:
        raise RuntimeError("four-spin dimer overlap convention changed")
    first = horizontal
    second = vertical - overlap * horizontal
    second /= np.linalg.norm(second)
    basis = np.column_stack((first, second))
    projector = basis @ basis.conj().T
    x_matrix = basis @ np.asarray([[0.0, 1.0], [1.0, 0.0]]) @ basis.conj().T
    z_matrix = basis @ np.asarray([[1.0, 0.0], [0.0, -1.0]]) @ basis.conj().T
    return {
        "projector": np.asarray(projector, dtype=np.complex128),
        "x": np.asarray(x_matrix, dtype=np.complex128),
        "z": np.asarray(z_matrix, dtype=np.complex128),
        "horizontal": np.asarray(horizontal, dtype=np.complex128),
        "vertical": np.asarray(vertical, dtype=np.complex128),
        "orthogonal_vertical": np.asarray(second, dtype=np.complex128),
    }


def rectangular_four_site_cells(
    rows: int, columns: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Return ordered 2x2 cells ``(top-left, top-right, bottom-left, bottom-right)``."""

    if isinstance(rows, bool) or not isinstance(rows, Integral):
        raise TypeError("rows must be an integer")
    if isinstance(columns, bool) or not isinstance(columns, Integral):
        raise TypeError("columns must be an integer")
    rows, columns = int(rows), int(columns)
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    return tuple(
        (
            row * columns + column,
            row * columns + column + 1,
            (row + 1) * columns + column,
            (row + 1) * columns + column + 1,
        )
        for row in range(rows - 1)
        for column in range(columns - 1)
    )


def apply_local_operator_full(
    state: np.ndarray,
    operator: np.ndarray,
    sites: Sequence[int],
) -> np.ndarray:
    """Apply one ordered four-site operator to a full computational vector."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError("state must be a one-dimensional full-Hilbert vector")
    n = int(round(np.log2(vector.size)))
    if 1 << n != vector.size:
        raise ValueError("state length must be a power of two")
    local = np.asarray(operator, dtype=np.complex128)
    if local.shape != (16, 16):
        raise ValueError("operator must be a 16x16 four-spin matrix")
    ordered = tuple(int(site) for site in sites)
    if len(ordered) != 4 or len(set(ordered)) != 4:
        raise ValueError("sites must contain four distinct entries")
    if any(not 0 <= site < n for site in ordered):
        raise ValueError("a local-operator site lies outside the state")
    remainder = tuple(site for site in range(n) if site not in set(ordered))
    permutation = ordered + remainder
    inverse = np.argsort(permutation)
    tensor = vector.reshape((2,) * n).transpose(permutation)
    tensor = tensor.reshape(16, -1)
    changed = (local @ tensor).reshape((2,) * n)
    return changed.transpose(tuple(int(value) for value in inverse)).reshape(-1)


def summed_local_operator_action(
    state: np.ndarray,
    operator: np.ndarray,
    cells: Sequence[Sequence[int]],
    *,
    weights: Sequence[Real] | None = None,
) -> np.ndarray:
    """Apply a weighted sum of identically oriented four-site operators."""

    checked = tuple(tuple(int(site) for site in cell) for cell in cells)
    if not checked:
        raise ValueError("cells must be nonempty")
    if weights is None:
        coefficients = np.ones(len(checked), dtype=np.float64)
    else:
        coefficients = np.asarray(tuple(float(value) for value in weights))
        if coefficients.shape != (len(checked),) or np.any(~np.isfinite(coefficients)):
            raise ValueError("weights must be finite and match the cells")
    result = np.zeros_like(np.asarray(state, dtype=np.complex128))
    for coefficient, cell in zip(coefficients, checked, strict=True):
        result += coefficient * apply_local_operator_full(state, operator, cell)
    return result


def projected_summed_local_operator(
    singlet_isometry: np.ndarray,
    operator: np.ndarray,
    cells: Sequence[Sequence[int]],
    *,
    weights: Sequence[Real] | None = None,
) -> tuple[np.ndarray, float]:
    """Project a summed local invariant into a complete global singlet basis.

    The returned residual is the largest norm of the action outside the
    supplied isometry.  It is an independent SU(2)-closure audit.
    """

    isometry = np.asarray(singlet_isometry, dtype=np.complex128)
    if isometry.ndim != 2 or isometry.shape[0] < isometry.shape[1]:
        raise ValueError("singlet_isometry must be a tall matrix")
    identity_error = float(
        np.max(
            np.abs(isometry.conj().T @ isometry - np.eye(isometry.shape[1])),
            initial=0.0,
        )
    )
    if identity_error > 1.0e-10:
        raise ValueError("singlet_isometry columns are not orthonormal")
    images = np.column_stack(
        [
            summed_local_operator_action(
                isometry[:, column], operator, cells, weights=weights
            )
            for column in range(isometry.shape[1])
        ]
    )
    projected = isometry.conj().T @ images
    residual = float(np.max(np.linalg.norm(images - isometry @ projected, axis=0)))
    return projected, residual


def full_computational_configurations(n: int) -> np.ndarray:
    """Return all binary spin configurations in computational integer order."""

    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be an integer")
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    integers = np.arange(1 << n, dtype=np.uint64)
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((integers[:, None] >> shifts[None, :]) & 1).astype(np.int32)


def full_heisenberg_graph_action(
    n: int,
    bonds: Sequence[tuple[int, int] | WeightedBond],
) -> FullGraphAction:
    """Compile ``sum_ij J_ij S_i.S_j`` without a dense full-space matrix."""

    configurations = full_computational_configurations(n)
    dimension = len(configurations)
    integers = np.arange(dimension, dtype=np.int64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    swaps: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        if isinstance(first, bool) or not isinstance(first, Integral):
            raise TypeError("bond sites must be integers")
        if isinstance(second, bool) or not isinstance(second, Integral):
            raise TypeError("bond sites must be integers")
        first, second = int(first), int(second)
        value = float(coupling)
        if not np.isfinite(value):
            raise ValueError("bond couplings must be finite")
        if first > second:
            first, second = second, first
        if not 0 <= first < second < n:
            raise ValueError("bond sites must satisfy 0 <= i < j < n")
        different = configurations[:, first] != configurations[:, second]
        diagonal += value * np.where(different, -0.25, 0.25)
        first_mask = 1 << (n - first - 1)
        second_mask = 1 << (n - second - 1)
        swaps.append(integers ^ (first_mask | second_mask))
        weights.append(0.5 * value * different.astype(np.float64))
    return FullGraphAction(
        n=n,
        diagonal=diagonal,
        swapped_indices=(
            np.stack(swaps, axis=0)
            if swaps
            else np.empty((0, dimension), dtype=np.int64)
        ),
        offdiagonal_weights=(
            np.stack(weights, axis=0)
            if weights
            else np.empty((0, dimension), dtype=np.float64)
        ),
    )


def apply_full_heisenberg_graph(
    action: FullGraphAction,
    state: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a compiled graph action to one full-Hilbert state vector."""

    vector = jnp.asarray(state)
    expected = (1 << action.n,)
    if vector.shape != expected:
        raise ValueError(f"expected state shape {expected}, got {vector.shape}")
    diagonal = jnp.asarray(action.diagonal)
    swapped = jnp.asarray(action.swapped_indices)
    weights = jnp.asarray(action.offdiagonal_weights)
    return diagonal * vector + jnp.sum(weights * vector[swapped], axis=0)


def _spectral_width(operator: object) -> float:
    dense = np.asarray(
        operator.toarray() if hasattr(operator, "toarray") else operator,
        dtype=np.complex128,
    )
    defect = float(np.max(np.abs(dense - dense.conj().T), initial=0.0))
    if defect > 1.0e-10:
        raise ValueError(f"observable is not Hermitian: defect={defect:.3e}")
    values = np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))
    width = float(values[-1] - values[0])
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("observable must have positive finite spectral width")
    return width


def natural_observable_scales(
    lattice: str,
    operators: Mapping[str, object],
    *,
    n_sites: int,
    triangular_plaquette_count: int | None = None,
) -> dict[str, float]:
    """Return fixed spectral-width scales for the physical error vector.

    Linear structure factors, including direction-averaged stripe operators,
    use the spectral width of the *composite operator actually measured*.
    Connected second moments use a sum of the sharp single-operator range
    bounds ``width**2 / 4`` together with the normalization used by
    ``lattice_observable_panel``.
    """

    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    widths = {name: _spectral_width(operator) for name, operator in operators.items()}
    if lattice == "square":
        required = {
            "neel",
            "stripe_x",
            "stripe_y",
            "b1g",
            "columnar_x",
            "columnar_y",
        }
        if not required <= set(widths):
            raise ValueError("square observable panel is incomplete")
        scales = {
            "neel_m2": widths["neel"],
            "stripe_m2_mean": _spectral_width(
                0.5 * (operators["stripe_x"] + operators["stripe_y"])
            ),
            "b1g_mean": widths["b1g"] / n_sites,
            "b1g_connected": widths["b1g"] ** 2 / (4.0 * n_sites**2),
            "columnar_dimer_connected": (
                widths["columnar_x"] ** 2 + widths["columnar_y"] ** 2
            )
            / (4.0 * n_sites**2),
        }
    elif lattice == "triangular":
        required = {
            "k",
            "m1",
            "m2",
            "m3",
            "nematic_1",
            "nematic_2",
            "chirality",
        }
        if not required <= set(widths):
            raise ValueError("triangular observable panel is incomplete")
        if triangular_plaquette_count is None or triangular_plaquette_count < 1:
            raise ValueError("triangular_plaquette_count must be positive")
        scales = {
            "k_m2": widths["k"],
            "stripe_m2_mean": _spectral_width(
                (operators["m1"] + operators["m2"] + operators["m3"])
                / 3.0
            ),
            "bond_nematic_connected": (
                widths["nematic_1"] ** 2 + widths["nematic_2"] ** 2
            )
            / (4.0 * n_sites**2),
            "chirality_mean": widths["chirality"] / triangular_plaquette_count,
            "chirality_connected": widths["chirality"] ** 2
            / (4.0 * triangular_plaquette_count**2),
        }
    else:
        raise ValueError("lattice must be 'square' or 'triangular'")
    if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise RuntimeError("derived observable scale is not positive and finite")
    return scales


def normalized_observable_error(
    panel: Mapping[str, Real],
    reference: Mapping[str, Real],
    scales: Mapping[str, Real],
) -> dict[str, object]:
    """Return the signed dimensionless error vector and its RMS norm."""

    names = tuple(scales)
    if not names:
        raise ValueError("at least one observable scale is required")
    values = np.asarray(
        [
            (float(panel[name]) - float(reference[name])) / float(scales[name])
            for name in names
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("observable errors must be finite")
    return {
        "names": names,
        "signed": values,
        "rms": float(np.linalg.norm(values) / np.sqrt(len(values))),
        "maximum_absolute": float(np.max(np.abs(values))),
    }


def normalized_response_direction(
    lower: Mapping[str, Real],
    upper: Mapping[str, Real],
    scales: Mapping[str, Real],
    *,
    coupling_step: float,
) -> np.ndarray:
    """Return a unit vector parallel to the exact finite-difference response."""

    delta = float(coupling_step)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("coupling_step must be positive and finite")
    vector = np.asarray(
        [
            (float(upper[name]) - float(lower[name]))
            / (2.0 * delta * float(scales[name]))
            for name in scales
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("the normalized response direction is numerically zero")
    return vector / norm


def response_parallel_error(
    signed_error: Sequence[Real], response_direction: Sequence[Real]
) -> float:
    """Return the RMS-normalized error component along a unit response vector."""

    error = np.asarray(signed_error, dtype=np.float64)
    response = np.asarray(response_direction, dtype=np.float64)
    if error.ndim != 1 or response.shape != error.shape or error.size == 0:
        raise ValueError("error and response must be equal nonempty vectors")
    norm = float(np.linalg.norm(response))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1.0e-10:
        raise ValueError("response_direction must have unit norm")
    return float(abs(np.dot(error, response)) / np.sqrt(error.size))


def first_crossing_interpolation(
    trajectory: Sequence[Mapping[str, Real]],
    target_fidelity: float,
    *,
    fidelity_key: str = "conditional_singlet_fidelity",
    value_keys: Sequence[str] = (
        "observable_error_rms",
        "response_parallel_error",
        "elapsed_seconds",
    ),
) -> dict[str, float] | None:
    """Interpolate scalar diagnostics at the first upward fidelity crossing.

    The operation never selects a checkpoint using an observable.  If the
    initial point already exceeds the target, that point is returned.  A
    downward crossing is ignored.
    """

    target = float(target_fidelity)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target_fidelity must lie in [0, 1]")
    rows = tuple(trajectory)
    if not rows:
        raise ValueError("trajectory must be nonempty")
    first = float(rows[0][fidelity_key])
    if first >= target:
        return {
            "target_fidelity": target,
            "lower_step": float(rows[0]["step"]),
            "upper_step": float(rows[0]["step"]),
            "interpolation_fraction": 0.0,
            **{key: float(rows[0][key]) for key in value_keys},
        }
    for lower, upper in zip(rows, rows[1:], strict=False):
        low = float(lower[fidelity_key])
        high = float(upper[fidelity_key])
        if low < target <= high and high > low:
            fraction = (target - low) / (high - low)
            return {
                "target_fidelity": target,
                "lower_step": float(lower["step"]),
                "upper_step": float(upper["step"]),
                "interpolation_fraction": float(fraction),
                **{
                    key: float(lower[key])
                    + fraction * (float(upper[key]) - float(lower[key]))
                    for key in value_keys
                },
            }
    return None


__all__ = [
    "FullGraphAction",
    "apply_local_operator_full",
    "apply_full_heisenberg_graph",
    "first_crossing_interpolation",
    "four_spin_multiplicity_operators",
    "full_computational_configurations",
    "full_heisenberg_graph_action",
    "natural_observable_scales",
    "normalized_observable_error",
    "normalized_response_direction",
    "projected_summed_local_operator",
    "rectangular_four_site_cells",
    "response_parallel_error",
    "summed_local_operator_action",
]
