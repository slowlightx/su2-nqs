"""SU(2)-invariant full counting statistics of collective block spins.

The primary observable is the probability distribution of doubled total-spin
labels carried by one or more physical site groups.  Two independent exact
representations are provided:

* computational-basis subsystem spin projectors, valid for arbitrary groups;
* internal labels of a fusion tree that first fuses each group.

Their agreement makes the group labels physical observables rather than an
artifact of one autoregressive ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import product

import numpy as np

from ..su2.fusion_tree import FusionNode, FusionTree, FusionTreeState, Leaf, TreeNode
from .exact_diagonalization import spin_squared_matrix

FCSKey = tuple[int, ...]


def _validated_groups(
    n_sites: int,
    groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    checked = tuple(tuple(int(site) for site in group) for group in groups)
    if not checked or any(not group for group in checked):
        raise ValueError("groups must be nonempty")
    flat = tuple(site for group in checked for site in group)
    if len(set(flat)) != len(flat):
        raise ValueError("measured groups must be disjoint")
    if any(not 0 <= site < n_sites for site in flat):
        raise ValueError("a group contains a site outside the system")
    return checked


@lru_cache(maxsize=None)
def subsystem_spin_eigenbasis(n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Return doubled-spin labels and eigenvectors of subsystem ``S^2``."""

    spin_squared, _ = spin_squared_matrix(n_sites)
    values, vectors = np.linalg.eigh(spin_squared)
    raw_q = np.sqrt(np.maximum(1.0 + 4.0 * values, 0.0)) - 1.0
    labels = np.rint(raw_q).astype(np.int32)
    expected = labels * (labels + 2) / 4.0
    if np.max(np.abs(values - expected)) > 1.0e-10:
        raise RuntimeError("subsystem S^2 eigenvalues do not match doubled spins")
    if np.max(np.abs(vectors.conj().T @ vectors - np.eye(2**n_sites))) > 1.0e-12:
        raise RuntimeError("subsystem S^2 eigenvectors are not orthonormal")
    return labels, vectors


def _rotate_groups(
    vector: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    n_sites = int(round(np.log2(np.asarray(vector).size)))
    if 2**n_sites != np.asarray(vector).size:
        raise ValueError("wavefunction length must be a power of two")
    measured = {site for group in groups for site in group}
    remainder = tuple(site for site in range(n_sites) if site not in measured)
    order = tuple(site for group in groups for site in group) + remainder
    shape = tuple(2 ** len(group) for group in groups)
    if remainder:
        shape += (2 ** len(remainder),)
    tensor = np.asarray(vector, dtype=np.complex128).reshape((2,) * n_sites)
    tensor = np.transpose(tensor, order).reshape(shape)
    labels = []
    for axis, group in enumerate(groups):
        q_labels, eigenvectors = subsystem_spin_eigenbasis(len(group))
        tensor = np.tensordot(eigenvectors.conj().T, tensor, axes=(1, axis))
        tensor = np.moveaxis(tensor, 0, axis)
        labels.append(q_labels)
    return tensor, tuple(labels)


def block_spin_fcs_with_tangent(
    wavefunction: np.ndarray,
    tangent: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> tuple[dict[FCSKey, float], dict[FCSKey, float]]:
    """Return ``p(q_groups)`` and its directional derivative.

    ``wavefunction`` and ``tangent`` use the full computational basis.  The
    wavefunction must have unit norm and the tangent must be a finite
    normalized-state derivative, ``2 Re <psi|dpsi> = 0``.
    """

    wavefunction = np.asarray(wavefunction, dtype=np.complex128)
    tangent = np.asarray(tangent, dtype=np.complex128)
    if wavefunction.shape != tangent.shape or wavefunction.ndim != 1:
        raise ValueError("wavefunction and tangent must be equal one-dimensional arrays")
    n_sites = int(round(np.log2(wavefunction.size)))
    checked = _validated_groups(n_sites, groups)
    norm = float(np.real(np.vdot(wavefunction, wavefunction)))
    if abs(norm - 1.0) > 1.0e-10:
        raise ValueError("wavefunction must be normalized")
    if np.any(~np.isfinite(wavefunction)) or np.any(~np.isfinite(tangent)):
        raise ValueError("wavefunction and tangent must be finite")
    norm_derivative = float(2.0 * np.real(np.vdot(wavefunction, tangent)))
    if abs(norm_derivative) > 1.0e-9:
        raise ValueError("tangent must preserve wavefunction normalization")
    rotated, label_arrays = _rotate_groups(wavefunction, checked)
    rotated_tangent, tangent_labels = _rotate_groups(tangent, checked)
    if any(not np.array_equal(a, b) for a, b in zip(label_arrays, tangent_labels, strict=True)):
        raise RuntimeError("inconsistent subsystem spin labels")
    probability_array = np.abs(rotated) ** 2
    derivative_array = 2.0 * np.real(np.conj(rotated) * rotated_tangent)
    if probability_array.ndim > len(checked):
        axes = tuple(range(len(checked), probability_array.ndim))
        probability_array = np.sum(probability_array, axis=axes)
        derivative_array = np.sum(derivative_array, axis=axes)

    unique_labels = tuple(tuple(int(q) for q in np.unique(values)) for values in label_arrays)
    probabilities: dict[FCSKey, float] = {}
    derivatives: dict[FCSKey, float] = {}
    for key in product(*unique_labels):
        indices = tuple(np.flatnonzero(values == q) for values, q in zip(label_arrays, key, strict=True))
        selector = np.ix_(*indices)
        probabilities[key] = float(np.sum(probability_array[selector]))
        derivatives[key] = float(np.sum(derivative_array[selector]))
    return probabilities, derivatives


def block_spin_fcs(
    wavefunction: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> dict[FCSKey, float]:
    """Return collective-spin probabilities for arbitrary disjoint groups."""

    tangent = np.zeros_like(np.asarray(wavefunction, dtype=np.complex128))
    probabilities, _ = block_spin_fcs_with_tangent(wavefunction, tangent, groups)
    return probabilities


def classical_fisher_information(
    probabilities: Mapping[FCSKey, float],
    derivatives: Mapping[FCSKey, float],
    *,
    probability_floor: float = 1.0e-14,
) -> tuple[float, float]:
    """Return classical Fisher information and probability mass below floor."""

    if probability_floor < 0.0:
        raise ValueError("probability_floor must be non-negative")
    if set(probabilities) != set(derivatives):
        raise ValueError("probability and derivative outcomes must match")
    information = 0.0
    dropped_mass = 0.0
    for key, probability in probabilities.items():
        derivative = float(derivatives[key])
        if probability > probability_floor:
            information += derivative * derivative / probability
        else:
            dropped_mass += max(float(probability), 0.0)
    return float(information), float(dropped_mass)


def moment_fisher_lower_bound(
    probabilities: Mapping[FCSKey, float],
    derivatives: Mapping[FCSKey, float],
) -> float:
    """Information captured by the first moments of block Casimirs.

    This is ``dmu.T @ Cov(features)^+ @ dmu`` for features
    ``S_g(S_g+1)``.  It is a lower bound on the full-distribution classical
    Fisher information and quantifies what is lost by plotting only a
    structure-factor-like mean.
    """

    diagnostics = moment_fisher_diagnostics(probabilities, derivatives)
    return diagnostics["information"]


def moment_fisher_diagnostics(
    probabilities: Mapping[FCSKey, float],
    derivatives: Mapping[FCSKey, float],
) -> dict[str, float | int]:
    """Return the Casimir-moment information and covariance rank."""

    keys = tuple(sorted(probabilities))
    if not keys or set(keys) != set(derivatives):
        raise ValueError("probability and derivative outcomes must be nonempty and match")
    p = np.asarray([probabilities[key] for key in keys], dtype=np.float64)
    dp = np.asarray([derivatives[key] for key in keys], dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(dp)):
        raise ValueError("probabilities and derivatives must be finite")
    if np.min(p) < -1.0e-12 or abs(float(np.sum(p)) - 1.0) > 1.0e-10:
        raise ValueError("probabilities must be normalized and non-negative")
    if abs(float(np.sum(dp))) > 1.0e-9:
        raise ValueError("probability derivatives must sum to zero")
    features = np.asarray(
        [[q * (q + 2) / 4.0 for q in key] for key in keys],
        dtype=np.float64,
    )
    mean = p @ features
    derivative_mean = dp @ features
    centered = features - mean
    covariance = (centered.T * p) @ centered
    information = float(
        derivative_mean
        @ np.linalg.pinv(covariance, rcond=1.0e-12)
        @ derivative_mean
    )
    return {
        "information": information,
        "covariance_rank": int(np.linalg.matrix_rank(covariance, tol=1.0e-12)),
    }


def fcs_score_projection_diagnostics(
    probabilities: Mapping[FCSKey, float],
    derivatives: Mapping[FCSKey, float],
    *,
    probability_floor: float = 1.0e-14,
    pseudoinverse_rcond: float = 1.0e-12,
) -> dict[str, object]:
    """Resolve FCS Fisher information into Casimir and shape responses.

    The exact score ``d log p / dg`` is orthogonally projected, with the
    probability distribution as metric, onto the centered block-Casimir
    features ``S_A(S_A+1)``.  The projected norm is the information available
    from the Casimir means.  The residual norm is therefore the response of
    the *shape* of the multiplet distribution that a structure-factor-like
    first moment cannot retain.

    Outcomes at or below ``probability_floor`` are excluded from the score,
    consistently with :func:`classical_fisher_information`.  Their total mass
    and the resulting Pythagorean closure error are reported explicitly.
    Doubled spins remain the only Hilbert labels accepted by this API.
    """

    if probability_floor < 0.0:
        raise ValueError("probability_floor must be non-negative")
    if pseudoinverse_rcond <= 0.0:
        raise ValueError("pseudoinverse_rcond must be positive")
    keys = tuple(sorted(probabilities))
    if not keys or set(keys) != set(derivatives):
        raise ValueError("probability and derivative outcomes must be nonempty and match")
    p = np.asarray([probabilities[key] for key in keys], dtype=np.float64)
    dp = np.asarray([derivatives[key] for key in keys], dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(dp)):
        raise ValueError("probabilities and derivatives must be finite")
    if np.min(p) < -1.0e-12 or abs(float(np.sum(p)) - 1.0) > 1.0e-10:
        raise ValueError("probabilities must be normalized and non-negative")
    if abs(float(np.sum(dp))) > 1.0e-9:
        raise ValueError("probability derivatives must sum to zero")

    features = np.asarray(
        [[q * (q + 2) / 4.0 for q in key] for key in keys],
        dtype=np.float64,
    )
    mean = p @ features
    derivative_mean = dp @ features
    centered = features - mean
    covariance = (centered.T * p) @ centered
    coefficients = (
        np.linalg.pinv(covariance, rcond=pseudoinverse_rcond) @ derivative_mean
    )
    projected_score = centered @ coefficients
    casimir_information = float(derivative_mean @ coefficients)

    active = p > probability_floor
    score = np.zeros_like(p)
    score[active] = dp[active] / p[active]
    residual_contributions = np.zeros_like(p)
    residual_contributions[active] = (
        p[active] * (score[active] - projected_score[active]) ** 2
    )
    full_information = float(np.sum(p[active] * score[active] ** 2))
    residual_information = float(np.sum(residual_contributions))
    cross_term = float(
        np.sum(
            p[active]
            * projected_score[active]
            * (score[active] - projected_score[active])
        )
    )
    closure_error = abs(
        full_information - casimir_information - residual_information
    )
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    retained_cutoff = pseudoinverse_rcond * float(
        np.max(np.abs(covariance_eigenvalues))
    )
    retained = covariance_eigenvalues[covariance_eigenvalues > retained_cutoff]
    return {
        "full_information": full_information,
        "casimir_information": casimir_information,
        "shape_information": residual_information,
        "pythagorean_closure_error": float(closure_error),
        "score_projection_cross_term": cross_term,
        "dropped_probability_mass": float(np.sum(np.maximum(p[~active], 0.0))),
        "covariance_rank": int(np.linalg.matrix_rank(covariance, tol=1.0e-12)),
        "pseudoinverse_rcond": float(pseudoinverse_rcond),
        "minimum_retained_covariance_eigenvalue": (
            float(np.min(retained)) if retained.size else 0.0
        ),
        "casimir_coefficients": [float(value) for value in coefficients],
        "outcome_score": {
            key: float(value) for key, value in zip(keys, score, strict=True)
        },
        "outcome_projected_score": {
            key: float(value)
            for key, value in zip(keys, projected_score, strict=True)
        },
        "outcome_shape_information": {
            key: float(value)
            for key, value in zip(keys, residual_contributions, strict=True)
        },
    }


def fcs_summary(
    probabilities: Mapping[FCSKey, float],
    derivatives: Mapping[FCSKey, float],
    *,
    quantum_fisher_information: float,
    probability_floor: float = 1.0e-14,
) -> dict[str, float]:
    """Summarize normalization, moments, and Fisher-information fractions."""

    normalization = float(sum(probabilities.values()))
    minimum_probability = float(min(probabilities.values()))
    derivative_sum = float(sum(derivatives.values()))
    if abs(normalization - 1.0) > 1.0e-10:
        raise ValueError("FCS probabilities must be normalized")
    if minimum_probability < -1.0e-12:
        raise ValueError("FCS probabilities must be non-negative")
    if abs(derivative_sum) > 1.0e-9:
        raise ValueError("FCS probability derivatives must sum to zero")
    classical, dropped = classical_fisher_information(
        probabilities, derivatives, probability_floor=probability_floor
    )
    moment_diagnostics = moment_fisher_diagnostics(probabilities, derivatives)
    moment = float(moment_diagnostics["information"])
    if moment > classical + max(1.0e-10, 1.0e-10 * classical):
        raise ValueError("Casimir-moment information exceeds full FCS information")
    if classical > quantum_fisher_information + max(
        1.0e-10, 1.0e-10 * quantum_fisher_information
    ):
        raise ValueError("classical Fisher information exceeds quantum Fisher information")
    entropy = -sum(
        probability * np.log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    if quantum_fisher_information <= 0.0:
        raise ValueError("quantum Fisher information must be positive")
    return {
        "normalization": normalization,
        "minimum_probability": minimum_probability,
        "derivative_sum": derivative_sum,
        "shannon_entropy": float(entropy),
        "classical_fisher_information": classical,
        "moment_fisher_lower_bound": moment,
        "moment_covariance_rank": int(moment_diagnostics["covariance_rank"]),
        "fisher_capture_fraction": classical / quantum_fisher_information,
        "moment_fisher_fraction_of_qfi": moment / quantum_fisher_information,
        "moment_fraction_of_fcs": moment / classical if classical > 0.0 else 0.0,
        "dropped_probability_mass": dropped,
    }


def _balanced_range(start: int, stop: int) -> TreeNode:
    if stop - start == 1:
        return Leaf(start)
    middle = (start + stop) // 2
    return FusionNode(_balanced_range(start, middle), _balanced_range(middle, stop))


def grouped_fusion_tree(
    group_sizes: Sequence[int],
    *,
    association: str = "left",
) -> tuple[FusionTree, tuple[int | None, ...]]:
    """Build a tree that fuses each contiguous physical group first.

    Returns the tree and postorder label indices of the group roots.  A
    singleton group has fixed doubled spin one and therefore index ``None``.
    """

    sizes = tuple(int(size) for size in group_sizes)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("group sizes must be positive")
    roots: list[TreeNode] = []
    start = 0
    for size in sizes:
        roots.append(_balanced_range(start, start + size))
        start += size
    if association == "left":
        root = roots[0]
        for child in roots[1:]:
            root = FusionNode(root, child)
    elif association == "right":
        root = roots[-1]
        for child in reversed(roots[:-1]):
            root = FusionNode(child, root)
    else:
        raise ValueError("association must be 'left' or 'right'")
    tree = FusionTree(root)
    indices = {node: index for index, node in enumerate(tree.internal_nodes)}
    return tree, tuple(None if isinstance(node, Leaf) else indices[node] for node in roots)


def tree_block_spin_fcs_with_tangent(
    amplitudes: np.ndarray,
    tangent: np.ndarray,
    states: Sequence[FusionTreeState],
    group_label_indices: Sequence[int | None],
) -> tuple[dict[FCSKey, float], dict[FCSKey, float]]:
    """Aggregate a grouped-tree state by its physical subtree-root labels."""

    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    tangent = np.asarray(tangent, dtype=np.complex128)
    states = tuple(states)
    indices = tuple(group_label_indices)
    if amplitudes.shape != tangent.shape or amplitudes.shape != (len(states),):
        raise ValueError("amplitudes, tangent, and tree-state count must match")
    probabilities: dict[FCSKey, float] = {}
    derivatives: dict[FCSKey, float] = {}
    for amplitude, derivative, state in zip(amplitudes, tangent, states, strict=True):
        key = tuple(1 if index is None else int(state.labels[index]) for index in indices)
        probabilities[key] = probabilities.get(key, 0.0) + float(abs(amplitude) ** 2)
        derivatives[key] = derivatives.get(key, 0.0) + float(
            2.0 * np.real(np.conj(amplitude) * derivative)
        )
    return dict(sorted(probabilities.items())), dict(sorted(derivatives.items()))


def doubled_spins_can_close_to_singlet(labels: Sequence[int]) -> bool:
    """Return whether two or three doubled spins can fuse to total zero."""

    values = tuple(int(value) for value in labels)
    if len(values) == 2:
        return values[0] == values[1]
    if len(values) == 3:
        first, second, third = values
        return (
            abs(first - second) <= third <= first + second
            and (first + second - third) % 2 == 0
        )
    raise ValueError("singlet closure is implemented for two or three groups")
