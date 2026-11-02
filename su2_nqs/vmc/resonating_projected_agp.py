"""Three-root algebra for a resonating bank of strict projected AGPs.

This module is an exact-sector *alternative carrier* and control.  It is not
part of the fusion-path NQS ansatz and must not be used as evidence for a
fusion-NQS advantage.

The nonlinear bank contains ``R`` independently projected AGPs.  Its lowest
three represented roots are obtained from one shared set of direct rows

``Phi[x, r] = psi_r(x)`` and ``B[x, r] = (H psi_r)(x)``.

All sampled moments use a positive external proposal and avoid both
``log(psi)`` and ``B / Phi``.  Exact nodes are therefore admissible.  Column
and quadrature rescalings are removed before products are formed, so the
generalized Ritz result is covariant under nonsingular changes of bank basis
and under the homogeneous scale gauge of each AGP.  Rescaling cannot rescue a
determinant that has already overflowed; production training must still fix
the norm of every pairing matrix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, NamedTuple, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ..models.projected_singlet_agp import ProjectedSingletAGPConfig
from .projected_agp import (
    ProjectedAGPHeisenbergConfig,
    projected_agp_heisenberg_action,
)
from .sampled_multistate import NumericalSubspaceRankError


class ResonatingProjectedAGPAction(NamedTuple):
    """Direct amplitudes and Hamiltonian actions for one carrier bank."""

    amplitudes: jax.Array
    hamiltonian_actions: jax.Array
    center_exact_nodes: jax.Array
    minimum_connected_cancellation_ratio: jax.Array


@dataclass(frozen=True)
class MatrixHermiticityAudit:
    """Roundoff or sampling departure from a Hermitian matrix."""

    maximum_absolute_error: float
    relative_frobenius_error: float


@dataclass(frozen=True)
class ThreeRootResonatingResult:
    """Fail-closed three-root solution in a sampled resonating-AGP span.

    ``coefficients`` are expressed in the original input-bank coordinates,
    while ``scaled_coefficients`` multiply the internally column-rescaled
    bank.  The three columns of ``root_amplitudes`` and ``root_actions`` have
    unit norm in the internally rescaled quadrature measure.
    """

    overlap: np.ndarray
    hamiltonian_raw: np.ndarray
    hamiltonian: np.ndarray
    action_gram: np.ndarray
    overlap_hermiticity: MatrixHermiticityAudit
    hamiltonian_hermiticity: MatrixHermiticityAudit
    action_gram_hermiticity: MatrixHermiticityAudit
    energies: np.ndarray
    coefficients: np.ndarray
    scaled_coefficients: np.ndarray
    root_amplitudes: np.ndarray
    root_actions: np.ndarray
    residual_gram: np.ndarray
    residual_norms: np.ndarray
    relative_residual_norms: np.ndarray
    spectral_intervals: np.ndarray
    projected_residual_matrix: np.ndarray
    overlap_eigenvalues: np.ndarray
    overlap_condition_number: float
    orthonormality_error: float
    column_rescales: np.ndarray
    log_quadrature_rescale: float
    root_state_weight_ess: np.ndarray
    root_state_weight_ess_fractions: np.ndarray
    sample_count: int
    bank_size: int
    root_count: int = 3


def _weighted_graph_counter(
    bonds: Sequence[tuple[int, int, float]],
) -> Counter[tuple[int, int, str]]:
    """Canonical multiset retaining duplicate periodic bonds exactly."""

    return Counter(
        (
            min(int(first), int(second)),
            max(int(first), int(second)),
            float(coupling).hex(),
        )
        for first, second, coupling in bonds
    )


def require_group_graph_compatibility(
    carrier_config: ProjectedSingletAGPConfig,
    graph_config: ProjectedAGPHeisenbergConfig,
) -> None:
    """Require every retained physical permutation to preserve ``H``.

    Abstract group closure alone does not establish that a supplied group is
    a symmetry of a particular weighted graph.  This inexpensive host check
    binds the exact ``Gamma.A1`` carrier to the Hamiltonian used for Ritz
    analysis.  It intentionally treats repeated small-torus bonds as a
    multiset.
    """

    if carrier_config.n_sites != graph_config.n_sites:
        raise ValueError("carrier and graph must have the same site count")
    reference = _weighted_graph_counter(graph_config.bonds)
    for image in carrier_config.group_images:
        transformed = _weighted_graph_counter(
            tuple(
                (image[first], image[second], coupling)
                for first, second, coupling in graph_config.bonds
            )
        )
        if transformed != reference:
            raise ValueError(
                "the projected site group is not an automorphism of the "
                "weighted Hamiltonian graph"
            )


def _validated_parameter_bank(
    parameter_bank: Mapping[str, jax.Array],
    config: ProjectedSingletAGPConfig,
) -> tuple[dict[str, jax.Array], int]:
    expected_keys = (
        {"pairing_real"}
        if config.parameter_mode == "real"
        else {"pairing_real", "pairing_imag"}
    )
    if set(parameter_bank) != expected_keys:
        raise ValueError("parameter-bank keys do not match the carrier mode")
    expected_width = config.parameter_count_per_real_component
    bank: dict[str, jax.Array] = {}
    bank_size: int | None = None
    for name in sorted(expected_keys):
        values = jnp.asarray(parameter_bank[name], dtype=jnp.float64)
        if values.ndim != 2 or values.shape[1] != expected_width:
            raise ValueError(
                f"{name} must have shape (bank_size, {expected_width})"
            )
        if bank_size is None:
            bank_size = int(values.shape[0])
        elif int(values.shape[0]) != bank_size:
            raise ValueError("all parameter-bank leaves must share bank_size")
        bank[name] = values
    assert bank_size is not None
    if bank_size < 1:
        raise ValueError("the resonating parameter bank must be nonempty")
    return bank, bank_size


def resonating_projected_agp_action(
    parameter_bank: Mapping[str, jax.Array],
    configurations: jax.Array | np.ndarray | Sequence[Sequence[int]],
    carrier_config: ProjectedSingletAGPConfig,
    graph_config: ProjectedAGPHeisenbergConfig,
    *,
    amplitude_microbatch_size: int | None = None,
) -> ResonatingProjectedAGPAction:
    """Stack direct ``Phi`` and ``H Phi`` columns for a projected-AGP bank."""

    require_group_graph_compatibility(carrier_config, graph_config)
    bank, bank_size = _validated_parameter_bank(parameter_bank, carrier_config)
    columns = []
    for index in range(bank_size):
        parameters = {name: values[index] for name, values in bank.items()}
        columns.append(
            projected_agp_heisenberg_action(
                parameters,
                configurations,
                carrier_config,
                graph_config,
                amplitude_microbatch_size=amplitude_microbatch_size,
            )
        )
    return ResonatingProjectedAGPAction(
        amplitudes=jnp.stack([column.amplitudes for column in columns], axis=1),
        hamiltonian_actions=jnp.stack(
            [column.hamiltonian_actions for column in columns], axis=1
        ),
        center_exact_nodes=jnp.stack(
            [column.center_exact_nodes for column in columns], axis=1
        ),
        minimum_connected_cancellation_ratio=jnp.stack(
            [
                column.minimum_connected_cancellation_ratio
                for column in columns
            ],
            axis=1,
        ),
    )


def _hermiticity_audit(matrix: np.ndarray) -> MatrixHermiticityAudit:
    difference = matrix - matrix.conj().T
    denominator = max(
        float(np.linalg.norm(matrix, ord="fro")),
        float(np.finfo(np.float64).tiny),
    )
    return MatrixHermiticityAudit(
        maximum_absolute_error=float(
            np.max(np.abs(difference), initial=0.0)
        ),
        relative_frobenius_error=float(
            np.linalg.norm(difference, ord="fro") / denominator
        ),
    )


def _hermitize(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (matrix + matrix.conj().T), dtype=np.complex128)


def _normalized_observation_weights(
    sample_count: int,
    observation_weights: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    if observation_weights is None:
        return np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
    values = np.asarray(observation_weights, dtype=np.float64)
    if values.shape != (sample_count,):
        raise ValueError("observation_weights must have one entry per row")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("observation_weights must be finite and nonnegative")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("observation_weights must have positive total")
    return values / total


def solve_three_root_resonating_subspace(
    action: ResonatingProjectedAGPAction,
    proposal_probabilities: np.ndarray | Sequence[float],
    *,
    observation_weights: np.ndarray | Sequence[float] | None = None,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float = 1.0e8,
    maximum_orthonormality_error: float = 1.0e-8,
) -> ThreeRootResonatingResult:
    """Return the lowest three roots in a direct-amplitude carrier bank.

    Samples drawn independently from normalized ``q`` use the default equal
    observation weights.  Exhaustive deterministic quadrature should pass
    ``observation_weights=q``.  The proposal must be strictly positive on the
    full support even though only its values on supplied rows can be checked
    here.
    """

    amplitudes = np.asarray(action.amplitudes, dtype=np.complex128)
    actions = np.asarray(action.hamiltonian_actions, dtype=np.complex128)
    if (
        amplitudes.ndim != 2
        or amplitudes.shape[0] < 1
        or amplitudes.shape[1] < 3
    ):
        raise NumericalSubspaceRankError(
            "a three-root result requires at least three nonempty bank columns"
        )
    if actions.shape != amplitudes.shape:
        raise ValueError("hamiltonian_actions must match amplitudes")
    if np.any(~np.isfinite(amplitudes)) or np.any(~np.isfinite(actions)):
        raise FloatingPointError(
            "amplitudes and Hamiltonian actions must be finite before rescaling"
        )
    if (
        not np.isfinite(relative_rank_tolerance)
        or not 0.0 < relative_rank_tolerance < 1.0
    ):
        raise ValueError("relative_rank_tolerance must lie between zero and one")
    if not np.isfinite(maximum_condition_number) or maximum_condition_number < 1.0:
        raise ValueError("maximum_condition_number must be finite and at least one")
    if (
        not np.isfinite(maximum_orthonormality_error)
        or maximum_orthonormality_error <= 0.0
    ):
        raise ValueError("maximum_orthonormality_error must be finite and positive")

    sample_count, bank_size = amplitudes.shape
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if proposal.shape != (sample_count,):
        raise ValueError("proposal_probabilities must have one entry per row")
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("proposal probabilities must be finite and positive")
    weights = _normalized_observation_weights(sample_count, observation_weights)

    column_rescales = np.maximum(
        np.maximum(np.max(np.abs(amplitudes), axis=0), np.max(np.abs(actions), axis=0)),
        np.finfo(np.float64).tiny,
    )
    scaled_amplitudes = amplitudes / column_rescales[None, :]
    scaled_actions = actions / column_rescales[None, :]
    with np.errstate(divide="ignore"):
        log_measure = np.where(
            weights > 0.0, np.log(weights) - np.log(proposal), -np.inf
        )
    log_quadrature_rescale = float(np.max(log_measure))
    if not np.isfinite(log_quadrature_rescale):
        raise ValueError("the supplied quadrature has no positive observation row")
    measure = np.where(
        np.isfinite(log_measure),
        np.exp(log_measure - log_quadrature_rescale),
        0.0,
    )

    def weighted_gram(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.einsum(
            "x,xa,xb->ab", measure, left.conj(), right, optimize=True
        )

    overlap_raw = weighted_gram(scaled_amplitudes, scaled_amplitudes)
    hamiltonian_raw = weighted_gram(scaled_amplitudes, scaled_actions)
    action_gram_raw = weighted_gram(scaled_actions, scaled_actions)
    overlap_audit = _hermiticity_audit(overlap_raw)
    hamiltonian_audit = _hermiticity_audit(hamiltonian_raw)
    action_gram_audit = _hermiticity_audit(action_gram_raw)
    overlap = _hermitize(overlap_raw)
    hamiltonian = _hermitize(hamiltonian_raw)
    action_gram = _hermitize(action_gram_raw)

    try:
        overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    except np.linalg.LinAlgError as error:
        raise NumericalSubspaceRankError(
            "overlap eigendecomposition failed"
        ) from error
    largest = float(overlap_values[-1])
    threshold = relative_rank_tolerance * largest
    rank = int(np.count_nonzero(overlap_values > threshold))
    if largest <= 0.0 or rank != bank_size:
        raise NumericalSubspaceRankError(
            "resonating bank is not full rank: "
            f"rank={rank}/{bank_size}, lambda_min={float(overlap_values[0]):.6e}, "
            f"lambda_max={largest:.6e}"
        )
    condition = largest / float(overlap_values[0])
    if condition > maximum_condition_number:
        raise NumericalSubspaceRankError(
            "resonating-bank overlap exceeds the condition-number gate: "
            f"{condition:.6e} > {maximum_condition_number:.6e}"
        )

    whitening = overlap_vectors / np.sqrt(overlap_values)[None, :]
    effective = _hermitize(whitening.conj().T @ hamiltonian @ whitening)
    try:
        all_energies, all_vectors = np.linalg.eigh(effective)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("effective Hamiltonian solve failed") from error
    scaled_coefficients = whitening @ all_vectors[:, :3]
    energies = np.asarray(all_energies[:3], dtype=np.float64)
    normalization = (
        scaled_coefficients.conj().T @ overlap @ scaled_coefficients
    )
    orthonormality_error = float(
        np.max(np.abs(normalization - np.eye(3)), initial=0.0)
    )
    if orthonormality_error > maximum_orthonormality_error:
        raise FloatingPointError(
            "three-root generalized Ritz normalization failed: "
            f"error={orthonormality_error:.6e}"
        )

    root_amplitudes = scaled_amplitudes @ scaled_coefficients
    root_actions = scaled_actions @ scaled_coefficients
    residual_rows = root_actions - root_amplitudes * energies[None, :]
    residual_gram = _hermitize(weighted_gram(residual_rows, residual_rows))
    squared = np.real(np.diag(residual_gram))
    negative_tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * max(float(np.max(np.abs(residual_gram), initial=0.0)), 1.0)
    )
    if np.any(squared < -negative_tolerance):
        raise FloatingPointError("a direct three-root residual norm is negative")
    residual_norms = np.sqrt(np.maximum(squared, 0.0))
    relative_residuals = residual_norms / np.maximum(
        np.abs(energies), np.finfo(np.float64).tiny
    )
    projected_residual = weighted_gram(root_amplitudes, residual_rows)

    root_state_weights = measure[:, None] * np.abs(root_amplitudes) ** 2
    weight_sums = np.sum(root_state_weights, axis=0)
    ess = weight_sums**2 / np.sum(root_state_weights**2, axis=0)
    coefficients = scaled_coefficients / column_rescales[:, None]
    return ThreeRootResonatingResult(
        overlap=overlap,
        hamiltonian_raw=hamiltonian_raw,
        hamiltonian=hamiltonian,
        action_gram=action_gram,
        overlap_hermiticity=overlap_audit,
        hamiltonian_hermiticity=hamiltonian_audit,
        action_gram_hermiticity=action_gram_audit,
        energies=energies,
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        scaled_coefficients=np.asarray(scaled_coefficients, dtype=np.complex128),
        root_amplitudes=np.asarray(root_amplitudes, dtype=np.complex128),
        root_actions=np.asarray(root_actions, dtype=np.complex128),
        residual_gram=np.asarray(residual_gram, dtype=np.complex128),
        residual_norms=np.asarray(residual_norms, dtype=np.float64),
        relative_residual_norms=np.asarray(relative_residuals, dtype=np.float64),
        spectral_intervals=np.stack(
            (energies - residual_norms, energies + residual_norms), axis=1
        ),
        projected_residual_matrix=np.asarray(
            projected_residual, dtype=np.complex128
        ),
        overlap_eigenvalues=np.asarray(overlap_values, dtype=np.float64),
        overlap_condition_number=float(condition),
        orthonormality_error=orthonormality_error,
        column_rescales=np.asarray(column_rescales, dtype=np.float64),
        log_quadrature_rescale=log_quadrature_rescale,
        root_state_weight_ess=np.asarray(ess, dtype=np.float64),
        root_state_weight_ess_fractions=np.asarray(
            ess / sample_count, dtype=np.float64
        ),
        sample_count=sample_count,
        bank_size=bank_size,
    )


__all__ = [
    "MatrixHermiticityAudit",
    "ResonatingProjectedAGPAction",
    "ThreeRootResonatingResult",
    "require_group_graph_compatibility",
    "resonating_projected_agp_action",
    "solve_three_root_resonating_subspace",
]
