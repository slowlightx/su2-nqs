"""Born-weighted tangent-space projections for causal NQS diagnostics.

The routines in this module diagnose the *instantaneous* variational tangent
space at one fixed wavefunction.  They do not optimize the model and their
projection fractions must not be interpreted as finite-training recovery.

For real parameters and complex log derivatives ``O[x, a]``, define the
Born-centered, realified tangent matrix

``A = [sqrt(2 p) Re(O - <O>_p); sqrt(2 p) Im(O - <O>_p)]``.

The column-space projection of an equally realified complex response is the
Fisher-whitened response that can be represented to first order.  In
particular, for the centered local-energy residual ``r = E_loc - E``,
``||P_A b_r||^2`` is the predicted energy decrease per unit natural-gradient
step in this convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .sparse_causal_tangent import HEAD_GROUPS, HEAD_NAMES
from ..models.sparse_causal_tree import (
    SparseCausalTreeGRUConfig,
    sparse_causal_tree_log_psi,
)


@dataclass(frozen=True)
class BornTangentProjection:
    """Stable SVD solution of one Born-weighted tangent projection."""

    normalized_weights: np.ndarray
    centered_target: np.ndarray
    design_matrix: np.ndarray
    target_vector: np.ndarray
    projected_target_vector: np.ndarray
    least_squares_coefficients: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int
    relative_rank_cutoff: float
    absolute_rank_cutoff: float
    tangent_condition_number: float | None
    fisher_condition_number: float | None
    target_norm_squared: float
    projected_norm_squared: float
    projection_fraction: float
    orthogonal_residual_norm_squared: float
    direct_normal_equation_response: float

    def scalar_summary(self) -> dict[str, object]:
        """Return JSON-safe scalar diagnostics without large dense arrays."""

        active = self.singular_values[: self.numerical_rank]
        return {
            "n_states": int(self.normalized_weights.size),
            "parameter_count": int(self.design_matrix.shape[1]),
            "realified_dimension": int(self.design_matrix.shape[0]),
            "numerical_rank": int(self.numerical_rank),
            "relative_rank_cutoff": float(self.relative_rank_cutoff),
            "absolute_rank_cutoff": float(self.absolute_rank_cutoff),
            "largest_singular_value": (
                float(active[0]) if active.size else 0.0
            ),
            "smallest_retained_singular_value": (
                float(active[-1]) if active.size else 0.0
            ),
            "tangent_condition_number": self.tangent_condition_number,
            "fisher_condition_number": self.fisher_condition_number,
            "target_norm_squared": float(self.target_norm_squared),
            "projected_norm_squared": float(self.projected_norm_squared),
            "projection_fraction": float(self.projection_fraction),
            "orthogonal_residual_norm_squared": float(
                self.orthogonal_residual_norm_squared
            ),
            "direct_normal_equation_response": float(
                self.direct_normal_equation_response
            ),
            "minimum_norm_parameter_direction_l2": float(
                np.linalg.norm(self.least_squares_coefficients)
            ),
            "minimum_norm_parameter_direction_max_abs": float(
                np.max(
                    np.abs(self.least_squares_coefficients), initial=0.0
                )
            ),
        }


@dataclass(frozen=True)
class BornTangentFactorization:
    """Reusable Fisher-whitened column-space factorization."""

    normalized_weights: np.ndarray
    design_matrix: np.ndarray
    retained_left_vectors: np.ndarray
    retained_singular_values: np.ndarray
    retained_right_vectors: np.ndarray
    singular_values: np.ndarray
    relative_rank_cutoff: float
    absolute_rank_cutoff: float

    @property
    def numerical_rank(self) -> int:
        return int(self.retained_singular_values.size)

    @property
    def tangent_condition_number(self) -> float | None:
        if not self.numerical_rank:
            return None
        return float(
            self.retained_singular_values[0]
            / self.retained_singular_values[-1]
        )

    def rank_at_relative_cutoff(self, relative_rank_cutoff: float) -> int:
        """Return numerical rank at a cutoff covered by this factorization."""

        if not np.isfinite(relative_rank_cutoff) or relative_rank_cutoff <= 0.0:
            raise ValueError("relative rank cutoff must be positive and finite")
        largest = float(self.singular_values[0]) if self.singular_values.size else 0.0
        rank = int(
            np.count_nonzero(
                self.singular_values > relative_rank_cutoff * largest
            )
        )
        if rank > self.numerical_rank:
            raise ValueError(
                "requested cutoff retains directions discarded by this factorization"
            )
        return rank

    def project(
        self,
        target: np.ndarray,
        *,
        relative_rank_cutoff: float | None = None,
    ) -> BornTangentProjection:
        """Project one response without recomputing the tangent SVD."""

        centered_target, target_vector = born_centered_real_target(
            target, self.normalized_weights
        )
        cutoff = (
            self.relative_rank_cutoff
            if relative_rank_cutoff is None
            else float(relative_rank_cutoff)
        )
        rank = self.rank_at_relative_cutoff(cutoff)
        singular = self.retained_singular_values[:rank]
        left = self.retained_left_vectors[:, :rank]
        right = self.retained_right_vectors[:rank]
        if rank:
            overlaps = left.T @ target_vector
            coefficients = right.T @ (
                overlaps / singular
            )
            projected = left @ overlaps
        else:
            coefficients = np.zeros(
                (self.design_matrix.shape[1],), dtype=np.float64
            )
            projected = np.zeros_like(target_vector)
        target_norm_squared = float(np.dot(target_vector, target_vector))
        projected_norm_squared = float(np.dot(projected, projected))
        fraction = (
            min(1.0, max(0.0, projected_norm_squared / target_norm_squared))
            if target_norm_squared > 0.0
            else 0.0
        )
        residual = target_vector - projected
        response = float(
            np.dot(self.design_matrix.T @ target_vector, coefficients)
        )
        condition = float(singular[0] / singular[-1]) if rank else None
        largest = float(self.singular_values[0]) if self.singular_values.size else 0.0
        return BornTangentProjection(
            normalized_weights=self.normalized_weights,
            centered_target=centered_target,
            design_matrix=self.design_matrix,
            target_vector=target_vector,
            projected_target_vector=projected,
            least_squares_coefficients=coefficients,
            singular_values=self.singular_values,
            numerical_rank=rank,
            relative_rank_cutoff=cutoff,
            absolute_rank_cutoff=cutoff * largest,
            tangent_condition_number=condition,
            fisher_condition_number=(
                float(condition**2) if condition is not None else None
            ),
            target_norm_squared=target_norm_squared,
            projected_norm_squared=projected_norm_squared,
            projection_fraction=fraction,
            orthogonal_residual_norm_squared=float(np.dot(residual, residual)),
            direct_normal_equation_response=response,
        )

    def damped_natural_gradient_response(
        self,
        target: np.ndarray,
        damping: float,
    ) -> dict[str, float]:
        """Predict a damped unit natural-gradient response in linear order."""

        if not np.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be positive and finite")
        _, target_vector = born_centered_real_target(
            target, self.normalized_weights
        )
        singular = self.retained_singular_values
        if singular.size:
            overlaps = self.retained_left_vectors.T @ target_vector
            filter_values = singular**2 / (singular**2 + damping)
            response = float(np.sum(filter_values * overlaps**2))
            direction_norm = float(
                np.linalg.norm(
                    singular * overlaps / (singular**2 + damping)
                )
            )
            induced_tangent_norm = float(
                np.linalg.norm(filter_values * overlaps)
            )
        else:
            response = 0.0
            direction_norm = 0.0
            induced_tangent_norm = 0.0
        target_norm_squared = float(np.dot(target_vector, target_vector))
        return {
            "damping": float(damping),
            "predicted_energy_decrease_per_unit_damped_step": response,
            "fraction_of_target_norm_in_damped_response": (
                response / target_norm_squared
                if target_norm_squared > 0.0
                else 0.0
            ),
            "parameter_direction_l2_norm": direction_norm,
            "induced_tangent_l2_norm": induced_tangent_norm,
        }


@dataclass(frozen=True)
class MessageHeadLogDerivative:
    """Materialized sparse-message-head log derivative and column metadata."""

    values: np.ndarray
    column_slices: Mapping[str, slice]
    elapsed_seconds: float
    differentiation_mode: str
    autodiff_block_size: int

    def columns(self, names: Sequence[str]) -> np.ndarray:
        """Return the derivative columns belonging to the requested heads."""

        indices = np.concatenate(
            [
                np.arange(
                    self.column_slices[name].start,
                    self.column_slices[name].stop,
                    dtype=np.int64,
                )
                for name in names
            ]
        )
        return self.values[:, indices]

    def group(self, name: str) -> np.ndarray:
        """Return one named component group or all six message heads."""

        if name == "all":
            names = HEAD_NAMES
        else:
            try:
                names = HEAD_GROUPS[name]
            except KeyError as exc:
                raise ValueError(f"unknown sparse tangent group: {name}") from exc
        return self.columns(names)


def _normalized_weights(weights: np.ndarray, n_states: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (n_states,):
        raise ValueError("Born weights require one value per state")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Born weights must be finite and non-negative")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Born weights must have positive total mass")
    return values / total


def born_centered_real_design(
    log_derivatives: np.ndarray,
    born_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized weights, centered derivatives, and real design.

    ``log_derivatives`` has shape ``(n_states, n_real_parameters)``.  The
    returned design uses the same factor-of-two convention as the exact SR
    implementation in :mod:`su2_nqs.vmc.sampled_sr`.
    """

    derivatives = np.asarray(log_derivatives, dtype=np.complex128)
    if derivatives.ndim != 2:
        raise ValueError("log derivatives must be a state-by-parameter matrix")
    if np.any(~np.isfinite(derivatives)):
        raise ValueError("log derivatives must be finite")
    weights = _normalized_weights(born_weights, derivatives.shape[0])
    mean = np.sum(weights[:, None] * derivatives, axis=0, keepdims=True)
    centered = derivatives - mean
    scale = np.sqrt(2.0 * weights)[:, None]
    design = np.concatenate(
        (scale * np.real(centered), scale * np.imag(centered)), axis=0
    )
    return weights, centered, design


def born_centered_real_target(
    target: np.ndarray,
    normalized_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Born-center and realify one complex statewise response."""

    values = np.asarray(target, dtype=np.complex128)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("target and normalized weights must be equal-length vectors")
    if np.any(~np.isfinite(values)):
        raise ValueError("target response must be finite")
    centered = values - np.sum(weights * values)
    scale = np.sqrt(2.0 * weights)
    realified = np.concatenate(
        (scale * np.real(centered), scale * np.imag(centered))
    )
    return centered, realified


def project_born_complex_tangent(
    log_derivatives: np.ndarray,
    born_weights: np.ndarray,
    target: np.ndarray,
    *,
    relative_rank_cutoff: float = 1.0e-10,
) -> BornTangentProjection:
    """Project a centered complex response onto a centered NQS tangent.

    A compact SVD supplies the Moore--Penrose solution and a rank-revealing
    Fisher whitening.  The result is invariant under any nonsingular linear
    reparameterization whose singular directions remain above the requested
    numerical cutoff.
    """

    factorization = factor_born_complex_tangent(
        log_derivatives,
        born_weights,
        relative_rank_cutoff=relative_rank_cutoff,
    )
    return factorization.project(target)


def factor_born_complex_tangent(
    log_derivatives: np.ndarray,
    born_weights: np.ndarray,
    *,
    relative_rank_cutoff: float = 1.0e-10,
) -> BornTangentFactorization:
    """Factor one centered tangent for reuse across physical targets."""

    if not np.isfinite(relative_rank_cutoff) or relative_rank_cutoff <= 0.0:
        raise ValueError("relative rank cutoff must be positive and finite")
    weights, _, design = born_centered_real_design(log_derivatives, born_weights)
    n_parameters = design.shape[1]
    if n_parameters == 0:
        singular_values = np.empty((0,), dtype=np.float64)
        absolute_cutoff = 0.0
        retained_left = np.empty((design.shape[0], 0), dtype=np.float64)
        retained_singular = np.empty((0,), dtype=np.float64)
        retained_right = np.empty((0, 0), dtype=np.float64)
    else:
        left, singular_values, right_h = np.linalg.svd(
            design, full_matrices=False
        )
        largest = float(singular_values[0]) if singular_values.size else 0.0
        absolute_cutoff = relative_rank_cutoff * largest
        rank = int(np.count_nonzero(singular_values > absolute_cutoff))
        retained_left = left[:, :rank]
        retained_singular = singular_values[:rank]
        retained_right = right_h[:rank]
    return BornTangentFactorization(
        normalized_weights=weights,
        design_matrix=design,
        retained_left_vectors=retained_left,
        retained_singular_values=retained_singular,
        retained_right_vectors=retained_right,
        singular_values=singular_values,
        relative_rank_cutoff=float(relative_rank_cutoff),
        absolute_rank_cutoff=float(absolute_cutoff),
    )


def local_energy_tangent_response(
    log_derivatives: np.ndarray,
    born_weights: np.ndarray,
    local_energies: np.ndarray,
    *,
    relative_rank_cutoff: float = 1.0e-10,
) -> dict[str, object]:
    """Return the instantaneous natural-gradient energy response.

    The predicted decrease is the first-order energy change for a unit step
    along the Moore--Penrose natural-gradient direction at the supplied base
    state.  It is neither a finite-step energy difference nor a training
    recovery rate.
    """

    projection = project_born_complex_tangent(
        log_derivatives,
        born_weights,
        local_energies,
        relative_rank_cutoff=relative_rank_cutoff,
    )
    summary = projection.scalar_summary()
    summary.update(
        {
            "local_energy_variance": float(
                projection.target_norm_squared / 2.0
            ),
            "captured_local_energy_variance": float(
                projection.projected_norm_squared / 2.0
            ),
            "predicted_energy_decrease_per_unit_natural_gradient_step": float(
                projection.projected_norm_squared
            ),
            "interpretation": "instantaneous_tangent_response_not_training_recovery",
        }
    )
    return summary


def materialize_message_head_log_derivatives(
    params: dict[str, object],
    config: SparseCausalTreeGRUConfig,
    states: jax.Array,
    *,
    head_names: Sequence[str] = HEAD_NAMES,
    differentiation_mode: str = "auto",
    autodiff_block_size: int = 16,
) -> MessageHeadLogDerivative:
    """Materialize derivatives with respect to the six sparse message heads.

    The base TreeGRU and message encoders are held fixed.  Forward columns or
    reverse rows are evaluated in bounded blocks so the N=16 full-support
    diagnostic does not construct a parameter-square Fisher matrix or a giant
    identity array.  ``head_names`` permits a component diagnostic without
    first differentiating every unused head.
    """

    if autodiff_block_size < 1:
        raise ValueError("autodiff block size must be positive")
    selected_names = tuple(head_names)
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise ValueError("head names must be non-empty and unique")
    if any(name not in HEAD_NAMES for name in selected_names):
        raise ValueError("unknown sparse message head")
    if differentiation_mode not in ("auto", "forward", "reverse"):
        raise ValueError("differentiation mode must be auto, forward, or reverse")
    labels = jnp.asarray(states, dtype=jnp.int32)
    if labels.ndim != 2 or labels.shape[1] != config.n - 1:
        raise ValueError("states have the wrong fusion-label shape")
    if "base" not in params or "message" not in params:
        raise ValueError("sparse parameters require base and message trees")
    if any(name not in params["message"] for name in selected_names):
        raise ValueError("sparse parameters are missing a message head")

    fixed_message = {
        name: value
        for name, value in params["message"].items()
        if name not in selected_names
    }
    head_tuple = tuple(params["message"][name] for name in selected_names)
    flat_heads, unravel_heads = ravel_pytree(head_tuple)
    if flat_heads.dtype != jnp.float64:
        raise ValueError("reference tangent materialization requires float64 heads")

    column_slices: dict[str, slice] = {}
    offset = 0
    for name, value in zip(selected_names, head_tuple, strict=True):
        size = int(np.prod(value.shape, dtype=np.int64))
        column_slices[name] = slice(offset, offset + size)
        offset += size

    def realified_logs(flat):
        candidate_message = dict(fixed_message)
        candidate_message.update(
            dict(zip(selected_names, unravel_heads(flat), strict=True))
        )
        candidate = {"base": params["base"], "message": candidate_message}
        # Anchored configurations are intentionally excluded: this diagnostic
        # is the plain six-head tangent used by the current sparse gate.
        if "anchor" in params:
            candidate["anchor"] = params["anchor"]
        logs = jax.vmap(
            lambda state: sparse_causal_tree_log_psi(candidate, state, config)
        )(labels)
        return jnp.concatenate((jnp.real(logs), jnp.imag(logs)))

    started = time.perf_counter()
    n_parameters = int(flat_heads.size)
    if differentiation_mode == "auto":
        # Forward columns are preferable for component ablations (48--2304
        # head parameters versus 2860 real outputs at N=16).  Reverse rows
        # remains available for future wider-head configurations.
        probe = realified_logs(flat_heads)
        differentiation_mode = (
            "forward" if n_parameters <= int(probe.size) else "reverse"
        )
    if differentiation_mode == "forward":
        outputs, pushforward = jax.linearize(realified_logs, flat_heads)
        n_outputs = int(outputs.size)

        @jax.jit
        def push_columns(tangents):
            return jax.vmap(pushforward)(tangents)

        blocks = []
        for start in range(0, n_parameters, autodiff_block_size):
            stop = min(start + autodiff_block_size, n_parameters)
            tangents = np.zeros(
                (autodiff_block_size, n_parameters), dtype=np.float64
            )
            columns = np.arange(start, stop)
            tangents[np.arange(stop - start), columns] = 1.0
            block = np.asarray(push_columns(jnp.asarray(tangents)))[
                : stop - start
            ]
            blocks.append(block)
        realified_jacobian = np.concatenate(blocks, axis=0).T
    else:
        outputs, pullback = jax.vjp(realified_logs, flat_heads)
        n_outputs = int(outputs.size)

        @jax.jit
        def pull_rows(cotangents):
            return jax.vmap(lambda row: pullback(row)[0])(cotangents)

        blocks = []
        for start in range(0, n_outputs, autodiff_block_size):
            stop = min(start + autodiff_block_size, n_outputs)
            cotangents = np.zeros(
                (autodiff_block_size, n_outputs), dtype=np.float64
            )
            rows = np.arange(start, stop)
            cotangents[np.arange(stop - start), rows] = 1.0
            block = np.asarray(pull_rows(jnp.asarray(cotangents)))[
                : stop - start
            ]
            blocks.append(block)
        realified_jacobian = np.concatenate(blocks, axis=0)
    n_states = labels.shape[0]
    values = (
        realified_jacobian[:n_states]
        + 1j * realified_jacobian[n_states:]
    )
    if np.any(~np.isfinite(values)):
        raise FloatingPointError("message-head log derivative contains non-finite values")
    return MessageHeadLogDerivative(
        values=np.asarray(values, dtype=np.complex128),
        column_slices=column_slices,
        elapsed_seconds=time.perf_counter() - started,
        differentiation_mode=differentiation_mode,
        autodiff_block_size=autodiff_block_size,
    )


__all__ = [
    "BornTangentFactorization",
    "BornTangentProjection",
    "MessageHeadLogDerivative",
    "born_centered_real_design",
    "born_centered_real_target",
    "factor_born_complex_tangent",
    "local_energy_tangent_response",
    "materialize_message_head_log_derivatives",
    "project_born_complex_tangent",
]
