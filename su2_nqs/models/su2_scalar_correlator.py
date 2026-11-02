"""Configuration-dependent scalar correlator on local-Schur data.

Every input is a reduced SU(2) label: local doubled spins/multiplicities and
inter-cell doubled spins.  The correlator never sees magnetic indices.  It is
therefore an SU(2) scalar by construction; an enclosing exact spatial
projector fixes the requested lattice irrep.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..su2.local_schur_fusion import local_schur_channels
from .su2_four_cell_sector_vit import SU2FourCellSectorViTConfig


@dataclass(frozen=True)
class SchurScalarFeatures:
    """Standardized physical scalar features and their semantic labels."""

    values: np.ndarray
    labels: tuple[str, ...]


def _casimir(doubled_q: np.ndarray) -> np.ndarray:
    values = np.asarray(doubled_q, dtype=np.float64)
    return 0.25 * values * (values + 2.0)


def schur_scalar_features(
    actions: np.ndarray,
    config: SU2FourCellSectorViTConfig,
) -> SchurScalarFeatures:
    """Return nonconstant standardized scalar features for every action row."""

    rows = np.asarray(actions)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError("actions must have shape (batch,6)")
    if rows.dtype == np.bool_ or not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("actions must use an integer dtype")
    channels = local_schur_channels(config.kind)
    cell_actions = rows[:, (0, 1, 3, 4)]
    if np.any(cell_actions < 0) or np.any(cell_actions >= len(channels)):
        raise ValueError("a local channel action lies outside the Schur alphabet")
    channel_q = np.asarray([q for q, _a in channels], dtype=np.int32)
    channel_a = np.asarray([a for _q, a in channels], dtype=np.int32)
    local_q = channel_q[cell_actions]
    local_a = channel_a[cell_actions]
    local_c = _casimir(local_q)
    left_q = rows[:, 2]
    right_q = rows[:, 5]
    if np.any(left_q < 0) or np.any(right_q < 0):
        raise ValueError("inter-cell doubled spins must be nonnegative")
    pair_c = _casimir(np.column_stack((left_q, right_q)))

    columns: list[np.ndarray] = []
    labels: list[str] = []

    def add(label: str, values: np.ndarray) -> None:
        labels.append(label)
        columns.append(np.asarray(values, dtype=np.float64))

    for channel in range(len(channels)):
        add(
            f"local_channel_count_{channel}",
            np.sum(cell_actions == channel, axis=1),
        )
    maximum_pair_q = max(int(np.max(left_q)), int(np.max(right_q)))
    for doubled_q in range(maximum_pair_q + 1):
        add(
            f"pair_q_count_{doubled_q}",
            np.sum(np.column_stack((left_q, right_q)) == doubled_q, axis=1),
        )
    add("local_casimir_sum", np.sum(local_c, axis=1))
    add("local_casimir_square_sum", np.sum(local_c**2, axis=1))
    add("local_multiplicity_sum", np.sum(local_a, axis=1))
    add("local_multiplicity_square_sum", np.sum(local_a**2, axis=1))
    add("pair_casimir_sum", np.sum(pair_c, axis=1))
    add("pair_casimir_product", np.prod(pair_c, axis=1))
    add("pair_casimir_difference_square", (pair_c[:, 0] - pair_c[:, 1]) ** 2)

    coordinates = np.asarray(config.cell_coordinates, dtype=np.float64)
    maximum_distance = max(
        np.linalg.norm(coordinates[first] - coordinates[second])
        for first in range(4)
        for second in range(first + 1, 4)
    )
    for first in range(4):
        for second in range(first + 1, 4):
            distance = float(
                np.linalg.norm(coordinates[first] - coordinates[second])
                / maximum_distance
            )
            weight = 1.0 / max(distance, 1.0e-12)
            prefix = f"cell_pair_{first}_{second}"
            add(prefix + "_casimir_product", weight * local_c[:, first] * local_c[:, second])
            add(prefix + "_q_difference_square", weight * (local_q[:, first] - local_q[:, second]) ** 2)
            add(
                prefix + "_multiplicity_resonance",
                weight
                * (
                    (local_q[:, first] == local_q[:, second])
                    & (local_a[:, first] == local_a[:, second])
                ),
            )

    raw = np.column_stack(columns)
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    keep = scale > 1.0e-12
    standardized = (raw[:, keep] - mean[keep]) / scale[keep]
    standardized.setflags(write=False)
    return SchurScalarFeatures(
        values=standardized,
        labels=tuple(label for label, selected in zip(labels, keep, strict=True) if selected),
    )


def init_schur_scalar_correlator(feature_count: int) -> dict[str, jnp.ndarray]:
    """Return a zero correction so all matched arms share the initial state."""

    if isinstance(feature_count, bool) or not isinstance(feature_count, int):
        raise TypeError("feature_count must be an integer")
    if feature_count < 1:
        raise ValueError("feature_count must be positive")
    return {
        "log_magnitude": jnp.zeros((feature_count,), dtype=jnp.float64),
        "phase": jnp.zeros((feature_count,), dtype=jnp.float64),
    }


def schur_scalar_log_correction(
    params: dict[str, jnp.ndarray],
    features: jnp.ndarray,
    *,
    phase_scale: float = 0.25,
) -> jnp.ndarray:
    """Return one complex log-correlator value per reduced configuration."""

    values = jnp.asarray(features, dtype=jnp.float64)
    if values.ndim != 2:
        raise ValueError("features must be a matrix")
    expected = (values.shape[1],)
    if set(params) != {"log_magnitude", "phase"}:
        raise ValueError("scalar-correlator parameters have unexpected keys")
    magnitude = jnp.asarray(params["log_magnitude"], dtype=jnp.float64)
    phase = jnp.asarray(params["phase"], dtype=jnp.float64)
    if magnitude.shape != expected or phase.shape != expected:
        raise ValueError("scalar-correlator parameter shape mismatch")
    real = values @ magnitude
    imaginary = float(phase_scale) * np.pi * jnp.tanh(values @ phase)
    return real + 1.0j * imaginary


__all__ = [
    "SchurScalarFeatures",
    "init_schur_scalar_correlator",
    "schur_scalar_features",
    "schur_scalar_log_correction",
]
