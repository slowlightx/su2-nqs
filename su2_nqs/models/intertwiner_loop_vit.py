"""Configuration-dependent SU(2)-scalar attention on virtual singlet loops.

This is deliberately not a direct fusion-path model.  Four complete local
Schur cells expose only reduced ``(q,a)`` labels.  In each pairing frame the
native tensor connects two pairs of equal cell irreps to ``pair_q=0``.  A
shared Spatial-ViT assigns configuration-dependent reduced coefficients to
those virtual singlet bonds.  Exact four-cell recoupling then coherently adds
the horizontal, vertical, and crossed frames in one canonical multiplicity
basis.

Magnetic indices never enter the learned model.  A lattice projector can be
applied to the returned canonical coefficient vector by the benchmark layer;
the model itself contains no Hamiltonian logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.four_cell_recoupling import (
    PairingOrder,
    four_cell_pairing_recoupling,
    pair_singlet_state_indices,
    validate_pairing_order,
)
from ..su2.local_schur import CellKind, cell_schur_spec
from ..su2.local_schur_fusion import four_cell_schur_states


jax.config.update("jax_enable_x64", True)


DEFAULT_PAIRING_FRAMES: tuple[PairingOrder, ...] = (
    (0, 1, 2, 3),
    (0, 2, 1, 3),
    (0, 3, 1, 2),
)


@dataclass(frozen=True)
class IntertwinerLoopViTConfig:
    """Shared reduced-attention architecture for four Schur cells."""

    kind: CellKind
    cell_coordinates: tuple[tuple[float, float], ...]
    frame_orders: tuple[PairingOrder, ...] = DEFAULT_PAIRING_FRAMES
    embedding_dim: int = 4
    head_count: int = 1
    depth: int = 1
    mlp_ratio: int = 2
    initialization_scale: float = 0.15

    def __post_init__(self) -> None:
        _ = cell_schur_spec(self.kind)
        coordinates = np.asarray(self.cell_coordinates, dtype=np.float64)
        if coordinates.shape != (4, 2) or np.any(~np.isfinite(coordinates)):
            raise ValueError("four finite two-dimensional cell coordinates are required")
        if len({tuple(row) for row in coordinates}) != 4:
            raise ValueError("cell coordinates must be distinct")
        object.__setattr__(
            self,
            "cell_coordinates",
            tuple(tuple(float(value) for value in row) for row in coordinates),
        )
        frames = tuple(validate_pairing_order(order) for order in self.frame_orders)
        if not frames or len(set(frames)) != len(frames):
            raise ValueError("pairing frames must be nonempty and distinct")
        object.__setattr__(self, "frame_orders", frames)
        for name in ("embedding_dim", "head_count", "depth", "mlp_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if not np.isfinite(self.initialization_scale) or self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive and finite")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio

    @property
    def coefficient_dimension(self) -> int:
        return len(four_cell_schur_states(self.kind))

    @property
    def loop_configuration_count(self) -> int:
        return len(pair_singlet_state_indices(self.kind))


def intertwiner_loop_labels(kind: CellKind) -> np.ndarray:
    """Return immutable local ``(q,a)`` labels for native pair singlets."""

    states = four_cell_schur_states(kind)
    indices = pair_singlet_state_indices(kind)
    result = np.asarray(
        [
            [
                (states[int(index)].cell_q[cell], states[int(index)].cell_a[cell])
                for cell in range(4)
            ]
            for index in indices
        ],
        dtype=np.int32,
    )
    result.setflags(write=False)
    return result


def _frame_geometry(
    config: IntertwinerLoopViTConfig,
    order: PairingOrder,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(config.cell_coordinates, dtype=np.float64)[
        np.asarray(order, dtype=np.int32)
    ]
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    scale = max(float(np.max(np.linalg.norm(centered, axis=1))), 1.0)
    centered /= scale
    delta = centered[None, :, :] - centered[:, None, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    paired = np.zeros((4, 4, 1), dtype=np.float64)
    for left, right in ((0, 1), (2, 3)):
        paired[left, right, 0] = 1.0
        paired[right, left, 0] = 1.0
    pair_features = np.concatenate((delta, distance, paired), axis=-1)
    centered.setflags(write=False)
    pair_features.setflags(write=False)
    return centered, pair_features


def _matrix(key: jax.Array, rows: int, columns: int, scale: float) -> jax.Array:
    standard = float(scale) * np.sqrt(2.0 / float(rows + columns))
    return standard * jax.random.normal(key, (rows, columns), dtype=jnp.float64)


def init_intertwiner_loop_vit(
    key: jax.Array,
    config: IntertwinerLoopViTConfig,
) -> dict[str, object]:
    """Initialize one parameter set shared by every pairing frame."""

    keys = iter(jax.random.split(key, 5 + 7 * config.depth))
    scale = float(config.initialization_scale)
    layers = []
    for _ in range(config.depth):
        layers.append(
            {
                "query": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "key": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "value": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "output": _matrix(next(keys), config.embedding_dim, config.embedding_dim, scale),
                "pair_weight": _matrix(next(keys), 4, config.head_count, scale),
                "mlp_in": _matrix(next(keys), config.embedding_dim, config.mlp_dim, scale),
                "mlp_out": _matrix(next(keys), config.mlp_dim, config.embedding_dim, scale),
            }
        )
    spec = cell_schur_spec(config.kind)
    return {
        "q_embedding": scale
        * jax.random.normal(
            next(keys), (max(spec.q_values) + 1, config.embedding_dim), dtype=jnp.float64
        ),
        "multiplicity_embedding": scale
        * jax.random.normal(
            next(keys),
            (spec.maximum_multiplicity, config.embedding_dim),
            dtype=jnp.float64,
        ),
        "position_projection": _matrix(next(keys), 2, config.embedding_dim, scale),
        "coefficient_head": _matrix(next(keys), config.embedding_dim, 1, scale)[:, 0],
        "coefficient_bias": jnp.asarray(scale, dtype=jnp.float64),
        "layers": tuple(layers),
    }


def _layer_norm(values: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8)


def intertwiner_loop_frame_coefficients(
    params: dict[str, object],
    config: IntertwinerLoopViTConfig,
    order: PairingOrder,
) -> jax.Array:
    """Return signed reduced coefficients for one native loop frame."""

    checked = validate_pairing_order(order)
    if checked not in config.frame_orders:
        raise ValueError("pairing order is not registered in the configuration")
    labels = jnp.asarray(intertwiner_loop_labels(config.kind))
    coordinates_np, pair_np = _frame_geometry(config, checked)
    coordinates = jnp.asarray(coordinates_np)
    pair = jnp.asarray(pair_np)
    q = labels[..., 0]
    multiplicity = labels[..., 1]
    tokens = (
        params["q_embedding"][q]
        + params["multiplicity_embedding"][multiplicity]
        + coordinates @ params["position_projection"]
    )
    histories = []
    for layer in params["layers"]:
        normalized = _layer_norm(tokens)
        query = (normalized @ layer["query"]).reshape(
            len(labels), 4, config.head_count, config.head_dim
        )
        key = (normalized @ layer["key"]).reshape(
            len(labels), 4, config.head_count, config.head_dim
        )
        value = (normalized @ layer["value"]).reshape(
            len(labels), 4, config.head_count, config.head_dim
        )
        logits = jnp.einsum("bihd,bjhd->bhij", query, key) / np.sqrt(
            config.head_dim
        )
        logits = logits + jnp.einsum(
            "ijf,fh->hij", pair, layer["pair_weight"]
        )[None, ...]
        attention = jax.nn.softmax(logits, axis=-1)
        histories.append(attention)
        attended = jnp.einsum("bhij,bjhd->bihd", attention, value).reshape(
            len(labels), 4, config.embedding_dim
        )
        tokens = tokens + attended @ layer["output"]
        normalized = _layer_norm(tokens)
        tokens = tokens + jax.nn.gelu(normalized @ layer["mlp_in"]) @ layer[
            "mlp_out"
        ]
    pooled = jnp.mean(_layer_norm(tokens), axis=1)
    return pooled @ params["coefficient_head"] + params["coefficient_bias"]


def intertwiner_loop_frame_contributions(
    params: dict[str, object],
    config: IntertwinerLoopViTConfig,
) -> tuple[jax.Array, ...]:
    """Return every frame recoupled into the canonical singlet basis."""

    selected = jnp.asarray(pair_singlet_state_indices(config.kind))
    contributions = []
    for order in config.frame_orders:
        coefficients = intertwiner_loop_frame_coefficients(params, config, order)
        recoupling = jnp.asarray(four_cell_pairing_recoupling(config.kind, order))
        contributions.append(recoupling[:, selected] @ coefficients)
    return tuple(contributions)


def intertwiner_loop_canonical_coefficients(
    params: dict[str, object],
    config: IntertwinerLoopViTConfig,
) -> jax.Array:
    """Coherently sum all virtual-loop frames in canonical coordinates."""

    contributions = intertwiner_loop_frame_contributions(params, config)
    return jnp.sum(jnp.stack(contributions, axis=0), axis=0) / np.sqrt(
        len(contributions)
    )


def project_intertwiner_loop_coefficients(
    canonical_coefficients: jax.Array,
    coefficient_sector_basis: jax.Array,
) -> jax.Array:
    """Project canonical coefficients into an exact registered sector basis."""

    vector = jnp.asarray(canonical_coefficients)
    basis = jnp.asarray(coefficient_sector_basis)
    if vector.ndim != 1 or basis.ndim != 2 or basis.shape[0] != vector.shape[0]:
        raise ValueError("canonical coefficients and sector basis are incompatible")
    return basis.conj().T @ vector


def intertwiner_loop_parameter_count(params: object) -> int:
    """Count real trainable scalar entries."""

    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))


__all__ = [
    "DEFAULT_PAIRING_FRAMES",
    "IntertwinerLoopViTConfig",
    "init_intertwiner_loop_vit",
    "intertwiner_loop_canonical_coefficients",
    "intertwiner_loop_frame_coefficients",
    "intertwiner_loop_frame_contributions",
    "intertwiner_loop_labels",
    "intertwiner_loop_parameter_count",
    "project_intertwiner_loop_coefficients",
]
