r"""Physical block-Jacobi preconditioners for real-parameter quantum Fisher solves.

The input score is the *centered* complex log-derivative for every real
parameter,

.. math::

   \widetilde O_{s,p} = \partial_{\theta_p}\log\psi(x_s)
       - \mathbb E[\partial_{\theta_p}\log\psi],

so the sampled real-parameter quantum Fisher is

.. math::

   S_{pq} = 2\,\mathrm{Re}\sum_s w_s
       \widetilde O^*_{s,p}\widetilde O_{s,q}.

This module never materializes ``S``.  The diagonal builder stores one inverse
variance per parameter.  The block builder materializes only explicitly
requested, mutually disjoint small blocks and applies their inverses in a
matrix-free global solve.

For the sparse causal SU(2) model, :func:`sparse_causal_head_block_specs`
groups the exact ``q_op=2`` endpoint carrier by ``(site, delta_q)`` and jointly
whitens its probability/phase coordinates.  This is a change of coordinates
for the SR linear solve, not an additional model input or representational
channel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np


PathEntry: TypeAlias = str | int
ParameterPath: TypeAlias = tuple[PathEntry, ...]


@dataclass(frozen=True)
class FisherBlockSpec:
    """A family of disjoint Fisher blocks in one or more parameter leaves.

    ``paths`` names parameter leaves.  All members must have the same sizes on
    ``independent_axes``.  Every index tuple along those axes defines one
    independent Jacobi block; all remaining coordinates from all member
    leaves are concatenated inside that block.

    For example, two endpoint-head leaves of shape ``(N, 3, width)`` with
    ``independent_axes=(0, 1)`` produce ``N * 3`` blocks of size
    ``2 * width``.
    """

    name: str
    paths: tuple[ParameterPath, ...]
    independent_axes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a Fisher block needs a non-empty name")
        if not self.paths:
            raise ValueError("a Fisher block needs at least one parameter path")
        normalized_paths = tuple(tuple(path) for path in self.paths)
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError(f"Fisher block {self.name!r} repeats a parameter path")
        if len(set(self.independent_axes)) != len(self.independent_axes):
            raise ValueError(
                f"Fisher block {self.name!r} repeats an independent axis"
            )
        if any(axis < 0 for axis in self.independent_axes):
            raise ValueError("independent axes must be non-negative")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _BuiltFisherBlock:
    name: str
    leaf_indices: tuple[int, ...]
    independent_axes: tuple[int, ...]
    independent_shape: tuple[int, ...]
    member_shapes: tuple[tuple[int, ...], ...]
    member_dimensions: tuple[int, ...]
    inverse: jax.Array

    def tree_flatten(self):
        children = (self.inverse,)
        metadata = (
            self.name,
            self.leaf_indices,
            self.independent_axes,
            self.independent_shape,
            self.member_shapes,
            self.member_dimensions,
        )
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        (
            name,
            leaf_indices,
            independent_axes,
            independent_shape,
            member_shapes,
            member_dimensions,
        ) = metadata
        (inverse,) = children
        return cls(
            name=name,
            leaf_indices=leaf_indices,
            independent_axes=independent_axes,
            independent_shape=independent_shape,
            member_shapes=member_shapes,
            member_dimensions=member_dimensions,
            inverse=inverse,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FisherPreconditioner:
    """Matrix-free global application of diagonal and small-block inverses.

    Instances are constructed by :func:`build_fisher_preconditioner`.  The
    stored ``inverse_diagonal`` follows the parameter pytree.  A leaf covered
    by a block uses the corresponding small inverse instead; uncovered leaves
    use the diagonal inverse.
    """

    inverse_diagonal: object
    blocks: tuple[_BuiltFisherBlock, ...]
    parameter_paths: tuple[ParameterPath, ...]
    parameter_shapes: tuple[tuple[int, ...], ...]
    damping: float
    sample_count: int

    def tree_flatten(self):
        children = (self.inverse_diagonal, self.blocks)
        metadata = (
            self.parameter_paths,
            self.parameter_shapes,
            self.damping,
            self.sample_count,
        )
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        parameter_paths, parameter_shapes, damping, sample_count = metadata
        inverse_diagonal, blocks = children
        return cls(
            inverse_diagonal=inverse_diagonal,
            blocks=blocks,
            parameter_paths=parameter_paths,
            parameter_shapes=parameter_shapes,
            damping=damping,
            sample_count=sample_count,
        )

    def apply(self, vector: object) -> object:
        """Apply the approximate ``(S + damping I)^-1`` to a real pytree."""

        vector_leaves, vector_structure = jax.tree_util.tree_flatten(vector)
        diagonal_leaves, diagonal_structure = jax.tree_util.tree_flatten(
            self.inverse_diagonal
        )
        if vector_structure != diagonal_structure:
            raise ValueError("preconditioner and vector pytree structures differ")
        if tuple(tuple(np.shape(leaf)) for leaf in vector_leaves) != self.parameter_shapes:
            raise ValueError("preconditioner and vector leaf shapes differ")

        output = [inverse * value for inverse, value in zip(diagonal_leaves, vector_leaves)]
        for block in self.blocks:
            views = [
                _parameter_block_view(
                    vector_leaves[index], block.independent_axes
                )
                for index in block.leaf_indices
            ]
            joined = jnp.concatenate(views, axis=-1)
            whitened = jnp.einsum("...ij,...j->...i", block.inverse, joined)
            start = 0
            for index, shape, dimension in zip(
                block.leaf_indices,
                block.member_shapes,
                block.member_dimensions,
            ):
                member = whitened[..., start : start + dimension]
                output[index] = _parameter_block_unview(
                    member, shape, block.independent_axes
                )
                start += dimension
        return jax.tree_util.tree_unflatten(vector_structure, output)

    __call__ = apply

    @property
    def block_sizes(self) -> dict[str, tuple[int, int]]:
        """Map block name to ``(number_of_blocks, coordinates_per_block)``."""

        result: dict[str, tuple[int, int]] = {}
        for block in self.blocks:
            count = int(np.prod(block.independent_shape, dtype=np.int64))
            result[block.name] = (count, int(sum(block.member_dimensions)))
        return result

    @property
    def stored_matrix_elements(self) -> int:
        """Number of scalar entries stored by the Jacobi approximation."""

        diagonal_count = sum(
            int(np.size(leaf))
            for leaf in jax.tree_util.tree_leaves(self.inverse_diagonal)
        )
        block_count = sum(int(block.inverse.size) for block in self.blocks)
        covered_count = sum(
            int(np.prod(shape, dtype=np.int64))
            for block in self.blocks
            for shape in block.member_shapes
        )
        # Diagonal entries for block-covered leaves are retained only to keep a
        # simple pytree template, but are not part of the effective storage
        # requirement and are never read by apply().
        return diagonal_count - covered_count + block_count


@dataclass(frozen=True)
class ChunkedFisherDiagnostics:
    """Bounded-memory construction diagnostics for a score accumulator."""

    sample_count: int
    chunk_count: int
    maximum_chunk_size: int
    parameter_count: int
    maximum_score_chunk_shape: tuple[int, int]
    maximum_real_imag_jacobian_shape: tuple[int, int, int]
    maximum_score_chunk_elements: int
    maximum_score_chunk_bytes: int
    estimated_peak_score_workspace_bytes: int
    largest_block_dimension: int
    stored_matrix_elements: int


def _canonical_key(key: object) -> PathEntry:
    if isinstance(key, jax.tree_util.DictKey):
        value = key.key
    elif isinstance(key, jax.tree_util.SequenceKey):
        value = key.idx
    elif isinstance(key, jax.tree_util.GetAttrKey):
        value = key.name
    elif isinstance(key, jax.tree_util.FlattenedIndexKey):
        value = key.key
    else:  # Defensive support for future JAX key-path entries.
        value = getattr(key, "key", getattr(key, "idx", str(key)))
    if isinstance(value, Integral):
        return int(value)
    return str(value)


def _tree_paths_and_leaves(tree: object) -> tuple[
    tuple[ParameterPath, ...], list[object], jax.tree_util.PyTreeDef
]:
    path_leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    paths = tuple(
        tuple(_canonical_key(entry) for entry in path) for path, _ in path_leaves
    )
    return paths, [leaf for _, leaf in path_leaves], structure


def _normalized_weights(
    sample_count: int, sample_weights: jax.Array | None
) -> jax.Array:
    if sample_weights is None:
        return jnp.full((sample_count,), 1.0 / sample_count, dtype=jnp.float64)
    weights = jnp.asarray(sample_weights, dtype=jnp.float64)
    if weights.shape != (sample_count,):
        raise ValueError("sample weights must have shape (sample_count,)")
    host_weights = np.asarray(weights)
    if not np.all(np.isfinite(host_weights)):
        raise ValueError("sample weights must be finite")
    if np.any(host_weights < 0.0):
        raise ValueError("sample weights must be non-negative")
    total = float(np.sum(host_weights))
    if not total > 0.0:
        raise ValueError("sample weights must have positive total weight")
    return weights / total


def _score_block_view(
    score: jax.Array, independent_axes: tuple[int, ...]
) -> jax.Array:
    rank = score.ndim - 1
    complementary_axes = tuple(
        axis for axis in range(rank) if axis not in independent_axes
    )
    permutation = (0,) + tuple(axis + 1 for axis in independent_axes) + tuple(
        axis + 1 for axis in complementary_axes
    )
    transposed = jnp.transpose(score, permutation)
    independent_shape = tuple(score.shape[axis + 1] for axis in independent_axes)
    dimension = int(
        np.prod(
            [score.shape[axis + 1] for axis in complementary_axes],
            dtype=np.int64,
        )
    )
    return jnp.reshape(transposed, (score.shape[0],) + independent_shape + (dimension,))


def _parameter_block_view(
    value: jax.Array, independent_axes: tuple[int, ...]
) -> jax.Array:
    rank = value.ndim
    complementary_axes = tuple(
        axis for axis in range(rank) if axis not in independent_axes
    )
    permutation = independent_axes + complementary_axes
    transposed = jnp.transpose(value, permutation)
    independent_shape = tuple(value.shape[axis] for axis in independent_axes)
    dimension = int(
        np.prod([value.shape[axis] for axis in complementary_axes], dtype=np.int64)
    )
    return jnp.reshape(transposed, independent_shape + (dimension,))


def _parameter_block_unview(
    view: jax.Array,
    original_shape: tuple[int, ...],
    independent_axes: tuple[int, ...],
) -> jax.Array:
    rank = len(original_shape)
    complementary_axes = tuple(
        axis for axis in range(rank) if axis not in independent_axes
    )
    permutation = independent_axes + complementary_axes
    transposed_shape = tuple(original_shape[axis] for axis in permutation)
    transposed = jnp.reshape(view, transposed_shape)
    inverse_permutation = tuple(int(axis) for axis in np.argsort(permutation))
    return jnp.transpose(transposed, inverse_permutation)


def _validate_inputs(
    params: object, centered_scores: object
) -> tuple[
    tuple[ParameterPath, ...],
    list[jax.Array],
    list[jax.Array],
    jax.tree_util.PyTreeDef,
    int,
]:
    paths, parameter_leaves_raw, parameter_structure = _tree_paths_and_leaves(params)
    score_paths, score_leaves_raw, score_structure = _tree_paths_and_leaves(
        centered_scores
    )
    if parameter_structure != score_structure or paths != score_paths:
        raise ValueError("parameter and score pytree structures differ")
    if not parameter_leaves_raw:
        raise ValueError("the parameter pytree may not be empty")

    parameter_leaves = [jnp.asarray(leaf) for leaf in parameter_leaves_raw]
    score_leaves = [jnp.asarray(leaf) for leaf in score_leaves_raw]
    sample_count: int | None = None
    for path, parameter, score in zip(paths, parameter_leaves, score_leaves):
        if not jnp.issubdtype(parameter.dtype, jnp.floating):
            raise TypeError(f"parameter leaf {path!r} must have a real floating dtype")
        if score.ndim != parameter.ndim + 1 or score.shape[1:] != parameter.shape:
            raise ValueError(
                f"score leaf {path!r} must have shape "
                f"(sample_count, {parameter.shape!r})"
            )
        if sample_count is None:
            sample_count = int(score.shape[0])
        elif score.shape[0] != sample_count:
            raise ValueError("all score leaves must have the same sample count")
    assert sample_count is not None
    if sample_count < 1:
        raise ValueError("at least one score sample is required")
    return (
        paths,
        parameter_leaves,
        score_leaves,
        parameter_structure,
        sample_count,
    )


def build_fisher_preconditioner(
    params: object,
    centered_scores: object,
    *,
    damping: float,
    block_specs: tuple[FisherBlockSpec, ...] = (),
    sample_weights: jax.Array | None = None,
) -> FisherPreconditioner:
    """Build a diagonal or physical block-Jacobi Fisher preconditioner.

    Args:
        params: Real parameter pytree.  Values provide shape and dtype only.
        centered_scores: Matching pytree with a leading sample axis containing
            centered complex log-derivatives.
        damping: Positive scalar ``lambda`` in ``S + lambda I``.
        block_specs: Mutually disjoint small-block families.  Leaves not named
            here retain diagonal Fisher whitening.
        sample_weights: Optional non-negative sample weights.  They are
            normalized internally.

    Returns:
        A reusable matrix-free preconditioner.  No global parameter-square
        matrix is constructed or stored.
    """

    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")
    (
        paths,
        parameter_leaves,
        score_leaves,
        parameter_structure,
        sample_count,
    ) = _validate_inputs(params, centered_scores)
    weights = _normalized_weights(sample_count, sample_weights)

    inverse_diagonal_leaves = []
    for parameter, score in zip(parameter_leaves, score_leaves):
        diagonal = 2.0 * jnp.real(
            jnp.sum(
                weights.reshape((sample_count,) + (1,) * parameter.ndim)
                * jnp.conj(score)
                * score,
                axis=0,
            )
        )
        # Roundoff can only make this non-negative quantity slightly negative.
        diagonal = jnp.maximum(diagonal, 0.0)
        inverse_diagonal_leaves.append(1.0 / (diagonal + damping))

    path_to_index = {path: index for index, path in enumerate(paths)}
    covered_indices: set[int] = set()
    built_blocks: list[_BuiltFisherBlock] = []
    for spec in block_specs:
        try:
            indices = tuple(path_to_index[tuple(path)] for path in spec.paths)
        except KeyError as error:
            raise ValueError(
                f"Fisher block {spec.name!r} names missing parameter path "
                f"{error.args[0]!r}"
            ) from error
        overlap = covered_indices.intersection(indices)
        if overlap:
            overlap_paths = tuple(paths[index] for index in sorted(overlap))
            raise ValueError(
                f"Fisher blocks may not overlap; repeated paths: {overlap_paths!r}"
            )
        covered_indices.update(indices)

        member_shapes = tuple(tuple(parameter_leaves[index].shape) for index in indices)
        ranks = tuple(len(shape) for shape in member_shapes)
        if any(axis >= rank for rank in ranks for axis in spec.independent_axes):
            raise ValueError(
                f"Fisher block {spec.name!r} has an out-of-range independent axis"
            )
        independent_shapes = tuple(
            tuple(shape[axis] for axis in spec.independent_axes)
            for shape in member_shapes
        )
        if any(shape != independent_shapes[0] for shape in independent_shapes[1:]):
            raise ValueError(
                f"Fisher block {spec.name!r} members disagree on independent sizes"
            )

        views = [
            _score_block_view(score_leaves[index], spec.independent_axes)
            for index in indices
        ]
        member_dimensions = tuple(int(view.shape[-1]) for view in views)
        joined = jnp.concatenate(views, axis=-1)
        fisher = 2.0 * jnp.real(
            jnp.einsum(
                "s...i,s...j,s->...ij",
                jnp.conj(joined),
                joined,
                weights,
            )
        )
        fisher = 0.5 * (fisher + jnp.swapaxes(fisher, -1, -2))
        eigenvalues, eigenvectors = jnp.linalg.eigh(fisher)
        # The exact matrix is PSD.  Clipping tiny negative roundoff before
        # adding lambda makes zero and near-null score directions harmless.
        inverse_eigenvalues = 1.0 / (jnp.maximum(eigenvalues, 0.0) + damping)
        inverse = jnp.einsum(
            "...ik,...k,...jk->...ij",
            eigenvectors,
            inverse_eigenvalues,
            eigenvectors,
        )
        built_blocks.append(
            _BuiltFisherBlock(
                name=spec.name,
                leaf_indices=indices,
                independent_axes=spec.independent_axes,
                independent_shape=independent_shapes[0],
                member_shapes=member_shapes,
                member_dimensions=member_dimensions,
                inverse=inverse,
            )
        )

    inverse_diagonal = jax.tree_util.tree_unflatten(
        parameter_structure, inverse_diagonal_leaves
    )
    return FisherPreconditioner(
        inverse_diagonal=inverse_diagonal,
        blocks=tuple(built_blocks),
        parameter_paths=paths,
        parameter_shapes=tuple(tuple(leaf.shape) for leaf in parameter_leaves),
        damping=float(damping),
        sample_count=sample_count,
    )


def build_diagonal_fisher_preconditioner(
    params: object,
    centered_scores: object,
    *,
    damping: float,
    sample_weights: jax.Array | None = None,
) -> FisherPreconditioner:
    """Build the stable diagonal ``(diag(S) + lambda)^-1`` preconditioner."""

    return build_fisher_preconditioner(
        params,
        centered_scores,
        damping=damping,
        sample_weights=sample_weights,
    )


def _numpy_parameter_block_view(
    value: np.ndarray, independent_axes: tuple[int, ...]
) -> np.ndarray:
    complementary_axes = tuple(
        axis for axis in range(value.ndim) if axis not in independent_axes
    )
    permutation = independent_axes + complementary_axes
    transposed = np.transpose(value, permutation)
    independent_shape = tuple(value.shape[axis] for axis in independent_axes)
    dimension = int(
        np.prod([value.shape[axis] for axis in complementary_axes], dtype=np.int64)
    )
    return np.reshape(transposed, independent_shape + (dimension,))


def _numpy_score_block_view(
    score: np.ndarray, independent_axes: tuple[int, ...]
) -> np.ndarray:
    rank = score.ndim - 1
    complementary_axes = tuple(
        axis for axis in range(rank) if axis not in independent_axes
    )
    permutation = (0,) + tuple(axis + 1 for axis in independent_axes) + tuple(
        axis + 1 for axis in complementary_axes
    )
    transposed = np.transpose(score, permutation)
    independent_shape = tuple(score.shape[axis + 1] for axis in independent_axes)
    dimension = int(
        np.prod(
            [score.shape[axis + 1] for axis in complementary_axes],
            dtype=np.int64,
        )
    )
    return np.reshape(
        transposed, (score.shape[0],) + independent_shape + (dimension,)
    )


def build_chunked_fisher_preconditioner(
    params: object,
    score_weight_chunks: Iterable[tuple[object, jax.Array]],
    *,
    damping: float,
    block_specs: tuple[FisherBlockSpec, ...] = (),
) -> tuple[FisherPreconditioner, ChunkedFisherDiagnostics]:
    """Accumulate a centered Fisher preconditioner without storing all scores.

    Each iterable entry contains an *uncentered* complex score pytree with a
    leading chunk axis and its non-negative sample weights.  The accumulator
    keeps only ``O(parameter_count)`` first/diagonal second moments and the
    explicitly requested small block Gram matrices.  It never allocates a
    global parameter-square Fisher matrix or a full ``sample x parameter``
    score matrix.
    """

    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")
    paths, parameter_leaves_raw, parameter_structure = _tree_paths_and_leaves(params)
    if not parameter_leaves_raw:
        raise ValueError("the parameter pytree may not be empty")
    parameter_leaves = [np.asarray(leaf) for leaf in parameter_leaves_raw]
    for path, parameter in zip(paths, parameter_leaves):
        if not np.issubdtype(parameter.dtype, np.floating):
            raise TypeError(f"parameter leaf {path!r} must have a real floating dtype")
    parameter_shapes = tuple(tuple(leaf.shape) for leaf in parameter_leaves)
    parameter_count = int(sum(leaf.size for leaf in parameter_leaves))
    path_to_index = {path: index for index, path in enumerate(paths)}

    block_layouts = []
    covered_indices: set[int] = set()
    for spec in block_specs:
        try:
            indices = tuple(path_to_index[tuple(path)] for path in spec.paths)
        except KeyError as error:
            raise ValueError(
                f"Fisher block {spec.name!r} names missing parameter path "
                f"{error.args[0]!r}"
            ) from error
        overlap = covered_indices.intersection(indices)
        if overlap:
            overlap_paths = tuple(paths[index] for index in sorted(overlap))
            raise ValueError(
                f"Fisher blocks may not overlap; repeated paths: {overlap_paths!r}"
            )
        covered_indices.update(indices)
        member_shapes = tuple(parameter_shapes[index] for index in indices)
        ranks = tuple(len(shape) for shape in member_shapes)
        if any(axis >= rank for rank in ranks for axis in spec.independent_axes):
            raise ValueError(
                f"Fisher block {spec.name!r} has an out-of-range independent axis"
            )
        independent_shapes = tuple(
            tuple(shape[axis] for axis in spec.independent_axes)
            for shape in member_shapes
        )
        if any(shape != independent_shapes[0] for shape in independent_shapes[1:]):
            raise ValueError(
                f"Fisher block {spec.name!r} members disagree on independent sizes"
            )
        dimensions = tuple(
            int(
                np.prod(
                    [
                        size
                        for axis, size in enumerate(shape)
                        if axis not in spec.independent_axes
                    ],
                    dtype=np.int64,
                )
            )
            for shape in member_shapes
        )
        block_dimension = int(sum(dimensions))
        raw_gram = np.zeros(
            independent_shapes[0] + (block_dimension, block_dimension),
            dtype=np.complex128,
        )
        block_layouts.append(
            {
                "spec": spec,
                "indices": indices,
                "member_shapes": member_shapes,
                "independent_shape": independent_shapes[0],
                "dimensions": dimensions,
                "raw_gram": raw_gram,
            }
        )

    raw_means = [np.zeros(shape, dtype=np.complex128) for shape in parameter_shapes]
    raw_diagonals = [np.zeros(shape, dtype=np.float64) for shape in parameter_shapes]
    total_weight = 0.0
    sample_count = 0
    chunk_count = 0
    maximum_chunk_size = 0
    maximum_chunk_bytes = 0
    for score_tree, chunk_weights in score_weight_chunks:
        score_paths, score_leaves_raw, score_structure = _tree_paths_and_leaves(
            score_tree
        )
        if score_structure != parameter_structure or score_paths != paths:
            raise ValueError("parameter and chunk score pytree structures differ")
        score_leaves = [np.asarray(leaf) for leaf in score_leaves_raw]
        if not score_leaves:
            raise ValueError("a score chunk may not be empty")
        chunk_size = int(score_leaves[0].shape[0])
        if chunk_size < 1:
            raise ValueError("score chunks must contain at least one sample")
        for path, parameter, score in zip(paths, parameter_leaves, score_leaves):
            if score.shape != (chunk_size,) + parameter.shape:
                raise ValueError(
                    f"score leaf {path!r} has shape {score.shape!r}, expected "
                    f"{(chunk_size,) + parameter.shape!r}"
                )
            if not np.all(np.isfinite(score)):
                raise ValueError(f"score leaf {path!r} contains a non-finite value")
        weights = np.asarray(chunk_weights, dtype=np.float64)
        if weights.shape != (chunk_size,):
            raise ValueError("chunk weights must have shape (chunk_size,)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("chunk weights must be finite and non-negative")
        chunk_weight = float(np.sum(weights))
        total_weight += chunk_weight
        sample_count += chunk_size
        chunk_count += 1
        maximum_chunk_size = max(maximum_chunk_size, chunk_size)
        maximum_chunk_bytes = max(
            maximum_chunk_bytes,
            int(sum(score.nbytes for score in score_leaves)),
        )
        for index, score in enumerate(score_leaves):
            reshaped_weights = weights.reshape(
                (chunk_size,) + (1,) * parameter_leaves[index].ndim
            )
            raw_means[index] += np.sum(reshaped_weights * score, axis=0)
            raw_diagonals[index] += np.sum(
                reshaped_weights * np.abs(score) ** 2, axis=0
            )
        for layout in block_layouts:
            views = [
                _numpy_score_block_view(
                    score_leaves[index], layout["spec"].independent_axes
                )
                for index in layout["indices"]
            ]
            joined = np.concatenate(views, axis=-1)
            layout["raw_gram"] += np.einsum(
                "s...i,s...j,s->...ij",
                np.conj(joined),
                joined,
                weights,
                optimize=True,
            )
        # Materialize this chunk's sufficient statistics before the producer
        # yields another device buffer.  No score chunk is retained here.
        del score_leaves

    if chunk_count < 1:
        raise ValueError("at least one score chunk is required")
    if not total_weight > 0.0:
        raise ValueError("score chunks must have positive total weight")
    means = [value / total_weight for value in raw_means]
    inverse_diagonal_leaves = []
    for raw_diagonal, mean in zip(raw_diagonals, means):
        covariance_diagonal = 2.0 * (
            raw_diagonal / total_weight - np.abs(mean) ** 2
        )
        covariance_diagonal = np.maximum(covariance_diagonal, 0.0)
        inverse_diagonal_leaves.append(
            jnp.asarray(1.0 / (covariance_diagonal + damping))
        )

    built_blocks = []
    largest_block_dimension = 0
    for layout in block_layouts:
        spec = layout["spec"]
        mean_views = [
            _numpy_parameter_block_view(means[index], spec.independent_axes)
            for index in layout["indices"]
        ]
        joined_mean = np.concatenate(mean_views, axis=-1)
        covariance = layout["raw_gram"] / total_weight - np.einsum(
            "...i,...j->...ij", np.conj(joined_mean), joined_mean
        )
        fisher = 2.0 * np.real(covariance)
        fisher = 0.5 * (fisher + np.swapaxes(fisher, -1, -2))
        eigenvalues, eigenvectors = np.linalg.eigh(fisher)
        inverse_eigenvalues = 1.0 / (np.maximum(eigenvalues, 0.0) + damping)
        inverse = np.einsum(
            "...ik,...k,...jk->...ij",
            eigenvectors,
            inverse_eigenvalues,
            eigenvectors,
            optimize=True,
        )
        block_dimension = int(sum(layout["dimensions"]))
        largest_block_dimension = max(largest_block_dimension, block_dimension)
        built_blocks.append(
            _BuiltFisherBlock(
                name=spec.name,
                leaf_indices=layout["indices"],
                independent_axes=spec.independent_axes,
                independent_shape=layout["independent_shape"],
                member_shapes=layout["member_shapes"],
                member_dimensions=layout["dimensions"],
                inverse=jnp.asarray(inverse),
            )
        )
    inverse_diagonal = jax.tree_util.tree_unflatten(
        parameter_structure, inverse_diagonal_leaves
    )
    preconditioner = FisherPreconditioner(
        inverse_diagonal=inverse_diagonal,
        blocks=tuple(built_blocks),
        parameter_paths=paths,
        parameter_shapes=parameter_shapes,
        damping=float(damping),
        sample_count=sample_count,
    )
    diagnostics = ChunkedFisherDiagnostics(
        sample_count=sample_count,
        chunk_count=chunk_count,
        maximum_chunk_size=maximum_chunk_size,
        parameter_count=parameter_count,
        maximum_score_chunk_shape=(maximum_chunk_size, parameter_count),
        maximum_real_imag_jacobian_shape=(
            maximum_chunk_size,
            2,
            parameter_count,
        ),
        maximum_score_chunk_elements=maximum_chunk_size * parameter_count,
        maximum_score_chunk_bytes=maximum_chunk_bytes,
        # ``log_derivative_scores`` constructs a real/imag Jacobian and its
        # complex output. They have equal byte counts for standard dtypes and
        # may coexist transiently. Model reverse-pass activations are backend
        # dependent and deliberately not hidden inside this estimate.
        estimated_peak_score_workspace_bytes=2 * maximum_chunk_bytes,
        largest_block_dimension=largest_block_dimension,
        stored_matrix_elements=preconditioner.stored_matrix_elements,
    )
    return preconditioner, diagnostics


def sparse_causal_head_block_specs(params: object) -> tuple[FisherBlockSpec, ...]:
    """Return the physical sparse-head block layout for ``params``.

    The endpoint probability and phase heads are paired at every
    ``(site, delta_q)`` carrier coordinate.  Closed and hidden scalar
    probability/phase heads form separate blocks.  The function only
    describes an existing parameterization; it creates no parameters and
    reads no lattice labels or samples.
    """

    paths, _, _ = _tree_paths_and_leaves(params)
    available = set(paths)

    def unique_suffix(name: str) -> ParameterPath:
        matches = tuple(path for path in available if path and path[-1] == name)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one sparse causal head named {name!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    probability_endpoint = unique_suffix("probability_endpoint_head")
    phase_endpoint = unique_suffix("phase_endpoint_head")
    probability_closed = unique_suffix("probability_closed_head")
    phase_closed = unique_suffix("phase_closed_head")
    probability_hidden = unique_suffix("probability_hidden_head")
    phase_hidden = unique_suffix("phase_hidden_head")
    return (
        FisherBlockSpec(
            name="qop2_endpoint_site_delta",
            paths=(probability_endpoint, phase_endpoint),
            independent_axes=(0, 1),
        ),
        FisherBlockSpec(
            name="closed_scalar_heads",
            paths=(probability_closed, phase_closed),
        ),
        FisherBlockSpec(
            name="hidden_scalar_heads",
            paths=(probability_hidden, phase_hidden),
        ),
    )


def exact_carrier_decoder_block_specs(
    params: object,
) -> tuple[FisherBlockSpec, ...]:
    """Return endpoint-``delta_q`` and per-decision exact-carrier blocks.

    Exact-carrier endpoint heads have shape ``(site, 3, context)``; the middle
    axis is the doubled ``delta_q=(-2,0,+2)`` channel. Closed heads have shape
    ``(decision, context)``. Probability and phase roles are whitened jointly
    inside each physical carrier coordinate.
    """

    paths, leaves, _ = _tree_paths_and_leaves(params)
    path_to_shape = {
        path: tuple(np.shape(leaf)) for path, leaf in zip(paths, leaves)
    }

    def unique_suffix(name: str) -> ParameterPath:
        matches = tuple(path for path in paths if path and path[-1] == name)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one exact-carrier head named {name!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    probability_endpoint = unique_suffix("probability_endpoint_head")
    phase_endpoint = unique_suffix("phase_endpoint_head")
    probability_closed = unique_suffix("probability_closed_head")
    phase_closed = unique_suffix("phase_closed_head")
    if path_to_shape[probability_endpoint] != path_to_shape[phase_endpoint]:
        raise ValueError("exact-carrier endpoint probability/phase shapes differ")
    if path_to_shape[probability_closed] != path_to_shape[phase_closed]:
        raise ValueError("exact-carrier closed probability/phase shapes differ")
    if len(path_to_shape[probability_endpoint]) != 3:
        raise ValueError("exact-carrier endpoint heads must have rank three")
    if path_to_shape[probability_endpoint][1] != 3:
        raise ValueError("exact-carrier endpoint heads require three delta_q channels")
    if len(path_to_shape[probability_closed]) != 2:
        raise ValueError("exact-carrier closed heads must have rank two")
    return (
        FisherBlockSpec(
            name="exact_qop2_endpoint_site_delta",
            paths=(probability_endpoint, phase_endpoint),
            independent_axes=(0, 1),
        ),
        FisherBlockSpec(
            name="exact_closed_decision",
            paths=(probability_closed, phase_closed),
            independent_axes=(0,),
        ),
    )


def free_k2_decoder_block_specs(
    params: object,
) -> tuple[FisherBlockSpec, ...]:
    """Return matched probability/phase blocks for free K2 histories.

    Each row is one graph-blind K1 or ordered K2 history group.  Jointly
    whitening its probability and phase context gives the same number and
    size of blocks as the N=10 exact-carrier decoder while using no physical
    recoupling coefficient.
    """

    paths, leaves, _ = _tree_paths_and_leaves(params)
    path_to_shape = {
        path: tuple(np.shape(leaf)) for path, leaf in zip(paths, leaves)
    }

    def unique_suffix(name: str) -> ParameterPath:
        matches = tuple(path for path in paths if path and path[-1] == name)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one free-K2 head named {name!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    probability = unique_suffix("probability_history_head")
    phase = unique_suffix("phase_history_head")
    if path_to_shape[probability] != path_to_shape[phase]:
        raise ValueError("free-K2 probability/phase head shapes differ")
    if len(path_to_shape[probability]) != 2:
        raise ValueError("free-K2 history heads must have rank two")
    return (
        FisherBlockSpec(
            name="free_k2_history_group",
            paths=(probability, phase),
            independent_axes=(0,),
        ),
    )


def k2_pair_carrier_decoder_block_specs(
    params: object,
) -> tuple[FisherBlockSpec, ...]:
    """Return one joint probability/phase block per physical K1/K2 group.

    The two heads have shape ``(feature_group, context)``.  At N=10 there
    are 39 groups, exactly matching the free-K2 control's block sizes while
    retaining the fixed SU(2) carrier and spatial-moment interpretation.
    """

    paths, leaves, _ = _tree_paths_and_leaves(params)
    path_to_shape = {
        path: tuple(np.shape(leaf)) for path, leaf in zip(paths, leaves)
    }

    def unique_suffix(name: str) -> ParameterPath:
        matches = tuple(path for path in paths if path and path[-1] == name)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one K2 pair-carrier head named {name!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    probability = unique_suffix("probability_physical_head")
    phase = unique_suffix("phase_physical_head")
    if path_to_shape[probability] != path_to_shape[phase]:
        raise ValueError("K2 pair-carrier probability/phase head shapes differ")
    if len(path_to_shape[probability]) != 2:
        raise ValueError("K2 pair-carrier physical heads must have rank two")
    return (
        FisherBlockSpec(
            name="k2_pair_physical_group",
            paths=(probability, phase),
            independent_axes=(0,),
        ),
    )


def physical_carrier_head_block_specs(
    params: object,
) -> tuple[FisherBlockSpec, ...]:
    """Dispatch to the exact- or learned-sparse carrier head layout."""

    paths, _, _ = _tree_paths_and_leaves(params)
    names = {path[-1] for path in paths if path}
    hidden = {"probability_hidden_head", "phase_hidden_head"}
    if hidden <= names:
        return sparse_causal_head_block_specs(params)
    return exact_carrier_decoder_block_specs(params)


__all__ = [
    "ChunkedFisherDiagnostics",
    "FisherBlockSpec",
    "FisherPreconditioner",
    "ParameterPath",
    "build_chunked_fisher_preconditioner",
    "build_diagonal_fisher_preconditioner",
    "build_fisher_preconditioner",
    "exact_carrier_decoder_block_specs",
    "free_k2_decoder_block_specs",
    "k2_pair_carrier_decoder_block_specs",
    "physical_carrier_head_block_specs",
    "sparse_causal_head_block_specs",
]
