"""Canonical SU(2) edge environments for a *raw* invariant tensor tree.

This module has no neural-model or Hamiltonian dependency. Each reduced
kernel has indices ``(a_out,a_left,a_right)`` and a doubled-spin key
``(q_left,q_right,q_out)``. Fixed, orthonormal CG tensors carry magnetic
indices. The root is a singlet. A Reynolds sum or additive carrier is NOT
one tree and must not be assigned these single-edge rank bounds.

The double layer includes both sides of an edge. A raw tensor SVD is not a
Schmidt spectrum when its virtual basis is nonorthogonal or rank deficient.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import numpy as np

from .fusion_tree import FusionTree, Leaf, TreeNode
from .quantum_numbers import coupling_range, validate_q

SpinBlocks = Mapping[int, np.ndarray]
ReducedKernels = Mapping[tuple[int, int, int], np.ndarray]


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2


def _positive_eigh(matrix: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(_hermitian(matrix))
    scale = max(float(np.max(abs(values), initial=0.)), np.finfo(float).tiny)
    if float(np.min(values, initial=0.)) < -tolerance * scale:
        raise ValueError("nonpositive reduced Gram/environment")
    # Derived double layers are mathematically positive. Roundoff zeros are
    # handled here; this is NOT repair of an estimated physical RDM.
    return np.maximum(values, 0.), vectors


@dataclass(frozen=True)
class ReducedSingletTree:
    """A static raw TTN with charge-blocked leaf metrics and untied kernels.

    ``leaf_grams[cell][q]`` is <L_a|L_b> for one fixed magnetic index.
    The singlet root contracts ``root_vector`` without complex conjugation.
    All arrays and q labels are validated; no magnetic multiplet is cut.
    """

    tree: FusionTree
    leaf_grams: tuple[SpinBlocks, ...]
    kernels: tuple[ReducedKernels, ...]
    root_vector: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.tree, FusionTree):
            raise TypeError("tree must be FusionTree")
        if len(self.leaf_grams) != self.tree.n_leaves or len(self.kernels) != self.tree.n_leaves-1:
            raise ValueError("one leaf metric and one kernel per tree node required")
        widths: dict[TreeNode, dict[int, int]] = {}
        for leaf, blocks in zip(self.tree.leaves, self.leaf_grams, strict=True):
            widths[leaf] = {}
            for q, gram in blocks.items():
                validate_q(q)
                array = np.asarray(gram)
                if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
                    raise ValueError("leaf Gram blocks must be nonempty square matrices")
                if np.any(~np.isfinite(array)) or not np.allclose(array, array.conj().T, atol=1e-13, rtol=1e-13):
                    raise ValueError("finite Hermitian leaf Gram blocks required")
                _positive_eigh(array, 1e-12)
                widths[leaf][q] = array.shape[0]
        for node, blocks in zip(self.tree.internal_nodes, self.kernels, strict=True):
            widths[node] = {}
            for key, tensor in blocks.items():
                if len(key) != 3:
                    raise ValueError("kernel keys must be (q_left,q_right,q_out)")
                ql, qr, qt = key
                for q in key:
                    validate_q(q)
                if qt not in coupling_range(ql, qr):
                    raise ValueError("kernel violates SU(2) triangle or parity")
                array = np.asarray(tensor)
                if ql not in widths[node.left] or qr not in widths[node.right]:
                    raise ValueError("kernel references an absent child charge")
                expected = (widths[node.left][ql], widths[node.right][qr])
                if array.ndim != 3 or array.shape[0] < 1 or array.shape[1:] != expected:
                    raise ValueError("inconsistent reduced tensor widths")
                if np.any(~np.isfinite(array)):
                    raise ValueError("finite reduced tensors required")
                if qt in widths[node] and widths[node][qt] != array.shape[0]:
                    raise ValueError("output multiplicity must agree within a q block")
                widths[node][qt] = array.shape[0]
        root = np.asarray(self.root_vector)
        if set(widths[self.tree.root]) != {0} or root.shape != (widths[self.tree.root][0],):
            raise ValueError("a singlet root and matching reduced root vector required")
        if np.any(~np.isfinite(root)):
            raise ValueError("finite root vector required")


@dataclass(frozen=True)
class EdgeEnvironments:
    """Reduced subtree G and complementary E with norm=Tr(G E)."""

    upward: Mapping[TreeNode, SpinBlocks]
    downward: Mapping[TreeNode, SpinBlocks]
    norm_squared: float
    maximum_relative_edge_norm_error: float


def reduced_tree_environments(state: ReducedSingletTree) -> EdgeEnvironments:
    """Contract exact double layers, polynomial in retained multiplicities.

    Upward CG norm is one for each fixed parent m. Downward CG sums give
    ``(q_parent+1)/(q_child+1)``; dropping this ratio corrupts p(q).
    E has ket-bra orientation, G has bra-ket orientation, so norm=Tr(G E).
    """
    up: dict[TreeNode, dict[int, np.ndarray]] = {
        leaf: {q: np.asarray(g, dtype=np.complex128).copy() for q, g in blocks.items()}
        for leaf, blocks in zip(state.tree.leaves, state.leaf_grams, strict=True)
    }
    for node, kernels in zip(state.tree.internal_nodes, state.kernels, strict=True):
        up[node] = {}
        for (ql, qr, qt), tensor in kernels.items():
            t = np.asarray(tensor, dtype=np.complex128)
            contribution = np.einsum("alr,buv,lu,rv->ab", t.conj(), t, up[node.left][ql], up[node.right][qr], optimize=True)
            up[node][qt] = up[node].get(qt, np.zeros_like(contribution)) + contribution
        up[node] = {q: _hermitian(g) for q, g in up[node].items()}
    down = {node: {q: np.zeros_like(g) for q, g in blocks.items()} for node, blocks in up.items()}
    root = np.asarray(state.root_vector, dtype=np.complex128)
    down[state.tree.root][0] = np.outer(root, root.conj())
    norm = float(np.trace(up[state.tree.root][0] @ down[state.tree.root][0]).real)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("raw tree has nonpositive or nonfinite norm")
    for node, kernels in reversed(tuple(zip(state.tree.internal_nodes, state.kernels, strict=True))):
        for (ql, qr, qt), tensor in kernels.items():
            t = np.asarray(tensor, dtype=np.complex128)
            parent = down[node][qt]
            left = np.einsum("alr,ab,buv,vr->lu", t, parent, t.conj(), up[node.right][qr], optimize=True)
            right = np.einsum("alr,ab,buv,ul->rv", t, parent, t.conj(), up[node.left][ql], optimize=True)
            down[node.left][ql] += (qt+1)/(ql+1) * left
            down[node.right][qr] += (qt+1)/(qr+1) * right
        down[node.left] = {q: _hermitian(e) for q, e in down[node.left].items()}
        down[node.right] = {q: _hermitian(e) for q, e in down[node.right].items()}
    discrepancies = [abs(sum((q+1)*np.trace(g @ down[node][q]).real for q, g in blocks.items())/norm-1.)
                     for node, blocks in up.items()]
    return EdgeEnvironments(up, down, norm, float(max(discrepancies, default=0.)))


@dataclass(frozen=True)
class CanonicalSpinBlock:
    """One canonical block: sigma eigenvalues include the magnetic factor."""

    q: int
    multiplicity_weights: np.ndarray
    magnetic_eigenvalues: np.ndarray
    metric_rank: int
    gauge: np.ndarray
    gauge_inverse: np.ndarray


@dataclass(frozen=True)
class CanonicalEdgeSpectrum:
    """True normalized raw-tree Schmidt spectrum on one complete subtree."""

    blocks: Mapping[int, CanonicalSpinBlock]
    norm_squared: float
    physical_bond_dimension: int
    entropy: float
    normalized_weight_error: float


def canonical_edge_spectrum(environments: EdgeEnvironments, node: TreeNode,
                            *, relative_tolerance: float = 1e-12) -> CanonicalEdgeSpectrum:
    """Whiten G before diagonalizing E; retain a reversible null-space gauge.

    A zero-norm virtual direction is not a physical Schmidt vector. The
    returned invertible gauge keeps such directions only for reversibility;
    their reported weights are zero and their metric_rank is explicit.
    """
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("positive relative tolerance below one required")
    blocks = {}
    entropy = 0.
    for q in sorted(environments.upward[node]):
        gram = environments.upward[node][q]
        outside = environments.downward[node][q]
        values, vectors = _positive_eigh(gram, relative_tolerance)
        maximum = float(np.max(values, initial=0.))
        active = np.flatnonzero(values > relative_tolerance * maximum)
        null = np.flatnonzero(values <= relative_tolerance * maximum)
        order = np.concatenate((active, null))
        scales = np.ones(len(values))
        scales[:len(active)] = 1/np.sqrt(values[active])
        whitening = vectors[:, order] * scales
        inverse = np.linalg.inv(whitening)
        canonical_outside = _hermitian(inverse @ outside @ inverse.conj().T)
        rotation = np.eye(len(values), dtype=np.complex128)
        weights = np.zeros(len(values))
        if len(active):
            eigenvalues, eigenvectors = _positive_eigh(canonical_outside[:len(active), :len(active)], relative_tolerance)
            selected = np.argsort(eigenvalues)[::-1]
            weights[:len(active)] = eigenvalues[selected]/environments.norm_squared
            rotation[:len(active), :len(active)] = eigenvectors[:, selected]
        gauge = whitening @ rotation
        gauge_inverse = rotation.conj().T @ inverse
        sigma = (q+1)*weights
        positive = weights[weights > 0]
        entropy -= (q+1)*float(np.sum(positive*np.log(positive)))
        blocks[q] = CanonicalSpinBlock(q, sigma, weights, len(active), gauge, gauge_inverse)
    dimension = sum((q+1)*len(block.multiplicity_weights) for q, block in blocks.items())
    return CanonicalEdgeSpectrum(blocks, environments.norm_squared, dimension, entropy,
                                 abs(sum(float(block.multiplicity_weights.sum()) for block in blocks.values())-1.))


def gauge_transform_edge(state: ReducedSingletTree, node_index: int,
                         gauges: Mapping[int, np.ndarray]) -> ReducedSingletTree:
    """Apply L->LZ and K->Z^-1 K on an INTERNAL nonroot edge; exact inverse."""
    if isinstance(node_index, bool) or not isinstance(node_index, Integral):
        raise TypeError("node_index must be integer")
    nodes = state.tree.internal_nodes
    if not 0 <= node_index < len(nodes)-1:
        raise ValueError("need an internal nonroot edge")
    node = nodes[node_index]
    parent_index = next(index for index, candidate in enumerate(nodes) if candidate.left == node or candidate.right == node)
    kernels = [{key: np.asarray(t, dtype=np.complex128).copy() for key, t in blocks.items()} for blocks in state.kernels]
    widths = {key[2]: value.shape[0] for key, value in kernels[node_index].items()}
    if set(gauges) != set(widths):
        raise ValueError("a gauge for every edge q block required")
    inverses = {}
    for q, width in widths.items():
        z = np.asarray(gauges[q], dtype=np.complex128)
        if z.shape != (width, width) or np.any(~np.isfinite(z)):
            raise ValueError("finite square edge gauges required")
        inverses[q] = np.linalg.inv(z)
    for key, tensor in kernels[node_index].items():
        kernels[node_index][key] = np.einsum("ilr,ij->jlr", tensor, gauges[key[2]], optimize=True)
    parent = nodes[parent_index]
    for key, tensor in kernels[parent_index].items():
        if parent.left == node:
            kernels[parent_index][key] = np.einsum("air,ji->ajr", tensor, inverses[key[0]], optimize=True)
        else:
            kernels[parent_index][key] = np.einsum("ali,ji->alj", tensor, inverses[key[1]], optimize=True)
    return ReducedSingletTree(state.tree, state.leaf_grams, tuple(kernels), np.asarray(state.root_vector).copy())


@dataclass(frozen=True)
class MultipletAllocation:
    """Optimal one-edge whole-multiplet compression of a FIXED saved state."""

    copies_by_q: Mapping[int, int]
    physical_bond_dimension: int
    retained_weight: float
    discarded_weight: float


def allocate_multiplets(spectrum: CanonicalEdgeSpectrum, physical_budget: int,
                        *, exact_budget: bool = True) -> MultipletAllocation:
    """Exact deterministic knapsack; a copy of doubled q costs q+1 dimensions.

    Only existing canonical copies may be retained. This does not choose a
    new representation or optimize an ansatz, nor estimate an energy gain.
    """
    return allocate_multiplet_weights({q: block.multiplicity_weights for q, block in spectrum.blocks.items()},
                                     physical_budget, exact_budget=exact_budget)


def allocate_multiplet_weights(weights_by_q: Mapping[int, np.ndarray], physical_budget: int,
                               *, exact_budget: bool = True) -> MultipletAllocation:
    """Allocate copies from already validated normalized physical sigma spectra.

    The caller must establish SU(2) invariance before interpreting these as
    physical Schmidt weights. No RDM repair or symmetry certification occurs
    here. Tiny signed numerical eigenvalues are retained, not clipped.
    """
    if isinstance(physical_budget, bool) or not isinstance(physical_budget, Integral) or physical_budget < 1:
        raise ValueError("physical budget must be a positive integer")
    for q, weights in weights_by_q.items():
        validate_q(q)
        values = np.asarray(weights)
        if values.ndim != 1 or np.any(~np.isfinite(values)) or values.dtype.kind == 'c':
            raise ValueError("finite real multiplicity spectra required")
        if np.any(values < -1e-12) or np.any(np.diff(values) > 1e-12):
            raise ValueError("nonnegative descending multiplicity spectra required")
    if abs(sum(float(np.sum(values)) for values in weights_by_q.values())-1.) > 1e-9:
        raise ValueError("normalized multiplicity spectra required")
    dynamic: dict[int, tuple[float, tuple[int, ...]]] = {0: (0., ())}
    qs = tuple(sorted(weights_by_q))
    for q in qs:
        prefixes = np.r_[0., np.cumsum(weights_by_q[q])]
        updated = {}
        for cost, (weight, counts) in dynamic.items():
            for copies, addition in enumerate(prefixes):
                new_cost = cost + copies*(q+1)
                if new_cost > physical_budget:
                    break
                value = (float(weight+addition), counts+(copies,))
                if new_cost not in updated or value[0] > updated[new_cost][0]:
                    updated[new_cost] = value
        dynamic = updated
    if exact_budget:
        if physical_budget not in dynamic:
            raise ValueError("exact physical budget is not realizable by available multiplets")
        cost = physical_budget
    else:
        cost = max(dynamic, key=lambda value: (dynamic[value][0], -value))
    weight, copies = dynamic[cost]
    return MultipletAllocation(dict(zip(qs, copies, strict=True)), cost, weight, max(0., 1-weight))


def truncate_internal_edge(state: ReducedSingletTree, node_index: int,
                           spectrum: CanonicalEdgeSpectrum,
                           allocation: MultipletAllocation) -> ReducedSingletTree:
    """Apply the canonical physical Schmidt projector on one edge.

    Whole magnetic multiplets are retained. Storage remains padded to the
    original width, so reduced *effective* D does not imply measured speedup.
    No normalization, energy optimization or spatial projection is performed.
    """
    if set(allocation.copies_by_q) != set(spectrum.blocks):
        raise ValueError("allocation and canonical spectrum charges disagree")
    transformed = gauge_transform_edge(state, node_index, {q: block.gauge for q, block in spectrum.blocks.items()})
    kernels = [{key: t.copy() for key, t in blocks.items()} for blocks in transformed.kernels]
    for key, tensor in kernels[node_index].items():
        q = key[2]
        keep = allocation.copies_by_q[q]
        if isinstance(keep, bool) or not isinstance(keep, Integral) or not 0 <= keep <= tensor.shape[0]:
            raise ValueError("invalid retained multiplicity count")
        tensor[keep:] = 0.
    return ReducedSingletTree(state.tree, state.leaf_grams, tuple(kernels), transformed.root_vector)


__all__ = ["ReducedSingletTree", "EdgeEnvironments", "CanonicalSpinBlock", "CanonicalEdgeSpectrum",
           "MultipletAllocation", "reduced_tree_environments", "canonical_edge_spectrum",
           "gauge_transform_edge", "allocate_multiplets", "allocate_multiplet_weights", "truncate_internal_edge"]
