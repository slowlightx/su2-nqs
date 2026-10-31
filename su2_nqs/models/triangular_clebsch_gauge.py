"""Offline canonical coordinates and physical edge spectra for triangular CT.

This adapter reuses the exact SU(2) reduced double layer. It acts on the
common physical-amplitude bank, after a binary core has been folded, and
does not change CG conventions or the production JAX contraction. All
charges are doubled integers. QR and environment diagonalization happen
outside differentiation and sampling.

The spectra describe ONE RAW TREE. A coherent lattice projection remains
unchanged by a gauge transformation, but its Schmidt spectra include cross
image terms and are not the spectra returned here. Moving the optimizer
center or changing widths requires a new optimizer state: Adam/SR coordinate
data cannot simply be reused. Untied edge gauges need not stay inside a
particular shared dictionary or local-C3 parameterization.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode
from ..su2.reduced_tree_environment import (
    CanonicalEdgeSpectrum, ReducedSingletTree, canonical_edge_spectrum,
    reduced_tree_environments,
)
from ..su2.triangle_blocks import nine_site_output, triangle_block_tables
from .triangular_clebsch_expansion import initialize_live_triangular_expansion
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout


def _nodes(config: MixedClebschConfig) -> tuple[FusionTree, tuple[TreeNode, ...]]:
    hierarchy = triangular_schur_hierarchy(config.length)
    nodes: list[TreeNode] = [Leaf(i) for i in range(len(hierarchy.blocks))]
    for left, right, parent in hierarchy.binary_merges:
        if parent != len(nodes):
            raise ValueError("noncanonical hierarchy indexing")
        nodes.append(FusionNode(nodes[left], nodes[right]))
    return FusionTree(nodes[-1]), tuple(nodes)


def _unpack(bank: Mapping, config: MixedClebschConfig) -> tuple[list, list]:
    if set(bank) != {"local", "top"}:
        raise ValueError("folded local/top physical-amplitude bank required")
    hierarchy = triangular_schur_hierarchy(config.length)
    if len(bank["local"]) != len(hierarchy.blocks):
        raise ValueError("one local bank per nine-site block required")
    local = []
    for values in bank["local"]:
        if len(values) != len(config.local_q):
            raise ValueError("local doubled-charge blocks disagree")
        blocks = {}
        for q, w in zip(config.local_q, values, strict=True):
            array = np.asarray(w, dtype=np.result_type(np.asarray(w).dtype, np.float64))
            expected = (triangle_block_tables()["nine"][q].shape[1],
                        nine_site_output(q, int(config.multiplicity_cap)).width)
            if array.shape != expected or np.any(~np.isfinite(array)):
                raise ValueError("finite local reduced matrix with declared widths required")
            blocks[q] = array.copy()
        local.append(blocks)
    top = []
    for groups, row in zip(bank["top"], mixed_clebsch_layout(config), strict=True):
        blocks = {}
        for values, group in zip(groups, row["groups"], strict=True):
            array = np.asarray(values, dtype=np.result_type(np.asarray(values).dtype, np.float64))
            if array.shape != group["shape"] or np.any(~np.isfinite(array)):
                raise ValueError("top charge-bucket shape disagrees")
            blocks.update({key: value.copy() for key, value in zip(group["keys"], array, strict=True)})
        top.append(blocks)
    return local, top


def _pack(local: list, top: list, config: MixedClebschConfig) -> dict:
    return {"local": tuple(tuple(node[q] for q in config.local_q) for node in local),
            "top": tuple(tuple(np.stack([node[key] for key in group["keys"]]) for group in row["groups"])
                         for node, row in zip(top, mixed_clebsch_layout(config), strict=True))}


def triangular_bank_reduced_tree(bank: Mapping, config: MixedClebschConfig) -> ReducedSingletTree:
    """Map a physical CT bank to its exact raw reduced tree and leaf Grams.

    Local nine-site Schur columns are orthonormal at each fixed magnetic
    index, hence the local Gram is W^dagger W without a magnetic factor.
    The factor q+1 is supplied by the exact complementary environment.
    """
    local, top = _unpack(bank, config)
    tree, nodes = _nodes(config)
    by_node = dict(zip(nodes[len(local):], top, strict=True))
    grams = tuple({q: w.conj().T @ w for q, w in blocks.items()} for blocks in local)
    return ReducedSingletTree(tree, grams, tuple(by_node[node] for node in tree.internal_nodes), np.ones(1))


def triangular_edge_spectrum(bank: Mapping, config: MixedClebschConfig,
                             edge: int) -> CanonicalEdgeSpectrum:
    """Physical raw-tree spectrum on hierarchy edge ``edge`` (never root).

    Reject incomplete Schmidt weight caused by an ill-conditioned virtual
    gauge. A small Gram eigenvalue alone does not imply small physical
    weight: the complementary environment can carry the inverse scale.
    Canonicalize the bank first rather than silently discarding such modes.
    """
    _, nodes = _nodes(config)
    if isinstance(edge, bool) or not isinstance(edge, Integral) or not 0 <= edge < len(nodes)-1:
        raise ValueError("nonroot hierarchy edge index required")
    state = triangular_bank_reduced_tree(bank, config)
    environment = reduced_tree_environments(state)
    spectrum = canonical_edge_spectrum(environment, nodes[edge])
    if (not np.isfinite(spectrum.normalized_weight_error)
            or spectrum.normalized_weight_error > 1e-10
            or environment.maximum_relative_edge_norm_error > 1e-10):
        raise ArithmeticError(
            "raw edge spectrum lost normalized physical weight; canonicalize "
            "the bank before interpreting or compressing this ill-conditioned gauge")
    return spectrum


def triangular_raw_overlap(left: Mapping, right: Mapping,
                           config: MixedClebschConfig) -> complex:
    """Exact <left|right> of two unprojected banks with the same layout.

    Unlike a configuration-frequency overlap this contracts all physical
    states through the fixed orthonormal CG identities. It does not include
    cross-image contractions needed by complete A1 projection.
    """
    ll, lt = _unpack(left, config)
    rl, rt = _unpack(right, config)
    up = [{q: x[q].conj().T @ y[q] for q in config.local_q}
          for x, y in zip(ll, rl, strict=True)]
    for left_blocks, right_blocks, row in zip(lt, rt, mixed_clebsch_layout(config), strict=True):
        current = {}
        for (ql, qr, qt), a in left_blocks.items():
            b = right_blocks[(ql, qr, qt)]
            value = np.einsum("alr,buv,lu,rv->ab", a.conj(), b,
                              up[row["left"]][ql], up[row["right"]][qr], optimize=True)
            current[qt] = current.get(qt, np.zeros_like(value))+value
        up.append(current)
    return complex(up[-1][0][0, 0])


def gauge_triangular_edge(bank: Mapping, config: MixedClebschConfig, edge: int,
                          gauges: Mapping[int, np.ndarray]) -> dict:
    """Apply basis |L> -> |L>Z, parent -> Z^-1 parent in each doubled-q block."""
    local, top = _unpack(bank, config)
    rows = mixed_clebsch_layout(config)
    count = len(local)
    if isinstance(edge, bool) or not isinstance(edge, Integral) or not 0 <= edge < count+len(top)-1:
        raise ValueError("nonroot hierarchy edge index required")
    widths = ({q: w.shape[1] for q, w in local[edge].items()} if edge < count
              else {key[2]: value.shape[0] for key, value in top[edge-count].items()})
    if set(gauges) != set(widths):
        raise ValueError("one square gauge per present doubled-q block required")
    inverse = {}
    for q, width in widths.items():
        z = np.asarray(gauges[q])
        if z.shape != (width, width) or not np.all(np.isfinite(z)):
            raise ValueError("finite square reduced gauge required")
        inverse[q] = np.linalg.inv(z)
    if edge < count:
        local[edge] = {q: w @ gauges[q] for q, w in local[edge].items()}
    else:
        top[edge-count] = {key: np.einsum("alr,ab->blr", w, gauges[key[2]], optimize=True)
                           for key, w in top[edge-count].items()}
    parent = next(i for i, row in enumerate(rows) if edge in (row["left"], row["right"]))
    is_left = rows[parent]["left"] == edge
    top[parent] = {key: (np.einsum("alr,ul->aur", w, inverse[key[0]], optimize=True) if is_left
                         else np.einsum("alr,vr->alv", w, inverse[key[1]], optimize=True))
                   for key, w in top[parent].items()}
    return _pack(local, top, config)


def canonicalize_triangular_bank(bank: Mapping, config: MixedClebschConfig) -> tuple[dict, dict]:
    """Move the raw-tree optimization center to the singlet root by block QR.

    No matrix inverse or rank threshold is used: rectangular QR retains all
    stored columns, including a completion of null directions. R factors
    are absorbed in the unique parent. Every physical amplitude and every
    coherent A1-projected amplitude is preserved. Root is left unnormalized.
    """
    local, top = _unpack(bank, config)
    rows = mixed_clebsch_layout(config)
    count = len(local)
    defects = []
    for edge in range(count+len(top)-1):
        factors = {}
        if edge < count:
            for q, matrix in local[edge].items():
                orthogonal, factor = np.linalg.qr(matrix, mode="reduced")
                if orthogonal.shape != matrix.shape:
                    raise ValueError("stored width exceeds complete local multiplicity")
                local[edge][q] = orthogonal
                factors[q] = factor
                defects.append(float(np.max(abs(orthogonal.conj().T @ orthogonal-np.eye(matrix.shape[1])))))
        else:
            node = top[edge-count]
            for q in sorted({key[2] for key in node}):
                keys = [key for key in node if key[2] == q]
                pieces = [node[key].reshape(node[key].shape[0], -1).T for key in keys]
                matrix = np.concatenate(pieces, axis=0)
                orthogonal, factor = np.linalg.qr(matrix, mode="reduced")
                if orthogonal.shape != matrix.shape:
                    raise ValueError("stored width exceeds incoming reduced product space")
                cursor = 0
                for key, piece in zip(keys, pieces, strict=True):
                    node[key] = orthogonal[cursor:cursor+len(piece)].T.reshape(node[key].shape)
                    cursor += len(piece)
                factors[q] = factor
                defects.append(float(np.max(abs(orthogonal.conj().T @ orthogonal-np.eye(matrix.shape[1])))))
        parent = next(i for i, row in enumerate(rows) if edge in (row["left"], row["right"]))
        is_left = rows[parent]["left"] == edge
        top[parent] = {key: (np.einsum("alr,ul->aur", w, factors[key[0]], optimize=True) if is_left
                             else np.einsum("alr,vr->alv", w, factors[key[1]], optimize=True))
                       for key, w in top[parent].items()}
    return _pack(local, top, config), {
        "center": "singlet_root", "raw_tree_only_metric": True,
        "complete_A1_amplitudes_preserved_by_linearity": True,
        "maximum_isometry_defect": max(defects, default=0.),
        "optimizer_coordinates_changed_require_reset": True,
        "spectra_of_projected_state_not_computed": True,
    }


def compress_triangular_edge(bank: Mapping, config: MixedClebschConfig, edge: int,
                             copies_by_q: Mapping[int, int]) -> tuple[dict, dict]:
    """Retain largest environment-weighted Schmidt copies on ONE raw edge.

    Arrays remain padded; this tests physical compression, not throughput.
    The raw normalized squared fidelity is the retained weight. Applying
    A1 after compression is legal but has a different fidelity/energy.
    """
    spectrum = triangular_edge_spectrum(bank, config, edge)
    if set(copies_by_q) != set(spectrum.blocks):
        raise ValueError("a retained count for each present doubled-q block required")
    for q, keep in copies_by_q.items():
        if (isinstance(keep, bool) or not isinstance(keep, Integral)
                or not 0 <= keep <= len(spectrum.blocks[q].multiplicity_weights)):
            raise ValueError("invalid retained multiplicity")
    gauged = gauge_triangular_edge(bank, config, edge, {q: b.gauge for q, b in spectrum.blocks.items()})
    local, top = _unpack(gauged, config)
    if edge < len(local):
        for q, w in local[edge].items():
            w[:, copies_by_q[q]:] = 0.
    else:
        for key, w in top[edge-len(local)].items():
            w[copies_by_q[key[2]]:] = 0.
    weight = sum(float(block.multiplicity_weights[:copies_by_q[q]].sum()) for q, block in spectrum.blocks.items())
    return _pack(local, top, config), {"raw_squared_fidelity": weight,
                                      "raw_discarded_weight": max(0., 1.-weight),
                                      "projected_fidelity_not_inferred": True,
                                      "padded_storage_unchanged": True}


def orthogonal_live_expansion(bank: Mapping, source_config: MixedClebschConfig,
                              target_config: MixedClebschConfig, key: jax.Array) -> tuple[dict, dict]:
    """Widen a folded joint bank with independent local complement columns.

    Reuses the tested one-sided live expansion: inherited parent rows
    annihilate new child features, so the old state is unchanged. New local
    columns are orthogonal to the old columns, rather than random near-
    duplicates. This does not infer useful missing directions from zero
    Schmidt weights: H/residual information and an energy test are needed.
    Native rank-limited binary parameters cannot be reconstructed silently.
    """
    source = jax.tree.map(jnp.asarray, dict(bank))
    expanded, report = initialize_live_triangular_expansion(
        source_config, source, target_config, key, mode="ternary_joint")
    local = []
    defects = []
    for old_node, new_node in zip(source["local"], expanded["local"], strict=True):
        values = []
        for old, new in zip(old_node, new_node, strict=True):
            old, new = np.asarray(old), np.array(new)
            added = new.shape[1]-old.shape[1]
            if added:
                q, _ = np.linalg.qr(old, mode="complete")
                complement = q[:, old.shape[1]:old.shape[1]+added]
                new[:, old.shape[1]:] = complement
                defects.append(float(np.max(abs(old.conj().T @ complement))))
            values.append(jnp.asarray(new))
        local.append(tuple(values))
    expanded["local"] = tuple(local)
    report.update(local_complement_orthogonality_error=max(defects, default=0.),
                  missing_direction_utility_not_predicted_from_parent_spectrum=True,
                  folded_joint_bank_only=True)
    return expanded, report


__all__ = ["triangular_bank_reduced_tree", "triangular_edge_spectrum", "triangular_raw_overlap", "gauge_triangular_edge",
           "canonicalize_triangular_bank", "compress_triangular_edge", "orthogonal_live_expansion"]
