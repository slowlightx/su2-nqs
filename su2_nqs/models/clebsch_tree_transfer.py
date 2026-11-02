"""Transfer reduced ClebschTree matrices without changing magnetic structure.

Same-tree ``embedding`` is an exact zero-padded inclusion into wider reduced
spaces.  Across different trees, ``initialization`` only transfers matching
subtree-size and doubled-charge blocks; unmatched coefficients remain the
independent target initialization.  This latter operation is a warm start,
not a claim that a projected small-system physical state has been embedded.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .literature_clebsch_tree import (
    LiteratureClebschTreeConfig, clebsch_layers, make_clebsch_features,
)
from ..su2.fusion_tree import Leaf, TreeNode


def _subtree_size(node: TreeNode) -> int:
    return 1 if isinstance(node, Leaf) else _subtree_size(node.left) + _subtree_size(node.right)


def _blocks(bank: jax.Array, layer: Any) -> dict[tuple[int, int, int], jax.Array]:
    result = {}
    offset = 0
    for key, shape in zip(layer.keys, layer.shapes, strict=True):
        count = int(np.prod(shape))
        result[key] = bank[offset:offset + count].reshape(shape)
        offset += count
    if bank.ndim != 1 or offset != bank.size:
        raise ValueError("reduced bank does not match the registered channel layout")
    return result


def transfer_clebsch_parameters(
    source_config: LiteratureClebschTreeConfig,
    source: Mapping[str, Any],
    target_config: LiteratureClebschTreeConfig,
    target: Mapping[str, Any],
    *, mode: str = "initialization",
) -> tuple[dict, dict]:
    """Copy matching ``(q_left,q_right,q_out)`` blocks using doubled labels.

    ``embedding`` requires the identical tree and target ``(q,m2)``, and only
    permits increased intermediate cutoffs/widths.  All newly opened entries
    are zero, giving exactly the original amplitudes.  ``initialization``
    retains every unmatched target coefficient (including positive-q banks)
    and leaves the target readout unchanged unless both trees are identical.
    """
    if mode not in ("initialization", "embedding"):
        raise ValueError("mode must be initialization or embedding")
    if (source_config.target_q, source_config.target_m2) != (
            target_config.target_q, target_config.target_m2):
        raise ValueError("transfer must preserve target doubled q and m")
    same_tree = source_config.tree == target_config.tree
    if mode == "embedding" and not same_tree:
        raise ValueError("exact embedding requires the identical tree")
    source_layers = clebsch_layers(source_config)
    target_layers = clebsch_layers(target_config)
    if len(source['merges']) != len(source_layers) or len(target['merges']) != len(target_layers):
        raise ValueError("parameter banks and registered layers disagree")
    lookup = {}
    for bank, layer in zip(source['merges'], source_layers, strict=True):
        node = layer.nodes[0]
        signature = (_subtree_size(node.left), _subtree_size(node.right))
        if signature in lookup:
            raise ValueError("source has ambiguous shared subtree-size banks")
        lookup[signature] = _blocks(bank, layer)
    merged = []
    rows = []
    copied = 0
    for bank, layer in zip(target['merges'], target_layers, strict=True):
        node = layer.nodes[0]
        signature = (_subtree_size(node.left), _subtree_size(node.right))
        existing = lookup.get(signature, {})
        target_blocks = _blocks(bank, layer)
        if mode == "embedding" and set(existing) - set(target_blocks):
            raise ValueError("embedding cannot remove source charge channels")
        transferred = []
        pieces = []
        for key, value in target_blocks.items():
            block = jnp.zeros_like(value) if mode == "embedding" else value
            if key in existing:
                previous = existing[key]
                if mode == "embedding" and any(a > b for a, b in zip(previous.shape, value.shape, strict=True)):
                    raise ValueError("embedding cannot truncate reduced widths")
                shape = tuple(min(a, b) for a, b in zip(previous.shape, value.shape, strict=True))
                selection = tuple(slice(n) for n in shape)
                block = block.at[selection].set(previous[selection])
                copied += int(np.prod(shape))
                transferred.append(dict(doubled_channels=list(key), copied_shape=list(shape)))
            pieces.append(block.reshape(-1))
        merged.append(jnp.concatenate(pieces))
        rows.append(dict(child_sites=list(signature), channels=transferred,
                         initialized_new_entries="zero" if mode == "embedding" else "retained_target_random"))
    readout = jnp.zeros_like(target['readout']) if mode == "embedding" else target['readout']
    if same_tree:
        if mode == "embedding" and len(source['readout']) > len(readout):
            raise ValueError("embedding cannot truncate the root reduced readout")
        count = min(len(source['readout']), len(readout))
        readout = readout.at[:count].set(source['readout'][:count])
        copied += count
    report = dict(mode=mode, exact_physical_ray_preserved=mode == "embedding",
                  source_sites=source_config.tree.n_leaves,
                  target_sites=target_config.tree.n_leaves,
                  copied_scalar_parameters=copied, layers=rows,
                  magnetic_index_parameters=False,
                  projected_small_state_embedding_claim=False)
    return dict(merges=tuple(merged), readout=readout), report


def calibrate_clebsch_parameters(
    config: LiteratureClebschTreeConfig, params: Mapping[str, Any], states: np.ndarray,
) -> tuple[dict, dict]:
    """Positive layer-wise standardization, preserving the actual physical ray.

    This is the original paper-style initialization scale calibration, not a
    fit, channel truncation, stochastic reconfiguration, or state projection.
    Every weight in one shared bank receives the same positive scalar.
    """
    rows = np.asarray(states)
    if rows.ndim != 2 or rows.shape[1] != config.tree.n_leaves or not np.isin(rows, [0, 1]).all():
        raise ValueError("calibration needs valid physical bit rows")
    layers = clebsch_layers(config)
    if len(params['merges']) != len(layers):
        raise ValueError("parameters disagree with the configured layers")
    current = dict(params)
    evaluate = jax.jit(make_clebsch_features(config))
    report = []
    for index, layer in enumerate(layers):
        out = np.asarray(evaluate(current, jnp.asarray(rows))[index])
        active = np.asarray([[k < width for k in range(out.shape[-1])] for width in layer.widths])
        std = float(np.std(out[:, :, active]))
        if not np.isfinite(std) or std <= 1e-15:
            raise FloatingPointError("vanishing/invalid transferred layer activation scale")
        banks = list(current['merges'])
        banks[index] = banks[index] / std
        current = dict(current, merges=tuple(banks))
        after = float(np.std(np.asarray(evaluate(current, jnp.asarray(rows))[index])[:, :, active]))
        report.append(dict(layer=index, positive_scale=1 / std, std_after=after))
    return current, dict(physical_ray_preserved=True, layers=report)
