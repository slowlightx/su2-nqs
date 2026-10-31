"""State-preserving capacity inclusion for the existing physical-spin ViT.

This changes neither the spatial attention rule nor the physical symmetries.
New feed-forward features have nonzero inputs and zero outgoing weights;
optional new residual encoders start as identities. Thus old states remain
available and the newly opened output directions need not be dead channels.
"""
from __future__ import annotations

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from .literature_spatial_vit import (
    LiteratureSpatialViTConfig, init_literature_spatial_vit,
)


def expand_spatial_vit_capacity(
    params: dict,
    source: LiteratureSpatialViTConfig,
    target: LiteratureSpatialViTConfig,
    key: jax.Array,
) -> dict:
    """Include one full fixed-doubled-m ViT in a wider MLP/deeper ViT.

    Geometry, patching, embedding/head dimensions, normalization, and doubled
    magnetization labels must be identical. Only ``mlp_ratio`` and ``depth``
    can increase. This is not optimizer-state continuation or exact SU(2).
    """
    before, after = asdict(source), asdict(target)
    for name in before:
        if name not in ('depth', 'mlp_ratio') and before[name] != after[name]:
            raise ValueError('capacity inclusion cannot change '+name)
    if target.depth < source.depth or target.mlp_ratio < source.mlp_ratio:
        raise ValueError('capacity inclusion cannot shrink the old network')
    template = init_literature_spatial_vit(key, source)
    actual, structure = jax.tree.flatten(params)
    expected, reference_structure = jax.tree.flatten(template)
    if structure != reference_structure or any(
        np.shape(a) != np.shape(b) or np.asarray(a).dtype != np.asarray(b).dtype
        or not np.isfinite(np.asarray(a)).all()
        for a, b in zip(actual, expected, strict=True)
    ):
        raise ValueError('parent arrays do not match the declared complete ViT')
    expanded = init_literature_spatial_vit(key, target)
    for name in ('patch_embedding', 'patch_embedding_bias', 'final_norm', 'log_cosh_head'):
        expanded[name] = jax.tree.map(lambda v: jnp.array(v), params[name])
    layers = []
    width = source.mlp_dim
    for index, fresh in enumerate(expanded['layers']):
        if index < source.depth:
            old = params['layers'][index]
            for name in old:
                if name not in ('mlp_in', 'mlp_in_bias', 'mlp_out'):
                    fresh[name] = jax.tree.map(jnp.array, old[name])
            fresh['mlp_in'] = fresh['mlp_in'].at[:, :width].set(old['mlp_in'])
            fresh['mlp_in_bias'] = fresh['mlp_in_bias'].at[:width].set(old['mlp_in_bias'])
            fresh['mlp_out'] = jnp.zeros_like(fresh['mlp_out']).at[:width].set(old['mlp_out'])
        else:
            for name in ('output', 'output_bias', 'mlp_out', 'mlp_out_bias'):
                fresh[name] = jnp.zeros_like(fresh[name])
        layers.append(fresh)
    expanded['layers'] = tuple(layers)
    return expanded
