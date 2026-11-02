"""Physics-based initial parameters for the unchanged complete Schur ViT."""
from __future__ import annotations

from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np

from .many_cell_schur_vit import ManyCellSchurViTConfig, init_many_cell_schur_vit


def deepen_many_cell_schur_vit(
    params: dict, source: ManyCellSchurViTConfig,
    target: ManyCellSchurViTConfig, key: jax.Array,
) -> dict:
    """Append identity residual blocks, preserving every reduced amplitude.

    Both causal probability and configuration-phase stacks grow. Only depth
    may change: doubled-q labels, masks, geometry and all existing parameters
    are untouched. New attention/MLP output matrices start at zero; their
    gradients can activate each block without destroying the parent state.
    This does not create a lattice-symmetry projector.
    """
    if target.depth <= source.depth:
        raise ValueError("target depth must exceed source depth")
    for field in fields(source):
        if field.name != "depth" and getattr(source, field.name) != getattr(target, field.name):
            raise ValueError("depth expansion changed " + field.name)
    template = init_many_cell_schur_vit(key, source)
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(template):
        raise ValueError("source parameter tree does not match declared architecture")
    for actual, expected in zip(jax.tree_util.tree_leaves(params),
                                jax.tree_util.tree_leaves(template), strict=True):
        if np.shape(actual) != np.shape(expected) or not np.isfinite(np.asarray(actual)).all():
            raise ValueError("invalid source parameter shape or value")
    initialized = init_many_cell_schur_vit(key, target)
    result = jax.tree_util.tree_map(jnp.asarray, params)
    for name in ("layers", "global_layers"):
        added = []
        for layer in initialized[name][source.depth:]:
            layer = dict(layer)
            layer["output"] = jnp.zeros_like(layer["output"])
            layer["mlp_out"] = jnp.zeros_like(layer["mlp_out"])
            added.append(layer)
        result[name] = tuple(result[name]) + tuple(added)
    return result


def afm_child_casimir_initialization(params: dict, inverse_temperature: float) -> dict:
    """Bias existing child-Casimir features without removing any doubled q.

    Existing features 3/4 are q(q+2)/[s(s+2)], where s is the number of
    spin-half leaves in the child. For a two-spin child their difference
    between q=0 and q=2 is one: the AFM pair's singlet-triplet gap J1=1.
    Thus the added logit gives the relative Boltzmann factor exp(-beta).
    Large inter-cell children use normalized Casimirs, not an extensive
    penalty or a q cutoff. This is an initializer, NOT a Gibbs wavefunction
    or a certificate of a ground-state/phase/space-group sector.
    """
    beta = float(inverse_temperature)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("inverse_temperature must be positive and finite")
    feature = jnp.asarray(params["logit_feature"])
    if feature.shape != (8,) or not jnp.issubdtype(feature.dtype,jnp.floating):
        raise ValueError("need the unchanged eight real candidate features")
    result = jax.tree_util.tree_map(lambda x:jnp.asarray(x),params)
    result["logit_feature"] = feature.at[3].add(-beta).at[4].add(-beta)
    return result


__all__ = ["afm_child_casimir_initialization", "deepen_many_cell_schur_vit"]
