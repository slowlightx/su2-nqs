"""Deterministic size transfer of factored Spatial-ViT parameters.

This initializes a NEW physical state, not a continuation or a tensor-product
construction. Only the relative-factor table depends on the number of patches.
The historical interpolation uses normalized periodic patch coordinates; an
optional lift preserves the residue-folded effective spatial kernel. Learned
distance decay remains in physical-site units. No SU(2) symmetry is added.
"""
from __future__ import annotations

from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np

from .literature_spatial_vit import LiteratureSpatialViTConfig, literature_relative_geometry


def transfer_spatial_vit_size(params: dict, source: LiteratureSpatialViTConfig,
                             target: LiteratureSpatialViTConfig, *,
                             rule: str = 'normalized_bilinear') -> dict:
    """Copy the backbone and lift relative tables by an explicit size rule.

    ``normalized_bilinear`` retains the historical initializer. The optional
    ``folded_spatial_kernel`` preserves each layer's effective linear spatial
    filter on configurations tiled from the source torus: target kernel
    weights with the same source displacement sum to the original weight.
    It preserves physical distance units and requires integer size multiples.
    This identity does not make the target wavefunction a tensor product or
    certify its energy; global pooling and full spatial projection still act
    on the target system.

    All architecture fields except physical rows/columns must agree. Equal
    grids replay bit-for-bit. In the historical bilinear rule, constant tables
    remain constant and new tables are convex combinations. The folded rule
    instead rescales the table to compensate for its changed envelope mass.
    Projection and sampler initialization must be rebuilt for the actual
    target Hamiltonian.
    """
    if rule not in {'normalized_bilinear', 'folded_spatial_kernel'}:
        raise ValueError('unknown Spatial-ViT size-transfer rule')
    old, new = asdict(source), asdict(target)
    for name in old:
        if name not in {'rows', 'columns'} and old[name] != new[name]:
            raise ValueError('size transfer changed architecture: '+name)
    result = jax.tree.map(lambda value: jnp.array(value), params)
    r0, c0 = source.patch_grid_rows, source.patch_grid_columns
    r1, c1 = target.patch_grid_rows, target.patch_grid_columns
    if (r0, c0) == (r1, c1):
        return result
    if rule == 'folded_spatial_kernel':
        if r1 % r0 or c1 % c0 or not source.use_distance_envelope:
            raise ValueError('folded spatial transfer needs integer size multiples and a distance envelope')
        alias = ((np.arange(r1)[:, None] % r0)*c0
                 + np.arange(c1)[None, :] % c0).ravel()
        source_distance = jnp.asarray(literature_relative_geometry(source)[1][0])
        target_distance = jnp.asarray(literature_relative_geometry(target)[1][0])
        membership = jnp.asarray(alias[:, None] == np.arange(r0*c0)[None, :])
        layers = []
        for layer in result['layers']:
            decay = jax.nn.softplus(layer['raw_distance_decay'])[:, None]
            source_log_envelope = jax.nn.log_softmax(-decay*source_distance[None], axis=-1)
            target_log_envelope = jax.nn.log_softmax(-decay*target_distance[None], axis=-1)
            # Log-space folding avoids underflow for remote displacements.
            folded_log_mass = jax.scipy.special.logsumexp(
                jnp.where(membership[None], target_log_envelope[:, :, None], -jnp.inf), axis=1)
            correction = jnp.exp(source_log_envelope-folded_log_mass)
            lifted = layer['relative_factor'][:, alias]*correction[:, alias]
            layers.append({**layer, 'relative_factor': lifted})
        return {**result, 'layers': tuple(layers)}
    rr = np.arange(r1)*r0/r1
    cc = np.arange(c1)*c0/c1
    ir, ic = np.floor(rr).astype(int), np.floor(cc).astype(int)
    dr, dc = jnp.asarray(rr-ir), jnp.asarray(cc-ic)
    layers = []
    for layer in result['layers']:
        table = layer['relative_factor'].reshape(source.head_count, r0, c0)
        a = table[:, ir[:, None], ic[None, :]]
        b = table[:, (ir[:, None]+1) % r0, ic[None, :]]
        c = table[:, ir[:, None], (ic[None, :]+1) % c0]
        d = table[:, (ir[:, None]+1) % r0, (ic[None, :]+1) % c0]
        resized = ((1-dr[:, None])*(1-dc[None, :])*a
                   + dr[:, None]*(1-dc[None, :])*b
                   + (1-dr[:, None])*dc[None, :]*c
                   + dr[:, None]*dc[None, :]*d)
        layers.append({**layer, 'relative_factor': resized.reshape(source.head_count, -1)})
    return {**result, 'layers': tuple(layers)}
