"""Finite-bond, invariant Schur tensor decoder with a neural tensor generator.

This is a spatial *cell tensor tree*, not the frozen left-path ansatz. Complete
four-spin Schur tensors sit on the physical legs. Internal edges retain whole
SU(2) multiplets q=0,2,...,q_max with chi copies; no magnetic-dependent cutoff
is allowed. Fixed CG tensors contract their magnetic indices. The CNN/attention
generates reduced tensors ONCE per parameter update, never once per physical
configuration or Hamiltonian connection.

Important expressivity boundary: the neural generator is contained in the
untied free reduced-tensor manifold at the same (q_max,chi). It is NOT a
lossless factorization of an arbitrary full-configuration Transformer. Its
advantage, if any, must be measured against that free-tensor control.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import Leaf, TreeNode
from ..su2.local_schur import cell_schur_tensor
from .cell_multiplicity_clebsch_tree import SpinCellMultiplicityConfig, cell_multiplicity_layouts
from .schur_cutset_compact import packed_cg_table
from .su2_schur_config_vit import _layer_norm, _matrix


@dataclass(frozen=True)
class FiniteSchurDecoderConfig:
    """Explicit finite multiplet budget on a periodic rectangular cell grid."""
    tree: SpinCellMultiplicityConfig
    cell_rows: int
    cell_columns: int
    embedding_dim: int = 16
    depth: int = 2
    initialization_scale: float = .2

    def __post_init__(self) -> None:
        if not isinstance(self.tree, SpinCellMultiplicityConfig):
            raise TypeError('tree must be a SpinCellMultiplicityConfig')
        if (self.tree.cell_kind != 'square_plaquette' or self.tree.merge_mode != 'full_joint'
                or self.tree.target_q != 0 or self.tree.target_m2 != 0):
            raise ValueError('need complete four-spin cells and a full-joint singlet target')
        q = self.tree.maximum_intermediate_q
        if q is None or q < 4:
            raise ValueError('explicit doubled q cutoff >=4 keeps ALL local Schur multiplets')
        for name in ('cell_rows', 'cell_columns', 'embedding_dim', 'depth'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, Integral) or v < 1:
                raise ValueError(name+' must be a positive integer')
        if self.cell_rows*self.cell_columns != self.tree.n_cells:
            raise ValueError('cell grid and tree disagree')
        if not np.isfinite(self.initialization_scale) or self.initialization_scale <= 0:
            raise ValueError('positive finite initialization scale required')


def _leaf_ids(node: TreeNode) -> tuple[int, ...]:
    if isinstance(node, Leaf):
        return (node.index,)
    return _leaf_ids(node.left)+_leaf_ids(node.right)


@lru_cache(maxsize=None)
def reduced_entry_metadata(config: FiniteSchurDecoderConfig) -> dict:
    """Deterministic active entries; q is doubled, a indexes multiplicities.

    Each descriptor is (node,q_left,q_right,q_total,a_out,a_left,a_right).
    Padding is absent from the parameter vector and is not counted as capacity.
    Different spatial nodes NEVER share free tensors merely because sizes agree.
    """
    labels, blocks, offset = [], [], 0
    for index, layout in enumerate(cell_multiplicity_layouts(config.tree)):
        node_blocks = []
        for key, lw, rw in zip(layout.keys, layout.left_widths, layout.right_widths, strict=True):
            shape = (config.tree.feature_dim, lw, rw)
            count = int(np.prod(shape))
            for k in range(shape[0]):
                for a in range(lw):
                    for b in range(rw):
                        labels.append((index, *key, k, a, b))
            node_blocks.append((offset, offset+count, shape))
            offset += count
        blocks.append(tuple(node_blocks))
    label_array = np.asarray(labels, dtype=np.int32); label_array.setflags(write=False)
    return dict(labels=label_array, blocks=tuple(blocks),
                complex_entries=offset, free_real_parameters=2*offset)


def pack_reduced_bank(entries: jax.Array, config: FiniteSchurDecoderConfig) -> dict:
    """Pack complex ACTIVE entries into an untied padded numeric tensor bank."""
    meta = reduced_entry_metadata(config)
    values = jnp.asarray(entries)
    if values.shape != (meta['complex_entries'],):
        raise ValueError('wrong active reduced-entry shape')
    nodes = []
    for blocks, layout in zip(meta['blocks'], cell_multiplicity_layouts(config.tree), strict=True):
        lw, rw = max(layout.left_widths), max(layout.right_widths)
        nodes.append(jnp.stack([jnp.pad(values[start:stop].reshape(shape),
            ((0, 0), (0, lw-shape[1]), (0, rw-shape[2]))) for start, stop, shape in blocks]))
    return dict(nodes=tuple(nodes))


def free_reduced_bank(params: dict, config: FiniteSchurDecoderConfig) -> dict:
    """Strong control: every complex reduced entry is independently trainable."""
    z = jnp.asarray(params['entries'])
    if z.shape != (reduced_entry_metadata(config)['complex_entries'], 2):
        raise ValueError('free parameters must have shape [active_entries,2]')
    return pack_reduced_bank(z[:, 0]+1j*z[:, 1], config)


def init_finite_schur_network(key: jax.Array, config: FiniteSchurDecoderConfig) -> dict:
    """Initialize the full CNN/attention/reduced-label generator, all trainable."""
    keys = iter(jax.random.split(key, 5+8*config.depth))
    d, scale = config.embedding_dim, config.initialization_scale
    layers = []
    for _ in range(config.depth):
        layers.append(dict(
            conv=scale*jax.random.normal(next(keys), (3, d, d), dtype=jnp.float64)/np.sqrt(3*d),
            **{name: _matrix(next(keys), d, d, scale) for name in ('query', 'key', 'value', 'output')},
            distance_bias=scale*jax.random.normal(next(keys), (), dtype=jnp.float64),
            mlp_in=_matrix(next(keys), d, 2*d, scale), mlp_out=_matrix(next(keys), 2*d, d, scale)))
    return dict(position=_matrix(next(keys), 5, d, scale),
        q_embedding=scale*jax.random.normal(next(keys), (3, config.tree.maximum_intermediate_q+1, d), dtype=jnp.float64),
        a_embedding=scale*jax.random.normal(next(keys), (3, max(3, config.tree.feature_dim), d), dtype=jnp.float64),
        readout=_matrix(next(keys), d, 2, scale), bias=scale*jax.random.normal(next(keys), (2,), dtype=jnp.float64),
        layers=tuple(layers))


def neural_reduced_entries(params: dict, config: FiniteSchurDecoderConfig) -> jax.Array:
    """CNN short-range context + spatial attention generates all reduced entries.

    Geometry and local kernel (q,a) labels only, no magnetic index or physical
    bit in any nonlinearity. Context is shared across tensor channels; this
    finite factorization does NOT claim unrestricted configuration-dependent
    long-range neural coefficients. Return the bank once, then reuse it.
    """
    rows, columns, d = config.cell_rows, config.cell_columns, config.embedding_dim
    rr, cc = np.indices((rows, columns))
    coords = np.stack((rr.ravel(), cc.ravel()), axis=1)
    features = np.stack((np.ones(rows*columns), np.sin(2*np.pi*rr.ravel()/rows),
        np.cos(2*np.pi*rr.ravel()/rows), np.sin(2*np.pi*cc.ravel()/columns),
        np.cos(2*np.pi*cc.ravel()/columns)), axis=1)
    t = jnp.asarray(features)@params['position']
    delta = abs(coords[:, None]-coords[None])
    delta = np.minimum(delta, np.asarray((rows, columns))-delta)
    distance2 = jnp.asarray(np.sum(delta**2, axis=-1), dtype=jnp.float64)
    for layer in params['layers']:
        grid = _layer_norm(t).reshape(rows, columns, d)
        local = (grid@layer['conv'][0]
            + .5*(jnp.roll(grid, 1, axis=0)+jnp.roll(grid, -1, axis=0))@layer['conv'][1]
            + .5*(jnp.roll(grid, 1, axis=1)+jnp.roll(grid, -1, axis=1))@layer['conv'][2])
        t += jax.nn.gelu(local.reshape(-1, d))
        norm = _layer_norm(t)
        query, key, value = (norm@layer[name] for name in ('query', 'key', 'value'))
        weights = jax.nn.softmax(query@key.T/np.sqrt(d)+layer['distance_bias']*distance2, axis=-1)
        t += (weights@value)@layer['output']
        t += jax.nn.gelu(_layer_norm(t)@layer['mlp_in'])@layer['mlp_out']
    contexts = jnp.stack([t[jnp.asarray(_leaf_ids(node))].mean(axis=0)
                         for node in config.tree.cell_tree.internal_nodes])
    z = jnp.asarray(reduced_entry_metadata(config)['labels'])
    token = contexts[z[:, 0]]
    for role in range(3):
        token += params['q_embedding'][role, z[:, 1+role]]
        token += params['a_embedding'][role, z[:, 4+role]]
    output = jnp.tanh(_layer_norm(token))@params['readout']+params['bias']
    # Fixed rescaling, not per-query/per-spin normalization or truncation.
    scale = np.concatenate([np.repeat(1/np.sqrt(shape[1]*shape[2]), stop-start)
        for blocks in reduced_entry_metadata(config)['blocks'] for start, stop, shape in blocks])
    return (output[:, 0]+1j*output[:, 1])*jnp.asarray(scale)


def neural_reduced_bank(params: dict, config: FiniteSchurDecoderConfig) -> dict:
    """Differentiable parameter-to-bank map, outside all physical query loops."""
    return pack_reduced_bank(neural_reduced_entries(params, config), config)


def shifted_neural_reduced_bank(params: dict, config: FiniteSchurDecoderConfig, *,
                                offset_entries: jax.Array, reference_entries: jax.Array) -> dict:
    """W0 + G(theta) - G(theta0), with an untrained full-rank reduced offset.

    SAME trainable CNN/attention parameters; no additional head or magnetic
    dependence. W0 may start with rich multiplicity participation instead of
    the generator's almost-product tensor bank. The result remains contained
    in the untied free tensor family. Offset/reference arrays are frozen;
    physical amplitudes still depend on configurations only through fixed CG.
    This initialization does not certify low energy or a particular phase.
    """
    count=reduced_entry_metadata(config)['complex_entries']
    offset,reference=map(jnp.asarray,(offset_entries,reference_entries))
    if offset.shape != (count,) or reference.shape != (count,):
        raise ValueError('offset and reference must contain every active complex reduced entry')
    entries=(jax.lax.stop_gradient(offset)+neural_reduced_entries(params,config)
             -jax.lax.stop_gradient(reference))
    return pack_reduced_bank(entries,config)


def _make_finite_schur_contraction(config: FiniteSchurDecoderConfig, *, features: bool) -> Callable:
    """Single CG convention for physical amplitudes and initialization probes."""
    tree, layouts = config.tree.cell_tree, cell_multiplicity_layouts(config.tree)
    schur = jnp.asarray(cell_schur_tensor('square_plaquette'))
    tables = tuple(jnp.asarray(packed_cg_table(row.left_size, row.right_size, row.keys)) for row in layouts)
    channels = tuple(jnp.asarray(row.keys, dtype=jnp.int32) for row in layouts)
    cap, chi = config.tree.maximum_intermediate_q, config.tree.feature_dim
    # Repeated-q scatter-add uses floating-point atomics on GPU. Identical
    # banks can then differ by a few ulps between physical reference calls,
    # spoiling exact carrier cancellation. Group the SAME channels statically
    # and use fixed reduction kernels; no CG channel or parameter is changed.
    by_total_q = tuple(tuple(jnp.asarray(np.flatnonzero(np.asarray(row.keys)[:,2]==q),dtype=jnp.int32)
                            for q in range(cap+1)) for row in layouts)

    def raw(bank: dict, spins: jax.Array) -> jax.Array:
        bits = jnp.asarray(spins)
        if bits.shape != (config.tree.n_sites,):
            raise ValueError('wrong physical spin shape')
        if not jnp.issubdtype(bits.dtype, jnp.integer):
            raise TypeError('computational bits must be integers')
        valid = jnp.all((bits == 0) | (bits == 1)); bits = jnp.clip(bits, 0, 1)
        values = {}
        for leaf in tree.leaves:
            local = bits[jnp.asarray(config.tree.cells[leaf.index])]
            m = jnp.sum(2*local-1)
            values[leaf] = (m, schur[jnp.dot(local, jnp.asarray([8, 4, 2, 1])), :, m+4, :])
        for index, (node, layout) in enumerate(zip(tree.internal_nodes, layouts, strict=True)):
            ml, left = values[node.left]; mr, right = values[node.right]
            ql, qr, qt = channels[index].T
            lw, rw = max(layout.left_widths), max(layout.right_widths)
            reduced = jnp.einsum('cl,cr,cklr->ck', left[ql, :lw], right[qr, :rw], bank['nodes'][index])
            cg = tables[index][:, ml+layout.left_size, mr+layout.right_size]
            terms=cg[:,None]*reduced
            result=jnp.stack([jnp.sum(terms[ids],axis=0) if len(ids)
                              else jnp.zeros((chi,),dtype=reduced.dtype)
                              for ids in by_total_q[index]])
            values[node] = (ml+mr, result)
        m, root = values[tree.root]
        if features:
            return tuple(jnp.where(valid, values[node][1], 0)
                         for node in tree.internal_nodes)
        return jnp.where(valid & (m == 0), root[0].sum()/np.sqrt(chi), 0j)
    return raw


def make_finite_schur_amplitude(config: FiniteSchurDecoderConfig) -> Callable:
    """Unnormalized raw(bank,bits), with fixed CG indices and a q=m=0 root.

    A physical row fixes subtree magnetic projections. Reduced kernels never
    depend on that row. Every parameter tangent remains an exact singlet;
    VMC uses a physical-basis sampler, not a fusion autoregressive law.
    """
    return _make_finite_schur_contraction(config, features=False)


def make_finite_schur_features(config: FiniteSchurDecoderConfig) -> Callable:
    """Return postorder (doubled-q,multiplicity) activations for a physical row.

    For each node m2 is the physical sum over its subtree, already included
    in its fixed CG factors. These features are initialization diagnostics,
    not probabilities, entanglement spectra or physical spin occupations.
    """
    return _make_finite_schur_contraction(config, features=True)


def init_layer_standardized_free_tensors(key: jax.Array, config: FiniteSchurDecoderConfig,
                                         calibration_states: np.ndarray) -> tuple[dict, dict]:
    """Independent uniform complex reduced matrices, with empirical layer scales.

    Inspired by Vieijra--Nys initialization, without claiming a paper replica:
    these nodes are untied, and leaves use a complete local Schur front end.
    One POSITIVE scalar per node standardizes activations on fixed probes.
    It changes only the raw state's overall amplitude, not signs or its ray.
    It conditions free parameter coordinates, not a new variational manifold.
    No q channel is deleted, clipped or reweighted separately. Initialization
    probes are not Born samples and never certify a good physical state.
    """
    bits = np.asarray(calibration_states)
    if (bits.ndim != 2 or bits.shape[1] != config.tree.n_sites or len(bits) < 2
            or not np.issubdtype(bits.dtype, np.integer)
            or not np.all((bits == 0) | (bits == 1))):
        raise ValueError('at least two legal integer physical calibration rows required')
    meta = reduced_entry_metadata(config)
    entries = jax.random.uniform(key, (meta['complex_entries'], 2),
        dtype=jnp.float64, minval=-1., maxval=1.)
    features = jax.jit(jax.vmap(make_finite_schur_features(config), in_axes=(None,0)))
    probes = jnp.asarray(bits, dtype=jnp.int32)
    scales, before = [], []
    for index, blocks in enumerate(meta['blocks']):
        value = np.asarray(features(free_reduced_bank({'entries':entries}, config), probes)[index])
        std = float(np.sqrt(np.mean(abs(value-value.mean())**2)))
        if not np.isfinite(std) or std < 1e-150:
            raise FloatingPointError('zero/nonfinite layer activation scale; no floor or channel repair')
        start, stop = blocks[0][0], blocks[-1][1]
        entries = entries.at[start:stop].divide(std)
        scales.append(1/std); before.append(std)
    final = features(free_reduced_bank({'entries':entries}, config), probes)
    after = [float(np.sqrt(np.mean(abs(np.asarray(v)-np.asarray(v).mean())**2))) for v in final]
    return {'entries':entries}, dict(calibration_rows=len(bits), node_scalar_scales=scales,
        sequential_activation_std_before=before, activation_std_after=after,
        raw_global_amplitude_scale=float(np.prod(scales)), physical_ray_unchanged_by_scaling=True,
        magnetic_index_dependent_weights=False, independent_reduced_entries=True,
        initialization_is_not_ground_state_certificate=True)


def centered_additive_amplitudes(
    residual: Callable, carrier: Callable, reference_bank: dict, *, scale: float = 1.,
) -> Callable:
    """C(bits)+scale*[D(bank,bits)-D(reference,bits)] in ONE physical representation.

    Carrier and residual must each have independent S=0/A1 certificates. No
    carrier zeros are frozen. This wrapper does not convert the carrier to
    global fusion coefficients and does not short-circuit initial gradients.
    """
    if not np.isfinite(scale):
        raise ValueError('finite additive scale required')
    frozen = jax.tree.map(jax.lax.stop_gradient, reference_bank)
    return lambda bank, states: carrier(states)+scale*(residual(bank, states)-residual(frozen, states))


__all__ = ['FiniteSchurDecoderConfig', 'reduced_entry_metadata', 'pack_reduced_bank',
    'free_reduced_bank', 'init_finite_schur_network', 'neural_reduced_entries',
    'neural_reduced_bank', 'make_finite_schur_amplitude', 'centered_additive_amplitudes']
__all__ += ['make_finite_schur_features', 'init_layer_standardized_free_tensors']
__all__ += ['shifted_neural_reduced_bank']
