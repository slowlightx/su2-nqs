"""Finite Schur-label-conditioned attention with exact physical SU(2) decoding.

One four-spin cell supplies six reduced labels (q=2j,a). A branch projects
onto a COMPLETE magnetic multiplet before the coherent branch sum. Attention
never receives physical bits or magnetic indices. This is a partially, not
fully, configuration-dependent unnormalised state; no Born sampler is implied.
All previous decoder sources remain unchanged for sealed-result reproducibility.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.local_schur import cell_schur_spec, cell_schur_tensor
from .cell_multiplicity_clebsch_tree import (
    SpinCellMultiplicityConfig, _reduced_merge, cell_multiplicity_layouts,
)
from .spin_clebsch_tree import spin_cg_tensor
from .su2_schur_config_vit import _layer_norm, _matrix


SCHUR_LABELS = ((0, 0), (0, 1), (2, 0), (2, 1), (2, 2), (4, 0))


@dataclass(frozen=True)
class CutsetAttentionConfig:
    """A single complete local cutset and a rank-R conditional reduced bank."""

    tree: SpinCellMultiplicityConfig
    cell_coordinates: tuple[tuple[float, float], ...]
    cutset_cell: int = 0
    embedding_dim: int = 4
    rank: int = 2
    scramble_seed: int = 2026100201

    def __post_init__(self) -> None:
        if (self.tree.cell_kind != 'square_plaquette' or self.tree.merge_mode != 'full_joint'
                or self.tree.target_q != 0 or self.tree.target_m2 != 0
                or self.tree.maximum_intermediate_q is not None or self.tree.projector is not None):
            raise ValueError('need an unprojected, full-q, full-joint four-spin-cell singlet tree')
        for name in ('cutset_cell', 'embedding_dim', 'rank', 'scramble_seed'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, Integral):
                raise TypeError(f'{name} must be an integer')
        if not 0 <= self.cutset_cell < self.tree.n_cells:
            raise ValueError('cutset cell is out of range')
        if self.embedding_dim < 2 or self.rank < 1 or self.scramble_seed < 0:
            raise ValueError('positive rank, embedding_dim>=2 and nonnegative seed required')
        coordinates = np.asarray(self.cell_coordinates, dtype=float)
        if coordinates.shape != (self.tree.n_cells, 2) or not np.isfinite(coordinates).all():
            raise ValueError('need a finite 2D coordinate for each cell')
        object.__setattr__(self, 'cell_coordinates', tuple(map(tuple, coordinates.tolist())))


def spatial_features(config: CutsetAttentionConfig, *, scrambled: bool = False) -> np.ndarray:
    """dx,dy,distance; scramble unordered edge features, not sites/H/CG masks.

    With two cells the sole edge cannot be scrambled: this control is then
    identically equal and has no physical discriminating power.
    """
    coordinates = np.asarray(config.cell_coordinates)
    delta = coordinates[None, :, :] - coordinates[:, None, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    features = np.concatenate((delta, distance), axis=-1) / max(1., float(distance.max()))
    if scrambled:
        i, j = np.triu_indices(config.tree.n_cells, 1)
        order = np.random.default_rng(config.scramble_seed).permutation(len(i))
        shuffled = features[i[order], j[order]].copy()
        features[i, j] = shuffled
        features[j, i] = shuffled * np.asarray((-1., -1., 1.))
    return features


def _node_cells(node) -> tuple[int, ...]:
    if hasattr(node, 'index'):
        return (node.index,)
    return _node_cells(node.left) + _node_cells(node.right)


def conditional_coefficients(
    attention: dict, config: CutsetAttentionConfig, *, scrambled: bool = False,
) -> jax.Array:
    """Six-label self-attention -> [branch, physical merge node, reduced rank]."""
    n = config.tree.n_cells
    tokens = jnp.broadcast_to(attention['mask'], (6, n, config.embedding_dim))
    tokens = tokens.at[:, config.cutset_cell].set(attention['labels'])
    norm = _layer_norm(tokens)
    query, key, value = (norm @ attention[k] for k in ('query', 'key', 'value'))
    logits = jnp.einsum('bid,bjd->bij', query, key) / np.sqrt(config.embedding_dim)
    logits += jnp.einsum('ijf,f->ij', jnp.asarray(spatial_features(config, scrambled=scrambled)),
                         attention['pair_weight'])[None]
    attended = jnp.einsum('bij,bjd->bid', jax.nn.softmax(logits, axis=-1), value)
    tokens = tokens + attended @ attention['output']
    tokens = tokens + jax.nn.gelu(_layer_norm(tokens) @ attention['mlp_in']) @ attention['mlp_out']
    tokens = _layer_norm(tokens)
    contexts = jnp.stack([tokens[:, jnp.asarray(_node_cells(node))].mean(axis=1)
                          for node in config.tree.cell_tree.internal_nodes], axis=1)
    return jnp.tanh(contexts @ attention['head'])


def init_cutset_attention(key: jax.Array, base: dict, config: CutsetAttentionConfig) -> dict:
    """Keep shared base tensors trainable; add node-specific rank-R directions."""
    keys = iter(jax.random.split(key, 12))
    d, r = config.embedding_dim, config.rank
    attention = {'labels': .3 * jax.random.normal(next(keys), (6, d), dtype=jnp.float64),
                 'mask': .3 * jax.random.normal(next(keys), (d,), dtype=jnp.float64)}
    for name in ('query', 'key', 'value', 'output'):
        attention[name] = _matrix(next(keys), d, d, .4)
    attention['pair_weight'] = .4 * jax.random.normal(next(keys), (3,), dtype=jnp.float64)
    attention['mlp_in'] = _matrix(next(keys), d, 2*d, .4)
    attention['mlp_out'] = _matrix(next(keys), 2*d, d, .4)
    attention['head'] = _matrix(next(keys), d, r, .4)
    layouts = cell_multiplicity_layouts(config.tree)
    template = {part: tuple(base[part]['merges'][layout.parameter_index] for layout in layouts)
                for part in ('real', 'imag')}
    leaves, structure = jax.tree_util.tree_flatten(template)
    random_keys = jax.random.split(next(keys), len(leaves))
    directions = jax.tree_util.tree_unflatten(structure, [
        .05 * jax.random.normal(k, (r,) + v.shape, dtype=jnp.float64)
        / np.sqrt(np.prod(v.shape[1:])) for k, v in zip(random_keys, leaves, strict=True)])
    return dict(base=base, directions=directions, attention=attention)


def _bank(base: dict, directions: dict, coefficients: jax.Array,
          config: CutsetAttentionConfig) -> dict:
    layouts = cell_multiplicity_layouts(config.tree)
    if coefficients.shape != (6, len(layouts), config.rank):
        raise ValueError('conditional coefficients have wrong branch/node/rank shape')
    bank = {}
    for part in ('real', 'imag'):
        merges = []
        for node, layout in enumerate(layouts):
            shared = base[part]['merges'][layout.parameter_index]
            merge = jax.tree_util.tree_map(
                lambda b, v: b[None] + jnp.einsum('zr,r...->z...', coefficients[:, node], v)
                / np.sqrt(config.rank), shared, directions[part][node])
            merges.append(merge)
        bank[part] = dict(merges=tuple(merges),
                          readout=jnp.broadcast_to(base[part]['readout'], (6, config.tree.feature_dim)))
    return bank


def make_branch_bank(params: dict, config: CutsetAttentionConfig, *, scrambled: bool = False) -> dict:
    """Precompute all six per-node kernels ONCE per parameter update, not per spin."""
    return _bank(params['base'], params['directions'],
                 conditional_coefficients(params['attention'], config, scrambled=scrambled), config)


def make_free_parameters(params: dict, config: CutsetAttentionConfig, *, scrambled: bool = False) -> dict:
    """Rank-matched free-coefficient control, initialized at exactly the same bank.

    Same rank/branch/kernel shapes do NOT imply exactly equal scalar counts.
    Record counts explicitly; no dummy parameters are added to manufacture a match.
    """
    return dict(base=params['base'], directions=params['directions'],
                coefficients=conditional_coefficients(params['attention'], config, scrambled=scrambled))


def make_free_branch_bank(params: dict, config: CutsetAttentionConfig) -> dict:
    """All base, directions and conditional coefficients are trainable."""
    return _bank(params['base'], params['directions'], params['coefficients'], config)


def branch_raw_amplitude(branch: dict, spins: jax.Array, config: CutsetAttentionConfig,
                         label: jax.Array | None = None) -> jax.Array:
    """One per-node tree, optionally P_(q,a) masked; no magnetic truncation.

    Reuses all existing Schur/CG/layout/merge conventions. The narrow new
    contraction adds only a complete-multiplet leaf mask and node-specific
    kernels; it does not modify the sealed, size-shared decoder.
    """
    tree_config = config.tree
    bits = jnp.asarray(spins, dtype=jnp.int32)
    if bits.shape != (tree_config.n_sites,):
        raise ValueError('wrong physical spin shape')
    valid = jnp.all((bits == 0) | (bits == 1)); bits = jnp.clip(bits, 0, 1)
    if label is not None:
        if isinstance(label, (tuple, list, np.ndarray)):
            concrete = np.asarray(label)
            if concrete.shape != (2,) or tuple(concrete.tolist()) not in SCHUR_LABELS:
                raise ValueError('absent local (q,a) multiplet; q uses doubled integers')
        label = jnp.asarray(label)
        if label.shape != (2,) or not jnp.issubdtype(label.dtype, jnp.integer):
            raise ValueError('a reduced label needs two integers, with doubled q')
        valid = valid & jnp.any(jnp.all(label == jnp.asarray(SCHUR_LABELS), axis=1))
    spec = cell_schur_spec(tree_config.cell_kind)
    schur = jnp.asarray(cell_schur_tensor(tree_config.cell_kind))
    powers = jnp.asarray(1 << np.arange(3, -1, -1), dtype=jnp.int32)
    tree = tree_config.cell_tree
    leaves = {}
    for leaf in tree.leaves:
        local_bits = bits[jnp.asarray(tree_config.cells[leaf.index])]
        m = jnp.sum(2*local_bits-1)
        local = schur[jnp.dot(local_bits, powers), :, m+4, :]
        if label is not None and leaf.index == config.cutset_cell:
            local = local * ((jnp.arange(5)[:, None] == label[0])
                             & (jnp.arange(spec.maximum_multiplicity)[None, :] == label[1]))
        leaves[leaf] = (m, local)
    outputs = []
    for part in ('real', 'imag'):
        values = dict(leaves)
        for index, (node, layout) in enumerate(zip(tree.internal_nodes, cell_multiplicity_layouts(tree_config), strict=True)):
            ml, left = values[node.left]; mr, right = values[node.right]
            cg = jnp.asarray(spin_cg_tensor(layout.left_size, layout.right_size))
            result = jnp.zeros((layout.size+1, tree_config.feature_dim), dtype=jnp.float64)
            for channel, ((ql, qr, qt), lw, rw) in enumerate(zip(layout.keys, layout.left_widths, layout.right_widths, strict=True)):
                reduced = _reduced_merge(branch[part]['merges'][index], channel, left[ql, :lw], right[qr, :rw], 'full_joint')
                result = result.at[qt].add(cg[ql, ml+layout.left_size, qr, mr+layout.right_size, qt] * reduced)
            values[node] = (ml+mr, result)
        m, root = values[tree.root]
        outputs.append(jnp.dot(root[0], branch[part]['readout']))
    return jnp.where(valid & (m == 0), outputs[0]+1j*outputs[1], 0j)


def cutset_raw_amplitude(bank: dict, spins: jax.Array, config: CutsetAttentionConfig) -> jax.Array:
    """Coherently sum six projected branches; never mix branch probabilities."""
    return jnp.sum(jax.vmap(lambda branch, label: branch_raw_amplitude(branch, spins, config, label))(
        bank, jnp.asarray(SCHUR_LABELS, dtype=jnp.int32)))


def make_centered_additive(carrier_fn: Callable, residual_fn: Callable,
                           reference_bank: dict, *, eta: float = 1.) -> Callable:
    """Carrier(bits)+eta*(D(bank,bits)-D(reference,bits)), with fixed reference.

    eta=1, bank=reference preserves the carrier with a nonzero possible bank
    tangent. eta=0 is a replay test only: every residual-parameter tangent is zero.
    Caller must supply singlet/A1 carrier and an already A1-projected residual.
    """
    if not np.isfinite(eta):
        raise ValueError('eta must be finite')
    reference = jax.tree_util.tree_map(jax.lax.stop_gradient, reference_bank)
    def amplitude(bank, bits):
        if eta == 0:
            return carrier_fn(bits)
        return carrier_fn(bits) + eta*(residual_fn(bank, bits)-residual_fn(reference, bits))
    return amplitude
