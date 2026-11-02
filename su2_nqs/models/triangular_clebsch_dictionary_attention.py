"""Static spatial-attention generation of full-rank SU(2) reduced tensors.

Only geometry, hierarchy and doubled-q/C3 metadata enter the coefficient
network. Queried magnetic bits and global reduced configurations NEVER enter.
A dictionary rank is across NODES, not a rank restriction on virtual legs.
At fixed virtual widths this remains inside the same free tensor-tree family.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.triangle_blocks import triangle_block_tables, nine_site_output
from .triangular_mixed_clebsch import (
    MixedClebschConfig, local_c3_bases, mixed_clebsch_layout,
)


@dataclass(frozen=True)
class TriangleDictionaryAttentionConfig:
    """Static hypernetwork; q=2j throughout and no magnetic indices learned."""
    model: MixedClebschConfig = MixedClebschConfig()
    dictionary_rank: int = 16
    embedding_dim: int = 16
    heads: int = 2
    depth: int = 1
    architecture: str = 'attention'
    local_symmetry: str = 'free_joint'
    geometry_harmonics: int = 2

    def __post_init__(self) -> None:
        for name in ('dictionary_rank', 'embedding_dim', 'heads', 'depth'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(name+' must be a positive integer')
        if self.embedding_dim % self.heads:
            raise ValueError('embedding dimension must divide into complete heads')
        if self.architecture not in ('attention', 'linear'):
            raise ValueError('architecture must be attention or linear')
        if self.local_symmetry not in ('free_joint', 'C3'):
            raise ValueError('local_symmetry must be free_joint or C3')
        if (isinstance(self.geometry_harmonics, bool)
                or not isinstance(self.geometry_harmonics, Integral)
                or self.geometry_harmonics < 1):
            raise ValueError('geometry_harmonics must be a positive integer')


@lru_cache(maxsize=None)
def dictionary_layout(config: TriangleDictionaryAttentionConfig) -> tuple[dict, ...]:
    """Group only identical full tensor shapes, charge keys and hierarchy scales."""
    cfg = config.model; tree = triangular_schur_hierarchy(cfg.length)
    local_shapes = (tuple((len(b),) for b in local_c3_bases(cfg)) if config.local_symmetry == 'C3'
        else tuple((triangle_block_tables()['nine'][q].shape[1],
                    nine_site_output(q, cfg.multiplicity_cap).width) for q in cfg.local_q))
    nodes = [dict(role='local', shapes=local_shapes, support=9, q=cfg.local_q)
             for _ in tree.blocks]
    for row in mixed_clebsch_layout(cfg):
        nodes.append(dict(role='top', shapes=tuple(g['shape'] for g in row['groups']),
            support=len(tree.coarse_supports[row['parent']]), q=tuple(row['widths']),
            charge_keys=tuple(g['keys'] for g in row['groups'])))
    types: dict[tuple, dict] = {}
    for node_id, node in enumerate(nodes):
        identity = (node['role'], node['support'], node['shapes'], node['q'], node.get('charge_keys'))
        if identity not in types:
            types[identity] = dict(node, node_ids=[], vector_size=sum(int(np.prod(s)) for s in node['shapes']))
        types[identity]['node_ids'].append(node_id)
    return tuple(dict(row, node_ids=tuple(row['node_ids']),
        rank=min(config.dictionary_rank, len(row['node_ids']))) for row in types.values())


@lru_cache(maxsize=None)
def dictionary_geometry(config: TriangleDictionaryAttentionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Immutable node features and periodic120-degree distances, never spin bits."""
    cfg = config.model; tree = triangular_schur_hierarchy(cfg.length)
    supports = tree.coarse_supports; layout = mixed_clebsch_layout(cfg)
    centers = []; tokens = []
    for index, support in enumerate(supports):
        coordinates = np.asarray([divmod(s, cfg.length) for s in support], float)
        circular = np.mean(np.exp(2j*np.pi*coordinates/cfg.length), axis=0)
        center = np.where(abs(circular) > 1e-12, np.angle(circular)*cfg.length/(2*np.pi), 0.)
        centers.append(center)
        fourier = [f(2*np.pi*(kr*center[0]+kc*center[1])/cfg.length)
                   for kr in range(config.geometry_harmonics+1)
                   for kc in range(config.geometry_harmonics+1)
                   if kr or kc for f in (np.cos, np.sin)]
        local = index < len(tree.blocks)
        charges = cfg.local_q if local else tuple(layout[index-len(tree.blocks)]['widths'])
        tokens.append([*fourier, np.log2(len(support))/np.log2(tree.n_sites),
            float(local), float(index == len(supports)-1), float(config.local_symmetry == 'C3'),
            *[float(q in charges) for q in range(cfg.maximum_q+1)]])
    centers = np.asarray(centers)
    delta = centers[:, None]-centers[None, :]
    distances = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            r, c = delta[..., 0]+a*cfg.length, delta[..., 1]+b*cfg.length
            distances.append(r*r+c*c-r*c)
    scale = np.sqrt(np.asarray([len(s) for s in supports], float))
    metric = np.sqrt(np.maximum(np.min(distances, axis=0), 0.))/np.sqrt(scale[:, None]*scale[None, :])
    return np.asarray(tokens, np.float64), np.asarray(metric, np.float64)


def _norm(values: jax.Array) -> jax.Array:
    return (values-values.mean(-1, keepdims=True))*jax.lax.rsqrt(values.var(-1, keepdims=True)+1e-6)


def dictionary_node_features(params: dict, config: TriangleDictionaryAttentionConfig) -> jax.Array:
    """Node features from fixed geometry only; no queried state/configuration argument."""
    tokens, distance = dictionary_geometry(config)
    if config.architecture == 'linear':
        hidden = jnp.asarray(tokens)
    else:
        hidden = jnp.asarray(tokens) @ params['encoder']['embedding']
        d, h = config.embedding_dim, config.heads
        for layer in params['encoder']['layers']:
            normalized = _norm(hidden)
            query, key, value = [(normalized @ layer[k]).reshape(len(hidden), h, d//h)
                                 for k in ('query', 'key', 'value')]
            logits = jnp.einsum('ihd,jhd->hij', query, key)/np.sqrt(d//h)
            logits -= jax.nn.softplus(layer['raw_distance_decay'])[:, None, None]*jnp.asarray(distance)[None]
            mixed = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, axis=-1), value).reshape(len(hidden), d)
            hidden = hidden+mixed @ layer['output']
            hidden = hidden+jax.nn.gelu(_norm(hidden) @ layer['mlp_in']) @ layer['mlp_out']
        hidden = _norm(hidden)
    return jnp.column_stack((hidden, jnp.ones(len(hidden), jnp.float64)))


def _bank_vectors(bank: dict, config: TriangleDictionaryAttentionConfig) -> tuple[np.ndarray, ...]:
    """Offline vectorization; reject, never silently project, a non-C3 parent."""
    cfg = config.model; values = []
    if config.local_symmetry == 'C3':
        bases = local_c3_bases(cfg)
        for node in bank['local']:
            coefficients = []
            for basis, weight in zip(bases, node, strict=True):
                c = np.einsum('pik,ik->p', basis, np.asarray(weight))
                residual = np.einsum('p,pik->ik', c, basis)-np.asarray(weight)
                if np.linalg.norm(residual) > 2e-11*max(np.linalg.norm(weight), 1e-100):
                    raise ValueError('parent is not in the requested local C3 intertwiner space')
                coefficients.append(c)
            values.append(np.concatenate([v.ravel() for v in coefficients]))
    else:
        values.extend(np.concatenate([np.asarray(w).ravel() for w in node]) for node in bank['local'])
    values.extend(np.concatenate([np.asarray(w).ravel() for w in node]) for node in bank['top'])
    if len(values) != len(triangular_schur_hierarchy(cfg.length).coarse_supports):
        raise ValueError('parent hierarchy does not match model')
    return tuple(values)


def init_dictionary_attention_from_bank(
    key: jax.Array, bank: dict, config: TriangleDictionaryAttentionConfig,
) -> tuple[dict, dict]:
    """Offline same-bank initialization if R>=node count; otherwise explicit SVD.

    Full-R dictionary entries are the original complete W tensors; an offline
    geometry-feature linear solve initializes c to the identity. No per-node
    full correction, carrier residual or hidden array is retained in params.
    """
    # Layer normalization makes each hidden row sum to zero. With the
    # appended bias, its design rank is at most embedding_dim, not d+1.
    # A full-node dictionary initialized with an identity coefficient matrix
    # needs this many independent design directions. Reject the impossible
    # case before allocating dictionaries or invoking an optimizer.
    if config.architecture == 'attention' and any(
            item['rank'] == len(item['node_ids']) > config.embedding_dim
            for item in dictionary_layout(config)):
        raise ValueError('attention feature rank cannot reproduce the full-node dictionary; '
                         'increase embedding capacity')
    tokens, _ = dictionary_geometry(config); d = config.embedding_dim
    keys = iter(jax.random.split(key, 1+6*config.depth))
    matrix = lambda a, b: jax.random.normal(next(keys), (a, b), dtype=jnp.float64)/np.sqrt(a)
    encoder = (dict(embedding=matrix(tokens.shape[1], d), layers=tuple(dict(
        **{name: matrix(d, d) for name in ('query', 'key', 'value', 'output')},
        mlp_in=matrix(d, 2*d), mlp_out=matrix(2*d, d),
        raw_distance_decay=jnp.ones(config.heads, jnp.float64)) for _ in range(config.depth)))
        if config.architecture == 'attention' else {})
    features = np.asarray(dictionary_node_features(dict(encoder=encoder), config))
    original = _bank_vectors(bank, config); dictionaries = []; heads = []; errors = []; conditions = []; feature_ranks = []
    for item in dictionary_layout(config):
        rows = np.asarray([original[i] for i in item['node_ids']])
        if rows.shape[1] != item['vector_size']:
            raise ValueError('parent full tensor shape does not match declared virtual widths')
        rank = item['rank']
        if rank == len(rows):
            basis, coefficients = rows, np.eye(rank)
        else:
            u, s, vh = np.linalg.svd(rows, full_matrices=False)
            basis, coefficients = vh[:rank], u[:, :rank]*s[:rank]
        design = features[np.asarray(item['node_ids'])]
        head, _, design_rank, _ = np.linalg.lstsq(design, coefficients, rcond=None)
        fit_error = np.linalg.norm(design @ head-coefficients)/max(np.linalg.norm(coefficients), 1e-100)
        if fit_error > 2e-11:
            raise ValueError('static feature readout lacks rank for same-bank initialization; increase embedding capacity')
        dictionaries.append(jnp.asarray(basis)); heads.append(jnp.asarray(head))
        errors.append(float(np.linalg.norm(coefficients @ basis-rows)/max(np.linalg.norm(rows), 1e-100)))
        conditions.append(float(np.linalg.cond(design)))
        feature_ranks.append(int(design_rank))
    params = dict(encoder=encoder, dictionaries=tuple(dictionaries), coefficient_heads=tuple(heads))
    return params, dict(exact_same_bank_expected=max(errors, default=0.) < 2e-11,
        per_type_node_sharing_approximation_errors=errors, feature_condition_numbers=conditions,
        feature_readout_ranks=feature_ranks,
        dictionary_ranks=[x['rank'] for x in dictionary_layout(config)],
        virtual_leg_rank_factorization=False, full_per_node_residual=False,
        local_symmetry=config.local_symmetry, geometry_only=True)


def dictionary_attention_bank(params: dict, config: TriangleDictionaryAttentionConfig) -> dict:
    """Generate one static complete reduced W bank, once per parameter update.

    Local C3 mode synthesizes verified intertwiners; free_joint mode can replay
    any existing joint bank and does not pretend that local C3 was imposed.
    Both keep the full three-spin q=1 multiplicity pair and the root singlet.
    Full physical A1 projection is still required outside this generator.
    """
    features = dictionary_node_features(params, config)
    nodes: list = [None]*len(triangular_schur_hierarchy(config.model.length).coarse_supports)
    for item, basis, head in zip(dictionary_layout(config), params['dictionaries'], params['coefficient_heads'], strict=True):
        coefficients = features[jnp.asarray(item['node_ids'])] @ head
        vectors = coefficients @ basis
        for local_id, node_id in enumerate(item['node_ids']):
            pieces = []; offset = 0
            for shape in item['shapes']:
                count = int(np.prod(shape)); pieces.append(vectors[local_id, offset:offset+count].reshape(shape)); offset += count
            if item['role'] == 'local' and config.local_symmetry == 'C3':
                pieces = [jnp.einsum('p,pik->ik', c, jnp.asarray(b))
                          for c, b in zip(pieces, local_c3_bases(config.model), strict=True)]
            nodes[node_id] = tuple(pieces)
    split = len(triangular_schur_hierarchy(config.model.length).blocks)
    return dict(local=tuple(nodes[:split]), top=tuple(nodes[split:]))


def dictionary_parameter_count(params: object) -> int:
    """Count ALL stored trainable scalars, including dictionary and coefficient network."""
    return sum(int(x.size) for x in jax.tree.leaves(params))
