"""Physical raw-Born sampler adapter for frozen active ClebschTree banks."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..models.triangular_clebsch_active_layout import mixed_active_layout
from ..models.triangular_clebsch_gauge import _nodes
from ..models.triangular_mixed_clebsch import MixedClebschConfig
from ..su2.reduced_tree_environment import ReducedSingletTree
from ..su2.triangle_blocks import triangle_block_tables
from .raw_tree_born import RawTreeBornSampler


def make_triangular_raw_born_sampler(bank: dict, config: MixedClebschConfig, *,
        active_top_widths: Sequence[Mapping[int, int]], memory_limit_bytes: int = 768*1024**2,
        shared_tensor_cache: dict | None = None) -> RawTreeBornSampler:
    """Build an exact physical sampler from an already-active isometric bank.

    This is NOT a hidden fusion-path draw or a product of leaf diagonals.
    Every internal q/multiplicity interference is retained by the sampler.
    """
    geometry = triangular_schur_hierarchy(config.length)
    tree, nodes = _nodes(config)
    layout = mixed_active_layout(config, active_top_widths)
    if len(bank['local']) != len(geometry.blocks) or len(bank['top']) != len(layout):
        raise ValueError('active bank must match the complete physical hierarchy')
    tables = triangle_block_tables()['nine']
    physical_m = np.asarray([2*i.bit_count()-9 for i in range(512)])
    grams = []; isometries = []; leaf_cache = {}
    for local in bank['local']:
        key = tuple((np.asarray(w).shape, np.asarray(w).dtype.str, np.asarray(w).tobytes()) for w in local)
        if key not in leaf_cache:
            gram = {}; iso = {}
            for q, weights in zip(config.local_q, local, strict=True):
                w = np.asarray(weights); value = tables[q]@w
                gram[q] = w.conj().T@w
                iso[q] = np.stack([value*(physical_m == m)[:, None] for m in range(-q, q+1, 2)], axis=-1)
            leaf_cache[key] = gram, iso
        gram, iso = leaf_cache[key]; grams.append(gram); isometries.append(iso)
    kernels = {}
    for row, groups in zip(layout, bank['top'], strict=True):
        data = {}
        for group, values in zip(row['groups'], groups, strict=True):
            array = np.asarray(values)
            if array.shape != group['shape']:
                raise ValueError('active bank must have the declared exact q-width profile')
            data.update(zip(group['keys'], array, strict=True))
        kernels[nodes[row['parent']]] = data
    state = ReducedSingletTree(tree, tuple(grams), tuple(kernels[n] for n in tree.internal_nodes), np.ones(1))
    return RawTreeBornSampler(state, isometries, tuple(block.sites for block in geometry.blocks),
        memory_limit_bytes=memory_limit_bytes, shared_tensor_cache=shared_tensor_cache)
