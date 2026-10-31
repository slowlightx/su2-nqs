"""Quality-preserving, first-order-live width expansion of a ClebschTree.

The physical tree, doubled ``q=2j`` cutoff and magnetic CG tensors are fixed.
Old reduced features are embedded exactly. Newly available output features
are initialized, but all old outputs annihilate newly opened input channels.
The old readout is zero-padded. Thus the initial physical state is unchanged,
while derivatives of new downstream weights can already access the new
features. This is not a new ansatz beyond the wider free reduced-tensor model.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .clebsch_tree_transfer import transfer_clebsch_parameters
from .literature_clebsch_tree import (
    LiteratureClebschTreeConfig, clebsch_layers, init_literature_clebsch_tree,
)


def initialize_live_clebsch_expansion(
    source_config: LiteratureClebschTreeConfig,
    source: Mapping[str, Any],
    target_config: LiteratureClebschTreeConfig,
    key: jax.Array,
    *, new_feature_scale: float = 0.2,
) -> tuple[dict, dict]:
    """Widen a same-tree, same-cutoff state without changing any amplitude.

    All labels use doubled integers. Only reduced multiplicity dimensions
    may increase. Existing coefficients and old physical feature directions
    are retained exactly; no layer rescaling, magnetic parameter, projection
    approximation or stochastic state fitting is performed.

    ``new_feature_scale`` controls only the initially unused output rows. A
    fresh optimizer must be registered for the larger parameter space; this
    function does not claim to preserve a narrow optimizer's state.
    """
    if (source_config.tree != target_config.tree
            or source_config.maximum_q != target_config.maximum_q
            or (source_config.target_q, source_config.target_m2)
            != (target_config.target_q, target_config.target_m2)):
        raise ValueError("live expansion requires the same tree, cutoff and target (q,m2)")
    if not np.isfinite(new_feature_scale) or new_feature_scale <= 0:
        raise ValueError("new_feature_scale must be positive and finite")
    old_layers, new_layers = clebsch_layers(source_config), clebsch_layers(target_config)
    if any(any(a > b for a, b in zip(old.widths, new.widths, strict=True))
           for old, new in zip(old_layers, new_layers, strict=True)):
        raise ValueError("live expansion cannot remove a reduced output channel")
    random, _ = init_literature_clebsch_tree(key, target_config)
    embedded, copy = transfer_clebsch_parameters(
        source_config, source, target_config, random, mode="embedding")
    banks, records = [], []
    for old, new, zero, seed in zip(old_layers, new_layers,
                                  embedded["merges"], random["merges"], strict=True):
        offset, pieces, initialized = 0, [], 0
        for (q_left, q_right, q_out), shape in zip(new.keys, new.shapes, strict=True):
            count = int(np.prod(shape))
            block = zero[offset:offset + count].reshape(shape)
            start = old.widths[q_out]
            if start < shape[0]:
                # Initialize only output rows orthogonal to the old coordinate
                # subspace. Old rows (including ALL new-child columns) stay as
                # in the exact zero-padded inclusion.
                fresh = seed[offset:offset + count].reshape(shape)
                fresh = fresh * (new_feature_scale / np.sqrt(shape[1] * shape[2]))
                block = block.at[start:, :, :].set(fresh[start:, :, :])
                initialized += (shape[0] - start) * shape[1] * shape[2]
            pieces.append(block.reshape(-1))
            offset += count
        banks.append(jnp.concatenate(pieces))
        records.append(dict(old_widths=list(old.widths), new_widths=list(new.widths),
                            initialized_unused_output_scalars=initialized))
    result = dict(merges=tuple(banks), readout=embedded["readout"])
    return result, dict(
        exact_same_physical_state=True, same_tree=True, same_doubled_cutoff=True,
        magnetic_parameters=False, requires_new_optimizer_registration=True,
        complete_A1_unchanged_by_linearity=True, new_feature_scale=new_feature_scale,
        new_rows_do_not_change_old_features=True, layer_rescaling_performed=False,
        source_parameters=sum(int(v.size) for v in jax.tree.leaves(source)),
        target_parameters=sum(int(v.size) for v in jax.tree.leaves(result)),
        copied_parameters=copy["copied_scalar_parameters"], layers=records,
        tangent_gain_is_to_be_measured_not_assumed=True,
    )
