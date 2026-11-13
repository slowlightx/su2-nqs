"""Offline whole-multiplet compression into actual compact CT arrays.

Only multiplicity coordinates are changed. Doubled spins ``q=2*j``, the
Condon--Shortley tensors, physical sites and final singlet are unchanged.
The environment criterion is for one raw tree, not its coherent A1 sum.
Input boundary-response vectors must use the supplied bank's virtual basis;
canonicalize before constructing those vectors, rather than after them.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .triangular_clebsch_gauge import (
    _pack, _unpack, gauge_triangular_edge, triangular_edge_spectrum,
    triangular_raw_overlap,
)
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout


def _edge_widths(config: MixedClebschConfig) -> tuple[dict[int, int], ...]:
    rows = mixed_clebsch_layout(config)
    count = len(rows) + 1
    from ..su2.triangle_blocks import nine_site_output
    local = {q: nine_site_output(q, int(config.multiplicity_cap)).width
             for q in config.local_q}
    return tuple([dict(local) for _ in range(count)] +
                 [dict(row["widths"]) for row in rows])


def _compatible(source: MixedClebschConfig, target: MixedClebschConfig) -> None:
    if source.length != target.length or source.maximum_q != target.maximum_q:
        raise ValueError("compression must retain the same geometry and doubled-q cutoff")
    if target.multiplicity_cap > source.multiplicity_cap:
        raise ValueError("compact compression cannot increase multiplicity cap")
    for old, new in zip(_edge_widths(source), _edge_widths(target), strict=True):
        if old.keys() != new.keys() or any(new[q] > old[q] for q in old):
            raise ValueError("all doubled-q sectors must survive with nonincreasing widths")


def repack_compact_triangular_bank(
    padded_bank: Mapping, source_config: MixedClebschConfig,
    target_config: MixedClebschConfig, *, zero_tolerance: float = 0.,
) -> dict:
    """Remove physically zero producer tails and repack native charge buckets.

    A child producer must vanish outside the target multiplicity width.
    Parent input tails may be nonzero: they multiply those zero features.
    No magnetic state within a retained doubled-q multiplet is removed.
    This function does not itself select a retained physical subspace.
    """
    _compatible(source_config, target_config)
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("finite nonnegative zero tolerance required")
    local, top = _unpack(padded_bank, source_config)
    widths = _edge_widths(target_config)
    for edge, node in enumerate(local):
        for q, tensor in node.items():
            keep = widths[edge][q]
            if np.max(abs(tensor[:, keep:]), initial=0.) > zero_tolerance:
                raise ValueError("local producer has a nonzero discarded multiplicity tail")
            node[q] = tensor[:, :keep].copy()
    count = len(local)
    for i, (node, row) in enumerate(zip(top, mixed_clebsch_layout(source_config), strict=True)):
        for key, tensor in node.items():
            ql, qr, qt = key
            keep = widths[count+i][qt]
            if np.max(abs(tensor[keep:]), initial=0.) > zero_tolerance:
                raise ValueError("upper producer has a nonzero discarded multiplicity tail")
            node[key] = tensor[:keep, :widths[row["left"]][ql],
                               :widths[row["right"]][qr]].copy()
    compact = _pack(local, top, target_config)
    _unpack(compact, target_config)
    return compact


def _protected_rotation(block, keep: int, protected: np.ndarray) -> tuple[np.ndarray, dict]:
    """Constrain a raw Schmidt selection by a physical response subspace."""
    width = len(block.multiplicity_weights)
    vectors = np.asarray(protected)
    if vectors.ndim != 2 or vectors.shape[0] != width or np.any(~np.isfinite(vectors)):
        raise ValueError("protected columns must be finite ket coefficients in the source edge basis")
    rank_metric = block.metric_rank
    canonical = block.gauge_inverse @ vectors
    active = canonical[:rank_metric]
    u, singular, _ = np.linalg.svd(active, full_matrices=False)
    threshold = 1e-12 * float(np.max(singular, initial=0.))
    rank = int(np.count_nonzero(singular > threshold))
    if rank > keep:
        raise ValueError("target width cannot contain the protected response rank")
    if rank_metric == 0 or rank == 0:
        return np.eye(width), {"protected_rank": 0, "protected_projector_defect": 0.}
    if keep >= rank_metric:
        return np.eye(width), {"protected_rank": rank, "protected_projector_defect": 0.}
    protected_basis = u[:, :rank]
    complete, _ = np.linalg.qr(protected_basis, mode="complete")
    complement = complete[:, rank:]
    weights = np.asarray(block.multiplicity_weights[:rank_metric])
    residual_density = complement.conj().T @ (weights[:, None]*complement)
    _, eigenvectors = np.linalg.eigh((residual_density+residual_density.conj().T)/2)
    selected = np.column_stack((protected_basis, complement @ eigenvectors[:, ::-1][:, :keep-rank]))
    active_rotation, _ = np.linalg.qr(selected, mode="complete")
    rotation = np.eye(width, dtype=np.result_type(active_rotation.dtype, np.float64))
    rotation[:rank_metric, :rank_metric] = active_rotation
    selected = active_rotation[:, :keep]
    defect = np.linalg.norm(active-selected @ (selected.conj().T @ active))
    relative = float(defect/max(float(np.linalg.norm(active)), np.finfo(float).tiny))
    if relative > 2e-10:
        raise ArithmeticError("protected boundary-response subspace was not retained")
    return rotation, {"protected_rank": rank, "protected_projector_defect": relative}


def compress_compact_triangular_bank(
    bank: Mapping, source_config: MixedClebschConfig,
    target_config: MixedClebschConfig, *,
    protected_subspaces: Mapping[int, Mapping[int, np.ndarray]] | None = None,
) -> tuple[dict, dict]:
    """Sequentially select whole q multiplets, then shrink actual array shapes.

    ``protected_subspaces[edge][q]`` contains column vectors in the INPUT
    bank's ket virtual basis on that edge. Protects their span inside the
    available physical edge space, then fills the remaining budget by the
    largest complementary raw Schmidt weight. Lower-edge truncation may
    already change an ancestor's physical support; this is not a guarantee
    that an arbitrary external H-response survives all compressions.

    All source q sectors remain, including legal q>0 below the singlet root.
    Source should be block-canonicalized first if its gauge is ill-conditioned.
    Canonicalizing or compressing changes optimizer coordinates and requires
    explicit optimizer reset/transport, never a silent old-moment copy.
    """
    _compatible(source_config, target_config)
    local, top = _unpack(bank, source_config)
    current = _pack(local, top, source_config)
    old_widths, new_widths = _edge_widths(source_config), _edge_widths(target_config)
    protected_subspaces = {} if protected_subspaces is None else dict(protected_subspaces)
    if any(isinstance(e, bool) or not isinstance(e, (int, np.integer)) or
           not 0 <= e < len(old_widths)-1 for e in protected_subspaces):
        raise ValueError("protected subspaces require a nonroot integer hierarchy edge")
    original_norm = triangular_raw_overlap(current, current, source_config).real
    if not np.isfinite(original_norm) or original_norm <= 0:
        raise ValueError("source raw tree must have a positive finite norm")
    records = []
    for edge, widths in enumerate(new_widths[:-1]):
        constraints = protected_subspaces.get(edge, {})
        if set(constraints)-set(widths):
            raise ValueError("protected subspace references an absent doubled-q block")
        for q, columns in constraints.items():
            values = np.asarray(columns)
            if (isinstance(q, bool) or not isinstance(q, (int, np.integer)) or
                    values.ndim != 2 or values.shape[0] != old_widths[edge][q] or
                    np.any(~np.isfinite(values))):
                raise ValueError("finite protected columns with integer doubled-q labels required")
        if old_widths[edge] == widths:
            continue
        spectrum = triangular_edge_spectrum(current, source_config, edge)
        gauges, summaries = {}, {}
        for q, block in spectrum.blocks.items():
            keep = widths[q]
            rotation = np.eye(len(block.multiplicity_weights))
            report = {"protected_rank": 0, "protected_projector_defect": 0.}
            if q in constraints:
                rotation, report = _protected_rotation(block, keep, constraints[q])
            gauges[q] = np.real_if_close(block.gauge @ rotation, tol=1000)
            retained = np.trace(rotation[:, :keep].conj().T @
                                (block.multiplicity_weights[:, None]*rotation[:, :keep])).real
            summaries[q] = {"old_copies": len(block.multiplicity_weights),
                            "retained_copies": keep, "retained_raw_weight": float(retained), **report}
        current = gauge_triangular_edge(current, source_config, edge, gauges)
        local, top = _unpack(current, source_config)
        if edge < len(local):
            for q, tensor in local[edge].items():
                tensor[:, widths[q]:] = 0.
        else:
            for key, tensor in top[edge-len(local)].items():
                tensor[widths[key[2]]:] = 0.
        current = _pack(local, top, source_config)
        records.append({"edge": edge, "q_blocks": summaries,
                        "conditional_raw_retained_weight": sum(x["retained_raw_weight"] for x in summaries.values())})
    compact = repack_compact_triangular_bank(current, source_config, target_config)
    new_norm = triangular_raw_overlap(current, current, source_config).real
    if not np.isfinite(new_norm) or new_norm <= 0:
        raise ArithmeticError("selected compact subspaces annihilated the raw state")
    overlap = triangular_raw_overlap(bank, current, source_config)
    fidelity = float(abs(overlap)**2/(original_norm*new_norm))
    old_count = sum(x.size for node in local for x in node.values()) + sum(x.size for node in top for x in node.values())
    clocal, ctop = _unpack(compact, target_config)
    new_count = sum(x.size for node in clocal for x in node.values()) + sum(x.size for node in ctop for x in node.values())
    return compact, {"source_cap": int(source_config.multiplicity_cap),
                     "target_cap": int(target_config.multiplicity_cap),
                     "maximum_q": int(source_config.maximum_q), "root_q": 0,
                     "all_source_q_sectors_retained": True,
                     "complete_magnetic_multiplets_retained": True,
                     "source_parameters": int(old_count), "target_parameters": int(new_count),
                     "padded_storage_unchanged": False,
                     "raw_squared_fidelity": fidelity,
                     "raw_norm_ratio": float(new_norm/original_norm),
                     "projected_fidelity_not_inferred": True,
                     "optimizer_coordinates_changed_require_reset": True,
                     "edge_selections": records}


__all__ = ["repack_compact_triangular_bank", "compress_compact_triangular_bank"]
