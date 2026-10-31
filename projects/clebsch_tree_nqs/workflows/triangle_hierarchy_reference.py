"""Supplementary matched-width Schmidt-copy reference for CT12 transfer.

This follows the already registered target-H initializer, without altering
its arms or evidence. Each raw CT6 parent is optimally Schmidt-compressed
across its existing 18+18 cut (per fixed q width). Four such RAW singlets
are tiled on the target cover; they are not four projected A1 parents and
not necessarily rectangular patches. All q>0 capacity remains available,
but the initial 72/144 product couples only the q0 first columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout
from su2_nqs.operators.su2_sector_merge import project_sector_maps
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_hierarchy_initialization import (
    _block_hash, complete_active_columns, native_bank_from_state, retained_widths,
    source_channels, split_bond_ledger,
)
from .triangle_sector_transfer import common_target_bank, extract_merge_map, load_source


def schmidt_source_maps(source: dict, active: int) -> tuple[tuple[dict, dict], dict, dict]:
    """Compress raw CT6 exactly by per-q SVD, preserving complex phases.

    The root tensor contracts without conjugation. Consequently C=U s Vh
    requires the RIGHT child map Vh.T (not Vh.conj().T).
    """
    cfg = MixedClebschConfig(6, 20, 10)
    layout = mixed_clebsch_layout(cfg)
    left = layout[0]["widths"]; right = layout[1]["widths"]
    root_channels = source_channels(left, right, 0)
    root = extract_merge_map(source, cfg, 2, 0, root_channels)[:, 0]
    lookup = {label: root[i] for i, label in enumerate(root_channels)}
    maps = ({}, {}); singular = {}; original_weight = retained_weight = 0.
    error = 0.
    for q in sorted(left):
        coefficients = np.asarray([[lookup[q, a, q, b] for b in range(right[q])]
                                   for a in range(left[q])])
        u, values, vh = np.linalg.svd(coefficients, full_matrices=False)
        count = min(active, len(values)); singular[q] = values[:count]
        original_weight += float(np.sum(values**2)); retained_weight += float(np.sum(values[:count]**2))
        error = max(error, float(np.max(abs(u @ np.diag(values) @ vh-coefficients))))
        for child, rotation in ((0, u[:, :count]), (1, vh[:count].T)):
            row = layout[child]
            leaf = lambda i: {qi: w.shape[1] for qi, w in zip(cfg.local_q, source["local"][i], strict=True)}
            channels = source_channels(leaf(row["left"]), leaf(row["right"]), q)
            maps[child][q] = extract_merge_map(source, cfg, child, q, channels) @ rotation
    if retained_weight <= 0 or original_weight <= 0:
        raise ArithmeticError("source raw CT6 has zero norm")
    singular = {q: x/np.sqrt(retained_weight) for q, x in singular.items()}
    return maps, singular, dict(source_raw_norm_squared=original_weight,
        retained_raw_norm_squared=retained_weight,
        normalized_raw_source_fidelity=retained_weight/original_weight,
        SVD_reconstruction_error=error, per_q_retained_singular_values={q: x.tolist() for q, x in singular.items()},
        exact_projected_A1_fidelity_known=False)


def run(output: Path, source_run: Path, active: int = 4) -> dict:
    """Construct two deterministic raw-product references, no optimization."""
    if active not in (4, 8):
        raise ValueError("active must be4 or8")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing supplementary reference")
    atomic_json(output/"registration.json", dict(task="same_width_Schmidt_raw_CT6_product_reference",
        active=active, couplings=(.125, 0.), seed=2610231511, source_run=str(source_run),
        workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_training=True, no_remote_submission=True, full_A1_energy_and_variance_not_established=True))
    tree = triangular_schur_hierarchy(12); cfg = MixedClebschConfig(12, 20, 10)
    reports = []; started = perf_counter()
    for j2 in (.125, 0.):
        tag = "j0125" if j2 else "j0"
        source, provenance = load_source(j2)
        old = load_checkpoint(source_run/tag/"physical"/"state.pkl")["params"]
        blocks = list(old["blocks"][:16]); local_maps, singular, quality = schmidt_source_maps(source, active)
        _, joins, bonds = split_bond_ledger(12, j2)
        state = dict(j2=j2, arm="schmidt_copy", active=active, blocks=blocks,
                     maps={}, channels={}, completed_nodes=0, records=[], physical_bonds=len(bonds))
        common = common_target_bank(source); cache = {}
        for node, (left, right, _) in enumerate(tree.binary_merges):
            l, r = blocks[left], blocks[right]
            keep = retained_widths(l.q_widths, r.q_widths, active, node == 14)
            channels = {q: source_channels(l.q_widths, r.q_widths, q) for q in keep}
            tick = perf_counter(); rng = np.random.default_rng(2610231511+node)
            if node < 8:
                maps = local_maps[node % 2]
            else:
                maps = {}
                for q, width in keep.items():
                    candidate = rng.normal(size=(len(channels[q]), width))
                    if q == 0:
                        first = np.zeros((len(channels[q]), 1))
                        if node < 12:
                            for i, (ql, a, qr, b) in enumerate(channels[q]):
                                if ql == qr and a == b:
                                    first[i, 0] = singular[ql][a]
                        else:
                            first[channels[q].index((0, 0, 0, 0)), 0] = 1.
                        maps[q] = complete_active_columns(first, candidate)
                    else:
                        maps[q] = np.linalg.qr(candidate, mode="reduced")[0]
            key = (_block_hash(l), _block_hash(r), joins[node],
                   tuple((q, hashlib.sha256(np.asarray(w).tobytes()).hexdigest()) for q, w in maps.items()))
            if key in cache:
                merged = cache[key]
            else:
                merged = project_sector_maps(l, r, joins[node], maps=maps, compact_boundary=True,
                    max_output_dimension=active*36, memory_limit_bytes=768*1024**2)
                cache[key] = merged
            blocks.append(merged); state["maps"][node] = maps; state["channels"][node] = channels
            state["completed_nodes"] += 1
            state["records"].append(dict(node=node, parent_sites=merged.n_sites, seconds=perf_counter()-tick))
        native, embedding = native_bank_from_state(source, state)
        report = dict(j2=j2, source=provenance, schmidt_compression=quality, embedding=embedding,
                      raw_tree_energy_per_site=float(blocks[-1].h_by_q[0][0, 0].real)/144,
                      raw_source_product_not_projected_parent_product=True,
                      A1_energy_and_variance_known=False, records=state["records"])
        save_checkpoint(output/tag/"state.pkl", state, metadata=report)
        save_checkpoint(output/tag/"initialized_bank.pkl", native, optimizer_state=None,
                        metadata=dict(report, model=dict(length=12, multiplicity_cap=20, maximum_q=10),
                                      optimizer_reset_required=True))
        atomic_json(output/tag/"native_metadata.json", report)
        reports.append(report)
        atomic_json(output/"results.json", dict(reports=reports, seconds=perf_counter()-started))
        print(j2, report["raw_tree_energy_per_site"], quality["normalized_raw_source_fidelity"], flush=True)
    return dict(reports=reports)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--active", type=int, default=4)
    a = p.parse_args(); run(a.output, a.source_run, a.active)
