"""Bounded L12/18/24 width-four raw-H initialization, with identical CT6 leaves.

This is bottom-subspace transfer, NOT transport of a learned CT12 whole state
or of its upper maps. Each new target hierarchy uses its complete PBC bond
ledger. Reuse is allowed only when ordered child-operator hashes and every
cross bond match. No sweep, attention, Born sampling, A1 energy, or physical
variance is inferred from these raw-tree initialization/cost probes.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from time import perf_counter, process_time

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.su2_sector_merge import project_sector_maps
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_hierarchy_initialization import (
    active_amplitude, active_bank_from_state, active_state_environment,
    initialize_arm, split_bond_ledger,
)
from .triangle_sector_transfer import load_source

ROOT = Path(__file__).resolve().parents[3]


def _verify(source: dict, state: dict, directory: Path) -> dict:
    """Independent root projection, all-edge norm and non-Born decoder replay."""
    tree = triangular_schur_hierarchy(state["length"])
    leaves, joins, bonds = split_bond_ledger(tree.length, state["j2"])
    node = len(tree.binary_merges)-1
    left, right, parent = tree.binary_merges[node]
    root = project_sector_maps(state["blocks"][left], state["blocks"][right], joins[node],
        maps=state["maps"][node], compact_boundary=True, max_output_dimension=1,
        memory_limit_bytes=768*1024**2)
    energy = float(root.h_by_q[0][0, 0].real)
    energy_error = abs(energy-float(state["blocks"][parent].h_by_q[0][0, 0].real))
    env, _ = active_state_environment(source, state)
    gram_error = max(float(np.max(abs(g-np.eye(len(g)))))
                     for row in env.upward.values() for g in row.values())
    map_error = max(float(np.max(abs(w.conj().T@w-np.eye(w.shape[1]))))
                    for maps in state["maps"].values() for w in maps.values())
    if max(energy_error, gram_error, map_error, abs(env.norm_squared-1)) > 2e-9:
        raise ArithmeticError("independent root-H or raw normalization gate failed")
    bank, profile = active_bank_from_state(source, state)
    cfg = MixedClebschConfig(tree.length, 20, 10)
    query = jax.jit(make_feature_lut_amplitude(cfg, active_top_widths=profile))
    prepared = prepare_feature_lut_bank(bank, cfg)
    rng = np.random.default_rng(2610265000+tree.length)
    rows = []
    for _ in range(4):
        row = np.zeros(tree.n_sites, np.int32)
        # Support-only configurations: each 18-site pair has zero m. These
        # are NOT samples from the wavefunction or an energy audit stream.
        for i, block in enumerate(tree.blocks):
            count = 4+i % 2
            row[list(block.sites)] = rng.permutation(np.r_[np.ones(count, np.int32), np.zeros(9-count, np.int32)])
        rows.append(row)
    started = perf_counter()
    first = float(query(prepared, rows[0])); cold = perf_counter()-started
    started = perf_counter()
    actual = np.asarray([float(query(prepared, row)) for row in rows])
    warm = perf_counter()-started
    expected = np.asarray([active_amplitude(source, state, row) for row in rows])
    relative_error = float(np.max(abs(actual-expected))/max(float(np.max(abs(expected))), 1e-300))
    if first != actual[0] or not np.any(expected != 0) or relative_error > 2e-10:
        raise ArithmeticError("generic active decoder disagrees with scalar CG contraction")
    metadata = dict(model=dict(length=tree.length, multiplicity_cap=20, maximum_q=10),
        active_top_widths=profile, representation="active bank; pass exact profile to decoder",
        raw_tree_energy_per_site=energy/tree.n_sites, raw_norm=env.norm_squared,
        A1_energy_known=False, physical_variance_known=False, optimizer_reset_required=True)
    save_checkpoint(directory/"active_bank.pkl", bank, metadata=metadata)
    np.savez(directory/"raw_decoder_replay.npz", configurations=np.asarray(rows), scalar=expected, decoder=actual)
    unique = [r for r in state["records"] if not r["cache_hit"]]
    return dict(independent_root_total_energy=energy, independent_root_H_error=energy_error,
        raw_tree_energy_per_site=energy/tree.n_sites, raw_norm_squared=env.norm_squared,
        maximum_upward_Gram_error=gram_error, maximum_map_isometry_error=map_error,
        maximum_relative_edge_norm_error=env.maximum_relative_edge_norm_error,
        scalar_vs_active_decoder_relative_error=relative_error,
        decoder_cold_seconds=cold, decoder_warm_four_rows_seconds=warm,
        replay_rows_are_not_Born_samples=True, complete_A1_queries=0, physical_H_queries=0,
        root_child_sites=[len(tree.coarse_supports[i]) for i in (left, right)],
        layer_counts=tree.layer_counts, root_cross_bonds=len(joins[-1]),
        bonds=dict(total=len(bonds), J1=sum(g == 1 for _, _, g in bonds),
            J2=sum(g == .125 for _, _, g in bonds), weight_sum=sum(g for _, _, g in bonds),
            leaf_counts=list(map(len, leaves)), merge_counts=list(map(len, joins))),
        all_target_bonds_used_once=True, all_nonroot_q_positive_sectors_present=True,
        retained_q_by_node=profile,
        unique_coarse_solves=len(unique), reused_coarse_nodes=len(state["records"])-len(unique),
        coarse_solve_H_matvecs=sum(s["matvec_count"] for r in unique for s in r["diagnostics"]["sectors"].values()),
        coarse_projection_H_columns=1+sum(sum(r["retain"].values()) for r in unique),
        coarse_matvecs_and_projection_columns_are_distinct_units=True,
        estimated_solver_peak_bytes=max(r["diagnostics"]["estimated_total_peak_bytes"] for r in state["records"]),
        estimated_solver_memory_excludes_caller_runtime_and_saved_states=True,
        maximum_retained_eigen_residual=max(s["eigen_residual_max"] for r in state["records"] for s in r["diagnostics"]["sectors"].values()),
        checkpoint_sha256=file_sha256(directory/"active_bank.pkl"),
        state_sha256=file_sha256(directory/"state.pkl"))


def run(output: Path, *, seconds_per_size: float = 600.) -> dict:
    """One immutable matched-width geometry sequence with bounded node solves."""
    if not 60 <= seconds_per_size <= 600:
        raise ValueError("registered wall cap is 60..600 seconds per size")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("new isolated output directory required")
    files = [Path(__file__), Path(__file__).with_name("triangle_hierarchy_initialization.py"),
        ROOT/"su2_nqs/lattices/triangular_schur_hierarchy.py", ROOT/"su2_nqs/operators/su2_sector_merge.py"]
    registration = dict(task="same_CT6_nine_site_subspace_L12_L18_L24_width4_cost_probe_v1",
        lengths=[12, 18, 24], active_width=4, j2=.125, maximum_q=10,
        bottom_q_widths=[20, 20, 20, 8, 1], root_q=0, seconds_per_size=seconds_per_size,
        geometry_rule="fixed 6x6 modules; recursively bisected Morton tile intervals; parent order height then start",
        transfer_scope="same four trained CT6 nine-site reduced spaces; NOT learned CT12 upper W or whole-state transfer",
        exact_cache_key="ordered child H/spin hashes + target cross bonds + retained q widths + arm",
        leaf_cache_key="ordered source local-map hash + base_index + target internal bonds",
        no_sweep_or_attention=True, no_MC_or_A1_energy=True, no_full_physical_variance=True,
        source_hashes={str(p.relative_to(ROOT)): file_sha256(p) for p in files})
    atomic_json(output/"registration.json", registration)
    frozen = output/"registered_sources"; frozen.mkdir()
    for file in files:
        shutil.copy2(file, frozen/file.name)
    started = perf_counter(); cpu = process_time()
    source, provenance = load_source(.125)
    atomic_json(output/"source.json", provenance)
    save_checkpoint(output/"canonical_CT6_source_bank.pkl", source, metadata=provenance)
    results = dict(registration=registration, source=provenance, sizes={})
    cache = {}; leaf_cache = {}
    for length in registration["lengths"]:
        folder = output/f"L{length}"
        tick = perf_counter(); cp = process_time()
        summary = initialize_arm(source, .125, "physical", 4, folder,
            tick+seconds_per_size, length=length, shared_cache=cache, shared_leaf_cache=leaf_cache)
        summary.update(initialization_wall_seconds=perf_counter()-tick,
                       initialization_CPU_seconds=process_time()-cp)
        state = load_checkpoint(folder/"state.pkl")["params"]
        if summary["status"] == "complete":
            cache = state["cache"]; leaf_cache = state["leaf_cache"]
            verify_tick = perf_counter(); verify_cpu = process_time()
            summary["verification"] = _verify(source, state, folder)
            summary.update(verification_wall_seconds=perf_counter()-verify_tick,
                           verification_CPU_seconds=process_time()-verify_cpu)
        results["sizes"][str(length)] = summary
        results.update(wall_seconds=perf_counter()-started, CPU_seconds=process_time()-cpu)
        atomic_json(output/"results.json", results)
        print(length, summary["status"], summary["raw_tree_energy_per_site"], flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds-per-size", type=float, default=600.)
    args = parser.parse_args()
    run(args.output, seconds_per_size=args.seconds_per_size)
