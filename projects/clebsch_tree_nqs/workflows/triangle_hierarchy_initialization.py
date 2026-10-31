"""Resumable Hamiltonian-informed CT12 initialization, without a physical basis.

All sixteen nine-spin child spaces come from one trusted strong CT6 parent.
Every target PBC bond enters exactly one leaf H or coarse merge. Complete
q>0 multiplets survive below the singlet root. Copy/random controls use the
same active upper width. Their coarse Hamiltonians retain all offdiagonals.
The recursively computed root E is the exact expectation of ONE RAW TREE;
coherent A1 energy and physical variance require the separate full-H audit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from itertools import chain
import json
from numbers import Integral
from pathlib import Path
import signal
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy import sparse

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout
from su2_nqs.models.triangular_clebsch_active_layout import mixed_active_layout
from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.operators.su2_sector_merge import (
    compact_reduced_block, project_sector_maps, reduced_block_from_projected,
    solve_sector_merge,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.quantum_numbers import coupling_range, validate_q
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_sector_transfer import common_target_bank, extract_merge_map, load_source
from .triangle_transfer_initialization import physical_pair_operator

ARMS = ("physical", "copy", "random")
QMAX = 10
BASE_SEED = 2610231400


def split_bond_ledger(length: int, j2: float) -> tuple[list, list, tuple]:
    """Partition every weighted physical bond once among leaves and LCA merges."""
    tree = triangular_schur_hierarchy(length)
    supports = tree.coarse_supports
    bonds = triangular_j1j2_bonds(length, length, j2=j2, periodic=True)
    leaves = []
    used = {}
    for sites in supports[:len(tree.blocks)]:
        position = {s: i for i, s in enumerate(sites)}
        rows = []
        for a, b, value in bonds:
            if a in position and b in position:
                rows.append((*sorted((position[a], position[b])), value))
                used[a, b] = used.get((a, b), 0)+1
        leaves.append(tuple(sorted(rows)))
    joins = []
    for left, right, _ in tree.binary_merges:
        lp = {s: i for i, s in enumerate(supports[left])}
        rp = {s: i for i, s in enumerate(supports[right])}
        cross = []
        for a, b, value in bonds:
            if a in lp and b in rp:
                cross.append((lp[a], rp[b], value)); used[a, b] = used.get((a, b), 0)+1
            elif b in lp and a in rp:
                cross.append((lp[b], rp[a], value)); used[a, b] = used.get((a, b), 0)+1
        joins.append(tuple(sorted(cross)))
    if set(used) != {(a, b) for a, b, _ in bonds} or any(v != 1 for v in used.values()):
        raise ArithmeticError("leaf/LCA ledger did not use every physical bond exactly once")
    return leaves, joins, bonds


def source_channels(left: dict[int, int], right: dict[int, int], q: int) -> tuple:
    """The fixed (qL,aL,qR,aR) order of one multiplicity source space."""
    return tuple((ql, a, qr, b) for ql, dl in sorted(left.items()) for qr, dr in sorted(right.items())
                 if q in coupling_range(ql, qr) for a in range(dl) for b in range(dr))


def _block_hash(block) -> str:
    h = hashlib.sha256(str((block.n_sites, sorted(block.q_widths.items()))).encode())
    arrays = [block.h_by_q[q] for q in sorted(block.q_widths)]
    for site in block.site_reduced_spin:
        for key in sorted(site):
            h.update(str(key).encode()); arrays.append(site[key])
    for array in arrays:
        x = np.ascontiguousarray(array)
        h.update(str((x.shape, x.dtype)).encode()); h.update(x.tobytes())
    return h.hexdigest()


def _real_matrix(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix)
    if np.max(abs(x.imag)) > 1e-8*max(1., float(np.linalg.norm(x))):
        raise ArithmeticError("the real time-reversal invariant initialization acquired complex maps")
    return x.real.copy()


def complete_active_columns(active: np.ndarray, old: np.ndarray) -> np.ndarray:
    """Complete an orthonormal map without changing even its active phases."""
    if active.shape[0] != old.shape[0] or active.shape[1] > old.shape[1]:
        raise ValueError("active source shape and native output width must agree")
    if np.max(abs(active.conj().T @ active-np.eye(active.shape[1]))) > 2e-8:
        raise ArithmeticError("active columns must already be isometric")
    columns = [active[:, k].copy() for k in range(active.shape[1])]
    # A generic QR of [active,old] may flip the active column phases. Such a
    # gauge would have to be absorbed in its parent; preserve them directly.
    candidates = chain(iter(old.T),
        (np.eye(1, len(old), k, dtype=old.dtype).ravel() for k in range(len(old))))
    for candidate in candidates:
        if len(columns) == old.shape[1]:
            break
        value = candidate.copy()
        for _ in range(2):
            for previous in columns:
                value -= previous*np.vdot(previous, value)
        norm = np.linalg.norm(value)
        if norm > 1e-11:
            columns.append(value/norm)
    if len(columns) != old.shape[1]:
        raise ArithmeticError("could not complete the native inactive columns")
    return np.column_stack(columns)


def retained_widths(left: dict[int, int], right: dict[int, int], active: int, root: bool) -> dict[int, int]:
    """Keep every reachable doubled charge through qmax below the root.

    Infer parity from actual children, rather than assuming even-spin
    intermediate regions. Complete q>0 multiplets survive whenever legal.
    """
    if isinstance(active, bool) or not isinstance(active, Integral) or active < 1:
        raise ValueError("active multiplicity must be a positive integer")
    for widths in (left, right):
        if not widths:
            raise ValueError("nonempty child sectors required")
        for q, width in widths.items():
            validate_q(q)
            if isinstance(width, bool) or not isinstance(width, Integral) or width < 1:
                raise ValueError("positive integer child multiplicities required")
    reachable = sorted({q for ql in left for qr in right for q in coupling_range(ql, qr) if q <= QMAX})
    if root:
        if 0 not in reachable:
            raise ValueError("child sectors cannot reach a singlet root")
        return {0: 1}
    return {q: min(active, len(source_channels(left, right, q))) for q in reachable}


def _leaf_block(local: tuple, inside: tuple):
    tables = triangle_block_tables()["nine"]
    m = np.asarray([2*i.bit_count()-9 for i in range(512)])
    sectors = {q: np.stack([(tables[q] @ np.asarray(w))*(m == value)[:, None]
                           for value in range(-q, q+1, 2)], axis=-1)
               for q, w in zip((1, 3, 5, 7, 9), local, strict=True)}
    h = sum((value*physical_pair_operator(9, a, b) for a, b, value in inside),
            sparse.csr_matrix((512, 512)))
    return compact_reduced_block(reduced_block_from_projected(project_block(BlockIsometry(9, sectors), h.toarray())))


@contextmanager
def _node_timeout(seconds: float):
    """Interrupt only this foreground CPU workflow at a Python/kernel boundary."""
    previous = signal.getsignal(signal.SIGALRM)
    def stop(*_):
        raise TimeoutError("fixed node/remaining wall budget reached")
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.01, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def initialize_arm(bank: dict, j2: float, arm: str, active: int, directory: Path,
                   deadline: float, *, resume: bool = False, length: int = 12,
                   shared_cache: dict | None = None, shared_leaf_cache: dict | None = None) -> dict:
    """Commit complete nodes atomically; interrupted nodes are recomputed."""
    tree = triangular_schur_hierarchy(length)
    if arm not in ARMS or (arm == "copy" and length != 12):
        raise ValueError("copy control is registered only for L=12; physical/random accept L=6*k")
    cfg = MixedClebschConfig(length, 20, QMAX)
    leaves, joins, all_bonds = split_bond_ledger(length, j2)
    path = directory/"state.pkl"; directory.mkdir(parents=True, exist_ok=True)
    common = common_target_bank(bank) if arm == "copy" else None
    if path.exists():
        if not resume:
            raise FileExistsError("existing arm state requires explicit --resume")
        state = load_checkpoint(path)["params"]
        if (state["j2"], state["arm"], state["active"], state.get("length", 12)) != (j2, arm, active, length):
            raise ValueError("saved hierarchy belongs to another registered arm")
    else:
        state = dict(j2=j2, arm=arm, active=active, length=length,
                     n_leaves=len(tree.blocks), n_merges=len(tree.binary_merges),
                     blocks=[], maps={}, channels={}, records=[],
                     cache={} if shared_cache is None else dict(shared_cache),
                     leaf_cache={} if shared_leaf_cache is None else dict(shared_leaf_cache), completed_nodes=0,
                     physical_bonds=len(all_bonds), target_bond_weights_sum=sum(x[2] for x in all_bonds))
    while len(state["blocks"]) < len(tree.blocks):
        index = len(state["blocks"])
        if perf_counter() > deadline-30:
            save_checkpoint(path, state); return dict(status="paused", **_state_summary(state))
        donor = tree.blocks[index].base_index
        # A donor number alone is not a transferable basis identity. Bind a
        # shared leaf cache to the actual ordered/gauged local reduced maps.
        digest = hashlib.sha256()
        for weights in bank["local"][donor]:
            array = np.ascontiguousarray(weights)
            digest.update(str((array.shape, array.dtype)).encode()); digest.update(array.tobytes())
        key = (digest.hexdigest(), donor, leaves[index])
        if key not in state["leaf_cache"]:
            state["leaf_cache"][key] = _leaf_block(bank["local"][donor], leaves[index])
        state["blocks"].append(state["leaf_cache"][key])
        save_checkpoint(path, state)
    for node in range(state["completed_nodes"], len(tree.binary_merges)):
        left, right, parent = tree.binary_merges[node]
        l, r = state["blocks"][left], state["blocks"][right]
        root = node == len(tree.binary_merges)-1
        retain = retained_widths(l.q_widths, r.q_widths, active, root)
        channel = {q: source_channels(l.q_widths, r.q_widths, q) for q in retain}
        seed = BASE_SEED+int(1000*j2)+100*ARMS.index(arm)+node
        key = (_block_hash(l), _block_hash(r), joins[node], tuple(retain.items()), arm)
        if arm != "physical":
            key = (*key, node)
        tick = perf_counter()
        if tick > deadline-40:
            save_checkpoint(path, state); return dict(status="paused", **_state_summary(state))
        try:
            with _node_timeout(min(300., deadline-perf_counter()-10)):
                cache_hit = key in state["cache"]
                if cache_hit:
                    merged, maps, diagnostics = state["cache"][key]
                else:
                    if arm == "physical":
                        solved = solve_sector_merge(l, r, joins[node], retain=retain,
                            seed=seed, tol=1e-7, maxiter=150, propagate_operators=False,
                            compact_boundary=True, memory_limit_bytes=768*1024**2,
                            max_output_dimension=active*36, dense_reference_limit=64)
                        maps = {q: _real_matrix(w) for q, w in solved.reduced_maps.items()}
                        diagnostics = dict(solved.diagnostics, eigen_residuals={q: x.tolist() for q, x in solved.eigen_residuals.items()})
                    else:
                        maps = {}
                        rng = np.random.default_rng(seed)
                        for q, count in retain.items():
                            raw = (extract_merge_map(common, cfg, node, q, channel[q])[:, :count]
                                   if arm == "copy" else rng.normal(size=(len(channel[q]), count)))
                            maps[q] = np.linalg.qr(raw, mode="reduced")[0][:, :count]
                        diagnostics = dict(naive_coefficient_copy=arm == "copy", learned_H_selection=False)
                    merged = project_sector_maps(l, r, joins[node], maps=maps,
                        compact_boundary=True, max_output_dimension=active*36,
                        memory_limit_bytes=768*1024**2)
                    state["cache"][key] = (merged, maps, diagnostics)
                if parent != len(state["blocks"]):
                    raise ArithmeticError("coarse hierarchy no longer follows target topological order")
                state["blocks"].append(merged)
                state["maps"][node] = maps; state["channels"][node] = channel
                state["completed_nodes"] += 1
                record = dict(node=node, parent_sites=merged.n_sites, retain=retain,
                    output_dimension=merged.dimension, cache_hit=cache_hit,
                    seconds=perf_counter()-tick, cross_bonds=len(joins[node]), diagnostics=diagnostics,
                    compact_storage_bytes=sum(a.nbytes for site in merged.site_reduced_spin for a in site.values()))
                state["records"].append(record)
                save_checkpoint(path, state, optimizer_state=dict(optimizer=None, reset_required=True,
                    RNG=np.random.default_rng(seed).bit_generator.state), metadata=dict(no_training_updates=True))
                atomic_json(directory/"progress.json", _state_summary(state))
                print(j2, arm, node, merged.n_sites, merged.dimension, record["seconds"], cache_hit, flush=True)
        except (TimeoutError, RuntimeError, MemoryError) as error:
            atomic_json(directory/"interruption.json", dict(node=node, error=str(error), seconds=perf_counter()-tick))
            save_checkpoint(path, state)
            return dict(status="bounded_incomplete", **_state_summary(state))
    return dict(status="complete", **_state_summary(state))


def _state_summary(state: dict) -> dict:
    final = state["blocks"][-1] if state["completed_nodes"] == state.get("n_merges", 15) else None
    length = state.get("length", 12)
    return dict(j2=state["j2"], arm=state["arm"], length=length, active_width=state["active"],
        completed_nodes=state["completed_nodes"], physical_bonds=state["physical_bonds"],
        raw_tree_energy_per_site=None if final is None else float(final.h_by_q[0][0, 0].real)/length**2,
        raw_variance_known=False, A1_energy_known=False, records=state["records"])


def native_bank_from_state(source: dict, state: dict) -> tuple[dict, dict]:
    """Embed active maps isometrically in width20, without changing the state."""
    if state.get("length", 12) != 12:
        raise ValueError("full-width native completion is registered only for L=12; use active_bank_from_state")
    cfg = MixedClebschConfig(12, 20, QMAX)
    common = common_target_bank(source)
    result = jax.tree.map(lambda x: np.asarray(x).copy(), common)
    defects = []
    for node, maps in state["maps"].items():
        layout = mixed_clebsch_layout(cfg)[node]
        left, right = layout["left"], layout["right"]
        full_l = ({q: x.shape[1] for q, x in zip(cfg.local_q, result["local"][left], strict=True)}
                  if left < 16 else mixed_clebsch_layout(cfg)[left-16]["widths"])
        full_r = ({q: x.shape[1] for q, x in zip(cfg.local_q, result["local"][right], strict=True)}
                  if right < 16 else mixed_clebsch_layout(cfg)[right-16]["widths"])
        for q, values in maps.items():
            channels = source_channels(full_l, full_r, q)
            position = {label: i for i, label in enumerate(channels)}
            active = np.zeros((len(channels), values.shape[1]))
            for i, label in enumerate(state["channels"][node][q]):
                active[position[label]] = values[i]
            old = extract_merge_map(common, cfg, node, q, channels)
            completed = complete_active_columns(active, old)
            # Root's output width is one; other q blocks all receive complete
            # inactive isometric columns, enabling one-sided live expansion.
            # Write into the already-owned copy. Copying the entire width20
            # native bank for each q would make packing artificially expensive.
            row_index = {label: i for i, label in enumerate(channels)}
            for bucket, group in zip(result["top"][node], layout["groups"], strict=True):
                for k, (ql, qr, qt) in enumerate(group["keys"]):
                    if qt != q:
                        continue
                    for a in range(bucket.shape[-2]):
                        for b in range(bucket.shape[-1]):
                            bucket[k, :, a, b] = completed[row_index[ql, a, qr, b]]
            defects.append(float(np.max(abs(completed.T @ completed-np.eye(completed.shape[1])))))
    return result, dict(maximum_reduced_isometry_error=max(defects),
                        active_upper_width=state["active"], stored_native_width=20,
                        active_top_widths={i: {q: w.shape[1] for q, w in maps.items()}
                                           for i, maps in state["maps"].items()},
                        inactive_columns_completed_not_both_zero=True,
                        state_preservation="parents couple only active child indices; completed inactive child columns do not enter root amplitude")


def active_bank_from_state(source: dict, state: dict) -> tuple[dict, tuple[dict[int, int], ...]]:
    """Pack any complete L=6*k raw hierarchy in its exact active layout.

    Unlike width-20 completion this stores only present output/child axes.
    Pass the returned doubled-q width profile to the feature-LUT decoder;
    native channel-opening gradients are not represented by this export.
    """
    tree = triangular_schur_hierarchy(state.get("length", 12))
    cfg = MixedClebschConfig(tree.length, 20, QMAX)
    if set(state["maps"]) != set(range(len(tree.binary_merges))):
        raise ValueError("complete target hierarchy required for active export")
    profile = tuple({q: w.shape[1] for q, w in state["maps"][node].items()}
                    for node in range(len(tree.binary_merges)))
    layout = mixed_active_layout(cfg, profile)
    top = []
    for node, row in enumerate(layout):
        lookup = {q: {label: i for i, label in enumerate(state["channels"][node][q])}
                  for q in profile[node]}
        groups = []
        for group in row["groups"]:
            bucket = np.zeros(group["shape"], dtype=np.result_type(*state["maps"][node].values()))
            for k, (ql, qr, q) in enumerate(group["keys"]):
                for a in range(bucket.shape[-2]):
                    for b in range(bucket.shape[-1]):
                        bucket[k, :, a, b] = state["maps"][node][q][lookup[q][ql, a, qr, b]]
            groups.append(bucket)
        top.append(tuple(groups))
    return dict(local=tuple(source["local"][block.base_index] for block in tree.blocks),
                top=tuple(top)), profile


def active_state_environment(source: dict, state: dict) -> tuple:
    """Exact all-edge raw norm environments for any complete stored hierarchy.

    The returned node tuple follows coarse IDs, not recursive postorder.
    Source local maps are transported only by the fixed tile/base lineage.
    """
    from su2_nqs.models.triangular_clebsch_gauge import _nodes
    from su2_nqs.su2.reduced_tree_environment import ReducedSingletTree, reduced_tree_environments
    geometry = triangular_schur_hierarchy(state.get("length", 12))
    tree, nodes = _nodes(MixedClebschConfig(geometry.length, 20, QMAX))
    grams = tuple({q: np.asarray(w).conj().T @ np.asarray(w)
                   for q, w in zip((1, 3, 5, 7, 9), source["local"][block.base_index], strict=True)}
                  for block in geometry.blocks)
    kernels = {}
    for node, maps in state["maps"].items():
        rows = {}
        for q, matrix in maps.items():
            channels = state["channels"][node][q]
            for ql, qr in sorted({(x[0], x[2]) for x in channels}):
                ids = [i for i, x in enumerate(channels) if x[0] == ql and x[2] == qr]
                dl = 1+max(channels[i][1] for i in ids); dr = 1+max(channels[i][3] for i in ids)
                rows[ql, qr, q] = np.asarray(matrix)[ids].reshape(dl, dr, -1).transpose(2, 0, 1)
        kernels[nodes[len(geometry.blocks)+node]] = rows
    reduced = ReducedSingletTree(tree, grams, tuple(kernels[n] for n in tree.internal_nodes), np.ones(1))
    return reduced_tree_environments(reduced), nodes


def active_amplitude(source: dict, state: dict, bits: np.ndarray) -> float:
    """Independent scalar contraction of the active variable-width tree."""
    row = np.asarray(bits)
    tree = triangular_schur_hierarchy(state.get("length", 12))
    if row.shape != (tree.n_sites,) or np.any((row != 0) & (row != 1)):
        raise ValueError("one valid target-size bit configuration is required")
    tables = triangle_block_tables()["nine"]
    values = []; magnetic = []
    for index, block in enumerate(tree.blocks):
        local = row[list(block.sites)]
        packed = int(local @ (1 << np.arange(8, -1, -1)))
        values.append({q: tables[q][packed] @ np.asarray(w)
                       for q, w in zip((1, 3, 5, 7, 9), source["local"][block.base_index], strict=True)})
        magnetic.append(int((2*local-1).sum()))
    for node, (left, right, parent) in enumerate(tree.binary_merges):
        ml, mr = magnetic[left], magnetic[right]
        out = {}
        for q, matrix in state["maps"][node].items():
            coeff = []
            for ql, a, qr, b in state["channels"][node][q]:
                cg = (clebsch_gordan(ql, ml, qr, mr, q, ml+mr)
                      if abs(ml) <= ql and abs(mr) <= qr and abs(ml+mr) <= q else 0.)
                coeff.append(cg*values[left][ql][a]*values[right][qr][b])
            out[q] = np.asarray(coeff) @ matrix
        values.append(out); magnetic.append(ml+mr)
    return float(values[-1][0][0]) if magnetic[-1] == 0 else 0.


def load_initialized_bank(directory: Path, arm: str = "physical", j2: float = .125) -> tuple[dict, dict]:
    """Load a materialized bank or reconstruct it from the small active ledger."""
    tag = "j0" if j2 == 0 else "j0125"
    folder = Path(directory)/tag/arm
    saved = folder/"initialized_bank.pkl"
    if saved.exists():
        payload = load_checkpoint(saved); return payload["params"], payload["metadata"]
    state = load_checkpoint(folder/"state.pkl")["params"]
    source, provenance = load_source(j2)
    bank, embedded = native_bank_from_state(source, state)
    return bank, dict(model=dict(length=12, multiplicity_cap=20, maximum_q=10),
                      arm=arm, j2=j2, source=provenance, embedding=embedded)


def run(output: Path, *, active: int = 4, seconds_cap: float = 1200., resume: bool = False) -> dict:
    """One fixed active width; a larger width requires a new registered run."""
    if active not in (4, 8) or not 60 <= seconds_cap <= 1200:
        raise ValueError("registered active width must be 4 or 8, wall cap 60..1200 seconds")
    output.mkdir(parents=True, exist_ok=True)
    registration = dict(task="full_target_CT12_hierarchy_initialization_v1",
        model=dict(length=12, multiplicity_cap=20, maximum_q=10), active_upper_width=active,
        bottom_q_widths=(20, 20, 20, 8, 1), nonroot_q=(0, 2, 4, 6, 8, 10), root_q=0,
        arms=ARMS, couplings=(.125, 0.), eig_tol=1e-7, eig_maxiter=150,
        wall_seconds=seconds_cap, seed=BASE_SEED,
        workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_2_to_N_physical_basis=True, no_remote_submission=True,
        optimizer="initialization only: old optimizer reset required; no training state reused")
    path = output/"registration.json"
    if path.exists():
        if not resume or json.loads(path.read_text()) != registration:
            raise FileExistsError("existing experiment requires matching explicit resume registration")
    else:
        if any(output.iterdir()):
            raise FileExistsError("refusing to overwrite prior artifacts")
        atomic_json(path, registration)
    started = perf_counter(); deadline = started+seconds_cap
    summaries = []
    # Finish the two physical target hierarchies before spending the remaining
    # cap on equally-wide copy and random controls.
    for arm in ARMS:
        for j2 in (.125, 0.):
            source, provenance = load_source(j2)
            tag = "j0125" if j2 == .125 else "j0"
            atomic_json(output/tag/"source.json", provenance)
            summary = initialize_arm(source, j2, arm, active, output/tag/arm, deadline, resume=resume)
            summaries.append(summary)
            if summary["status"] == "complete":
                state = load_checkpoint(output/tag/arm/"state.pkl")["params"]
                native, embedded = native_bank_from_state(source, state)
                metadata = dict(model=registration["model"], arm=arm, j2=j2, source=provenance,
                                raw_tree_energy_per_site=summary["raw_tree_energy_per_site"], embedding=embedded,
                                full_A1_energy_not_yet_audited=True, optimizer_reset_required=True)
                if arm == "physical":
                    save_checkpoint(output/tag/arm/"initialized_bank.pkl", native,
                                    optimizer_state=None, metadata=metadata)
                atomic_json(output/tag/arm/"native_metadata.json", metadata)
            atomic_json(output/"results.json", dict(registration=registration, arms=summaries,
                current_session_seconds=perf_counter()-started,
                physical_variance_not_from_compressed_H=True, no_training_or_remote_jobs=True))
            if perf_counter() > deadline-30:
                return dict(arms=summaries, status="paused")
    return dict(arms=summaries, status="complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active", type=int, default=4)
    parser.add_argument("--seconds-cap", type=float, default=1200.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output, active=args.active, seconds_cap=args.seconds_cap, resume=args.resume)
