"""Bounded target-sector transfer using actual saved CT nine-site blocks.

No 2**18 physical basis is constructed. Child H and every physical boundary
spin come from the real saved CT9 isometries; their target-geometry cross
bonds define a sector-specific coarse Hamiltonian. Native CT12 amplitude
checks are developmental costs and are not an energy audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.lattices.triangular_hnf import C6, hnf_site_permutation
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_amplitude, make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_layout,
)
from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.checkpoint import load_checkpoint
from .triangle_transfer_initialization import physical_pair_operator

ROOT = Path(__file__).resolve().parents[3]
PARENT_DIRECTORY = ROOT/"projects/clebsch_tree_nqs/results/remote_review_20261021/optimized_endpoints/ct6_continue_20261021"
PARENTS = {0.: "j0_ternary_joint_w20_continue32", .125: "j0125_binary_truncated_w20_continue32"}


def load_source(j2: float) -> tuple[dict, dict]:
    """Canonicalize one trusted w20 parent without changing its raw state."""
    if j2 not in PARENTS:
        raise ValueError("this fixed probe has only the two registered couplings")
    path = PARENT_DIRECTORY/PARENTS[j2]/"checkpoint.pkl"
    raw = load_checkpoint(path)["params"]
    config = MixedClebschConfig(6, 20, 10)
    arity = "binary_truncated" if "six" in raw else "ternary_joint"
    bank, diagnostics = canonicalize_triangular_bank(arity_clebsch_bank(raw, config, arity, 8), config)
    return bank, dict(checkpoint=str(path.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arity=arity, canonicalization=diagnostics)


def target_child_blocks(bank: dict, j2: float, *, cap: int = 20) -> tuple:
    """Project the two ACTUAL CT12 first nine-site supports and boundary spins."""
    tree = triangular_schur_hierarchy(12)
    target = tuple(tuple(block.sites) for block in tree.blocks[:2])
    weighted_bonds = triangular_j1j2_bonds(12, 12, j2=j2, periodic=True)
    tables = triangle_block_tables()["nine"]
    config = MixedClebschConfig(6, 20, 10)
    magnetization = np.asarray([2*i.bit_count()-9 for i in range(512)])
    children = []; notes = []
    for index, sites in enumerate(target):
        lookup = {site: local for local, site in enumerate(sites)}
        inside = [(lookup[a], lookup[b], value) for a, b, value in weighted_bonds
                  if a in lookup and b in lookup]
        h = sum((value*physical_pair_operator(9, a, b) for a, b, value in inside),
                sparse.csr_matrix((512, 512)))
        sectors = {}
        for q, weights in zip(config.local_q, bank["local"][index], strict=True):
            matrix = tables[q] @ np.asarray(weights)[:, :min(cap, weights.shape[1])]
            sectors[q] = np.stack([matrix*(magnetization == m)[:, None]
                                   for m in range(-q, q+1, 2)], axis=-1)
        tick = perf_counter()
        projected = project_block(BlockIsometry(9, sectors), h.toarray())
        children.append(projected)
        notes.append(dict(target_sites=sites, internal_bonds=inside,
                          projection_seconds=perf_counter()-tick,
                          retained_dimension=len(projected.labels),
                          q_widths={q: u.shape[1] for q, u in sectors.items()},
                          projection_diagnostics=projected.diagnostics))
    left, right = ({site: i for i, site in enumerate(sites)} for sites in target)
    cross = []
    for a, b, value in weighted_bonds:
        if a in left and b in right:
            cross.append((left[a], right[b], value))
        elif b in left and a in right:
            cross.append((left[b], right[a], value))
    return tuple(children), tuple(cross), notes


def extract_merge_map(bank: dict, config: MixedClebschConfig, node: int,
                      target_q: int, channels: tuple) -> np.ndarray:
    """Read native W in the declared (qL,aL,qR,aR) reduced source order."""
    layout = mixed_clebsch_layout(config)[node]
    tensors = {key: np.asarray(values[k]) for values, group in
               zip(bank["top"][node], layout["groups"], strict=True)
               for k, key in enumerate(group["keys"])}
    return np.stack([tensors[ql, qr, target_q][:, al, ar]
                     for ql, al, qr, ar in channels])


def replace_merge_map(bank: dict, config: MixedClebschConfig, node: int,
                      target_q: int, channels: tuple, matrix: np.ndarray) -> dict:
    """Install one full native reduced map; all magnetic CG data stay fixed."""
    result = jax.tree.map(lambda x: np.asarray(x).copy(), bank)
    layout = mixed_clebsch_layout(config)[node]
    if matrix.shape != (len(channels), layout["widths"][target_q]):
        raise ValueError("full output width and source channel ordering must agree")
    rows = {(ql, al, qr, ar): matrix[i] for i, (ql, al, qr, ar) in enumerate(channels)}
    for values, group in zip(result["top"][node], layout["groups"], strict=True):
        for k, (ql, qr, q) in enumerate(group["keys"]):
            if q != target_q:
                continue
            for al in range(values.shape[-2]):
                for ar in range(values.shape[-1]):
                    values[k, :, al, ar] = rows[ql, al, qr, ar]
    return result


def complete_with_copy(low_vectors: np.ndarray, copied: np.ndarray) -> np.ndarray:
    """Retain new low vectors, then orthogonally complete from the old map."""
    width = copied.shape[1]
    if low_vectors.shape[0] != copied.shape[0] or not 0 < low_vectors.shape[1] <= width:
        raise ValueError("new directions must fit the declared retained width")
    combined = np.column_stack((low_vectors, copied))
    q, _ = np.linalg.qr(combined, mode="reduced")
    return q[:, :width]


def common_target_bank(source: dict, seed: int = 26102391) -> dict:
    """One common target bank with copied bottom/18-site tensors and new top.

    New high layers are seeded once. This copies RAW tensor modules, not four
    individually projected CT6 wavefunctions, and is not a physical-state-
    preserving 6-to-12 expansion.
    """
    cfg = MixedClebschConfig(12, 20, 10)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(seed), cfg)["ternary_free"]
    bank = jax.tree.map(lambda x: np.asarray(x).copy(), bank)
    return dict(local=tuple(source["local"][i % 4] for i in range(16)),
                top=tuple(source["top"][i % 2] if i < 8 else values
                          for i, values in enumerate(bank["top"])))


def amplitude_cost(baseline: dict, candidate: dict, output: Path) -> dict:
    """Four fixed rows, raw and complete A1 amplitudes; never an energy audit."""
    cfg = MixedClebschConfig(12, 20, 10)
    tick = perf_counter()
    images = np.asarray(triangular_hnf_gamma_a1_group(((12, 0), (0, 12))), np.int32)
    group_seconds = perf_counter()-tick
    raw = jax.jit(make_feature_lut_amplitude(cfg))
    projected = jax.jit(make_feature_lut_projected_amplitude(cfg, images, chunk=4))
    rng = np.random.default_rng(26102392)
    rows = np.asarray([rng.permutation(np.r_[np.ones(72, np.int32), np.zeros(72, np.int32)])
                       for _ in range(4)])
    reports = []; saved = {"rows": rows}
    symmetry_maps = {
        "translation": np.asarray(hnf_site_permutation(((12, 0), (0, 12)), translation=(1, 0))),
        "C6_rotation": np.asarray(hnf_site_permutation(((12, 0), (0, 12)), operation=C6)),
    }
    for name, bank in (("common_copy", baseline), ("sector_initialized", candidate)):
        prepared = prepare_feature_lut_bank(bank, cfg)
        timings = {}; values = {}
        for kind, query in (("raw", raw), ("complete_A1", projected)):
            started = perf_counter()
            first = np.asarray(query(prepared, rows[0])); timings[kind+"_first_seconds"] = perf_counter()-started
            started = perf_counter()
            values[kind] = np.asarray([query(prepared, x) for x in rows])
            timings[kind+"_warm_four_rows_seconds"] = perf_counter()-started
            saved[name+"_"+kind] = values[kind]
        tick = perf_counter()
        symmetry = {}
        reference = values["complete_A1"][0]
        for label, image in symmetry_maps.items():
            changed = complex(np.asarray(projected(prepared, rows[0][image])))
            raw_changed = complex(np.asarray(raw(prepared, rows[0][image])))
            symmetry[label] = dict(A1_relative_amplitude_error=float(abs(changed-reference)/max(abs(reference), 1e-200)),
                                   raw_relative_amplitude_change=float(abs(raw_changed-values["raw"][0])/max(abs(values["raw"][0]), 1e-200)))
        timings["two_symmetry_checks_seconds"] = perf_counter()-tick
        reports.append(dict(arm=name, **timings, symmetry=symmetry,
                            raw_finite=bool(np.isfinite(values["raw"]).all()),
                            A1_finite=bool(np.isfinite(values["complete_A1"]).all()),
                            A1_nonzero_rows=int(np.count_nonzero(values["complete_A1"]))))
    np.savez_compressed(output/"fixed_amplitudes.npz", **saved)
    return dict(group_order=len(images), group_build_seconds=group_seconds, reports=reports,
                actual_full_H_centers=0, wavefunction_norm_computed=False,
                statement="Four deterministic Mz=0 configurations plus translation/C6 checks. The group mean normalizes the PROJECTOR, not the physical wavefunction; no sampling/quality claim.")


def run(output: Path, seconds_cap: float = 600., amplitude: bool = True) -> dict:
    """Evaluate registered active/full target sectors within a shared deadline."""
    from su2_nqs.operators.su2_sector_merge import reduced_block_from_projected, SectorMergeOperator
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a completed or partial experiment")
    if not np.isfinite(seconds_cap) or not 10 <= seconds_cap <= 600:
        raise ValueError("the fixed CPU wall cap must be 10..600 seconds")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"registration.json", dict(task="actual_CT9_target_sector_transfer_v2",
        couplings=(0., .125), active_cap=2, full_cap=20, full_target_sectors={"0.0": (0,), "0.125": (0, 2)},
        new_low_vectors=2, native_output_width=20, eigen_tolerance=1e-7,
        max_matvecs_per_sector=1200, eigen_ncv=32, CPU_seconds_cap=seconds_cap,
        seeds=(26102391, 26102392, 26102393), amplitude_four_rows=amplitude,
        no_2_to_18_allocation=True, no_remote_jobs=True,
        workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    started = perf_counter(); deadline = started+seconds_cap; reports = []; source_notes = []; packed = {}
    for j2 in (0., .125):
        bank, source = load_source(j2); source_notes.append(dict(j2=j2, **source))
        for cap in (2, 20):
            if perf_counter() > deadline-10:
                reports.append(dict(j2=j2, cap=cap, status="not_started_due_to_budget")); continue
            children, cross, notes = target_child_blocks(bank, j2, cap=cap)
            left, right = (reduced_block_from_projected(x) for x in children)
            for q in ((0,) if cap == 2 or j2 == 0 else (0, 2)):
                op = SectorMergeOperator(left, right, cross, q)
                calls = 0; sector_started = perf_counter()
                def apply(v):
                    nonlocal calls
                    if calls >= 1200 or perf_counter() >= deadline-10:
                        raise TimeoutError("frozen matvec/deadline budget reached")
                    calls += 1
                    value = op.matvec(v)
                    if np.max(abs(np.asarray(value).imag)) > 1e-10*max(1., float(np.linalg.norm(value))):
                        raise ArithmeticError("the actual real scalar Hamiltonian acquired an imaginary action")
                    return np.asarray(value).real
                a = LinearOperator((op.dimension,)*2, matvec=apply, dtype=np.float64)
                report = dict(j2=j2, cap=cap, target_q=q, dimension=op.dimension,
                              target_cross_bonds=cross, child_notes=notes,
                              operator_diagnostics=op.diagnostics)
                try:
                    first = np.random.default_rng(26102393).normal(size=op.dimension)
                    tick = perf_counter(); apply(first); report["first_matvec_seconds"] = perf_counter()-tick
                    copied = (extract_merge_map(bank, MixedClebschConfig(6, 20, 10), 0, q, op.channels)
                              if cap == 20 else np.linalg.qr(np.random.default_rng(26102393).normal(size=(op.dimension, 2)))[0])
                    if cap == 20:
                        action = np.column_stack([apply(x) for x in copied.T])
                        values, axes = np.linalg.eigh(copied.conj().T @ action)
                        report["copy_Ritz_ground_energy"] = float(values[0])
                        first = copied @ axes[:, 0]
                        random = np.linalg.qr(np.random.default_rng(26102393).normal(size=copied.shape))[0]
                        rh = np.column_stack([apply(x) for x in random.T])
                        report["random_same_width_Ritz_ground_energy"] = float(np.linalg.eigvalsh(random.conj().T @ rh)[0])
                    values, vectors = eigsh(a, k=min(2, op.dimension-2), which="SA", tol=1e-7,
                                            ncv=min(32, op.dimension), maxiter=80, v0=first)
                    order = np.argsort(values); values, vectors = values[order], vectors[:, order]
                    report.update(status="converged", low_energies=values.tolist(),
                        eigen_residuals=[float(np.linalg.norm(apply(v)-e*v)) for e, v in zip(values, vectors.T, strict=True)])
                    report["lowest_state_child_q_weights"] = {
                        str(ql): float(sum(abs(vectors[i, 0])**2 for i, row in enumerate(op.channels)
                                           if row[0] == ql)) for ql in sorted(left.q_widths)}
                    np.savez_compressed(output/f"j{j2}_cap{cap}_q{q}.npz", low_energies=values,
                                        low_vectors=vectors, channels=np.asarray(op.channels))
                    if cap == 20:
                        if j2 in packed:
                            target, candidate = packed[j2]
                        else:
                            target = common_target_bank(bank); candidate = target
                        new = complete_with_copy(vectors, copied)
                        candidate = replace_merge_map(candidate, MixedClebschConfig(12, 20, 10), 0, q, op.channels, new)
                        packed[j2] = (target, candidate)
                        np.savez_compressed(output/f"native_update_j{j2}_q{q}.npz",
                                            copied_map=copied, completed_map=new,
                                            channels=np.asarray(op.channels))
                        report["completed_native_map_Gram_error"] = float(np.max(abs(new.conj().T @ new-np.eye(20))))
                except (TimeoutError, ArpackNoConvergence) as error:
                    report.update(status="bounded_incomplete", error=str(error))
                report.update(matvec_calls=calls, sector_seconds=perf_counter()-sector_started)
                reports.append(report); atomic_json(output/"partial_results.json", dict(reports=reports, sources=source_notes))
                print(j2, cap, q, report["status"], calls, report["sector_seconds"], flush=True)
    amplitude_reports = []
    if amplitude:
        for j2, (baseline, candidate) in packed.items():
            if perf_counter() > deadline-60:
                amplitude_reports.append(dict(j2=j2, status="not_started_due_to_budget")); continue
            child = output/f"amplitude_j{j2}"; child.mkdir()
            amplitude_reports.append(dict(j2=j2, **amplitude_cost(baseline, candidate, child)))
    atomic_json(output/"candidate_recipe.json", dict(target_model=dict(length=12, multiplicity_cap=20, maximum_q=10),
        common_new_upper_seed=26102391, common_bank_factory="common_target_bank",
        source_banks=source_notes, replacement_node=0,
        updated_maps=[str(path.name) for path in output.glob("native_update_*.npz")],
        numpy_version=np.__version__, jax_version=jax.__version__,
        statement="Rebuild canonical source bank, seed common new upper layers, then install each completed native map. Folded physical bank, not a proof of native binary-rank8 parameterization."))
    result = dict(reports=reports, sources=source_notes, amplitude=amplitude_reports,
        total_seconds=perf_counter()-started, full_global_H_centers=0,
        interpretation="Actual CT9 retained spaces and CT12 target bonds. Sector Ritz energies are coarse-block energies, not global variational audits; fixed-row full A1 is only a cost/symmetry preflight.")
    atomic_json(output/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds-cap", type=float, default=600.)
    parser.add_argument("--no-amplitude", action="store_true")
    args = parser.parse_args(); run(args.output, args.seconds_cap, not args.no_amplitude)
