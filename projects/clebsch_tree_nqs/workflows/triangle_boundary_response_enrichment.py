"""Same-width response enrichment of one saved CT12 merge, no new head.

Frozen raw-root q0 density weights boundary spin response. Only node8 q2
changes. Ancestor overlaps/QR export the normalized projection of the old
state, NOT a state-preserving transformation. Full-A1 quality is separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from shutil import copyfile
import signal
from time import perf_counter, process_time
from typing import Literal, Sequence

import numpy as np
from scipy.sparse.linalg import eigsh

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.retained_boundary_response import (
    change_child_coordinates, correlated_bath_response_factor, density_protection_count,
    fixed_q_response_frame, response_capture,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_clebsch_gauge import _nodes, triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.su2_sector_merge import ReducedBlockData, SectorMergeOperator, project_sector_maps
from su2_nqs.operators.su2_three_block_center import ThreeBlockSingletOperator, parent_anchored_center_ritz
from su2_nqs.su2.reduced_tree_environment import ReducedSingletTree, reduced_tree_environments
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_hierarchy_initialization import native_bank_from_state, split_bond_ledger
from .triangle_sector_transfer import load_source

TREE = triangular_schur_hierarchy(12)
NODE = 8
SEED = 2610233800


def active_environment(state: dict, source: dict):
    """Exact raw double layer of the ACTIVE variable-width stored tree."""
    tree, nodes = _nodes(MixedClebschConfig(12, 20, 10)); by_node = {}
    grams = tuple({q: np.asarray(w).conj().T @ np.asarray(w)
                   for q, w in zip((1, 3, 5, 7, 9), source["local"][i % 4], strict=True)}
                  for i in range(16))
    for node, maps in state["maps"].items():
        rows = {}
        for q, matrix in maps.items():
            channels = state["channels"][node][q]
            for ql, qr in sorted({(x[0], x[2]) for x in channels}):
                ids = [i for i, x in enumerate(channels) if x[0] == ql and x[2] == qr]
                dl = 1+max(channels[i][1] for i in ids); dr = 1+max(channels[i][3] for i in ids)
                rows[ql, qr, q] = np.asarray(matrix)[ids].reshape(dl, dr, -1).transpose(2, 0, 1)
        by_node[nodes[16+node]] = rows
    reduced = ReducedSingletTree(tree, grams, tuple(by_node[n] for n in tree.internal_nodes), np.ones(1))
    environment = reduced_tree_environments(reduced)
    defect = max(float(np.max(abs(g-np.eye(len(g))))) for row in environment.upward.values() for g in row.values())
    if defect > 2e-11 or abs(environment.norm_squared-1.) > 2e-11:
        raise ArithmeticError("saved active state is not a normalized isometric raw tree")
    return environment, nodes, defect


def frozen_responses(state: dict, source: dict, *,
                     boundary_sites: Sequence[int] | None = None,
                     boundary_bonds: Sequence[tuple[int, int, float]] | None = None) -> tuple[dict, dict]:
    """Frozen q=0→2 response; default remains the historical J1-only cut.

    Explicit sites allow a verified cache to be supplemented without
    reevaluating its existing columns. Explicit bonds instead select every
    inside endpoint of a nonzero crossing bond. Neither option changes CG
    conventions or the conditional raw-density weighting.
    """
    if boundary_sites is not None and boundary_bonds is not None:
        raise ValueError("provide explicit boundary sites or bonds, not both")
    env, nodes, defect = active_environment(state, source)
    parent = 16+NODE; left, right, _ = TREE.binary_merges[NODE]
    down = env.downward[nodes[parent]]
    rho = np.asarray(down[0]); p0 = float(np.trace(rho).real/env.norm_squared)
    if p0 <= 1e-12:
        raise ArithmeticError("frozen q0 sector has no meaningful weight")
    rho = (rho+rho.conj().T)/(2*np.trace(rho).real)
    values, vectors = np.linalg.eigh(rho)
    if values.min() < -1e-12:
        raise ArithmeticError("exact conditional root density is not positive")
    root = (vectors*np.sqrt(np.maximum(values, 0.))) @ vectors.conj().T
    op0 = SectorMergeOperator(state["blocks"][left], state["blocks"][right], (), 0)
    op2 = SectorMergeOperator(state["blocks"][left], state["blocks"][right], (), 2)
    initial = state["maps"][NODE][0] @ root
    product = [op0.expand(initial[:, i]) for i in range(initial.shape[1])]
    support = TREE.coarse_supports[parent]; inside = set(support); position = {s: i for i, s in enumerate(support)}
    bonds = (triangular_j1j2_bonds(12, 12, j2=0., periodic=True)
             if boundary_bonds is None else boundary_bonds)
    boundary = (sorted({a if a in inside else b for a, b, j in bonds
                        if j != 0 and (a in inside) != (b in inside)})
                if boundary_sites is None else list(boundary_sites))
    if (not boundary or len(set(boundary)) != len(boundary)
            or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s not in inside
                   for s in boundary)):
        raise ValueError("distinct physical boundary sites inside the frozen region required")
    responses = []; covariance_error = 0.
    for site in boundary:
        i = position[site]; columns = []
        for psi in product:
            y = op0.left.p[i] @ psi if i < 18 else (op0.right.p[i-18] @ psi.T).T
            column = op2.project(y); columns.append(column)
            covariance_error = max(covariance_error, float(np.linalg.norm(op2.expand(column)-y)))
        responses.append(np.column_stack(columns))
    train = np.concatenate(responses[::2], axis=1)
    heldout = (np.concatenate(responses[1::2], axis=1) if len(responses) > 1
               else np.empty((op2.dimension, 0), dtype=train.dtype))
    if max(np.max(abs(train.imag), initial=0.), np.max(abs(heldout.imag), initial=0.)) > 1e-10:
        raise ArithmeticError("real parent produced a complex boundary response")
    data = dict(train=train.real, heldout=heldout.real, all=np.concatenate(responses, axis=1).real)
    p = {q: float((q+1)*np.trace(e).real/env.norm_squared) for q, e in down.items()}
    notes = dict(frozen_raw_q0_probability=p0, conditional_q0_trace=float(np.trace(rho).real),
        conditional_q0_eigenvalues=values.tolist(), global_region_q_weights=p,
        maximum_upward_Gram_defect=defect, raw_norm_squared=env.norm_squared,
        boundary_sites=boundary, train_sites=boundary[::2], heldout_sites=boundary[1::2],
        boundary_definition=("all J1-only boundary sites, fixed identically at both couplings"
                             if boundary_sites is None and boundary_bonds is None else
                             "explicit inside endpoints of supplied nonzero crossing bonds"
                             if boundary_bonds is not None else "explicit physical site subset"),
        response_definition="P_child S_i^+ P_child acting on columns Q0 sqrt(rho36,q0); conditional q0 density from frozen raw144 state",
        scalar_conversion="sumxyz=3/2 sumSplus; denominator is retained-child response, NOT physical onsite3/4",
        covariance_q2_absolute_error=covariance_error,
        source_is_global_raw_RDM_not_single_block_ground=True)
    return data, dict(notes=notes, down_q2=np.asarray(down[2])/env.norm_squared)


def _independent_node_map_copy(maps: dict) -> dict:
    """Detach EACH node, even when cached isomorphic nodes shared a dict.

    A generic deepcopy preserves alias relationships. Subsequent per-node
    assignments must not rotate translated peers without updating their
    parents and Hamiltonian blocks. Arrays are independently copied too.
    """
    return {node: {q: np.array(w, copy=True) for q, w in row.items()}
            for node, row in maps.items()}


def _checked_changed_nodes(original: dict, updated: dict, *, allowed: set[int]) -> list[int]:
    """Fail closed if a local update also changes a cached peer or its shape."""
    if original.keys() != updated.keys():
        raise ArithmeticError("local projection changed hierarchy node labels")
    changed = []
    for node, maps in original.items():
        if maps.keys() != updated[node].keys():
            raise ArithmeticError("local projection changed doubled-q labels")
        if any(w.shape != updated[node][q].shape for q, w in maps.items()):
            raise ArithmeticError("local projection changed retained multiplicity width")
        if any(not np.array_equal(w, updated[node][q]) for q, w in maps.items()):
            changed.append(node)
    if set(changed)-allowed:
        raise ArithmeticError(f"unintended cached-peer map updates: {set(changed)-allowed}")
    return changed


def project_candidate(state: dict, frame: np.ndarray, j2: float, down_q2: np.ndarray) -> tuple[dict, dict]:
    """Project the old RAW state onto one new edge subspace, then QR ancestors."""
    candidate = deepcopy(state); candidate["cache"] = {}; candidate["leaf_cache"] = {}
    candidate["maps"] = _independent_node_map_copy(state["maps"])
    candidate["parent_initialization_records"] = candidate.pop("records", [])
    candidate["records"] = []
    candidate["arm"] = "boundary_response_projected_old_state"
    old = np.asarray(state["maps"][NODE][2]); overlap = frame.conj().T @ old
    retained_probability = float(3*np.trace(down_q2 @ (overlap.conj().T @ overlap)).real)
    old_probability = float(3*np.trace(down_q2).real)
    predicted_norm = 1-old_probability+retained_probability
    candidate["maps"][NODE][2] = frame.copy()
    _, joins, _ = split_bond_ledger(12, j2)
    left, right, parent = TREE.binary_merges[NODE]
    candidate["blocks"][parent] = project_sector_maps(candidate["blocks"][left], candidate["blocks"][right], joins[NODE],
        maps=candidate["maps"][NODE], compact_boundary=True, max_output_dimension=288, memory_limit_bytes=768*1024**2)
    # Node8 is the LEFT36 child of node12. Only this ancestor and root change.
    ancestor = 12; al, ar, ap = TREE.binary_merges[ancestor]
    if al != parent:
        raise ArithmeticError("registered node8 ancestor layout changed")
    gauges = {}; ranks = {}
    for q, matrix in state["maps"][ancestor].items():
        projected = change_child_coordinates(matrix, state["channels"][ancestor][q], left={2: overlap})
        qr, r = np.linalg.qr(projected, mode="reduced")
        candidate["maps"][ancestor][q] = qr; gauges[q] = r
        ranks[q] = float(np.linalg.svd(projected, compute_uv=False).min())
    candidate["blocks"][ap] = project_sector_maps(candidate["blocks"][al], candidate["blocks"][ar], joins[ancestor],
        maps=candidate["maps"][ancestor], compact_boundary=True, max_output_dimension=288, memory_limit_bytes=768*1024**2)
    root = len(TREE.binary_merges)-1; rl, rr, rp = TREE.binary_merges[root]
    if rl != ap:
        raise ArithmeticError("registered ancestor/root layout changed")
    root_map = change_child_coordinates(state["maps"][root][0], state["channels"][root][0], left=gauges)
    norm = float(np.vdot(root_map, root_map).real)
    if abs(norm-predicted_norm) > 2e-10 or not 0 < norm <= 1+2e-10:
        raise ArithmeticError("direct root norm and edge-density projection fidelity disagree")
    candidate["maps"][root][0] = root_map/np.sqrt(norm)
    changed_nodes = _checked_changed_nodes(state["maps"], candidate["maps"], allowed={NODE, ancestor, root})
    candidate["blocks"][rp] = project_sector_maps(candidate["blocks"][rl], candidate["blocks"][rr], joins[root],
        maps=candidate["maps"][root], compact_boundary=True, max_output_dimension=288, memory_limit_bytes=768*1024**2)
    raw_energy = float(candidate["blocks"][rp].h_by_q[0][0, 0].real)/144
    notes = dict(raw_projection_norm_squared=norm, normalized_RAW_fidelity_to_parent=norm,
        predicted_norm_from_frozen_q2_density=predicted_norm, raw_norm_prediction_error=abs(norm-predicted_norm),
        parent_q2_probability=old_probability, retained_unnormalized_q2_probability=retained_probability,
        new_q2_probability=retained_probability/norm, raw_energy_per_site=raw_energy,
        ancestor_minimum_singular_values=ranks, state_preserved=False,
        exact_normalized_projection_not_fullA1_fidelity=True, upper_QR_gauges_absorbed_in_root=True,
        stored_parent_solver_records_are_not_new_candidate_eigenstates=True,
        node_map_aliases_detached_before_updates=True,
        actually_changed_nodes=changed_nodes,
        reduced_H_columns_evaluated=97, MC_full_H_centers=0, full_A1_energy_known=False)
    return candidate, notes


def run(output: Path, source_run: Path, seconds_cap: float = 180., *,
        protection: Literal["energy", "density", "adaptive"] = "energy") -> dict:
    """One pre-registered same-width, three-arm, two-coupling local experiment."""
    if not 30 <= seconds_cap <= 180:
        raise ValueError("fixed local wallcap must lie in30..180 seconds")
    if protection not in ("energy", "density", "adaptive"):
        raise ValueError("protection must be energy, density or adaptive")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("choose a new output; frozen evidence is never overwritten")
    inputs = {tag: str(source_run/tag/"physical/state.pkl") for tag in ("j0", "j0125")}
    response_arm = {"energy": "low4_response4", "density": "density4_response4", "adaptive": "densityauto_response"}[protection]
    random_arm = {"energy": "low4_random4", "density": "density4_random4", "adaptive": "densityauto_random"}[protection]
    fidelity_threshold = {"energy": .95, "density": .99, "adaptive": .995}[protection]
    registration = dict(task="frozen_global_raw_q0_boundary_response_same_width_v3", source_inputs=inputs,
        source_sha256={tag: file_sha256(Path(path)) for tag, path in inputs.items()},
        node=NODE, changed_q=2, width=8, protected_columns="frozen density tail rule" if protection == "adaptive" else 4, protection=protection,
        protection_source=("minimal k with p(q2)*sum_{i>k}lambda_i<=.005; no candidate energy or heldout data" if protection == "adaptive" else
                           "four leading frozen physical raw rho_q2 eigenvectors" if protection == "density" else "four lowest isolated energy vectors"),
        maximum_lost_global_RAW_norm=.005 if protection == "adaptive" else None,
        couplings=(.125, 0.), arms=("energy8", response_arm, random_arm), seed=SEED,
        response_weighting="conditional q0 density of frozen raw144 root state",
        heldout_split="alternating sorted J1-boundary sites; spatial sites, NOT independent environments",
        full_A1_quality_not_established=True, no_neural_network=True, no_new_large_eigensolve=True,
        serial_development_revision_not_independent_confirmation=protection != "energy",
        diagnostic_thresholds=dict(heldout_fraction_gain=.001, raw_fidelity=fidelity_threshold, maximum_raw_energy_increase_per_site=.002),
        wall_seconds=seconds_cap, workflow_sha256=file_sha256(Path(__file__)),
        helper_sha256=file_sha256(Path(__file__).parents[3]/"su2_nqs/frontier/retained_boundary_response.py"))
    atomic_json(output/"registration.json", registration)
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("fixed180s-or-lower boundary-response experiment budget")
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, seconds_cap)
    started = perf_counter(); reports = []
    try:
        for j2 in (.125, 0.):
            tag = "j0125" if j2 else "j0"; source, provenance = load_source(j2)
            (output/tag).mkdir(parents=True, exist_ok=True)
            saved = load_checkpoint(inputs[tag]); state = saved["params"]
            data, weighting = frozen_responses(state, source)
            frozen = weighting["notes"]
            np.savez_compressed(output/tag/"frozen_response_columns.npz", **data, down_q2=weighting["down_q2"])
            energy_frame = state["maps"][NODE][2]
            rho2 = weighting["down_q2"]
            rho2 = (rho2+rho2.conj().T)/(2*np.trace(rho2).real)
            spectrum, axes = np.linalg.eigh(rho2); order = np.argsort(spectrum)[::-1]
            spectrum, axes = spectrum[order], axes[:, order]
            if spectrum.min() < -1e-12:
                raise ArithmeticError("frozen q2 conditional density is not positive")
            if np.max(abs(axes.imag)) > 1e-10:
                raise ArithmeticError("real parent produced complex density axes")
            protected_reference = energy_frame @ axes.real if protection != "energy" else energy_frame
            frozen["conditional_q2_eigenvalues_descending"] = spectrum.tolist()
            frozen["physical_q2_probability"] = frozen["global_region_q_weights"][2]
            frozen["density4_unprotected_global_norm_weight"] = float(frozen["physical_q2_probability"]*np.sum(spectrum[4:]))
            frozen["density4_theoretical_fidelity_lower_bound"] = 1-frozen["density4_unprotected_global_norm_weight"]
            frozen["density4_bound_scope"] = "Keeping the four leading rho_q2 directions bounds LOST global raw norm ABOVE by p(q2)*tail4. Extra response directions can only reduce this loss; best possible loss is0. No projected-A1 bound."
            count = (density_protection_count(frozen["physical_q2_probability"], spectrum, maximum_lost_norm=.005)
                     if protection == "adaptive" else 4)
            frozen["protected_count"] = count
            frozen["remaining_response_capacity"] = 8-count
            frozen["allocation_uses_frozen_density_only"] = protection == "adaptive"
            frozen["selected_tail_global_norm_loss_bound"] = float(frozen["physical_q2_probability"]*spectrum[count:].sum())
            frozen["selected_RAW_fidelity_lower_bound"] = 1-frozen["selected_tail_global_norm_loss_bound"]
            frozen["no_response_freedom_at_fixed_width"] = count == 8
            atomic_json(output/tag/"frozen_source.json", dict(frozen, source=provenance))
            response = fixed_q_response_frame(2, protected_reference, data["train"], protect=count)
            random = fixed_q_response_frame(2, protected_reference, data["train"], protect=count, seed=SEED+int(1000*j2))
            frames = {"energy8": energy_frame, response_arm: response.frame, random_arm: random.frame}
            energy_reference = float(state["blocks"][-1].h_by_q[0][0, 0].real)/144
            parent_native, _ = native_bank_from_state(source, state)
            native_config = MixedClebschConfig(12, 20, 10)
            parent_native_norm = triangular_raw_overlap(parent_native, parent_native, native_config)
            arm_reports = {}
            for arm, frame in frames.items():
                tick = perf_counter()
                readout = {split: response_capture(frame, columns) for split, columns in data.items()}
                if arm == "energy8":
                    candidate, quality = project_candidate(state, frame, j2, weighting["down_q2"])
                    if (abs(quality["raw_energy_per_site"]-energy_reference) > 1e-10
                            or abs(quality["normalized_RAW_fidelity_to_parent"]-1.) > 2e-11):
                        raise ArithmeticError("identity overlap/QR/projected-H roundtrip changed the parent")
                    quality["state_preserved"] = True
                    quality["identity_roundtrip_energy_error"] = abs(quality["raw_energy_per_site"]-energy_reference)
                else:
                    candidate, quality = project_candidate(state, frame, j2, weighting["down_q2"])
                    if (protection != "energy" and quality["normalized_RAW_fidelity_to_parent"]
                            < frozen["selected_RAW_fidelity_lower_bound"]-2e-10):
                        raise ArithmeticError("physical density protection violated its prospective raw fidelity bound")
                    save_checkpoint(output/tag/arm/"state.pkl", candidate, optimizer_state=None,
                        metadata=dict(j2=j2, parent_state_sha256=registration["source_sha256"][tag],
                            optimizer_reset_required=True, no_training_updates=True, **quality))
                    if arm == response_arm:
                        native, embedding = native_bank_from_state(source, candidate)
                        native_norm = triangular_raw_overlap(native, native, native_config)
                        native_overlap = triangular_raw_overlap(parent_native, native, native_config)
                        actual_fidelity = float(abs(native_overlap)**2/(parent_native_norm.real*native_norm.real))
                        native_error = abs(actual_fidelity-quality["normalized_RAW_fidelity_to_parent"])
                        if (native_error > 2e-10 or abs(native_norm-1) > 2e-10
                                or abs(parent_native_norm-1) > 2e-10):
                            raise ArithmeticError("exported physical bank violates independently contracted raw fidelity")
                        quality.update(native_export_overlap_squared=actual_fidelity,
                            native_export_certificate_error=native_error,
                            native_export_norm_real=float(native_norm.real),
                            native_export_norm_imag=float(native_norm.imag))
                        save_checkpoint(output/tag/arm/"initialized_bank.pkl", native, optimizer_state=None,
                            metadata=dict(model=dict(length=12, multiplicity_cap=20, maximum_q=10),
                                j2=j2, arm=arm, protection=protection, embedding=embedding, optimizer_reset_required=True,
                                effective_folded_bank_not_rank8_binary=True, **quality))
                hq = np.asarray(candidate["blocks"][16+NODE].h_by_q[2])
                notes = dict(response=readout, **quality,
                    q2_projected_H_trace_mean=float(np.trace(hq).real)/len(hq),
                    q2_projected_H_eigenvalues=np.linalg.eigvalsh(hq).tolist(),
                    protected_columns_error=(None if arm == "energy8" and protection != "energy" else
                                              float(np.max(abs(frame[:, :count]-protected_reference[:, :count]), initial=0))),
                    frame_Gram_error=float(np.max(abs(frame.conj().T @ frame-np.eye(8)))),
                    seconds=perf_counter()-tick)
                np.savez_compressed(output/tag/(arm+"_thin_frame.npz"), q2_frame=frame)
                atomic_json(output/tag/(arm+".json"), notes); arm_reports[arm] = notes
                print(tag, arm, quality["raw_energy_per_site"], quality["normalized_RAW_fidelity_to_parent"],readout["heldout"]["fraction"],flush=True)
            gain = arm_reports[response_arm]["response"]["heldout"]["fraction"]-arm_reports["energy8"]["response"]["heldout"]["fraction"]
            quality = arm_reports[response_arm]
            flags = dict(heldout_response_gain=gain,
                heldout_response_gain_pass=gain >= .001,
                raw_fidelity_pass=quality["normalized_RAW_fidelity_to_parent"] >= fidelity_threshold,
                raw_energy_increase_pass=quality["raw_energy_per_site"]-energy_reference <= .002,
                full_A1_gate_passed=False)
            reports.append(dict(j2=j2, frozen_source=frozen, arms=arm_reports, diagnostic_flags=flags,
                response_residual_singular_values=response.singular_values.tolist(),
                response_fallback_columns=response.fallback_columns))
            atomic_json(output/"results.json", dict(registration=registration, reports=reports,
                seconds=perf_counter()-started, all_candidates_reported=True, no_outcome_dependent_tuning=True))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
    return dict(reports=reports)


def boundary_coupling_matrix(support: Sequence[int],
                             bonds: Sequence[tuple[int, int, float]]) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Return real J[inside,outside] for each crossing bond exactly once.

    Duplicate undirected bonds are rejected rather than silently doubled.
    Both positive and negative finite real couplings are allowed; PSD of
    J G.T J.T does not require ferromagnetic or antiferromagnetic signs.
    """
    sites = tuple(support)
    if (not sites or len(set(sites)) != len(sites)
            or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in sites)):
        raise ValueError("distinct nonnegative integer region sites required")
    inside = set(sites); seen = set(); cut = []
    for a, b, strength in bonds:
        if (any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in (a, b))
                or a == b or np.iscomplexobj(strength) or not np.isfinite(strength)):
            raise ValueError("physical bonds need distinct integer endpoints and finite real strength")
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError("duplicate undirected physical bond")
        seen.add(key)
        if strength and ((a in inside) != (b in inside)):
            cut.append((a, b, float(strength)) if a in inside else (b, a, float(strength)))
    if not cut:
        raise ValueError("the supplied Hamiltonian has no nonzero crossing bond")
    rows = tuple(sorted({a for a, _, _ in cut})); columns = tuple(sorted({b for _, b, _ in cut}))
    row_index = {s: i for i, s in enumerate(rows)}; col_index = {s: i for i, s in enumerate(columns)}
    coupling = np.zeros((len(rows), len(columns)))
    for a, b, strength in cut:
        coupling[row_index[a], col_index[b]] = strength
    return coupling, rows, columns


def _assemble_response_sites(cached: np.ndarray, cached_sites: Sequence[int],
                             added: np.ndarray | None, added_sites: Sequence[int],
                             target_sites: Sequence[int], source_width: int) -> np.ndarray:
    """Merge complete site-major frozen columns without changing their gauge."""
    old = tuple(cached_sites); new = tuple(added_sites); target = tuple(target_sites)
    if (len(set(old)) != len(old) or len(set(new)) != len(new) or set(old) & set(new)
            or set(old) | set(new) != set(target) or len(set(target)) != len(target)):
        raise ValueError("cached and added site sets must partition the exact target boundary")
    a = np.asarray(cached)
    if (source_width < 1 or a.ndim != 2 or a.shape[1] != len(old)*source_width
            or not np.isfinite(a).all()):
        raise ValueError("cached site-major response column shape is inconsistent")
    b = (np.empty((len(a), 0), dtype=a.dtype) if added is None else np.asarray(added))
    if b.shape != (len(a), len(new)*source_width) or not np.isfinite(b).all():
        raise ValueError("supplementary response columns do not match the same child/source space")
    lookup = {s: a[:, k*source_width:(k+1)*source_width] for k, s in enumerate(old)}
    lookup.update({s: b[:, k*source_width:(k+1)*source_width] for k, s in enumerate(new)})
    return np.stack([lookup[s] for s in target])


def root_ritz_probe(left: ReducedBlockData, right: ReducedBlockData,
                    bonds: Sequence[tuple[int, int, float]], initial: np.ndarray, *,
                    seed: int, tolerance: float = 1e-9, maxiter: int = 80,
                    krylov_dimension: int = 12) -> tuple[np.ndarray, dict]:
    """Optimize only the existing raw singlet root in the same child spaces.

    The reported residual is within this reduced product space. It is NOT
    physical full-H variance. Root-cross energy refers to the final two
    children, not necessarily to the smaller edge whose frame was changed.
    """
    if (not np.isfinite(tolerance) or tolerance <= 0 or isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer)) or seed < 0
            or isinstance(maxiter, bool) or maxiter < 1 or krylov_dimension < 3):
        raise ValueError("finite tolerance and bounded positive Ritz controls required")
    tick = perf_counter()
    op = SectorMergeOperator(left, right, bonds, 0, memory_limit_bytes=768*1024**2)
    parent = np.asarray(initial).reshape(-1)
    if parent.shape != (op.dimension,) or not np.isfinite(parent).all() or np.linalg.norm(parent) == 0:
        raise ValueError("initial normalized root must use exactly the declared singlet channels")
    parent = parent/np.linalg.norm(parent)
    hp = op.matvec(parent); old_energy = float(np.vdot(parent, hp).real)
    if op.dimension <= 64:
        dense = op.dense_reference(max_dimension=64)
        values, vectors = np.linalg.eigh(dense)
        vector = vectors[:, 0]; solver = "exact_small_sector_eigh"
    else:
        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=op.dimension)
        if op.dtype.kind == "c":
            v0 = v0+1j*rng.normal(size=op.dimension)
        _, vectors = eigsh(op.linear_operator(), k=1, which="SA", v0=v0,
                           tol=tolerance, maxiter=maxiter, ncv=min(op.dimension, krylov_dimension))
        vector = vectors[:, 0]; solver = "bounded_matrix_free_eigsh"
    vector = vector/np.linalg.norm(vector)
    overlap = np.vdot(parent, vector)
    if abs(overlap) > 1e-14:
        vector = vector*np.conj(overlap)/abs(overlap)
    h = op.matvec(vector); energy_complex = np.vdot(vector, h)
    energy = float(energy_complex.real); residual = float(np.linalg.norm(h-energy*vector))
    if abs(energy_complex.imag) > 1e-9 or residual > max(100*tolerance, 1e-8)*max(1., abs(energy)):
        raise ArithmeticError("raw root Ritz residual or imaginary energy exceeds tolerance")
    if energy > old_energy+max(1e-8, 10*tolerance*abs(old_energy)):
        raise ArithmeticError("root Ritz energy increased relative to the feasible parent")
    psi = op.expand(vector)
    internal = op.left.h @ psi+(op.right.h @ psi.T).T
    internal_energy = float(np.vdot(psi, internal).real)
    n_sites = left.n_sites+right.n_sites
    return vector[:, None], dict(solver=solver, total_q=0, reduced_dimension=op.dimension,
        seed=int(seed), tolerance=tolerance, maxiter=maxiter, krylov_dimension=krylov_dimension,
        raw_energy_before=old_energy, raw_energy_after=energy,
        raw_energy_per_site_before=old_energy/n_sites, raw_energy_per_site_after=energy/n_sites,
        root_children_internal_energy=internal_energy, root_cross_energy=energy-internal_energy,
        root_cross_energy_scope="crossing the final root cut only; not the modified36-site edge cut",
        reduced_product_space_residual_norm=residual,
        reduced_product_space_residual_squared=residual**2,
        physical_full_H_variance_known=False, full_A1_energy_known=False,
        lower_tensors_fixed=True, root_normalization_error=float(abs(np.vdot(vector, vector)-1)),
        same_child_basis_root_overlap_squared=float(abs(np.vdot(parent, vector))**2),
        reduced_H_matvec_count=op.matvec_count, reduced_H_matvec_seconds=op.matvec_seconds,
        seconds=perf_counter()-tick)


def oriented_root_coefficients(root_map: np.ndarray, root_channels: tuple, *,
                               center_on_left: bool) -> dict[int, np.ndarray]:
    """Put the selected root child first, including the SU(2) swap phase.

    Labels are doubled q. Swapping two equal-q multiplets coupled to a
    singlet gives (-1)**q. This is +1 for the even-spin CT12 halves, but the
    explicit factor also makes the coordinate helper valid for odd blocks.
    """
    from su2_nqs.operators.su2_conditional_response import root_coefficient_matrices
    if not isinstance(center_on_left, bool):
        raise ValueError('center_on_left must be boolean')
    roots = root_coefficient_matrices(root_map, root_channels)
    return {q: (value.copy() if center_on_left else (-1)**q * value.T.copy())
            for q, value in roots.items()}


def make_joint_upper_root_operator(state: dict, j2: float, *, upper_node: int = 12
                                  ) -> tuple[ThreeBlockSingletOperator,np.ndarray,dict]:
    """Expose the affected72/root center without freezing its72-dimensional map.

    Either existing72 node may be the center. Its two36 children and the
    complementary72 block remain fixed. No environment bond is discarded.
    Selecting13 enables an alternating upper sweep, not a new ansatz/head.
    """
    if isinstance(upper_node, bool) or upper_node not in (12, 13):
        raise ValueError('the upper center must be node12 or13')
    ancestor,root=upper_node,14
    a,b,ab=TREE.binary_merges[ancestor];rl,rr,total=TREE.binary_merges[root]
    center_on_left=ab==rl; e=rr if center_on_left else rl
    if (rl,rr,total)!=(28,29,30) or ab not in (rl,rr):
        raise ArithmeticError('registered36/36/72 hierarchy changed')
    _,joins,all_bonds=split_bond_ledger(12,j2)
    bonds_ae=[];bonds_be=[]
    for i,j,g in joins[root]:
        local,environment=(i,j) if center_on_left else (j,i)
        (bonds_ae if local<36 else bonds_be).append((local if local<36 else local-36,environment,g))
    op=ThreeBlockSingletOperator(state['blocks'][a],state['blocks'][b],state['blocks'][e],
        joins[ancestor],bonds_ae,bonds_be,memory_limit_bytes=768*1024**2,max_dimension=100_000)
    roots=oriented_root_coefficients(state['maps'][root][0],state['channels'][root][0],
                                    center_on_left=center_on_left)
    center=op.encode(state['maps'][ancestor],roots)
    if abs(np.vdot(center,center).real-1)>2e-10:
        raise ArithmeticError('parent joint center is not normalized')
    return op,center,dict(affected_upper_node=ancestor,root_node=root,center_on_left=center_on_left,
        physical_block_sites=[36,36,72],cross_AB_bonds=len(joins[ancestor]),
        cross_AE_bonds=len(bonds_ae),cross_BE_bonds=len(bonds_be),
        target_total_physical_bonds=len(all_bonds),all_weighted_target_bonds_accounted=True,
        raw_parent_energy_expected=float(np.asarray(state['blocks'][total].h_by_q[0])[0,0].real),
        lower_subtrees_fixed=True,affected72_and_root_jointly_variable=True,**op.diagnostics)


def joint_upper_root_probe(state: dict, source: dict, j2: float, *,
                          operator: ThreeBlockSingletOperator | None = None,
                          center: np.ndarray | None = None,
                          initial_product: np.ndarray | None = None,
                          krylov_vectors: int = 4,
                          original_q2_frame: np.ndarray | None = None,
                          upper_node: int = 12) -> tuple[dict,dict]:
    """One exact raw-H joint72/root update; caller owns registration and budget.

    This optional development path changes no production training. An old8
    control and nested8+2 arm must receive the same frozen Krylov budget.
    Complete-A1 amplitudes/energy must be audited separately after export.
    """
    if isinstance(upper_node, bool) or upper_node not in (12, 13):
        raise ValueError('the upper center must be node12 or13')
    ancestor,root=upper_node,14
    a,b,parent=TREE.binary_merges[ancestor]
    center_on_left=parent==TREE.binary_merges[root][0]
    if (operator is None)!=(center is None):
        raise ValueError('prepared operator and center must be supplied together')
    if operator is None:
        operator,center,preparation=make_joint_upper_root_operator(state,j2,upper_node=ancestor)
    else:
        preparation=dict(affected_upper_node=ancestor,root_node=root,
            raw_parent_energy_expected=float(state['blocks'][30].h_by_q[0][0,0].real),
            caller_prepared_exact_three_block_operator=True)
    hparent=operator.matvec(center) if initial_product is None else np.asarray(initial_product)
    parent_energy=np.vdot(center,hparent)
    discrepancy=abs(parent_energy-preparation['raw_parent_energy_expected'])
    if discrepancy>2e-8:
        raise ArithmeticError('joint full-target H disagrees with the stored physical parent energy')
    optimized,quality=parent_anchored_center_ritz(operator,center,krylov_vectors=krylov_vectors,
                                                initial_product=hparent)
    widths={q:w.shape[1] for q,w in state['maps'][ancestor].items()}
    maps,roots,factorization=operator.factor(optimized,widths=widths)
    maximum_imaginary=max(float(np.max(abs(x.imag),initial=0.))
                          for group in (maps,roots) for x in group.values())
    # The deployed native bank exporter is explicitly real. Never silently
    # discard a legitimate complex update to make it fit that interface.
    if maximum_imaginary>2e-11:
        raise ArithmeticError('native real exporter cannot represent this complex joint-center update')
    candidate=deepcopy(state);candidate['maps']=_independent_node_map_copy(state['maps'])
    candidate['cache']={};candidate['leaf_cache']={}
    candidate['parent_initialization_records']=candidate.pop('records',[]);candidate['records']=[]
    candidate['maps'][ancestor]={q:np.asarray(w.real).copy() for q,w in maps.items()}
    candidate['channels']=deepcopy(state['channels'])
    candidate['channels'][ancestor]={q:operator.pairs['AB'][q][0].channels for q in maps}
    _,joins,_=split_bond_ledger(12,j2)
    candidate['blocks'][parent]=project_sector_maps(candidate['blocks'][a],candidate['blocks'][b],joins[ancestor],
        maps=candidate['maps'][ancestor],compact_boundary=True,max_output_dimension=288,
        memory_limit_bytes=768*1024**2)
    root_channels=state['channels'][root][0]
    candidate['maps'][root][0]=np.asarray([
        (roots[ql][il,ir] if center_on_left else (-1)**ql*roots[ql][ir,il]).real
        for ql,il,qr,ir in root_channels])[:,None]
    # A one-dimensional singlet root has identically zero projected vector
    # operators. Its scalar H is the exact just-evaluated center expectation.
    candidate['blocks'][30]=replace(candidate['blocks'][30],
        h_by_q={0:np.asarray([[quality['raw_energy_after']]])})
    changed=_checked_changed_nodes(state['maps'],candidate['maps'],allowed={ancestor,root})
    replay_roots=oriented_root_coefficients(candidate['maps'][root][0],root_channels,
                                           center_on_left=center_on_left)
    replay=operator.encode(candidate['maps'][ancestor],replay_roots)
    replay_error=float(np.max(abs(replay-optimized)))
    if replay_error>2e-10:raise ArithmeticError('native upper/root coordinates lost the optimized center')
    occupations={}
    for name,current in (('before',state),('after',candidate)):
        env,nodes,defect=active_environment(current,source)
        down=np.asarray(env.downward[nodes[24]][2]);frame=np.asarray(current['maps'][8][2])
        item=dict(q2_weight=float(3*np.trace(down).real/env.norm_squared),raw_norm_squared=env.norm_squared,
                  upward_Gram_error=defect)
        if original_q2_frame is not None:
            overlap=frame.conj().T@np.asarray(original_q2_frame)
            outside=np.eye(frame.shape[1])-overlap@overlap.conj().T
            item['physical_raw_weight_outside_original_q2_subspace']=float(
                3*np.trace(outside@down).real/env.norm_squared)
        occupations[name]=item
    return candidate,dict(preparation=preparation,joint_Ritz=quality,factorization=factorization,
        parent_H_energy_replay_error=float(discrepancy),coordinate_replay_error=replay_error,
        maximum_discarded_imaginary_roundoff=maximum_imaginary,actually_changed_nodes=changed,
        occupations=occupations,source_lower_tensors_array_equal=True,
        complete_A1_energy_known=False,full_physical_H_variance_known=False,
        effective_H_scope='all target physical bonds projected into fixedA36/B36/E72 child spaces',
        upper72_output_widths=widths,
        additional_upper72_coarse_H_columns=sum(widths.values()),
        additional_root_coarse_H_columns=0,no_new_head=True)


def embed_in_nested_boundary_frame(state: dict, expanded: dict) -> tuple[dict,dict]:
    """Exactly zero-pad an old8 state into the same active layout as nested10.

    This is for matched compiled full-A1 auditing, not a physical update.
    The new36 frame MUST retain every old column bit-for-bit as its prefix.
    """
    old=np.asarray(state['maps'][8][2]);new=np.asarray(expanded['maps'][8][2])
    if new.shape[0]!=old.shape[0] or new.shape[1]<old.shape[1] or not np.array_equal(new[:,:old.shape[1]],old):
        raise ValueError('nested boundary frame must exactly retain the full old-frame prefix')
    result=deepcopy(state);result['maps']=_independent_node_map_copy(state['maps'])
    result['channels']=deepcopy(state['channels']);result['cache']={};result['leaf_cache']={}
    result['maps'][8][2]=new.copy();result['blocks'][24]=expanded['blocks'][24]
    for q,matrix in state['maps'][12].items():
        labels=expanded['channels'][12][q];index={label:i for i,label in enumerate(labels)}
        embedded=np.zeros((len(labels),matrix.shape[1]),dtype=matrix.dtype)
        for i,label in enumerate(state['channels'][12][q]):embedded[index[label]]=matrix[i]
        result['maps'][12][q]=embedded;result['channels'][12][q]=labels
    return result,dict(physical_state_preserved_by_exact_zero_padding=True,
        original_boundary_width=old.shape[1],audit_boundary_width=new.shape[1],
        new_boundary_paths_have_exact_zero_parent_coefficients=True)


def run_joint_upper_root_environment_probe(output: Path, parent_state: Path, expanded_state: Path, *,
                                           j2: float = .125, seconds_cap: float = 600.,
                                           krylov_vectors: int = 4) -> dict:
    """Registered serial STEP2: old8 versus nested10, identical exact-H protocol.

    PREPARATION ONLY until the caller has explicitly approved STEP1. No CLI
    automatically invokes this function, and no production task is changed.
    """
    if j2!=.125 or krylov_vectors!=4 or not np.isfinite(seconds_cap) or not 0<seconds_cap<=600:
        raise ValueError('frozen J2=.125, four-vector protocol and at most600seconds required')
    if output.exists() and any(output.iterdir()):raise FileExistsError('refusing to overwrite a registered comparison')
    output.mkdir(parents=True,exist_ok=True)
    source,provenance=load_source(j2)
    files=[Path(__file__),Path(__file__).parents[3]/'su2_nqs/operators/su2_three_block_center.py',
           Path(__file__).parents[3]/'su2_nqs/operators/su2_sector_merge.py']
    registration=dict(task='joint72_root_same_protocol_old8_vs_nested10_v1',j2=j2,
        parent_state=str(parent_state),parent_state_sha256=file_sha256(parent_state),
        expanded_state=str(expanded_state),expanded_state_sha256=file_sha256(expanded_state),
        source_parent=provenance,source_sha256={str(p):file_sha256(p) for p in files},
        maximum_wall_seconds=seconds_cap,fixed_center_Krylov_vectors_per_arm=4,
        arms=['old8_joint','nested10_joint'],lower_subtrees_fixed=True,
        full_target_weighted_raw_H=True,MC_queries=0,full_A1_energy_not_yet_known=True,
        no_new_head=True,optimization_parent_in_both_trial_spaces=True,
        cost_gate='time one exact center H per arm; run remaining3 each only with projected sufficient common budget',
        fullA1_audit_export_same_nested_active_layout=True)
    atomic_json(output/'registration.json',registration)
    for p in files:copyfile(p,output/('registered_'+p.name))
    started=perf_counter();result=dict(registration=registration,status='started',arms={})
    previous=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('joint72/root comparison reached the frozen wall bound')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds_cap)
    try:
        original=load_checkpoint(parent_state)['params'];expanded=load_checkpoint(expanded_state)['params']
        # This verifies exact nesting before either H cost measurement.
        padded,embed_notes=embed_in_nested_boundary_frame(original,expanded)
        for node in set(original['maps'])-{8,12,14}:
            if any(not np.array_equal(original['maps'][node][q],expanded['maps'][node][q])
                   for q in original['maps'][node]):
                raise ArithmeticError('STEP1 changed an unrelated lower subtree')
        prepared={};costs={}
        for name,state in (('old8_joint',original),('nested10_joint',expanded)):
            op,center,notes=make_joint_upper_root_operator(state,j2)
            tick=perf_counter();hp=op.matvec(center);cost=perf_counter()-tick
            error=abs(np.vdot(center,hp)-notes['raw_parent_energy_expected'])
            if error>2e-8:raise ArithmeticError('preflight full-target center H failed parent replay')
            prepared[name]=(op,center,hp,state);costs[name]=cost
            result['arms'][name]=dict(preflight=notes,parent_H_seconds=cost,parent_energy_replay_error=float(error))
            atomic_json(output/'results.json',result)
        if abs(prepared['old8_joint'][3]['blocks'][30].h_by_q[0][0,0]-
               prepared['nested10_joint'][3]['blocks'][30].h_by_q[0][0,0])>2e-9:
            raise ArithmeticError('nested STEP1 no longer has the same physical parent energy')
        remaining=seconds_cap-(perf_counter()-started)
        forecast=1.5*3*sum(costs.values())+120.
        result['common_cost_gate']=dict(remaining_seconds=remaining,forecast_seconds=forecast,
                                       same_four_products_for_both=True)
        if forecast>remaining:
            result['status']='cost_only_no_optimizer_released';return result
        endpoints={}
        for name,(op,center,hp,state) in prepared.items():
            candidate,notes=joint_upper_root_probe(state,source,j2,operator=op,center=center,
                initial_product=hp,krylov_vectors=4,original_q2_frame=original['maps'][8][2])
            endpoints[name]=candidate;result['arms'][name]['optimization']=notes
            result['arms'][name]['center_H_matvecs_including_preflight']=op.matvec_count
            save_checkpoint(output/name/'state.pkl',candidate,optimizer_state=None,
                metadata=dict(task=registration['task'],arm=name,raw_only=True,**notes))
            atomic_json(output/'results.json',result)
        exports={'parent':padded,'preserved_expanded':expanded,
                 'old8_joint':embed_in_nested_boundary_frame(endpoints['old8_joint'],expanded)[0],
                 'nested10_joint':endpoints['nested10_joint']}
        for name,state in exports.items():
            maximum_imaginary=max(float(np.max(abs(np.asarray(w).imag),initial=0.))
                                  for row in state['maps'].values() for w in row.values())
            if maximum_imaginary>2e-11:raise ArithmeticError('native exporter would discard a complex state')
            realstate=deepcopy(state);realstate['maps']={node:{q:np.asarray(w.real).copy() for q,w in row.items()}
                                                      for node,row in state['maps'].items()}
            bank,embedding=native_bank_from_state(source,realstate)
            metadata=dict(task=registration['task'],arm=name,j2=j2,
                model=dict(length=12,multiplicity_cap=20,maximum_q=10),embedding=embedding,
                source=provenance,full_A1_energy_not_yet_audited=True,optimizer_reset_required=True,
                raw_energy_per_site=float(state['blocks'][30].h_by_q[0][0,0].real)/144)
            save_checkpoint(output/name/'initialized_bank.pkl',bank,optimizer_state=None,metadata=metadata)
            atomic_json(output/name/'native_metadata.json',metadata)
        result.update(status='complete_raw_joint_comparison',exact_parent_embedding=embed_notes,
            expanded_minus_old_joint_raw_energy_per_site=(
                result['arms']['nested10_joint']['optimization']['joint_Ritz']['raw_energy_after']-
                result['arms']['old8_joint']['optimization']['joint_Ritz']['raw_energy_after'])/144)
        return result
    except Exception as error:
        result.update(status='stopped_or_failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result['wall_seconds']=perf_counter()-started
        atomic_json(output/'results.json',result)


def run_effective_environment_probe(output: Path, source_run: Path, response_cache: Path, *,
                                    seconds_cap: float = 180.) -> dict:
    """Bounded cached same-width test of incoherent versus geometric baths.

    The bath is G=I, not an inferred correlated environment. Its offdiagonal
    boundary kernel arises solely from shared outside physical sites. All
    actual J1/J2 cut bonds are included. Both arms retain six identical
    density directions and change only two q=2 multiplicity directions.
    A fixed protocol then optimizes their raw singlet root; no MC, full-A1
    Hamiltonian query, new neural head or physical-variance claim is made.
    """
    if not 30 <= seconds_cap <= 180:
        raise ValueError("the fixed CPU and wall budgets must lie in30..180 seconds")
    output = Path(output); source_run = Path(source_run); response_cache = Path(response_cache)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a fresh output directory; evidence is immutable")
    inputs = {tag: source_run/tag/"physical/state.pkl" for tag in ("j0125", "j0")}
    cache_registration = json.loads((response_cache/"registration.json").read_text())
    input_hashes = {tag: file_sha256(path) for tag, path in inputs.items()}
    if cache_registration.get("source_sha256") != input_hashes:
        raise ValueError("cached response source hashes do not identify these exact frozen parents")
    cache_files = {tag: response_cache/tag/"frozen_response_columns.npz" for tag in inputs}
    metadata_files = {tag: response_cache/tag/"frozen_source.json" for tag in inputs}
    registration = dict(task="same_width_geometry_coherent_boundary_bath_v1", node=NODE,
        changed_q=2, width=8, protected_density_columns=6, couplings=[.125, 0.],
        arms=["degree_diagonal", "geometry_coherent"], cached_parent_reference="energy8",
        source_inputs={tag: str(path) for tag, path in inputs.items()}, source_sha256=input_hashes,
        response_cache=str(response_cache), response_cache_sha256={tag: file_sha256(path) for tag, path in cache_files.items()},
        response_metadata_sha256={tag: file_sha256(path) for tag, path in metadata_files.items()},
        response_source="frozen raw144 conditional regionq0 density; q0to2 only",
        full_actual_J1_J2_cut_included=True, product_bath_Gram="identity; uncorrelated isotropic bath model",
        kernel_convention="K=J G.T J.T; diagonal arm drops only the offdiagonal of this K",
        bath_model_not_measured_quantum_environment=True, no_full_A1_or_MC_H_queries=True,
        root_Ritz=dict(seed=2610234800, tolerance=1e-9, maxiter=80, krylov_dimension=12),
        cpu_seconds_cap=float(seconds_cap), wall_seconds_cap=float(seconds_cap),
        stop_rule="fixed budget or exception; retain all completed arms without outcome-based extension",
        no_neural_network=True, physical_variance_unknown=True, projected_A1_quality_unknown=True,
        workflow_sha256=file_sha256(Path(__file__)),
        helper_sha256=file_sha256(Path(__file__).parents[3]/"su2_nqs/frontier/retained_boundary_response.py"),
        merge_kernel_sha256=file_sha256(Path(__file__).parents[3]/"su2_nqs/operators/su2_sector_merge.py"))
    output.mkdir(parents=True, exist_ok=True)
    for original, name, registered_hash in (
            (Path(__file__), "registered_workflow.py", registration["workflow_sha256"]),
            (Path(__file__).parents[3]/"su2_nqs/frontier/retained_boundary_response.py",
             "registered_response_helper.py", registration["helper_sha256"]),
            (Path(__file__).parents[3]/"su2_nqs/operators/su2_sector_merge.py",
             "registered_sector_merge.py", registration["merge_kernel_sha256"])):
        copyfile(original, output/name)
        if file_sha256(output/name) != registered_hash:
            raise ArithmeticError("source changed while freezing the pre-run code snapshot")
    atomic_json(output/"registration.json", registration)
    started = perf_counter(); cpu_started = process_time(); reports = []
    previous_alarm = signal.getsignal(signal.SIGALRM); previous_prof = signal.getsignal(signal.SIGPROF)
    def timeout(*_):
        raise TimeoutError("fixed effective-environment CPU/wall budget exhausted")
    signal.signal(signal.SIGALRM, timeout); signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds_cap); signal.setitimer(signal.ITIMER_PROF, seconds_cap)
    result = dict(registration=registration, reports=reports, status="running")
    try:
        for j2 in (.125, 0.):
            tag = "j0125" if j2 else "j0"; directory = output/tag; directory.mkdir()
            state = load_checkpoint(inputs[tag])["params"]
            source, source_provenance = load_source(j2)
            cached_meta = json.loads(metadata_files[tag].read_text())
            if cached_meta.get("source", {}).get("checkpoint_sha256") != source_provenance["checkpoint_sha256"]:
                raise ValueError("physical leaf checkpoint does not match the frozen response cache")
            with np.load(cache_files[tag]) as loaded:
                cached_columns = np.array(loaded["all"]); down_q2 = np.array(loaded["down_q2"])
            J, inside_sites, outside_sites = boundary_coupling_matrix(TREE.coarse_supports[16+NODE],
                triangular_j1j2_bonds(12, 12, j2=j2, periodic=True))
            cached_sites = tuple(cached_meta["boundary_sites"])
            missing = tuple(s for s in inside_sites if s not in cached_sites)
            added = None
            if missing:
                supplement, weight = frozen_responses(state, source, boundary_sites=missing)
                if np.max(abs(weight["down_q2"]-down_q2)) > 2e-12:
                    raise ArithmeticError("cached and supplementary response source gauges disagree")
                added = supplement["all"]
            source_width = state["maps"][NODE][0].shape[1]
            responses = _assemble_response_sites(cached_columns, cached_sites, added, missing,
                                                 inside_sites, source_width)
            G = np.eye(len(outside_sites)); K = J @ J.T
            coherent, coherent_notes = correlated_bath_response_factor(2, responses, couplings=J, bath_gram=G)
            diagonal, diagonal_notes = correlated_bath_response_factor(2, responses,
                couplings=np.diag(np.sqrt(np.diag(K))), bath_gram=np.eye(len(inside_sites)))
            factors = dict(degree_diagonal=diagonal, geometry_coherent=coherent)
            energy_frame = np.asarray(state["maps"][NODE][2])
            rho = (down_q2+down_q2.conj().T)/(2*np.trace(down_q2).real)
            values, vectors = np.linalg.eigh(rho); order = np.argsort(values)[::-1]
            values, vectors = values[order], vectors[:, order]
            if len(values) != 8 or energy_frame.shape[1] != 8 or values.min() < -1e-12:
                raise ValueError("registered width8 conditional physical density is invalid")
            protected = energy_frame @ vectors
            p2 = float(3*np.trace(down_q2).real)
            loss_bound = float(p2*values[6:].sum())
            if loss_bound > .005+2e-12:
                raise ValueError("fixed six-column protection no longer meets its registered raw norm bound")
            _, joins, _ = split_bond_ledger(12, j2)
            root = len(TREE.binary_merges)-1; rl, rr, rp = TREE.binary_merges[root]
            parent_root, parent_ritz = root_ritz_probe(state["blocks"][rl], state["blocks"][rr], joins[root],
                state["maps"][root][0], **registration["root_Ritz"])
            row = dict(j2=j2, source_provenance=source_provenance, source_hash=input_hashes[tag],
                cached_response_sites=list(cached_sites), newly_computed_response_sites=list(missing),
                boundary_inside_sites=list(inside_sites), boundary_outside_sites=list(outside_sites),
                crossing_bonds=int(np.count_nonzero(J)), response_shape=list(responses.shape),
                geometry_kernel_rank=int(np.linalg.matrix_rank(J)),
                kernel_offdiagonal_Frobenius=float(np.linalg.norm(K-np.diag(np.diag(K)))),
                kernel_offdiagonal_absolute_sum=float(np.sum(abs(K-np.diag(np.diag(K))))),
                conditional_q2_eigenvalues_descending=values.tolist(), physical_q2_probability=p2,
                protected_RAW_norm_loss_upper_bound=loss_bound,
                factor_diagnostics=dict(degree_diagonal=diagonal_notes, geometry_coherent=coherent_notes),
                parent_root_Ritz=parent_ritz, arms={})
            reports.append(row)
            np.savez_compressed(directory/"frozen_environment.npz", responses=responses, coupling_matrix=J,
                bath_Gram=G, kernel=K, down_q2=down_q2, protected_reference=protected,
                parent_Ritz_root=parent_root)
            atomic_json(output/"results.json", result)
            for arm in registration["arms"]:
                tick = perf_counter()
                selected = fixed_q_response_frame(2, protected, factors[arm], protect=6)
                candidate, quality = project_candidate(state, selected.frame, j2, down_q2)
                if quality["normalized_RAW_fidelity_to_parent"] < 1-loss_bound-2e-10:
                    raise ArithmeticError("fixed protected density subspace violated the exact raw norm bound")
                new_root, ritz = root_ritz_probe(candidate["blocks"][rl], candidate["blocks"][rr], joins[root],
                    candidate["maps"][root][0], **registration["root_Ritz"])
                candidate["maps"][root][0] = new_root
                candidate["blocks"][rp] = replace(candidate["blocks"][rp],
                    h_by_q={0: np.asarray([[ritz["raw_energy_after"]]])})
                env, _, gram_defect = active_environment(candidate, source)
                if abs(env.norm_squared-1.) > 2e-11:
                    raise ArithmeticError("root-optimized candidate raw normalization failed")
                details = dict(projection=quality, root_Ritz=ritz,
                    raw_energy_delta_to_equally_root_optimized_parent_per_site=(
                        ritz["raw_energy_per_site_after"]-parent_ritz["raw_energy_per_site_after"]),
                    capture_under_both_frozen_kernels={name: response_capture(selected.frame, factor)
                                                     for name, factor in factors.items()},
                    retained_response_fraction_is_training_objective_not_validation=True,
                    frame_Gram_error=selected.isometry_error, raw_Gram_error=gram_defect,
                    raw_norm_squared=env.norm_squared,
                    full_physical_H_variance_known=False, complete_A1_energy_known=False,
                    seconds=perf_counter()-tick)
                save_checkpoint(directory/arm/"state.pkl", candidate, optimizer_state=None,
                    metadata=dict(task=registration["task"], parent_sha256=input_hashes[tag],
                        same_width=True, no_MC_training=True, root_Ritz_only=True, optimizer_reset_required=True,
                        **details))
                np.savez_compressed(directory/(arm+"_frame_and_root.npz"), q2_frame=selected.frame, root=new_root)
                row["arms"][arm] = details
                atomic_json(directory/(arm+".json"), details)
                result.update(cpu_seconds=process_time()-cpu_started, wall_seconds=perf_counter()-started)
                atomic_json(output/"results.json", result)
            row["coherent_minus_diagonal_raw_energy_per_site"] = (
                row["arms"]["geometry_coherent"]["root_Ritz"]["raw_energy_per_site_after"]
                -row["arms"]["degree_diagonal"]["root_Ritz"]["raw_energy_per_site_after"])
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error_type=type(error).__name__, error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGALRM, previous_alarm); signal.signal(signal.SIGPROF, previous_prof)
        result.update(cpu_seconds=process_time()-cpu_started, wall_seconds=perf_counter()-started,
            MC_full_H_queries=0, complete_A1_H_queries=0, physical_full_H_variance_known=False)
        atomic_json(output/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--seconds-cap", type=float, default=180.)
    parser.add_argument("--protection", choices=("energy", "density", "adaptive"), default="energy")
    args = parser.parse_args(); run(args.output, args.source_run, args.seconds_cap, protection=args.protection)
