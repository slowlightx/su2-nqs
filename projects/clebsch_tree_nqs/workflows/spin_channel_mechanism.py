"""Non-saturated physical channel ablations around a complete K2 anchor.

No model training occurs here. Operators act on physical amplitudes before
exact A1 compression. A fixed family is tested with all directions retained;
there is no energy-driven selection, truncation or hyperparameter search.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.benchmarks.spectral_error_tomography import lowest_subspace_state, orthonormal_column_space
from su2_nqs.benchmarks.su2_entanglement_spectrum import (
    su2_resolved_entanglement, compare_entanglement_reports,
    json_ready_entanglement_report, singlet_entropy_components,
)
from su2_nqs.benchmarks.triangular_state_manifold import TRIANGULAR_N16_RHOMBUS_SUPPORTS
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.operators.cell_spin_channels import cell_casimir_action, four_site_spin_modes, weighted_cell_exchange_action
from su2_nqs.operators.cell_spin_transitions import cell_spin_preserving_action, four_site_dimer_action
from .spin_n16_physical_fmove_carrier_gate import ROOT, _load_source, _fixed_sz_configurations, _metrics
from .spin_channel_attention import _krylov


ARMS = {
    "q_transfer": (0, 1, 2),
    "q_preserving_exchange": (3, 4, 5),
    "multiplicity_dimer": (6, 7),
    "q_transfer_plus_multiplicity": (0, 1, 2, 6, 7),
    "q_preserving_plus_multiplicity": (3, 4, 5, 6, 7),
    "split_exchange": (0, 1, 2, 3, 4, 5),
    "all_channels": tuple(range(8)),
}


def build_transition_bank(phi: np.ndarray, basis: tuple) -> tuple[np.ndarray, dict, dict]:
    """Build three q-transfer, three q-keep and two pure multiplicity scalars."""
    pairs = FixedSzPairActions(basis)
    calls = {"batch_calls": 0, "vector_equivalents": 0}

    def pair(v: np.ndarray, i: int, j: int) -> np.ndarray:
        calls["batch_calls"] += 1
        calls["vector_equivalents"] += v.shape[1] if v.ndim == 2 else 1
        return .5 * v[pairs.indices(i, j)] - .25 * v

    a, b = TRIANGULAR_N16_RHOMBUS_SUPPORTS[:2]
    changed, kept, dimers = [], [], []
    singles = {}
    costs = {}
    checks = {"raw_hermiticity_max": 0., "spin_raising_relative_max": 0., "q_preserving_commutator_relative_max": 0.}

    def compress(image: np.ndarray) -> np.ndarray:
        block = phi.conj().T @ image
        checks["raw_hermiticity_max"] = max(checks["raw_hermiticity_max"], float(np.max(np.abs(block-block.conj().T))))
        if np.linalg.norm(image[:, 0]) > 1e-12:
            checks["spin_raising_relative_max"] = max(checks["spin_raising_relative_max"], spin_raising_relative_norm(image[:, 0], basis))
        return (block+block.conj().T)/2

    for mu, weights in enumerate(four_site_spin_modes()[1:]):
        action = lambda v, w=weights: weighted_cell_exchange_action(v, pair, a, b, w, w)
        started, before = perf_counter(), dict(calls)
        raw = action(phi)
        costs[f"raw_exchange_{mu}"] = {k: calls[k]-before[k] for k in calls}
        costs[f"raw_exchange_{mu}"]["wall_seconds"] = perf_counter()-started
        started, before = perf_counter(), dict(calls)
        keep = cell_spin_preserving_action(phi, action, pair, (a, b))
        costs[f"pinched_exchange_{mu}"] = {k: calls[k]-before[k] for k in calls}
        costs[f"pinched_exchange_{mu}"]["wall_seconds"] = perf_counter()-started
        changed.append(compress(raw-keep))
        kept.append(compress(keep))
        probe = phi[:, 0]
        for cell in (a, b):
            comm = cell_casimir_action(keep[:, 0], pair, cell) - cell_spin_preserving_action(cell_casimir_action(probe, pair, cell), action, pair, (a, b))
            checks["q_preserving_commutator_relative_max"] = max(checks["q_preserving_commutator_relative_max"], float(np.linalg.norm(comm)/np.linalg.norm(probe)))
    for mu in (0, 1):
        started, before = perf_counter(), dict(calls)
        left = four_site_dimer_action(phi, pair, a, mu)
        right = four_site_dimer_action(phi, pair, b, mu)
        product_image = four_site_dimer_action(right, pair, a, mu)
        costs[f"dimer_pair_{mu}_including_single_observables"] = {k: calls[k]-before[k] for k in calls}
        costs[f"dimer_pair_{mu}_including_single_observables"]["wall_seconds"] = perf_counter()-started
        dimers.append(compress(product_image))
        singles[f"a{mu}"] = compress(left)
        singles[f"b{mu}"] = compress(right)
    checks["pair_action_counts_including_validation"] = dict(calls)
    return np.asarray(changed+kept+dimers), singles, {"checks": checks, "construction_costs": costs}


def rotate_complement(anchor: np.ndarray, seed: int) -> np.ndarray:
    """Orthogonal scramble fixing EVERY vector in the complete anchor space."""
    if np.max(np.abs(anchor.imag)) > 1e-12:
        raise ValueError("registered scramble expects a real anchor")
    q, _ = np.linalg.qr(anchor.real, mode="complete")
    rank = anchor.shape[1]
    rest = q[:, rank:]
    random = np.random.default_rng(seed).normal(size=(rest.shape[1], rest.shape[1]))
    rotation, r = np.linalg.qr(random)
    rotation = rotation @ np.diag(np.sign(np.diag(r)))
    return q[:, :rank] @ q[:, :rank].T + rest @ rotation @ rest.T


def _observables(state: np.ndarray, phi: np.ndarray, basis: tuple, reference: list,
                 h1: np.ndarray, h2: np.ndarray, bank: np.ndarray, singles: dict) -> dict:
    physical = phi @ state
    a, b = TRIANGULAR_N16_RHOMBUS_SUPPORTS[:2]
    result = {}
    for name, sites, ref in zip(("one_cell", "two_cells"), (a, a+b), reference, strict=True):
        report = su2_resolved_entanglement(physical, basis, sites)
        result[name] = json_ready_entanglement_report(report)
        result[name]["entropy_components"] = singlet_entropy_components(report)
        result[name]["errors_to_exact"] = compare_entanglement_reports(report, ref)
    result["h1_per_site"] = float(np.vdot(state, h1@state).real/16)
    result["h2_per_site"] = float(np.vdot(state, h2@state).real/16)
    result["connected_dimer_correlations"] = [float(
        np.vdot(state, bank[6+mu]@state).real
        - np.vdot(state, singles[f"a{mu}"]@state).real*np.vdot(state, singles[f"b{mu}"]@state).real
    ) for mu in (0, 1)]
    return result


def run_spin_channel_mechanism(config_path: Path, output: Path, *, overwrite: bool = False) -> dict:
    """Run the registered non-saturated operator mechanism experiment."""
    spec = json.loads(config_path.read_text())
    if spec.get("task") != "spin_channel_mechanism" or spec.get("preregistered_before_run") is not True or spec.get("remote_training_on_pass") is not False:
        raise ValueError("expected registered local mechanism experiment")
    if set(spec["arms"]) != set(ARMS) | {"unsplit_exchange"} or spec["krylov_anchor_rank"] != 3:
        raise ValueError("do not silently change the registered arms or anchor")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"results.json").exists() and not overwrite:
        raise FileExistsError(output/"results.json")
    _, arrays = _load_source(spec)
    phi, basis = arrays["physical_a1_basis"], _fixed_sz_configurations()
    bank, singles, diagnostics = build_transition_bank(phi, basis)
    if max(v for k,v in diagnostics["checks"].items() if k != "pair_action_counts_including_validation") > 2e-10:
        raise RuntimeError("exact operator checks failed before variational comparison")
    h1 = arrays["coupling_0_h_a1"]
    h2 = (arrays["coupling_0.125_h_a1"]-h1)/.125
    pi = arrays["coupling_0.125_carrier_a1"]
    cases, stored = {}, {"operator_bank": bank, **singles}
    tolerance = float(spec["relative_rank_tolerance"])
    for coupling in spec["couplings"]:
        h = h1 + float(coupling)*h2
        anchor = _krylov(pi, h, 3)
        carrier, _ = lowest_subspace_state(anchor, h)
        _, vecs = np.linalg.eigh(h)
        exact = vecs[:, 0]
        a,b = TRIANGULAR_N16_RHOMBUS_SUPPORTS[:2]
        reference = [su2_resolved_entanglement(phi@exact, basis, cut) for cut in (a,a+b)]
        images = np.einsum("kij,j->ki", bank, carrier)
        case = {"arms": {}, "controls": {}, "a1_dimension": int(h.shape[0])}
        for name,state in (("two_lanczos",carrier),("exact_a1",exact)):
            case["controls"][name] = dict(_metrics(state,h), physics=_observables(state,phi,basis,reference,h1,h2,bank,singles))
            stored[f"j{coupling}_{name}"] = state
        for name in spec["arms"]:
            directions = images[:3]+images[3:6] if name == "unsplit_exchange" else images[list(ARMS[name])]
            columns = np.column_stack((anchor,directions.T))
            state, rank = lowest_subspace_state(columns,h,relative_tolerance=tolerance)
            if rank > int(spec["maximum_allowed_rank"]) or rank >= h.shape[0]:
                raise RuntimeError("registered non-saturation boundary exceeded")
            control, control_rank = lowest_subspace_state(_krylov(pi,h,rank),h)
            row = dict(_metrics(state,h), rank=rank, physics=_observables(state,phi,basis,reference,h1,h2,bank,singles))
            row["matched_krylov"] = dict(_metrics(control,h),rank=control_rank,physics=_observables(control,phi,basis,reference,h1,h2,bank,singles))
            row["scrambles"] = []
            gram_error = anchor_error = 0.
            for seed in spec["scramble_seeds"]:
                rotation = rotate_complement(anchor,seed)
                rotated = directions@rotation.T
                gram_error = max(gram_error,float(np.max(np.abs(rotated@rotated.conj().T-directions@directions.conj().T))))
                anchor_error = max(anchor_error,float(np.max(np.abs(rotation@anchor-anchor))))
                candidate, srank = lowest_subspace_state(np.column_stack((anchor,rotated.T)),h,relative_tolerance=tolerance)
                if srank != rank:
                    raise RuntimeError("scramble changed variational dimension")
                row["scrambles"].append(dict(_metrics(candidate,h),rank=srank,seed=seed))
            row["scramble_gram_error"] = gram_error
            row["scramble_anchor_error"] = anchor_error
            row["energy_minus_matched_krylov_per_site"] = row["energy_per_site"]-row["matched_krylov"]["energy_per_site"]
            row["beats_matched_krylov_energy_and_variance"] = row["energy_minus_matched_krylov_per_site"] < -1e-12 and row["variance_total"] < row["matched_krylov"]["variance_total"]
            row["scramble_joint_wins"] = sum(row["energy_per_site"] < s["energy_per_site"] and row["variance_total"] < s["variance_total"] for s in row["scrambles"])
            # Exact projected-H work for Ritz, not physical full-H evaluator work.
            row["ritz_projected_h_matvecs"] = rank
            case["arms"][name] = row
            stored[f"j{coupling}_{name}"] = state
            stored[f"j{coupling}_{name}_krylov"] = control
            print(f"J2={coupling} {name}: rank={rank}, E/N={row['energy_per_site']:.12f}, Var={row['variance_total']:.6g}, deltaK={row['energy_minus_matched_krylov_per_site']:+.4g}, scramble={row['scramble_joint_wins']}/5",flush=True)
        cases[str(coupling)] = case
    np.savez_compressed(output/"states.npz",**stored)
    result = {
        "task": "spin_channel_mechanism", "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec, "diagnostics": diagnostics, "cases": cases,
        "model": "triangular 4x4 PBC, S=0 Gamma-A1, original pi-flux K2 anchor",
        "claim_boundary": "Pre-A1 local-q mechanism decomposition; no neural training or scalable efficiency claim. Same rank is not same pair-action cost. Spectral pinching is intentionally an exact-small diagnostic. No phase diagram from this torus (K absent).",
        "remote_training_licensed": False,
        "bundle_sha256": file_sha256(output/"states.npz"),
        "source_manifest": transitive_local_source_manifest((Path(__file__).resolve(),),root=ROOT,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs")),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(output/"results.json",result)
    return result
