"""Physical local-irrep transitions plus conditional scalar attention oracle."""

from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.channel_attention import train_channel_schedule
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.benchmarks.spectral_error_tomography import lowest_subspace_state
from su2_nqs.benchmarks.su2_entanglement_spectrum import (
    su2_resolved_entanglement, compare_entanglement_reports,
    json_ready_entanglement_report, singlet_entropy_components,
)
from su2_nqs.benchmarks.triangular_state_manifold import TRIANGULAR_N16_RHOMBUS_SUPPORTS
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.operators.cell_spin_channels import (
    cell_spin_projector_action, cell_casimir_action,
    weighted_cell_exchange_action, four_site_spin_modes,
)
from .spin_n16_physical_fmove_carrier_gate import ROOT, _load_source, _fixed_sz_configurations, _metrics


def _krylov(carrier: np.ndarray, h: np.ndarray, rank: int) -> np.ndarray:
    """Twice reorthogonalized Krylov basis for small matched-rank controls."""
    columns = [carrier / np.linalg.norm(carrier)]
    for _ in range(rank - 1):
        v = h @ columns[-1]
        for _ in range(2):
            for q in columns:
                v = v - q * np.vdot(q, v)
        if np.linalg.norm(v) < 1e-12:
            break
        columns.append(v / np.linalg.norm(v))
    return np.column_stack(columns)


def _operator_bank(phi, basis):
    pairs = FixedSzPairActions(basis)
    pair = lambda v, i, j: 0.5 * v[pairs.indices(i, j)] - 0.25 * v
    a, b, c, d = TRIANGULAR_N16_RHOMBUS_SUPPORTS
    contexts = np.asarray(list(product(range(4), (0, 2, 4), (0, 2, 4))), dtype=np.int32)
    matrices = []
    hermiticity = splus = sum_error = 0.0
    commutators = []
    for mode in four_site_spin_modes():
        image = weighted_cell_exchange_action(phi, pair, a, b, mode, mode)
        comm = cell_casimir_action(image, pair, a) - weighted_cell_exchange_action(cell_casimir_action(phi, pair, a), pair, a, b, mode, mode)
        commutators.append(float(np.linalg.norm(comm) / np.linalg.norm(phi)))
        summed = np.zeros_like(image)
        for qc, qd in product((0, 2, 4), repeat=2):
            partial = cell_spin_projector_action(image, pair, c, qc)
            partial = cell_spin_projector_action(partial, pair, d, qd)
            summed += partial
            block = phi.conj().T @ partial
            hermiticity = max(hermiticity, float(np.max(np.abs(block - block.conj().T))))
            if np.linalg.norm(partial[:, 0]) > 1e-12:
                splus = max(splus, spin_raising_relative_norm(partial[:, 0], basis))
            matrices.append(0.5 * (block + block.conj().T))
        sum_error = max(sum_error, float(np.max(np.abs(summed - image))))
    bank = np.asarray(matrices)
    # One common scale preserves sum_{q,r} P_q P_r = I in local-only ablation.
    scale = max(np.linalg.norm(np.sum(bank[9 * m:9 * (m + 1)], axis=0), ord=2) for m in range(4))
    return bank / scale, contexts, {
        "raw_hermiticity_error": hermiticity,
        "unprojected_splus_relative_norm": float(splus),
        "spectator_projector_sum_error": sum_error,
        "local_casimir_commutator_rms_by_mode": commutators,
        "common_operator_scale": float(scale),
    }


def _scramble_fixing_carrier(carrier: np.ndarray, seed: int) -> np.ndarray:
    # Same bank eigenvalues, residual Gram matrix, rank and exact starting state.
    # This is a nonlocal isospectral control, not a local lattice permutation.
    c = np.asarray(carrier).real
    if np.linalg.norm(np.asarray(carrier).imag) > 1e-12:
        raise ValueError("this registered control requires a real carrier")
    _, _, vh = np.linalg.svd(c[None, :], full_matrices=True)
    complement = vh[1:].T
    gaussian = np.random.default_rng(seed).normal(size=(c.size - 1, c.size - 1))
    rotation, triangular = np.linalg.qr(gaussian)
    rotation = rotation @ np.diag(np.sign(np.diag(triangular)))
    return np.outer(c, c) + complement @ rotation @ complement.T


def _physics(state, phi, basis, reference):
    report = su2_resolved_entanglement(phi @ state, basis, TRIANGULAR_N16_RHOMBUS_SUPPORTS[0])
    result = json_ready_entanglement_report(report)
    result["entropy_components"] = singlet_entropy_components(report)
    result["errors_to_exact"] = compare_entanglement_reports(report, reference)
    return result


def run_spin_channel_attention(config_path: Path, output: Path, *, overwrite: bool = False) -> dict:
    spec = json.loads(config_path.read_text())
    if spec.get("task") != "spin_channel_attention" or spec.get("preregistered_before_run") is not True or spec.get("remote_training_on_pass") is not False:
        raise ValueError("invalid local-only registered experiment")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists() and not overwrite:
        raise FileExistsError(output / "results.json")
    _, arrays = _load_source(spec)
    phi = arrays["physical_a1_basis"]
    basis = _fixed_sz_configurations()
    bank, contexts, checks = _operator_bank(phi, basis)
    if max(checks[k] for k in ("raw_hermiticity_error", "unprojected_splus_relative_norm", "spectator_projector_sum_error")) > 1e-10:
        raise RuntimeError("operator symmetry/projector check failed before training")
    coordinates = np.asarray(((0., 0.), (1., 0.), (0., 1.), (1., 1.)))
    cases, states = {}, {"operator_bank": bank, "contexts": contexts}
    for coupling in spec["couplings"]:
        key = str(coupling)
        h0, h125 = arrays["coupling_0_h_a1"], arrays["coupling_0.125_h_a1"]
        h = h0 + (h125 - h0) * (float(coupling) / 0.125)
        pi = arrays["coupling_0.125_carrier_a1"]
        k2, _ = lowest_subspace_state(_krylov(pi, h, 3), h)
        exact_e, exact_v = np.linalg.eigh(h)
        reference = su2_resolved_entanglement(phi @ exact_v[:, 0], basis, TRIANGULAR_N16_RHOMBUS_SUPPORTS[0])
        images = np.einsum("kij,j->ki", bank, k2)
        local_images = images.reshape(4, 9, -1).sum(axis=1)
        oracle, rank = lowest_subspace_state(np.column_stack((k2, images.T)), h)
        local_oracle, local_rank = lowest_subspace_state(np.column_stack((k2, local_images.T)), h)
        kmatched, _ = lowest_subspace_state(_krylov(pi, h, rank), h)
        klocal, _ = lowest_subspace_state(_krylov(pi, h, local_rank), h)
        k4, _ = lowest_subspace_state(_krylov(pi, h, 5), h)
        controls = {}
        for name, state in (("pi_flux", pi), ("two_lanczos", k2), ("degree_four_krylov", k4), ("conditional_linear_oracle", oracle), ("local_linear_oracle", local_oracle), ("rank_matched_krylov", kmatched), ("local_rank_matched_krylov", klocal), ("exact_a1", exact_v[:, 0])):
            controls[name] = dict(_metrics(state, h), physics=_physics(state, phi, basis, reference))
            states[f"j{key}_{name}"] = state
        runs = []
        for seed in spec["seeds"]:
            rotation = _scramble_fixing_carrier(k2, seed + int(spec["scramble_seed_offset"]))
            scrambled_images = images @ rotation.T
            for name, selected, schedule in (
                ("local_only", images, "local_only"),
                ("joint", images, "joint"),
                ("staged", images, "staged"),
                ("scrambled_joint", scrambled_images, "joint"),
                ("scrambled_staged", scrambled_images, "staged"),
            ):
                state, training = train_channel_schedule(
                    k2, selected, h, contexts, coordinates, seed=seed,
                    width=int(spec["width"]), steps=int(spec["steps"]),
                    warmup=int(spec["warmup"]), learning_rate=float(spec["learning_rate"]),
                    schedule=schedule, checkpoint=output / "checkpoints" / f"j{key}_{name}_{seed}.pkl",
                )
                row = {"arm": name, **training, **_metrics(state, h), "physics": _physics(state, phi, basis, reference)}
                runs.append(row)
                states[f"j{key}_{name}_{seed}"] = state
                print(f"J2={key} {name} seed={seed}: E/N={row['energy_per_site']:.12f}, Var={row['variance_total']:.7g}", flush=True)
        comparisons = {}
        for first, second in (("staged", "joint"), ("joint", "scrambled_joint"), ("staged", "scrambled_staged")):
            diffs, vd = [], []
            for seed in spec["seeds"]:
                a = next(r for r in runs if r["arm"] == first and r["seed"] == seed)
                b = next(r for r in runs if r["arm"] == second and r["seed"] == seed)
                diffs.append(a["energy_per_site"] - b["energy_per_site"])
                vd.append(a["variance_total"] - b["variance_total"])
            comparisons[f"{first}_minus_{second}"] = {
                "paired_energy_differences": diffs, "mean_energy_difference": float(np.mean(diffs)),
                "paired_seed_standard_error": float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))),
                "paired_variance_differences": vd,
                "interpretation": "seed variability only; energies are exhaustive, not MCMC estimates",
            }
        cases[key] = {
            "controls": controls, "runs": runs, "comparisons": comparisons,
            "conditional_span_rank": rank, "local_span_rank": local_rank,
            "a1_dimension": int(h.shape[0]), "conditional_span_saturated": rank == h.shape[0],
            "initial_carrier_energy_per_site": float(np.vdot(k2, h @ k2).real / 16),
        }
    np.savez_compressed(output / "states.npz", **states)
    result = {
        "task": "spin_channel_attention", "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec, "model": "triangular 4x4 PBC J1-J2, exact S=0 Gamma-A1",
        "checks": checks, "cases": cases,
        "claim_boundary": "q-conditioned physical-amplitude scalar operator oracle; no a-conditioning, scalable sampler, fixed-Sz arm, GPU comparison or QSL phase claim. The exact A1 block has only 14 states. Conditional saturation must not be called a variational scaling win. Scramble is nonlocal and isospectral, not a local geometry permutation.",
        "remote_training_licensed": False,
        "bundle_sha256": file_sha256(output / "states.npz"),
        "source_manifest": transitive_local_source_manifest((Path(__file__).resolve(),), root=ROOT, package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs")),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(output / "results.json", result)
    return result
