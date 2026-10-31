"""Bounded eight-spin boundary/tree comparison; all starts and arms reported.

Run with ``python -m projects.clebsch_tree_nqs.workflows.ct_boundary_probe``.
This exact development probe does not update production energy notebooks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from su2_nqs.benchmarks.ct_boundary_probe import (
    exchange_commutator, exchange_gate, four_by_four_singlets,
    physical_swap, rank_one_energy_gradient,
    factors_to_isometric,
)
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph


def run_probe(output: Path, *, isometric: bool = True) -> dict:
    """Run a frozen finite benchmark, without sampling or production training."""
    output.mkdir(parents=True, exist_ok=True)
    preregister = {
        "model": "triangular 2x4 OBC J1=1", "n_sites": 8,
        "couplings": [0.0, 0.125], "seeds": [71, 103, 211, 307],
        "cut_order": {"compact": [0, 1, 4, 5], "rows": [0, 1, 2, 3],
                      "zigzag": [0, 1, 4, 6]},
        "retained_multiplicities": {"0": 1, "2": 1, "4": 1},
        "arms": ["compact", "rows", "zigzag", "compact_boundary_gate", "compact_within_gate"],
        "boundary_gate_sites": [1, 2, 6], "within_gate_sites": [0, 1, 5],
        "spatial_groups": ["none", "inversion_even"],
        "spatial_caveat": "Only inversion of this OBC patch; not periodic triangular full A1.",
        "optimizer": "BFGS analytic exact gradient", "maxiter": 300,
        "parameterization": "unit-column sphere coordinates" if isometric else "redundant real factors",
        "initialization": "same normalized physical state as twelve-factor seeded draw",
        "gtol": 1e-10, "initial_gate_angle": 0.0,
        "claim_boundary": "Small real singlet representation/optimization probe; not QSL or large-size accuracy evidence.",
        "family_dimensions": {"fixed_cut_normalized_real": 8, "boundary_gate_upper_bound": 9,
                              "within_gate": 8, "complete_real_singlet": 13},
    }
    registration = output / "registration.json"
    if registration.exists() and json.loads(registration.read_text()) != preregister:
        raise ValueError("existing registration differs; use a new experiment directory")
    registration.write_text(json.dumps(preregister, indent=2) + "\n")
    start = perf_counter()
    inversion = np.zeros((256, 256))
    for i in range(256):
        inversion[int(f"{i:08b}"[::-1], 2), i] = 1
    s2, _ = spin_squared_matrix(8)
    reference_basis = four_by_four_singlets((0, 1, 4, 5))
    exact_checks = {
        "orthonormality_max": float(np.max(abs(reference_basis.T @ reference_basis - np.eye(14)))),
        "singlet_S2_max": float(np.max(abs(s2 @ reference_basis))),
    }
    cases = []
    for j2 in preregister["couplings"]:
        bonds = triangular_j1j2_bonds(2, 4, j2=j2, periodic=False)
        physical_h = sum(c * (.5 * physical_swap(8, a, b) - .25 * np.eye(256))
                         for a, b, c in bonds)
        graph = SzHeisenbergGraph(8, bonds, n_up=4)
        ids = np.array([sum(b << (7-i) for i, b in enumerate(bits)) for bits in graph.basis])
        exact_checks[f"J2={j2}:H_independent_connectivity_max"] = float(np.max(
            abs(physical_h[np.ix_(ids, ids)] - graph.to_dense())))
        h0 = reference_basis.T @ physical_h @ reference_basis
        evals, evecs = np.linalg.eigh(h0)
        ground_vectors = reference_basis @ evecs[:, abs(evals-evals[0]) < 1e-10]
        exact_checks[f"J2={j2}:full_Sz_vs_singlet_energy"] = float(abs(
            np.linalg.eigvalsh(graph.to_dense())[0] - evals[0]))
        exact_checks[f"J2={j2}:inversion_commutator_max"] = float(np.max(
            abs(inversion @ physical_h - physical_h @ inversion)))
        for symmetry in preregister["spatial_groups"]:
            arms = []
            for arm in preregister["arms"]:
                cut = preregister["cut_order"][arm if arm in preregister["cut_order"] else "compact"]
                basis = four_by_four_singlets(tuple(cut))
                h = basis.T @ physical_h @ basis
                generator = None
                if arm == "compact_boundary_gate":
                    generator = exchange_commutator(basis, (1, 2, 6))
                if arm == "compact_within_gate":
                    generator = exchange_commutator(basis, (0, 1, 5))
                projector = (.5 * (np.eye(14) + basis.T @ inversion @ basis)
                             if symmetry == "inversion_even" else None)
                if generator is not None:
                    gate, _ = exchange_gate(generator, .31)
                    exact_checks[f"{arm}:unitary_max"] = float(np.max(abs(gate.T @ gate-np.eye(14))))
                runs = []
                for seed in preregister["seeds"]:
                    x = np.random.default_rng(seed).normal(size=12)
                    if isometric:
                        x = factors_to_isometric(x)
                    if generator is not None:
                        x = np.r_[x, 0.0]

                    def objective(p):
                        e, g, _ = rank_one_energy_gradient(p, h, generator=generator,
                                                          projector=projector, isometric=isometric)
                        return e, g

                    clock = perf_counter()
                    fit = minimize(objective, x, jac=True, method="BFGS",
                                   options={"maxiter": preregister["maxiter"], "gtol": preregister["gtol"]})
                    energy, gradient, state = rank_one_energy_gradient(
                        fit.x, h, generator=generator, projector=projector, isometric=isometric)
                    residual = h @ state - energy * state
                    physical = basis @ state
                    runs.append({"seed": seed, "energy_per_site": energy/8,
                                 "variance_per_site": float(residual @ residual)/8,
                                 "ground_space_fidelity": float(np.linalg.norm(ground_vectors.T @ physical)**2),
                                 "gradient_norm": float(np.linalg.norm(gradient)),
                                 "inversion_defect": float(np.linalg.norm(inversion @ physical-physical)),
                                 "iterations": int(fit.nit), "objective_evaluations": int(fit.nfev),
                                 "success": bool(fit.success), "message": str(fit.message),
                                 "angle": float(fit.x[-1]) if generator is not None else None,
                                 "parameters": fit.x.tolist(),
                                 "normalized_singlet_coefficients": state.tolist(),
                                 "cut_multiplicity_weights": {
                                     "0": (np.linalg.svd(state[:4].reshape(2, 2), compute_uv=False)**2).tolist(),
                                     "2": (np.linalg.svd(state[4:13].reshape(3, 3), compute_uv=False)**2).tolist(),
                                     "4": [float(state[13]**2)]},
                                 "seconds": perf_counter()-clock})
                arms.append({"arm": arm, "left_sites": cut, "runs": runs,
                             "best_energy_per_site": min(r["energy_per_site"] for r in runs),
                             "worst_energy_per_site": max(r["energy_per_site"] for r in runs)})
            cases.append({"j2": j2, "spatial_symmetry": symmetry,
                          "exact_energy_per_site": float(evals[0]/8),
                          "inversion_even_singlet_dimension": int(round(np.trace(
                              .5 * (np.eye(14) + reference_basis.T @ inversion @ reference_basis)))),
                          "bonds": bonds, "arms": arms})
    result = {"registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
              "source_sha256": {
                  str(path.relative_to(Path(__file__).resolve().parents[3])):
                  hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__).resolve(),
                               Path(__file__).resolve().parents[3] / "su2_nqs/benchmarks/ct_boundary_probe.py")},
              "exact_checks": exact_checks, "cases": cases, "seconds": perf_counter()-start,
              "limitations": ["Tiny open patch does not host or identify a bulk QSL.",
                              "Boundary gate adds one physical parameter; within-block control is redundant.",
                              "Tree cuts have equal unprojected retained widths; inversion projection can increase Schmidt rank.",
                              "Dense local benchmark costs do not estimate large-system H-ratio cost.",
                              "Every seed is reported; best endpoint alone is not an optimizer robustness claim."]}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    directory = (Path(__file__).resolve().parents[1] / "results" / "ct_boundary_probe_20261021"
                 / "matched_isometric_coordinates")
    data = run_probe(directory)
    print(json.dumps({"seconds": data["seconds"], "exact_checks": data["exact_checks"],
                     "cases": [{"j2": c["j2"], "symmetry": c["spatial_symmetry"],
                                "exact": c["exact_energy_per_site"],
                                "arms": {a["arm"]: [a["best_energy_per_site"], a["worst_energy_per_site"]]
                                         for a in c["arms"]}} for c in data["cases"]]}, indent=2))
