"""Read-only spectral-readiness diagnostics from existing physical-source panels.

No new wavefunction, H query, sample, gap, or independently optimized branch is
created. All energies/widths below are in TOTAL-energy units. Source centroids
relative to an approximate parent must not be relabeled as excitation gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json


def source_diagnostics(psi: np.ndarray, hpsi: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    """Return [Re E, Im E, sigma_H, norm/parent_norm, E-parent_E] per channel.

    Physical products and proposal probabilities are used directly, so target
    nodes with nonzero Hpsi remain in the energy-width calculation. Channel
    zero is the parent. The last row is the INCOHERENT direct sum of channels
    2 onward (the three distinct M momenta), not a superposition of them.
    """
    p, h, q = np.asarray(psi), np.asarray(hpsi), np.asarray(proposal)
    if (p.ndim != 2 or p.shape != h.shape or p.shape[1] < 3 or q.shape != (len(p),)
            or not np.isfinite(p).all() or not np.isfinite(h).all()
            or not np.isfinite(q).all() or np.iscomplexobj(q) or np.any(q <= 0)):
        raise ValueError("finite parent/source products and positive exact Q required")
    weights = 1/q[:, None]
    norm = np.mean(abs(p)**2*weights, axis=0)
    if np.any(norm <= 0): raise ValueError("every observed source norm must be positive")
    energy = np.mean(p.conj()*h*weights, axis=0)/norm
    residual = np.mean(abs(h-energy[None]*p)**2*weights, axis=0)/norm
    values = np.stack((energy.real, energy.imag, np.sqrt(residual),
                       norm/norm[0], energy.real-energy[0].real), axis=1)
    star_norm = norm[2:].sum()
    star_energy = np.sum(norm[2:]*energy[2:])/star_norm
    star_variance = np.mean(np.sum(abs(h[:, 2:]-star_energy*p[:, 2:])**2/q[:, None], axis=1))/star_norm
    star = np.asarray([star_energy.real, star_energy.imag, np.sqrt(star_variance),
                       star_norm/norm[0], star_energy.real-energy[0].real])
    return np.vstack((values, star))


def summarize(directory: Path) -> dict:
    """Reprocess the original 128-row audit, preserving its original parent."""
    record = json.loads((directory/"results.json").read_text())
    panel = directory/"audit_source_products.npz"
    with np.load(panel) as data:
        p, h, q = data["psi"], data["hpsi"], data["proposal"]
    if len(p) != 128: raise ValueError("this diagnostic expects the registered 128-row panel")
    labels = ["parent", *record["source_channels"], "M_star_direct_sum"]
    point = source_diagnostics(p, h, q)
    groups = 16; ids = np.repeat(np.arange(groups), len(p)//groups)
    leave = np.asarray([source_diagnostics(p[ids != g], h[ids != g], q[ids != g]) for g in range(groups)])
    errors = np.sqrt((groups-1)/groups*np.sum((leave-leave.mean(0))**2, axis=0))
    # Independent reconstruction of the existing endpoint energies and widths.
    discrepancy = 0.
    for i, name in enumerate(record["source_channels"], 1):
        old = record["source_seed_moments"][name]["endpoint_values"][1]
        discrepancy = max(discrepancy, abs(point[i, 0]/36-old[0]),
                          abs(point[i, 2]**2/36-old[2]))
    if discrepancy > 1e-12: raise ArithmeticError("saved physical-source moments failed to replay")
    return dict(j2=record["j2"], original_parent_global_step=record["parent_global_step"],
        original_parent_parameter_sha256=record["parent_parameter_sha256"],
        source_panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
        reused_existing_audit_as_read_only_analysis_not_new_precision=True,
        sample_records=len(p), independent_groups=groups,
        metrics=["total_energy", "imaginary_total_energy", "total_energy_width",
                 "source_norm_over_parent_norm", "source_centroid_relative_to_parent"],
        values={name: point[i].tolist() for i, name in enumerate(labels)},
        jackknife_SE={name: errors[i].tolist() for i, name in enumerate(labels)},
        maximum_replay_discrepancy=discrepancy,
        interpretation=dict(source_total_q=2, source_total_m2=0,
            strict_spin_requires_actual_singlet_parent=True,
            parent_not_assumed_H_eigenstate=True, not_lowest_branch_or_gap=True,
            M_star_is_incoherent_sum_not_selected_lowest_M=True,
            widths_are_spectral_spread_not_energy_error_bars=True,
            physical_Sk_is_three_times_norm_ratio_for_this_singlet_source=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    base = root/"projects/clebsch_tree_nqs/results/ct_sector_physical_seed_20261029"
    report = dict(task="strict_triplet_source_readiness_not_new_spectroscopy",
        no_sampling_or_H_queries=True,
        runs={tag: summarize(base/tag) for tag in ("j0_v2", "j0125_v2")})
    if args.output.exists(): raise FileExistsError("do not overwrite a frozen analysis")
    atomic_json(args.output, report)
    for tag, run in report["runs"].items():
        for label in ("K", "M_star_direct_sum"):
            x, s = run["values"][label], run["jackknife_SE"][label]
            print(tag, label, "centroid", x[4], "+/-", s[4], "width", x[2], "+/-", s[2])


if __name__ == "__main__": main()
