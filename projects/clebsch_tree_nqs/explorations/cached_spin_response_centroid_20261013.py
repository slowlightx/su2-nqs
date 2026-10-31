"""Actual longitudinal first-response moments from sealed N36 data only.

This uses physical-space A1 to equate spatial NN/J2 bond orbits, never SU2
to replace transverse spins by longitudinal spins.  It prints JSON only.
The parent is approximate and not strictly singlet: no true gap or strict
triplet label follows.  All uncertainties resample whole chains together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


def small_identity_check() -> float:
    """Check general complex k/parent identities in the full N4 Hilbert space."""
    n = 4
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    coordinates = np.array([divmod(site, 2) for site in range(n)])
    bonds = triangular_j1j2_bonds(2, 2, j2=0.125, periodic=True)
    h = sum((j * spin_dot_matrix(n, i, k) for i, k, j in bonds), start=np.zeros((16, 16)))
    maximum = 0.0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        psi = rng.normal(size=16) + 1j * rng.normal(size=16)
        psi /= np.linalg.norm(psi)
        energy = np.vdot(psi, h @ psi).real
        for momentum in ((0.5, 0.0), (1/3, 1/3), (0.17, 0.29)):
            phases = np.exp(2j * np.pi * coordinates @ momentum)
            diagonal = (bits - 0.5) @ phases / np.sqrt(n)
            op = np.diag(diagonal)
            a = np.abs(diagonal)**2
            g = np.vdot(psi, a * psi).real
            covariance = np.vdot(a * psi, h @ psi - energy * psi).real
            f = 0.0
            for first, second, coupling in bonds:
                zz = np.diag((bits[:, first] - 0.5) * (bits[:, second] - 0.5))
                transverse = spin_dot_matrix(n, first, second) - zz
                f -= coupling * abs(phases[first] - phases[second])**2 / (2*n) * np.vdot(psi, transverse @ psi).real
            numerator = 0.5 * (np.vdot(op @ psi, h @ op @ psi).real + np.vdot(op.conj() @ psi, h @ op.conj() @ psi).real)
            maximum = max(maximum, abs(numerator - energy * g - f - covariance))
    assert maximum < 1e-12
    return maximum


def response_estimator(means: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Return first moments without dropping approximate-parent covariance."""
    norm, second_a, energy, imaginary_energy, ae, imaginary_ae, t1, t2 = means
    f = -float(factors @ np.array((t1, t2)))
    covariance = ae - norm * energy
    return np.array((norm, second_a - norm**2, f, f / norm, covariance / norm,
                     (f + covariance) / norm, energy + (f + covariance) / norm,
                     imaginary_energy, (imaginary_ae - norm * imaginary_energy) / norm))


def analyze_cached_parent(
    regional: Path, stream_directory: Path, parent_registration: Path,
    *, expected_step: int, j2: float, entry: dict | None = None,
) -> dict:
    """Replay one sealed parent; never evaluate a new amplitude or H action.

    ``j2=0`` has no transverse J2 estimator: that slot and its geometric
    factor are exactly zero, rather than an empty-shell mean or a ratio.
    """
    start = time.monotonic()
    registered = json.loads(parent_registration.read_text())
    report = json.loads((regional / "results.json").read_text())
    registration = json.loads((regional / "registration.json").read_text())
    configuration = registration["config"]
    if entry is not None:
        for filename, field in (("results.json", "results_sha256"),
                                ("registration.json", "registration_sha256"),
                                ("physics.npz", "arrays_sha256")):
            assert hashlib.sha256((regional / filename).read_bytes()).hexdigest() == entry[field]
    assert report["full_A1"] is True and report["strict_SU2"] is False
    assert report["step"] == configuration["step"] == expected_step
    assert report["params_sha256"] == configuration["params_sha256"]
    assert report["checkpoint_sha256"] == configuration["checkpoint_sha256"]
    assert configuration["problem"]["geometry"] == "triangular"
    assert configuration["problem"]["rows"] == configuration["problem"]["columns"] == 6
    assert configuration["problem"]["j2"] == j2
    assert configuration["problem"]["boundary"] == "PBC_both_physical"
    assert registered["config"]["j2_over_j1"] == j2
    assert registered["config"]["coordinate_convention"] == "physical120"
    assert registered["config"]["model"]["geometry"] == "triangular"
    assert registered["config"]["model"]["rows"] == registered["config"]["model"]["columns"] == 6
    assert registered["result_sha256"] == configuration["parent_registration_sha256"]
    assert hashlib.sha256(parent_registration.read_bytes()).hexdigest() in registration["inputs_sha256"].values()
    streams = []
    manifests = []
    for seed_string, frozen_digest in configuration["stream_sha256"].items():
        seed = int(seed_string)
        path = stream_directory / f"stream{seed}.npz"
        metadata = json.loads(path.with_suffix(".json").read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == metadata["raw_sha256"] == frozen_digest
        assert metadata["params_sha256"] == report["params_sha256"]
        assert metadata["registration_sha256"] == configuration["parent_registration_sha256"]
        assert metadata["records"] == metadata["full_H_calls"] == 8192
        with np.load(path) as data:
            bits, energies = np.array(data["states"]), np.array(data["energies"])
            assert bits.shape == (128, 64, 36) and energies.shape == (128, 64)
            assert energies.dtype == np.complex128 and np.isfinite(energies).all()
            assert np.all((bits == 0) | (bits == 1)) and np.all(bits.sum(axis=-1) == 18)
            streams.append((bits, energies))
        manifests.append({"path": str(path), "sha256": digest})
    assert len(streams) == 2
    raw = regional / "physics.npz"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == report["raw_sha256"]
    with np.load(raw) as data:
        bits_regional = np.array(data["states"])
        chain_ids = np.array(data["compact6__chain_ids"])
        sites = np.array(data["compact6__region_sites"])
        matrices = np.array(data["compact6__per_record_matrices"])
        stored_rho = np.array(data["compact6__rho"])
    indices = configuration["snapshot_indices"]
    expected_bits = np.concatenate([bits[indices].reshape(-1, 36) for bits, _ in streams])
    expected_chain_ids = np.concatenate([np.tile(np.arange(64) + offset*64, len(indices)) for offset in range(2)])
    assert np.array_equal(bits_regional, expected_bits)
    assert np.array_equal(chain_ids, expected_chain_ids)
    assert np.max(np.abs(matrices.mean(axis=0) - stored_rho)) < 1e-12

    bonds = triangular_j1j2_bonds(6, 6, j2=j2, periodic=True)
    assert report["bonds"] == [list(bond) for bond in bonds]
    weights = {(i, j): coupling for i, j, coupling in bonds}
    local_bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    operators = {1.0: [], 0.125: []}
    local_pairs = {1.0: [], 0.125: []}
    for first in range(6):
        for second in range(first + 1, 6):
            edge = tuple(sorted((int(sites[first]), int(sites[second]))))
            if edge not in weights:
                continue
            coupling = weights[edge]
            zz = np.diag((local_bits[:, first] - 0.5) * (local_bits[:, second] - 0.5))
            operators[coupling].append(spin_dot_matrix(6, first, second) - zz)
            local_pairs[coupling].append(list(edge))
    assert len(operators[1.0]) == 9
    assert len(operators[0.125]) == (3 if j2 else 0)
    transverse = np.column_stack([
        np.einsum("rij,ji->r", matrices, np.mean(operators[coupling], axis=0)).real
        if operators[coupling] else np.zeros(len(matrices))
        for coupling in (1.0, 0.125)
    ])
    chain_transverse = np.array([transverse[chain_ids == chain].mean(axis=0) for chain in range(128)])
    coordinates = np.array([divmod(site, 6) for site in range(36)])
    momenta = {"K": (1/3, 1/3), "M1": (0.5, 0.0), "M2": (0.0, 0.5), "M3": (0.5, 0.5)}
    estimates = {}
    combined_bank = {}
    factor_bank = {}
    replica_bank = {}
    sample_a_bank = {}
    for label, momentum in momenta.items():
        phases = np.exp(2j * np.pi * coordinates @ momentum)
        factors = np.array([sum(coupling * abs(phases[i] - phases[j])**2 / 72 for i, j, coupling in bonds if coupling == shell) for shell in (1.0, 0.125)])
        statistics = []
        raw_a = []
        for bits, energies in streams:
            a = np.abs((bits - 0.5) @ phases / 6)**2
            raw_a.append(a.reshape(-1))
            statistics.append(np.column_stack((a.mean(axis=0), (a**2).mean(axis=0), energies.real.mean(axis=0), energies.imag.mean(axis=0), (a * energies.real).mean(axis=0), (a * energies.imag).mean(axis=0))))
        chain_statistics = np.concatenate(statistics, axis=0)
        combined = np.column_stack((chain_statistics, chain_transverse))
        combined_bank[label] = combined
        sample_a_bank[label] = np.concatenate(raw_a)
        factor_bank[label] = factors
        means = combined.mean(axis=0)
        values = response_estimator(means, factors)
        leave_out = (combined.sum(axis=0) - combined) / 127
        replicas = np.array([response_estimator(row, factors) for row in leave_out])
        replica_bank[label] = replicas
        errors = np.sqrt(127/128 * np.sum((replicas - replicas.mean(axis=0))**2, axis=0))
        names = ("norm_squared", "diagonal_A_variance", "longitudinal_double_commutator", "static_f_over_norm", "parent_covariance_correction_over_norm", "response_centroid_minus_parent", "response_centroid_total", "parent_energy_imaginary", "parent_covariance_imaginary_over_norm")
        estimates[label] = {
            "momentum_fractional": list(momentum), "shell_geometric_factors": factors.tolist(),
            "quantities": {name: {"value": float(value), "paired_whole_chain_JK_SE": float(error)} for name, value, error in zip(names, values, errors)},
            "denominator_exceeds_10_chain_SE": bool(values[0] > 10 * errors[0]),
            "scope": "symmetrized +/-K first moment; exact physical-space inversion makes branches equal" if label == "K" else "Hermitian M source first moment",
        }
    m_combined = np.mean([combined_bank[name] for name in ("M1", "M2", "M3")], axis=0)
    m_values = response_estimator(m_combined.mean(axis=0), factor_bank["M1"])
    m_replicas = np.array([response_estimator(row, factor_bank["M1"]) for row in (m_combined.sum(axis=0) - m_combined) / 127])
    m_errors = np.sqrt(127/128 * np.sum((m_replicas - m_replicas.mean(axis=0))**2, axis=0))
    estimates["M_star"] = {"scope": "norm-weighted direct-sum average of all three M responses; not three independent modes", "quantities": {name: {"value": float(value), "paired_whole_chain_JK_SE": float(error)} for name, value, error in zip(names, m_values, m_errors)}}
    difference_replicas = m_replicas[:, 5] - replica_bank["K"][:, 5]
    contrast = {
        "definition": "M_star centroid minus K centroid, both relative to the same approximate parent",
        "value": float(m_values[5] - estimates["K"]["quantities"]["response_centroid_minus_parent"]["value"]),
        "paired_whole_chain_JK_SE": float(np.sqrt(127/128 * np.sum((difference_replicas - difference_replicas.mean())**2))),
        "QSL_specific": False, "lowest_eigenvalue_contrast": False,
    }
    # Retain the observed covariance tail sensitivity without trimming,
    # winsorizing, or pretending this measures unobserved-tail bias.
    raw_energy = np.concatenate([energies.reshape(-1) for _, energies in streams])
    centered_energy = raw_energy - raw_energy.mean()
    tail_indices = np.argsort(abs(centered_energy)**2)[-int(np.ceil(.01*len(raw_energy))):]
    tail = {"definition": "top ceil(1% records) ranked by |E_loc-mean(E_loc)|^2; descriptive, no trimming",
            "record_count": len(tail_indices),
            "sampled_energy_variance_share": float(np.sum(abs(centered_energy[tail_indices])**2)/np.sum(abs(centered_energy)**2))}
    tail["covariance_summands"] = {}
    sample_a_bank["M_star"] = np.mean([sample_a_bank[name] for name in ("M1", "M2", "M3")], axis=0)
    for name in ("K", "M_star"):
        a = sample_a_bank[name]
        summand = (a-a.mean())*centered_energy.real
        tail["covariance_summands"][name] = dict(
            squared_summand_tail_share=float(np.sum(summand[tail_indices]**2)/np.sum(summand**2)),
            signed_tail_contribution_to_covariance=float(summand[tail_indices].sum()/len(summand)),
            total_covariance=float(summand.mean()))
    return {
        "N": 36, "j2": j2, "parent_step": expected_step,
        "params_sha256": report["params_sha256"], "checkpoint_sha256": report["checkpoint_sha256"],
        "source_streams": manifests,
        "regional_coherent_npz": {"path": str(raw), "sha256": report["raw_sha256"]},
        "local_shell_bonds": {str(coupling): pairs for coupling, pairs in local_pairs.items()},
        "local_transverse_orbit_means": chain_transverse.mean(axis=0).tolist(),
        "paired_whole_chains": 128, "Born_records": 16384, "conditional_RDM_records": 512,
        "same_params_and_chain_pairing_verified": True,
        "physical_full_A1_used_for_spatial_bond_orbits_only": True,
        "strict_SU2": False, "strict_triplet_label": False,
        "exact_ground_state_gap_upper_bound": False, "response_residual_available": False,
        "H_variance_interval_certificate_available": False, "mixing_certified": False,
        "new_amplitude_queries": 0, "new_H_calls": 0, "new_sampling": 0,
        "N4_complex_parent_and_momentum_identity_max_error": small_identity_check(),
        "estimates": estimates,
        "physical_first_moment_contrast": contrast,
        "observed_tail_sensitivity": tail,
        "limitations": ["first-response centroids, not spectral thresholds or lowest eigenvalues", "unknown E_parent-E0", "whole-chain SE does not certify unseen tails or equilibrium bias", "nonlinear ratio plug-in finite-chain bias is not removed", "three M results are symmetry-equivalent true-state quantities, not independent channels", "S_z is diagonal: its response vanishes at every parent node, unlike a generic dimer response"],
        "elapsed_seconds": time.monotonic() - start,
    }


def quality_series() -> dict:
    """The same first-moment mechanism at six fixed saved endpoints only."""
    base = Path("projects/clebsch_tree_nqs")
    config_path = base / "configs/validation/triangle_symmetry_contrast_20261013.json"
    config = json.loads(config_path.read_text())
    synced = base / "results/triangle_strip_fcs_20261013_inputs"
    continuation = synced / "parent_registrations"
    locations = {
        500: (synced / "tri_j0_step500", continuation / "nqs-wavefunctions/nqs/tri_j0_repair_v6_single_gpu/seed1/registration.json"),
        750: (synced / "tri_j0_step750", continuation / "nqs-wavefunctions/nqs/tri_j0_repair_v6_continue_1000/seed1/registration.json"),
        1000: (synced / "tri_j0_step1000", continuation / "nqs-wavefunctions/nqs/tri_j0_repair_v6_continue_1000/seed1/registration.json"),
        8600: (synced / "tri_j0125_step8600", continuation / "spin_spatial_vit_gpu_continuation_8600_v1/seed1/registration.json"),
        9600: (base / "results/spin_resonance_response_quality_20261011/inputs/tri_9600", continuation / "spin_spatial_vit_gpu_continuation_9600_v1/seed1/registration.json"),
        10600: (base / "results/spin_spatial_vit_gpu_continuation_10600_v1/seed1/audits/step10600", base / "results/spin_spatial_vit_gpu_continuation_10600_v1/seed1/registration.json"),
    }
    rows = []
    for entry in config["inputs"]:
        streams, parent = locations[entry["step"]]
        result = analyze_cached_parent(Path(entry["directory"]), streams, parent,
                                      expected_step=entry["step"], j2=entry["j2"], entry=entry)
        # Report only the two frozen physical channels. Single-M values are
        # internal orbit checks, not candidates for post-selected minima.
        result["estimates"] = {name: result["estimates"][name] for name in ("K", "M_star")}
        result.pop("physical_first_moment_contrast")
        rows.append(result)
    multiplier = 2.241402727604947  # two-sided Bonferroni normal z, alpha=.05, family size2
    comparisons = {}
    for name in ("K", "M_star"):
        series = [[row["estimates"][name]["quantities"]["response_centroid_minus_parent"]
                   for row in rows if row["j2"] == coupling] for coupling in (0, .125)]
        values = [[point["value"] for point in group] for group in series]
        error = np.hypot(series[0][-1]["paired_whole_chain_JK_SE"], series[1][-1]["paired_whole_chain_JK_SE"])
        difference = values[1][-1]-values[0][-1]
        drift = [float(np.ptp(group)) for group in values]
        threshold = sum(drift)+multiplier*error
        comparisons[name] = dict(latest_cross_J_difference=float(difference),
            independent_whole_chain_SE=float(error), own_late_endpoint_ranges=drift,
            descriptive_threshold=float(threshold), exceeds_descriptive_threshold=bool(abs(difference)>threshold))
    return dict(scope="same longitudinal response first moments at fixed saved quality endpoints; no phase or gap assignment",
        source_config=str(config_path), source_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_channels=["K", "M_star"], rows=rows, latest_cross_J_contrasts=comparisons,
        gate=dict(normal_multiplier=multiplier, family_size=2, alpha=.05,
                  retrospective_descriptive=True, quality_ranges_are_bias_bounds=False,
                  normal_SE_is_interval_certificate=False, cross_J_is_same_H_competition=False),
        new_amplitude_queries=0, new_H_calls=0, new_sampling=0,
        strict_SU2=False, strict_triplet_label=False, gap_claim=False, QSL_specific=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-series", action="store_true")
    args = parser.parse_args()
    if args.quality_series:
        result = quality_series()
    else:
        root = Path("projects/clebsch_tree_nqs/results/spin_spatial_vit_gpu_continuation_10600_v1/seed1")
        result = analyze_cached_parent(
            Path("projects/clebsch_tree_nqs/results/spin_resonance_response_quality_20261011/inputs/tri_10600/regional"),
            root / "audits/step10600", root / "registration.json", expected_step=10600, j2=.125)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
