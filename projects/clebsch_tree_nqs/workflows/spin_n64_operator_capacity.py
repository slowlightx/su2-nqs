"""CPU-local one-layer Operator-ViT capacity/runtime bridge at N=64."""

from __future__ import annotations

from datetime import datetime, timezone
from math import comb
import json
from pathlib import Path
import time

import numpy as np

from su2_nqs.benchmarks.operator_tangent_response import (
    scalar_operator_tangent_statistics,
    triangular_a1_local_ratios,
)
from su2_nqs.benchmarks.triangular_induced_parton_n64 import (
    A1_COEFFICIENTS,
    BOUNDARY_ORBIT,
    build_n64_pi_zero_flux_orbit,
    projected_n64_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.frontier.artifacts import array_sha256, atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.operators.triangular_a1_scalar import TRIANGULAR_A1_SCALAR_LABELS
from su2_nqs.vmc.fixed_wavefunction_mcmc import (
    chain_mean_standard_error,
    sample_fixed_wavefunction_exchange,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n64_operator_capacity"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N64 operator-capacity identity")
    sampler = spec.get("sampler")
    if not isinstance(sampler, dict):
        raise TypeError("sampler must be an object")
    for key in (
        "replicates",
        "chain_count",
        "burn_sweeps",
        "measurement_blocks",
        "decorrelation_sweeps",
        "center_batch_size",
        "amplitude_batch_size",
    ):
        if int(sampler.get(key, 0)) < 1:
            raise ValueError(f"sampler.{key} must be positive")


def _orthogonal_fractions(metric: np.ndarray, coupling: float) -> dict[str, float]:
    direction = np.asarray((1.0, coupling, 0.0, 0.0, 0.0))
    h_norm = float(direction @ metric @ direction)
    result = {}
    for index in range(2, 5):
        overlap = float(metric[index] @ direction)
        result[TRIANGULAR_A1_SCALAR_LABELS[index]] = 1.0 - overlap**2 / (
            float(metric[index, index]) * h_norm
        )
    return result


def run_spin_n64_operator_capacity(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    sampler = spec["sampler"]
    couplings = tuple(float(value) for value in spec["couplings"])
    orbit = build_n64_pi_zero_flux_orbit()

    def amplitudes(states: np.ndarray) -> np.ndarray:
        return projected_n64_pi_zero_flux_a1_amplitudes(
            states,
            orbit=orbit,
            microbatch_size=int(sampler["amplitude_batch_size"]),
        ).amplitudes

    rows = []
    for replicate in range(int(sampler["replicates"])):
        started = time.perf_counter()
        seed = int(spec["seed"]) + 1000 * replicate
        samples = sample_fixed_wavefunction_exchange(
            amplitudes,
            n_sites=64,
            n_up=32,
            chain_count=int(sampler["chain_count"]),
            burn_sweeps=int(sampler["burn_sweeps"]),
            measurement_blocks=int(sampler["measurement_blocks"]),
            decorrelation_sweeps=int(sampler["decorrelation_sweeps"]),
            seed=seed,
        )
        ratio_started = time.perf_counter()
        ratios, ledger = triangular_a1_local_ratios(
            amplitudes,
            samples.configurations,
            samples.amplitudes,
            ((8, 0), (0, 8)),
            center_batch_size=int(sampler["center_batch_size"]),
            amplitude_batch_size=int(sampler["amplitude_batch_size"]),
        )
        ratio_seconds = time.perf_counter() - ratio_started
        response = scalar_operator_tangent_statistics(ratios, couplings)
        energy_rows = []
        for coupling in couplings:
            local_energy = ratios[:, 0] + coupling * ratios[:, 1]
            coupling_row = next(
                row
                for row in response["couplings"]
                if np.isclose(float(row["j2_over_j1"]), coupling)
            )
            energy_rows.append(
                {
                    "j2_over_j1": coupling,
                    "bare_carrier_energy_per_site": float(
                        coupling_row["mean_energy_total"]
                    )
                    / 64.0,
                    "chain_mean_mcse_per_site": chain_mean_standard_error(
                        local_energy,
                        chain_count=samples.chain_count,
                        measurement_blocks=samples.measurement_blocks,
                    )
                    / 64.0,
                    "variance_per_site": float(coupling_row["variance_total"])
                    / 64.0,
                    "resonance_tangent_independence": _orthogonal_fractions(
                        np.asarray(response["fubini_study_metric"]), coupling
                    ),
                }
            )
        row = {
            "replicate_index": replicate,
            "seed": seed,
            "sample_count": len(samples.configurations),
            "sample_sha256": array_sha256(samples.configurations),
            "acceptance_fraction": samples.acceptance_fraction,
            "burn_acceptance_fraction": samples.burn_acceptance_fraction,
            "energy_and_capacity": energy_rows,
            "operator_response": response,
            "evaluation_ledger": ledger,
            "one_layer_ratio_seconds": ratio_seconds,
            "one_layer_connected_amplitudes_per_second": (
                float(ledger["unique_connected_amplitude_evaluations"])
                / ratio_seconds
            ),
            "wall_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        atomic_json(output / f"replicate_{replicate}.json", row)
        print(
            f"replicate={replicate} samples={len(samples.configurations)} "
            f"one-layer={ratio_seconds:.2f}s unique="
            f"{ledger['unique_connected_amplitude_evaluations']}",
            flush=True,
        )

    aggregate = []
    for coupling in couplings:
        energy_rows = [
            next(
                item
                for item in row["energy_and_capacity"]
                if np.isclose(float(item["j2_over_j1"]), coupling)
            )
            for row in rows
        ]
        channels = {}
        for label in TRIANGULAR_A1_SCALAR_LABELS[2:]:
            values = np.asarray(
                [item["resonance_tangent_independence"][label] for item in energy_rows]
            )
            channels[label] = {
                "mean_fraction_orthogonal_to_hamiltonian_tangent": float(
                    np.mean(values)
                ),
                "replicate_standard_error": float(
                    np.std(values, ddof=1) / np.sqrt(len(values))
                ),
            }
        energies = np.asarray(
            [item["bare_carrier_energy_per_site"] for item in energy_rows]
        )
        aggregate.append(
            {
                "j2_over_j1": coupling,
                "mean_bare_carrier_energy_per_site": float(np.mean(energies)),
                "replicate_standard_error_energy_per_site": float(
                    np.std(energies, ddof=1) / np.sqrt(len(energies))
                ),
                "resonance_tangent_independence": channels,
            }
        )

    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "model_case": {
            "hamiltonian": "spin-1/2 triangular J1-J2 Heisenberg",
            "boundary": "8x8 periodic torus",
            "n_sites": 64,
            "fixed_sz_zero_dimension": comb(64, 32),
            "contains_120_degree_K_momentum": False,
            "role": "ED-intractable capacity/runtime bridge, not phase evidence",
            "target_sector": "exact S=0, Gamma-A1",
        },
        "carrier": {
            "boundary_orbit": [list(value) for value in BOUNDARY_ORBIT],
            "a1_coefficients": A1_COEFFICIENTS.tolist(),
            "one_body_gaps": orbit.one_body_gaps.tolist(),
            "excluded_pp_gap": orbit.excluded_pp_gap,
            "compressed_terms": 3,
            "full_reynolds_terms": 768,
        },
        "replicates": rows,
        "aggregate": aggregate,
        "runtime": {
            "mean_one_layer_ratio_seconds": float(
                np.mean([row["one_layer_ratio_seconds"] for row in rows])
            ),
            "mean_connected_amplitudes_per_second": float(
                np.mean(
                    [row["one_layer_connected_amplitudes_per_second"] for row in rows]
                )
            ),
            "mean_unique_connected_evaluations_per_sample": float(
                np.mean(
                    [
                        row["evaluation_ledger"][
                            "unique_connected_amplitude_evaluations"
                        ]
                        / row["sample_count"]
                        for row in rows
                    ]
                )
            ),
        },
        "interpretation_boundary": (
            "one-layer physical capacity and CPU runtime only; no trained N64 "
            "energy and no phase inference because 8x8 lacks K"
        ),
        "hamiltonian_diagonalization_performed": False,
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_n64_operator_capacity"]
