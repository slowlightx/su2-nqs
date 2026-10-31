"""Independent N=36 energy benchmark for the projected pi/0-flux state."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    induced_representation_audit,
    projected_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.frontier.artifacts import array_sha256, atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.fixed_wavefunction_mcmc import (
    chain_mean_standard_error,
    fixed_wavefunction_local_energies,
    sample_fixed_wavefunction_exchange,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_projected_pi_flux_benchmark"
EXACT_ENERGY_PER_SITE = -0.515564
GCNN_ENERGY_PER_SITE = -0.515386
GUTZWILLER_LS_ENERGY_PER_SITE = -0.512503
DSL_ZERO_LANCZOS_ENERGY_PER_SITE = -0.501788
DSL_ONE_LANCZOS_ENERGY_PER_SITE = -0.510344


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid projected-pi-flux benchmark identity")
    sampler = spec.get("sampler")
    if not isinstance(sampler, dict):
        raise TypeError("sampler must be an object")
    for name in (
        "replicates",
        "chain_count",
        "burn_sweeps",
        "measurement_blocks",
        "decorrelation_sweeps",
        "local_energy_batch_size",
    ):
        if int(sampler.get(name, 0)) < 1:
            raise ValueError(f"sampler.{name} must be positive")
    if int(sampler["replicates"]) < 2:
        raise ValueError("multiple independent replicates are required")


def _replicate_standard_error(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def run_spin_n36_pi_flux_benchmark(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    sampler = spec["sampler"]
    orbit = build_n36_pi_zero_flux_orbit()
    audit = induced_representation_audit()
    bonds = triangular_j1j2_bonds(6, 6, j1=1.0, j2=0.125, periodic=True)

    def amplitudes(states: np.ndarray) -> np.ndarray:
        return projected_pi_zero_flux_a1_amplitudes(
            states,
            orbit=orbit,
            microbatch_size=int(spec.get("amplitude_microbatch_size", 64)),
        ).amplitudes

    rows = []
    for replicate in range(int(sampler["replicates"])):
        started = time.perf_counter()
        seed = int(spec["seed"]) + 1000 * replicate
        samples = sample_fixed_wavefunction_exchange(
            amplitudes,
            n_sites=36,
            n_up=18,
            chain_count=int(sampler["chain_count"]),
            burn_sweeps=int(sampler["burn_sweeps"]),
            measurement_blocks=int(sampler["measurement_blocks"]),
            decorrelation_sweeps=int(sampler["decorrelation_sweeps"]),
            seed=seed,
        )
        local = fixed_wavefunction_local_energies(
            amplitudes,
            samples.configurations,
            bonds,
            batch_size=int(sampler["local_energy_batch_size"]),
        )
        energy_total = complex(np.mean(local))
        variance_total = float(np.mean(np.abs(local - energy_total) ** 2).real)
        energy_per_site = float(energy_total.real / 36.0)
        row = {
            "replicate_index": replicate,
            "seed": seed,
            "sample_count": len(local),
            "sample_sha256": array_sha256(samples.configurations),
            "energy_total": float(energy_total.real),
            "energy_per_site": energy_per_site,
            "chain_mean_mcse_per_site": chain_mean_standard_error(
                local,
                chain_count=samples.chain_count,
                measurement_blocks=samples.measurement_blocks,
            )
            / 36.0,
            "variance_total": variance_total,
            "variance_per_site": variance_total / 36.0,
            "maximum_abs_imaginary_local_energy": float(
                np.max(np.abs(local.imag))
            ),
            "acceptance_fraction": samples.acceptance_fraction,
            "burn_acceptance_fraction": samples.burn_acceptance_fraction,
            "gap_above_exact_per_site": energy_per_site - EXACT_ENERGY_PER_SITE,
            "wall_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        atomic_json(output / f"replicate_{replicate}.json", row)
        print(
            f"replicate={replicate} E/N={energy_per_site:.8f} "
            f"mcse={row['chain_mean_mcse_per_site']:.2e} "
            f"accept={samples.acceptance_fraction:.3f}",
            flush=True,
        )

    energies = np.asarray([row["energy_per_site"] for row in rows])
    mean = float(np.mean(energies))
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
            "boundary": "6x6 periodic torus",
            "j1": 1.0,
            "j2_over_j1": 0.125,
            "target_sector": "exact S=0, M=0, Gamma-A1",
            "j1_bond_count": 108,
            "j2_bond_count": 108,
        },
        "ansatz": {
            "kind": "Gutzwiller-projected spin-independent pi/0-flux Slater state",
            "trainable_parameter_count": 0,
            "boundary_orbit": ["PP", "AP", "PA"],
            "a1_coefficients": [1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
            "one_body_gaps": orbit.one_body_gaps.tolist(),
            "excluded_aa_gap": orbit.excluded_aa_gap,
            "full_d6_projector_rank": int(audit["projector_rank"]),
            "exact_su2_and_space_group_from_analytic_construction": True,
        },
        "literature_targets": {
            "exact_36_site_energy_per_site": EXACT_ENERGY_PER_SITE,
            "space_group_gcnn_energy_per_site": GCNN_ENERGY_PER_SITE,
            "dsl_zero_lanczos_energy_per_site": DSL_ZERO_LANCZOS_ENERGY_PER_SITE,
            "dsl_one_lanczos_energy_per_site": DSL_ONE_LANCZOS_ENERGY_PER_SITE,
            "dsl_two_lanczos_energy_per_site": GUTZWILLER_LS_ENERGY_PER_SITE,
            "citations": [
                "Y. Iqbal et al., Phys. Rev. B 93, 144411 (2016), Table III",
                "C. Roth, A. Szabo, and A. H. MacDonald, Phys. Rev. B 108, 054410 (2023), Table III"
            ],
        },
        "replicates": rows,
        "aggregate": {
            "mean_energy_per_site": mean,
            "replicate_standard_error_per_site": _replicate_standard_error(energies),
            "gap_above_exact_per_site": mean - EXACT_ENERGY_PER_SITE,
            "relative_error_to_exact_magnitude": (
                (mean - EXACT_ENERGY_PER_SITE) / abs(EXACT_ENERGY_PER_SITE)
            ),
            "mean_variance_per_site": float(
                np.mean([row["variance_per_site"] for row in rows])
            ),
            "mean_acceptance_fraction": float(
                np.mean([row["acceptance_fraction"] for row in rows])
            ),
        },
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


__all__ = ["run_spin_n36_pi_flux_benchmark"]
