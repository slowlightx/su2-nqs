"""Transfer the exact-SU2/A1 physical-operator response from N=16 to N=36.

This is a deliberately local-compute experiment.  It does not optimize a
large neural network.  Instead it asks the prerequisite physics question for
such a calculation: on the same projected pi/0-flux carrier, do resonance
operators retain an energy-lowering tangent component beyond the ordinary
Hamiltonian/Lanczos direction when the Hilbert space is no longer enumerated?
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from su2_nqs.benchmarks.operator_tangent_response import (
    cross_validated_one_layer_operator_ritz,
    cross_validated_resonance_channel_ritz,
    sampled_one_layer_operator_ritz,
    sampled_resonance_channel_ritz,
    scalar_operator_tangent_statistics,
    triangular_a1_local_ratios,
    triangular_a1_shell_word_local_ratios,
)
from su2_nqs.benchmarks.qsl_three_arm import (
    TRIANGULAR_4X4_J2_0125_EXACT_ENERGY,
)
from su2_nqs.benchmarks.scalar_operator_frontier import (
    triangular_a1_resonance_operators,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import (
    SMALL_CASES,
    configuration_from_repository_integer,
    fixed_sz_sparse_hamiltonian,
    physical_seed_amplitudes,
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
TASK = "spin_n36_operator_tangent_transfer"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 operator-tangent transfer identity")
    sampler = spec.get("sampler")
    if not isinstance(sampler, dict):
        raise TypeError("sampler must be an object")
    for name in (
        "replicates",
        "chain_count",
        "burn_sweeps",
        "measurement_blocks",
        "decorrelation_sweeps",
        "center_batch_size",
        "amplitude_batch_size",
    ):
        if int(sampler.get(name, 0)) < 1:
            raise ValueError(f"sampler.{name} must be positive")
    if int(sampler["replicates"]) < 2:
        raise ValueError("at least two independent replicates are required")
    couplings = np.asarray(spec.get("couplings", ()), dtype=np.float64)
    if couplings.ndim != 1 or len(couplings) < 1 or np.any(~np.isfinite(couplings)):
        raise ValueError("couplings must be a nonempty finite sequence")


def _n16_exact_response(couplings: tuple[float, ...]) -> dict[str, object]:
    case = SMALL_CASES["square4"]
    h1, basis = fixed_sz_sparse_hamiltonian(case, 0.0)
    h_at_one, basis_at_one = fixed_sz_sparse_hamiltonian(case, 1.0)
    if basis != basis_at_one:
        raise RuntimeError("N16 shell bases differ")
    resonance = triangular_a1_resonance_operators(case, basis)
    operators = (h1, h_at_one - h1, *resonance.values())
    configurations = np.asarray(
        [
            configuration_from_repository_integer(value, case.n_sites)
            for value in basis
        ],
        dtype=np.int8,
    )
    carrier = np.asarray(
        physical_seed_amplitudes(configurations, case)[:, 0], dtype=np.float64
    )
    carrier /= np.linalg.norm(carrier)
    support = np.abs(carrier) > 1.0e-14
    responses = np.column_stack(
        [np.asarray(operator @ carrier) for operator in operators]
    )
    ratios = responses[support] / carrier[support, None]
    probabilities = np.abs(carrier[support]) ** 2
    report = scalar_operator_tangent_statistics(
        ratios,
        couplings,
        weights=probabilities,
    )
    report["support_size"] = int(np.count_nonzero(support))
    report["fixed_sz_dimension"] = len(carrier)
    report["carrier_norm_error"] = abs(float(np.vdot(carrier, carrier).real) - 1.0)
    words = np.empty((np.count_nonzero(support), 2, 5), dtype=np.complex128)
    for shell in range(2):
        for token in range(5):
            words[:, shell, token] = np.asarray(
                operators[shell] @ (operators[token] @ carrier)
            )[support] / carrier[support]
    report["one_layer_ritz"] = sampled_one_layer_operator_ritz(
        ratios,
        words,
        couplings,
        weights=probabilities,
    )
    report["single_resonance_channel_ritz"] = sampled_resonance_channel_ritz(
        ratios,
        words,
        couplings,
        weights=probabilities,
    )
    return report


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _coupling_row(report: dict[str, object], coupling: float) -> dict[str, object]:
    for row in report["couplings"]:
        if np.isclose(float(row["j2_over_j1"]), coupling, atol=1.0e-14):
            return row
    raise KeyError(coupling)


def run_spin_n36_operator_tangent_transfer(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    previous_payload = None
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    if result_path.exists() and overwrite:
        candidate = json.loads(result_path.read_text())
        if candidate.get("config") == spec:
            previous_payload = candidate
    sampler = spec["sampler"]
    couplings = tuple(float(value) for value in spec["couplings"])
    prefix_payload = None
    prefix_directory = None
    if spec.get("reuse_prefix_result") is not None:
        prefix_result = ROOT / str(spec["reuse_prefix_result"])
        prefix_payload = json.loads(prefix_result.read_text())
        prefix_directory = prefix_result.parent
        prefix_spec = prefix_payload.get("config", {})
        comparable_sampler_keys = (
            "chain_count",
            "burn_sweeps",
            "measurement_blocks",
            "decorrelation_sweeps",
            "center_batch_size",
            "amplitude_batch_size",
        )
        if (
            prefix_payload.get("task") != TASK
            or int(prefix_spec.get("seed", -1)) != int(spec["seed"])
            or tuple(float(value) for value in prefix_spec.get("couplings", ()))
            != couplings
            or any(
                int(prefix_spec.get("sampler", {}).get(key, -1))
                != int(sampler[key])
                for key in comparable_sampler_keys
            )
        ):
            raise ValueError("reuse_prefix_result is incompatible with this config")
    n16 = _n16_exact_response(couplings)
    orbit = build_n36_pi_zero_flux_orbit()

    def amplitudes(states: np.ndarray) -> np.ndarray:
        return projected_pi_zero_flux_a1_amplitudes(
            states,
            orbit=orbit,
            microbatch_size=int(sampler["amplitude_batch_size"]),
        ).amplitudes

    replicates = []
    ratio_replicates = []
    word_replicates = []
    for replicate in range(int(sampler["replicates"])):
        started = time.perf_counter()
        seed = int(spec["seed"]) + 1000 * replicate
        cache_path = output / f"replicate_{replicate}_local_ratios.npz"
        cached_payload = None
        cached_path = None
        if previous_payload is not None and cache_path.is_file():
            cached_payload = previous_payload
            cached_path = cache_path
        elif (
            prefix_payload is not None
            and replicate < len(prefix_payload.get("n36_replicates", ()))
        ):
            candidate = prefix_directory / f"replicate_{replicate}_local_ratios.npz"
            if candidate.is_file():
                cached_payload = prefix_payload
                cached_path = candidate
        if cached_payload is not None and cached_path is not None:
            cached = np.load(cached_path)
            ratios = np.asarray(cached["local_ratios"])
            words = np.asarray(cached["shell_word_ratios"])
            old_row = cached_payload["n36_replicates"][replicate]
            if (
                array_sha256(ratios) != old_row["local_ratio_sha256"]
                or array_sha256(words) != old_row["shell_word_ratio_sha256"]
            ):
                raise RuntimeError("cached local-ratio hash mismatch")
            if cached_path != cache_path:
                np.savez_compressed(
                    cache_path,
                    local_ratios=ratios,
                    shell_word_ratios=words,
                )
            response = scalar_operator_tangent_statistics(ratios, couplings)
            ritz = sampled_one_layer_operator_ritz(ratios, words, couplings)
            replicate_row = {
                **old_row,
                "operator_response": response,
                "finite_ritz_comparison": ritz,
                "cache_reused_for_analysis": True,
                "cache_source_result": str(
                    cached_path.parent.relative_to(ROOT) / "results.json"
                ),
            }
            ratio_replicates.append(ratios)
            word_replicates.append(words)
            replicates.append(replicate_row)
            atomic_json(output / f"replicate_{replicate}.json", replicate_row)
            print(f"replicate={replicate} reused verified ratio cache", flush=True)
            continue
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
        ratios, evaluation = triangular_a1_local_ratios(
            amplitudes,
            samples.configurations,
            samples.amplitudes,
            ((6, 0), (0, 6)),
            center_batch_size=int(sampler["center_batch_size"]),
            amplitude_batch_size=int(sampler["amplitude_batch_size"]),
        )
        response = scalar_operator_tangent_statistics(ratios, couplings)
        words, word_evaluation = triangular_a1_shell_word_local_ratios(
            amplitudes,
            samples.configurations,
            samples.amplitudes,
            ((6, 0), (0, 6)),
            amplitude_batch_size=int(sampler["amplitude_batch_size"]),
        )
        ritz = sampled_one_layer_operator_ritz(ratios, words, couplings)
        ratio_replicates.append(ratios)
        word_replicates.append(words)
        np.savez_compressed(
            output / f"replicate_{replicate}_local_ratios.npz",
            local_ratios=ratios,
            shell_word_ratios=words,
        )
        energy_rows = []
        for coupling in couplings:
            local_energy = ratios[:, 0] + coupling * ratios[:, 1]
            row = _coupling_row(response, coupling)
            energy_rows.append(
                {
                    "j2_over_j1": coupling,
                    "energy_per_site": float(row["mean_energy_total"]) / 36.0,
                    "chain_mean_mcse_per_site": chain_mean_standard_error(
                        local_energy,
                        chain_count=samples.chain_count,
                        measurement_blocks=samples.measurement_blocks,
                    )
                    / 36.0,
                    "variance_per_site": float(row["variance_total"]) / 36.0,
                }
            )
        replicate_row = {
            "replicate_index": replicate,
            "seed": seed,
            "sample_sha256": array_sha256(samples.configurations),
            "local_ratio_sha256": array_sha256(ratios),
            "shell_word_ratio_sha256": array_sha256(words),
            "acceptance_fraction": samples.acceptance_fraction,
            "burn_acceptance_fraction": samples.burn_acceptance_fraction,
            "energy_estimates": energy_rows,
            "operator_response": response,
            "finite_ritz_comparison": ritz,
            "evaluation_ledger": {
                "one_layer_local_ratios": evaluation,
                "depth_two_shell_words": word_evaluation,
            },
            "wall_seconds": time.perf_counter() - started,
        }
        replicates.append(replicate_row)
        atomic_json(output / f"replicate_{replicate}.json", replicate_row)
        focus = _coupling_row(response, 0.125 if 0.125 in couplings else couplings[0])
        print(
            f"replicate={replicate} samples={len(samples.configurations)} "
            f"full/H response={focus['full_to_hamiltonian_squared_slope_ratio']:.4f} "
            f"wall={replicate_row['wall_seconds']:.1f}s",
            flush=True,
        )

    cross_validation = cross_validated_one_layer_operator_ritz(
        ratio_replicates,
        word_replicates,
        couplings,
    )
    channel_cross_validation = cross_validated_resonance_channel_ritz(
        ratio_replicates,
        word_replicates,
        couplings,
    )

    aggregate_rows = []
    for coupling in couplings:
        rows = [
            _coupling_row(replicate["operator_response"], coupling)
            for replicate in replicates
        ]
        ritz_rows = [
            _coupling_row(replicate["finite_ritz_comparison"], coupling)
            for replicate in replicates
        ]
        energies = [
            next(
                energy["energy_per_site"]
                for energy in replicate["energy_estimates"]
                if np.isclose(energy["j2_over_j1"], coupling)
            )
            for replicate in replicates
        ]
        ratios = [float(row["full_to_hamiltonian_squared_slope_ratio"]) for row in rows]
        incremental = [
            float(row["incremental_squared_slope_fraction_beyond_two_shells"])
            for row in rows
        ]
        n16_row = _coupling_row(n16, coupling)
        n16_ratio = float(n16_row["full_to_hamiltonian_squared_slope_ratio"])
        n16_ritz = _coupling_row(n16["one_layer_ritz"], coupling)
        in_sample_ritz_differences = [
            float(row["operator_minus_krylov_energy_total"]) / 36.0
            for row in ritz_rows
        ]
        cross_validated_row = _coupling_row(cross_validation, coupling)
        ritz_differences = [
            float(fold["operator_minus_krylov_energy_total"]) / 36.0
            for fold in cross_validated_row["folds"]
        ]
        n16_ritz_difference = (
            float(n16_ritz["operator_minus_krylov_energy_total"]) / 16.0
        )
        n16_channel_row = _coupling_row(
            n16["single_resonance_channel_ritz"], coupling
        )
        preregistered_channel = min(
            n16_channel_row["resonance_channels"],
            key=lambda channel: float(
                channel["augmented_minus_krylov_energy_total"]
            ),
        )
        channel_row = _coupling_row(channel_cross_validation, coupling)
        held_out_channel = next(
            channel
            for channel in channel_row["resonance_channels"]
            if channel["token_label"] == preregistered_channel["token_label"]
        )
        channel_differences = [
            float(fold["augmented_minus_krylov_energy_total"]) / 36.0
            for fold in held_out_channel["folds"]
        ]
        hamiltonian_direction = np.asarray(
            (1.0, coupling, 0.0, 0.0, 0.0), dtype=np.float64
        )
        independence = {}
        for token_index in range(2, 5):
            fractions = []
            for replicate in replicates:
                metric = np.asarray(
                    replicate["operator_response"]["fubini_study_metric"],
                    dtype=np.float64,
                )
                numerator = float(metric[token_index] @ hamiltonian_direction) ** 2
                denominator = float(metric[token_index, token_index]) * float(
                    hamiltonian_direction @ metric @ hamiltonian_direction
                )
                fractions.append(1.0 - numerator / denominator)
            n16_metric = np.asarray(n16["fubini_study_metric"], dtype=np.float64)
            n16_numerator = float(
                n16_metric[token_index] @ hamiltonian_direction
            ) ** 2
            n16_denominator = float(n16_metric[token_index, token_index]) * float(
                hamiltonian_direction @ n16_metric @ hamiltonian_direction
            )
            independence[TRIANGULAR_A1_SCALAR_LABELS[token_index]] = {
                "n16_exact_fraction_orthogonal_to_hamiltonian_tangent": (
                    1.0 - n16_numerator / n16_denominator
                ),
                "n36_mean_fraction_orthogonal_to_hamiltonian_tangent": float(
                    np.mean(fractions)
                ),
                "n36_replicate_standard_error": _standard_error(fractions),
            }
        aggregate_rows.append(
            {
                "j2_over_j1": coupling,
                "mean_energy_per_site": float(np.mean(energies)),
                "replicate_standard_error_energy_per_site": _standard_error(energies),
                "mean_full_to_hamiltonian_squared_slope_ratio": float(
                    np.mean(ratios)
                ),
                "replicate_standard_error_response_ratio": _standard_error(ratios),
                "mean_incremental_squared_slope_fraction_beyond_two_shells": float(
                    np.mean(incremental)
                ),
                "replicate_standard_error_incremental_fraction": _standard_error(
                    incremental
                ),
                "n16_exact_full_to_hamiltonian_squared_slope_ratio": n16_ratio,
                "n36_to_n16_response_ratio": float(np.mean(ratios)) / n16_ratio,
                "mean_operator_minus_one_lanczos_energy_per_site": float(
                    np.mean(ritz_differences)
                ),
                "replicate_standard_error_operator_minus_one_lanczos_per_site": (
                    _standard_error(ritz_differences)
                ),
                "operator_below_one_lanczos_replicates": int(
                    np.count_nonzero(np.asarray(ritz_differences) < 0.0)
                ),
                "mean_in_sample_operator_minus_one_lanczos_energy_per_site": (
                    float(np.mean(in_sample_ritz_differences))
                ),
                "n16_exact_operator_minus_one_lanczos_energy_per_site": (
                    n16_ritz_difference
                ),
                "n36_to_n16_absolute_ritz_gain_ratio": (
                    abs(float(np.mean(ritz_differences)))
                    / abs(n16_ritz_difference)
                    if abs(n16_ritz_difference) > 1.0e-15
                    else None
                ),
                "n16_preregistered_best_single_resonance_channel": (
                    preregistered_channel["token_label"]
                ),
                "n16_preregistered_channel_gain_per_site": float(
                    preregistered_channel[
                        "augmented_minus_krylov_energy_total"
                    ]
                )
                / 16.0,
                "n36_held_out_preregistered_channel_gain_per_site": float(
                    np.mean(channel_differences)
                ),
                "n36_held_out_preregistered_channel_standard_error": (
                    _standard_error(channel_differences)
                ),
                "n36_preregistered_channel_below_lanczos_replicates": int(
                    np.count_nonzero(np.asarray(channel_differences) < 0.0)
                ),
                "resonance_tangent_independence": independence,
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
        "physics_question": (
            "Do exact-SU2/Gamma-A1 resonance operators retain an independent "
            "energy-lowering tangent beyond the Hamiltonian/Lanczos direction "
            "from N=16 to the non-enumerated N=36 system?"
        ),
        "model_case": {
            "hamiltonian": "spin-1/2 triangular J1-J2 Heisenberg",
            "n36_boundary": "6x6 periodic torus",
            "target_sector": "exact S=0, M=0, Gamma-A1",
            "carrier": "Gutzwiller-projected spin-independent pi/0-flux state",
            "operator_tokens": list(TRIANGULAR_A1_SCALAR_LABELS),
        },
        "n16_exhaustive_reference": n16,
        "n16_exact_ground_energy_anchor_at_j2_0125": (
            TRIANGULAR_4X4_J2_0125_EXACT_ENERGY / 16.0
        ),
        "n36_replicates": replicates,
        "n36_leave_one_replicate_out_ritz": cross_validation,
        "n36_leave_one_replicate_out_single_resonance_channels": (
            channel_cross_validation
        ),
        "aggregate": aggregate_rows,
        "interpretation_boundary": (
            "The held-out finite Ritz rows are the only energy-transfer test. "
            "In-sample Ritz energies are retained as finite-sample bias controls. "
            "Tangent orthogonality establishes independent capacity, not an "
            "optimized N36 finite-depth Operator-ViT energy."
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


__all__ = ["TASK", "run_spin_n36_operator_tangent_transfer"]
