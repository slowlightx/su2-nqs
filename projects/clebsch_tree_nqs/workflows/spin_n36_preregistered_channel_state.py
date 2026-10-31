"""Independent-state audit of N36 preregistered resonance corrections.

Channel identities are selected only by the exhaustive N16 result.  Their
coefficients are fitted on the separate carrier-Born L1 data, after which a
new mixture proposal samples the actual one-Lanczos and resonance-augmented
states.  Energies are evaluated from their un-divided amplitudes/actions on
common configurations, avoiding in-sample Ritz and carrier-node assumptions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from su2_nqs.benchmarks.operator_tangent_response import (
    one_lanczos_and_resonance_state_actions,
    sampled_resonance_channel_ritz,
    triangular_a1_operator_actions,
    triangular_a1_shell_word_actions,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.operators.triangular_a1_scalar import TRIANGULAR_A1_SCALAR_LABELS
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_preregistered_channel_state_audit"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid preregistered-channel audit identity")
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
    if int(sampler["replicates"]) < 2:
        raise ValueError("at least two independent audit replicates are required")


def _coefficients(row: dict[str, object]) -> np.ndarray:
    return np.asarray(row["coefficients_real"], dtype=np.float64) + 1j * np.asarray(
        row["coefficients_imag"], dtype=np.float64
    )


def _state_estimate(
    state: np.ndarray,
    action: np.ndarray,
    proposal: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    denominator_terms = np.abs(state / proposal) ** 2
    denominator = float(np.sum(denominator_terms))
    if not np.isfinite(denominator) or denominator <= 1.0e-15:
        raise FloatingPointError("state has no weight under the mixture proposal")
    numerator_terms = np.conj(state) * action / np.abs(proposal) ** 2
    energy = complex(np.sum(numerator_terms) / denominator)
    variance = float(
        np.sum(np.abs(action - energy * state) ** 2 / np.abs(proposal) ** 2).real
        / denominator
    )
    normalized = denominator_terms / denominator
    ess = float(1.0 / np.sum(normalized**2))
    return {
        "energy_total": float(energy.real),
        "energy_per_site": float(energy.real / 36.0),
        "variance_total": variance,
        "variance_per_site": variance / 36.0,
        "importance_ess_fraction": ess / len(state),
        "maximum_normalized_weight": float(np.max(normalized)),
        "absolute_imaginary_energy": abs(float(energy.imag)),
    }, numerator_terms


def _chain_difference_standard_error(
    krylov_state: np.ndarray,
    krylov_action: np.ndarray,
    resonance_state: np.ndarray,
    resonance_action: np.ndarray,
    proposal: np.ndarray,
    *,
    chain_count: int,
    measurement_blocks: int,
) -> tuple[float, list[float]]:
    differences = []
    for chain in range(chain_count):
        indices = np.arange(chain, chain_count * measurement_blocks, chain_count)
        energies = []
        for state, action in (
            (krylov_state, krylov_action),
            (resonance_state, resonance_action),
        ):
            denominator = np.sum(np.abs(state[indices] / proposal[indices]) ** 2)
            numerator = np.sum(
                np.conj(state[indices])
                * action[indices]
                / np.abs(proposal[indices]) ** 2
            )
            energies.append(float((numerator / denominator).real))
        differences.append((energies[1] - energies[0]) / 36.0)
    standard_error = float(
        np.std(differences, ddof=1) / np.sqrt(len(differences))
    )
    return standard_error, differences


def _load_training_data(
    result_path: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    payload = json.loads(result_path.read_text())
    if payload.get("status") != "complete":
        raise ValueError("training response artifact must be complete")
    ratios = []
    words = []
    for replicate in payload["n36_replicates"]:
        index = int(replicate["replicate_index"])
        cached = np.load(result_path.parent / f"replicate_{index}_local_ratios.npz")
        ratios.append(np.asarray(cached["local_ratios"]))
        words.append(np.asarray(cached["shell_word_ratios"]))
    return payload, np.concatenate(ratios), np.concatenate(words)


def run_spin_n36_preregistered_channel_state(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    training_result_path = ROOT / str(spec["training_response_result"])
    training_payload, training_ratios, training_words = _load_training_data(
        training_result_path
    )
    couplings = tuple(float(value) for value in spec["couplings"])
    fitted = sampled_resonance_channel_ritz(
        training_ratios, training_words, couplings
    )
    orbit = build_n36_pi_zero_flux_orbit()
    sampler = spec["sampler"]

    def carrier(states: np.ndarray) -> np.ndarray:
        return projected_pi_zero_flux_a1_amplitudes(
            states,
            orbit=orbit,
            microbatch_size=int(sampler["amplitude_batch_size"]),
        ).amplitudes

    coupling_results = []
    for coupling_index, coupling in enumerate(couplings):
        aggregate_row = next(
            row
            for row in training_payload["aggregate"]
            if np.isclose(float(row["j2_over_j1"]), coupling)
        )
        label = str(aggregate_row["n16_preregistered_best_single_resonance_channel"])
        token_index = TRIANGULAR_A1_SCALAR_LABELS.index(label)
        fitted_row = next(
            row
            for row in fitted["couplings"]
            if np.isclose(float(row["j2_over_j1"]), coupling)
        )
        channel = next(
            row for row in fitted_row["resonance_channels"] if row["token_label"] == label
        )
        krylov_coefficients = _coefficients(fitted_row["one_lanczos_krylov"])
        resonance_coefficients = _coefficients(channel["augmented_ritz"])

        def proposal(states: np.ndarray) -> np.ndarray:
            centers = np.asarray(carrier(states), dtype=np.complex128)
            actions, _ = triangular_a1_operator_actions(
                carrier,
                states,
                ((6, 0), (0, 6)),
                center_batch_size=int(sampler["center_batch_size"]),
                amplitude_batch_size=int(sampler["amplitude_batch_size"]),
            )
            h_carrier = actions[:, 0] + coupling * actions[:, 1]
            krylov_state = (
                krylov_coefficients[0] * centers
                + krylov_coefficients[1] * h_carrier
            )
            resonance_state = (
                resonance_coefficients[0] * centers
                + resonance_coefficients[1] * h_carrier
                + resonance_coefficients[2] * actions[:, token_index]
            )
            return np.sqrt(np.abs(krylov_state) ** 2 + np.abs(resonance_state) ** 2)

        replicate_rows = []
        for replicate in range(int(sampler["replicates"])):
            started = time.perf_counter()
            seed = int(spec["seed"]) + 10000 * coupling_index + 1000 * replicate
            samples = sample_fixed_wavefunction_exchange(
                proposal,
                n_sites=36,
                n_up=18,
                chain_count=int(sampler["chain_count"]),
                burn_sweeps=int(sampler["burn_sweeps"]),
                measurement_blocks=int(sampler["measurement_blocks"]),
                decorrelation_sweeps=int(sampler["decorrelation_sweeps"]),
                seed=seed,
            )
            centers = np.asarray(carrier(samples.configurations), dtype=np.complex128)
            actions, action_ledger = triangular_a1_operator_actions(
                carrier,
                samples.configurations,
                ((6, 0), (0, 6)),
                center_batch_size=int(sampler["center_batch_size"]),
                amplitude_batch_size=int(sampler["amplitude_batch_size"]),
            )
            words, word_ledger = triangular_a1_shell_word_actions(
                carrier,
                samples.configurations,
                ((6, 0), (0, 6)),
                amplitude_batch_size=int(sampler["amplitude_batch_size"]),
            )
            states_and_actions = one_lanczos_and_resonance_state_actions(
                centers,
                actions,
                words,
                coupling,
                token_index,
                krylov_coefficients,
                resonance_coefficients,
            )
            krylov, _ = _state_estimate(
                states_and_actions[0], states_and_actions[1], samples.amplitudes
            )
            resonance, _ = _state_estimate(
                states_and_actions[2], states_and_actions[3], samples.amplitudes
            )
            difference = resonance["energy_per_site"] - krylov["energy_per_site"]
            chain_se, chain_differences = _chain_difference_standard_error(
                *states_and_actions,
                samples.amplitudes,
                chain_count=samples.chain_count,
                measurement_blocks=samples.measurement_blocks,
            )
            row = {
                "replicate_index": replicate,
                "seed": seed,
                "sample_count": len(samples.configurations),
                "acceptance_fraction": samples.acceptance_fraction,
                "burn_acceptance_fraction": samples.burn_acceptance_fraction,
                "one_lanczos": krylov,
                "preregistered_resonance": resonance,
                "resonance_minus_lanczos_energy_per_site": difference,
                "chain_mean_standard_error_difference_per_site": chain_se,
                "chain_differences_per_site": chain_differences,
                "evaluation_ledger": {
                    "one_layer": action_ledger,
                    "depth_two": word_ledger,
                },
                "wall_seconds": time.perf_counter() - started,
            }
            replicate_rows.append(row)
            atomic_json(
                output / f"coupling_{coupling_index}_replicate_{replicate}.json",
                row,
            )
            print(
                f"J2={coupling:.3f} replicate={replicate} "
                f"dE/N={difference:+.6e} ESS="
                f"{resonance['importance_ess_fraction']:.3f}",
                flush=True,
            )
        differences = np.asarray(
            [row["resonance_minus_lanczos_energy_per_site"] for row in replicate_rows]
        )
        coupling_results.append(
            {
                "j2_over_j1": coupling,
                "preregistered_token_index": token_index,
                "preregistered_token_label": label,
                "krylov_coefficients_real": np.real(krylov_coefficients).tolist(),
                "krylov_coefficients_imag": np.imag(krylov_coefficients).tolist(),
                "resonance_coefficients_real": np.real(
                    resonance_coefficients
                ).tolist(),
                "resonance_coefficients_imag": np.imag(
                    resonance_coefficients
                ).tolist(),
                "replicates": replicate_rows,
                "mean_resonance_minus_lanczos_energy_per_site": float(
                    np.mean(differences)
                ),
                "replicate_standard_error_difference_per_site": float(
                    np.std(differences, ddof=1) / np.sqrt(len(differences))
                ),
                "negative_difference_replicates": int(
                    np.count_nonzero(differences < 0.0)
                ),
                "minimum_resonance_importance_ess_fraction": float(
                    min(
                        row["preregistered_resonance"]["importance_ess_fraction"]
                        for row in replicate_rows
                    )
                ),
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
        "training_response_result": str(training_result_path.relative_to(ROOT)),
        "training_response_result_sha256": training_payload["result_sha256"],
        "model_case": {
            "hamiltonian": "spin-1/2 triangular J1-J2 Heisenberg",
            "boundary": "6x6 periodic torus",
            "target_sector": "exact S=0, Gamma-A1",
            "carrier": "projected pi/0-flux",
            "comparison": "one Lanczos versus N16-preregistered resonance",
        },
        "couplings": coupling_results,
        "selection_protocol": (
            "token identity from exhaustive N16 only; coefficients from separate "
            "L1 carrier samples; energies from new common mixture samples"
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


__all__ = ["TASK", "run_spin_n36_preregistered_channel_state"]
