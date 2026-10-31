#!/usr/bin/env python3
"""Cross-fitted three-coordinate N=36 adaptive-carrier training canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from su2_nqs.benchmarks.adaptive_projected_pair_carrier import (  # noqa: E402
    triangular_physical_pairing_bank,
)
from su2_nqs.benchmarks.projected_singlet_agp import (  # noqa: E402
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.benchmarks.triangular_induced_parton import (  # noqa: E402
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import atomic_json  # noqa: E402
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds  # noqa: E402
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig  # noqa: E402
from su2_nqs.vmc.fixed_wavefunction_mcmc import (  # noqa: E402
    sample_fixed_wavefunction_exchange,
)
from su2_nqs.vmc.projected_agp import (  # noqa: E402
    ProjectedAGPHeisenbergConfig,
    projected_agp_heisenberg_action,
)


def _adam_update(parameters, gradient, first, second, step, learning_rate):
    first = 0.9 * first + 0.1 * gradient
    second = 0.999 * second + 0.001 * gradient**2
    corrected_first = first / (1.0 - 0.9**step)
    corrected_second = second / (1.0 - 0.999**step)
    parameters = parameters - learning_rate * corrected_first / (
        np.sqrt(corrected_second) + 1.0e-8
    )
    return np.clip(parameters, -0.35, 0.35), first, second


def _summary(amplitudes, actions, proposal, chains):
    weight = np.abs(amplitudes) ** 2 / proposal
    energy = complex(np.sum(np.conj(amplitudes) * actions / proposal) / np.sum(weight))
    residual = actions - energy * amplitudes
    variance = float(np.sum(np.abs(residual) ** 2 / proposal) / np.sum(weight))
    ess = float(np.sum(weight) ** 2 / np.sum(weight**2))
    chain_values = []
    for chain in range(chains):
        index = np.arange(chain, len(amplitudes), chains)
        chain_values.append(
            float(
                np.sum(np.conj(amplitudes[index]) * actions[index] / proposal[index]).real
                / np.sum(weight[index])
                / 36.0
            )
        )
    return {
        "energy_total": float(energy.real),
        "energy_per_site": float(energy.real / 36.0),
        "variance_total": variance,
        "variance_per_site": variance / 36.0,
        "importance_ess": ess,
        "importance_ess_fraction": ess / len(weight),
        "chain_energy_per_site": chain_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--ridge", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("this canary requires the GPU backend")

    j2 = 0.125
    chains, blocks = 16, 24
    orbit = build_n36_pi_zero_flux_orbit()

    @jax.jit
    def base_kernel(states):
        return projected_pi_zero_flux_a1_amplitudes_jax(states, orbit=orbit)

    def base_amplitudes(states):
        return np.asarray(base_kernel(jnp.asarray(states, dtype=jnp.int8)))

    train_samples = sample_fixed_wavefunction_exchange(
        base_amplitudes,
        n_sites=36,
        n_up=18,
        chain_count=chains,
        burn_sweeps=80,
        measurement_blocks=blocks,
        decorrelation_sweeps=8,
        seed=args.seed,
    )
    audit_samples = sample_fixed_wavefunction_exchange(
        base_amplitudes,
        n_sites=36,
        n_up=18,
        chain_count=chains,
        burn_sweeps=80,
        measurement_blocks=blocks,
        decorrelation_sweeps=8,
        seed=args.seed + 1,
    )
    train_proposal = np.abs(train_samples.amplitudes) ** 2
    audit_proposal = np.abs(audit_samples.amplitudes) ** 2

    base, directions, labels = triangular_physical_pairing_bank(6)
    upper = np.triu_indices(36, 1)
    packed_base = jnp.asarray(base[upper], dtype=jnp.float64)
    packed_directions = jnp.asarray(
        np.stack([direction[upper] for direction in directions]), dtype=jnp.float64
    )
    group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    carrier = ProjectedSingletAGPConfig(
        n_sites=36,
        group_images=tuple(tuple(int(site) for site in image) for image in group),
    )
    bonds = triangular_j1j2_bonds(6, 6, j1=1.0, j2=j2, periodic=True)
    graph = ProjectedAGPHeisenbergConfig(
        n_sites=36,
        bonds=tuple((int(i), int(j), float(coupling)) for i, j, coupling in bonds),
    )

    def objective(parameters, states, proposal):
        packed = packed_base + parameters @ packed_directions
        action = projected_agp_heisenberg_action(
            {"pairing_real": packed},
            states,
            carrier,
            graph,
            amplitude_microbatch_size=512,
        )
        amplitude_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(action.amplitudes)),
                jnp.max(jnp.abs(action.hamiltonian_actions)),
            ),
            jnp.finfo(jnp.float64).tiny,
        )
        amplitudes = action.amplitudes / amplitude_scale
        hamiltonian_actions = action.hamiltonian_actions / amplitude_scale
        log_measure = -jnp.log(proposal)
        measure = jnp.exp(log_measure - jnp.max(log_measure))
        state_weights = measure * jnp.abs(amplitudes) ** 2
        denominator = jnp.sum(state_weights)
        energy = jnp.real(
            jnp.sum(measure * jnp.conj(amplitudes) * hamiltonian_actions)
            / denominator
        )
        ess = jnp.sum(state_weights) ** 2 / jnp.sum(state_weights**2)
        loss = energy + args.ridge * jnp.sum(parameters**2)
        return loss, (energy, ess / len(proposal))

    value_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))
    generator = np.random.default_rng(args.seed + 2)
    fold_rows = []
    trained_parameters = []
    started = time.perf_counter()
    for fold in range(2):
        pool = np.arange(fold, len(train_samples.configurations), 2)
        parameters = np.zeros(3, dtype=np.float64)
        first = np.zeros(3, dtype=np.float64)
        second = np.zeros(3, dtype=np.float64)
        history = []
        for step in range(1, args.steps + 1):
            chosen = generator.choice(pool, size=16, replace=False)
            (loss, (energy, ess)), gradient = value_gradient(
                jnp.asarray(parameters),
                jnp.asarray(train_samples.configurations[chosen]),
                jnp.asarray(train_proposal[chosen]),
            )
            gradient_host = np.asarray(gradient)
            parameters, first, second = _adam_update(
                parameters,
                gradient_host,
                first,
                second,
                step,
                args.learning_rate,
            )
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "energy_total": float(energy),
                    "minibatch_ess_fraction": float(ess),
                    "gradient_norm": float(np.linalg.norm(gradient_host)),
                    "parameters": parameters.tolist(),
                }
            )
        trained_parameters.append(parameters)
        fold_rows.append(
            {
                "fold": fold,
                "train_pool_size": len(pool),
                "parameters": parameters.tolist(),
                "history": history,
            }
        )

    def evaluate(parameters):
        packed = packed_base + jnp.asarray(parameters) @ packed_directions
        amplitude_pieces = []
        action_pieces = []
        for start in range(0, len(audit_samples.configurations), 16):
            action = projected_agp_heisenberg_action(
                {"pairing_real": packed},
                audit_samples.configurations[start : start + 16],
                carrier,
                graph,
                amplitude_microbatch_size=512,
            )
            amplitude_pieces.append(np.asarray(action.amplitudes))
            action_pieces.append(np.asarray(action.hamiltonian_actions))
        return np.concatenate(amplitude_pieces), np.concatenate(action_pieces)

    baseline_amplitudes, baseline_actions = evaluate(np.zeros(3))
    baseline = _summary(
        baseline_amplitudes, baseline_actions, audit_proposal, chains
    )
    audits = []
    for fold, parameters in enumerate(trained_parameters):
        amplitudes, actions = evaluate(parameters)
        summary = _summary(amplitudes, actions, audit_proposal, chains)
        paired = np.asarray(summary["chain_energy_per_site"]) - np.asarray(
            baseline["chain_energy_per_site"]
        )
        audits.append(
            {
                "fold": fold,
                "parameters": parameters.tolist(),
                **summary,
                "paired_minus_baseline_per_site": float(np.mean(paired)),
                "paired_chain_standard_error_per_site": float(
                    np.std(paired, ddof=1) / np.sqrt(len(paired))
                ),
            }
        )
    parameter_distance = float(
        np.linalg.norm(trained_parameters[0] - trained_parameters[1])
    )
    payload = {
        "schema_version": 1,
        "task": "n36_adaptive_pair_crossfit_training_canary",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "model": "triangular_J1_J2_Heisenberg_6x6_PBC",
        "j2_over_j1": j2,
        "target_sector": "exact_S0_Gamma_A1",
        "parameter_labels": list(labels),
        "training": {
            "folds": 2,
            "steps_per_fold": args.steps,
            "minibatch_size": 16,
            "learning_rate": args.learning_rate,
            "ridge": args.ridge,
            "coordinate_bound": 0.35,
        },
        "train_sample_count": len(train_samples.configurations),
        "untouched_audit_sample_count": len(audit_samples.configurations),
        "baseline_audit": baseline,
        "fold_training": fold_rows,
        "fold_audits": audits,
        "cross_fold_parameter_distance": parameter_distance,
        "wall_seconds": time.perf_counter() - started,
        "decision": {
            "both_folds_lower_audit_energy": all(
                row["paired_minus_baseline_per_site"] < 0.0 for row in audits
            ),
            "both_folds_two_sigma_resolved": all(
                row["paired_minus_baseline_per_site"]
                + 2.0 * row["paired_chain_standard_error_per_site"]
                < 0.0
                for row in audits
            ),
            "both_folds_audit_ess_above_0p2": all(
                row["importance_ess_fraction"] > 0.2 for row in audits
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
