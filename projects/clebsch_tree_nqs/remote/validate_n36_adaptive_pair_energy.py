#!/usr/bin/env python3
"""Paired-mixture N=36 energy audit for a transferred adaptive carrier."""

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
from su2_nqs.models.projected_singlet_agp import (  # noqa: E402
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)
from su2_nqs.vmc.fixed_wavefunction_mcmc import (  # noqa: E402
    fixed_wavefunction_local_energies,
    sample_fixed_wavefunction_exchange,
)


TRANSFER_PARAMETERS = np.asarray(
    [-0.37892179225984485, -0.24983280650264278, 0.4064207615603173],
    dtype=np.float64,
)


def _weighted_summary(local: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    normalized = weights / np.sum(weights)
    energy = complex(np.sum(normalized * local))
    variance = float(np.sum(normalized * np.abs(local - energy) ** 2).real)
    ess = float(np.sum(weights) ** 2 / np.sum(weights**2))
    return {
        "energy_total": float(energy.real),
        "energy_per_site": float(energy.real / 36.0),
        "variance_total": variance,
        "variance_per_site": variance / 36.0,
        "importance_ess": ess,
        "importance_ess_fraction": ess / len(weights),
        "maximum_abs_imaginary_local_energy": float(np.max(np.abs(local.imag))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--j2", type=float, default=0.125)
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--burn", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--decorrelation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("this audit requires the GPU backend")

    orbit = build_n36_pi_zero_flux_orbit()
    base, directions, labels = triangular_physical_pairing_bank(6)
    pairing = base + sum(
        coefficient * direction
        for coefficient, direction in zip(
            TRANSFER_PARAMETERS, directions, strict=True
        )
    )
    upper = np.triu_indices(36, 1)
    group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    carrier_config = ProjectedSingletAGPConfig(
        n_sites=36,
        group_images=tuple(tuple(int(site) for site in image) for image in group),
    )
    packed_pairing = jnp.asarray(pairing[upper], dtype=jnp.float64)

    @jax.jit
    def fixed_kernel(states):
        return projected_pi_zero_flux_a1_amplitudes_jax(states, orbit=orbit)

    @jax.jit
    def adaptive_kernel(states):
        return projected_singlet_agp_amplitudes(
            {"pairing_real": packed_pairing}, states, carrier_config
        ).amplitudes

    def fixed(states: np.ndarray) -> np.ndarray:
        return np.asarray(fixed_kernel(jnp.asarray(states, dtype=jnp.int8)))

    def adaptive_unscaled(states: np.ndarray) -> np.ndarray:
        return np.asarray(adaptive_kernel(jnp.asarray(states, dtype=jnp.int8)))

    # Fix one proposal-only scale before sampling.  It does not change either
    # variational state, and prevents one unnormalised determinant convention
    # from monopolising the mixture proposal.
    rng = np.random.default_rng(args.seed - 1)
    pilot = np.zeros((256, 36), dtype=np.int8)
    for row in pilot:
        row[rng.choice(36, 18, replace=False)] = 1
    pilot_fixed = fixed(pilot)
    pilot_adaptive = adaptive_unscaled(pilot)
    proposal_scale = float(
        np.sqrt(
            np.mean(np.abs(pilot_fixed) ** 2)
            / np.mean(np.abs(pilot_adaptive) ** 2)
        )
    )

    def adaptive(states: np.ndarray) -> np.ndarray:
        return proposal_scale * adaptive_unscaled(states)

    def mixture(states: np.ndarray) -> np.ndarray:
        left, right = fixed(states), adaptive(states)
        return np.sqrt(np.abs(left) ** 2 + np.abs(right) ** 2)

    started = time.perf_counter()
    samples = sample_fixed_wavefunction_exchange(
        mixture,
        n_sites=36,
        n_up=18,
        chain_count=args.chains,
        burn_sweeps=args.burn,
        measurement_blocks=args.blocks,
        decorrelation_sweeps=args.decorrelation,
        seed=args.seed,
    )
    bonds = triangular_j1j2_bonds(
        6, 6, j1=1.0, j2=args.j2, periodic=True
    )
    local_fixed = fixed_wavefunction_local_energies(
        fixed, samples.configurations, bonds, batch_size=16
    )
    local_adaptive = fixed_wavefunction_local_energies(
        adaptive, samples.configurations, bonds, batch_size=16
    )
    amp_fixed = fixed(samples.configurations)
    amp_adaptive = adaptive(samples.configurations)
    proposal_probability = np.abs(amp_fixed) ** 2 + np.abs(amp_adaptive) ** 2
    weights_fixed = np.abs(amp_fixed) ** 2 / proposal_probability
    weights_adaptive = np.abs(amp_adaptive) ** 2 / proposal_probability
    fixed_summary = _weighted_summary(local_fixed, weights_fixed)
    adaptive_summary = _weighted_summary(local_adaptive, weights_adaptive)

    fixed_by_chain = []
    adaptive_by_chain = []
    for chain in range(args.chains):
        indices = np.arange(chain, len(samples.configurations), args.chains)
        wf = weights_fixed[indices]
        wa = weights_adaptive[indices]
        fixed_by_chain.append(
            float(np.sum(wf * local_fixed[indices]).real / np.sum(wf) / 36.0)
        )
        adaptive_by_chain.append(
            float(np.sum(wa * local_adaptive[indices]).real / np.sum(wa) / 36.0)
        )
    paired = np.asarray(adaptive_by_chain) - np.asarray(fixed_by_chain)
    overlap_numerator = np.mean(
        np.conj(amp_fixed) * amp_adaptive / proposal_probability
    )
    overlap_denominator = np.sqrt(
        np.mean(np.abs(amp_fixed) ** 2 / proposal_probability)
        * np.mean(np.abs(amp_adaptive) ** 2 / proposal_probability)
    )
    payload = {
        "schema_version": 1,
        "task": "n36_adaptive_pair_paired_energy_canary",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "model": "triangular_J1_J2_Heisenberg_6x6_PBC",
        "j2_over_j1": args.j2,
        "target_sector": "exact_S0_Gamma_A1",
        "parameter_labels": list(labels),
        "transferred_n16_parameters": TRANSFER_PARAMETERS.tolist(),
        "full_reynolds_group_order": len(group),
        "proposal_scale_applied_only_to_adaptive_amplitude": proposal_scale,
        "sample_count": len(samples.configurations),
        "chain_count": args.chains,
        "measurement_blocks": args.blocks,
        "acceptance_fraction": samples.acceptance_fraction,
        "burn_acceptance_fraction": samples.burn_acceptance_fraction,
        "fixed_carrier": fixed_summary,
        "adaptive_carrier": adaptive_summary,
        "paired_adaptive_minus_fixed_per_site": {
            "mean": float(np.mean(paired)),
            "chain_standard_error": float(
                np.std(paired, ddof=1) / np.sqrt(len(paired))
            ),
            "chain_values": paired.tolist(),
        },
        "squared_overlap": float(abs(overlap_numerator / overlap_denominator) ** 2),
        "wall_seconds": time.perf_counter() - started,
        "decision": {
            "adaptive_energy_lower": adaptive_summary["energy_per_site"]
            < fixed_summary["energy_per_site"],
            "paired_two_sigma_resolved": float(np.mean(paired))
            + 2.0 * float(np.std(paired, ddof=1) / np.sqrt(len(paired)))
            < 0.0,
            "variance_lower": adaptive_summary["variance_per_site"]
            < fixed_summary["variance_per_site"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
