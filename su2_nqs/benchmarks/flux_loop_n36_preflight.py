"""Bounded triangular-6x6 preflight for the flux/loop mechanism.

This module deliberately stops before a variational-energy fit.  Fresh
pi/0-flux Born chains probe whether three full-Reynolds paired-Z2 carriers
have usable support and whether the exact-A1 dimer-resonance action produces
configuration-dependent directions distinct from the Hamiltonian tangent.
Passing licenses a small matched-energy pilot, never production training.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np

from ..lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from ..models.projected_singlet_agp import (
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)
from ..operators.triangular_a1_scalar import (
    TRIANGULAR_A1_SCALAR_LABELS,
    triangular_a1_scalar_token_connectivity,
)
from ..vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from .paired_z2_carrier import PairedZ2Kernel, paired_z2_kernel_bank
from .triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes,
)


SUPERCELL = ((6, 0), (0, 6))
N_SITES = 36
J2_OVER_J1 = 0.125
DIMER_LABEL = "rhombus_dimer_resonance"


@dataclass(frozen=True)
class ProjectedPairedBatch:
    """Projected carrier amplitudes and cancellation diagnostics."""

    amplitudes: np.ndarray
    cancellation_ratios: np.ndarray
    exact_nodes: np.ndarray


def _selected_operator_actions(
    amplitude_fn,
    configurations: np.ndarray,
    labels: Sequence[str],
    *,
    center_batch_size: int,
    amplitude_batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evaluate only the registered tokens needed by this bounded preflight."""

    selected = tuple(str(label) for label in labels)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(label not in TRIANGULAR_A1_SCALAR_LABELS for label in selected)
    ):
        raise ValueError("labels must be a nonempty unique registered subset")
    states = np.asarray(configurations, dtype=np.int8)
    actions = np.zeros((len(states), len(selected)), dtype=np.complex128)
    connected_requests = 0
    unique_evaluations = 0
    maximum_union_size = 0
    for start in range(0, len(states), center_batch_size):
        stop = min(len(states), start + center_batch_size)
        records = []
        union: dict[tuple[int, ...], int] = {}
        for state in states[start:stop]:
            tokens = triangular_a1_scalar_token_connectivity(state, SUPERCELL)
            row = []
            for label in selected:
                token_states, elements = tokens[label]
                row.append((token_states, elements))
                connected_requests += len(token_states)
                for connected in token_states:
                    if connected not in union:
                        union[connected] = len(union)
            records.append(row)
        ordered = np.asarray(tuple(union), dtype=np.int8)
        pieces = []
        for amplitude_start in range(0, len(ordered), amplitude_batch_size):
            batch = ordered[amplitude_start : amplitude_start + amplitude_batch_size]
            observed = np.asarray(amplitude_fn(batch), dtype=np.complex128)
            if observed.shape != (len(batch),) or np.any(~np.isfinite(observed)):
                raise FloatingPointError("amplitude_fn returned invalid values")
            pieces.append(observed)
        values = np.concatenate(pieces)
        unique_evaluations += len(ordered)
        maximum_union_size = max(maximum_union_size, len(ordered))
        for local_row, row in enumerate(records):
            for token_index, (token_states, elements) in enumerate(row):
                positions = np.fromiter(
                    (union[connected] for connected in token_states),
                    dtype=np.int64,
                    count=len(token_states),
                )
                actions[start + local_row, token_index] = np.dot(
                    elements, values[positions]
                )
    return actions, {
        "sample_count": len(states),
        "connected_configuration_requests": connected_requests,
        "unique_connected_amplitude_evaluations": unique_evaluations,
        "deduplication_fraction": 1.0
        - unique_evaluations / max(1, connected_requests),
        "maximum_center_batch_union_size": maximum_union_size,
        "token_labels": list(selected),
    }


def _projected_config() -> ProjectedSingletAGPConfig:
    group = triangular_hnf_gamma_a1_group(SUPERCELL)
    return ProjectedSingletAGPConfig(
        n_sites=N_SITES,
        group_images=tuple(tuple(int(site) for site in image) for image in group),
        parameter_mode="real",
    )


def projected_paired_z2_n36_batch(
    configurations: np.ndarray,
    kernel: PairedZ2Kernel,
    *,
    microbatch_size: int = 16,
) -> ProjectedPairedBatch:
    """Evaluate one full-432-image paired carrier on physical spins."""

    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[1] != N_SITES:
        raise ValueError("configurations must have shape (batch, 36)")
    first, second = np.triu_indices(N_SITES, k=1)
    parameters = {"pairing_real": np.asarray(kernel.pairing[first, second])}
    batch = projected_singlet_agp_amplitudes(
        parameters,
        states,
        _projected_config(),
        microbatch_size=microbatch_size,
    )
    return ProjectedPairedBatch(
        amplitudes=np.asarray(batch.amplitudes, dtype=np.complex128),
        cancellation_ratios=np.asarray(batch.cancellation_ratios, dtype=np.float64),
        exact_nodes=np.asarray(batch.exact_nodes, dtype=bool),
    )


def _importance_row(
    carrier: np.ndarray, proposal: np.ndarray, cancellation: np.ndarray
) -> dict[str, float | int]:
    if np.any(np.abs(proposal) <= 0.0):
        raise FloatingPointError("the sampled proposal contains an exact node")
    ratios = np.asarray(carrier) / np.asarray(proposal)
    weights = np.abs(ratios) ** 2
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("paired carrier has zero or nonfinite proposal weight")
    normalized = weights / total
    finite_cancellation = cancellation[cancellation > 0.0]
    ratio_rms = float(np.sqrt(np.mean(np.abs(ratios) ** 2)))
    ratio_standard_deviation = float(
        np.sqrt(np.mean(np.abs(ratios - np.mean(ratios)) ** 2))
    )
    return {
        "sample_count": len(carrier),
        "exact_node_count": int(np.count_nonzero(np.abs(carrier) == 0.0)),
        "exact_node_fraction": float(np.mean(np.abs(carrier) == 0.0)),
        "importance_ess_fraction": float(
            1.0 / np.sum(normalized**2) / len(normalized)
        ),
        "maximum_normalized_importance_weight": float(np.max(normalized)),
        "median_projection_cancellation_ratio": float(np.median(cancellation)),
        "minimum_nonzero_projection_cancellation_ratio": (
            float(np.min(finite_cancellation)) if len(finite_cancellation) else 0.0
        ),
        "ratio_complex_standard_deviation": ratio_standard_deviation,
        "ratio_relative_standard_deviation": ratio_standard_deviation / ratio_rms,
    }


def _sample_gram(
    proposal: np.ndarray, paired: np.ndarray
) -> dict[str, object]:
    ratios = np.column_stack(
        (np.ones(len(proposal), dtype=np.complex128), paired / proposal[:, None])
    )
    scales = np.sqrt(np.mean(np.abs(ratios) ** 2, axis=0))
    normalized = ratios / scales[None, :]
    gram = normalized.conj().T @ normalized / len(normalized)
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.conj().T))
    threshold = max(float(eigenvalues[-1]) * 1.0e-8, 1.0e-12)
    positive = eigenvalues[eigenvalues > threshold]
    return {
        "normalized_overlap_matrix_real": gram.real.tolist(),
        "normalized_overlap_matrix_imag": gram.imag.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "sample_rank_at_1e-8": int(len(positive)),
        "retained_condition_number": (
            float(positive[-1] / positive[0]) if len(positive) else float("inf")
        ),
    }


def _orthogonal_fraction(direction: np.ndarray, reference: np.ndarray) -> float:
    left = np.asarray(direction, dtype=np.complex128)
    right = np.asarray(reference, dtype=np.complex128)
    left = left - np.mean(left)
    right = right - np.mean(right)
    left_norm = float(np.vdot(left, left).real)
    right_norm = float(np.vdot(right, right).real)
    if left_norm <= 1.0e-18 or right_norm <= 1.0e-18:
        return 0.0
    coefficient = np.vdot(right, left) / right_norm
    residual = left - coefficient * right
    return float(np.vdot(residual, residual).real / left_norm)


def triangular_n36_flux_loop_preflight(
    *,
    seed: int,
    chain_count: int,
    burn_sweeps: int,
    measurement_blocks: int,
    decorrelation_sweeps: int,
    action_center_count: int,
    amplitude_microbatch_size: int,
    center_batch_size: int,
    gates: dict[str, float | int],
) -> dict[str, object]:
    """Run fresh sampling, carrier-support, and dimer-action diagnostics."""

    if action_center_count > chain_count * measurement_blocks:
        raise ValueError("action_center_count exceeds the sampled support")
    orbit = build_n36_pi_zero_flux_orbit()

    def proposal(states: np.ndarray) -> np.ndarray:
        return projected_pi_zero_flux_a1_amplitudes(
            states, orbit=orbit, microbatch_size=amplitude_microbatch_size
        ).amplitudes

    sampling_started = time.perf_counter()
    samples = sample_fixed_wavefunction_exchange(
        proposal,
        n_sites=N_SITES,
        n_up=N_SITES // 2,
        chain_count=chain_count,
        burn_sweeps=burn_sweeps,
        measurement_blocks=measurement_blocks,
        decorrelation_sweeps=decorrelation_sweeps,
        seed=int(seed),
    )
    sampling_seconds = time.perf_counter() - sampling_started
    states = np.asarray(samples.configurations, dtype=np.int8)
    pi_amplitudes = np.asarray(samples.amplitudes, dtype=np.complex128)
    group = triangular_hnf_gamma_a1_group(SUPERCELL)
    symmetry_image = np.asarray(group[-1], dtype=np.int64)

    kernels = paired_z2_kernel_bank("triangular_n36")
    carrier_columns = []
    carrier_rows = []
    full_projection_seconds = 0.0
    for kernel in kernels:
        started = time.perf_counter()
        batch = projected_paired_z2_n36_batch(
            states, kernel, microbatch_size=amplitude_microbatch_size
        )
        transformed = projected_paired_z2_n36_batch(
            states[: min(4, len(states)), symmetry_image],
            kernel,
            microbatch_size=amplitude_microbatch_size,
        )
        reference = batch.amplitudes[: len(transformed.amplitudes)]
        full_projection_seconds += time.perf_counter() - started
        row = _importance_row(
            batch.amplitudes, pi_amplitudes, batch.cancellation_ratios
        )
        row.update(
            {
                "label": kernel.label,
                "quasiparticle_gap": kernel.quasiparticle_gap,
                "u_condition_number": kernel.u_condition_number,
                "continuous_igg_nullity": kernel.continuous_igg_nullity,
                "gamma_a1_transformed_amplitude_relative_residual": float(
                    np.linalg.norm(transformed.amplitudes - reference)
                    / max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
                ),
            }
        )
        carrier_columns.append(batch.amplitudes)
        carrier_rows.append(row)
    paired = np.column_stack(carrier_columns)
    gram = _sample_gram(pi_amplitudes, paired)

    # The first measurement block contains one row from every independent
    # chain.  Restricting the action canary to it avoids pseudo-replication.
    center_count = min(action_center_count, chain_count)
    center_states = states[:center_count]
    action_rows = []
    action_seconds = 0.0
    labels = ("J1_shell", "J2_shell", DIMER_LABEL)
    for kernel in kernels:
        def amplitude(configurations: np.ndarray, *, _kernel=kernel) -> np.ndarray:
            return projected_paired_z2_n36_batch(
                configurations,
                _kernel,
                microbatch_size=amplitude_microbatch_size,
            ).amplitudes

        centers = amplitude(center_states)
        if np.any(np.abs(centers) <= 0.0):
            raise FloatingPointError("an action center is an exact paired-carrier node")
        started = time.perf_counter()
        actions, ledger = _selected_operator_actions(
            amplitude,
            center_states,
            labels,
            center_batch_size=center_batch_size,
            amplitude_batch_size=amplitude_microbatch_size,
        )
        elapsed = time.perf_counter() - started
        action_seconds += elapsed
        ratios = actions / centers[:, None]
        hamiltonian = ratios[:, 0] + J2_OVER_J1 * ratios[:, 1]
        dimer = ratios[:, 2]
        action_rows.append(
            {
                "label": kernel.label,
                "center_count": center_count,
                "wall_seconds": elapsed,
                "dimer_ratio_mean_real": float(np.mean(dimer).real),
                "dimer_ratio_mean_imag": float(np.mean(dimer).imag),
                "dimer_ratio_complex_standard_deviation": float(
                    np.sqrt(np.mean(np.abs(dimer - np.mean(dimer)) ** 2))
                ),
                "hamiltonian_ratio_complex_standard_deviation": float(
                    np.sqrt(
                        np.mean(
                            np.abs(hamiltonian - np.mean(hamiltonian)) ** 2
                        )
                    )
                ),
                "dimer_fraction_orthogonal_to_hamiltonian_tangent": (
                    _orthogonal_fraction(dimer, hamiltonian)
                ),
                "ledger": ledger,
            }
        )

    carrier_gate = bool(
        gram["sample_rank_at_1e-8"] >= int(gates["minimum_sample_rank"])
        and all(
            row["importance_ess_fraction"]
            >= float(gates["minimum_importance_ess_fraction"])
            and row["exact_node_fraction"]
            <= float(gates["maximum_exact_node_fraction"])
            and row["median_projection_cancellation_ratio"]
            >= float(gates["minimum_median_cancellation_ratio"])
            and row["gamma_a1_transformed_amplitude_relative_residual"]
            <= float(gates["maximum_gamma_a1_residual"])
            for row in carrier_rows
        )
    )
    action_gate = bool(
        all(
            row["dimer_ratio_complex_standard_deviation"]
            >= float(gates["minimum_dimer_ratio_standard_deviation"])
            and row["dimer_fraction_orthogonal_to_hamiltonian_tangent"]
            >= float(gates["minimum_dimer_orthogonal_fraction"])
            for row in action_rows
        )
    )
    return {
        "case": "triangular_6x6_j2_0.125",
        "n_sites": N_SITES,
        "space_group_order": len(group),
        "sample_count": len(states),
        "fresh_pi_flux_sampling": {
            "seed": int(seed),
            "chain_count": chain_count,
            "measurement_blocks": measurement_blocks,
            "acceptance_fraction": samples.acceptance_fraction,
            "burn_acceptance_fraction": samples.burn_acceptance_fraction,
            "wall_seconds": sampling_seconds,
        },
        "paired_carriers": carrier_rows,
        "sampled_carrier_bank": gram,
        "dimer_action": action_rows,
        "timing": {
            "full_projection_seconds": full_projection_seconds,
            "dimer_action_seconds": action_seconds,
            "total_measured_seconds": (
                sampling_seconds + full_projection_seconds + action_seconds
            ),
        },
        "gates": {
            "paired_carrier_support_gate_passed": carrier_gate,
            "configuration_dependent_dimer_action_gate_passed": action_gate,
            "matched_energy_pilot_licensed": carrier_gate and action_gate,
            "gpu_production_licensed": False,
        },
        "claim_boundary": (
            "Support, symmetry, action independence, and local CPU cost only; "
            "no variational energy, scaling, phase, or QSL claim."
        ),
    }


__all__ = [
    "ProjectedPairedBatch",
    "projected_paired_z2_n36_batch",
    "triangular_n36_flux_loop_preflight",
]
