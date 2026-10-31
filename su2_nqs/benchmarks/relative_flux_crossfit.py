"""Cross-fitted relative-flux response gate on the triangular 6x6 torus.

Four exact ``S=0, Gamma-A1`` carriers are compared in the common physical
spin representation: the projected pi/0-flux determinant and three projected
paired-Z2 states.  Development chains alone fix a well-conditioned set of
collective/relative carrier modes.  Independent audit chains then test
whether the dimer-resonance response contains a reproducible direction beyond
the carrier plus one-Hamiltonian tangent space (and ``H^2|pi>``).

This is deliberately a representation/capacity gate, not an energy result.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import time

import numpy as np

from ..models.projected_singlet_agp import projected_singlet_agp_amplitudes
from ..vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from .flux_loop_n36_preflight import (
    DIMER_LABEL,
    J2_OVER_J1,
    N_SITES,
    SUPERCELL,
    _projected_config,
    _selected_operator_actions,
    projected_paired_z2_n36_batch,
)
from .operator_tangent_response import triangular_a1_word_actions
from .paired_z2_carrier import PairedZ2Kernel, paired_z2_kernel_bank
from .triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes,
)


SHELL_LABELS = ("J1_shell", "J2_shell")
ACTION_LABELS = (*SHELL_LABELS, DIMER_LABEL)
H2_WORDS = (
    ("J1_shell", "J1_shell"),
    ("J1_shell", "J2_shell"),
    ("J2_shell", "J1_shell"),
    ("J2_shell", "J2_shell"),
)


@dataclass(frozen=True)
class RelativeFluxBasis:
    """Development-frozen carrier normalization and spectral transform."""

    scales: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    retained_count: int


@dataclass(frozen=True)
class SampledResponse:
    """Common-proposal carrier and scalar-action ratios for one split."""

    carrier_ratios: np.ndarray
    hamiltonian_action_ratios: np.ndarray
    dimer_action_ratios: np.ndarray
    h2_pi_ratio: np.ndarray
    sampling: dict[str, object]
    support: list[dict[str, object]]
    action_ledgers: list[dict[str, object]]
    timing: dict[str, float]


def fit_relative_flux_basis(
    carrier_ratios: np.ndarray,
    *,
    relative_eigenvalue_cutoff: float,
) -> RelativeFluxBasis:
    """Fit a deterministic, column-scale-invariant carrier basis."""

    ratios = np.asarray(carrier_ratios, dtype=np.complex128)
    if ratios.ndim != 2 or ratios.shape[0] < ratios.shape[1]:
        raise ValueError("carrier_ratios must be a tall rank-two matrix")
    if not 0.0 < relative_eigenvalue_cutoff < 1.0:
        raise ValueError("relative_eigenvalue_cutoff must lie in (0, 1)")
    scales = np.sqrt(np.mean(np.abs(ratios) ** 2, axis=0))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise FloatingPointError("carrier columns must have finite nonzero norm")
    normalized = ratios / scales[None, :]
    gram = normalized.conj().T @ normalized / len(normalized)
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order].real, dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    retained = eigenvalues >= relative_eigenvalue_cutoff * eigenvalues[0]
    retained_count = int(np.count_nonzero(retained))
    if retained_count < 1:
        raise FloatingPointError("spectral truncation removed every carrier mode")
    return RelativeFluxBasis(
        scales=scales,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        retained_count=retained_count,
    )


def apply_relative_flux_basis(
    values_relative_to_proposal: np.ndarray,
    basis: RelativeFluxBasis,
) -> np.ndarray:
    """Apply the development-frozen whitening transform to states/actions."""

    values = np.asarray(values_relative_to_proposal, dtype=np.complex128)
    if values.ndim != 2 or values.shape[1] != len(basis.scales):
        raise ValueError("values do not match the fitted carrier bank")
    count_retained = basis.retained_count
    transform = basis.eigenvectors[:, :count_retained] / np.sqrt(
        basis.eigenvalues[:count_retained]
    )[None, :]
    return (values / basis.scales[None, :]) @ transform


def _column_rms(values: np.ndarray) -> np.ndarray:
    scales = np.sqrt(np.mean(np.abs(values) ** 2, axis=0))
    if np.any(scales <= 1.0e-15) or np.any(~np.isfinite(scales)):
        raise FloatingPointError("regression contains a null/nonfinite column")
    return scales


def _fit_transfer(
    development_base: np.ndarray,
    development_target: np.ndarray,
    audit_base: np.ndarray,
    audit_target: np.ndarray,
    *,
    relative_singular_value_cutoff: float,
) -> dict[str, object]:
    """Fit on development rows and report unmodified audit residuals."""

    base_scale = _column_rms(development_base)
    target_scale = _column_rms(development_target)
    left = development_base / base_scale[None, :]
    target = development_target / target_scale[None, :]
    u, singular_values, vh = np.linalg.svd(left, full_matrices=False)
    keep = singular_values >= relative_singular_value_cutoff * singular_values[0]
    inverse = np.zeros_like(singular_values)
    inverse[keep] = 1.0 / singular_values[keep]
    coefficients = (vh.conj().T * inverse[None, :]) @ (u.conj().T @ target)

    def residual_rows(base: np.ndarray, observed: np.ndarray) -> np.ndarray:
        normalized = observed / target_scale[None, :]
        predicted = (base / base_scale[None, :]) @ coefficients
        return normalized - predicted

    development_residual = residual_rows(development_base, development_target)
    audit_residual = residual_rows(audit_base, audit_target)

    def fractions(residual: np.ndarray, observed: np.ndarray) -> tuple[list[float], float]:
        normalized = observed / target_scale[None, :]
        numerator = np.sum(np.abs(residual) ** 2, axis=0)
        denominator = np.sum(np.abs(normalized) ** 2, axis=0)
        per_column = numerator / np.maximum(denominator, np.finfo(float).tiny)
        joint = float(np.sum(numerator) / max(float(np.sum(denominator)), np.finfo(float).tiny))
        return per_column.tolist(), joint

    development_per_column, development_joint = fractions(
        development_residual, development_target
    )
    audit_per_column, audit_joint = fractions(audit_residual, audit_target)
    return {
        "base_rank": int(np.count_nonzero(keep)),
        "base_singular_values": singular_values.tolist(),
        "base_retained_condition_number": float(
            singular_values[keep][0] / singular_values[keep][-1]
        ),
        "development_residual_fractions": development_per_column,
        "development_joint_residual_fraction": development_joint,
        "audit_residual_fractions": audit_per_column,
        "audit_joint_residual_fraction": audit_joint,
        "audit_residual_matrix": audit_residual,
        "audit_target_scale": target_scale,
    }


def _scrambled_transfer_controls(
    development_base: np.ndarray,
    development_target: np.ndarray,
    audit_base: np.ndarray,
    audit_target: np.ndarray,
    *,
    relative_singular_value_cutoff: float,
    seed: int,
    count_controls: int,
) -> list[float]:
    """Break configuration/response pairing independently in both splits."""

    generator = np.random.default_rng(int(seed))
    controls = []
    for _ in range(count_controls):
        development_order = generator.permutation(len(development_target))
        audit_order = generator.permutation(len(audit_target))
        fitted = _fit_transfer(
            development_base,
            development_target[development_order],
            audit_base,
            audit_target[audit_order],
            relative_singular_value_cutoff=relative_singular_value_cutoff,
        )
        controls.append(float(fitted["audit_joint_residual_fraction"]))
    return controls


def _support_row(
    label: str,
    carrier: np.ndarray,
    proposal: np.ndarray,
    cancellation: np.ndarray | None,
) -> dict[str, object]:
    ratios = carrier / proposal
    weights = np.abs(ratios) ** 2
    normalized = weights / np.sum(weights)
    row: dict[str, object] = {
        "label": label,
        "exact_node_count": int(np.count_nonzero(np.abs(carrier) == 0.0)),
        "importance_ess_fraction": float(
            1.0 / np.sum(normalized**2) / len(normalized)
        ),
        "ratio_rms": float(np.sqrt(np.mean(np.abs(ratios) ** 2))),
    }
    if cancellation is not None:
        row["median_projection_cancellation_ratio"] = float(
            np.median(cancellation)
        )
    return row


def _sample_response_split(
    *,
    seed: int,
    chain_count: int,
    burn_sweeps: int,
    measurement_blocks: int,
    decorrelation_sweeps: int,
    action_center_count: int,
    amplitude_microbatch_size: int,
    center_batch_size: int,
) -> SampledResponse:
    orbit = build_n36_pi_zero_flux_orbit()

    def pi_amplitude(states: np.ndarray) -> np.ndarray:
        return projected_pi_zero_flux_a1_amplitudes(
            states, orbit=orbit, microbatch_size=amplitude_microbatch_size
        ).amplitudes

    started = time.perf_counter()
    sample = sample_fixed_wavefunction_exchange(
        pi_amplitude,
        n_sites=N_SITES,
        n_up=N_SITES // 2,
        chain_count=chain_count,
        burn_sweeps=burn_sweeps,
        measurement_blocks=measurement_blocks,
        decorrelation_sweeps=decorrelation_sweeps,
        seed=int(seed),
    )
    sampling_seconds = time.perf_counter() - started
    states = np.asarray(sample.configurations, dtype=np.int8)
    proposal = np.asarray(sample.amplitudes, dtype=np.complex128)
    if action_center_count > chain_count or action_center_count > len(states):
        raise ValueError("action centers must fit in the first independent-chain block")
    center_states = states[:action_center_count]
    center_proposal = proposal[:action_center_count]

    kernels = paired_z2_kernel_bank("triangular_n36")
    carrier_columns = [center_proposal]
    support = [_support_row("pi_zero_flux", proposal, proposal, None)]
    paired_functions = []
    projection_started = time.perf_counter()
    for kernel in kernels:
        full = projected_paired_z2_n36_batch(
            states, kernel, microbatch_size=amplitude_microbatch_size
        )
        carrier_columns.append(full.amplitudes[:action_center_count])
        support.append(
            _support_row(
                kernel.label,
                full.amplitudes,
                proposal,
                full.cancellation_ratios,
            )
        )

        def paired_amplitude(
            configurations: np.ndarray, *, _kernel: PairedZ2Kernel = kernel
        ) -> np.ndarray:
            return projected_paired_z2_n36_batch(
                configurations,
                _kernel,
                microbatch_size=amplitude_microbatch_size,
            ).amplitudes

        paired_functions.append(paired_amplitude)
    projection_seconds = time.perf_counter() - projection_started
    carriers = np.column_stack(carrier_columns)
    if np.any(np.abs(carriers) <= 0.0) or np.any(np.abs(center_proposal) <= 0.0):
        raise FloatingPointError("an action center is a carrier/proposal node")

    h_columns = []
    dimer_columns = []
    action_ledgers = []
    action_started = time.perf_counter()
    for label, amplitude in zip(
        ("pi_zero_flux", *(kernel.label for kernel in kernels)),
        (pi_amplitude, *paired_functions),
        strict=True,
    ):
        actions, ledger = _selected_operator_actions(
            amplitude,
            center_states,
            ACTION_LABELS,
            center_batch_size=center_batch_size,
            amplitude_batch_size=amplitude_microbatch_size,
        )
        h_columns.append(
            (actions[:, 0] + J2_OVER_J1 * actions[:, 1]) / center_proposal
        )
        dimer_columns.append(actions[:, 2] / center_proposal)
        action_ledgers.append({"label": label, **ledger})
    action_seconds = time.perf_counter() - action_started

    h2_started = time.perf_counter()
    h2_words, h2_ledger = triangular_a1_word_actions(
        pi_amplitude,
        center_states,
        SUPERCELL,
        H2_WORDS,
        amplitude_batch_size=amplitude_microbatch_size,
    )
    h2_pi = (
        h2_words[:, 0]
        + J2_OVER_J1 * (h2_words[:, 1] + h2_words[:, 2])
        + J2_OVER_J1**2 * h2_words[:, 3]
    ) / center_proposal
    h2_seconds = time.perf_counter() - h2_started
    action_ledgers.append({"label": "h2_pi_zero_flux", **h2_ledger})
    return SampledResponse(
        carrier_ratios=carriers / center_proposal[:, None],
        hamiltonian_action_ratios=np.column_stack(h_columns),
        dimer_action_ratios=np.column_stack(dimer_columns),
        h2_pi_ratio=h2_pi,
        sampling={
            "seed": int(seed),
            "chain_count": chain_count,
            "sample_count": len(states),
            "action_center_count": action_center_count,
            "acceptance_fraction": sample.acceptance_fraction,
            "burn_acceptance_fraction": sample.burn_acceptance_fraction,
        },
        support=support,
        action_ledgers=action_ledgers,
        timing={
            "sampling_seconds": sampling_seconds,
            "paired_projection_seconds": projection_seconds,
            "one_action_seconds": action_seconds,
            "h2_pi_seconds": h2_seconds,
            "total_measured_seconds": (
                sampling_seconds + projection_seconds + action_seconds + h2_seconds
            ),
        },
    )


def _audit_mode_gram(modes: np.ndarray) -> dict[str, object]:
    gram = modes.conj().T @ modes / len(modes)
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "matrix_real": gram.real.tolist(),
        "matrix_imag": gram.imag.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "identity_frobenius_deviation": float(
            np.linalg.norm(gram - np.eye(len(gram)), ord="fro")
        ),
    }


def triangular_n36_relative_flux_crossfit(
    *,
    development_seed: int,
    audit_seed: int,
    control_seed: int,
    chain_count: int,
    burn_sweeps: int,
    measurement_blocks: int,
    decorrelation_sweeps: int,
    action_center_count: int,
    amplitude_microbatch_size: int,
    center_batch_size: int,
    relative_eigenvalue_cutoff: float,
    relative_singular_value_cutoff: float,
    control_count: int,
    gates: dict[str, float | int],
) -> dict[str, object]:
    """Run the registered independent development/audit response gate."""

    common = dict(
        chain_count=chain_count,
        burn_sweeps=burn_sweeps,
        measurement_blocks=measurement_blocks,
        decorrelation_sweeps=decorrelation_sweeps,
        action_center_count=action_center_count,
        amplitude_microbatch_size=amplitude_microbatch_size,
        center_batch_size=center_batch_size,
    )
    development = _sample_response_split(seed=development_seed, **common)
    basis = fit_relative_flux_basis(
        development.carrier_ratios,
        relative_eigenvalue_cutoff=relative_eigenvalue_cutoff,
    )
    audit = _sample_response_split(seed=audit_seed, **common)

    development_modes = apply_relative_flux_basis(
        development.carrier_ratios, basis
    )
    audit_modes = apply_relative_flux_basis(audit.carrier_ratios, basis)
    development_h = apply_relative_flux_basis(
        development.hamiltonian_action_ratios, basis
    )
    audit_h = apply_relative_flux_basis(audit.hamiltonian_action_ratios, basis)
    development_dimer = apply_relative_flux_basis(
        development.dimer_action_ratios, basis
    )
    audit_dimer = apply_relative_flux_basis(audit.dimer_action_ratios, basis)

    # Mode zero is the largest-overlap collective carrier.  All remaining
    # retained modes are fixed, development-selected relative-flux channels.
    relative_slice = slice(1, basis.retained_count)
    development_target = development_dimer[:, relative_slice]
    audit_target = audit_dimer[:, relative_slice]
    development_base = np.column_stack(
        (development_modes, development_h, development.h2_pi_ratio)
    )
    audit_base = np.column_stack((audit_modes, audit_h, audit.h2_pi_ratio))
    transfer = _fit_transfer(
        development_base,
        development_target,
        audit_base,
        audit_target,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
    )
    controls = _scrambled_transfer_controls(
        development_base,
        development_target,
        audit_base,
        audit_target,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        seed=control_seed,
        count_controls=control_count,
    )
    physical_residual = float(transfer["audit_joint_residual_fraction"])
    median_control = float(np.median(controls))
    physical_to_control = physical_residual / median_control
    audit_gram = _audit_mode_gram(audit_modes)
    retained_condition = float(
        basis.eigenvalues[0] / basis.eigenvalues[basis.retained_count - 1]
    )
    modes_gate = bool(
        basis.retained_count >= int(gates["minimum_retained_mode_count"])
        and retained_condition
        <= float(gates["maximum_development_condition_number"])
        and audit_gram["condition_number"]
        <= float(gates["maximum_audit_condition_number"])
    )
    response_gate = bool(
        physical_residual
        >= float(gates["minimum_audit_nonkrylov_residual_fraction"])
        and physical_to_control
        <= float(gates["maximum_physical_to_scrambled_residual_ratio"])
    )
    pilot_licensed = modes_gate and response_gate
    transfer_public = {
        key: value
        for key, value in transfer.items()
        if key not in {"audit_residual_matrix", "audit_target_scale"}
    }
    return {
        "case": "triangular_6x6_j2_0.125_relative_flux_crossfit",
        "n_sites": N_SITES,
        "carrier_labels": [
            "pi_zero_flux",
            *(kernel.label for kernel in paired_z2_kernel_bank("triangular_n36")),
        ],
        "development": {
            "sampling": development.sampling,
            "support": development.support,
            "action_ledgers": development.action_ledgers,
            "timing": development.timing,
        },
        "audit": {
            "sampling": audit.sampling,
            "support": audit.support,
            "action_ledgers": audit.action_ledgers,
            "timing": audit.timing,
        },
        "development_frozen_basis": {
            "column_scales": basis.scales.tolist(),
            "eigenvalues_descending": basis.eigenvalues.tolist(),
            "eigenvectors_real": basis.eigenvectors.real.tolist(),
            "eigenvectors_imag": basis.eigenvectors.imag.tolist(),
            "relative_eigenvalue_cutoff": relative_eigenvalue_cutoff,
            "retained_mode_count": basis.retained_count,
            "relative_mode_count": max(0, basis.retained_count - 1),
            "retained_condition_number": retained_condition,
        },
        "audit_frozen_mode_gram": audit_gram,
        "cross_fitted_dimer_response": transfer_public,
        "scrambled_configuration_controls": {
            "seed": int(control_seed),
            "count": control_count,
            "audit_joint_residual_fractions": controls,
            "median_audit_joint_residual_fraction": median_control,
            "physical_to_median_scrambled_residual_ratio": physical_to_control,
        },
        "gates": {
            "stable_relative_flux_modes_gate_passed": modes_gate,
            "cross_fitted_nonkrylov_response_gate_passed": response_gate,
            "matched_energy_pilot_licensed": pilot_licensed,
            "gpu_production_licensed": False,
        },
        "claim_boundary": (
            "Cross-fitted sampled state-space capacity only; no optimized "
            "energy, variance, phase-boundary, scaling, or QSL claim."
        ),
    }


__all__ = [
    "RelativeFluxBasis",
    "apply_relative_flux_basis",
    "fit_relative_flux_basis",
    "triangular_n36_relative_flux_crossfit",
]
