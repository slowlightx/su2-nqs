#!/usr/bin/env python3
"""Exploratory cross-fitted ridge audit of an existing G1 component cache.

This script may be run on an already inspected audit set only to freeze the
selection rule and forecast a new experiment.  Its output is explicitly not a
variational claim; production must use a newly seeded untouched audit set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.remote.analyze_g1 import (  # noqa: E402
    _cached_rows,
    _matrices,
    _ratio_summary,
)
from projects.clebsch_tree_nqs.workflows.spin_g1_operator_vit import (  # noqa: E402
    _effective_contract,
    _paired_chain_difference,
)
from su2_nqs.benchmarks.crossfit_regularized_operator import (  # noqa: E402
    crossfit_select_ridge,
    krylov_control_fit,
    regularized_generalized_ritz,
)
from su2_nqs.benchmarks.operator_vit_sampled import (  # noqa: E402
    carrier_standardization_for_labels,
    centered_component_coefficients_to_raw_words,
    centered_scalar_component_matrices,
    krylov_control_basis,
    scalar_component_words,
)
from su2_nqs.benchmarks.qsl_three_arm import (  # noqa: E402
    _energy_variance,
    triangular_pi_flux_operator_components,
)
from su2_nqs.frontier.artifacts import atomic_json  # noqa: E402
from su2_nqs.operators.triangular_a1_scalar import (  # noqa: E402
    TRIANGULAR_A1_SCALAR_LABELS,
)


RIDGE_GRID = (
    0.0,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
    100.0,
    1.0e4,
)


def _exact_small_capacity(coupling: float) -> dict[str, object]:
    """Enumerate the same nested shell/ring spaces on triangular 4x4."""

    data = triangular_pi_flux_operator_components(coupling)
    output: dict[str, object] = {}
    for name, token_count in (("shell_resolved", 2), ("shell_plus_ring", 3)):
        selected = [0]
        selected.extend(1 + index for index in range(token_count))
        selected.extend(
            1 + 5 + 5 * outer + inner
            for outer in range(token_count)
            for inner in range(token_count)
        )
        components = np.asarray(data.component_matrix[:, selected])
        action = np.asarray(data.hamiltonian @ components)
        weights = np.zeros((token_count,), dtype=np.float64)
        weights[:2] = (1.0, coupling)
        control_basis = krylov_control_basis(
            hamiltonian_weights=weights,
            means=np.asarray(data.centering_means[:token_count]),
            scales=np.asarray(data.response_scales[:token_count]),
        )
        control = krylov_control_fit(
            components, action, control_basis=control_basis
        )
        candidate = regularized_generalized_ritz(
            components, action, ridge=0.0, control_basis=control_basis
        )
        control_energy, control_variance = _energy_variance(
            components @ control.coefficients, data.hamiltonian
        )
        candidate_energy, candidate_variance = _energy_variance(
            components @ candidate.coefficients, data.hamiltonian
        )
        output[name] = {
            "component_count": components.shape[1],
            "two_lanczos_energy_per_site": control_energy / 16.0,
            "two_lanczos_variance_total": control_variance,
            "full_space_energy_per_site": candidate_energy / 16.0,
            "full_space_variance_total": candidate_variance,
            "gain_per_site": (candidate_energy - control_energy) / 16.0,
            "non_krylov_norm_fraction": candidate.resonance_norm_squared,
        }
    return output


def _exact_small_transferred_polynomial(coupling: float) -> dict[tuple[str, ...], float]:
    """Return the N=16-selected physical three-token polynomial."""

    data = triangular_pi_flux_operator_components(coupling)
    token_count = 3
    selected = [0]
    selected.extend(1 + index for index in range(token_count))
    selected.extend(
        1 + 5 + 5 * outer + inner
        for outer in range(token_count)
        for inner in range(token_count)
    )
    components = np.asarray(data.component_matrix[:, selected])
    action = np.asarray(data.hamiltonian @ components)
    control_basis = krylov_control_basis(
        hamiltonian_weights=(1.0, coupling, 0.0),
        means=np.asarray(data.centering_means[:token_count]),
        scales=np.asarray(data.response_scales[:token_count]),
    )
    fit = regularized_generalized_ritz(
        components, action, ridge=0.0, control_basis=control_basis
    )
    return centered_component_coefficients_to_raw_words(
        fit.coefficients,
        labels=TRIANGULAR_A1_SCALAR_LABELS[:token_count],
        means=np.asarray(data.centering_means[:token_count]),
        scales=np.asarray(data.response_scales[:token_count]),
    )


def analyze(root: Path) -> dict[str, object]:
    source = json.loads((root / "results.json").read_text())
    if source.get("status") != "complete":
        raise ValueError("G1 result must be complete")
    spec = source["config"]
    expansions, _, effective_words, base_words = _effective_contract(spec)
    counts = source["sample_identity"]
    train_raw = _cached_rows(root, "train", int(counts["train_count"]))
    audit_raw = _cached_rows(root, "audit", int(counts["audit_count"]))
    means = np.asarray(source["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(source["standardization"]["scales"], dtype=np.float64)
    train_components, train_h = _matrices(
        train_raw,
        base_words=base_words,
        effective_words=effective_words,
        expansions=expansions,
        means=means,
        scales=scales,
    )
    audit_components, audit_h = _matrices(
        audit_raw,
        base_words=base_words,
        effective_words=effective_words,
        expansions=expansions,
        means=means,
        scales=scales,
    )
    chain_count = int(spec["sampling"]["chain_count"])
    fit, crossfit = crossfit_select_ridge(
        train_components,
        train_h,
        chain_count=chain_count,
        fold_count=4,
        ridge_grid=RIDGE_GRID,
    )
    control = krylov_control_fit(train_components, train_h)
    trial_state, trial_action = (
        audit_components @ fit.coefficients,
        audit_h @ fit.coefficients,
    )
    control_state, control_action = (
        audit_components @ control.coefficients,
        audit_h @ control.coefficients,
    )
    trial = _ratio_summary(
        trial_state, trial_action, chain_count=chain_count, n_sites=36
    )
    baseline = _ratio_summary(
        control_state, control_action, chain_count=chain_count, n_sites=36
    )
    paired = _paired_chain_difference(
        trial_state,
        trial_action,
        control_state,
        control_action,
        chain_count=chain_count,
    )
    shell_labels = TRIANGULAR_A1_SCALAR_LABELS[:2]
    shell_components_words = scalar_component_words(shell_labels)
    shell_words = tuple(
        dict.fromkeys(
            (
                *shell_components_words,
                *(
                    (label, *word)
                    for label in shell_labels
                    for word in shell_components_words
                ),
            )
        )
    )
    base_positions = {word: index for index, word in enumerate(base_words)}
    if any(word not in base_positions for word in shell_words):
        raise ValueError("source cache does not contain the shell-resolved word basis")
    train_shell_raw = train_raw[
        :, [base_positions[word] for word in shell_words]
    ]
    audit_shell_raw = audit_raw[
        :, [base_positions[word] for word in shell_words]
    ]
    shell_means, shell_scales, _ = carrier_standardization_for_labels(
        train_shell_raw, shell_words, shell_labels
    )
    shell_h_weights = (1.0, float(spec["j2_over_j1"]))
    train_shell_components, train_shell_h = centered_scalar_component_matrices(
        train_shell_raw,
        shell_words,
        labels=shell_labels,
        hamiltonian_weights=shell_h_weights,
        means=shell_means,
        scales=shell_scales,
    )
    audit_shell_components, audit_shell_h = centered_scalar_component_matrices(
        audit_shell_raw,
        shell_words,
        labels=shell_labels,
        hamiltonian_weights=shell_h_weights,
        means=shell_means,
        scales=shell_scales,
    )
    train_shell_centers = train_shell_raw[:, shell_words.index(())]
    audit_shell_centers = audit_shell_raw[:, shell_words.index(())]
    train_shell_components /= train_shell_centers[:, None]
    train_shell_h /= train_shell_centers[:, None]
    audit_shell_components /= audit_shell_centers[:, None]
    audit_shell_h /= audit_shell_centers[:, None]
    shell_control_basis = krylov_control_basis(
        hamiltonian_weights=shell_h_weights,
        means=shell_means,
        scales=shell_scales,
    )
    shell_fit, shell_crossfit = crossfit_select_ridge(
        train_shell_components,
        train_shell_h,
        chain_count=chain_count,
        fold_count=4,
        ridge_grid=RIDGE_GRID,
        control_basis=shell_control_basis,
    )
    shell_control = krylov_control_fit(
        train_shell_components,
        train_shell_h,
        control_basis=shell_control_basis,
    )
    shell_trial_state = audit_shell_components @ shell_fit.coefficients
    shell_trial_action = audit_shell_h @ shell_fit.coefficients
    shell_control_state = audit_shell_components @ shell_control.coefficients
    shell_control_action = audit_shell_h @ shell_control.coefficients
    shell_trial = _ratio_summary(
        shell_trial_state,
        shell_trial_action,
        chain_count=chain_count,
        n_sites=36,
    )
    shell_baseline = _ratio_summary(
        shell_control_state,
        shell_control_action,
        chain_count=chain_count,
        n_sites=36,
    )
    shell_paired = _paired_chain_difference(
        shell_trial_state,
        shell_trial_action,
        shell_control_state,
        shell_control_action,
        chain_count=chain_count,
    )
    pooled_shell_components = np.concatenate(
        (train_shell_components, audit_shell_components), axis=0
    )
    pooled_shell_h = np.concatenate((train_shell_h, audit_shell_h), axis=0)
    pooled_shell_fit, pooled_shell_crossfit = crossfit_select_ridge(
        pooled_shell_components,
        pooled_shell_h,
        chain_count=2 * chain_count,
        fold_count=8,
        ridge_grid=RIDGE_GRID,
        control_basis=shell_control_basis,
    )
    shell_audit_grid = []
    for ridge in RIDGE_GRID:
        ridge_fit = regularized_generalized_ritz(
            train_shell_components,
            train_shell_h,
            ridge=ridge,
            control_basis=shell_control_basis,
        )
        ridge_state = audit_shell_components @ ridge_fit.coefficients
        ridge_action = audit_shell_h @ ridge_fit.coefficients
        ridge_paired = _paired_chain_difference(
            ridge_state,
            ridge_action,
            shell_control_state,
            shell_control_action,
            chain_count=chain_count,
        )
        shell_audit_grid.append(
            {
                "ridge": ridge,
                "difference_per_site": ridge_paired["difference_per_site"],
                "standard_error_per_site": ridge_paired[
                    "chain_standard_error_per_site"
                ],
                "non_krylov_norm_fraction": ridge_fit.resonance_norm_squared,
            }
        )
    # A minimally enlarged physical family: resolve the J1/J2 shells and add
    # one A1 rhombus-resonance token.  The Hamiltonian still has zero ring
    # weight, so the same I,H,H^2 control is exactly nested in the 13 columns.
    three_labels = TRIANGULAR_A1_SCALAR_LABELS[:3]
    three_component_words = scalar_component_words(three_labels)
    three_words = tuple(
        dict.fromkeys(
            (
                *three_component_words,
                *(
                    (label, *word)
                    for label in shell_labels
                    for word in three_component_words
                ),
            )
        )
    )
    if any(word not in base_positions for word in three_words):
        raise ValueError("source cache does not contain the three-token word basis")
    three_positions = [base_positions[word] for word in three_words]
    train_three_raw = train_raw[:, three_positions]
    audit_three_raw = audit_raw[:, three_positions]
    three_means, three_scales, _ = carrier_standardization_for_labels(
        train_three_raw, three_words, three_labels
    )
    three_h_weights = (1.0, float(spec["j2_over_j1"]), 0.0)
    train_three_components, train_three_h = centered_scalar_component_matrices(
        train_three_raw,
        three_words,
        labels=three_labels,
        hamiltonian_weights=three_h_weights,
        means=three_means,
        scales=three_scales,
    )
    audit_three_components, audit_three_h = centered_scalar_component_matrices(
        audit_three_raw,
        three_words,
        labels=three_labels,
        hamiltonian_weights=three_h_weights,
        means=three_means,
        scales=three_scales,
    )
    train_three_centers = train_three_raw[:, three_words.index(())]
    audit_three_centers = audit_three_raw[:, three_words.index(())]
    train_three_components /= train_three_centers[:, None]
    train_three_h /= train_three_centers[:, None]
    audit_three_components /= audit_three_centers[:, None]
    audit_three_h /= audit_three_centers[:, None]
    three_control_basis = krylov_control_basis(
        hamiltonian_weights=three_h_weights,
        means=three_means,
        scales=three_scales,
    )
    three_fit, three_crossfit = crossfit_select_ridge(
        train_three_components,
        train_three_h,
        chain_count=chain_count,
        fold_count=4,
        ridge_grid=RIDGE_GRID,
        control_basis=three_control_basis,
    )
    three_control = krylov_control_fit(
        train_three_components,
        train_three_h,
        control_basis=three_control_basis,
    )
    three_trial_state = audit_three_components @ three_fit.coefficients
    three_trial_action = audit_three_h @ three_fit.coefficients
    three_control_state = audit_three_components @ three_control.coefficients
    three_control_action = audit_three_h @ three_control.coefficients
    three_trial = _ratio_summary(
        three_trial_state, three_trial_action, chain_count=chain_count, n_sites=36
    )
    three_baseline = _ratio_summary(
        three_control_state,
        three_control_action,
        chain_count=chain_count,
        n_sites=36,
    )
    three_paired = _paired_chain_difference(
        three_trial_state,
        three_trial_action,
        three_control_state,
        three_control_action,
        chain_count=chain_count,
    )
    pooled_three_components = np.concatenate(
        (train_three_components, audit_three_components), axis=0
    )
    pooled_three_h = np.concatenate((train_three_h, audit_three_h), axis=0)
    pooled_three_fit, pooled_three_crossfit = crossfit_select_ridge(
        pooled_three_components,
        pooled_three_h,
        chain_count=2 * chain_count,
        fold_count=8,
        ridge_grid=RIDGE_GRID,
        control_basis=three_control_basis,
    )
    three_audit_grid = []
    for ridge in RIDGE_GRID:
        ridge_fit = regularized_generalized_ritz(
            train_three_components,
            train_three_h,
            ridge=ridge,
            control_basis=three_control_basis,
        )
        ridge_paired = _paired_chain_difference(
            audit_three_components @ ridge_fit.coefficients,
            audit_three_h @ ridge_fit.coefficients,
            three_control_state,
            three_control_action,
            chain_count=chain_count,
        )
        three_audit_grid.append(
            {
                "ridge": ridge,
                "difference_per_site": ridge_paired["difference_per_site"],
                "standard_error_per_site": ridge_paired[
                    "chain_standard_error_per_site"
                ],
                "non_krylov_norm_fraction": ridge_fit.resonance_norm_squared,
            }
        )
    # Physics-selected four-dimensional transfer: freeze the optimal N=16
    # shell+ring polynomial in *raw physical operator units*, then fit only its
    # mixing with I,H,H^2 on N=36.  This removes nine data-selected nuisance
    # directions while retaining the short-range resonance mechanism.
    transferred_raw_coefficients = _exact_small_transferred_polynomial(
        float(spec["j2_over_j1"])
    )

    def transferred_vectors(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = {word: index for index, word in enumerate(three_words)}
        state = np.zeros((len(raw),), dtype=np.complex128)
        action = np.zeros((len(raw),), dtype=np.complex128)
        coupling = float(spec["j2_over_j1"])
        for word, coefficient in transferred_raw_coefficients.items():
            state += coefficient * raw[:, positions[word]]
            action += coefficient * (
                raw[:, positions[(shell_labels[0], *word)]]
                + coupling * raw[:, positions[(shell_labels[1], *word)]]
            )
        center = raw[:, positions[()]]
        return state / center, action / center

    train_transferred, train_h_transferred = transferred_vectors(train_three_raw)
    audit_transferred, audit_h_transferred = transferred_vectors(audit_three_raw)
    train_transfer_components = np.column_stack(
        (train_three_components @ three_control_basis, train_transferred)
    )
    train_transfer_h = np.column_stack(
        (train_three_h @ three_control_basis, train_h_transferred)
    )
    audit_transfer_components = np.column_stack(
        (audit_three_components @ three_control_basis, audit_transferred)
    )
    audit_transfer_h = np.column_stack(
        (audit_three_h @ three_control_basis, audit_h_transferred)
    )
    transfer_control_basis = np.zeros((4, 3), dtype=np.float64)
    transfer_control_basis[:3] = np.eye(3)
    transfer_fit, transfer_crossfit = crossfit_select_ridge(
        train_transfer_components,
        train_transfer_h,
        chain_count=chain_count,
        fold_count=4,
        ridge_grid=RIDGE_GRID,
        control_basis=transfer_control_basis,
    )
    transfer_control = krylov_control_fit(
        train_transfer_components,
        train_transfer_h,
        control_basis=transfer_control_basis,
    )
    transfer_trial_state = audit_transfer_components @ transfer_fit.coefficients
    transfer_trial_action = audit_transfer_h @ transfer_fit.coefficients
    transfer_control_state = audit_transfer_components @ transfer_control.coefficients
    transfer_control_action = audit_transfer_h @ transfer_control.coefficients
    transfer_trial = _ratio_summary(
        transfer_trial_state,
        transfer_trial_action,
        chain_count=chain_count,
        n_sites=36,
    )
    transfer_baseline = _ratio_summary(
        transfer_control_state,
        transfer_control_action,
        chain_count=chain_count,
        n_sites=36,
    )
    transfer_paired = _paired_chain_difference(
        transfer_trial_state,
        transfer_trial_action,
        transfer_control_state,
        transfer_control_action,
        chain_count=chain_count,
    )
    return {
        "schema_version": 1,
        "task": "spin_g1_crossfit_regularized_exploration",
        "source_result_sha256": source["result_sha256"],
        "claim_status": "exploration_only_audit_previously_inspected",
        "ridge_grid": list(RIDGE_GRID),
        "exact_small_capacity": _exact_small_capacity(
            float(spec["j2_over_j1"])
        ),
        "crossfit": crossfit,
        "selected_fit": {
            "coefficients": fit.coefficients.tolist(),
            "training_energy_per_site": fit.training_energy / 36.0,
            "resonance_norm_fraction": fit.resonance_norm_squared,
        },
        "audit_regularized_operator": trial,
        "audit_two_lanczos": baseline,
        "paired_regularized_minus_two_lanczos": paired,
        "shell_resolved_crossfit": {
            "means": shell_means.tolist(),
            "scales": shell_scales.tolist(),
            "control_basis": shell_control_basis.tolist(),
            "crossfit": shell_crossfit,
            "selected_fit": {
                "coefficients": shell_fit.coefficients.tolist(),
                "training_energy_per_site": shell_fit.training_energy / 36.0,
                "non_krylov_norm_fraction": shell_fit.resonance_norm_squared,
            },
            "audit_regularized_operator": shell_trial,
            "audit_two_lanczos": shell_baseline,
            "paired_regularized_minus_two_lanczos": shell_paired,
            "pooled_previously_inspected_crossfit": pooled_shell_crossfit,
            "pooled_selected_non_krylov_norm_fraction": (
                pooled_shell_fit.resonance_norm_squared
            ),
            "audit_grid_post_selection_diagnostic_only": shell_audit_grid,
        },
        "shell_plus_ring_crossfit": {
            "labels": list(three_labels),
            "means": three_means.tolist(),
            "scales": three_scales.tolist(),
            "control_basis": three_control_basis.tolist(),
            "crossfit": three_crossfit,
            "selected_fit": {
                "coefficients": three_fit.coefficients.tolist(),
                "training_energy_per_site": three_fit.training_energy / 36.0,
                "non_krylov_norm_fraction": three_fit.resonance_norm_squared,
            },
            "audit_regularized_operator": three_trial,
            "audit_two_lanczos": three_baseline,
            "paired_regularized_minus_two_lanczos": three_paired,
            "pooled_previously_inspected_crossfit": pooled_three_crossfit,
            "pooled_selected_non_krylov_norm_fraction": (
                pooled_three_fit.resonance_norm_squared
            ),
            "audit_grid_post_selection_diagnostic_only": three_audit_grid,
        },
        "n16_physical_polynomial_transfer": {
            "dimension": 4,
            "raw_word_coefficients": {
                "/".join(word) if word else "I": value
                for word, value in transferred_raw_coefficients.items()
            },
            "crossfit": transfer_crossfit,
            "selected_fit": {
                "coefficients": transfer_fit.coefficients.tolist(),
                "non_krylov_norm_fraction": transfer_fit.resonance_norm_squared,
            },
            "audit_regularized_transfer": transfer_trial,
            "audit_two_lanczos": transfer_baseline,
            "paired_transfer_minus_two_lanczos": transfer_paired,
        },
        "decision_forecast": {
            "point_energy_below_control": paired["difference_per_site"] < 0.0,
            "resolved_at_2se": paired["difference_per_site"]
            + 2.0 * paired["chain_standard_error_per_site"]
            < 0.0,
            "variance_lower": trial["variance_total"] < baseline["variance_total"],
            "new_untouched_audit_licensed": False,
            "shell_resolved_point_energy_below_control": shell_paired[
                "difference_per_site"
            ]
            < 0.0,
            "shell_resolved_at_2se": shell_paired["difference_per_site"]
            + 2.0 * shell_paired["chain_standard_error_per_site"]
            < 0.0,
            "shell_resolved_variance_lower": shell_trial["variance_total"]
            < shell_baseline["variance_total"],
            "shell_plus_ring_point_energy_below_control": three_paired[
                "difference_per_site"
            ]
            < 0.0,
            "shell_plus_ring_at_2se": three_paired["difference_per_site"]
            + 2.0 * three_paired["chain_standard_error_per_site"]
            < 0.0,
            "shell_plus_ring_variance_lower": three_trial["variance_total"]
            < three_baseline["variance_total"],
            "n16_transfer_point_energy_below_control": transfer_paired[
                "difference_per_site"
            ]
            < 0.0,
            "n16_transfer_at_2se": transfer_paired["difference_per_site"]
            + 2.0 * transfer_paired["chain_standard_error_per_site"]
            < 0.0,
            "n16_transfer_variance_lower": transfer_trial["variance_total"]
            < transfer_baseline["variance_total"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = analyze(args.result_root)
    atomic_json(args.output, payload)
    print(json.dumps(payload["decision_forecast"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
