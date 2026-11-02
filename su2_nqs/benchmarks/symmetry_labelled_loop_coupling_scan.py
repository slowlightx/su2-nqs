"""Coupling-resolved virtual-loop response on exact-SU(2) Operator-ViT states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .qsl_three_arm import (
    _operator_vit_trials,
    triangular_pi_flux_operator_components,
)
from .symmetry_labelled_loop_resonance import (
    LabelledLoopOperators,
    _covariance_diagnostics,
    _labelled_return_columns,
    _residual_capture,
    _ritz_row,
    triangular_labelled_loop_operators,
)


COUPLINGS = (0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175)
FRESH_SEEDS = (1411, 1412, 1413)
HELDOUT_QSL_COUPLINGS = (0.075, 0.1, 0.15)
HELDOUT_ORDERED_COUPLINGS = (0.0, 0.05, 0.175)


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state must have finite nonzero norm")
    return value / norm


def _moment_row(
    state: np.ndarray,
    components: Sequence[np.ndarray],
    hamiltonian,
) -> dict[str, float]:
    """Return equal-time weight, SMA gap, and susceptibility proxy."""

    vector = _normalize(state)
    energy = float(np.vdot(vector, hamiltonian @ vector).real)
    images = tuple(np.asarray(component) for component in components)
    m0 = float(sum(np.vdot(image, image).real for image in images))
    if m0 <= 1.0e-14:
        raise RuntimeError("a labelled response has zero equal-time weight")
    m1 = float(
        sum(
            np.vdot(image, hamiltonian @ image).real - energy * np.vdot(image, image).real
            for image in images
        )
    )
    gap = m1 / m0
    return {
        "m0_total": m0,
        "m0_per_site": m0 / 16.0,
        "m1_total": m1,
        "sma_gap": gap,
        "sma_susceptibility_proxy_per_site": (
            float(2.0 * m0 / gap / 16.0) if gap > 0.0 else float("inf")
        ),
    }


def _response_metrics(
    state: np.ndarray,
    operators: LabelledLoopOperators,
    hamiltonian,
) -> dict[str, object]:
    """Return channel moments, residual capture, and matched Ritz tests."""

    vector = _normalize(state)
    first_order = {
        "E2_rhombus_ring": tuple(operator @ vector for operator in operators.e2_ring),
        "E2_rhombus_dimer": tuple(operator @ vector for operator in operators.e2_dimer),
        "A2_directed_ring": (operators.a2_current @ vector,),
    }
    returns = dict(
        zip(
            operators.labels,
            _labelled_return_columns(vector, operators),
            strict=True,
        )
    )
    moments = {
        label: _moment_row(vector, images, hamiltonian)
        for label, images in first_order.items()
    }
    captures = {
        label: _residual_capture(vector, (returns[return_label],), hamiltonian)
        for label, return_label in zip(
            first_order,
            operators.labels,
            strict=True,
        )
    }
    a1_directions = tuple(operator @ vector for operator in operators.a1_resonance)
    captures["combined_E2_A2"] = _residual_capture(
        vector, tuple(returns.values()), hamiltonian
    )
    captures["existing_A1_loops"] = _residual_capture(
        vector, a1_directions, hamiltonian
    )
    h_state = np.asarray(hamiltonian @ vector)
    h2_state = np.asarray(hamiltonian @ h_state)
    h3_state = np.asarray(hamiltonian @ h2_state)
    h4_state = np.asarray(hamiltonian @ h3_state)
    ritz = {
        "one_lanczos": _ritz_row(
            "one_lanczos", (vector, h_state), hamiltonian
        ),
        "A2_return": _ritz_row(
            "A2_return",
            (vector, h_state, returns["A2_directed_ring_return"]),
            hamiltonian,
        ),
        "matched_degree2_krylov": _ritz_row(
            "matched_degree2_krylov", (vector, h_state, h2_state), hamiltonian
        ),
        "labelled_E2_A2_bank": _ritz_row(
            "labelled_E2_A2_bank",
            (vector, h_state, *returns.values()),
            hamiltonian,
        ),
        "matched_degree4_krylov": _ritz_row(
            "matched_degree4_krylov",
            (vector, h_state, h2_state, h3_state, h4_state),
            hamiltonian,
        ),
    }
    static_a2 = float(
        abs(np.vdot(vector, first_order["A2_directed_ring"][0]))
    )
    return {
        "moments": moments,
        "residual_capture": captures,
        "ritz": ritz,
        "static_A2_expectation_abs": static_a2,
    }


def _mean(rows: Sequence[dict[str, object]], getter) -> float:
    return float(np.mean([float(getter(row)) for row in rows]))


def symmetry_labelled_loop_coupling_scan(
    *,
    couplings: tuple[float, ...],
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    """Train fresh exact-SU2 states and test phase-selective loop response."""

    if couplings != COUPLINGS:
        raise ValueError("the loop-response coupling grid is frozen")
    if seeds != FRESH_SEEDS:
        raise ValueError("the loop-response fresh seeds are frozen")
    if steps != 160 or learning_rate != 0.015:
        raise ValueError("the loop-response training hyperparameters are frozen")

    operators = triangular_labelled_loop_operators()
    rows: list[dict[str, object]] = []
    discovery_certificate = None
    for coupling in couplings:
        data = triangular_pi_flux_operator_components(coupling)
        trials = _operator_vit_trials(
            data,
            seeds=seeds,
            steps=steps,
            learning_rate=learning_rate,
            retain_state=True,
        )
        for trial in trials:
            metadata = dict(trial)
            state = _normalize(np.asarray(metadata.pop("state")))
            response = _response_metrics(state, operators, data.hamiltonian)
            if coupling == 0.125 and int(trial["seed"]) == seeds[0]:
                discovery_certificate = _covariance_diagnostics(
                    state,
                    operators,
                    data.configurations,
                    data.integer_basis,
                )
            rows.append(
                {
                    "j2_over_j1": coupling,
                    "seed": int(trial["seed"]),
                    "operator_vit": metadata,
                    "response": response,
                }
            )
    if discovery_certificate is None:
        raise RuntimeError("the discovery-point certificate was not constructed")

    summaries = []
    for coupling in couplings:
        selected = [row for row in rows if row["j2_over_j1"] == coupling]
        summaries.append(
            {
                "j2_over_j1": coupling,
                "mean_A2_residual_capture": _mean(
                    selected,
                    lambda row: row["response"]["residual_capture"][
                        "A2_directed_ring"
                    ],
                ),
                "mean_E2_ring_residual_capture": _mean(
                    selected,
                    lambda row: row["response"]["residual_capture"][
                        "E2_rhombus_ring"
                    ],
                ),
                "mean_E2_dimer_residual_capture": _mean(
                    selected,
                    lambda row: row["response"]["residual_capture"][
                        "E2_rhombus_dimer"
                    ],
                ),
                "mean_A2_m0_per_site": _mean(
                    selected,
                    lambda row: row["response"]["moments"][
                        "A2_directed_ring"
                    ]["m0_per_site"],
                ),
                "mean_A2_sma_gap": _mean(
                    selected,
                    lambda row: row["response"]["moments"][
                        "A2_directed_ring"
                    ]["sma_gap"],
                ),
                "mean_A2_sma_susceptibility_proxy_per_site": _mean(
                    selected,
                    lambda row: row["response"]["moments"][
                        "A2_directed_ring"
                    ]["sma_susceptibility_proxy_per_site"],
                ),
                "A2_dominant_seed_count": sum(
                    row["response"]["residual_capture"]["A2_directed_ring"]
                    > max(
                        row["response"]["residual_capture"]["E2_rhombus_ring"],
                        row["response"]["residual_capture"]["E2_rhombus_dimer"],
                    )
                    for row in selected
                ),
                "A2_beats_degree2_krylov_seed_count": sum(
                    row["response"]["ritz"]["A2_return"]["energy_total"]
                    < row["response"]["ritz"]["matched_degree2_krylov"][
                        "energy_total"
                    ]
                    for row in selected
                ),
                "labelled_beats_degree4_krylov_seed_count": sum(
                    row["response"]["ritz"]["labelled_E2_A2_bank"][
                        "energy_total"
                    ]
                    < row["response"]["ritz"]["matched_degree4_krylov"][
                        "energy_total"
                    ]
                    for row in selected
                ),
            }
        )

    heldout_qsl = [
        row for row in rows if row["j2_over_j1"] in HELDOUT_QSL_COUPLINGS
    ]
    heldout_ordered = [
        row for row in rows if row["j2_over_j1"] in HELDOUT_ORDERED_COUPLINGS
    ]

    def phase_mean(panel, field, subfield=None):
        if field == "capture":
            return _mean(
                panel,
                lambda row: row["response"]["residual_capture"][subfield],
            )
        return _mean(
            panel,
            lambda row: row["response"]["moments"]["A2_directed_ring"][field],
        )

    qsl_capture = phase_mean(heldout_qsl, "capture", "A2_directed_ring")
    ordered_capture = phase_mean(
        heldout_ordered, "capture", "A2_directed_ring"
    )
    qsl_gap = phase_mean(heldout_qsl, "sma_gap")
    ordered_gap = phase_mean(heldout_ordered, "sma_gap")
    qsl_susceptibility = phase_mean(
        heldout_qsl, "sma_susceptibility_proxy_per_site"
    )
    ordered_susceptibility = phase_mean(
        heldout_ordered, "sma_susceptibility_proxy_per_site"
    )
    heldout = heldout_qsl + heldout_ordered
    exact_errors = (
        discovery_certificate["e2_ring_covariance_max_relative_error"],
        discovery_certificate["e2_dimer_covariance_max_relative_error"],
        discovery_certificate["a2_covariance_max_relative_error"],
        discovery_certificate["returned_a1_max_relative_error"],
        discovery_certificate["maximum_splus_relative_norm"],
        discovery_certificate["maximum_elementary_hermiticity_error"],
        max(float(row["operator_vit"]["splus_relative_norm"]) for row in rows),
        max(float(row["response"]["static_A2_expectation_abs"]) for row in rows),
    )
    response_gates = {
        "all_exact_symmetry_checks_below_1e-10": max(exact_errors) < 1.0e-10,
        "fresh_discovery_replication_A2_dominant_three_of_three": next(
            row for row in summaries if row["j2_over_j1"] == 0.125
        )["A2_dominant_seed_count"]
        == 3,
        "heldout_A2_dominant_at_least_15_of_18": sum(
            row["response"]["residual_capture"]["A2_directed_ring"]
            > max(
                row["response"]["residual_capture"]["E2_rhombus_ring"],
                row["response"]["residual_capture"]["E2_rhombus_dimer"],
            )
            for row in heldout
        )
        >= 15,
        "heldout_qsl_capture_ratio_above_1p25": qsl_capture
        > 1.25 * ordered_capture,
        "heldout_qsl_sma_gap_ratio_below_0p90": qsl_gap < 0.90 * ordered_gap,
        "heldout_qsl_susceptibility_ratio_above_1p25": qsl_susceptibility
        > 1.25 * ordered_susceptibility,
    }
    variational_gates = {
        "heldout_A2_beats_degree2_krylov_at_least_15_of_18": sum(
            row["response"]["ritz"]["A2_return"]["energy_total"]
            < row["response"]["ritz"]["matched_degree2_krylov"]["energy_total"]
            for row in heldout
        )
        >= 15,
        "heldout_labelled_bank_beats_degree4_at_least_15_of_18": sum(
            row["response"]["ritz"]["labelled_E2_A2_bank"]["energy_total"]
            < row["response"]["ritz"]["matched_degree4_krylov"]["energy_total"]
            for row in heldout
        )
        >= 15,
    }
    return {
        "couplings": list(couplings),
        "fresh_seeds": list(seeds),
        "steps": steps,
        "learning_rate": learning_rate,
        "discovery_coupling": 0.125,
        "heldout_qsl_couplings": list(HELDOUT_QSL_COUPLINGS),
        "heldout_ordered_couplings": list(HELDOUT_ORDERED_COUPLINGS),
        "rows": rows,
        "coupling_summary": summaries,
        "discovery_certificate": discovery_certificate,
        "phase_contrast": {
            "qsl_mean_A2_residual_capture": qsl_capture,
            "ordered_mean_A2_residual_capture": ordered_capture,
            "qsl_to_ordered_capture_ratio": qsl_capture / ordered_capture,
            "qsl_mean_A2_sma_gap": qsl_gap,
            "ordered_mean_A2_sma_gap": ordered_gap,
            "qsl_to_ordered_sma_gap_ratio": qsl_gap / ordered_gap,
            "qsl_mean_A2_sma_susceptibility_proxy_per_site": qsl_susceptibility,
            "ordered_mean_A2_sma_susceptibility_proxy_per_site": ordered_susceptibility,
            "qsl_to_ordered_susceptibility_ratio": qsl_susceptibility
            / ordered_susceptibility,
        },
        "response_gates": response_gates,
        "response_signal_passed": bool(all(response_gates.values())),
        "variational_gates": variational_gates,
        "variational_promotion_passed": bool(all(variational_gates.values())),
        "claim_boundary": (
            "J2/J1=0.125 generated the A2 hypothesis. Phase-selectivity gates "
            "use only the six other couplings and three fresh seeds. m0 is an "
            "equal-time fluctuation and 2m0/Delta_SMA a response proxy, not an "
            "exact thermodynamic susceptibility. No large-system ED is used."
        ),
    }


__all__ = [
    "COUPLINGS",
    "FRESH_SEEDS",
    "HELDOUT_ORDERED_COUPLINGS",
    "HELDOUT_QSL_COUPLINGS",
    "symmetry_labelled_loop_coupling_scan",
]
