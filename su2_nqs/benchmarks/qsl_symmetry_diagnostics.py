"""Symmetry-resolved physics diagnostics for the triangular QSL frontier.

The routines in this module deliberately act on already trained states.  They
do not change a carrier, optimizer, or variational family.  Their purpose is
to test whether an energy improvement also moves physically interpretable
spin, bond-resonance, and chirality channels toward the exact-small answer.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.sparse.linalg import eigsh

from ..lattices.ordering_pair_form_factors import ordering_pair_form_factors
from ..operators.triangular_resonance import triangular_e2_moments
from .physical_interpretation import (
    FixedSzPairActions,
    triangular_periodic_observable_panel,
    uniform_chirality_action,
)
from .qsl_three_arm import OperatorComponentData


TRIANGULAR_4X4_SUPERCELL = ((4, 0), (0, 4))


def _normalize(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError("state has zero numerical norm")
    return vector / norm


def triangular_4x4_exact_ground(
    data: OperatorComponentData,
    *,
    tolerance: float = 1.0e-13,
) -> tuple[float, np.ndarray, float]:
    """Return the fixed-``M=0`` exact ground state for the common support."""

    values, vectors = eigsh(
        data.hamiltonian,
        k=1,
        which="SA",
        tol=float(tolerance),
        maxiter=200_000,
    )
    index = int(np.argmin(values))
    energy = float(values[index])
    state = _normalize(vectors[:, index])
    residual = float(np.linalg.norm(data.hamiltonian @ state - energy * state))
    return energy, state, residual


def _spin_correlation_matrix(
    state: np.ndarray,
    pair_actions: FixedSzPairActions,
) -> np.ndarray:
    vector = _normalize(state)
    n_sites = pair_actions.n_sites
    correlations = np.eye(n_sites, dtype=np.float64) * 0.75
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            value = float(
                np.vdot(vector, pair_actions.pair(vector, first, second)).real
            )
            correlations[first, second] = correlations[second, first] = value
    return correlations


def _ordering_structure_factors(correlations: np.ndarray) -> dict[str, float]:
    names, form_factors = ordering_pair_form_factors(
        "triangular", 4, 4, channel_set="phase_competition"
    )
    return {
        name: float(np.sum(kernel * correlations) / correlations.shape[0])
        for name, kernel in zip(names, form_factors, strict=True)
    }


def _relative_error(value: float, reference: float) -> float:
    return float(abs(value - reference) / max(abs(reference), 1.0e-14))


def triangular_qsl_symmetry_diagnostics(
    data: OperatorComponentData,
    states: Mapping[str, np.ndarray],
    exact_state: np.ndarray,
) -> dict[str, object]:
    """Compare energy and symmetry-resolved observables on triangular ``4x4``.

    ``120_K_star`` is a real-space three-sublattice form factor.  The ``4x4``
    torus does not contain the K momentum, so it is reported as a local
    ordering tendency rather than as a momentum-sector structure factor.
    """

    if not states:
        raise ValueError("at least one variational state is required")
    exact = _normalize(exact_state)
    if exact.shape != data.carrier.shape:
        raise ValueError("exact_state does not match the common support")
    basis = tuple(tuple(int(bit) for bit in row) for row in data.configurations)
    pair_actions = FixedSzPairActions(basis)
    n_sites = pair_actions.n_sites

    def evaluate(state: np.ndarray) -> dict[str, object]:
        vector = _normalize(state)
        if vector.shape != exact.shape:
            raise ValueError("a variational state has the wrong support")
        action = np.asarray(data.hamiltonian @ vector)
        energy = float(np.vdot(vector, action).real)
        residual = action - energy * vector
        e2 = triangular_e2_moments(
            vector,
            pair_actions.pair,
            TRIANGULAR_4X4_SUPERCELL,
            lambda candidate: np.asarray(data.hamiltonian @ candidate),
        )
        periodic = triangular_periodic_observable_panel(
            vector,
            basis,
            TRIANGULAR_4X4_SUPERCELL,
            pair_actions=pair_actions,
        )
        chirality_image = uniform_chirality_action(
            vector, pair_actions, TRIANGULAR_4X4_SUPERCELL
        )
        chirality_mean = np.vdot(vector, chirality_image)
        chirality_source = chirality_image - chirality_mean * vector
        chirality_m0 = float(np.vdot(chirality_source, chirality_source).real)
        chirality_m1 = float(
            np.vdot(chirality_source, data.hamiltonian @ chirality_source).real
            - energy * chirality_m0
        )
        structure = _ordering_structure_factors(
            _spin_correlation_matrix(vector, pair_actions)
        )
        dimer_values = np.asarray(
            [
                periodic["dimer_M"][name]["leading_strength"]
                for name in ("M1", "M2", "M3")
            ],
            dtype=np.float64,
        )
        return {
            "energy_per_site": energy / n_sites,
            "variance_per_site": float(np.vdot(residual, residual).real) / n_sites,
            "exact_fidelity": float(abs(np.vdot(exact, vector)) ** 2),
            "e2_equal_time_weight_per_site_squared": e2.m0 / n_sites**2,
            "e2_state_referenced_sma_gap": e2.sma_gap,
            "e2_component_weight_spread_fraction": float(
                abs(e2.component_m0[0] - e2.component_m0[1])
                / max(e2.m0, 1.0e-14)
            ),
            "chirality_connected_per_triangle_squared": periodic[
                "chirality_connected_per_triangle_squared"
            ],
            "chirality_state_referenced_sma_gap": (
                chirality_m1 / chirality_m0
                if chirality_m0 > 1.0e-14
                else float("inf")
            ),
            "e2_over_chirality_sma_gap_ratio": (
                e2.sma_gap / (chirality_m1 / chirality_m0)
                if chirality_m0 > 1.0e-14 and abs(chirality_m1) > 1.0e-14
                else float("inf")
            ),
            "bond_nematic_connected_total": periodic[
                "bond_nematic_connected_total"
            ],
            "dimer_M_leading_mean": float(np.mean(dimer_values)),
            "dimer_M_leading_spread": float(np.ptp(dimer_values)),
            "ordering_structure_factors": structure,
        }

    exact_row = evaluate(exact)
    variational = {name: evaluate(state) for name, state in states.items()}
    comparison_keys = (
        "e2_equal_time_weight_per_site_squared",
        "chirality_connected_per_triangle_squared",
        "bond_nematic_connected_total",
        "dimer_M_leading_mean",
    )
    for row in variational.values():
        row["relative_errors_to_exact"] = {
            key: _relative_error(float(row[key]), float(exact_row[key]))
            for key in comparison_keys
        }
        row["ordering_relative_errors_to_exact"] = {
            key: _relative_error(
                float(row["ordering_structure_factors"][key]),
                float(exact_row["ordering_structure_factors"][key]),
            )
            for key in exact_row["ordering_structure_factors"]
            if key != "Gamma"
        }
        errors = tuple(row["relative_errors_to_exact"].values()) + tuple(
            row["ordering_relative_errors_to_exact"].values()
        )
        row["median_symmetry_observable_relative_error"] = float(
            np.median(errors)
        )

    comparisons: dict[str, object] = {}
    gates: dict[str, bool] = {}
    operator_name = "pi_flux_exact_su2_a1_operator_vit"
    baseline_name = "optimized_pi_flux_jastrow_two_lanczos"
    fixed_sz_name = "pi_flux_fixed_sz_gamma_a1_spatial_vit"
    if operator_name in variational and baseline_name in variational:
        operator = variational[operator_name]
        baseline = variational[baseline_name]
        operator_error = float(operator["median_symmetry_observable_relative_error"])
        baseline_error = float(baseline["median_symmetry_observable_relative_error"])
        exact_energy = float(exact_row["energy_per_site"])
        comparisons.update(
            {
                "operator_minus_two_lanczos_energy_per_site": float(
                    operator["energy_per_site"] - baseline["energy_per_site"]
                ),
                "operator_over_two_lanczos_variance_ratio": float(
                    operator["variance_per_site"] / baseline["variance_per_site"]
                ),
                "operator_over_two_lanczos_median_observable_error_ratio": float(
                    operator_error / baseline_error
                ),
                "operator_exact_gap_fraction_of_two_lanczos": float(
                    (operator["energy_per_site"] - exact_energy)
                    / (baseline["energy_per_site"] - exact_energy)
                ),
                "operator_minus_two_lanczos_exact_fidelity": float(
                    operator["exact_fidelity"] - baseline["exact_fidelity"]
                ),
            }
        )
        gates.update(
            {
                "operator_beats_two_lanczos_energy": (
                    operator["energy_per_site"] < baseline["energy_per_site"]
                ),
                "operator_beats_two_lanczos_variance": (
                    operator["variance_per_site"] < baseline["variance_per_site"]
                ),
                "operator_improves_median_symmetry_observable_error": (
                    operator_error < baseline_error
                ),
            }
        )
        gates["local_energy_and_physics_corroboration_pass"] = all(gates.values())
    if operator_name in variational and fixed_sz_name in variational:
        operator = variational[operator_name]
        fixed_sz = variational[fixed_sz_name]
        comparisons.update(
            {
                "operator_minus_fixed_sz_vit_energy_per_site": float(
                    operator["energy_per_site"] - fixed_sz["energy_per_site"]
                ),
                "operator_over_fixed_sz_vit_median_observable_error_ratio": float(
                    operator["median_symmetry_observable_relative_error"]
                    / fixed_sz["median_symmetry_observable_relative_error"]
                ),
            }
        )

    return {
        "geometry": "triangular_4x4_periodic",
        "j2_over_j1": 0.125,
        "exact": exact_row,
        "variational": variational,
        "comparisons": comparisons,
        "gates": gates,
        "diagnostic_contract": {
            "energy_and_variance_are_primary": True,
            "qsl_identification_from_4x4_is_forbidden": True,
            "k_star_is_noncommensurate_real_space_form_factor": True,
            "sma_gap_for_variational_states_is_state_referenced": True,
            "promotion_rule": (
                "A variational energy win is physically corroborated only when "
                "the same state improves the pre-registered symmetry-observable "
                "error over the strongest optimized VMC control."
            ),
        },
    }


__all__ = [
    "TRIANGULAR_4X4_SUPERCELL",
    "triangular_4x4_exact_ground",
    "triangular_qsl_symmetry_diagnostics",
]
