"""Matched exact-small frontier for projected paired-Z2 carrier banks."""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.su2_four_cell_sector_vit import (
    su2_four_cell_sector_log_amplitudes,
)
from ..vmc.adam import train_exact_adam
from .adaptive_projected_pair_carrier import batched_singlet_agp_amplitudes
from .clebsch_tree_exact import coo_matvec, reynolds_project_state
from .four_cell_additive_carrier import _case_data, _init_params
from .four_cell_matched_frontier import _metrics, _parameter_count
from .four_cell_schur import four_cell_schur_sector_basis_matrix
from .four_cell_schur_frontier import (
    SQUARE_N16_SUPPORTS,
    TRIANGULAR_N12_SUPPORTS,
)
from .paired_z2_carrier import paired_z2_kernel_bank


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "no_carrier",
    "paired_z2_additive_carrier",
    "scrambled_paired_z2_carrier",
)


@dataclass(frozen=True)
class PairedZ2CaseData:
    """Static exact-small arrays shared by three matched carrier arms."""

    common: dict[str, object]
    physical_carriers: jax.Array
    scrambled_carriers: jax.Array
    carrier_diagnostics: tuple[dict[str, object], ...]
    scramble_seed: int


def _paired_case_data(name: str) -> PairedZ2CaseData:
    common = _case_data(name)
    if name == "triangular_n12_j2_0125":
        geometry = "triangular_n12"
        kind = "triangular_triangle"
        supports = TRIANGULAR_N12_SUPPORTS
    elif name == "square_n16_j2_05":
        geometry = "square_n16"
        kind = "square_plaquette"
        supports = SQUARE_N16_SUPPORTS
    else:
        raise ValueError(f"unknown paired-Z2 case: {name}")
    basis, configurations = four_cell_schur_sector_basis_matrix(
        kind, 0, target_m2=0, supports=supports
    )
    if not np.array_equal(basis, common["singlet_basis"]):
        raise RuntimeError("paired-Z2 and common singlet bases disagree")
    states = np.asarray(configurations, dtype=np.int8)
    coefficient_columns = []
    diagnostics = []
    for carrier in paired_z2_kernel_bank(geometry):  # type: ignore[arg-type]
        raw = batched_singlet_agp_amplitudes(states, carrier.pairing)
        projected = np.asarray(
            reynolds_project_state(jnp.asarray(raw), common["indices"])
        )
        coefficients = basis.conj().T @ projected
        norm = float(np.linalg.norm(coefficients))
        if norm <= 1.0e-14:
            raise RuntimeError(f"paired-Z2 carrier {carrier.label} projects to zero")
        coefficients /= norm
        coefficient_columns.append(coefficients)
        state = basis @ coefficients
        state /= np.linalg.norm(state)
        diagnostics.append(
            {
                "label": carrier.label,
                "quasiparticle_gap": carrier.quasiparticle_gap,
                "u_condition_number": carrier.u_condition_number,
                "pairing_symmetry_residual": carrier.pairing_symmetry_residual,
                "continuous_igg_nullity": carrier.continuous_igg_nullity,
                "smallest_nonzero_igg_singular_value": (
                    carrier.smallest_nonzero_igg_singular_value
                ),
                "metrics": _metrics(
                    state,
                    common["hamiltonian"],
                    common["exact_state"],
                    basis,
                    common["indices_numpy"],
                ),
            }
        )
    physical = np.column_stack(coefficient_columns)
    scramble_seed = 2026092201 if geometry == "triangular_n12" else 2026092202
    permutation = np.random.default_rng(scramble_seed).permutation(len(physical))
    scrambled = physical[permutation]
    return PairedZ2CaseData(
        common=common,
        physical_carriers=jnp.asarray(physical),
        scrambled_carriers=jnp.asarray(scrambled),
        carrier_diagnostics=tuple(diagnostics),
        scramble_seed=scramble_seed,
    )


def _state(params: object, data: PairedZ2CaseData, carriers) -> jax.Array:
    common = data.common
    residual = jnp.exp(
        su2_four_cell_sector_log_amplitudes(
            params["backbone"], common["actions"], common["config"]
        )
    )
    if carriers is None:
        coefficients = residual
    else:
        weights = jax.nn.softmax(params["carrier_logits"])
        weights = weights * jnp.exp(1.0j * params["carrier_phases"])
        carrier = carriers @ weights
        residual_scale = jnp.exp(
            params["residual_log_scale"] + 1.0j * params["residual_phase"]
        )
        coefficients = carrier + residual_scale * residual
    physical = common["basis"] @ coefficients
    return reynolds_project_state(physical, common["indices"])


def paired_z2_frontier(
    *,
    seeds: tuple[int, ...],
    steps: int = 1000,
) -> dict[str, object]:
    """Compare paired-Z2, coefficient-scrambled, and raw Schur-ViT arms."""

    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct seeds are required")
    if steps < 1:
        raise ValueError("steps must be positive")
    cases = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _paired_case_data(case_name)
        common = data.common
        carrier_count = data.physical_carriers.shape[1]
        runs = {arm: [] for arm in ARM_IDS}
        for seed in seeds:
            for arm_id, carriers in (
                (ARM_IDS[0], None),
                (ARM_IDS[1], data.physical_carriers),
                (ARM_IDS[2], data.scrambled_carriers),
            ):
                params = _init_params(
                    jax.random.PRNGKey(seed),
                    common["config"],
                    carrier_count,
                    additive=carriers is not None,
                )

                def state_fn(candidate):
                    return _state(candidate, data, carriers)

                def loss(candidate):
                    state = state_fn(candidate)
                    action = coo_matvec(
                        common["rows"],
                        common["columns"],
                        common["values"],
                        state,
                    )
                    return jnp.vdot(state, action).real / jnp.vdot(state, state).real

                initial = float(loss(params))
                started = time.perf_counter()
                trained, optimizer = train_exact_adam(
                    params,
                    loss_fn=loss,
                    steps=steps,
                    learning_rate=1.0e-2,
                    record_every=max(1, steps // 5),
                )
                state = np.array(state_fn(trained), copy=True)
                state /= np.linalg.norm(state)
                row = {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "parameter_count": _parameter_count(trained),
                    "initial_energy_total": initial,
                    "wall_seconds": time.perf_counter() - started,
                    "full_h_calls": int(steps + 2),
                    "optimizer": optimizer,
                    "metrics": _metrics(
                        state,
                        common["hamiltonian"],
                        common["exact_state"],
                        common["singlet_basis"],
                        common["indices_numpy"],
                    ),
                }
                if carriers is not None:
                    row["carrier_weights"] = np.asarray(
                        jax.nn.softmax(trained["carrier_logits"])
                    ).tolist()
                    row["residual_scale_magnitude"] = float(
                        np.exp(np.asarray(trained["residual_log_scale"]))
                    )
                runs[arm_id].append(row)

        exact = float(common["exact_energy_total"])
        summary = {}
        for arm_id, rows in runs.items():
            errors = np.asarray(
                [row["metrics"]["energy_total"] - exact for row in rows]
            )
            summary[arm_id] = {
                "median_energy_error_total": float(np.median(errors)),
                "maximum_energy_error_total": float(np.max(errors)),
                "all_seed_energy_errors_total": errors.tolist(),
                "median_variance_total": float(
                    np.median([row["metrics"]["variance_total"] for row in rows])
                ),
                "median_wall_seconds": float(
                    np.median([row["wall_seconds"] for row in rows])
                ),
            }
        physical = summary[ARM_IDS[1]]
        raw = summary[ARM_IDS[0]]
        scrambled = summary[ARM_IDS[2]]
        paired_wins = [
            physical_error < scrambled_error
            for physical_error, scrambled_error in zip(
                physical["all_seed_energy_errors_total"],
                scrambled["all_seed_energy_errors_total"],
                strict=True,
            )
        ]
        gates = {
            "physical_improves_raw_median": physical[
                "median_energy_error_total"
            ]
            < raw["median_energy_error_total"],
            "physical_improves_raw_worst_seed": physical[
                "maximum_energy_error_total"
            ]
            < raw["maximum_energy_error_total"],
            "physical_beats_scrambled_median": physical[
                "median_energy_error_total"
            ]
            < scrambled["median_energy_error_total"],
            "physical_beats_scrambled_every_paired_seed": bool(all(paired_wins)),
            "physical_all_seeds_below_1e-5": physical[
                "maximum_energy_error_total"
            ]
            < 1.0e-5,
        }
        cases[case_name] = {
            "n_sites": common["n_sites"],
            "exact_energy_total": exact,
            "steps": int(steps),
            "scramble_seed": data.scramble_seed,
            "carrier_diagnostics": list(data.carrier_diagnostics),
            "runs": runs,
            "summary": summary,
            "paired_physical_win_flags": paired_wins,
            "gates": gates,
            "gates_passed": bool(all(gates.values())),
        }
    return {
        "seeds": list(seeds),
        "cases": cases,
        "all_cases_passed": bool(
            all(case["gates_passed"] for case in cases.values())
        ),
        "claim_boundary": (
            "The carrier is a number- and Gutzwiller-projected, real singlet "
            "BdG state with finite mean-field gap and zero continuous-IGG "
            "nullity. This labels a paired-Z2 mean-field mechanism; it does "
            "not by itself establish projected-state topological order."
        ),
    }


__all__ = ["ARM_IDS", "paired_z2_frontier"]
