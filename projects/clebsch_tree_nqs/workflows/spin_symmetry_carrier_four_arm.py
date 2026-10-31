"""Exact-small four-arm audit for spatial carriers on one SU(2)-ViT backbone.

The comparison freezes each cached exact-SU(2) Route-ViT state and changes
only the low-dimensional carrier span:

``baseline``
    the unmodified backbone state;
``A1``
    three orientation-uniform J1/J1, J1/J2, and J2/J2 bond pairs;
``A1+E2``
    the same three directions plus the corresponding
    ``E2 x E2 -> A1`` contractions;
``generic``
    Hamiltonian-blind Reynolds averages of generic two-bond strings, with
    three- and six-direction budgets matched to the physical arms.

The generic sources are fixed from lattice geometry and explicit seeds before
the Hamiltonian is consulted.  This makes the workflow a causal variational
test, rather than a relabelled Krylov comparison.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import physical_permutation_index_table
from su2_nqs.benchmarks.physical_interpretation import (
    FixedSzPairActions,
    hnf_oriented_bond_families,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity
from su2_nqs.lattices.triangular_hnf import J1_POSITIVE, J2_POSITIVE
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.operators.reynolds_bond_carriers import (
    bond_pair_orbit,
    canonical_bond,
    connected_gram_schmidt,
    reynolds_bond_pair_action,
)
from su2_nqs.operators.triangular_resonance import (
    connected_triangular_a1_pair_action,
    connected_triangular_e2_pair_action,
    triangular_e2_moments,
)

from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_e2_resonance_audit import (
    _normalized,
    _ritz,
    _state_metrics,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_symmetry_carrier_four_arm_audit"


def _bond_pool(supercell, displacements) -> tuple[tuple[int, int], ...]:
    n_sites = int(supercell[0][0] * supercell[1][1])
    return tuple(
        sorted(
            {
                canonical_bond(bond, n_sites)
                for family in hnf_oriented_bond_families(
                    supercell, displacements
                )
                for bond in family
            }
        )
    )


def _generic_source_bank(
    *,
    seed: int,
    supercell,
    group: Sequence[Sequence[int]],
    screen_state: np.ndarray | None = None,
    screen_actions: FixedSzPairActions | None = None,
) -> tuple[dict[str, object], ...]:
    """Draw two sources per shell pair without using a Hamiltonian.

    When a fixed geometry-only singlet ``screen_state`` is supplied, a source
    is retained only if it increases the sequential carrier rank on that
    state.  This prevents accidental duplicate controls while remaining blind
    to every trained backbone and to the Hamiltonian.
    """

    if (screen_state is None) != (screen_actions is None):
        raise ValueError("screen_state and screen_actions must be supplied together")

    pools = {
        "11": (_bond_pool(supercell, J1_POSITIVE),) * 2,
        "12": (
            _bond_pool(supercell, J1_POSITIVE),
            _bond_pool(supercell, J2_POSITIVE),
        ),
        "22": (_bond_pool(supercell, J2_POSITIVE),) * 2,
    }
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    screen_images: list[np.ndarray] = []
    seen: set[tuple[str, tuple[tuple[tuple[int, int], tuple[int, int]], ...]]] = set()
    # The prefix has one source from each shell pair.  The second triplet adds
    # the same amount of work in the same order, making G3/G6 nested controls.
    for shell_pair in ("11", "12", "22", "11", "12", "22"):
        left_pool, right_pool = pools[shell_pair]
        for _attempt in range(10_000):
            left = left_pool[int(rng.integers(len(left_pool)))]
            right = right_pool[int(rng.integers(len(right_pool)))]
            if shell_pair in {"11", "22"} and left == right:
                continue
            orbit = bond_pair_orbit(left, right, group)
            identity = (shell_pair, orbit)
            if identity in seen:
                continue
            candidate_image = None
            if screen_state is not None and screen_actions is not None:
                candidate_image = reynolds_bond_pair_action(
                    screen_state,
                    screen_actions.pair,
                    left,
                    right,
                    group,
                )
                previous_rank = len(
                    connected_gram_schmidt(screen_state, screen_images)
                )
                next_rank = len(
                    connected_gram_schmidt(
                        screen_state, (*screen_images, candidate_image)
                    )
                )
                if next_rank != previous_rank + 1:
                    continue
            seen.add(identity)
            rows.append(
                {
                    "shell_pair": shell_pair,
                    "first_bond": list(left),
                    "second_bond": list(right),
                    "orbit_size": len(orbit),
                    "orbit_sha256": canonical_sha256(orbit),
                }
            )
            if candidate_image is not None:
                screen_images.append(candidate_image)
            break
        else:
            raise RuntimeError("could not draw a distinct generic carrier orbit")
    return tuple(rows)


def _geometry_only_singlet_screen(
    basis: Sequence[Sequence[int]], permutation_table: np.ndarray
) -> np.ndarray:
    """Return a Gamma-A1 projection of a fixed nearest-pair singlet product."""

    rows = np.asarray(basis, dtype=np.int8)
    if rows.ndim != 2 or rows.shape[1] % 2:
        raise ValueError("screen basis must have an even positive site count")
    state = np.ones((rows.shape[0],), dtype=np.float64)
    for first in range(0, rows.shape[1], 2):
        left, right = rows[:, first], rows[:, first + 1]
        state *= np.where(
            (left == 0) & (right == 1),
            1.0,
            np.where((left == 1) & (right == 0), -1.0, 0.0),
        )
    state = np.mean(state[np.asarray(permutation_table)], axis=0)
    norm = float(np.linalg.norm(state))
    if norm <= 1.0e-14:
        raise RuntimeError("the geometry-only singlet screen vanished under projection")
    return np.asarray(state / norm, dtype=np.complex128)


def _generic_directions(
    state: np.ndarray,
    sources: Sequence[dict[str, object]],
    *,
    actions: FixedSzPairActions,
    group: Sequence[Sequence[int]],
) -> tuple[np.ndarray, ...]:
    raw = [
        reynolds_bond_pair_action(
            state,
            actions.pair,
            row["first_bond"],
            row["second_bond"],
            group,
        )
        for row in sources
    ]
    whitened = connected_gram_schmidt(state, raw)
    if len(whitened) != len(sources):
        raise RuntimeError(
            "the preregistered generic bank is rank deficient on this backbone"
        )
    return whitened


def _physical_directions(
    state: np.ndarray,
    actions: FixedSzPairActions,
    supercell,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    a1 = (
        connected_triangular_a1_pair_action(
            state, actions.pair, supercell, first_displacements=J1_POSITIVE
        )[0],
        connected_triangular_a1_pair_action(
            state,
            actions.pair,
            supercell,
            first_displacements=J1_POSITIVE,
            second_displacements=J2_POSITIVE,
        )[0],
        connected_triangular_a1_pair_action(
            state, actions.pair, supercell, first_displacements=J2_POSITIVE
        )[0],
    )
    e2 = (
        connected_triangular_e2_pair_action(
            state, actions.pair, supercell, first_displacements=J1_POSITIVE
        )[0],
        connected_triangular_e2_pair_action(
            state,
            actions.pair,
            supercell,
            first_displacements=J1_POSITIVE,
            second_displacements=J2_POSITIVE,
        )[0],
        connected_triangular_e2_pair_action(
            state, actions.pair, supercell, first_displacements=J2_POSITIVE
        )[0],
    )
    if len(connected_gram_schmidt(state, a1)) != 3:
        raise RuntimeError("the A1 carrier bank lost rank")
    if len(connected_gram_schmidt(state, (*a1, *e2))) != 6:
        raise RuntimeError("the A1+E2 carrier bank lost rank")
    return a1, e2


def _subspace_overlap(
    state: np.ndarray,
    physical: Sequence[np.ndarray],
    generic: Sequence[np.ndarray],
) -> dict[str, object]:
    left = np.column_stack(connected_gram_schmidt(state, physical))
    right = np.column_stack(connected_gram_schmidt(state, generic))
    singular = np.linalg.svd(left.conj().T @ right, compute_uv=False)
    joint_rank = np.linalg.matrix_rank(np.column_stack((left, right)), tol=1.0e-10)
    return {
        "physical_rank": int(left.shape[1]),
        "generic_rank": int(right.shape[1]),
        "joint_rank": int(joint_rank),
        "generic_rank_outside_physical_span": int(joint_rank - left.shape[1]),
        "principal_overlap_singular_values": singular.tolist(),
        "maximum_principal_overlap": float(np.max(singular)),
    }


def _gamma_leakage(
    directions: Sequence[np.ndarray], permutation_table: np.ndarray
) -> float:
    rows = []
    for direction in directions:
        denominator = max(float(np.linalg.norm(direction)), 1.0e-30)
        rows.extend(
            float(np.linalg.norm(direction[indices] - direction) / denominator)
            for indices in permutation_table
        )
    return max(rows, default=0.0)


def _metrics(variant, *, exact, exact_energy, exact_moments, data, actions, supercell):
    state = np.asarray(variant["state"], dtype=np.complex128)
    row = {
        key: value for key, value in variant.items() if key != "state"
    }
    row.update(
        _state_metrics(
            state,
            exact=exact,
            exact_energy=exact_energy,
            exact_m0=exact_moments.m0,
            exact_sma_gap=exact_moments.sma_gap,
            data=data,
            pair_actions=actions,
            supercell=supercell,
        )
    )
    spin_image = np.asarray(data["spin_squared"] @ _normalized(state))
    row["spin_squared_expectation"] = float(
        np.vdot(_normalized(state), spin_image).real
    )
    row["spin_squared_residual"] = float(np.linalg.norm(spin_image))
    return row


def run_spin_symmetry_carrier_four_arm(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the exact-small same-backbone/same-dimension carrier audit."""

    spec = json.loads(config_path.read_text())
    if (
        spec.get("schema_version"),
        spec.get("domain"),
        spec.get("task"),
    ) != (1, "spin", TASK):
        raise ValueError("invalid four-arm carrier config identity")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    case = dict(spec["case"])
    if (case.get("geometry"), case.get("rows"), case.get("columns")) != (
        "triangular",
        4,
        4,
    ):
        raise ValueError("the exact local four-arm gate is triangular 4x4")
    seeds = tuple(int(value) for value in spec["state_seeds"])
    bank_seeds = tuple(int(value) for value in spec["generic_bank_seeds"])
    if len(seeds) < 3 or len(bank_seeds) < 3 or len(set(bank_seeds)) != len(bank_seeds):
        raise ValueError("at least three distinct state and generic-bank seeds are required")

    state_path = ROOT / str(spec["state_bundle"])
    data = _prepare_case(case, "fixed_m0")
    supercell = ((4, 0), (0, 4))
    group = triangular_hnf_gamma_a1_group(supercell)
    permutation_table = physical_permutation_index_table(data["basis"], group)
    actions = FixedSzPairActions(data["basis"])
    screen_state = _geometry_only_singlet_screen(data["basis"], permutation_table)
    bundle = np.load(state_path)
    if not np.array_equal(np.asarray(bundle["basis"]), np.asarray(data["basis"])):
        raise RuntimeError("cached state bundle and reconstructed basis differ")
    exact = _normalized(bundle["exact_ground"])
    exact_energy = float(np.vdot(exact, data["hamiltonian"] @ exact).real)
    if abs(exact_energy - float(case["exact_energy_anchor"])) > 1.0e-10:
        raise RuntimeError("cached exact state fails the registered energy anchor")
    exact_moments = triangular_e2_moments(
        exact,
        actions.pair,
        supercell,
        lambda candidate: data["hamiltonian"] @ candidate,
    )
    source_banks = {
        str(seed): _generic_source_bank(
            seed=seed,
            supercell=supercell,
            group=group,
            screen_state=screen_state,
            screen_actions=actions,
        )
        for seed in bank_seeds
    }

    trials = []
    for seed in seeds:
        key = f"seed{seed}_gamma_a1_route_vit_conditioned_d4"
        if key not in bundle:
            raise KeyError(f"cached state is absent: {key}")
        baseline = _normalized(bundle[key])
        a1, e2 = _physical_directions(baseline, actions, supercell)
        fixed = {
            "baseline": {"state": baseline, "retained_dimension": 1},
            "A1": _ritz(baseline, a1, data["hamiltonian"]),
            "A1_plus_E2": _ritz(baseline, (*a1, *e2), data["hamiltonian"]),
        }
        fixed_metrics = {
            name: _metrics(
                value,
                exact=exact,
                exact_energy=exact_energy,
                exact_moments=exact_moments,
                data=data,
                actions=actions,
                supercell=supercell,
            )
            for name, value in fixed.items()
        }
        generic_rows = []
        for bank_seed in bank_seeds:
            directions = _generic_directions(
                baseline,
                source_banks[str(bank_seed)],
                actions=actions,
                group=group,
            )
            generic3 = _metrics(
                _ritz(baseline, directions[:3], data["hamiltonian"]),
                exact=exact,
                exact_energy=exact_energy,
                exact_moments=exact_moments,
                data=data,
                actions=actions,
                supercell=supercell,
            )
            generic6 = _metrics(
                _ritz(baseline, directions, data["hamiltonian"]),
                exact=exact,
                exact_energy=exact_energy,
                exact_moments=exact_moments,
                data=data,
                actions=actions,
                supercell=supercell,
            )
            generic_rows.append(
                {
                    "bank_seed": bank_seed,
                    "generic3": generic3,
                    "generic6": generic6,
                    "subspace_overlap": _subspace_overlap(
                        baseline, (*a1, *e2), directions
                    ),
                    "gamma_a1_relative_leakage": _gamma_leakage(
                        directions, permutation_table
                    ),
                }
            )
        primary = generic_rows[0]
        trials.append(
            {
                "state_seed": seed,
                "state_bundle_key": key,
                "arms": {
                    **fixed_metrics,
                    "generic3": primary["generic3"],
                    "generic6": primary["generic6"],
                },
                "generic_bank_robustness": generic_rows,
                "physical_carrier_gamma_a1_relative_leakage": _gamma_leakage(
                    (*a1, *e2), permutation_table
                ),
                "same_work_gates": {
                    "A1_beats_generic3_energy": fixed_metrics["A1"]["energy"]
                    < primary["generic3"]["energy"] - 1.0e-12,
                    "A1_plus_E2_beats_generic6_energy": fixed_metrics["A1_plus_E2"]["energy"]
                    < primary["generic6"]["energy"] - 1.0e-12,
                    "E2_marginal_energy_gain_exceeds_generic_second_triplet": (
                        fixed_metrics["A1"]["energy"]
                        - fixed_metrics["A1_plus_E2"]["energy"]
                        > primary["generic3"]["energy"]
                        - primary["generic6"]["energy"]
                    ),
                    "A1_plus_E2_repairs_nematic_m0_vs_A1": fixed_metrics["A1_plus_E2"]["e2_m0_abs_error"]
                    < fixed_metrics["A1"]["e2_m0_abs_error"],
                    "all_arms_exact_singlet_to_1e_minus_9": max(
                        row["spin_squared_residual"]
                        for row in (
                            *fixed_metrics.values(),
                            primary["generic3"],
                            primary["generic6"],
                        )
                    )
                    < 1.0e-9,
                },
            }
        )

    gates = tuple(row["same_work_gates"] for row in trials)
    all_generic = [
        bank
        for trial in trials
        for bank in trial["generic_bank_robustness"]
    ]
    robust_pairs = [
        (trial, bank)
        for trial in trials
        for bank in trial["generic_bank_robustness"]
    ]
    summary = {
        "trial_count": len(trials),
        "generic_bank_count_per_trial": len(bank_seeds),
        "A1_beats_primary_generic3_count": sum(
            row["A1_beats_generic3_energy"] for row in gates
        ),
        "A1_plus_E2_beats_primary_generic6_count": sum(
            row["A1_plus_E2_beats_generic6_energy"] for row in gates
        ),
        "E2_marginal_beats_generic_marginal_count": sum(
            row["E2_marginal_energy_gain_exceeds_generic_second_triplet"]
            for row in gates
        ),
        "robust_comparison_count": len(robust_pairs),
        "A1_beats_all_generic3_count": sum(
            trial["arms"]["A1"]["energy"]
            < bank["generic3"]["energy"] - 1.0e-12
            for trial, bank in robust_pairs
        ),
        "A1_plus_E2_beats_all_generic6_count": sum(
            trial["arms"]["A1_plus_E2"]["energy"]
            < bank["generic6"]["energy"] - 1.0e-12
            for trial, bank in robust_pairs
        ),
        "E2_marginal_beats_all_generic_marginals_count": sum(
            trial["arms"]["A1"]["energy"]
            - trial["arms"]["A1_plus_E2"]["energy"]
            > bank["generic3"]["energy"] - bank["generic6"]["energy"]
            for trial, bank in robust_pairs
        ),
        "A1_plus_E2_repairs_nematic_m0_vs_A1_count": sum(
            row["A1_plus_E2_repairs_nematic_m0_vs_A1"] for row in gates
        ),
        "all_primary_arms_exact_singlets": all(
            row["all_arms_exact_singlet_to_1e_minus_9"] for row in gates
        ),
        "maximum_generic_gamma_a1_relative_leakage": max(
            row["gamma_a1_relative_leakage"] for row in all_generic
        ),
        "minimum_generic_rank_outside_physical_span": min(
            row["subspace_overlap"]["generic_rank_outside_physical_span"]
            for row in all_generic
        ),
        "median_energy_gain": {
            arm: float(
                np.median(
                    [
                        row["arms"]["baseline"]["energy"]
                        - row["arms"][arm]["energy"]
                        for row in trials
                    ]
                )
            )
            for arm in ("A1", "A1_plus_E2", "generic3", "generic6")
        },
    }
    result = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "state_bundle": {
            "path": str(state_path.relative_to(ROOT)),
            "sha256": file_sha256(state_path),
            "reused_without_training": True,
        },
        "backbone_contract": {
            "family": "exact_SU2_Gamma_A1_Route_ViT",
            "same_cached_backbone_in_every_arm": True,
            "same_work_pairs": {
                "dimension_4": ["A1", "generic3"],
                "dimension_7": ["A1_plus_E2", "generic6"],
            },
            "generic_sources_see_hamiltonian_or_energy": False,
            "generic_rank_screen": (
                "fixed nearest-pair singlet product projected to Gamma-A1; "
                "independent of trained backbones and Hamiltonian"
            ),
        },
        "generic_source_banks": {
            key: list(value) for key, value in source_banks.items()
        },
        "exact_reference": {
            "energy": exact_energy,
            "e2_m0": exact_moments.m0,
            "e2_sma_gap": exact_moments.sma_gap,
        },
        "trials": trials,
        "summary": summary,
        "interpretation_boundary": {
            "exact_support_n16_not_thermodynamic": True,
            "frozen_backbone_ritz_not_end_to_end_training": True,
            "n36_sampler_gate_is_separate_and_already_passed": True,
            "promotion_requires_trainable_equal_capacity_n36_replication": True,
            "no_new_ed_diagonalization": True,
        },
        "runtime_identity": numerical_runtime_identity(),
    }
    result["result_sha256"] = canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "results.json", result)
    return result


__all__ = ["TASK", "run_spin_symmetry_carrier_four_arm"]
