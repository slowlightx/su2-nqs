from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs import run
from projects.clebsch_tree_nqs.workflows import (
    spin_triangle_patch_n36_low_spin_resonance as profile,
)
from su2_nqs.frontier.triangle_patch_physical_multiplet_init import (
    init_triangle_patch_cluster_multiplets,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.models.triangle_patch_clebsch_tree import triangle_patch_6x6_canary_plan
from su2_nqs.vmc.spin_observables import (
    prepare_spin_structure_factor,
    spin_structure_factor_local_values,
)


jax.config.update("jax_enable_x64", True)

CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/"
    "spin_triangle_patch_n36_low_spin_resonance_profile.json"
)


def _spec() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_profile_config_freezes_nested_causal_question_and_external_guides() -> None:
    spec = _spec()
    profile._validate(spec)
    assert spec["claim_boundary"] == profile.CLAIM_BOUNDARY
    assert [arm["expected_dimension"] for arm in spec["optimizer"]["arms"]] == [
        0,
        10,
        8,
        18,
    ]
    assert spec["optimizer"]["name"] == "explicit_small_active_qgt_sr"
    assert spec["optimizer"]["persistent_solver_parity_in_profile"] is False
    for case in spec["cases"]:
        assert (
            case["literature_reference_energy_per_site"]
            == case["exact_energy_per_site"]
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda spec: spec.update(task="wrong"),
        lambda spec: spec["initialization"].update(
            local_q2_channel_1="random_triplet"
        ),
        lambda spec: spec["optimizer"]["arms"][2].update(expected_dimension=9),
        lambda spec: spec["cases"][0].update(
            literature_reference_energy_per_site=-0.5
        ),
    ),
)
def test_profile_config_fails_closed(mutation) -> None:
    spec = copy.deepcopy(_spec())
    mutation(spec)
    with pytest.raises((TypeError, ValueError)):
        profile._validate(spec)


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_physical_multiplet_reference_gives_nested_10_8_18_charts(
    geometry: str, ratio: float
) -> None:
    plan = triangle_patch_6x6_canary_plan(geometry)
    params = init_triangle_patch_cluster_multiplets(
        jax.random.PRNGKey(20260917),
        plan,
        j2_over_j1=ratio,
        background_scale=0.05,
        route_gate_scale=0.05,
    )
    spaces = profile._arm_subspaces(params, plan)
    assert {name: 0 if chart is None else chart.dimension for name, chart in spaces.items()} == {
        "A0_frozen": 0,
        "A1_q0": 10,
        "A2_q2_increment": 8,
        "A12_q0_q2": 18,
    }
    complete = spaces["A12_q0_q2"]
    assert complete is not None
    q0 = spaces["A1_q0"]
    q2 = spaces["A2_q2_increment"]
    assert q0 is not None and q2 is not None
    assert set(q0.active_flat_indices).isdisjoint(q2.active_flat_indices)
    assert set(q0.active_flat_indices) | set(q2.active_flat_indices) == set(
        complete.active_flat_indices
    )


def test_qgt_summary_and_training_preselection_are_quantitative_not_hard_gates() -> None:
    qgt = np.diag([0.0, 2.0, 8.0])
    summary = profile._qgt_summary(qgt, 0.5)
    assert summary["numerical_rank"] == 2
    np.testing.assert_allclose(summary["stable_rank"], 68.0 / 64.0)
    np.testing.assert_allclose(summary["damped_condition_number"], 17.0)

    reference = np.zeros(4, dtype=np.complex128)
    # Upweight the two lower-energy configurations at the first scale.
    candidates = {
        1.0: np.asarray([0.3, 0.3, -0.3, -0.3], dtype=np.complex128),
        0.5: np.zeros(4, dtype=np.complex128),
    }
    scale, attempts = profile._training_preselect(
        reference,
        candidates,
        np.asarray([-2.0, -1.0, 1.0, 2.0]),
        minimum_ess_fraction=0.5,
    )
    assert scale == 1.0
    assert attempts[0]["energy_difference"] < 0.0
    assert attempts[0]["effective_sample_size_fraction"] >= 0.5


def test_explicit_small_active_sr_solves_damped_system_and_applies_fisher_radius() -> None:
    qgt = np.diag([1.0, 4.0])
    gradient = np.asarray([2.0, -5.0])
    direction, diagnostics = profile._explicit_active_sr_direction(
        qgt,
        gradient,
        damping=0.5,
        step_size=1.0,
        fisher_trust_radius=0.2,
    )
    raw = np.linalg.solve(qgt + 0.5 * np.eye(2), gradient)
    raw_fisher = np.sqrt(raw @ qgt @ raw)
    expected_scale = min(1.0, 0.2 / raw_fisher)
    np.testing.assert_allclose(direction, expected_scale * raw, atol=2e-15)
    np.testing.assert_allclose(diagnostics["raw_fisher_norm"], raw_fisher)
    assert diagnostics["linear_residual_norm"] < 1e-14
    assert diagnostics["persistent_matrix_free_parity_skipped_by_profile"] is True


def test_exact_singlet_diagonal_structure_factor_matches_two_spin_answer() -> None:
    samples = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    modes = (
        ("gamma", np.asarray([1.0, 1.0], dtype=np.complex128)),
        ("pi", np.asarray([1.0, -1.0], dtype=np.complex128)),
    )
    values = profile._singlet_diagonal_structure_factors(samples, modes)
    np.testing.assert_allclose(values[:, 0], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(values[:, 1], 1.5, atol=0.0, rtol=0.0)

    # Independent full-SU(2) local-action comparison on |01>-|10>.
    table = jnp.asarray([0.0, 1.0, -1.0, 0.0], dtype=jnp.float64) / np.sqrt(2.0)

    def amplitude(candidate, bits):
        index = 2 * bits[0] + bits[1]
        return candidate[index]

    full_spec = prepare_spin_structure_factor(modes[1][1])
    full_local = spin_structure_factor_local_values(
        table,
        jnp.asarray(samples),
        full_spec,
        amplitude_fn=amplitude,
    )
    np.testing.assert_allclose(np.mean(full_local), np.mean(values[:, 1]), atol=1e-14)


def test_directional_modes_partition_square_hamiltonian_exactly() -> None:
    bonds = square_j1j2_bonds(6, 6, j1=1.0, j2=0.5, periodic=True)
    modes = profile._directional_bond_modes("square", bonds)
    flattened = sorted(bond for _, _, grouped in modes for bond in grouped)
    assert flattened == sorted(bonds)
    assert len(modes) == 4


def test_cli_exposes_case_selection_and_explicit_execution_tiers() -> None:
    arguments = run._build_parser().parse_args(
        [
            "triangle-patch-n36-low-spin-resonance",
            "--config",
            str(CONFIG),
            "--case-id",
            "square_6x6_j2_0p5",
            "--smoke",
        ]
    )
    assert arguments.case_id == "square_6x6_j2_0p5"
    assert arguments.smoke is True
    assert arguments.local_pilot is False

    pilot = run._build_parser().parse_args(
        [
            "triangle-patch-n36-low-spin-resonance",
            "--config",
            str(CONFIG),
            "--local-pilot",
        ]
    )
    assert pilot.smoke is False
    assert pilot.local_pilot is True


def test_execution_tiers_are_mutually_exclusive_before_runtime(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        profile.run_triangle_patch_n36_low_spin_resonance(
            CONFIG,
            tmp_path / "invalid",
            smoke=True,
            local_pilot=True,
        )
