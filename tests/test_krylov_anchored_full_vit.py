from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_n16_krylov_anchored_full_vit import (
    _validate,
)
from projects.clebsch_tree_nqs.workflows.spin_n16_full_vit_sr_reachability import (
    _validate as validate_sr_reachability,
)
from projects.clebsch_tree_nqs.workflows.spin_n16_full_vit_sr_continuation import (
    _validate as validate_sr_continuation,
)
from su2_nqs.benchmarks.krylov_anchored_full_vit import (
    a1_orbit_coordinates,
    a1_orbit_hamiltonian,
    a1_orbit_reduction,
    anchored_unit_residual_state,
    lowest_krylov_state,
    matched_backbone_initializations,
    scrambled_coordinate_layout,
)


def test_registered_config_preserves_four_arm_contract() -> None:
    path = Path(
        "projects/clebsch_tree_nqs/configs/validation/"
        "spin_n16_krylov_anchored_full_vit.json"
    )
    spec = json.loads(path.read_text())
    _validate(spec)
    assert spec["optimization"]["all_backbone_parameters_trainable"] is True
    assert spec["architecture"]["use_distance_envelope"] is False

    sr_path = Path(
        "projects/clebsch_tree_nqs/configs/validation/"
        "spin_n16_full_vit_sr_reachability.json"
    )
    sr_spec = json.loads(sr_path.read_text())
    validate_sr_reachability(sr_spec)
    assert sr_spec["development_only"] is True

    continuation_path = Path(
        "projects/clebsch_tree_nqs/configs/validation/"
        "spin_n16_full_vit_sr_continuation.json"
    )
    continuation = json.loads(continuation_path.read_text())
    validate_sr_continuation(continuation)
    assert continuation["final_development_optimizer_attempt"] is True


def test_orbit_reduction_matches_explicit_normalized_orbit_basis() -> None:
    permutations = np.asarray(
        (
            (0, 1, 2, 3),
            (1, 0, 3, 2),
        ),
        dtype=np.int32,
    )
    reduction = a1_orbit_reduction(permutations)
    np.testing.assert_array_equal(reduction.orbit_sizes, (2, 2))
    amplitudes = jnp.asarray((1.0, 3.0, 2.0, 6.0))
    coordinates = np.asarray(a1_orbit_coordinates(amplitudes, reduction))
    np.testing.assert_allclose(coordinates, (4.0 / np.sqrt(2), 8.0 / np.sqrt(2)))

    hamiltonian = np.asarray(
        (
            (2.0, 1.0, 0.0, 0.0),
            (1.0, 2.0, 0.0, 0.0),
            (0.0, 0.0, 4.0, -1.0),
            (0.0, 0.0, -1.0, 4.0),
        )
    )
    rows, columns = np.nonzero(hamiltonian)
    reduced = a1_orbit_hamiltonian(
        rows, columns, hamiltonian[rows, columns], reduction
    )
    np.testing.assert_allclose(reduced, np.diag((3.0, 3.0)), atol=1.0e-14)


def test_anchor_residual_is_carrier_orthogonal_and_krylov_is_variational() -> None:
    carrier = jnp.asarray((1.0, 1.0, 0.0))
    residual = jnp.asarray((1.0, -1.0, 2.0))
    state = np.asarray(anchored_unit_residual_state(carrier, residual, 0.3))
    anchor = np.asarray(carrier) / np.linalg.norm(carrier)
    difference = state - anchor
    assert abs(np.vdot(anchor, difference)) < 2.0e-14
    np.testing.assert_allclose(np.linalg.norm(difference), 0.3, atol=2.0e-14)

    hamiltonian = np.diag((-2.0, -1.0, 1.0))
    ritz, rank = lowest_krylov_state(
        np.asarray((1.0, 1.0, 1.0)), hamiltonian, degree=2
    )
    assert rank == 3
    assert float(np.vdot(ritz, hamiltonian @ ritz).real) == np.min(
        np.diag(hamiltonian)
    )


def test_su2_and_fixed_sz_backbones_are_bitwise_identical() -> None:
    su2, fixed, su2_config, _fixed_config = matched_backbone_initializations(
        17,
        embedding_dim=8,
        head_count=2,
        depth=1,
        mlp_ratio=2,
        phase_scale=0.25,
        initialization_scale=0.15,
        use_distance_envelope=False,
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(su2),
        jax.tree_util.tree_leaves(fixed),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)

    permutation = np.random.default_rng(3).permutation(16)
    scrambled = scrambled_coordinate_layout(su2_config.layout, permutation)
    assert scrambled.tree == su2_config.layout.tree
    assert scrambled.site_order == su2_config.layout.site_order
    assert scrambled.site_coordinates != su2_config.layout.site_coordinates
