import jax
import numpy as np

from su2_nqs.models.many_cell_schur_vit import (
    many_cell_schur_log_amplitudes,
    sample_many_cell_schur_vit,
)
from su2_nqs.models.many_cell_simplex_vit import (
    ManyCellSimplexViTConfig,
    init_many_cell_simplex_vit,
    local_f_move_feature_table,
    many_cell_simplex_geometry_features,
    many_cell_simplex_log_amplitudes,
    sample_many_cell_simplex_vit,
)
from su2_nqs.su2.many_cell_schur import (
    scramble_simplex_coordinates,
    triangular_six_by_six_simplex_layout,
    validate_layout_state,
)
from su2_nqs.su2.wigner6j import f_move
from su2_nqs.vmc.many_cell_simplex import (
    audit_sampled_local_f_moves,
    evaluate_many_cell_simplex_additive_local_energy,
    evaluate_many_cell_simplex_local_energy,
    sample_many_cell_simplex_additive_proposal,
)
from su2_nqs.vmc.many_cell_tree import (
    build_many_cell_tree_connectivity,
    evaluate_many_cell_tree_local_energy,
)


def _config(layout=None):
    return ManyCellSimplexViTConfig(
        triangular_six_by_six_simplex_layout() if layout is None else layout,
        embedding_dim=8,
        head_count=2,
        depth=1,
    )


def test_simplex_layout_is_a_complete_n36_singlet_tree() -> None:
    layout = triangular_six_by_six_simplex_layout()
    assert layout.n_sites == 36
    assert layout.n_simplexes == 3
    assert tuple(cell for group in layout.cell_groups for cell in group) == tuple(
        range(9)
    )
    assert len(set(layout.pair_indices + layout.root_indices)) == 6
    assert layout.schur.tree.target_is_reachable(0)


def test_local_f_features_are_built_from_normalized_exact_columns() -> None:
    table = local_f_move_feature_table()
    matrix, left, _ = f_move(2, 2, 2, 2)
    for column, q12 in enumerate(left):
        probabilities = matrix[:, column] ** 2
        expected_uniform = np.sum(matrix[:, column]) / np.sqrt(len(matrix))
        np.testing.assert_allclose(
            table[2, 2, 2, 2, q12, 0], expected_uniform, atol=2.0e-14
        )
        np.testing.assert_allclose(np.sum(probabilities), 1.0, atol=2.0e-14)


def test_zero_simplex_head_reproduces_backbone_sampling_and_amplitudes() -> None:
    config = _config()
    params = init_many_cell_simplex_vit(jax.random.PRNGKey(7), config)
    key = jax.random.PRNGKey(8)
    expected = np.asarray(
        sample_many_cell_schur_vit(params["backbone"], key, 4, config.base_config)
    )
    actual = np.asarray(sample_many_cell_simplex_vit(params, key, 4, config))
    np.testing.assert_array_equal(actual, expected)
    for row in actual:
        validate_layout_state(config.simplex_layout.schur, row, target_q=0)
    expected_logs = np.asarray(
        many_cell_schur_log_amplitudes(
            params["backbone"], actual, config.base_config
        )
    )
    actual_logs = np.asarray(
        many_cell_simplex_log_amplitudes(params, actual, config)
    )
    np.testing.assert_allclose(actual_logs, expected_logs, atol=2.0e-12)
    audit = audit_sampled_local_f_moves(actual, config)
    assert audit["maximum_f_column_norm_error"] < 2.0e-14
    assert audit["f_move_column_count"] == 12


def test_zero_head_h_ratios_match_generic_tree_reference() -> None:
    config = _config()
    params = init_many_cell_simplex_vit(jax.random.PRNGKey(11), config)
    labels = np.asarray(
        sample_many_cell_simplex_vit(
            params, jax.random.PRNGKey(12), 2, config
        )
    )
    bonds = ((0, 1, 1.0), (1, 7, 0.125), (8, 14, -0.2), (18, 19, 0.7))
    workload = build_many_cell_tree_connectivity(
        labels, config.base_config, bonds
    )
    expected = evaluate_many_cell_tree_local_energy(
        params["backbone"], workload, config.base_config
    )
    actual = evaluate_many_cell_simplex_local_energy(
        {
            "simplex_logit": params["simplex_logit"],
            "simplex_root_logit": params["simplex_root_logit"],
            "inter_simplex_logit": params["inter_simplex_logit"],
            "simplex_phase": params["simplex_phase"],
        },
        params["backbone"],
        workload,
        config,
    )
    np.testing.assert_allclose(actual.ratios, expected.ratios, atol=3.0e-12)
    np.testing.assert_allclose(
        actual.local_energies, expected.local_energies, atol=3.0e-12
    )


def test_scramble_changes_geometry_not_su2_tree_or_parameter_shapes() -> None:
    physical = triangular_six_by_six_simplex_layout()
    scrambled = scramble_simplex_coordinates(
        physical, (4, 0, 7, 2, 8, 3, 6, 1, 5)
    )
    assert scrambled.schur.tree == physical.schur.tree
    assert scrambled.schur.site_order == physical.schur.site_order
    assert scrambled.pair_indices == physical.pair_indices
    physical_config = _config(physical)
    scrambled_config = _config(scrambled)
    assert not np.allclose(
        many_cell_simplex_geometry_features(physical_config),
        many_cell_simplex_geometry_features(scrambled_config),
    )
    left = init_many_cell_simplex_vit(jax.random.PRNGKey(13), physical_config)
    right = init_many_cell_simplex_vit(jax.random.PRNGKey(13), scrambled_config)
    assert [leaf.shape for leaf in jax.tree_util.tree_leaves(left)] == [
        leaf.shape for leaf in jax.tree_util.tree_leaves(right)
    ]


def test_additive_simplex_channel_exactly_nests_carrier_and_matches_ratios() -> None:
    config = _config()
    carrier_model = init_many_cell_simplex_vit(jax.random.PRNGKey(20), config)
    residual = init_many_cell_simplex_vit(jax.random.PRNGKey(21), config)
    carrier = carrier_model["backbone"]
    proposal = sample_many_cell_simplex_additive_proposal(
        carrier,
        residual,
        0.2,
        jax.random.PRNGKey(22),
        3,
        config.base_config,
        config,
        carrier_probability=0.4,
    )
    labels = np.asarray(proposal.labels)
    carrier_logs = np.asarray(
        many_cell_schur_log_amplitudes(carrier, labels, config.base_config)
    )
    residual_logs = np.asarray(
        many_cell_simplex_log_amplitudes(residual, labels, config)
    )
    expected_proposal = np.logaddexp(
        np.log(0.4) + 2.0 * carrier_logs.real,
        np.log(0.6) + 2.0 * residual_logs.real,
    )
    np.testing.assert_allclose(
        np.asarray(proposal.log_proposal_probability),
        expected_proposal,
        atol=3.0e-12,
    )
    bonds = ((0, 1, 1.0), (1, 7, 0.125), (18, 19, 0.7))
    workload = build_many_cell_tree_connectivity(
        labels, config.base_config, bonds
    )
    nested = evaluate_many_cell_simplex_additive_local_energy(
        carrier,
        residual,
        0.0,
        workload,
        config.base_config,
        config,
    )
    carrier_reference = evaluate_many_cell_tree_local_energy(
        carrier, workload, config.base_config
    )
    np.testing.assert_allclose(
        nested.local_energies, carrier_reference.local_energies, atol=3.0e-12
    )

    scale = 0.2
    additive = evaluate_many_cell_simplex_additive_local_energy(
        carrier,
        residual,
        scale,
        workload,
        config.base_config,
        config,
    )
    connected = workload.connected_labels.reshape((-1, config.n - 1))
    carrier_connected = np.exp(
        np.asarray(
            many_cell_schur_log_amplitudes(
                carrier, connected, config.base_config
            )
        )
    ).reshape(workload.mask.shape)
    residual_connected = np.exp(
        np.asarray(many_cell_simplex_log_amplitudes(residual, connected, config))
    ).reshape(workload.mask.shape)
    expected_center = np.exp(carrier_logs) + scale * np.exp(residual_logs)
    expected_connected = carrier_connected + scale * residual_connected
    expected_ratios = np.where(
        workload.mask, expected_connected / expected_center[:, None], 0.0
    )
    expected_local = np.sum(workload.coefficients * expected_ratios, axis=1)
    np.testing.assert_allclose(additive.center_amplitudes, expected_center, atol=3.0e-12)
    np.testing.assert_allclose(additive.ratios, expected_ratios, atol=3.0e-12)
    np.testing.assert_allclose(additive.local_energies, expected_local, atol=3.0e-12)
