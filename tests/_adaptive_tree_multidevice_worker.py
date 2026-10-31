"""Subprocess-only multi-device correctness gate for adaptive tree VMC."""

import jax
import numpy as np

from su2_nqs.models.compact_tree_gru import (
    CompactTreeGRUConfig,
    init_compact_tree_gru,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.adaptive_tree import (
    adaptive_compact_tree_vmc_energy_gradient_from_connectivity,
    build_bucketed_tree_connectivity,
    sample_compact_tree_data_parallel,
)
from su2_nqs.vmc.tree_local_energy import (
    build_padded_tree_connectivity,
    padded_compact_tree_importance_vmc_energy_gradient,
    padded_compact_tree_vmc_energy_gradient,
)


def main() -> None:
    devices = tuple(jax.local_devices())
    if len(devices) != 4:
        raise RuntimeError(f"expected four virtual devices, found {devices!r}")
    tree = FusionTree.balanced(6)
    config = CompactTreeGRUConfig(tree, 0, 2, 4)
    params = init_compact_tree_gru(jax.random.PRNGKey(317), config)
    labels = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    # Five real samples force one dummy on two devices and three on four.
    bonds = tuple((site, (site + 1) % 6, 1.0) for site in range(6)) + (
        (0, 3, 0.4),
        (1, 4, -0.2),
    )
    legacy_connectivity = build_padded_tree_connectivity(
        labels, config, bonds, maximum_width=16
    )
    bucketed = build_bucketed_tree_connectivity(
        labels, config, bonds, registered_widths=(4, 8, 16)
    )
    importance = np.asarray((1.0e-3, 0.02, 1.0, 9.0, 200.0))
    samples_two = sample_compact_tree_data_parallel(
        params,
        jax.random.PRNGKey(941),
        5,
        config,
        global_index_start=17,
        microbatch_size=2,
        devices=devices[:2],
    )
    samples_four = sample_compact_tree_data_parallel(
        params,
        jax.random.PRNGKey(941),
        5,
        config,
        global_index_start=17,
        microbatch_size=1,
        devices=devices,
    )
    np.testing.assert_array_equal(samples_two.labels, samples_four.labels)
    np.testing.assert_array_equal(
        samples_two.global_indices, samples_four.global_indices
    )
    for weighting in ("uniform", "exhaustive_born", "importance"):
        if weighting == "importance":
            expected = padded_compact_tree_importance_vmc_energy_gradient(
                params, legacy_connectivity, config, importance
            )
            supplied = importance
        else:
            expected = padded_compact_tree_vmc_energy_gradient(
                params,
                legacy_connectivity,
                config,
                born_weighted=(weighting == "exhaustive_born"),
            )
            supplied = None
        for selected in (devices[:2], devices):
            actual = adaptive_compact_tree_vmc_energy_gradient_from_connectivity(
                params,
                bucketed,
                config,
                weighting=weighting,
                importance_weights=supplied,
                microbatch_size=1,
                devices=selected,
            )
            np.testing.assert_allclose(
                actual.energy, expected.energy, atol=2.0e-11
            )
            np.testing.assert_allclose(
                actual.variance, expected.variance, atol=2.0e-11
            )
            np.testing.assert_allclose(
                actual.enumerated_normalization,
                expected.enumerated_normalization,
                atol=2.0e-11,
            )
            for observed, reference in zip(
                jax.tree_util.tree_leaves(actual.gradient),
                jax.tree_util.tree_leaves(expected.gradient),
                strict=True,
            ):
                np.testing.assert_allclose(observed, reference, atol=2.0e-11)
            assert actual.execution.device_count == len(selected)
            assert actual.execution.real_sample_count == 5
            assert actual.execution.dummy_sample_count == (
                (len(selected) - 5 % len(selected)) % len(selected)
            )


if __name__ == "__main__":
    main()
