import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    init_compact_tree_gru,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.adaptive_tree import (
    adaptive_compact_tree_vmc_energy_gradient_from_connectivity,
    build_bucketed_tree_connectivity,
    compact_tree_vmc_energy_gradient_auto,
    sample_compact_tree_data_parallel,
)
from su2_nqs.vmc.tree_local_energy import (
    build_padded_tree_connectivity,
    padded_compact_tree_importance_vmc_energy_gradient,
    padded_compact_tree_vmc_energy_gradient,
)


def _problem():
    tree = FusionTree.from_nested(((0, 1), (2, (3, (4, 5)))))
    config = CompactTreeGRUConfig(tree, 0, 3, 5)
    params = init_compact_tree_gru(jax.random.PRNGKey(1907), config)
    labels = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    bonds = (
        (0, 1, 1.0),
        (1, 2, 0.7),
        (2, 3, -0.2),
        (3, 4, 0.5),
        (4, 5, 1.1),
        (5, 0, 0.9),
        (1, 4, 0.3),
    )
    return config, params, labels, bonds


def _assert_gradients_close(observed, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(observed),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual_leaf, expected_leaf, atol=2.0e-11, rtol=2.0e-11
        )


@pytest.mark.parametrize(
    "weighting", ("uniform", "exhaustive_born", "importance")
)
@pytest.mark.parametrize("microbatch_size", (1, 3))
def test_adaptive_gradient_matches_legacy_for_every_weighting_and_microbatch(
    weighting, microbatch_size
):
    config, params, labels, bonds = _problem()
    legacy_connectivity = build_padded_tree_connectivity(
        labels, config, bonds, maximum_width=64
    )
    bucketed = build_bucketed_tree_connectivity(
        labels, config, bonds, registered_widths=(4, 8, 16, 32, 64)
    )
    importance = None
    if weighting == "importance":
        logs = jax.vmap(
            lambda state: compact_tree_log_psi(params, state, config)
        )(jnp.asarray(labels))
        importance = np.asarray(jnp.exp(2.0 * jnp.real(logs))) + 0.07
        expected = padded_compact_tree_importance_vmc_energy_gradient(
            params, legacy_connectivity, config, importance
        )
    else:
        expected = padded_compact_tree_vmc_energy_gradient(
            params,
            legacy_connectivity,
            config,
            born_weighted=(weighting == "exhaustive_born"),
        )
    actual = adaptive_compact_tree_vmc_energy_gradient_from_connectivity(
        params,
        bucketed,
        config,
        weighting=weighting,
        importance_weights=importance,
        microbatch_size=microbatch_size,
        devices=(jax.local_devices()[0],),
    )
    np.testing.assert_allclose(actual.energy, expected.energy, atol=2.0e-11)
    np.testing.assert_allclose(actual.variance, expected.variance, atol=2.0e-11)
    np.testing.assert_allclose(
        actual.enumerated_normalization,
        expected.enumerated_normalization,
        atol=2.0e-11,
    )
    _assert_gradients_close(actual.gradient, expected.gradient)
    assert actual.execution.selected_mode == "adaptive"
    assert actual.execution.real_sample_count == len(labels)
    assert actual.execution.allocated_connectivity_slots < (
        len(labels) * 64
    )
    assert actual.execution.dummy_sample_count >= 0


def test_auto_preserves_legacy_cpu_path_and_explicit_adaptive_is_available():
    config, params, labels, bonds = _problem()
    automatic = compact_tree_vmc_energy_gradient_auto(
        params,
        labels,
        config,
        bonds,
        execution_mode="auto",
        registered_widths=(4, 8, 16, 32, 64),
    )
    if jax.local_devices()[0].platform == "cpu":
        assert automatic.execution.selected_mode == "legacy"
    forced = compact_tree_vmc_energy_gradient_auto(
        params,
        labels,
        config,
        bonds,
        execution_mode="adaptive",
        registered_widths=(4, 8, 16, 32, 64),
        microbatch_size=3,
        devices=(jax.local_devices()[0],),
    )
    np.testing.assert_allclose(forced.energy, automatic.energy, atol=2.0e-11)


def test_bucket_registry_fails_closed_instead_of_truncating_connectivity():
    config, _, labels, bonds = _problem()
    with pytest.raises(ValueError, match="exceeds largest registered width"):
        build_bucketed_tree_connectivity(
            labels, config, bonds, registered_widths=(1,)
        )


def test_parallel_sampler_matches_global_index_stream_across_microbatches():
    config, params, _, _ = _problem()
    key = jax.random.PRNGKey(601)
    first = sample_compact_tree_data_parallel(
        params,
        key,
        7,
        config,
        global_index_start=13,
        microbatch_size=2,
        devices=(jax.local_devices()[0],),
    )
    second = sample_compact_tree_data_parallel(
        params,
        key,
        7,
        config,
        global_index_start=13,
        microbatch_size=5,
        devices=(jax.local_devices()[0],),
    )
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.global_indices, np.arange(13, 20))
    assert first.dummy_sample_count == 1
    assert second.dummy_sample_count == 3


def test_two_and_four_device_nondivisible_batch_worker():
    worker = Path(__file__).with_name("_adaptive_tree_multidevice_worker.py")
    environment = os.environ.copy()
    flags = environment.get("XLA_FLAGS", "")
    forced = "--xla_force_host_platform_device_count=4"
    environment["XLA_FLAGS"] = f"{flags} {forced}".strip()
    environment["JAX_PLATFORM_NAME"] = "cpu"
    environment["JAX_ENABLE_X64"] = "1"
    completed = subprocess.run(
        [sys.executable, str(worker)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
