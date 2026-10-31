from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.sz_gru import SzGRUConfig, init_sz_gru
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.induced_three_carrier_proposal import (
    CARRIER_LABELS,
    FourStreamCursors,
    FourStreamSeeds,
)
from su2_nqs.vmc.induced_three_carrier_proposal_mixture import (
    exact_root_mixture_log_probabilities,
    fit_root_mixture_step,
    initialize_root_mixture_calibration,
    load_latest_root_mixture_checkpoint,
    sample_indexed_exact_root_mixture,
    save_root_mixture_checkpoint_generation,
)


def _config() -> SzGRUConfig:
    return SzGRUConfig(n=6, n_up=3, embedding_dim=3, hidden_dim=5)


def _params() -> dict[str, object]:
    return {
        label: init_sz_gru(jax.random.key(40100 + root), _config())
        for root, label in enumerate(CARRIER_LABELS)
    }


def _target(states: jax.Array) -> np.ndarray:
    bits = jnp.asarray(states, dtype=jnp.float64)
    positions = jnp.arange(bits.shape[1], dtype=jnp.float64)
    return np.asarray(
        jnp.stack(
            (
                0.15 * (bits @ positions),
                -0.12 * (bits @ positions) + 0.2 * bits[:, 0],
                0.08 * (bits @ positions[::-1]) - 0.3 * bits[:, -1],
            ),
            axis=1,
        )
    )


def _identity(seeds: FourStreamSeeds) -> dict[str, object]:
    return {
        "carrier_bank": {"labels": list(CARRIER_LABELS)},
        "proposal": {"kind": "three-root-test"},
        "source_sha256": {"module": "c" * 64},
        "config_sha256": "d" * 64,
        "stream_seeds": seeds.as_dict(),
    }


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for first, second in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_exact_root_mixture_is_normalized_and_sampling_is_chunk_invariant() -> None:
    config = _config()
    params = _params()
    basis = np.asarray(enumerate_sz_configs(6, 3), dtype=np.int8)
    logs = exact_root_mixture_log_probabilities(params, basis, config, 0.1)
    assert float(np.sum(np.exp(np.asarray(logs)))) == pytest.approx(
        1.0, abs=2.0e-14
    )
    key = jax.random.key(40110)
    whole = sample_indexed_exact_root_mixture(
        params, key, start_index=17, n_samples=100, config=config, epsilon=0.1
    )
    first = sample_indexed_exact_root_mixture(
        params, key, start_index=17, n_samples=37, config=config, epsilon=0.1
    )
    second = sample_indexed_exact_root_mixture(
        params, key, start_index=54, n_samples=63, config=config, epsilon=0.1
    )
    np.testing.assert_array_equal(
        whole.configurations,
        np.concatenate((first.configurations, second.configurations)),
    )
    np.testing.assert_array_equal(
        whole.proposal_components,
        np.concatenate((first.proposal_components, second.proposal_components)),
    )
    np.testing.assert_allclose(
        whole.log_probabilities,
        exact_root_mixture_log_probabilities(
            params, whole.configurations, config, 0.1
        ),
        atol=0.0,
    )


def test_root_mixture_resume_is_bitwise_identical(tmp_path: Path) -> None:
    config = _config()
    params = _params()
    seeds = FourStreamSeeds(
        calibration=40120, fit=40121, validation=40122, audit=40123
    )
    initial = initialize_root_mixture_calibration(
        params,
        stream_seeds=seeds,
        config=config,
        epsilon=0.1,
        calibration_base_sample_count=32,
        target_log_numerators=_target,
        tail_fraction=0.1,
    )
    keywords = {
        "stream_seeds": seeds,
        "config": config,
        "epsilon": 0.1,
        "fit_batch_size_per_root": 20,
        "validation_batch_size": 24,
        "learning_rate": 0.01,
        "target_log_numerators": _target,
        "tail_fraction": 0.1,
    }
    continuous_one = fit_root_mixture_step(initial, **keywords)
    continuous_two = fit_root_mixture_step(continuous_one, **keywords)
    identity = _identity(seeds)
    save_root_mixture_checkpoint_generation(
        tmp_path,
        continuous_one,
        run_identity=identity,
        stream_seeds=seeds,
    )
    resumed_one = load_latest_root_mixture_checkpoint(
        tmp_path, run_identity=identity, stream_seeds=seeds
    )
    resumed_two = fit_root_mixture_step(resumed_one, **keywords)
    _assert_tree_equal(continuous_two.params_by_root, resumed_two.params_by_root)
    _assert_tree_equal(
        continuous_two.optimizer_state_by_root,
        resumed_two.optimizer_state_by_root,
    )
    assert resumed_two.cursors == FourStreamCursors(
        calibration=64, fit=40, validation=48, audit=0
    )
    assert resumed_two.cursors == continuous_two.cursors
    assert resumed_two.history == continuous_two.history
