from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    compressed_collective_tower_a1,
    projected_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.models.sz_gru import SzGRUConfig, init_sz_gru
from su2_nqs.vmc.constrained_uniform import sample_uniform_sz_configs
from su2_nqs.vmc.induced_three_carrier_proposal import (
    CARRIER_LABELS,
    FourStreamCursors,
    FourStreamSeeds,
    carrier_support_diagnostics,
    evaluate_n36_three_carrier_batch,
    fit_induced_proposal_step,
    initialize_calibrated_fit_state,
    load_latest_induced_proposal_checkpoint,
    save_induced_proposal_checkpoint_generation,
)


def _synthetic_three_root_logs(states: jax.Array) -> np.ndarray:
    bits = jnp.asarray(states, dtype=jnp.float64)
    positions = jnp.arange(bits.shape[1], dtype=jnp.float64)
    first = 0.17 * (bits @ positions)
    second = -0.11 * (bits @ positions) + 0.3 * bits[:, 0] * bits[:, -1]
    third = 0.09 * (bits @ positions[::-1]) - 0.2 * bits[:, 1] * bits[:, 3]
    return np.asarray(jnp.stack((first, second, third), axis=1))


def _identity(seeds: FourStreamSeeds) -> dict[str, object]:
    return {
        "carrier_bank": {"labels": list(CARRIER_LABELS), "n": 6},
        "proposal": {"kind": "test", "epsilon": 0.1},
        "source_sha256": {"module": "a" * 64},
        "config_sha256": "b" * 64,
        "stream_seeds": seeds.as_dict(),
    }


def _assert_tree_bitwise_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for first, second in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_n36_target_batch_is_exactly_the_registered_three_carrier_bank() -> None:
    states = np.asarray(
        sample_uniform_sz_configs(jax.random.key(32101), 17, 36, 18),
        dtype=np.int8,
    )
    orbit = build_n36_pi_zero_flux_orbit()
    observed = evaluate_n36_three_carrier_batch(
        states, orbit=orbit, microbatch_size=8
    )
    dsl = projected_pi_zero_flux_a1_amplitudes(
        states, orbit=orbit, microbatch_size=8
    )
    tower = compressed_collective_tower_a1(states, family="120")
    stripe = compressed_collective_tower_a1(states, family="stripe")
    expected = np.stack((dsl.amplitudes, tower.amplitudes, stripe.amplitudes), axis=1)
    np.testing.assert_array_equal(observed.amplitudes, expected)
    np.testing.assert_array_equal(observed.exact_nodes, expected == 0.0)
    finite = expected != 0.0
    np.testing.assert_allclose(
        observed.log_state_numerators[finite],
        2.0 * np.log(np.abs(expected[finite])),
        atol=0.0,
    )
    support = carrier_support_diagnostics(
        observed, near_cancellation_threshold=1.0e-10
    )
    assert support["labels"] == list(CARRIER_LABELS)
    assert support["sample_count"] == len(states)
    assert support["union_support_fraction"] == pytest.approx(1.0)


def test_four_stream_seed_and_target_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="distinct"):
        FourStreamSeeds(calibration=1, fit=1, validation=2, audit=3)
    invalid = np.zeros((2, 36), dtype=np.int8)
    with pytest.raises(ValueError, match="N_up=18"):
        evaluate_n36_three_carrier_batch(invalid)


def test_calibrated_fit_checkpoint_resume_is_bitwise_identical(
    tmp_path: Path,
) -> None:
    config = SzGRUConfig(n=6, n_up=3, embedding_dim=3, hidden_dim=5)
    params = init_sz_gru(jax.random.key(32200), config)
    seeds = FourStreamSeeds(
        calibration=32201, fit=32202, validation=32203, audit=32204
    )
    initialize_keywords = {
        "stream_seeds": seeds,
        "config": config,
        "epsilon": 0.1,
        "calibration_base_sample_count": 32,
        "target_log_numerators": _synthetic_three_root_logs,
        "tail_fraction": 0.1,
    }
    step_keywords = {
        "stream_seeds": seeds,
        "config": config,
        "epsilon": 0.1,
        "fit_batch_size": 24,
        "validation_batch_size": 20,
        "learning_rate": 0.01,
        "target_log_numerators": _synthetic_three_root_logs,
        "tail_fraction": 0.1,
    }
    continuous_zero = initialize_calibrated_fit_state(
        params, **initialize_keywords
    )
    continuous_one = fit_induced_proposal_step(
        continuous_zero, **step_keywords
    )
    continuous_two = fit_induced_proposal_step(
        continuous_one, **step_keywords
    )

    identity = _identity(seeds)
    generation_zero = save_induced_proposal_checkpoint_generation(
        tmp_path,
        continuous_zero,
        run_identity=identity,
        stream_seeds=seeds,
    )
    generation_one = save_induced_proposal_checkpoint_generation(
        tmp_path,
        continuous_one,
        run_identity=identity,
        stream_seeds=seeds,
    )
    assert generation_zero != generation_one
    assert generation_zero.is_file() and generation_one.is_file()
    resumed_one = load_latest_induced_proposal_checkpoint(
        tmp_path, run_identity=identity, stream_seeds=seeds
    )
    resumed_two = fit_induced_proposal_step(resumed_one, **step_keywords)
    _assert_tree_bitwise_equal(continuous_two.params, resumed_two.params)
    _assert_tree_bitwise_equal(
        continuous_two.optimizer_state, resumed_two.optimizer_state
    )
    np.testing.assert_array_equal(
        continuous_two.frozen_root_log_normalizers,
        resumed_two.frozen_root_log_normalizers,
    )
    assert resumed_two.cursors == FourStreamCursors(
        calibration=64, fit=48, validation=40, audit=0
    )
    assert resumed_two.cursors == continuous_two.cursors
    assert resumed_two.history == continuous_two.history
    assert all(row["audit_stream_cursor"] == 0 for row in resumed_two.history)

    changed = dict(identity)
    changed["proposal"] = {"kind": "changed", "epsilon": 0.1}
    with pytest.raises(ValueError, match="identity does not match"):
        load_latest_induced_proposal_checkpoint(
            tmp_path, run_identity=changed, stream_seeds=seeds
        )
