from __future__ import annotations

from math import exp
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.sz_gru import SzGRUConfig, init_sz_gru
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.sz_proposal_preflight import (
    ProposalFitState,
    ProposalStreamCursors,
    ProposalStreamSeeds,
    fit_step_from_log_target,
    full_support_sz_mixture_log_probabilities,
    initialize_proposal_adam,
    load_latest_proposal_fit_checkpoint,
    log_weight_diagnostics,
    m_doubling_state_weight_audit,
    sample_indexed_full_support_sz_mixture,
    save_proposal_fit_checkpoint_generation,
)


def _config() -> SzGRUConfig:
    return SzGRUConfig(n=6, n_up=3, embedding_dim=3, hidden_dim=5)


def _params():
    return init_sz_gru(jax.random.key(1701), _config())


def _identity(seeds: ProposalStreamSeeds) -> dict[str, object]:
    return {
        "graph": {"name": "six-site-test", "bonds_sha256": "a" * 64},
        "group": {"name": "identity", "images_sha256": "b" * 64},
        "bank": {"kind": "synthetic", "parameters_sha256": "c" * 64},
        "proposal": {"epsilon": 0.1, "n": 6, "n_up": 3},
        "source_sha256": {"module": "d" * 64, "runner": "e" * 64},
        "stream_seeds": seeds.as_dict(),
    }


def _synthetic_root_logs(states: jax.Array) -> jax.Array:
    bits = jnp.asarray(states, dtype=jnp.float64)
    positions = jnp.arange(bits.shape[1], dtype=jnp.float64)
    first = 0.35 * (bits @ positions) + 0.2 * bits[:, 0] * bits[:, -1]
    second = -0.22 * (bits @ positions) + 0.4 * bits[:, 1] * bits[:, 3]
    return jnp.stack((first, second), axis=1)


def _assert_tree_identical(first: object, second: object) -> None:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for left, right in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_full_support_mixture_is_exactly_normalized_and_zero_off_sector() -> None:
    config = _config()
    params = _params()
    basis = jnp.asarray(enumerate_sz_configs(config.n, config.n_up))
    logs = full_support_sz_mixture_log_probabilities(
        params, basis, config, epsilon=0.13
    )
    np.testing.assert_allclose(np.sum(np.exp(np.asarray(logs))), 1.0, atol=2.0e-14)
    assert np.all(np.isfinite(np.asarray(logs)))
    invalid = jnp.asarray([[0] * config.n, [0, 0, 0, 1, 1, 2]])
    assert np.all(np.isneginf(np.asarray(full_support_sz_mixture_log_probabilities(
        params, invalid, config, epsilon=0.13
    ))))


def test_indexed_sampling_is_chunk_invariant_and_matches_evaluated_q() -> None:
    config = _config()
    params = _params()
    key = jax.random.key(1702)
    whole = sample_indexed_full_support_sz_mixture(
        params,
        key,
        start_index=19,
        n_samples=1200,
        config=config,
        epsilon=0.17,
    )
    first = sample_indexed_full_support_sz_mixture(
        params,
        key,
        start_index=19,
        n_samples=431,
        config=config,
        epsilon=0.17,
    )
    second = sample_indexed_full_support_sz_mixture(
        params,
        key,
        start_index=450,
        n_samples=769,
        config=config,
        epsilon=0.17,
    )
    np.testing.assert_array_equal(
        whole.configurations,
        np.concatenate((np.asarray(first.configurations), np.asarray(second.configurations))),
    )
    np.testing.assert_array_equal(
        whole.uniform_component,
        np.concatenate((np.asarray(first.uniform_component), np.asarray(second.uniform_component))),
    )
    np.testing.assert_array_equal(whole.sample_indices, np.arange(19, 1219))
    np.testing.assert_allclose(
        whole.log_probabilities,
        full_support_sz_mixture_log_probabilities(
            params, whole.configurations, config, epsilon=0.17
        ),
        atol=0.0,
    )
    assert np.all(np.asarray(whole.configurations).sum(axis=1) == config.n_up)
    assert abs(float(np.mean(np.asarray(whole.uniform_component))) - 0.17) < 0.04


def test_log_weight_and_nested_m_doubling_diagnostics_are_stable() -> None:
    extreme = log_weight_diagnostics(
        np.asarray([1000.0, 999.0, 997.0, -np.inf]), tail_fraction=0.25
    )
    assert np.isfinite(extreme.log_normalizer)
    assert 0.0 < extreme.effective_sample_size_fraction <= 1.0
    assert extreme.positive_weight_count == 3

    count = 64
    proposal = np.log(np.linspace(1.0, 2.0, 2 * count))
    proposal -= np.log(np.sum(np.exp(proposal)))
    roots = np.stack((proposal + 2.0, proposal - 3.0), axis=1)
    audit = m_doubling_state_weight_audit(
        roots,
        proposal,
        base_sample_count=count,
        root_log_normalizers=np.asarray([2.0, -3.0]),
        tail_fraction=0.05,
        minimum_ess_fraction=0.999999999,
        maximum_ratio_to_mean=1.000000001,
        maximum_top_tail_mass=0.06,
        maximum_absolute_log_normalizer_shift=1.0e-12,
        standard_error_multiplier=3.0,
    )
    assert audit["pass"]
    assert audit["doubled_2M"]["equal_normalized_bank"][
        "effective_sample_size_fraction"
    ] == pytest.approx(1.0)


def test_step_checkpoint_resume_preserves_parameters_and_all_stream_cursors(
    tmp_path: Path,
) -> None:
    config = _config()
    seeds = ProposalStreamSeeds(fit=1801, validation=1802, audit=1803)
    params = _params()
    initial = ProposalFitState(
        params=params,
        optimizer_state=initialize_proposal_adam(params),
        completed_steps=0,
        cursors=ProposalStreamCursors(),
        history=(),
    )
    step_keywords = {
        "stream_seeds": seeds,
        "config": config,
        "epsilon": 0.1,
        "fit_batch_size": 24,
        "validation_batch_size": 20,
        "learning_rate": 0.01,
        "target_log_numerators": _synthetic_root_logs,
        "tail_fraction": 0.1,
    }
    continuous_one = fit_step_from_log_target(initial, **step_keywords)
    continuous_two = fit_step_from_log_target(continuous_one, **step_keywords)

    identity = _identity(seeds)
    save_proposal_fit_checkpoint_generation(
        tmp_path,
        continuous_one,
        run_identity=identity,
        stream_seeds=seeds,
    )
    resumed_one = load_latest_proposal_fit_checkpoint(
        tmp_path, run_identity=identity, stream_seeds=seeds
    )
    resumed_two = fit_step_from_log_target(resumed_one, **step_keywords)
    _assert_tree_identical(continuous_two.params, resumed_two.params)
    _assert_tree_identical(
        continuous_two.optimizer_state, resumed_two.optimizer_state
    )
    assert continuous_two.cursors == ProposalStreamCursors(
        fit=48, validation=40, audit=0
    )
    assert resumed_two.cursors == continuous_two.cursors
    assert resumed_two.history == continuous_two.history
    assert all(row["audit_stream_cursor"] == 0 for row in resumed_two.history)

    changed = dict(identity)
    changed["graph"] = {"name": "changed", "bonds_sha256": "f" * 64}
    with pytest.raises(ValueError, match="identity does not match"):
        load_latest_proposal_fit_checkpoint(
            tmp_path, run_identity=changed, stream_seeds=seeds
        )


def test_stream_contract_and_checkpoint_fail_closed() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ProposalStreamSeeds(fit=1, validation=1, audit=2)
    with pytest.raises(ValueError, match="between zero and one"):
        full_support_sz_mixture_log_probabilities(
            _params(), jnp.zeros((1, 6)), _config(), epsilon=0.0
        )
    row = log_weight_diagnostics(np.asarray([0.0, 0.0]))
    assert exp(row.log_normalizer) == pytest.approx(1.0)
