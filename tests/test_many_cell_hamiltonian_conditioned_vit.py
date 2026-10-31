from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.hamiltonian_conditioned_tree import HamiltonianBond
from su2_nqs.models.many_cell_hamiltonian_conditioned_vit import (
    ManyCellHamiltonianConditionedViTConfig,
    condition_many_cell_schur_vit,
    many_cell_hamiltonian_conditioned_log_amplitudes,
    many_cell_hamiltonian_conditioned_head_log_amplitudes,
    prepare_many_cell_hamiltonian_conditioned_head_design,
    sample_many_cell_hamiltonian_conditioned_vit,
)
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout


def _model():
    layout = rectangular_four_spin_layout(2, 2, triangular_coordinates=True)
    base_config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    conditioned_config = ManyCellHamiltonianConditionedViTConfig(
        base_config,
        (
            HamiltonianBond(0, 1, 1.0, 0),
            HamiltonianBond(0, 3, 0.2, 1),
        ),
        n_channels=2,
        local_embedding_dim=3,
        local_hidden_dim=5,
    )
    base = init_many_cell_schur_vit(jax.random.PRNGKey(4), base_config)
    params = condition_many_cell_schur_vit(
        base, jax.random.PRNGKey(5), conditioned_config
    )
    labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    return base_config, conditioned_config, base, params, labels


def test_zero_output_conditioning_head_preserves_base_amplitudes() -> None:
    base_config, config, base, params, labels = _model()
    expected = many_cell_schur_log_amplitudes(base, jnp.asarray(labels), base_config)
    actual = many_cell_hamiltonian_conditioned_log_amplitudes(
        params, jnp.asarray(labels), config
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2.0e-14)


def test_conditioned_probabilities_remain_exhaustively_normalized() -> None:
    _, config, _, params, labels = _model()
    params["local"]["probability_head"] = params["local"][
        "probability_head"
    ].at[0, 0].set(0.37)
    params["local"]["phase_head"] = params["local"]["phase_head"].at[1, 1].set(-0.2)
    logs = many_cell_hamiltonian_conditioned_log_amplitudes(
        params, jnp.asarray(labels), config
    )
    normalization = float(jnp.sum(jnp.exp(2.0 * jnp.real(logs))))
    assert abs(normalization - 1.0) < 2.0e-12
    assert np.all(np.isfinite(np.asarray(logs)))


def test_cached_active_head_design_is_exact_for_finite_heads() -> None:
    _, config, base, params, labels = _model()
    params["local"]["probability_head"] = 0.2 * jax.random.normal(
        jax.random.PRNGKey(31), params["local"]["probability_head"].shape
    )
    params["local"]["phase_head"] = 0.2 * jax.random.normal(
        jax.random.PRNGKey(37), params["local"]["phase_head"].shape
    )
    expected = many_cell_hamiltonian_conditioned_log_amplitudes(
        params, jnp.asarray(labels), config
    )
    design = prepare_many_cell_hamiltonian_conditioned_head_design(
        base, params["local"], jnp.asarray(labels), config
    )
    actual = many_cell_hamiltonian_conditioned_head_log_amplitudes(
        {
            "probability_head": params["local"]["probability_head"],
            "phase_head": params["local"]["phase_head"],
        },
        design,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2.0e-14)


def test_conditioned_sampler_matches_exhaustive_born_probabilities() -> None:
    _, config, _, params, labels = _model()
    params["local"]["probability_head"] = params["local"][
        "probability_head"
    ].at[0, 0].set(0.55)
    logs = many_cell_hamiltonian_conditioned_log_amplitudes(
        params, jnp.asarray(labels), config
    )
    expected = np.exp(2.0 * np.real(np.asarray(logs)))
    samples = np.asarray(
        sample_many_cell_hamiltonian_conditioned_vit(
            params, jax.random.PRNGKey(19), 20000, config
        )
    )
    lookup = {tuple(row): index for index, row in enumerate(labels)}
    counts = np.zeros(len(labels), dtype=np.int64)
    for row in samples:
        counts[lookup[tuple(row)]] += 1
    observed = counts / len(samples)
    np.testing.assert_allclose(observed, expected, atol=0.012)
