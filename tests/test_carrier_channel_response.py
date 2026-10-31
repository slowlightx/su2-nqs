import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.carrier_channel_response import (
    exact_carrier_channel_directional_derivatives,
    split_plan_sources_by_shell,
    weighted_centered_projection,
    weighted_centered_projection_family,
)
from su2_nqs.benchmarks.exact_carrier_tangent import (
    exact_carrier_decoder_log_derivatives,
)
from su2_nqs.benchmarks.exact_causal_carrier import propagate_exact_causal_carrier
from su2_nqs.benchmarks.sparse_causal_recurrence import (
    compile_sparse_causal_recurrence_plan,
)
from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig
from su2_nqs.models.exact_carrier_tree import (
    ExactCarrierTreeGRUConfig,
    init_exact_carrier_tree_gru,
)
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.fusion_tree import FusionTree


jax.config.update("jax_enable_x64", True)


def _shells(n: int):
    nearest = tuple((site, site + 1, 1.0, 0) for site in range(n - 1))
    frustrated = tuple((site, site + 2, 0.37, 1) for site in range(n - 2))
    return nearest, frustrated


@pytest.mark.parametrize("n", (4, 6, 8))
def test_shell_carriers_add_to_combined_final_scalar(n: int):
    first, second = _shells(n)
    combined = compile_sparse_causal_recurrence_plan(n, first + second)
    first_plan = compile_sparse_causal_recurrence_plan(n, first)
    second_plan = compile_sparse_causal_recurrence_plan(n, second)
    np.testing.assert_allclose(
        [
            propagate_exact_causal_carrier(combined, path).closed_scalar[-1]
            for path in enumerate_paths(n, 0)
        ],
        [
            propagate_exact_causal_carrier(first_plan, path).closed_scalar[-1]
            + propagate_exact_causal_carrier(second_plan, path).closed_scalar[-1]
            for path in enumerate_paths(n, 0)
        ],
        atol=8.0e-13,
    )


def test_event_cut_delta_ledger_sums_to_exact_analytic_tangent():
    n = 6
    first, second = _shells(n)
    tree = FusionTree.left_associated(n)
    base = CompactTreeGRUConfig(tree, 0, embedding_dim=3, hidden_dim=5)
    combined_config = ExactCarrierTreeGRUConfig(
        base, compile_sparse_causal_recurrence_plan(n, first + second)
    )
    shell_configs = tuple(
        ExactCarrierTreeGRUConfig(base, plan)
        for plan in split_plan_sources_by_shell(
            combined_config.recurrence_plan
        ).values()
    )
    params = init_exact_carrier_tree_gru(jax.random.PRNGKey(17), combined_config)
    role_size = n * 3 * (base.hidden_dim + 1) + (n - 1) * (base.hidden_dim + 1)
    generator = np.random.default_rng(818)
    probability_direction = jnp.asarray(generator.normal(size=role_size))
    phase_direction = jnp.asarray(generator.normal(size=role_size))
    for path in enumerate_paths(n, 0):
        labels = jnp.asarray(path[1:])
        probability, phase = exact_carrier_decoder_log_derivatives(
            params, labels, combined_config
        )
        resolved = [
            exact_carrier_channel_directional_derivatives(
                params,
                labels,
                config,
                probability_direction,
                phase_direction,
            )
            for config in shell_configs
        ]
        np.testing.assert_allclose(
            sum(float(jnp.sum(item[0])) for item in resolved),
            float(probability @ probability_direction),
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            sum(float(jnp.sum(item[1])) for item in resolved),
            float(phase @ phase_direction),
            atol=2.0e-12,
        )


def test_weighted_projection_is_centered_pythagorean_and_gauge_invariant():
    generator = np.random.default_rng(91)
    features = generator.normal(size=(13, 5))
    weights = generator.uniform(size=13)
    target = generator.normal(size=13)
    first = weighted_centered_projection(features, weights, target)
    second = weighted_centered_projection(
        features + np.arange(5)[None, :], weights, target + 7.3
    )
    assert first.numerical_rank == 5
    np.testing.assert_allclose(first.fitted, second.fitted, atol=2.0e-13)
    assert first.target_norm_squared == pytest.approx(
        first.projected_norm_squared + first.residual_norm_squared, abs=2.0e-13
    )


def test_weighted_projection_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="rank-two"):
        weighted_centered_projection(np.ones(3), np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="positive and finite"):
        weighted_centered_projection(
            np.ones((3, 1)), np.ones(3), np.ones(3), relative_cutoff=0.0
        )


def test_projection_family_reuses_definition_at_each_cutoff():
    generator = np.random.default_rng(93)
    features = generator.normal(size=(11, 7))
    features[:, -1] *= 1.0e-9
    weights = generator.uniform(size=11)
    target = generator.normal(size=11)
    family = weighted_centered_projection_family(
        features, weights, target, relative_cutoffs=(1.0e-7, 1.0e-11)
    )
    for cutoff, observed in family.items():
        expected = weighted_centered_projection(
            features, weights, target, relative_cutoff=cutoff
        )
        np.testing.assert_allclose(observed.fitted, expected.fitted, atol=2.0e-13)
        assert observed.numerical_rank == expected.numerical_rank
