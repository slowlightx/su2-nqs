import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.sparse_causal_recurrence import (
    compile_sparse_causal_recurrence_plan,
)
from su2_nqs.benchmarks.sparse_causal_tangent import (
    exact_message_head_tangent_diagnostics,
)
from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig
from su2_nqs.models.sparse_causal_tree import (
    SparseCausalTreeGRUConfig,
    init_sparse_causal_tree_gru,
    sparse_causal_tree_log_psi,
)
from su2_nqs.su2.fusion_tree import FusionTree


def test_cut_message_rhs_sums_to_full_head_rhs_and_is_finite():
    n = 6
    bonds = tuple((site, site + 1, 1.0, 0) for site in range(n - 1)) + tuple(
        (site, site + 2, 0.47, 1) for site in range(n - 2)
    )
    plan = compile_sparse_causal_recurrence_plan(n, bonds)
    base = CompactTreeGRUConfig(
        FusionTree.left_associated(n), 0, embedding_dim=3, hidden_dim=7
    )
    config = SparseCausalTreeGRUConfig(base, plan, message_hidden_dim=8)
    params = init_sparse_causal_tree_gru(jax.random.PRNGKey(551), config)
    states = jnp.asarray(
        [state.labels for state in config.tree.enumerate_states(0)]
    )
    logs = jax.vmap(
        lambda state: sparse_causal_tree_log_psi(params, state, config)
    )(states)
    weights = jnp.exp(2.0 * jnp.real(logs))
    target = jnp.asarray(
        np.linspace(-0.21, 0.27, len(states))
        + 1j * np.linspace(0.18, -0.16, len(states))
    )
    result = exact_message_head_tangent_diagnostics(
        params, config, states, weights, target
    )

    assert result["all_finite"]
    assert result["maximum_cut_sum_error"] < 2.0e-13
    assert result["aggregate_rhs_norms"]["total"] > 1.0e-8
    assert len(result["cut_rhs_norms"]) == n - 1
    assert all(row["total"] >= 0.0 for row in result["cut_rhs_norms"])

