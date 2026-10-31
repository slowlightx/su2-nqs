"""Independent algebraic checks for static, whole-multiplet capacity actions."""
import itertools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.capacity_attention import (
    CapacityControllerConfig, init_capacity_controller,
    predict_capacity_coefficients, capacity_quadratic_loss,
)
from su2_nqs.benchmarks.projected_capacity import (
    budgeted_multiplet_actions, tangent_moments,
    restricted_natural_direction, heldout_direction_metrics,
    conditional_greedy_actions,
)


def test_physical_moments_are_gauge_centered_and_complex_correct():
    rng = np.random.default_rng(74120)
    scores = rng.normal(size=(48, 7))+1j*rng.normal(size=(48, 7))
    energies = rng.normal(size=48)+1j*rng.normal(size=48)
    gram, force = tangent_moments(scores, energies)
    shifted = tangent_moments(scores+(3+7j), energies+18-2j)
    np.testing.assert_allclose(gram, shifted[0], atol=3e-15)
    np.testing.assert_allclose(force, shifted[1], atol=3e-15)
    d = rng.normal(size=7)
    np.testing.assert_allclose(d @ gram @ d, np.var(scores @ d), atol=2e-14)
    direct = 2*np.real(np.vdot(scores @ d-(scores @ d).mean(), energies-energies.mean()))/48
    np.testing.assert_allclose(2*force @ d, direct, atol=2e-14)


def test_force_convention_agrees_with_exact_normalized_energy_derivative():
    # Enumerate a complete real basis, using explicit Born weights by
    # replicating equally weighted amplitudes. Independent dense H reference.
    rng = np.random.default_rng(74124)
    psi = np.ones(12)/np.sqrt(12)
    h = rng.normal(size=(12, 12)); h = (h+h.T)/2
    tangent = rng.normal(size=(12, 4))
    local_energy = (h @ psi)/psi
    gram, force = tangent_moments(tangent, local_energy)
    direction = rng.normal(size=4)
    delta = psi*(tangent @ direction)

    def energy(step):
        state = psi+step*delta
        return state @ h @ state/(state @ state)

    finite_difference = (energy(1e-6)-energy(-1e-6))/(2e-6)
    np.testing.assert_allclose(2*force @ direction, finite_difference, atol=2e-9)
    from projects.clebsch_tree_nqs.workflows.triangle_projected_capacity import tangent_moments as producer_moments
    producer_metric, producer_derivative = producer_moments(tangent, local_energy)
    np.testing.assert_allclose(producer_metric, gram, atol=2e-15)
    np.testing.assert_allclose(producer_derivative, 2*force, atol=2e-15)


def test_exact_budget_selector_matches_exhaustive_full_multiplet_control():
    merits = np.array([.8, .7, .4, 1.3, .2, .9])
    costs = np.array([12, 7, 10, 11, 3, 9])
    qs = [1, 3, 1, 5, 0, 2]
    mask = np.array([True, True, False, True, True, True])
    got = budgeted_multiplet_actions(merits, costs, qs, native_budget=24,
                                     magnetic_budget=7, allowed=mask)
    candidates = []
    for bits in itertools.product((0, 1), repeat=6):
        ids = tuple(np.flatnonzero(bits))
        if any(not mask[i] for i in ids):
            continue
        native, magnetic = sum(costs[i] for i in ids), sum(qs[i]+1 for i in ids)
        if native <= 24 and magnetic <= 7:
            candidates.append((-sum(merits[i] for i in ids), native, magnetic, ids))
    assert tuple(got) == min(candidates)[3]
    assert not len(budgeted_multiplet_actions(merits, costs, qs, native_budget=0, magnetic_budget=7))
    with pytest.raises(TypeError):
        budgeted_multiplet_actions(merits, costs, [1., 3, 1, 5, 0, 2], native_budget=24, magnetic_budget=7)


def test_restricted_direction_has_exact_requested_fs_and_only_selected_support():
    rng = np.random.default_rng(74121)
    x = rng.normal(size=(60, 8))
    gram = x.T @ x/60
    force = rng.normal(size=8)
    direction = restricted_natural_direction(gram, force, [1, 4, 6], fs_radius=.001)
    np.testing.assert_array_equal(direction[[0, 2, 3, 5, 7]], 0)
    result = heldout_direction_metrics(direction, gram, force)
    assert result["energy_slope_total"] < 0
    np.testing.assert_allclose(result["fs_tangent"], .001, atol=3e-18)
    # Coefficient-coordinate rescaling cannot manufacture a physical gain.
    scale = np.exp(rng.normal(size=8))
    transformed = restricted_natural_direction(gram*scale[:, None]*scale[None, :],
        force*scale, [1, 4, 6], fs_radius=.001)
    np.testing.assert_allclose(transformed*scale, direction, atol=2e-17)


def test_conditional_control_does_not_buy_the_same_residual_twice():
    # Actions 0 and 1 are exact duplicates. Independent merits would choose
    # both, while the conditional full-metric control keeps the independent 2.
    gram = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
    force = np.array([1., 1., .7])
    selected = conditional_greedy_actions(gram, force, [3, 3, 3], [1, 1, 1],
        native_budget=6, magnetic_budget=4, ridge=0.)
    np.testing.assert_array_equal(selected, [0, 2])


@pytest.mark.parametrize("mixing", ("linear", "bilinear", "pointwise", "pooled", "signed_graph", "attention"))
def test_static_controller_starts_identically_masks_before_mixing_and_is_equivariant(mixing):
    rng = np.random.default_rng(74122)
    config = CapacityControllerConfig(6, width=4, relation_dim=2, mixing=mixing)
    params = init_capacity_controller(jax.random.PRNGKey(74122), config)
    x = jnp.asarray(rng.normal(size=(9, 6)))
    allowed = jnp.asarray([True]*8+[False])
    relations = jnp.asarray(rng.normal(size=(9, 9, 2)))
    np.testing.assert_array_equal(predict_capacity_coefficients(params, x, config, allowed=allowed, relations=relations), 0)
    params["readout"] = jnp.ones_like(params["readout"])*.1
    output = predict_capacity_coefficients(params, x, config, allowed=allowed, relations=relations)
    assert output[-1] == 0
    corrupted = x.at[-1].set(1e20)
    np.testing.assert_allclose(output, predict_capacity_coefficients(params, corrupted, config, allowed=allowed, relations=relations), atol=2e-15)
    permutation = np.array([4, 2, 8, 0, 1, 7, 3, 6, 5])
    permuted = predict_capacity_coefficients(params, x[permutation], config,
        allowed=allowed[permutation], relations=relations[permutation][:, permutation])
    np.testing.assert_allclose(permuted, output[permutation], atol=2e-15)
    grad = jax.grad(lambda p: capacity_quadratic_loss(predict_capacity_coefficients(
        p, x, config, allowed=allowed, relations=relations), jnp.eye(9), jnp.arange(9.)))(params)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves(grad))


def test_attention_uses_other_tokens_but_pointwise_control_does_not():
    x = jnp.asarray(np.random.default_rng(74123).normal(size=(6, 5)))
    changes = []
    for mode in ("pointwise", "attention"):
        config = CapacityControllerConfig(5, mixing=mode)
        params = init_capacity_controller(jax.random.PRNGKey(74123), config)
        params["readout"] = jnp.ones(4)*.1
        before = predict_capacity_coefficients(params, x, config)
        after = predict_capacity_coefficients(params, x.at[5].add(2), config)
        changes.append(float(abs(after[0]-before[0])))
    assert changes[0] == 0
    assert changes[1] > 1e-7


def test_signed_graph_preserves_screening_sign_and_zero_rows_are_safe():
    config = CapacityControllerConfig(5, mixing="signed_graph", relation_dim=1)
    params = init_capacity_controller(jax.random.PRNGKey(74126), config)
    params["readout"] = jnp.ones(4)*.1
    x = jnp.asarray(np.random.default_rng(74126).normal(size=(6, 5)))
    relation = jnp.zeros((6, 6, 1)).at[0, 3, 0].set(2.)
    plus = predict_capacity_coefficients(params, x, config, relations=relation)
    minus = predict_capacity_coefficients(params, x, config, relations=-relation)
    assert abs(float(plus[0]-minus[0])) > 1e-7
    np.testing.assert_allclose(plus[1:], minus[1:], atol=2e-15)
    assert np.isfinite(plus).all() and np.isfinite(minus).all()


def test_producer_cache_relations_and_feature_names_cannot_be_silently_dropped(tmp_path):
    import json
    from projects.clebsch_tree_nqs.workflows.projected_capacity_attention import prepare_panel
    rng = np.random.default_rng(74125)
    path = tmp_path/"dataset.npz"
    relations = rng.normal(size=(3, 3, 4))
    np.savez(path, scores=rng.normal(size=(16, 3)), local_energies=rng.normal(size=16),
        fit_mask=np.arange(16) < 12, chain_labels=np.repeat(np.arange(8), 2),
        token_features=rng.normal(size=(3, 2)), token_cost=np.array([2, 3, 4]),
        token_q=np.array([1, 3, 5]), token_nodes=np.arange(3), pair_relations=relations)
    (tmp_path/"registration.json").write_text(json.dumps({"feature_names": ["q", "mode_weight"], "j2": .125}))
    panel = prepare_panel(path)
    np.testing.assert_array_equal(panel["token_pair_relations"], relations)
    assert list(panel["feature_names"]) == ["q", "mode_weight", "j2_over_j1"]
    np.testing.assert_array_equal(panel["token_features"][:, -1], .125)


def test_screening_pool_respects_signed_context_q_kind_and_whole_node_permutation():
    from projects.clebsch_tree_nqs.workflows.projected_capacity_attention import _screening_feature_means
    rng = np.random.default_rng(74127)
    panel = dict(token_nodes=np.repeat(np.arange(4), 6),
                 token_q=np.tile([1, 1, 3, 3, 5, 5], 4),
                 token_kind=np.tile([0, 1, 0, 1, 0, 1], 4))
    screening = rng.normal(size=(24, 24))
    original = _screening_feature_means(panel, screening)
    np.testing.assert_allclose(_screening_feature_means(panel, -screening), -original, atol=2e-15)
    own_edge = panel["token_nodes"][:, None] == panel["token_nodes"][None, :]
    np.testing.assert_array_equal(_screening_feature_means(panel, screening+1e7*own_edge), original)
    permutation = np.arange(24).reshape(4, 6)[[3, 1, 0, 2]].ravel()
    moved = {name: value[permutation] for name, value in panel.items()}
    np.testing.assert_allclose(_screening_feature_means(moved, screening[permutation][:, permutation]),
                               original[permutation], atol=2e-15)
