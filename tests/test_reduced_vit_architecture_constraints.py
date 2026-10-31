"""Independent diagnostics of the native Schur-ViT architecture.

These checks document parameterization and locality priors, not a failure of
the fixed-CG singlet basis or normalized autoregressive sampling.  In
particular, a locality table aligned to the input history would be a different
inductive bias; it is not a correction to an SU(2) identity.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.compact_tree_gru import split_tables
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    _candidate_features,
    _candidate_scores,
    _global_configuration_phase,
    _support_torus_distances,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    sample_many_cell_schur_vit,
    causal_history_geometry,
    _context_tokens,
)
from su2_nqs.models.schur_initialization import deepen_many_cell_schur_vit
from su2_nqs.models.tree_gru import tree_decisions
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.su2.tree_rotations import leaf_signature


@pytest.mark.parametrize(
    "side,width,depth,channel_count",
    [(4, 8, 1, 5), (4, 16, 8, 5), (6, 16, 4, 9), (6, 32, 8, 9)],
)
def test_root_born_logits_remain_quadratic_when_backbone_grows(
    side, width, depth, channel_count
):
    """At target doubled q=0, all root scores lie in span{1,q,q²}."""
    config = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(side, side),
        embedding_dim=width,
        head_count=2,
        depth=depth,
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(120 + side), config)
    left, right, valid = split_tables(config.layout.tree)
    mask = valid[0, 0]
    q = left[0, 0, mask].astype(np.float64)
    np.testing.assert_array_equal(q, right[0, 0, mask])
    features = _candidate_features(config)[0, 0, mask]
    assert len(q) == channel_count
    assert np.linalg.matrix_rank(features, tol=1e-12) == 3
    assert np.linalg.matrix_rank(features - features[0], tol=1e-12) == 2

    # Arbitrary backbone context makes the statement independent of the
    # initialized network's depth, attention weights, or hidden width.
    context = jax.random.normal(
        jax.random.PRNGKey(440 + side), (1, 1, width), dtype=jnp.float64
    )
    scores = np.asarray(
        _candidate_scores(params, context, jnp.asarray(features[None, None]), config)
    )[0, 0]
    polynomial = np.stack((np.ones_like(q), q, q * q), axis=1)
    fitted = polynomial @ np.linalg.lstsq(polynomial, scores, rcond=None)[0]
    np.testing.assert_allclose(scores, fitted, atol=5e-13, rtol=0)
    probabilities = np.asarray(jax.nn.softmax(jnp.asarray(scores)))
    log_ratios = np.log(probabilities / probabilities[0])
    centered_basis = polynomial[:, 1:] - polynomial[0, 1:]
    fitted_ratios = centered_basis @ np.linalg.lstsq(
        centered_basis, log_ratios, rcond=None
    )[0]
    np.testing.assert_allclose(log_ratios, fitted_ratios, atol=5e-13, rtol=0)


def test_global_phase_is_not_confined_to_root_quadratic_logit_family():
    """The nonlinear bidirectional phase map can distinguish root channels.

    This is a scalar-network feature probe, not a batch of valid Hilbert
    configurations: all non-root features are deliberately held fixed.  It
    establishes that the root Born-score rank restriction is not also a
    degree-two restriction on the global phase map.
    """
    config = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(4, 4),
        embedding_dim=8,
        head_count=2,
        depth=1,
        initialization_scale=0.8,
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(88), config)
    left, _, valid = split_tables(config.layout.tree)
    mask = valid[0, 0]
    q = left[0, 0, mask].astype(np.float64)
    chosen = np.zeros((len(q), config.decision_count, 8), dtype=np.float64)
    chosen[:, 0] = _candidate_features(config)[0, 0, mask]
    phase = np.asarray(_global_configuration_phase(params, jnp.asarray(chosen), config))
    polynomial = np.stack((np.ones_like(q), q, q * q), axis=1)
    residual = phase - polynomial @ np.linalg.lstsq(polynomial, phase, rcond=None)[0]
    assert np.max(np.abs(residual)) > 1e-7


def test_zero_first_order_phase_force_does_not_certify_correct_signs():
    """A real positive state can be a phase saddle of a real Hamiltonian."""
    # An independent two-channel example: the positive state has E=+1,
    # while a relative pi phase gives the exact ground state with E=-1.
    hamiltonian = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float64)

    def energy(relative_phase):
        coefficients = jnp.asarray([jnp.exp(1j * relative_phase), 1.0]) / np.sqrt(2.0)
        return jnp.real(jnp.vdot(coefficients, hamiltonian @ coefficients))

    np.testing.assert_allclose(jax.grad(energy)(0.0), 0.0, atol=2e-14, rtol=0)
    np.testing.assert_allclose(jax.grad(jax.grad(energy))(0.0), -1.0, atol=2e-14, rtol=0)
    np.testing.assert_allclose(energy(0.0) - energy(np.pi), 2.0, atol=2e-14, rtol=0)


def test_depth_four_to_eight_preserves_all_n16_amplitudes_and_normalization():
    """The proposed size extension starts from the identical complete state."""
    source = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(4, 4),
        embedding_dim=16,
        head_count=2,
        depth=4,
        direct_phase_initialization_scale=0.1,
    )
    target = replace(source, depth=8)
    params = init_many_cell_schur_vit(jax.random.PRNGKey(122), source)
    grown = deepen_many_cell_schur_vit(params, source, target, jax.random.PRNGKey(123))
    labels = np.asarray([state.labels for state in source.layout.tree.enumerate_states(0)])
    assert labels.shape == (1430, 15)
    old_logs = np.asarray(many_cell_schur_log_amplitudes(params, labels, source))
    new_logs = np.asarray(many_cell_schur_log_amplitudes(grown, labels, target))
    np.testing.assert_allclose(new_logs, old_logs, atol=2e-14, rtol=0)
    np.testing.assert_allclose(np.exp(2 * new_logs.real).sum(), 1.0, atol=2e-13, rtol=0)
    assert sum(leaf.size for leaf in jax.tree_util.tree_leaves(grown)) == 33770
    for name in ("layers", "global_layers"):
        for layer in grown[name][source.depth:]:
            np.testing.assert_array_equal(layer["output"], 0.0)
            np.testing.assert_array_equal(layer["mlp_out"], 0.0)


def _independent_support_distances(config):
    """Direct rectangular-torus distances of physical leaf supports."""
    side = int(np.sqrt(config.n))
    assert side * side == config.n
    decisions = tree_decisions(config.layout.tree)
    supports = [
        tuple(
            config.layout.site_order[leaf]
            for leaf in leaf_signature(config.layout.tree.internal_nodes[decision.node_index])
        )
        for decision in decisions
    ]

    def site_distance(first, second):
        y, x = divmod(first, side)
        yy, xx = divmod(second, side)
        basis = (
            np.asarray([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2]])
            if config.geometry == "triangular" else np.eye(2)
        )
        return min(
            np.linalg.norm(basis @ np.asarray([xx - x + dx * side, yy - y + dy * side]))
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        )

    return np.asarray(
        [[min(site_distance(a, b) for a in left for b in right) for right in supports]
         for left in supports],
        dtype=np.float64,
    )


@pytest.mark.parametrize("side", [4, 6])
@pytest.mark.parametrize("geometry", ["square", "triangular"])
def test_causal_input_history_and_prediction_supports_are_distinct(side, geometry):
    """Token j contains action j-1, but current locality keys name decision j.

    A prospective no-parameter alternative shifts only key columns of the
    causal locality table.  The BOS column has no action support and is set
    to zero.  The global configuration phase must retain the unshifted table.
    Both choices preserve the causal mask and normalized AR definition.
    """
    config = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(side, side),
        embedding_dim=8,
        head_count=2,
        depth=1,
        geometry=geometry,
    )
    independent = _independent_support_distances(config)
    current = _support_torus_distances(config)
    np.testing.assert_allclose(current, independent, atol=2e-14, rtol=0)
    history_aligned = np.zeros_like(independent)
    history_aligned[:, 1:] = independent[:, :-1]
    causal = np.tril(np.ones_like(current, dtype=bool))
    assert np.max(np.abs((current - history_aligned)[causal])) >= 1.0
    assert not np.array_equal(history_aligned, history_aligned.T)
    np.testing.assert_array_equal(history_aligned[:, 0], 0.0)


def test_aligned_joint_scoring_breaks_the_legacy_root_probability_rank():
    config=ManyCellSchurViTConfig(rectangular_four_spin_layout(6,6),embedding_dim=16,
        head_count=2,depth=1,initialization_scale=1.3,conditional_mode='aligned_joint')
    params=init_many_cell_schur_vit(jax.random.PRNGKey(125),config)
    left,_,valid=split_tables(config.layout.tree);mask=valid[0,0]
    q=left[0,0,mask].astype(float)
    f=jnp.asarray(_candidate_features(config)[0,0,mask][None,None])
    context=jax.random.normal(jax.random.PRNGKey(126),(1,1,16),dtype=jnp.float64)
    scores=np.asarray(_candidate_scores(params,context,f,config))[0,0]
    basis=np.stack([np.ones_like(q),q,q*q],axis=1)
    residual=scores-basis@np.linalg.lstsq(basis,scores,rcond=None)[0]
    assert np.max(abs(residual))>1e-5
    jac=np.asarray(jax.jacrev(lambda w:_candidate_scores({**params,'logit_feature':w},context,f,config)[0,0])
        (params['logit_feature']))
    assert np.linalg.matrix_rank(jac[1:]-jac[:1],tol=1e-11)==8


def test_aligned_joint_history_geometry_and_causality_are_explicit():
    config=ManyCellSchurViTConfig(rectangular_four_spin_layout(4,4),embedding_dim=8,
        head_count=2,depth=1,conditional_mode='aligned_joint')
    params=init_many_cell_schur_vit(jax.random.PRNGKey(127),config)
    _,distance=causal_history_geometry(config)
    expected=_independent_support_distances(config)
    np.testing.assert_allclose(distance[:,1:],expected[:,:-1],atol=2e-14,rtol=0)
    np.testing.assert_array_equal(distance[:,0],0.)
    chosen=jax.random.normal(jax.random.PRNGKey(128),(1,15,8),dtype=jnp.float64)
    following=chosen.at[:,3:].add(1.7)
    before=_context_tokens(params,chosen,config);after=_context_tokens(params,following,config)
    np.testing.assert_allclose(before[:,:4],after[:,:4],atol=1e-13,rtol=0)
    assert np.max(abs(np.asarray(before[:,4:]-after[:,4:])))>1e-5


def test_aligned_joint_native_normalization_frequency_and_singlet_remain_exact():
    from su2_nqs.su2.recoupling import tree_basis_matrix
    from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
    config=ManyCellSchurViTConfig(rectangular_four_spin_layout(4,2),embedding_dim=8,
        head_count=2,depth=1,conditional_mode='aligned_joint',direct_phase_initialization_scale=.7)
    params=init_many_cell_schur_vit(jax.random.PRNGKey(129),config)
    labels=np.array([s.labels for s in config.layout.tree.enumerate_states(0)])
    coefficients=np.exp(np.asarray(many_cell_schur_log_amplitudes(params,labels,config)))
    probability=abs(coefficients)**2
    np.testing.assert_allclose(probability.sum(),1.,atol=4e-13,rtol=0)
    u,_=tree_basis_matrix(config.layout.tree,0,0);s2,_=spin_squared_matrix(8)
    np.testing.assert_allclose(s2@(u@coefficients),0.,atol=8e-13,rtol=0)
    draws=np.asarray(jax.jit(lambda k:sample_many_cell_schur_vit(params,k,4096,config))(jax.random.PRNGKey(130)))
    frequencies=np.array([np.mean(np.all(draws==row,axis=1)) for row in labels])
    assert abs(frequencies-probability).max()<.04
    phase_grad=jax.grad(lambda p:jnp.sum(many_cell_schur_log_amplitudes(p,labels,config).imag))(params)
    assert np.linalg.norm(phase_grad['layers'][0]['output'])>1e-8


def test_neural_state_migration_loss_removes_only_global_phase_gauge():
    from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_revision import joint_log_state_matching_loss
    reference=jnp.array([.5*jnp.log(.2)+.1j,.5*jnp.log(.8)+2.2j])
    np.testing.assert_allclose(joint_log_state_matching_loss(reference+7j,reference),0.,atol=2e-14,rtol=0)
    candidate=reference+jnp.array([.3j,-.2j])
    assert float(joint_log_state_matching_loss(candidate,reference))>.03
    gradient=jax.grad(lambda p:joint_log_state_matching_loss(reference+1j*p,reference))(jnp.array([.3,-.2]))
    np.testing.assert_allclose(gradient.sum(),0.,atol=2e-14,rtol=0)
