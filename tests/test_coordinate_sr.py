import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.coordinate_sr import (
    rescaled_real_sr_direction, diagonal_regularized_real_sr_direction,
)
from su2_nqs.models.clebsch_tree_transfer import calibrate_clebsch_parameters
from su2_nqs.models.literature_clebsch_tree import LiteratureClebschTreeConfig,init_literature_clebsch_tree,make_clebsch_amplitudes
from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis

jax.config.update('jax_enable_x64',True)


def test_rescaled_dual_solve_matches_independent_dense_metric():
    rng=np.random.default_rng(13); scores=rng.normal(size=(17,9)); energy=rng.normal(size=17)
    scale=np.exp(rng.normal(size=9)); ridge=.01
    direction,info=rescaled_real_sr_direction(scores,energy,scale,damping=ridge)
    centered=(scores/scale)-(scores/scale).mean(axis=0)
    metric=centered.T@centered/len(scores)
    gradient=2*centered.T@(energy-energy.mean())/len(scores)
    expected=-np.linalg.solve(metric+ridge*np.eye(9),gradient)/scale
    np.testing.assert_allclose(direction,expected,atol=2e-12,rtol=2e-12)
    assert float(info['relative_SR_residual']) < 2e-12
    assert float(np.dot(2*(scores-scores.mean(0)).T@(energy-energy.mean())/len(scores),direction)) < 0


def test_uniform_scaling_changes_effective_ridge_not_physical_definition():
    rng=np.random.default_rng(14); scores=rng.normal(size=(9,5));energy=rng.normal(size=9)
    a,_=rescaled_real_sr_direction(scores,energy,np.full(5,3.),damping=.01)
    b,_=rescaled_real_sr_direction(scores,energy,np.ones(5),damping=.09)
    np.testing.assert_allclose(a,b,rtol=3e-12,atol=3e-12)
    bad,info=rescaled_real_sr_direction(scores,energy,np.zeros(5),damping=.01)
    assert not bool(info['valid'])
    assert not np.isfinite(bad).any()
    with pytest.raises(ValueError):rescaled_real_sr_direction(scores,energy,np.ones(4),damping=.01)


@pytest.mark.parametrize('relative_shift', [0., .01, .1])
def test_feature_diagonal_regularizer_matches_independent_dense_full_fisher(relative_shift):
    rng = np.random.default_rng(19001)
    scores = rng.normal(size=(23, 9))*np.logspace(-2, 2, 9)
    scores[:, 0] = 1.  # An exact normalization gauge must stay inactive.
    energy = rng.normal(size=23)
    centered = scores-scores.mean(0)
    fisher = centered.T@centered/len(scores)
    gradient = 2*centered.T@(energy-energy.mean())/len(scores)
    metric = fisher+relative_shift*np.diag(np.diag(fisher))+.001*np.eye(9)
    expected = -np.linalg.solve(metric, gradient)
    direction, info = diagonal_regularized_real_sr_direction(scores, energy,
        relative_shift=relative_shift, absolute_shift=.001)
    np.testing.assert_allclose(direction, expected, rtol=3e-9, atol=3e-10)
    assert float(direction[0]) == 0.
    assert float(np.linalg.norm(metric@np.asarray(direction)+gradient)/np.linalg.norm(gradient)) < 2e-10
    assert float(gradient@direction) < 0
    assert bool(info['valid'])


def test_diagonal_regularizer_rejects_unregistered_or_complex_metric():
    for relative, absolute in [(-.1, .01), (.01, 0.), (float('nan'), .01)]:
        with pytest.raises(ValueError):
            diagonal_regularized_real_sr_direction(np.ones((4, 2)), np.ones(4),
                relative_shift=relative, absolute_shift=absolute)
    with pytest.raises(ValueError, match='real matrix'):
        diagonal_regularized_real_sr_direction(np.ones((4, 2), complex), np.ones(4),
            relative_shift=.01, absolute_shift=.001)


def test_actual_raw_and_coherent_projected_ray_and_score_coordinate_rule():
    from jax.flatten_util import ravel_pytree
    cfg=LiteratureClebschTreeConfig(FusionTree.balanced(8),multiplicity_cap=3,maximum_q=4)
    p,_=init_literature_clebsch_tree(jax.random.PRNGKey(15),cfg)
    x=np.asarray(fixed_m_spin_basis(8,0));balanced,report=calibrate_clebsch_parameters(cfg,p,x)
    scales=dict(merges=tuple(jnp.ones_like(v)*r['positive_scale'] for v,r in zip(p['merges'],report['layers'],strict=True)),
        readout=jnp.ones_like(p['readout']))
    d,_=ravel_pytree(scales);theta,unravel=ravel_pytree(p);newtheta,_=ravel_pytree(balanced)
    images=np.asarray([np.roll(np.arange(8),i) for i in range(8)])
    raw=make_clebsch_amplitudes(cfg)
    projected,_=make_projected_clebsch(cfg,tuple(range(8)),images,orbit_chunk=2,fold_root=False)
    for fn in (raw,projected):
        a=np.asarray(fn(p,x));b=np.asarray(fn(balanced,x))
        np.testing.assert_allclose(a/np.linalg.norm(a),b/np.linalg.norm(b),atol=3e-13)
        ids=np.flatnonzero((np.abs(a)>1e-10)&(np.abs(b)>1e-10))[:4]
        gradient=jax.grad(lambda t,row:jnp.log(jnp.abs(fn(unravel(t),row[None])[0])))
        first=jax.vmap(gradient,in_axes=(None,0))(theta,x[ids])
        second=jax.vmap(gradient,in_axes=(None,0))(newtheta,x[ids])
        np.testing.assert_allclose(second,first/d,rtol=3e-11,atol=3e-11)
