"""Independent budget, full-state gradient and symmetry-reuse checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.bounded_exact import train_bounded_exact
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_full_vit_budget import prepare_objective


def test_bounded_optimizer_improves_and_replays_checkpoint(tmp_path):
    initial = {"x": jnp.array([3., -2.])}
    loss = lambda p: jnp.sum((p["x"]-jnp.array([1., 2.]))**2)
    kwargs = dict(loss_fn=loss, identity={"test": 1}, checkpoint=tmp_path/"ckpt.pkl",
                  adam_steps=5, lbfgs_steps=10, seconds_per_stage=20., record_every=1)
    params, report = train_bounded_exact(initial, **kwargs)
    assert report["best_loss"] < 1e-12
    assert report["adam_completed"] == 5
    assert report["evaluations"] >= 7
    assert report["adam_best_loss"] > report["best_loss"]
    with pytest.raises(FileExistsError):
        train_bounded_exact(initial, **kwargs)
    replay, resumed = train_bounded_exact(initial, resume=True, **kwargs)
    np.testing.assert_allclose(replay["x"], params["x"], atol=1e-10)
    assert resumed["resumed"]
    assert resumed["evaluations"] == report["evaluations"]
    with pytest.raises(ValueError, match="contract"):
        train_bounded_exact(initial, resume=True, **{**kwargs, "adam_steps": 6})


def test_resume_during_lbfgs_preserves_adam_boundary(tmp_path):
    initial = {"x": jnp.array([-1.2, 1.])}
    def loss(p):
        x,y=p["x"]
        return (1-x)**2 + 100*(y-x*x)**2
    def interrupt(p,row):
        if row["stage"]=="lbfgs" and row["step"]==3:
            raise RuntimeError("simulated interruption")
        return {}
    kwargs=dict(loss_fn=loss,identity={},checkpoint=tmp_path/"partial.pkl",
                adam_steps=3,lbfgs_steps=20,seconds_per_stage=20.,record_every=1)
    with pytest.raises(RuntimeError,match="simulated"):
        train_bounded_exact(initial,callback=interrupt,**kwargs)
    boundary=load_checkpoint(kwargs["checkpoint"])["metadata"]
    assert boundary["lbfgs_completed"]==2
    _,report=train_bounded_exact(initial,resume=True,**kwargs)
    assert report["adam_terminal_loss"]==boundary["adam_terminal_loss"]
    assert report["adam_best_loss"]==boundary["adam_best_loss"]
    assert report["lbfgs_curvature_restarted"]
    assert len([r for r in report["history"] if r["stage"]=="adam_terminal"])==1


def test_near_zero_time_budget_is_reported_not_hidden(tmp_path):
    p, r = train_bounded_exact({"x": jnp.array([3.])}, loss_fn=lambda p: jnp.sum(p["x"]**2),
        identity={}, checkpoint=tmp_path/"tiny.pkl", adam_steps=100, lbfgs_steps=100,
        seconds_per_stage=1e-12)
    assert r["adam_completed"] == 1
    assert r["adam_stop"] == "time"
    assert r["lbfgs_stop"] == "time"
    assert np.isfinite(p["x"]).all()


@pytest.mark.parametrize("geometry", ["square", "triangular"])
def test_coarse_reuse_and_reduced_H_equal_full_physical_action(geometry):
    case = dict(geometry=geometry, rows=2, columns=2, j2_over_j1=.125,
                exact_energy_anchor=-10.)
    data = _prepare_case(case, "fixed_m0")
    config = FixedSzSpatialViTConfig(geometry=geometry, rows=2, columns=2,
        patch_rows=1, patch_columns=2, embedding_dim=4, head_count=2)
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(42), config)
    problem = prepare_objective(data, config)
    assert max(problem["validate"](params).values()) < 1e-12
    leaves, structure = jax.tree_util.tree_flatten(params)
    tangent = jax.tree_util.tree_unflatten(structure, [jnp.ones_like(x)*.01 for x in leaves])
    value, derivative = jax.jvp(problem["loss"], (params,), (tangent,))
    shifted = jax.tree_util.tree_map(lambda p,t: p+1e-4*t, params, tangent)
    opposite = jax.tree_util.tree_map(lambda p,t: p-1e-4*t, params, tangent)
    finite_difference = (problem["loss"](shifted)-problem["loss"](opposite))/2e-4
    np.testing.assert_allclose(derivative, finite_difference, atol=2e-8, rtol=1e-5)
    assert np.isfinite(value)
