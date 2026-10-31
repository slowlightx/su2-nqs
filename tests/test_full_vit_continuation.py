"""Checkpoint fidelity and untouched-chain statistics for the full ViT baseline."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_full_vit_pilot import (
    CONTINUATION_CONFIG, audit_statistics, make_sampled_adam_update,
    sampled_transition, validate_parent,
)
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from su2_nqs.benchmarks.spatial_vit_sampling import (
    make_full_heisenberg_ratios, make_log_exchange_sampler, projected_log_function,
)
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from su2_nqs.vmc.adam import _manual_adam_init, _manual_adam_update
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.frontier.artifacts import pytree_sha256


def _small_problem():
    config = FixedSzSpatialViTConfig(geometry="square",rows=2,columns=2,
        patch_rows=1,patch_columns=1,embedding_dim=4,head_count=2,depth=2)
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(15),config)
    data = _prepare_case(dict(geometry="square",rows=2,columns=2,
        j2_over_j1=.5,exact_energy_anchor=-10.),"fixed_m0")
    fn,_ = projected_log_function(config,data["groups"]["gamma_a1"])
    log_batch = jax.jit(jax.vmap(fn,in_axes=(None,0)))
    local = make_full_heisenberg_ratios(fn,data["bonds"])
    sampler = make_log_exchange_sampler(fn,8)
    spec = dict(learning_rate=.0003,gradient_clip_norm=1.)
    updater = make_sampled_adam_update(log_batch,spec)
    state = dict(model=params,key=jax.random.PRNGKey(14),
        chains=initialize_spin_exchange_chains(jax.random.PRNGKey(13),n_chains=16,n_sites=4,n_up=2))
    return state,local,sampler,updater,log_batch,data,spec


def test_checkpoint_continuation_replays_weights_moments_chains_and_key(tmp_path):
    state,local,sample,update,_,_,_ = _small_problem()
    start = copy.deepcopy(state)
    moments = _manual_adam_init(state["model"])
    for _ in range(6):
        state,moments,*_ = sampled_transition(state,moments,sample,local,update)
    expected = pytree_sha256((state,moments))
    moments = _manual_adam_init(start["model"])
    for _ in range(2):
        start,moments,*_ = sampled_transition(start,moments,sample,local,update)
    file = tmp_path/"checkpoint.pkl"
    save_checkpoint(file,start,optimizer_state=moments,metadata={"step":2})
    saved = load_checkpoint(file)
    start,moments = saved["params"],saved["optimizer_state"]
    # Compilation warm replay must be discarded, not advance optimizer/RNG.
    sampled_transition(start,moments,sample,local,update)
    # Resume in a fresh interpreter, not just through a same-process pickle.
    subprocess.run([sys.executable,"-c",r'''
import runpy,sys
m=runpy.run_path("tests/test_full_vit_continuation.py")
_,local,sample,update,_,_,_=m["_small_problem"]()
saved=m["load_checkpoint"](sys.argv[1])
state,moments=saved["params"],saved["optimizer_state"]
for _ in range(4):
    state,moments,*_=m["sampled_transition"](state,moments,sample,local,update)
m["save_checkpoint"](sys.argv[1],state,optimizer_state=moments,metadata={"step":6})
''',str(file)],check=True,cwd=Path(__file__).resolve().parents[1],timeout=90)
    resumed=load_checkpoint(file)
    assert pytree_sha256((resumed["params"],resumed["optimizer_state"])) == expected
    assert int(resumed["optimizer_state"][0]) == 6
    assert np.all(np.asarray(resumed["params"]["chains"]).sum(axis=1)==2)


def test_complex_vmc_update_matches_direct_covariance_gradient():
    state,local,_,update,logs,data,spec = _small_problem()
    p = state["model"]
    x = jnp.asarray(data["basis"])
    energies = local(p,x)
    # Independent real/imag derivative form, not the conjugated-log expression.
    centered = jax.lax.stop_gradient(energies-energies.mean())
    def covariance(candidate):
        z = logs(candidate,x)
        return 2*jnp.mean(z.real*centered.real+z.imag*centered.imag)
    grad = jax.grad(covariance)(p)
    norm = jnp.sqrt(sum(jnp.sum(v*v) for v in jax.tree_util.tree_leaves(grad)))
    grad = jax.tree_util.tree_map(lambda v:v*jnp.minimum(1.,1./jnp.maximum(norm,1e-30)),grad)
    expected,_ = _manual_adam_update(p,grad,_manual_adam_init(p),spec["learning_rate"])
    actual,_,_ = update(p,_manual_adam_init(p),x,energies)
    for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a,b,atol=1e-13,rtol=1e-13)


def test_statistics_resample_chains_not_correlated_snapshots():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(3,8,12))+1j*rng.normal(size=(3,8,12))
    result = audit_statistics(values,4,bootstrap_seed=41)
    assert result["variance_per_site"] == pytest.approx(np.mean(abs(values-values.mean())**2)/4)
    repeated = audit_statistics(np.repeat(values,2,axis=1),4,bootstrap_seed=41)
    assert repeated["chain_mean_standard_error"] == pytest.approx(result["chain_mean_standard_error"])
    np.testing.assert_allclose(repeated["energy_chain_bootstrap_95"],result["energy_chain_bootstrap_95"])
    assert result["independent_chains"] == 36
    assert not result["mixing_certified"]
    values[0,0,0] = np.nan
    with pytest.raises(ValueError,match="finite"):
        audit_statistics(values,4,bootstrap_seed=1)


def test_parent_identity_and_fixed_precommitted_budget():
    spec = json.loads(CONTINUATION_CONFIG.read_text())
    assert spec["additional_steps"] == 1000
    seen = []
    for geometry in ("triangular","square"):
        parent,result = validate_parent(spec,geometry)
        assert int(parent["optimizer_state"][0]) == 64
        assert result["parameter_count"] == 4032
        for seeds in spec["audit"]["seeds"][geometry].values():
            assert len(seeds) == 3
            seen.extend(seeds)
    assert len(set(seen)) == 12
    bad = copy.deepcopy(spec)
    bad["learning_rate"] *= 2
    with pytest.raises(ValueError,match="learning_rate"):
        validate_parent(bad,"triangular")
    bad = copy.deepcopy(spec)
    bad["parents"]["square"]["checkpoint_sha256"] = "stale"
    with pytest.raises(ValueError,match="parent artifacts"):
        validate_parent(bad,"square")


def test_nonfinite_proposal_fails_before_checkpoint_replacement():
    state,local,sample,update,_,_,_ = _small_problem()
    original = pytree_sha256(state)
    def invalid(p,x):
        return jnp.full(len(x),jnp.nan+0j)
    with pytest.raises(FloatingPointError,match="nonfinite"):
        sampled_transition(state,_manual_adam_init(state["model"]),sample,invalid,update)
    assert pytree_sha256(state) == original
