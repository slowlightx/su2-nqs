"""Exact frozen-parent cache oracles; no CT result files or accelerators needed."""
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.cached_amplitude_operators import (
    build_cached_amplitude_jacobian, cached_derivative_memory_plan,
    cached_projected_operators, load_cached_amplitude_jacobian,
    solve_cached_projected_sr,
)
from su2_nqs.vmc.importance_amplitude_sr import solve_projected_block_sr
from su2_nqs.vmc.matrix_free_lsqr import damped_lsqr


def fixture(row_batch=2, path=None):
    rng = np.random.default_rng(26103191)
    jac = rng.normal(size=(7, 11))
    theta = rng.normal(size=11)
    psi = jac@theta
    psi[2] = 0.  # exact target node, but all eleven derivatives remain live
    shift = psi-jac@theta
    rows = np.arange(7, dtype=np.int32)[:, None]
    q = np.exp(rng.normal(size=7))
    hp = rng.normal(size=7)
    batch = lambda x, r: jnp.asarray(jac)[r[:, 0]]@x+jnp.asarray(shift)[r[:, 0]]
    cache = build_cached_amplitude_jacobian(batch, jnp.asarray(theta), rows, psi, q,
                row_batch_size=row_batch, maximum_host_bytes=10**7,
                source_identity="independent-linear-node-oracle", storage_path=path)
    return cache, theta, rows, psi, q, hp, jac, batch


@pytest.mark.parametrize("row_batch", [1, 2, 4])
def test_cached_full_native_derivatives_and_host_dual_match_independent_oracle(row_batch):
    cache, theta, rows, psi, q, hp, jac, _ = fixture(row_batch)
    np.testing.assert_allclose(cache.jacobian, jac, atol=2e-15, rtol=2e-15)
    cache.verify(theta, rows, psi, q, source_identity="independent-linear-node-oracle")
    scale = 1/np.sqrt(q*np.sum(psi**2/q))
    u = psi*scale
    a = (np.eye(len(psi))-np.outer(u, u))@(jac*scale[:, None])
    energy = np.sum(psi*hp/q)/np.sum(psi**2/q)
    force = 2*(hp-energy*psi)*scale
    damping = .23
    expected = np.linalg.solve(a.T@a+damping*np.eye(len(theta)), a.T@force)
    direction, gradient, info = solve_cached_projected_sr(cache, psi, hp, q,
        damping=damping, parameter_chunk_size=4, maximum_device_bytes=10**7)
    np.testing.assert_allclose(direction, expected, atol=5e-13, rtol=5e-13)
    np.testing.assert_allclose(gradient, a.T@force, atol=5e-13, rtol=5e-13)
    np.testing.assert_allclose(info["chosen_direction_tangent"], a@expected, atol=5e-13)
    assert info["metric_is_F_not_two_F"] and info["no_AD_recomputed_in_solve"]
    assert info["relative_linear_residual"] < 1e-12
    assert info["exact_node_records"] == 1
    legacy, legacy_gradient, legacy_info = solve_projected_block_sr(
        jnp.asarray(jac), jnp.asarray(psi), jnp.asarray(hp), jnp.asarray(q),
        damping=damping, parameter_chunk_size=4)
    np.testing.assert_allclose(direction, legacy, atol=5e-13, rtol=5e-13)
    np.testing.assert_allclose(gradient, legacy_gradient, atol=5e-13, rtol=5e-13)
    np.testing.assert_allclose(info["raw_fisher_norm"], legacy_info["raw_fisher_norm"], atol=5e-13)


def test_resident_cached_operators_are_exact_adjoint_and_match_existing_lsqr():
    cache, _, _, psi, q, hp, jac, _ = fixture()
    forward, adjoint = cached_projected_operators(cache, psi, q, maximum_device_bytes=10**7)
    rng = np.random.default_rng(5)
    v, w = rng.normal(size=11), rng.normal(size=7)
    np.testing.assert_allclose(np.vdot(w, forward(jnp.asarray(v))),
                               np.vdot(adjoint(jnp.asarray(w)), v), atol=2e-13)
    scale = 1/np.sqrt(q*np.sum(psi**2/q))
    energy = np.sum(psi*hp/q)/np.sum(psi**2/q)
    force = 2*(hp-energy*psi)*scale
    solved = damped_lsqr(forward, adjoint, jnp.asarray(force), damping=.2,
                         max_iterations=40, relative_tolerance=1e-11)
    host, _, _ = solve_cached_projected_sr(cache, psi, hp, q, damping=.2,
                         parameter_chunk_size=3, maximum_device_bytes=10**7)
    np.testing.assert_allclose(solved.solution, host, atol=2e-11, rtol=2e-11)
    assert float(solved.relative_normal_residual) < 1e-10


def test_memmap_restore_hash_and_identity(tmp_path):
    path = tmp_path/"frozen_raw_J.npy"
    cache, theta, rows, psi, q, hp, jac, _ = fixture(path=path)
    assert isinstance(cache.jacobian, np.memmap) and not cache.jacobian.flags.writeable
    restored = load_cached_amplitude_jacobian(path, maximum_host_bytes=10**7)
    restored.verify(theta, rows, psi, q, source_identity="independent-linear-node-oracle")
    np.testing.assert_array_equal(restored.jacobian, jac)
    first, _, _ = solve_cached_projected_sr(cache, psi, hp, q, damping=.3,
                         parameter_chunk_size=4, maximum_device_bytes=10**7)
    second, _, _ = solve_cached_projected_sr(restored, psi, hp, q, damping=.3,
                         parameter_chunk_size=4, maximum_device_bytes=10**7)
    np.testing.assert_array_equal(first, second)
    changed = np.load(path, mmap_mode="r+")
    changed[0, 0] += 1
    changed.flush()
    del changed
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_cached_amplitude_jacobian(path, maximum_host_bytes=10**7)


def test_actual_fresh_process_restore_and_solve(tmp_path):
    path = tmp_path/"frozen_J.npy"
    cache, theta, rows, psi, q, hp, _, _ = fixture(path=path)
    expected, _, _ = solve_cached_projected_sr(cache, psi, hp, q, damping=.3,
                         parameter_chunk_size=4, maximum_device_bytes=10**7)
    payload = json.dumps(dict(theta=theta.tolist(), rows=rows.tolist(), psi=psi.tolist(),
                              q=q.tolist(), hp=hp.tolist()))
    script = """
import json,sys,numpy as np
from su2_nqs.vmc.cached_amplitude_operators import load_cached_amplitude_jacobian,solve_cached_projected_sr
p=json.loads(sys.argv[2]);p={k:np.asarray(v,np.int32 if k=='rows' else np.float64) for k,v in p.items()}
c=load_cached_amplitude_jacobian(sys.argv[1],maximum_host_bytes=10**7)
c.verify(p['theta'],p['rows'],p['psi'],p['q'],source_identity='independent-linear-node-oracle')
d,g,info=solve_cached_projected_sr(c,p['psi'],p['hp'],p['q'],damping=.3,parameter_chunk_size=4,maximum_device_bytes=10**7)
print(json.dumps(dict(direction=np.asarray(d).tolist(),dtype=str(d.dtype),residual=info['relative_linear_residual'])))
"""
    env = dict(os.environ, JAX_ENABLE_X64="True", JAX_PLATFORMS="cpu",
               JAX_PLATFORM_NAME="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script, str(path), payload],
        cwd=Path(__file__).resolve().parents[1], env=env, check=True,
        capture_output=True, text=True, timeout=60)
    actual = json.loads(result.stdout)
    np.testing.assert_array_equal(actual["direction"], np.asarray(expected))
    assert actual["dtype"] == "float64" and actual["residual"] < 1e-12


@pytest.mark.parametrize("field", ["psi", "density"])
def test_restore_checks_saved_amplitude_and_Q_hash_even_without_raw_hash_pass(tmp_path, field):
    path = tmp_path/"frozen_J.npy"
    fixture(path=path)
    manifest_path = Path(str(path)+".json")
    manifest = json.loads(manifest_path.read_text())
    manifest[field][0] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="amplitude/Q content SHA256"):
        load_cached_amplitude_jacobian(path, maximum_host_bytes=10**7, verify_content=False)


def test_x64_disabled_fresh_process_rejects_every_numerical_entry_before_work(tmp_path):
    script = """
import jax
assert not jax.config.x64_enabled
from su2_nqs.vmc.cached_amplitude_operators import build_cached_amplitude_jacobian,load_cached_amplitude_jacobian,cached_projected_operators,solve_cached_projected_sr
calls=[lambda:build_cached_amplitude_jacobian(None,None,None,None,None,maximum_host_bytes=100),lambda:load_cached_amplitude_jacobian('does-not-exist.npy',maximum_host_bytes=100),lambda:cached_projected_operators(None,None,None,maximum_device_bytes=100),lambda:solve_cached_projected_sr(None,None,None,None,damping=.1,maximum_device_bytes=100)]
for call in calls:
 try:call()
 except RuntimeError as error:assert 'x64' in str(error)
 else:raise AssertionError('silently accepted x64 disabled')
print('all four entries rejected')
"""
    env = dict(os.environ, JAX_ENABLE_X64="False", JAX_PLATFORMS="cpu",
               JAX_PLATFORM_NAME="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1], env=env, check=True,
        capture_output=True, text=True, timeout=60)
    assert result.stdout.strip() == "all four entries rejected"


@pytest.mark.parametrize("field", ["theta", "rows", "psi", "q", "source"])
def test_stale_parent_or_record_cache_is_rejected(field):
    cache, theta, rows, psi, q, _, _, _ = fixture()
    values = dict(theta=theta.copy(), rows=rows.copy(), psi=psi.copy(), density=q.copy(),
                  source_identity="independent-linear-node-oracle")
    names = dict(theta="theta", rows="rows", psi="psi", q="density")
    if field == "source":
        values["source_identity"] += "-changed"
    else:
        values[names[field]].flat[0] += 1
    with pytest.raises(ValueError, match="does not match"):
        cache.verify(**values)


def test_invalid_measures_complex_outputs_and_memory_budgets_fail_closed(tmp_path):
    cache, theta, rows, psi, q, hp, _, batch = fixture()
    bad = q.copy(); bad[2] = 0
    with pytest.raises(ValueError, match="positive Q"):
        build_cached_amplitude_jacobian(batch, theta, rows, psi, bad, maximum_host_bytes=10**7)
    with pytest.raises(MemoryError, match="host array budget"):
        build_cached_amplitude_jacobian(batch, theta, rows, psi, q, maximum_host_bytes=1)
    with pytest.raises(MemoryError, match="device array budget"):
        cached_projected_operators(cache, psi, q, maximum_device_bytes=1)
    with pytest.raises(MemoryError, match="device array budget"):
        solve_cached_projected_sr(cache, psi, hp, q, damping=.2, maximum_device_bytes=1)
    with pytest.raises(MemoryError, match="host array budget"):
        solve_cached_projected_sr(cache, psi, hp, q, damping=.2,
                                  maximum_device_bytes=10**7, maximum_host_bytes=1)
    with pytest.raises(ValueError, match="not complex"):
        build_cached_amplitude_jacobian(lambda x, r: batch(x, r).astype(jnp.complex128),
            theta, rows, psi, q, maximum_host_bytes=10**7)
    with pytest.raises(ValueError, match="identical frozen"):
        solve_cached_projected_sr(cache, psi*2, hp, q, damping=.2, maximum_device_bytes=10**7)
    path = tmp_path/"do_not_replace.npy"
    path.touch()
    with pytest.raises(FileExistsError):
        build_cached_amplitude_jacobian(batch, theta, rows, psi, q,
            maximum_host_bytes=10**7, storage_path=path)


def test_cache_replay_is_checked_in_importance_weighted_geometry():
    theta = np.zeros(1, np.float64)
    psi = np.array([1., 0.])
    density = np.array([1., 1e-24])
    rows = np.arange(2, dtype=np.int32)[:, None]
    # Raw norm mismatch is only 1e-11, but normalized Q-weighted mismatch is 10.
    batch = lambda x, r: jnp.array([1., 1e-11])[r[:, 0]]+0*x[0]
    with pytest.raises(ArithmeticError, match="disagree"):
        build_cached_amplitude_jacobian(batch, theta, rows, psi, density,
                                        maximum_host_bytes=10**7)


def test_1024_record_shape_plan_has_honest_host_cost_and_bounded_device_arrays():
    wide = cached_derivative_memory_plan(1024, 20_748_160, row_batch_size=2,
                                         parameter_chunk_size=16384)
    old = cached_derivative_memory_plan(1024, 1_495_080, row_batch_size=2,
                                        parameter_chunk_size=16384)
    assert wide["raw_host_cache_bytes"] == 169_968_926_720
    assert wide["raw_host_cache_bytes"] > 150*1024**3
    assert wide["streamed_solve_device_array_bound_bytes"] < 2*1024**3
    assert old["raw_host_cache_bytes"] < 12*1024**3
    assert wide["exact_solver_Jacobian_passes"] == 3
