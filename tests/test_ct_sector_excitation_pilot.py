"""Sector-pilot contracts and tiny independent oracles, not CT6 optimization."""
from copy import deepcopy
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct_sector_excitation_pilot as w
from su2_nqs.frontier.artifacts import canonical_sha256, pytree_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint


def plan(monkeypatch, *, sector="q2_K", profile="local"):
    monkeypatch.setattr(w, "file_sha256", lambda _: "a" * 64)
    return w.make_plan(.125, sector, profile=profile)


@pytest.mark.parametrize("sector", ("q2_K", "q0_X"))
def test_exact_spin_momentum_and_bounded_authority(monkeypatch, sector):
    p = plan(monkeypatch, sector=sector)
    w.validate_plan(p)
    assert p["sector"]["target_m2"] == 0
    assert p["sector"]["momentum"] == ([[1, 3], [1, 3]] if sector == "q2_K" else [[-1, 6], [-1, 6]])
    with pytest.raises(PermissionError):
        w.validate_plan(p, executing=True)
    p.update(execution_authorized=True, exact_invariants_passed=True)
    w.validate_plan(p, executing=True)
    for field, value in (("steps", 3), ("wall_cap_seconds", 601), ("fit_records", 4096),
                         ("epsilon", 0.), ("automatic_continuation", True),
                         ("all_nonroot_W_and_target_root_trainable", False), ("score_chunk", True)):
        with pytest.raises(ValueError):
            w.validate_plan(dict(p, **{field: value}))
    q = deepcopy(p); q["sector"]["target_q"] = 4
    with pytest.raises(ValueError):
        w.validate_plan(q)


def test_atomic_optimizer_and_next_rng_restore_and_tamper(tmp_path):
    rng = dict(generator=np.random.default_rng(987).bit_generator.state, used_seeds=[], excluded_seeds=[123], roles=[])
    params = {"backbone": {"w": jnp.array([1.3, .4])}, "root": (jnp.array([[.8]]),)}
    moments = (19, jax.tree.map(lambda x: .1 * x, params), jax.tree.map(lambda x: .2 * x, params))
    original = (params, moments, rng)
    path = tmp_path / "state.pkl"
    w.save_state(path, original, completed_updates=1, metadata=dict(parent="synthetic"))
    restored, meta = w.restore_state(path)
    assert meta["completed_updates"] == 1
    assert pytree_sha256(original[:2]) == pytree_sha256(restored[:2])
    seed1, next1 = w.next_seed(original[2], "future_fit")
    seed2, next2 = w.next_seed(restored[2], "future_fit")
    assert seed1 == seed2 and next1 == next2
    assert seed1 not in (123,)
    original_saved = load_checkpoint(path)
    original_saved["metadata"]["next_rng"]["used_seeds"] = [42]
    save_checkpoint(path, original_saved["params"], optimizer_state=original_saved["optimizer_state"],
                    metadata=original_saved["metadata"])
    with pytest.raises(ValueError, match="seal mismatch"):
        w.restore_state(path)


def mock_initialization(monkeypatch, *, primary_zero=False, unresolved=False):
    from su2_nqs.models import triangular_arity_clebsch as arity
    from su2_nqs.models import triangular_target_root as root
    monkeypatch.setattr(root, "target_root_layout", lambda *_: ((2, 2, 2, 2),))
    primary = np.zeros((2, 2)) if primary_zero else np.array([[.2, .4], [.6, .8]])
    monkeypatch.setattr(root, "singlet_root_coefficients", lambda *_: (primary,))
    monkeypatch.setattr(root, "collective_spin_root_coefficients", lambda *_: (primary,))
    monkeypatch.setattr(arity, "arity_clebsch_bank", lambda p, *_: p)
    native = dict(local=(jnp.array([.3, .5]),), six=(jnp.array([1., .2]),),
                  top=((jnp.array([.7, .9]),), (jnp.array([99.]),)))
    moments = (12, jax.tree.map(lambda x: 2 * x, native), jax.tree.map(lambda x: 3 * x, native))
    saved = dict(params=native, optimizer_state=moments, metadata=dict(rng=dict(consumed_role_seeds=[17], excluded_seeds=[18])))
    calls = []
    def amplitude(p, rows):
        calls.append(p)
        if unresolved:
            return np.full(len(rows), 1e-90 + 0j)
        return np.full(len(rows), np.asarray(p["root"][0]).sum() * (1 + .3j))
    runtime = dict(cfg=object(), amplitudes=amplitude)
    return saved, runtime, calls, primary


@pytest.mark.parametrize("sector", ("q2_K", "q0_X"))
def test_physical_root_first_inherits_all_nonroot_w_and_optimizer(monkeypatch, sector):
    p = plan(monkeypatch, sector=sector)
    saved, runtime, calls, primary = mock_initialization(monkeypatch)
    before = pytree_sha256(saved)
    rows = np.zeros((16, 36), np.int32)
    state, report = w.initialize_target(p, saved, runtime, rows, np.full(16, .01))
    params, moments, rng = state
    assert report["selected_attempt"] == 0 and report["attempts"][0]["seed"] is None
    np.testing.assert_array_equal(params["root"][0], primary)
    assert set(params["backbone"]) == {"local", "six", "top"}
    assert len(params["backbone"]["top"]) == 1
    assert params["backbone"]["six"] is saved["params"]["six"]
    assert moments[0] == 12
    assert np.count_nonzero(np.asarray(moments[1]["root"][0])) == 0
    np.testing.assert_array_equal(moments[1]["backbone"]["local"][0], 2 * saved["params"]["local"][0])
    assert pytree_sha256(saved) == before and len(calls) == 1
    assert {17, 18}.issubset(rng["excluded_seeds"])


def test_only_exact_zero_norm_uses_preregistered_root_fallback(monkeypatch):
    p = plan(monkeypatch)
    saved, runtime, calls, _ = mock_initialization(monkeypatch, primary_zero=True)
    state, report = w.initialize_target(p, saved, runtime, np.zeros((16, 36), np.int32), np.full(16, .01))
    assert report["selected_attempt"] == 1 and len(calls) == 2
    rng = np.random.default_rng(p["initialization_seed"])
    np.testing.assert_array_equal(state[0]["root"][0], rng.normal(size=(2, 2)) / 2)
    assert not report["seed_selection_uses_energy"]
    saved, runtime, _, _ = mock_initialization(monkeypatch, unresolved=True)
    with pytest.raises(ArithmeticError, match="unresolved small"):
        w.initialize_target(p, saved, runtime, np.zeros((16, 36), np.int32), np.full(16, .01))


def test_cost_estimate_complex_J_and_frozen_audit_budget(monkeypatch):
    p = plan(monkeypatch)
    cost = w.fit_cost_estimate(p, amplitude_seconds=.01, H_seconds=1., jacobian_seconds=2., Q_seconds=.2)
    assert cost["estimated_audit_seconds"] == pytest.approx(1.5 * 64 * 2.2)
    assert cost["estimated_all_seconds"] > cost["estimated_audit_seconds"]
    with pytest.raises(ValueError):
        w.fit_cost_estimate(p, amplitude_seconds=np.nan, H_seconds=1., jacobian_seconds=2., Q_seconds=.2)


def tiny_runtime():
    """Independent two-level complex amplitude, embedded only in bit labels."""
    def vector(p):
        x = p["backbone"]["w"]
        root = p["root"][0]
        return jnp.array([x[0] * root[0] + .2j * x[1], x[1] * root[1] + .1j * x[0]])
    def scalar(p, row):
        return vector(p)[row[0]]
    def H(p, rows):
        v = np.asarray(vector(p)); h = np.array([[.3, -.4], [-.4, -.5]]) @ v
        return v[rows[:, 0]], h[rows[:, 0]]
    def draw(count, seed):
        rows = np.zeros((count, 36), np.int32)
        rows[:, 0] = np.random.default_rng(seed).integers(0, 2, count)
        return rows, dict(seed=seed)
    def Q(rows):
        return np.full(len(rows), .5), (np.ones(len(rows)), -.2 + .05 * rows[:, 0],
                                      np.full(len(rows), .5), np.full(len(rows), .5))
    return dict(prepare=lambda p: p, amplitude=scalar,
                amplitudes=lambda p, rows: np.asarray(jax.vmap(scalar, in_axes=(None, 0))(p, jnp.asarray(rows))),
                H=H, draw=draw, Q=Q)


def test_actual_full_parameter_complex_SR_finite_guard_and_fresh_H_audit(monkeypatch, tmp_path):
    p = plan(monkeypatch)
    p.update(target_FS=.0035, maximum_FS=.009, tangent_FS_radius=.003)
    params = dict(backbone=dict(w=jnp.array([.9, .7])), root=(jnp.array([.8, .6]),))
    moments = (7, jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))
    rng = dict(generator=np.random.default_rng(8842).bit_generator.state, used_seeds=[], excluded_seeds=[], roles=[])
    state, report = w.transition(p, tiny_runtime(), (params, moments, rng), tmp_path / "step1", index=1)
    assert report["all_W_trainable"] and not report["root_only_Ritz"] and report["accepted"]
    assert pytree_sha256(state[0]["backbone"]) != pytree_sha256(params["backbone"])
    assert pytree_sha256(state[0]["root"]) != pytree_sha256(params["root"])
    assert int(state[1][0]) == 8
    assert set(state[2]["used_seeds"]) == {report["fit_seed"], report["calibration_seed"]}
    audit, following = w.paired_final_audit(p, tiny_runtime(), params, state[0], state[2], tmp_path / "audit")
    assert audit["independent_from_fit_and_finite_geometry"] and audit["gap_not_certified"]
    assert audit["seed"] not in state[2]["used_seeds"]
    assert following["used_seeds"][-1] == audit["seed"]
    assert np.isfinite(audit["moments"]["endpoint_values"]).all()
    # The toy test is a driver oracle, never a CT6 energy/sector result.
    assert audit["moments"]["candidate_minus_control"][0] < 0


def test_complex_finite_radius_uses_global_ray_phase_not_local_phase():
    q = np.full(64, .5)
    old = np.tile([1 + .2j, .3 + .6j], 32)
    global_phase = old * np.exp(.7j)
    local_phase = old * np.tile([np.exp(.7j), np.exp(-.7j)], 32)
    from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS
    assert common_proposal_FS(np.column_stack((old, global_phase)), q, groups=4)["FS_radians"] < 1e-14
    assert common_proposal_FS(np.column_stack((old, local_phase)), q, groups=4)["FS_radians"] > .1


def test_no_queries_or_registration_without_invariant_authority(monkeypatch, tmp_path):
    p = plan(monkeypatch)
    monkeypatch.setattr(w, "checked_parent", lambda *_: pytest.fail("must refuse before loading parent"))
    with pytest.raises(PermissionError):
        w.run(p, tmp_path / "not_created", stage="prepare")
    assert not (tmp_path / "not_created").exists()


def test_complex_checkpoint_replay_includes_nodes_phase_H_and_Q():
    rows = np.zeros((4, 36), np.int32)
    p = np.array([0, 1 + .2j, -.3j, .4])
    h = np.array([.1j, -.8, .3 + .2j, .05])
    q = np.array([.1, .2, .3, .4])
    reference = dict(rows=rows, psi=p, hpsi=h, proposal=q)
    rt = dict(H=lambda *_: (p + np.array([1e-16j, 0, 0, 0]), h), Q=lambda *_: (q, ()))
    report = w.replay_reference(rt, {}, reference, require_H=True)
    assert report["passed"] and not report["phase_aligned"] and report["full_H_replayed"]
    rt["H"] = lambda *_: (p * np.exp(.1j), h * np.exp(.1j))
    with pytest.raises(ArithmeticError, match="fixed complex-scale"):
        w.replay_reference(rt, {}, reference, require_H=True)
    with pytest.raises(ValueError, match="full-H/Q"):
        w.replay_reference(rt, {}, dict(rows=rows, psi=p), require_H=True)


def test_full_driver_atomic_role_reservation_and_interrupted_audit(monkeypatch, tmp_path):
    p = plan(monkeypatch)
    p.update(exact_invariants_passed=True, execution_authorized=True)
    monkeypatch.setattr(w, "BASE", tmp_path)
    monkeypatch.setattr(w, "transitive_local_source_manifest", lambda *a, **k: {})
    monkeypatch.setattr(w, "checked_parent", lambda *_: {})
    rt = dict(tiny_runtime(), proposal_metadata={}, normalization={}, work={})
    monkeypatch.setattr(w, "make_runtime", lambda *_: rt)
    params = dict(backbone=dict(w=jnp.array([.9, .7])), root=(jnp.array([.8, .6]),))
    moments = (7, jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))
    rng = dict(generator=np.random.default_rng(8842).bit_generator.state, used_seeds=[], excluded_seeds=[], roles=[])
    monkeypatch.setattr(w, "initialize_target", lambda *_: ((params, moments, rng), dict(primary=True)))
    out = tmp_path / "arm"
    w.run(p, out, stage="prepare")
    cost = dict(estimated_retained_complex_J_bytes=64, estimated_all_seconds=.2,
                estimated_one_update_seconds=.01, estimated_audit_seconds=.01)
    (out / "preflight.json").write_text(json.dumps(cost))
    seen = {}
    def transition(plan, runtime, state, directory, *, index):
        committed, _ = w.restore_state(out / "checkpoint.pkl")
        assert len(state[2]["used_seeds"]) == 0
        assert len(committed[2]["used_seeds"]) == 2
        directory.mkdir()
        seen["fit_cal_reserved_before_query"] = True
        return (jax.tree.map(lambda x: x + .0001, state[0]), state[1], state[2]), dict(accepted=True)
    monkeypatch.setattr(w, "transition", transition)
    def audit(plan, runtime, old, new, original_rng, directory):
        committed, _ = w.restore_state(out / "checkpoint.pkl")
        assert len(original_rng["used_seeds"]) == 2
        assert len(committed[2]["used_seeds"]) == 3
        directory.mkdir()
        seen["audit_seed"] = committed[2]["used_seeds"][-1]
        raise TimeoutError("synthetic interrupted audit after reserved seed")
    monkeypatch.setattr(w, "paired_final_audit", audit)
    report = w.run(p, out, stage="train")
    assert report["status"] == "wall_cost_gate_checkpoint_only_not_final_precision"
    assert seen["fit_cal_reserved_before_query"]
    committed, _ = w.restore_state(out / "checkpoint.pkl")
    assert len(committed[2]["used_seeds"]) == 3 and seen["audit_seed"] in committed[2]["used_seeds"]
    future, _ = w.next_seed(committed[2], "genuinely_new_audit")
    assert future != seen["audit_seed"]
    with pytest.raises(ValueError, match="incomplete transaction"):
        w.run(p, out, stage="resume")


def test_gpu_restart_keeps_state_and_disables_execution(monkeypatch, tmp_path):
    p = plan(monkeypatch)
    monkeypatch.setattr(w, "BASE", tmp_path)
    source = tmp_path / "local_arm"; source.mkdir()
    (source / "plan.json").write_text(json.dumps(p))
    params = {"backbone": {"w": jnp.array([.1, .3])}, "root": (jnp.ones((1, 1)),)}
    moments = (12, jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))
    rng = dict(generator=np.random.default_rng(71).bit_generator.state, used_seeds=[77], excluded_seeds=[], roles=[])
    for name in ("initial.pkl", "checkpoint.pkl"):
        w.save_state(source / name, (params, moments, rng), completed_updates=0,
                     metadata=dict(plan_sha256=canonical_sha256(p), initialization=dict(primary=True)))
    target = w.write_gpu_restart_plan(source)
    gpu = json.loads(target.read_text())
    assert gpu["profile"] == "gpu" and not gpu["execution_authorized"] and not gpu["exact_invariants_passed"]
    assert gpu["restart"]["source_plan_sha256"] == canonical_sha256(p)
    state, _ = w.restore_state(source / "checkpoint.pkl")
    assert state[2] == rng and pytree_sha256(state[:2]) == pytree_sha256((params, moments))
