"""Continuation protocol tests without carrier, Hamiltonian or sampler calls.

Synthetic orchestration fixtures exercise committed stage ordering and failure
paths; optional artifact tests replay only previously saved arrays.
"""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_resonance_continuation as workflow


def _spec():
    return json.loads(workflow.CONFIG.read_text())


def _forbid(*args, **kwargs):
    raise AssertionError("protocol test reached a forbidden physical route")


def test_fixed_question_contains_eight_arms_and_one_new_audit():
    spec = _spec()
    workflow.validate_protocol(spec)
    assert len(workflow.ARMS) == len(set(workflow.ARMS)) == 8
    assert workflow.ARMS[:2] == ("continued_additive110", "continued_free360")
    assert workflow.ARMS[2:] == tuple(x for i in range(3) for x in (f"matched_free{i}", f"attention{i}"))
    assert spec["training_steps"] == 60 and spec["step_bisections"] + 1 == 12
    assert spec["step_fs_radius"] == .001
    assert spec["audit_blocks"] * spec["chains"] == 128
    assert spec["audit_seed"] == 2026101601
    assert spec["bootstrap_seed"] == 2026101602
    assert spec["distillation_teacher"] == "additive110"
    assert "initialization labels are not independent samples" in spec["primary"]
    assert "not a larger manifold" in spec["claim_boundary"]
    assert "no best-checkpoint selection" in spec["protocol"]


@pytest.mark.parametrize("field,value", [
    ("task", "new_search"), ("starts", ["old_endpoint"]),
    ("distillation_teacher", "free360"), ("training_steps", 61),
    ("step_fs_radius", .002), ("step_max_alpha", 1.), ("step_bisections", 12),
    ("audit_seed", 1), ("bootstrap_seed", 1), ("bootstrap_replicates", 5001),
    ("chains", 8), ("audit_blocks", 9), ("feature_batches_per_worker", 9),
    ("max_feature_workers", 17), ("remote_allowed", True),
    ("wall_seconds_cap", 1801), ("wall_seconds_cap", 0),
    ("peak_rss_gib_cap", float("nan")), ("peak_rss_gib_cap", 4.01),
])
def test_protocol_change_fails_before_runtime_or_physics(monkeypatch, field, value):
    spec = _spec()
    spec[field] = value
    monkeypatch.setattr(workflow, "numerical_runtime_identity", _forbid)
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", _forbid)
    monkeypatch.setattr(workflow, "draw", _forbid)
    with pytest.raises(ValueError):
        workflow.registration(spec)


def _fixture(tmp_path, monkeypatch, *, registered=True, sealed=True):
    spec = _spec()
    spec.update(output="new", source_run="old", distillation_run="distilled")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec))
    source = tmp_path / "old"
    source.mkdir()
    (source / "registration.json").write_text(json.dumps({"config": {"synthetic": True}}))
    (source / "scale.json").write_text(json.dumps({"carrier_scale": 2.}))
    output = tmp_path / "new"
    output.mkdir()
    reg = {"config": spec, "synthetic": True}
    if registered:
        (output / "registration.json").write_text(json.dumps(reg))
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "registration", lambda _: reg)
    monkeypatch.setattr(workflow, "peak_rss_gib", lambda: .01)
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", _forbid)
    monkeypatch.setattr(workflow, "draw", _forbid)
    monkeypatch.setattr(workflow, "initial_arms", _forbid)
    monkeypatch.setattr(workflow, "load_corpus", _forbid)
    monkeypatch.setattr(workflow, "run_cached_energy_continuation", _forbid)
    if sealed:
        (output / "endpoints_seal.json").write_text("{}")
        monkeypatch.setattr(workflow, "require_endpoint_seal", lambda path: {})
    return config, output, spec, reg


def _saved_samples(output):
    data = dict(states=np.zeros((128, 36), dtype=np.int8), q=np.ones(128),
                groups=np.repeat(np.arange(16), 8), carrier=np.ones(128),
                hcarrier=-np.ones(128))
    workflow._save_npz(output / "audit_samples.npz", **data)
    info = dict(sha256=workflow.file_sha256(output / "audit_samples.npz"),
                endpoint_seal_sha256=workflow.file_sha256(output / "endpoints_seal.json"))
    (output / "audit_sampling.json").write_text(json.dumps(info))
    return data


def test_first_stage_cannot_draw_audit(tmp_path, monkeypatch):
    config, output, _, _ = _fixture(tmp_path, monkeypatch, registered=False, sealed=False)
    with pytest.raises(ValueError, match="first stage"):
        workflow.run_stage(config, "samples")
    assert not (output / "audit_samples.npz").exists()


def test_existing_attempt_requires_explicit_resume(tmp_path, monkeypatch):
    config, _, _, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError, match="explicit --resume"):
        workflow.run_stage(config, "samples")


def test_resume_cannot_change_registered_inputs(tmp_path, monkeypatch):
    config, output, _, reg = _fixture(tmp_path, monkeypatch)
    (output / "registration.json").write_text(json.dumps({**reg, "changed": True}))
    with pytest.raises(ValueError, match="sources/config/inputs changed"):
        workflow.run_stage(config, "samples", resume=True)


def test_audit_requires_every_initial_and_final_endpoint_seal(tmp_path, monkeypatch):
    config, output, _, _ = _fixture(tmp_path, monkeypatch, sealed=False)
    with pytest.raises(FileNotFoundError):
        workflow.run_stage(config, "samples", resume=True)
    assert not (output / "audit_samples.npz").exists()
    assert json.loads((output / "failure.json").read_text())["automatic_budget_extension"] is False


def test_endpoint_seal_covers_all_thirteen_portable_checkpoints(tmp_path):
    names = [*(arm + "_endpoint" for arm in workflow.ARMS),
             "initial_additive", "initial_free", "initial0", "initial1", "initial2"]
    for name in names:
        for ext in ("json", "npz"):
            (tmp_path / f"{name}.{ext}").write_bytes(f"{name}:{ext}".encode())
    seals = workflow._seal_endpoints(tmp_path)
    assert len(seals) == 13 and set(seals) == set(names)
    (tmp_path / "endpoints_seal.json").write_text(json.dumps(seals))
    assert workflow.require_endpoint_seal(tmp_path) == seals
    (tmp_path / "initial1.npz").write_bytes(b"tampered initial")
    with pytest.raises(ValueError, match="checkpoint seal changed"):
        workflow.require_endpoint_seal(tmp_path)


def test_saved_audit_is_reused_without_drawing_or_requerying(tmp_path, monkeypatch):
    config, output, _, _ = _fixture(tmp_path, monkeypatch)
    _saved_samples(output)
    result = workflow.run_stage(config, "samples", resume=True)
    assert result["status"] == "samples_already_saved"


def test_tampered_draw_is_never_silently_resampled(tmp_path, monkeypatch):
    config, output, _, _ = _fixture(tmp_path, monkeypatch)
    _saved_samples(output)
    (output / "audit_samples.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sample hash mismatch"):
        workflow.run_stage(config, "samples", resume=True)


def test_new_audit_draw_uses_original_proposal_and_frozen_scale(tmp_path, monkeypatch):
    config, output, spec, _ = _fixture(tmp_path, monkeypatch)
    events = []
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", lambda oldspec: SimpleNamespace(carrier="carrier", bonds="bonds"))
    def draw(oldspec, n, carrier, bonds, scale, seed, blocks):
        assert (output / "endpoints_seal.json").exists()
        assert oldspec == {"synthetic": True}
        assert (n, carrier, bonds, scale, seed, blocks) == (36, "carrier", "bonds", 2., spec["audit_seed"], 8)
        events.append("one new draw")
        return dict(states=np.zeros((128, 36)), q=np.ones(128), groups=np.repeat(np.arange(16), 8)), {"synthetic": True}
    monkeypatch.setattr(workflow, "draw", draw)
    assert workflow.run_stage(config, "samples", resume=True)["status"] == "samples_complete"
    assert workflow.run_stage(config, "samples", resume=True)["status"] == "samples_already_saved"
    assert events == ["one new draw"]
    sampling = json.loads((output / "audit_sampling.json").read_text())
    assert sampling["seed"] == spec["audit_seed"]
    assert sampling["endpoint_seal_sha256"] == workflow.file_sha256(output / "endpoints_seal.json")


def test_exhausted_cumulative_time_cannot_be_reset_by_resume(tmp_path, monkeypatch):
    config, output, spec, _ = _fixture(tmp_path, monkeypatch)
    (output / "progress.json").write_text(json.dumps({"seconds": spec["wall_seconds_cap"]}))
    with pytest.raises(TimeoutError, match="cumulative registered cap exhausted"):
        workflow.run_stage(config, "samples", resume=True)


def test_audit_cannot_be_opened_before_training_finishes(tmp_path, monkeypatch):
    config, output, _, _ = _fixture(tmp_path, monkeypatch, sealed=False)
    (output / "audit_samples.npz").write_bytes(b"premature audit")
    with pytest.raises(ValueError, match="audit exists before endpoint sealing"):
        workflow.run_stage(config, "train", resume=True)


def test_feature_worker_limit_cannot_be_extended_on_resume(tmp_path, monkeypatch):
    config, output, spec, _ = _fixture(tmp_path, monkeypatch)
    _saved_samples(output)
    for i in range(spec["max_feature_workers"]):
        (output / f"feature_worker_{i:02d}.json").write_text('{"status":"already committed"}')
    with pytest.raises(TimeoutError, match="feature worker limit"):
        workflow.run_stage(config, "features", resume=True)
    assert not (output / "feature_worker_16.json").exists()


@pytest.mark.parametrize("finishing", [False, True])
def test_worker_boundary_does_not_bypass_final_independent_carrier_check(tmp_path, monkeypatch, finishing):
    config, output, _, _ = _fixture(tmp_path, monkeypatch)
    _saved_samples(output)
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", lambda _: object())
    events = []
    def build(engine, data, scale, path, progress):
        for completed in range(114 if finishing else 2, 129 if finishing else 17, 2):
            progress("audit_features", completed=completed, total=128)
        events.append("independent C/HC validation")
    monkeypatch.setattr(workflow, "build_corpus", build)
    result = workflow.run_stage(config, "features", resume=True)
    assert result["status"] == ("complete" if finishing else "committed_batch_boundary")
    assert events == (["independent C/HC validation"] if finishing else [])
    record = json.loads((output / "feature_worker_00.json").read_text())
    assert record["committed_batches"] == 8


def test_external_timeout_exhausts_budget_even_if_last_progress_is_old(tmp_path, monkeypatch):
    config, output, _, _ = _fixture(tmp_path, monkeypatch)
    (output / "progress.json").write_text('{"seconds":12.0}')
    calls = []
    def timeout(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(workflow.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        workflow.orchestrate(config, resume=True)
    assert len(calls) == 1
    marker = json.loads((output / "external_budget_exhausted.json").read_text())
    assert marker["automatic_extension"] is False
    with pytest.raises(TimeoutError, match="resume denied"):
        workflow.run_stage(config, "samples", resume=True)


def test_memory_failure_preserves_committed_batch_and_cumulative_time(tmp_path, monkeypatch):
    config, output, _, _ = _fixture(tmp_path, monkeypatch)
    _saved_samples(output)
    (output / "progress.json").write_text('{"seconds":123.0}')
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", lambda _: object())
    monkeypatch.setattr(workflow, "peak_rss_gib", lambda: 4.125)
    def build(engine, data, scale, path, progress):
        workflow._save_npz(path, carrier=np.ones(2))
        progress("audit_features", completed=2, total=128)
    monkeypatch.setattr(workflow, "build_corpus", build)
    with pytest.raises(MemoryError, match="post-commit"):
        workflow.run_stage(config, "features", resume=True)
    with np.load(output / "audit_features.npz") as saved:
        assert len(saved["carrier"]) == 2
    failure = json.loads((output / "failure.json").read_text())
    assert failure["seconds"] >= 123. and failure["peak_rss_gib"] == 4.125
    assert failure["automatic_budget_extension"] is False
    record = json.loads((output / "feature_worker_00.json").read_text())
    assert record["committed_batches"] == 1
    assert record["status"] == "post_commit_memory_guard"


def test_all_eight_fixed_endpoints_commit_before_audit_is_allowed(tmp_path, monkeypatch):
    config, output, _, _ = _fixture(tmp_path, monkeypatch, sealed=False)
    initials = {name: dict(parameters=jnp.zeros(n), generator_table=jnp.zeros((6, 6, 10)), coefficients=jnp.zeros(1176))
                for name, n in [("initial_additive", 110), ("initial_free", 360),
                                ("initial0", 730), ("initial1", 730), ("initial2", 730)]}
    arms = {}
    for arm in workflow.ARMS:
        parent = "initial_additive" if arm == "continued_additive110" else "initial_free" if arm == "continued_free360" else "initial" + arm[-1]
        n = 110 if arm == "continued_additive110" else 730 if arm.startswith("attention") else 360
        arms[arm] = (jnp.zeros(n), lambda x: jnp.zeros((6, 6, 10)), parent)
    monkeypatch.setattr(workflow, "initial_arms", lambda _: (arms, initials))
    corpus = dict(carrier=jnp.ones(4), hcarrier=-jnp.ones(4), features=jnp.zeros((4, 1176)), hfeatures=jnp.zeros((4, 1176)), q=jnp.ones(4))
    monkeypatch.setattr(workflow, "load_corpus", lambda path, phase: corpus if phase == "train" else _forbid())
    monkeypatch.setattr(workflow, "multicontext_feature_coefficients", lambda *args: jnp.zeros(1176))
    events = []
    def fit(x0, state, q, cfg, checkpoint, identity, *, resume, progress):
        assert not (output / "audit_samples.npz").exists()
        assert not (output / "endpoints_seal.json").exists()
        assert cfg.steps == 60 and cfg.target_fs == .001 and cfg.bisection_steps == 11
        assert resume is False
        events.append(identity["arm"])
        p, hp = state(x0)
        return SimpleNamespace(parameters=x0, amplitudes=p, history=[{"step": i + 1} for i in range(60)],
                               ledger={"new_samples": 0, "new_H_queries": 0, "completed_gradient_calls": 60}, active_seconds=.01)
    monkeypatch.setattr(workflow, "run_cached_energy_continuation", fit)
    assert workflow.run_stage(config, "train", resume=True)["status"] == "training_complete"
    assert events == list(workflow.ARMS)
    assert len(workflow.require_endpoint_seal(output)) == 13
    assert not (output / "audit_samples.npz").exists()
    training = json.loads((output / "training.json").read_text())
    assert max(training["initial_replay_max"].values()) == 0
    assert [len(training["arms"][arm]["history"]) for arm in workflow.ARMS] == [60] * 8


def test_three_matched_free_initial_tables_equal_frozen_neural_tables_without_physics():
    arms, initials = workflow.initial_arms(_spec())
    assert tuple(arms) == workflow.ARMS
    assert len(initials) == 5
    for i in range(3):
        free, nn = arms[f"matched_free{i}"], arms[f"attention{i}"]
        assert free[2] == nn[2] == f"initial{i}"
        assert free[0].shape == (360,) and nn[0].shape == (730,)
        assert np.array_equal(np.asarray(free[1](free[0])), np.asarray(initials[f"initial{i}"]["generator_table"]))
        np.testing.assert_allclose(free[1](free[0]), nn[1](nn[0]), atol=1e-12, rtol=1e-12)


def test_completed_artifact_seals_and_saved_feature_replay():
    output = workflow.ROOT / _spec()["output"]
    if not (output / "results.json").exists():
        pytest.skip("bounded continuation has not completed")
    result = json.loads((output / "results.json").read_text())
    unsigned = dict(result)
    seal = unsigned.pop("result_sha256")
    assert workflow.canonical_sha256(unsigned) == seal
    assert workflow.file_sha256(output / "registration.json") == result["registration_sha256"]
    assert result["status"] == "complete" and result["fresh_audit"]
    assert not result["variational_win_claimed"] and not result["remote_released"]
    assert result["physical_H_feature_centers"] == result["additional_cheap_carrier_H_centers"] == 128
    for name, sha in result["artifacts"].items():
        assert workflow.file_sha256(output / name) == sha
    workflow.require_endpoint_seal(output)
    with np.load(output / "audit_values.npz") as values, np.load(output / "audit_features.npz") as features, np.load(output / "audit_samples.npz") as samples:
        assert values["psi"].shape == values["hpsi"].shape == (128, 14)
        assert np.array_equal(values["q"], samples["q"])
        assert np.array_equal(values["groups"], samples["groups"])
        np.testing.assert_allclose(values["psi"], features["carrier"][:, None] + features["features"] @ values["coefficients"], atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(values["hpsi"], features["hcarrier"][:, None] + features["hfeatures"] @ values["coefficients"], atol=1e-12, rtol=1e-13)
        assert len(np.unique(values["groups"])) == 16
    for arm in workflow.ARMS:
        record = result["training"]["arms"][arm]
        assert len(record["history"]) == 60
        assert record["ledger"]["completed_gradient_calls"] == 60
        assert record["ledger"]["completed_step_candidate_calls"] == 720
        assert record["ledger"]["new_samples"] == record["ledger"]["new_H_queries"] == 0


def _completed_artifacts():
    output = workflow.ROOT / _spec()["output"]
    if not (output / "results.json").exists():
        pytest.skip("bounded continuation has not completed")
    return output, json.loads((output / "results.json").read_text())


def test_completed_source_seals_and_audit_randomness_are_unchanged():
    output, result = _completed_artifacts()
    registration = json.loads((output / "registration.json").read_text())
    assert registration["config"] == result["config"] == _spec()
    assert workflow.canonical_sha256({k: v for k, v in registration.items() if k != "registration_sha256"}) == registration["registration_sha256"]
    for name, sha in registration["sources"].items():
        assert workflow.file_sha256(workflow.ROOT / name) == sha
    for name, sha in registration["inputs"].items():
        assert workflow.file_sha256(workflow.ROOT / name) == sha
    source = workflow.ROOT / _spec()["source_run"]
    old = json.loads((source / "registration.json").read_text())["config"]
    assert result["sampling"]["seed"] not in (old["train_seed"], old["audit_seed"])
    assert result["sampling"]["endpoint_seal_sha256"] == workflow.file_sha256(output / "endpoints_seal.json")
    with np.load(output / "audit_samples.npz") as samples, np.load(output / "audit_features.npz") as features:
        assert samples["states"].shape == (128, 36)
        assert np.all(samples["states"].sum(axis=1) == 18)
        assert np.all(np.isfinite(samples["q"])) and np.all(samples["q"] > 0)
        _, counts = np.unique(samples["groups"], return_counts=True)
        assert len(counts) == 16 and np.all(counts == 8)
        assert str(features["sample_seal"]) == workflow.canonical_sha256({k: samples[k].tolist() for k in ("states", "q", "groups")})
        records = json.loads(str(features["records"]))
        assert len(records) == 64 and sum(r["full_H_centers"] for r in records) == 128
        assert all(r["group_order"] == 432 and r["bond_count"] == 216 for r in records)
        np.testing.assert_allclose(features["carrier"], samples["carrier"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(features["hcarrier"], 36 * samples["hcarrier"], rtol=1e-10, atol=1e-11)
    workers = [json.loads(path.read_text()) for path in output.glob("feature_worker_*.json")]
    assert len(workers) <= _spec()["max_feature_workers"]
    assert sum(w.get("committed_batches", 0) for w in workers) == 64
    assert all(w.get("committed_batches", 0) <= 8 for w in workers)
    assert result["seconds"] <= _spec()["wall_seconds_cap"]
    # The post-commit RSS guard may overshoot. Never erase/forbid a recorded
    # old worker peak just because a subsequent clean worker was smaller.
    assert all(np.isfinite(w["peak_rss_gib"]) and w["peak_rss_gib"] > 0 for w in workers)


def test_completed_direct_node_safe_statistics_and_joint_primary_replay():
    output, result = _completed_artifacts()
    with np.load(output / "audit_values.npz") as saved:
        psi, hp, q, groups = (saved[k].copy() for k in ("psi", "hpsi", "q", "groups"))
        names = saved["names"].tolist()
    z = abs(psi)**2 / q[:, None]
    numerator = psi.conj() * hp / q[:, None]
    second = abs(hp)**2 / q[:, None]
    norm = z.sum(axis=0)
    e = numerator.sum(axis=0) / norm
    residual = abs(hp - psi * e)**2 / q[:, None]
    v = residual.sum(axis=0) / norm / 36
    for i, name in enumerate(names):
        report = result["audit"]["arms"][name]
        np.testing.assert_allclose(report["energy_per_site"]["value"], e[i].real / 36, atol=2e-15, rtol=1e-13)
        np.testing.assert_allclose(report["variance_per_site"]["value"], v[i], atol=3e-13, rtol=1e-11)
        np.testing.assert_allclose(report["weight_ess_fraction"], norm[i]**2 / (len(q) * np.sum(z[:, i]**2)), atol=1e-14)
        np.testing.assert_allclose(report["max_residual_chain_fraction"], max(residual[groups == g, i].sum() for g in np.unique(groups)) / residual[:, i].sum(), atol=1e-13)
    labels = np.unique(groups)
    blocks = np.stack([np.stack((z[groups == g].sum(axis=0), numerator[groups == g].sum(axis=0), second[groups == g].sum(axis=0))) for g in labels])
    draws = np.random.default_rng(_spec()["bootstrap_seed"]).multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=_spec()["bootstrap_replicates"])
    moments = np.einsum("rg,gka->rka", draws, blocks)
    re = moments[:, 1] / moments[:, 0].real
    rv = (moments[:, 2].real / moments[:, 0].real - abs(re)**2) / 36
    points = []
    replicates = []
    for i in range(3):
        free, nn = names.index(f"matched_free{i}"), names.index(f"attention{i}")
        points.extend(((e[nn].real - e[free].real) / 36, v[nn] - v[free]))
        replicates.extend(((re[:, nn].real - re[:, free].real) / 36, rv[:, nn] - rv[:, free]))
    replicates = np.stack(replicates, axis=1)
    points = np.r_[points, np.asarray(points).reshape(3, 2).mean(axis=0)]
    replicates = np.c_[replicates, replicates.reshape(-1, 3, 2).mean(axis=1)]
    keys = [f"pair{i}.{metric}" for i in range(3) for metric in ("energy", "variance")] + ["mean.energy", "mean.variance"]
    for i, key in enumerate(keys):
        report = result["primary_audit"]["metrics"][key]
        np.testing.assert_allclose(report["value"], points[i], atol=4e-13, rtol=1e-10)
        np.testing.assert_allclose(report["chain_bootstrap_se"], replicates[:, i].std(ddof=1), atol=4e-13, rtol=1e-10)
        np.testing.assert_allclose(report["chain_bootstrap_95"], np.quantile(replicates[:, i], [.025, .975]), atol=4e-13, rtol=1e-10)
        if i < 6:
            np.testing.assert_allclose(report["bonferroni_six_comparisons_interval"], np.quantile(replicates[:, i], [.05 / 12, 1 - .05 / 12]), atol=4e-13, rtol=1e-10)
    assert result["primary_audit"]["primary"] == "mean.energy"
