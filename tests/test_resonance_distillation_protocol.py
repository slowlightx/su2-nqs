"""Saved-feature distillation protocol checks, without a physical state call.

Orchestration tests use constant synthetic feature arrays and an optimizer
stub. They never call an N36 carrier, Hamiltonian constructor or sampler.
"""

import ast
from collections import namedtuple
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_resonance_distillation as workflow


def _spec():
    return json.loads(workflow.CONFIG.read_text())


def _forbid(*args, **kwargs):
    raise AssertionError("protocol test reached a forbidden numerical/data route")


def test_registered_question_and_budget_do_not_create_new_physical_evidence():
    spec = _spec()
    assert workflow.validate_protocol(spec) is None
    assert spec["teachers"] == ["additive110", "free360"]
    assert spec["starts"] == ["old_endpoint", "original_initialization", "independent_initialization"]
    assert spec["fit_count"] == 2 * (2 * 3 + 1) == 14
    assert spec["max_iterations_per_fit"] == 1000
    assert spec["max_function_evaluations_per_fit"] == 1500
    assert spec["max_line_search_steps"] == 25 and spec["lbfgs_memory"] == 30
    assert spec["physical_loss_weights"] == [.5, .5]
    assert spec["wall_seconds_cap"] <= 900 and spec["peak_rss_gib_cap"] <= 4.
    assert spec["new_physical_H_calls"] == spec["new_carrier_queries"] == spec["new_samples"] == 0
    assert spec["remote_allowed"] is False
    assert "ONLY original128 training records" in spec["protocol"]
    assert "already seen" in spec["evaluation"]
    assert "not an all-Hilbert-space certificate" in spec["interpretation"]


@pytest.mark.parametrize("field,value", [
    ("task", "new_state_search"), ("teachers", ["free360"]),
    ("starts", ["old_endpoint"]), ("independent_initialization_seed", 13),
    ("max_iterations_per_fit", 1001), ("max_function_evaluations_per_fit", 1501),
    ("physical_loss_weights", [1., 0.]), ("fit_count", 15),
    ("new_physical_H_calls", 1), ("new_carrier_queries", 1), ("new_samples", 1),
    ("remote_allowed", True), ("wall_seconds_cap", 901),
    ("peak_rss_gib_cap", float("nan")),
])
def test_protocol_mutation_fails_before_reading_numeric_inputs(tmp_path, monkeypatch, field, value):
    spec = _spec()
    spec[field] = value
    config = tmp_path / "bad.json"
    config.write_text(json.dumps(spec))
    monkeypatch.setattr(workflow, "numerical_runtime_identity", _forbid)
    monkeypatch.setattr(workflow, "load_saved_problem", _forbid)
    monkeypatch.setattr(workflow, "load_corpus", _forbid)
    with pytest.raises(ValueError):
        workflow.run(config)


def test_workflow_has_no_direct_physical_query_or_sampling_entrypoint():
    tree = ast.parse(Path(workflow.__file__).read_text())
    forbidden = {"draw", "compile_physical_h_queries", "compile_a1_overlapping_carrier_queries",
                 "FixedCarrierFeatures", "sample_fixed_wavefunction_exchange",
                 "make_amplitude_heisenberg_action", "build_n36_pi_zero_flux_orbit",
                 "projected_singlet_agp_amplitudes", "slater_minor_updates"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not forbidden.intersection(names)
    # Pure nonlinear coefficient evaluation of saved features is allowed.
    assert "multicontext_feature_coefficients" in names


def _mock_run(tmp_path, monkeypatch, *, optimizer=None):
    spec = _spec()
    spec["output"] = "synthetic_distillation"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec))
    output = tmp_path / spec["output"]
    events = []
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "registration", lambda _: {"config": spec, "synthetic": True})
    initial = {"toy": jnp.zeros(1)}
    reference = jnp.zeros((6, 6, 10))
    table = lambda x: jnp.zeros((6, 6, 10)) + 0 * jnp.sum(x)
    teachers = {name: {"parameters": jnp.zeros(1), "generator_table": reference,
                      "coefficients": jnp.zeros(1176)}
                for name in ("additive110", "free360", "attention730")}
    monkeypatch.setattr(workflow, "load_saved_problem", lambda _: (
        workflow.MulticontextResonanceAttentionConfig(), jnp.zeros((2, 2)), initial,
        reference, jnp.zeros(1), table, teachers))
    monkeypatch.setattr(workflow, "init_multicontext_resonance_attention", lambda *args: initial)
    monkeypatch.setattr(workflow, "multicontext_feature_coefficients", lambda *args: jnp.zeros(1176))
    monkeypatch.setattr(workflow, "gauge_invariant_table_loss", lambda table, target: jnp.sum(table**2))
    Physical = namedtuple("Physical", "innovation_relative_mse h_response_relative_mse")
    Table = namedtuple("Table", "relative_mse")
    monkeypatch.setattr(workflow, "physical_distillation_metrics", lambda p, *args: Physical(jnp.sum(p*0).real, jnp.sum(p*0).real))
    monkeypatch.setattr(workflow, "table_distillation_metrics", lambda *args: Table(jnp.asarray(0.)))
    monkeypatch.setattr(workflow, "head_rank_certificate", lambda *args, **kwargs: {"synthetic_fixture": True})

    def load_corpus(_source, phase):
        if phase == "audit":
            assert events == ["train", *("fit" for _ in range(14))]
            seals = json.loads((output / "endpoints_seal.json").read_text())
            assert len(seals) == 14
            for name, pair in seals.items():
                for ext, sha in pair.items():
                    assert workflow.file_sha256(output / f"{name}.{ext}") == sha
        events.append(phase)
        return {"carrier": jnp.ones(4), "hcarrier": -jnp.ones(4),
                "features": jnp.zeros((4, 1176)), "hfeatures": jnp.zeros((4, 1176)),
                "q": jnp.ones(4)}
    monkeypatch.setattr(workflow, "load_corpus", load_corpus)

    def minimize(fun, x, *, jac, method, callback, options):
        assert "audit" not in events
        assert jac is True and method == "L-BFGS-B"
        assert options["maxiter"] == 1000 and options["maxfun"] == 1500
        events.append("fit")
        if optimizer is not None:
            return optimizer(fun, x, callback)
        fun(x)
        callback(x)
        return SimpleNamespace(x=x, success=True, message="synthetic optimizer termination")
    monkeypatch.setattr(workflow, "minimize", minimize)
    return config, output, events


def test_all_fourteen_final_endpoints_are_sealed_before_seen_transfer_is_loaded(tmp_path, monkeypatch):
    config, output, events = _mock_run(tmp_path, monkeypatch)
    result = workflow.run(config)
    assert events == ["train", *("fit" for _ in range(14)), "audit"]
    expected = {f"{teacher}__{start}__{phase}" for teacher in _spec()["teachers"]
                for start in _spec()["starts"] for phase in ("table", "physical")}
    expected |= {f"{teacher}__direct_physical" for teacher in _spec()["teachers"]}
    assert set(result["fits"]) == expected
    assert result["status"] == "complete"
    assert not result["fresh_audit"] and not result["variational_win_claimed"]
    assert not result["remote_released"]
    assert result["new_physical_H_calls"] == result["new_carrier_queries"] == result["new_samples"] == 0
    for record in result["fits"].values():
        assert record["optimizer_function_evaluations"] == 2
        assert record["final_diagnostic_evaluations"] == 1
    assert not (output / "failure.json").exists()


def test_existing_distillation_attempt_is_not_overwritten(tmp_path, monkeypatch):
    config, output, _ = _mock_run(tmp_path, monkeypatch)
    output.mkdir()
    sentinel = output / "registration.json"
    sentinel.write_text('{"sealed":true}')
    monkeypatch.setattr(workflow, "load_saved_problem", _forbid)
    with pytest.raises(FileExistsError):
        workflow.run(config)
    assert sentinel.read_text() == '{"sealed":true}'


@pytest.mark.parametrize("exception", (MemoryError, TimeoutError))
def test_resource_failure_stops_without_transfer_or_automatic_extension(tmp_path, monkeypatch, exception):
    def fail(*args):
        raise exception("synthetic bounded failure")
    config, output, events = _mock_run(tmp_path, monkeypatch, optimizer=fail)
    with pytest.raises(exception):
        workflow.run(config)
    assert events == ["train", "fit"]
    assert not (output / "results.json").exists()
    assert json.loads((output / "failure.json").read_text())["automatic_extension"] is False


def test_optimizer_function_cap_preserves_last_accepted_not_latest_trial(tmp_path, monkeypatch):
    def capped(fun, x, callback):
        accepted = np.asarray(x) + .1
        fun(accepted)
        callback(accepted)
        # Simulate a line-search trial, then the wrapper's strict budget exit.
        fun(accepted + .2)
        raise StopIteration("strict function budget")
    config, output, events = _mock_run(tmp_path, monkeypatch, optimizer=capped)
    result = workflow.run(config)
    assert events[-1] == "audit" and len(result["fits"]) == 14
    first = "additive110__old_endpoint__table"
    manifest = json.loads((output / f"{first}.json").read_text())
    assert manifest["identity"]["selection"] == "optimizer final accepted iterate"
    with np.load(output / f"{first}.npz", allow_pickle=False) as data:
        # Sorted artifact keys: coefficients, generator_table, parameters.
        np.testing.assert_allclose(data["leaf_0002"], [.1], atol=1e-15)
    assert all(not record["optimizer_success"] for record in result["fits"].values())


def _completed_artifacts():
    folder = workflow.ROOT / _spec()["output"]
    if not (folder / "results.json").exists():
        pytest.skip("distillation artifacts are not present in this checkout")
    result = json.loads((folder / "results.json").read_text())
    registration = json.loads((folder / "registration.json").read_text())
    assert result["status"] == "complete"
    return folder, result, registration


def _saved_endpoint(path, count):
    values = workflow.load_parameter_artifact(path, dict(
        parameters=jnp.zeros(count), generator_table=jnp.zeros((6, 6, 10)), coefficients=jnp.zeros(1176)))
    return {key: np.asarray(value) for key, value in values.items()}


def test_completed_distillation_seals_all_endpoints_sources_and_budget():
    folder, result, registration = _completed_artifacts()
    assert result["config"] == registration["config"] == _spec()
    assert workflow.file_sha256(folder / "registration.json") == result["registration_sha256"]
    assert workflow.canonical_sha256({k: v for k, v in registration.items() if k != "registration_sha256"}) == registration["registration_sha256"]
    assert workflow.canonical_sha256({k: v for k, v in result.items() if k != "result_sha256"}) == result["result_sha256"]
    for collection in ("sources", "inputs"):
        for path, digest in registration[collection].items():
            assert workflow.file_sha256(workflow.ROOT / path) == digest
    for name, digest in result["artifacts"].items():
        assert workflow.file_sha256(folder / name) == digest
    seals = json.loads((folder / "endpoints_seal.json").read_text())
    assert len(seals) == len(result["fits"]) == 14
    assert set(seals) == set(result["fits"])
    for name, pair in seals.items():
        for ext, digest in pair.items():
            assert workflow.file_sha256(folder / f"{name}.{ext}") == digest
        fit = result["fits"][name]
        assert 0 <= fit["iterations"] <= 1000
        assert len(fit["history"]) == fit["iterations"]
        assert 1 <= fit["optimizer_function_evaluations"] <= 1500
        assert fit["final_diagnostic_evaluations"] == 1
        assert np.isfinite(fit["final_loss"]) and np.isfinite(fit["final_gradient_norm"])
        assert json.loads((folder / f"{name}.json").read_text())["identity"]["selection"] == "optimizer final accepted iterate"
    assert result["new_physical_H_calls"] == result["new_carrier_queries"] == result["new_samples"] == 0
    assert result["reused_train_records"] == result["reused_seen_transfer_records"] == 128
    assert not result["fresh_audit"] and not result["variational_win_claimed"] and not result["remote_released"]
    assert result["seconds"] <= _spec()["wall_seconds_cap"]


def _numpy_table_metrics(candidate, target):
    identity = np.broadcast_to(np.asarray([1., 1., 0., 1., 1., 1., 0., 0., 0., 1.]), candidate.shape)
    def remove(value, weight):
        shift = np.sum(weight*value*identity) / np.sum(weight*identity**2)
        return value - shift*identity, shift
    error, shift = remove(candidate-target, np.ones_like(candidate))
    teacher, _ = remove(target, np.ones_like(target))
    dimensions = np.asarray([1., 1., 3., 3., 3., 5.])
    active = np.asarray([1., 1., 1., 3., 3., 3., 3., 3., 3., 5.])
    weight = dimensions[:, None, None]*dimensions[None, :, None]*active[None, None, :]
    hs_error, _ = remove(candidate-target, weight)
    hs_teacher, _ = remove(target, weight)
    return {"relative_mse": np.sum(error**2)/np.sum(teacher**2),
            "hs_relative_mse": np.sum(weight*hs_error**2)/np.sum(weight*hs_teacher**2),
            "absolute_mse": np.mean(error**2), "removed_identity_shift": shift, "defined": True}


def _numpy_physical_metrics(psi, hp, teacher, ht, c, hc, q):
    inner = lambda a, b: np.sum(a.conj()*b/q)
    norm = lambda a: inner(a, a).real
    nc, np_, nt = norm(c), norm(psi), norm(teacher)
    bp, bt = inner(c, psi)/nc, inner(c, teacher)/nc
    dp, dt = psi/bp-c, teacher/bt-c
    hdp, hdt = hp/bp-hc, ht/bt-hc
    shift = (inner(teacher, ht)/nt).real
    rp, rt = hdp-shift*dp, hdt-shift*dt
    error, herror = norm(dp-dt), norm(rp-rt)
    e = inner(psi, hp)/np_
    return dict(ray_infidelity=norm(psi-teacher*inner(teacher, psi)/nt)/np_,
        innovation_relative_mse=error/norm(dt), h_response_relative_mse=herror/norm(rt),
        innovation_absolute_mse=error/nc, h_response_absolute_mse=herror/nc,
        candidate_carrier_fidelity=abs(bp)**2*nc/np_, target_carrier_fidelity=abs(bt)**2*nc/nt,
        chart_valid=True, innovation_defined=True, h_response_defined=True, energy_shift=shift,
        energy_per_site=e.real/36, variance_per_site=norm(hp-e*psi)/np_/36)


def test_saved_physical_and_gauge_metrics_replay_independently_without_model_calls():
    folder, result, _ = _completed_artifacts()
    source = workflow.ROOT / _spec()["source_run"]
    corpora = {}
    for reported, phase in (("train", "train"), ("seen_transfer", "audit")):
        with np.load(source / f"{phase}_features.npz", allow_pickle=False) as f, np.load(
            source / f"{phase}_samples.npz", allow_pickle=False
        ) as s:
            corpora[reported] = {key: f[key].copy() for key in ("carrier", "hcarrier", "features", "hfeatures")}
            corpora[reported]["q"] = s["q"].copy()
    for teacher_name, count in (("additive110", 110), ("free360", 360)):
        teacher = _saved_endpoint(source / f"{teacher_name}_endpoint.json", count)
        for name, report in result["evaluations"][teacher_name].items():
            if name == "teacher":
                candidate = teacher
            elif name == "old_attention":
                candidate = _saved_endpoint(source / "attention730_endpoint.json", 730)
            else:
                candidate = _saved_endpoint(folder / f"{name}.json", 730)
            tm = _numpy_table_metrics(candidate["generator_table"], teacher["generator_table"])
            for key, value in tm.items():
                assert report["table"][key] == pytest.approx(value, abs=2e-12, rel=2e-11)
            for phase, corpus in corpora.items():
                c, hc, f, hf, q = (corpus[key] for key in ("carrier", "hcarrier", "features", "hfeatures", "q"))
                psi, hp = c+f@candidate["coefficients"], hc+hf@candidate["coefficients"]
                target, ht = c+f@teacher["coefficients"], hc+hf@teacher["coefficients"]
                metrics = _numpy_physical_metrics(psi, hp, target, ht, c, hc, q)
                for key, value in metrics.items():
                    assert report[phase][key] == pytest.approx(value, abs=3e-11, rel=3e-10)
            if name in result["fits"]:
                expected_loss = (tm["relative_mse"] if name.endswith("__table") else
                    .5*(report["train"]["innovation_relative_mse"] + report["train"]["h_response_relative_mse"]))
                assert result["fits"][name]["final_loss"] == pytest.approx(expected_loss, abs=2e-11, rel=2e-10)


def test_rank8_necessary_table_bound_is_reproduced_but_not_used_as_physical_no_go():
    folder, result, _ = _completed_artifacts()
    source = workflow.ROOT / _spec()["source_run"]
    with np.load(source / "initial.npz", allow_pickle=False) as saved:
        anchors = [saved[key] for key in saved.files if saved[key].shape == (6, 6, 10)]
        assert len(anchors) == 1
        anchor = anchors[0]
    for name, count in (("additive110", 110), ("free360", 360)):
        teacher = _saved_endpoint(source / f"{name}_endpoint.json", count)["generator_table"]
        raw = (teacher+anchor).reshape(36, 10)
        singular = np.linalg.svd(raw-raw.mean(axis=0), compute_uv=False)
        bound = np.linalg.norm(singular[8:])
        certificate = result["rank_certificates"][name]
        np.testing.assert_allclose(certificate["weighted_singular_values"], singular, atol=2e-14, rtol=2e-12)
        assert certificate["weighted_frobenius_error_lower_bound"] == pytest.approx(bound, abs=2e-14)
        assert certificate["weighted_mean_square_error_lower_bound"] == pytest.approx(bound**2/360, abs=2e-15)
        assert certificate["context_centered_raw_rank_limit"] == 8
        assert certificate["physical_state_replay_excluded"] is False
        assert certificate["nonlinear_trunk_achievability_certified"] is False
        for fit in result["fits"]:
            if fit.startswith(name + "__"):
                candidate = _saved_endpoint(folder / f"{fit}.json", 730)["generator_table"]
                actual_raw = (candidate+anchor).reshape(36, 10)
                actual_singular = np.linalg.svd(actual_raw-actual_raw.mean(axis=0), compute_uv=False)
                assert np.linalg.norm(actual_singular[8:]) <= 2e-12 * max(1., np.linalg.norm(actual_singular))
                metrics = _numpy_table_metrics(candidate, teacher)
                assert metrics["absolute_mse"] >= bound**2/360 - 2e-13
