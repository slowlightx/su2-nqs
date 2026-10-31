"""Fail-closed bounded-training protocol checks; never execute an N36 model.

Resource tests replace the entire physical engine before it is constructed.
These checks do not draw configurations, prepare H features, or fit a state.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_resonance_training as workflow


def _spec():
    return json.loads(workflow.CONFIG.read_text())


def _write_config(path: Path, spec: dict) -> Path:
    path.write_text(json.dumps(spec))
    return path


def _forbid_engine(*args, **kwargs):
    raise AssertionError("test reached forbidden N36 physical engine")


def _isolated_run(tmp_path, monkeypatch):
    """Use synthetic registration; all physical source access is disabled."""
    spec = _spec()
    spec["output"] = "bounded_attempt"
    config = _write_config(tmp_path / "config.json", spec)
    registered = {"config": spec, "test_identity": "no physical evaluation"}
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "registration", lambda _: registered)
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", _forbid_engine)
    return spec, config, tmp_path / spec["output"], registered


def test_registered_design_is_shared_physical_parent_and_fresh_fixed_endpoint_audit():
    spec = _spec()
    assert workflow.validate_protocol(spec) is None
    assert (spec["geometry"], spec["length"], spec["j2_over_j1"]) == ("triangular", 6, .125)
    assert spec["arms"] == ["additive110", "free360", "attention730"]
    assert spec["chains"] * spec["train_blocks"] == 128
    assert spec["chains"] * spec["audit_blocks"] == 128
    assert spec["training_steps"] == 60
    assert spec["step_bisections"] + 1 == 12
    assert spec["step_fs_radius"] == .002
    assert spec["burn_proposals_per_chain"] == 4096
    assert spec["decorrelation_proposals_per_record"] == 128
    assert spec["uniform_floor_relative_to_pilot_mean"] == .2
    assert spec["bridge_fraction"] == .5
    assert spec["uniform_scale_samples"] == 4096
    assert spec["primary_contrast"] == "free360-minus-additive110"
    assert spec["remote_allowed"] is False
    assert spec["wall_seconds_cap"] <= 3600
    assert spec["peak_rss_gib_cap"] <= 4.
    assert "BOTH bra and ket" in spec["objective"]
    assert "No claim that finite-Q training energies are variational bounds" in spec["objective"]
    assert "ONLY after all fixed60-step endpoints are sealed" in spec["audit"]
    assert "Do not call bounded null fully optimized" in spec["audit"]


@pytest.mark.parametrize("field,value", [
    ("task", "different_task"), ("geometry", "square"), ("length", 8),
    ("j2_over_j1", .1), ("arms", ["free360", "additive110", "attention730"]),
    ("remote_allowed", True), ("training_steps", 61), ("chains", 8),
    ("train_blocks", 16), ("audit_blocks", 16), ("step_bisections", 12),
    ("step_fs_radius", .004), ("primary_contrast", "attention730-minus-additive110"),
    ("microbatch", 4), ("minor_max_rank", 6), ("wall_seconds_cap", 0),
    ("peak_rss_gib_cap", float("nan")), ("minor_batch_size", 0),
])
def test_protocol_mutation_is_rejected_before_numerical_runtime(tmp_path, monkeypatch, field, value):
    spec = _spec()
    spec[field] = value
    path = _write_config(tmp_path / "invalid.json", spec)
    monkeypatch.setattr(workflow, "numerical_runtime_identity", _forbid_engine)
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", _forbid_engine)
    with pytest.raises(ValueError):
        workflow.run(path)


def test_train_audit_and_initialization_seeds_are_not_reused():
    spec = _spec()
    fields = ("initialization_seed", "scale_seed", "train_seed", "audit_seed", "bootstrap_seed")
    assert len({spec[key] for key in fields}) == len(fields)
    for key in fields:
        changed = dict(spec)
        changed[key] = spec["train_seed"] if key != "train_seed" else spec["audit_seed"]
        with pytest.raises(ValueError):
            workflow.validate_protocol(changed)


def test_registered_attempt_cannot_be_overwritten(tmp_path, monkeypatch):
    _, config, output, registered = _isolated_run(tmp_path, monkeypatch)
    output.mkdir()
    path = output / "registration.json"
    path.write_text(json.dumps(registered))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        workflow.run(config)
    assert path.read_bytes() == before


def test_resume_rejects_changed_registration_before_engine(tmp_path, monkeypatch):
    _, config, output, registered = _isolated_run(tmp_path, monkeypatch)
    output.mkdir()
    (output / "registration.json").write_text(json.dumps({**registered, "test_identity": "other"}))
    with pytest.raises(ValueError, match="changed"):
        workflow.run(config, resume=True)


def test_orphan_outputs_are_not_adopted_under_a_new_registration(tmp_path, monkeypatch):
    _, config, output, _ = _isolated_run(tmp_path, monkeypatch)
    output.mkdir()
    sentinel = output / "train_samples.npz"
    sentinel.write_bytes(b"unregistered prior artifact")
    with pytest.raises(FileExistsError):
        workflow.run(config)
    assert sentinel.read_bytes() == b"unregistered prior artifact"
    assert not (output / "registration.json").exists()


def test_resume_counts_failure_time_and_stops_before_engine_when_budget_exhausted(tmp_path, monkeypatch):
    spec, config, output, registered = _isolated_run(tmp_path, monkeypatch)
    output.mkdir()
    (output / "registration.json").write_text(json.dumps(registered))
    (output / "progress.json").write_text(json.dumps({"seconds": 10.}))
    (output / "failure.json").write_text(json.dumps({"seconds": spec["wall_seconds_cap"] + 1.}))
    with pytest.raises(TimeoutError):
        workflow.run(config, resume=True)
    assert not (output / "results.json").exists()


@pytest.mark.parametrize("exception", (MemoryError, TimeoutError))
def test_resource_failure_is_recorded_without_automatic_retry_or_result(tmp_path, monkeypatch, exception):
    _, config, output, _ = _isolated_run(tmp_path, monkeypatch)
    calls = []
    def fail_engine(_):
        calls.append("once")
        raise exception("synthetic resource cap; no physical work")
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", fail_engine)
    with pytest.raises(exception):
        workflow.run(config)
    assert calls == ["once"]
    report = json.loads((output / "failure.json").read_text())
    assert report["automatic_extension"] is False
    assert report["seconds"] >= 0 and np.isfinite(report["seconds"])
    assert not (output / "results.json").exists()
    assert not (output / "audit_samples.npz").exists()


def test_endpoint_structure_has_sample_independent_bound_and_no_observable_claim():
    table = np.zeros((6, 6, 10))
    # A reduced q0 diagonal is a unit matrix element in every context.
    table[..., 0] = .03
    record = workflow.endpoint_structure(table)
    assert record["generator_max_spectral_norm"] == pytest.approx(.03)
    assert record["rigorous_relative_state_change_bound"] == pytest.approx(np.expm1(.06))
    assert record["joint_generator_norm"] < 1e-14
    assert not {"energy", "entanglement", "physical_correlation"}.intersection(record)


def _mock_completed_pipeline(tmp_path, monkeypatch, *, tamper=None):
    """Exercise orchestration using constants, never a physical amplitude."""
    spec, config, output, _ = _isolated_run(tmp_path, monkeypatch)
    events = []
    class ConstantEngine:
        def __init__(self, _):
            self.carrier = lambda x: np.ones(len(x))
            self.bonds = ()
    monkeypatch.setattr(workflow, "FixedCarrierFeatures", ConstantEngine)
    monkeypatch.setattr(workflow, "init_multicontext_resonance_attention", lambda *args: {"toy": np.zeros(1)})
    monkeypatch.setattr(workflow, "make_multicontext_reference", lambda *args: np.zeros((6, 6, 10)))
    monkeypatch.setattr(workflow, "peak_rss_gib", lambda: .1)

    def draw(_spec, _n, _carrier, _bonds, _scale, seed, blocks):
        phase = "train" if seed == spec["train_seed"] else "audit"
        assert seed == spec[phase + "_seed"]
        if phase == "audit":
            seal = json.loads((output / "endpoints_seal.json").read_text())
            for arm in workflow.ARMS:
                for ext in ("json", "npz"):
                    assert seal[arm][ext] == workflow.file_sha256(output / f"{arm}_endpoint.{ext}")
            assert events == ["train_draw", *workflow.ARMS]
        events.append(phase + "_draw")
        count = blocks * spec["chains"]
        return dict(states=np.zeros((count, 36), dtype=np.int8), q=np.ones(count),
                    carrier=np.ones(count), hcarrier=np.full(count, -.5),
                    groups=np.tile(np.arange(spec["chains"]), blocks)), {
                        "synthetic_fixture": True, "carrier_H_centers": count}

    def corpus(_engine, data, _scale, path, _progress):
        if path.stem == "audit_features" and tamper is not None:
            changed = output / f"free360_endpoint.{tamper}"
            changed.write_bytes(changed.read_bytes() + b"changed after seal")
        count = len(data["states"])
        return dict(carrier=np.ones(count), hcarrier=np.full(count, -18.),
                    features=np.zeros((count, 1176)), hfeatures=np.zeros((count, 1176)))

    def train(arm, _spec, _data, _corpus, folder, _initial, _reference, _progress):
        assert _spec["training_steps"] == 60
        assert "audit_draw" not in events
        events.append(arm)
        endpoint = folder / f"{arm}_endpoint.json"
        workflow.save_parameter_artifact(endpoint, {"coefficients": np.zeros(1176)},
            identity={"arm": arm, "selection": "fixed final step"})
        return {"fixed_endpoint": True}, np.zeros(1176)

    def audit(*args):
        assert events[-1] == "audit_draw"
        events.append("independent_audit")
        return {"paired": {spec["primary_contrast"]: {
            "energy_per_site": {"chain_bootstrap_95": [-.001, .001]},
            "variance_per_site": {"chain_bootstrap_95": [-.001, .001]},
        }}, "arms": {arm: {"max_residual_chain_fraction": .1} for arm in workflow.ARMS}}

    monkeypatch.setattr(workflow, "draw", draw)
    monkeypatch.setattr(workflow, "build_corpus", corpus)
    monkeypatch.setattr(workflow, "train_arm", train)
    monkeypatch.setattr(workflow, "paired_chain_audit", audit)
    return config, output, events


def test_audit_draw_occurs_only_after_all_three_endpoint_json_and_npz_seals(tmp_path, monkeypatch):
    config, output, events = _mock_completed_pipeline(tmp_path, monkeypatch)
    result = workflow.run(config)
    assert events == ["train_draw", *workflow.ARMS, "audit_draw", "independent_audit"]
    assert result["status"] == "complete"
    assert result["physical_H_prepared_centers"] == 256
    assert result["feature_coordinates"] == 1176 and result["group_order"] == 432
    assert result["variational_win_claimed"] is False and result["remote_released"] is False
    assert result["primary_energy_interval_below_zero"] is False
    assert not (output / "failure.json").exists()


@pytest.mark.parametrize("extension", ("json", "npz"))
def test_endpoint_mutation_during_audit_is_rejected_before_statistics(tmp_path, monkeypatch, extension):
    config, output, events = _mock_completed_pipeline(tmp_path, monkeypatch, tamper=extension)
    with pytest.raises(ValueError, match="endpoint changed"):
        workflow.run(config)
    assert events[-1] == "audit_draw" and "independent_audit" not in events
    assert not (output / "results.json").exists()
    assert json.loads((output / "failure.json").read_text())["automatic_extension"] is False


def test_interrupted_single_file_checkpoint_commit_preserves_previous_whole_step(tmp_path, monkeypatch):
    path = tmp_path / "free360_checkpoint.npz"
    old = dict(x=np.asarray([1., 2.]), m=np.asarray([.1, .2]), v=np.asarray([.01, .04]),
               arm=np.asarray("free360"), completed_steps=np.asarray(1),
               history=np.asarray(json.dumps([{"step": 1}])))
    workflow._save_npz(path, **old)
    before = path.read_bytes()
    original = Path.replace
    def interrupted_replace(source, target):
        if source == path.with_suffix(".tmp"):
            raise OSError("synthetic interruption before atomic replacement")
        return original(source, target)
    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="synthetic interruption"):
        workflow._save_npz(path, **{**old, "x": np.asarray([8., 9.]), "completed_steps": np.asarray(2)})
    assert path.read_bytes() == before
    with np.load(path, allow_pickle=False) as saved:
        for key, value in old.items():
            np.testing.assert_array_equal(saved[key], value)


@pytest.mark.parametrize("steps", ([1, 3], list(range(1, 62))))
def test_invalid_checkpoint_history_fails_before_objective_evaluation(tmp_path, steps):
    spec = _spec()
    workflow._save_npz(tmp_path / "free360_checkpoint.npz", x=np.zeros(360), m=np.zeros(360), v=np.zeros(360),
        arm=np.asarray("free360"), completed_steps=np.asarray(len(steps)),
        history=np.asarray(json.dumps([{"step": step} for step in steps])))
    corpus = dict(carrier=np.ones(2), hcarrier=np.ones(2),
                  features=np.zeros((2, 1176)), hfeatures=np.zeros((2, 1176)))
    with pytest.raises(ValueError, match="ordered bounded trajectory"):
        workflow.train_arm("free360", spec, {"q": np.ones(2)}, corpus, tmp_path,
                           {}, np.zeros((6, 6, 10)), lambda *args, **kwargs: None)
    assert not (tmp_path / "free360_endpoint.json").exists()


def _completed_training_artifacts():
    spec = _spec()
    folder = workflow.ROOT / spec["output"]
    if not (folder / "results.json").exists():
        pytest.skip("bounded training has not produced a completed artifact")
    result = json.loads((folder / "results.json").read_text())
    assert result["status"] == "complete"
    registration = json.loads((folder / "registration.json").read_text())
    return folder, result, registration


def test_completed_training_provenance_and_fixed_work_ledger():
    folder, result, registration = _completed_training_artifacts()
    assert registration["config"] == result["config"] == _spec()
    registered_hash = registration["registration_sha256"]
    assert workflow.canonical_sha256({k: v for k, v in registration.items()
                                     if k != "registration_sha256"}) == registered_hash
    assert workflow.file_sha256(folder / "registration.json") == result["registration_sha256"]
    assert workflow.canonical_sha256({k: v for k, v in result.items()
                                     if k != "result_sha256"}) == result["result_sha256"]
    for relative, digest in registration["inputs"].items():
        assert workflow.file_sha256(workflow.ROOT / relative) == digest
    for relative, digest in registration["sources"].items():
        assert workflow.file_sha256(workflow.ROOT / relative) == digest
    for name, digest in result["artifacts"].items():
        assert workflow.file_sha256(folder / name) == digest
    assert result["physical_H_prepared_centers"] == 256
    assert result["additional_cheap_carrier_H_centers"] == 256
    assert result["full_H_feature_cost_shared_across_arms"]
    assert result["feature_coordinates"] == 1176 and result["group_order"] == 432
    assert result["bonds"] == 216
    assert result["seconds"] <= _spec()["wall_seconds_cap"]
    assert result["peak_rss_gib"] <= _spec()["peak_rss_gib_cap"]
    assert not result["equilibrium_certified"] and not result["variational_win_claimed"]
    assert result["remote_released"] is False
    for arm, dimension in zip(workflow.ARMS, (110, 360, 730), strict=True):
        report = result["training"][arm]
        assert report["parameter_count"] == dimension
        assert report["gradient_calls"] == 60 and report["step_candidate_calls"] == 720
        assert report["cached_state_pair_evaluations"] == 901
        assert [row["step"] for row in report["history"]] == list(range(1, 61))
        for row in report["history"]:
            step = row["physical_step"]
            assert step["evaluation_count"] == len(step["records"]) == 12
            assert step["distance"]["fs"] <= .002 + 2e-14
            assert row["gradient_norm"] >= 0 and np.isfinite(row["gradient_norm"])
        with np.load(folder / f"{arm}_checkpoint.npz", allow_pickle=False) as checkpoint:
            assert int(checkpoint["completed_steps"]) == 60 and str(checkpoint["arm"]) == arm
            assert json.loads(str(checkpoint["history"])) == report["history"]
            assert checkpoint["x"].shape == checkpoint["m"].shape == checkpoint["v"].shape == (dimension,)


def test_completed_features_and_endpoints_replay_without_new_amplitudes_or_h():
    folder, result, _ = _completed_training_artifacts()
    seals = json.loads((folder / "endpoints_seal.json").read_text())
    coefficients = [np.zeros(1176)]
    for arm, dimension in zip(workflow.ARMS, (110, 360, 730), strict=True):
        for extension in ("json", "npz"):
            assert workflow.file_sha256(folder / f"{arm}_endpoint.{extension}") == seals[arm][extension]
        assert seals[arm]["json"] == result["training"][arm]["endpoint_sha256"]
        endpoint = workflow.load_parameter_artifact(folder / f"{arm}_endpoint.json",
            {"parameters": np.zeros(dimension), "generator_table": np.zeros((6, 6, 10)),
             "coefficients": np.zeros(1176)},
            expected_identity={"arm": arm, "completed_steps": 60, "selection": "fixed final step"})
        coefficients.append(np.asarray(endpoint["coefficients"]))
    coefficients = np.stack(coefficients, axis=1)
    for phase in ("train", "audit"):
        with np.load(folder / f"{phase}_samples.npz", allow_pickle=False) as samples, np.load(
            folder / f"{phase}_features.npz", allow_pickle=False
        ) as features:
            assert samples["states"].shape == (128, 36)
            assert np.all(samples["states"].sum(axis=1) == 18)
            assert np.all(np.isfinite(samples["q"])) and np.all(samples["q"] > 0)
            labels, counts = np.unique(samples["groups"], return_counts=True)
            assert len(labels) == 16 and np.all(counts == 8)
            assert features["features"].shape == features["hfeatures"].shape == (128, 1176)
            assert features["carrier"].shape == features["hcarrier"].shape == (128,)
            seal = workflow.canonical_sha256({key: samples[key].tolist() for key in ("states", "q", "groups")})
            assert str(features["sample_seal"]) == seal
            records = json.loads(str(features["records"]))
            assert len(records) == 64 and sum(row["full_H_centers"] for row in records) == 128
            assert all(row["group_order"] == 432 and row["bond_count"] == 216 for row in records)
            np.testing.assert_allclose(features["carrier"], samples["carrier"], atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(features["hcarrier"], 36 * samples["hcarrier"], atol=1e-11, rtol=1e-10)
            if phase == "audit":
                with np.load(folder / "audit_values.npz", allow_pickle=False) as audit:
                    np.testing.assert_array_equal(audit["coefficients"], coefficients)
                    np.testing.assert_array_equal(audit["q"], samples["q"])
                    np.testing.assert_array_equal(audit["groups"], samples["groups"])
                    np.testing.assert_allclose(audit["psi"], features["carrier"][:, None] + features["features"] @ coefficients,
                                               atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(audit["hpsi"], features["hcarrier"][:, None] + features["hfeatures"] @ coefficients,
                                               atol=1e-11, rtol=1e-12)
    assert _spec()["train_seed"] != _spec()["audit_seed"]
    # Independent random draws may share a configuration. Do not enforce
    # artificially disjoint bitstrings or drop such legitimate coincidences.


def test_completed_manual_resource_resume_keeps_history_and_cumulative_budget():
    folder, result, _ = _completed_training_artifacts()
    failure_path = folder / "failure.json"
    if not failure_path.exists():
        pytest.skip("this artifact did not require a manual resource resume")
    failure = json.loads(failure_path.read_text())
    assert failure["automatic_extension"] is False
    assert failure["seconds"] < result["seconds"] <= _spec()["wall_seconds_cap"]
    historical_peaks = [float(failure["peak_rss_gib"]), float(result["peak_rss_gib"])]
    prepared_seconds = 0.
    for phase in ("train", "audit"):
        with np.load(folder / f"{phase}_features.npz", allow_pickle=False) as saved:
            records = json.loads(str(saved["records"]))
            assert len(records) == 64 and sum(record["full_H_centers"] for record in records) == 128
            historical_peaks.extend(float(record["peak_rss_gib"]) for record in records)
            prepared_seconds += sum(record["seconds"] for record in records)
    assert prepared_seconds <= result["seconds"]
    if "MemoryError" in failure["error"]:
        # The cap is checked after each bounded two-center preparation. A
        # finite guard overshoot is preserved, never silently recast as a
        # whole-experiment peak below the cap after process isolation.
        assert max(historical_peaks) >= failure["peak_rss_gib"] > _spec()["peak_rss_gib_cap"]
    assert not result["remote_released"]


def test_completed_paired_moments_uncertainty_and_tails_independent_numpy_replay():
    folder, result, _ = _completed_training_artifacts()
    with np.load(folder / "audit_values.npz", allow_pickle=False) as saved:
        psi, hp, q, groups = (saved[key] for key in ("psi", "hpsi", "q", "groups"))
        assert psi.shape == hp.shape == (128, 4)
        names = ("carrier", *workflow.ARMS)
        observations = np.stack((abs(psi)**2 / q[:, None], psi.conj()*hp / q[:, None],
                                 abs(hp)**2 / q[:, None]), axis=-1)
        labels = np.unique(groups)
        grouped = np.stack([observations[groups == label].sum(axis=0) for label in labels])
        total = observations.sum(axis=0)

        def moments(raw):
            norm = raw[..., 0].real
            energy = raw[..., 1] / norm
            variance = (raw[..., 2].real / norm - abs(energy)**2) / 36
            return energy / 36, variance

        energy, variance = moments(total)
        leave_e, leave_v = moments(total[None] - grouped)
        multiplicities = np.random.default_rng(_spec()["bootstrap_seed"]).multinomial(
            len(labels), np.full(len(labels), 1/len(labels)), size=_spec()["bootstrap_replicates"])
        boot_e, boot_v = moments(np.einsum("bc,cam->bam", multiplicities, grouped))
        for index, name in enumerate(names):
            report = result["audit"]["arms"][name]
            assert report["energy_per_site"]["value"] == pytest.approx(energy[index].real, abs=2e-13)
            assert report["imaginary_energy_per_site"]["value"] == pytest.approx(energy[index].imag, abs=2e-13)
            assert report["variance_per_site"]["value"] == pytest.approx(variance[index], abs=5e-12)
            weight = observations[:, index, 0].real
            residual = abs(hp[:, index] - 36*energy[index]*psi[:, index])**2 / q
            assert report["stable_residual_variance_per_site"] == pytest.approx(residual.sum()/weight.sum()/36, abs=3e-13)
            assert report["weight_ess_fraction"] == pytest.approx(weight.sum()**2/(128*np.sum(weight**2)), abs=3e-14)
            assert report["exact_node_samples"] == np.count_nonzero(psi[:, index] == 0)
            if residual.sum() > 0:
                assert report["max_residual_sample_fraction"] == pytest.approx(residual.max()/residual.sum(), abs=2e-12)
                assert report["max_residual_chain_fraction"] == pytest.approx(
                    max(residual[groups == label].sum() for label in labels)/residual.sum(), abs=2e-12)
        for old, new in combinations(range(4), 2):
            report = result["audit"]["paired"][f"{names[new]}-minus-{names[old]}"]
            for metric, point, leave, boot in (
                ("energy_per_site", energy.real, leave_e.real, boot_e.real),
                ("variance_per_site", variance, leave_v, boot_v),
            ):
                point_delta = point[new] - point[old]
                leave_delta = leave[:, new] - leave[:, old]
                boot_delta = boot[:, new] - boot[:, old]
                jackknife = np.sqrt((len(labels)-1)/len(labels) * np.sum((leave_delta-leave_delta.mean())**2))
                assert report[metric]["value"] == pytest.approx(point_delta, abs=5e-12)
                assert report[metric]["chain_jackknife_se"] == pytest.approx(jackknife, abs=5e-12)
                assert report[metric]["chain_bootstrap_se"] == pytest.approx(boot_delta.std(ddof=1), abs=5e-12)
                np.testing.assert_allclose(report[metric]["chain_bootstrap_95"], np.quantile(boot_delta, [.025, .975]),
                                           atol=5e-12, rtol=1e-10)
                np.testing.assert_allclose(report[metric]["leave_one_chain_out_range"],
                                           [leave_delta.min(), leave_delta.max()], atol=5e-12, rtol=1e-10)
