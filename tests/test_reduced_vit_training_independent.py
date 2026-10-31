"""Independent physical oracles for the native singlet ViT training canary.

These tests use an analytic four-spin singlet isometry and explicit bit-basis
exchange matrix. They do not infer correctness from an N=36 energy curve or
advertise the native autoregressive Born law as a spatially projected law.
"""

from itertools import product

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    sample_many_cell_schur_vit,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.dual_sampled_sr import make_dual_sampled_sr_update
from su2_nqs.vmc.many_cell_tree import (
    apply_many_cell_tree_connectivity,
    build_many_cell_tree_connectivity,
)


def _bit_hamiltonian(bonds):
    """Explicit full-physical-basis H[new,old], bit zero = down."""
    bits = np.asarray(list(product((0, 1), repeat=4)), dtype=np.int32)
    lookup = {tuple(row): index for index, row in enumerate(bits)}
    h = np.zeros((16, 16), dtype=np.float64)
    for old, row in enumerate(bits):
        for first, second, coupling in bonds:
            if row[first] == row[second]:
                h[old, old] += coupling / 4
            else:
                h[old, old] -= coupling / 4
                flipped = row.copy()
                flipped[first], flipped[second] = flipped[second], flipped[first]
                h[lookup[tuple(flipped)], old] += coupling / 2
    return h


def _analytic_singlet_problem():
    config = ManyCellSchurViTConfig(
        rectangular_four_spin_layout(2, 2), embedding_dim=8, head_count=2,
        depth=1, geometry="square", direct_phase_initialization_scale=0.4,
    )
    labels = np.asarray([[0, 0, 0], [2, 2, 0]], dtype=np.int32)
    # Explicit singlet x singlet, followed by triplet x triplet -> singlet.
    # This is not computed by the CG/recoupling implementation under test.
    u = np.zeros((16, 2), dtype=np.float64)
    u[[5, 6, 9, 10], 0] = [.5, -.5, -.5, .5]
    u[[3, 12], 1] = 1 / np.sqrt(3)
    u[[5, 6, 9, 10], 1] = -1 / np.sqrt(12)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=2e-15, rtol=0)
    s2 = 3 * np.eye(16) + 2 * _bit_hamiltonian(
        [(a, b, 1.) for a in range(4) for b in range(a + 1, 4)]
    )
    np.testing.assert_allclose(s2 @ u, 0., atol=2e-15, rtol=0)
    bonds = [(0, 1, 1.), (1, 3, .7), (3, 2, -.2), (2, 0, .5), (0, 3, .125)]
    physical = _bit_hamiltonian(bonds)
    h = u.T @ physical @ u
    workload = build_many_cell_tree_connectivity(labels, config, bonds)
    for column in range(2):
        actual = apply_many_cell_tree_connectivity(workload, np.eye(2)[:, column])
        np.testing.assert_allclose(actual, h[:, column], atol=2e-14, rtol=0)
    return config, labels, u, s2, physical, jnp.asarray(h)


def _uniform_born_params(config):
    params = init_many_cell_schur_vit(jax.random.PRNGKey(2026100811), config)
    # Retain the full nonzero backbone and nontrivial phase, but make this
    # two-state exact reference uniformly weighted so a balanced sample is
    # an independent exhaustive Born quadrature for both-sided derivatives.
    params["logit_context"] = jnp.zeros_like(params["logit_context"])
    params["logit_feature"] = jnp.zeros_like(params["logit_feature"])
    return params


def test_full_network_native_sr_has_physical_descent_and_actual_fs_geometry():
    config, labels, u, s2, physical, h = _analytic_singlet_problem()
    params = _uniform_born_params(config)
    balanced = jnp.asarray(np.tile(labels, (32, 1)))
    log_batch = lambda p, x: many_cell_schur_log_amplitudes(p, x, config)
    log_one = lambda p, x: log_batch(p, x[None, :])[0]
    old_c = jnp.exp(log_batch(params, labels))
    np.testing.assert_allclose(abs(old_c)**2, [.5, .5], atol=3e-14, rtol=0)
    local = h @ old_c / old_c
    update = make_dual_sampled_sr_update(
        log_one, params, damping=.13, step_size=.03, score_chunk_size=4,
        fisher_trust_radius=.04, maximum_overlap_distance=.04,
        minimum_ess_fraction=.8,
    )
    changed, diagnostics = update(params, balanced, jnp.tile(local, 32))
    assert bool(diagnostics["accepted"])
    assert float(diagnostics["parameter_step_norm"]) > 0
    assert int(diagnostics["score_real_entries"]) == 64 * 2 * len(ravel_pytree(params)[0])

    def energy(p):
        c = jnp.exp(log_batch(p, labels))
        return jnp.real(jnp.vdot(c, h @ c) / jnp.vdot(c, c))

    gradient = ravel_pytree(jax.grad(energy)(params))[0]
    np.testing.assert_allclose(
        diagnostics["gradient_norm"], jnp.linalg.norm(gradient), atol=3e-12, rtol=3e-12,
    )
    assert float(energy(changed)) < float(energy(params))
    new_c = np.exp(np.asarray(log_batch(changed, labels)))
    np.testing.assert_allclose(np.vdot(new_c, new_c), 1., atol=3e-14, rtol=0)
    old_psi, new_psi = u @ np.asarray(old_c), u @ new_c
    exact_fs = np.sqrt(max(0., 1. - abs(np.vdot(old_psi, new_psi))**2))
    np.testing.assert_allclose(diagnostics["sampled_fs_sine"], exact_fs, atol=3e-13, rtol=0)
    assert exact_fs <= .04 * (1 + 1e-12)
    np.testing.assert_allclose(s2 @ new_psi, 0., atol=3e-14, rtol=0)
    np.testing.assert_allclose(
        np.vdot(new_psi, physical @ new_psi), energy(changed), atol=3e-14, rtol=0,
    )


def test_trained_checkpoint_restores_optimizer_rng_and_next_complete_update(tmp_path):
    config, labels, _, _, _, h = _analytic_singlet_problem()
    params = _uniform_born_params(config)
    log_batch = lambda p, x: many_cell_schur_log_amplitudes(p, x, config)
    settings = dict(damping=.13, step_size=.03, score_chunk_size=4,
                    fisher_trust_radius=.04, maximum_overlap_distance=.04,
                    minimum_ess_fraction=.8)
    update = make_dual_sampled_sr_update(lambda p, x: log_batch(p, x[None, :])[0], params, **settings)
    local = h @ jnp.exp(log_batch(params, labels)) / jnp.exp(log_batch(params, labels))
    trained, first = update(params, jnp.asarray(np.tile(labels, (32, 1))), jnp.tile(local, 32))
    assert bool(first["accepted"])
    assert float(first["parameter_step_norm"]) > 0
    rng_key = jax.random.PRNGKey(2026100812)
    path = tmp_path / "trained.pkl"
    save_checkpoint(
        path, jax.tree_util.tree_map(np.asarray, trained),
        optimizer_state={"step": 1, "kind": "stateless_ridge_SR", "settings": settings},
        metadata={"next_training_key": np.asarray(rng_key), "exact_SU2": True, "full_A1": False},
    )
    restored = load_checkpoint(path)
    assert restored["optimizer_state"]["step"] == 1
    assert restored["optimizer_state"]["settings"] == settings
    assert restored["metadata"]["full_A1"] is False

    def continuation(p, key):
        next_key, draw_key = jax.random.split(key)
        draws = sample_many_cell_schur_vit(p, draw_key, 64, config)
        c = jnp.exp(log_batch(p, labels))
        local_energies = h @ c / c
        # Labels have analytically known singlet/triplet-pair row index.
        sampled_local = local_energies[draws[:, 0] // 2]
        newer, diagnostic = update(p, draws, sampled_local)
        return newer, next_key, draws, diagnostic

    uninterrupted = continuation(trained, rng_key)
    replayed = continuation(restored["params"], jnp.asarray(restored["metadata"]["next_training_key"]))
    for expected, actual in zip(
        jax.tree_util.tree_leaves(uninterrupted), jax.tree_util.tree_leaves(replayed), strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert float(uninterrupted[3]["parameter_step_norm"]) > 0


def test_complex_full_h_second_moment_matches_explicit_physical_residual():
    config, labels, u, _, physical, h = _analytic_singlet_problem()
    params = _uniform_born_params(config)
    c = np.exp(np.asarray(many_cell_schur_log_amplitudes(params, labels, config)))
    probability = abs(c)**2
    local = np.asarray(h) @ c / c
    mean = probability @ local
    variance = probability @ abs(local - mean)**2
    psi = u @ c
    explicit = np.linalg.norm(physical @ psi - mean * psi)**2
    np.testing.assert_allclose(variance, explicit, atol=3e-14, rtol=0)
    assert abs(mean.imag) < 3e-14
    # Real-part-only local-energy variance would omit genuine phase-sensitive
    # H psi fluctuations and is not the quantum variance Var(H).
    assert variance - probability @ (local.real - mean.real)**2 > 1e-5


def test_independent_geometry_backtracks_the_same_direction_not_the_audit():
    from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_train import calibrated_sr_update

    params = jnp.asarray([0.], dtype=jnp.float64)
    train = jnp.asarray([[-1.], [1.]])
    calibration = jnp.asarray([[-10.], [10.]])
    energy = jnp.asarray([1., -1.])
    logs = lambda p, x: (p[0] * x[:, 0]).astype(jnp.complex128)
    raw_update = make_dual_sampled_sr_update(
        lambda p, x: logs(p, x[None, :])[0], params,
        damping=.1, step_size=10., fisher_trust_radius=.3,
        maximum_overlap_distance=.05, minimum_ess_fraction=.8,
        score_chunk_size=1,
    )
    raw_candidate, raw_diag = raw_update(params, train, energy)
    assert bool(raw_diag["accepted"])
    raw_calibration_fs = np.sqrt(1 - abs(np.mean(np.exp(
        np.asarray(logs(raw_candidate, calibration)))))**2 / np.mean(np.exp(
        2 * np.asarray(logs(raw_candidate, calibration)).real)))
    assert raw_calibration_fs > .05
    settings = dict(geometry_backtracking_steps=6, maximum_overlap_distance=.05,
                    minimum_ess_fraction=.8)
    changed, diag = calibrated_sr_update(
        params, train, energy, calibration, raw_update, logs, settings,
    )
    assert diag["accepted"]
    assert 0 < diag["extra_applied_fraction"] < 1
    assert diag["independent_geometry_backtracking_evaluations"] > 1
    np.testing.assert_allclose(
        changed, params + diag["extra_applied_fraction"] * (raw_candidate - params),
        atol=1e-15, rtol=0,
    )
    for sample in (train, calibration):
        ratio = np.exp(np.asarray(logs(changed, sample) - logs(params, sample)))
        independent_fs = np.sqrt(max(0., 1 - abs(ratio.mean())**2 / np.mean(abs(ratio)**2)))
        assert independent_fs <= .05 * (1 + 1e-12)
    assert diag["geometry_is_global_certificate"] is False


def test_audit_statistics_use_complex_variance_and_unbiased_iid_errors():
    from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_train import local_energy_statistics

    values = np.asarray([-3 + 4j, -1 - 1j, 2 + 3j, 6 - 2j], dtype=np.complex128)
    result = local_energy_statistics(values, 36)
    center = values.mean()
    np.testing.assert_allclose(result["energy_per_site"], center.real / 36, atol=1e-16)
    np.testing.assert_allclose(
        result["energy_stderr_per_site"], np.sqrt(np.sum((values.real-center.real)**2) / 12) / 36,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        result["variance_per_site"], np.sum(abs(values-center)**2) / (3*36), atol=1e-16,
    )
    assert result["variance_per_site"] > values.real.var(ddof=1) / 36
    assert result["samples_are_native_iid"] is True
    assert result["full_A1"] is False


def test_registered_budget_and_rng_streams_reject_invalid_independence():
    import json
    from pathlib import Path
    import pytest
    from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_train import _validate

    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "projects/clebsch_tree_nqs/configs/validation/spin_reduced_vit_train.json").read_text())
    _validate(settings)
    assert settings["samples_per_step"] == 64
    assert settings["steps"] == 20
    assert settings["audit_samples"] == 512
    assert settings["total_seconds"] <= 3600
    assert settings["training_seconds"] <= 2700
    assert settings["full_A1"] is False
    with pytest.raises(ValueError, match="distinct"):
        _validate({**settings, "audit_seed": settings["training_seed"]})
    with pytest.raises(ValueError, match="one hour"):
        _validate({**settings, "total_seconds": 3601})
    with pytest.raises(ValueError, match="full A1"):
        _validate({**settings, "full_A1": True})


def test_independent_direction_slope_matches_explicit_complex_rayleigh_derivative():
    from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_train import directional_energy_statistics

    # Explicit four-dimensional physical vector, with no CG/model routine in
    # the oracle. Uniform Born support makes these four rows exact quadrature.
    psi = np.exp(1j*np.asarray([.3,-.8,.1,.7]))/2
    tangent = np.asarray([.3+.4j,-.7+.8j,.2-.1j,.6-.5j])
    h = np.asarray([[.2, .7j, -.6, .1],[-.7j,.9,.3,.2j],
                    [-.6,.3,-.1,.4j],[.1,-.2j,-.4j,.7]])
    local = h@psi/psi
    actual = directional_energy_statistics(tangent,local,4)
    def rayleigh(t):
        state = psi*np.exp(t*tangent)
        return np.vdot(state,h@state).real/np.vdot(state,state).real/4
    finite_difference = (rayleigh(1e-6)-rayleigh(-1e-6))/(2e-6)
    delta = psi*tangent
    explicit_derivative = 2*(np.vdot(delta,h@psi).real-
        np.vdot(psi,h@psi).real*np.vdot(delta,psi).real)/4
    # IID estimator uses Bessel correction; deterministic exhaustive quadrature
    # is the corresponding population covariance, with that factor removed.
    np.testing.assert_allclose(actual["energy_slope_per_site"]*3/4,
                               explicit_derivative,atol=2e-16,rtol=0)
    np.testing.assert_allclose(explicit_derivative,finite_difference,atol=1e-9,rtol=0)
    np.testing.assert_allclose(actual["magnitude_slope_per_site"]+actual["phase_slope_per_site"],
                               actual["energy_slope_per_site"],atol=2e-16,rtol=0)
    assert abs(actual["phase_slope_per_site"]) > .01
    shifted = directional_energy_statistics(tangent+3+7j,local+20-4j,4)
    np.testing.assert_allclose(shifted["energy_slope_per_site"],actual["energy_slope_per_site"],
                               atol=2e-15,rtol=0)
    assert not actual["used_for_update_selection"]


def test_statistics_continuation_preregisters_independent_streams_and_bounded_work():
    import json
    from pathlib import Path
    import pytest
    from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_train import _validate
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root/"projects/clebsch_tree_nqs/configs/remote/spin_reduced_vit_statistics_n36_20261011.json").read_text())
    _validate(settings)
    assert settings["samples_per_step"] == 256 and settings["steps"] == 20
    assert settings["audit_samples"] == 512 and settings["total_seconds"] == 3300
    assert settings["direction_check_steps"] == [1,5,10,15,20]
    assert "capacity_expansion" not in settings and not settings["full_A1"]
    with pytest.raises(ValueError,match="distinct"):
        _validate({**settings,"parent_audit_seed":settings["audit_seed"]})


def test_frozen_n36_trained_artifact_has_real_updates_untouched_audit_and_full_work_ledger():
    import hashlib
    import json
    from pathlib import Path
    import pytest
    from su2_nqs.frontier.artifacts import pytree_sha256

    root = Path(__file__).resolve().parents[1]
    directory = root / "projects/clebsch_tree_nqs/results/spin_reduced_vit_train_v1"
    if not (directory / "results.json").exists():
        pytest.skip("optional completed native N36 training canary absent")
    result = json.loads((directory / "results.json").read_text())
    registration = json.loads((directory / "registration.json").read_text())
    frozen = json.loads((directory / "frozen_endpoint.json").read_text())
    completion = json.loads((directory / "completion.json").read_text())
    settings = registration["config"]
    # Independent reviewer pins these two inputs as well: the original
    # runner's completion seal covers results.json, not the raw audit archive.
    for name, digest in {
        "registration.json": "498372998d011bfa528d7a199ecbeb1960bbb42a4416aa9d39f2864d109f15d9",
        "untouched_audit.npz": "f7a7e15996fb20bfecd142a79dce77e64094dcc2e52a1817930a45ce0fac7e14",
    }.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((directory / "results.json").read_bytes()).hexdigest() == completion["results_sha256"]
    assert hashlib.sha256((directory / "trained_checkpoint.pkl").read_bytes()).hexdigest() == frozen["checkpoint_sha256"]
    assert hashlib.sha256((root / settings["parent_checkpoint"]).read_bytes()).hexdigest() == registration["parent_checkpoint_sha256"]
    for relative, digest in registration["sources"].items():
        assert hashlib.sha256((directory / "source_snapshot" / relative).read_bytes()).hexdigest() == digest
    assert result["status"] == "completed"
    assert result["training_steps"] == result["accepted_steps"] == 20
    assert result["parameters_changed"]
    assert result["initial_params_sha256"] != result["terminal_params_sha256"]
    assert result["optimized_parameter_count"] == 4986
    assert result["all_parameters_trainable"]
    assert result["exact_SU2"] and not result["full_A1"]
    assert result["native_normalized_Born"]
    assert completion["exit_code"] == 0
    assert completion["audit_complete"]
    assert completion["compute_wall_seconds"] <= 3600
    ledger = result["ledger"]
    assert ledger["stage_H_calls"] == {"training": 1280, "checkpoint_replay": 68, "untouched_audit": 512}
    assert ledger["full_H_calls"] == 1860
    assert ledger["full_H_bond_calls"] == 1860 * 216
    assert ledger["training_draws"] == ledger["geometry_draws"] == 1280
    assert ledger["audit_draws"] == 512
    assert ledger["exact_capacity_escalations"] == []
    for row in result["history"]:
        assert row["sr"]["accepted"]
        assert row["sr"]["final_parameter_step_norm"] > 0
        assert row["sr"]["independent_geometry_overlap"]["sampled_fs_sine"] <= .05
        assert row["sr"]["relative_linear_residual"] <= 1e-7
    resumed = load_checkpoint(directory / "trained_checkpoint.pkl")
    assert resumed["optimizer_state"]["step"] == resumed["optimizer_state"]["accepted_steps"] == 20
    assert set(resumed["metadata"]["rng"]) == {"training", "geometry"}
    assert pytree_sha256(resumed["params"]) == result["terminal_params_sha256"] == frozen["params_sha256"]
    replay = result["checkpoint_replay"]
    assert replay["trained_checkpoint_step"] == 20
    assert replay["optimizer_state_identical"] and replay["optimizer_reconstructed_from_saved_settings"]
    assert replay["next_rng_identical"] and replay["next_training_samples_identical"]
    assert replay["next_geometry_samples_identical"]
    assert replay["amplitude_max_error"] == replay["H_ratio_max_error"] == replay["next_full_network_SR_update_max_error"] == 0
    assert replay["speculative_update_not_applied"]
    with np.load(directory / "untouched_audit.npz") as arrays:
        energies = arrays["local_energies"]
        labels = arrays["labels"]
        assert labels.shape == (512, 35)
        assert np.all(labels[:, -1] == 0)
        assert arrays["frozen_params_sha256"].item() == frozen["params_sha256"]
        assert arrays["audit_seed"].item() == settings["audit_seed"]
    mean = energies.mean()
    expected_variance = np.sum(abs(energies-mean)**2) / (511 * 36)
    np.testing.assert_allclose(result["audit"]["energy_per_site"], mean.real / 36, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result["audit"]["energy_stderr_per_site"], energies.real.std(ddof=1) / np.sqrt(512) / 36, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result["audit"]["variance_per_site"], expected_variance, atol=1e-15, rtol=0)
    assert result["audit"]["selection_or_training_used_audit"] is False
    assert result["audit"]["initial_or_training_energy_is_not_paired_audit"] is True
