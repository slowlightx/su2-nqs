"""Independent small physical oracles for the initialization experiment."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_transfer_initialization import (
    COORDS6, RETAIN, _hamiltonian, _projector_vector, build_dataset,
    controller, evaluate_twelve, fit_controller, init_controller,
    physical_bonds, physical_pair_operator, schur_multiplets,
    run,
    initial_tree_parameters, make_refinement_step, refinement_measure,
    tree_coefficient_vector, twelve_singlet_basis, _twelve_reference,
)
from su2_nqs.operators.su2_block_transfer import (
    BlockIsometry, merge_projected_blocks, project_block,
)


def test_pair_action_matches_independent_pauli_and_singlet():
    sx = np.array([[0, 1], [1, 0]])/2
    sy = np.array([[0, 1j], [-1j, 0]])/2
    sz = np.diag([-.5, .5])
    expected = sum(np.kron(x, x) for x in (sx, sy, sz))
    np.testing.assert_allclose(physical_pair_operator(2, 0, 1).toarray(), expected, atol=1e-15)
    singlet = np.array([0, -1, 1, 0])/np.sqrt(2)
    np.testing.assert_allclose(expected @ singlet, -.75*singlet, atol=1e-15)


def test_actual_two_triangle_coarse_H_and_retained_projectors_match_physical_oracle():
    full_bonds = physical_bonds(COORDS6, .125, .6, .12)
    h = _hamiltonian(6, full_bonds).toarray()
    iso = BlockIsometry(3, schur_multiplets(3))
    left = project_block(iso, _hamiltonian(3, [(a, b, g) for a, b, g in full_bonds if b < 3]).toarray())
    right = project_block(iso, _hamiltonian(3, [(a-3, b-3, g) for a, b, g in full_bonds if a >= 3]).toarray())
    merged = merge_projected_blocks(left, right, [(a, b-3, g) for a, b, g in full_bonds if a < 3 <= b], retain=RETAIN)
    for q, u in schur_multiplets(6).items():
        values, vectors = np.linalg.eigh(u[:, :, -1].T @ h @ u[:, :, -1])
        exact = u[:, :, -1] @ vectors[:, :RETAIN[q]]
        observed = merged.block.isometry.sectors[q][:, :, -1]
        np.testing.assert_allclose(observed @ observed.conj().T, exact @ exact.T, atol=2e-13)
        np.testing.assert_allclose(merged.energies[q], values[:RETAIN[q]], atol=2e-13)
    assert merged.diagnostics["scalar_error"] < 2e-13


def test_physical_projector_does_not_depend_on_virtual_frame_rotation():
    rng = np.random.default_rng(26102359)
    for q, u in schur_multiplets(6).items():
        width = u.shape[1]
        frame = np.linalg.qr(rng.normal(size=(width, RETAIN[q])))[0]
        rotation = np.linalg.qr(rng.normal(size=(width, width)))[0]
        p = frame @ frame.T
        for k in range(q+1):
            physical = u[:, :, k] @ p @ u[:, :, k].T
            changed_u = u[:, :, k] @ rotation
            changed_p = rotation.T @ p @ rotation
            np.testing.assert_allclose(changed_u @ changed_p @ changed_u.T, physical, atol=2e-14)


def test_twelve_spin_full_H_variance_detects_compressed_space_leakage():
    data = build_dataset()
    result = evaluate_twelve(data["parent_frames"], .125, 1.)
    locked = evaluate_twelve(data["parent_frames"], .125, 1., q0_only=True)
    assert result["energy_per_site"] >= result["exact_reference_energy_per_site"]-1e-12
    assert result["variance_per_site"] > 1e-6
    assert result["compressed_H_variance_per_site"] < 1e-20
    assert sum(result["q_weights"].values()) > 1-1e-12
    assert sum(result["q_weights"][q] for q in (2, 4, 6)) > 1e-4
    assert result["energy_per_site"] < locked["energy_per_site"]-1e-3


def test_attention_projector_regression_gradient_matches_finite_difference():
    key = jax.random.PRNGKey(26102361)
    params = init_controller("attention", key, (12, 8), 131)
    x = jax.random.normal(key, (12, 8), dtype=jnp.float64)
    direction = jax.tree.map(lambda p: jnp.ones_like(p)/np.sqrt(p.size), params)
    def objective(p):
        return jnp.sum(controller(p, x, "attention")**2)
    derivative = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(jax.grad(objective)(params)),
                                                  jax.tree.leaves(direction), strict=True))
    eps = 1e-5
    plus = jax.tree.map(lambda p, d: p+eps*d, params, direction)
    minus = jax.tree.map(lambda p, d: p-eps*d, params, direction)
    np.testing.assert_allclose(derivative, (objective(plus)-objective(minus))/(2*eps), atol=3e-9, rtol=3e-7)


def test_fixed_budget_controller_restores_optimizer_and_rng(tmp_path):
    data = build_dataset()
    prediction, report = fit_controller(data, "linear", 26102341, 3, tmp_path)
    assert prediction.shape == data["targets"].shape
    assert report["next_step_replay_error"] == 0
    assert np.isfinite(report["heldout_projector_error"])
    assert len(_projector_vector(data["parent_frames"])) == 131


def test_normalized_geometry_is_retained_and_scramble_changes_prediction():
    data = build_dataset()
    train = np.asarray([i for i, row in enumerate(data["notes"]) if row["split"] == "train"])
    x = data["inputs"]
    normalized = (x-x[train].mean((0, 1)))/np.maximum(x[train].std((0, 1)), .1)
    assert np.std(normalized[0, :, 1:7]) > .1
    p = init_controller("attention", jax.random.PRNGKey(26102362), (12, 8), 131)
    difference = controller(p, jnp.asarray(normalized[0]), "attention")-controller(
        p, jnp.asarray(normalized[0]), "scrambled_attention")
    assert np.linalg.norm(difference) > 1e-6


def test_experiment_rejects_nonempty_output_and_invalid_budget(tmp_path):
    (tmp_path/"preserved.txt").write_text("existing evidence")
    with pytest.raises(FileExistsError):
        run(tmp_path)
    with pytest.raises(ValueError):
        run(tmp_path, 0)


def test_complete_reference_and_78_parameter_tree_replay_actual_physical_state():
    basis = twelve_singlet_basis()
    np.testing.assert_allclose(basis.T @ basis, np.eye(132), atol=3e-14)
    data = build_dataset()
    h, _, _, _, _ = _twelve_reference(.125, 1., 0.)
    hp = h @ basis; reduced = basis.T @ hp
    theta = initial_tree_parameters(data["parent_frames"], reduced)
    assert theta.shape == (78,)
    values, coeff = refinement_measure(theta, reduced, hp)
    actual = evaluate_twelve(data["parent_frames"], .125, 1.)
    np.testing.assert_allclose(values["energy_per_site"], actual["energy_per_site"], atol=3e-14)
    np.testing.assert_allclose(values["variance_per_site"], actual["variance_per_site"], atol=3e-14)
    psi = basis @ coeff
    np.testing.assert_allclose(np.vdot(psi, h @ psi)/12, actual["energy_per_site"], atol=3e-14)
    direction = np.random.default_rng(26102366).normal(size=78)
    jac = np.asarray(jax.jacfwd(tree_coefficient_vector)(jnp.asarray(theta)))
    np.testing.assert_allclose(np.vdot(basis @ (jac @ direction), basis @ (jac @ direction)),
                               np.vdot(jac @ direction, jac @ direction), atol=3e-14)


def test_same_capacity_natural_gradient_respects_physical_FS_and_unfreezes_q2():
    data = build_dataset(); basis = twelve_singlet_basis()
    h, _, _, _, _ = _twelve_reference(.125, 1., 0.)
    hp = h @ basis; reduced = basis.T @ hp
    theta = initial_tree_parameters(data["parent_frames"], reduced, q0_only=True)
    old = np.asarray(tree_coefficient_vector(theta))
    updated, info = make_refinement_step(reduced, hp)(theta)
    new = np.asarray(tree_coefficient_vector(updated))
    assert info["accepted"]
    assert 0 < info["actual_FS_sine"] <= .020000000001
    assert np.linalg.norm(old[25:]) < 1e-14
    assert np.linalg.norm(new[25:106]) > 1e-6
    assert info["energy_per_site"] < refinement_measure(theta, reduced, hp)[0]["energy_per_site"]
