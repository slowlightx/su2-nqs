"""Independent checks of unchanged-model oracle diagnosis, not VMC wins."""

import jax
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_diagnosis import (
    oracle_fit, phase_only_fidelity, reference_support_amplitude_oracle,
    root_channel_diagnosis,
)
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig, _candidate_features, init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.models.compact_tree_gru import split_tables
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout


def test_phase_oracle_separates_exact_sign_error_and_global_gauge():
    c = np.asarray([1., -1.]) / np.sqrt(2.)
    assert phase_only_fidelity(c, np.zeros(2)) < 1e-30
    assert abs(phase_only_fidelity(c, np.asarray([0., np.pi])) - 1) < 3e-15
    assert abs(phase_only_fidelity(1j*c, np.asarray([0., np.pi])) - 1) < 3e-15


def test_singlet_root_logits_have_only_two_physical_degrees_of_freedom():
    for side in (4, 6):
        config = ManyCellSchurViTConfig(rectangular_four_spin_layout(side, side))
        left, _, valid = split_tables(config.layout.tree)
        mask = valid[0, 0]
        q = left[0, 0, mask].astype(float)
        features = _candidate_features(config)[0, 0, mask]
        independent_basis = np.stack((np.ones_like(q), q, q*q), axis=1)
        fit = independent_basis @ np.linalg.lstsq(independent_basis, features, rcond=None)[0]
        np.testing.assert_allclose(fit, features, atol=3e-15, rtol=0)
        assert np.linalg.matrix_rank(features, tol=1e-12) == 3


def test_root_oracle_reproduces_a_known_quadratic_distribution():
    layout = rectangular_four_spin_layout(4, 4)
    config = ManyCellSchurViTConfig(layout)
    labels = np.asarray([s.labels for s in layout.tree.enumerate_states(0)])
    # Choose a reference by assigning each root sector the same prescribed
    # probability divided by its exact multiplicity, independent of model.
    from su2_nqs.models.tree_gru import tree_decisions
    q = labels[:, tree_decisions(layout.tree)[0].left_index]
    choices = np.unique(q)
    prescribed = np.exp(-.1*choices-.02*choices**2)
    prescribed /= prescribed.sum()
    p = np.zeros(len(labels))
    for value, probability in zip(choices, prescribed, strict=True):
        p[q == value] = probability / np.sum(q == value)
    result = root_channel_diagnosis(config, labels, np.sqrt(p))
    assert result["physical_probability_rank"] == 2
    assert abs(result["root_KL"]) < 2e-13
    assert result["root_total_variation"] < 3e-7


def test_phase_fit_freezes_all_born_parameters_including_shared_projection():
    layout = rectangular_four_spin_layout(2, 2)
    config = ManyCellSchurViTConfig(layout, embedding_dim=8, head_count=2, depth=1)
    labels = np.asarray([s.labels for s in layout.tree.enumerate_states(0)])
    parent = init_many_cell_schur_vit(jax.random.PRNGKey(14), config)
    old = np.asarray(many_cell_schur_log_amplitudes(parent, labels, config))
    trained, metrics = oracle_fit(parent, labels, np.asarray([1., -1.]), config,
        arm="phase_supervision", steps=5, learning_rate=.01, maximum_seconds=30)
    new = np.asarray(many_cell_schur_log_amplitudes(trained, labels, config))
    np.testing.assert_allclose(new.real, old.real, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(trained["decision_projection"], parent["decision_projection"])
    assert np.linalg.norm(new.imag-old.imag) > 1e-3
    assert metrics["completed_steps"] == 5


def test_reciprocal_oracle_does_not_invent_phases_on_reference_zeros():
    reference = np.asarray([1., -1., 0.]) / np.sqrt(2.)
    candidate = np.asarray([1., 1., 1.]) / np.sqrt(3.)
    state, info = reference_support_amplitude_oracle(reference, candidate,
        zero_tolerance=1e-12, maximum_ambiguous_mass=.001)
    np.testing.assert_allclose(state, reference, atol=3e-15, rtol=0)
    assert abs(info["removed_candidate_mass"] - 1/3) < 2e-15
    assert not info["within_registered_ambiguous_mass_budget"]
    assert not info["full_state_reference_phase_energy_reportable"]
    assert info["conditional_state_not_original_candidate"]
    assert state[2] == 0
    with pytest.raises(ValueError, match="no weight"):
        reference_support_amplitude_oracle(reference, np.asarray([0., 0., 1.]),
            zero_tolerance=1e-12, maximum_ambiguous_mass=.001)


def test_reciprocal_oracle_is_unambiguous_only_with_full_reference_support():
    reference = np.asarray([1., -2j]) / np.sqrt(5.)
    model = np.asarray([1j, 1.]) / np.sqrt(2.)
    state, info = reference_support_amplitude_oracle(reference, model,
        zero_tolerance=0., maximum_ambiguous_mass=.001)
    np.testing.assert_allclose(abs(state), abs(model), atol=3e-15, rtol=0)
    assert info["removed_candidate_mass"] == 0
    assert info["full_state_reference_phase_energy_reportable"]
    assert abs(abs(np.vdot(reference, state))**2 -
        np.sum(abs(reference)*abs(model))**2) < 3e-15
    for invalid in (-1., float("nan")):
        with pytest.raises(ValueError):
            reference_support_amplitude_oracle(reference, model,
                zero_tolerance=invalid, maximum_ambiguous_mass=.001)
