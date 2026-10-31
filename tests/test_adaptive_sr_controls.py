"""Independent monitoring controls; synthetic data are not physics results."""
from copy import deepcopy
from dataclasses import replace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.adaptive_sr_controls import (
    AdaptiveSRPolicy, AdaptiveSRState, SRMonitor, SRResourceEnvelope,
    advance_sr_controls, make_controlled_importance_sr, restore_sr_controls,
    select_controlled_finite_grid, sr_control_checkpoint,
)
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr


def policy():
    return AdaptiveSRPolicy(fit_levels=(8, 16, 32))


def panel(index=1, **changes):
    values = dict(panel_id=f'frozen-monitor-{index}', source_role='independent_training_monitor',
        sample_seed=100 + 2 * index, statistics_seed=101 + 2 * index,
        completed_step=4 * index, parent_sha256=str(index - 1) * 64,
        endpoint_sha256=str(index) * 64, delta_energy=-1.e-4, energy_se=1.e-5,
        delta_variance=-2.e-4, variance_se=1.e-5, predicted_decrease=1.e-4,
        geometry_ok=True, coverage_ok=True)
    return SRMonitor(**(values | changes))


def envelope(**changes):
    values = dict(parameter_count=10, maximum_device_bytes=10**9, reserved_device_bytes=1000,
        remaining_updates=4, remaining_fit_H_records=128, remaining_training_seconds=100.,
        seconds_per_fit_record=.1, other_seconds_per_step=1.)
    return SRResourceEnvelope(**(values | changes))


def observe(state, monitor, resources=None, p=None):
    return advance_sr_controls(p or policy(), state, monitor, resources or envelope(),
        reserved_seeds=(1, 2, 3, 4, 5))


def test_two_reliable_panels_relax_only_the_next_block():
    p = policy(); initial = AdaptiveSRState.initial(p)
    first, report = observe(initial, panel())
    assert first.fs_radius == initial.fs_radius and first.good_streak == 1
    second, report = observe(first, panel(2))
    assert second.fs_radius == pytest.approx(.00125)
    assert second.damping == pytest.approx(.04)
    assert second.fit_records == 8 and second.good_streak == 0
    assert report['applies_to_future_blocks_only'] and report['can_train_next_block']
    assert not report['terminal_audit_used'] and not report['convergence_claimed']
    assert initial.looks == 0  # immutable source protocol/state


@pytest.mark.parametrize('changes,reason',[
    (dict(delta_energy=5.e-5), 'resolved_harm'),
    (dict(delta_variance=5.e-5), 'resolved_harm'),
    (dict(delta_energy=0.), 'missed_resolved_prediction'),
    (dict(geometry_ok=False), 'geometry_failure'),
])
def test_harm_or_failed_geometry_shrinks_radius_and_changes_ridge(changes, reason):
    state, report = observe(AdaptiveSRState.initial(policy()), panel(**changes))
    assert state.fs_radius == .0005 and state.damping == .1
    assert report['reason'] == reason and state.fit_records == 8


def noisy(index):
    return panel(index, delta_energy=-1.e-6, predicted_decrease=1.e-6,
        directional_slope=-2.e-6, directional_slope_se=1.e-5)


def test_plateau_alone_does_not_claim_stationarity_or_increase_samples():
    s = AdaptiveSRState.initial(policy())
    for i in range(1, 4):
        s, report = observe(s, replace(noisy(i), directional_slope=None, directional_slope_se=None))
    assert s.fit_records == 8 and s.noise_streak == 0
    assert not report['convergence_claimed']
    assert report['reason'] == 'unresolved_or_moderate_change_hold_not_stationarity'


def test_noisy_independent_slope_and_energy_increase_records_only_after_patience():
    s = AdaptiveSRState.initial(policy())
    s, _ = observe(s, noisy(1)); assert s.fit_records == 8
    s, report = observe(s, noisy(2))
    assert s.fit_records == 16 and s.fs_radius == .001 and s.damping == .05
    assert report['reason'] == 'increase_next_block_training_records'


@pytest.mark.parametrize('limits',[
    dict(maximum_device_bytes=2000),  # 8 fits; 16 does not.
    dict(remaining_fit_H_records=32),
    dict(remaining_training_seconds=7.5),
])
def test_growth_cannot_spend_reserved_memory_H_work_or_terminal_wall(limits):
    s, _ = observe(AdaptiveSRState.initial(policy()), noisy(1))
    s, report = observe(s, noisy(2), envelope(**limits))
    assert s.fit_records == 8
    assert report['reason'] == 'sample_growth_blocked_by_reserved_memory_work_or_wall'


def test_actual_CT6_6144_row_growth_fails_70GiB_storage_check():
    resource = envelope(parameter_count=1535560, maximum_device_bytes=70 * 1024**3,
        reserved_device_bytes=3 * 1024**3, remaining_fit_H_records=100000,
        remaining_training_seconds=100000.)
    assert resource.permits(5120)
    assert not resource.permits(6144)


def test_incomplete_coverage_never_prompts_more_of_the_same_or_grants_training():
    state, report = observe(AdaptiveSRState.initial(policy()), panel(coverage_ok=False))
    assert state.fit_records == 8 and state.fs_radius == .001
    assert not report['can_train_next_block']
    assert report['reason'] == 'coverage_not_verified_hold_and_diagnose'


@pytest.mark.parametrize('role', ['fit', 'geometry', 'terminal_audit', 'independent_audit'])
def test_terminal_audit_or_fitting_cannot_drive_schedule(role):
    with pytest.raises(ValueError, match='not monitor panels'):
        observe(AdaptiveSRState.initial(policy()), panel(source_role=role))


@pytest.mark.parametrize('damage', ['sample_seed', 'statistics_seed', 'same_seed', 'energy_se',
                                  'slope', 'hash', 'nan', 'completed'])
def test_monitor_provenance_and_numerics_fail_closed(damage):
    values = dict(sample_seed=dict(sample_seed=2), statistics_seed=dict(statistics_seed=5),
        same_seed=dict(statistics_seed=102), energy_se=dict(energy_se=-1.),
        slope=dict(directional_slope=1., directional_slope_se=None),
        hash=dict(endpoint_sha256='unsealed'), nan=dict(delta_energy=float('nan')),
        completed=dict(completed_step=0))
    with pytest.raises(ValueError):
        observe(AdaptiveSRState.initial(policy()), panel(**values[damage]))


def test_repeated_panel_seed_block_and_endpoint_search_are_rejected():
    state, _ = observe(AdaptiveSRState.initial(policy()), panel())
    for m in (panel(), panel(2, sample_seed=102), panel(2, parent_sha256='0' * 64),
              panel(2, completed_step=4)):
        with pytest.raises(ValueError): observe(state, m)


def test_budgeted_monitor_count_is_not_an_indefinite_adaptive_loop():
    p = replace(policy(), maximum_monitor_looks=1)
    s, _ = observe(AdaptiveSRState.initial(p), panel(), p=p)
    with pytest.raises(ValueError, match='look budget'):
        observe(s, panel(2), p=p)


def test_control_extremes_are_clipped_and_exact_zero_is_not_noisy_plateau():
    p = replace(policy(), maximum_fs_radius=.001, minimum_damping=.05,
        minimum_fs_radius=.0005, maximum_damping=.1)
    s = AdaptiveSRState.initial(p)
    for i in (1, 2):
        s, _ = observe(s, panel(i, delta_energy=1.e-3), p=p)
    assert s.fs_radius == .0005 and s.damping == .1
    for i in (3, 4, 5, 6):
        s, _ = observe(s, panel(i), p=p)
    assert s.fs_radius <= p.maximum_fs_radius and s.damping >= p.minimum_damping
    s, report = observe(s, panel(7, delta_energy=0., predicted_decrease=0., energy_se=0.,
        directional_slope=0., directional_slope_se=0.), p=p)
    assert s.fit_records == 8 and not report['convergence_claimed']


def test_state_policy_restore_preserves_next_decision_and_rejects_tampering():
    p = policy(); s, _ = observe(AdaptiveSRState.initial(p), panel())
    payload = sr_control_checkpoint(p, s)
    restored = restore_sr_controls(p, payload)
    assert restored == s
    assert observe(s, panel(2)) == observe(restored, panel(2))
    bad = deepcopy(payload); bad['state']['damping'] = .0001
    with pytest.raises(ValueError): restore_sr_controls(p, bad)
    with pytest.raises(ValueError): restore_sr_controls(replace(p, maximum_fs_radius=.004), payload)


def test_real_SR_binding_matches_old_update_and_restores_physical_optimizer_rng(tmp_path):
    p = policy(); controls = AdaptiveSRState.initial(p)
    params = jnp.array([.15, -.11]); rows = jnp.arange(8)[:, None]
    features = jnp.stack((jnp.linspace(-1., 1., 8), jnp.cos(jnp.arange(8))), axis=1)
    amplitude = lambda theta, row: jnp.exp(features[row[0]] @ theta)
    hpsi = jnp.linspace(-2., .5, 8); density = jnp.ones(8) / 8
    moments = _manual_adam_init(params)
    adaptive = make_controlled_importance_sr(amplitude, params, p, parameter_chunk_size=1)
    old = make_importance_amplitude_sr(amplitude, params, damping=.05,
        sr_tangent_fs_radius=.001, parameter_chunk_size=1)
    first = adaptive(params, moments, rows, hpsi, density, .03, controls=controls)
    expected = old(params, moments, rows, hpsi, density, .03)
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    controls, _ = observe(controls, panel())
    controls, _ = observe(controls, panel(2))
    path = tmp_path / 'adaptive.pkl'
    rng = np.random.default_rng(2801).bit_generator.state
    save_checkpoint(path, first[0], optimizer_state=first[1], metadata=dict(
        adaptive_sr=sr_control_checkpoint(p, controls), next_training_rng=rng))
    saved = load_checkpoint(path)
    restored = restore_sr_controls(p, saved['metadata']['adaptive_sr'])
    continued = adaptive(saved['params'], saved['optimizer_state'], rows, hpsi, density, .03, controls=restored)
    direct = make_importance_amplitude_sr(amplitude, params, damping=.04,
        sr_tangent_fs_radius=.00125, parameter_chunk_size=1)(first[0], first[1], rows, hpsi, density, .03)
    for a, b in zip(jax.tree.leaves(continued), jax.tree.leaves(direct), strict=True):
        np.testing.assert_array_equal(a, b)
    r1 = np.random.default_rng(); r1.bit_generator.state = rng
    r2 = np.random.default_rng(); r2.bit_generator.state = saved['metadata']['next_training_rng']
    np.testing.assert_array_equal(r1.integers(0, 2**32, 16), r2.integers(0, 2**32, 16))
    with pytest.raises(ValueError, match='record count'):
        adaptive(params, moments, rows[:4], hpsi[:4], density[:4], .03, controls=controls)


def test_controlled_realized_FS_guard_not_just_parameter_or_tangent_norm():
    p = policy(); state = AdaptiveSRState.initial(p)
    vector = np.tile([1., -1., 1., -1.], 4)
    angles = np.array([.002, .0005, 0.])
    # Each half is an exact two-component quadrature. Candidate zero exceeds
    # the physical radius; candidate one passes on selection AND guard.
    phi = np.cos(angles)[None, :] + vector[:, None] * np.sin(angles)[None, :]
    chosen, report = select_controlled_finite_grid(p, state, phi, np.ones(16),
        np.array([1., .25, 0.]), split=8, groups=4)
    assert chosen == 1 and report['accepted']
    assert report['guard']['nominal_upper'] <= state.fs_radius
    assert not report['guard_used_to_select_smaller_fraction']
    # A disagreement on guard is a veto, not another search/energy look.
    broken = phi.copy(); broken[8:, 1] = np.cos(.004) + vector[8:] * np.sin(.004)
    chosen, report = select_controlled_finite_grid(p, state, broken, np.ones(16),
        np.array([1., .25, 0.]), split=8, groups=4)
    assert chosen == 2 and not report['accepted']
