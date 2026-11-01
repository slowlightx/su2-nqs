"""CPU-only authorization/product tests, never a new VMC/remote experiment."""
from copy import deepcopy
import json
from math import comb
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct6_one_extra_segment_20261031 as extra
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import paired_iid_moments
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.full_support_proposal import draw_full_support_sz0


@pytest.fixture
def products(tmp_path):
    rng = np.random.default_rng(410103)
    def callback(n, seed):
        random = np.random.default_rng(seed)
        return np.asarray([random.permutation([0]*18+[1]*18) for _ in range(n)], np.int32), {}
    rows, draw = draw_full_support_sz0(callback, 8192, n_sites=36, seed=3930294082)
    guide = (rng.random((8192, 2, 2))+.5)/comb(36, 18)
    q = .875*guide.mean((1, 2))+.125/comb(36, 18)
    wave = rng.normal(size=8192)+1j*rng.normal(size=8192)
    p = np.sqrt(q[:, None])*np.repeat(wave[:, None], 2, axis=1)
    h0 = -20*p[:, 0]+np.sqrt(q)*(.12*rng.normal(size=8192)+.05j*rng.normal(size=8192))
    h = np.stack((h0, h0-.2*p[:, 0]), axis=1)
    p[0] = 0; h[0] = np.sqrt(q[0])*np.asarray([.1, .1])  # True node, Hpsi is NOT zero.
    path = tmp_path/'products.npz'
    np.savez(path, rows=rows, psi=p, hpsi=h, proposal=q, guide_components=guide)
    m = paired_iid_moments(p, h, q, np.repeat(np.arange(128), 64), n_sites=36, seed=82)
    audit = dict(records=8192, moments=m)
    return path, audit, draw


def test_separate_one_extra_profile_does_not_open_old_campaign():
    plan = extra.make_plan(); extra.validate_plan(plan, executing=True)
    with pytest.raises(ValueError): extra.campaign.validate_plan(plan, executing=True)
    with pytest.raises(ValueError): extra.base.validate_plan(plan, executing=True)
    for change in (dict(steps=64), dict(maximum_extra_allocations=2), dict(wall_cap_seconds=86401),
                   dict(j2=0.), dict(automatic_next_allocation=True), dict(one_extra_index=2),
                   dict(fit_records=1024), dict(parent_directory='/tmp/other'), dict(segment_index=5)):
        with pytest.raises(ValueError): extra.validate_plan(plan | change)
    with pytest.raises(PermissionError): extra.validate_plan(plan | dict(execution_authorized=False), executing=True)
    config = extra.ROOT/'configs/remote/ct6_one_extra_segment_20261031/plan.json'
    assert json.loads(config.read_text()) == plan


def test_independent_node_H2_Q_and_both_errors_match_reference(products):
    path, audit, _ = products
    review = extra.independent_products(path, audit, extra.make_plan())
    assert review['maximum_relative_Q_error'] == 0.
    assert review['maximum_scaled_moment_replay_error'] < 2e-12
    assert review['recomputed']['exact_target_node_records'] == [1, 1]
    assert min(review['recomputed']['target_node_H2_contributions']) > 0
    np.testing.assert_array_equal(review['paired_maximum_SE'], np.maximum(
        audit['moments']['paired_iid_delta_SE'], review['paired_delete128_group_SE']))
    assert review['energy_upper_two_SE'] < 0 and review['new_draws'] == review['new_H_centers'] == 0


def test_tiny_density_requires_relative_not_absolute_replay(products, tmp_path):
    path, audit, _ = products
    with np.load(path) as f: arrays = dict(f)
    arrays['proposal'] = arrays['proposal']*1.001
    assert np.max(arrays['proposal']) < 2e-10  # An absolute2e-10 test would silently pass.
    bad = tmp_path/'badQ.npz'; np.savez(bad, **arrays)
    with pytest.raises(ArithmeticError, match='relative'): extra.independent_products(bad, audit, extra.make_plan())
    arrays['proposal'] /= 1.001
    arrays['guide_components'] = arrays['guide_components'].astype(complex)+1e-13j
    np.savez(bad, **arrays)
    with pytest.raises(ValueError, match='imaginary'): extra.independent_products(bad, audit, extra.make_plan())


def test_no_gain_corrupt_report_or_false_coverage_cannot_pass(products, tmp_path):
    path, audit, _ = products
    bad = deepcopy(audit); bad['moments']['H2_concentration']['maximum_sample_fraction'][0] = .9
    with pytest.raises(ArithmeticError): extra.independent_products(path, bad, extra.make_plan())
    with np.load(path) as f: arrays = dict(f)
    arrays['hpsi'][:, 1] = arrays['hpsi'][:, 0]
    audit = dict(records=8192, moments=paired_iid_moments(arrays['psi'], arrays['hpsi'], arrays['proposal'],
        np.repeat(np.arange(128), 64), n_sites=36, seed=83))
    badpath = tmp_path/'nogain.npz'; np.savez(badpath, **arrays)
    with pytest.raises(ValueError, match='energy gain'): extra.independent_products(badpath, audit, extra.make_plan())


def test_raw_variance_harm_and_concentrated_node_weight_are_vetoed(products, tmp_path):
    path, _, _ = products
    with np.load(path) as f: arrays = dict(f)
    base = arrays['hpsi'][:, 0].copy()
    p = arrays['psi'][:, 0]
    arrays['hpsi'][:, 1] = -20.2*p+5*(base+20*p)
    badpath = tmp_path/'variance.npz'
    def audit():
        return dict(records=8192, moments=paired_iid_moments(arrays['psi'], arrays['hpsi'],
            arrays['proposal'], np.repeat(np.arange(128), 64), n_sites=36, seed=84))
    np.savez(badpath, **arrays)
    with pytest.raises(ValueError, match='variance resolves harm'):
        extra.independent_products(badpath, audit(), extra.make_plan())
    arrays['hpsi'][:, 1] = base-.2*p
    arrays['hpsi'][0, :] = 1000*np.sqrt(arrays['proposal'][0])
    np.savez(badpath, **arrays)
    with pytest.raises(ValueError, match='coverage'):
        extra.independent_products(badpath, audit(), extra.make_plan())


@pytest.fixture
def parent(products, tmp_path, monkeypatch):
    path, audit, draw = products
    src = tmp_path/'segment04'/'run01'; snapshot = src/'parent_snapshot'
    snapshot.mkdir(parents=True); (src/'terminal_audit').mkdir()
    old = json.loads(extra.TEMPLATE.read_text())
    previous = old | dict(segment_index=3)
    params = dict(w=jnp.asarray([1., .2, -.1, 0.]))
    zero = jax.tree.map(jnp.zeros_like, params)
    rng = dict(generator=np.random.default_rng(71).bit_generator.state, next_step=1,
        consumed_role_seeds=[], excluded_seeds=list(old['seeds'].values()))
    progress = extra.base.initial_progress()
    for i in range(32):
        _, rng = extra.base.c6.role_seeds(rng, rng['next_step'])
        progress = extra.base.consume_attempt(progress, dict(attempt=i+1, accepted=True,
            work=dict(fit_H_centers=5120), predicted_decrease_per_site=0.))
    initial = params, (jnp.asarray(604), zero, zero), rng
    controls = extra.base.AdaptiveSRState(fs_radius=.00025, damping=.4, fit_records=5120)
    preg = dict(plan=previous, parent_global_step=572, native_origin=476, initial_native_cursor=1)
    atomic_json(snapshot/'registration.json', preg)
    extra.base.save_adaptive_state(snapshot/'checkpoint.pkl', initial, controls, progress,
        plan=previous, registration=preg)
    reg = dict(plan=old, parent_global_step=604, native_origin=476, initial_native_cursor=33,
        parent_seals={p.name:file_sha256(p) for p in snapshot.iterdir()}, sources={},
        original_parameter_sha256=pytree_sha256(params), original_optimizer_sha256=pytree_sha256(initial[1]),
        original_rng_sha256=canonical_sha256(rng), campaign_seed_history=list(old['seeds'].values()),
        inherited_control_values=dict(fs_radius=.00025, damping=.4, fit_records=5120))
    atomic_json(src/'registration.json', reg)
    extra.base.legacy.save_state(src/'parent.pkl', initial, 0, 604)
    extra.base.save_adaptive_state(src/'initial.pkl', initial, controls, extra.base.initial_progress(),
        plan=old, registration=reg)
    rng = deepcopy(rng); progress = extra.base.initial_progress()
    for i in range(2):
        _, rng = extra.base.c6.role_seeds(rng, rng['next_step'])
        progress = extra.base.consume_attempt(progress, dict(attempt=i+1, accepted=True,
            work=dict(fit_H_centers=5120), predicted_decrease_per_site=0.))
    progress['stop_reason'] = 'wall_gate_terminal_and_monitor_reserves_preserved'
    state = dict(w=params['w']+.001), (jnp.asarray(606), zero, zero), rng
    extra.base.save_adaptive_state(src/'checkpoint.pkl', state, controls, progress, plan=old, registration=reg)
    frozen = extra.base.freeze_endpoint(old, src, reg)
    audit.update(status='complete', endpoint_params_sha256=pytree_sha256(state[0]),
        seed=old['seeds']['audit'], statistics_seed=old['seeds']['audit_statistics'],
        no_endpoint_selection=True, comparison_parent_global_step=604, full_support_epsilon=.125)
    atomic_json(src/'terminal_audit/audit.json', audit)
    (src/'terminal_audit/products.npz').write_bytes(path.read_bytes())
    with np.load(path) as f: rows = f['rows']
    extra.base.c6.save_draw(src/'terminal_audit/draw', rows, draw)
    final = {k:frozen[k] for k in ('attempts', 'accepted_updates', 'absolute_completed_steps',
        'parameter_sha256', 'optimizer_sha256', 'rng_sha256', 'controller_sha256', 'checkpoint_sha256')}
    final.update(status='complete_fixed_prefix_independent_audit', terminal_audit=audit,
        independent_from_training_geometry_and_adaptation=True)
    atomic_json(src/'audit_result.json', final)
    monkeypatch.setattr(extra, 'PARENT_DIRECTORY', str(src))
    monkeypatch.setattr(extra, 'PARENT_REGISTRATION_SHA256', file_sha256(src/'registration.json'))
    return extra.make_plan(), src, state, controls


def test_actual_control_and_endpoint_lineage_then_exact_binding(parent, tmp_path):
    plan, src, state, controls = parent
    gate = extra.inspect_parent(plan)
    assert gate['parent_attempts'] == 2 and gate['parent_global_step'] == 606
    assert gate['prefix_classification'] == 'clean_resource_bounded_prefix'
    out = tmp_path/'extra'; assert extra.checked_gate(plan, out)['status'] == 'passed'
    run = out/'run01'; run.mkdir(); binding = extra.bind_parent(plan, run)
    inherited, _ = extra.base.legacy.load_state(run/'parent.pkl')
    assert pytree_sha256(inherited[:2]) == pytree_sha256(state[:2])
    assert canonical_sha256(inherited[2]) == canonical_sha256(state[2])
    current = extra.initial_controls(plan, binding)
    assert (current.fs_radius, current.damping, current.fit_records) == (.00025, .4, 5120)
    assert current.looks == current.good_streak == current.noise_streak == 0
    assert binding['no_successor'] and binding['controller_history'][-1]['archived_full_controller']['state']['fs_radius'] == .00025


def test_cpu_gate_refuses_moving_or_missing_terminal_without_GPU_or_draw(tmp_path, monkeypatch):
    monkeypatch.setattr(extra, 'PARENT_DIRECTORY', str(tmp_path/'missing_parent'))
    plan = extra.make_plan(); path = tmp_path/'plan.json'; atomic_json(path, plan)
    monkeypatch.setattr(extra.base, 'run', lambda *a, **k:pytest.fail('GPU consumer must not be called'))
    output = tmp_path/'extra'
    result = extra.run(path, output, 'prepare')
    assert result['status'] == 'refused_no_training' and result['new_draws'] == result['new_H_centers'] == 0
    receipt = (output/'parent_gate.json').read_bytes()
    assert extra.run(path, output, 'train')['status'] == 'refused_no_training'
    assert (output/'parent_gate.json').read_bytes() == receipt
    assert not (output/'run01').exists()


def test_changed_state_or_uncommitted_attempt_cannot_reuse_gate(parent, tmp_path):
    plan, src, _, _ = parent
    (src/'step3').mkdir()
    with pytest.raises(ValueError, match='uncommitted'): extra.inspect_parent(plan)
    (src/'step3').rmdir()
    output = tmp_path/'extra'; extra.checked_gate(plan, output)
    path = src/'terminal_audit/audit.json'; path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='seal changed'): extra.checked_gate(plan, output)


def test_unknown_or_scientific_prefix_stop_cannot_bypass_campaign(tmp_path):
    progress = dict(attempts=8, stop_reason='geometry_failure', monitor_history=[])
    with pytest.raises(ValueError): extra._prefix_safety(tmp_path, progress, dict(reason='geometry_failure'))
    progress['stop_reason'] = 'independent_monitor_coverage_or_reserved_resource_gate'
    progress['monitor_history'] = [dict(monitor=dict(delta_energy=.01, energy_se=.001,
        delta_variance=0., variance_se=.001, coverage_ok=True, geometry_ok=True),
        resources=dict(parameter_count=100, maximum_device_bytes=1024, remaining_updates=8,
            remaining_fit_H_records=0, remaining_training_seconds=0, seconds_per_fit_record=1., other_seconds_per_step=1.),
        reason='resolved_harm', can_train_next_block=False, next_controls=dict(fit_records=5120))]
    with pytest.raises(ValueError): extra._prefix_safety(tmp_path, progress, dict(reason=progress['stop_reason']))


def test_launcher_is_one_external_immutable_allocation_only():
    text = (extra.ROOT/'configs/remote/ct6_one_extra_segment_20261031/run.sbatch').read_text()
    assert '#SBATCH --time=24:00:00' in text and '#SBATCH --mem=128G' in text
    assert '#SBATCH --cpus-per-task=4' in text and '#SBATCH --no-requeue' in text
    assert '--nodelist' not in text and '--dependency' not in text
    assert 'rev-parse HEAD' in text and 'status --porcelain' in text
    assert 'PYTHONDONTWRITEBYTECODE=1' in text
    assert 'export XLA_CLIENT_MEM_FRACTION=.90' in text
    assert text.index('JAX_PLATFORMS=cpu') < text.index('srun --ntasks=1')
    assert 'sbatch ' not in text and 'scontrol requeue' not in text
