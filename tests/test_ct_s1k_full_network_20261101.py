"""Independent tiny derivative/cost/lineage oracles, not a CT6 energy result."""
from copy import deepcopy
import json

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct_s1k_full_network_20261101 as w
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import save_checkpoint
from su2_nqs.vmc.complex_importance_amplitude_sr import make_complex_importance_amplitude_sr


@pytest.fixture
def plan():
    value = json.loads(w.CONFIG.read_text())
    value.update(execution_authorized=False, exact_invariants_passed=False)
    return value


def test_separate_task_preserves_full_statistics_sector_budget_and_authority(plan):
    w.validate_plan(plan)
    with pytest.raises(PermissionError): w.validate_plan(plan, executing=True)
    authorized = plan | dict(execution_authorized=True, exact_invariants_passed=True)
    w.validate_plan(authorized, executing=True)
    with pytest.raises(ValueError): w.old.validate_plan(authorized)
    for key, value in [('fit_records', 16), ('FS_records', 64), ('bilateral_records', 64),
        ('terminal_records', 128), ('target_q', 0), ('fixed_attempts', 9), ('maximum_updates', 16),
        ('warm_repetitions', 0), ('preflight_records', 2), ('no_successor', False),
        ('S0X_after_independent_S1_outcome_only', False), ('cold_cost_multiplied_by_records', True)]:
        with pytest.raises(ValueError): w.validate_plan(plan | {key:value})
    changed = deepcopy(plan); changed['resources']['requested_node'] = 'fixed-node'
    with pytest.raises(ValueError): w.validate_plan(changed)
    changed = deepcopy(plan); changed['resources']['allocation_wall_seconds'] += 1
    with pytest.raises(ValueError): w.validate_plan(changed)
    with pytest.raises(TypeError): w.validate_plan(plan | dict(target_q=2.))
    with pytest.raises(TypeError): w.validate_plan(plan | dict(fixed_attempts=8.))


def test_one_cold_plus_repeated_actual128_warm_does_not_extrapolate_compile(plan):
    times = iter([0., 1000., 1000., 1002., 1002., 1005.])
    calls = []
    _, measurement = w.measure_cold_and_warm(lambda:calls.append(128), clock=lambda:next(times))
    assert calls == [128, 128, 128]
    assert measurement['cold_compile_plus_execution_seconds'] == 1000.
    assert measurement['repeated_warm_seconds'] == [2., 3.]
    assert measurement['warm_seconds_per_record'] == 3/128
    profile = dict(measurements={name:measurement for name in
        ('source_Q', 'CT_H', 'CT_amp', 'physical_amp', 'H_JVP', 'complex_SR')},
        process_replay_reserve_seconds=1800.)
    cost = w.phase_costs(plan, profile, draw_seconds=0.)
    assert cost['fit'] == pytest.approx(1.5*(3*3+3))
    assert cost['FS'] == pytest.approx(1.5*(3+6*3))
    assert cost['independent'] == pytest.approx(1.5*(3+3*3+3))
    assert cost['terminal'] == pytest.approx(1.5*4*(3+3*3))
    assert cost['process_replay'] == 1800.
    assert sum(cost.values()) < 3000.  # Not 128 times a cold1000s execution.


def test_persistent_complex_H_JVP_keeps_all_native_prepare_derivatives():
    traces = []
    def prepare(p):
        traces.append('trace')
        x, y, z, r = [p[k][0] for k in ('local', 'six', 'top', 'root')]
        return jnp.array([x*y+1j*z*r, z+x*r+1j*y*z])
    matrix = jnp.array([[.2, -.7j], [.7j, -.4]])
    def action(v, rows):
        return jnp.stack((v[rows[:, 0]], (matrix@v)[rows[:, 0]]), axis=1)
    p = {key:jnp.array([value]) for key, value in zip(('local', 'six', 'top', 'root'), (.8, .6, .4, .7))}
    d = {key:jnp.array([value]) for key, value in zip(('local', 'six', 'top', 'root'), (.1, -.2, .3, .11))}
    rows = jnp.asarray([[0], [1]]*64); source = jnp.ones((128, 2), dtype=complex)*(.3+.4j)
    reference = jnp.zeros((128, 2), dtype=complex)
    compiled = w.make_persistent_H_jvp(prepare, action, 3.2)
    values, tangent = jax.block_until_ready(compiled(p, d, rows, source, reference))
    trace_count = len(traces)
    again = jax.tree.map(lambda v:v+.02, p)
    _ = jax.block_until_ready(compiled(again, d, rows, source, reference))
    assert len(traces) == trace_count  # Same executable, not a nested newly created jit.
    def direct(parameters): return source+3.2*(action(prepare(parameters), rows)-reference)
    plus = jax.tree.map(lambda a, b:a+1e-6*b, p, d)
    minus = jax.tree.map(lambda a, b:a-1e-6*b, p, d)
    finite = (np.asarray(direct(plus))-np.asarray(direct(minus)))/2e-6
    np.testing.assert_allclose(tangent, finite, atol=6e-10, rtol=2e-9)
    assert np.max(abs(np.asarray(tangent).imag)) > .1
    for family in p:
        direction = jax.tree.map(jnp.zeros_like, p); direction[family] = jnp.ones(1)
        _, derivative = compiled(p, direction, rows, source, reference)
        assert np.linalg.norm(derivative) > 0  # local/six/top/root, not root-only.


def test_unknown_normalizer_mass_is_not_mislabeled_as_absolute_norm():
    actual = w.clean(dict(sampled_parent_norm=jnp.array(7.), nested=dict(norm_estimates=np.array([1., 3.]))))
    assert actual == dict(sampled_relative_mass=7., nested=dict(relative_mass_estimates=[1., 3.]))


def test_exact_node_replay_uses_nonzero_frozen_scale_and_rejects_nonfinite():
    assert w.fixed_scale_replay_error(np.zeros(2, complex), np.zeros(2, complex), .01) == 0.
    assert w.fixed_scale_replay_error(np.array([1e-16j, 0.]), np.zeros(2), .01) == pytest.approx(1e-14)
    with pytest.raises(ArithmeticError): w.fixed_scale_replay_error(np.zeros(2), np.zeros(2), 0.)
    with pytest.raises(ArithmeticError): w.fixed_scale_replay_error(np.array([np.nan, 0]), np.zeros(2), .01)


def test_no_queries_without_exact_authority_hash_and_allocated_gpu(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'load_bound', lambda *_:pytest.fail('must refuse before input loading'))
    out = tmp_path/'never-created'
    with pytest.raises(PermissionError): w.run(plan, out, approved_plan_sha256=canonical_sha256(plan))
    plan.update(execution_authorized=True, exact_invariants_passed=True)
    with pytest.raises(PermissionError): w.run(plan, out, approved_plan_sha256='wrong')
    monkeypatch.setattr(w.jax, 'default_backend', lambda:'cpu')
    with pytest.raises(RuntimeError): w.run(plan, out, approved_plan_sha256=canonical_sha256(plan))
    assert not out.exists()


def test_bound_zero_update_parent_inherits_exact_old_reserved_rng_and_moments(plan, tmp_path):
    native = dict(w=jnp.array([.7, .2, -.1, .9]))
    moments = (jnp.array(23), jax.tree.map(lambda x:2*x, native), jax.tree.map(lambda x:3*x, native))
    rng = dict(generator=np.random.default_rng(1201).bit_generator.state, used_seeds=[17, 31],
               excluded_seeds=[71, 99], roles=[dict(role='old_fit128_reserved', seed=17), dict(role='old_terminal512_reserved', seed=31)])
    for key in ('fixed_source_carrier', 'fixed_CT_reference'):
        path = tmp_path/(key+'.pkl'); save_checkpoint(path, native)
        plan[key] = str(path); plan[key+'_sha256'] = file_sha256(path)
    replay = tmp_path/'replay.npz'; np.savez(replay, rows=np.zeros((2, 36), np.int32))
    plan['runtime_replay'] = str(replay); plan['runtime_replay_sha256'] = file_sha256(replay)
    previous = dict(task=w.old.TASK, fixed_source_carrier_sha256=plan['fixed_source_carrier_sha256'],
                    fixed_CT_reference_sha256=plan['fixed_CT_reference_sha256'])
    meta = dict(plan=previous, plan_sha256=canonical_sha256(previous), fixed_scale=plan['fixed_scale'])
    path = tmp_path/'checkpoint.pkl'; w.old.save_state(path, (native, moments, rng), completed_updates=0, metadata=meta)
    plan['parent_checkpoint'] = str(path); plan['parent_checkpoint_sha256'] = file_sha256(path)
    plan.update(parent_parameter_sha256=pytree_sha256(native), parent_optimizer_sha256=pytree_sha256(moments),
                parent_rng_sha256=canonical_sha256(rng), real_native_parameters=4)
    result = dict(status='zero_update_budget_gate_before128_fit', committed_updates=0, work=dict(IID_rows=0, full_W_complex_J_rows=0))
    for key, contents in (('parent_result', result), ('parent_registration', dict(plan_sha256=meta['plan_sha256']))):
        path = tmp_path/(key+'.json'); atomic_json(path, contents)
        plan[key] = str(path); plan[key+'_sha256'] = file_sha256(path)
    loaded, *_ = w.load_bound(plan)
    assert pytree_sha256(loaded[:2]) == pytree_sha256((native, moments))
    assert canonical_sha256(loaded[2]) == canonical_sha256(rng)
    seed_a, next_a = w.old.next_seed(rng, 'new_fit128')
    seed_b, next_b = w.old.next_seed(loaded[2], 'new_fit128')
    assert seed_a == seed_b and next_a == next_b and seed_a not in {17, 31, 71, 99}
    result['committed_updates'] = 1; atomic_json(tmp_path/'parent_result.json', result)
    plan['parent_result_sha256'] = file_sha256(tmp_path/'parent_result.json')
    with pytest.raises(ValueError, match='zero-update'): w.load_bound(plan)


def test_incomplete_train_cannot_release_or_start_replay(plan, tmp_path, monkeypatch):
    plan.update(execution_authorized=True, exact_invariants_passed=True)
    monkeypatch.setattr(w.jax, 'default_backend', lambda:'gpu')
    atomic_json(tmp_path/'result.json', dict(status='bounded_partial_unreleased_no_automatic_retry'))
    monkeypatch.setattr(w, 'load_bound', lambda *_:pytest.fail('incomplete result must not query physics'))
    assert w.replay(plan, tmp_path)['status'] == 'refused_incomplete_run_no_release'
    assert not (tmp_path/'release_certificate.json').exists()


def test_single_allocation_wrapper_and_freshprocess_replay_are_explicit():
    text = (w.CONFIG.parent/'run.sbatch').read_text()
    assert '#SBATCH --time=12:00:00' in text and '#SBATCH --mem=128G' in text
    assert '--nodelist' not in text and '#SBATCH --no-requeue' in text
    assert '--phase train' in text and '--phase replay' in text
    assert text.count('/usr/local/slurm/current/bin/srun') == 2
    assert 'rev-parse HEAD' in text and 'status --porcelain' in text
    assert 'sbatch ' not in text and 'S0X' in text


def test_complete_tiny_driver_transaction_keeps_128_512_and_release_replay(plan, tmp_path, monkeypatch):
    """An independent two-level driver oracle, explicitly NOT CT6 physics."""
    native = dict(backbone=dict(local=jnp.array([.9, .7]), six=jnp.array([.8]), top=jnp.array([.6])),
                  root=(jnp.array([.5]),))
    zero = jax.tree.map(jnp.zeros_like, native)
    moments = (jnp.array(23), zero, zero)
    rng = dict(generator=np.random.default_rng(5501).bit_generator.state, used_seeds=[17], excluded_seeds=[99],
               roles=[dict(role='old_reserved', seed=17)])
    state = (native, moments, rng)
    matrix = jnp.array([[-.4, -.6], [-.6, .2]], dtype=complex)
    def vector(p):
        return jnp.array([p['backbone']['local'][0]*p['root'][0][0]+.2j*p['backbone']['local'][1],
                          p['backbone']['six'][0]+.1j*p['backbone']['top'][0]])
    def scalar(p, row): return vector(p)[row[0]]
    class Tiny:
        def __init__(self, *_):
            self.counts = dict(full_W_complex_J_rows=0, native_CT_H_centers=0,
                native_CT_amplitude_centers=0, source_full_H_Q_centers=0, complete_H_JVP_centers=0)
            self.base_seeds = []; self.forbidden_base_seeds = set()
            self.raw = dict(prepare=lambda p:p)
            self.amplitude = scalar
            self.update = make_complex_importance_amplitude_sr(scalar, native, damping=.05,
                score_chunk=1, parameter_chunk_size=8192, sr_tangent_fs_radius=.01)
        def check_resources(self): pass
        def source(self, rows):
            v = np.asarray(vector(native)); h = np.asarray(matrix)@v
            target = np.column_stack((v[rows[:, 0]], h[rows[:, 0]]))
            return np.stack((target, target), axis=1), np.full(len(rows), .5), np.full(len(rows), .5)
        def H(self, p, rows, source, ref=None):
            v = np.asarray(vector(p)); h = np.asarray(matrix)@v
            actual = np.column_stack((v[rows[:, 0]], h[rows[:, 0]]))
            return actual, np.zeros_like(actual) if ref is None else ref
        def amplitudes(self, p, rows, source, ref=None):
            return np.asarray(vector(p))[rows[:, 0]], np.zeros(len(rows))
        def physical_amplitudes(self, p, rows):
            return np.asarray(vector(p))[rows[:, 0]]
        def H_JVP(self, p, d, rows, source, ref):
            def function(values):
                v = vector(values)
                return jnp.stack((v[rows[:, 0]], (matrix@v)[rows[:, 0]]), axis=1)
            a, b = jax.jvp(function, (p,), (d,))
            return np.asarray(a), np.asarray(b)
        def draw(self, count, seed):
            rows = np.zeros((count, 36), np.int32); rows[:, 0] = np.arange(count)%2
            return rows, dict(base_batch_seeds=[], synthetic_driver_oracle=True, seed=seed)
    rt = Tiny(); rows = np.zeros((2, 36), np.int32); rows[1, 0] = 1
    source, base, q = rt.source(rows); physical, _ = rt.H(native, rows, source[:, 1])
    golden = dict(rows=rows, source_products=source[:, 1], physical_products=physical,
                  normalized_base_density=base, relative_density=q)
    def bound(_): return state, {}, {}, native, golden
    monkeypatch.setattr(w, 'load_bound', bound)
    monkeypatch.setattr(w, 'Runtime', Tiny)
    monkeypatch.setattr(w.jax, 'default_backend', lambda:'gpu')
    class Device: device_kind = 'A100 synthetic driver oracle'
    monkeypatch.setattr(w.jax, 'devices', lambda:[Device()])
    monkeypatch.setattr(w, 'physical_checks', lambda *a:dict(synthetic_driver_oracle_only=True))
    def profile(*_):
        return dict(measurements={k:dict(warm_seconds_per_record=1e-8) for k in
            ('source_Q', 'CT_H', 'CT_amp', 'physical_amp', 'H_JVP', 'complex_SR')}, process_replay_reserve_seconds=1800.)
    monkeypatch.setattr(w, 'preflight', profile)
    plan.update(execution_authorized=True, exact_invariants_passed=True)
    output = tmp_path/'synthetic_only'
    result = w.run(plan, output, approved_plan_sha256=canonical_sha256(plan))
    assert result['status'] == 'complete_unreleased_pending_fresh_process_replay'
    assert result['accepted_updates'] >= 1 and result['terminal_records_completed'] == 512
    assert not result['released'] and not (output/'release_certificate.json').exists()
    with np.load(output/'attempt01/fit128.npz') as f: assert len(f['rows']) == 128
    with np.load(output/'attempt01/FS128.npz') as f: assert len(f['rows']) == 128
    with np.load(output/'attempt01/bilateral128.npz') as f: assert len(f['rows']) == 128
    with np.load(output/'terminal512.npz') as f: assert len(f['rows']) == 512
    restored, _ = w.old.restore_state(output/'checkpoint.pkl')
    assert int(restored[1][0]) == 23+result['accepted_updates']
    assert set(rng['used_seeds']).issubset(restored[2]['used_seeds'])
    certificate = w.replay(plan, output)
    assert certificate['status'] == 'released_complete_fixed_terminal_and_freshprocess_replay'
    assert certificate['checks']['first128_parameter_max_absolute_error'] < 2e-11
    assert certificate['checks']['final_physical_H_relative_error'] < 2e-11
    with pytest.raises(FileExistsError): w.replay(plan, output)
    monkeypatch.setattr(w.old, 'coverage', lambda *_:dict(passed=False, synthetic_driver_oracle=True))
    parent_only = tmp_path/'synthetic_parent_only'
    result = w.run(plan, parent_only, approved_plan_sha256=canonical_sha256(plan))
    assert result['status'] == 'complete_unchanged_source_parent_only_pending_fresh_process_replay'
    assert result['accepted_updates'] == 0 and not result['new_optimized_state_created']
    certificate = w.replay(plan, parent_only)
    assert certificate['status'] == 'source_parent_only_no_new_optimized_state'
    assert not certificate['new_optimized_state_released']
    assert not certificate['terminal_coverage_passed'] and certificate['terminal_precision_not_certified_if_failed']
