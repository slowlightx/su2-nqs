"""Bounded same-D free/linear-dictionary/attention-dictionary development.

Reconstruct a saved joint bank, then train all parameters toward that bank from
one COMMON perturbed physical state. This is a gauge-dependent reconstruction
test, NOT VMC, an independent energy audit, or a variational advantage claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_dictionary_attention import (
    TriangleDictionaryAttentionConfig, dictionary_attention_bank, dictionary_parameter_count,
    dictionary_layout, init_dictionary_attention_from_bank,
)
from su2_nqs.models.triangular_clebsch_feature_lut import (
    prepare_feature_lut_bank, make_feature_lut_projected_amplitude,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, make_mixed_clebsch_amplitude, make_mixed_projected_amplitude,
)
from su2_nqs.vmc.adam import _manual_adam_init, _manual_adam_update
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from .triangle_scale_batch_cost import relative_vector_error

ROOT = Path(__file__).resolve().parents[3]


def run(config_path: Path) -> dict:
    """Fixed-budget reconstruction, exact physical queries and checkpoint replay."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text()); output = ROOT/spec['output']
    if output.exists():
        raise FileExistsError('development outputs are immutable')
    output.mkdir(parents=True)
    start = perf_counter()
    result = dict(status='running', arms={}, energy_optimization=False, independent_energy_audit=False,
        variational_win_claim=False, architecture_is_static_geometry_hypernetwork=True,
        training_objective='relative full-bank Frobenius reconstruction; gauge dependent',
        config_sha256=file_sha256(config_path), backend=jax.default_backend())
    def save():
        result['elapsed_seconds'] = perf_counter()-start
        atomic_json(output/'results.json', result)
    def timeout(*_):
        result.update(status='censored', reason='registered local time cap'); save(); raise SystemExit(2)
    signal.signal(signal.SIGALRM, timeout); signal.alarm(spec['seconds_cap'])
    source = ROOT/spec['source']['checkpoint']
    if file_sha256(source) != spec['source']['checkpoint_sha256']:
        raise ValueError('frozen joint parent checkpoint changed')
    saved = load_checkpoint(source); target = saved['params']
    if pytree_sha256(target) != spec['source']['params_sha256']:
        raise ValueError('frozen parent parameters changed')
    parent_registration = json.loads((source.parent/'registration.json').read_text())
    model = MixedClebschConfig(**spec['model'])
    if parent_registration['config']['model'] != spec['model'] or parent_registration['config']['arm'] != 'ternary_joint':
        raise ValueError('same-D joint physical bank required')
    sources = transitive_local_source_manifest([Path(__file__), ROOT/'su2_nqs/vmc/local_energy.py'], root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    atomic_json(output/'registration.json', dict(config=spec, sources=sources))
    with np.load(source.parent/'terminal_audit.npz') as data:
        rows = jnp.asarray(data['bits'][:spec['physical_probe_rows']], jnp.int32)
        held_rows = jnp.asarray(data['bits'][-spec['physical_probe_rows']:], jnp.int32)
    images = np.asarray(triangular_hnf_gamma_a1_group(((model.length, 0), (0, model.length))), np.int32)
    old = make_mixed_projected_amplitude(model, images, chunk=4)
    cached = make_feature_lut_projected_amplitude(model, images, chunk=4)
    bonds = tuple(triangular_j1j2_bonds(model.length, model.length, j2=spec['j2_over_j1'], periodic=True))
    action = make_amplitude_heisenberg_action(cached, bonds, n_sites=model.length**2)
    reference_amplitudes = jax.jit(jax.vmap(old, in_axes=(None, 0)))(target, rows)
    reference_H = jax.block_until_ready(action(prepare_feature_lut_bank(target, model), rows[:1]))
    key = jax.random.PRNGKey(spec['seed'])
    leaves, definition = jax.tree.flatten(target); keys = jax.random.split(key, len(leaves)+1)
    noisy = jax.tree.unflatten(definition, [w+spec['common_bank_perturbation']*jnp.linalg.norm(w)
        /np.sqrt(w.size)*jax.random.normal(k, w.shape, dtype=jnp.float64)
        for w, k in zip(leaves, keys[1:], strict=True)])
    target_norm2 = sum(jnp.sum(w*w) for w in jax.tree.leaves(target))
    result.update(model=spec['model'], source_checkpoint_sha256=file_sha256(source),
        source_params_sha256=pytree_sha256(target), common_perturbed_bank_sha256=pytree_sha256(noisy),
        full_A1_images=len(images), Hamiltonian_bonds=len(bonds),
        notes=['No ordinary Spatial-ViT checkpoint/model is modified.',
            'The same-R full-node replay is not a parameter-compression result.',
            'Different generated banks at fixed D remain in the same free-tree family.',
            'A1 image cross terms are retained; attention weights are not entanglement spectra.'])
    for arm in ('free', 'linear', 'attention'):
        config = TriangleDictionaryAttentionConfig(model=model, architecture='linear' if arm == 'linear' else 'attention',
            **spec['generator'])
        generate = (lambda p: p) if arm == 'free' else lambda p: dictionary_attention_bank(p, config)
        if arm == 'free':
            params, initialization = noisy, dict(exact_same_bank_expected=True, free_tensor_control=True)
            exact_params = target
        else:
            params, initialization = init_dictionary_attention_from_bank(keys[0], noisy, config)
            exact_params, exact_note = init_dictionary_attention_from_bank(keys[0], target, config)
            initialization['target_replay_initialization'] = exact_note
        generated = generate(exact_params)
        bank_error = relative_vector_error(np.concatenate([np.asarray(w).ravel() for w in jax.tree.leaves(generated)]),
            np.concatenate([np.asarray(w).ravel() for w in jax.tree.leaves(target)]))
        prepare = jax.jit(lambda p: prepare_feature_lut_bank(generate(p), model))
        evaluate = jax.jit(lambda p, x: jax.vmap(cached, in_axes=(None, 0))(prepare(p), x))
        actual_amp = evaluate(exact_params, rows)
        fresh = jax.block_until_ready(prepare(exact_params)); actual_H = jax.block_until_ready(action(fresh, rows[:1]))
        tick = perf_counter(); fresh = jax.block_until_ready(prepare(exact_params))
        jax.block_until_ready(action(fresh, rows[:1])); h_seconds = perf_counter()-tick
        errors = dict(bank=bank_error, full_A1=relative_vector_error(actual_amp, reference_amplitudes),
            full_H=max(relative_vector_error(a, b) for a, b in zip(actual_H, reference_H, strict=True)))
        if not np.isfinite(list(errors.values())).all() or max(errors.values()) > spec['tolerance']:
            raise ArithmeticError('same-bank physical replay failed: '+arm)
        def loss(p):
            return sum(jnp.sum((a-b)**2) for a, b in zip(jax.tree.leaves(generate(p)),
                jax.tree.leaves(target), strict=True))/target_norm2
        loss = jax.jit(loss); value_and_gradient = jax.jit(jax.value_and_grad(loss))
        initial_bank = generate(params)
        initial_error = relative_vector_error(np.concatenate([np.asarray(w).ravel() for w in jax.tree.leaves(initial_bank)]),
            np.concatenate([np.asarray(w).ravel() for w in jax.tree.leaves(noisy)]))
        if initial_error > spec['tolerance']:
            raise ArithmeticError('reconstruction arms do not share the initial physical bank')
        optimizer = _manual_adam_init(params); history = []; initial_params = params
        for step in range(spec['reconstruction_steps']):
            value, gradient = value_and_gradient(params)
            proposal, next_optimizer = _manual_adam_update(params, gradient, optimizer, spec['learning_rate'])
            accepted = False
            for halve in range(8):
                alpha = .5**halve
                candidate = jax.tree.map(lambda a, b: a+alpha*(b-a), params, proposal)
                candidate_loss = float(loss(candidate))
                if np.isfinite(candidate_loss) and candidate_loss < float(value):
                    params, optimizer, accepted = candidate, next_optimizer, True
                    break
            history.append(dict(step=step+1, loss_before=float(value), loss_after=float(loss(params)), accepted=accepted))
        checkpoint = output/arm/'checkpoint.pkl'
        save_checkpoint(checkpoint, params, optimizer_state=optimizer, metadata=dict(
            rng=np.asarray(keys[0]), source_params_sha256=pytree_sha256(target),
            objective='bank_reconstruction_not_energy', history=history, model_config=asdict(config)))
        restored = load_checkpoint(checkpoint)
        expected = _manual_adam_update(params, value_and_gradient(params)[1], optimizer, spec['learning_rate'])
        replay = _manual_adam_update(restored['params'], value_and_gradient(restored['params'])[1],
            restored['optimizer_state'], spec['learning_rate'])
        checkpoint_exact = (pytree_sha256(expected) == pytree_sha256(replay)
            and np.array_equal(restored['metadata']['rng'], np.asarray(keys[0])))
        target_held = jax.jit(jax.vmap(old, in_axes=(None, 0)))(target, held_rows)
        initial_held, final_held = evaluate(initial_params, held_rows), evaluate(params, held_rows)
        record = dict(initialization=initialization, parameter_count=dictionary_parameter_count(params),
            exact_target_replay_errors=errors, initial_common_bank_error=initial_error,
            H_warm_seconds_including_fresh_generator_and_LUT=h_seconds, reconstruction_history=history,
            initial_loss=history[0]['loss_before'], terminal_loss=history[-1]['loss_after'],
            held_physical_amplitude_relative_error_before=relative_vector_error(initial_held, target_held),
            held_physical_amplitude_relative_error_after=relative_vector_error(final_held, target_held),
            checkpoint_optimizer_parameter_RNG_next_update_replay_exact=checkpoint_exact,
            checkpoint_sha256=file_sha256(checkpoint), active_original_parameter_leaves=len(jax.tree.leaves(params)))
        if arm != 'free':
            record['tensor_types'] = [{k: v for k, v in item.items() if k != 'charge_keys'} for item in dictionary_layout(config)]
        result['arms'][arm] = record; save(); print(arm, json.dumps(record), flush=True)
    signal.alarm(0)
    result.update(status='completed', checkpoint_replay_all=all(
        a['checkpoint_optimizer_parameter_RNG_next_update_replay_exact'] for a in result['arms'].values()))
    save()
    return result


def diagnose_completed(config_path: Path) -> dict:
    """Reuse saved endpoints to prove joint training and physical AD, no updates."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text()); output = ROOT/spec['output']
    original = json.loads((output/'results.json').read_text())
    if original['status'] != 'completed' or (output/'additional_validation.json').exists():
        raise ValueError('one immutable supplement requires a completed reconstruction run')
    source = ROOT/spec['source']['checkpoint']; target = load_checkpoint(source)['params']
    model = MixedClebschConfig(**spec['model']); key = jax.random.PRNGKey(spec['seed'])
    leaves, definition = jax.tree.flatten(target); keys = jax.random.split(key, len(leaves)+1)
    noisy = jax.tree.unflatten(definition, [w+spec['common_bank_perturbation']*jnp.linalg.norm(w)
        /np.sqrt(w.size)*jax.random.normal(k, w.shape, dtype=jnp.float64)
        for w, k in zip(leaves, keys[1:], strict=True)])
    if pytree_sha256(noisy) != original['common_perturbed_bank_sha256']:
        raise ValueError('the common initial state cannot be replayed')
    with np.load(source.parent/'terminal_audit.npz') as data:
        row = jnp.asarray(data['bits'][-1], jnp.int32)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    old = make_mixed_projected_amplitude(model, images, chunk=4)
    new = make_feature_lut_projected_amplitude(model, images, chunk=4)
    denominator = sum(jnp.sum(w*w) for w in jax.tree.leaves(target))
    result = dict(status='running', new_training_steps=0, arms={}, variational_win_claim=False,
        source_results_sha256=file_sha256(output/'results.json'),
        sources=transitive_local_source_manifest([Path(__file__)], root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    start = perf_counter()
    for arm in ('free', 'linear', 'attention'):
        saved = load_checkpoint(output/arm/'checkpoint.pkl'); params = saved['params']
        cfg = TriangleDictionaryAttentionConfig(model=model, architecture='linear' if arm == 'linear' else 'attention', **spec['generator'])
        generate = (lambda p: p) if arm == 'free' else lambda p: dictionary_attention_bank(p, cfg)
        initial = noisy if arm == 'free' else init_dictionary_attention_from_bank(keys[0], noisy, cfg)[0]
        loss = lambda p: sum(jnp.sum((a-b)**2) for a, b in zip(jax.tree.leaves(generate(p)),
            jax.tree.leaves(target), strict=True))/denominator
        gradient = jax.jit(jax.grad(loss))(params)
        norm = lambda tree: float(np.sqrt(sum(np.sum(abs(np.asarray(w))**2) for w in jax.tree.leaves(tree))))
        component_gradients = {k: norm(v) for k, v in gradient.items()}
        component_changes = {k: norm(jax.tree.map(lambda a, b: a-b, params[k], initial[k])) for k in params}
        moment_norms = {k: norm(v) for k, v in saved['optimizer_state'][1].items()}
        scalar0 = lambda p: old(generate(p), row)
        scalar1 = lambda p: new(prepare_feature_lut_bank(generate(p), model), row)
        tangent = jax.tree.map(lambda w: .0001*jnp.sin(jnp.arange(w.size).reshape(w.shape)+.3), params)
        d0 = jax.jit(lambda p, v: jax.jvp(scalar0, (p,), (v,))[1])
        d1 = jax.jit(lambda p, v: jax.jvp(scalar1, (p,), (v,))[1])
        expected, actual = d0(params, tangent), d1(params, tangent)
        tick = perf_counter(); jax.block_until_ready(d1(params, tangent)); jvp_seconds = perf_counter()-tick
        g0, g1 = jax.jit(jax.grad(scalar0))(params), jax.jit(jax.grad(scalar1))(params)
        error = relative_vector_error(np.concatenate([np.asarray(w).ravel() for w in jax.tree.leaves(g1)]),
            np.concatenate([np.asarray(w).ravel() for w in jax.tree.leaves(g0)]))
        jvp_error = relative_vector_error(actual, expected)
        if max(error, jvp_error) > spec['tolerance']:
            raise ArithmeticError('saved-state original-parameter AD replay failed')
        result['arms'][arm] = dict(terminal_loss_gradient_norm_by_component=component_gradients,
            actual_parameter_change_norm_by_component=component_changes,
            saved_Adam_first_moment_norm_by_component=moment_norms,
            full_physical_parameter_JVP_relative_error=jvp_error,
            full_physical_parameter_gradient_relative_error=error,
            warm_JVP_seconds_including_generator_and_LUT=jvp_seconds)
        result['elapsed_seconds'] = perf_counter()-start
        atomic_json(output/'additional_validation.json', result)
    result['status'] = 'passed'; atomic_json(output/'additional_validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--diagnose', action='store_true')
    args = parser.parse_args()
    print(json.dumps(diagnose_completed(args.config) if args.diagnose else run(args.config), indent=2))
