"""Bounded CPU VMC for a static geometry dictionary and same-D free controls.

The earlier ``triangle_dictionary_local`` reconstruction experiment is kept
independent and immutable. Here every optimization batch is newly sampled,
the complete coherent A1 and Heisenberg action are used, and a fixed endpoint
is audited with separate chains and RNG. No physical spin enters the generator.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_dictionary_attention import (
    TriangleDictionaryAttentionConfig, dictionary_attention_bank,
    dictionary_parameter_count, init_dictionary_attention_from_bank,
)
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.bounded_physical_pilot import energy_summary
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
from su2_nqs.vmc.parameter_block_sr import solve_parameter_block_sr
from su2_nqs.vmc.physical_exchange import (
    initialize_spin_exchange_chains, run_exchange_sweeps,
)
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

ROOT = Path(__file__).resolve().parents[3]


def make_prepared_sr_update(
    prepare: Callable, log_prepared: Callable, template_params: object, *,
    damping: float = .1, step_size: float = .03,
    fisher_trust_radius: float = .005, maximum_overlap_distance: float = .005,
    minimum_ess_fraction: float = .8, backtracking_steps: int = 8,
    score_chunk_size: int = 4, parameter_chunk_size: int | None = None,
) -> Callable:
    """Ridge SR with one differentiable static-bank preparation per score batch.

    ``prepare`` is differentiated by a shared VJP, so every original parameter
    (including the dictionary and encoder) participates. Per-configuration
    scores include this pullback. ``local_E`` is detached, as required for the
    center-score VMC covariance. S=2 Re Cov(O,O), g=2 Re Cov(O,E).
    The nonlinear trust guard checks the first passing overlap/ESS halving;
    candidate energies and independent audits never select the update.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('prepared SR requires x64')
    for value in (damping, step_size, fisher_trust_radius, maximum_overlap_distance):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('ridge and trust scales must be positive and finite')
    if not 0 < minimum_ess_fraction <= 1:
        raise ValueError('minimum_ess_fraction must lie in (0,1]')
    for value in (backtracking_steps, score_chunk_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('positive integer chunk and backtracking counts required')
    if parameter_chunk_size is not None and (isinstance(parameter_chunk_size, bool)
            or not isinstance(parameter_chunk_size, int) or parameter_chunk_size < 1):
        raise ValueError('parameter_chunk_size must be a positive integer or None')
    leaves = jax.tree.leaves(template_params)
    if not leaves or any(np.asarray(w).dtype != np.dtype('float64') or
                         not np.isfinite(w).all() for w in leaves):
        raise ValueError('parameters must have finite real float64 leaves')
    _, unravel = ravel_pytree(template_params)
    log_batch = jax.vmap(log_prepared, in_axes=(None, 0))

    @jax.jit
    def update(params, samples, local_E):
        count = len(samples)
        if count < 2 or local_E.shape != (count,):
            raise ValueError('need >=2 samples and matching local energies')
        flat, _ = ravel_pytree(params)
        prepared, pullback = jax.vjp(lambda v: prepare(unravel(v)), flat)

        def score(row):
            def components(bank):
                z = log_prepared(bank, row)
                return jnp.stack((z.real, z.imag))
            bank_scores = jax.jacrev(components)(prepared)
            return jax.vmap(lambda cotangent: pullback(cotangent)[0])(bank_scores)

        scores = jax.lax.map(score, samples, batch_size=score_chunk_size)
        e = jax.lax.stop_gradient(local_E-local_E.mean())
        b = jnp.sqrt(2./count)*jnp.concatenate((e.real, e.imag))
        if parameter_chunk_size is None:
            centered = scores-scores.mean(0, keepdims=True)
            a = jnp.sqrt(2./count)*jnp.concatenate((centered[:, 0], centered[:, 1]))
            gram = a @ a.T
            coefficients = jax.scipy.linalg.solve(
                gram+damping*jnp.eye(2*count), b, assume_a='pos')
            direction = a.T @ coefficients
            gradient = a.T @ b
            residual = a.T @ (a @ direction)+damping*direction-gradient
            gradient_norm = jnp.linalg.norm(gradient)
            relative_residual = jnp.linalg.norm(residual)/jnp.maximum(gradient_norm, 1e-30)
            raw_fisher_norm = jnp.linalg.norm(a @ direction)
        else:
            direction, gradient, blocked = solve_parameter_block_sr(scores, b,
                damping=damping, score_scale=(2./count)**.5, parameter_chunk_size=parameter_chunk_size)
            gradient_norm = blocked['gradient_norm']
            relative_residual = blocked['relative_linear_residual']
            raw_fisher_norm = blocked['raw_fisher_norm']
        scale = step_size*jnp.minimum(1., fisher_trust_radius/
                                     jnp.maximum(step_size*raw_fisher_norm, 1e-30))
        old_logs = log_batch(prepared, samples)
        initial = (flat, jnp.asarray(False), jnp.asarray(0.), jnp.asarray(1.),
                   jnp.asarray(0.), jnp.asarray(0))

        def trial(k, carry):
            selected, accepted, fs, ess, used, attempts = carry

            def evaluate(_):
                amount = scale*.5**k
                candidate = flat-amount*direction
                ratios = log_batch(prepare(unravel(candidate)), samples)-old_logs
                diagnostics = sampled_overlap_diagnostics(ratios)
                passed = (jnp.all(jnp.isfinite(scores)) & jnp.all(jnp.isfinite(local_E)) &
                          jnp.all(jnp.isfinite(ratios)) & jnp.all(jnp.isfinite(candidate)) &
                          (relative_residual <= 1e-7) &
                          (diagnostics['sampled_fs_sine'] <= maximum_overlap_distance) &
                          (diagnostics['reweighting_ess_fraction'] >= minimum_ess_fraction))
                return (jnp.where(passed, candidate, selected), passed,
                        jnp.where(passed, diagnostics['sampled_fs_sine'], 0.),
                        jnp.where(passed, diagnostics['reweighting_ess_fraction'], 1.),
                        jnp.where(passed, amount, 0.), attempts+1)
            return jax.lax.cond(accepted, lambda _: carry, evaluate, operand=None)

        chosen, accepted, fs, ess, used, attempts = jax.lax.fori_loop(
            0, backtracking_steps, trial, initial)
        diagnostics = dict(gradient_norm=gradient_norm,
            relative_linear_residual=relative_residual,
            raw_direction_norm=jnp.linalg.norm(direction), raw_fisher_norm=raw_fisher_norm,
            fisher_norm=used*raw_fisher_norm, applied_scale=used,
            parameter_step_norm=jnp.linalg.norm(chosen-flat), sampled_fs_sine=fs,
            reweighting_ess_fraction=ess, accepted=accepted,
            backtracking_evaluations=attempts, score_real_entries=scores.size,
            sample_matrix_dimension=2*count,
            finite_nonzero_sample_centers=jnp.all(jnp.isfinite(old_logs)))
        # Component norms are audit information, not optimization controls.
        diagnostics['gradient_norm_by_component'] = {
            name: jnp.sqrt(sum(jnp.sum(v*v) for v in jax.tree.leaves(value)))
            for name, value in unravel(gradient).items()
        } if isinstance(params, dict) else {}
        if parameter_chunk_size is not None:
            diagnostics.update({key: blocked[key] for key in (
                'parameter_blocks', 'maximum_normalized_score_block_entries', 'retained_raw_score_entries')})
        return unravel(chosen), diagnostics
    return update


def make_record_sampler(
    amplitude: Callable, *, records: int, burn_proposals: int,
    proposals_between: int, exchange_edges: tuple[tuple[int, int], ...],
    bond_proposal_probability: float = .75, lazy_probability: float = .05,
) -> Callable:
    """Native symmetric bond/global MH, returning [time,chain,site] records.

    A proposal is one attempt per chain, not an implicit n-site sweep. Current
    amplitudes are refreshed after each parameter update; only within this
    immutable-state call are they cached. Exact zero centers are not floored.
    """
    if records < 1 or burn_proposals < 1 or proposals_between < 1:
        raise ValueError('positive record and proposal counts required')

    @jax.jit
    def sample(prepared, key, states):
        def advance(rng, x, attempts, amplitudes=None):
            return run_exchange_sweeps(prepared, rng, x, amplitude_fn=amplitude,
                particle_kind='spin', sweeps=attempts, amplitudes=amplitudes,
                exchange_edges=exchange_edges, bond_proposal_probability=bond_proposal_probability,
                lazy_probability=lazy_probability)
        states, amplitudes, key, burn = advance(key, states, burn_proposals)

        def body(carry, _):
            x, values, rng = carry
            x, values, rng, info = advance(rng, x, proposals_between, values)
            return (x, values, rng), (x, info)
        (states, amplitudes, key), (rows, info) = jax.lax.scan(
            body, (states, amplitudes, key), xs=None, length=records)
        return states, key, rows, dict(
            burn=burn, records=jax.tree.map(jnp.mean, info),
            terminal_zero_centers=jnp.sum(abs(amplitudes) == 0))
    return sample


def summarize_born_audit(local_energy: np.ndarray, bits: np.ndarray, *, n_sites: int) -> dict:
    """Unclipped unweighted Born estimates, retaining independent-chain blocks.

    Chains have separate random initial configurations but finite burn-in.
    Chain-block standard errors do not certify equilibrium or unseen tails.
    """
    energy = np.asarray(local_energy, np.complex128)
    rows = np.asarray(bits)
    if energy.ndim != 2 or min(energy.shape) < 2 or rows.shape[:2] != energy.shape:
        raise ValueError('audit requires [time>=2,chains>=2] energies and matching bits')
    if not np.isfinite(energy).all():
        raise FloatingPointError('nonfinite full-H energy in audit')
    summary = energy_summary(energy, n_sites)
    mean = energy.mean()
    chain_means = energy.mean(0)
    chain_se = summary['energy_chain_se']
    residual = abs(energy-mean)**2
    chain_variance = residual.mean(0)/n_sites
    centered = energy.real-energy.real.mean(0, keepdims=True)
    denominator = np.sum(centered[:-1]**2+centered[1:]**2)
    lag_one = float(2*np.sum(centered[:-1]*centered[1:])/denominator) if denominator else 0.
    within = np.var(energy.real, axis=0, ddof=1).mean()
    between = energy.shape[0]*np.var(chain_means.real, ddof=1)
    rhat = np.sqrt(((energy.shape[0]-1)*within/energy.shape[0]+
                    between/energy.shape[0])/within) if within else None
    return dict(**summary,
        chain_block_standard_error_per_site=float(chain_se),
        approximate_95_interval_per_site=[float(mean.real/n_sites-1.96*chain_se),
                                         float(mean.real/n_sites+1.96*chain_se)],
        variance_chain_block_standard_error=float(chain_variance.std(ddof=1)/np.sqrt(energy.shape[1])),
        chain_energy_means_per_site=(chain_means.real/n_sites).tolist(),
        local_energy_range_per_site=[float(energy.real.min()/n_sites), float(energy.real.max()/n_sites)],
        records_per_chain=energy.shape[0], independent_chains=energy.shape[1],
        sample_records=int(energy.size), unique_states=int(len(np.unique(rows.reshape(-1, rows.shape[-1]), axis=0))),
        pooled_lag_one_energy_correlation=lag_one,
        unsplit_energy_Rhat=None if rhat is None else float(rhat),
        clipping=False, importance_reweighting=False, finite_burn_equilibrium_certified=False,
        unseen_tail_coverage_certified=False,
        variance_is_Born_local_energy_estimate=True)


def _plain(tree):
    return jax.tree.map(lambda x: np.asarray(x).item() if np.asarray(x).ndim == 0
                        else np.asarray(x).tolist(), tree)


def _tree_relative_error(actual, expected) -> float:
    differences = sum(np.sum(abs(np.asarray(a)-np.asarray(b))**2)
                      for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True))
    norm = sum(np.sum(abs(np.asarray(b))**2) for b in jax.tree.leaves(expected))
    return float(np.sqrt(differences/max(norm, 1e-300)))


def _physical_runtime(spec: dict, model: MixedClebschConfig) -> tuple:
    """Share the exact query construction between training and process replay."""
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=0., periodic=True))
    amplitude = make_feature_lut_projected_amplitude(model, images, chunk=spec['projection_chunk'])
    action = make_cached_projected_heisenberg_action(model, images, bonds,
        projection_chunk=spec['projection_chunk'])

    @jax.jit
    def evaluate(prepared, rows):
        def one(row):
            psi, hp = action(prepared, row[None])
            return jnp.stack((psi[0], hp[0]))
        values = jax.lax.map(one, rows, batch_size=spec['action_chunk'])
        return values[:, 0], values[:, 1]
    return images, bonds, amplitude, evaluate


def _transition_factory(prepare, sampler, evaluate, update) -> Callable:
    """One fresh sampler/full-H/SR attempt, with no mutable hidden state."""
    prepared_fn = jax.jit(prepare)

    def transition(params, states, key):
        bank = prepared_fn(params)
        states, key, rows, sampling = jax.block_until_ready(sampler(bank, key, states))
        psi, hp = jax.block_until_ready(evaluate(bank, rows.reshape(-1, 36)))
        if np.any(np.asarray(psi) == 0) or not np.isfinite(np.asarray(hp/psi)).all():
            raise FloatingPointError('zero or nonfinite sampled full-H center')
        energies = jax.lax.stop_gradient(hp/psi)
        changed, diagnostics = jax.block_until_ready(update(params, rows.reshape(-1, 36), energies))
        return changed, states, key, rows, energies, sampling, diagnostics
    return transition


def replay_completed(config_path: Path) -> dict:
    """Reload frozen checkpoints in a new process and replay the next full step.

    Invoke this CLI only after the training process exits. No optimization
    update is committed, no new endpoint is selected, and audits are untouched.
    """
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text())
    output = ROOT/spec['output']
    original = json.loads((output/'results.json').read_text())
    registration = json.loads((output/'registration.json').read_text())
    destination = output/'process_replay.json'
    if original['status'] != 'completed' or destination.exists():
        raise ValueError('replay requires one completed run and a new immutable supplement')
    if registration['config_sha256'] != file_sha256(config_path) or any(
            file_sha256(ROOT/path) != digest for path, digest in registration['sources'].items()):
        raise ValueError('registered config or source changed before process replay')
    model = MixedClebschConfig(**spec['model'])
    _, bonds, amplitude, evaluate = _physical_runtime(spec, model)
    training = spec['training']
    sampler = make_record_sampler(amplitude, records=training['records_per_chain'],
        burn_proposals=training['refresh_proposals'], proposals_between=training['proposals_between'],
        exchange_edges=tuple((a, b) for a, b, J in bonds if J != 0), **spec['sampler'])
    result = dict(status='running', updates_committed=0, audits_reused_as_training=False,
        original_results_sha256=file_sha256(output/'results.json'), arms={})
    start = perf_counter()
    for arm in ('free', 'linear', 'attention'):
        saved = load_checkpoint(output/arm/'checkpoint.pkl')
        cfg = TriangleDictionaryAttentionConfig(model=model,
            architecture='linear' if arm == 'linear' else 'attention', **spec['generator'])
        generate = (lambda p: p) if arm == 'free' else lambda p, cfg=cfg: dictionary_attention_bank(p, cfg)
        prepare = lambda p, generate=generate: prepare_feature_lut_bank(generate(p), model)
        logs = lambda bank, bits: jnp.log(amplitude(bank, bits).astype(jnp.complex128))
        update = make_prepared_sr_update(prepare, logs, saved['params'], **spec['sr'])
        transition = _transition_factory(prepare, sampler, evaluate, update)
        continued = transition(saved['params'], jnp.asarray(saved['metadata']['sampler_states']),
            jnp.asarray(saved['metadata']['rng']))
        actual = pytree_sha256(continued)
        expected = original['arms'][arm]['next_transition_replay_sha256']
        result['arms'][arm] = dict(full_next_transition_bitwise_exact=actual == expected,
            next_transition_sha256=actual,
            checkpoint_optimizer_step=int(saved['optimizer_state']['completed_steps']))
        result['elapsed_seconds'] = perf_counter()-start
        atomic_json(destination, result)
        if actual != expected:
            result['status'] = 'failed'
            atomic_json(destination, result)
            raise ArithmeticError('new-process sampler/full-H/SR continuation differs: '+arm)
    result['status'] = 'passed'
    atomic_json(destination, result)
    return result


def run(config_path: Path) -> dict:
    """Run a preregistered local fixed-budget VMC and immutable endpoint audits."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text())
    if spec['j2_over_j1'] != 0 or spec['model']['length'] != 6:
        raise ValueError('this bounded local validation is N36, J2=0')
    output = ROOT/spec['output']
    if output.exists():
        raise FileExistsError('registered local VMC output is immutable')
    source = ROOT/spec['source']['checkpoint']
    if file_sha256(source) != spec['source']['checkpoint_sha256']:
        raise ValueError('frozen parent checkpoint hash mismatch')
    saved = load_checkpoint(source)
    parent = saved['params']
    if pytree_sha256(parent) != spec['source']['params_sha256']:
        raise ValueError('frozen parent parameter hash mismatch')
    parent_registration = json.loads((source.parent/'registration.json').read_text())
    if ('registration_sha256' in spec['source'] and
            parent_registration['sha256'] != spec['source']['registration_sha256']):
        raise ValueError('frozen parent registration seal mismatch')
    if ('audit_sha256' in spec['source'] and
            file_sha256(source.parent/'terminal_audit.json') != spec['source']['audit_sha256']):
        raise ValueError('frozen parent audit hash mismatch')
    if 'audit_sha256' in spec['source']:
        parent_audit = json.loads((source.parent/'terminal_audit.json').read_text())
        if (parent_audit['params_sha256'] != spec['source']['params_sha256'] or
                parent_audit['registration_sha256'] != parent_registration['sha256']):
            raise ValueError('frozen source audit does not identify the same parent state')
    if (parent_registration['config']['model'] != spec['model'] or
            parent_registration['config']['j2_over_j1'] != 0 or
            parent_registration['config']['arm'] != 'ternary_joint'):
        raise ValueError('need one same-D joint J2=0 parent for all arms')
    model = MixedClebschConfig(**spec['model'])
    images, bonds, amplitude, evaluate = _physical_runtime(spec, model)
    edges = tuple((a, b) for a, b, J in bonds if J != 0)
    train_spec, audit_spec = spec['training'], spec['audit']
    sampler_options = dict(exchange_edges=edges, **spec['sampler'])
    sampler = make_record_sampler(amplitude, records=train_spec['records_per_chain'],
        burn_proposals=train_spec['refresh_proposals'],
        proposals_between=train_spec['proposals_between'], **sampler_options)
    audit_sampler = make_record_sampler(amplitude, records=audit_spec['records_per_chain'],
        burn_proposals=audit_spec['burn_proposals'],
        proposals_between=audit_spec['proposals_between'], **sampler_options)
    initial_sampler = make_record_sampler(amplitude, records=1,
        burn_proposals=train_spec['initial_burn_proposals'], proposals_between=1, **sampler_options)
    output.mkdir(parents=True)
    sources = transitive_local_source_manifest([Path(__file__), ROOT/'su2_nqs/vmc/local_energy.py'],
        root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    for relative in sources:
        destination = output/'source_snapshots'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
    registration = dict(config=spec, config_sha256=file_sha256(config_path), sources=sources,
        runtime=numerical_runtime_identity(), source_registration_sha256=file_sha256(source.parent/'registration.json'))
    registration['sha256'] = canonical_sha256(registration)
    atomic_json(output/'registration.json', registration)
    start = perf_counter()
    result = dict(status='running', energy_optimization=True, arms={},
        independent_endpoint_audits=True, variational_win_claim=False,
        architecture_is_static_geometry_hypernetwork=True,
        optimizer='stateless sample-space ridge SR; fresh attempt counter and trust diagnostics',
        parent_optimizer_reused=False, parent_sampler_reused=False, parent_rng_reused=False,
        common_initial_bank_sha256=pytree_sha256(parent), model=asdict(model),
        full_A1_images=len(images), Hamiltonian_bonds=len(bonds),
        all_parameter_count_is_not_tiny=True, backend=jax.default_backend(),
        registration_sha256=registration['sha256'],
        notes=['Full-rank dictionary is across nodes; this is not parameter compression.',
               'Finite-sample FS and ESS guards do not certify a global trust bound.',
               'Audit means and chain-block uncertainty require equilibration; finite burn-in is not certified.',
               'A fixed terminal step is audited without energy-based checkpoint selection.'])

    def save():
        result['elapsed_seconds'] = perf_counter()-start
        atomic_json(output/'results.json', result)

    def timeout(*_):
        result.update(status='censored', reason='registered local hard time cap')
        save()
        raise SystemExit(2)
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(spec['hard_seconds_cap'])
    save()

    def audit(bank, label, seed):
        tick = perf_counter()
        key, init_key = jax.random.split(jax.random.PRNGKey(seed))
        states = initialize_spin_exchange_chains(init_key, n_chains=audit_spec['chains'],
            n_sites=36, n_up=18)
        prepared = jax.jit(lambda b: prepare_feature_lut_bank(b, model))(bank)
        states, key, rows, info = jax.block_until_ready(audit_sampler(prepared, key, states))
        psi, hp = jax.block_until_ready(evaluate(prepared, rows.reshape(-1, 36)))
        if np.any(np.asarray(psi) == 0):
            raise FloatingPointError('exact zero Born audit centers; no flooring allowed')
        energy = np.asarray(hp/psi).reshape(rows.shape[:2])
        record = summarize_born_audit(energy, np.asarray(rows), n_sites=36)
        record.update(sampler=_plain(info), elapsed_seconds=perf_counter()-tick,
            params_sha256=pytree_sha256(bank), seed=seed,
            independent_of_training=True, endpoint_selection_used_audit=False)
        directory = output/label
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory/'audit_samples.npz', bits=np.asarray(rows),
            psi=np.asarray(psi).reshape(rows.shape[:2]), Hpsi=np.asarray(hp).reshape(rows.shape[:2]),
            local_energy=energy, terminal_chains=np.asarray(states), rng=np.asarray(key))
        atomic_json(directory/'audit.json', record)
        print(label, 'audit', json.dumps(record), flush=True)
        return record

    try:
        result['parent_audit'] = audit(parent, 'parent', spec['audit_seed'])
        save()
        for arm_index, arm in enumerate(('free', 'linear', 'attention')):
            cfg = TriangleDictionaryAttentionConfig(model=model,
                architecture='linear' if arm == 'linear' else 'attention', **spec['generator'])
            if arm == 'free':
                params, initialization = parent, dict(exact_same_bank_expected=True)
                generate = lambda p: p
            else:
                params, initialization = init_dictionary_attention_from_bank(
                    jax.random.PRNGKey(spec['generator_seed']), parent, cfg)
                generate = lambda p, cfg=cfg: dictionary_attention_bank(p, cfg)
            initial_params = params
            initial_bank_error = _tree_relative_error(generate(params), parent)
            if initial_bank_error > spec['tolerance']:
                raise ArithmeticError('arms do not replay the same initial physical bank')
            prepare = lambda p, generate=generate: prepare_feature_lut_bank(generate(p), model)
            prepared_fn = jax.jit(prepare)
            log_prepared = lambda bank, bits: jnp.log(amplitude(bank, bits).astype(jnp.complex128))
            update = make_prepared_sr_update(prepare, log_prepared, params, **spec['sr'])
            key, init_key = jax.random.split(jax.random.PRNGKey(spec['training_seed']))
            states = initialize_spin_exchange_chains(init_key, n_chains=train_spec['chains'], n_sites=36, n_up=18)
            prepared = prepared_fn(params)
            states, key, _, initial_sampling = jax.block_until_ready(initial_sampler(prepared, key, states))
            history = []
            optimizer = dict(completed_steps=jnp.asarray(0, jnp.int64), damping=jnp.asarray(spec['sr']['damping']))
            directory = output/arm
            checkpoint = directory/'checkpoint.pkl'
            result['arms'][arm] = dict(status='running', parameter_count=dictionary_parameter_count(params),
                parameter_count_by_component={name: dictionary_parameter_count(value)
                    for name, value in params.items()},
                initialization=initialization, initial_common_bank_relative_error=initial_bank_error,
                generator_config=asdict(cfg), history=history, initial_sampling=_plain(initial_sampling))
            save()

            transition = _transition_factory(prepare, sampler, evaluate, update)
            for step in range(train_spec['steps']):
                tick = perf_counter()
                params, states, key, rows, energies, sampling, diagnostics = transition(params, states, key)
                optimizer = dict(completed_steps=jnp.asarray(step+1, jnp.int64),
                    damping=jnp.asarray(spec['sr']['damping']), last_diagnostics=diagnostics)
                history.append(dict(step=step+1, training_energy_per_site=float(energies.real.mean()/36),
                    training_local_energy_variance_per_site=float(jnp.mean(abs(energies-energies.mean())**2)/36),
                    elapsed_seconds=perf_counter()-tick, sampler=_plain(sampling), sr=_plain(diagnostics)))
                save_checkpoint(checkpoint, params, optimizer_state=optimizer, metadata=dict(
                    sampler_states=np.asarray(states), rng=np.asarray(key), arm=arm,
                    registration_sha256=registration['sha256'], params_sha256=pytree_sha256(params),
                    model_config=asdict(cfg), objective='fresh_Born_VMC_full_H', history=history,
                    optimizer='stateless_ridge_SR_new_counter_no_parent_Adam_moments'))
                np.savez_compressed(directory/'last_training_samples.npz', bits=np.asarray(rows),
                    local_energy=np.asarray(energies).reshape(rows.shape[:2]))
                print(arm, 'step', step+1, json.dumps(history[-1]), flush=True)
                save()
            # The endpoint is frozen before any audit is inspected.
            endpoint_hash = pytree_sha256(params)
            terminal = audit(generate(params), arm, spec['audit_seed']+arm_index+1)
            restored = load_checkpoint(checkpoint)
            continued = transition(params, states, key)
            replay = transition(restored['params'], jnp.asarray(restored['metadata']['sampler_states']),
                jnp.asarray(restored['metadata']['rng']))
            replay_exact = (pytree_sha256(continued) == pytree_sha256(replay) and
                pytree_sha256(restored['optimizer_state']) == pytree_sha256(optimizer))
            if not replay_exact or pytree_sha256(params) != endpoint_hash:
                raise ArithmeticError('endpoint/checkpoint continuation replay failed')
            changes = {name: float(np.sqrt(sum(np.sum(abs(np.asarray(a)-np.asarray(b))**2)
                for a, b in zip(jax.tree.leaves(params[name]), jax.tree.leaves(initial_params[name]), strict=True))))
                for name in params}
            result['arms'][arm].update(status='completed', terminal_audit=terminal,
                trained_parameter_change_norm_by_component=changes,
                trained_parameter_leaves=int(sum(np.any(np.asarray(a) != np.asarray(b))
                    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(initial_params), strict=True))),
                total_parameter_leaves=len(jax.tree.leaves(params)),
                endpoint_params_sha256=endpoint_hash, checkpoint_sha256=file_sha256(checkpoint),
                checkpoint_params_optimizer_sampler_RNG_full_next_update_replay_exact=replay_exact,
                next_transition_replay_sha256=pytree_sha256(continued),
                replay_updates_committed=0, additional_replay_full_H_batches=2)
            save()
        result['status'] = 'completed'
        save()
        return result
    except BaseException as error:
        if result['status'] != 'censored':
            result.update(status='failed', error=repr(error))
            save()
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--replay-completed', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(replay_completed(arguments.config) if arguments.replay_completed
                     else run(arguments.config), indent=2))
