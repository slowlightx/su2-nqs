"""Bounded three-arm N36 training with exact fixed-carrier feature reuse.

Every feature includes the full space projector and physical H. Only their
nonlinear coefficients are optimized. Audit samples are drawn after endpoint
sealing. Small fixed probes validate the representation, never choose a state.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.a1_carrier_queries import compile_a1_overlapping_carrier_queries
from su2_nqs.benchmarks.multicontext_resonance_features import (
    features_from_plan, multicontext_feature_coefficients,
)
from su2_nqs.benchmarks.multicontext_resonance_queries import multicontext_carrier_amplitudes
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit, triangular_pi_zero_flux_hopping, BOUNDARY_ORBIT,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact, load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.additive_environment import additive_environment_generators
from su2_nqs.models.conditional_carrier_resonance import local_scalar_components
from su2_nqs.models.multicontext_resonance_attention import (
    MulticontextResonanceAttentionConfig, init_multicontext_resonance_attention,
    make_multicontext_reference, multicontext_resonance_generators, interaction_contrast,
)
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig, projected_singlet_agp_amplitudes
from su2_nqs.models.slater_minor_updates import (
    SlaterMinorPlan, compile_slater_minor_plan, slater_minor_updates, closed_shell_amplitudes,
)
from su2_nqs.vmc.cutset_training import paired_chain_audit, weighted_state_distance
from su2_nqs.vmc.fixed_proposal_rayleigh import weighted_full_h_loss, weighted_full_h_moments
from su2_nqs.vmc.matched_state_step import matched_state_step
from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
from .spin_overlap_attention_cost import _save_npz, _relative, peak_rss_gib
from .spin_schur_carrier_control import draw, batched

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_resonance_training.json'
ARMS = ('additive110', 'free360', 'attention730')


def validate_protocol(spec: dict) -> None:
    """One fixed experiment; reject silent budget, geometry or audit changes."""
    if (spec['task'] != 'spin_resonance_training' or tuple(spec['arms']) != ARMS
            or (spec['geometry'], spec['length'], spec['j2_over_j1']) != ('triangular', 6, .125)
            or spec['remote_allowed'] or spec['training_steps'] != 60
            or (spec['chains'], spec['train_blocks'], spec['audit_blocks']) != (16, 8, 8)
            or spec['step_bisections'] != 11 or spec['step_fs_radius'] != .002
            or spec['primary_contrast'] != 'free360-minus-additive110'):
        raise ValueError('not the registered bounded three-arm N36 experiment')
    seeds = [spec[key] for key in ('initialization_seed', 'scale_seed', 'train_seed', 'audit_seed', 'bootstrap_seed')]
    if len(set(seeds)) != len(seeds):
        raise ValueError('independent initialization, training and audit seeds required')
    for name in ('wall_seconds_cap', 'peak_rss_gib_cap', 'microbatch', 'minor_batch_size'):
        if not np.isfinite(spec[name]) or spec[name] <= 0:
            raise ValueError('positive finite resource caps required')
    if spec['microbatch'] != 2 or spec['minor_max_rank'] != 7:
        raise ValueError('fixed two-center exact query batch required')


def registration(spec: dict) -> dict:
    validate_protocol(spec)
    source = ROOT/spec['source_run']
    parent_registration = json.loads((source/'registration.json').read_text())
    ancestor = ROOT/parent_registration['config']['source_run']/'results.json'
    if file_sha256(ancestor) != parent_registration['config']['source_results_sha256']:
        raise ValueError('ancestor carrier/cell configuration changed')
    result = dict(config=spec, runtime=numerical_runtime_identity(),
        inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in
                (source/'results.json', source/'registration.json', source/'replay.npz', ancestor)},
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    result['registration_sha256'] = canonical_sha256(result)
    return result


class FixedCarrierFeatures:
    """Exact preparation reused across every arm and optimizer step."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
        self.bonds = triangular_hnf_j1j2_bonds(((6, 0), (0, 6)), j2=.125)
        parent = json.loads((ROOT/spec['source_run']/'results.json').read_text())
        old_path = ROOT/json.loads((ROOT/spec['source_run']/'registration.json').read_text())['config']['source_run']
        old = json.loads((old_path/'results.json').read_text())
        self.cells = old['config']['physical_cells']
        orbit = build_n36_pi_zero_flux_orbit()
        agp = ProjectedSingletAGPConfig(36, (tuple(range(36)),))
        pairs = tuple(jnp.asarray(p) for p in orbit.pairings)
        @jax.jit
        def carrier(bits):
            v = jnp.stack([projected_singlet_agp_amplitudes({'pairing_real':p}, bits, agp).amplitudes for p in pairs])
            return jnp.asarray([1., -1., -1.])@v/3
        self.carrier = carrier
        occupied = jnp.asarray(np.stack([np.linalg.eigh(triangular_pi_zero_flux_hopping(
            6, boundary_x=x, boundary_y=y))[1][:, :18] for x, y in BOUNDARY_ORBIT]))
        @jax.jit
        def minor(reference_bits, target_bits, reference_indices, reference_rows,
                  added_sites, replacement_positions, active, sorting_signs, ranks, physical_signs):
            plan = SlaterMinorPlan(reference_bits, target_bits, reference_indices, reference_rows,
                added_sites, replacement_positions, active, sorting_signs, ranks, physical_signs, 7)
            pieces = [slater_minor_updates(occupied[k], plan, condition_limit=spec['minor_condition_limit']) for k in range(3)]
            values = jnp.stack([closed_shell_amplitudes(piece, plan) for piece in pieces])
            return jnp.asarray([1., -1., -1.])@values/3, jnp.stack([p.dense_fallback for p in pieces])
        self.minor = minor
        self.parent = parent

    def prepare(self, centers: np.ndarray, scale: float, *, verify: bool = False):
        """Return node-safe C, HC, 1176 features and H features for two centers."""
        if centers.shape != (2, 36):
            raise ValueError('fixed two-center batch required')
        start = perf_counter(); spec = self.spec
        hp = compile_physical_h_queries(centers, self.bonds)
        plan = compile_a1_overlapping_carrier_queries(hp.configurations, self.group, cells=self.cells,
            carrier_sector='Gamma-A1', max_unique_queries=spec['max_unique_queries'],
            max_terminal_requests=spec['max_terminal_requests'], max_plan_bytes=spec['max_plan_bytes'])
        planning = perf_counter()-start
        tick = perf_counter(); values = np.empty(len(plan.configurations), dtype=np.complex128)
        fallback = 0; batch = spec['minor_batch_size']
        names = ('reference_bits', 'target_bits', 'reference_indices', 'reference_rows', 'added_sites',
                 'replacement_positions', 'active', 'sorting_signs', 'ranks', 'physical_signs')
        for i in range(0, len(values), batch):
            count = min(batch, len(values)-i)
            target = np.pad(plan.configurations[i:i+count], ((0, batch-count), (0, 0)), mode='edge')
            refs = np.argmin(np.sum(target[:, None] != centers[None], axis=-1), axis=1)
            mp = compile_slater_minor_plan(centers, target, refs, max_rank=7)
            v, fb = self.minor(*(jnp.asarray(getattr(mp, n)) for n in names))
            values[i:i+count] = np.asarray(v)[:count]/scale
            fallback += int(np.count_nonzero(np.asarray(fb)[:, :count]))
        carrier_seconds = perf_counter()-tick
        tick = perf_counter()
        feature = features_from_plan(plan, values, image_batch_size=128)
        allf = np.asarray(feature.features); c = np.asarray(feature.projected_carrier)
        center_f = allf[hp.center_indices]
        hf = np.sum(allf[hp.connected_indices]*hp.elements[..., None]*hp.valid[..., None], axis=1)
        cc = c[hp.center_indices]
        hc = np.sum(c[hp.connected_indices]*hp.elements*hp.valid, axis=1)
        record = dict(planning_seconds=planning, carrier_seconds=carrier_seconds,
            feature_seconds=perf_counter()-tick, full_H_centers=2, H_outputs=len(hp.configurations),
            carrier_queries=len(values), fallback_components=fallback, group_order=len(self.group),
            bond_count=len(self.bonds), peak_rss_gib=peak_rss_gib())
        if verify:
            t = np.random.default_rng(2026101406).normal(size=(6, 6, 10))*.01
            direct = np.asarray(multicontext_carrier_amplitudes(jnp.asarray(t), jnp.asarray(t), plan, jnp.asarray(values)))
            coef = np.asarray(multicontext_feature_coefficients(jnp.asarray(t), jnp.asarray(t)))
            reconstructed = c+allf@coef
            record['amplitude_relative_error'] = _relative(reconstructed, direct)
            record['carrier_dense_relative_error'] = _relative(cc, batched(self.carrier, centers, 2)/scale)
            record['full_H_relative_error'] = _relative(hc+hf@coef,
                np.sum(direct[hp.connected_indices]*hp.elements*hp.valid, axis=1))
            if max(record[k] for k in ('amplitude_relative_error', 'carrier_dense_relative_error', 'full_H_relative_error')) > spec['amplitude_relative_tolerance']:
                raise FloatingPointError('exact feature or dynamic minor replay failed')
        record['seconds'] = perf_counter()-start
        return dict(carrier=cc, hcarrier=hc, features=center_f, hfeatures=hf), record


def build_corpus(engine: FixedCarrierFeatures, data: dict, scale: float, path: Path, progress) -> dict:
    """Checkpoint feature preparation; never discard a difficult sample."""
    arrays = {k:[] for k in ('carrier', 'hcarrier', 'features', 'hfeatures')}; records = []
    sample_seal = canonical_sha256({k:np.asarray(data[k]).tolist() for k in ('states', 'q', 'groups')})
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['sample_seal']) != sample_seal: raise ValueError('feature checkpoint sample mismatch')
            arrays = {k:[saved[k].copy()] for k in arrays}
            records = json.loads(str(saved['records']))
    done = sum(len(v) for v in arrays['carrier'])
    for begin in range(done, len(data['states']), 2):
        result, record = engine.prepare(data['states'][begin:begin+2], scale, verify=begin == 0)
        for key in arrays: arrays[key].append(result[key])
        records.append(record)
        _save_npz(path, **{k:np.concatenate(v) for k, v in arrays.items()},
            sample_seal=np.asarray(sample_seal), records=np.asarray(json.dumps(records)))
        progress(path.stem, completed=begin+2, total=len(data['states']))
        gc.collect()
    result = {k:np.concatenate(v) for k, v in arrays.items()}
    # Independent cheap carrier/HC from the positive-Q sampler, including total-H units.
    for key, expected in (('carrier', data['carrier']), ('hcarrier', data['hcarrier']*36)):
        if _relative(result[key], expected) > engine.spec['amplitude_relative_tolerance']:
            raise FloatingPointError('carrier/HC feature corpus disagrees with sampler reference')
    return result


def endpoint_structure(table: np.ndarray) -> dict:
    """Deterministic whole-Hilbert operator bound; not a physical observable."""
    rho = max(float(np.max(abs(np.linalg.eigvalsh(np.einsum('ijr,rab->ijab', table, basis)))))
              for basis in local_scalar_components().reduced_basis.values())
    joint = np.asarray(interaction_contrast(jnp.asarray(table)))
    return dict(generator_max_spectral_norm=rho,
        rigorous_relative_state_change_bound=float(np.expm1(2*rho)),
        joint_generator_norm=float(np.linalg.norm(joint)),
        generator_norm=float(np.linalg.norm(table)))


def train_arm(arm: str, spec: dict, data: dict, corpus: dict, output: Path, initial, reference, progress):
    """Independent parameters; full-H empirical loss, fixed work and endpoint."""
    cfg = MulticontextResonanceAttentionConfig(); positions = jnp.asarray(spec['positions'])
    if arm == 'attention730':
        x0, unravel = ravel_pytree(initial)
        table = lambda x: multicontext_resonance_generators(unravel(x), reference, positions, cfg)
    elif arm == 'additive110':
        x0 = jnp.zeros(110); table = additive_environment_generators
    else:
        x0 = jnp.zeros(360); table = lambda x:x.reshape(6, 6, 10)
    c, hc, f, hf = (jnp.asarray(corpus[k]) for k in ('carrier', 'hcarrier', 'features', 'hfeatures'))
    q = jnp.asarray(data['q'])
    def state(x):
        t = table(x); coefficients = multicontext_feature_coefficients(t, t)
        return c+f@coefficients, hc+hf@coefficients
    state_jit = jax.jit(state)
    def loss(x):
        psi, hp = state(x)
        return weighted_full_h_loss(psi, hp/36, q)
    loss_gradient = jax.jit(jax.value_and_grad(loss))
    path = output/(arm+'_checkpoint.npz'); history_path = output/(arm+'_history.json')
    x = x0; m = jnp.zeros_like(x); v = jnp.zeros_like(x); history = []
    if path.exists():
        with np.load(path, allow_pickle=False) as checkpoint:
            x, m, v = (jnp.asarray(checkpoint[k]) for k in ('x', 'm', 'v'))
            history = json.loads(str(checkpoint['history']))
            if len(history) != int(checkpoint['completed_steps']) or str(checkpoint['arm']) != arm:
                raise ValueError('checkpoint step or arm mismatch')
            if not 0 <= len(history) <= spec['training_steps'] or [h['step'] for h in history] != list(range(1,len(history)+1)):
                raise ValueError('checkpoint history is not an ordered bounded trajectory')
            if any(value.shape != x0.shape or not np.isfinite(np.asarray(value)).all() for value in (x,m,v)):
                raise ValueError('checkpoint parameter shape/finiteness mismatch')
    start = perf_counter()
    for step in range(len(history), spec['training_steps']):
        loss, gradient = loss_gradient(x); jax.block_until_ready(gradient)
        if not np.isfinite(np.asarray(gradient)).all(): raise FloatingPointError('nonfinite full gradient')
        m = .9*m+.1*gradient; v = .999*v+.001*gradient**2
        d = -(m/(1-.9**(step+1)))/(jnp.sqrt(v/(1-.999**(step+1)))+1e-8)
        d /= jnp.maximum(jnp.linalg.norm(d), 1e-300)
        before, _ = state_jit(x)
        matched = matched_state_step(lambda alpha:np.asarray(state_jit(x+alpha*d)[0]),
            np.asarray(before), data['q'], spec['step_fs_radius'], spec['step_max_alpha'],
            spec['step_bisections'])
        x = x+matched.alpha*d
        psi, hp = state_jit(x)
        moments = weighted_full_h_moments(psi, hp, q)
        history.append(dict(step=step+1, loss_before_per_site=float(loss),
            energy_per_site=float(moments.energy_complex.real/36),
            imaginary_energy_per_site=float(moments.energy_complex.imag/36),
            variance_per_site=float(moments.variance/36), gradient_norm=float(jnp.linalg.norm(gradient)),
            physical_step=matched.as_dict(), seconds=perf_counter()-start))
        _save_npz(path, x=np.asarray(x), m=np.asarray(m), v=np.asarray(v),
            arm=np.asarray(arm), completed_steps=np.asarray(step+1), history=np.asarray(json.dumps(history)))
        atomic_json(history_path, history)
        if (step+1)%10 == 0: progress('train_'+arm, completed=step+1)
    psi, hp = state_jit(x)
    t = np.asarray(table(x)); coef = np.asarray(multicontext_feature_coefficients(t, t))
    endpoint = output/(arm+'_endpoint.json')
    endpoint_values = {'parameters':x, 'generator_table':t, 'coefficients':coef}
    identity = dict(arm=arm, completed_steps=spec['training_steps'], selection='fixed final step')
    if endpoint.exists():
        restored = load_parameter_artifact(endpoint, endpoint_values, expected_identity=identity)
        if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(endpoint_values))):
            raise ValueError('frozen endpoint differs from completed checkpoint')
    else: save_parameter_artifact(endpoint, endpoint_values, identity=identity)
    return dict(parameter_count=len(x), seconds=perf_counter()-start, history=history,
        structure=endpoint_structure(t), endpoint_fs=weighted_state_distance(corpus['carrier'], np.asarray(psi), data['q']),
        endpoint_sha256=file_sha256(endpoint), gradient_calls=spec['training_steps'],
        step_candidate_calls=spec['training_steps']*(spec['step_bisections']+1),
        cached_state_pair_evaluations=spec['training_steps']*(spec['step_bisections']+4)+1,
        work_note='Each state pair contracts cached psi/Hpsi features; counts include gradient objective, before, after, all step candidates and final endpoint. Not fresh carrier queries.'), coef


def run(config: Path = CONFIG, *, resume: bool = False) -> dict:
    spec = json.loads(config.read_text()); registered = registration(spec)
    output = ROOT/spec['output']; output.mkdir(parents=True, exist_ok=True)
    reg_path = output/'registration.json'
    if reg_path.exists():
        if not resume: raise FileExistsError('registered attempt exists; use --resume without changing its protocol')
        if json.loads(reg_path.read_text()) != registered: raise ValueError('registered code/config/inputs changed')
        if (output/'results.json').exists(): return json.loads((output/'results.json').read_text())
    else:
        if any(output.iterdir()): raise FileExistsError('refusing to adopt unregistered existing artifacts')
        atomic_json(reg_path, registered)
    start = perf_counter()
    previous_seconds = max([0., *(float(json.loads((output/name).read_text()).get('seconds', 0.))
        for name in ('progress.json', 'failure.json') if (output/name).exists())])
    if previous_seconds >= spec['wall_seconds_cap']:
        raise TimeoutError('cumulative registered budget exhausted; resume cannot extend it')
    def progress(stage, **kw):
        seconds = previous_seconds+perf_counter()-start
        record = dict(stage=stage, seconds=seconds, peak_rss_gib=peak_rss_gib(), **kw)
        atomic_json(output/'progress.json', record)
        print(json.dumps(record), flush=True)
        if seconds > spec['wall_seconds_cap']: raise TimeoutError('registered cumulative wall cap')
        if record['peak_rss_gib'] > spec['peak_rss_gib_cap']: raise MemoryError('registered process RSS cap')
    def timeout(*_): raise TimeoutError('registered wall cap; no automatic budget extension')
    handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, int(spec['wall_seconds_cap']-previous_seconds)))
    try:
        engine = FixedCarrierFeatures(spec)
        cfg = MulticontextResonanceAttentionConfig()
        initial = init_multicontext_resonance_attention(jax.random.PRNGKey(spec['initialization_seed']), cfg)
        reference = make_multicontext_reference(initial, jnp.asarray(spec['positions']), cfg)
        if not (output/'initial.json').exists():
            save_parameter_artifact(output/'initial.json', {'attention':initial, 'reference':reference},
                identity=dict(seed=spec['initialization_seed'], same_physical_state='G=0 carrier'))
        else:
            restored = load_parameter_artifact(output/'initial.json', {'attention':initial, 'reference':reference},
                expected_identity=dict(seed=spec['initialization_seed'], same_physical_state='G=0 carrier'))
            initial, reference = restored['attention'], restored['reference']
        scale_path = output/'scale.json'
        if scale_path.exists(): scale = json.loads(scale_path.read_text())['carrier_scale']
        else:
            rng = np.random.default_rng(spec['scale_seed']); uniform = np.zeros((spec['uniform_scale_samples'], 36), np.int8)
            for row in uniform: row[rng.choice(36, 18, replace=False)] = 1
            scale = float(np.sqrt(np.mean(abs(batched(engine.carrier, uniform, 256))**2)))
            if not np.isfinite(scale) or scale <= 0: raise FloatingPointError('invalid fixed carrier scale')
            atomic_json(scale_path, dict(carrier_scale=scale, samples=len(uniform)))
        progress('setup')
        def samples(phase):
            path = output/(phase+'_samples.npz'); info_path = output/(phase+'_sampling.json')
            if path.exists():
                info = json.loads(info_path.read_text())
                if info['sha256'] != file_sha256(path): raise ValueError('saved sample seal changed')
                return dict(np.load(path, allow_pickle=False)), info
            values, info = draw(spec, 36, engine.carrier, engine.bonds, scale, spec[phase+'_seed'], spec[phase+'_blocks'])
            _save_npz(path, **values); info['sha256'] = file_sha256(path)
            atomic_json(info_path, info); progress(phase+'_samples_saved')
            return values, info
        data, sampling = samples('train')
        corpus = build_corpus(engine, data, scale, output/'train_features.npz', progress)
        summaries = {}; coefficients = []
        for arm in ARMS:
            summaries[arm], coef = train_arm(arm, spec, data, corpus, output, initial, reference, progress)
            coefficients.append(coef)
        endpoint_seal = {arm:{ext:file_sha256(output/(arm+'_endpoint.'+ext)) for ext in ('json','npz')} for arm in ARMS}
        seal_path = output/'endpoints_seal.json'
        if seal_path.exists():
            if json.loads(seal_path.read_text()) != endpoint_seal: raise ValueError('frozen audit endpoint seal changed')
        else:
            if (output/'audit_samples.npz').exists(): raise ValueError('audit exists without prior endpoint seal')
            atomic_json(seal_path, endpoint_seal)
        progress('all_fixed_endpoints_sealed')
        audit_data, audit_sampling = samples('audit')
        audit_corpus = build_corpus(engine, audit_data, scale, output/'audit_features.npz', progress)
        for arm in ARMS:
            if any(file_sha256(output/(arm+'_endpoint.'+ext)) != endpoint_seal[arm][ext] for ext in ('json','npz')):
                raise ValueError('endpoint changed during untouched audit')
        coefs = np.stack([np.zeros(1176), *coefficients], axis=1)
        psi = audit_corpus['carrier'][:, None]+audit_corpus['features']@coefs
        hpsi = audit_corpus['hcarrier'][:, None]+audit_corpus['hfeatures']@coefs
        _save_npz(output/'audit_values.npz', psi=psi, hpsi=hpsi, coefficients=coefs,
            q=audit_data['q'], groups=audit_data['groups'])
        audit = paired_chain_audit(psi, hpsi, audit_data['q'], audit_data['groups'], 36,
            spec['bootstrap_replicates'], spec['bootstrap_seed'], ('carrier', *ARMS))
        primary = audit['paired'][spec['primary_contrast']]
        concentration_ok = all(audit['arms'][arm]['max_residual_chain_fraction'] < .5 for arm in ARMS)
        result = dict(status='complete', config=spec, training=summaries, audit=audit,
            sampling=dict(train=sampling, audit=audit_sampling), carrier_scale=scale,
            endpoint_audit_distances={arm:weighted_state_distance(psi[:, 0], psi[:, i+1], audit_data['q']) for i, arm in enumerate(ARMS)},
            primary_energy_interval_below_zero=primary['energy_per_site']['chain_bootstrap_95'][1] < 0,
            primary_variance_interval_nonpositive=primary['variance_per_site']['chain_bootstrap_95'][1] <= 0,
            no_single_audit_chain_dominates_residual=concentration_ok,
            equilibrium_certified=False, variational_win_claimed=False, remote_released=False,
            physical_H_prepared_centers=len(data['states'])+len(audit_data['states']),
            additional_cheap_carrier_H_centers=sampling['carrier_H_centers']+audit_sampling['carrier_H_centers'],
            full_H_feature_cost_shared_across_arms=True,
            feature_coordinates=1176, group_order=432, bonds=216,
            seconds=previous_seconds+perf_counter()-start, peak_rss_gib=peak_rss_gib(),
            registration_sha256=file_sha256(reg_path),
            artifacts={p.name:file_sha256(p) for p in output.glob('*.npz')},
            interpretation='Bounded fixed-Q training and independent-chain audit; finite-sample ratios are not rigorous energy bounds. No saturation, tail, equilibrium, GPU, or PRL win inferred automatically.')
        result['result_sha256'] = canonical_sha256(result)
        progress('complete')
        atomic_json(output/'results.json', result)
        return result
    except Exception as error:
        atomic_json(output/'failure.json', dict(error=repr(error), seconds=previous_seconds+perf_counter()-start,
            peak_rss_gib=peak_rss_gib(), automatic_extension=False))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); run(args.config, resume=args.resume)
