"""Bounded same-state N36/N64 SU(2) carrier-control force experiment.

One registration, one immutable parent, six outcome-independent full-model
JVPs, independent metric calibration and whole-chain paired noise audit.
The compact evaluator changes no model parameters, allowed q, or A1 images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.triangular_induced_parton import build_n36_pi_zero_flux_orbit
from su2_nqs.benchmarks.triangular_induced_parton_n64 import build_n64_pi_zero_flux_orbit
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.cell_singlet_initialization import cell_product_parameters, perturb_cell_product
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig, projected_singlet_agp_amplitudes
from su2_nqs.models.schur_cutset_attention import CutsetAttentionConfig, init_cutset_attention, make_branch_bank
from su2_nqs.models.schur_cutset_compact import compact_cg_storage, make_compact_cutset_amplitude, make_full_a1_batch
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from su2_nqs.vmc.singlet_carrier_control import parent_directional_force
from su2_nqs.vmc.singlet_diagonal_force import three_hz_local_energies
from .spin import _bonds, _groups
from .spin_decoder_cost import ROOT, physical_config, peak_rss_bytes
from .spin_vap_decoder import init_decoder

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_carrier_control.json'


def registration(spec: dict) -> dict:
    """Seal all numerical code and reference tensors before inspecting new data."""
    interface = json.loads((ROOT/spec['interface_config']).read_text())
    parent = ROOT/interface['parent']/'triangular'
    paths = [ROOT/spec['interface_config'], parent/'results.json', parent/'parameters.json', parent/'parameters.npz']
    result = dict(config=spec, inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    result['registration_sha256'] = canonical_sha256(result)
    return result


def setup(spec: dict, length: int):
    """Identical architecture and physical initialization policy at either size."""
    if length not in (6, 8):
        raise ValueError('only independently certified N36/N64 carriers supported')
    interface = json.loads((ROOT/spec['interface_config']).read_text())
    tree = physical_config(length, length, interface['width'])
    parent = ROOT/interface['parent']/'triangular'
    if length == 6:
        base = load_parameter_artifact(parent/'parameters.json', init_decoder(2026093003, tree))
    else:
        coefficients = np.asarray(json.loads((parent/'results.json').read_text())['coefficients'])
        base = perturb_cell_product(cell_product_parameters(tree, coefficients), seed=2026093021, scale=.05)
    coords = tuple((float(np.mean(np.asarray(c) % length)), float(np.mean(np.asarray(c)//length))) for c in tree.cells)
    cfg = CutsetAttentionConfig(tree, coords, cutset_cell=interface['selected_cell'],
        embedding_dim=interface['attention_width'], rank=interface['condition_rank'], scramble_seed=interface['scramble_seed'])
    params = init_cutset_attention(jax.random.PRNGKey(interface['seed']), base, cfg)
    group = _groups('triangular', length, length)['gamma_a1']
    residual = make_full_a1_batch(make_compact_cutset_amplitude(cfg), PhysicalSpatialProjector(group),
                                 chunk_size=spec['projection_chunk_size'])
    orbit = build_n36_pi_zero_flux_orbit() if length == 6 else build_n64_pi_zero_flux_orbit()
    agp = ProjectedSingletAGPConfig(n_sites=length**2, group_images=(tuple(range(length**2)),))
    coefficients = jnp.asarray([1., -1., -1.] if length == 6 else [1., 1., 1.])/3
    def carrier(x):
        components = jnp.stack([projected_singlet_agp_amplitudes(
            {'pairing_real': jnp.asarray(p)}, x, agp).amplitudes for p in orbit.pairings], axis=1)
        return components@coefficients
    return params, cfg, residual, jax.jit(carrier), _bonds('triangular', length, length, spec['j2_over_j1']), len(group)


def parameter_direction(params: dict, block: str, seed: int) -> dict:
    """Seeded block-balanced direction; no access to amplitudes or energy."""
    if block not in ('joint', 'base', 'attention'):
        raise ValueError('unknown direction block')
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('base', 'directions', 'attention'):
        leaves, structure = jax.tree_util.tree_flatten(params[name])
        active = block == 'joint' or block == name
        arrays = [rng.normal(size=v.shape) if active else np.zeros(v.shape) for v in leaves]
        norm = np.sqrt(sum(np.sum(v*v) for v in arrays))
        scale = max(norm, 1e-300)*(np.sqrt(3.) if block == 'joint' else 1.)
        out[name] = jax.tree_util.tree_unflatten(structure, [jnp.asarray(v/scale) for v in arrays])
    return out


def batched(fn, states: np.ndarray, size: int) -> np.ndarray:
    return np.concatenate([np.asarray(fn(jnp.asarray(states[i:i+size], dtype=jnp.int32)))
                           for i in range(0, len(states), size)])


def draw(spec: dict, n: int, carrier, bonds, cscale: float, seed: int, blocks: int):
    """Positive-Q MH followed by a fixed-probability, bijective exchange bridge."""
    floor = spec['uniform_floor_relative_to_pilot_mean']; beta = spec['bridge_fraction']
    density = jax.jit(lambda x: jnp.sqrt(abs(carrier(x)/cscale)**2+floor))
    tick = perf_counter()
    sampled = sample_fixed_wavefunction_exchange(lambda x: np.asarray(density(x)), n_sites=n, n_up=n//2,
        chain_count=spec['chains'], burn_sweeps=spec['burn_proposals_per_chain'], measurement_blocks=blocks,
        decorrelation_sweeps=spec['decorrelation_proposals_per_record'], seed=seed)
    sampling_seconds = perf_counter()-tick
    states = sampled.configurations.copy(); rng = np.random.default_rng(seed+100000)
    edges = np.asarray([(a, b) for a, b, _ in bonds]); coupling = np.asarray([j for _, _, j in bonds])
    probability = abs(coupling)/abs(coupling).sum(); labels = rng.choice(len(bonds), len(states), p=probability)
    bridge_used = rng.random(len(states)) < beta; rows = np.flatnonzero(bridge_used)
    a, b = edges[labels[rows]].T
    states[rows, a], states[rows, b] = states[rows, b].copy(), states[rows, a].copy()
    tick = perf_counter(); c = batched(lambda x: carrier(x)/cscale, states, 32)
    q = []; hc = []
    for x, value in zip(states, c, strict=True):
        swapped = np.broadcast_to(x, (len(bonds), n)).copy(); rows = np.arange(len(bonds))
        swapped[rows, edges[:, 0]], swapped[rows, edges[:, 1]] = x[edges[:, 1]], x[edges[:, 0]]
        connected = batched(lambda y: carrier(y)/cscale, swapped, 32)
        q.append((1-beta)*abs(value)**2+beta*np.dot(probability, abs(connected)**2)+floor)
        unlike = x[edges[:, 0]] != x[edges[:, 1]]
        hc.append(np.sum(coupling*(np.where(unlike, -.25, .25)*value+.5*unlike*connected))/n)
    return dict(states=states, q=np.asarray(q), carrier=c, hcarrier=np.asarray(hc),
        diagonal=np.asarray(three_hz_local_energies(states, bonds))/n,
        groups=np.tile(np.arange(spec['chains']), blocks), bridge_used=bridge_used, bridge_labels=labels), dict(
            sampling_seconds=sampling_seconds, density_and_HC_seconds=perf_counter()-tick,
            acceptance=sampled.acceptance_fraction, burn_acceptance=sampled.burn_acceptance_fraction,
            carrier_H_centers=len(states), normalization_unknown=True, equilibrium_certified=False,
            carrier_determinants=3*(spec['chains']*(1+spec['burn_proposals_per_chain']+
                blocks*spec['decorrelation_proposals_per_record'])+len(states)*(len(bonds)+1)))


def projective_metric(c: np.ndarray, t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Physical tangent norm modulo the complex ray, with fixed quadrature."""
    norm = np.sum(abs(c)**2/q)
    overlap = np.sum(c.conj()[:, None]*t/q[:, None], axis=0)/norm
    return np.sum(abs(t-c[:, None]*overlap)**2/q[:, None], axis=0)/norm


def force_summary(data: dict, tangents: np.ndarray, scales: np.ndarray, spec: dict) -> dict:
    """Paired whole-chain uncertainty; samples within one chain are NOT IID."""
    c, hc, q, d, groups = (data[k] for k in ('carrier', 'hcarrier', 'q', 'diagonal', 'groups'))
    t = tangents/scales[None, :]; labels = np.unique(groups)
    forces = {}
    compiled = jax.jit(parent_directional_force)
    for name, action in (('diagonal', d*c), ('control', hc)):
        def measured(index):
            return np.asarray(compiled(c[index], action[index], t[index], q[index]))
        chain = np.stack([measured(groups == label) for label in labels])
        pooled = measured(np.arange(len(c)))
        jack = np.stack([measured(groups != label) for label in labels])
        if not all(np.isfinite(value).all() for value in (chain, pooled, jack)):
            raise FloatingPointError('nonfinite chain force/norm; do not drop or replace a chain')
        se = np.sqrt((len(labels)-1)*np.mean((jack-jack.mean(axis=0))**2, axis=0))
        weights = abs(c)**2/q; norm = weights.sum(); energy = np.sum(c.conj()*action/q)/norm
        overlap = np.sum(c.conj()[:, None]*t/q[:, None], axis=0)/norm
        tangent_perp = t-c[:, None]*overlap
        influence = (2*np.real(tangent_perp.conj()*(action-energy*c)[:, None])/q[:, None]
                     - weights[:, None]*pooled[None, :])
        squared = np.sum(influence**2, axis=1)
        forces[name] = dict(chain_forces=chain.tolist(), chain_mean=chain.mean(axis=0).tolist(),
            chain_mean_se=(chain.std(axis=0, ddof=1)/np.sqrt(len(labels))).tolist(),
            pooled=pooled.tolist(), pooled_jackknife_se=se.tolist(),
            covariance_trace=float(np.var(chain, axis=0, ddof=1).sum()),
            energy_per_site=[float(energy.real), float(energy.imag)],
            max_sample_squared_influence_fraction=float(squared.max()/max(squared.sum(), 1e-300)),
            max_chain_squared_influence_fraction=float(max(squared[groups == g].sum() for g in labels)/max(squared.sum(), 1e-300)))
    diagonal = np.asarray(forces['diagonal']['chain_forces']); control = np.asarray(forces['control']['chain_forces'])
    rng = np.random.default_rng(spec['bootstrap_seed'])
    draws = rng.integers(len(labels), size=(spec['bootstrap_replicates'], len(labels)))
    ratio = np.var(diagonal[draws], axis=1, ddof=1).sum(axis=1)/np.maximum(
        np.var(control[draws], axis=1, ddof=1).sum(axis=1), 1e-300)
    weights = abs(c)**2/q; norm = weights.sum()
    overlap = np.sum(c.conj()[:, None]*t/q[:, None], axis=0)/norm
    perpendicular = abs(t-c[:, None]*overlap)**2/q[:, None]
    return dict(forces=forces, variance_reduction=forces['diagonal']['covariance_trace']/max(forces['control']['covariance_trace'], 1e-300),
        variance_reduction_bootstrap95=np.quantile(ratio, [.025, .975]).tolist(),
        paired_force_difference=(control-diagonal).mean(axis=0).tolist(),
        paired_force_difference_se=((control-diagonal).std(axis=0, ddof=1)/np.sqrt(len(labels))).tolist(),
        audit_projective_norm_squared=projective_metric(c, t, q).tolist(),
        tangent_max_sample_norm_fraction=(perpendicular.max(axis=0)/np.maximum(perpendicular.sum(axis=0), 1e-300)).tolist(),
        carrier_norm_ess_fraction=float(norm**2/np.sum(weights**2)/len(c)),
        carrier_nodes=int(np.count_nonzero(c == 0)), samples=len(c), independent_chains=len(labels),
        calibration_scales=scales.tolist(), inference='Frozen directional-force evidence only; no optimized energy, no Var(H), no equilibrium certificate.')


def run(path: Path, length: int) -> dict:
    spec = json.loads(path.read_text()); root = ROOT/spec['output']; dest = root/f'n{length**2}'
    registered = json.loads((root/'registration.json').read_text())
    if registered != registration(spec):
        raise ValueError('source, input or protocol changed after registration')
    if length not in spec['sizes']:
        raise ValueError('size not registered')
    if (dest/'progress.json').exists():
        raise FileExistsError('one bounded attempt per size; do not silently restart')
    dest.mkdir(parents=True, exist_ok=True); start = perf_counter(); timings = {}
    def progress(event, **extra):
        record = dict(event=event, seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes(), **extra)
        atomic_json(dest/'progress.json', record)
        print(length, event, round(record['seconds'], 2), 's', round(record['peak_rss_bytes']/2**30, 2), 'GiB', flush=True)
        if record['peak_rss_bytes'] > spec['peak_rss_limit_gib']*2**30:
            raise MemoryError('registered RSS limit')
    def alarm(_s, _f):
        raise TimeoutError('registered wall limit; no automatic extension')
    previous = signal.signal(signal.SIGALRM, alarm); signal.alarm(spec['wall_seconds_per_size'])
    try:
        params, cfg, residual, carrier, bonds, group_order = setup(spec, length)
        identity = dict(registration_sha256=registered['registration_sha256'], length=length)
        save_parameter_artifact(dest/'parameters.json', params, identity=identity)
        bank = make_branch_bank(params, cfg); directions = []; names = []
        for block in spec['direction_blocks']:
            for seed in spec['direction_seeds']:
                tangent = parameter_direction(params, block, seed)
                directions.append(jax.jvp(lambda p: make_branch_bank(p, cfg), (params,), (tangent,))[1])
                names.append(f'{block}:{seed}')
        derivative = jax.jit(lambda b, db, x: jax.jvp(lambda v: residual(v, x), (b,), (db,)))
        rng = np.random.default_rng(spec['scale_seed']); uniform = np.zeros((spec['uniform_scale_samples'], length**2), dtype=np.int8)
        for row in uniform:
            row[rng.choice(length**2, length**2//2, replace=False)] = 1
        cprobe = batched(carrier, uniform, 32); cscale = float(np.sqrt(np.mean(abs(cprobe)**2)))
        if not np.isfinite(cscale) or cscale <= 0:
            raise FloatingPointError('invalid carrier scale')
        progress('setup_and_fixed_directions')
        scales = None; metadata = {}; counts = {}
        for phase in ('calibration', 'audit'):
            data, metadata[phase] = draw(spec, length**2, carrier, bonds, cscale, spec[phase+'_seed'], spec[phase+'_blocks'])
            np.savez_compressed(dest/(phase+'_samples.npz'), **data)
            progress(phase+'_samples_saved')
            samples = len(data['states']); tangent = np.empty((samples, len(directions)), dtype=np.complex128)
            amplitudes = np.empty(samples, dtype=np.complex128); times = []
            for k, dbank in enumerate(directions):
                tick = perf_counter()
                for i in range(0, samples, spec['microbatch']):
                    x = jnp.asarray(data['states'][i:i+spec['microbatch']], dtype=jnp.int32)
                    value, dt = derivative(bank, dbank, x)
                    amplitudes[i:i+len(x)], tangent[i:i+len(x), k] = np.asarray(value), np.asarray(dt)
                times.append(perf_counter()-tick)
                np.savez_compressed(dest/(phase+'_jvp_checkpoint.npz'), amplitudes=amplitudes,
                                    tangents=tangent[:, :k+1], completed_direction_count=k+1)
                progress(phase+'_direction_'+str(k+1))
            if not np.isfinite(tangent).all():
                raise FloatingPointError('nonfinite directional amplitude; no direction replacement')
            metric = projective_metric(data['carrier'], tangent, data['q'])
            if phase == 'calibration':
                if np.any(metric <= 1e-300) or not np.isfinite(metric).all():
                    raise FloatingPointError('null/invalid preregistered tangent; no replacement allowed')
                scales = np.sqrt(metric)
            np.savez_compressed(dest/(phase+'_values.npz'), **data, amplitudes=amplitudes, tangents=tangent, scales=scales)
            timings[phase+'_directions_seconds'] = times; counts[phase] = samples
            if phase == 'audit':
                summary = force_summary(data, tangent, scales, spec)
        result = dict(identity=identity, summary=summary, direction_names=names, sampling=metadata,
            timings=timings, carrier_scale=cscale, parameter_count=sum(v.size for v in jax.tree_util.tree_leaves(params)),
            parameters_sha256=pytree_sha256(params), group_order=group_order, bond_count=len(bonds),
            cg_storage=compact_cg_storage(cfg), seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes(),
            training_steps=0, residual_H_centers=0, carrier_H_centers=sum(counts.values()),
            directional_amplitude_centers=sum(counts.values())*len(directions),
            carrier_determinants=3*len(cprobe)+sum(v['carrier_determinants'] for v in metadata.values()),
            cost_note='Both estimators share Q/H-connection preprocessing and the same six JVPs. HC adds no determinant queries to this bridge. Costs include CPU compilation; not a GPU forecast.',
            artifacts={p.name:file_sha256(p) for p in dest.glob('*.npz')})
        result['result_sha256'] = canonical_sha256(result); atomic_json(dest/'results.json', result)
        progress('complete'); print(json.dumps(summary, indent=2), flush=True); return result
    except Exception as error:
        atomic_json(dest/'failure.json', dict(error=repr(error), seconds=perf_counter()-start, timings=timings,
                    peak_rss_bytes=peak_rss_bytes()))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--length', type=int, choices=(6, 8))
    args = parser.parse_args(); spec = json.loads(args.config.read_text())
    if args.register:
        path = ROOT/spec['output']/'registration.json'
        if path.exists():
            raise FileExistsError(path)
        record = registration(spec); atomic_json(path, record); print(record['registration_sha256'])
    elif args.length is None:
        parser.error('--length required unless --register')
    else:
        run(args.config, args.length)


if __name__ == '__main__':
    main()
