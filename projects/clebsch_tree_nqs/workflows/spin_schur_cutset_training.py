"""One bounded joint-training experiment, reusing the sealed cutset model.

Training is a fixed defensive quadrature of the exact-singlet 3Hz functional.
Only terminal independent-chain full-H measurements are physical variance.
No previous model, source manifest or numerical result is rewritten.
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

from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.schur_cutset_attention import (
    CutsetAttentionConfig, cutset_raw_amplitude, init_cutset_attention,
    make_branch_bank, make_free_parameters, make_free_branch_bank,
)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.cutset_training import weighted_state_distance, paired_chain_audit
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from su2_nqs.vmc.projected_spin_work import make_chunked_a1_amplitudes
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint
from su2_nqs.vmc.singlet_diagonal_force import three_hz_local_energies
from .spin import _bonds, _groups
from .spin_decoder_cost import ROOT, physical_config, peak_rss_bytes
from .spin_vap_decoder import init_decoder

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_cutset_training.json'


def registration(spec: dict) -> dict:
    """Seal protocol and all numerical dependencies before preparing samples."""
    interface = json.loads((ROOT/spec['interface_config']).read_text())
    paths = [ROOT/spec['interface_config']]
    paths += [ROOT/interface['parent']/'triangular'/f'parameters.{ext}' for ext in ('json', 'npz')]
    result = dict(config=spec, inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    result['registration_sha256'] = canonical_sha256(result)
    return result


def setup(spec: dict):
    interface = json.loads((ROOT/spec['interface_config']).read_text())
    tree = physical_config(6, 6, interface['width'])
    base = load_parameter_artifact(ROOT/interface['parent']/'triangular/parameters.json',
                                  init_decoder(2026093003, tree))
    coords = tuple((float(np.mean(np.asarray(c) % 6)), float(np.mean(np.asarray(c) // 6))) for c in tree.cells)
    config = CutsetAttentionConfig(tree, coords, cutset_cell=interface['selected_cell'],
        embedding_dim=interface['attention_width'], rank=interface['condition_rank'],
        scramble_seed=interface['scramble_seed'])
    attention = init_cutset_attention(jax.random.PRNGKey(interface['seed']), base, config)
    params = dict(attention=attention, scrambled=attention,
                  rank_free=make_free_parameters(attention, config))
    generators = dict(attention=lambda p: make_branch_bank(p, config),
                      scrambled=lambda p: make_branch_bank(p, config, scrambled=True),
                      rank_free=lambda p: make_free_branch_bank(p, config))
    residual = make_chunked_a1_amplitudes(lambda p, x: cutset_raw_amplitude(p, x, config),
        PhysicalSpatialProjector(_groups('triangular', 6, 6)['gamma_a1']),
        chunk_size=interface['projection_chunk_size'])
    orbit = build_n36_pi_zero_flux_orbit()
    carrier = jax.jit(lambda x: projected_pi_zero_flux_a1_amplitudes_jax(x, orbit=orbit))
    return params, generators, residual, carrier, _bonds('triangular', 6, 6, .125)


def _batch(fn, states: np.ndarray, size: int = 8) -> np.ndarray:
    return np.concatenate([np.asarray(fn(jnp.asarray(states[i:i+size], dtype=jnp.int32)))
                           for i in range(0, len(states), size)])


def draw(spec: dict, carrier, bonds, carrier_scale: float, seed: int, blocks: int):
    """MCMC base then fixed-probability exchange pushforward; no IID assertion."""
    floor = spec['uniform_floor_relative_to_pilot_mean']
    q0_amplitude = jax.jit(lambda x: jnp.sqrt(abs(carrier(x)/carrier_scale)**2+floor))
    sampled = sample_fixed_wavefunction_exchange(lambda x: np.asarray(q0_amplitude(x)),
        n_sites=36, n_up=18, chain_count=spec['chains'],
        burn_sweeps=spec['burn_proposals_per_chain'], measurement_blocks=blocks,
        decorrelation_sweeps=spec['decorrelation_proposals_per_record'], seed=seed)
    states = sampled.configurations.copy()
    rng = np.random.default_rng(seed+100000)
    edges = np.asarray([(a,b) for a,b,_ in bonds]); couplings = np.asarray([j for _,_,j in bonds])
    p = abs(couplings)/abs(couplings).sum()
    labels = rng.choice(len(bonds), len(states), p=p)
    use = rng.random(len(states)) < spec['bridge_fraction']
    rows = np.flatnonzero(use); a,b = edges[labels[rows]].T
    states[rows,a], states[rows,b] = states[rows,b].copy(), states[rows,a].copy()
    centers = _batch(lambda x: carrier(x)/carrier_scale, states)
    probability = []; hc = []
    for x, c in zip(states, centers, strict=True):
        swaps = np.broadcast_to(x, (len(bonds),36)).copy(); r = np.arange(len(bonds))
        swaps[r,edges[:,0]], swaps[r,edges[:,1]] = x[edges[:,1]], x[edges[:,0]]
        connected = _batch(lambda y: carrier(y)/carrier_scale, swaps, 32)
        qbase = abs(c)**2+floor
        pushed = np.dot(p, abs(connected)**2)+floor
        probability.append((1-spec['bridge_fraction'])*qbase+spec['bridge_fraction']*pushed)
        unlike = x[edges[:,0]] != x[edges[:,1]]
        hc.append(np.sum(couplings*(np.where(unlike,-.25,.25)*c+.5*unlike*connected)))
    return dict(states=states, q=np.asarray(probability), carrier=centers,
        carrier_h=np.asarray(hc), groups=np.tile(np.arange(spec['chains']), blocks),
        bridge_used=use, bridge_labels=labels), dict(
            acceptance=sampled.acceptance_fraction, burn_acceptance=sampled.burn_acceptance_fraction,
            samples=len(states), chains=spec['chains'], seed=seed,
            carrier_density_connection_centers=len(states),
            carrier_determinants=3*(spec['chains']*(1+spec['burn_proposals_per_chain']+
                blocks*spec['decorrelation_proposals_per_record'])+len(states)*(1+len(bonds))),
            normalization_unknown=True, equilibrium_certified=False)


def run(stage: str, arm: str | None) -> None:
    spec = json.loads(CONFIG.read_text()); output = ROOT/spec['output']
    registered = json.loads((output/'registration.json').read_text())
    if registered != registration(spec):
        raise ValueError('numeric source/config/input changed since registration')
    identity = dict(registration_sha256=registered['registration_sha256'], stage=stage, arm=arm)
    dest = output/(arm if stage == 'train' else stage); dest.mkdir(parents=True, exist_ok=True)
    if (dest/'results.json').exists():
        raise FileExistsError('stage is sealed; do not overwrite')
    start = perf_counter(); prior_seconds = 0.; ledger = dict(residual_amplitude_centers=0, residual_gradient_centers=0,
                                         residual_H_centers=0, carrier_density_connection_centers=0,
                                         shared_carrier_H_centers=0, carrier_determinants=0)
    def progress(event: str, **extra):
        elapsed = prior_seconds+perf_counter()-start
        atomic_json(dest/'progress.json', dict(event=event, seconds=elapsed,
            peak_rss_bytes=peak_rss_bytes(), ledger=ledger, **extra))
        print(stage, arm, event, round(elapsed,1), 's', round(peak_rss_bytes()/2**30,2),'GiB', flush=True)
        if peak_rss_bytes()>spec['peak_rss_limit_gib']*2**30:
            raise MemoryError('registered RSS cap')
    def alarm(_s,_f): raise TimeoutError('registered stage wall cap; no automatic extension')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(spec[{'prepare':'prepare_wall_seconds','train':'training_wall_seconds_per_arm','audit':'audit_wall_seconds'}[stage]])
    params, generators, residual, carrier, bonds = setup(spec)
    evaluate = jax.jit(residual)
    def values(bank, states):
        answer = _batch(lambda x: evaluate(bank, x), states, spec['microbatch'])
        ledger['residual_amplitude_centers'] += len(states)
        return answer
    progress('setup')
    if stage == 'prepare':
        rng = np.random.default_rng(spec['scale_seed'])
        uniform = np.zeros((spec['uniform_calibration_samples'],36), dtype=np.int8)
        for x in uniform: x[rng.choice(36,18,replace=False)] = 1
        c = _batch(carrier, uniform, 32)
        ledger['carrier_determinants'] += 3*len(uniform)
        cscale = float(np.sqrt(np.mean(abs(c)**2)))
        if not cscale>0: raise FloatingPointError('zero carrier scale')
        metadata = dict(carrier_scale=cscale, uniform_scale_relative_se=float(np.std(abs(c)**2,ddof=1)/
                         np.sqrt(len(c))/np.mean(abs(c)**2)), sampling={})
        for label, blocks in [('calibration',spec['calibration_blocks']),('training',spec['training_blocks'])]:
            data, info = draw(spec, carrier, bonds, cscale, spec[label+'_seed'], blocks)
            ledger['carrier_density_connection_centers'] += len(data['states'])
            ledger['shared_carrier_H_centers'] += len(data['states'])
            ledger['carrier_determinants'] += info['carrier_determinants']
            metadata['sampling'][label] = info
            for name in ('attention','scrambled'):
                data[name+'_reference'] = values(generators[name](params[name]),data['states'])
            data['rank_free_reference'] = data['attention_reference']
            data['three_hz_per_site'] = np.asarray(three_hz_local_energies(data['states'],bonds))/36
            np.savez_compressed(dest/(label+'.npz'), **data)
            if label == 'calibration':
                metadata['residual_scale'] = float(np.sqrt(np.sum(abs(data['attention_reference'])**2/data['q'])/
                                                        np.sum(abs(data['carrier'])**2/data['q'])))
            progress(label+'_prepared')
        metadata.update(identity=identity, ledger=ledger, seconds=perf_counter()-start,
                        peak_rss_bytes=peak_rss_bytes())
        atomic_json(dest/'results.json',metadata)
    elif stage == 'train':
        prepared = json.loads((output/'prepare/results.json').read_text())
        training = dict(np.load(output/'prepare/training.npz'))
        calibration = dict(np.load(output/'prepare/calibration.npz'))
        scale = prepared['residual_scale']; generate = jax.jit(generators[arm])
        initial = params[arm]; current = initial
        reference = training[arm+'_reference']/scale
        cref = calibration[arm+'_reference']/scale
        history = []; m = jax.tree.map(jnp.zeros_like,current); v = jax.tree.map(jnp.zeros_like,current)
        if (dest/'checkpoint.pkl').exists():
            saved = load_resumable_checkpoint(dest/'checkpoint.pkl', expected_run_identity=identity)
            current=saved['params'];m,v=saved['optimizer_state'];history=saved['history']
            ledger.update(saved['metadata']['ledger'])
            prior_seconds = float(saved['metadata']['active_seconds'])
            remaining = spec['training_wall_seconds_per_arm']-prior_seconds-(perf_counter()-start)
            if remaining <= 0: raise TimeoutError('cumulative training budget exhausted')
            signal.alarm(max(1,int(remaining)))
        def micro_loss(bank, x, c, ref, q, diagonal, energy, norm_sum):
            psi = c+residual(bank,x)/scale-ref
            return jnp.sum(abs(psi)**2/q*(diagonal-energy))/norm_sum
        gradient = jax.jit(jax.grad(micro_loss))
        bank = generate(current)
        oldcal = calibration['carrier']+values(bank,calibration['states'])/scale-cref
        for step in range(len(history), spec['steps']):
            progress(f'gradient_{step+1}_begin')
            bank, pullback = jax.vjp(generate, current)
            psi = training['carrier']+values(bank,training['states'])/scale-reference
            norm = np.sum(abs(psi)**2/training['q'])
            energy = float(np.sum(abs(psi)**2/training['q']*training['three_hz_per_site'])/norm)
            gb = jax.tree.map(jnp.zeros_like,bank)
            for i in range(0,len(psi),spec['microbatch']):
                sl=slice(i,i+spec['microbatch'])
                gg=gradient(bank,training['states'][sl],training['carrier'][sl],reference[sl],
                    training['q'][sl],training['three_hz_per_site'][sl],energy,norm)
                gb=jax.tree.map(lambda a,b:a+b,gb,gg)
            g=pullback(gb)[0]; jax.block_until_ready(g)
            ledger['residual_gradient_centers'] += len(psi)
            if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(g)):
                raise FloatingPointError('nonfinite gradient')
            m=jax.tree.map(lambda a,b:.9*a+.1*b,m,g)
            v=jax.tree.map(lambda a,b:.999*a+.001*b*b,v,g)
            direction=jax.tree.map(lambda a,b:spec['learning_rate']*(a/(1-.9**(step+1)))/
                                  (jnp.sqrt(b/(1-.999**(step+1)))+1e-8),m,v)
            accepted=0.; attempts=[]; nextcal=oldcal
            for factor in spec['backtrack_factors']:
                candidate=jax.tree.map(lambda a,b:a-factor*b,current,direction)
                candidatecal=calibration['carrier']+values(generate(candidate),calibration['states'])/scale-cref
                distance=weighted_state_distance(oldcal,candidatecal,calibration['q'])
                parent_distance=weighted_state_distance(calibration['carrier'],candidatecal,calibration['q'])
                attempts.append(dict(factor=factor,distance=distance,parent_distance=parent_distance))
                if (distance['fs']<=spec['step_fs_cap'] and parent_distance['fs']<=spec['parent_fs_cap']
                    and distance['ess_fraction']>=spec['minimum_norm_weight_ess_fraction']):
                    current=candidate;nextcal=candidatecal;accepted=factor;break
            oldcal=nextcal
            top_grad={k:float(np.sqrt(sum(np.sum(np.asarray(x)**2) for x in jax.tree.leaves(g[k])))) for k in g}
            history.append(dict(step=step+1,training_energy_3hz_per_site_before=energy,
                accepted_factor=accepted,attempts=attempts,gradient_norm_by_block=top_grad,
                calibration_parent_distance=weighted_state_distance(calibration['carrier'],oldcal,calibration['q'])))
            save_resumable_checkpoint(dest/'checkpoint.pkl',jax.device_get(current),optimizer_state=jax.device_get((m,v)),
                completed_steps=step+1,next_sample_index=0,run_identity=identity,history=history,
                metadata=dict(ledger=ledger,fixed_training_corpus=True,reference_fixed=True,
                              active_seconds=prior_seconds+perf_counter()-start))
            progress(f'step_{step+1}_complete',accepted_factor=accepted,history=history)
        bank=generate(current)
        psi=training['carrier']+values(bank,training['states'])/scale-reference
        final_train=float(np.sum(abs(psi)**2/training['q']*training['three_hz_per_site'])/
                          np.sum(abs(psi)**2/training['q']))
        save_parameter_artifact(dest/'parameters.json',current,identity=identity)
        replay=load_parameter_artifact(dest/'parameters.json',initial,expected_identity=identity)
        if pytree_sha256(replay)!=pytree_sha256(current):raise AssertionError('checkpoint mismatch')
        np.savez_compressed(dest/'values.npz',training_psi=psi,calibration_psi=oldcal)
        changes={key:float(np.sqrt(sum(np.sum((np.asarray(a)-np.asarray(b))**2) for a,b in
            zip(jax.tree.leaves(current[key]),jax.tree.leaves(initial[key]),strict=True)))) for key in current}
        atomic_json(dest/'results.json',dict(identity=identity,history=history,
            final_training_energy_3hz_per_site=final_train,parameter_changes_by_block=changes,
            parameters=sum(x.size for x in jax.tree.leaves(current)),checkpoint_exact=True,
            ledger=ledger,seconds=prior_seconds+perf_counter()-start,peak_rss_bytes=peak_rss_bytes(),
            parameter_sha256=pytree_sha256(current),fixed_endpoint=True))
    else:
        # Do not even create audit configurations before all endpoint artifacts exist.
        for name in spec['arms']:
            if not (output/name/'results.json').exists(): raise ValueError('all endpoints must be sealed')
        prepared=json.loads((output/'prepare/results.json').read_text())
        data, info=draw(spec,carrier,bonds,prepared['carrier_scale'],spec['audit_seed'],spec['audit_blocks'])
        ledger['carrier_density_connection_centers']+=len(data['states'])
        ledger['shared_carrier_H_centers']+=len(data['states'])
        ledger['carrier_determinants']+=info['carrier_determinants']
        scale=prepared['residual_scale']
        action=make_amplitude_heisenberg_action(lambda p,x: residual(p,x[None])[0]/scale,bonds,n_sites=36)
        raw=dict(data); allpsi=[data['carrier']]; allh=[data['carrier_h']]
        references={}
        def hvalues(bank,label):
            parts=[];hs=[]
            for i in range(0,len(data['states']),2):
                a,b=map(np.asarray,action(bank,data['states'][i:i+2]))
                parts.append(a);hs.append(b);ledger['residual_H_centers']+=len(a)
                progress(label+f'_H_{i+len(a)}')
            return np.concatenate(parts),np.concatenate(hs)
        for name in ('attention','scrambled'):
            references[name]=hvalues(generators[name](params[name]),name+'_reference')
        for name in spec['arms']:
            parameters=load_parameter_artifact(output/name/'parameters.json',params[name])
            d,hd=hvalues(generators[name](parameters),name+'_endpoint')
            rd,rhd=references['attention' if name=='rank_free' else name]
            allpsi.append(data['carrier']+d-rd);allh.append(data['carrier_h']+hd-rhd)
            raw[name+'_residual']=d;raw[name+'_Hresidual']=hd
        psi=np.column_stack(allpsi);hp=np.column_stack(allh)
        raw.update(psi=psi,hpsi=hp)
        for name,(d,h) in references.items():raw[name+'_reference']=d;raw[name+'_Hreference']=h
        np.savez_compressed(dest/'values.npz',**raw)
        report=paired_chain_audit(psi,hp,data['q'],data['groups'],n_sites=36,
            bootstrap_replicates=spec['bootstrap_replicates'],seed=spec['bootstrap_seed'],
            arm_names=['carrier']+spec['arms'])
        report.update(identity=identity,sampling=info,ledger=ledger,seconds=perf_counter()-start,
                      peak_rss_bytes=peak_rss_bytes(),untouched_fixed_endpoint=True)
        atomic_json(dest/'results.json',report)
    signal.alarm(0)
    progress('completed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('register','prepare','train','audit'))
    parser.add_argument('--arm',choices=('attention','rank_free','scrambled'));args=parser.parse_args()
    if args.stage=='register':
        spec=json.loads(CONFIG.read_text()); path=ROOT/spec['output']/'registration.json'
        if path.exists():raise FileExistsError('registration already sealed')
        atomic_json(path,registration(spec));print(path)
    else:
        if args.stage=='train' and args.arm is None:parser.error('train requires --arm')
        run(args.stage,args.arm)


if __name__=='__main__':main()
