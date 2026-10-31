"""Fixed-endpoint safety gate before the bounded triangular CT continuation.

Uses a frozen parent Born sample for a correctly reweighted FULL-H candidate
energy, and fresh candidate Born chains. Reweighting parent local energies
alone would not be a candidate-energy estimator and is not used here.
No parameters, optimizer, training RNG, or training chains are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,mixed_clebsch_bank,make_mixed_projected_amplitude
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains,run_exchange_sweeps

ROOT=Path(__file__).resolve().parents[3]
jax.config.update('jax_enable_x64',True)


def gate_failures(history: list[dict],replay: dict,born: dict,correlated: dict) -> list[str]:
    """Preregistered broad SAFETY checks, not significance or winner selection."""
    failures=[]
    if len(history)!=2:failures.append('pilot must finish exactly two updates')
    if not any(h['accepted'] for h in history):failures.append('all pilot updates rejected')
    for h in history:
        values=[h[k] for k in ('FS_sine','ESS_fraction','energy_before_update_per_site',
            'variance_before_update_per_site','top_one_percent_residual_weight')]
        if not np.isfinite(values).all():failures.append('nonfinite pilot training statistics')
        if h['FS_sine']>.025+1e-12 or h['ESS_fraction']<.95-1e-12:
            failures.append('pilot trust-region failure')
        if h['top_one_percent_residual_weight']>=.9:failures.append('pilot energy tails dominate')
    if not replay.get('checkpoint_rng_sampler_replay_exact') or not replay.get('checkpoint_optimizer_exact'):
        failures.append('pilot optimizer/RNG/sampler replay failure')
    numbers=[born[k] for k in ('energy_per_site','energy_chain_SE','variance_per_site')]
    numbers += [correlated[k] for k in ('energy_per_site','energy_chain_SE','weight_ESS_fraction')]
    if not np.isfinite(numbers).all():failures.append('nonfinite fresh/reweighted probe')
    if correlated['weight_ESS_fraction']<.25:failures.append('parent reweighting no longer supported')
    tolerance=max(.15,10*np.hypot(born['energy_chain_SE'],correlated['energy_chain_SE']))
    if abs(born['energy_per_site']-correlated['energy_per_site'])>tolerance:
        failures.append('fresh Born versus full-H reweighted candidate energy mismatch')
    reference=max([h['variance_before_update_per_site'] for h in history]+[1e-3])
    if born['variance_per_site']>100*reference:failures.append('fresh second moment jumped by two orders of magnitude')
    return failures


def run_gate(spec: dict) -> dict:
    """Audit frozen pilot endpoint and leave the scientific training state intact."""
    if jax.default_backend()!='gpu':raise RuntimeError('allocated GPU required')
    output=Path(spec['output']);cp=output/'checkpoint.pkl';before=file_sha256(cp)
    registration=json.loads((output/'registration.json').read_text())
    if registration['config']!=spec or registration['sha256']!=canonical_sha256({
            k:v for k,v in registration.items() if k!='sha256'}):
        raise ValueError('pilot registration mismatch')
    for name,digest in registration['sources'].items():
        if file_sha256(ROOT/name)!=digest:raise ValueError('pilot source changed: '+name)
    saved=load_checkpoint(cp);meta=saved['metadata'];p=saved['params']
    if (meta['completed_steps']!=2 or meta['registration_sha256']!=registration['sha256']
            or meta['params_sha256']!=pytree_sha256(p)
            or meta['optimizer_sha256']!=pytree_sha256(saved['optimizer_state'])
            or int(saved['optimizer_state'][0])!=meta['optimizer_step']
            or meta['sampler_rng_sha256']!=pytree_sha256((jnp.asarray(meta['rng']),jnp.asarray(meta['sampler_states'])))):
        raise ValueError('pilot checkpoint parameter/optimizer/RNG mismatch')
    result=json.loads((output/'results.json').read_text())
    if result['completed_steps']!=2 or result['registration_sha256']!=registration['sha256']:
        raise ValueError('pilot result mismatch')
    cfg=MixedClebschConfig(**spec['model']);n=cfg.length**2
    images=np.asarray(triangular_hnf_gamma_a1_group(((cfg.length,0),(0,cfg.length))),np.int32)
    amp=make_mixed_projected_amplitude(cfg,images,chunk=spec['projection_chunk'])
    bank=mixed_clebsch_bank(p,cfg,spec['arm'])
    bonds=triangular_j1j2_bonds(cfg.length,cfg.length,j2=spec['j2_over_j1'],periodic=True)
    h=make_amplitude_heisenberg_action(amp,bonds,n_sites=n)
    @jax.jit
    def action(b,rows):
        a,hp=jax.lax.map(lambda x:h(b,x),rows.reshape(-1,2,n))
        return a.reshape(-1),hp.reshape(-1)
    parent=Path(spec['trained_parent']['directory'])
    parent_path=parent/'checkpoint.pkl'
    if file_sha256(parent_path)!=spec['trained_parent']['checkpoint_sha256']:
        raise ValueError('frozen parent changed')
    parent_p=load_checkpoint(parent_path)['params']
    parent_bank=mixed_clebsch_bank(parent_p,cfg,'ternary_c3')
    archive=parent/'terminal_audit.npz'
    with np.load(archive) as data:
        old_rows=jnp.asarray(data['bits'][:128]);old_labels=np.asarray(data['chain_labels'][:128])
    old_amp=jax.jit(jax.vmap(amp,in_axes=(None,0)))(parent_bank,old_rows)
    new_amp,new_h=action(bank,old_rows)
    ratios=np.asarray(new_amp/old_amp);weights=abs(ratios)**2
    local=np.asarray(new_h/new_amp)/n
    if not np.isfinite(ratios).all() or not np.isfinite(local).all() or np.any(np.asarray(old_amp)==0):
        raise FloatingPointError('nonfinite correlated full-H probe')
    numerator=weights*local;mean=float(numerator.sum()/weights.sum())
    leave=np.asarray([numerator[old_labels!=c].sum()/weights[old_labels!=c].sum() for c in np.unique(old_labels)])
    se=float(np.sqrt((len(leave)-1)/len(leave)*np.sum((leave-leave.mean())**2)))
    correlated=dict(energy_per_site=mean,energy_chain_SE=se,
        weight_ESS_fraction=float(weights.sum()**2/(len(weights)*np.sum(weights**2))),
        samples=len(weights),parent_archive_sha256=file_sha256(archive),
        estimator='sum |psi_new/psi_parent|^2*(H psi_new/psi_new) / sum weights; full H of candidate')
    edges=[(a,b) for a,b,_ in triangular_j1j2_bonds(cfg.length,cfg.length,j2=0.,periodic=True)]
    @jax.jit(static_argnames=('proposals',))
    def sample(key,states,proposals):
        return run_exchange_sweeps(bank,key,states,amplitude_fn=amp,particle_kind='spin',sweeps=proposals,exchange_edges=edges)
    seed=2026101604
    key,ki=jax.random.split(jax.random.PRNGKey(seed))
    states=initialize_spin_exchange_chains(ki,n_chains=32,n_sites=n,n_up=n//2)
    states,_,key,_=sample(key,states,proposals=720)
    batches=[]
    for _ in range(4):
        states,_,key,_=sample(key,states,proposals=36);batches.append(states)
    rows=jnp.stack(batches).reshape(-1,n);a,hp=action(bank,rows)
    e=np.asarray(hp/a).reshape(4,32)
    born=dict(energy_per_site=float(e.mean()/n),energy_chain_SE=float(e.mean(0).std(ddof=1)/np.sqrt(32)/n),
        variance_per_site=float(e.var()/n),samples=128,chains=32,seed=seed,
        independent_of_training=True,mixing_certified=False)
    failures=gate_failures(meta['history'],result,born,correlated)
    if file_sha256(cp)!=before:raise ArithmeticError('gate modified the training checkpoint')
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    gate=dict(status='passed' if not failures else 'failed',failures=failures,
        pilot_checkpoint_sha256=before,params_sha256=meta['params_sha256'],
        pilot_history=meta['history'],correlated=correlated,fresh_born=born,
        checkpoint_unmodified=True,optimizer_rng_sampler_replay=result,
        pilot_registration_sha256=registration['sha256'],sources=sources,
        interpretation='Broad numerical safety only; not an energy improvement or mixing certificate.')
    np.savez_compressed(output/'pilot_gate.npz',parent_rows=np.asarray(old_rows),parent_chain_labels=old_labels,
        parent_amplitude=np.asarray(old_amp),candidate_amplitude=np.asarray(new_amp),candidate_Hpsi=np.asarray(new_h),
        fresh_rows=np.asarray(rows),fresh_local_energies=e)
    atomic_json(output/'pilot_gate.json',gate)
    return gate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();gate=run_gate(json.loads(args.config.read_text()))
    print(json.dumps(gate),flush=True)
    if gate['status']!='passed':raise SystemExit('pilot safety gate failed; continuation not released')
