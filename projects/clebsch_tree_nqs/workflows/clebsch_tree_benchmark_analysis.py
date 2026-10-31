"""Replay saved ClebschTree audits and checkpoints, without training/H calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
jax.config.update('jax_enable_x64', True)


def triangle_summary(folder: Path) -> dict:
    """Check all fixed endpoints, raw chain errors and optimizer/RNG identities."""
    records=[];initial_energies=[];largest=0.
    for arm in ('binary_free','ternary_free','ternary_c3'):
        path=folder/'train20'/arm
        reg=json.loads((path/'registration.json').read_text())
        if canonical_sha256({k:v for k,v in reg.items() if k!='sha256'})!=reg['sha256']:
            raise ValueError('triangle registration hash mismatch')
        result=json.loads((path/'results.json').read_text());cp=load_checkpoint(path/'checkpoint.pkl')
        meta=cp['metadata']
        if (result['completed_steps']!=20 or result['status']!='fixed_steps_complete'
            or result['registration_sha256']!=reg['sha256']
            or meta['registration_sha256']!=reg['sha256']
            or meta['params_sha256']!=pytree_sha256(cp['params'])
            or meta['optimizer_sha256']!=pytree_sha256(cp['optimizer_state'])
            or meta['sampler_rng_sha256']!=pytree_sha256((meta['rng'],meta['sampler_states']))):
            raise ValueError('triangle trained checkpoint mismatch')
        audits={}
        for name in ('initial_audit','terminal_audit'):
            a=json.loads((path/(name+'.json')).read_text())
            if a['registration_sha256']!=reg['sha256']:raise ValueError('audit registration mismatch')
            with np.load(path/(name+'.npz')) as z:
                e=z['local_energies'].reshape(-1,a['chains']);labels=z['chain_labels']
                if not np.array_equal(labels,np.tile(np.arange(a['chains']),len(e))):
                    raise ValueError('audit chain order changed')
            leave=np.array([np.delete(e,c,axis=1).var()/36 for c in range(a['chains'])])
            replay=dict(energy_per_site=float(e.mean()/36),
                energy_chain_SE=float(e.mean(0).std(ddof=1)/np.sqrt(a['chains'])/36),
                variance_per_site=float(e.var()/36),
                variance_chain_jackknife_SE=float(np.sqrt((len(leave)-1)/len(leave)*np.sum((leave-leave.mean())**2))))
            error=max(abs(a[k]-v) for k,v in replay.items());largest=max(largest,error)
            if error>2e-13:raise ValueError('audit raw moment replay failed')
            audits[name]=a
        if audits['terminal_audit']['params_sha256']!=meta['params_sha256']:
            raise ValueError('terminal audit is not the trained checkpoint')
        initial_energies.append(audits['initial_audit']['energy_per_site'])
        hist=result['history'];terminal=audits['terminal_audit'];initial=audits['initial_audit']
        records.append(dict(arm=arm,parameters=result['parameter_count'],audits=audits,
            accepted_updates=sum(h['accepted'] for h in hist),full_H_training_calls=sum(h['full_H_calls'] for h in hist),
            maximum_FS_sine=max(h['FS_sine'] for h in hist),
            maximum_SR_residual=max(h['relative_SR_residual'] for h in hist),
            energy_change_per_site=terminal['energy_per_site']-initial['energy_per_site'],
            energy_change_independent_chain_SE=float(np.hypot(terminal['energy_chain_SE'],initial['energy_chain_SE'])),
            checkpoint_optimizer_and_RNG_identity_verified=True))
    b,f,c=records
    contrast=c['audits']['terminal_audit']['energy_per_site']-f['audits']['terminal_audit']['energy_per_site']
    se=float(np.hypot(c['audits']['terminal_audit']['energy_chain_SE'],f['audits']['terminal_audit']['energy_chain_SE']))
    return dict(records=records,maximum_raw_audit_replay_discrepancy=largest,
        initial_audit_energy_spread=max(initial_energies)-min(initial_energies),
        C3_minus_free_terminal_energy=contrast,C3_minus_free_independent_SE=se,
        interpretation='Real bounded optimization completed; C3 vs ternary-free energy difference unresolved. All endpoints are far from competitive; neither symmetry benefit nor an expressivity no-go established by20 updates.',
        phase_or_variational_win_claim=False)


def analyze(output: Path) -> dict:
    """Preserve terminal audits separately from a running square snapshot."""
    triangle=triangle_summary(ROOT/'projects/clebsch_tree_nqs/results/triangle_c3_architecture_20261015')
    base=output/'square'
    audit=json.loads((base/'audit100/audit.json').read_text())
    with np.load(base/'audit100/local_energies.npz') as z:
        keys=list(z.files)
    snapshot_path=base/'checkpoint_metadata_snapshot.json'
    snapshot=json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    progress=json.loads((base/'remat_snapshot/progress.json').read_text())
    hist=snapshot['history'][-20:] if snapshot else [progress['last']]
    seconds=np.mean([h['seconds'] for h in hist])
    square=dict(independent_audit100=audit['statistics'],running_progress=progress,
        raw_audit_array_keys=keys,training_statistics_NOT_endpoint_audit=True,
        last20_training_energy_mean=float(np.mean([h['energy_before_update_per_site'] for h in hist])),
        last20_training_variance_mean=float(np.mean([h['variance_before_update_per_site'] for h in hist])),
        last20_seconds_per_update=float(seconds),
        last20_H_fraction=float(sum(h['H_seconds'] for h in hist)/sum(h['seconds'] for h in hist)),
        throughput_extrapolated_hours_to1000=float((1000-progress['completed_steps'])*seconds/3600),
        throughput_extrapolation_not_a_resource_request=True,
        paper_target_per_site=-31.90850/64,
        paper_reference='https://arxiv.org/html/2104.14869#A1.T1',
        limitation='100-step audit remains weak; live later means are not independently audited. The general Appendix-B schedule and FS safeguards are not an exact reproduction of the1024-sample1000-step Fig3 sweep. No new training submitted here.')
    previous=json.loads((ROOT/'projects/clebsch_tree_nqs/results/vieijra_nys_reproduction_20261014/comparison.json').read_text())
    square['previous_exact4x4_validation_NOT_new_optimization']=previous['endpoints'][-1]
    report=dict(triangle=triangle,square=square,new_training=0,new_H_calls=0)
    atomic_json(output/'comparison.json',report)
    print(json.dumps(dict(triangle_replay_error=triangle['maximum_raw_audit_replay_discrepancy'],
        triangle_contrast=(triangle['C3_minus_free_terminal_energy'],triangle['C3_minus_free_independent_SE']),
        square_audit100=audit['statistics']['energy_per_site'],square_training_step=progress['completed_steps']),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'projects/clebsch_tree_nqs/results/clebsch_benchmark_review_20261015')
    analyze(parser.parse_args().output)
