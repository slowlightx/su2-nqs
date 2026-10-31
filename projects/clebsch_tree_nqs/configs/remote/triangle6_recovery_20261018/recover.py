"""Repair only continuation provenance; reuse the unchanged triangular trainer.

For each stage, archive the complete audited parent in a new immutable directory
and measure a fresh target-H certificate on THAT checkpoint. The original source,
parent, preflight and guard are never overwritten. No architecture or optimizer
change is introduced; the scientific endpoints remain 152 and 200.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, make_mixed_projected_amplitude, make_mixed_clebsch_amplitude,
)
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import (
    ROOT, checked_arity_endpoint, checked_target_H_cost_certificate, train,
)

jax.config.update('jax_enable_x64', True)


def snapshot_parent(source: Path, destination: Path, expected_step: int) -> dict:
    """Copy sealed completed state/optimizer/RNG/audit, never a live checkpoint."""
    registration = json.loads((source/'registration.json').read_text())
    result = json.loads((source/'results.json').read_text())
    audit = json.loads((source/'terminal_audit.json').read_text())
    seal = registration['sha256']
    if (seal != canonical_sha256({k:v for k,v in registration.items() if k!='sha256'})
            or result.get('status') != 'fixed_steps_complete'
            or result.get('completed_steps') != expected_step
            or result.get('registration_sha256') != seal
            or result.get('terminal_audit') != audit
            or audit.get('registration_sha256') != seal
            or audit.get('independent_from_training') is not True
            or result.get('checkpoint_rng_sampler_replay_exact') is not True
            or result.get('checkpoint_optimizer_exact') is not True):
        raise ValueError('complete independently audited and replayed parent required')
    request = dict(directory=str(destination), registration_sha256=seal,
        completed_steps=expected_step, checkpoint_sha256=file_sha256(source/'checkpoint.pkl'),
        params_sha256=audit['params_sha256'], audit_sha256=file_sha256(source/'terminal_audit.json'))
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('registration.json','results.json','checkpoint.pkl','terminal_audit.json'):
        target = destination/name
        if target.exists():
            if file_sha256(target) != file_sha256(source/name):
                raise ValueError('immutable archived parent changed: '+name)
        else:
            shutil.copy2(source/name, target)
    note = dict(source=str(source), immutable_snapshot=str(destination),
        original_preflight_not_relabelled=True, parent=request)
    old = source/'preflight.json'
    if old.exists():
        target = destination/'inherited_preflight.json'
        if target.exists() and file_sha256(target) != file_sha256(old):
            raise ValueError('inherited historical preflight changed')
        if not target.exists():shutil.copy2(old, target)
        note['inherited_preflight_sha256'] = file_sha256(old)
    target = destination/'snapshot.json'
    if target.exists() and json.loads(target.read_text()) != note:
        raise ValueError('parent snapshot provenance changed')
    if not target.exists():atomic_json(target,note)
    return request


def certify(spec: dict, parent: Path) -> dict:
    """Independently test full432-image/H action on the actual archived parent."""
    cfg = MixedClebschConfig(**spec['model'])
    _, _, endpoint = checked_arity_endpoint(cfg,spec)
    saved = endpoint['saved']; p = saved['params']
    expected = spec['arity_endpoint_parent']
    identity = dict(parent=expected, all_execution_sources_checked=True,
        model=spec['model'], j2_over_j1=spec['j2_over_j1'])
    path = parent/'preflight.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('fresh_actual_endpoint_identity') != identity:
            raise ValueError('fresh certificate checkpoint identity changed')
        return existing
    bank=arity_clebsch_bank(p,cfg,spec['arm'],spec['arity_comparison']['binary_rank'])
    images=np.asarray(triangular_hnf_gamma_a1_group(((cfg.length,0),(0,cfg.length))),np.int32)
    projected=make_mixed_projected_amplitude(cfg,images,chunk=spec['projection_chunk'])
    evaluate=jax.jit(jax.vmap(projected,in_axes=(None,0)))
    rows=jnp.asarray(saved['metadata']['sampler_states'][:2])
    bonds=triangular_j1j2_bonds(cfg.length,cfg.length,j2=spec['j2_over_j1'],periodic=True)
    full_h=jax.jit(make_amplitude_heisenberg_action(projected,bonds,n_sites=cfg.length**2))
    begin=perf_counter(); values=np.asarray(evaluate(bank,rows))
    psi,hpsi=jax.block_until_ready(full_h(bank,rows))
    graph=SzHeisenbergGraph(cfg.length**2,tuple(bonds),n_up=cfg.length**2//2)
    oracle=[]
    for row in np.asarray(rows):
        connected,elements=graph.get_conn(tuple(row))
        # Unlike full_h, assemble connectivity independently, then query every
        # complete projected physical amplitude. No ratio or image pruning.
        oracle.append(np.dot(elements,np.asarray(evaluate(bank,jnp.asarray(connected)))))
    oracle=np.asarray(oracle)
    norm=max(np.linalg.norm(oracle),np.linalg.norm(np.asarray(hpsi)),1e-100)
    h_error=float(np.linalg.norm(np.asarray(hpsi)-oracle)/norm)
    transformed=np.asarray(evaluate(bank,rows[:,images[37]]))
    a1=float(np.linalg.norm(transformed-values)/max(np.linalg.norm(values),1e-100))
    bra=np.asarray(rows[0]).copy();bra[np.flatnonzero(bra==0)[0]]=1
    n=cfg.length**2
    lowered=np.tile(bra,(n//2+1,1));lowered[np.arange(n//2+1),np.flatnonzero(bra)]=0
    raw=make_mixed_clebsch_amplitude(cfg)
    ward=np.asarray(jax.jit(jax.vmap(raw,in_axes=(None,0)))(bank,jnp.asarray(lowered)))
    ward_error=float(abs(ward.sum())/max(np.linalg.norm(ward),1e-100))
    bank_fn=lambda z:arity_clebsch_bank(z,cfg,spec['arm'],spec['arity_comparison']['binary_rank'])
    grad=jax.jit(jax.grad(lambda z:jnp.log(jnp.abs(projected(bank_fn(z),rows[0])))))(p)
    finite=(np.isfinite(values).all() and np.all(values!=0) and np.linalg.norm(ward)>1e-100
        and np.isfinite(np.asarray(psi)).all()
        and np.isfinite(np.asarray(hpsi)).all() and np.isfinite(ward).all()
        and all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad)))
    checks=dict(target_H_oracle_relative_error=h_error,A1_relative_error=a1,
        S_plus_relative_error=ward_error,H_finite=bool(finite),gradient_finite=bool(finite))
    passed=finite and all(np.isfinite(e) and e<=2e-11 for e in (h_error,a1,ward_error))
    record=dict(status='passed' if passed else 'failed',
        registration_sha256=expected['registration_sha256'],validation=checks,
        target_H_independent_oracle=True,group_order=len(images),physical_bonds=len(bonds),
        fresh_actual_endpoint_identity=identity,full_H_centers=2,
        seconds=perf_counter()-begin,calibration_rows_are_not_Born_energy_samples=True,
        original_historical_preflight_preserved=True,new_training_updates=0)
    atomic_json(path,record)
    if not passed:raise ArithmeticError('fresh endpoint-H certificate failed')
    return record


def prepare(plan: dict, arm: str, stage: int, plan_sha256: str) -> dict:
    """Only provenance paths/statistics change; original physics and SR persist."""
    if arm not in plan['base_plans'] or stage not in (0,1):
        raise ValueError('unregistered arm/stage')
    original=plan['base_plans'][arm];path=ROOT/original['path']
    if file_sha256(path) != original['sha256']:raise ValueError('original plan changed')
    base=json.loads(path.read_text())
    for name,digest in base['execution_sources'].items():
        if file_sha256(ROOT/name)!=digest:raise ValueError('frozen numerical source changed: '+name)
    helper=Path(__file__).resolve()
    if file_sha256(helper) != plan['helper_sha256']:raise ValueError('recovery helper changed')
    target=Path(plan['output_root'])/arm
    source=Path(base['parent']['directory']) if stage==0 else target/'stage0'
    expected_step=104+48*stage
    source_reg=json.loads((source/'registration.json').read_text())
    if stage==0:
        if source_reg['sha256']!=base['parent']['registration_sha256']:
            raise ValueError('wrong originally registered104 parent')
    elif source_reg['config']!=json.loads((target/'stage0_config.json').read_text()):
        raise ValueError('parent is not this exact preceding recovery stage')
    request=snapshot_parent(source,target/('parent'+str(expected_step)),expected_step)
    spec=deepcopy(base['template_config'])
    # The obsolete certificate is not deleted to bypass a guard: it is
    # replaced below by a freshly measured certificate for the actual parent.
    spec.pop('target_H_cost_certificate',None)
    spec.update(arity_endpoint_parent=request,endpoint_protocol='same_H_full_state_continuation',
        additional_steps=48,steps=expected_step+48,samples=1024,audit_samples=(8192,16384)[stage],
        audit_chains=128,audit_burn_proposals=1440,terminal_audit_seed=plan['audit_seeds'][arm][stage],
        output=str(target/('stage'+str(stage))),recovery_protocol=dict(
            plan_sha256=plan_sha256,helper_sha256=plan['helper_sha256'],stage=stage,
            fresh_parent_H_certificate=True,core_sources_unchanged=True,
            optimizer_sampler_RNG_reset=False,original_parent_preserved=True))
    certify(spec,Path(request['directory']))
    cert=Path(request['directory'])/'preflight.json'
    spec['target_H_cost_certificate']=dict(directory=request['directory'],
        registration_sha256=request['registration_sha256'],preflight_sha256=file_sha256(cert))
    if not checked_target_H_cost_certificate(spec):raise AssertionError('strict certificate guard bypassed')
    # Run the original full-state loader again after installing the certificate.
    checked_arity_endpoint(MixedClebschConfig(**spec['model']),spec)
    frozen=target/('stage'+str(stage)+'_config.json')
    if frozen.exists() and json.loads(frozen.read_text())!=spec:raise ValueError('frozen recovery stage changed')
    if not frozen.exists():atomic_json(frozen,spec)
    print(json.dumps(dict(preflight='passed',arm=arm,stage=stage,
        actual_parent_steps=expected_step,endpoint=spec['steps'],certificate_sha256=file_sha256(cert))),flush=True)
    return spec


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--arm',choices=('binary_truncated','ternary_joint'),required=True)
    parser.add_argument('--stage',type=int,choices=(0,1),required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if file_sha256(args.plan)!=args.plan_sha256:raise ValueError('recovery plan changed')
    if jax.default_backend()!='gpu':raise RuntimeError('fresh actual-state preflight requires allocated A100')
    plan=json.loads(args.plan.read_text());spec=prepare(plan,args.arm,args.stage,args.plan_sha256)
    if not args.prepare_only:
        result=train(spec,Path(spec['output']),session_seconds=plan['training_session_seconds'])
        if result['status']!='fixed_steps_complete':
            raise RuntimeError('safe checkpoint preserved; no incomplete endpoint releases a later stage')
        print(json.dumps(dict(status=result['status'],completed_steps=result['completed_steps'],
            terminal_audit=result['terminal_audit'])),flush=True)
