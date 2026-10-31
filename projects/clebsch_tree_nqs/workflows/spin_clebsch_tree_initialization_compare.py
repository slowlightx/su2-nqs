"""Bounded cold-versus-small-system initialization control using existing VMC.

No new optimizer or neural head: both arms use the same strict-S0/full-A1
ClebschTree, sample/update budgets and independent final energy audit.  A
pilot records actual cost before the fixed continuation is released.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.models.literature_clebsch_tree import (
    LiteratureClebschTreeConfig, init_literature_clebsch_tree,
    make_clebsch_amplitudes, make_clebsch_features, clebsch_layers,
)
from su2_nqs.models.clebsch_tree_transfer import transfer_clebsch_parameters, calibrate_clebsch_parameters
from su2_nqs.lattices.geometry_tree import square_geometry_tree
from su2_nqs.benchmarks.clebsch_tree_exact import graph_hamiltonian_coo, coo_matvec, physical_permutation_index_table, reynolds_project_state
from .spin import _groups, _bonds
from .spin_clebsch_tree_sampled import ROOT, cost
from .spin_clebsch_tree_training import train
from .spin_clebsch_tree_audit import audit

jax.config.update('jax_enable_x64', True)


def diagnose_parent(parent: Path, output: Path) -> dict:
    """Exact cached-small raw/projected moments, plus size-transfer coverage.

    Evaluate the existing physical support and sparse H; never diagonalize H.
    Intermediate feature norms are diagnostic activations, NOT Born sector
    probabilities or a measured amount of inter-cluster entanglement.
    """
    record=json.loads((parent/'registration.json').read_text())
    saved=load_checkpoint(parent/'checkpoint.pkl'); p=saved['params']
    spec=record['config']
    if (spec['rows'],spec['columns']) != (4,4):
        raise ValueError('this diagnostic only reuses the saved square4 reference')
    if saved['metadata']['registration_sha256'] != record['sha256']:
        raise ValueError('saved small state and registration disagree')
    tree,order=square_geometry_tree(4,4)
    cfg=LiteratureClebschTreeConfig(tree,multiplicity_cap=spec['multiplicity_cap'],maximum_q=spec['maximum_q'])
    r,c,v,basis=graph_hamiltonian_coo(16,0,_bonds('square',4,4,.5))
    raw=np.asarray(jax.jit(make_clebsch_amplitudes(cfg))(p,np.asarray(basis)[:,order]))
    permutations=physical_permutation_index_table(basis,_groups('square',4,4)['gamma_a1'])
    projected=np.asarray(reynolds_project_state(raw,permutations))
    moments=[]
    for psi in (raw,projected):
        hpsi=np.asarray(coo_matvec(r,c,v,psi)); norm=float(np.vdot(psi,psi).real)
        energy=float(np.vdot(psi,hpsi).real/norm)
        moments.append(dict(energy_per_site=energy/16,
            variance_per_site=float(np.vdot(hpsi,hpsi).real/norm-energy**2)/16,norm=norm))
    transfers=[]
    for length,width,high in ((6,8,None),(8,16,8)):
        target_tree,target_order=square_geometry_tree(length,length)
        target=LiteratureClebschTreeConfig(target_tree,multiplicity_cap=width,maximum_q=8,high_spin_cap=high)
        rng=np.random.default_rng(2026101782)
        bits=np.asarray([rng.permutation([0]*(length**2//2)+[1]*(length**2//2)) for _ in range(1000)])
        random,initial=init_literature_clebsch_tree(jax.random.PRNGKey(2026101781),target,bits[:,target_order])
        moved,report=transfer_clebsch_parameters(cfg,p,target,random)
        moved,scale=calibrate_clebsch_parameters(target,moved,bits[:,target_order])
        features=jax.jit(make_clebsch_features(target))(moved,bits[:16,target_order])
        finite=all(np.isfinite(x).all() for x in jax.tree.leaves(moved))
        activity=[]
        for layer,values in zip(clebsch_layers(target),features,strict=True):
            values=np.asarray(values)
            activity.append(dict(child_bank_parameters=layer.parameter_size,
                positive_q_with_nonzero_features=[q for q,t in enumerate(layer.widths)
                    if q>0 and t and np.any(values[:,:,q,:t] != 0)]))
        amplitudes=np.asarray(jax.jit(make_clebsch_amplitudes(target))(moved,bits[:16,target_order]))
        transfers.append(dict(length=length,parameters=initial['parameter_count'],
            copied_scalar_parameters=report['copied_scalar_parameters'],
            copied_fraction=report['copied_scalar_parameters']/initial['parameter_count'],
            bank_transfer=report,positive_q_feature_support=activity,
            feature_activity_is_not_Born_sector_weight=True,all_parameters_finite=finite,
            nonzero_raw_probe_amplitudes=int(np.count_nonzero(amplitudes)),probe_rows=16,
            recalibration=scale))
    stored=json.loads((parent/'results.json').read_text())
    result=dict(parent=str(parent),checkpoint_sha256=file_sha256(parent/'checkpoint.pkl'),
        parameters_sha256=pytree_sha256(p),raw=moments[0],full_A1=moments[1],
        exact_projected_replay_energy_absolute_error=abs(moments[1]['energy_per_site']-stored['energy_per_site']),
        A1_projection_probability=moments[1]['norm']/moments[0]['norm'],
        new_eigensolve=False,support_size=len(basis),transfers=transfers,
        interpretation='Raw reduced-bank transfer does not embed the high-quality locally A1-projected physical state; compare initial target quality before long training.')
    atomic_json(output,result)
    return result


def run(plan: dict, arm: str, stage: str) -> dict:
    """Execute one pre-registered initialization arm; never extend its budget."""
    if plan.get('status','active') != 'active':
        raise ValueError('this weak size-transfer training plan was withheld before submission')
    if arm not in plan['arms'] or stage not in ('pilot', 'continue'):
        raise ValueError('unregistered arm/stage')
    base = ROOT / Path(plan['output']) / arm
    base.mkdir(parents=True, exist_ok=True)
    seal = canonical_sha256(plan)
    registration = dict(plan=plan, sha256=seal)
    if (base/'plan.json').exists():
        if json.loads((base/'plan.json').read_text()) != registration:
            raise ValueError('fixed initialization comparison plan changed')
    else:
        atomic_json(base/'plan.json', registration)
    spec = dict(plan['physical'], **plan['arms'][arm], output=str(base/'cost'))
    if stage == 'pilot' and not (base/'cost/results.json').exists():
        cost(spec, base/'cost')
    gate = json.loads((base/'cost/results.json').read_text())
    if gate['status'] != 'passed' or gate['config'] != spec:
        raise ValueError('actual initialization has not passed the cost/invariant gate')
    training = dict(spec, **plan['training'], output=str(base/'training'),
        cost_parent=dict(directory=str(base/'cost'),
            checkpoint_sha256=gate['checkpoint_sha256'],
            registration_sha256=gate['registration_sha256']))
    if (base/'training_config.json').exists():
        if json.loads((base/'training_config.json').read_text()) != training:
            raise ValueError('training configuration changed')
    else:
        atomic_json(base/'training_config.json', training)
    if stage == 'pilot':
        result = train(training, base/'training', plan['pilot_steps'], plan['pilot_training_seconds'])
        rows = result['history']
        if (result['completed_steps'] != plan['pilot_steps'] or not all(
                r['accepted'] and r['relative_SR_residual'] < 1e-7
                and np.isfinite(r['energy_before_update_per_site'])
                and r['FS_sine'] <= training['maximum_FS_sine'] * (1+1e-10)
                for r in rows)):
            raise ArithmeticError('initialization pilot did not finish safely')
        # Retain the pilot boundary even if the live checkpoint advances.
        shutil.copy2(base/'training/checkpoint.pkl', base/'pilot_checkpoint.pkl')
        warm = rows[-1]['seconds']
        estimate = (training['steps'] - plan['pilot_steps']) * warm
        estimate += (plan['audit']['streams'] * plan['audit']['chains_per_stream']
            * plan['audit']['snapshots_per_stream'] / training['warm_samples']) * warm
        estimate += plan['continuation_overhead_seconds']
        result = dict(status='pilot_passed', arm=arm, plan_sha256=seal,
            pilot_steps=plan['pilot_steps'], warm_step_seconds=warm,
            estimated_continuation_and_full_audit_seconds=estimate,
            time_release_permitted=estimate <= plan['continuation_seconds'],
            pilot_checkpoint_sha256=file_sha256(base/'pilot_checkpoint.pkl'),
            training_moments_not_independent_audit=True,
            initial_parameters_sha256=pytree_sha256(load_checkpoint(base/'cost/checkpoint.pkl')['params']))
        atomic_json(base/'pilot.json', result)
        return result
    pilots = [json.loads((ROOT/Path(plan['output'])/name/'pilot.json').read_text())
              for name in plan['arms']]
    if not all(p['plan_sha256'] == seal and p['time_release_permitted'] for p in pilots):
        raise RuntimeError('both matched pilots must fit the full pre-registered audit budget')
    midpoint = plan['resume_check_step']
    half_session=(plan['continuation_seconds']-plan['continuation_overhead_seconds'])/2
    first = train(training, base/'training', midpoint, half_session)
    restored = load_checkpoint(base/'training/checkpoint.pkl')
    if (first['completed_steps'] != midpoint or restored['metadata']['completed_steps'] != midpoint
            or restored['metadata']['params_sha256'] != pytree_sha256(restored['params'])
            or restored['metadata']['optimizer_step'] != int(restored['optimizer_state'][0])):
        raise ArithmeticError('midpoint optimizer/parameters/RNG checkpoint is not resumable')
    rng_hash = canonical_sha256(np.asarray(restored['metadata']['rng']).tolist())
    final = train(training, base/'training', training['steps'], half_session)
    if final['completed_steps'] != training['steps']:
        raise RuntimeError('fixed endpoint not reached; retain checkpoint, do not shrink the audit')
    independent = dict(plan['audit'], training_directory=str(base/'training'),
        endpoint_step=training['steps'], output=str(base/'audit'))
    measured = audit(independent, base/'audit')
    result = dict(status='complete', arm=arm, plan_sha256=seal,
        full_updates=training['steps'], midpoint_step=midpoint,
        midpoint_rng_sha256=rng_hash, independent_statistics=measured['statistics'],
        audit_checkpoint_sha256=measured['checkpoint_sha256'],
        initial_state_transfer_is_not_a_physical_wavefunction_embedding=True)
    atomic_json(base/'comparison_endpoint.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--arm')
    parser.add_argument('--stage', choices=('pilot', 'continue', 'diagnose'), required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.stage == 'diagnose':
        if args.parent is None or args.output is None:parser.error('diagnose needs --parent and --output')
        result=diagnose_parent(args.parent,args.output)
    else:
        if args.plan is None or args.arm is None:parser.error('training needs --plan and --arm')
        result=run(json.loads(args.plan.read_text()),args.arm,args.stage)
    print(json.dumps(result,indent=2),flush=True)
