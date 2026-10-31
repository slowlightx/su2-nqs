"""Post-calibration, PRE-AUDIT bounded probe of an energy-compatible direction.

Original strict fold refusal remains sealed. No direction refit, new samples,
head or change of numerical sources. Probe uses the unused one-candidate H
allowance; evaluates on the SAME future parent experiment's untouched audit.
This is explicitly a diagnostic extension, not the original preregistration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256
from su2_nqs.frontier.provenance import file_sha256,numerical_runtime_identity,transitive_local_source_manifest
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact,load_parameter_artifact
from su2_nqs.vmc.carrier_protection import CarrierProtection,carrier_quality_gate,physical_distance
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step
from su2_nqs.vmc.cutset_training import paired_chain_audit
from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from .spin_finite_schur_reanchor import CONFIG,configuration,loaded,quality
from .spin_finite_schur_optimize import setup,full_products,check_resources
from .spin_finite_schur_direction import candidate_products
from .spin_finite_schur_generalization import trust_regions
from .spin_finite_schur_rayleigh import chain_regions
from .spin_schur_carrier_control import batched
from .spin_decoder_cost import ROOT,peak_rss_bytes


def registration(spec: dict,output: Path) -> dict:
    """Declare the diagnostic from seen calibration, before ANY new audit exists."""
    paths=[output/n for n in ('registration.json','prepared.json','fit.npz','geometry.npz','calibration.npz','calibration.json','filter.json')]
    for arm in ('network','free'):paths += [output/arm/n for n in ('proposal.json','direction.npz','products.npz')]
    row=dict(parent_registration=json.loads((output/'registration.json').read_text())['registration_sha256'],
        phase='post_calibration_pre_untouched_audit_diagnostic',config=spec,
        protocol='Only directions with E slopes negative on full/even/odd, full R slope negative and original refusal ONLY residual_descent_all_folds. No sign flip/refit. One fixed full-response bound alpha<=.5*(-dR_full)/response_full and .1;13 physical-FS probes cap all fit/geometry/calibration chains .01. Preserve original rejection. Free energy-uphill direction logically skipped. Same future128-record audit, not a new draw. No original commit or production release from this probe.',
        inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    row['registration_sha256']=canonical_sha256(row);return row


def build(spec: dict,reg: dict,output: Path,folder: Path) -> None:
    """One finite actual-parameter probe, no new derivative or fitting records."""
    if (output/'audit.npz').exists():raise ValueError('probe must be fixed BEFORE untouched audit')
    start=perf_counter();fit,geo,new=(loaded(output/(k+'.npz')) for k in ('fit','geometry','calibration'))
    prep=json.loads((output/'prepared.json').read_text());scale=prep['residual_scale'];_,initial,generators,projected,_,_,plan,evaluate=setup(spec)
    rows={}
    for arm in ('network','free'):
        original=json.loads((output/arm/'proposal.json').read_text());guard=original['trial']['eligibility'];full=guard['regions']['full']
        eligible=guard['rejected_by']==['residual_descent_all_folds'] and guard['checks']['energy_descent_all_folds'] and full['variance_slope_per_site']<0 and full['residual_response_per_site']>0
        trial=dict(raw_candidate_constructed=False,accepted=False,diagnostic_eligible=bool(eligible),original_gate_rejected=True)
        if eligible:
            saved=loaded(output/arm/'direction.npz');products=loaded(output/arm/'products.npz');d=saved['direction']
            flat,unravel=ravel_pytree(initial[arm]);generator=generators[arm]
            maximum=min(.1,.5*(-full['variance_slope_per_site'])/full['residual_response_per_site'])
            states=np.r_[fit['states'],geo['states'],new['states']];q=np.r_[fit['q'],geo['q'],new['q']];parent=np.r_[fit['carrier'],geo['carrier'],new['carrier']]
            reference=np.r_[fit['reference_values'],saved['geometry_reference_values'],products['reference_values']]
            def amplitude(alpha):return parent+scale*(batched(lambda x:projected(generator(unravel(flat-alpha*jnp.asarray(d))),x),states,8)-reference)
            match=matched_chain_state_step(amplitude,parent,q,trust_regions([fit,geo,new]),.01,maximum,12,relative_tolerance=.01)
            candidate=unravel(flat-match.alpha*jnp.asarray(d));centers=np.r_[fit['states'],new['states']]
            plans=[plan(jnp.asarray(centers[i:i+spec['H_microbatch']])) for i in range(0,len(centers),spec['H_microbatch'])]
            data=dict(carrier=np.r_[fit['carrier'],new['carrier']],hcarrier=np.r_[fit['hcarrier'],new['hcarrier']],
                reference_values=np.r_[fit['reference_values'],products['reference_values']],reference_H_total=np.r_[fit['reference_H_total'],products['reference_H_total']])
            p,hp,work=candidate_products(generator(candidate),data,plans,evaluate,scale,36)
            replay=float(np.max(abs(p-np.r_[match.amplitudes[:64],match.amplitudes[192:]])))
            if replay>2e-10*max(1.,float(np.max(abs(p)))):raise FloatingPointError('probe amplitude/H replay fails')
            gates=quality(new,p[64:],hp[64:],spec)
            destination=folder/arm;destination.mkdir()
            save_parameter_artifact(destination/'candidate.json',candidate,identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,kind='post_cal_pre_audit_diagnostic'))
            np.savez_compressed(destination/'control.npz',psi=p,hpsi=hp)
            trial.update(raw_candidate_constructed=True,matching=match.as_dict(),quality_gates=gates,
                full_only_response_maximum_alpha=maximum,all_fold_quality_pass=all(g['accepted'] for g in gates),
                full_calibration_quality_pass=gates[0]['accepted'],candidate_work=work,amplitude_H_replay_error=replay,
                original_gate_not_overridden=True,underfilled_FS_not_equal_distance=not match.search['matched'])
        rows[arm]=trial;check_resources(spec,start)
    atomic_json(folder/'proposals.json',dict(registration_sha256=reg['registration_sha256'],arms=rows,
        new_samples=0,new_forces=0,work_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes()))


def audit_probe(spec: dict,reg: dict,output: Path,folder: Path) -> None:
    """Evaluate the frozen probe on the parent's ONE already drawn untouched set."""
    if (folder/'results.json').exists():raise FileExistsError('diagnostic audit sealed')
    start=perf_counter();proposals=json.loads((folder/'proposals.json').read_text());audit=json.loads((output/'audit.json').read_text());data=loaded(output/'audit.npz')
    prep=json.loads((output/'prepared.json').read_text());n,initial,generators,_,_,_,plan,evaluate=setup(spec);c,hc=data['carrier'],36*data['hcarrier']
    f=audit['names'].index('filter');filter_p,filter_h=data['psi'][:,f],data['hpsi'][:,f]
    positions={tuple(map(int,x)):i for i,x in enumerate(data['states'])}
    def action(x):
        ids=[positions[tuple(map(int,y))] for y in np.asarray(x)];return c[ids],hc[ids]
    rows={};policy=CarrierProtection(**spec['protection'])
    for arm,trial in proposals['arms'].items():
        if not trial['raw_candidate_constructed']:continue
        params=load_parameter_artifact(folder/arm/'candidate.json',initial[arm])
        p,hp,work=full_products(generators[arm](params),generators['network'](initial['network']),data['states'],action,plan,evaluate,scale=prep['residual_scale'],microbatch=spec['H_microbatch'])
        gate=carrier_quality_gate(c,hc,c,hc,p,hp,data['q'],data['groups'],n,policy);gate['control_used_for_training']=False
        compare=carrier_quality_gate(filter_p,filter_h,c,hc,p,hp,data['q'],data['groups'],n,policy)
        distances={k:physical_distance(c[m],p[m],data['q'][m]) for k,m in chain_regions(data['groups']).items()}
        metrics=paired_chain_audit(np.c_[c,filter_p,p],np.c_[hc,filter_h,hp],data['q'],data['groups'],n,1024,spec['seeds']['bootstrap'],['C1','filter','diagnostic_neural'])
        np.savez_compressed(folder/arm/'audit.npz',psi=p,hpsi=hp)
        row=dict(C1_gate=gate,filter_comparison=compare,distances=distances,metrics=metrics,full_H_work=work,
            untouched_joint_quality_pass=bool(gate['accepted'] and max(v['fs'] for v in distances.values())<=.01*(1+1e-8)),
            extra_gain_over_filter=bool(compare['delta_energy_vs_carrier']+2*compare['paired_chain_jackknife_se']['energy_per_site']<0 and compare['delta_variance_vs_carrier']<=0),
            original_parent_commit=False,post_audit_refit=False,product_sha256=file_sha256(folder/arm/'audit.npz'))
        # Retain the attempted diagnostic candidate separately; original endpoint
        # remains exactly the parent, regardless of this observational result.
        save_gated_checkpoint(folder/arm/'checkpoint.pkl',params,optimizer_state={},run_identity=dict(registration_sha256=reg['registration_sha256'],arm=arm,stage='diagnostic'),
            decision=dict(raw_candidate_constructed=True,accepted=False,diagnostic_only=True))
        rows[arm]=row;check_resources(spec,start)
    atomic_json(folder/'results.json',dict(registration_sha256=reg['registration_sha256'],status='complete',arms=rows,
        new_samples=0,new_fitting_records=0,same_untouched_audit_sha256=file_sha256(output/'audit.npz'),
        post_calibration_pre_audit_extension=True,original_gate_and_endpoints_unchanged=True,
        work_seconds=perf_counter()-start,peak_rss_bytes=peak_rss_bytes()))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('register','build','audit'),required=True);args=parser.parse_args()
    spec=configuration(CONFIG);output=ROOT/spec['output'];folder=output/'diagnostic';folder.mkdir(exist_ok=True);reg=registration(spec,output)
    if args.stage=='register':
        if (folder/'registration.json').exists() or (output/'audit.npz').exists():raise FileExistsError('diagnostic must be sealed before audit')
        atomic_json(folder/'registration.json',reg);print(reg['registration_sha256']);return
    if reg!=json.loads((folder/'registration.json').read_text()):raise ValueError('diagnostic numeric registration mismatch')
    if args.stage=='build':build(spec,reg,output,folder)
    else:audit_probe(spec,reg,output,folder)


if __name__=='__main__':main()
