"""One frozen cached-state study: geometry, local SU2 mechanism, moment tests."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from shutil import copy2
from time import perf_counter

import numpy as np
from scipy.stats import norm

from su2_nqs.benchmarks.multipartite_geometry import (
    partial_trace_sites, induced_geometry, local_schur_channel,
    low_order_observables, lift_invariant_model, ppt_mixture_moment_fit,
    biseparable_moment_fit,
    triplet_screening_certificate,
    spin_measurement_branches, measurement_witness_pullback,
    solve_physical_multipartite_witness, covering_region_density,
    biseparable_residual_ball,
    localizable_block_negativity,
)
from su2_nqs.benchmarks.su2_multipartite_witness import (
    solve_su2_multipartite_witness, validate_witness_certificate,
)
from su2_nqs.benchmarks.su2_maximum_entropy import invariant_constraint_space, fit_invariant_maximum_entropy
from su2_nqs.benchmarks.spin_sector_entanglement import spin_sector_entanglement
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.benchmarks.spin_rotation_calibration import singlet_covering_state, lieb_mattis_singlet, pure_state_rdm
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

ROOT=Path(__file__).resolve().parents[3]
MODES=('raw','charge','magnetic','multiplicity')


def _write(path: Path, payload: dict) -> None:
    atomic_json(path,{**payload,'result_sha256':canonical_sha256(payload)})


def load_quality(cfg: dict) -> tuple[list,dict,dict]:
    """Verify producer/array seals and reconstruct actual whole-chain means."""
    inputs={};records=[]
    def checked(path, expected=None):
        path=ROOT/path; digest=file_sha256(path)
        if expected is not None and expected!=digest: raise ValueError('input seal changed: '+str(path))
        inputs[str(path.relative_to(ROOT))]=digest
        return path
    spec=json.loads(checked(cfg['source_config']).read_text())
    base=json.loads(checked(spec['nqs_input_config'],spec['nqs_input_config_sha256']).read_text())
    all_bonds={}
    for entry in base['inputs']:
        folder=Path(entry['directory']); payload={}
        for name,key in [('results.json','results_sha256'),('registration.json','registration_sha256')]:
            row=json.loads(checked(folder/name,entry[key]).read_text())
            if canonical_sha256({k:v for k,v in row.items() if k!='result_sha256'})!=row['result_sha256']:
                raise ValueError('canonical producer seal differs')
            payload[name]=row
        producer,reg=payload['results.json'],payload['registration.json']
        problem=reg['config']['problem']
        if (problem['geometry']!='triangular' or problem['j2']!=entry['j2'] or
            (problem['rows'],problem['columns'])!=(6,6) or problem['boundary']!='PBC_both_physical' or
            not producer['full_A1'] or producer['step']!=entry['step'] or
            producer['registration_sha256']!=reg['result_sha256'] or
            producer['params_sha256']!=reg['config']['params_sha256'] or producer['raw_sha256']!=entry['arrays_sha256']):
            raise ValueError('producer state identity differs')
        all_bonds[entry['j2']]=producer['bonds']
        with np.load(checked(folder/'physics.npz',entry['arrays_sha256'])) as arrays:
            for cut,sites in base['regions'].items():
                if problem['regions'][cut]!=sites: raise ValueError('site order mismatch')
                ids=arrays[cut+'__chain_ids']; unique,counts=np.unique(ids,return_counts=True)
                if not np.array_equal(unique,np.arange(128)) or not np.all(counts==4):
                    raise ValueError('expected128 independent equal-weight chains')
                per=arrays[cut+'__per_record_matrices']
                chains=np.array([per[ids==i].mean(0) for i in unique]); rho=arrays[cut+'__rho']
                if np.max(abs(chains.mean(0)-rho))>2e-12: raise ValueError('chain/RDM mean mismatch')
                records.append(dict(label=entry['label'],method='ViT',j2=entry['j2'],step=entry['step'],
                    chi=None,parent=cut,sites=sites,rho=rho,chains=chains))
    for entry in spec['dmrg_series']:
        folder=Path(entry['directory']); series=json.loads(checked(folder/'results.json',entry['results_sha256']).read_text())
        with np.load(checked(folder/'readouts.npz',entry['arrays_sha256'])) as arrays:
            for row in series['records']:
                if row['label'] not in entry['labels'] or row['cut'] not in base['regions']: continue
                sites=row.get('sites',base['regions'][row['cut']])
                if sites!=base['regions'][row['cut']]: raise ValueError('reference sites differ')
                records.append(dict(label=row['label'],method='DMRG',j2=.125,step=row.get('sweep') or int(row['label'].split('_')[2]),
                    chi=row['chi'],parent=row['cut'],sites=sites,
                    rho=arrays[row['label']+'__'+row['cut']+'__rho'],chains=None))
    return records,inputs,all_bonds


def _controls() -> dict[str,np.ndarray]:
    first=singlet_covering_state(6,((0,1),(2,3),(4,5)))
    second=singlet_covering_state(6,((1,2),(3,4),(5,0)))
    ring=first+second;ring/=np.linalg.norm(ring)
    ordered=lieb_mattis_singlet(12,(0,2,4,6,8,10))
    return dict(single_dimer_covering=np.outer(first,first.conj()),
                coherent_two_covering=np.outer(ring,ring.conj()),
                incoherent_two_covering=(np.outer(first,first.conj())+np.outer(second,second.conj()))/2,
                ordered_singlet_N12_reduction=pure_state_rdm(ordered,tuple(range(6))))


def _linear_audit(rho,chains,w):
    value=-float(np.trace(rho@w).real)
    if chains is None: return dict(value=value,SE=None,chains=0)
    vals=-np.einsum('cij,ji->c',chains,w).real
    return dict(value=float(vals.mean()),SE=float(vals.std(ddof=1)/np.sqrt(len(vals))),chains=len(vals))


def analyze(config: Path) -> dict:
    cfg=json.loads(config.read_text()); output=ROOT/cfg['output']
    if output.exists(): raise FileExistsError('completed/partial studies immutable; use fresh version')
    records,inputs,bonds=load_quality(cfg); inputs[str(config.relative_to(ROOT))]=file_sha256(config)
    sources=transitive_local_source_manifest([Path(__file__),ROOT/'su2_nqs/frontier/provenance.py'],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    output.mkdir(parents=True)
    for name in sources:
        target=output/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True);copy2(ROOT/name,target)
    _write(output/'registration.json',dict(config=cfg,inputs_sha256=inputs,sources=sources,
        prospective=False,design_frozen_before_fitting=True,input_states_repaired=False))
    started=perf_counter(); evaluations=[];fits=[];geometry=[];mechanisms=[];moment_rows=[];saved={}
    controls=_controls();maxent_cache={}
    def budget():
        if perf_counter()-started>cfg['seconds_cap']: raise TimeoutError('registered study cap exhausted')
    def progress(stage):
        _write(output/'progress.json',dict(stage=stage,seconds=perf_counter()-started,fits=fits,moment_tests=moment_rows,complete=False))
    for case in cfg['cases']:
        budget(); name=case['name']; keep=case['keep']; n=len(keep);parties=case['parties']
        selected=[]
        for row in records:
            if row['parent']!=case['parent']: continue
            selected.append(dict(**{k:v for k,v in row.items() if k not in ('rho','chains')},
                rho=partial_trace_sites(row['rho'],keep),
                chains=None if row['chains'] is None else partial_trace_sites(row['chains'],keep)))
        for j2 in (0.,.125):
            sites=[selected[0]['sites'][i] for i in keep]
            geometry.append(dict(case=name,j2=j2,parties=parties,**induced_geometry(sites,bonds[j2],j2)))
        for trainer in cfg['training_endpoints']:
            budget(); train=next(r for r in selected if r['label']==trainer)
            training_rho=train['rho'] if train['chains'] is None else train['chains'][:64].mean(0)
            seconds=cfg['six_individual_solver_seconds'] if n==6 and len(parties)==6 else cfg['solver_seconds']
            fit,cert=solve_su2_multipartite_witness(training_rho,parties,time_reversal_even=True,
                solver_options=dict(time_limit_secs=seconds,max_iters=10000,eps=2e-6))
            validate_witness_certificate(cert)
            key=name+'__'+trainer
            saved[key+'__W']=cert.witness;saved[key+'__Q']=np.array(cert.q_operators)
            fits.append(dict(key=key,case=name,trainer=trainer,cuts=cert.cuts,fit=fit,
                training_chains=None if train['chains'] is None else list(range(64)),
                audit_training_endpoint_chains=None if train['chains'] is None else list(range(64,128)),
                deterministic_reference_fit=train['chains'] is None))
            maps={m:cert.witness if m=='raw' else local_schur_channel(cert.witness,parties,m) for m in MODES}
            for mode,w in maps.items(): saved[key+'__'+mode+'__W']=w
            for row in selected:
                matrix=row['rho']; chains=row['chains']
                if row['label']==trainer and chains is not None: chains=chains[64:];matrix=chains.mean(0)
                audits={mode:_linear_audit(matrix,chains,w) for mode,w in maps.items()}
                meta={k:row[k] for k in ('label','method','j2','step','chi')}
                evaluations.append(dict(**meta,case=name,key=key,trainer=trainer,audits=audits,
                    same_endpoint_fit_audit_overlap=0,own_reference_fit=row['label']==trainer and chains is None))
            control_rows=[]
            for label,rho in controls.items():
                # Synthetic six-spin controls share parties/site order, not a lattice Hamiltonian.
                rho=partial_trace_sites(rho,keep)
                control_rows.append(dict(label=label,values={m:-float(np.trace(rho@w).real) for m,w in maps.items()},
                    QSL=False,not_ground_state_reference=True))
            fits[-1]['controls']=control_rows
            progress('geometry '+key)
            print(json.dumps(dict(stage='witness',case=name,trainer=trainer,bound=fit['entanglement_lower_bound'],seconds=perf_counter()-started)),flush=True)
        # Max-entropy point predictions retain same frozen W for every quality point.
        for row in selected:
            budget(); cache=(row['label'],case['parent'],tuple(keep))
            if cache not in maxent_cache:
                space=invariant_constraint_space(n);omega=space.data(row['rho'])
                report,model,_=fit_invariant_maximum_entropy(omega,space)
                if not report['certified_moment_fit']: raise RuntimeError('maximum entropy constraints failed')
                maxent_cache[cache]=(report,lift_invariant_model(model,n))
            report,model=maxent_cache[cache]
            vals={}
            for trainer in cfg['training_endpoints']:
                w=saved[name+'__'+trainer+'__W']
                vals[trainer]=dict(actual=-float(np.trace(row['rho']@w).real),
                                  maximum_entropy=-float(np.trace(model@w).real))
            mechanisms.append(dict(case=name,label=row['label'],method=row['method'],j2=row['j2'],
                maximum_entropy=report,point_predictions=vals,fit_uncertainty_propagated=False,
                point_comparison_is_not_significance_test=True))
    # Stronger low-order feasibility/exclusion analysis at one fixed reference.
    for case in cfg['cases']:
        if case['name'] not in cfg['moment_test_cases']: continue
        budget(); name=case['name'];base=next(r for r in records if r['parent']==case['parent'] and
            r['label'] in (cfg['moment_test_parent'],cfg['moment_test_parent']+'_hexagon'))
        rho=partial_trace_sites(base['rho'],case['keep']);n=len(case['keep'])
        ops=low_order_observables(n);target=np.einsum('kij,ji->k',ops,rho).real
        fit,cert=solve_su2_multipartite_witness(rho,case['parties'],time_reversal_even=True,
            operator_span='low_order_and_spin_weights',solver_options=dict(time_limit_secs=30.,max_iters=10000,eps=2e-6))
        saved[name+'__low_W']=cert.witness;saved[name+'__low_Q']=np.array(cert.q_operators)
        # Repair identity shift stays inside the exact declared low-order algebra.
        coeff=np.linalg.lstsq(ops.reshape(len(ops),-1).T,cert.witness.reshape(-1),rcond=1e-11)[0]
        span_error=float(np.linalg.norm(cert.witness-np.einsum('k,kij->ij',coeff,ops)))
        exclusion=fit['entanglement_lower_bound']>1e-6 and span_error<1e-10
        entry=dict(case=name,parent=base['label'],low_order_witness=fit,
            span_residual=span_error,biseparability_excluded_for_nominal_moments=bool(exclusion),
            numerical_not_interval_certificate=True,DMRG_quality_bias_not_bounded=True)
        if exclusion:
            entry['stronger_search']='Feasible low-order witness already excludes all biseparable completions of these nominal constraints.'
        else:
            ppt,ppt_arrays=ppt_mixture_moment_fit(target,ops,case['parties'],seconds=30.)
            entry['PPT_outer']=ppt;saved[name+'__PPT_components']=ppt_arrays
            inner,arrays=biseparable_moment_fit(target,ops,case['parties'],
                iterations=cfg['biseparable_iterations'],tolerance=cfg['biseparable_moment_tolerance'])
            entry['explicit_biseparable']=inner
            for k,v in arrays.items():saved[name+'__biseparable_'+k]=v
        saved[name+'__moment_target']=target;moment_rows.append(entry);progress('moments '+name)
        print(json.dumps(dict(stage='moments',case=name,exclusion=exclusion,inner=entry.get('explicit_biseparable'),seconds=perf_counter()-started)),flush=True)
    # Actual 3+3 local-multiplicity readout, distinct from three two-spin parties.
    local=[]
    for row in records:
        budget();report=spin_sector_entanglement(row['rho'],(0,1,2))
        local.append(dict(label=row['label'],parent=row['parent'],method=row['method'],j2=row['j2'],
            report=report,nonlinear_bias_removed=False,uncertainty_not_evaluated_here=True))
    for label,rho in controls.items():
        local.append(dict(label=label,method='control',report=spin_sector_entanglement(rho,(0,1,2))))
    critical=float(norm.ppf(1-cfg['family_alpha']/(2*cfg['primary_family_size'])))
    comparisons=[]
    for fit in fits:
        these=[r for r in evaluations if r['key']==fit['key']]
        a=sorted([r for r in these if r['method']=='ViT' and r['j2']==0],key=lambda r:r['step'])
        b=sorted([r for r in these if r['method']=='ViT' and r['j2']==.125],key=lambda r:r['step'])
        c=sorted([r for r in these if r['method']=='DMRG'],key=lambda r:(r['chi'],r['step']))
        for kind,left,right in [('coupling',a,b),('cross_method',b,c)]:
            if not left or not right:continue
            for mode in MODES:
                delta=right[-1]['audits'][mode]['value']-left[-1]['audits'][mode]['value']
                se=np.hypot(left[-1]['audits'][mode]['SE'] or 0,right[-1]['audits'][mode]['SE'] or 0)
                drift=sum(np.ptp([r['audits'][mode]['value'] for r in series]) for series in (left,right))
                comparisons.append(dict(key=fit['key'],case=fit['case'],mode=mode,type=kind,
                    difference=delta,SE=float(se),own_quality_range_sum=float(drift),threshold=float(drift+critical*se),
                    passes_descriptive_gate=bool(abs(delta)>drift+critical*se),
                    DMRG_SE_missing_is_not_zero_error=kind=='cross_method'))
    np.savez_compressed(output/'arrays.npz',**saved)
    result=dict(geometry=geometry,fits=fits,evaluations=evaluations,maximum_entropy=mechanisms,
        low_order_tests=moment_rows,local_schur=local,comparisons=comparisons,
        critical_SE=critical,actual_comparisons=len(comparisons),registered_family_size=cfg['primary_family_size'],
        seconds=perf_counter()-started,raw_sha256=file_sha256(output/'arrays.npz'),
        new_training=0,new_samples=0,new_H_calls=0,new_amplitude_queries=0,
        strict_SU2_input_assumed=False,phase_claim=False,
        caveats='Feasible fixed-witness lower estimates, not certified optimal GMN. Zero inconclusive. Local channel lower-bound differences do not measure exact entanglement loss. MaxEnt differences are point predictions with fitting uncertainty unpropagated. PPT mixtures are outer relaxation; only explicitly saved product mixtures prove biseparability to reported moment precision. Quality ranges and chain SE do not certify remaining bias.')
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),raw_sha256=result['raw_sha256']))
    return result


def refine(config: Path) -> dict:
    """Bounded numerical follow-up: refit local-channel W, full JK and inner LP.

    No new physical observable or state is selected. Reoptimizing a witness
    after the declared channel tests retention beyond the original witness's
    detection power. Maximum-entropy uncertainty includes whole-chain refits.
    """
    cfg=json.loads(config.read_text()); primary=ROOT/cfg['primary_directory']
    completion=json.loads((primary/'completion.json').read_text())
    for name,key in [('results.json','results_sha256'),('registration.json','registration_sha256'),('arrays.npz','raw_sha256')]:
        if file_sha256(primary/name)!=completion[key]:raise ValueError('primary study seal differs')
    base=json.loads((primary/'registration.json').read_text())['config']
    records,inputs,_=load_quality(base);output=ROOT/cfg['output']
    if output.exists():raise FileExistsError('use a fresh follow-up output')
    sources=transitive_local_source_manifest([Path(__file__),ROOT/'su2_nqs/frontier/provenance.py'],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    output.mkdir(parents=True)
    for name in sources:
        dst=output/'source_snapshots'/name;dst.parent.mkdir(parents=True,exist_ok=True);copy2(ROOT/name,dst)
    _write(output/'registration.json',dict(config=cfg,inputs=inputs,sources=sources,
        primary_completion_sha256=file_sha256(primary/'completion.json'),
        retrospective_numerical_followup=True,selection_scope='fixed compact6 mechanism and two unfinished coarse-party inner tests'))
    start=perf_counter(); saved={};channel_rows=[];uncertainty=[];inner_rows=[]
    def check():
        if perf_counter()-start>cfg['seconds_cap']:raise TimeoutError('follow-up cap reached')
    parties=((0,3),(1,4),(2,5))
    charge_projectors={}
    for qs in product((0,2),repeat=3):
        p=np.eye(64)
        for q,pair in zip(qs,parties):
            singlet=np.eye(64)/4-spin_dot_matrix(6,*pair)
            p=p@(singlet if q==0 else np.eye(64)-singlet)
        charge_projectors[','.join(map(str,qs))]=p
    for trainer in base['training_endpoints']:
        check();r=next(r for r in records if r['parent']=='compact6' and r['label']==trainer)
        fit,cert=solve_su2_multipartite_witness(local_schur_channel(r['chains'][:64].mean(0),parties,'charge'),
            parties,time_reversal_even=True,solver_options=dict(time_limit_secs=30.,max_iters=10000,eps=2e-6))
        pulled=local_schur_channel(cert.witness,parties,'charge')
        saved[trainer+'__charge_fit_W']=cert.witness;saved[trainer+'__charge_fit_Q']=np.array(cert.q_operators)
        saved[trainer+'__pullback_W']=pulled
        audits=[]
        for row in records:
            if row['parent']!='compact6':continue
            chains=row['chains'];rho=row['rho']
            if row['label']==trainer:chains=chains[64:];rho=chains.mean(0)
            sectors={}
            for labels,p in charge_projectors.items():
                probability=_linear_audit(rho,chains,-p)
                contribution=_linear_audit(rho,chains,p@cert.witness@p)
                sectors[labels]=dict(probability=probability,signed_witness_contribution=contribution,
                    can_support_three_party_GME=labels=='2,2,2')
            total=_linear_audit(rho,chains,pulled)
            if abs(sum(s['signed_witness_contribution']['value'] for s in sectors.values())-total['value'])>1e-11:
                raise RuntimeError('local-charge witness decomposition failed')
            audits.append(dict(label=row['label'],method=row['method'],j2=row['j2'],step=row['step'],chi=row['chi'],
                raw=_linear_audit(rho,chains,cert.witness),after_local_charge=total,
                local_charge_sectors=sectors))
        channel_rows.append(dict(trainer=trainer,fit=fit,cuts=cert.cuts,audits=audits))
        print(json.dumps(dict(stage='charge-refit',trainer=trainer,bound=fit['entanglement_lower_bound'])),flush=True)
    with np.load(primary/'arrays.npz') as archive:
        witnesses=np.array([archive['compact6_coarse__'+t+'__W'] for t in base['training_endpoints']])
    space=invariant_constraint_space(6)
    for label in cfg['maximum_entropy_endpoints']:
        check();row=next(r for r in records if r['parent']=='compact6' and r['label']==label)
        rho=row['rho'];chains=row['chains'];n=len(chains)
        def prediction(x):
            check(); report,model,_=fit_invariant_maximum_entropy(space.data(x),space)
            if not report['certified_moment_fit']:raise RuntimeError('JK moment fit failed')
            lifted=lift_invariant_model(model,6)
            return np.array([-np.einsum('kij,ji->k',witnesses,x).real,
                             -np.einsum('kij,ji->k',witnesses,lifted).real])
        point=prediction(rho)
        deleted=np.array([prediction((n*rho-c)/(n-1)) for c in chains])
        delta=deleted[:,0]-deleted[:,1]
        err=lambda a:np.sqrt((n-1)/n*np.sum((a-a.mean(0))**2,axis=0))
        uncertainty.append(dict(label=label,actual=point[0].tolist(),maximum_entropy=point[1].tolist(),
            difference=(point[0]-point[1]).tolist(),paired_refit_SE=err(delta).tolist(),
            actual_SE=err(deleted[:,0]).tolist(),maximum_entropy_SE=err(deleted[:,1]).tolist(),
            whole_chains=n,nonlinear_bias_removed=False))
        saved[label+'__maximum_entropy_JK']=deleted
        print(json.dumps(dict(stage='maxent-JK',label=label,difference=uncertainty[-1]['difference'],SE=uncertainty[-1]['paired_refit_SE'])),flush=True)
    for name in cfg['constructive_refinement_cases']:
        check();case=next(c for c in base['cases'] if c['name']==name)
        row=next(r for r in records if r['parent']==case['parent'] and r['label'] in ('DMRG_10240_32','DMRG_10240_32_hexagon'))
        ops=low_order_observables(6);target=np.einsum('kij,ji->k',ops,row['rho']).real
        report,arrays=biseparable_moment_fit(target,ops,case['parties'],iterations=cfg['constructive_iterations'],tolerance=1e-7)
        inner_rows.append(dict(case=name,parent=row['label'],report=report))
        for k,v in arrays.items():saved[name+'__'+k]=v
        print(json.dumps(dict(stage='constructive-refinement',case=name,report=report)),flush=True)
    np.savez_compressed(output/'arrays.npz',**saved)
    result=dict(channels=channel_rows,maximum_entropy_uncertainty=uncertainty,constructive=inner_rows,
        seconds=perf_counter()-start,raw_sha256=file_sha256(output/'arrays.npz'),
        new_samples=0,new_H_calls=0,new_training=0,phase_claim=False,
        charge_mechanism='For three two-spin parties, every local-q outcome with one q=0 has a one-dimensional factor and is biseparable. Only local (q1,q2,q3)=(2,2,2) can retain genuine tripartite entanglement after charge measurement. Outcome projections are LOCAL, not a nonlocal total-singlet filter. All outcomes and their signed witness contributions retained; no conditional ratio estimator is used.')
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),raw_sha256=result['raw_sha256']))
    return result


def mechanism_audit(config: Path) -> dict:
    """Analytic triplet screening plus independent product-mixture replay."""
    cfg=json.loads(config.read_text());primary=ROOT/cfg['primary_directory'];follow=ROOT/cfg['output']
    for folder in (primary,follow):
        seal=json.loads((folder/'completion.json').read_text())
        for name,key in [('results.json','results_sha256'),('registration.json','registration_sha256'),('arrays.npz','raw_sha256')]:
            if file_sha256(folder/name)!=seal[key]:raise ValueError('completed analysis seal failed')
    base=json.loads((primary/'registration.json').read_text())['config'];records,inputs,_=load_quality(base)
    output=primary/'analytic_mechanism'
    if output.exists():raise FileExistsError('analytic audit immutable')
    output.mkdir(parents=True)
    sources=transitive_local_source_manifest([Path(__file__),ROOT/'su2_nqs/frontier/provenance.py'],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    for name in sources:
        p=output/'source_snapshots'/name;p.parent.mkdir(parents=True,exist_ok=True);copy2(ROOT/name,p)
    _write(output/'registration.json',dict(inputs=inputs,sources=sources,
        operator='W=P222/2-3P222,total0/2; q_i=2 means each physical pair is a spin-1 triplet.',
        parties=[[0,3],[1,4],[2,5]],secondary_exploratory=True,
        no_operator_fit=True,no_nonlocal_postselected_state=True,
        prior_results_sha256=[file_sha256(p/'results.json') for p in (primary,follow)]))
    cert,ops=triplet_screening_certificate(((0,3),(1,4),(2,5)));validate_witness_certificate(cert)
    saved=dict(**ops,witness=cert.witness,Q=np.array(cert.q_operators));rows=[]
    for r in records:
        if r['parent']!='compact6':continue
        row={k:r[k] for k in ('label','method','j2','step','chi')}
        row['GME_bound']=_linear_audit(r['rho'],r['chains'],cert.witness)
        row['p222']=_linear_audit(r['rho'],r['chains'],-ops['P222'])
        row['p222_total0']=_linear_audit(r['rho'],r['chains'],-ops['P222_total0'])
        rows.append(row)
    control_rows=[]
    for name,rho in _controls().items():
        control_rows.append(dict(label=name,GME_bound=-float(np.trace(rho@cert.witness).real),
            p222=float(np.trace(rho@ops['P222']).real),p222_total0=float(np.trace(rho@ops['P222_total0']).real)))
    # Independent bit-SWAP spin operators and polynomial spin projectors.
    from itertools import combinations_with_replacement,combinations
    d=64; dots=[]; eye=np.eye(d)
    for i,j in combinations(range(6),2):
        swap=np.zeros((d,d))
        for x in range(d):
            flip=(((x>>(5-i))^(x>>(5-j)))&1)*((1<<(5-i))|(1<<(5-j)))
            swap[x^flip,x]=1
        dots.append(.5*swap-.25*eye)
    s2=4.5*eye+2*sum(dots); projectors=[]
    for val in (0,2,6,12):
        p=eye.copy()
        for other in (0,2,6,12):
            if other!=val:p=p@(s2-other*eye)/(val-other)
        projectors.append(p)
    bank=np.array([eye,*projectors,*dots,*[(a@b+b@a)/2 for a,b in combinations_with_replacement(dots,2)]])
    replay=[]
    with np.load(primary/'arrays.npz') as main,np.load(follow/'arrays.npz') as extra:
        for case in base['cases']:
            if case['name'] not in base['moment_test_cases'] or case['name']=='compact6_coarse':continue
            name=case['name'];container=extra if name.endswith('coarse') else main
            prefix=name+'__' if name.endswith('coarse') else name+'__biseparable_'
            weights=container[prefix+'weights'];vectors=container[prefix+'vectors'];cuts=container[prefix+'cuts']
            reconstructed=np.einsum('a,ai,aj->ij',weights,vectors,vectors.conj())
            parent=next(r for r in records if r['parent']==case['parent'] and r['label'] in ('DMRG_10240_32','DMRG_10240_32_hexagon'))
            factor_error=0.
            for mask,vector in zip(cuts,vectors):
                cut=tuple(i for i in range(6) if int(mask)&(1<<i));order=cut+tuple(i for i in range(6) if i not in cut)
                singular=np.linalg.svd(vector.reshape((2,)*6).transpose(order).reshape(2**len(cut),-1),compute_uv=False)
                factor_error=max(factor_error,float(np.max(singular[1:],initial=0)))
            moment_error=float(np.max(abs(np.einsum('kij,ji->k',bank,reconstructed-parent['rho']))))
            if weights.min()<0 or abs(weights.sum()-1)>1e-12 or factor_error>1e-12 or moment_error>1e-7:
                raise RuntimeError('independent product-mixture certificate failed')
            replay.append(dict(case=name,component_count=len(weights),raw_moment_max_error=moment_error,
                product_Schmidt_tail_max=factor_error,minimum_weight=float(weights.min()),
                compatible_regional_completion_not_same_full_RDM=True))
    comparisons=[]
    a=[r for r in rows if r['method']=='ViT' and r['j2']==0]
    b=[r for r in rows if r['method']=='ViT' and r['j2']==.125]
    c=sorted([r for r in rows if r['method']=='DMRG'],key=lambda r:(r['chi'],r['step']))
    for kind,left,right in [('coupling',a,b),('cross_method',b,c)]:
        delta=right[-1]['GME_bound']['value']-left[-1]['GME_bound']['value']
        se=float(np.hypot(left[-1]['GME_bound']['SE'] or 0,right[-1]['GME_bound']['SE'] or 0))
        drift=sum(np.ptp([r['GME_bound']['value'] for r in x]) for x in (left,right))
        threshold=float(drift+norm.ppf(1-.05/4)*se)
        comparisons.append(dict(type=kind,difference=delta,SE=se,quality_range_sum=float(drift),threshold=threshold,
            passes_descriptive_gate=bool(abs(delta)>threshold),secondary_exploratory=True))
    np.savez_compressed(output/'arrays.npz',**saved)
    result=dict(records=rows,controls=control_rows,comparisons=comparisons,
        independent_biseparable_replay=replay,certificate=cert.diagnostics,
        raw_sha256=file_sha256(output/'arrays.npz'),new_samples=0,new_H_calls=0,phase_claim=False,
        interpretation='Analytic three spin-1 screening witness. A measured total-singlet probability is part of a valid unconditioned witness, not a postselection argument. Independent dimer coverings can also be detected across these coarse parties. Regional low-moment compatibility does not assert that the candidate equals the original RDM or has a consistent global ground-state extension.')
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),raw_sha256=result['raw_sha256']))
    return result


def local_probe(config: Path) -> dict:
    """Frozen site loss/local measurement study on the existing quality series.

    Conditional RDMs generally break SU2. Physical witnesses are fitted in
    computational coordinates, not scalar-only coordinates. Outcome audits
    use linear parent pullbacks, keeping both doubled m=-1/+1 outcomes.
    """
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.benchmarks.spin_sector import spin_squared_sz_sparse, singlet_projector_action
    from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
    from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns
    cfg=json.loads(config.read_text());primary=ROOT/cfg['primary_directory'];output=ROOT/cfg['output']
    if output.exists():raise FileExistsError('local probes immutable; use a new version')
    completion=json.loads((primary/'completion.json').read_text())
    for name,key in [('results.json','results_sha256'),('registration.json','registration_sha256'),('arrays.npz','raw_sha256')]:
        if file_sha256(primary/name)!=completion[key]:raise ValueError('primary data seal differs')
    base=json.loads((primary/'registration.json').read_text())['config'];records,inputs,bonds=load_quality(base)
    inputs[str(config.relative_to(ROOT))]=file_sha256(config)
    positive=ROOT/cfg['positive_calibration_directory'];positive_result=json.loads((positive/'results.json').read_text())
    if file_sha256(positive/'readouts.npz')!=positive_result['raw_sha256']:
        raise ValueError('positive calibration seal differs')
    inputs[str((positive/'readouts.npz').relative_to(ROOT))]=file_sha256(positive/'readouts.npz')
    output.mkdir(parents=True)
    sources=transitive_local_source_manifest([Path(__file__),ROOT/'su2_nqs/frontier/provenance.py'],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    for name in sources:
        dst=output/'source_snapshots'/name;dst.parent.mkdir(parents=True,exist_ok=True);copy2(ROOT/name,dst)
    _write(output/'registration.json',dict(config=cfg,inputs=inputs,sources=sources,
        design_frozen_before_fitting=True,prospective=False,
        fit='equal-parent pooled first64 chains of J0/500 and J.125/8600; later endpoints all128, those parents last64 only',
        real_physical_detector='Explicit real-symmetric physical operators; original imaginary coherences retained. No SU2 restriction on child witness.',
        controls_not_energy_matched=True,postselected_outcomes_dropped=False))
    start=perf_counter();saved={};fits=[];audit=[];controls=[];structure=[];geometry=[]
    def check():
        if perf_counter()-start>cfg['seconds_cap']:raise TimeoutError('local study cap reached')
    def progress(stage):
        _write(output/'progress.json',dict(stage=stage,seconds=perf_counter()-start,fits=len(fits),audit_records=len(audit),complete=False))
    # Exact N36 symmetry-invariant dimer ENSEMBLE, not a coherent A1 cat.
    group=triangular_hnf_gamma_a1_group(((6,0),(0,6)))
    parent_cover=tuple((r*6+c,r*6+c+1) for r in range(6) for c in (0,2,4))
    # The existing independent physical singlet polynomial defines an ordered control.
    s2,states=spin_squared_sz_sparse(12,n_up=6)
    packed=np.array([sum(bit<<(11-i) for i,bit in enumerate(bits)) for bits in states])
    with np.load(positive/'readouts.npz') as arrays:
        known=np.array(arrays['kagome18__known_family_RVB__A6__rho'])
    for case in cfg['cases']:
        check();name=case['name'];parent=case['parent'];parties=case['parties']
        selected=[r for r in records if r['parent']==parent]
        if 'keep' in case:
            kept=case['keep']
            selected=[dict(**{k:v for k,v in r.items() if k not in ('rho','chains','sites')},
                sites=[r['sites'][i] for i in kept],rho=partial_trace_sites(r['rho'],kept),
                chains=None if r['chains'] is None else partial_trace_sites(r['chains'],kept)) for r in selected]
        sites=selected[0]['sites'];n=len(sites)
        training=[next(r for r in selected if r['label']==label) for label in cfg['training_endpoints']]
        train_rho=sum(r['chains'][:64].mean(0) for r in training)/len(training)
        ctl={label:partial_trace_sites(rho,case['keep']) if 'keep' in case else rho
             for label,rho in _controls().items()}
        ctl['A1_dimer_ensemble_N36']=sum(covering_region_density(36,[(p[i],p[j]) for i,j in parent_cover],sites)
                                                for p in group)/len(group)
        # Complete the actual six-site colour pattern to four spins per colour.
        colours=[(s//6+s%6)%3 for s in sites]
        full=colours+[c for c in range(3) for _ in range(4-colours.count(c))]
        product_state=np.array([1.],complex)
        for c in full:product_state=np.kron(product_state,np.array([np.exp(2j*np.pi*c/3),1])/np.sqrt(2))
        projected=singlet_projector_action(product_state[packed],s2,maximum_spin=6)
        projected/=np.linalg.norm(projected);vector=np.zeros(4096,complex);vector[packed]=projected
        ctl['S0_restored120_N12_region']=pure_state_rdm(vector,tuple(range(n)))
        ctl['finite_kagome_RVB_N18_different_geometry']=partial_trace_sites(known,case['keep']) if 'keep' in case else known
        for label,rho in ctl.items():saved[name+'__control__'+label+'__rho']=rho
        for j2 in (0.,.125):geometry.append(dict(case=name,j2=j2,parties=parties,**induced_geometry(sites,bonds[j2],j2)))
        if 'baseline_key' in case:
            prior=json.loads((primary/'results.json').read_text())
            fit=next(x for x in prior['fits'] if x['key']==case['baseline_key'])
            with np.load(primary/'arrays.npz') as arrays:baseline=np.array(arrays[fit['key']+'__W'])
            baseline_report=dict(reused=True,source=case['baseline_key'],fit=fit['fit'])
        else:
            report,cert=solve_su2_multipartite_witness(train_rho,parties,time_reversal_even=True,
                solver_options=dict(time_limit_secs=cfg['baseline_solver_seconds'],max_iters=10000,eps=2e-6))
            validate_witness_certificate(cert);baseline=cert.witness;baseline_report=dict(reused=False,fit=report)
            saved[name+'__baseline__Q']=np.array(cert.q_operators)
        saved[name+'__baseline__W']=baseline;fits.append(dict(key=name+'__baseline',**baseline_report))
        for row in selected:
            chains=row['chains'];rho=row['rho']
            if row['label'] in cfg['training_endpoints']:chains=chains[64:];rho=chains.mean(0)
            meta={k:row[k] for k in ('label','method','j2','step','chi')}
            audit.append(dict(case=name,operation='baseline',site=None,**meta,
                score=_linear_audit(rho,chains,baseline)))
        for label,rho in ctl.items():controls.append(dict(case=name,label=label,operation='baseline',site=None,
            score=-float(np.trace(rho@baseline).real),QSL_claim=False,geometry_matched='kagome' not in label))
        for site in cfg['measured_sites']:
            check();keep=tuple(i for i in range(n) if i!=site)
            children=[[keep.index(i) for i in p if i!=site] for p in parties]
            children=[p for p in children if p]
            for axis in ('discard',*cfg['measurement_axes']):
                check();key=name+'__'+str(site)+'__'+axis;pullbacks=[];branch_ops=[];branch_fits=[]
                if axis=='discard':
                    targets=[partial_trace_sites(train_rho,keep)]
                else:
                    targets,prob=spin_measurement_branches(train_rho,site,axis)
                    targets=[x/p for x,p in zip(targets,prob)]
                for k,target in enumerate(targets):
                    report,cert=solve_physical_multipartite_witness(target,children,seconds=cfg['child_solver_seconds'],
                        real_operators=cfg['real_physical_witnesses'])
                    validate_witness_certificate(cert);branch_ops.append(cert.witness);branch_fits.append(report)
                    saved[key+'__'+str(k)+'__W']=cert.witness;saved[key+'__'+str(k)+'__Q']=np.array(cert.q_operators)
                    if axis!='discard':pullbacks.append(measurement_witness_pullback(cert.witness,site,axis,(-1,1)[k]))
                fits.append(dict(key=key,parties=children,branch_fits=branch_fits))
                for row in selected:
                    chains=row['chains'];rho=row['rho']
                    if row['label'] in cfg['training_endpoints']:chains=chains[64:];rho=chains.mean(0)
                    meta={k:row[k] for k in ('label','method','j2','step','chi')}
                    if axis=='discard':
                        child=partial_trace_sites(rho,keep);chain_child=None if chains is None else partial_trace_sites(chains,keep)
                        score=_linear_audit(child,chain_child,branch_ops[0]);outcomes=[]
                    else:
                        branch,ps=spin_measurement_branches(rho,site,axis)
                        outcomes=[]
                        for k,m in enumerate((-1,1)):
                            probability=_linear_audit(rho,chains,-measurement_witness_pullback(np.eye(2**(n-1)),site,axis,m))
                            signed=_linear_audit(rho,chains,pullbacks[k])
                            outcomes.append(dict(m=m,probability=probability,weighted_score=signed,
                                conditional_score=signed['value']/ps[k],conditional_ratio_SE_not_reported=True))
                        score=_linear_audit(rho,chains,sum(pullbacks))
                        if abs(ps.sum()-1)>2e-12 or np.max(abs(branch.sum(0)-partial_trace_sites(rho,keep)))>2e-12:
                            raise RuntimeError('all-outcome closure failed')
                    audit.append(dict(case=name,operation=axis,site=site,physical_site=sites[site],**meta,
                        score=score,outcomes=outcomes,score_is_signed_conservative_lower_estimate=True,
                        difference_from_baseline_is_not_exact_GMN_loss=True,
                        remaining_party_count=len(children),same_party_count=len(children)==len(parties)))
                for label,rho in ctl.items():
                    if axis=='discard':score=-float(np.trace(partial_trace_sites(rho,keep)@branch_ops[0]).real);outcomes=[]
                    else:
                        _,ps=spin_measurement_branches(rho,site,axis)
                        ws=[-float(np.trace(rho@w).real) for w in pullbacks];score=sum(ws)
                        outcomes=[dict(m=m,probability=float(p),weighted_score=v) for m,p,v in zip((-1,1),ps,ws)]
                    controls.append(dict(case=name,label=label,operation=axis,site=site,score=score,outcomes=outcomes,
                        QSL_claim=False,geometry_matched='kagome' not in label))
                progress(key);print(json.dumps(dict(case=name,site=site,axis=axis,seconds=perf_counter()-start)),flush=True)
        # Fixed original and discarded-spin scalar/multiplicity data; no child symmetry repair.
        for row in selected:
            for site in [None,*cfg['measured_sites']]:
                rho=row['rho'] if site is None else partial_trace_sites(row['rho'],tuple(i for i in range(n) if i!=site))
                report=physical_spin_block_readout(rho)
                u,labels=regional_schur_columns(int(np.log2(len(rho))))
                blocks={}
                for q in sorted({x[0] for x in labels}):
                    rr=sum(u[:,[i for i,x in enumerate(labels) if x[:2]==(q,m)]].conj().T@rho@
                        u[:,[i for i,x in enumerate(labels) if x[:2]==(q,m)]] for m in range(-q,q+1,2))
                    blocks[str(q)]=dict(weight=float(np.trace(rr).real),reduced_eigenvalues=np.linalg.eigvalsh(rr).tolist(),
                        interpretation='Invariant-observable reduced data; strict entanglement spectrum only if raw SU2 eligibility passed.')
                    saved[name+'__'+row['label']+'__'+str(site)+'__q'+str(q)]=rr
                structure.append(dict(case=name,label=row['label'],site=site,method=row['method'],j2=row['j2'],
                    strict_su2_eligible=report['strict_su2_eligible'],su2_defect=report['su2_factorization_frobenius'],blocks=blocks))
    # Contrast complete fixed detector inventory, comparing each own quality range.
    comparisons=[];critical=norm.ppf(1-cfg['family_alpha']/(2*cfg['comparison_family_size']))
    for case in cfg['cases']:
        contexts=[('baseline',None)]+[(op,site) for op in ('discard',*cfg['measurement_axes']) for site in cfg['measured_sites']]
        for op,site in contexts:
            these=[r for r in audit if r['case']==case['name'] and r['operation']==op and r['site']==site]
            a=sorted([r for r in these if r['method']=='ViT' and r['j2']==0],key=lambda r:r['step'])
            b=sorted([r for r in these if r['method']=='ViT' and r['j2']==.125],key=lambda r:r['step'])
            c=sorted([r for r in these if r['method']=='DMRG'],key=lambda r:(r['chi'],r['step']))
            for kind,left,right in [('coupling',a,b),('cross_method',b,c)]:
                difference=right[-1]['score']['value']-left[-1]['score']['value']
                se=float(np.hypot(left[-1]['score']['SE'] or 0,right[-1]['score']['SE'] or 0))
                drift=float(sum(np.ptp([r['score']['value'] for r in rr]) for rr in (left,right)))
                comparisons.append(dict(case=case['name'],operation=op,site=site,type=kind,difference=difference,SE=se,
                    own_quality_range_sum=drift,threshold=float(drift+critical*se),passes_descriptive_gate=bool(abs(difference)>drift+critical*se)))
    if len(comparisons)!=cfg['comparison_family_size']:raise RuntimeError('frozen comparison count differs')
    np.savez_compressed(output/'arrays.npz',**saved)
    result=dict(geometry=geometry,fits=fits,audits=audit,controls=controls,structure=structure,comparisons=comparisons,
        seconds=perf_counter()-start,raw_sha256=file_sha256(output/'arrays.npz'),new_training=0,new_samples=0,new_H_calls=0,
        new_amplitude_queries=0,phase_claim=False,missing_geometries=cfg['missing_geometries'],
        caveats='Frozen feasible witnesses, not optimal GMN. Signed sum over all measurement branches is a conservative weighted lower estimate. Conditional ratios descriptive, uncertainty evaluated on linear pullbacks. Real-operator restriction explicit. Old caches retrospective. Finite-kagome RVB is different geometry and not a triangular phase label. Quality ranges do not bound all remaining error. Non-SU2 reduced spectra are observable data, not a strict entanglement spectrum.')
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),raw_sha256=result['raw_sha256']))
    return result


def full_rdm_audit(config: Path) -> dict:
    """Explicit biseparable full-RDM certificates, stronger than moment fits."""
    from su2_nqs.benchmarks.su2_multipartite_witness import su2_invariant_witness_basis
    cfg=json.loads(config.read_text());primary=ROOT/cfg['primary_directory'];output=ROOT/cfg['output']
    if output.exists():raise FileExistsError('full-RDM audits immutable')
    base=json.loads((primary/'registration.json').read_text())['config'];quality,inputs,bonds=load_quality(base)
    selected=[r for r in quality if r['method']=='DMRG' and r['label'] in cfg['six_site_labels']]
    for case in cfg.get('derived_cases',[]):
        for source in quality:
            if source['method']!='DMRG' or source['parent']!=case['parent'] or source['label'] not in case['labels']:continue
            selected.append(dict(label=source['label'],parent=case['name'],chi=source['chi'],
                rho=partial_trace_sites(source['rho'],case['keep']),
                sites=[source['sites'][i] for i in case['keep']],parties=case['parties']))
    for folder in cfg['fork_directories']:
        folder=ROOT/folder;done=json.loads((folder/'completion.json').read_text())
        report=json.loads((folder/'results.json').read_text());row=report['regions'][0]
        if (report['status']!='complete' or file_sha256(folder/'results.json')!=done['results_sha256'] or
                file_sha256(folder/'regions/fork4/rdm.npz')!=row['rdm_sha256']):
            raise ValueError('completed fork seals differ')
        with np.load(folder/'regions/fork4/rdm.npz') as a:rho=a['rho']
        inputs[str((folder/'results.json').relative_to(ROOT))]=file_sha256(folder/'results.json')
        inputs[str((folder/'regions/fork4/rdm.npz').relative_to(ROOT))]=row['rdm_sha256']
        selected.append(dict(label='DMRG_'+str(row['actual_chi'])+'_fork4',rho=rho,parent='fork4',sites=row['region']['sites'],
            j2=.125,chi=row['actual_chi']))
    output.mkdir(parents=True)
    sources=transitive_local_source_manifest([Path(__file__),ROOT/'su2_nqs/frontier/provenance.py'],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    for name in sources:
        dst=output/'source_snapshots'/name;dst.parent.mkdir(parents=True,exist_ok=True);copy2(ROOT/name,dst)
    _write(output/'registration.json',dict(config=cfg,inputs=inputs,sources=sources,
        scope='Full original RDM certified only by explicit product mixture plus a residual separable-ball check. No physical input averaging.',
        exclusion_of_ViT='Its measured RDM SU2 defects around .03 exceed the available residual-ball radius for this small inflation. Do not repeat a logically impossible average-candidate test.'))
    start=perf_counter();saved={};rows=[]
    for row in selected:
        if perf_counter()-start>cfg['seconds_cap']:raise TimeoutError('full-RDM audit cap exhausted')
        rho=row['rho'];d=len(rho);n=d.bit_length()-1;parties=row.get('parties',tuple((i,) for i in range(n)));t=cfg['inflation']
        if n==4:
            bank=[]
            for i in range(d):
                x=np.zeros((d,d));x[i,i]=1;bank.append(x)
                for j in range(i+1,d):
                    x=np.zeros((d,d));x[i,j]=x[j,i]=1/np.sqrt(2);bank.append(x)
            operators=np.array(bank);average=False
        else:
            operators=su2_invariant_witness_basis(n,True);average=True
        target=np.eye(d)/d+t*(rho-np.eye(d)/d)
        fit,atoms=biseparable_moment_fit(np.einsum('kij,ji->k',operators,target).real,operators,parties,
            iterations=cfg['iterations'],tolerance=cfg['moment_tolerance'])
        cert,completion=biseparable_residual_ball(rho,atoms['weights'],atoms['vectors'],atoms['cuts'],parties,
            inflation=t,shared_su2_average=average)
        key=row['label']+'__'+row['parent']
        for name,array in {**atoms,**completion,'original_rho':rho}.items():saved[key+'__'+name]=array
        geometry=induced_geometry(row['sites'],bonds[.125],.125)
        rows.append(dict(label=row['label'],parent=row['parent'],chi=row['chi'],geometry=geometry,
            inner_fit=fit,certificate=cert,complete_RDM_not_only_low_moments=True))
        print(json.dumps(dict(label=row['label'],region=row['parent'],certificate=cert,seconds=perf_counter()-start)),flush=True)
        _write(output/'progress.json',dict(rows=rows,seconds=perf_counter()-start,complete=False))
    np.savez_compressed(output/'arrays.npz',**saved)
    result=dict(records=rows,seconds=perf_counter()-start,raw_sha256=file_sha256(output/'arrays.npz'),
        new_training=0,new_H_calls=0,new_ED=0,QSL_fingerprint=False,
        interpretation='Biseparability of the stated physical parties in measured RDMs, not full separability and not no global entanglement. Exact decomposition into an explicit biseparable eta and a separable-ball tau closes residuals. Numerical certificate, not interval arithmetic. Finite-chi and sampled-state errors remain separate.')
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),raw_sha256=result['raw_sha256']))
    return result


def published_calibration(config: Path) -> dict:
    """Read the authors' actual RDMs and calibrate the same local instruments.

    MATLAB/HDF5 matrices are transposed back to the published row-major order;
    parties follow verify_rdms.m, converting only one-based labels to zero.
    Provided scripts are read as conventions, never executed.
    """
    from zipfile import ZipFile
    from io import BytesIO
    import h5py
    cfg=json.loads(config.read_text());output=ROOT/cfg['output'];archive=ROOT/cfg['archive']
    if output.exists():raise FileExistsError('published calibration immutable')
    output.mkdir(parents=True)
    sources=transitive_local_source_manifest([Path(__file__),ROOT/'su2_nqs/frontier/provenance.py'],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    for name in sources:
        dst=output/'source_snapshots'/name;dst.parent.mkdir(parents=True,exist_ok=True);copy2(ROOT/name,dst)
    _write(output/'registration.json',dict(config=cfg,archive_sha256=file_sha256(archive),sources=sources,
        interpretation='Independent published Kagome N36 calibration, not a triangular phase label. Source scripts not executed.'))
    start=perf_counter();saved={};records=[]
    with ZipFile(archive) as zipped:
        for case in cfg['cases']:
            with h5py.File(BytesIO(zipped.read('fig5/'+case['rdm_file']))) as f:
                couplings=f['J2List'][()].reshape(-1);k=int(np.argmin(abs(couplings-case['j2'])))
                if abs(couplings[k]-case['j2'])>1e-12:raise ValueError('reference coupling missing')
                array=f[f['rdmList'][()].reshape(-1)[k]][()].T
                rho=array['real']+1j*array['imag'] if array.dtype.names else np.asarray(array,complex)
            with h5py.File(BytesIO(zipped.read('fig5/'+case['gmn_file']))) as f:
                ref=float(f['gmnList'][()].reshape(-1)[k])
            name=case['name'];parties=case['parties'];saved[name+'__rho']=rho
            fit,cert=solve_su2_multipartite_witness(rho,parties,time_reversal_even=True,
                solver_options=dict(time_limit_secs=cfg['baseline_solver_seconds'],max_iters=20000,eps=5e-7))
            validate_witness_certificate(cert);saved[name+'__baseline__W']=cert.witness
            saved[name+'__baseline__Q']=np.array(cert.q_operators)
            row=dict(name=name,j2=case['j2'],reference_GMN=ref,recovered_lower_bound=fit['entanglement_lower_bound'],
                discrepancy=fit['entanglement_lower_bound']-ref,fit=fit,local_probes=[])
            if case.get('measure_all_sites'):
                n=len(rho).bit_length()-1
                for site in range(n):
                    keep=tuple(i for i in range(n) if i!=site)
                    child_parties=[[keep.index(i) for i in p if i!=site] for p in parties]
                    child_parties=[p for p in child_parties if p]
                    for axis in ('discard',*cfg['axes']):
                        if perf_counter()-start>cfg['seconds_cap']:raise TimeoutError('reference calibration cap')
                        if axis=='discard':targets=[partial_trace_sites(rho,keep)];prob=np.array([1.])
                        else:targets,prob=spin_measurement_branches(rho,site,axis)
                        outcomes=[]
                        for k,(target,p) in enumerate(zip(targets,prob)):
                            fit,cert=solve_physical_multipartite_witness(target/p,child_parties,
                                seconds=cfg['child_solver_seconds'],real_operators=True)
                            validate_witness_certificate(cert)
                            key=name+'__'+str(site)+'__'+axis+'__'+str(k)
                            saved[key+'__W']=cert.witness;saved[key+'__Q']=np.array(cert.q_operators)
                            outcomes.append(dict(m=None if axis=='discard' else (-1,1)[k],probability=float(p),
                                conditional_bound=fit['entanglement_lower_bound'],
                                signed_weighted_score=float(-p*fit['expectation']),fit=fit))
                        row['local_probes'].append(dict(site=site,operation=axis,remaining_party_count=len(child_parties),
                            same_party_count=len(child_parties)==len(parties),outcomes=outcomes,
                            signed_score=sum(r['signed_weighted_score'] for r in outcomes)))
            records.append(row)
            print(json.dumps(dict(name=name,reference=ref,recovered=row['recovered_lower_bound'],seconds=perf_counter()-start)),flush=True)
    np.savez_compressed(output/'arrays.npz',**saved)
    result=dict(records=records,seconds=perf_counter()-start,raw_sha256=file_sha256(output/'arrays.npz'),
        source='https://doi.org/10.6084/m9.figshare.29386484',paper='https://www.nature.com/articles/s41467-026-71113-8',
        original_RDMs_repaired=False,triangular_QSL_claim=False,new_optimization=0,new_H_calls=0,
        caveats='Same detector normalization, different model and geometry. Positive repaired lower bounds certify entanglement. Zero does not certify absence. A measured one-spin party disappears: resulting bipartite negativity must not be compared as retained tripartite GME.')
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),raw_sha256=result['raw_sha256']))
    return result


def assistance_analysis(config: Path) -> dict:
    """Physical central-spin assistance and local SU2 charge coherence."""
    cfg=json.loads(config.read_text());primary=ROOT/cfg['primary_directory'];output=ROOT/cfg['output']
    if output.exists():raise FileExistsError('assistance analyses immutable')
    base=json.loads((primary/'registration.json').read_text())['config'];quality,inputs,_=load_quality(base)
    output.mkdir(parents=True)
    sources=transitive_local_source_manifest([Path(__file__),ROOT/'su2_nqs/frontier/provenance.py'],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    for name in sources:
        dst=output/'source_snapshots'/name;dst.parent.mkdir(parents=True,exist_ok=True);copy2(ROOT/name,dst)
    _write(output/'registration.json',dict(config=cfg,inputs=inputs,sources=sources,
        scope='Retrospective fixed physical instrument, no fit, ratios, total-singlet filtering or input repair. Assistance is bipartite, not by itself GME.'))
    start=perf_counter();records=[];controls=[];saved={}
    keys=('assisted_negativity','after_local_charge_negativity','assistance_gain','charge_coherence_loss')
    for case in cfg['cases']:
        for row in quality:
            if row['parent']!=case['parent']:continue
            if perf_counter()-start>cfg['seconds_cap']:raise TimeoutError('assistance budget')
            rho=partial_trace_sites(row['rho'],case['keep'])
            report=localizable_block_negativity(rho,case['center'],case['left'],cfg['axes'])
            se={};jack=[]
            if row['chains'] is not None:
                chains=partial_trace_sites(row['chains'],case['keep']);k=len(chains)
                for matrix in (k*rho-chains)/(k-1):
                    jack.append(localizable_block_negativity(matrix,case['center'],case['left'],cfg['axes']))
                for axis in cfg['axes']:
                    se[axis]={}
                    for key in keys:
                        x=np.array([r['axes'][axis][key] for r in jack]);se[axis][key]=float(np.sqrt((k-1)/k*np.sum((x-x.mean())**2)))
                        saved[case['name']+'__'+row['label']+'__'+axis+'__'+key]=x
            records.append(dict(case=case['name'],**{k:row[k] for k in ('label','method','j2','step','chi')},
                readout=report,whole_chain_SE=se,nonlinear_bias_removed=False,ratio_estimator_used=False))
        folder=ROOT/cfg['control_directory'];done=json.loads((folder/'completion.json').read_text())
        if file_sha256(folder/'arrays.npz')!=done['raw_sha256']:raise ValueError('control seals changed')
        with np.load(folder/'arrays.npz') as arrays:
            prefix=case['name']+'__control__'
            for name in arrays.files:
                if name.startswith(prefix) and name.endswith('__rho'):
                    report=localizable_block_negativity(arrays[name],case['center'],case['left'],cfg['axes'])
                    controls.append(dict(case=case['name'],label=name[len(prefix):-5],readout=report,QSL_claim=False))
    comparisons=[];critical=norm.ppf(1-cfg['alpha']/(2*cfg['family_size']))
    for case in cfg['cases']:
        r=[x for x in records if x['case']==case['name']]
        a=sorted([x for x in r if x['method']=='ViT' and x['j2']==0],key=lambda x:x['step'])
        b=sorted([x for x in r if x['method']=='ViT' and x['j2']==.125],key=lambda x:x['step'])
        c=sorted([x for x in r if x['method']=='DMRG'],key=lambda x:(x['chi'],x['step']))
        for axis in cfg['axes']:
            for key in cfg['primary_readouts']:
                for kind,left,right in [('coupling',a,b),('cross_method',b,c)]:
                    value=lambda x:x['readout']['axes'][axis][key]
                    delta=value(right[-1])-value(left[-1]);drift=sum(np.ptp([value(x) for x in rr]) for rr in (left,right))
                    error=float(np.hypot(left[-1]['whole_chain_SE'].get(axis,{}).get(key,0),right[-1]['whole_chain_SE'].get(axis,{}).get(key,0)))
                    comparisons.append(dict(case=case['name'],axis=axis,readout=key,type=kind,difference=float(delta),SE=error,
                        quality_range_sum=float(drift),threshold=float(drift+critical*error),passes_descriptive_gate=bool(abs(delta)>drift+critical*error)))
    if len(comparisons)!=cfg['family_size']:raise RuntimeError('assistance family count changed')
    np.savez_compressed(output/'arrays.npz',**saved)
    result=dict(records=records,controls=controls,comparisons=comparisons,seconds=perf_counter()-start,
        raw_sha256=file_sha256(output/'arrays.npz'),new_H_calls=0,new_samples=0,new_training=0,QSL_fingerprint=False,
        interpretation='Center outcomes reveal bipartite entanglement hidden by discarding the record. Comparing local q pinching tests dependence of detected negativity on singlet/triplet coherence. Nonlinear plug-in negativity may have positive finite-sampling bias; jackknife SE does not remove it. A vanishing negativity can leave PPT entanglement. No local pair has nontrivial multiplicity.')
    _write(output/'results.json',result)
    _write(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),raw_sha256=result['raw_sha256']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--mechanism-audit',action='store_true')
    parser.add_argument('--local-probe',action='store_true')
    parser.add_argument('--full-rdm-audit',action='store_true')
    parser.add_argument('--published-calibration',action='store_true')
    parser.add_argument('--assistance',action='store_true')
    args=parser.parse_args();result=(assistance_analysis if args.assistance else published_calibration if args.published_calibration else full_rdm_audit if args.full_rdm_audit else local_probe if args.local_probe else mechanism_audit if args.mechanism_audit else refine if args.refine else analyze)(args.config.resolve())
    print(json.dumps(dict(completed=True,seconds=result.get('seconds'))))
