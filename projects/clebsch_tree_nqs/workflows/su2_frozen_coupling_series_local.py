"""Same-region frozen scalar channels across saved coupling/quality series.

Only cached physical compact6 matrices/amplitudes are consumed. No new state,
sampling, Hamiltonian action, spectral repair or remote access is performed.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from su2_nqs.benchmarks.su2_multiplicity_intervention import pair_spin_projectors
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'projects/clebsch_tree_nqs/results'
OLD=BASE/'su2_mechanism_local_20261028'
OUT=OLD/'coupling_series'
SITES=(12,13,14,18,19,20)
TARGETS=(0.,.025,.05,.075,.10,.125,.15,.175)


def panel_summary(values: np.ndarray, leaves: np.ndarray|None, names: tuple[str,...]) -> dict:
    """Actual linear means and joint delete-group covariance; no invented SE."""
    values=np.asarray(values,float)
    if values.shape!=(len(names),) or not np.isfinite(values).all():raise ValueError('finite matching means required')
    result=dict(observable_order=list(names),means=dict(zip(names,values.tolist())),SE_available=leaves is not None)
    if leaves is not None:
        leaves=np.asarray(leaves,float)
        if leaves.ndim!=2 or leaves.shape[1]!=len(names) or len(leaves)<4 or not np.isfinite(leaves).all():
            raise ValueError('at least four finite shared deletion groups required')
        centered=leaves-leaves.mean(0)
        covariance=(len(leaves)-1)/len(leaves)*(centered.T@centered)
        result.update(jackknife_SE=dict(zip(names,np.sqrt(np.diag(covariance)).tolist())),
            joint_jackknife_covariance=covariance.tolist(),deletion_groups=len(leaves))
    else:result['uncertainty_status']='Original mean RDM available, but no retained chain/group deletions for these new observables. Existing unrelated scalar SEs are not reused.'
    for label,numerator,denominator in [('tau_q0','T0','P110'),('tau_q2','T2','P112'),
            ('reference_Pstar_within_q0','Pstar','Ptotal0'),('delta03_Pstar_within_q0','delta03_Pstar','Ptotal0')]:
        i,j=names.index(numerator),names.index(denominator)
        point=float(values[i]/values[j]) if values[j]>1e-12 else None
        result[label]=dict(value=point)
        if leaves is not None and point is not None and np.all(leaves[:,j]>1e-12):
            ratios=leaves[:,i]/leaves[:,j]
            result[label]['jackknife_SE']=float(np.sqrt((len(ratios)-1)/len(ratios)*np.sum((ratios-ratios.mean())**2)))
    return result


def iid_cached_panel(rows: np.ndarray,phi: np.ndarray,q: np.ndarray,ops: np.ndarray,groups: int=32) -> tuple[np.ndarray,np.ndarray]:
    """Node-safe linear observable means/leaves from existing IID query cache."""
    rows,phi,q,ops=map(np.asarray,(rows,phi,q,ops))
    if (rows.ndim!=2 or rows.shape[1]<=max(SITES) or rows.dtype.kind not in 'biu'
            or not np.isin(rows,(0,1)).all() or phi.shape!=(len(rows),64)
            or q.shape!=(len(rows),) or np.iscomplexobj(q) or not np.isfinite(q).all()
            or not np.isfinite(phi).all() or ops.ndim!=3 or ops.shape[1:]!=(64,64)
            or not np.isfinite(ops).all() or isinstance(groups,bool) or not isinstance(groups,int)
            or groups<4 or not len(rows) or len(rows)%groups or np.any(q<=0)):
        raise ValueError('finite64-state physical panel, positive density and >=4 equal IID groups required')
    packed=rows[:,SITES]@(1<<np.arange(5,-1,-1))
    parent=phi[np.arange(len(rows)),packed]
    mass=abs(parent)**2/q
    if not mass.sum()>0:raise ValueError('valid cached IID groups required')
    action=np.einsum('kni,ni->nk',ops[:,packed,:],phi)
    numerators=parent.conj()[:,None]*action/q[:,None]
    point=numerators.sum(0)/mass.sum()
    numgroup=numerators.reshape(groups,-1,len(ops)).sum(1)
    massgroup=mass.reshape(groups,-1).sum(1)
    if np.any(mass.sum()-massgroup<=0):raise ValueError('a delete-group target norm is zero')
    leaves=(numerators.sum(0)-numgroup)/(mass.sum()-massgroup)[:,None]
    return point.real,leaves.real


def main(output: Path|None=None) -> None:
    start=time.monotonic()
    destination=(output or OUT).resolve()
    if not destination.is_relative_to(BASE.resolve()):raise ValueError('analysis output must remain in project results')
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('use a fresh output directory; existing analysis is immutable')
    destination.mkdir(parents=True,exist_ok=True)
    sources={};panels={};records={};inventory=[]
    def seal(path):
        path=Path(path);sources[str(path.relative_to(ROOT))]=file_sha256(path)
    op_path=OLD/'interventions/frozen_operators.npz';six_path=OLD/'interventions/frozen_six_body.npz'
    star_path=OLD/'regions/frozen_projectors.npz'
    for path in (op_path,six_path,star_path):seal(path)
    with np.load(op_path) as z,np.load(six_path) as s,np.load(star_path) as p:
        bank=dict(T=s['T'],T0=s['T_qtotal0'],T2=s['T_qtotal2'],kappa=z['vector_chirality_A_dot_B'],
            P110=z['P_1_1_0'],P112=z['P_1_1_2'],Ptotal0=z['P_1_1_0']+z['P_3_3_0'],
            Pstar=p['compact6_q0_reference_main_projector'])
    ps=pair_spin_projectors(6,(0,3))
    bank.update({'delta03_'+name:sum(x@op@x for x in ps.values())-op for name,op in list(bank.items())})
    names=tuple(bank);ops=np.stack(list(bank.values()))
    def add(label,method,j2,quality,rho,source,key,identity,leaves=None,extra=None):
        if j2 not in TARGETS:raise ValueError('unregistered coupling')
        values=np.einsum('ij,kji->k',rho,ops).real
        summary=panel_summary(values,leaves,names)
        summary.update(label=label,method=method,j2=j2,quality=quality,sites=list(SITES),
            source=str(source.relative_to(ROOT)),source_key=key,identity=identity,
            RDM_trace_error=float(abs(np.trace(rho)-1)),
            RDM_minimum_Hermitian_eigenvalue=float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min()),
            no_strict_entanglement_spectrum=True)
        if extra:summary.update(extra)
        records[label]=summary
        if leaves is not None:panels[label+'__leaves']=leaves
        seal(source)
    # Sealed previously assembled means: intermediate couplings are here even
    # when their original physics.npz files were not synchronized locally.
    merged=BASE/'coupling_quality_review_20261012/latest_su2_series_v1'
    completion=json.loads((merged/'completion.json').read_text())
    for name in ('results.json','readouts.npz'):
        if file_sha256(merged/name)!=completion[name]:raise ValueError('merged source seal changed')
        seal(merged/name)
    metadata=json.loads((merged/'results.json').read_text())
    magnetic_metadata={}
    with np.load(merged/'readouts.npz') as archive:
        for item in metadata['records']:
            problem=item.get('problem',{})
            if item.get('cut')!='compact6' or problem.get('geometry')!='triangular':continue
            if item.get('sites')!=list(SITES) or (problem['rows'],problem['columns'])!=(6,6):
                raise ValueError('do not replace fixed geometry')
            label=item['label'];key=label+'__compact6__rho'
            identity={k:item[k] for k in ('params_sha256','checkpoint_sha256','physical_problem_signature') if k in item}
            quality={k:item[k] for k in ('step','chi','sweep','energy') if k in item}
            add(label,item['method'],float(problem['j2']),quality,archive[key],merged/'readouts.npz',key,identity)
            magnetic_metadata[label]=dict(full_system_structure_factors=item.get('structure_factors'),
                same_parent_energy=item.get('energy'),original_scalar_norm='S(k)=N^-1 sum_ij exp(ik.(ri-rj)) <Si.Sj>, includes3/4 self term',
                estimator='Full vector scalar including coherent transverse exchange; not3*SzSz. See spatial_vit_physics.scalar_structure_records/make_scalar_structure_factors.',
                connected_status='Full scalar preserved as provided; analytic zero-one-point certification is attached only when matching A1/fixed-Sz parent registration is available.')
    # Attach actual retained chain deletions only after matching the same rho.
    physical=[('triangular_0_750',BASE/'coupling_quality_review_20261012/latest_inputs/readouts/tri_j0_continue_step00750_regional/physics.npz'),
        ('triangular_0_1000',BASE/'coupling_quality_review_20261012/latest_inputs/readouts/tri_j0_continue_step01000_regional/physics.npz'),
        ('ViT_J0125_6600',BASE/'spin_spatial_vit_gpu_continuation_6600_v1/analysis/six_site_readout/physics.npz'),
        ('ViT_J0125_7600',BASE/'spin_spatial_vit_gpu_continuation_7600_v1/analysis/witness_regions_v1/physics.npz')]
    for label,path in physical:
        with np.load(path) as data:
            if data['compact6__region_sites'].tolist()!=list(SITES):raise ValueError('physical order mismatch')
            rho=data['compact6__rho'];leaves=np.einsum('nij,kji->nk',data['compact6__jackknife_rhos'],ops).real
        if label in records:
            with np.load(ROOT/records[label]['source']) as source:
                if not np.allclose(rho,source[records[label]['source_key']],rtol=0,atol=2e-14):raise ValueError('different parent RDMs')
            old=records[label];quality=old['quality'];identity=old['identity'];j2=old['j2']
        else:
            reg=json.loads((path.parent/'registration.json').read_text());seal(path.parent/'registration.json')
            cfg=reg['config'];j2=float(cfg['problem']['j2']);quality=dict(step=cfg.get('step'))
            identity={k:reg.get(k,cfg.get(k)) for k in ('params_sha256','checkpoint_sha256')}
        add(label,'NQS',j2,quality,rho,path,'compact6__rho',identity,leaves)
    # Late DMRG sweeps retain genuine same-cut raw RDMs and their own energy.
    # Include them as quality refinements, never as competing physical phases.
    dmrg_dirs=[BASE/'spin_tensor_network_continuation_v2'/name for name in
        ('chi2048_relax','chi3072','chi4096','chi4096_relax','chi5120_probe')]
    dmrg_dirs.append(BASE/'coupling_quality_review_20261012/latest_inputs/dmrg/work/chi10240_from8192_v1')
    for directory in dmrg_dirs:
        registration=json.loads((directory/'registration.json').read_text())
        cfg=registration['config']
        if cfg['regions']['compact6']!=list(SITES) or (cfg['rows'],cfg['columns'],cfg['j2'])!=(6,6,.125):
            raise ValueError('DMRG same-region physical registration mismatch')
        seal(directory/'registration.json')
        for sidecar in sorted((directory/'sweep_readouts').glob('sweep*.json')):
            meta=json.loads(sidecar.read_text());path=sidecar.with_suffix('.npz');seal(sidecar)
            if file_sha256(path)!=meta['rdms_sha256']:raise ValueError('DMRG sweep RDM changed')
            chi,sweep=int(meta['chi']),int(meta['global_sweep']);label=f'DMRG_{chi}_{sweep}';key=f'chi{chi}_compact6'
            with np.load(path) as data:rho=data[key]
            if label in records:
                old=records[label]
                with np.load(ROOT/old['source']) as data:
                    if not np.allclose(rho,data[old['source_key']],rtol=0,atol=2e-14):raise ValueError('DMRG mean/sweep mismatch')
            add(label,'DMRG',.125,dict(chi=chi,sweep=sweep,energy=dict(energy_per_site=meta['energy_per_site'])),
                rho,path,key,{k:meta[k] for k in ('checkpoint_sha256','rdms_sha256') if k in meta})
    # Native physical CT readout caches from the actual last audited states.
    for path in sorted(OLD.glob('ct_A1_compact6*/ct6_*_physical_region.npz')):
        report=json.loads((path.parent/'results.json').read_text())['results'][0]
        j2=.125 if report['tag']=='ct6_j0125' else 0.
        with np.load(path) as data:
            if data['sites'].tolist()!=list(SITES):raise ValueError('CT physical geometry mismatch')
            values,leaves=iid_cached_panel(data['rows'],data['region_amplitudes'],data['proposal'],ops)
            rho=data['rho_Hermitian']
        if np.max(abs(values-np.einsum('ij,kji->k',rho,ops).real))>2e-12:raise ValueError('CT direct/readout mismatch')
        add(report['tag'],'CT',j2,dict(global_step=report['global_step']),rho,path,'rho_Hermitian',
            {k:report[k] for k in ('checkpoint_sha256','parameter_sha256','source_audit_sha256')},leaves,
            dict(cached_IID_groups=True,independent_energy_audit=False,exploratory_readout_only=True))
        seal(path.parent/'results.json')
    # Match full-system benchmarks by exact parameter SHA, not merely by J2.
    audited={}
    for path in (BASE/'remote_readout_review_20261013/inputs/nqs').glob('*/seed1/audits/step*/results.json'):
        meta=json.loads(path.read_text());parent_registration=path.parents[2]/'registration.json'
        if not parent_registration.exists():continue
        reg=json.loads(parent_registration.read_text());cfg=reg.get('config',reg)
        if meta.get('audit_projection')=='gamma_a1' and cfg.get('model',{}).get('target_m2')==0:
            audited[meta.get('params_sha256')]=(path,parent_registration,meta)
    for label,item in records.items():
        item.update(magnetic_metadata.get(label,dict(full_system_structure_factors=None,
            connected_status='No same-parent full-system magnetic cache attached; regional_K is not a substitute.')))
        digest=item['identity'].get('params_sha256')
        if digest in audited and audited[digest][2].get('checkpoint_sha256')==item['identity'].get('checkpoint_sha256'):
            audit_path,parent_path,meta=audited[digest]
            item['full_system_structure_factors']=meta.get('structure_factors')
            item['same_parent_energy']=meta.get('energy')
            item['connected_status']='Analytically equal to the full scalar: translation-A1 plus fixed total Sz=0 gives every local vector mean zero. Self terms are retained; no empirical noisy subtraction or SU2=3SzSz assumption.'
            item['magnetic_parent_evidence']=dict(audit=str(audit_path.relative_to(ROOT)),audit_sha256=file_sha256(audit_path),
                parent_registration=str(parent_path.relative_to(ROOT)),parent_registration_sha256=file_sha256(parent_path))
            seal(audit_path);seal(parent_path)
    seal(ROOT/'su2_nqs/benchmarks/spatial_vit_physics.py')
    for j2 in TARGETS:
        found=[r for r in records.values() if r['j2']==j2]
        inventory.append(dict(j2=j2,available_labels=[r['label'] for r in found],
            same_compact6_physical_RDM=bool(found),has_retained_statistical_panels=any(r['SE_available'] for r in found),
            missing_input=None if found else 'Need actual6x6 PBC parent/checkpoint identity plus compact6 physical rho on sites[12,13,14,18,19,20], preferably original per-chain deletions or IID amplitude/proposal panel. iPEPS compact4 and prefix18/bump18 observables do not substitute.'))
    missing_panels=[]
    for directory in (BASE/'coupling_quality_review_20261012/latest_inputs/readouts').iterdir():
        report_path,registration_path=directory/'results.json',directory/'registration.json'
        if not report_path.is_file() or not registration_path.is_file():continue
        report=json.loads(report_path.read_text());registration=json.loads(registration_path.read_text())
        matching=[r for r in records.values() if r['method']=='NQS' and not r['SE_available']
            and r['identity'].get('params_sha256')==report.get('params_sha256')
            and r['identity'].get('checkpoint_sha256')==report.get('checkpoint_sha256')]
        if matching:
            remote=registration['config']['output'].rstrip('/')+'/physics.npz'
            expected=metadata['inputs_sha256'].get(remote) or report.get('raw_sha256')
            cached=directory/'physics.npz'
            if cached.exists():
                if expected is None or file_sha256(cached)!=expected:raise ValueError('newly synchronized frozen panel SHA mismatch')
                with np.load(cached) as data:
                    if data['compact6__region_sites'].tolist()!=list(SITES):raise ValueError('newly synchronized physical order mismatch')
                    rho=data['compact6__rho'];leaves=np.einsum('nij,kji->nk',data['compact6__jackknife_rhos'],ops).real
                for item in matching:
                    with np.load(ROOT/item['source']) as data:
                        if not np.allclose(rho,data[item['source_key']],rtol=0,atol=2e-14):raise ValueError('synchronized panel is another physical state')
                    item.update(panel_summary(np.einsum('ij,kji->k',rho,ops).real,leaves,names))
                    item['statistical_panel_source']=str(cached.relative_to(ROOT));panels[item['label']+'__leaves']=leaves
                seal(cached)
            else:
                missing_panels.append(dict(labels=[r['label'] for r in matching],
                    original_registered_physics_path=remote,expected_sha256=expected,
                    needed_keys=['compact6__rho','compact6__jackknife_rhos','compact6__region_sites'],
                    local_destination=str(cached.relative_to(ROOT)),
                    action='Synchronize existing frozen file if still available; no new training, samples or H calls are intrinsically required.'))
            seal(report_path);seal(registration_path)
    # These older regional outputs were archived at /share, not the newer
    # home readout tree. Their exact raw-file hashes survive in the merged
    # provenance even when their sidecar folders were not copied locally.
    for step in (8600,9600,10600):
        label=f'triangular_0.125_{step}';item=records[label]
        if item['SE_available']:continue
        remote=f'/share/kim/yx789/nqs-physics-readouts/tri_{step}/regional/physics.npz'
        expected=metadata['inputs_sha256'][remote]
        cached=BASE/f'coupling_quality_review_20261012/latest_inputs/readouts/tri_{step}_regional/physics.npz'
        if cached.exists():
            if file_sha256(cached)!=expected:raise ValueError('legacy registered regional panel SHA mismatch')
            with np.load(cached) as data:
                if data['compact6__region_sites'].tolist()!=list(SITES):raise ValueError('legacy regional order mismatch')
                rho=data['compact6__rho'];leaves=np.einsum('nij,kji->nk',data['compact6__jackknife_rhos'],ops).real
            with np.load(ROOT/item['source']) as data:
                if not np.allclose(rho,data[item['source_key']],rtol=0,atol=2e-14):raise ValueError('legacy panel parent mismatch')
            item.update(panel_summary(np.einsum('ij,kji->k',rho,ops).real,leaves,names))
            item['statistical_panel_source']=str(cached.relative_to(ROOT));panels[label+'__leaves']=leaves;seal(cached)
        else:missing_panels.append(dict(labels=[label],original_registered_physics_path=remote,
            expected_sha256=expected,local_destination=str(cached.relative_to(ROOT)),
            needed_keys=['compact6__rho','compact6__jackknife_rhos','compact6__region_sites'],
            action='Synchronize original registered frozen regional panel, not a newer checkpoint with the same name.'))
    for item in inventory:
        item['has_retained_statistical_panels']=any(r['j2']==item['j2'] and r['SE_available'] for r in records.values())
    groups=defaultdict(list)
    for item in records.values():groups[item['method'],item['j2']].append(item)
    drift=[]
    for (method,j2),items in groups.items():
        items.sort(key=lambda r:(r['quality'].get('step',r['quality'].get('global_step',0)),r['quality'].get('chi',0),r['quality'].get('sweep',0)))
        for a,b in zip(items[:-1],items[1:]):
            sa,sb=a.get('full_system_structure_factors') or {},b.get('full_system_structure_factors') or {}
            magnetic_delta={}
            for key in ('K','M1','M2','M3'):
                va=sa.get(key,{}).get('scalar',{}).get('value');vb=sb.get(key,{}).get('scalar',{}).get('value')
                magnetic_delta[key]=vb-va if va is not None and vb is not None else None
            drift.append(dict(method=method,j2=j2,parent=a['label'],endpoint=b['label'],
                delta={k:b['means'][k]-a['means'][k] for k in names},
                full_system_scalar_structure_delta=magnetic_delta,
                warning='Empirical own-state quality drift; not a statistical error or convergence certificate.'))
    np.savez_compressed(destination/'chain_panels.npz',**panels)
    atomic_json(destination/'results.json',dict(task='fixed_compact6_coupling_channels_v1',records=list(records.values()),
        coupling_inventory=inventory,own_quality_drift=drift,sources_sha256=sources,
        definitions=dict(physical_sites=list(SITES),A=[12,18,19],B=[13,20,14],doubled_spin_labels=True,
            T='(16/3)chi_Achi_B; T=T0+T2',
            T0='P(qA=1,qB=1,qtotal=0) T P; physical total spin S=0',
            T2='P(qA=1,qB=1,qtotal=2) T P; physical total spin S=1',
            kappa='kappaA dot kappaB = -9/4*T0+3/4*T2; four-spin scalar, not pure-six-body',
            delta03='Tr[D_pair03(rho) O]-Tr[rho O], D_pair03=(rho+Swap03 rho Swap03)/2; all outcomes retained',
            intervention_scope='Regional CPTP map: not a global pure variational state or full-A1-restored update. No out-of-region Hamiltonian or distant-response claim.',
            Pstar='Physical qtotal0 rank1 projector frozen on DMRGchi10240, not reselected on each candidate',
            global_structure='N^-1 sum_ij exp(ik.(ri-rj)) connected vector spin correlations, retaining self terms; analytic zero local moments required for equality to the raw scalar'),
        chain_panels_sha256=file_sha256(destination/'chain_panels.npz'),wall_seconds=time.monotonic()-start,
        no_new_training=True,no_new_samples=True,no_new_H_queries=True,
        warning='Frozen low-order kappa and pure-six-body T are different observables. No QSL uniqueness, thermodynamic phase, or precise unmeasured intermediate point is inferred.',
        source_manifest=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))))
    atomic_json(destination/'missing_inputs.json',dict(missing_compact6_couplings=[x for x in inventory if not x['same_compact6_physical_RDM']],
        existing_registered_files_needed_for_covariance=missing_panels,
        required_geometry=dict(rows=6,columns=6,boundary='PBC_both_physical',sites=list(SITES))))
    with (destination/'summary.tsv').open('w') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['label','method','J2','T','T_SE','T0','T2','kappa','kappa_SE','P110','P112','tau0','tau2','delta_Pstar','energy_per_site','full_S_K','full_S_M1','full_S_M2','full_S_M3'])
        for r in sorted(records.values(),key=lambda x:(x['j2'],x['method'],str(x['quality']))):
            m=r['means'];se=r.get('jackknife_SE',{})
            energy=r.get('same_parent_energy') or r['quality'].get('energy') or {}
            sf=r.get('full_system_structure_factors') or {}
            writer.writerow([r['label'],r['method'],r['j2'],m['T'],se.get('T'),m['T0'],m['T2'],m['kappa'],se.get('kappa'),m['P110'],m['P112'],r['tau_q0']['value'],r['tau_q2']['value'],m['delta03_Pstar'],
                energy.get('energy_per_site'),*[sf.get(k,{}).get('scalar',{}).get('value') for k in ('K','M1','M2','M3')]])
    print(json.dumps(dict(records=len(records),inventory=inventory,wall_seconds=time.monotonic()-start),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT,help='Fresh local output directory under project results; existing outputs are never overwritten.')
    main(parser.parse_args().output)
