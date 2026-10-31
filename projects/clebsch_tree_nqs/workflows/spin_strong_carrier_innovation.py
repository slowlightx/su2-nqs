"""Frozen-operator transfer to reoptimized strong carriers, without new ED/MCMC."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.strong_carrier_innovation import operator_innovation
from su2_nqs.benchmarks.spectral_error_tomography import lowest_subspace_state, orthonormal_column_space
from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from .spin_channel_attention import _krylov
from .spin_channel_mechanism import rotate_complement
from .spin_environment_complement import partition_operators
from .spin_n16_physical_fmove_carrier_gate import ROOT, _load_source, _fixed_sz_configurations, _metrics


def _bundle(reference: dict) -> tuple[dict, dict]:
    directory=ROOT/reference['directory'];r=json.loads((directory/'results.json').read_text())
    if r['result_sha256']!=reference['result_sha256'] or r['result_sha256']!=canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'}):
        raise RuntimeError('reference result seal changed')
    if file_sha256(directory/'states.npz')!=r['bundle_sha256']:raise RuntimeError('reference bundle changed')
    for name,digest in r['source_manifest'].items():
        if file_sha256(ROOT/name)!=digest:raise RuntimeError(f'sealed operator source changed: {name}')
    with np.load(directory/'states.npz',allow_pickle=False) as b:arrays={k:b[k] for k in b.files}
    return r,arrays


def run_spin_strong_carrier_innovation(config_path: Path, output: Path) -> dict:
    """Evaluate all preregistered matched-rank cells; never choose a winner to rerun."""
    spec=json.loads(config_path.read_text())
    if spec['task']!='spin_strong_carrier_innovation' or not spec['preregistered_before_run'] or spec['new_monte_carlo_samples'] or spec['remote_training_on_pass']:
        raise ValueError('expected frozen registered exact-small innovation test')
    if spec['anchors']!=['K2','K4','K2_three_mode'] or spec['maximum_rank']!=7:
        raise ValueError('registered anchor/rank family changed')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists():raise FileExistsError(output/'results.json')
    started=perf_counter()
    _,source=_load_source(spec)
    _,old_env=_bundle(spec['environment_reference'])
    old_complement,old_comp=_bundle(spec['complement_reference'])
    phi=source['physical_a1_basis'];basis=_fixed_sz_configurations()
    h1=source['coupling_0_h_a1'];h2=(source['coupling_0.125_h_a1']-h1)/.125
    pi=source['coupling_0.125_carrier_a1']
    operators=[old_env[f'O_mode_{m}'] for m in (1,2,3)];collective=sum(operators)/np.sqrt(3)
    checks=dict(a1_orthonormality_max=float(np.max(abs(phi.conj().T@phi-np.eye(phi.shape[1])))),
                cached_exact_residual_max=0.,old_strong_state_overlap_error_max=0.)
    # Replay the physical product before A1 compression, not P O P times P F P.
    for partition in spec['partitions']:
        cells=tuple(tuple(x) for x in old_complement['config']['partitions'][partition])
        f,of,diag=partition_operators(phi,basis,cells)
        checks[partition]=dict(diag,F_replay_max=float(np.max(abs(f-old_comp[partition+'_F']))),
                               OF_replay_max=float(np.max(abs(of-old_comp[partition+'_OF']))))
        if max(checks[partition].values())>2e-10:raise RuntimeError('physical operator replay failed')
    tolerance=spec['relative_rank_tolerance'];cases={};stored={}
    for coupling in spec['couplings']:
        h=h1+coupling*h2;exact=old_env[f'j{coupling}_exact']
        exact_metrics=_metrics(exact,h)
        error=float(np.linalg.norm(h@exact-exact_metrics['energy_total']*exact))
        checks['cached_exact_residual_max']=max(checks['cached_exact_residual_max'],error)
        if error>2e-10:raise RuntimeError('cached exact-state validation failed')
        k2=_krylov(pi,h,3);c2,_=lowest_subspace_state(k2,h)
        strong,_,_=orthonormal_column_space(np.column_stack((k2,*[op@c2 for op in operators])),relative_tolerance=tolerance)
        anchors=dict(K2=k2,K4=_krylov(pi,h,5),K2_three_mode=strong)
        strong_state,_=lowest_subspace_state(strong,h)
        overlap_error=float(abs(1-abs(np.vdot(strong_state,old_comp[f'original/j{coupling}/base']))))
        checks['old_strong_state_overlap_error_max']=max(checks['old_strong_state_overlap_error_max'],overlap_error)
        if overlap_error>2e-10:raise RuntimeError('old strong-state replay failed')
        subsystem=tuple(range(8))
        exact_rho=reduced_density_matrix(phi@exact,basis,subsystem)
        for partition in spec['partitions']:
            f,of=old_comp[partition+'_F'],old_comp[partition+'_OF']
            def describe(v: np.ndarray, rank: int) -> dict:
                v=v/np.linalg.norm(v);m=_metrics(v,h)
                rho=reduced_density_matrix(phi@v,basis,subsystem)
                m.update(rank=int(rank),energy_error_per_site=m['energy_per_site']-exact_metrics['energy_per_site'],
                    ground_fidelity=float(abs(np.vdot(exact,v))**2),
                    two_cell_rdm_trace_distance=float(.5*np.sum(abs(np.linalg.eigvalsh(rho-exact_rho)))),
                    h1_per_site=float(np.vdot(v,h1@v).real/16),h2_per_site=float(np.vdot(v,h2@v).real/16),
                    connected_OF=float((np.vdot(v,of@v)-np.vdot(v,collective@v)*np.vdot(v,f@v)).real))
                return m
            exact_observable=float((np.vdot(exact,of@exact)-np.vdot(exact,collective@exact)*np.vdot(exact,f@exact)).real)
            for name,anchor in anchors.items():
                key=f'{partition}/j{coupling}/{name}'
                trial=operator_innovation(h,anchor,of,tolerance=tolerance);diag=trial['diagnostics']
                ranks={k:diag[v] for k,v in dict(carrier='anchor_rank',direct='direct_rank',conditional='conditional_rank',h_once='h_once_rank',h_twice='h_twice_rank').items()}
                if max(ranks.values())>spec['maximum_rank'] or max(ranks.values())>=len(h):raise RuntimeError('non-saturation bound failed')
                states={k:trial[k] for k in ranks}
                for label,columns in (
                    ('rank_matched_direct_krylov',_krylov(pi,h,ranks['direct'])),
                    ('rank_matched_conditional_krylov',_krylov(pi,h,ranks['conditional'])),
                    ('F_only',np.column_stack((trial['anchor'],f@trial['carrier']))),
                    ('old_K2_input',np.column_stack((trial['anchor'],of@c2))),
                ):
                    states[label],ranks[label]=lowest_subspace_state(columns,h,relative_tolerance=tolerance)
                row=dict(diagnostics=diag,states={k:describe(v,ranks[k]) for k,v in states.items()},
                         exact_energy_per_site=exact_metrics['energy_per_site'],exact_connected_OF=exact_observable,
                         input_changed_from_K2=bool(name!='K2'),scrambles=[],checks={})
                # Baseline-fixed scrambles only test the independent remainder after H.
                for seed in spec['scramble_seeds']:
                    rotation=rotate_complement(trial['h_basis'],seed)
                    residual=rotation@trial['innovation']
                    candidate,rank=lowest_subspace_state(np.column_stack((trial['h_basis'],residual)),h,relative_tolerance=tolerance)
                    if rank!=ranks['conditional']:raise RuntimeError('scramble rank changed')
                    row['scrambles'].append(dict(_metrics(candidate,h),rank=rank,seed=seed,
                        anchor_error=float(np.max(abs(rotation@trial['h_basis']-trial['h_basis']))),
                        norm_error=float(abs(np.linalg.norm(residual)-np.linalg.norm(trial['innovation'])))))
                def wins(left: str, right: str) -> bool:
                    a,b=row['states'][left],row['states'][right]
                    return bool(a['rank']==b['rank'] and a['energy_per_site']<b['energy_per_site']-spec['energy_margin_per_site'] and a['variance_total']<=b['variance_total']+spec['variance_tolerance_total'])
                row['gates']=dict(direct_beats_H= wins('direct','h_once'),direct_beats_global_Krylov=wins('direct','rank_matched_direct_krylov'),
                    conditional_beats_second_H=wins('conditional','h_twice'),conditional_beats_global_Krylov=wins('conditional','rank_matched_conditional_krylov'))
                conditional=row['states']['conditional']
                row['scramble_joint_wins']=sum(conditional['energy_per_site']<r['energy_per_site']-spec['energy_margin_per_site'] and conditional['variance_total']<=r['variance_total']+spec['variance_tolerance_total'] for r in row['scrambles'])
                row['checks']['anchor_contained_direct_error']=float(np.max(abs(trial['anchor'].conj().T@trial['anchor_residual'])))
                for state in row['states'].values():
                    if state['energy_error_per_site'] < -2e-11:raise RuntimeError('energy below cached exact reference')
                if conditional['energy_per_site']>row['states']['h_once']['energy_per_site']+1e-12:raise RuntimeError('nested-space variational monotonicity failed')
                for label,v in states.items():stored[key+'/'+label]=v
                for label in ('anchor','h_basis','innovation','direction'):stored[key+'/'+label]=trial[label]
                cases[key]=row
                print(f"{key}: outside H={diag['outside_anchor_and_h_fraction']:.4g}; direct-K={row['states']['direct']['energy_per_site']-row['states']['rank_matched_direct_krylov']['energy_per_site']:+.4g}; innovation-K={conditional['energy_per_site']-row['states']['rank_matched_conditional_krylov']['energy_per_site']:+.4g}; joint={row['gates']['conditional_beats_global_Krylov']}; scrambles={row['scramble_joint_wins']}/5",flush=True)
    np.savez_compressed(output/'states.npz',**stored)
    result=dict(task=spec['task'],status='complete',config=spec,config_sha256=canonical_sha256(spec),
        created_utc=datetime.now(timezone.utc).isoformat(),model='triangular 4x4 PBC J1-J2, exact S=0 Gamma-A1, dimension 14',
        checks=checks,cases=cases,summary={k:sum(row['gates'][k] for row in cases.values()) for k in next(iter(cases.values()))['gates']},
        primary_case=spec['primary_case'],case_count=len(cases),wall_seconds=perf_counter()-started,
        full_sector_hamiltonian_diagonalization_performed=False,new_monte_carlo_samples=0,remote_training_licensed=False,
        claim_boundary='Exact-small rank-matched operator innovation after strong-carrier reoptimization. No full-H-work/GPU matching, new ED, training, 6x6 extrapolation or QSL phase identification.',
        bundle_sha256=file_sha256(output/'states.npz'),runtime_identity=numerical_runtime_identity(),
        source_manifest=transitive_local_source_manifest((Path(__file__).resolve(),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    result['result_sha256']=canonical_sha256(result);atomic_json(output/'results.json',result)
    print('Summary:',result['summary'], 'seconds:',result['wall_seconds'],flush=True)
    return result
