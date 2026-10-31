"""Compare frozen rotor, response and initialization diagnostics without training.

This reporting workflow never assigns a bulk phase. Actual total-spin content
comes from the physical Casimir, not from virtual charge or level degeneracy.
All original worker records, checkpoints and source snapshots remain immutable.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from .dmrg_region_probe import _canonical


def _worker(path: Path) -> dict:
    record = json.loads(path.read_text())
    expected = _canonical({k: v for k, v in record.items() if k != 'record_sha256'})
    if record.get('record_sha256') != expected or record.get('status') != 'complete':
        raise ValueError(f'incomplete or changed physical worker: {path}')
    if record['parent_fingerprint_before'] != record['parent_fingerprint_after']:
        raise ValueError('readout changed its physical parent')
    return record


def analyze_large_training(root: Path) -> dict:
    """Read independent N144 audits without choosing a favorable stream."""
    def sealed(path):
        d=json.loads(path.read_text())
        payload={k:v for k,v in d.items() if k!='result_sha256'}
        expected=_canonical(payload)
        if expected!=d.get('result_sha256'):raise ValueError(f'changed neural result: {path}')
        return d
    paths=[root/'scale_staged_20261014'/case/'seed1' for case in ('tri_j0_12x12','tri_j0125_12x12')]
    paths+=[root/'scale_opt_20261014/tri_j0_12x12/seed1']
    records=[];inputs={}
    for directory in paths:
        progress_path=directory/'progress.json';progress=sealed(progress_path)
        inputs[str(progress_path)]=file_sha256(progress_path)
        audits=[]
        for p in sorted((directory/'audits').glob('*/energy.json')):
            a=sealed(p);inputs[str(p)]=file_sha256(p)
            audits.append(dict(step=a['step'],projection=a['audit_projection'],
                params_sha256=a['params_sha256'],checkpoint_sha256=a['checkpoint_sha256'],
                energy=a['energy'],frozen_before_audit=a['frozen_before_audit'],
                audit_never_used_for_training=a['audit_never_used_for_training']))
        records.append(dict(case=directory.parent.name,stage=directory.parent.parent.name,
            completed_steps=progress['completed_steps'],status=progress['status'],
            parameter_count=progress['parameter_count'],active_training_seconds=progress['active_training_seconds'],
            last_training_record=progress['history'][-1],audits=audits,
            latest_endpoint_independently_audited=any(a['step']==progress['completed_steps'] for a in audits)))
    earlier=records[0]['audits'][-1]['energy'];later=records[2]['audits'][-1]['energy']
    se=np.hypot(earlier['chain_mean_standard_error'],later['chain_mean_standard_error'])
    result=dict(records=records,inputs=inputs,J0_latest_minus_previous_energy=later['energy_per_site']-earlier['energy_per_site'],
        independent_combined_standard_error=float(se),
        improvement_in_combined_SE=float((earlier['energy_per_site']-later['energy_per_site'])/se),
        J0_latest_variance_tail_share=later['top_one_percent_variance_share'],
        J0125_remaining_staged_steps=700-records[1]['completed_steps'],
        new_training=0,new_samples=0,competitive_accuracy_certified=False,phase_claim=False)
    output=root/'large_training_review_20261015';output.mkdir(exist_ok=True)
    atomic_json(output/'comparison.json',result)
    return result


def analyze_bond_quadrupoles(root: Path) -> dict:
    """Compare frozen rank-two winding-pi rows with full-support replica traces.

    Aggregate the first TWO pi multiplets before interpreting a row texture:
    near-degenerate even/odd mixtures are not individually stable channels.
    All row combinations in the replica comparison retain paired-chain
    numerator/denominator covariance. No new amplitudes or samples are used.
    """
    from su2_nqs.benchmarks.replica_spin_response import summarize_swap_response
    from su2_nqs.frontier.artifacts import canonical_sha256
    inputs={};mps=[];vit=[]
    for path in sorted((root/'mps').glob('*/worker_result.json')):
        w=_worker(path);r=w['readout'];nrow=r['cut']//6
        q=r['two_copy_bond_quadrupole']
        matrix=np.array(q['purity_normalized_real'])+1j*np.array(q['purity_normalized_imag'])
        sectors=r['leading_spin_content']['sectors'];sp=r['leading_spatial_content']
        if sectors['0']['leading_levels'][0]['higher_spin_weight_upper_bound_nominal']>1e-5:
            raise ValueError('leading m=0 vector does not resolve a singlet')
        pi=np.array(sp['4']['momentum_weights'])[:,3]
        mirror=np.array(sp['4']['mirror_matrix']['real'])+1j*np.array(sp['4']['mirror_matrix']['imag'])
        rows=r['bond_quadrupole_transitions']['rows']
        branches=[]
        for i,x in enumerate(sectors['4']['leading_levels'][:len(pi)]):
            if pi[i]>.95 and x['higher_spin_weight_upper_bound_nominal']<=1e-5:
                amplitude=np.array([complex(a['real'][i][0],a['imag'][i][0]) for a in rows.values()])
                branches.append(dict(index=i,xi=x['entanglement_energy'],pi_weight=float(pi[i]),
                    mirror_expectation=float(mirror[i,i].real),row_strength=(abs(amplitude)**2).tolist(),
                    row_cross_real=np.outer(amplitude.conj(),amplitude).real.tolist()))
        if len(branches)<2:raise ValueError('two resolved pi multiplets required')
        strengths=np.sum([x['row_strength'] for x in branches[:2]],axis=0)
        diag=matrix.diagonal().real
        record=dict(case=path.parent.name,coupling=0. if path.parent.name.startswith('ordered') else .125,
            cut=r['cut'],chi=w['actual_chi'],checkpoint_sha256=w['checkpoint_sha256'],
            purity=r['purity'],rows=list(rows),branches=branches,
            first_two_pi_row_strength=strengths.tolist(),
            first_two_pi_energy_span=branches[1]['xi']-branches[0]['xi'],
            full_two_copy_matrix_real=matrix.real.tolist(),full_two_copy_matrix_imag=matrix.imag.tolist(),
            matrix_hermiticity_error=float(np.max(abs(matrix-matrix.conj().T))),
            physical_vector_orthonormality_error=max(s['physical_orthonormality_error'] for s in sp.values()))
        if nrow==3:
            record.update(full_boundary_average=float((diag[0]+diag[2])/2),full_interior=float(diag[1]),
                full_boundary_minus_interior=float((diag[0]+diag[2])/2-diag[1]),
                full_interior_over_boundary=float(diag[1]/((diag[0]+diag[2])/2)),
                first_two_interior_over_boundary=float(strengths[1]/((strengths[0]+strengths[2])/2)))
        mps.append(record);inputs[str(path)]=file_sha256(path)
    if len(mps)!=6:raise ValueError('all six fixed MPS quality/cut cases required')
    paths=sorted((root/'vit').glob('*/results.json'))
    exchanged=sorted((root/'vit_exchange').glob('*/results.json'))
    if len(paths)!=4 or (exchanged and len(exchanged)!=4):
        raise ValueError('both frozen ViT quality endpoints at each coupling required for each estimator')
    for path in paths+exchanged:
        result=json.loads(path.read_text());rp=path.parent/'registration.json';reg=json.loads(rp.read_text())
        if (result['registration_sha256']!=reg['sha256'] or
                canonical_sha256({k:v for k,v in reg.items() if k!='sha256'})!=reg['sha256']
                or not result['all_inputs_unchanged']):raise ValueError('neural readout seal changed')
        inputs[str(path)]=file_sha256(path);inputs[str(rp)]=file_sha256(rp)
        record=dict(case=path.parent.name,estimator=path.parent.parent.name,
            coupling=0. if path.parent.name.startswith('tri_j0_') else .125,
            checkpoint_sha256=reg['settings']['checkpoint_sha256'],step=reg['settings']['step'],
            energy=result['energy_audit'],regions={})
        for name,raw in result['summaries'].items():
            p=path.parent/(name+'.npz');inputs[str(p)]=file_sha256(p)
            with np.load(p) as z:w=z['swap'];n=z['numerator'];labels=z['chain_pair_labels']
            row_count=len(raw['quadrupole_rows']);mat=n.reshape(-1,row_count,row_count)
            # Real symmetric combinations follow the Hermiticity of the
            # measured quadratic form; never clip negative sampled diagonals.
            combos=[mat[:,j,j] for j in range(row_count)];names=[f'row{j}' for j in range(row_count)]
            if row_count==3:
                boundary=(mat[:,0,0]+mat[:,2,2])/2
                combos += [boundary,mat[:,1,1],boundary-mat[:,1,1],(mat[:,0,2]+mat[:,2,0])/2]
                names += ['boundary_average','interior','boundary_minus_interior','boundary_cross']
            summary=summarize_swap_response(w,np.column_stack(combos),labels)
            summary.update(channel_order=names,definition='Tr(rho Q_a-- rho Q_b++)/Tr(rho^2), exact fixed-Sz but no assumed SU2')
            record['regions'][name]=summary
        vit.append(record)
    result=dict(mps=mps,vit=vit,inputs=inputs,source_sha256=file_sha256(Path(__file__)),
        unique_physical_parents=len({x['checkpoint_sha256'] for x in mps+vit}),
        mps_state_cut_records=len(mps),vit_state_records=len(paths),
        vit_estimator_records=len(vit),
        branch_selector='first two actual Casimir-S2, winding-pi highest-weight vectors; leading physical S0 initial vector',
        boundary_comparison='per-row boundary mean versus single interior row; avoids counting two boundaries as one',
        mirror_selection='Q_row,pi++ has parity(-1)^(row-1); single-branch darkness can be kinematic',
        full_schmidt_trace_not_low_branch_truncation=True,independent_couplings=True,
        new_samples=0,new_training=0,new_ED=0,QSL_fingerprint_claim=False)
    out=root/'analysis';out.mkdir(exist_ok=True);atomic_json(out/'comparison.json',result)
    return result


def analyze_rotor_families(root: Path) -> dict:
    """Frozen S0/S1 parameters predict high-spin levels and spatial channels."""
    from su2_nqs.benchmarks.rotor_family_scan import compare_frozen_spatial_rotor
    records=[];inputs={}
    for path in sorted((root/'mps').glob('*/worker_result.json')):
        w=_worker(path);r=w['readout']
        if not w['spec'].get('total_spin_ladder') or w['spec'].get('spatial_levels')!=16:
            raise ValueError('frozen extended spatial/ladder protocol required')
        result=compare_frozen_spatial_rotor(r)
        record=dict(case=path.parent.name,coupling=0. if path.parent.name.startswith('ordered') else .125,
            cut=r['cut'],chi=w['actual_chi'],checkpoint_sha256=w['checkpoint_sha256'],comparison=result,
            physical_vector_orthonormality_error=max(x['physical_orthonormality_error']
                for x in r['leading_spatial_content'].values()))
        if result['status']=='compared':
            q4=result['observed'][4];x0=result['anchor_xi']
            first={key:next((x for x in q4 if x.get('spatial_weights',{}).get(key,0)>.95),None)
                for key in ('K_star','k_pi')}
            record['spin2_channel_competition']={key:None if x is None else
                dict(index=x['index'],gap=x['xi']-x0,weight=x['spatial_weights'][key])
                for key,x in first.items()}
            if all(first.values()):record['spin2_pi_minus_K_gap']=first['k_pi']['xi']-first['K_star']['xi']
        records.append(record);inputs[str(path)]=file_sha256(path)
    if len(records)!=6:raise ValueError('all six frozen quality/cut readouts required')
    out=root/'analysis';out.mkdir(exist_ok=True)
    result=dict(records=records,inputs=inputs,source_sha256=file_sha256(Path(__file__)),
        rotor_model_source_sha256=file_sha256(Path(inspect.getfile(compare_frozen_spatial_rotor))),
        parameters_fit_only_to_S0_S1=True,full_2D_momentum_assigned=False,
        physical_excitation_gaps_measured=False,phase_claim=False,new_ED=0,new_training=0,new_samples=0)
    atomic_json(out/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'k0_mirror+1':'tab:blue','k0_mirror-1':'tab:blue','K_star':'tab:orange',
        'k_pi':'tab:purple','k_pi_over3_star':'tab:green'}
    latest=[max((x for x in records if x['coupling']==j and x['cut']==cut),key=lambda x:x['chi'])
        for j in (0.,.125) for cut in (18,12)]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained',sharex=True)
    for ax,record in zip(axes.ravel(),latest):
        r=record['comparison'];x0=r['anchor_xi']
        for q in (2,4,6):
            prediction=r['predictions'][str(q)]
            ax.scatter([q/2]*len(prediction),[p['gap'] for p in prediction],facecolors='none',
                edgecolors='black',marker='s',s=65,zorder=2)
            actual=r['observed'][q][:q+1]
            for i,row in enumerate(actual):
                w=row.get('spatial_weights',{});label=max(w,key=w.get) if w else ''
                color=colors.get(label,'0.6') if w.get(label,0)>.95 else '0.6'
                ax.scatter(q/2+.055*(i-(len(actual)-1)/2),row['xi']-x0,color=color,s=35,zorder=3)
        ax.set(title=f'J2={record["coupling"]}, cut={record["cut"]}, chi={record["chi"]}',
            xticks=[1,2,3],xlabel='Physical total spin S',ylabel=r'$\xi-\xi_{S=0}$ (not an H gap)')
        ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker='s',mfc='none',color='black',ls='',label='Frozen rotor prediction')]
    handles += [Line2D([],[],marker='o',color=c,ls='',label=label) for label,c in
        [('k_parallel=0','tab:blue'),('+/-2pi/3','tab:orange'),('pi','tab:purple'),('+/-pi/3','tab:green'),('mixed','0.6')]]
    fig.legend(handles=handles,loc='outside lower center',ncols=3,fontsize=9)
    fig.suptitle('S0/S1 fit does not set S2/S3: count independent multiplets and their spatial channels')
    for ext in ('png','pdf'):fig.savefig(out/f'frozen_rotor_prediction.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    keys=('k0_mirror+1','k0_mirror-1','K_star','k_pi','k_pi_over3_star')
    for ax,cut in zip(axes,(18,12)):
        selected=[r for r in latest if r['cut']==cut]
        data=[];labels=[]
        for r in selected:
            for i,row in enumerate(r['comparison']['observed'][4][:5]):
                data.append([row['spatial_weights'][k] for k in keys]);labels.append(f'J2={r["coupling"]}, S2 #{i+1}')
        ax.imshow(data,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(xticks=range(5),xticklabels=['0,R+','0,R-','+/-2pi/3','pi','+/-pi/3'],
            yticks=range(10),yticklabels=labels,title=f'cut={cut}: first five spin-2 multiplets')
        ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('Non-collective winding channels enter the low spin-2 window; not a phase certificate')
    for ext in ('png','pdf'):fig.savefig(out/f'spin2_spatial_competition.{ext}',dpi=180)
    plt.close(fig)
    return result


def analyze_spatial_labels(root: Path) -> dict:
    """Assign only the spatial labels that actually preserve each strip cut.

    Use physical projectors, not a presumed label from an operator matrix
    element. Real near-degenerate K vectors have both +/-k; label their star
    and retain the measured family translation matrix. No full2D momentum.
    """
    names=('ordered_J0_chi2048_cut18','ordered_J0_chi4096_cut12',
           'ordered_J0_chi4096_cut18','chi8192_cut18','chi10240_cut12','chi10240_cut18')
    records=[];inputs={}
    for name in names:
        p=root/'mps'/name/'worker_result.json';worker=_worker(p)
        r=worker['readout'];s=r['leading_spatial_content'];spin=r['leading_spin_content']
        if worker['spec'].get('spatial_strip_columns')!=6 or r['cut'] not in (12,18):
            raise ValueError('same six-column physical strip required')
        ground=s['0'];triplet=s['2']
        parity=(1 if ground['gamma_parallel_even_weights'][0]>ground['gamma_parallel_odd_weights'][0] else -1)
        g=[ground['gamma_parallel_even_weights'][0],ground['gamma_parallel_odd_weights'][0],
           ground['momentum_weights'][0][2]+ground['momentum_weights'][0][4]]
        initial=spin['sectors']['0']['leading_levels'][0]
        matrices=spin['raising_transitions']['0_to_2']['site_matrices']
        matrix=matrices['row1_kappa_z']
        chirality=(abs(np.array(matrix['real'])+1j*np.array(matrix['imag']))[:,0]**2)[:3]
        branches=[];weights=[g]
        for i,level in enumerate(spin['sectors']['2']['leading_levels'][:3]):
            w=triplet['momentum_weights'][i];even=triplet['gamma_parallel_even_weights'][i];odd=triplet['gamma_parallel_odd_weights'][i]
            star=w[2]+w[4]
            label=('k_parallel=0,mirror+' if even>.95 else 'k_parallel=0,mirror-' if odd>.95
                   else 'k_parallel=+/-2pi/3 star' if star>.95 else 'mixed; no sharp label')
            branches.append(dict(index=i,dominant_spatial_label=label,
                zero_momentum_even_weight=even,zero_momentum_odd_weight=odd,K_parallel_star_weight=star,
                opposite_to_initial_mirror_weight=(odd if parity==1 else even),
                delta_xi=level['entanglement_energy']-initial['entanglement_energy'],
                physical_casimir_expectation=level['physical_casimir_expectation'],
                vector_chirality_squared_matrix_element=float(chirality[i])))
            weights.append([even,odd,star])
        active=int(np.argmax(chirality));a=branches[active]
        family=np.array(triplet['translation_matrix']['real'])+1j*np.array(triplet['translation_matrix']['imag'])
        eigenvalues=np.linalg.eigvals(family[:3,:3])
        records.append(dict(name=name,coupling=0. if name.startswith('ordered') else .125,
            chi=worker['actual_chi'],cut=r['cut'],checkpoint_sha256=worker['checkpoint_sha256'],
            initial_mirror_dominant_sign=parity,initial_spatial_weights=g,
            initial_physical_Casimir=initial['physical_casimir_expectation'],branches=branches,
            plot_weight_rows=weights,chirality_active_index=active,
            chiral_to_other_two_gap_ratio=a['delta_xi']/np.mean([b['delta_xi'] for b in branches if b['index']!=active]),
            family_translation_eigenvalues=dict(real=eigenvalues.real.tolist(),imag=eigenvalues.imag.tolist()),
            family_translation_leakage=(1-np.sum(abs(family[:3,:3])**2,axis=0)).tolist(),
            physical_vector_orthonormality_error=max(v['physical_orthonormality_error'] for v in s.values()),
            full_2D_momentum_assigned=False,exact_spatial_eigenvector_claim=False))
        inputs[str(p)]=file_sha256(p)
    out=root/'analysis';out.mkdir(exist_ok=True)
    result=dict(records=records,inputs=inputs,source_sha256=file_sha256(Path(__file__)),
        group='dihedral group generated by winding translation T and strip-preserving mirror R, not the full2D space group',
        both_couplings_keep_K_star_and_zero_momentum_opposite_mirror_channels=all(
            sum(b['K_parallel_star_weight'] for b in r['branches'])>1.9
            and r['branches'][r['chirality_active_index']]['opposite_to_initial_mirror_weight']>.95
            for r in records),
        one_spin_darkness_has_spatial_selection_rule=True,
        new_low_energy_family_established=False,QSL_identified=False,new_training=0,new_ED=0,new_samples=0)
    atomic_json(out/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for row,coupling in enumerate((0.,.125)):
        for col,cut in enumerate((18,12)):
            r=[x for x in records if x['coupling']==coupling and x['cut']==cut][-1]
            a=np.asarray(r['plot_weight_rows']);ax=axes[row,col]
            im=ax.imshow(a,vmin=0,vmax=1,cmap='Blues',aspect='auto')
            for i in range(4):
                for j in range(3):ax.text(j,i,f'{a[i,j]:.4f}',ha='center',va='center',color='white' if a[i,j]>.6 else 'black',fontsize=9)
            ax.set(xticks=range(3),xticklabels=[r'$k_\parallel=0,R+$',r'$k_\parallel=0,R-$',r'$k_\parallel=\pm2\pi/3$ star'],
                yticks=range(4),yticklabels=['Leading singlet','Triplet1','Triplet2','Triplet3'],
                title=f'J2={coupling}, cut={cut}, chi={r["chi"]}')
            ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('Actual physical Schmidt vectors: the same channels reorder; mirror reference changes with cut')
    fig.colorbar(im,ax=axes,label='Measured physical sector-projector weight',shrink=.75)
    for ext in ('png','pdf'):fig.savefig(out/f'actual_spatial_labels.{ext}',dpi=180)
    plt.close(fig)
    return result


def analyze_chirality_cut(root: Path) -> dict:
    """Frozen vector-chirality readout: quality drift and identical-operator cuts.

    Cross-coupling/endpoint chains are independent; cuts within one endpoint
    reuse pairs and therefore require a paired, denominator-aware jackknife.
    Neither positive replica moments nor Schmidt gaps identify a QSL/roton.
    """
    from su2_nqs.benchmarks.replica_spin_response import summarize_swap_response
    inputs={}; vit={}; mps={}; cuts=('prefix18','bump18','prefix12')
    loop=[[i,6+(i-5)%6] for i in range(6,12)]
    for case,steps in (('tri_j0',(750,1000)),('tri_j0125',(100,200))):
        sequence=[]
        for step in steps:
            path=root/'vit'/f'{case}_{step}'
            r=json.loads((path/'results.json').read_text())
            reg=json.loads((path/'registration.json').read_text())
            if (reg['sha256'] != _canonical({k:v for k,v in reg.items() if k!='sha256'})
                    or r['registration_sha256'] != reg['sha256'] or not r['all_inputs_unchanged']
                    or reg['settings']['step'] != step):
                raise ValueError('changed or mismatched frozen ViT parent')
            data={}; record=dict(step=step,checkpoint_sha256=reg['settings']['checkpoint_sha256'],
                energy_audit=r['energy_audit'],cuts={},paired_cut_contrasts={})
            for cut in cuts:
                s=r['summaries'][cut]
                if s['chirality_bonds']['row1_kappa_z'] != loop:
                    raise ValueError('same physical chirality operator required across all cuts')
                with np.load(path/(cut+'.npz')) as z:
                    w=z['swap'];n=z['numerator'];labels=z['chain_pair_labels']
                a=summarize_swap_response(w,n,labels)
                order=s['channel_order'];index=order.index('row1_kappa_z')
                data[cut]=(w,n[:,index],labels)
                record['cuts'][cut]=dict(channels=dict(zip(order,a['response'])),
                    chain_pair_SE=dict(zip(order,a['response_chain_pair_SE'])),
                    purity=a['purity'],purity_SE=a['purity_chain_pair_SE'],
                    denominator_resolved_at_3SE=a['denominator_resolved_at_3SE'],
                    imaginary_numerator_means=a['imaginary_numerator_means'],
                    region_sites=s['region_sites'],independent_pairs=a['independent_chain_pairs'],
                    swap_tail_share=a['top_one_percent_absolute_swap_share'])
                inputs[str(path/(cut+'.npz'))]=file_sha256(path/(cut+'.npz'))
            for cut in cuts[1:]:
                w0,n0,labels=data['prefix18'];w1,n1,l1=data[cut]
                if not np.array_equal(labels,l1):raise ValueError('paired cuts require identical chain pairing')
                groups=np.unique(labels)
                a=np.array([[w0[labels==g].real.mean(),n0[labels==g].real.mean(),
                             w1[labels==g].real.mean(),n1[labels==g].real.mean()] for g in groups])
                leave=(a.sum(axis=0)-a)/(len(a)-1)
                if np.any(leave[:,[0,2]]<=0):raise ValueError('unresolved paired-cut purity')
                jack=leave[:,3]/leave[:,2]-leave[:,1]/leave[:,0]
                value=a[:,3].mean()/a[:,2].mean()-a[:,1].mean()/a[:,0].mean()
                record['paired_cut_contrasts'][cut]=dict(row1_response_minus_prefix18=float(value),
                    paired_SE=float(np.sqrt((len(a)-1)/len(a)*np.sum((jack-jack.mean())**2))))
            for filename in ('results.json','registration.json'):
                inputs[str(path/filename)]=file_sha256(path/filename)
            sequence.append(record)
        vit[case]=sequence
    for case,names in (('tri_j0',('ordered_J0_chi2048','ordered_J0_chi4096')),
                       ('tri_j0125',('chi8192','chi10240'))):
        sequence=[]
        for name in names:
            for cut in (12,18):
                p=root/'mps'/f'{name}_cut{cut}'/'worker_result.json'
                worker=_worker(p);r=worker['readout'];spec=worker['spec']
                if spec['cut'] != cut or spec['chirality_groups']['row1_kappa_z'] != loop:
                    raise ValueError('physical MPS cut or operator changed')
                channels={k:v['purity_normalized'] for k,v in r['two_copy_chirality_z_response']['channels'].items()}
                spin=r['leading_spin_content'];matrices=spin['raising_transitions']['0_to_2']['site_matrices']
                strength={k:(abs(np.asarray(v['real'])+1j*np.asarray(v['imag']))[:,0]**2).tolist()
                          for k,v in matrices.items()}
                sequence.append(dict(name=name,chi=worker['actual_chi'],cut=cut,channels=channels,
                    K=r['two_copy_spin_response']['modes']['K']['purity_normalized'],
                    initial_Casimir=spin['sectors']['0']['leading_levels'][0]['physical_casimir_expectation'],
                    initial_entanglement_energy=spin['sectors']['0']['leading_levels'][0]['entanglement_energy'],
                    triplet_levels=spin['sectors']['2']['leading_levels'],
                    vector_plus_strength_from_lowest_m0=strength,
                    checkpoint_sha256=worker['checkpoint_sha256']))
                inputs[str(p)]=file_sha256(p)
        mps[case]=sequence
    comparisons={}
    for cut in cuts:
        common=set(vit['tri_j0'][-1]['cuts'][cut]['channels']) & set(vit['tri_j0125'][-1]['cuts'][cut]['channels'])
        comparisons[cut]={}
        for channel in sorted(common):
            values=[vit[c][-1]['cuts'][cut]['channels'][channel] for c in vit]
            se=[vit[c][-1]['cuts'][cut]['chain_pair_SE'][channel] for c in vit]
            drift=[vit[c][-1]['cuts'][cut]['channels'][channel]-vit[c][0]['cuts'][cut]['channels'][channel] for c in vit]
            total_se=float(np.linalg.norm(se));delta=values[1]-values[0]
            comparisons[cut][channel]=dict(J0125_minus_J0=delta,independent_SE=total_se,
                signal_over_SE=delta/total_se if total_se else None,
                same_coupling_observed_drifts=drift,
                contrast_exceeds_sum_of_observed_drifts=abs(delta)>sum(abs(d) for d in drift),
                drift_is_not_a_certified_remaining_error_bound=True)
    out=root/'analysis';out.mkdir(exist_ok=True)
    result=dict(vit_quality_sequences=vit,mps_quality_sequences=mps,coupling_contrasts=comparisons,
        inputs=inputs,analysis_source_sha256=file_sha256(Path(__file__)),
        vector_chirality_not_scalar=True,roton_identified=False,QSL_certified=False,
        entanglement_not_Hamiltonian_response=True,new_ED=0,new_training=0,new_Born_samples=0)
    atomic_json(out/'comparison.json',result)
    _chirality_cut_figures(out,result)
    return result


def _chirality_cut_figures(out: Path, result: dict) -> None:
    """Display errors, quality drift and cuts without bar charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    colors={'tri_j0':'#2878b5','tri_j0125':'#c25b27'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,(case,rows) in zip(axes,result['vit_quality_sequences'].items()):
        x=[r['energy_audit']['energy_per_site'] for r in rows]
        ax.errorbar(x,[r['cuts']['prefix18']['channels']['full18_kappa_z'] for r in rows],
            xerr=[r['energy_audit']['chain_mean_standard_error'] for r in rows],
            yerr=[r['cuts']['prefix18']['chain_pair_SE']['full18_kappa_z'] for r in rows],
            fmt='o-',capsize=3,color=colors[case],label='Ordinary ViT, two saved endpoints')
        reference=[r for r in result['mps_quality_sequences'][case] if r['cut']==18]
        ax.axhspan(min(r['channels']['full18_kappa_z'] for r in reference),
                   max(r['channels']['full18_kappa_z'] for r in reference),alpha=.15,color=colors[case],
                   label='Observed DMRG chi drift (not error bound)')
        ax.axhline(reference[-1]['channels']['full18_kappa_z'],color=colors[case],ls=':')
        for i,(energy,r) in enumerate(zip(x,rows)):ax.annotate(str(r['step']),(energy,r['cuts']['prefix18']['channels']['full18_kappa_z']),xytext=((-24,6) if i==0 else (6,6)),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Independent energy / site',ylabel=r'$Tr(\rho\kappa_z\rho\kappa_z)/Tr\rho^2$',title=('$J_2=0$' if case=='tri_j0' else '$J_2=0.125$'))
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(out/f'direct_vector_chirality_quality.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    cuts=['prefix18','bump18','prefix12']
    for ax,(case,rows) in zip(axes,result['vit_quality_sequences'].items()):
        for j,r in enumerate(rows):
            ax.errorbar(np.arange(3)+(j-.5)*.07,[r['cuts'][c]['channels']['row1_kappa_z'] for c in cuts],
                yerr=[r['cuts'][c]['chain_pair_SE']['row1_kappa_z'] for c in cuts],
                fmt=('o--','s-')[j],capsize=3,alpha=(.5,1)[j],color=colors[case],label=f'ViT step{r["step"]}')
        for j,cut in ((0,18),(2,12)):
            reference=[r for r in result['mps_quality_sequences'][case] if r['cut']==cut]
            ax.plot([j,j],[r['channels']['row1_kappa_z'] for r in reference],'x:',color='black',label='DMRG chi sequence' if j==0 else None)
        ax.set(xticks=range(3),xticklabels=['18-site strip','18-site boundary bump','12-site strip'],
            ylabel='Two-copy response of IDENTICAL six-bond operator',title=case.replace('tri_',''))
        ax.grid(alpha=.2);ax.legend(fontsize=7);ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('Fixed row1 vector chirality; cut-dependent observable, not a bulk phase invariant')
    for ext in ('png','pdf'):fig.savefig(out/f'fixed_operator_cut_dependence.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    for ax,cut in zip(axes,(18,12)):
        for j,case in enumerate(('tri_j0','tri_j0125')):
            r=[r for r in result['mps_quality_sequences'][case] if r['cut']==cut][-1]
            strengths=np.array(r['vector_plus_strength_from_lowest_m0']['row1_kappa_z'][:3])
            gap=np.array([l['entanglement_energy'] for l in r['triplet_levels'][:3]])
            # Remove the common initial level using the separately measured
            # lowest m0 energy. Saving it explicitly keeps the plot traceable.
            gap-=r['initial_entanglement_energy']
            bright=int(np.argmax(strengths))
            positions=j+np.array([-.08,0,.08])
            ax.scatter(positions,gap,s=25+65*strengths/(max(strengths)+1e-300),color=colors[case])
            ax.scatter(positions[bright],gap[bright],s=150,facecolors='none',edgecolors='black')
            ax.annotate('chirality-active',(positions[bright],gap[bright]),xytext=(-25,10),textcoords='offset points',fontsize=7)
        ax.set(xticks=[0,1],xticklabels=['$J_2=0$','$J_2=0.125$'],ylabel=r'$\xi_t-\xi_{s}$ (NOT a physical gap)')
        ax.set_title(f'{cut}-site strip')
        ax.margins(y=.12)
        ax.grid(alpha=.2)
    fig.suptitle('Branch order survives; extreme separation is strongly cut dependent')
    for ext in ('png','pdf'):fig.savefig(out/f'chiral_branch_cut_reordering.{ext}',dpi=180)
    plt.close(fig)


def analyze_schmidt_response(root: Path) -> dict:
    """Resolve physical spin textures and replica response at one fixed cut.

    The first three S~1 vectors are also reported as a family: individual
    eigenvectors in a near-degenerate group have arbitrary basis rotations.
    No entanglement energy is interpreted as a Hamiltonian excitation gap.
    """
    from su2_nqs.benchmarks.replica_spin_response import summarize_swap_response, explicit_half_rdm_cost
    groups = {'J0': ['ordered_J0_chi2048','ordered_J0_chi4096'],
              'J0125': ['chi8192','chi10240']}
    momenta = dict(Gamma=[0,0],K=[1/3,1/3],minus_K=[-1/3,-1/3],
                   M1=[.5,0],M2=[0,.5],M3=[.5,.5])
    coords = np.array([[i%6,i//6] for i in range(18)])
    phases = np.exp(2j*np.pi*coords @ np.array(list(momenta.values())).T)
    k_basis,_ = np.linalg.qr(phases[:,[1,2]])
    inputs = {}; states = {}
    for case, names in groups.items():
        rows=[]
        for name in names:
            path=root/'mps'/name/'worker_result.json'
            worker=_worker(path); data=worker['readout']; spec=worker['spec']
            if (data['cut'] != 18 or data['n_sites'] != 36 or spec['columns'] != 6
                    or data['transition_sites'] != list(range(18)) or spec['spin_momenta'] != momenta):
                raise ValueError('identical physical prefix18, complete sites and momentum convention required')
            spin=data['leading_spin_content']; levels=spin['sectors']['2']['leading_levels']
            initial=spin['sectors']['0']['leading_levels'][0]
            if initial['physical_casimir_expectation'] > 1e-5:
                raise ValueError('initial physical Schmidt vector is not singlet dominated')
            matrices=spin['raising_transitions']['0_to_2']['site_matrices']
            f=np.array([np.asarray(matrices[str(i)]['real'])+1j*np.asarray(matrices[str(i)]['imag'])
                        for i in range(18)])[:,:,0]
            branches=[]
            for index, level in enumerate(levels):
                if level['higher_spin_weight_upper_bound_nominal'] > 1e-5: continue
                vector=f[:,index]; norm=float(np.vdot(vector,vector).real)
                pivot=int(np.argmax(abs(vector))); aligned=vector*np.exp(-1j*np.angle(vector[pivot]))
                branches.append(dict(level=index,
                    delta_xi=level['entanglement_energy']-initial['entanglement_energy'],
                    casimir=level['physical_casimir_expectation'],
                    site_squared_norm=norm,
                    K_shape_fraction=float(np.linalg.norm(k_basis.conj().T @ vector)**2/norm) if norm>0 else None,
                    boundary_squared_fraction=float(np.sum(abs(vector.reshape(3,6)[[0,2]])**2)/norm) if norm>0 else None,
                    fourier_strength={k:float(abs(phases[:,j] @ vector)**2) for j,k in enumerate(momenta)},
                    phase_aligned_site_real=aligned.real.tolist(),phase_aligned_site_imag=aligned.imag.tolist()))
            family=f[:,[b['level'] for b in branches[:3]]]
            total=float(np.linalg.norm(family)**2)
            chiral_path=root/'chirality'/name/'worker_result.json'
            chirality=None
            if chiral_path.exists():
                chiral_worker=_worker(chiral_path)
                if chiral_worker['checkpoint_sha256'] != worker['checkpoint_sha256']:
                    raise ValueError('spin and chirality operators must use the same physical parent')
                chiral_spin=chiral_worker['readout']['leading_spin_content']
                for label in ('0','2'):
                    before=[v['eigenvalue'] for v in spin['sectors'][label]['leading_levels']]
                    after=[v['eigenvalue'] for v in chiral_spin['sectors'][label]['leading_levels']]
                    np.testing.assert_allclose(before,after,atol=1e-12,rtol=0)
                matrices=chiral_spin['raising_transitions']['0_to_2']['site_matrices']
                a=np.array([np.asarray(matrices[f'kappa_row{r}']['real'])
                    +1j*np.asarray(matrices[f'kappa_row{r}']['imag']) for r in range(3)])[:,:,0]
                strengths=abs(a.sum(axis=0))**2
                for branch in branches:
                    index=branch['level']
                    branch['oriented_row_vector_chirality_strength']=float(strengths[index])
                    branch['vector_chirality_row_matrix_elements']=dict(real=a[:,index].real.tolist(),imag=a[:,index].imag.tolist())
                # Fixed comparison: first three triplets, not an arbitrary
                # selection of a higher level to improve an apparent contrast.
                first=branches[:3]
                chiral=max(first,key=lambda b:b['oriented_row_vector_chirality_strength'])
                magnetic=sorted(first,key=lambda b:b['fourier_strength']['K'],reverse=True)[:2]
                chirality=dict(chirality_active_level=chiral['level'],
                    chiral_to_bright_magnetic_xi_ratio=chiral['delta_xi']/np.mean([b['delta_xi'] for b in magnetic]),
                    chiral_strength=chiral['oriented_row_vector_chirality_strength'],
                    definition='sum over 18 oriented x-nearest-neighbor bonds of (Si x Sj)^+; VECTOR chirality, not scalar chirality',
                    thermodynamic_chiral_order_claimed=False)
                inputs[str(chiral_path.relative_to(root))]=file_sha256(chiral_path)
            row=dict(chi=worker['actual_chi'],parent_checkpoint_sha256=worker['checkpoint_sha256'],
                lowest_triplet_branches=branches,
                vector_chirality_mechanism=chirality,
                lowest_three_family_K_fraction=float(np.linalg.norm(k_basis.conj().T @ family)**2/total),
                lowest_three_family_site_weight=(np.sum(abs(family)**2,axis=1)/total).tolist(),
                two_copy_response=data['two_copy_spin_response'],purity=data['purity'],
                rotation_defect=data['purity_normalized_commutator_squared'],
                wall_seconds=worker['full_environment_and_operator_seconds'],
                peak_GiB=worker['worker_peak_rss_bytes']/2**30)
            rows.append(row);inputs[str(path.relative_to(root))]=file_sha256(path)
        states[case]=rows
    replicas={}
    replica_source=('vit_orbit' if all((root/'vit_orbit'/name/'results.json').exists()
                         for name in ('tri_j0','tri_j0125')) else 'vit')
    for name in ('tri_j0','tri_j0125'):
        folder=root/replica_source/name
        result=json.loads((folder/'results.json').read_text())
        with np.load(folder/'prefix18.npz') as z:
            w,n,labels,c=z['swap'],z['numerator'],z['chain_pair_labels'],z['ordinary']
        order=result['summaries']['prefix18']['momentum_order']
        k=[order.index(key) for key in ('K','minus_K')]
        numerator=n[:,k].mean(axis=1)
        ordinary=c[:,k].mean(axis=1)
        sym=summarize_swap_response(w,numerator[:,None],labels)
        eta=float(numerator.real.mean()/w.real.mean()/ordinary.mean())
        jack=[]
        for label in np.unique(labels):
            keep=labels != label
            jack.append(numerator[keep].real.mean()/w[keep].real.mean()/ordinary[keep].mean())
        sym.update(K_star_response=sym['response'][0],K_star_chain_pair_SE=sym['response_chain_pair_SE'][0],
            ordinary_K_SzSz=float(ordinary.mean()),K_response_over_ordinary=eta,
            K_response_over_ordinary_chain_pair_SE=float(np.sqrt((len(jack)-1)*np.var(jack))),
            original_result=result)
        replicas[name]=sym
        for path in (folder/'results.json',folder/'prefix18.npz',folder/'compact6.npz'):
            inputs[str(path.relative_to(root))]=file_sha256(path)
    k0=replicas['tri_j0']; k1=replicas['tri_j0125']
    diff=k1['K_star_response']-k0['K_star_response']
    se=float(np.hypot(k1['K_star_chain_pair_SE'],k0['K_star_chain_pair_SE']))
    result=dict(mps_quality_sequences=states,ordinary_ViT_replicas=replicas,
        replica_input_version=replica_source,
        momentum_convention='reciprocal components (column,row): M1 alternates columns, M2 rows, M3 both; M1/M2 names swap relative to older row-column plots',
        ViT_K_response_coupling_contrast=dict(value=diff,independent_pair_SE=se,signal_over_SE=abs(diff)/se),
        inputs=inputs,analysis_source_sha256=file_sha256(Path(__file__)),
        analysis_support_source_sha256=file_sha256(Path(__file__).resolve().parents[3]/'su2_nqs/benchmarks/replica_spin_response.py'),
        explicit18_cost=explicit_half_rdm_cost(),new_training=0,new_Born_samples=0,new_ED=0,
        exact_SU2_assumed=False,QSL_certified=False,
        caveats=['Schmidt transition weights are NOT dynamical structure-factor poles or physical gaps',
          'Region cut breaks rotations; its different M directions need not agree',
          'K shape of one branch is not a global momentum eigenvalue',
          'Uniform singlet-triplet transition zero is kinematic; the full uniform replica moment is not zero',
          'The last chi change is empirical quality drift, not a certified error bar',
          'Replica errors are whole-chain-pair jackknife, not IID or a mixing certificate',
          'An exact positive response can have a negative noisy individual Monte Carlo estimate; no positivity clipping',
          'Only one neural endpoint per coupling here; neural quality drift not yet established'])
    out=root/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'comparison.json',result)
    _schmidt_response_figures(out,result)
    return result


def _schmidt_response_figures(out: Path, result: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for case,color,label in [('J0','#2878b5',r'$J_2=0$'),('J0125','#c25b27',r'$J_2=0.125$')]:
        rows=result['mps_quality_sequences'][case]
        for j,row in enumerate(rows):
            profile=row['lowest_triplet_branches'][0]['phase_aligned_site_real']
            axes[0].plot(range(18),profile,'o-',ms=3,alpha=.35+.65*j,label=label+fr', $\chi={row["chi"]}$',color=color)
        b=rows[-1]['lowest_triplet_branches'][:8]
        axes[1].plot([v['delta_xi'] for v in b],[v['fourier_strength']['K'] for v in b],'o',label=label,color=color)
        mps=rows[-1]['two_copy_response']['modes']['K']['purity_normalized']
        old=rows[-2]['two_copy_response']['modes']['K']['purity_normalized']
        x=0 if case=='J0' else .125
        axes[2].plot([x,x],[old,mps],'-',lw=3,color=color,alpha=.4)
        axes[2].plot(x,mps,'s',mfc='white',color=color)
        neural=result['ordinary_ViT_replicas']['tri_j0' if case=='J0' else 'tri_j0125']
        axes[2].errorbar(x,neural['K_star_response'],yerr=neural['K_star_chain_pair_SE'],fmt='o',color=color,capsize=3)
    for boundary in (5.5,11.5): axes[0].axvline(boundary,color='.75',lw=1)
    axes[0].set(xlabel='Physical prefix site (three rows)',ylabel=r'$\langle t|s_i^+|s\rangle$ (phase aligned)',title='Softest triplet transition texture')
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel=r'$\xi_t-\xi_s$ (not a physical gap)',ylabel=r'$|\langle t|S_A^+(K)|s\rangle|^2$',title='K weight across higher branches')
    axes[1].legend(fontsize=8)
    axes[2].set(xlabel=r'$J_2/J_1$',ylabel=r'$Q_A(K)$, 18-site cut',title='Squares: MPS; circles: ordinary ViT')
    for ax in axes: ax.grid(alpha=.12)
    for ext in ('png','pdf'):fig.savefig(out/f'transition_replica_bridge.{ext}',dpi=180)
    plt.close(fig)
    if all(row['vector_chirality_mechanism'] is not None
           for rows in result['mps_quality_sequences'].values() for row in rows):
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for j,(case,coupling) in enumerate((('J0',0.),('J0125',.125))):
            rows=result['mps_quality_sequences'][case]
            for field,color,label in (('magnetic','#2878b5','K-bright magnetic pair'),
                                      ('chiral','#c25b27','Vector-chirality-active branch')):
                values=[]
                for row in rows:
                    first=row['lowest_triplet_branches'][:3]
                    if field=='magnetic':
                        selected=sorted(first,key=lambda b:b['fourier_strength']['K'],reverse=True)[:2]
                        value=np.mean([b['delta_xi'] for b in selected])
                    else:
                        selected=max(first,key=lambda b:b['oriented_row_vector_chirality_strength'])
                        value=selected['delta_xi']
                    values.append(value)
                axes[0].plot([coupling]*2,values,'-',lw=4,color=color,alpha=.35)
                axes[0].plot(coupling,values[-1],'o',color=color,label=label if j==0 else None)
            first=rows[-1]['lowest_triplet_branches'][:3]
            for key,color,label in (('K','#2878b5','Spin K'),('chiral','#c25b27','Vector chirality')):
                values=[b['fourier_strength']['K'] if key=='K'
                        else b['oriented_row_vector_chirality_strength'] for b in first]
                axes[j+1].plot([b['delta_xi'] for b in first],values,'o-',ms=5,color=color,label=label)
            axes[j+1].set(xlabel=r'$\xi_t-\xi_s$ (entanglement units)',
                ylabel='Squared physical transition amplitude',title=fr'$J_2={coupling}$, $\chi={rows[-1]["chi"]}$')
            axes[j+1].legend(fontsize=8)
        axes[0].set(xlabel=r'$J_2/J_1$',ylabel=r'$\xi_t-\xi_s$ (NOT an excitation gap)',
                    title='Same first-three-triplet family, reordered')
        axes[0].legend(fontsize=7)
        for ax in axes: ax.grid(alpha=.12)
        for ext in ('png','pdf'):fig.savefig(out/f'magnetic_chiral_reorganization.{ext}',dpi=180)
        plt.close(fig)


def compare_ordered_spatial_controls(root: Path) -> dict:
    """Compare actual J2=0/.125 parents at the SAME physical prefix18.

    Read doubled magnetic sectors m=0,2,4,6 and bound higher-spin admixture
    of each highest-weight vector via its measured Casimir. No RDM twirl,
    exact SU2 assignment, eigensolve, training or phase classification occurs.
    Changes between saved chi endpoints are empirical drift, not error bounds.
    """
    names={
        'J0': [f'tower/ordered_J0_chi{chi}_spatial_v1/worker_result.json' for chi in (2048,4096)],
        'J0125': ['tower/local_chi5120_spatial_v1/worker_result.json']+
                 [f'tower/chi{chi}_spatial_v1/worker_result.json' for chi in (8192,10240)]}
    cases={}; inputs={}
    groups=('boundary_row0','interior_row1','boundary_row2')
    for name, paths in names.items():
        rows=[]
        for relative in paths:
            path=root/relative; worker=_worker(path); data=worker['readout']
            if (data['n_sites']!=36 or data['cut']!=18 or not data['groups_partition_prefix']
                    or any(data['spatial_spin_groups'][g]!=list(range(6*i,6*(i+1)))
                           for i,g in enumerate(groups))):
                raise ValueError('identical physical row-major prefix18 required')
            inputs[relative]=file_sha256(path)
            sectors=data['leading_spin_content']['sectors']
            levels={q:[v for v in sectors[str(q)]['leading_levels']
                       if -1e-8 <= v['higher_spin_weight_upper_bound_nominal'] < 1e-5]
                    for q in (0,2,4,6)}
            if not levels[0] or len(levels[2])<3 or not levels[4]:
                raise ValueError('insufficient Casimir-supported low branches')
            if levels[0][0]!=sectors['0']['leading_levels'][0]:
                raise ValueError('lowest magnetic-sector vector is not singlet dominated')
            base=levels[0][0]['entanglement_energy']
            gap1=levels[2][0]['entanglement_energy']-base
            gap2=levels[4][0]['entanglement_energy']-base
            spread=levels[2][2]['entanglement_energy']-levels[2][0]['entanglement_energy']
            if gap1<=0: raise ValueError('positive singlet-to-triplet branch separation required')
            profile=[levels[2][0]['physical_observable_expectations'][g] for g in groups]
            if abs(sum(profile)-1)>1e-8:
                raise ValueError('triplet physical Sz profile failed closure')
            rows.append(dict(chi=worker['actual_chi'],checkpoint_sha256=worker['checkpoint_sha256'],
                rotation_defect=data['purity_normalized_commutator_squared'],
                first_triplet_profile=profile,interior_polarization=profile[1],
                singlet_triplet_xi_separation=gap1,singlet_spin2_xi_separation=gap2,
                spin2_to_triplet_gap_ratio=gap2/gap1,
                first_three_triplet_spread=spread,triplet_spread_over_lowest_gap=spread/gap1,
                dominant_spin_tolerance=1e-5,strict_SU2_spectrum_claimed=False,
                first_triplet_higher_spin_weight_bound=levels[2][0]['higher_spin_weight_upper_bound_nominal'],
                relative_entanglement_levels={q:[v['entanglement_energy']-base for v in values]
                                             for q,values in levels.items()},
                lowest_branch_Casimirs={q:values[0]['physical_casimir_expectation']
                                       for q,values in levels.items() if values}))
        cases[name]=rows
    metrics=('interior_polarization','singlet_triplet_xi_separation',
             'spin2_to_triplet_gap_ratio','triplet_spread_over_lowest_gap')
    comparisons={}
    for metric in metrics:
        delta=cases['J0125'][-1][metric]-cases['J0'][-1][metric]
        drifts={name:abs(rows[-1][metric]-rows[-2][metric]) for name,rows in cases.items()}
        comparisons[metric]=dict(J0125_minus_J0=delta,last_observed_absolute_drifts=drifts,
            exceeds_sum_of_last_observed_drifts=abs(delta)>sum(drifts.values()),
            drift_is_not_a_certified_quality_error_bound=True)
    result=dict(physical_question='Collective-like triplet cluster versus isolated screened branch at the same half-torus cut',
        cases=cases,comparisons=comparisons,input_sha256=inputs,
        analysis_source_sha256=file_sha256(Path(__file__)),
        actual_Hamiltonian_couplings=[0.,.125],same_geometry=True,
        new_training=0,new_Born_samples=0,new_ED=0,
        exact_SU2_spectra_claimed=False,QSL_certified=False,
        caveats=['This is the physical Schmidt spectrum, not a Hamiltonian excitation spectrum.',
            'Each plotted spin is Casimir-supported dominant spin, not an exact irrep of a repaired RDM.',
            'Anisotropic/finite-region magnetic rotors need not have a small triplet spread.',
            'The J0 control also need not obey the ideal quadratic-rotor inequality.',
            'Same-cut differences exceeding observed chi drift establish finite-parent structure, not a bulk QSL fingerprint.'])
    output=root/'spatial/ordered_comparison';output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for offset,(name,rows) in zip((-.07,.07),cases.items(),strict=True):
        color={'J0':'#2878b5','J0125':'#c25b27'}[name]
        label={'J0':r'$J_2=0$','J0125':r'$J_2=0.125$'}[name]
        profiles=np.asarray([r['first_triplet_profile'] for r in rows])
        axes[0].fill_between(range(3),profiles.min(axis=0),profiles.max(axis=0),color=color,alpha=.14)
        axes[0].plot(range(3),profiles[-1],'o-',color=color,label=label)
        for q,values in rows[-1]['relative_entanglement_levels'].items():
            axes[1].plot([int(q)/2+offset]*min(8,len(values)),values[:8],'.',color=color,
                         label=label if int(q)==2 else None)
    axes[0].axhline(0,color='#777777',lw=.7)
    axes[0].set(xticks=[0,1,2],xticklabels=['Boundary 0','Interior','Boundary 2'],
                ylabel=r'$\langle S^z_{row}\rangle$ in lowest mostly-triplet vector',
                title='Same physical cut; shading: observed chi variation')
    axes[1].set(xlabel='Casimir-supported dominant spin S',ylabel=r'$\xi-\xi_0$',
                xticks=[0,1,2,3],ylim=(-.1,7),title='Physical Schmidt branch organization')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    for ext in ('png','pdf'):fig.savefig(output/f'ordered_vs_frustrated.{ext}',dpi=180)
    plt.close(fig)
    return result


def run(root: Path) -> dict:
    """Collect source seals and explicit interpretation limits, then plot them."""
    sources = []

    def read(relative: str) -> dict:
        path = root / relative
        sources.append(path)
        record = json.loads(path.read_text())
        if 'result_sha256' in record and record['result_sha256'] != _canonical(
                {k: v for k, v in record.items() if k != 'result_sha256'}):
            raise ValueError(f'changed result seal: {path}')
        return record

    controls = read('tower/controls_and_actual_spectra.json')
    bound = read('tower/positive_rotor_bound_quality_v1.json')
    tower = []
    for state in bound['actual']:
        chi = state['chi']
        relative = (f'tower/local_chi{chi}_covariance_v1/worker_result.json'
                    if chi == 5120 else f'tower/chi{chi}_covariance_v1/worker_result.json')
        path = root / relative
        sources.append(path)
        record = _worker(path)
        if record['checkpoint_sha256'] != state['checkpoint_sha256']:
            raise ValueError('Casimir and spectrum have different physical parents')
        data = record['readout']
        leading = {m: data['leading_spin_content']['sectors'][m]['leading_levels'][0]
                   for m in ('0', '2', '4')}
        test = state['conditional_test']
        discrepancy = max(abs(leading[m]['entanglement_energy'] -
                              test['magnetic_sector_minima'][m]) for m in leading)
        if discrepancy > 1e-8:
            raise ValueError('physical Schmidt spectra differ between independent readouts')
        dominant_spin_supported = all(
            -1e-8 <= v['higher_spin_weight_upper_bound_nominal'] < 1e-6
            for v in leading.values())
        tower.append(dict(chi=chi, cut=18, checkpoint_sha256=record['checkpoint_sha256'],
            spin1_gap=test['magnetic_m1_gap'], spin2_gap=test['magnetic_m2_gap'],
            quadratic_rotor_spin2_upper_bound=test['positive_rotor_bound_if_premises_hold'],
            excess_above_quadratic_rotor_bound=test['excess_above_bound_if_premises_hold'],
            spin2_to_spin1_gap_ratio=test['ratio_if_premises_hold'],
            leading_physical_casimir={m: v['physical_casimir_expectation'] for m, v in leading.items()},
            leading_higher_spin_weight_upper_bound_nominal={
                m: v['higher_spin_weight_upper_bound_nominal'] for m, v in leading.items()},
            purity_normalized_rotation_commutator_squared=data['purity_normalized_commutator_squared'],
            maximum_omitted_Gram_weight=max(s['omitted_physical_density_weight']
                for s in data['leading_spin_content']['sectors'].values()),
            physical_spectrum_replay_max_error=discrepancy,
            exact_q_eigenvectors_claimed=False,
            dominant_spin_premises_numerically_supported=dominant_spin_supported,
            dominant_spin_reporting_tolerance=1e-6,
            pure_rotor_template_inconsistent=bool(dominant_spin_supported and
                test['excess_above_bound_if_premises_hold'] > 0),
            bulk_magnetism_excluded=False, QSL_certified=False))
    response = {name: read(f'response/{name}/results.json')
                for name in ('j0_s1000', 'j0125_s200')}
    response_audit = read('response/independent_chain_audit_v2.json')
    rotor_counterexample = read('tower/rotor_with_boundary_counterexample_v1.json')
    teacher = read('architecture/joint_teacher_initialization_v1/results.json')
    topology = read('topology/actual_branch_calibration_v1.json')
    connected = read('connected_structure.json')
    validation = read('validation_and_resources.json')
    spatial = read('spatial/spatial_analysis.json') if (root/'spatial/spatial_analysis.json').exists() else None
    spatial_validation = (read('spatial/validation_and_resources.json')
                          if (root/'spatial/validation_and_resources.json').exists() else None)
    summary = dict(
        physical_question='Does J2=.125 retain a 120-degree collective rotor, or a different low-multiplet organization?',
        geometry='triangular 6x6 PBC; DMRG prefix A=18 is a two-boundary half torus',
        tower_quality_sequence=tower,
        rotor_hypothesis='complete low family H_E=sum_a A_a J_a^2 with positive inertia, all S=0,1,2 present',
        analytic_necessary_bound='Delta_xi(S=2)_min <= 3 Delta_xi(S=1)_min; not sufficient for magnetic order',
        caveats=['leading Casimir bounds are nominal numerical bounds, not certified intervals',
            'quadratic-rotor exclusion at this finite region is not exclusion of every ordered/proximate state',
            'singlet-triplet near-quartets also occur for ordinary dimers or two effective boundary spins',
            'a higher collective-rotation family can coexist with a non-rotor lowest quartet',
            'same-geometry J2=0 large-region spectrum and region/system evolution remain unmeasured',
            'two-vector response Ritz quadrature is source-accessible and has no measured m4/residual',
            'ordinary ViT source is overwhelmingly but not exactly spin one',
            'existing paired kernels are not four verified holonomy sectors of one fixed kernel'],
        response=response,
        response_paired_separation_contrast=response_audit['M_minus_K_separation_coupling_contrast'],
        rotor_with_boundary_counterexample=rotor_counterexample,
        initialization_arms={name: dict(initial=arm['initial'], endpoint=arm['endpoint'],
            completed_steps=arm['completed_steps'], training_seconds=arm['training_seconds'])
            for name, arm in teacher['arms'].items()},
        initialization_model_parameters=teacher['parameter_count'],
        connected_convention=connected['connected_definition'],
        onsite_subtraction_is_separate=connected['offsite_definition'],
        topology_new_phase_claim=topology['QSL_claim'],
        validation_and_resources=validation,
        spatial_followup=spatial, spatial_validation=spatial_validation,
        total_new_ED=0, total_new_Born_sampling=0,
        phase_identification=False,
        inputs={str(p.relative_to(root)): file_sha256(p) for p in sources},
    )
    atomic_json(root / 'summary.json', summary)
    _figures(root, summary, controls, teacher)
    return summary


def _figures(root: Path, summary: dict, controls: dict, teacher: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tower = summary['tower_quality_sequence']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout='constrained')
    x = [r['chi'] for r in tower]
    for field, label, color in (
        ('spin1_gap', r'mostly $S_A=1$: measured Casimir', '#2878b5'),
        ('spin2_gap', r'mostly $S_A=2$: measured Casimir', '#c25b27'),
        ('quadratic_rotor_spin2_upper_bound', r'quadratic rotor bound $3\Delta\xi_1$', '#666666')):
        axes[0].plot(x, [r[field] for r in tower], 'o-', color=color, label=label)
    axes[0].set(xlabel=r'DMRG bond dimension $\chi$', ylabel=r'$\Delta\xi=-\log(\lambda/\lambda_0)$',
        title=r'Actual $J_2/J_1=0.125$, 18-site half torus')
    axes[0].legend(fontsize=8, loc='center right')
    baseline = controls['reference']['q_sectors']['0']['entanglement_energies'][0]
    for q in (0, 2, 4, 6):
        levels = controls['reference']['q_sectors'][str(q)]['entanglement_energies']
        axes[1].plot([q/2]*len(levels), np.array(levels)-baseline, 'o', color='#2878b5',
            ms=5, label='Collective 120-degree singlet' if q == 0 else None)
    actual = tower[-1]
    axes[1].plot([0, 1, 2], [0, actual['spin1_gap'], actual['spin2_gap']],
        's--', color='#c25b27', label=r'Actual $\chi=10240$: lowest branches')
    axes[1].axhline(0, color='#888888', ls=':', label='Crossing-dimer control: flat')
    axes[1].set(xlabel=r'$S_A$ (actual: Casimir-validated dominant spin)', ylabel=r'$\xi-\xi_0$',
        title='Ordered-singlet and dimer controls', ylim=(-.15, 5.8))
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.15)
    for extension in ('png', 'pdf'): fig.savefig(root/f'entanglement_rotor_test.{extension}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    for name, color in (('Kstar', '#2878b5'), ('Mstar', '#c25b27')):
        for ax, metric, ylabel in (
            (axes[0], 'lowest_ritz_from_parent', 'Lowest two-vector response Ritz energy / J1'),
            (axes[1], 'lowest_source_weight', 'Source weight of lower Ritz branch')):
            rows = [summary['response'][s]['response_stars'][name]['metrics'][metric]
                    for s in ('j0_s1000', 'j0125_s200')]
            ax.errorbar([0, .125], [r['value'] for r in rows], yerr=[r['chain_SE'] for r in rows],
                fmt='o-', color=color, capsize=3, label=name)
            ax.set(xlabel=r'$J_2/J_1$', ylabel=ylabel); ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    axes[1].set_ylim(.78, 1.01)
    fig.suptitle('Saved ordinary ViT sources: variational quadrature, NOT certified gaps')
    for extension in ('png', 'pdf'): fig.savefig(root/f'response_ritz_comparison.{extension}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout='constrained')
    for index, (name, arm) in enumerate(teacher['arms'].items()):
        history = arm['history']; color = ('#2878b5', '#c25b27')[index]
        label = 'Preserved parent' if index == 0 else 'Same Born weights, phase jitter'
        for ax, field, title in ((axes[0], 'energy_per_site', 'Independent physical H energy'),
                                (axes[1], 'variance_per_site', 'Physical H variance / N'),
                                (axes[2], 'all_cells_q0', 'All plaquettes simultaneously singlet')):
            ax.plot([r['step'] for r in history], [r[field] for r in history], 'o-',
                ms=3, color=color, label=label)
            ax.set(xlabel='Teacher-training step', title=title); ax.grid(alpha=.15)
    axes[0].axhline(teacher['convention_validation']['reference_energy_per_site'],
        color='#777777', ls=':', label='Saved teacher')
    axes[0].legend(fontsize=7)
    axes[2].set_ylim(0, 1.05)
    fig.suptitle('Unchanged strict-SU2 model: joint magnitude/phase fit, N=16 diagnostic only')
    for extension in ('png', 'pdf'): fig.savefig(root/f'initialization_diagnostic.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--ordered-controls',action='store_true',
                        help='replay the same-cut actual J0 versus .125 comparison')
    parser.add_argument('--schmidt-response',action='store_true',
                        help='resolve physical K/M transitions and two-copy neural response')
    parser.add_argument('--chirality-cut',action='store_true',
                        help='frozen direct vector chirality, quality drift and non-equivalent cuts')
    parser.add_argument('--spatial-labels',action='store_true',
                        help='actual physical Schmidt-vector translation/mirror labels')
    parser.add_argument('--rotor-families',action='store_true',
                        help='frozen S0/S1 fit predicts S2/S3 and spatial multiplicities')
    parser.add_argument('--large-training',action='store_true',help='sealed N144 independent-audit status')
    parser.add_argument('--bond-quadrupoles',action='store_true',help='row-resolved S2/pi and matching two-copy response')
    args = parser.parse_args()
    if args.bond_quadrupoles:
        result=analyze_bond_quadrupoles(args.input);count=len(result['mps'])+len(result['vit'])
    elif args.large_training:
        result=analyze_large_training(args.input);count=len(result['records'])
    elif args.rotor_families:
        result=analyze_rotor_families(args.input);count=len(result['records'])
    elif args.spatial_labels:
        result=analyze_spatial_labels(args.input);count=len(result['records'])
    elif args.chirality_cut:
        result=analyze_chirality_cut(args.input)
        count=sum(len(rows) for rows in result['vit_quality_sequences'].values())
    elif args.schmidt_response:
        result=analyze_schmidt_response(args.input)
        count=sum(len(rows) for rows in result['mps_quality_sequences'].values())
    elif args.ordered_controls:
        result=compare_ordered_spatial_controls(args.input)
        count=sum(len(rows) for rows in result['cases'].values())
    else:
        result=run(args.input)
        count=len(result['tower_quality_sequence'])
    if args.bond_quadrupoles:
        print(json.dumps(dict(state_cut_estimator_records=count,
            unique_physical_parents=result['unique_physical_parents'],phase_identification=False)))
    else:
        print(json.dumps(dict(states=count,phase_identification=False)))
