"""Fresh full-A1 refinement of an existing enriched18|126 coefficient cut.

The learned9->18 projectors are fixed. A three-state raw-orthonormal span
contains the parent, an old-column direction about that SAME enriched parent,
and an additional-column direction. Coherent36/72/root restoration is exact.
Fit, amplitude-only calibration, and final audit use separate fresh IID draws.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter, process_time

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_active_layout import repack_active_clebsch_bank
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.su2_complement import descend_singlet_complement, restore_singlet_parent
from su2_nqs.operators.su2_conditional_response import root_coefficient_matrices
from su2_nqs.operators.su2_sector_merge import SectorMergeOperator
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_orbit_exchange_bridge_products
from su2_nqs.vmc.linear_orbit_refinement import (make_orbit_amplitude_products,
    protected_center_directions, raw_normalized_line_coefficients, calibrate_shared_projected_radius)
from su2_nqs.vmc.projected_spin_linear import fit_parent_anchored_linear_update
from su2_nqs.vmc.triangular_raw_born import make_triangular_raw_born_sampler
from .triangle_hierarchy_audit import checked_active_widths
from .triangle_hierarchy_initialization import native_bank_from_state, split_bond_ledger, _node_timeout
from .triangle_iid_orbit_audit import paired_iid_moments
from .triangle_iid_orbit_overlap import paired_projected_overlap
from .triangle_lower_path import ROOT, TREE, clone, cross_bonds, real_maps, reproject
from .triangle_sector_transfer import load_source

CFG = MixedClebschConfig(12, 20, 10)
GUIDE_WEIGHTS = np.asarray([.8, .1, .1])
PARENT_SHA = '6525f98a05a061af2d7e59ea8f024cf0fd2aab46e0d684e19faccf79a16d1a31'


def _restore(context: dict, center: np.ndarray, source: dict, joins: list) -> tuple[dict, dict, dict]:
    """Restore an18-cut coefficient vector through both ancestors, without truncation."""
    parent = context['parent']; env18 = context['env18']; env36 = context['env36']
    coefficients = {q: np.zeros_like(c, dtype=np.result_type(center, c))
                    for q, c in env18.coefficients.items()}
    for value, (q, a, qr, b) in zip(center, context['channels'], strict=True):
        if q != qr: raise ArithmeticError('singlet center has mismatched doubled q')
        coefficients[q][a, b] = value
    maps36, root36, proof36 = restore_singlet_parent(env18, coefficients,
        parent['blocks'][16].q_widths, parent_widths={q: w.shape[1] for q, w in parent['maps'][8].items()})
    maps72, root72, proof72 = restore_singlet_parent(env36, root36,
        {q: w.shape[1] for q, w in maps36.items()},
        parent_widths={q: w.shape[1] for q, w in parent['maps'][12].items()})
    state = clone(parent); state['maps'][8] = real_maps(maps36); state['maps'][12] = real_maps(maps72)
    root = np.asarray([root72[q][a, b] for q, a, qr, b in parent['channels'][14][0]])[:, None]
    state['maps'][14][0] = real_maps({0: root})[0]
    for node in (8, 12, 14): reproject(state, node, joins)
    bank, embedding = native_bank_from_state(source, state)
    occupation = {q: float(np.sum(abs(coefficients[q][8:])**2)) for q in (0, 2, 4)}
    return state, bank, dict(embedding=embedding, restore36=proof36, restore72=proof72,
        raw_energy=float(state['blocks'][30].h_by_q[0][0, 0].real),
        raw_added18_probability_by_q=occupation, raw_added18_probability=sum(occupation.values()),
        occupancy_is_raw_not_projected=True, existing18_W_unchanged=True)


def _freeze(output: Path, extra: dict) -> dict:
    paths = [Path(__file__), ROOT/'su2_nqs/vmc/linear_orbit_refinement.py',
        ROOT/'su2_nqs/vmc/projected_spin_linear.py', ROOT/'su2_nqs/vmc/raw_tree_born.py',
        ROOT/'su2_nqs/vmc/triangular_raw_born.py', ROOT/'su2_nqs/vmc/exchange_bridge_importance.py',
        ROOT/'su2_nqs/operators/su2_complement.py',
        ROOT/'projects/clebsch_tree_nqs/workflows/triangle_iid_orbit_audit.py',
        ROOT/'projects/clebsch_tree_nqs/workflows/triangle_iid_orbit_overlap.py',
        ROOT/'su2_nqs/models/triangular_clebsch_exchange_cache.py',
        ROOT/'su2_nqs/models/triangular_clebsch_feature_lut.py',
        ROOT/'su2_nqs/operators/su2_sector_merge.py',
        ROOT/'projects/clebsch_tree_nqs/workflows/triangle_hierarchy_initialization.py',
        ROOT/'projects/clebsch_tree_nqs/workflows/triangle_lower_path.py']
    registration = dict(extra, source_hashes={str(p.relative_to(ROOT)): file_sha256(p) for p in paths})
    atomic_json(output/'registration.json', registration)
    frozen = output/'registered_sources'; frozen.mkdir()
    for path in paths: shutil.copy2(path, frozen/path.name)
    return registration


def prepare(parent_directory: Path, output: Path) -> dict:
    """Bounded actual CT12 export/replay preflight; no full-A1 H queries or fitting."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('fresh preflight directory required')
    parent_bank = parent_directory/'matched_radius/initialized_bank.pkl'
    if file_sha256(parent_bank) != PARENT_SHA: raise ValueError('frozen enriched parent identity mismatch')
    registration = _freeze(output, dict(task='CT12_exact_existing18cut_three_state_preflight',
        parent_checkpoint=str(parent_bank.resolve()), parent_sha256=PARENT_SHA,
        parent_state=str((parent_directory/'matched_radius/state.pkl').resolve()),
        parent_state_sha256=file_sha256(parent_directory/'matched_radius/state.pkl'),
        old_rows='a<8 in q0,2,4; all rows in other retainedq',
        centers=[18], restored_ancestors=[36,72,144], old18_W_fixed=True,
        maximum_wall_seconds=600, no_prior_audit_products_used=True, full_A1_H_queries=0))
    start = perf_counter(); cpu = process_time(); report = dict(registration=registration, status='started')
    with _node_timeout(600):
        source, provenance = load_source(.125)
        parent = load_checkpoint(registration['parent_state'])['params']
        _, joins, bonds = split_bond_ledger(12, .125); supports = TREE.coarse_supports
        roots = root_coefficient_matrices(parent['maps'][14][0], parent['channels'][14][0])
        env36 = descend_singlet_complement(parent['blocks'][24], parent['blocks'][25], parent['blocks'][29],
            parent['maps'][12], roots, cross_bonds(supports[25], supports[29], bonds))
        support108 = supports[25]+supports[29]
        env18 = descend_singlet_complement(parent['blocks'][16], parent['blocks'][17], env36.complement,
            parent['maps'][8], env36.coefficients, cross_bonds(supports[17], support108, bonds))
        op = SectorMergeOperator(parent['blocks'][16], env18.complement,
            cross_bonds(supports[16], supports[17]+support108, bonds), 0, memory_limit_bytes=768*1024**2)
        center = np.asarray([env18.coefficients[q][a,b] for q,a,qr,b in op.channels])
        hcenter = op.matvec(center)
        mask = np.asarray([q not in (0,2,4) or a < 8 for q,a,qr,b in op.channels])
        basis, direction_report = protected_center_directions(center, hcenter, mask)
        if np.max(abs(basis.imag), initial=0.) > 2e-11: raise ArithmeticError('real native family cannot discard complex tangents')
        basis = basis.real
        hbasis = np.column_stack((hcenter, op.matvec(basis[:,1]), op.matvec(basis[:,2])))
        context = dict(parent=parent, env36=env36, env18=env18, channels=op.channels, basis=basis, Hbasis=hbasis)
        banks = []; proofs = []; states = []
        for index in range(3):
            state, bank, proof = _restore(context, basis[:,index], source, joins)
            expected = np.vdot(basis[:,index], hbasis[:,index]).real
            if abs(proof['raw_energy']-expected) > 2e-8: raise ArithmeticError('restored root H disagrees with exact cut H')
            proof['independent_root_H_error'] = float(abs(proof['raw_energy']-expected))
            save_checkpoint(output/f'basis_{index}.pkl', bank, metadata=dict(embedding=proof['embedding'],
                model=dict(length=12,multiplicity_cap=20,maximum_q=10), source_parent_sha=PARENT_SHA))
            banks.append(bank); proofs.append(proof); states.append(state)
        gram = np.asarray([[triangular_raw_overlap(a,b,CFG) for b in banks] for a in banks])
        original = load_checkpoint(parent_bank)['params']
        parent_overlap = triangular_raw_overlap(original, banks[0], CFG)
        if max(np.max(abs(gram-np.eye(3))), abs(parent_overlap-1)) > 2e-10:
            raise ArithmeticError('native exported span or parent lost exact raw state')
        profile = checked_active_widths(proofs[0]['embedding']['active_top_widths'], nodes=15)
        active = [repack_active_clebsch_bank(b,CFG,profile) for b in banks]
        sampler = make_triangular_raw_born_sampler(active[0],CFG,active_top_widths=profile)
        rng = np.random.default_rng(2610269401); rows=np.asarray([sampler.draw(rng).bits for _ in range(4)])
        coefficient = np.asarray([.99,.08,-.05]); coefficient /= np.linalg.norm(coefficient)
        test_state, test_bank, proof = _restore(context,basis@coefficient,source,joins)
        raw = jax.jit(jax.vmap(make_feature_lut_amplitude(CFG,active_top_widths=profile),in_axes=(None,0)))
        basis_amplitudes = np.column_stack([np.asarray(raw(prepare_feature_lut_bank(b,CFG),rows)) for b in active])
        actual = np.asarray(raw(prepare_feature_lut_bank(repack_active_clebsch_bank(test_bank,CFG,profile),CFG),rows))
        expected = basis_amplitudes@coefficient
        error = float(np.max(abs(actual-expected))/max(np.max(abs(expected)),1e-300))
        energy_error = abs(proof['raw_energy']-np.vdot(coefficient,basis.T@hbasis@coefficient).real)
        if max(error,abs(triangular_raw_overlap(test_bank,test_bank,CFG)-1)) > 2e-10 or energy_error > 2e-8:
            raise ArithmeticError('actual CT12 linear combination export failed physical replay')
        save_checkpoint(output/'context.pkl',context,metadata=dict(registration=registration))
        np.savez(output/'span_replay.npz',basis=basis,Hbasis=hbasis,raw_gram=gram,rows=rows,
            basis_amplitudes=basis_amplitudes,test_coefficients=coefficient,test_amplitude=actual)
        report.update(status='complete_exact18cut_span_preflight',source=provenance,directions=direction_report,
            center_dimension=op.dimension,basis=proofs,maximum_raw_Gram_error=float(np.max(abs(gram-np.eye(3)))),
            exported_parent_overlap_error=float(abs(parent_overlap-1)),linear_amplitude_relative_error=error,
            linear_combination_root_H_error=float(energy_error),coarse_cut_H_matvecs=3,
            context_sha256=file_sha256(output/'context.pkl'),
            basis_checkpoint_sha256=[file_sha256(output/f'basis_{i}.pkl') for i in range(3)],
            active_profile=profile,full_A1_H_queries=0,fit_records=0)
    report.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu)
    atomic_json(output/'results.json',report)
    return report


def run(preflight: Path, output: Path, *, seconds_cap: float = 3600.) -> dict:
    """Registered24 fit /32 amplitude-calibration /cost-selected final IID audit."""
    if not 600 <= seconds_cap <= 3600: raise ValueError('600..3600second total numerical cap required')
    prior = json.loads((preflight/'results.json').read_text())
    if prior['status'] != 'complete_exact18cut_span_preflight': raise ValueError('successful exact export preflight required')
    if file_sha256(preflight/'context.pkl') != prior['context_sha256']:
        raise ValueError('exact cut/export context identity changed')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('new immutable numerical run directory required')
    for i, expected in enumerate(prior['basis_checkpoint_sha256']):
        if file_sha256(preflight/f'basis_{i}.pkl') != expected: raise ValueError('basis identity changed')
    controls = dict(damping=.1,metric_rcond=1e-10,maximum_fs_sine=float(np.sin(.03)),
        initial_step=1.,maximum_backtracks=10,variance_weight=0.,maximum_relative_variance_increase=.05)
    registration = _freeze(output,dict(task='CT12_single_existing18cut_full_A1_refinement_v1',
        preflight=str(preflight.resolve()),preflight_results_sha256=file_sha256(preflight/'results.json'),
        parent_sha256=PARENT_SHA,basis_checkpoint_sha256=prior['basis_checkpoint_sha256'],
        guide_weights=GUIDE_WEIGHTS.tolist(),bridge_fraction=.5,physical_bonds=864,images=1728,
        fit_records=24,calibration_records=32,audit_records_candidates=[96,80,64],
        audit_cost_rule='largestN with1.2*(3*(24+N)+2)*maxwarmH+180seconds below remaining total numerical cap',
        fit_seed=2610269411,calibration_seed=2610269421,audit_seed=2610269431,
        calibration_bootstrap_seed=2610269422,audit_bootstrap_seed=2610269441,
        total_numerical_seconds_cap=seconds_cap,preflight_seconds_charged=prior['wall_seconds'],
        context_sha256=prior['context_sha256'],
        fit_controls=controls,raw_FS_cap=.08,calibration_A1_FS_cap=.03,
        prior128audit_never_loaded=True,old_control_is_old_direction_about_same_enriched_parent=True,
        both_arms_exact_parent_skip_fresh_H_audit=True,
        W18_fixed=True,changed_maps=[8,12,14],no_remote_job_no_attention_no_neighboring_path=True))
    start=perf_counter();cpu=process_time();end=start+seconds_cap-prior['wall_seconds']
    result=dict(registration=registration,status='started',timings={},full_H_queries=0,amplitude_only_queries=0)
    def timed(name,callback):
        if perf_counter() >= end: raise TimeoutError('total numerical wall gate before '+name)
        atomic_json(output/'progress.json',dict(result,operation=name,wall_seconds=perf_counter()-start))
        tick=perf_counter();value=callback();jax.block_until_ready(value)
        result['timings'][name]=perf_counter()-tick
        return value
    try:
        native=[load_checkpoint(preflight/f'basis_{i}.pkl')['params'] for i in range(3)]
        profile=checked_active_widths(prior['active_profile'],nodes=15)
        norms=np.asarray([complex(triangular_raw_overlap(b,b,CFG)).real for b in native])
        if np.max(abs(norms-1))>2e-10: raise ArithmeticError('raw guide basis norm changed')
        active=[repack_active_clebsch_bank(b,CFG,profile) for b in native]
        prepared=[prepare_feature_lut_bank(b,CFG) for b in active]
        images=np.asarray(timed('complete_A1_group',lambda:triangular_hnf_gamma_a1_group(((12,0),(0,12)))),np.int32)
        inverse=np.argsort(images,axis=1); bonds=triangular_j1j2_bonds(12,12,j2=.125,periodic=True)
        edges=np.asarray([(a,b) for a,b,_ in bonds]);bp=np.abs([v for _,_,v in bonds]);bp/=bp.sum()
        action,metadata=make_orbit_exchange_bridge_products(
            lambda e:make_feature_lut_exchange_queries(CFG,e,active_top_widths=profile),images,bonds,bp,n_sites=144,projection_chunk=4)
        amp_action=make_orbit_amplitude_products(make_feature_lut_amplitude(CFG,active_top_widths=profile),images,projection_chunk=4)
        result['kernel_metadata']=metadata
        probe=np.load(preflight/'span_replay.npz')['rows'][:1]
        warm=[]
        for i in (0,2):
            for kind in ('cold','warm'):
                value=timed(f'cost_H_{i}_{kind}',lambda i=i:action(prepared[i],probe));result['full_H_queries']+=1
                if not all(np.isfinite(np.asarray(x)).all() for x in value): raise ArithmeticError('nonfinite cost products')
                if kind=='warm':warm.append(result['timings'][f'cost_H_{i}_{kind}'])
        candidates=[n for n in (96,80,64) if 1.2*(3*(24+n)+2)*max(warm)+180 < end-perf_counter()]
        if not candidates: raise TimeoutError('registered full-H forecast cannot fit64auditrecords')
        count=candidates[0];result['chosen_audit_records']=count
        atomic_json(output/'cost_only_allocation_before_fit.json',dict(audit_records=count,warm_seconds=warm,
            remaining_seconds=end-perf_counter(),fit_not_evaluated=True,rule=registration['audit_cost_rule']))

        def draw_rows(name,n,seed,bridge,group_count):
            cache={};samplers=[make_triangular_raw_born_sampler(b,CFG,active_top_widths=profile,
                shared_tensor_cache=cache) for b in active]
            rows=[];groups=[];latent=[];errors=[]
            for group,child in enumerate(np.random.SeedSequence(seed).spawn(group_count)):
                rng=np.random.default_rng(child)
                for _ in range(n//group_count):
                    index=int(rng.choice(3,p=GUIDE_WEIGHTS));draw=samplers[index].draw(rng)
                    image=int(rng.integers(len(images)));row=draw.bits[inverse[image]].copy()
                    use=bool(bridge and rng.random()<.5);bond=int(rng.choice(len(bonds),p=bp))
                    if use:
                        a,b=edges[bond];row[a],row[b]=row[b],row[a]
                    rows.append(row);groups.append(group);latent.append((index,image,int(use),bond))
                    errors.append(draw.log_probability_replay_error)
            rows=np.asarray(rows,np.int32);groups=np.asarray(groups)
            np.savez(output/f'{name}_draws.npz',rows=rows,groups=groups,latent=np.asarray(latent),log_Born_errors=errors)
            del samplers,cache;gc.collect()
            return rows,groups

        def products(name,rows,full_H):
            values=[]
            for k,row in enumerate(rows):
                record=[]
                for i in range(3):
                    call=action if full_H else amp_action
                    value=timed(f'{name}_{k}_{i}',lambda i=i,row=row:call(prepared[i],row[None]))
                    result['full_H_queries' if full_H else 'amplitude_only_queries']+=1
                    record.append([complex(x[0]) for x in value])
                values.append(record)
                np.savez(output/f'{name}_partial_products.npz',products=np.asarray(values))
            values=np.asarray(values)
            if not np.isfinite(values).all():raise ArithmeticError('nonfinite untouched products')
            phi=values[:,:,0]/np.sqrt(norms)
            if full_H:
                hp=values[:,:,1]/np.sqrt(norms);orbit=values[:,:,2].real/norms;push=values[:,:,3].real/norms
                q=((orbit+push)@GUIDE_WEIGHTS)/2
            else:
                hp=None;orbit=values[:,:,1].real/norms;push=None;q=orbit@GUIDE_WEIGHTS
            if np.any(orbit<0) or np.any(q<=0) or (push is not None and np.any(push<0)):
                raise ArithmeticError('positive exact proposal density required')
            guide=values[:,:,2:] if full_H else values[:,:,1:]
            if np.max(abs(guide.imag))>2e-10*max(float(np.max(abs(guide))),1e-300):
                raise ArithmeticError('positive guide unexpectedly has a material imaginary component')
            fields=dict(phi=phi,proposal=q,raw_products=values,rows=rows)
            if full_H:fields['Hphi']=hp
            np.savez(output/f'{name}_products.npz',**fields)
            return phi,hp,q

        fit_rows,_=draw_rows('fit',24,registration['fit_seed'],True,8)
        fit_phi,fit_hp,fit_q=products('fit',fit_rows,True)
        fitted=[];fit_reports=[];raw_fit_norms=[]
        for width in (2,3):
            parent=np.r_[1.,np.zeros(width-1)]
            coefficient,fit=fit_parent_anchored_linear_update(fit_phi[:,:width],fit_hp[:,:width],fit_q,parent,**controls)
            coefficient=np.pad(coefficient,(0,3-width));raw_fit_norms.append(float(np.linalg.norm(coefficient)))
            fitted.append(raw_normalized_line_coefficients(coefficient,raw_angle_cap=.08));fit_reports.append(fit)
        result.update(fit_reports=fit_reports,fit_returned_raw_coefficient_norms=raw_fit_norms)
        calibration_rows,cal_groups=draw_rows('calibration',32,registration['calibration_seed'],False,16)
        cal_phi,_,cal_q=products('calibration',calibration_rows,False)
        selected,calibration=calibrate_shared_projected_radius(cal_phi,cal_q,cal_groups,np.asarray(fitted),
            seed=registration['calibration_bootstrap_seed'],maximum_angle=.03)
        coefficients=np.vstack(([1.,0.,0.],selected));result['calibration']=calibration
        if np.all(np.linalg.norm(selected[:,1:],axis=1)<=1e-14):
            atomic_json(output/'frozen_candidates_before_audit.json',dict(coefficients=coefficients.tolist(),
                both_arms_parent=True,parent_checkpoint_sha256=PARENT_SHA,fit_reports=fit_reports,
                calibration=calibration,audit_H_records_evaluated=0))
            result.update(status='no_nonzero_calibrated_update_parent_retained',coefficients=coefficients.tolist(),
                optimization_candidate_nonzero=[False,False],audit_records=0,
                planned_audit_records=count,no_new_audit_claim=True,
                parent_prior_audit_not_reinterpreted=True,attention_updates=0,neighboring_paths_changed=0)
            return result
        if file_sha256(preflight/'context.pkl') != prior['context_sha256']:
            raise ValueError('exact cut/export context changed during fit')
        context=load_checkpoint(preflight/'context.pkl')['params'];source,source_provenance=load_source(.125)
        if source_provenance['checkpoint_sha256'] != prior['source']['checkpoint_sha256']:
            raise ValueError('trusted small-block source changed during refinement')
        _,joins,_=split_bond_ledger(12,.125)
        candidates=[];exports=[]
        for name,c in zip(('control','enriched'),selected,strict=True):
            state,bank,proof=_restore(context,context['basis']@c,source,joins)
            expected=np.vdot(c,context['basis'].T@context['Hbasis']@c).real
            norm=complex(triangular_raw_overlap(bank,bank,CFG))
            overlaps=np.asarray([triangular_raw_overlap(b,bank,CFG) for b in native])
            coefficient_error=float(np.max(abs(overlaps-c)))
            if abs(proof['raw_energy']-expected)>2e-8 or max(abs(norm-1),coefficient_error)>2e-10:
                raise ArithmeticError('final coherent ancestor restoration failed independent raw H/norm')
            save_checkpoint(output/f'{name}_state.pkl',state,metadata=dict(parent_sha=PARENT_SHA,coefficients=c.tolist()))
            save_checkpoint(output/f'{name}_bank.pkl',bank,metadata=dict(embedding=proof['embedding'],
                model=dict(length=12,multiplicity_cap=20,maximum_q=10),j2=.125,parent_sha=PARENT_SHA,
                fitted_then_amplitude_calibrated=True,coefficients=c.tolist(),optimizer_reset_required=True))
            proof.update(raw_norm=float(norm.real),coefficient_raw_norm=float(np.linalg.norm(c)),
                exact_raw_overlap_coefficient_error=coefficient_error,
                raw_basis_overlaps_real=overlaps.real.tolist(),raw_basis_overlaps_imaginary=overlaps.imag.tolist(),
                checkpoint_sha256=file_sha256(output/f'{name}_bank.pkl'))
            candidates.append(bank);exports.append(proof)
        result['exports']=exports
        atomic_json(output/'frozen_candidates_before_audit.json',dict(coefficients=coefficients.tolist(),
            checkpoint_sha256=[p['checkpoint_sha256'] for p in exports],fit_reports=fit_reports,
            calibration=calibration,audit_H_records_evaluated=0))
        for index,bank in enumerate(candidates):
            packed=prepare_feature_lut_bank(repack_active_clebsch_bank(bank,CFG,profile),CFG)
            actual=timed(f'final_linear_full_A1_H_replay_{index}',lambda packed=packed:action(packed,fit_rows[:1]));result['full_H_queries']+=1
            expected=np.asarray([fit_phi[0]@selected[index],fit_hp[0]@selected[index]])
            error=float(np.max(abs(np.asarray([actual[0][0],actual[1][0]])-expected))/max(np.max(abs(expected)),1e-300))
            if error>2e-10:raise ArithmeticError('exported bank full-A1/H differs from linear span')
            result['exports'][index]['full_A1_H_linear_replay_relative_error']=error
        del context,candidates,native;gc.collect()
        audit_rows,audit_groups=draw_rows('audit',count,registration['audit_seed'],True,16)
        audit_phi,audit_hp,audit_q=products('audit',audit_rows,True)
        psi=audit_phi@coefficients.T;hp=audit_hp@coefficients.T
        bound=2*np.sum(coefficients**2/GUIDE_WEIGHTS[None],axis=1)
        if np.any(np.max(abs(psi)**2/audit_q[:,None],axis=0)>bound*(1+2e-10)):
            raise ArithmeticError('coefficient-specific positive-guide bound violated')
        pairs={}
        for name,(a,b) in {'control_minus_parent':(0,1),'enriched_minus_parent':(0,2),'enriched_minus_control':(1,2)}.items():
            pairs[name]=dict(moments=paired_iid_moments(psi[:,[a,b]],hp[:,[a,b]],audit_q,audit_groups,
                n_sites=144,seed=registration['audit_bootstrap_seed']),
                overlap=paired_projected_overlap(psi[:,[a,b]],audit_q,audit_groups,seed=registration['audit_bootstrap_seed']+1))
        np.savez(output/'untouched_audit_products.npz',basis_phi=audit_phi,basis_Hphi=audit_hp,
            proposal=audit_q,psi=psi,Hpsi=hp,groups=audit_groups,rows=audit_rows,coefficients=coefficients)
        result.update(status='complete_fresh_single_path_full_A1_refinement_and_audit',audit=pairs,
            coefficients=coefficients.tolist(),norm_weight_upper_bounds=bound.tolist(),
            optimization_candidate_nonzero=[bool(np.linalg.norm(c[1:])>1e-14) for c in selected],
            no_post_audit_selection_or_parameter_change=True,fit_calibration_audit_seeds_disjoint=True,
            audit_records=count,attention_updates=0,neighboring_paths_changed=0)
    except TimeoutError as error:
        result.update(status='bounded_incomplete_refinement',error=str(error))
    except Exception as error:
        result.update(status='failed_validation',error=repr(error));raise
    finally:
        result.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu,
            total_numerical_wall_seconds=prior['wall_seconds']+perf_counter()-start,
            peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('prepare','run'),required=True)
    parser.add_argument('--parent',type=Path);parser.add_argument('--preflight',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seconds-cap',type=float,default=3600.)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.parent,args.output)
    else:run(args.preflight,args.output,seconds_cap=args.seconds_cap)
