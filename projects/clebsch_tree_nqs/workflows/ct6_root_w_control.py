"""Actual native-root W control of the frozen CT6 cross-branch gate curve.

Only the original singlet-root tensor changes. Consumed32 gate-fit rows make
one development direction;32 full-H direction centers calibrate its finite
line. A NEW128-record common proposal matches physical FS using amplitudes
alone. No result here is an untouched energy/variance audit.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from time import perf_counter,process_time
import shutil

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_gauge import _unpack,triangular_raw_overlap
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_amplitude,make_feature_lut_projected_amplitude,prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import mixed_clebsch_layout
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from su2_nqs.vmc.projected_spin_linear import make_terminal_tensor_products
from su2_nqs.vmc.linear_orbit_refinement import make_orbit_amplitude_products
from su2_nqs.vmc.root_linear_control import (
    projected_root_residual_direction,root_line_diagnostics,match_root_line_radius,native_root_matrix,
)
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action
from su2_nqs.vmc.triangular_raw_born import make_triangular_raw_born_sampler
from su2_nqs.dressing.physical_gate_amplitude import jax_plaquette_unitary
from .ct6_crossbranch_gate import CFG,PARENT,SITES,VECTOR,projected_entry_query,local_entry_tables
from .triangle_hierarchy_initialization import _node_timeout
from .triangle_iid_orbit_overlap import paired_projected_overlap

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'projects/clebsch_tree_nqs/results/ct6_crossbranch_gate_20261027'
THETA=.1303001548550192
CALIBRATION_SEED=2610271801


def root_child_frames(canonical:dict,original:dict) -> tuple[dict,dict]:
    """Exact cross double layers for the fixed CT6 native/canonical children."""
    al,at=_unpack(canonical,CFG);bl,bt=_unpack(original,CFG)
    up=[{q:x[q].conj().T@y[q] for q in CFG.local_q} for x,y in zip(al,bl,strict=True)]
    layout=mixed_clebsch_layout(CFG)
    for ca,cb,row in zip(at[:-1],bt[:-1],layout[:-1],strict=True):
        current={}
        for (ql,qr,qt),value in ca.items():
            term=np.einsum('alr,buv,lu,rv->ab',value.conj(),cb[(ql,qr,qt)],
                up[row['left']][ql],up[row['right']][qr],optimize=True)
            current[qt]=current.get(qt,np.zeros_like(term))+term
        up.append(current)
    left,right=up[layout[-1]['left']],up[layout[-1]['right']]
    condition=max(float(np.linalg.cond(x)) for block in (left,right) for x in block.values())
    minimum=min(float(np.linalg.svd(x,compute_uv=False)[-1]) for block in (left,right) for x in block.values())
    if not np.isfinite(condition) or condition>1e8 or minimum<=0:
        raise ArithmeticError('native root frame is not safely invertible; no completion allowed')
    return {q:(left[q],right[q]) for q in left},dict(maximum_child_condition=condition,
        minimum_child_singular_value=minimum,right_ket_transport='ordinary transpose')


def export_native_root(native:dict,canonical_top:tuple,frames:dict) -> dict:
    """Restore only native top[-1]; local/six/two nonroot top nodes untouched."""
    groups=mixed_clebsch_layout(CFG)[-1]['groups'];new=[]
    for data,group in zip(canonical_top,groups,strict=True):
        values=[]
        for c,(ql,qr,qt) in zip(data,group['keys'],strict=True):
            if ql!=qr or qt!=0 or c.shape[0]!=1:raise ArithmeticError('singlet root layout changed')
            left,right=frames[ql]
            value=native_root_matrix(c[0],left,right)
            if np.max(abs(value.imag))>1e-11 if np.iscomplexobj(value) else False:
                raise ArithmeticError('real native parent acquired a complex root')
            values.append(value.real[None])
        new.append(np.asarray(values))
    return dict(native,top=tuple(native['top'][:-1])+(tuple(new),))


def _per_site(record:dict) -> dict:
    result=dict(record)
    for name in ('energy_real','energy_imaginary','energy_first','energy_second','residual_variance','arclength_slope','arclength_curvature'):
        result[name+'_per_site']=None if record[name] is None else record[name]/36
    return result


def run(output:Path) -> dict:
    """One900-second local development/control run; never an audit."""
    if output.exists() and any(output.iterdir()):raise FileExistsError('fresh isolated output required')
    output.mkdir(parents=True,exist_ok=True)
    dependencies=[Path(__file__),ROOT/'su2_nqs/vmc/root_linear_control.py',
        ROOT/'su2_nqs/vmc/projected_spin_linear.py',ROOT/'su2_nqs/vmc/linear_orbit_refinement.py',
        ROOT/'su2_nqs/models/triangular_clebsch_gauge.py',ROOT/'su2_nqs/models/triangular_arity_clebsch.py',
        ROOT/'su2_nqs/models/triangular_clebsch_feature_lut.py',ROOT/'su2_nqs/models/triangular_clebsch_exchange_cache.py',
        ROOT/'su2_nqs/vmc/triangular_clebsch_cached_action.py',ROOT/'su2_nqs/vmc/triangular_raw_born.py',
        ROOT/'su2_nqs/vmc/raw_tree_born.py',Path(__file__).with_name('ct6_crossbranch_gate.py')]
    registration=dict(task='actual_CT6_original_root_W_calibration_control_v1',parent_checkpoint_sha256=file_sha256(PARENT),
        canonical_parent_sha256=file_sha256(OLD/'normalized_canonical_parent.pkl'),
        consumed_fit_products_sha256=file_sha256(OLD/'finite_fitting_products.npz'),
        fixed_gate_theta=THETA,gate_angle_vector=VECTOR.tolist(),gate_sites=SITES,
        root_real_coordinates=4704,root_only_not_all_original_W=True,metric_rank_cap=8,relative_ridge=.1,metric_rcond=1e-10,
        consumed_development_fit_rows=32,maximum_new_full_H_centers=32,new_calibration_rows=128,
        calibration_seed=CALIBRATION_SEED,calibration_Q='(gatecore+parent_raw_orbit+direction_raw_orbit)/3',
        each_raw_guide_norm_one=True,calibration_amplitudes_only_no_H=True,calibration_not_untouched_audit=True,
        direction_sign='negative exact bilateral consumed-fit line slope at zero',
        FS_match_fixed_signed_bracket=[0.,.5],FS_match_bisection_iterations=16,
        parent_normalized_native_root_scale_only=True,original_binary_rank8_and_all_nonroot_W_preserved=True,
        no_remote_change=True,no_new_gate_or_head=True,no_fresh_energy_audit=True,numerical_wall_cap_seconds=900,
        old_fit_Q_not_certified_for_new_root_line=True,old128_gate_audit_not_used_for_new_control_selection=True,
        source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in dependencies})
    atomic_json(output/'registration.json',registration)
    frozen=output/'registered_sources';frozen.mkdir()
    for path in dependencies:shutil.copy2(path,frozen/path.name)
    start=perf_counter();cpu=process_time();result=dict(status='started',new_full_H_centers=0,
        new_calibration_records=0,registration=registration,timings={})
    try:
        with _node_timeout(900):
            native=load_checkpoint(PARENT)['params'];parent=load_checkpoint(OLD/'normalized_canonical_parent.pkl')['params']
            original=arity_clebsch_bank(native,CFG,'binary_truncated',8)
            frames,result['native_transport']=root_child_frames(parent,original)
            images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
            prepared=prepare_feature_lut_bank(parent,CFG)
            scalar=make_feature_lut_projected_amplitude(CFG,images,chunk=4)
            amplitude=jax.jit(jax.vmap(scalar,in_axes=(None,0)))
            dummy=lambda bank,rows:(amplitude(bank,rows),jnp.zeros(len(rows),jnp.complex128))
            c0,replace,columns=make_terminal_tensor_products(dummy,prepared)
            c0=np.asarray(c0);_,unravel=ravel_pytree(parent['top'][-1])
            if len(c0)!=4704 or abs(c0@c0-1)>2e-9:raise ArithmeticError('canonical root is not4704 raw-isometric unit coordinates')
            fit=np.load(OLD/'finite_fitting_products.npz');rows=np.load(OLD/'finite_registered_rows.npz')['rows'][:32]
            _,pairs,_,_,_=local_entry_tables();identity=np.eye(16)[pairs[:,0],pairs[:,1]]
            hp0=fit['HPhi70']@identity;psi0=fit['Phi70']@identity;qfit=fit['proposal']
            tick=perf_counter();features=np.asarray(columns(prepared,rows)[0]);result['timings']['root_feature32']=perf_counter()-tick
            replay=float(np.max(abs(features@c0-psi0))/max(np.max(abs(psi0)),1e-300))
            if replay>2e-10:raise ArithmeticError('canonical root features disagree with frozen fit parent')
            direction,notes=projected_root_residual_direction(features,c0,hp0,qfit)
            result.update(direction=notes,parent_fit_amplitude_replay_relative_error=replay,root_feature_centers=32)
            dtop=unravel(jnp.asarray(direction));direction_bank=dict(parent,top=tuple(parent['top'][:-1])+(dtop,))
            native_parent=export_native_root(native,parent['top'][-1],frames)
            native_direction=export_native_root(native,dtop,frames)
            folded_parent=arity_clebsch_bank(native_parent,CFG,'binary_truncated',8)
            folded_direction=arity_clebsch_bank(native_direction,CFG,'binary_truncated',8)
            gram=np.array([[triangular_raw_overlap(a,b,CFG) for b in (parent,direction_bank)] for a in (folded_parent,folded_direction)])
            if np.max(abs(gram-np.eye(2)))>3e-10:raise ArithmeticError('native direction/parent exact raw frame replay failed')
            result['native_raw_cross_Gram_error']=float(np.max(abs(gram-np.eye(2))))
            save_checkpoint(output/'canonical_direction.pkl',direction_bank,metadata=dict(role='raw_unit_original_root_direction',parent_sha256=registration['canonical_parent_sha256']))
            save_checkpoint(output/'native_direction.pkl',native_direction,metadata=dict(role='raw_unit_original_root_direction',binary_rank=8))
            np.savez(output/'frozen_direction.npz',parent_coefficients=c0,direction_coefficients=direction,fit_rows=rows,root_features=features,
                Phi_parent=psi0,HPhi_parent=hp0,proposal=qfit)
            atomic_json(output/'frozen_direction.json',dict(result,wall_seconds=perf_counter()-start,
                direction_sha256=file_sha256(output/'canonical_direction.pkl'),native_direction_sha256=file_sha256(output/'native_direction.pkl')))
            print('root direction frozen; new H centers0',flush=True)
            haction=make_cached_projected_heisenberg_action(CFG,images,triangular_j1j2_bonds(6,6,j2=.125,periodic=True),projection_chunk=4)
            dprepared=prepare_feature_lut_bank(direction_bank,CFG);hd=[];phid=[];times=[]
            for index,row in enumerate(rows):
                tick=perf_counter();p,h=haction(dprepared,row[None]);jax.block_until_ready(h)
                times.append(perf_counter()-tick);phid.append(float(p[0]));hd.append(float(h[0]));result['new_full_H_centers']=index+1
                if index%8==7:
                    atomic_json(output/'progress.json',dict(result,wall_seconds=perf_counter()-start))
                    print('root direction full H',index+1,'/32',flush=True)
            phid=np.asarray(phid);hd=np.asarray(hd);expected=features@direction
            error=float(np.max(abs(phid-expected))/max(np.max(abs(expected)),1e-300))
            if error>2e-10:raise ArithmeticError('finite root direction differs from exact JVP/linear columns')
            phi2=np.column_stack((psi0,phid));hphi2=np.column_stack((hp0,hd))
            line0=root_line_diagnostics(phi2,hphi2,qfit,0.)
            sign=1 if line0['energy_first']<0 else -1
            result.update(fit_line_at_zero=_per_site(line0),fit_selected_sign=sign,direction_H_amplitude_replay_error=error)
            result['timings']['direction_full_H_centers']=times
            np.savez(output/'development_line_products.npz',Phi=phi2,HPhi=hphi2,proposal=qfit,rows=rows)
            atomic_json(output/'development_H_line.json',dict(result,wall_seconds=perf_counter()-start))
            # Fresh amplitude-only common-law calibration; all three sources
            # cover the root line and gate without assuming parent support.
            tick=perf_counter();profile=[row['widths'] for row in mixed_clebsch_layout(CFG)];cache={}
            samplers=[make_triangular_raw_born_sampler(bank,CFG,active_top_widths=profile,
                memory_limit_bytes=2*1024**3,shared_tensor_cache=cache) for bank in (parent,direction_bank)]
            result['timings']['calibration_sampler_factories']=perf_counter()-tick
            bit_table,_,_,_,_=local_entry_tables();inverse=np.argsort(images,axis=1)
            rng=np.random.default_rng(CALIBRATION_SEED);calrows=[];latent=[];rawrows=[];born_errors=[]
            tick=perf_counter()
            for _ in range(128):
                component=int(rng.integers(3));draw=samplers[int(component==2)].draw(rng)
                row=draw.bits.copy();rawrows.append(row.copy());replacement=-1
                if component==0:
                    choices=np.flatnonzero(bit_table.sum(1)==row[list(SITES)].sum())
                    replacement=int(rng.choice(choices));row[list(SITES)]=bit_table[replacement]
                image=int(rng.integers(len(images)));calrows.append(row[inverse[image]])
                latent.append((component,replacement,image));born_errors.append(draw.log_probability_replay_error)
            result['timings']['calibration_IID_draws']=perf_counter()-tick
            result['calibration_draw_rng_state']=rng.bit_generator.state;result['maximum_Born_replay_error']=max(born_errors)
            calrows=np.asarray(calrows,np.int32)
            np.savez(output/'calibration_registered_rows.npz',rows=calrows,raw_rows=np.asarray(rawrows),latent=np.asarray(latent),Born_replay=np.asarray(born_errors))
            del samplers,cache;gc.collect()
            gate_query,entry_pairs=projected_entry_query(CFG,images)
            raw=make_feature_lut_amplitude(CFG);orbit=make_orbit_amplitude_products(raw,images,projection_chunk=4)
            gate_coeff=np.asarray(jax_plaquette_unitary(THETA*jnp.asarray(VECTOR)))[entry_pairs[:,0],entry_pairs[:,1]]
            calp=[];calq=[];identity_errors=[];tick=perf_counter()
            for index,row in enumerate(calrows):
                entries,gq=gate_query(prepared,row[None]);p,pq=orbit(prepared,row[None]);d,dq=orbit(dprepared,row[None])
                e=np.asarray(entries[0]);p,d=float(p[0]),float(d[0]);q0=(float(gq[0])+float(pq[0])+float(dq[0]))/3
                if not np.isfinite(q0) or q0<=0:raise ArithmeticError('common calibration proposal is nonpositive')
                calp.append((p,d,float(e@gate_coeff)));calq.append(q0);identity_errors.append(abs(p-e@identity))
                result['new_calibration_records']=index+1
                if index%16==15:print('root/gate common amplitude calibration',index+1,'/128',flush=True)
            result['timings']['calibration_amplitude_products']=perf_counter()-tick
            calp,calq=np.asarray(calp),np.asarray(calq);groups=np.repeat(np.arange(16),8)
            if max(identity_errors)>2e-10*max(np.max(abs(calp[:,0])),1e-300):raise ArithmeticError('gate identity and independent parent calibration disagree')
            gate_overlap=paired_projected_overlap(calp[:,[0,2]],calq,groups,seed=2610271802)
            radius=float(gate_overlap['values'][-1])
            t,match=match_root_line_radius(calp[:,:2],calq,radius,sign=sign,bracket=.5,iterations=16)
            coefficient=(c0+t*direction)/np.sqrt(1+t*t)
            candidate=dict(parent,top=tuple(parent['top'][:-1])+(unravel(jnp.asarray(coefficient)),))
            native_candidate=export_native_root(native,candidate['top'][-1],frames)
            native_folded=arity_clebsch_bank(native_candidate,CFG,'binary_truncated',8)
            raw_norm=float(triangular_raw_overlap(native_folded,native_folded,CFG).real)
            raw_overlap=triangular_raw_overlap(native_folded,candidate,CFG)
            if max(abs(raw_norm-1),abs(raw_overlap-1))>3e-10:raise ArithmeticError('finite native checkpoint left the exact normalized root line')
            before_nonroot=dict(local=native['local'],six=native['six'],top=native['top'][:-1])
            after_nonroot=dict(local=native_candidate['local'],six=native_candidate['six'],top=native_candidate['top'][:-1])
            if pytree_sha256(before_nonroot)!=pytree_sha256(after_nonroot):raise ArithmeticError('nonroot native W changed')
            checkpoint=output/'native_root_W_candidate.pkl'
            save_checkpoint(checkpoint,native_candidate,optimizer_state=None,metadata=dict(binary_rank=8,
                parent_checkpoint_sha256=registration['parent_checkpoint_sha256'],canonical_parent_sha256=registration['canonical_parent_sha256'],
                root_only_original_W=True,root_line_t=t,optimizer_reset_required=True,all_nonroot_W_unchanged=True,
                matching_calibration_not_audit=True,raw_norm=raw_norm))
            save_checkpoint(output/'canonical_root_W_candidate.pkl',candidate,metadata=dict(root_line_t=t,raw_norm=raw_norm))
            restored=load_checkpoint(checkpoint)['params']
            if pytree_sha256(restored)!=pytree_sha256(native_candidate):raise ArithmeticError('checkpoint parameter replay failed')
            check_rows=calrows[:4]
            candidate_values=np.asarray(amplitude(prepare_feature_lut_bank(native_folded,CFG),check_rows))
            expected_values=(calp[:4,0]+t*calp[:4,1])/np.sqrt(1+t*t)
            finite_error=float(np.max(abs(candidate_values-expected_values))/max(np.max(abs(expected_values)),1e-300))
            if finite_error>2e-10:raise ArithmeticError('native finite full-A1 export is not its declared root line')
            root_values=(calp[:,0]+t*calp[:,1])/np.sqrt(1+t*t)
            root_overlap=paired_projected_overlap(np.column_stack((calp[:,0],root_values)),calq,groups,seed=2610271803)
            np.savez(output/'calibration_amplitude_products.npz',rows=calrows,Phi_parent_direction_gate=calp,proposal=calq,groups=groups,
                native_candidate_amplitudes=candidate_values,finite_expected=expected_values)
            result.update(status='complete_development_root_W_control_and_common_FS_calibration_no_energy_audit',
                fixed_gate_theta=THETA,matched_root_t=t,calibration_match=match,gate_overlap=gate_overlap,root_overlap=root_overlap,
                development_candidate_line=_per_site(root_line_diagnostics(phi2,hphi2,qfit,t)),
                native_candidate_sha256=file_sha256(checkpoint),canonical_candidate_sha256=file_sha256(output/'canonical_root_W_candidate.pkl'),
                native_candidate_raw_norm=raw_norm,native_candidate_cross_overlap_real=float(raw_overlap.real),
                native_full_A1_finite_replay_relative_error=finite_error,native_nonroot_parameters_unchanged=True,
                native_candidate_parameter_sha256=pytree_sha256(native_candidate),no_untouched_energy_claim=True,
                amplitude_role_counts=dict(root_column_centers=32,common_calibration_centers=128,
                    gate_entry_centers=128,parent_orbit_centers=128,direction_orbit_centers=128,native_finite_replay_centers=4),
                future_full_H_audit_requires_half_Q0_half_all_bond_push=True)
    except (TimeoutError,MemoryError) as error:
        result.update(status='bounded_incomplete',error=repr(error))
    finally:
        result.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu)
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(run(args.output)['status'],flush=True)
