"""Serial root-W force-consistency repair; development and amplitude calibration only.

Reuse the v1 native export, physical kernels, exact raw sampler, and overlap
postprocessor. Full root H columns fix the FINITE-SAMPLE bilateral force, not
the population one-sided identity. Neither consumed fitting rows nor the fresh
amplitude-only matching panel constitute an energy audit.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter,process_time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from . import ct6_root_w_control as v1
from su2_nqs.vmc.root_linear_control import (
    projected_root_residual_direction,root_line_diagnostics,match_root_line_radius,
)

V1=v1.ROOT/'projects/clebsch_tree_nqs/results/ct6_root_w_control_20261027'
SEED=2610271901


def _rss_bytes() -> int:
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def run(output:Path) -> dict:
    """At most32 full root-H feature centers plus128 amplitude-only records."""
    if output.exists() and any(output.iterdir()):raise FileExistsError('fresh isolated output required')
    output.mkdir(parents=True,exist_ok=True)
    import json
    old_registration=json.loads((V1/'registration.json').read_text())
    old_helper=V1/'registered_sources/root_linear_control.py'
    if v1.file_sha256(old_helper)!=old_registration['source_hashes']['su2_nqs/vmc/root_linear_control.py']:
        raise ArithmeticError('v1 frozen helper seal changed')
    dependencies=[Path(__file__),Path(v1.__file__)]+[
        v1.ROOT/p for p in old_registration['source_hashes'] if p!=str(Path(v1.__file__).relative_to(v1.ROOT))]
    reg=dict(task='CT6_native_root_W_bilateral_development_v2',source_hashes={str(p.relative_to(v1.ROOT)):v1.file_sha256(p) for p in dependencies},
        v1_registration_sha256=v1.file_sha256(V1/'registration.json'),v1_results_sha256=v1.file_sha256(V1/'results.json'),
        v1_frozen_helper_sha256=v1.file_sha256(old_helper),current_helper_differs_intentionally=True,
        parent_checkpoint_sha256=v1.file_sha256(v1.PARENT),canonical_parent_sha256=v1.file_sha256(v1.OLD/'normalized_canonical_parent.pkl'),
        old_direction_sha256=v1.file_sha256(V1/'canonical_direction.pkl'),consumed_fit_products_sha256=v1.file_sha256(v1.OLD/'finite_fitting_products.npz'),
        maximum_full_H_ROOT_feature_centers=32,root_coordinate_count=4704,root_only=True,rank_cap=8,ridge_relative=.1,
        calibration_rows=128,calibration_seed=SEED,calibration_Q='(gatecore+parent_raw_orbit+D_one_side_raw_orbit+D_bilateral_raw_orbit)/4',
        calibration_amplitudes_only=True,no_fresh_energy_audit=True,fixed_gate_theta=v1.THETA,
        sign='negative complete bilateral consumed-fit slope at zero',radius_bracket=.5,bisection_iterations=16,
        first_center_gate='31*first_center_wall+elapsed+150 <=900 and peak_RSS <=8GiB; conservative cold-cost forecast',
        wall_cap=900,no_width_or_gate_or_remote_change=True,all_nonroot_W_unchanged=True,
        old_fit_is_development_not_whole_line_support_certificate=True)
    v1.atomic_json(output/'registration.json',reg)
    frozen=output/'registered_sources';frozen.mkdir()
    for path in dependencies:shutil.copy2(path,frozen/path.name)
    start=perf_counter();cpu=process_time();r=dict(status='started',registration=reg,new_full_H_ROOT_feature_centers=0,new_calibration_records=0,timings={})
    try:
        with v1._node_timeout(900):
            native=v1.load_checkpoint(v1.PARENT)['params'];parent=v1.load_checkpoint(v1.OLD/'normalized_canonical_parent.pkl')['params']
            original=v1.arity_clebsch_bank(native,v1.CFG,'binary_truncated',8)
            frames,r['native_transport']=v1.root_child_frames(parent,original)
            old=np.load(V1/'frozen_direction.npz');c0=old['parent_coefficients'];d1=old['direction_coefficients']
            rows=old['fit_rows'];qfit=old['proposal'];psi0=old['Phi_parent'];hp0=old['HPhi_parent']
            _,unravel=ravel_pytree(parent['top'][-1])
            prepared=v1.prepare_feature_lut_bank(parent,v1.CFG)
            images=np.asarray(v1.triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
            action=v1.make_cached_projected_heisenberg_action(v1.CFG,images,
                v1.triangular_j1j2_bonds(6,6,j2=.125,periodic=True),projection_chunk=4)
            coords,_,columns=v1.make_terminal_tensor_products(action,prepared)
            if np.max(abs(np.asarray(coords)-c0))>1e-14:raise ArithmeticError('v1 canonical parent coordinates changed')
            features=[];hfeatures=[];times=[]
            for index,row in enumerate(rows):
                tick=perf_counter();f,h=columns(prepared,row[None]);jax.block_until_ready(h)
                times.append(perf_counter()-tick);features.append(np.asarray(f[0]));hfeatures.append(np.asarray(h[0]))
                r['new_full_H_ROOT_feature_centers']=index+1
                if index==0:
                    forecast=perf_counter()-start+31*times[0]+150
                    probe=dict(wall_seconds=times[0],peak_RSS_bytes=_rss_bytes(),forecast_total_seconds=forecast,
                        root_columns=4704,full_images=len(images),weighted_bonds=216)
                    r['first_root_H_feature_center']=probe
                    np.savez(output/'first_root_H_feature_center.npz',row=row,Phi=f,HPhi=h)
                    v1.atomic_json(output/'first_root_H_feature_center.json',probe)
                    print('FIRST_ROOT_H_FEATURE_CENTER',probe,flush=True)
                    if forecast>900 or probe['peak_RSS_bytes']>8*1024**3:
                        r['status']='cost_gate_stopped_after_one_root_H_feature_center'
                        return r
                if index%8==7:
                    v1.atomic_json(output/'progress.json',dict(r,wall_seconds=perf_counter()-start))
                    print('full root H features',index+1,'/32',flush=True)
            f,hf=np.asarray(features),np.asarray(hfeatures);r['timings']['root_H_feature_centers']=times
            ferr=float(np.max(abs(f-old['root_features']))/max(np.max(abs(old['root_features'])),1e-300))
            herr=float(np.max(abs(hf@c0-hp0))/max(np.max(abs(hp0)),1e-300))
            oldhd=np.load(V1/'development_line_products.npz')['HPhi'][:,1]
            d1herr=float(np.max(abs(hf@d1-oldhd))/max(np.max(abs(oldhd)),1e-300))
            if max(ferr,herr,d1herr)>3e-10:raise ArithmeticError('complete root features disagree with saved parent/direction products')
            d2,notes=projected_root_residual_direction(f,c0,hp0,qfit,hfeatures=hf)
            ds=(d1,d2);labels=('one_side','bilateral');banks=[];preps=[];signs=[];fitlines=[]
            r.update(bilateral_direction=notes,parent_amplitude_feature_replay=ferr,parent_H_feature_replay=herr,
                v1_direction_H_feature_replay=d1herr,
                raw_D1_D2_overlap=float(d1@d2),directions={})
            # Separate both numerator force contributions; no sampled-H symmetrization.
            x=f/np.sqrt(qfit[:,None]);y=hf/np.sqrt(qfit[:,None]);p=x@c0;hp=y@c0
            z=float(np.vdot(p,p).real);e=float(np.vdot(p,hp).real/z)
            g1=np.real(x.conj().T@(hp-e*p))/z
            g2=np.real(y.conj().T@p-e*(x.conj().T@p))/z
            r['force_decomposition']=dict(g1_norm=float(np.linalg.norm(g1)),g2_norm=float(np.linalg.norm(g2)),
                sum_norm=float(np.linalg.norm(g1+g2)),difference_norm=float(np.linalg.norm(g1-g2)),
                interpretation='bilateral finite-sample consistency; population one-sided identity is not contradicted')
            for label,d in zip(labels,ds,strict=True):
                bank=dict(parent,top=tuple(parent['top'][:-1])+(unravel(jnp.asarray(d)),))
                banks.append(bank);preps.append(v1.prepare_feature_lut_bank(bank,v1.CFG))
                phi=np.column_stack((f@c0,f@d));hphi=np.column_stack((hf@c0,hf@d))
                line=root_line_diagnostics(phi,hphi,qfit,0.);fitlines.append((phi,hphi))
                sign=1 if line['energy_first']<0 else -1;signs.append(sign)
                decomposition=dict(g1_dot_direction=float(g1@d),g2_dot_direction=float(g2@d),
                    full_force_dot_direction=float((g1+g2)@d),line_slope=line['energy_first'])
                if abs(decomposition['full_force_dot_direction']-line['energy_first'])>2e-10:
                    raise ArithmeticError('full root force is not its finite-line JVP')
                r['directions'][label]=dict(fit_at_zero=v1._per_site(line),sign=sign,force_decomposition=decomposition)
                v1.save_checkpoint(output/f'canonical_{label}_direction.pkl',bank,metadata=dict(role='raw_unit_root_direction',development_only=True))
            gram=np.array([[v1.triangular_raw_overlap(a,b,v1.CFG) for b in (parent,*banks)] for a in (parent,*banks)])
            expected=np.array([[1,0,0],[0,1,d1@d2],[0,d1@d2,1]])
            if np.max(abs(gram-expected))>3e-10:raise ArithmeticError('exact raw direction Gram failed')
            r['raw_Gram_error']=float(np.max(abs(gram-expected)))
            np.savez(output/'development_full_root_products.npz',rows=rows,Phi=f,HPhi=hf,proposal=qfit,
                parent=c0,D_one_side=d1,D_bilateral=d2,g1=g1,g2=g2)
            v1.atomic_json(output/'frozen_directions.json',dict(r,wall_seconds=perf_counter()-start))
            print('both root directions and signs frozen; amplitude-only calibration next',flush=True)
            profile=[row['widths'] for row in v1.mixed_clebsch_layout(v1.CFG)];cache={};tick=perf_counter()
            samplers=[v1.make_triangular_raw_born_sampler(bank,v1.CFG,active_top_widths=profile,
                memory_limit_bytes=2*1024**3,shared_tensor_cache=cache) for bank in (parent,*banks)]
            r['timings']['sampler_factories']=perf_counter()-tick
            bit_table,pairs,_,_,_=v1.local_entry_tables();identity=np.eye(16)[pairs[:,0],pairs[:,1]]
            inverse=np.argsort(images,axis=1);rng=np.random.default_rng(SEED);calrows=[];rawrows=[];latent=[];errors=[]
            tick=perf_counter()
            for _ in range(128):
                component=int(rng.integers(4));draw=samplers[max(0,component-1)].draw(rng)
                row=draw.bits.copy();rawrows.append(row.copy());replacement=-1
                if component==0:
                    choices=np.flatnonzero(bit_table.sum(1)==row[list(v1.SITES)].sum())
                    replacement=int(rng.choice(choices));row[list(v1.SITES)]=bit_table[replacement]
                image=int(rng.integers(len(images)));calrows.append(row[inverse[image]])
                latent.append((component,replacement,image));errors.append(draw.log_probability_replay_error)
            r['timings']['IID_draws']=perf_counter()-tick;r['maximum_Born_replay_error']=max(errors)
            r['calibration_rng_state']=rng.bit_generator.state;calrows=np.asarray(calrows,np.int32)
            np.savez(output/'calibration_registered_rows.npz',rows=calrows,raw_rows=rawrows,latent=latent,Born_replay=errors)
            del samplers,cache;gc.collect()
            gate_query,entry_pairs=v1.projected_entry_query(v1.CFG,images)
            orbit=v1.make_orbit_amplitude_products(v1.make_feature_lut_amplitude(v1.CFG),images,projection_chunk=4)
            gate_coeff=np.asarray(v1.jax_plaquette_unitary(v1.THETA*jnp.asarray(v1.VECTOR)))[entry_pairs[:,0],entry_pairs[:,1]]
            calp=[];calq=[];identity_errors=[];tick=perf_counter()
            for index,row in enumerate(calrows):
                entries,gq=gate_query(prepared,row[None]);values=[];densities=[float(gq[0])]
                for bank in (prepared,*preps):
                    value,density=orbit(bank,row[None]);values.append(float(value[0]));densities.append(float(density[0]))
                q0=sum(densities)/4
                if not np.isfinite(q0) or q0<=0:raise ArithmeticError('common calibration proposal is nonpositive')
                values.append(float(np.asarray(entries[0])@gate_coeff));calp.append(values);calq.append(q0)
                identity_errors.append(abs(values[0]-np.asarray(entries[0])@identity));r['new_calibration_records']=index+1
                if index%32==31:print('common gate/two-root calibration',index+1,'/128',flush=True)
            r['timings']['amplitude_products']=perf_counter()-tick
            calp,calq=np.asarray(calp),np.asarray(calq);groups=np.repeat(np.arange(16),8)
            if max(identity_errors)>2e-10*max(np.max(abs(calp[:,0])),1e-300):raise ArithmeticError('gate identity replay failed')
            gate_overlap=v1.paired_projected_overlap(calp[:,[0,3]],calq,groups,seed=SEED+1)
            radius=float(gate_overlap['values'][-1]);r['gate_overlap']=gate_overlap
            amplitude=jax.jit(jax.vmap(v1.make_feature_lut_projected_amplitude(v1.CFG,images,chunk=4),in_axes=(None,0)))
            nonroot=dict(local=native['local'],six=native['six'],top=native['top'][:-1]);replays=[]
            for i,(label,d) in enumerate(zip(labels,ds,strict=True)):
                cp=calp[:,[0,i+1]];t,match=match_root_line_radius(cp,calq,radius,sign=signs[i],bracket=.5,iterations=16)
                coefficients=(c0+t*d)/np.sqrt(1+t*t)
                candidate=dict(parent,top=tuple(parent['top'][:-1])+(unravel(jnp.asarray(coefficients)),))
                nc=v1.export_native_root(native,candidate['top'][-1],frames)
                folded=v1.arity_clebsch_bank(nc,v1.CFG,'binary_truncated',8)
                norm=float(v1.triangular_raw_overlap(folded,folded,v1.CFG).real)
                overlap=v1.triangular_raw_overlap(folded,candidate,v1.CFG)
                if max(abs(norm-1),abs(overlap-1))>3e-10:raise ArithmeticError('native root export norm/span failed')
                if v1.pytree_sha256(nonroot)!=v1.pytree_sha256(dict(local=nc['local'],six=nc['six'],top=nc['top'][:-1])):
                    raise ArithmeticError('nonroot W changed')
                path=output/f'native_{label}_root_W_candidate.pkl'
                v1.save_checkpoint(path,nc,optimizer_state=None,metadata=dict(binary_rank=8,root_only=True,
                    parent_checkpoint_sha256=reg['parent_checkpoint_sha256'],root_line_t=t,optimizer_reset_required=True,
                    development_and_calibration_not_audit=True,all_nonroot_W_unchanged=True))
                v1.save_checkpoint(output/f'canonical_{label}_root_W_candidate.pkl',candidate,metadata=dict(root_line_t=t,raw_norm=norm))
                if v1.pytree_sha256(v1.load_checkpoint(path)['params'])!=v1.pytree_sha256(nc):raise ArithmeticError('checkpoint replay failed')
                expected_values=(cp[:,0]+t*cp[:,1])/np.sqrt(1+t*t)
                actual=np.asarray(amplitude(v1.prepare_feature_lut_bank(folded,v1.CFG),calrows[:4]))
                error=float(np.max(abs(actual-expected_values[:4]))/max(np.max(abs(expected_values[:4])),1e-300))
                if error>3e-10:raise ArithmeticError('full-A1 native candidate finite replay failed')
                overlap_report=v1.paired_projected_overlap(np.column_stack((calp[:,0],expected_values)),calq,groups,seed=SEED+2+i)
                g=cp.conj().T@(cp/calq[:,None])/len(cp)
                calmetric=float(np.linalg.det(g).real/g[0,0].real**2)
                dev=v1._per_site(root_line_diagnostics(*fitlines[i],qfit,t))
                r['directions'][label].update(t=t,calibration_match=match,overlap=overlap_report,
                    calibration_metric_at_zero=calmetric,fit_to_calibration_metric_ratio=r['directions'][label]['fit_at_zero']['FS_metric']/calmetric,
                    development_candidate=dev,fit_energy_decreased=dev['energy_real']<r['directions'][label]['fit_at_zero']['energy_real'],
                    native_checkpoint_sha256=v1.file_sha256(path),native_raw_norm=norm,native_canonical_overlap_real=float(overlap.real),
                    finite_A1_replay_relative_error=error,all_nonroot_W_unchanged=True)
                replays.append(actual)
            np.savez(output/'calibration_amplitude_products.npz',rows=calrows,Phi_parent_D1_D2_gate=calp,
                proposal=calq,groups=groups,native_candidate_replays=replays)
            r.update(status='complete_bilateral_repair_and_common_FS_calibration_no_energy_audit',
                peak_RSS_bytes=_rss_bytes(),no_untouched_energy_claim=True,future_H_audit_requires_half_Q0_half_all_bond_push=True,
                role_counts=dict(full_H_ROOT_feature_centers=32,root_columns_per_center=4704,amplitude_calibration_centers=128,
                    gate_entry_centers=128,raw_orbit_centers=384,native_finite_A1_replay_centers=8))
    except (TimeoutError,MemoryError) as error:
        r.update(status='bounded_incomplete',error=repr(error))
    finally:
        r.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu)
        v1.atomic_json(output/'results.json',r)
    return r


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['status'],flush=True)
