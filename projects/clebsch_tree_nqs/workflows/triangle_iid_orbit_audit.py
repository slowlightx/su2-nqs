"""Fresh IID full-A1 energy/H-squared audit using exact raw-TTN orbit draws.

Q0=mean_k mean_g|raw_k(gx)|²/Zraw_k for the sealed endpoints.
Q=(Q0+KQ0)/2, with every nonzero physical H bond in the fixed exchange K.
Sampling measures physical leaves coherently, never hidden fusion labels.
Target amplitudes and H actions always sum all spatial images coherently.
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
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_amplitude, make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_orbit_exchange_bridge_products
from su2_nqs.vmc.triangular_raw_born import make_triangular_raw_born_sampler
from .triangle_hierarchy_audit import checked_active_widths

ROOT = Path(__file__).resolve().parents[3]


def paired_iid_moments(psi: np.ndarray, hpsi: np.ndarray, density: np.ndarray,
                       groups: np.ndarray, *, n_sites: int, seed: int) -> dict:
    """Node-safe ratios, paired IID influence errors and independent-block errors.

    Norm/energy/H² raw numerators are unbiased under the normalized proposal;
    normalized energy/variance ratios have finite-sample bias. Groups must be
    equal-size independent batches of draws from the same fixed proposal.
    No target-zero record is removed, and no energy is forced real.
    """
    p, h = np.asarray(psi, complex), np.asarray(hpsi, complex)
    q, g = np.asarray(density, float), np.asarray(groups)
    if (p.ndim != 2 or p.shape[1] != 2 or h.shape != p.shape or q.shape != (len(p),)
            or g.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(h).all()
            or not np.isfinite(q).all() or np.any(q <= 0) or n_sites < 1):
        raise ValueError('finite paired products and strictly positive proposal density required')
    labels, counts = np.unique(g, return_counts=True)
    if len(labels) < 4 or len(set(counts)) != 1:
        raise ValueError('at least four equal-size independent sample groups required')
    z = abs(p)**2/q[:, None]; e = p.conj()*h/q[:, None]; second = abs(h)**2/q[:, None]
    def moment(ids):
        norm = z[ids].mean(0)
        if np.any(norm <= 0): raise FloatingPointError('sampled target norm is zero')
        energy = e[ids].mean(0)/norm
        variance = (abs(h[ids]-energy*p[ids])**2/q[ids, None]).mean(0)/norm
        return np.stack((energy.real/n_sites, energy.imag/n_sites, variance/n_sites, norm), axis=1)
    ids = np.arange(len(p)); point = moment(ids)
    leave = np.asarray([moment(ids[g != label]) for label in labels])
    jackknife = np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(0))**2, axis=0))
    delta = point[1]-point[0]; leave_delta = leave[:, 1]-leave[:, 0]
    delta_jk = np.sqrt((len(labels)-1)/len(labels)*np.sum((leave_delta-leave_delta.mean(0))**2, axis=0))
    norm = z.mean(0); energy = e.mean(0)/norm; h2 = second.mean(0)/norm
    ue = (e-energy*z)/norm; um = (second-h2*z)/norm
    influence = np.stack((ue.real/n_sites, ue.imag/n_sites,
        (um-2*np.real(energy.conj()*ue))/n_sites, z-norm), axis=-1)
    iid_se = influence.std(0, ddof=1)/np.sqrt(len(p))
    paired_iid_se = (influence[:, 1]-influence[:, 0]).std(0, ddof=1)/np.sqrt(len(p))
    rng = np.random.default_rng(seed); by_group = [ids[g == label] for label in labels]
    bootstrap = np.asarray([moment(np.concatenate([by_group[i] for i in rng.integers(len(labels), size=len(labels))]))
                            for _ in range(512)])
    residual = abs(h-energy*p)**2/q[:, None]
    def concentration(values):
        total = values.sum(0)
        return dict(effective_contributions=(total**2/np.sum(values**2, axis=0)).tolist(),
            maximum_sample_fraction=(values.max(0)/total).tolist(),
            maximum_group_fraction=(np.asarray([values[g == label].sum(0) for label in labels]).max(0)/total).tolist())
    return dict(metric_order=['energy_per_site', 'imaginary_energy_per_site', 'variance_per_site', 'projected_norm'],
        endpoint_values=point.tolist(), endpoint_iid_delta_SE=iid_se.tolist(),
        endpoint_independent_block_jackknife_SE=jackknife.tolist(),
        endpoint_block_bootstrap_95=np.quantile(bootstrap, [.025, .975], axis=0).transpose(1, 2, 0).tolist(),
        candidate_minus_control=delta.tolist(), paired_iid_delta_SE=paired_iid_se.tolist(),
        paired_independent_block_jackknife_SE=delta_jk.tolist(),
        paired_block_bootstrap_95=np.quantile(bootstrap[:, 1]-bootstrap[:, 0], [.025, .975], axis=0).T.tolist(),
        norm_concentration=concentration(z), H2_concentration=concentration(second),
        residual_concentration=concentration(residual), exact_target_node_records=np.sum(p == 0, axis=0).tolist(),
        target_node_H2_contributions=np.sum(np.where(p == 0, second, 0), axis=0).tolist(),
        norm_weight_maximum=z.max(0).tolist(), sample_records=len(p), independent_groups=len(labels),
        raw_norm_numerators_unbiased=True, normalized_ratios_finite_sample_biased=True,
        IID_physical_orbit_exchange_draws=True, no_MH_equilibration_assumption=True,
        imaginary_energy_not_forced_zero=True, unseen_tail_precision_not_certified=True,
        full_H_squared_uses_node_safe_Hpsi_norm=True)


def common_active_profile(profiles: list) -> list[dict[int, int]]:
    """Union exact doubled-q layouts without discarding endpoint coordinates.

    Native banks must still pass exact inactive-child closure when repacked.
    This union changes execution padding only, never either physical state.
    """
    if not profiles or not profiles[0] or any(len(p) != len(profiles[0]) for p in profiles):
        raise ValueError('equal nonempty endpoint hierarchy lengths required')
    return [{q: max(p[node].get(q, 0) for p in profiles)
             for q in sorted(set().union(*(p[node] for p in profiles)))}
            for node in range(len(profiles[0]))]


def run(parent: Path, output: Path, *, seconds_cap: float = 3600.,
        checkpoint_paths: tuple[Path, ...] | None = None,
        records: int = 128, independent_blocks: int = 16,
        seed: int = 2610269017) -> dict:
    """Frozen IID paired audit; all rows are fixed before any H evaluation.

    The historical default retains its sealed endpoint identities and seeds.
    Explicit endpoints require a new RNG seed and are sealed before sampling.
    Different active profiles are embedded exactly in their common union.
    """
    if not 300 <= seconds_cap <= 7200: raise ValueError('registered wall cap is300..7200seconds')
    if (isinstance(records, bool) or not isinstance(records, int) or records < 16
            or isinstance(independent_blocks, bool) or not isinstance(independent_blocks, int)
            or independent_blocks < 4 or records % independent_blocks):
        raise ValueError('fixed records require at least four equal independent blocks')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32-10:
        raise ValueError('explicit uint32 seed with room for independent postprocessing required')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('new immutable audit directory required')
    historical = checkpoint_paths is None
    if historical:
        if (records, independent_blocks, seed) != (128, 16, 2610269017):
            raise ValueError('historical endpoint audit retains its original fixed protocol')
        paths = [parent/'common_layout/old18_bank.pkl', parent/'matched_radius/initialized_bank.pkl']
        expected_hashes = ['8f9c9eaa6e43999f0ecbba918542ac44953bdbf23a16aa643ed2f006ac7d0750',
            '6525f98a05a061af2d7e59ea8f024cf0fd2aab46e0d684e19faccf79a16d1a31']
    else:
        if not 2 <= len(checkpoint_paths) <= 4 or seed == 2610269017:
            raise ValueError('two to four frozen endpoints and a new audit seed required')
        paths = [Path(p).resolve() for p in checkpoint_paths]
        expected_hashes = [file_sha256(p) for p in paths]
    if [file_sha256(p) for p in paths] != expected_hashes: raise ValueError('frozen endpoint identity mismatch')
    files = [Path(__file__), ROOT/'su2_nqs/vmc/raw_tree_born.py', ROOT/'su2_nqs/vmc/triangular_raw_born.py',
        ROOT/'su2_nqs/vmc/exchange_bridge_importance.py', ROOT/'su2_nqs/models/triangular_clebsch_exchange_cache.py']
    registration = dict(task='CT12_fresh_exact_IID_raw_orbit_all_H_bridge_audit_v1',
        checkpoints=[str(p.resolve()) for p in paths], checkpoint_sha256=expected_hashes,
        model=dict(length=12,multiplicity_cap=20,maximum_q=10), j2=.125,
        records=records, independent_blocks=independent_blocks,
        draws_per_block=records//independent_blocks, seed=seed, bootstrap_seed=seed+4,
        seconds_cap=seconds_cap, bridge_fraction=.5,
        endpoint_probabilities=[1/len(paths)]*len(paths),
        all_H_bond_probabilities='abs(Jb)/sum(abs(Jb)); includes equal-spin/null swaps',
        proposal='Q=sum_k(Qorbit_k/Zraw_k+Qpush_k/Zraw_k)/(2*endpoint_count)',
        complete_A1=True, projection_chunk=4, H_microbatch=1, fixed_audit_endpoint=True,
        exact_nodes_retained_for_H_squared=True, no_MH_burn_or_hidden_label_sampling=True,
        no_ESS_cancellation_or_energy_based_stopping=True, no_training_or_attention=True,
        wall_budget_checked_between_operations_not_preemptive=True,
        source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in files})
    atomic_json(output/'registration.json',registration)
    frozen=output/'registered_sources';frozen.mkdir()
    for path in files: shutil.copy2(path,frozen/path.name)
    start=perf_counter();cpu=process_time();cap_end=start+seconds_cap
    result=dict(registration=registration,status='started',timings={},full_H_queries=0,completed_audit_records=0)
    def timed(label,callback):
        if perf_counter() >= cap_end: raise TimeoutError('wall gate before '+label)
        atomic_json(output/'progress.json',dict(result,operation=label,wall_seconds=perf_counter()-start))
        tick=perf_counter();value=callback();jax.block_until_ready(value)
        result['timings'][label]=perf_counter()-tick
        return value
    try:
        cfg=MixedClebschConfig(**registration['model']);banks=[];natives=[];profiles=[];norms=[];samplers=[];cache={}
        for path in paths:
            saved=load_checkpoint(path);native=saved['params']
            if any(saved['metadata'].get('model', {}).get(k, v) != v
                   for k, v in registration['model'].items()):
                raise ValueError('endpoint model differs from the registered CT12 decoder')
            if saved['metadata'].get('j2', .125) != .125:
                raise ValueError('this frozen transfer audit requires J2=.125 endpoints')
            profile=checked_active_widths(saved['metadata']['embedding']['active_top_widths'],nodes=15)
            norm=complex(triangular_raw_overlap(native,native,cfg))
            if not np.isfinite(norm) or abs(norm.imag)>2e-10 or abs(norm.real-1)>2e-10:
                raise ArithmeticError('frozen raw endpoint is not a normalized canonical state')
            bank=repack_active_clebsch_bank(native,cfg,profile);banks.append(bank);natives.append(native);profiles.append(profile);norms.append(norm.real)
            samplers.append(make_triangular_raw_born_sampler(bank,cfg,active_top_widths=profile,shared_tensor_cache=cache))
        profile=common_active_profile(profiles)
        banks=[repack_active_clebsch_bank(native,cfg,profile) for native in natives]
        result['individual_active_profiles']=profiles;result['common_active_profile']=profile
        del natives
        del saved,native
        result['raw_norms']=norms;result['sampler_diagnostics']=[s.diagnostics for s in samplers]
        images=np.asarray(timed('complete_A1_group',lambda:triangular_hnf_gamma_a1_group(((12,0),(0,12)))),np.int32)
        inverse_images=np.argsort(images,axis=1)
        bonds=triangular_j1j2_bonds(12,12,j2=.125,periodic=True)
        probability=np.abs(np.asarray([v for _,_,v in bonds]));probability/=probability.sum()
        edges=np.asarray([(a,b) for a,b,_ in bonds]);children=np.random.SeedSequence(registration['seed']).spawn(independent_blocks)
        rows=[];raw_rows=[];latent=[];log_replay=[];groups=[]
        tick=perf_counter()
        for group,child in enumerate(children):
            rng=np.random.default_rng(child)
            for _ in range(records//independent_blocks):
                if perf_counter() >= cap_end:
                    raise TimeoutError('wall budget before exact physical IID draw')
                endpoint=int(rng.integers(len(paths)));draw=samplers[endpoint].draw(rng)
                image=int(rng.integers(len(images)));row=draw.bits[inverse_images[image]].copy()
                use=bool(rng.random()<.5);bond=int(rng.choice(len(bonds),p=probability))
                if use:
                    a,b=edges[bond];row[a],row[b]=row[b],row[a]
                rows.append(row);raw_rows.append(draw.bits);groups.append(group)
                latent.append((endpoint,image,int(use),bond))
                log_replay.append(draw.log_probability_replay_error)
        result['timings'][str(records)+'_exact_IID_physical_draws']=perf_counter()-tick
        result['maximum_sampling_log_Born_replay_error']=max(log_replay)
        rows=np.asarray(rows,np.int32);groups=np.asarray(groups);latent=np.asarray(latent)
        np.savez(output/'registered_IID_rows.npz',rows=rows,raw_rows=np.asarray(raw_rows),groups=groups,
            endpoint_image_bridge_bond=latent,log_Born_replay_error=np.asarray(log_replay),raw_norms=np.asarray(norms))
        result['raw_draws']=len(rows);result['group_images']=len(images);result['physical_bonds']=len(bonds)
        del samplers,cache;gc.collect()
        prepared=tuple(prepare_feature_lut_bank(bank,cfg) for bank in banks)
        factory=lambda edges:make_feature_lut_exchange_queries(cfg,edges,active_top_widths=profile)
        action,metadata=make_orbit_exchange_bridge_products(factory,images,bonds,probability,
            n_sites=144,projection_chunk=4)
        result['kernel_metadata']=metadata
        amplitude=jax.jit(make_feature_lut_projected_amplitude(cfg,images,chunk=4,active_top_widths=profile))
        probe=rows[:1]
        for endpoint in range(len(paths)):
            outputs=[]
            for kind in ('cold','warm'):
                values=timed(f'H_cost_{endpoint}_{kind}',lambda endpoint=endpoint:action(prepared[endpoint],probe))
                outputs.append(np.asarray([complex(v[0]) for v in values]));result['full_H_queries']+=1
            if not np.isfinite(outputs).all() or not np.array_equal(outputs[0],outputs[1]):
                raise ArithmeticError('full coherent target/positive-guide cost replay failed')
            independent=complex(timed(f'independent_A1_probe_{endpoint}',lambda endpoint=endpoint:amplitude(prepared[endpoint],probe[0])))
            if abs(outputs[0][0]-independent)>2e-10*max(abs(independent),1e-300):
                raise ArithmeticError('H action center differs from independent complete A1 contraction')
        forecast=records*sum(result['timings'][f'H_cost_{k}_warm'] for k in range(len(paths)))
        result['measured_remaining_H_forecast_seconds']=forecast
        if 1.15*forecast>cap_end-perf_counter():
            raise TimeoutError('measured fixed-record full-H forecast exceeds registered audit wall budget')
        products=[]
        for index,row in enumerate(rows):
            pair=[]
            for endpoint in range(len(paths)):
                values=timed(f'audit_H_{index}_{endpoint}',lambda endpoint=endpoint,row=row:action(prepared[endpoint],row[None]))
                result['full_H_queries']+=1
                pair.append([complex(v[0]) for v in values])
            products.append(pair);result['completed_audit_records']=index+1
            np.savez(output/'partial_products.npz',products=np.asarray(products))
        data=np.asarray(products);norms=np.asarray(norms)
        psi=data[:,:,0]/np.sqrt(norms);hpsi=data[:,:,1]/np.sqrt(norms)
        orbit=data[:,:,2].real/norms;push=data[:,:,3].real/norms
        density=(orbit.sum(1)+push.sum(1))/(2*len(paths))
        if (not np.isfinite(data).all() or np.max(abs(data[:,:,2:].imag))>1e-10*np.max(abs(data[:,:,2:]))
                or np.any(orbit<0) or np.any(push<0) or np.any(density<=0)):
            raise ArithmeticError('finite positive complete orbit-bridge density required at every retained record')
        weight=abs(psi)**2/density[:,None]
        if np.max(weight)>2*len(paths)*(1+2e-10): raise ArithmeticError('normalizedraw orbit-mixture pointwise bound violated')
        if [file_sha256(path) for path in paths] != expected_hashes:
            raise ValueError('a frozen endpoint checkpoint changed during the audit')
        result['endpoint_checkpoint_hashes_rechecked_after_H']=True
        comparisons={f'{b}_minus_{a}': paired_iid_moments(psi[:,[a,b]],hpsi[:,[a,b]],density,groups,
            n_sites=144,seed=registration['bootstrap_seed'])
            for a in range(len(paths)) for b in range(a+1,len(paths))}
        result.update(status='complete_fresh_IID_full_A1_energy_variance_audit',comparisons=comparisons,
            observed_bridge_fraction=float(latent[:,2].mean()),endpoint_draw_counts=np.bincount(latent[:,0],minlength=len(paths)).tolist(),
            physical_variance_evaluated_with_all_H_nodes=True,no_missing_q_zero_records=True,
            precision_not_assumed_from_weight_bound=True,attention_release=False,optimization_updates=0)
        if len(paths)==2:result['moments']=comparisons['1_minus_0']
        np.savez(output/'untouched_products.npz',rows=rows,groups=groups,psi=psi,Hpsi=hpsi,
            Qorbit=orbit,Qpush=push,proposal_density=density,raw_norms=norms,latent_labels=latent)
    except TimeoutError as error:
        result.update(status='bounded_incomplete_IID_audit',error=str(error))
    except Exception as error:
        result.update(status='failed_validation',error=repr(error));raise
    finally:
        result.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu,
            peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
            no_training_no_attention=True,all_records_fixed_before_H=True)
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seconds-cap',type=float,default=3600.)
    parser.add_argument('--checkpoints',type=Path,nargs='+')
    parser.add_argument('--records',type=int,default=128)
    parser.add_argument('--independent-blocks',type=int,default=16)
    parser.add_argument('--seed',type=int,default=2610269017)
    args=parser.parse_args();run(args.parent,args.output,seconds_cap=args.seconds_cap,
        checkpoint_paths=None if args.checkpoints is None else tuple(args.checkpoints),
        records=args.records,independent_blocks=args.independent_blocks,seed=args.seed)
