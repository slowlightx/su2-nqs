"""Fresh two-phase stratified audit of one frozen SU2/A1 finite-step state."""

from datetime import datetime,timezone
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from su2_nqs.benchmarks.stratified_finite_audit import (select_strata,carrier_moments,correction_moments,
    stratified_correction,physical_values_and_jacobian,summarize_independent_pools,NAMES)
from su2_nqs.benchmarks.parton_swap_batch import ScannedClosedShellSwapEvaluator
from su2_nqs.benchmarks.triangular_induced_parton import (build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes_jax,triangular_pi_zero_flux_hopping,BOUNDARY_ORBIT,A1_COEFFICIENTS)
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest,numerical_runtime_identity
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.spin_physical import padded_spin_connectivity
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from .spin_finite_environment_dressing import ROOT,_bank
from .spin_finite_environment_audit import _sealed,_load_arrays


class FrozenEnvironmentEvaluator:
    """Reusable physical-amplitude/H interface for the sealed analytic carrier."""

    def __init__(self, physical_source: dict, spec: dict) -> None:
        self.old=physical_source['config'];self.kappa=physical_source['proposal_kappa']
        orbit=build_n36_pi_zero_flux_orbit()
        self.compiled=jax.jit(lambda s:projected_pi_zero_flux_a1_amplitudes_jax(s,orbit=orbit))
        p,c,self.bank=_bank(6,self.old['cells'])
        v=np.asarray([np.linalg.eigh(triangular_pi_zero_flux_hopping(6,boundary_x=x,boundary_y=y))[1][:,:18] for x,y in BOUNDARY_ORBIT])
        self.operator=ScannedClosedShellSwapEvaluator(v,A1_COEFFICIENTS,p,c,self.amplitude,
            chunk_size=spec['word_chunk_size'],sample_batch_size=spec['sample_batch_size'],condition_limit=self.old['condition_limit'])
        self.bonds=triangular_j1j2_bonds(6,6,j1=1.,j2=self.old['j2_over_j1'],periodic=True)

    def amplitude(self, states: np.ndarray) -> np.ndarray:
        """Frozen analytic physical amplitudes, never floored."""
        values=[];size=self.old['amplitude_batch_size']
        for start in range(0,len(states),size):
            batch=states[start:start+size];pad=np.repeat(batch[:1],size,axis=0);pad[:len(batch)]=batch
            values.append(np.asarray(self.compiled(pad))[:len(batch)])
        return np.concatenate(values)

    def h_images(self, states: np.ndarray, *, dressed: bool=False) -> tuple:
        """Full deterministic H action; optional exact projected direction action."""
        connected,elements,valid,_=padded_spin_connectivity(states,self.bonds)
        values=np.zeros(elements.shape,complex);values[valid]=self.amplitude(connected[valid])
        h=np.sum(elements*values,axis=1)
        if not dressed:return h,None,None
        direction,cost=self.operator(connected[valid]);values[valid]=direction[:,0]
        return h,np.sum(elements*values,axis=1),cost


def _summarize_pool(pool: dict, selected: dict, selection: dict, epsilon: float, reference: float) -> dict:
    cheap=carrier_moments(pool['psi'],pool['hpsi'],pool['proposal_squared'],reference)
    correction=correction_moments(*(selected[k] for k in ('psi','direction','hpsi','hdirection','proposal_squared')),epsilon,reference)
    counts=np.asarray([s['population_count'] for s in selection['strata']],dtype=int)
    estimate=stratified_correction(correction,selected['stratum'],counts)
    moments=np.r_[cheap.mean(axis=0),estimate['mean']]
    cov=np.zeros((6,6));cov[3:,3:]=estimate['covariance']
    values,jac=physical_values_and_jacobian(moments,reference)
    contributions=[]
    # Telescoping diagnostic: add each estimated stratum correction in a fixed order.
    previous=np.r_[cheap.mean(axis=0),np.zeros(3)]
    for s in estimate['strata']:
        after=previous.copy();after[3:]+=s['weighted_moment_contribution']
        v0,_=physical_values_and_jacobian(previous,reference);v1,_=physical_values_and_jacobian(after,reference)
        contributions.append(dict(stratum=s['stratum'],energy_change_total=float(v1[5]-v0[5]),variance_change_total=float(v1[6]-v0[6])))
        previous=after
    return dict(moments=moments.tolist(),conditional_covariance=cov.tolist(),
                physical=dict(zip(NAMES,map(float,values))),strata=estimate['strata'],
                telescoping_strata=contributions,
                conditional_design_se=dict(zip(NAMES,map(float,np.sqrt(np.maximum(np.diag(jac@cov@jac.T),0))))))


def run_spin_stratified_environment_audit(config_path: Path, output: Path, *, pool_index: int | None=None) -> dict:
    """Run one sealed independent pool, or aggregate all completed pools."""
    spec=json.loads(config_path.read_text())
    if spec['task']!='spin_stratified_environment_audit' or not spec['preregistered_before_run'] or spec['optimization_steps'] or spec['remote_training_on_pass']:
        raise ValueError('registered frozen-state protocol required')
    parent_dir=ROOT/spec['source_directory'];parent=_sealed(parent_dir,spec['source_result_sha256'])
    original_dir=ROOT/parent['config']['source_directory'];original=_sealed(original_dir,parent['config']['source_result_sha256'])
    if spec['primary_epsilon']!=parent['config']['primary_epsilon'] or len(set(spec['pool_seeds']))!=len(spec['pool_seeds']):raise ValueError('fixed step and independent seeds required')
    if set(spec['pool_seeds']) & (set(parent['config']['replicate_seeds'])|set(original['config']['replicate_seeds'])):raise ValueError('reused audit seed')
    for path,digest in parent['source_manifest'].items():
        if file_sha256(ROOT/path)!=digest:raise RuntimeError(f'frozen source changed: {path}')
    manifest=transitive_local_source_manifest((Path(__file__).resolve(),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    identity=dict(config_sha256=canonical_sha256(spec),source_manifest_sha256=canonical_sha256(manifest))
    output.mkdir(parents=True,exist_ok=True)
    eps=spec['primary_epsilon'];reference=parent['tail']['reference_energy'];cut=np.asarray(parent['tail']['thresholds'])
    if pool_index is None:
        rows=[]
        for i in range(len(spec['pool_seeds'])):
            row=json.loads((output/f'replicate_{i}.json').read_text())
            if any(row[k]!=v for k,v in identity.items()):raise RuntimeError('replicate identity mismatch')
            _load_arrays(output,row)
            if file_sha256(output/f'pool_{i}.npz')!=row['pool_bundle_sha256']:raise RuntimeError('pool seal mismatch')
            rows.append(row)
        moments=np.asarray([r['summary']['moments'] for r in rows]);cov=np.asarray([r['summary']['conditional_covariance'] for r in rows])
        summary=summarize_independent_pools(moments,cov,reference)
        result=dict(task=spec['task'],status='complete',created_utc=datetime.now(timezone.utc).isoformat(),config=spec,
            **identity,source_manifest=manifest,aggregate=summary,replicates=rows,reference_energy=reference,thresholds=cut.tolist(),
            sample_count=sum(r['sample_count'] for r in rows),pool_sample_count=sum(r['pool_sample_count'] for r in rows),
            independent_chain_count=len(rows)*spec['chain_count'],independent_pool_count=len(rows),
            stratum_population_counts=np.sum([r['population_counts'] for r in rows],axis=0).tolist(),
            stratum_sample_counts=np.sum([r['selected_counts'] for r in rows],axis=0).tolist(),
            total_worker_seconds=sum(r['wall_seconds_this_invocation'] for r in rows),remote_training_licensed=False,
            runtime_identity=numerical_runtime_identity(),claim_boundary=spec['scope'])
        result['result_sha256']=canonical_sha256(result);atomic_json(output/'results.json',result)
        print('Stratified audit complete:',summary,flush=True);return result
    index=int(pool_index)
    if not 0<=index<len(spec['pool_seeds']):raise ValueError('invalid pool index')
    result_path=output/f'replicate_{index}.json'
    if result_path.exists():
        row=json.loads(result_path.read_text())
        if any(row[k]!=v for k,v in identity.items()):raise RuntimeError('completed pool identity mismatch')
        _load_arrays(output,row);return row
    start_time=perf_counter();evaluator=FrozenEnvironmentEvaluator(original,spec)
    old=_load_arrays(original_dir,original['replicates'][0]);states=old['configurations'][:2]
    d,_=evaluator.operator(states);h,hd,_=evaluator.h_images(states,dressed=True)
    replay={k:float(np.linalg.norm(x-old[k][:2])/np.linalg.norm(old[k][:2])) for k,x in (('psi',evaluator.amplitude(states)),('direction',d[:,0]),('hpsi',h),('hdirection',hd))}
    if max(replay.values())>2e-10:raise RuntimeError('physical-state replay failed')
    print(f'pool {index}: sealed replay {replay}',flush=True)
    pool_path=output/f'pool_{index}.npz';pool_meta_path=output/f'pool_{index}.json'
    if pool_meta_path.exists():
        meta=json.loads(pool_meta_path.read_text())
        if any(meta[k]!=v for k,v in identity.items()):raise RuntimeError('pool identity mismatch')
        pool=_load_arrays(output,meta,'pool')
    else:
        proposal=lambda s:np.sqrt(abs(evaluator.amplitude(s))**2+evaluator.kappa)
        samples=sample_fixed_wavefunction_exchange(proposal,n_sites=36,n_up=18,chain_count=spec['chain_count'],
            burn_sweeps=spec['burn_exchange_steps_per_chain'],measurement_blocks=spec['measurement_blocks'],
            decorrelation_sweeps=spec['decorrelation_exchange_steps'],seed=spec['pool_seeds'][index])
        s=samples.configurations;p=evaluator.amplitude(s)
        h=np.concatenate([evaluator.h_images(s[j:j+8])[0] for j in range(0,len(s),8)])
        q=abs(p)**2+evaluator.kappa;score=abs(h-reference*p)**2/q;labels=np.searchsorted(cut,score,side='right')
        pool=dict(configurations=s,psi=p,hpsi=h,proposal_squared=q,stratum=labels,score=score)
        np.savez_compressed(pool_path,**pool)
        meta=dict(index=index,**identity,bundle_sha256=file_sha256(pool_path),acceptance_fraction=samples.acceptance_fraction,burn_acceptance_fraction=samples.burn_acceptance_fraction)
        atomic_json(pool_meta_path,meta)
    selection=select_strata(pool['stratum'],spec['stratum_allocation'],seed=spec['selection_seeds'][index]);ix=selection['indices']
    if len(ix)>spec['maximum_selected_per_pool']:raise RuntimeError('registered resource cap; no tail truncation allowed')
    print(f'pool {index}: fixed selected counts {[r["sample_count"] for r in selection["strata"]]}',flush=True)
    partial_path=output/f'partial_{index}.npz';partial_meta_path=output/f'partial_{index}.json';ds=[];hds=[];fallback=0
    if partial_meta_path.exists():
        partial=json.loads(partial_meta_path.read_text())
        if any(partial[k]!=v for k,v in identity.items()) or file_sha256(partial_path)!=partial['bundle_sha256']:raise RuntimeError('partial seal mismatch')
        with np.load(partial_path) as b:ds=list(b['direction']);hds=list(b['hdirection'])
        fallback=partial['direct_fallback_count']
    batch=spec['checkpoint_batch_size']
    for position in range(len(ds),len(ix),batch):
        if perf_counter()-start_time>spec['worker_wall_limit_seconds']:raise TimeoutError('registered wall cap; partial checkpoint retained')
        chosen=ix[position:position+batch];s=pool['configurations'][chosen]
        d,c0=evaluator.operator(s);h,hd,c1=evaluator.h_images(s,dressed=True)
        np.testing.assert_allclose(h,pool['hpsi'][chosen],atol=1e-24,rtol=2e-11)
        ds.extend(d[:,0]);hds.extend(hd);fallback+=c0['direct_fallback_count']+c1['direct_fallback_count']
        np.savez_compressed(partial_path,direction=np.asarray(ds),hdirection=np.asarray(hds))
        atomic_json(partial_meta_path,dict(index=index,**identity,bundle_sha256=file_sha256(partial_path),completed_centers=len(ds),direct_fallback_count=fallback))
        print(f'pool {index}: {len(ds)}/{len(ix)} complete full-H centers; elapsed {perf_counter()-start_time:.1f}s',flush=True)
    selected={k:pool[k][ix] for k in ('configurations','psi','hpsi','proposal_squared','stratum','score')}
    selected.update(direction=np.asarray(ds),hdirection=np.asarray(hds),pool_indices=ix,inclusion_probability=selection['inclusion_probability'])
    summary=_summarize_pool(pool,selected,selection,eps,reference)
    path=output/f'replicate_{index}.npz';np.savez_compressed(path,**selected)
    row=dict(index=index,seed=spec['pool_seeds'][index],selection_seed=spec['selection_seeds'][index],**identity,
        bundle_sha256=file_sha256(path),pool_bundle_sha256=file_sha256(pool_path),sample_count=len(ix),pool_sample_count=len(pool['psi']),
        population_counts=[s['population_count'] for s in selection['strata']],selected_counts=[s['sample_count'] for s in selection['strata']],
        summary=summary,replay_errors=replay,bank_cost=evaluator.bank,direct_fallback_count=fallback,wall_seconds_this_invocation=perf_counter()-start_time)
    atomic_json(result_path,row);print(f'pool {index}: COMPLETE',flush=True);return row
