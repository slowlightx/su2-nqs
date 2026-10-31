"""Frozen-state enlarged audit; cheap baseline pools expose variance tails."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from su2_nqs.benchmarks.finite_audit_statistics import FIELDS, paired_chain_summary, residual_tail_diagnostics
from su2_nqs.benchmarks.finite_scalar_dressing import finite_step_moments
from su2_nqs.benchmarks.parton_swap_batch import ScannedClosedShellSwapEvaluator
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax,
    triangular_pi_zero_flux_hopping, BOUNDARY_ORBIT, A1_COEFFICIENTS,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from su2_nqs.vmc.spin_physical import padded_spin_connectivity
from .spin_finite_environment_dressing import _bank, ROOT


def _sealed(directory: Path, expected: str) -> dict:
    result = json.loads((directory/'results.json').read_text())
    if result['result_sha256'] != expected or canonical_sha256({k:v for k,v in result.items() if k!='result_sha256'}) != expected:
        raise RuntimeError('source result seal mismatch')
    return result


def _load_arrays(directory: Path, row: dict, prefix: str = 'replicate') -> dict:
    path = directory/f"{prefix}_{row['index']}.npz"
    if file_sha256(path) != row['bundle_sha256']:
        raise RuntimeError('array seal mismatch')
    with np.load(path,allow_pickle=False) as b:
        return {k:b[k] for k in b.files}


def _baseline_data(data: dict) -> dict:
    return dict(psi=data['psi'], hpsi=data['hpsi'], proposal_squared=data['proposal_squared'],
                direction=np.zeros_like(data['psi']), hdirection=np.zeros_like(data['psi']))


def run_spin_finite_environment_audit(config_path: Path, output: Path, *, stream_index: int | None = None) -> dict:
    """Run one restartable independent stream, or finalize four completed streams."""
    spec = json.loads(config_path.read_text())
    if spec['task']!='spin_finite_environment_audit' or not spec['preregistered_before_run'] or spec['optimization_steps'] or spec['remote_training_on_pass']:
        raise ValueError('fixed local audit protocol required')
    source_dir = ROOT/spec['source_directory'];source = _sealed(source_dir,spec['source_result_sha256'])
    old = source['config']
    if spec['primary_epsilon'] != old['primary_epsilon'] or set(spec['replicate_seeds']) & set(old['replicate_seeds']):
        raise ValueError('primary step must stay fixed and seeds must be new')
    for relative, digest in source['source_manifest'].items():
        if file_sha256(ROOT/relative) != digest:
            raise RuntimeError(f'frozen physical implementation changed: {relative}')
    reference_dir = ROOT/spec['tail_reference_directory']
    reference = _sealed(reference_dir,spec['tail_reference_result_sha256'])
    if set(spec['replicate_seeds']) & set(reference['config']['replicate_seeds']):
        raise ValueError('tail reference and audit seeds overlap')
    output.mkdir(parents=True,exist_ok=True)
    manifest = transitive_local_source_manifest((Path(__file__).resolve(),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    identity = dict(config_sha256=canonical_sha256(spec),source_manifest_sha256=canonical_sha256(manifest))
    eps = spec['primary_epsilon'];nc = spec['chain_count'];nb = len(spec['dressed_block_indices'])
    if stream_index is None:
        rows,pools,streams = [],[],[]
        for index in range(len(spec['replicate_seeds'])):
            row = json.loads((output/f'replicate_{index}.json').read_text())
            if any(row[k]!=v for k,v in identity.items()):raise RuntimeError('stream identity mismatch')
            rows.append(row);streams.append(_load_arrays(output,row))
            pool_meta = json.loads((output/f'pool_{index}.json').read_text())
            if any(pool_meta[k]!=v for k,v in identity.items()):raise RuntimeError('pool identity mismatch')
            pools.append(_load_arrays(output,pool_meta,'pool'))
        data = {k:np.concatenate([s[k] for s in streams]) for k in FIELDS}
        chain_ids = np.concatenate([np.tile(np.arange(nc)+i*nc,nb) for i in range(len(streams))])
        summary = paired_chain_summary(data,chain_ids,eps,replicates=spec['bootstrap_replicates'],seed=spec['bootstrap_seed'])
        baseline = {k:np.concatenate([s[k] for s in pools]) for k in ('psi','hpsi','proposal_squared')}
        pool_chains = np.concatenate([np.tile(np.arange(nc)+i*nc,spec['pool_measurement_blocks']) for i in range(len(pools))])
        baseline_summary = paired_chain_summary(_baseline_data(baseline),pool_chains,0.,replicates=spec['bootstrap_replicates'],seed=spec['bootstrap_seed']+1)
        historical = [_load_arrays(reference_dir,row) for row in reference['replicates']]
        refp,refh,refq = [np.concatenate([s[k] for s in historical]) for k in ('psi','hpsi','proposal_squared')]
        refe = reference['aggregate']['energy_total']
        thresholds = np.quantile(abs(refh-refe*refp)**2/refq,spec['tail_quantiles'])
        tail = residual_tail_diagnostics(data,eps,reference_energy=refe,thresholds=thresholds)
        pool_tail = residual_tail_diagnostics(_baseline_data(baseline),0.,reference_energy=refe,thresholds=thresholds)
        cumulative=[]
        for blocks in range(1,nb+1):
            select=np.concatenate([np.arange(i*nc*nb,i*nc*nb+nc*blocks) for i in range(len(streams))])
            values=finite_step_moments(*(data[k][select] for k in FIELDS),eps)
            cumulative.append(dict(sample_count=len(select),**values))
        per_stream=[]
        for i,s in enumerate(streams):
            per_stream.append(paired_chain_summary(s,np.tile(np.arange(nc),nb),eps,replicates=spec['bootstrap_replicates'],seed=spec['bootstrap_seed']+10+i))
        result=dict(task=spec['task'],status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
                    config=spec,**identity,source_manifest=manifest,replicates=rows,aggregate=summary,
                    per_stream=per_stream,baseline_pool=baseline_summary,tail=tail,baseline_pool_tail=pool_tail,
                    cumulative=cumulative,remote_training_licensed=False,optimization_steps=0,
                    source_carrier=old['carrier'],source_direction=old['direction'],proposal_kappa=source['proposal_kappa'],
                    total_worker_seconds=sum(r['wall_seconds_this_invocation'] for r in rows),runtime_identity=numerical_runtime_identity(),
                    claim_boundary='Fixed operator/carrier/epsilon and full projection; 1024 fresh finite states, 16384 cheap baseline states. Pool data do not substitute for dressed H actions. No trimming, no training, no adaptive selection. Bootstrap cannot certify absent rare configurations.')
        result['result_sha256']=canonical_sha256(result);atomic_json(output/'results.json',result)
        print('Audit complete:',summary,flush=True)
        return result
    index=int(stream_index)
    if not 0<=index<len(spec['replicate_seeds']):raise ValueError('invalid stream index')
    meta_path=output/f'replicate_{index}.json'
    if meta_path.exists():
        row=json.loads(meta_path.read_text())
        if any(row[k]!=v for k,v in identity.items()):raise RuntimeError('completed stream identity mismatch')
        _load_arrays(output,row);return row
    started=perf_counter();orbit=build_n36_pi_zero_flux_orbit()
    carrier=jax.jit(lambda s:projected_pi_zero_flux_a1_amplitudes_jax(s,orbit=orbit))
    def amplitude(states):
        answer=[];size=old['amplitude_batch_size']
        for start in range(0,len(states),size):
            batch=states[start:start+size];padded=np.repeat(batch[:1],size,axis=0);padded[:len(batch)]=batch
            answer.append(np.asarray(carrier(padded))[:len(batch)])
        return np.concatenate(answer)
    permutations,coefficients,bank=_bank(6,old['cells'])
    orbitals=np.asarray([np.linalg.eigh(triangular_pi_zero_flux_hopping(6,boundary_x=x,boundary_y=y))[1][:,:18] for x,y in BOUNDARY_ORBIT])
    evaluator=ScannedClosedShellSwapEvaluator(orbitals,A1_COEFFICIENTS,permutations,coefficients,amplitude,
        chunk_size=spec['word_chunk_size'],sample_batch_size=spec['sample_batch_size'],condition_limit=old['condition_limit'])
    bonds=triangular_j1j2_bonds(6,6,j1=1.,j2=old['j2_over_j1'],periodic=True)
    def h_images(states, dressed=False):
        connected,elements,valid,_=padded_spin_connectivity(states,bonds)
        values=np.zeros(elements.shape,complex);values[valid]=amplitude(connected[valid])
        base=np.sum(elements*values,axis=1)
        if not dressed:return base
        image,cost=evaluator(connected[valid]);values[valid]=image[:,0]
        return base,np.sum(elements*values,axis=1),cost
    replay_data=_load_arrays(source_dir,source['replicates'][0])
    test_states=replay_data['configurations'][:2]
    t=perf_counter();d,_=evaluator(test_states);h,hd,_=h_images(test_states,True)
    errors={k:float(np.linalg.norm(observed-replay_data[k][:2])/np.linalg.norm(replay_data[k][:2])) for k,observed in (('psi',amplitude(test_states)),('direction',d[:,0]),('hpsi',h),('hdirection',hd))}
    if max(errors.values())>2e-10:raise RuntimeError('batched kernel disagrees with sealed physical state')
    replay_seconds=perf_counter()-t
    print(f'stream {index}: sealed amplitude/H replay {errors}; {replay_seconds:.1f}s',flush=True)
    pool_path=output/f'pool_{index}.npz';pool_meta_path=output/f'pool_{index}.json'
    if pool_meta_path.exists():
        pool_meta=json.loads(pool_meta_path.read_text())
        if any(pool_meta[k]!=v for k,v in identity.items()):raise RuntimeError('pool identity mismatch')
        pool=_load_arrays(output,pool_meta,'pool')
    else:
        kappa=source['proposal_kappa'];proposal=lambda s:np.sqrt(abs(amplitude(s))**2+kappa)
        samples=sample_fixed_wavefunction_exchange(proposal,n_sites=36,n_up=18,chain_count=nc,
            burn_sweeps=spec['burn_exchange_steps_per_chain'],measurement_blocks=spec['pool_measurement_blocks'],
            decorrelation_sweeps=spec['pool_decorrelation_exchange_steps'],seed=spec['replicate_seeds'][index])
        states=samples.configurations;p=amplitude(states)
        h=np.concatenate([h_images(states[j:j+8]) for j in range(0,len(states),8)])
        pool=dict(configurations=states,psi=p,hpsi=h,proposal_squared=abs(p)**2+kappa)
        np.savez_compressed(pool_path,**pool)
        pool_meta=dict(index=index,**identity,bundle_sha256=file_sha256(pool_path),acceptance_fraction=samples.acceptance_fraction,burn_acceptance_fraction=samples.burn_acceptance_fraction)
        atomic_json(pool_meta_path,pool_meta)
    select=np.concatenate([np.arange(block*nc,(block+1)*nc) for block in spec['dressed_block_indices']])
    states=pool['configurations'][select]
    partial_path=output/f'partial_{index}.npz';partial_meta_path=output/f'partial_{index}.json'
    ds,hs=[],[];fallback=0;maximum=0.
    if partial_meta_path.exists():
        meta=json.loads(partial_meta_path.read_text())
        if any(meta[k]!=v for k,v in identity.items()) or meta['bundle_sha256']!=file_sha256(partial_path):raise RuntimeError('partial checkpoint mismatch')
        with np.load(partial_path) as b:ds=list(b['direction']);hs=list(b['hdirection'])
        fallback,maximum=meta['direct_fallback_count'],meta['maximum_component_condition']
    batch=spec['center_checkpoint_batch_size']
    for start in range(len(ds),len(states),batch):
        if perf_counter()-started>spec['worker_wall_limit_seconds']:raise TimeoutError('worker cap; resume from partial checkpoint')
        current=states[start:start+batch];d,c0=evaluator(current);h,hd,c1=h_images(current,True)
        np.testing.assert_allclose(h,pool['hpsi'][select[start:start+len(current)]],atol=1e-24,rtol=2e-11)
        ds.extend(d[:,0]);hs.extend(hd)
        fallback+=c0['direct_fallback_count']+c1['direct_fallback_count']
        maximum=max(maximum,c0['maximum_component_condition'],c1['maximum_component_condition'])
        np.savez_compressed(partial_path,direction=np.asarray(ds),hdirection=np.asarray(hs))
        atomic_json(partial_meta_path,dict(index=index,**identity,bundle_sha256=file_sha256(partial_path),completed_centers=len(ds),direct_fallback_count=fallback,maximum_component_condition=maximum))
        print(f'stream {index}: {len(ds)}/{len(states)} full-H finite centers; elapsed {perf_counter()-started:.1f}s',flush=True)
    data=dict(configurations=states,psi=pool['psi'][select],hpsi=pool['hpsi'][select],proposal_squared=pool['proposal_squared'][select],direction=np.asarray(ds),hdirection=np.asarray(hs),pool_indices=select)
    path=output/f'replicate_{index}.npz';np.savez_compressed(path,**data)
    row=dict(index=index,seed=spec['replicate_seeds'][index],**identity,bundle_sha256=file_sha256(path),pool_bundle_sha256=pool_meta['bundle_sha256'],sample_count=len(states),pool_sample_count=len(pool['psi']),replay_errors=errors,replay_seconds=replay_seconds,bank_cost=bank,direct_fallback_count=fallback,maximum_component_condition=maximum,wall_seconds_this_invocation=perf_counter()-started)
    atomic_json(meta_path,row)
    print(f'stream {index}: COMPLETE (energies withheld until all streams finish)',flush=True)
    return row
