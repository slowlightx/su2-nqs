"""N36 physical CG decoder cost-only preflight; no training or Hilbert enumeration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import signal
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact,load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest,numerical_runtime_identity
from su2_nqs.models.cell_multiplicity_clebsch_tree import (
    SpinCellMultiplicityConfig,spin_cell_multiplicity_tree_raw_amplitude,
    cell_multiplicity_layouts,spin_cell_multiplicity_parameter_count,
)
from su2_nqs.models.spin_clebsch_tree import spin_cg_tensor
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector,spatial_projected_amplitude
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from .spin import _groups,_bonds
from .spin_vap_decoder import init_decoder

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_decoder_cost.json'


def physical_config(rows: int,columns: int,width: int) -> SpinCellMultiplicityConfig:
    """Even rectangular four-spin cells; full doubled-q channels, no truncation."""
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (rows,columns,width)):
        raise TypeError('rows, columns and width must be integers')
    if rows<2 or columns<2 or rows%2 or columns%2 or rows*columns<8:
        raise ValueError('need even sides and at least two complete cells')
    cells=tuple(tuple((r+dr)*columns+c+dc for dr,dc in ((0,0),(0,1),(1,0),(1,1)))
                for r in range(0,rows,2)for c in range(0,columns,2))
    return SpinCellMultiplicityConfig('square_plaquette',cells,feature_dim=width,merge_mode='full_joint')


def amplitude_functions(config: SpinCellMultiplicityConfig,group: tuple) -> dict:
    """Reuse physical contractions and complete coherent A1 projector unchanged."""
    projector=PhysicalSpatialProjector(group)
    def raw(params,bits,model_config=config):
        return (spin_cell_multiplicity_tree_raw_amplitude(params['real'],bits,model_config)
                +1j*spin_cell_multiplicity_tree_raw_amplitude(params['imag'],bits,model_config))
    def projected(params,bits):
        return spatial_projected_amplitude(params,bits,config,projector,amplitude_fn=raw)
    return dict(raw=raw,a1=projected,projector=projector)


def peak_rss_bytes() -> int:
    """Process high-water RSS, not a per-kernel allocation or GPU memory meter."""
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value*(1 if sys.platform=='darwin' else 1024))


def benchmark(fn,params: dict,states: np.ndarray,repeats: int) -> tuple[dict,object,np.ndarray]:
    """Synchronize every execution; keep compilation separate from warmed runtime."""
    before=peak_rss_bytes();started=perf_counter()
    lowered=jax.jit(fn).lower(params,jnp.asarray(states))
    lowering=perf_counter()-started;started=perf_counter()
    executable=lowered.compile();compilation=perf_counter()-started
    memory=executable.memory_analysis()
    fields=('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes',
            'alias_size_in_bytes','generated_code_size_in_bytes')
    estimates={key:int(getattr(memory,key)) for key in fields if memory is not None and hasattr(memory,key)}
    started=perf_counter();first=np.asarray(jax.block_until_ready(executable(params,jnp.asarray(states))))
    first_seconds=perf_counter()-started;times=[];replay=0.
    for _ in range(repeats):
        started=perf_counter();value=np.asarray(jax.block_until_ready(executable(params,jnp.asarray(states))))
        times.append(perf_counter()-started)
        replay=max(replay,float(np.max(abs(value-first))))
    report=dict(batch=len(states),lowering_seconds=lowering,compile_seconds=compilation,
                first_seconds=first_seconds,execution_seconds=times,replay_max=replay,
                peak_rss_before_bytes=before,peak_rss_after_bytes=peak_rss_bytes(),
                compiler_memory_estimates=estimates,finite=bool(np.isfinite(first).all()))
    return report,executable,first


def batched_host_amplitudes(executable,params: dict,states: np.ndarray,batch: int=8) -> np.ndarray:
    """Chunk host validation inputs through the already timed fixed-batch executable."""
    states=np.asarray(states,dtype=np.int32);values=[]
    for start in range(0,len(states),batch):
        chunk=states[start:start+batch];count=len(chunk)
        if count<batch: chunk=np.concatenate((chunk,np.repeat(chunk[-1:],batch-count,axis=0)))
        values.extend(np.asarray(executable(params,jnp.asarray(chunk)))[:count])
    return np.asarray(values)


def registration(spec: dict) -> dict:
    """Seal probes, all numerical sources and runtime before measurements."""
    record=dict(config=spec,probe_sha256=file_sha256(ROOT/spec['probe_file']),
        runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
                    package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    record['registration_sha256']=canonical_sha256(record)
    return record


def run(config_path: Path,geometry: str) -> dict:
    """One isolated geometry, fresh parameters, frozen configurations, no optimization."""
    spec=json.loads(config_path.read_text());output=ROOT/spec['output']
    reg=json.loads((output/'registration.json').read_text())
    if reg!=registration(spec): raise ValueError('registration mismatch')
    cases=[c for c in spec['cases'] if c['geometry']==geometry]
    if len(cases)!=1: raise ValueError('unregistered geometry')
    case=cases[0];destination=output/geometry
    if (destination/'progress.json').exists(): raise FileExistsError(destination)
    started=perf_counter();measurements={};validation={}
    def persist(status: str):
        atomic_json(destination/'progress.json',dict(status=status,geometry=geometry,
            registration_sha256=reg['registration_sha256'],measurements=measurements,
            validation=validation,elapsed_seconds=perf_counter()-started,peak_rss_bytes=peak_rss_bytes()))
        print(geometry,status,round(perf_counter()-started,3),'s',round(peak_rss_bytes()/2**30,3),'GiB',flush=True)
        if peak_rss_bytes()>spec['peak_rss_limit_gib']*2**30: raise MemoryError('registered RSS cap reached')
    def alarm(_signum,_frame): raise TimeoutError('registered process wall cap reached')
    old_handler=signal.signal(signal.SIGALRM,alarm)
    signal.alarm(spec['wall_limit_seconds_per_process'])
    persist('started')
    try:
        config=physical_config(6,6,spec['width']);groups=_groups(geometry,6,6)
        bonds=_bonds(geometry,6,6,case['j2_over_j1']);n_images=len(groups['gamma_a1'])
        template=init_decoder(spec['seed'],config)
        save_parameter_artifact(destination/'parameters.json',template,
                               identity=dict(registration_sha256=reg['registration_sha256'],geometry=geometry))
        params=load_parameter_artifact(destination/'parameters.json',template,
                               expected_identity=dict(registration_sha256=reg['registration_sha256'],geometry=geometry))
        validation['parameter_reload_identical']=pytree_sha256(params)==pytree_sha256(template)
        with np.load(ROOT/spec['probe_file']) as archive: states=archive[geometry+'_states'].astype(np.int32)
        assert states.shape==(8,36) and np.all(states.sum(axis=1)==18)
        layouts=cell_multiplicity_layouts(config);cg_tables=[]
        for left,right in sorted(set((x.left_size,x.right_size)for x in layouts)):
            begin=perf_counter();table=spin_cg_tensor(left,right)
            cg_tables.append(dict(left=left,right=right,bytes=int(table.nbytes),seconds=perf_counter()-begin))
            persist(f'CG_{left}_{right}_ready')
        functions=amplitude_functions(config,groups['gamma_a1']);outputs={};amp_executables={}
        for mode in ('raw','a1'):
            amplitude=functions[mode]
            batched=jax.vmap(amplitude,in_axes=(None,0))
            for batch in spec['batches']:
                key=f'{mode}_amplitude_b{batch}';persist(key+'_begin')
                m,executable,value=benchmark(batched,params,states[:batch],spec['timing_repeats'])
                measurements[key]=m;outputs[key]=value
                if batch==8: amp_executables[mode]=executable
                else: del executable
                persist(key+'_done')
        for mode in ('raw','a1'):
            amplitude=functions[mode]
            log_amplitude=lambda p,x:jnp.log(amplitude(p,x))
            ratios=make_full_heisenberg_ratios(log_amplitude,bonds)
            for batch in spec['batches']:
                key=f'{mode}_H_b{batch}';persist(key+'_begin')
                m,executable,value=benchmark(ratios,params,states[:batch],spec['timing_repeats'])
                measurements[key]=m;outputs[key]=value
                del executable
                persist(key+'_done')
        persist('validation_begin')
        begin=perf_counter()
        graph=SzHeisenbergGraph(36,tuple(bonds),n_up=18)
        connected,elements=graph.get_conn(tuple(map(int,states[0])))
        for mode in ('raw','a1'):
            values=batched_host_amplitudes(amp_executables[mode],params,np.asarray(connected))
            oracle=np.dot(elements,values)/outputs[f'{mode}_amplitude_b8'][0]
            measured=outputs[f'{mode}_H_b8'][0]
            validation[mode+'_H_oracle_scaled_error']=float(abs(oracle-measured)/(1+abs(oracle)))
            outputs[mode+'_H_oracle']=np.asarray(oracle)
            validation[mode+'_amplitude_batch_replay']=float(abs(outputs[f'{mode}_amplitude_b1'][0]-outputs[f'{mode}_amplitude_b8'][0])/max(abs(outputs[f'{mode}_amplitude_b8'][0]),1e-300))
            validation[mode+'_H_batch_replay_scaled']=float(abs(outputs[f'{mode}_H_b1'][0]-measured)/(1+abs(measured)))
        # Total spin on one frozen nonzero raw amplitude: independent unlike-spin swaps.
        center=states[0];up=np.flatnonzero(center);down=np.flatnonzero(1-center)
        swaps=[]
        for i in up:
            for j in down:
                row=center.copy();row[i],row[j]=row[j],row[i];swaps.append(row)
        values=batched_host_amplitudes(amp_executables['raw'],params,np.asarray(swaps))
        center_value=outputs['raw_amplitude_b8'][0]
        s2=18*center_value+sum(values)
        validation['raw_S2_action_scaled_residual']=float(abs(s2)/(18*abs(center_value)+np.sum(abs(values))))
        # Never use measured near-invariance to delete projector images.
        translations=np.asarray(groups['translation'])
        selected=translations[(translations[:,0]//6%2==0)&(translations[:,0]%6%2==0)]
        transformed=states[:,selected].reshape(-1,36)
        values=batched_host_amplitudes(amp_executables['raw'],params,transformed).reshape(8,len(selected))
        outputs['raw_patch_translation_amplitudes']=values
        raw=outputs['raw_amplitude_b8']
        validation['coarse_translation_relative_rms_max']=float(np.max(np.linalg.norm(values-raw[:,None],axis=0)/np.linalg.norm(raw)))
        images=np.asarray(groups['gamma_a1'])[[0,n_images//3,n_images//2+1,n_images-1]]
        values=batched_host_amplitudes(amp_executables['a1'],params,states[:,images].reshape(-1,36)).reshape(8,4)
        outputs['projected_image_amplitudes']=values
        projected=outputs['a1_amplitude_b8']
        validation['A1_image_relative_rms_max']=float(np.max(np.linalg.norm(values-projected[:,None],axis=0)/np.linalg.norm(projected)))
        validation['seconds']=perf_counter()-begin
        passed=all(m['finite']for m in measurements.values()) and validation['parameter_reload_identical']
        for key,value in validation.items():
            if key.endswith(('scaled_error','batch_replay','replay_scaled','scaled_residual','relative_rms_max')) and not key.startswith('coarse_translation'):
                passed=passed and np.isfinite(value) and value<1e-9
        path=destination/'values.npz';temporary=destination/'values.pending.npz'
        np.savez_compressed(temporary,states=states,**outputs);temporary.replace(path)
        result=dict(geometry=geometry,case=case,registration_sha256=reg['registration_sha256'],
            parameters_sha256=pytree_sha256(params),parameter_count=spin_cell_multiplicity_parameter_count(params),
            images=n_images,bonds=len(bonds),measurements=measurements,validation=validation,passed=bool(passed),
            cg_tables=cg_tables,raw_values_sha256=file_sha256(path),wall_seconds=perf_counter()-started,
            peak_rss_bytes=peak_rss_bytes(),backend=jax.default_backend(),training_updates=0,new_samples=0,
            new_ED=False,remote_submitted=False,weights_transferred_from_N16=False,spatial_quotient_used=False)
        result['result_sha256']=canonical_sha256(result);atomic_json(destination/'results.json',result)
        persist('complete' if passed else 'validation_failed')
        return result
    except Exception as error:
        atomic_json(destination/'failure.json',dict(type=type(error).__name__,message=str(error),
            elapsed_seconds=perf_counter()-started,measurements=measurements,peak_rss_bytes=peak_rss_bytes()))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG);parser.add_argument('--register',action='store_true')
    parser.add_argument('--geometry',choices=('triangular','square'));args=parser.parse_args()
    spec=json.loads(args.config.read_text())
    if args.register:
        path=ROOT/spec['output']/'registration.json'
        if path.exists(): raise FileExistsError(path)
        reg=registration(spec);atomic_json(path,reg);print(reg['registration_sha256'])
    else: run(args.config,args.geometry)


if __name__=='__main__': main()
