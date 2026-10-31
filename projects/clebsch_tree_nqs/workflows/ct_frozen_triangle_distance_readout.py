"""Bounded full-A1 far-triangle readout of an existing CT6 IID prefix.

No training, no new sampling and no H query. The only new expensive work is
complete432-image amplitude completion of the frozen six-site far region.
Every coupling has its own600-second wall envelope and initial four-row
throughput gate. Incomplete queries are never promoted to an RDM estimate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
jax.config.update('jax_enable_x64',True)
import numpy as np

from su2_nqs.benchmarks.cached_region_amplitudes import cached_region_readout,region_completion_rows,reconstruct_region_queries
from su2_nqs.benchmarks.triangle_pair_geometry import triangle_pair_geometry,frozen_six_site_order
from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_projected_amplitude,prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from .ct6_native_direction_calibration import normalized_raw_bank

ROOT=Path(__file__).resolve().parents[3]
CAMPAIGN=ROOT/'projects/clebsch_tree_nqs/results/su2_mechanism_local_20261028'
PARENTS=ROOT/'projects/clebsch_tree_nqs/results/ct_status_20261028/verified_next_parents'
SITES=(12,14,15,18,19,21)


@contextmanager
def deadline(seconds: float):
    """Foreground hard wall guard, plus explicit reserves at chunk boundaries."""
    previous=signal.getsignal(signal.SIGALRM)
    def stop(*_):raise TimeoutError('per-coupling600-second foreground wall envelope')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def frozen_operators() -> tuple[dict,dict]:
    """Geometry-independent scalars only; never transplant supported H or K/M."""
    paths=dict(six=CAMPAIGN/'interventions/frozen_six_body.npz',
        joint=CAMPAIGN/'interventions/frozen_operators.npz',
        reference=CAMPAIGN/'regions/frozen_projectors.npz')
    with np.load(paths['six']) as f:ops={k:f[k] for k in f.files}
    with np.load(paths['joint']) as f:
        for k in f.files:
            if k.startswith('P_') or k in ('vector_chirality_qtotal0','vector_chirality_qtotal2'):
                ops[k]=f[k]
    with np.load(paths['reference']) as f:
        ops['transplanted_local_label_Pstar']=f['compact6_q0_reference_main_projector']
        ops['total_region_q0_projector']=f['compact6_q0_projector']
    return ops,{k:dict(path=str(v.relative_to(ROOT)),sha256=file_sha256(v)) for k,v in paths.items()}


def run(tag: str,output: Path) -> dict:
    if tag not in ('ct6_j0','ct6_j0125'):raise ValueError('one registered frozen parent required')
    if output.exists():raise FileExistsError('new immutable output required')
    if frozen_six_site_order((12,18,19),(14,21,15))!=SITES:raise ArithmeticError('far motif mapping changed')
    geometry=triangle_pair_geometry((12,18,19),(14,21,15))
    if geometry['shortest_center_distance_squared_numerator_over9']!=57:raise ArithmeticError('far geometry mismatch')
    started=perf_counter();output.mkdir(parents=True)
    ops,seals=frozen_operators();step=466 if tag=='ct6_j0' else 492
    sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    atomic_json(output/'registration.json',dict(task='CT_frozen_far_triangle_A1_prefix512_v1',tag=tag,global_step=step,
        sites=SITES,geometry=geometry,records=512,independent_IID_groups=32,wall_cap_seconds=600,
        physical_A=[12,18,19],physical_B=[14,21,15],source_operators=seals,sources=sources,
        actual_new_H_queries=0,actual_new_sampler_records=0,
        transplanted_Pstar_not_original_same_region_spectrum=True,
        cached_rows_not_a_new_untouched_audit=True))
    note=dict(tag=tag,global_step=step,status='started',timings={})
    try:
        with deadline(max(.001,600-(perf_counter()-started))):
            directory=PARENTS/tag;checkpoint=directory/'parent.pkl';products=directory/'parent_snapshot/products.npz'
            audit_path=directory/'parent_snapshot/audit.json'
            saved=load_checkpoint(checkpoint);native=saved['params'];audit=json.loads(audit_path.read_text())
            if audit['endpoint_params_sha256']!=pytree_sha256(native):raise ValueError('frozen audit/native state mismatch')
            config=MixedClebschConfig(6,28,10);arity='ternary_joint' if tag=='ct6_j0' else 'binary_truncated'
            bank,norm=normalized_raw_bank(native,dict(cfg=config,bank=lambda p:arity_clebsch_bank(p,config,arity,8)))
            prepared=prepare_feature_lut_bank(bank,config)
            images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
            if len(images)!=432:raise ArithmeticError('complete lattice group missing')
            scalar=make_feature_lut_projected_amplitude(config,images,chunk=4)
            batch=jax.jit(lambda p,x:jax.lax.map(lambda row:scalar(p,row),x,batch_size=4))
            with np.load(products) as f:
                rows=f['rows'][:512];golden=f['psi'][:512,1];q=f['proposal'][:512];guide=f['guide_components'][:512]
            reconstructed=full_support_sz0_density(guide[:,:,0].mean(1),guide[:,:,1].mean(1),n_sites=36,epsilon=.125)
            if not np.allclose(q,reconstructed,atol=0,rtol=2e-14):raise ValueError('original Q density mismatch')
            tick=perf_counter();probe=np.asarray(batch(prepared,rows[:4]));note['timings']['compile_four_rows']=perf_counter()-tick
            replay=float(np.max(abs(probe-golden[:4]))/max(np.max(abs(golden[:4])),1e-300))
            if replay>2e-11:raise ArithmeticError('saved-state complete A1 replay failed')
            tick=perf_counter();np.asarray(batch(prepared,rows[:4]));warm=(perf_counter()-tick)/4
            queries,lookup=region_completion_rows(rows,SITES)
            note.update(complete_A1_images=432,normalization=norm,four_row_replay_relative_error=replay,
                estimated_warm_seconds_per_amplitude=warm,unique_amplitude_queries=len(queries),
                checkpoint_sha256=file_sha256(checkpoint),source_products_sha256=file_sha256(products),
                source_audit_sha256=file_sha256(audit_path),parameter_sha256=pytree_sha256(native))
            atomic_json(output/'progress.json',note)
            if perf_counter()-started+1.5*warm*len(queries)>585:
                note['status']='cost_gate_before_offdiagonal_queries'
            else:
                tick=perf_counter();pieces=[]
                for first in range(0,len(queries),128):
                    if perf_counter()-started>585:raise TimeoutError('chunk reserve before wall cap')
                    pieces.append(np.asarray(batch(prepared,queries[first:first+128])))
                    if first%1024==0:
                        atomic_json(output/'progress.json',dict(note,completed_queries=min(first+128,len(queries)),
                            elapsed_seconds=perf_counter()-started))
                note['timings']['offdiagonal_region_amplitudes']=perf_counter()-tick
                amplitudes=np.concatenate(pieces);phi=reconstruct_region_queries(amplitudes,lookup)
                packed=rows[:,SITES]@(2**np.arange(5,-1,-1))
                error=float(np.max(abs(phi[np.arange(512),packed]-golden))/max(np.max(abs(golden)),1e-300))
                if error>2e-11:raise ArithmeticError('all512 parent centers must replay')
                report,rho=cached_region_readout(rows,SITES,phi,q,ops,groups=32)
                path=output/(tag+'_physical_region.npz')
                np.savez_compressed(path,rows=rows,sites=np.asarray(SITES),proposal=q,region_amplitudes=phi,
                    rho_Hermitian=rho,query_rows=queries,query_values=amplitudes)
                note.update(status='complete_exploratory_full_A1_far_readout',readout=report,
                    all_parent_amplitude_relative_error=error,physical_region_sha256=file_sha256(path))
    except TimeoutError as error:
        note.update(status='bounded_incomplete_no_physics_estimate',stop_reason=str(error))
    note.update(wall_seconds=perf_counter()-started,actual_new_H_queries=0,actual_new_sampler_records=0,
        geometry=geometry,sites=list(SITES),cached_rows_not_a_new_untouched_audit=True,
        source_operators_unchanged=all(file_sha256(ROOT/v['path'])==v['sha256'] for v in seals.values()),
        source_manifest_unchanged=all(file_sha256(ROOT/name)==digest for name,digest in sources.items()))
    atomic_json(output/(tag+'_results.json'),note)
    return note


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',choices=('ct6_j0','ct6_j0125'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args.tag,args.output)
    print(json.dumps(dict(tag=args.tag,status=report['status'],wall_seconds=report['wall_seconds'],output=str(args.output))),flush=True)
