"""Confirm paired held-out fidelity change; no new H or training queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_target_root import collective_spin_root_coefficients
from su2_nqs.models.triangular_sector_amplitude import make_sector_prepare,pack_sector_parameters,make_prepared_sector_amplitude
from .ct_physical_spin_seed_20261029 import ROOT,PARENTS,fidelity


def run(directory:Path)->dict:
    """The selected endpoint was frozen BEFORE these128 rows were observed."""
    report=json.loads((directory/'results.json').read_text())
    frozen=json.loads((directory/'endpoint_frozen.json').read_text())
    if report['status']!='complete_physical_seed_and_heldout_full_W_initialization':raise ValueError('complete source experiment required')
    path=directory/'initialized.pkl'
    if file_sha256(path)!=frozen['checkpoint_sha256']:raise ValueError('initialization checkpoint changed')
    saved=load_checkpoint(path);final=saved['params']
    if pytree_sha256(final)!=frozen['parameter_sha256']:raise ValueError('frozen parameter identity changed')
    tag='ct6_j0' if report['j2']==0 else 'ct6_j0125';arity='ternary_joint' if report['j2']==0 else 'binary_truncated'
    parent=load_checkpoint(PARENTS/tag/'checkpoint.pkl')['params'];cfg=MixedClebschConfig(6,28,10)
    if pytree_sha256(parent)!=report['parent_parameter_sha256']:raise ValueError('audited singlet parent changed')
    initial=pack_sector_parameters(parent,tuple(jnp.asarray(x) for x in collective_spin_root_coefficients(arity_clebsch_bank(parent,cfg,arity,8),cfg,2)))
    prepare=jax.jit(make_sector_prepare(cfg,arity=arity,binary_rank=8))
    amp=make_prepared_sector_amplitude(cfg,2,0,((1,3),(1,3)),chunk=4)
    query=jax.jit(lambda p,rows:jax.lax.map(lambda x:amp(p,x),rows))
    with np.load(directory/'audit_source_products.npz') as raw:
        rows,Q,target=raw['rows'],raw['proposal'],raw['psi'][:,1]
    phi=np.column_stack([np.asarray(query(prepare(p),jnp.asarray(rows))) for p in (initial,final)])
    groups=np.repeat(np.arange(16),8)
    def estimates(ids):return np.asarray([fidelity(phi[ids,k],target[ids],Q[ids]) for k in (0,1)])
    point=estimates(np.arange(len(rows)));leave=np.asarray([estimates(np.flatnonzero(groups!=g)) for g in range(16)])
    delta=leave[:,1]-leave[:,0];se=np.sqrt(15/16*np.sum((delta-delta.mean())**2))
    np.savez_compressed(directory/'heldout_CT_amplitudes.npz',rows=rows,proposal=Q,target=target,psi=phi)
    result=dict(task='paired_heldout_physical_seed_fidelity_change',j2=report['j2'],
        initial_F=float(point[0]),final_F=float(point[1]),delta_F=float(point[1]-point[0]),
        paired_independent_group_jackknife_SE=float(se),records=128,groups=16,
        new_amplitude_centers=256,new_H_centers=0,new_optimization_steps=0,
        fixed_endpoint_initialized_checkpoint_sha256=frozen['checkpoint_sha256'],
        no_excitation_gap_or_initialized_CT_energy_claim=True)
    atomic_json(directory/'paired_fidelity_confirmation.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();print(json.dumps(run(args.directory),indent=2))
