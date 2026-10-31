"""Read frozen physical-source products without new amplitude/H queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256


def statistics(rows:np.ndarray,phi:np.ndarray,hp:np.ndarray,Q:np.ndarray)->np.ndarray:
    """Total-energy centroid/width and exact carrier-residual decomposition."""
    norm=np.sum(abs(phi)**2/Q[:,None],axis=0)
    energy=np.sum(phi.conj()*hp/Q[:,None],axis=0)/norm
    coords=np.asarray([divmod(i,6) for i in range(36)])
    momenta=np.asarray([[1/3,1/3],[.5,0.],[0.,.5],[.5,.5]])
    diagonal=(rows-.5)@np.exp(2j*np.pi*momenta@coords.T).T/6.
    parent_residual=diagonal*(hp[:,0]-energy[0]*phi[:,0])[:,None]
    commutator=hp[:,1:]-diagonal*hp[:,0,None]
    a=np.sum(abs(commutator)**2/Q[:,None],axis=0)/norm[1:]
    b=np.sum(abs(parent_residual)**2/Q[:,None],axis=0)/norm[1:]
    cross=np.sum(2*(commutator.conj()*parent_residual).real/Q[:,None],axis=0)/norm[1:]
    shift=energy[1:]-energy[0]
    variance=np.sum(abs(hp[:,1:]-energy[1:]*phi[:,1:])**2/Q[:,None],axis=0)/norm[1:]
    closure=a+b+cross-abs(shift)**2
    if not np.allclose(variance,closure,rtol=2e-10,atol=2e-12):
        raise ArithmeticError('commutator plus imperfect-carrier residual failed')
    values=np.column_stack((3*norm[1:]/norm[0],energy[1:].real/36.,
        shift.real,variance,np.sqrt(variance),a,b,cross))
    # Three orthogonal M momenta: a norm-weighted direct-sum centroid, never
    # selecting the lowest noisy estimate or counting three independent gaps.
    weights=norm[2:]/norm[2:].sum();em=weights@energy[2:]
    vm=np.sum(abs(hp[:,2:]-em*phi[:,2:])**2/Q[:,None])/norm[2:].sum()
    mstar=np.asarray([norm[2:].sum()/norm[0],em.real/36.,(em-energy[0]).real,
        vm,np.sqrt(vm),weights@a[1:],weights@b[1:],weights@cross[1:]])
    return np.vstack((values,mstar))


def run(directory:Path)->dict:
    """Propagate the same independent groups through every derived quantity."""
    receipt=json.loads((directory/'results.json').read_text())
    if receipt['status']!='complete_physical_seed_and_heldout_full_W_initialization':
        raise ValueError('complete frozen initialization and terminal products required')
    p=directory/'audit_source_products.npz'
    with np.load(p) as raw:
        x,phi,hp,Q=[raw[k] for k in ('rows','psi','hpsi','proposal')]
    groups=np.repeat(np.arange(16),len(x)//16)
    value=statistics(x,phi,hp,Q)
    replicas=np.asarray([statistics(x[groups!=g],phi[groups!=g],hp[groups!=g],Q[groups!=g]) for g in range(16)])
    error=np.sqrt(15/16*np.sum((replicas-replicas.mean(0))**2,axis=0))
    report=dict(task='strict_triplet_source_centroid_not_excitation_gap',j2=receipt['j2'],
        parent_step=receipt['parent_global_step'],source_file_sha256=file_sha256(p),
        metric_order=['connected_scalar_Sk','response_trial_energy_per_site',
            'centroid_minus_approximate_parent_total_energy','trial_total_energy_variance',
            'trial_total_energy_width','commutator_second_about_parent',
            'carrier_residual_second','commutator_carrier_cross'],
        channels={name:dict(values=v.tolist(),independent_group_jackknife_SE=e.tolist())
            for name,v,e in zip(('K','M1','M2','M3','M_star_direct_sum'),value,error,strict=True)},
        normalized_ratios_have_finite_sample_bias=True,independent_groups=16,records=len(x),
        total_spin_q=2,total_projection_m2=0,
        strict_spin_from_rank_one_on_exact_singlet=True,
        connected_subtracts_one_point_products_onsite_self_correlation_retained=True,
        lower_gap_or_spinon_identification=False,carrier_variance_not_dropped=True,
        supervised_initialized_CT_overlap=receipt['initial_and_final_F_with_physical_K_seed'],
        new_amplitude_or_H_queries=0)
    atomic_json(directory/'physical_summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();print(json.dumps(run(args.directory),indent=2))
