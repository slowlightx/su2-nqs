"""Read-only external iPEPS adapter; CTM contraction only, no optimization.

Use the existing torch environment and peps-torch checkout. An infinite
one-site spiral state is NOT a finite singlet torus. Save physical compact
RDMs with restored site-dependent rotations and an explicit basis conversion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
from time import perf_counter

import numpy as np
from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout

ROOT=Path(__file__).resolve().parents[3]


def run(state_file: Path, checkout: Path, output: Path, chis: tuple[int,...]=(16,24,32), *, j2: float=.125) -> dict:
    """Contract one unchanged external D5 state at predeclared CTM widths."""
    if not np.isfinite(j2) or j2 < 0:
        raise ValueError('j2 must be a finite nonnegative coupling')
    if not chis or any(isinstance(c,bool) or not isinstance(c,int) or c<2 for c in chis) or len(set(chis))!=len(chis):
        raise ValueError('chis must be unique integers >=2')
    output=Path(output)
    if (output/'results.json').exists():
        raise FileExistsError('completed iPEPS readout is immutable')
    output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(checkout))
    import torch
    import config as cfg
    from ipeps.ipeps import read_ipeps
    from ctm.generic.env import ENV,init_env
    from ctm.generic import ctmrg,rdm
    from models.spin_triangular import J1J2J4_1SITE
    torch.set_num_threads(1)
    cfg.configure(cfg.get_args_parser().parse_args(['--out_prefix',str(output/'ctm')]))
    cfg.global_args.torch_dtype=torch.complex128; cfg.global_args.dtype='complex128'
    cfg.global_args.device='cpu'
    cfg.ctm_args.ctm_max_iter=60; cfg.ctm_args.ctm_conv_tol=1e-8
    cfg.ctm_args.ctm_env_init_type='CTMRG'; cfg.ctm_args.projector_svd_method='DEFAULT'
    state=read_ipeps(str(state_file))
    model=J1J2J4_1SITE(j1=1.,j2=float(j2))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sources=[Path(__file__),ROOT/'su2_nqs/benchmarks/tensor_network_readout.py',
             ROOT/'su2_nqs/benchmarks/regional_su2_readout.py',
             checkout/'ipeps/ipeps.py',checkout/'ctm/generic/rdm.py',
             checkout/'ctm/generic/ctmrg.py',checkout/'ctm/generic/env.py',
             checkout/'models/spin_triangular.py',checkout/'groups/su2.py',checkout/'config.py']
    # The log has a different basename from the state JSON.
    log=Path(str(state_file).replace('_state.json','.out'))
    registration=dict(state=str(state_file),state_sha256=digest(state_file),log_sha256=digest(log),
        external_sources={str(p):digest(p) for p in sources},chis=list(chis),ctm_max_iter=60,
        seconds_cap=600,torch=torch.__version__,numpy=np.__version__,
        j2=float(j2),boundary='infinite iPEPS; NOT finite6x6',pitch=[-1/3,-1/3],
        protocol='No tensor optimization, SU2 twirl or eigenvalue clipping. Library RDM contracts Hermitian part and trace-normalizes; sym_pos_def=False.',
        geometry='Native peps-torch (x,y) -> repository (row=-y,column=x); native spin up/down -> down/up.')
    (output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    start=perf_counter(); records=[]; arrays={}
    old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded CTM budget')))
    signal.alarm(600)
    try:
        with torch.no_grad():
            for chi in chis:
                env=ENV(chi,state); init_env(state,env)
                def convergence(_state,_env,history,ctm_args):
                    vector=torch.cat([torch.linalg.svdvals(c)/torch.linalg.svdvals(c)[0] for c in _env.C.values()]).cpu().numpy()
                    if history is None: history=dict(previous=vector,deltas=[],iterations=0)
                    delta=float(np.linalg.norm(vector-history['previous']))
                    history['iterations']+=1; history['deltas'].append(delta); history['previous']=vector
                    return history['iterations']>=8 and delta<1e-8,history
                env,history,ctm_seconds,_=ctmrg.run(state,env,conv_check=convergence)
                raw=rdm.rdm2x2((0,0),state,env,unroll=True,sym_pos_def=False).reshape(16,16).cpu().numpy()
                # U(x,y)=R**(x-y). Native RDM order is (0,0),(1,0),(0,1),(1,1).
                r=model.R.cpu().numpy(); unitary=np.eye(1,dtype=complex)
                for exponent in (0,1,-1,0): unitary=np.kron(unitary,np.linalg.matrix_power(r,exponent))
                physical=unitary@raw@unitary.conj().T
                physical=physical[::-1,::-1] # all physical sites: up/down -> down/up
                info=physical_spin_block_readout(physical)
                energy=float(model.energy_per_site(state,env,unroll=True).real)
                one=rdm.rdm1x1((0,0),state,env,sym_pos_def=False).cpu().numpy()[::-1,::-1]
                spins=[np.array([[0,.5],[.5,0]]),np.array([[0,.5j],[-.5j,0]]),np.diag([-.5,.5])]
                moment=[float(np.trace(one@s).real) for s in spins]
                records.append(dict(chi=chi,ctm_iterations=history['iterations'],
                    last_corner_spectrum_change=history['deltas'][-1],ctm_seconds=ctm_seconds,
                    energy_per_site=energy,local_moment=moment,local_moment_norm=float(np.linalg.norm(moment)),
                    compact4=info,regional_unitary_error=float(np.linalg.norm(unitary.conj().T@unitary-np.eye(16)))))
                arrays[f'chi{chi}_native_compact4']=raw; arrays[f'chi{chi}_physical_compact4']=physical
                torch.save(dict(C=env.C,T=env.T,chi=chi,state_sha256=digest(state_file)),output/f'env_chi{chi}.pt')
                print('ipeps chi',chi,'E',energy,'m',np.linalg.norm(moment),'SU2 defect',info['su2_factorization_frobenius'],flush=True)
        with (output/'rdms.npz').open('wb') as handle: np.savez_compressed(handle,**arrays)
        result=dict(status='external_iPEPS_readout_complete',registration=registration,records=records,
            seconds=perf_counter()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,
            rdms_sha256=digest(output/'rdms.npz'),new_ED=0,new_optimization=0,phase_claim=False,
            disposition='Infinite symmetry-broken ordered-candidate calibration, not a finite singlet competitive pair. No multiplicity entanglement interpretation unless invariant.')
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True); p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--j2',type=float,default=.125)
    p.add_argument('--chis',type=int,nargs='+',default=[16,24,32])
    args=p.parse_args(); run(args.state,args.checkout,args.output,tuple(args.chis),j2=args.j2)
