"""Independent operator/moment replay of fresh sealed confirmation archives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from fnqs.artifacts import load_checkpoint,write_json
from fnqs.pilots.lineage import sha,read_node
from fnqs.variants.size_features import SizeFeatureConfig,log_psi
from fnqs.scripts.run_frozen_confirmation import verified
from fnqs.scripts.review_l16_repair import sparse_tfim


def independent_local_energy(params: dict,bits: np.ndarray,g: np.ndarray,cfg: SizeFeatureConfig) -> np.ndarray:
    """Enumerate one-spin flips independently, restricted to this TFIM review."""
    if not np.array_equal(g[:3],[1.,0.,0.]):raise ValueError('TFIM only')
    b=np.asarray(bits,dtype=np.int32);n=b.shape[1];flips=[]
    for site in range(n):
        changed=b.copy();changed[:,site]=1-changed[:,site];flips.append(changed)
    logs=np.asarray(log_psi(params,jnp.asarray(b),jnp.asarray(g),cfg))
    flip_logs=np.asarray(log_psi(params,jnp.asarray(np.concatenate(flips)),jnp.asarray(g),cfg)).reshape(n,len(b))
    diagonal=np.zeros(len(b))
    for site in range(n):diagonal-=np.where(b[:,site]==b[:,(site+1)%n],1.,-1.)
    return diagonal-g[3]*np.exp(flip_logs-logs[None,:]).sum(axis=0)


def review(output: Path) -> dict:
    """Check every label/hash/moment and independent H without new Born draws."""
    p=verified(output);cfg=SizeFeatureConfig(**p['model']);seal=json.loads((output/'endpoint_seal.json').read_text())['nodes']
    diagonal,v=sparse_tfim(16);action=0.;moment=0.;corr_error=0.;norm_error=0.;fd_error=0.;cov_error=0.;count=0;bindings={}
    costs=dict(born_samples=0,hamiltonian_actions=0,source_actions=0,exhaustive_basis_rows=0,local_energy_rows=0)
    for seed in p['seeds']:
        for n in p['sizes']:
            folder=output/'runs'/f'L{n}_seed_{seed}';node=folder/'fast/node.json';graph=read_node(node);meta=graph[node.resolve()]
            if seal[folder.name]!=sha(node):raise ValueError('sealed confirmation endpoint changed')
            params,_,_,_=load_checkpoint(folder/'fast/checkpoint.npz')
            data=json.loads((folder/'audit.json').read_text())
            points=p['reserved_points']+p['calibration_points']
            if len(data['points'])!=len(points):raise ValueError('confirmation point count changed')
            for i,(row,g) in enumerate(zip(data['points'],points)):
                path=folder/'audit'/f'point_{i}.npz'
                split='reserved_confirmation' if i<len(p['reserved_points']) else 'training_field_calibration'
                if row['n']!=n or row['couplings']!=g or row['split']!=split or row['archive_sha256']!=sha(path):raise ValueError('confirmation row/binding mismatch')
                if row['checkpoint_sha256']!=meta['checkpoint_sha256'] or row['protocol_sha256']!=sha(output/'protocol.json'):raise ValueError('checkpoint/protocol audit binding')
                with np.load(path,allow_pickle=False) as z:
                    bits=z['bits'];s=z['score'];e=z['energy'];fd=z['fd_scores'];x=z['couplings']
                    if not np.array_equal(x,g) or bits.shape[1]!=n or not np.all((bits==0)|(bits==1)):raise ValueError('raw confirmation labels')
                    if n==16:
                        psi=z['psi'];hs=z['h_score'];source=z['source'];w=psi**2;norm=psi@psi;w=w/norm
                        labels=np.arange(1<<n);expected=(labels[:,None]>>np.arange(n-1,-1,-1))&1
                        if not np.array_equal(bits,expected):raise ValueError('MSB exhaustive convention')
                        vp=v@psi;hp=diagonal*psi+g[3]*vp;hsp=diagonal*(psi*s)+g[3]*(v@(psi*s))
                        action=max(action,float(np.max(abs(hp-psi*e))),float(np.max(abs(hsp-psi*hs))),float(np.max(abs(vp-psi*source))))
                        energy=float(psi@hp/norm);mu=float(w@s);tau=psi*(s-mu)
                        defect=hsp-mu*hp-energy*tau+vp;defect-=psi*(psi@defect/norm)
                        rho=np.linalg.norm(defect)/np.linalg.norm(vp-psi*(psi@vp/norm))
                        moment=max(moment,abs(rho-row['response_relative_residual']));norm_error=max(norm_error,abs(norm-1))
                        costs['exhaustive_basis_rows']+=len(bits);costs['hamiltonian_actions']+=2;costs['source_actions']+=1
                    else:
                        if len(bits)!=p['audit']['born_samples'] or len(e)!=p['audit']['energy_rows'] or len(fd)!=p['audit']['fd_rows']:raise ValueError('MC prefix budget')
                        w=np.full(len(bits),1/len(bits));energy=float(e.mean())
                        manual=independent_local_energy(params,bits[:32],x,cfg)
                        action=max(action,float(np.max(abs(manual-e[:32]))));costs['hamiltonian_actions']+=1;costs['local_energy_rows']+=32
                        influences=np.column_stack(((s-s.mean())**2,np.stack([np.mean((2*bits.astype(float)-1)*(2*np.roll(bits,-r,axis=1).astype(float)-1),axis=1) for r in range(n)],axis=1)))
                        centered=influences-influences.mean(axis=0);cov=centered.T@centered/(len(s)*(len(s)-1))
                        cov_error=max(cov_error,float(np.max(abs(cov-z['joint_covariance']))))
                        boot=float(z['chi_bootstrap_replicates'].std(ddof=1));se=max(boot,float(np.sqrt(max(cov[0,0],0))))
                        moment=max(moment,abs(se-row['chi_se']),abs(e.std(ddof=1)/np.sqrt(len(e))-row['energy_se']))
                    chi=float(w@(s-w@s)**2);moment=max(moment,abs(chi-row['chi_total']),abs(energy-row['energy_total']))
                    zz=2*bits.astype(float)-1
                    corr=np.asarray([w@np.mean(zz*np.roll(zz,-r,axis=1),axis=1) for r in range(n)])
                    corr_error=max(corr_error,float(np.max(abs(corr-row['correlations']))))
                    ss=s[:len(fd)];fw=w[:len(fd)];fw=fw/fw.sum();d=fd-fw@fd-(ss-fw@ss)[:,None]
                    fd_error=max(fd_error,float(np.max(np.sqrt(fw@(d*d)/(fw@(ss-fw@ss)**2)))))
                count+=1;bindings[str(path)]=sha(path)
            print('replay',folder.name,flush=True)
    if action>2e-11 or moment>2e-11 or corr_error>1e-12 or norm_error>1e-10 or fd_error>5e-4 or cov_error>1e-10:
        raise RuntimeError('independent fresh-confirmation replay failed')
    result=dict(complete=True,points=count,operator_max_discrepancy=action,moment_max_discrepancy=moment,
        full_correlation_max_discrepancy=corr_error,normalization_max_error=norm_error,fd_relative_max=fd_error,
        joint_covariance_max_discrepancy=cov_error,tolerances=dict(operator=2e-11,moment=2e-11,correlation=1e-12,
        normalization=1e-10,fd_relative=5e-4,joint_covariance=1e-10),cost=costs,
        validated_sha256=bindings,protocol_sha256=sha(output/'protocol.json'),review_source_sha256=sha(Path(__file__)))
    write_json(output/'independent_replay.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    review(parser.parse_args().output)
