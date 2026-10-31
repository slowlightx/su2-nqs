"""Read-only fetch and independent NumPy check of completed CT12/B20032."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256,canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'projects/clebsch_tree_nqs/results/ct_sector_physical_seed_20261029/ct12_completed'
REMOTE='/home/yx789/nqs-wavefunctions/nqs/ct12_production32_20261028/j0125/run01'
NAMES=('checkpoint.pkl','registration.json','endpoint_frozen.json','audit_result.json',
    'train_result.json','terminal_audit/audit.json','terminal_audit/untouched_products.npz')


def run()->dict:
    """Only completed frozen products; no new physical queries or remote writes."""
    if DEST.exists():raise FileExistsError('new immutable mirror required')
    code=f'import hashlib,json;from pathlib import Path;p=Path({REMOTE!r});names={NAMES!r};print(json.dumps({{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names}}))'
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','unicorn','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=60)
    manifest=json.loads(r.stdout);DEST.mkdir()
    atomic_json(DEST/'remote_manifest.json',manifest)
    subprocess.run(['rsync','-aR',*[f'unicorn:{REMOTE}/./{n}' for n in NAMES],str(DEST)+'/'],check=True,timeout=180)
    assert all(file_sha256(DEST/n)==s for n,s in manifest.items())
    saved=load_checkpoint(DEST/'checkpoint.pkl');meta=saved['metadata']
    audit=json.loads((DEST/'terminal_audit/audit.json').read_text());frozen=json.loads((DEST/'endpoint_frozen.json').read_text())
    result=json.loads((DEST/'audit_result.json').read_text())
    digest=pytree_sha256(saved['params'])
    assert digest==audit['endpoint_params_sha256']==frozen['parameter_sha256']==result['parameter_sha256']
    assert file_sha256(DEST/'checkpoint.pkl')==frozen['checkpoint_sha256']==result['checkpoint_sha256']
    assert pytree_sha256(saved['optimizer_state'])==meta['optimizer_sha256']
    assert canonical_sha256(meta['rng'])==meta['rng_sha256']
    with np.load(DEST/'terminal_audit/untouched_products.npz') as raw:
        rows,products=raw['rows'],raw['products']
    assert rows.shape==(4096,144) and np.all(rows.sum(1)==72)
    psi,hp=products[:,:,0],products[:,:,1]
    Q=full_support_sz0_density(products[:,:,2].real.mean(1),products[:,:,3].real.mean(1),n_sites=144,epsilon=.125)
    def estimate(ids):
        z=(abs(psi[ids])**2/Q[ids,None]).mean(0)
        e=(psi[ids].conj()*hp[ids]/Q[ids,None]).mean(0)/z
        v=(abs(hp[ids]-e*psi[ids])**2/Q[ids,None]).mean(0)/z
        return np.stack((e.real/144,e.imag/144,v/144,z),axis=1)
    ids=np.arange(len(rows));labels=np.repeat(np.arange(64),64);point=estimate(ids)
    leave=np.asarray([estimate(ids[labels!=g]) for g in range(64)])
    error=np.sqrt(63/64*np.sum((leave-leave.mean(0))**2,axis=0))
    dl=leave[:,1]-leave[:,0];ds=np.sqrt(63/64*np.sum((dl-dl.mean(0))**2,axis=0))
    diffs=[np.max(abs(point-np.asarray(audit['moments']['endpoint_values']))),
        np.max(abs(error-np.asarray(audit['moments']['endpoint_independent_block_jackknife_SE']))),
        np.max(abs(ds-np.asarray(audit['moments']['paired_independent_block_jackknife_SE'])))]
    assert max(diffs)<2e-10
    report=dict(status='completed_CT12_independently_recomputed',job_id=735301,
        completed_updates=32,absolute_step=37,energy_per_site=float(point[1,0]),
        energy_SE=float(error[1,0]),variance_per_site=float(point[1,2]),
        paired_delta=(point[1]-point[0]).tolist(),paired_SE=ds.tolist(),
        metric_order=['energy_per_site','imaginary_energy_per_site','variance_per_site','norm'],
        maximum_numeric_difference=float(max(diffs)),all_remote_SHA_verified=True,
        checkpoint_optimizer_RNG_verified=True,parameter_sha256=digest,
        records=4096,independent_groups=64,still_developmental_not_selected_ground_benchmark=True)
    atomic_json(DEST/'independent_review.json',report)
    return report


if __name__=='__main__':print(json.dumps(run(),indent=2))
