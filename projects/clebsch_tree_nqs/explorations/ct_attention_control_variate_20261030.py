"""Retrospective actual-CT6 expensive-H control-variate gate, not training.

Six older complete-A1 parent caches supply 32 H labels each. Two later parent
caches are never fitted. Cheap physical features predict scalar E_local;
an explicit probability-sampling residual correction preserves the finite
panel's force, with whole chains as sampling units. No new H query is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT/'projects/clebsch_tree_nqs/results/ct_projected_residual_capacity_20261022/v2'
TRAIN = ('j0_w16','j0_w18','j0_w20','j0125_w16','j0125_w18','j0125_w20')
TARGET = ('j0','j0125')


def cheap_features(data: dict, j2: float) -> tuple[np.ndarray, np.ndarray]:
    """Physical block scalars/bits only: no H label or virtual gauge coordinate."""
    bits=np.asarray(data['bits']); spins=2*bits-1
    blocks=triangular_schur_hierarchy(6).blocks
    actual=triangular_j1j2_bonds(6,6,j2=j2,periodic=True)
    strength={(i,j):v for i,j,v in actual}
    bonds=triangular_j1j2_bonds(6,6,j2=1.,periodic=True)
    pairs=np.asarray([(i,j) for i,j,_ in bonds]); coupling=np.asarray([strength.get((i,j),0.) for i,j,_ in bonds])
    diagonal=(spins[:,pairs[:,0]]*spins[:,pairs[:,1]])*(coupling/4)[None,:]
    tokens=[]
    for b,block in enumerate(blocks):
        sites=np.asarray(block.sites)
        local=spins[:,sites]
        quadratic=np.stack([local[:,i]*local[:,j] for i in range(9) for j in range(i+1,9)],axis=1)
        internal=np.isin(pairs[:,0],sites)&np.isin(pairs[:,1],sites)
        incident=np.isin(pairs[:,0],sites)^np.isin(pairs[:,1],sites)
        state=data['token_features'][6*b+np.asarray([0,2,4])][:,[1,2,3,4,5]].ravel()
        state=np.broadcast_to(np.r_[state,j2],(len(bits),len(state)+1))
        tokens.append(np.column_stack((local,quadratic,diagonal[:,internal].sum(1),
                                       diagonal[:,incident].sum(1),state)))
    return np.stack(tokens,axis=1),diagonal


def initialize(seed: int, dim: int, width: int = 8) -> dict:
    """Identical, fully active parameter trees for attention and nonlinear control."""
    keys=iter(jax.random.split(jax.random.PRNGKey(seed),8))
    def w(a,b): return jax.random.normal(next(keys),(a,b),dtype=jnp.float64)/np.sqrt(a)
    return dict(embedding=w(dim,width),query=w(width,width),key=w(width,width),value=w(width,width),
        output=w(width,width),local_in=w(width,2*width),local_out=w(2*width,width),
        readout=jnp.zeros(width),bias=jnp.zeros(()))


def predict(p: dict, x: jax.Array, mode: str) -> jax.Array:
    """Scalar energy surrogate; magnetic structure of the CT state is untouched."""
    h=jnp.tanh(x@p['embedding']); h=h+jnp.tanh(h@p['local_in'])@p['local_out']
    q,k,v=(h@p[a] for a in ('query','key','value'))
    if mode == 'attention':
        mix=jax.nn.softmax(q@jnp.swapaxes(k,-1,-2)/np.sqrt(h.shape[-1]),axis=-1)@v
    else:
        # All-token context reaches every nonattention token as well.
        mix=jax.nn.sigmoid((q+h.mean(-2,keepdims=True))*k/np.sqrt(h.shape[-1]))*v
    h=h+jnp.tanh(mix@p['output'])
    return h.mean(-2)@p['readout']+p['bias']


def train(x: np.ndarray, y: np.ndarray, mode: str, seed: int,
          steps: int) -> tuple[dict, dict]:
    """Fixed small teacher budget and final endpoint; no target labels accepted."""
    start=perf_counter(); p=initialize(seed,x.shape[-1]); m=jax.tree.map(jnp.zeros_like,p); v=m
    def objective(p):
        return jnp.mean((predict(p,jnp.asarray(x),mode)-y)**2)+1e-3*sum(jnp.mean(z*z) for z in jax.tree.leaves(p))
    @jax.jit
    def step(p,m,v,k):
        l,g=jax.value_and_grad(objective)(p)
        m=jax.tree.map(lambda a,b:.9*a+.1*b,m,g); v=jax.tree.map(lambda a,b:.999*a+.001*b*b,v,g)
        p=jax.tree.map(lambda a,b,c:a-.005*b/(1-.9**k)/(jnp.sqrt(c/(1-.999**k))+1e-8),p,m,v)
        return p,m,v,l
    for k in range(1,steps+1): p,m,v,l=step(p,m,v,jnp.asarray(k))
    jax.block_until_ready(p)
    return p,dict(mode=mode,seed=seed,steps=steps,parameters=sum(z.size for z in jax.tree.leaves(p)),
                  training_seconds=perf_counter()-start,training_MSE=float(l))


def corrected_force(centered_scores: np.ndarray, energy: np.ndarray,
                    prediction: np.ndarray, groups: np.ndarray,
                    reference: int, correction: np.ndarray) -> np.ndarray:
    """Unbiased finite-population force conditional on a known reference chain.

    One uniformly chosen chain supplies a reference E0 and exact contribution;
    k other chains are sampled uniformly without replacement. The omitted
    population has known size. The prediction uses no population H labels.
    This statement is about the cached finite panel, not physical Born tails.
    """
    o,e,f,g=map(np.asarray,(centered_scores,energy,prediction,groups))
    ids=np.unique(g); other=ids[ids!=reference]
    if (o.ndim!=2 or e.shape!=(len(o),) or f.shape!=e.shape or g.shape!=e.shape
            or reference not in ids or len(correction)<1 or len(set(correction))!=len(correction)
            or np.any(~np.isin(correction,other))):
        raise ValueError('aligned panel, one reference, distinct nonreference correction chains required')
    counts=np.array([np.sum(g==a) for a in ids])
    if not np.all(counts==counts[0]): raise ValueError('equal record counts per chain required')
    e0=e[g==reference].mean()
    residual=o*(e-e0-f)[:,None]
    reference_value=residual[g==reference].mean(0)
    correction_value=np.mean([residual[g==a].mean(0) for a in correction],axis=0)
    return (o*f[:,None]).mean(0)+(reference_value+(len(ids)-1)*correction_value)/len(ids)


def natural_direction(gram: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Common full 24-dimensional physical solve, dimensionless ridge=.05."""
    d=np.sqrt(np.maximum(np.diag(gram),1e-20)); normalized=gram/d[:,None]/d[None,:]
    result=-np.linalg.solve(normalized+.05*np.eye(len(d)),force/d)/d
    return result/(np.sqrt(max(float(result@gram@result),1e-30)))


def score_target(data: dict, predictions: dict[str,np.ndarray], seed: int = 26103031) -> dict:
    """Final retrospective benchmark, after all predictions are frozen."""
    fit=data['fit_mask']; scores=data['scores'][fit]; energy=data['local_energies'][fit]
    groups=data['chain_labels'][fit]; o=scores-scores.mean(0)
    gram=o.T@o/len(o); true_force=(o*energy[:,None]).mean(0)
    ids=np.unique(groups); per_chain=np.sum(groups==ids[0]); rng=np.random.default_rng(seed)
    draws=[rng.choice(ids,size=8,replace=False) for _ in range(256)]
    audit=data['scores'][~fit]; ae=data['local_energies'][~fit]
    ao=audit-audit.mean(0); ag=ao.T@ao/len(ao); af=(ao*ae[:,None]).mean(0)
    inv=np.linalg.pinv(gram+.05*np.diag(np.diag(gram)),rcond=1e-12)
    baseline_errors=None; results={}
    for name,prediction in predictions.items():
        fs=[]; slopes=[]; error=[]; directions=[]
        pred=np.asarray(prediction)[fit]
        for draw in draws:
            estimated=corrected_force(o,energy,pred,groups,int(draw[0]),draw[1:])
            delta=estimated-true_force; error.append(float(delta@inv@delta))
            direction=natural_direction(gram,estimated); directions.append(direction)
            fs.append(float(np.sqrt(max(direction@ag@direction,0.))))
            slopes.append(float(2*af@direction/max(fs[-1],1e-30)))
        errors=np.asarray(error)
        if name=='no_surrogate': baseline_errors=errors
        results[name]=dict(mean_whitened_force_squared_error=float(errors.mean()),
            median_whitened_force_squared_error=float(np.median(errors)),
            empirical_force_query_variance_gain=float(baseline_errors.mean()/errors.mean()),
            paired_error_difference_randomization_SE=float(np.std(baseline_errors-errors,ddof=1)/np.sqrt(len(errors))),
            audit_mean_tangent_energy_slope_per_FS=float(np.mean(slopes)),
            audit_descent_fraction=float(np.mean(np.asarray(slopes)<0)),
            audit_mean_FS_over_fit_FS=float(np.mean(fs)),
            per_resample_error=errors.tolist(),per_resample_audit_slope_per_FS=slopes)
    return dict(fit_rows=int(fit.sum()),fit_chains=len(ids),audit_rows=int((~fit).sum()),
        audit_chains=len(np.unique(data['chain_labels'][~fit])),
        target_H_queries_per_repeat=8*int(per_chain),reference_chains=1,correction_chains=7,
        correction_is_finite_panel_probability_sample_not_new_MC=True,
        full_fit_reference_force=true_force.tolist(),models=results)


def run(output: Path, steps: int = 300) -> dict:
    """Freeze feature/label access, fit all arms, then inspect target H caches."""
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('use an empty immutable result directory')
    start=perf_counter(); registration=dict(task='actual_CT6_expensive_H_control_variate_retrospective_gate',
        teacher_parents=TRAIN,whole_parent_targets=TARGET,teacher_records_per_parent=32,
        teacher_seed=26103011,training_seeds=(26103021,26103022),steps=steps,
        target_reference_chain_count=1,target_correction_chain_count=7,query_randomizations=256,
        correction_seed=26103031,same_full_24_direction_Gram_and_natural_solve=True,
        masks='source teacher labels only from old fit chains; target H only in final benchmark/counted residual queries',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='old actual CT6 complete-A1 Born-chain panels, not CT12, not full-W or finite energy training',
        new_full_H_calls=0,new_amplitude_calls=0,wall_cap_seconds=300)
    atomic_json(output/'registration.json',registration)
    arrays={}; provenance={}; features={}; diagonal={}; teacher_indices={}; source_x=[]; source_y=[]; source_diag=[]
    rng=np.random.default_rng(registration['teacher_seed'])
    for name in TRAIN+TARGET:
        directory=CACHE/name
        with np.load(directory/'dataset.npz') as z: arrays[name]={k:z[k] for k in z.files}
        old=json.loads((directory/'registration.json').read_text()); result=json.loads((directory/'results.json').read_text())
        provenance[name]=dict(dataset_sha256=hashlib.sha256((directory/'dataset.npz').read_bytes()).hexdigest(),
            parent_parameter_sha256=old['parent_params_sha256'],parent_checkpoint_sha256=old['parent_checkpoint_sha256'],
            original_audit_sha256=old['parent_audit_sha256'],rows=len(arrays[name]['bits']),
            original_derivative_cache_seconds=result['wall_seconds'],j2=old['j2'])
        features[name],diagonal[name]=cheap_features(arrays[name],old['j2'])
        if name in TRAIN:
            data=arrays[name]; groups=np.unique(data['chain_labels'][data['fit_mask']]); per=np.sum(data['chain_labels']==groups[0])
            selected=rng.choice(groups,size=32//per,replace=False); ids=np.flatnonzero(np.isin(data['chain_labels'],selected))
            y=data['local_energies'][ids]; y=y-y.mean()
            source_x.append(features[name][ids]); source_y.append(y); source_diag.append(diagonal[name][ids]); teacher_indices[name]=ids
    x=np.concatenate(source_x); y=np.concatenate(source_y); dx=np.concatenate(source_diag)
    mean=x.mean((0,1)); scale=x.std((0,1))+1e-6; x=(x-mean)/scale
    for name in features: features[name]=(features[name]-mean)/scale
    dm=dx.mean(0); ds=dx.std(0)+1e-6; design=np.column_stack((np.ones(len(dx)),(dx-dm)/ds))
    # The J0 and J.125 pair lists have different lengths. Pad J0-only with zero J2 slots.
    coefficient=np.linalg.solve(design.T@design+10*np.eye(design.shape[1]),design.T@y)
    predictions={name:dict(no_surrogate=np.zeros(len(features[name]))) for name in TARGET}; fits=[]
    for name in TARGET:
        d=diagonal[name]; z=np.column_stack((np.ones(len(d)),(d-dm)/ds))
        predictions[name]['linear_bond_rule']=z@coefficient
        predictions[name]['direct_H_diagonal']=d.sum(1)-d.sum(1).mean()
    for mode in ('pointwise','attention'):
        for seed in (26103021,26103022):
            p,info=train(x,y,mode,seed,steps); fits.append(info)
            if perf_counter()-start>300: raise TimeoutError('fixed 300-second wall cap exceeded')
            np.savez_compressed(output/f'{mode}_{seed}_parameters.npz',**{k:np.asarray(v) for k,v in p.items()})
            for name in TARGET:
                predictions[name][f'{mode}_{seed}']=np.asarray(predict(p,jnp.asarray(features[name]),mode))
    shuffled=y[np.random.default_rng(26103012).permutation(len(y))]
    p,info=train(x,shuffled,'attention',26103021,steps); info['label_control']='random_teacher_permutation'; fits.append(info)
    for name in TARGET: predictions[name]['attention_random_teacher']=np.asarray(predict(p,jnp.asarray(features[name]),'attention'))
    # Freeze all candidate outputs before the final benchmark touches target H labels.
    np.savez_compressed(output/'frozen_predictions.npz',**{f'{name}__{mode}':v for name,arms in predictions.items() for mode,v in arms.items()})
    atomic_json(output/'prediction_freeze.json',dict(source_sha256=registration['source_sha256'],
        prediction_sha256=hashlib.sha256((output/'frozen_predictions.npz').read_bytes()).hexdigest(),
        target_H_used_in_fitting=False,teacher_H_labels=sum(len(v) for v in teacher_indices.values())))
    targets={name:score_target(arrays[name],predictions[name]) for name in TARGET}
    result=dict(registration=registration,provenance=provenance,teacher_indices={k:v.tolist() for k,v in teacher_indices.items()},
        fits=fits,targets=targets,wall_seconds=perf_counter()-start,
        work=dict(new_H_calls=0,new_amplitude_calls=0,historical_cached_H_rows_loaded=8*512,
                  charged_teacher_H_labels=192,charged_target_H_labels_per_proposed_update=32,
                  target_full_reference_H_rows_scoring_only=2*512,
                  inherited_complete_A1_derivative_cache_seconds=sum(v['original_derivative_cache_seconds'] for v in provenance.values())),
        tenfold_gate=dict(variance_ratio_required_for_10x_H_query_saving=.1,
            baseline_H_records_per_target=384,target_H_records=32,
            two_target_H_query_speedup_ceiling_with_192_teacher_and_no_fixed_cost=768/(192+64),
            asymptotic_no_teacher_H_query_ceiling=12,
            maximum_unchanged_cost_over_baseline_H_cost_for_asymptotic_10x=(1-10/12)/9,
            teacher_amortization_target_count_for_H_only_10x=30,
            comments='30 targets is algebraic lower bound:384N/(192+32N)>=10. Fixed A1 derivatives, MC, training, validation and uncertainty only raise it.'),
        limits=['Old panels were used in historical research; this is retrospective screening, not new independent audit.',
                'Whole-chain finite-panel residual correction is unbiased; physical Born/mixing/tail precision is not thereby certified.',
                '24 cached tangent directions cannot replace all-W training or certify finite-step energy/variance.',
                'Scalar physical configuration features preserve virtual-gauge covariance of the force estimator; they are not an SU2 wavefunction.',
                'No end-to-end runtime gain is measured by reading a cache.'])
    atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300); a=p.parse_args(); run(a.output,a.steps)
