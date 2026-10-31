"""Bounded actual-CT6 cross-branch SU(2) gate development, no remote writes.

The physical state is P_A1 U(theta)|raw CT>. No complete support is enumerated.
Tangent fits are explicitly empirical, not a proof about the full CT tangent.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.dressing.multiplicity_gate import plaquette_generators
from su2_nqs.dressing.physical_gate_amplitude import jax_plaquette_unitary
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_amplitude, make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank, triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT/'projects/clebsch_tree_nqs/results/ct6_status_20261025/j0125_final/checkpoint.pkl'
CFG = MixedClebschConfig(6,28,10)
SITES = (0,1,8,9)
VECTOR = np.asarray((.5,-.5,-1/(2*np.sqrt(2)),np.sqrt(3/8)))
SEED = 2610271101


def local_entry_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All70 four-spin matrix entries conserving doubled magnetization."""
    bits = ((np.arange(16)[:,None] >> np.arange(3,-1,-1))&1).astype(np.int32)
    pairs = np.asarray([(a,b) for a in range(16) for b in range(16)
        if bits[a].sum()==bits[b].sum()],np.int32)
    ids = {tuple(pair):i for i,pair in enumerate(pairs)}
    old = np.zeros((16,6),np.int32); valid=np.zeros((16,6),bool); indices=np.zeros((16,6),np.int32)
    for a in range(16):
        values=np.flatnonzero(bits.sum(1)==bits[a].sum())
        old[a]=a; old[a,:len(values)]=values; valid[a,:len(values)]=True
        indices[a]=ids[a,a];indices[a,:len(values)]=[ids[a,b] for b in values]
    return bits,pairs,old,valid,indices


def projected_entry_query(config: MixedClebschConfig, images: np.ndarray):
    """Return full coherent gate-entry amplitudes and positive raw guide."""
    bits,pairs,old,valid,indices=local_entry_tables()
    bit_table,old,valid,indices=map(jnp.asarray,(bits,old,valid,indices))
    sites=jnp.asarray(SITES);powers=jnp.asarray((8,4,2,1));maps=jnp.asarray(images.reshape(-1,4,36))
    raw=make_feature_lut_amplitude(config)
    def one(bank,row):
        def image(y):
            a=y[sites]@powers
            rows=jnp.broadcast_to(y,(6,36)).at[:,sites].set(bit_table[old[a]])
            values=jax.vmap(raw,in_axes=(None,0))(bank,rows)
            entries=jnp.zeros(70).at[indices[a]].add(jnp.where(valid[a],values,0.))
            q=jnp.sum(jnp.where(valid[a],values**2,0.))/jnp.sum(valid[a])
            return entries,q
        def group(perms):
            entries,q=jax.vmap(image)(row[perms])
            return entries.sum(0),q.sum()
        entries,q=jax.lax.map(group,maps)
        return entries.sum(0)/len(images),q.sum()/len(images)
    return jax.jit(jax.vmap(one,in_axes=(None,0))),pairs


def prepare(output: Path) -> dict:
    """Freeze a trusted parent and measure actual coherent amplitude cost."""
    output.mkdir(parents=True,exist_ok=True)
    if (output/'prepare.json').exists(): raise FileExistsError('immutable preparation already exists')
    start=perf_counter();saved=load_checkpoint(PARENT);native=saved['params']
    original=arity_clebsch_bank(native,CFG,'binary_truncated',8)
    norm=float(triangular_raw_overlap(original,original,CFG).real)
    canonical,diagnostics=canonicalize_triangular_bank(original,CFG)
    # Constant global normalization only; no channel removed or optimizer reused.
    canonical=jax.tree.map(lambda a:np.asarray(a).copy(),canonical)
    canonical['top']=(*canonical['top'][:-1],tuple(x/np.sqrt(norm) for x in canonical['top'][-1]))
    cnorm=float(triangular_raw_overlap(canonical,canonical,CFG).real)
    if abs(cnorm-1)>2e-9: raise ArithmeticError('normalized raw source failed')
    save_checkpoint(output/'normalized_canonical_parent.pkl',canonical,metadata=dict(
        original_checkpoint_sha256=file_sha256(PARENT),original_raw_norm=norm,
        optimizer_reset=True,original_optimizer_untouched=True))
    images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
    audit=np.load(PARENT.parent/'terminal_audit.npz');labels=audit['chain_labels']
    chains=np.r_[np.arange(32),np.arange(128,160)]
    indices=np.asarray([np.flatnonzero(labels==label)[0] for label in chains])
    rows=audit['bits'][indices]
    np.savez(output/'tangent_rows.npz',rows=rows,chain_labels=chains,source_indices=indices)
    prepared=prepare_feature_lut_bank(canonical,CFG)
    query,pairs=projected_entry_query(CFG,images)
    ticks={};values=[]
    for label,row in zip(('cold','warm'),rows[:2],strict=True):
        tick=perf_counter();entry,q=query(prepared,row[None]);jax.block_until_ready(entry)
        ticks[label]=perf_counter()-tick;values.append((np.asarray(entry[0]),float(q[0])))
    plain=jax.jit(make_feature_lut_projected_amplitude(CFG,images,chunk=4))
    eye=np.eye(16)[pairs[:,0],pairs[:,1]]
    g=np.einsum('i,ijk->jk',VECTOR,np.asarray(plaquette_generators()).real)[pairs[:,0],pairs[:,1]]
    references=np.asarray([float(plain(prepared,row)) for row in rows[:2]])
    actual=np.asarray([v[0]@eye for v in values]);scale=max(np.max(abs(references)),1e-300)
    error=float(np.max(abs(actual-references))/scale)
    if error>2e-10: raise ArithmeticError('identity differs from trusted fullA1 parent')
    # Full native parameters, not folded free tensors: keep original rank8 constraint.
    flat,unravel=ravel_pytree(native)
    def score(vector,row):
        bank=arity_clebsch_bank(unravel(vector),CFG,'binary_truncated',8)
        return jnp.log(jnp.abs(plain(prepare_feature_lut_bank(bank,CFG),row)/np.sqrt(norm)))
    grad=jax.jit(jax.grad(score));tick=perf_counter();first=np.asarray(grad(flat,rows[0]));cold=perf_counter()-tick
    tick=perf_counter();second=np.asarray(grad(flat,rows[1]));warm=perf_counter()-tick
    if not np.isfinite(first).all() or not np.isfinite(second).all(): raise ArithmeticError('nonfinite full-native score')
    np.savez(output/'jacobian_first_rows.npz',rows=rows[:2],scores=np.stack((first,second)))
    report=dict(status='prepared_actual_CT6_parent_and_complete_A1_gate',model=dict(length=6,multiplicity_cap=28,maximum_q=10),
        original_checkpoint=str(PARENT.relative_to(ROOT)),original_checkpoint_sha256=file_sha256(PARENT),
        absolute_step=saved['metadata']['absolute_completed_steps'],params_sha256=pytree_sha256(native),
        original_raw_norm=norm,normalized_raw_norm=cnorm,canonicalization=diagnostics,
        sites=SITES,angle_vector=VECTOR.tolist(),gate_before_full_A1=True,images=len(images),j2=.125,
        raw_amplitude_bank_parameter_count=int(flat.size),projected_entry_timing=ticks,
        full_native_gradient_cold_seconds=cold,full_native_gradient_warm_seconds=warm,
        full_A1_identity_relative_error=error,first_gate_scores=[float(v[0]@g/(v[0]@eye)) for v in values],
        tangent_training_rows=32,tangent_validation_rows=32,reused_old_audit_not_new_untouched_samples=True,
        ridge_relative_values=[1e-8,1e-6,1e-4,1e-2],
        full_tangent_certificate_not_claimed=True,remote_jobs_unchanged=True,
        full_H_centers=0,wall_seconds=perf_counter()-start,
        source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in (Path(__file__),
            ROOT/'su2_nqs/dressing/physical_gate_amplitude.py',ROOT/'su2_nqs/dressing/multiplicity_gate.py',
            ROOT/'su2_nqs/models/triangular_arity_clebsch.py',ROOT/'su2_nqs/models/triangular_clebsch_feature_lut.py')})
    atomic_json(output/'prepare.json',report)
    return report


def tangent(output: Path, count: int = 32) -> dict:
    """Cross-chain regularized full-native tangent fit with known-W controls."""
    import json
    report=json.loads((output/'prepare.json').read_text());start=perf_counter()
    if count not in (32,128): raise ValueError('bounded tangent rows per split:32 or128')
    artifact_label='tangent' if count==32 else 'tangent128'
    if (output/f'{artifact_label}.json').exists(): raise FileExistsError('immutable tangent review exists')
    native=load_checkpoint(PARENT)['params'];flat,unravel=ravel_pytree(native)
    bank=load_checkpoint(output/'normalized_canonical_parent.pkl')['params']
    images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
    if count==32:
        rows=np.load(output/'tangent_rows.npz')['rows']
    else:
        # One bounded calibration repair after known-W control fails at32.
        # No new MC or H queries; all256 old independent-chain labels once.
        audit=np.load(PARENT.parent/'terminal_audit.npz');chains=audit['chain_labels']
        rows=audit['bits'][[np.flatnonzero(chains==i)[0] for i in range(256)]]
        np.savez(output/'tangent128_rows.npz',rows=rows,chain_labels=np.arange(256))
    plain=make_feature_lut_projected_amplitude(CFG,images,chunk=4)
    def score(vector,row):
        b=arity_clebsch_bank(unravel(vector),CFG,'binary_truncated',8)
        return jnp.log(jnp.abs(plain(prepare_feature_lut_bank(b,CFG),row)/np.sqrt(report['original_raw_norm'])))
    grad=jax.jit(jax.grad(score));first=np.load(output/'jacobian_first_rows.npz')['scores']
    jac=np.empty((2*count,len(flat)),np.float64);jac[:2]=first
    if count==128:
        # The oldfirsttwoarechain0/1, also firsttwo of this expanded fixedsplit.
        if not np.array_equal(rows[:2],np.load(output/'tangent_rows.npz')['rows'][:2]):
            raise ArithmeticError('source-chain identity changed')
    for i in range(2,2*count):
        jac[i]=np.asarray(grad(flat,rows[i]))
        if i%8==7: print('native tangent rows',i+1,flush=True)
    query,pairs=projected_entry_query(CFG,images);prepared=prepare_feature_lut_bank(bank,CFG)
    entries=[]
    for i,row in enumerate(rows):
        e,_=query(prepared,row[None]);entries.append(np.asarray(e[0]))
        if i%16==15: print('gate tangent rows',i+1,flush=True)
    entries=np.asarray(entries);eye=np.eye(16)[pairs[:,0],pairs[:,1]]
    g=np.einsum('i,ijk->jk',VECTOR,np.asarray(plaquette_generators()).real)[pairs[:,0],pairs[:,1]]
    psi=entries@eye;added=(entries@g)/psi
    jac-=jac[:count].mean(0);added-=added[:count].mean()
    train,test=jac[:count],jac[count:];k=train@train.T;cross=test@train.T
    values,axes=np.linalg.eigh((k+k.T)/2);largest=float(values.max())
    if largest<=0: raise ArithmeticError('full native physical score kernel empty')
    rng=np.random.default_rng(SEED);broad=rng.normal(size=len(flat));broad/=np.linalg.norm(broad)
    known=jac@broad;known/=np.sqrt(np.mean(known[:count]**2))
    representer=rng.normal(size=count);replay=np.r_[k@representer,cross@representer]
    replay/=np.sqrt(np.mean(replay[:count]**2))
    targets=dict(gate=added/np.sqrt(np.mean(added[:count]**2)),known_broad_free_W=known,known_representer_free_W=replay)
    fits={}
    for label,target in targets.items():
        fits[label]=[]
        for relative in report['ridge_relative_values']:
            inv=1/(np.maximum(values,0)+relative*largest)
            alpha=axes@(inv*(axes.T@target[:count]));prediction=cross@alpha
            fits[label].append(dict(relative_ridge=relative,
                train_relative_MSE=float(np.mean((k@alpha-target[:count])**2)/np.mean(target[:count]**2)),
                validation_relative_MSE=float(np.mean((prediction-target[count:])**2)/np.mean(target[count:]**2)),
                effective_kernel_DOF=float(np.sum(np.maximum(values,0)*inv))))
    broad_best=min(x['validation_relative_MSE'] for x in fits['known_broad_free_W'])
    gate_best=min(x['validation_relative_MSE'] for x in fits['gate'])
    result=dict(status='complete_empirical_full_native_tangent_diagnostic',fit=fits,
        original_parameters=int(flat.size),train_rows=count,validation_rows=count,
        kernel_eigenvalues=values.tolist(),gate_score_train_RMS=float(np.sqrt(np.mean(added[:count]**2))),
        gate_score_validation_RMS=float(np.sqrt(np.mean(added[count:]**2))),
        known_W_generalization_calibrated=broad_best<.2,
        empirical_increment_supported=broad_best<.2 and gate_best>.5,
        outside_complete_exact_CT_tangent_proved=False,
        underidentified_fit_not_a_no_novelty_proof=True,full_H_centers=0,
        wall_seconds=perf_counter()-start)
    np.savez(output/f'{artifact_label}_products.npz',entries=entries,parent_psi=psi,gate_score=added,
        kernel=k,cross_kernel=cross,known_free_W_score=known,known_representer_score=replay)
    atomic_json(output/f'{artifact_label}.json',result)
    return result


def cost(output: Path) -> dict:
    """Measure one unchanged finite gate family with actual complete H action."""
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.vmc.ct_four_spin_gate_products import make_projected_gate_products
    from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action
    from .triangle_hierarchy_initialization import _node_timeout
    if (output/'cost.json').exists(): raise FileExistsError('immutable actual-H cost already exists')
    started=perf_counter();bank=load_checkpoint(output/'normalized_canonical_parent.pkl')['params']
    images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
    bonds=triangular_j1j2_bonds(6,6,j2=.125,periodic=True)
    prepared=prepare_feature_lut_bank(bank,CFG);row=np.load(output/'tangent_rows.npz')['rows'][0]
    tick=perf_counter();action,metadata=make_projected_gate_products(CFG,images,bonds,SITES,projection_chunk=1)
    factory=perf_counter()-tick;timings={};products=[]
    with _node_timeout(600):
        for name in ('cold','warm'):
            tick=perf_counter();value=action(prepared,row[None]);jax.block_until_ready(value)
            timings[name]=perf_counter()-tick;products.append([np.asarray(x[0]) for x in value])
        standard=make_cached_projected_heisenberg_action(CFG,images,bonds,projection_chunk=4)
        tick=perf_counter();reference=np.asarray([float(x[0]) for x in standard(prepared,row[None])])
        timings['parent_reference_cold']=perf_counter()-tick
        tick=perf_counter();reference2=np.asarray([float(x[0]) for x in standard(prepared,row[None])])
        timings['parent_reference_warm']=perf_counter()-tick
    pairs=np.asarray(metadata['entry_pairs']);eye=np.eye(16)[pairs[:,0],pairs[:,1]]
    generated=np.asarray([products[0][i]@eye for i in (0,1)])
    relative=float(np.max(abs(generated-reference))/max(np.max(abs(reference)),1e-300))
    replay=max(float(np.max(abs(a-b))) for a,b in zip(products[0],products[1],strict=True))
    if relative>2e-10 or replay!=0 or not np.array_equal(reference,reference2):
        raise ArithmeticError('gate fullH identity/replay differs from trusted parent')
    for i in (2,3):
        if products[0][i]<=0 or not np.isfinite(products[0][i]): raise ArithmeticError('nonpositive core/bridge density')
    np.savez(output/'cost_products.npz',row=row,Phi70=products[0][0],HPhi70=products[0][1],
        Qcore=products[0][2],Qpush=products[0][3],reference=reference)
    result=dict(status='actual_CT6_complete_A1_full_H_cost_and_identity_passed',kernel=metadata,
        factory_seconds=factory,timings=timings,identity_relative_error=relative,replay_max_error=replay,
        full_H_feature_centers=2,parent_reference_H_centers=2,
        no_energy_estimate_from_cost_row=True,no_native_CT_training=True,
        source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in (Path(__file__),
            ROOT/'su2_nqs/vmc/ct_four_spin_gate_products.py')},wall_seconds=perf_counter()-started)
    atomic_json(output/'cost.json',result);return result


def finite_gate(output: Path) -> dict:
    """Fixed32 fitting /128 untouched IID audit rows, one identity-start gate.

    This is a one-angle structural canary, not a claim of projected tangent
    novelty or a fair optimization win against all original free W variables.
    Twenty offline angle updates reuse32 full-H feature centers exactly.
    """
    import gc
    import json
    import resource
    import shutil
    import sys
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.vmc.ct_four_spin_gate_products import make_projected_gate_products
    from su2_nqs.vmc.triangular_raw_born import make_triangular_raw_born_sampler
    from .triangle_hierarchy_initialization import _node_timeout
    from .triangle_iid_orbit_audit import paired_iid_moments
    from .triangle_iid_orbit_overlap import paired_projected_overlap
    if (output/'finite_gate_registration.json').exists(): raise FileExistsError('immutable finite-gate run exists')
    raw_certificate=json.loads((output/'raw_tangent_certificate.json').read_text())
    tangent_record=json.loads((output/'tangent128.json').read_text())
    cost_record=json.loads((output/'cost.json').read_text())
    sources=[Path(__file__),ROOT/'su2_nqs/dressing/physical_gate_amplitude.py',
        ROOT/'su2_nqs/vmc/ct_four_spin_gate_products.py',ROOT/'su2_nqs/vmc/triangular_raw_born.py',
        ROOT/'su2_nqs/vmc/raw_tree_born.py',Path(__file__).with_name('triangle_iid_orbit_audit.py'),
        Path(__file__).with_name('triangle_iid_orbit_overlap.py')]
    registration=dict(task='actual_CT6_crossbranch_SU2_gate_complete_A1_finite_canary_v1',
        sites=SITES,angle_vector=VECTOR.tolist(),j2=.125,maximum_absolute_theta=.15,
        fitting_records=32,untouched_audit_records=128,audit_groups=16,records_per_group=8,
        seed=2610271201,audit_seed=2610271202,optimizer='scalar_Adam',offline_angle_updates=20,
        learning_rate=.02,training_FS_radius_cap=.01,maximum_FS_backtrack_halvings=8,
        exact_state='P_A1 U(theta*v) raw_CT; W frozen; source root normalized only',
        no_original_W_retraining_or_complete_projected_tangent_novelty_claim=True,
        raw_tangent_certificate_available=True,
        known_W_control_failed_to_calibrate_projected_novelty=not tangent_record['known_W_generalization_calibrated'],
        bridge_fraction=.5,proposal='Q=.5*(Qcore+Qpush); normalized exactrawBorn, uniformsame-m4spin replacement, uniformfullspacegroup, allweightedHbondbridge includingnulls',
        normalization_not_assumed_after_A1=True,all_H_connections_in_second_moment=True,
        fixed_terminal_angle_not_best_training_checkpoint=True,no_audit_based_model_or_step_selection=True,
        numerical_wall_cap_seconds=3600,full_H_center_forecast_seconds=cost_record['timings']['warm'],
        parent_checkpoint_sha256=file_sha256(PARENT),
        canonical_parent_sha256=file_sha256(output/'normalized_canonical_parent.pkl'),
        raw_certificate_sha256=file_sha256(output/'raw_tangent_certificate.json'),
        source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in sources})
    atomic_json(output/'finite_gate_registration.json',registration)
    frozen=output/'registered_sources_finite_gate';frozen.mkdir()
    for path in sources: shutil.copy2(path,frozen/path.name)
    start=perf_counter();result=dict(status='started',full_H_feature_centers=0,completed_fit_rows=0,completed_audit_rows=0,
        registration=registration)
    timings=[]
    try:
        with _node_timeout(3600):
            bank=load_checkpoint(output/'normalized_canonical_parent.pkl')['params']
            images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
            inverse_images=np.argsort(images,axis=1)
            bonds=triangular_j1j2_bonds(6,6,j2=.125,periodic=True)
            edge=np.asarray([(a,b) for a,b,_ in bonds],np.int32)
            bp=np.abs(np.asarray([g for _,_,g in bonds]));bp/=bp.sum()
            profile=[row['widths'] for row in mixed_clebsch_layout(CFG)]
            tick=perf_counter();sampler=make_triangular_raw_born_sampler(bank,CFG,
                active_top_widths=profile,memory_limit_bytes=2*1024**3)
            result['sampler_factory_seconds']=perf_counter()-tick
            result['sampler_diagnostics']=sampler.diagnostics
            bit_table,_,_,_,_=local_entry_tables();rows=[];latent=[];raws=[];replay=[]
            for label,seed,count in (('fit',registration['seed'],32),('audit',registration['audit_seed'],128)):
                rng=np.random.default_rng(seed)
                for _ in range(count):
                    draw=sampler.draw(rng);raw=draw.bits.copy();row=raw.copy()
                    cohort=np.flatnonzero(bit_table.sum(1)==row[list(SITES)].sum())
                    old=int(rng.choice(cohort));row[list(SITES)]=bit_table[old]
                    image=int(rng.integers(len(images)));row=row[inverse_images[image]].copy()
                    use=bool(rng.random()<.5);bond=int(rng.choice(len(bonds),p=bp))
                    if use:
                        a,b=edge[bond];row[a],row[b]=row[b],row[a]
                    rows.append(row);raws.append(raw);latent.append((old,image,int(use),bond));replay.append(draw.log_probability_replay_error)
                result[label+'_draw_rng_state']=rng.bit_generator.state
            rows=np.asarray(rows,np.int32)
            np.savez(output/'finite_registered_rows.npz',rows=rows,raw_rows=np.asarray(raws),
                latent=np.asarray(latent),log_Born_replay=np.asarray(replay),audit_groups=np.repeat(np.arange(16),8))
            result['maximum_raw_Born_replay_error']=max(replay)
            del sampler;gc.collect()
            prepared=prepare_feature_lut_bank(bank,CFG)
            action,metadata=make_projected_gate_products(CFG,images,bonds,SITES,projection_chunk=1)
            result['kernel']=metadata
            pairs=np.asarray(metadata['entry_pairs']);eye=np.eye(16)[pairs[:,0],pairs[:,1]]
            def coefficients(theta):
                u=jax_plaquette_unitary(theta*jnp.asarray(VECTOR))
                return u[pairs[:,0],pairs[:,1]]
            fit=[]
            for i,row in enumerate(rows[:32]):
                tick=perf_counter();values=action(prepared,row[None]);jax.block_until_ready(values)
                timings.append(perf_counter()-tick);fit.append([np.asarray(x[0]) for x in values])
                result['full_H_feature_centers']+=1;result['completed_fit_rows']=i+1
                if i%4==3:
                    atomic_json(output/'finite_progress.json',dict(result,wall_seconds=perf_counter()-start))
                    print('finite gate fitting H rows',i+1,'/',32,flush=True)
            fit_phi=np.asarray([r[0] for r in fit]);fit_h=np.asarray([r[1] for r in fit])
            fit_q=np.asarray([.5*(r[2]+r[3]) for r in fit])
            np.savez(output/'finite_fitting_products.npz',Phi70=fit_phi,HPhi70=fit_h,proposal=fit_q)
            if np.any(fit_q<=0): raise ArithmeticError('positive normalized fitting proposal required')
            p0=fit_phi@eye
            fp,fh,fq=map(jnp.asarray,(fit_phi,fit_h,fit_q))
            def energy(theta):
                c=coefficients(theta);p=fp@c;h=fh@c
                return jnp.sum(p*h/fq)/jnp.sum(p*p/fq)/36
            def distance(theta):
                p=fit_phi@np.asarray(coefficients(theta))
                overlap=np.sum(p0*p/fit_q)/np.sqrt(np.sum(p0*p0/fit_q)*np.sum(p*p/fit_q))
                return float(np.arccos(np.clip(abs(overlap),0,1)))
            gradient=jax.jit(jax.value_and_grad(energy));theta=0.;m=v=0.;history=[]
            for step in range(1,21):
                value,grad=map(float,gradient(theta));m=.9*m+.1*grad;v=.999*v+.001*grad*grad
                update=.02*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
                proposed=float(np.clip(theta-update,-.15,.15));halves=0
                while distance(proposed)>.01 and halves<8:
                    proposed=(theta+proposed)/2;halves+=1
                if distance(proposed)>.01: proposed=theta
                theta=proposed
                history.append(dict(step=step,theta=theta,preupdate_fit_E_per_site=value,gradient=grad,
                    estimated_parent_FS=distance(theta),FS_halvings=halves))
            endpoint=output/'finite_gate_checkpoint.pkl'
            save_checkpoint(endpoint,dict(theta=np.asarray(theta)),optimizer_state=dict(m=m,v=v,count=20),
                metadata=dict(parent=str(output/'normalized_canonical_parent.pkl'),sites=SITES,
                    angle_vector=VECTOR.tolist(),state_definition='complete_A1_gate_before_projection',
                    fitting_rows_sha256=file_sha256(output/'finite_fitting_products.npz'),
                    audit_rng_state=result['audit_draw_rng_state'],history=history))
            restored=load_checkpoint(endpoint);restored_theta=float(restored['params']['theta'])
            _,next_grad=map(float,gradient(theta));restored_grad=float(gradient(restored_theta)[1])
            expected_m=.9*m+.1*next_grad;expected_v=.999*v+.001*next_grad**2
            rs=restored['optimizer_state'];replay_update=max(abs(expected_m-(.9*rs['m']+.1*restored_grad)),
                abs(expected_v-(.999*rs['v']+.001*restored_grad**2)),abs(theta-restored_theta))
            if replay_update!=0: raise ArithmeticError('finite gate optimizer checkpoint replay differs')
            result.update(fixed_terminal_theta=theta,history=history,optimizer_next_step_replay_error=replay_update,
                checkpoint_sha256=file_sha256(endpoint),fit_endpoint_E_per_site=float(energy(theta)),
                fit_parent_E_per_site=float(energy(0.)),fit_parent_FS=distance(theta))
            atomic_json(output/'finite_frozen_endpoint.json',result)
            # No endpoint decision can use any of the following128 H products.
            coefficients_pair=np.stack((eye,np.asarray(coefficients(theta))))
            products=[]
            for i,row in enumerate(rows[32:]):
                tick=perf_counter();values=action(prepared,row[None]);jax.block_until_ready(values)
                timings.append(perf_counter()-tick);products.append([np.asarray(x[0]) for x in values])
                result['full_H_feature_centers']+=1;result['completed_audit_rows']=i+1
                if i%8==7:
                    atomic_json(output/'finite_progress.json',dict(result,wall_seconds=perf_counter()-start))
                    print('finite gate untouched H rows',i+1,'/',128,flush=True)
            phi=np.asarray([r[0] for r in products]);hp=np.asarray([r[1] for r in products])
            density=np.asarray([.5*(r[2]+r[3]) for r in products]);groups=np.repeat(np.arange(16),8)
            psi=phi@coefficients_pair.T;hpsi=hp@coefficients_pair.T
            if not np.isfinite([phi,hp]).all() or np.any(density<=0): raise ArithmeticError('audit product/density failure')
            if np.max(psi*psi/density[:,None])>12*(1+2e-9): raise ArithmeticError('unitary/guide support bound violated')
            np.savez(output/'finite_untouched_products.npz',rows=rows[32:],groups=groups,
                Phi70=phi,HPhi70=hp,psi=psi,Hpsi=hpsi,proposal_density=density,
                Qcore=np.asarray([r[2] for r in products]),Qpush=np.asarray([r[3] for r in products]))
            result['moments']=paired_iid_moments(psi,hpsi,density,groups,n_sites=36,seed=2610271211)
            result['overlap']=paired_projected_overlap(psi,density,groups,seed=2610271212)
            result['status']='complete_one_angle_gate_with_frozen_untouched_full_H_audit'
            result['coefficient_entry_queries_amortized_over_offline_angle_updates']=True
            result['no_production_or_formal_benchmark_update']=True
    except (TimeoutError,MemoryError) as error:
        result.update(status='bounded_incomplete',error=repr(error))
    finally:
        result.update(wall_seconds=perf_counter()-start,H_feature_center_seconds=timings,
            peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        atomic_json(output/'finite_gate_results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','tangent','tangent128','cost','finite'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    answer=(prepare(args.output) if args.stage=='prepare' else finite_gate(args.output) if args.stage=='finite'
        else cost(args.output) if args.stage=='cost'
        else tangent(args.output,128 if args.stage=='tangent128' else 32))
    print(answer['status'],flush=True)
