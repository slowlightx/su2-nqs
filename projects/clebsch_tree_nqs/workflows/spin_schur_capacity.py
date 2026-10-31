"""Serial chi=2/4 free-tensor capacity trial: two-node Ritz/SVD sweeps.

Reuse the immutable carrier, units, known positive-Q records and full-H
planner. This is bounded local optimization, not a stationarity certificate.
No neural architecture, response head, ED, remote job or route 2/3 is built.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact, load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import (
    FiniteSchurDecoderConfig, make_finite_schur_amplitude, free_reduced_bank,
    reduced_entry_metadata)
from su2_nqs.models.schur_two_site import (
    embed_reduced_bank, internal_edges, two_site_layout, join_reduced_pair,
    split_reduced_pair, make_two_site_features)
from su2_nqs.models.schur_cutset_compact import make_full_a1_batch
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products
from su2_nqs.vmc.carrier_subspace import fit_carrier_subspace
from su2_nqs.vmc.cutset_training import paired_chain_audit, weighted_state_distance
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products
from su2_nqs.vmc.projected_spin_linear import make_projected_linear_products
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint
from .spin import _groups, _bonds
from .spin_decoder_cost import ROOT, physical_config, peak_rss_bytes
from .spin_finite_schur_optimize import setup as old_setup
from .spin_finite_schur_population import loaded, carrier_h2
from .spin_schur_carrier_control import batched, draw

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_capacity.json'


def configuration(path: Path) -> dict:
    return json.loads(path.read_text())


def parent_setup(spec: dict):
    base = configuration(ROOT/spec['base_config'])
    n, initial, generators, _, carrier, bonds, _, _ = old_setup(base)
    cfg = FiniteSchurDecoderConfig(replace(physical_config(6, 6, 2), maximum_intermediate_q=4), 3, 3, 16, 2)
    bank = generators['free'](initial['free'])
    source = ROOT/spec['carrier_source']
    old = configuration(source/'results.json')
    if not old['accepted']['filter']:
        raise ValueError('require retained C2 carrier')
    coef = np.asarray(configuration(source/'filter.json')['coefficients'])
    units = configuration(source/'prepared.json')
    projector = PhysicalSpatialProjector(_groups('triangular', 6, 6)['gamma_a1'])
    return cfg, bank, carrier, bonds, projector, coef, units


def resources(spec: dict, start: float) -> None:
    if perf_counter()-start > spec['wall_limit_seconds_per_process']:
        raise TimeoutError('fixed process wall cap reached; checkpoint retained')
    if peak_rss_bytes() > spec['peak_rss_limit_gib']*2**30:
        raise MemoryError('fixed memory cap reached; checkpoint retained')


def registration(spec: dict) -> dict:
    if spec['widths'] != [2, 4] or spec['maximum_q'] != 4 or spec['sweeps'] != 2:
        raise ValueError('fixed two-width, two-sweep capacity experiment required')
    if len(set(spec['audit_seeds'])) != 2 or spec['basis_columns'] != 2:
        raise ValueError('two distinct audit streams and two local search columns required')
    source = ROOT/spec['carrier_source']; parent_setup(spec)
    paths = [CONFIG, ROOT/spec['base_config']]
    paths += [source/p for p in ('results.json', 'filter.json', 'filter_checkpoint.pkl',
                                 'prepared.json', 'fit.npz', 'geometry.npz')]
    base = configuration(ROOT/spec['base_config'])
    paths += [ROOT/base['parent']/p for p in ('free_tensors.json', 'free_tensors.npz', 'network.json', 'network.npz')]
    row = dict(config=spec, runtime=numerical_runtime_identity(),
        inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in paths},
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    row['registration_sha256'] = canonical_sha256(row)
    return row


def bank_functions(cfg, projector, bonds, spec):
    raw = make_finite_schur_amplitude(cfg)
    projected = jax.jit(make_full_a1_batch(raw, projector, chunk_size=32))
    plan, evaluate, _ = make_projected_heisenberg_products(raw, projector, bonds, chunk_size=64)
    return projected, jax.jit(plan), jax.jit(evaluate)


def products(bank, states, plan, evaluate, size=4):
    p, hp, records = [], [], []
    tick = perf_counter()
    for i in range(0, len(states), size):
        prepared = plan(jnp.asarray(states[i:i+size])); h, a, count, padded = evaluate(bank, prepared)
        p.append(np.asarray(a)); hp.append(np.asarray(h))
        records.append(dict(unique=int(count), padded=int(padded)))
    return np.concatenate(p), np.concatenate(hp), dict(seconds=perf_counter()-tick,
        centers=len(states), raw_queries=sum(v['unique'] for v in records), padded_queries=sum(v['padded'] for v in records))


def prepare(spec, reg, output):
    cfg, bank, _, bonds, projector, coef, units = parent_setup(spec)
    source = ROOT/spec['carrier_source']; fit, geo = (loaded(source/(k+'.npz')) for k in ('fit', 'geometry'))
    data = {k: np.concatenate((fit[k], geo[k])) for k in ('states', 'q')}
    powers = np.concatenate((fit['base_products'], geo['base_products']))
    data['carrier'], data['hcarrier'] = polynomial_carrier_products(powers, coef)
    data['fit_indices'] = np.arange(spec['H_centers_per_block'])
    data['control_indices'] = np.arange(64, 64+spec['H_centers_per_block'])
    data['H_indices'] = np.r_[data['fit_indices'], data['control_indices']]
    projected, plan, evaluate = bank_functions(cfg, projector, bonds, spec)
    data['reference'] = batched(lambda x: projected(bank, x), data['states'], 4)
    rp, rh, work = products(bank, data['states'][data['H_indices']], plan, evaluate)
    np.testing.assert_allclose(rp, data['reference'][data['H_indices']], rtol=2e-10, atol=2e-16)
    data['reference_H'] = rh
    np.savez_compressed(output/'known.npz', **data)
    save_parameter_artifact(output/'reference.json', bank, identity={'registration_sha256': reg['registration_sha256']})
    atomic_json(output/'prepared.json', dict(registration_sha256=reg['registration_sha256'],
        coefficients=coef.tolist(), cscale=units['cscale'], residual_scale=units['residual_scale'],
        reference_work=work, known_records=len(data['q']), new_training_samples=0,
        known_records_not_untouched=True, full_A1_images=projector.group_order, bonds=len(bonds)))
    print('prepared',len(data['q']),'known records; reference H',work,flush=True)


def search_directions(features, parent, hparent, q, scale, spec):
    """Whole-block regularized imaginary-time solve plus leading metric mode.

    These two directions search the full joined tensor, not a new model head.
    The sweep is not complete optimization of that tensor or whole manifold.
    """
    norm = np.sum(abs(parent)**2/q)
    overlap = np.sum(parent.conj()[:, None]*features/q[:, None], axis=0)/norm
    a = scale*(features-parent[:, None]*overlap)/np.sqrt(q[:, None]*norm)
    gram = a@a.conj().T
    values, vectors = np.linalg.eigh((gram+gram.conj().T)/2)
    active = values > max(values[-1]*spec['metric_cutoff'], 1e-24)
    if not np.any(active):
        return np.zeros((features.shape[1], 0), complex), dict(rank=0)
    energy = np.sum(parent.conj()*hparent/q).real/norm
    target = -(hparent-energy*parent)/np.sqrt(q*norm)
    v, lam = vectors[:, active], values[active]
    aligned = a.conj().T@(v@((v.conj().T@target)/(lam+spec['metric_damping']*lam[-1])))
    leading = a.conj().T@v[:, -1]/lam[-1]
    candidates = [aligned, leading]; result = []
    for d in candidates:
        d = d.copy()
        for previous in result:
            d -= previous*np.vdot(a@previous, a@d)
        speed = np.linalg.norm(a@d)
        if speed > 1e-12:
            result.append(d/speed)
    return np.column_stack(result), dict(rank=int(active.sum()), complete_block_parameters=2*features.shape[1],
        searched_columns=len(result), largest_metric_eigenvalue=float(lam[-1]),
        stationarity_certified=False)


def moment(p, hp, q):
    norm = np.sum(abs(p)**2/q); e = np.sum(p.conj()*hp/q)/norm
    return dict(energy_per_site=float(e.real/36), imaginary_energy_per_site=float(e.imag/36),
        variance_per_site=float(np.sum(abs(hp-e*p)**2/q)/norm/36))


def train(spec, reg, output, width):
    start = perf_counter(); small, initial, _, bonds, projector, _, units = parent_setup(spec)
    cfg = replace(small, tree=replace(small.tree, feature_dim=width))
    reference = embed_reduced_bank(initial, small, cfg)
    bank = reference
    folder = output/f'chi{width}'; folder.mkdir(exist_ok=True)
    identity = dict(registration_sha256=reg['registration_sha256'], width=width)
    known = loaded(output/'known.npz'); scale = units['residual_scale']; x = known['states']; q = known['q']
    hi = known['H_indices']; nf = spec['H_centers_per_block']
    projected, plan, evaluate = bank_functions(cfg, projector, bonds, spec)
    history, completed = [], 0
    if (folder/'checkpoint.pkl').exists():
        saved = load_resumable_checkpoint(folder/'checkpoint.pkl', expected_run_identity=identity)
        bank, history, completed = saved['params'], saved['history'], saved['completed_steps']
    initial_phi = batched(lambda z: projected(reference,z), x, 4)
    initial_error = float(np.max(abs(scale*(initial_phi-known['reference']))))
    if initial_error > 2e-9: raise FloatingPointError('width embedding changes physical parent')
    c, hc = known['carrier'], known['hcarrier']
    phi = batched(lambda z: projected(bank,z), x, 4)
    ph, hph, _ = products(bank, x[hi], plan, evaluate)
    p = c+scale*(phi-known['reference']); hp = hc[hi]+scale*(hph-known['reference_H'])
    edges = internal_edges(cfg); schedule = list(edges)+list(reversed(edges))
    for attempt, pair in enumerate(schedule):
        if attempt < completed: continue
        tick = perf_counter(); layout = two_site_layout(cfg,*pair); joined = join_reduced_pair(bank,layout)
        features = make_two_site_features(layout)
        linear_plan, linear_action, feature_projector = make_projected_linear_products(
            features, projector, bonds, chunk_size=spec['linear_H_chunk'])
        feature_projector, linear_plan, linear_action = map(jax.jit,(feature_projector,linear_plan,linear_action))
        f = batched(lambda z:feature_projector(bank,z),x,2)
        error = float(np.max(abs(scale*(f@joined-phi))))
        if error > 2e-8: raise FloatingPointError('joined physical features fail replay')
        # Operator geometry uses the 32 known full-H rows; all 192 known rows
        # constrain actual finite-step FS. These are optimization data, never
        # independent validation. No other H values are invented.
        geometry_p, geometry_hp = p[hi], hp
        dirs, metric = search_directions(f[hi],geometry_p,geometry_hp,q[hi],scale,spec)
        row = dict(attempt=attempt,pair=list(pair),metric=metric,feature_replay_error=error,
                   accepted=False,full_parameter_update=False,physical_embedding_error=initial_error)
        if dirs.shape[1]:
            hdirs, amplitudes, work = [], [], []
            for i in range(0,len(hi),spec['H_microbatch']):
                prepared = linear_plan(jnp.asarray(x[hi[i:i+spec['H_microbatch']]]))
                h,a,count,padded = linear_action(bank,jnp.asarray(dirs),prepared)
                amplitudes.append(np.asarray(a));hdirs.append(np.asarray(h))
                work.append(dict(unique=int(count),padded=int(padded)))
                resources(spec,start)
            t, ht = scale*np.concatenate(amplitudes),scale*np.concatenate(hdirs)
            basis_p, basis_h = np.c_[p[hi],t],np.c_[hp,ht]
            fit = fit_carrier_subspace(basis_p[:nf],basis_h[:nf],q[hi[:nf]],n_sites=36,
                relative_rank_tolerance=1e-8,maximum_condition_number=1e8)
            beta = fit.coefficients
            row.update(ritz=fit.as_dict(),linear_H_centers=len(hi),linear_H_columns=dirs.shape[1],
                linear_raw_queries=sum(v['unique'] for v in work),
                full_H_equivalent_columns=len(hi)*dirs.shape[1])
            selected = None
            if abs(beta[0]) > .1:
                delta = dirs@(beta[1:]/beta[0])
                for alpha in spec['step_fractions']:
                    candidate,trunc = split_reduced_pair(joined+alpha*delta,bank,layout)
                    cz = join_reduced_pair(candidate,layout); candidate_phi=f@cz
                    candidate_p=c+scale*(candidate_phi-known['reference'])
                    distances = [weighted_state_distance(p[m],candidate_p[m],q[m]) for m in (slice(0,64),slice(64,None))]
                    total = [weighted_state_distance(c[m],candidate_p[m],q[m]) for m in (slice(0,64),slice(64,None))]
                    if (max(v['fs'] for v in distances)<=spec['maximum_step_fs']
                        and max(v['fs'] for v in total)<=spec['maximum_parent_fs']
                        and min(v['ess_fraction'] for v in distances)>=spec['minimum_ess_fraction']):
                        selected=(candidate,candidate_phi,candidate_p,alpha,trunc,distances,total);break
                if selected is not None:
                    candidate,candidate_phi,candidate_p,alpha,trunc,distances,total=selected
                    cp,ch,cwork=products(candidate,x[hi],plan,evaluate)
                    candidate_hp=hc[hi]+scale*(ch-known['reference_H'])
                    replay=float(np.max(abs(scale*(cp-candidate_phi[hi]))))
                    if replay>2e-8:raise FloatingPointError('SVD retraction changes feature-based amplitude')
                    old_m=[moment(p[hi][m],hp[m],q[hi][m]) for m in (slice(0,nf),slice(nf,None))]
                    new_m=[moment(candidate_p[hi][m],candidate_hp[m],q[hi][m]) for m in (slice(0,nf),slice(nf,None))]
                    base_m=[moment(c[hi][m],hc[hi][m],q[hi][m]) for m in (slice(0,nf),slice(nf,None))]
                    accept=(new_m[0]['energy_per_site']<old_m[0]['energy_per_site']
                        and new_m[1]['energy_per_site']<=old_m[1]['energy_per_site']+spec['control_energy_tolerance']
                        and all(n['variance_per_site']<=spec['maximum_variance_ratio']*b['variance_per_site'] for n,b in zip(new_m,base_m)))
                    uncompressed=p+scale*f@(alpha*delta)
                    trunc_distance=weighted_state_distance(uncompressed,candidate_p,q)
                    row.update(accepted=bool(accept),alpha=alpha,truncation=trunc,
                        physical_truncation_distance=trunc_distance,distances=distances,parent_distances=total,
                        before=old_m,after=new_m,candidate_work=cwork,retraction_H_amplitude_error=replay)
                    if accept:bank,phi,p,hp=candidate,candidate_phi,candidate_p,candidate_hp
        row['seconds']=perf_counter()-tick;row['peak_rss_bytes']=peak_rss_bytes();history.append(row)
        save_resumable_checkpoint(folder/'checkpoint.pkl',bank,optimizer_state={'reference':reference},
            completed_steps=attempt+1,next_sample_index=0,run_identity=identity,history=history,
            metadata={'no_new_fit_samples':True,'stationarity_certified':False})
        print('chi',width,'attempt',attempt+1,'/',len(schedule),'accepted',row['accepted'],
              'FIT',moment(p[hi[:nf]],hp[:nf],q[hi[:nf]]),'sec',round(row['seconds'],2),flush=True)
        jax.clear_caches();resources(spec,start)
    save_parameter_artifact(folder/'endpoint.json',bank,identity=identity)
    np.savez_compressed(folder/'endpoint_known.npz',psi=p,hpsi=hp,phi=phi)
    atomic_json(folder/'results.json',dict(registration_sha256=reg['registration_sha256'],width=width,
        full_free_parameters=reduced_entry_metadata(cfg)['free_real_parameters'],attempts=len(history),
        accepted=sum(v['accepted'] for v in history),history=history,seconds=perf_counter()-start,
        endpoint_sha256=pytree_sha256(bank),peak_rss_bytes=peak_rss_bytes(),
        stationarity_certified=False,optimization_limited_not_capacity_no_go=True,
        endpoint_sealed_before_fresh_audit=True))


def audit(spec, reg, output):
    small, initial, carrier, bonds, projector, coef, units=parent_setup(spec)
    seals={str(w):file_sha256(output/f'chi{w}'/'endpoint.npz') for w in spec['widths']}
    records=[];start=perf_counter()
    old_spec=configuration(ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_population.json')
    old_spec.update(wall_limit_seconds_per_process=spec['wall_limit_seconds_per_process'])
    for index,seed in enumerate(spec['audit_seeds']):
        sample,info=draw({**spec['audit'], 'uniform_floor_relative_to_pilot_mean':.2,'bridge_fraction':.5},
            36,carrier,bonds,units['cscale'],seed,spec['audit']['records'])
        powers,pwork=carrier_h2(old_spec,sample['states'],carrier,bonds,units['cscale'],start)
        p,hp=polynomial_carrier_products(powers,coef);sample['carrier']=p;sample['hcarrier']=hp
        arms=[p];actions=[hp];work={}
        for width in spec['widths']:
            cfg=replace(small,tree=replace(small.tree,feature_dim=width))
            reference=embed_reduced_bank(initial,small,cfg)
            bank=load_parameter_artifact(output/f'chi{width}'/'endpoint.json',reference)
            _,plan,evaluate=bank_functions(cfg,projector,bonds,spec)
            a,ha,w=products(bank,sample['states'],plan,evaluate)
            r,hr,rw=products(reference,sample['states'],plan,evaluate)
            arms.append(p+units['residual_scale']*(a-r));actions.append(hp+units['residual_scale']*(ha-hr))
            work[str(width)]=dict(endpoint=w,reference=rw)
            resources(spec,start)
        sample.update(psi=np.column_stack(arms),hpsi=np.column_stack(actions),base_products=powers)
        np.savez_compressed(output/f'audit{index}.npz',**sample)
        report=paired_chain_audit(sample['psi'],sample['hpsi'],sample['q'],sample['groups'],36,
            spec['audit']['bootstrap_replicates'],spec['bootstrap_seed']+index,('carrier','chi2','chi4'))
        records.append(dict(seed=seed,sampling=info,carrier_H2_work=pwork,residual_work=work,audit=report))
        print('audit stream',index,'sealed',flush=True)
        jax.clear_caches()
    combined={k:np.concatenate([loaded(output/f'audit{i}.npz')[k] for i in range(2)]) for k in ('psi','hpsi','q','groups')}
    combined['groups'][len(combined['groups'])//2:]+=spec['audit']['chains']
    report=paired_chain_audit(combined['psi'],combined['hpsi'],combined['q'],combined['groups'],36,
        spec['audit']['bootstrap_replicates'],spec['bootstrap_seed']+2,('carrier','chi2','chi4'))
    d=report['paired']['chi4-minus-chi2']['energy_per_site'];r=report['paired']['chi4-minus-chi2']['variance_per_site']
    capacity_signal=d['value'] < -2*d['chain_jackknife_se'] and r['value']<=0
    row=dict(registration_sha256=reg['registration_sha256'],endpoint_seals=seals,streams=records,combined_audit=report,
        resolved_capacity_signal=bool(capacity_signal),physical_capacity_exhausted=False,
        stationary_free_tensors=False,route2_implemented=False,route3_implemented=False,
        route_decision='remain_route1_capacity_supported' if capacity_signal else 'route1_optimization_unresolved_no_architecture_switch',
        no_post_audit_refit=True,new_H2_centers=combined['psi'].shape[0],audit_seconds=perf_counter()-start,
        peak_rss_bytes=peak_rss_bytes(),gpu_hours=0,large_ED_calls=0)
    atomic_json(output/'results.json',row)
    print('capacity signal',capacity_signal,report['paired'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','train','audit'))
    parser.add_argument('--width',type=int,choices=(2,4));parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args();spec=configuration(args.config);output=ROOT/spec['output']
    if args.stage=='prepare':
        output.mkdir(exist_ok=False);reg=registration(spec);atomic_json(output/'registration.json',reg);prepare(spec,reg,output)
    else:
        reg=configuration(output/'registration.json')
        for path,digest in {**reg['inputs'],**reg['sources']}.items():
            if file_sha256(ROOT/path)!=digest:raise ValueError('registered input/source changed: '+path)
        if reg['config']!=spec:raise ValueError('registered config changed')
        if args.stage=='train':
            if args.width is None:parser.error('--width required')
            train(spec,reg,output,args.width)
        else:audit(spec,reg,output)


if __name__=='__main__':main()
