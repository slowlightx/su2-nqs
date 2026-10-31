"""Bounded COMPLETE-A1 refinement of existing CT root coordinates only.

Fixed lower tensors, no new head/width, no raw-H optimization substituted
for the projected physical objective. The initial IID proposal is a classical
leaf-product/image mixture; the registered fallback samples an unnormalized
A1-root-trace guide by finite-burn MH. Neither is the parent Born density.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_clebsch_active_layout import repack_active_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_clebsch_gauge import triangular_bank_reduced_tree
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.su2.reduced_tree_environment import reduced_tree_environments
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.conditioned_patch_proposal import physical_leaf_diagonals, make_conditioned_patch_proposal
from su2_nqs.vmc.dual_sampled_sr import importance_overlap_diagnostics
from su2_nqs.vmc.exchange_bridge_importance import importance_audit
from su2_nqs.vmc.projected_spin_linear import make_terminal_tensor_products
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action
from .triangle_hierarchy_audit import checked_active_widths, finite_relative_error
from .triangle_dictionary_vmc_local import make_record_sampler


def terminal_union_leaf_diagonals(state, features) -> tuple[np.ndarray, dict]:
    """Exact raw I/P root-ensemble marginals for canonical child bases.

    The P orthonormal S=0 columns are labelled(q,a_left,a_right), counted
    ONCE each; no extra q+1 weight. Magnetic factors enter complementary
    environments as d_other/[P(q+1)]. This is a positive mixed-state guiding
    ensemble, NOT a coherent superposition or a change to the target state.
    """
    # Reuse complete fixed-m feature/Gram and positivity validation.
    physical_leaf_diagonals(state, features)
    env = reduced_tree_environments(state); up = env.upward
    root = state.tree.root
    if root not in state.tree.internal_nodes:
        raise ValueError("a nontrivial binary singlet root is required")
    left, right = root.left, root.right
    charges = sorted(set(up[left]) & set(up[right]))
    if set(state.kernels[-1]) != {(q, q, 0) for q in charges}:
        raise ValueError("complete singlet root buckets required")
    defect = max(float(np.max(abs(g-np.eye(len(g)))))
                 for node, rows in up.items() if node != root for g in rows.values())
    if defect > 2e-10:
        raise ValueError("root-union identity requires canonical orthonormal child spaces")
    dimension = sum(len(up[left][q])*len(up[right][q]) for q in charges)
    down = {node: {q: np.zeros_like(g, dtype=np.complex128) for q, g in row.items()}
            for node, row in up.items()}
    for q in charges:
        dl, dr = len(up[left][q]), len(up[right][q])
        down[left][q] = np.eye(dl)*dr/(dimension*(q+1))
        down[right][q] = np.eye(dr)*dl/(dimension*(q+1))
    for node, kernels in reversed(tuple(zip(state.tree.internal_nodes[:-1], state.kernels[:-1], strict=True))):
        for (ql, qr, qt), tensor in kernels.items():
            t = np.asarray(tensor, complex); parent = down[node][qt]
            a = np.einsum("alr,ab,buv,vr->lu", t, parent, t.conj(), up[node.right][qr], optimize=True)
            b = np.einsum("alr,ab,buv,ul->rv", t, parent, t.conj(), up[node.left][ql], optimize=True)
            down[node.left][ql] += (qt+1)/(ql+1)*a
            down[node.right][qr] += (qt+1)/(qr+1)*b
    edge_error = max(abs(sum((q+1)*np.trace(up[node][q] @ e).real for q,e in rows.items())-1)
                     for node, rows in down.items() if node != root)
    diagonals = []
    for leaf, block in zip(state.tree.leaves, features, strict=True):
        value = sum(np.einsum("xa,ab,xb->x", np.asarray(w), down[leaf][q], np.asarray(w).conj(), optimize=True)
                    for q, w in block.items())
        if abs(value.imag).max() > 1e-11 or value.real.min() < -1e-11 or abs(value.real.sum()-1) > 1e-10:
            raise ArithmeticError("mixed-root physical marginal is not normalized positive")
        row = np.maximum(value.real, 0.); diagonals.append(row/row.sum())
    if edge_error > 2e-10:
        raise ArithmeticError("mixed root complementary edge traces disagree")
    return np.asarray(diagonals), dict(root_coordinates=dimension,
        maximum_nonroot_Gram_error=defect, maximum_mixed_edge_norm_error=float(edge_error),
        root_q_probability={q:len(up[left][q])*len(up[right][q])/dimension for q in charges},
        root_identity_is_in_active_physical_basis=True, extra_magnetic_degeneracy_weight=False,
        guiding_ensemble_is_coherent_A1=False)


def cost_selected_fit_count(warm_seconds: float, remaining_seconds: float, *, audit_count: int = 64) -> int:
    """Cost-only fit size; never consult energies, gradients or readout quality."""
    if not np.isfinite(warm_seconds) or warm_seconds <= 0 or not np.isfinite(remaining_seconds) or remaining_seconds <= 0:
        raise ValueError("finite measured positive timing and remaining budget required")
    # Keep at least a third of the remaining wall budget for the untouched64.
    if 1.5*audit_count*warm_seconds > remaining_seconds/1.5:
        return 0
    affordable = int(remaining_seconds/(1.5*warm_seconds))-audit_count
    return 32 if affordable >= 32 else 0


def norm_weight_diagnostics(phi: np.ndarray, coefficients: np.ndarray, logq: np.ndarray) -> dict:
    """Parent/union common-Q concentration, with numerical rescaling only."""
    phi = np.asarray(phi, complex); c = np.asarray(coefficients, float); logq = np.asarray(logq, float)
    if phi.ndim != 2 or phi.shape[1] != len(c) or logq.shape != (len(phi),) or not np.isfinite(phi).all() or not np.isfinite(logq).all():
        raise ValueError("finite matching root features and logQ required")
    result = {}
    for name, density in (("parent", abs(phi @ c)**2), ("root_union", np.sum(abs(phi)**2, axis=1)/phi.shape[1])):
        valid = density > 0
        if not valid.any():
            result[name] = dict(effective_contributions=0., maximum_fraction=1., log_norm_estimate=None)
            continue
        logs = np.full(len(phi), -np.inf); logs[valid] = np.log(density[valid])-logq[valid]
        shift = logs.max(); w = np.exp(logs-shift); total = w.sum()
        result[name] = dict(effective_contributions=float(total**2/(w @ w)),
            maximum_fraction=float(w.max()/total), log_norm_estimate=float(shift+np.log(total/len(w))))
    return result


def paired_audit_difference(phi, hphi, logq, parent, candidate, groups, *, n_sites=144) -> dict:
    """Paired energy/positive residual-variance errors on SAME fresh groups."""
    phi,hphi=np.asarray(phi),np.asarray(hphi); q=np.exp(np.asarray(logq)); groups=np.asarray(groups)
    def moments(indices,c):
        p=phi[indices]@c; hp=hphi[indices]@c; w=q[indices]
        norm=np.sum(abs(p)**2/w)
        e=np.sum(p.conj()*hp/w)/norm
        v=np.sum(abs(hp-e*p)**2/w)/norm
        return np.asarray([e.real/n_sites,v/n_sites])
    ids=np.arange(len(phi)); labels=np.unique(groups)
    def delta(indices):
        return moments(indices,candidate)-moments(indices,parent)
    point=delta(ids); leave=np.asarray([delta(ids[groups!=g]) for g in labels])
    se=np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(0))**2,axis=0))
    rng=np.random.default_rng(2610234106)
    boot=np.asarray([delta(np.concatenate([ids[groups==g] for g in rng.choice(labels,len(labels),replace=True)]))
                     for _ in range(300)])
    return dict(optimized_minus_parent=dict(energy_per_site=float(point[0]),variance_per_site=float(point[1])),
        paired_group_jackknife_SE=dict(energy_per_site=float(se[0]),variance_per_site=float(se[1])),
        paired_group_bootstrap_95=np.percentile(boot,[2.5,97.5],axis=0).T.tolist(),
        same_groups_for_both_endpoints=True,audit_used_for_selection=False)


def finite_chain_rhat(values: np.ndarray) -> dict:
    """Unsplit/split variance diagnostics only, never an equilibrium proof."""
    x=np.asarray(values,float)
    if x.ndim!=2 or min(x.shape)<4 or not np.isfinite(x).all():
        raise ValueError('finite[time>=4,chain>=4] observable required')
    def rhat(y):
        n=len(y);within=np.var(y,axis=0,ddof=1).mean();between=n*np.var(y.mean(0),ddof=1)
        return float(np.sqrt(((n-1)*within/n+between/n)/within)) if within>0 else None
    half=len(x)//2
    return dict(unsplit=rhat(x),split=rhat(np.concatenate([x[:half],x[-half:]],axis=1)),
                finite_burn_equilibrium_certified=False)


def run(spec: dict, output: Path) -> dict:
    """One root-only bounded update with disjoint fit, distance and audit data."""
    # Generic optimizer is imported only when an actual workflow starts.
    from su2_nqs.vmc.projected_spin_linear import fit_parent_anchored_linear_update
    jax.config.update("jax_enable_x64", True)
    if spec.get('j2_over_j1')!=.125 or spec.get('model')!={'length':12,'multiplicity_cap':20,'maximum_q':10}:
        raise ValueError('this frozen development workflow only permits J2=.125/L12/w20/qmax10')
    guiding_mode=spec.get('guiding_mode','IID')
    if guiding_mode not in ('IID','root_union_MH'):
        raise ValueError('guiding mode must be IID or the one preregistered root_union_MH fallback')
    reuse_fit='frozen_fit_npz' in spec
    if reuse_fit:
        if guiding_mode!='root_union_MH':
            raise ValueError('frozen-fit calibration continuation uses the same root-union MH guide')
        for field in ('frozen_fit_npz','frozen_fit_report'):
            if file_sha256(Path(spec[field]))!=spec[field+'_sha256']:
                raise ValueError('immutable fitted direction input changed: '+field)
    checkpoint = Path(spec["checkpoint"])
    if spec["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("frozen candidate checkpoint SHA changed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a fresh immutable refinement output")
    output.mkdir(parents=True, exist_ok=True)
    cap = float(spec.get("seconds_cap", 1800))
    previous_seconds=float(spec.get('previous_numerical_seconds',0))
    if not 120<=cap<=1800 or previous_seconds<0 or cap+previous_seconds>1800+1e-8:
        raise ValueError('primary plus fixed fallback numerical wall caps must not exceed1800s')
    cfg = MixedClebschConfig(12, 20, 10)
    saved = load_checkpoint(checkpoint); native = saved["params"]; metadata = saved["metadata"]
    active = checked_active_widths(metadata["embedding"]["active_top_widths"], nodes=15)
    if any(row != {q:8 for q in range(0,11,2)} for row in active[:-1]) or active[-1] != {0:1}:
        raise ValueError("fixed tau8 complete active closure required")
    execution = repack_active_clebsch_bank(native, cfg, active)
    registration = dict(spec=spec, workflow_sha256=file_sha256(Path(__file__)),
        source_params_sha256=pytree_sha256(native), complete_A1_order=1728,
        fixed_lower_tensors=True, root_coordinates=384, norm_pilot=64,
        fit_samples="reused exact previous32; ZEROrefit" if reuse_fit else "32if measuredcostaffords64fresh audit, otherwise0",
        calibration_samples=64 if reuse_fit else 16, untouched_audit_samples=64,
        seed_streams=dict(probe=2610234300 if reuse_fit else 2610234100,pilot=2610234101,fit=2610234102,
            calibration=2610234303 if reuse_fit else 2610234103,audit=2610234304 if reuse_fit else 2610234104),
        frozen_fit_reused_without_direction_damping_or_step_change=reuse_fit,
        serial_calibration_coverage_repair_not_independent_confirmation=reuse_fit,
        maximum_data_resolved_rank=8, calibration_selection="FSonly; no calibrationenergyselection",
        maximum_FS_sine=.03, initial_IID_uniform_mixture_fraction=.05,
        MH_floor_coefficient=.05/1728 if guiding_mode=='root_union_MH' else None,
        no_width_head_or_attention=True,
        guiding_mode=guiding_mode, independent_IID_full_support_Q=guiding_mode=='IID', Q_is_coherent_A1_Born=False,
        whole_wall_seconds=cap, previous_numerical_seconds=previous_seconds, combined_numerical_cap=1800,
        first_H_columns_cost_cap_seconds=120,
        fallback_if_IID_pilot_fails=dict(kind='existing exchange MH with actual A1 union-root trace guide',
            unnormalized_density='sum(abs(Phi_A1)**2)/384 + (.05/1728)*uniform_fixed_Sz_probability',
            fit_calibration_audit_chains='8 separately initialized chains for EACH split, disjointRNG',
            burn_proposals=64,between_proposals=8,
            records=dict(fit=0 if reuse_fit else 4,calibration=8 if reuse_fit else 2,audit=8),
            unknown_Z_cancels_in_ratio_moments=True,finite_burn_not_IID=True,
            equilibrium_or_node_tail_coverage_not_certified=True,
            gate='measure cheap guide-amplitude MH cost first; proceed only withinremaining1800s'),
        dependency_sha256={p:file_sha256(Path(p)) for p in (
            'su2_nqs/vmc/projected_spin_linear.py','su2_nqs/vmc/conditioned_patch_proposal.py',
            'su2_nqs/vmc/triangular_clebsch_cached_action.py','su2_nqs/models/triangular_clebsch_active_layout.py')},
        full_global_H_variance_from_HPcolumn_residual_not_compressed_H_squared=True)
    atomic_json(output/"registration.json", registration)
    started = perf_counter(); result = dict(status="started", timings={}, work=dict(amplitude_column_centers=0, H_column_centers=0,scalar_H_centers=0,guide_amplitude_centers=0),
        work_counts_are_completed_centers_not_interrupted_kernel_work=True,energy_estimate=False)
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("fixed1800s root refinement budget")
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, cap)
    def timed(name, callback, *args):
        if perf_counter()-started >= cap:
            raise TimeoutError("whole wall cap before "+name)
        atomic_json(output/"progress.json", dict(status=result["status"], operation=name, seconds=perf_counter()-started))
        tick = perf_counter(); value = callback(*args); jax.block_until_ready(value)
        result["timings"][name] = perf_counter()-tick
        return value
    try:
        images = np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))), np.int32)
        bonds = triangular_j1j2_bonds(12,12,j2=.125,periodic=True)
        prepared = prepare_feature_lut_bank(execution,cfg)
        amplitude = jax.jit(jax.vmap(make_feature_lut_projected_amplitude(cfg,images,chunk=4,active_top_widths=active),in_axes=(None,0)))
        # Dummy second output makes an amplitude-only derivative, no H query.
        amp_action = lambda b,x:(amplitude(b,x),jnp.zeros(len(x),jnp.complex128))
        coordinates, replace, phi_only = make_terminal_tensor_products(amp_action,prepared)
        if len(coordinates) != 384:
            raise ArithmeticError("existing active root is not384coordinates")
        state = triangular_bank_reduced_tree(native,cfg)
        # Parent validation uses native width20. Mixed-root guiding must instead
        # see ACTIVE root spaces: reconstruct generic tree from active maps.
        from su2_nqs.su2.reduced_tree_environment import ReducedSingletTree
        tree = state.tree
        # Reduced kernels are native padded. Slice all active multiplicity axes
        # with the same q/width metadata already validated by exact repacking.
        from su2_nqs.models.triangular_clebsch_gauge import _nodes
        _, nodes = _nodes(cfg)
        from su2_nqs.models.triangular_mixed_clebsch import mixed_clebsch_layout
        rows = mixed_clebsch_layout(cfg); by_node = dict(zip(tree.internal_nodes,state.kernels,strict=True))
        active_kernels = {}
        for i,row in enumerate(rows):
            target = {}
            for (ql,qr,qt),w in by_node[nodes[16+i]].items():
                dl = w.shape[1] if row["left"] < 16 else active[row["left"]-16].get(ql,0)
                dr = w.shape[2] if row["right"] < 16 else active[row["right"]-16].get(qr,0)
                do = active[i].get(qt,0)
                if dl and dr and do:
                    target[ql,qr,qt] = w[:do,:dl,:dr]
            active_kernels[nodes[16+i]] = target
        active_state = ReducedSingletTree(tree,state.leaf_grams,tuple(active_kernels[n] for n in tree.internal_nodes),np.ones(1))
        features = tuple({q:np.asarray(x) for q,x in zip(cfg.local_q,row,strict=True)} for row in prepared["local_features"])
        diagonals, union = terminal_union_leaf_diagonals(active_state,features)
        if union["root_coordinates"] != 384:
            raise ArithmeticError("mixed guiding included padded root coordinates")
        patches = np.asarray([b.sites for b in triangular_schur_hierarchy(12).blocks],np.int32)
        logQ,draw,qmetadata = make_conditioned_patch_proposal(patches,diagonals,images,n_up=72,uniform_fraction=.05,image_batch_size=4)
        atomic_json(output/"guiding_proposal.json",dict(qmetadata,union=union))
        np.savez_compressed(output/"guiding_proposal.npz",patches=patches,diagonals=diagonals,images=images)
        sampling_info={};sampling_forecast=0.
        if guiding_mode=='root_union_MH':
            from math import lgamma
            log_uniform=-(lgamma(145)-2*lgamma(73))
            floor_mass=.05/1728
            @jax.jit
            def guide_amplitude(b,row):
                feature,_=phi_only(b,row[None])
                return jnp.sqrt(jnp.sum(abs(feature[0])**2)/384+floor_mass*jnp.exp(log_uniform))
            guide_batch=jax.jit(jax.vmap(guide_amplitude,in_axes=(None,0)))
            short_sampler=make_record_sampler(guide_amplitude,records=1,burn_proposals=1,proposals_between=1,
                exchange_edges=tuple((a,b) for a,b,_ in bonds))
            init=draw(jax.random.PRNGKey(2610234190),8)
            timed('MH_guide_cost_first',short_sampler,prepared,jax.random.PRNGKey(2610234191),init)
            timed('MH_guide_cost_warm',short_sampler,prepared,jax.random.PRNGKey(2610234192),init)
            result['work']['guide_amplitude_centers']+=48
            per_batch=result['timings']['MH_guide_cost_warm']/3
            sampling_forecast=1.5*(3*64+(4+2+8)*8+3)*per_batch+45
            atomic_json(output/'MH_guide_cost.json',dict(guide='.mean(abs(Phi_A1)**2)+(.05/1728)*U_Sz',
                floor_coefficient=floor_mass,floor_is_not_a_known_mixture_probability=True,log_uniform_probability=log_uniform,
                measured_8_chain_proposal_batch_seconds=per_batch,sampling_forecast_seconds=sampling_forecast,
                guide_normalization_unknown=True,finite_burn_not_IID=True))
            if sampling_forecast>=.6*(cap-(perf_counter()-started)):
                result.update(status='fixed_MH_guide_cost_leaves_insufficient_fullH_budget');return result
        def draw_split(name,count):
            seed = registration["seed_streams"][name]
            if guiding_mode=='root_union_MH':
                if count%8:
                    raise ValueError('whole8chain record batches required')
                initial_key,chain_key=jax.random.split(jax.random.PRNGKey(seed))
                initial=draw(initial_key,8)
                sampler=make_record_sampler(guide_amplitude,records=count//8,burn_proposals=64,
                    proposals_between=8,exchange_edges=tuple((a,b) for a,b,_ in bonds))
                last,key,recorded,info=timed(name+'_MH',sampler,prepared,chain_key,initial)
                rows=np.asarray(recorded).reshape(count,144)
                g=np.asarray(timed(name+'_guide_density',guide_batch,prepared,jnp.asarray(rows)))
                logs=2*np.log(g)
                sampling_info[name]=dict(info=jax.tree.map(lambda x:np.asarray(x).tolist(),info),
                    next_RNG=np.asarray(key).tolist(),last_states=np.asarray(last).tolist(),
                    count=count,records=count//8,chains=8,burn=64,between=8)
                result['work']['guide_amplitude_centers']+=8*(64+(count//8)*8+1)+count
                atomic_json(output/(name+'_MH.json'),sampling_info[name])
                return rows,logs
            rows = timed(name+"_draw",lambda k:draw(k,count),jax.random.PRNGKey(seed))
            logs = np.asarray(timed(name+"_Q",logQ,rows))
            return np.asarray(rows),logs
        def query_rows(name,rows,callback,H):
            a=[];b=[]
            for i in range(len(rows)):
                p,h = timed(name+"_"+str(i),callback,prepared,jnp.asarray(rows[i:i+1]))
                a.append(np.asarray(p)[0]);b.append(np.asarray(h)[0])
                result["work"]["H_column_centers" if H else "amplitude_column_centers"] += 1
            return np.asarray(a),np.asarray(b)
        if guiding_mode=='IID':
            pilot_rows,pilot_logq = draw_split("pilot",64)
            pilot_phi,_ = query_rows("pilot_Phi",pilot_rows,phi_only,False)
            pilot = norm_weight_diagnostics(pilot_phi,np.asarray(coordinates),pilot_logq)
            atomic_json(output/"norm_pilot.json",pilot)
            np.savez_compressed(output/"norm_pilot.npz",rows=pilot_rows,Phi=pilot_phi,logQ=pilot_logq)
            if any(x["effective_contributions"] < 8 or x["maximum_fraction"] > .5 for x in pilot.values()):
                result.update(status="IID_guiding_pilot_failed_before_H",norm_pilot=pilot)
                return result
        else:
            pilot_path=Path(spec['previous_pilot_results'])
            if file_sha256(pilot_path)!=spec['previous_pilot_results_sha256']:
                raise ValueError('frozen IID failure provenance changed')
            import json
            pilot=json.loads(pilot_path.read_text())['norm_pilot']
            atomic_json(output/'reused_failed_IID_pilot.json',dict(pilot=pilot,used_for_fit=False,
                used_for_fallback_parameter_selection=False,serial_development_not_independent_confirmation=True))
        action = make_cached_projected_heisenberg_action(cfg,images,bonds,projection_chunk=4,active_top_widths=active)
        _,replace,Hcolumns = make_terminal_tensor_products(action,prepared)
        # Cost probes are NOT fit/calibration/audit observations.
        probe_rows=np.asarray(draw(jax.random.PRNGKey(registration['seed_streams']['probe']),1))
        if reuse_fit:
            import json
            fit_report=json.loads(Path(spec['frozen_fit_report']).read_text())
            frozen=np.load(spec['frozen_fit_npz'])
            if not np.array_equal(np.asarray(coordinates),frozen['parent']):
                raise ArithmeticError('frozen fit belongs to different physical parent root')
            candidate=np.asarray(frozen['proposed']).copy()
            fit_count=len(frozen['rows']);second=None
            atomic_json(output/'frozen_fit_provenance.json',dict(
                inputs={k:spec[k] for k in ('frozen_fit_npz','frozen_fit_report')},
                input_hashes={k:spec[k+'_sha256'] for k in ('frozen_fit_npz','frozen_fit_report')},
                new_fit_H_queries=0,refit_performed=False,damping_and_direction_unchanged=True))
            atomic_json(output/'fit.json',fit_report)
        else:
            cost_start=perf_counter()
            signal.setitimer(signal.ITIMER_REAL,min(120.,cap-(perf_counter()-started)))
            try:
                first=timed("full_H_columns_first",Hcolumns,prepared,jnp.asarray(probe_rows))
                second=timed("full_H_columns_warm",Hcolumns,prepared,jnp.asarray(probe_rows))
            finally:
                signal.setitimer(signal.ITIMER_REAL,max(.01,cap-(perf_counter()-started)))
            result["work"]["H_column_centers"]+=2
            if perf_counter()-cost_start > 120:
                result.update(status="H_column_cost_preflight_exceeded120s")
                return result
            np.testing.assert_allclose(first,second,rtol=2e-12,atol=1e-100)
            direct=timed("direct_parent_full_H",action,prepared,jnp.asarray(probe_rows))
            result['work']['scalar_H_centers']+=1
            replay=finite_relative_error(np.asarray([second[0]@coordinates,second[1]@coordinates]),np.asarray(direct))
            if replay>2e-10:
                raise ArithmeticError("full A1 root columns failed parent H replay")
            fit_count=cost_selected_fit_count(result["timings"]["full_H_columns_warm"],cap-(perf_counter()-started)-sampling_forecast)
            atomic_json(output/"cost_selection.json",dict(fit_count=fit_count,calibration_count=16,audit_count=64,
                full_H_columns_warm_seconds=result["timings"]["full_H_columns_warm"],root_replay_error=replay,
                selection_used_energy=False))
            if not fit_count:
                result.update(status="insufficient_measured_budget_for_fit_and64fresh_audit")
                return result
            fit_rows,fit_logq=draw_split("fit",fit_count)
            fit_phi,fit_hphi=query_rows("fit_Hcolumns",fit_rows,Hcolumns,True)
            fit_coverage=norm_weight_diagnostics(fit_phi,np.asarray(coordinates),fit_logq)
            rank=min(8,int(fit_coverage['parent']['effective_contributions']//4))
            atomic_json(output/'fit_coverage.json',dict(fit_coverage,selected_rank=rank,selection_uses_energy=False))
            if rank<1:
                result.update(status='fit_norm_support_too_concentrated_for_one_resolved_direction')
                return result
            candidate,fit_report=fit_parent_anchored_linear_update(fit_phi,fit_hphi,np.exp(fit_logq),np.asarray(coordinates),
                damping=.1,maximum_fs_sine=.03,maximum_tangent_rank=rank)
            atomic_json(output/"fit.json",fit_report)
            np.savez_compressed(output/"fit.npz",rows=fit_rows,logQ=fit_logq,Phi=fit_phi,HPhi=fit_hphi,parent=np.asarray(coordinates),proposed=candidate)
        if not fit_report['accepted'] or np.linalg.norm(candidate-np.asarray(coordinates))<=1e-14:
            result.update(status='no_nonzero_fit_update_accepted',optimizer_updates=0,fit=fit_report)
            return result
        cal_count=registration['calibration_samples']
        cal_rows,cal_logq=draw_split("calibration",cal_count)
        cal_phi,_=query_rows("calibration_Phi",cal_rows,phi_only,False)
        old=np.asarray(coordinates); cal_old=cal_phi@old
        selected=None; calibration=[]
        for k in range(11):
            fraction=2.**(-k); values=old+fraction*(candidate-old)
            d=importance_overlap_diagnostics(cal_old,cal_phi@values,np.exp(cal_logq))
            calibration.append(dict(fraction=fraction,overlap=d))
            if d["sampled_fs_sine"]<=.03 and min(d["first"]["effective_contributions"],d["second"]["effective_contributions"])>=4:
                selected=values;break
        atomic_json(output/"calibration.json",dict(attempts=calibration,energy_used_for_selection=False,accepted=selected is not None))
        np.savez_compressed(output/"calibration.npz",rows=cal_rows,logQ=cal_logq,Phi=cal_phi)
        if selected is None:
            result.update(status="no_cross_sample_physical_distance_calibrated_update")
            return result
        # Repack only the EXISTING active root entries into the padded bank.
        updated_execution=replace(execution,jnp.asarray(selected))
        updated_native=jax.tree.map(lambda x:np.asarray(x).copy(),native)
        for full,small in zip(updated_native["top"][-1],updated_execution["top"][-1],strict=True):
            full.fill(0.)
            full[tuple(slice(0,n) for n in small.shape)]=np.asarray(small)
        if any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(native['local']),jax.tree.leaves(updated_native['local']),strict=True)):
            raise ArithmeticError('root-only update changed local tensors')
        if any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(native['top'][:-1]),jax.tree.leaves(updated_native['top'][:-1]),strict=True)):
            raise ArithmeticError('root-only update changed one of fourteen frozen lower nodes')
        optimizer_state=dict(direction=fit_report['direction'],fit_step=fit_report['selected_step'],
            calibration_fraction=calibration[-1]['fraction'],parent_coordinates=old,
            optimized_coordinates=selected,optimizer_updates=1,stage='frozen_before_untouched_audit',
            seed_streams=registration['seed_streams'],draw_counts=dict(probe=1,pilot=0,fit=0 if reuse_fit else fit_count,calibration=cal_count,audit=0),
            prior_IID_pilot64_reused_by_provenance=True,frozen_fit_inputs=spec.get('frozen_fit_npz'))
        optimizer_state['sampler_states_and_next_rng']=sampling_info.copy()
        optimizer_state['guiding_mode']=guiding_mode
        save_checkpoint(output/"optimized_bank.pkl",updated_native,optimizer_state=optimizer_state,
            metadata=dict(model=spec['model'],j2=.125,embedding=metadata['embedding'],parent_metadata=metadata,
                energy_and_variance_audit_pending=True,root_refinement=dict(source_checkpoint_sha256=spec["checkpoint_sha256"],
                complete_A1=True,fixed_lower_tensors=True,no_new_width_head=True,fit_report=fit_report,
                calibration_fraction=calibration[-1]["fraction"],fresh_audit_not_used_for_selection=True),optimizer_reset_required=True))
        restored=load_checkpoint(output/'optimized_bank.pkl')
        if pytree_sha256(restored['params'])!=pytree_sha256(updated_native):
            raise ArithmeticError('restored optimized parameters changed')
        if (pytree_sha256(restored['optimizer_state'])!=pytree_sha256(optimizer_state)
                or restored['optimizer_state']['seed_streams']!=optimizer_state['seed_streams']
                or restored['optimizer_state']['draw_counts']!=optimizer_state['draw_counts']):
            raise ArithmeticError('restored optimizer/RNG draw state changed')
        restored_active=repack_active_clebsch_bank(restored['params'],cfg,active)
        restored_prepared=prepare_feature_lut_bank(restored_active,cfg)
        if second is None:
            # This is a replay, AFTER the fitted root and calibration are frozen;
            # no calibration energy is evaluated or used to choose the step.
            signal.setitimer(signal.ITIMER_REAL,min(120.,cap-(perf_counter()-started)))
            try:
                second=timed('frozen_fit_full_H_column_replay',Hcolumns,prepared,jnp.asarray(probe_rows))
                result['work']['H_column_centers']+=1
            finally:
                signal.setitimer(signal.ITIMER_REAL,max(.01,cap-(perf_counter()-started)))
        native_amplitude=jax.jit(make_feature_lut_projected_amplitude(cfg,images,chunk=4))
        native_value=timed('restored_native_A1_amplitude',native_amplitude,prepare_feature_lut_bank(restored['params'],cfg),jnp.asarray(probe_rows[0]))
        active_value=second[0]@selected
        if finite_relative_error(np.asarray(native_value).reshape(1),np.asarray(active_value).reshape(1))>2e-10:
            raise ArithmeticError('saved native root packing changed completeA1 amplitude')
        restored_values=timed('restored_active_full_H',action,restored_prepared,jnp.asarray(probe_rows))
        result['work']['scalar_H_centers']+=1
        saved_replay_error=finite_relative_error(np.asarray(restored_values),np.asarray([second[0]@selected,second[1]@selected]))
        if saved_replay_error>2e-10:
            raise ArithmeticError('saved restored root does not replay completeA1/fullH columns')
        atomic_json(output/'restored_checkpoint_replay.json',dict(complete_A1_amplitude_native_replay=True,
            complete_full_H_restored_active_replay_relative_error=saved_replay_error,
            root_padding_checked_by_exact_active_repack=True,optimizer_and_RNG_streams_restored=True))
        audit_rows,audit_logq=draw_split("audit",64)
        audit_phi,audit_hphi=query_rows("untouched_audit_Hcolumns",audit_rows,Hcolumns,True)
        groups=np.repeat(np.arange(8),8) if guiding_mode=='IID' else np.tile(np.arange(8),8)
        reports={}
        for name,c in (("parent",old),("optimized",selected)):
            reports[name]=importance_audit(audit_phi@c,audit_hphi@c,np.exp(audit_logq),groups,
                n_sites=144,bootstrap_replicates=300,seed=2610234105)
            if guiding_mode=='root_union_MH':
                report=reports[name]
                report['relative_norm_against_unnormalized_guide']=report.pop('norm_estimate')
                report.pop('norm_iid_se',None)
                for metric in report['metrics'].values():
                    metric.pop('iid_delta_se',None)
                report.update(IID_standard_errors_suppressed=True,
                    guide_normalization_unknown=True,absolute_physical_norm_estimated=False,
                    finite_burn_stationarity_not_certified=True,
                    effective_contributions_do_not_include_temporal_autocorrelation=True)
        np.savez_compressed(output/"untouched_audit.npz",rows=audit_rows,logQ=audit_logq,Phi=audit_phi,HPhi=audit_hphi,
            parent=old,optimized=selected,groups=groups)
        atomic_json(output/"untouched_audit.json",reports)
        if guiding_mode=='root_union_MH':
            audit_q=np.exp(audit_logq); parent_norm=abs(audit_phi@old)**2/audit_q
            candidate_norm=abs(audit_phi@selected)**2/audit_q
            rhat=dict(log_guide=finite_chain_rhat(audit_logq.reshape(8,8)),
                parent_norm=finite_chain_rhat(parent_norm.reshape(8,8)),
                candidate_norm=finite_chain_rhat(candidate_norm.reshape(8,8)),
                IID_standard_errors_invalid_for_MH=True,whole_chain_groups_used_for_jackknife_and_bootstrap=True,
                unknown_normalization_cancels_only_in_ratio_moments=True)
            atomic_json(output/'MH_audit_chain_diagnostics.json',rhat)
            result['MH_chain_diagnostics']=rhat
        paired=paired_audit_difference(audit_phi,audit_hphi,audit_logq,old,selected,groups)
        atomic_json(output/'paired_untouched_audit.json',paired)
        result.update(status="actual_root_update_and_fresh_development_audit_complete",energy_estimate=True,
            norm_pilot=pilot,fit=fit_report,audit=reports,paired_audit=paired,optimizer_updates=1,
            publication_quality_gate_passed=False,node_complete_H2_in_expectation=True,
            finite_burn_not_IID=guiding_mode=='root_union_MH',guiding_normalization_known=guiding_mode=='IID',
            all_root_stationarity_claimed=False,unseen_tail_coverage_certified=False)
        return result
    except TimeoutError as error:
        result.update(status="bounded_incomplete",error=str(error));return result
    except Exception as error:
        result.update(status='failed_validation',error=repr(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result["total_seconds"]=perf_counter()-started
        atomic_json(output/"results.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import json
    run(json.loads(args.plan.read_text()),args.output)
