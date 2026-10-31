"""Frozen local attention/control experiment for CT6 -> L12/18/24 initialization.

Learn scalar, gauge-covariant boundary-response filters in legal q blocks,
not arbitrary eigenvectors or the already cheap raw density environment.
Only actual CT6 child operators enter training. All larger target hierarchies
are built after freezing every model; no target energy selects parameters.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter, process_time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.operator_filter_learning import (
    apply_filter_controller, build_boundary_residual_sketch, filtered_columns,
    filter_trace_energy, init_filter_controller, occupied_seed_rotations, sketch_ritz_columns,
)
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude, prepare_feature_lut_bank
from su2_nqs.operators.su2_sector_merge import SectorMergeOperator, project_sector_maps, solve_sector_merge
from su2_nqs.operators.su2_conditional_response import root_coefficient_matrices
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_hierarchy_initialization import (
    _block_hash, _leaf_block, _node_timeout, _real_matrix, active_amplitude,
    active_bank_from_state, active_state_environment, native_bank_from_state,
    retained_widths, source_channels, split_bond_ledger,
)
from .triangle_sector_transfer import extract_merge_map, load_source

ROOT = Path(__file__).resolve().parents[3]
SEED = 2610269801
Q = (0, 2, 4, 6, 8, 10)
ARMS = ('copy', 'constant', 'linear', 'mlp', 'attention', 'ritz', 'physical')
TRAIN_SCALES = tuple((a, b) for a in (.75, 1., 1.25) for b in (.5, 1., 1.5))
HELD_SCALES = ((.6, 1.75), (1.4, .25))
_SOURCE_SEEDS = None


def learned_source_seeds(source: dict) -> tuple[dict, dict]:
    """Occupation-ranked CT6 donor columns; source identity remains immutable."""
    global _SOURCE_SEEDS
    if _SOURCE_SEEDS is not None and _SOURCE_SEEDS[0] is source:
        return _SOURCE_SEEDS[1:]
    cfg = MixedClebschConfig(6,20,10); full = []
    for node, pair in enumerate(((0,1),(2,3))):
        widths = [{q:w.shape[1] for q,w in zip((1,3,5,7,9),source['local'][i],strict=True)} for i in pair]
        full.append({q:extract_merge_map(source,cfg,node,q,source_channels(*widths,q)) for q in Q})
    roots = source_channels({q:w.shape[1] for q,w in full[0].items()}, {q:w.shape[1] for q,w in full[1].items()},0)
    coefficients = root_coefficient_matrices(extract_merge_map(source,cfg,2,0,roots),roots)
    original_norm = sum(float(np.vdot(c,c).real) for c in coefficients.values())
    if not np.isfinite(original_norm) or original_norm <= 0: raise ArithmeticError('source root has invalid raw norm')
    from su2_nqs.models.triangular_clebsch_gauge import triangular_bank_reduced_tree
    from su2_nqs.su2.reduced_tree_environment import reduced_tree_environments
    env_norm = reduced_tree_environments(triangular_bank_reduced_tree(source,cfg)).norm_squared
    if abs(env_norm/original_norm-1)>2e-10: raise ArithmeticError('root coefficients disagree with raw tree norm')
    # Canonicalization preserves the original amplitude scale. Normalize only
    # this density analysis, never mutate the stored donor wavefunction.
    coefficients = {q:c/np.sqrt(original_norm) for q,c in coefficients.items()}
    frames = {0:{},1:{}}; notes = {}
    for q in Q:
        left,right,notes[q] = occupied_seed_rotations(coefficients[q],4)
        frames[0][q],frames[1][q] = _real_matrix(full[0][q] @ left),_real_matrix(full[1][q] @ right)
    norm = sum(x['sector_weight'] for x in notes.values())
    if abs(norm-1)>2e-10: raise ArithmeticError('CT6 singlet-channel occupation does not sum to raw norm one')
    report = dict(by_q=notes, normalized_singlet_channel_mass=norm,
        original_source_raw_norm=original_norm, root_environment_norm_relative_error=abs(env_norm/original_norm-1),
        total_retained_left=sum(x['retained_weight_left'] for x in notes.values()),
        total_retained_right=sum(x['retained_weight_right'] for x in notes.values()),
        transfer_is_truncating_initialization_not_global_state_preservation=True)
    _SOURCE_SEEDS = (source,frames,report)
    return frames,report


def internal_action(left, right, channels: tuple, columns: np.ndarray) -> np.ndarray:
    """Apply child internal H in exactly the declared reduced channel order."""
    result = np.zeros_like(columns, dtype=np.result_type(columns, complex))
    for ql, qr in sorted({(x[0], x[2]) for x in channels}):
        ids = [i for i, x in enumerate(channels) if (x[0], x[2]) == (ql, qr)]
        tensor = columns[ids].reshape(left.q_widths[ql], right.q_widths[qr], -1)
        result[ids] = (np.einsum('ab,bct->act', left.h_by_q[ql], tensor)
            + np.einsum('ab,cbt->cat', right.h_by_q[qr], tensor)).reshape(len(ids), -1)
    return np.real_if_close(result)


def separable_seed(left, right, q: int, width: int) -> tuple[np.ndarray, dict]:
    """Lowest internal-product columns, with no cross H or singlet-root copy.

    At a degenerate cutoff this selects one deterministic stored-gauge frame;
    filter covariance is conditional on transporting that chosen seed.
    """
    channels = source_channels(left.q_widths, right.q_widths, q)
    spectra_l = {j: np.linalg.eigh(h) for j, h in left.h_by_q.items()}
    spectra_r = {j: np.linalg.eigh(h) for j, h in right.h_by_q.items()}
    candidates = []
    for ql, qr in sorted({(x[0], x[2]) for x in channels}):
        el, _ = spectra_l[ql]; er, _ = spectra_r[qr]
        candidates.extend((float(a+b), ql, qr, i, j) for i, a in enumerate(el) for j, b in enumerate(er))
    candidates.sort()
    values = []
    for _, ql, qr, a, b in candidates[:width]:
        col = np.zeros(len(channels), complex)
        vl, vr = spectra_l[ql][1][:, a], spectra_r[qr][1][:, b]
        for i, (jl, al, jr, ar) in enumerate(channels):
            if (jl, jr) == (ql, qr): col[i] = vl[al]*vr[ar]
        values.append(col)
    gap = candidates[width][0]-candidates[width-1][0] if width < len(candidates) else None
    return _real_matrix(np.column_stack(values)), dict(kind='separable_child_H_product',
        cutoff_gap=gap, degenerate_cutoff=gap is not None and gap < 1e-10,
        covariance_conditional_on_transported_seed=True)


def operator_signature(left, right, cross: tuple) -> str:
    """Exact ordered operator/bond identity, not a node index or size label."""
    return hashlib.sha256(repr((_block_hash(left), _block_hash(right), cross)).encode()).hexdigest()


def seed_maps(source: dict, left, right, base_pair: tuple | None) -> tuple[dict, dict]:
    maps, notes = {}, {}
    for q in Q:
        if base_pair is not None:
            if base_pair not in ((0, 1), (2, 3)):
                raise ValueError('learned CT6 copy requires the exact ordered local donor lineage')
            frames, occupation = learned_source_seeds(source)
            maps[q] = frames[base_pair[0]//2][q].copy()
            notes[q] = dict(kind='occupation_ranked_learned_CT6_18site_columns', ordered_base_pair=list(base_pair),
                occupation=occupation['by_q'][q])
        else:
            maps[q], notes[q] = separable_seed(left, right, q, 4)
    return maps, notes


def bundle(left, right, cross: tuple, seeds: dict) -> tuple[dict, np.ndarray, dict]:
    """One q-token bundle from actual full-cross-H operator moments only."""
    extrema = [np.concatenate([np.linalg.eigvalsh(h) for h in b.h_by_q.values()]) for b in (left, right)]
    boundary_bound = .75*sum(abs(g) for _, _, g in cross)
    lower = sum(float(x.min()) for x in extrema)-1e-10
    upper = sum(float(x.max()) for x in extrema)+1e-10
    sketches, tokens, reports = {}, [], {}
    for q in Q:
        factory_start = perf_counter()
        op = SectorMergeOperator(left, right, cross, q, memory_limit_bytes=768*1024**2)
        factory_seconds = perf_counter()-factory_start
        internal = lambda x: internal_action(left, right, op.channels, x[:, None])[:, 0]
        sketch_start = perf_counter()
        sketch = build_boundary_residual_sketch(op.matvec, internal, seeds[q], lower, upper, max(boundary_bound, 1e-10))
        sketch_seconds = perf_counter()-sketch_start
        sketches[q] = sketch
        ritz_start = perf_counter()
        _, ritz = sketch_ritz_columns(sketch)
        ritz_seconds = perf_counter()-ritz_start
        k = seeds[q].shape[1]
        h0 = internal_action(left, right, op.channels, seeds[q])
        norms = [float(np.trace(sketch.gram[i*k:(i+1)*k, i*k:(i+1)*k]).real/k) for i in range(1, 4)]
        gradients = [float(np.trace(sketch.energy[:k, i*k:(i+1)*k]).real/k) for i in range(1, 4)]
        cross_inner = float(np.trace(sketch.gram[k:2*k, 2*k:3*k]).real/k)
        tokens.append([q/10, np.log1p(left.n_sites), np.log1p(right.n_sites), np.log1p(op.dimension),
            len(cross)/(left.n_sites+right.n_sites), sum(abs(g) for _, _, g in cross)/(left.n_sites+right.n_sites),
            sketch.center/sketch.radius, *norms,
            float(np.trace(sketch.energy[:k, :k]).real/k), *gradients, cross_inner,
            float(boundary_bound/sketch.radius),
            float(np.linalg.norm(h0-seeds[q] @ (seeds[q].conj().T @ h0))/np.sqrt(k)/sketch.radius)])
        reports[q] = dict(ritz=ritz, dimension=op.dimension, H_columns=op.matvec_count,
            H_seconds=op.matvec_seconds, internal_reduced_H_columns=4*k,
            operator_factory_seconds=factory_seconds, sketch_seconds=sketch_seconds,
            diagnostic_ritz_seconds=ritz_seconds, CG_normalization_error=op.diagnostics['CG_normalization_error'])
        if op.matvec_count != 5*k: raise ArithmeticError('full-H sketch accounting mismatch')
    return sketches, np.asarray(tokens), dict(sectors=reports, internal_spectral_bounds=[lower, upper], cross_operator_norm_bound=boundary_bound,
        H_columns=sum(x['H_columns'] for x in reports.values()), tokens_are_not_raw_density_environments=True)


def make_dataset(source: dict, output: Path, costs: dict) -> tuple[dict, set]:
    """Deduplicated actual CT6 operators with fixed cross-coupling perturbations."""
    leaf_bonds, joins, _ = split_bond_ledger(6, .125)
    leaves = [_leaf_block(source['local'][i], leaf_bonds[i]) for i in range(4)]
    families = []
    for pair in ((0, 1), (2, 3)):
        l, r = (leaves[i] for i in pair); seeds, _ = seed_maps(source, l, r, pair)
        families.append((l, r, joins[pair[0]//2], seeds, f'CT6_9plus9_pair{pair[0]//2}'))
    modules = [project_sector_maps(l, r, cross, maps=seeds, compact_boundary=True,
        max_output_dimension=144, memory_limit_bytes=768*1024**2) for l, r, cross, seeds, _ in families]
    costs['dataset_projection_H_columns'] += 48
    seeds, _ = seed_maps(source, *modules, None)
    families.append((*modules, joins[2], seeds, 'CT6_18plus18_PBC'))
    rows, seen = [], set()
    arrays = dict(tokens=[], gram=[], energy=[], teacher=[], split=[])
    for l, r, cross, seeds, family in families:
        for split, scales in (('train', TRAIN_SCALES), ('boundary_coupling_holdout', HELD_SCALES)):
            for first, second in scales:
                bonds = tuple((i, j, g*(first if g == 1. else second)) for i, j, g in cross)
                signature = operator_signature(l, r, bonds)
                if signature in seen: raise ArithmeticError('fixed dataset unexpectedly repeats an operator signature')
                seen.add(signature); tick = perf_counter()
                sketches, tokens, report = bundle(l, r, bonds, seeds)
                costs['teacher_sketch_H_columns'] += report['H_columns']
                for name, value in dict(tokens=tokens, gram=np.asarray([sketches[q].gram for q in Q]),
                        energy=np.asarray([sketches[q].energy for q in Q]),
                        teacher=np.asarray([report['sectors'][q]['ritz']['normalized_trace_energy'] for q in Q]), split=split).items():
                    arrays[name].append(value)
                rows.append(dict(family=family, child_sites=[l.n_sites, r.n_sites], split=split,
                    cross_scales=[first, second], signature=signature, seconds=perf_counter()-tick, diagnostics=report))
                print('teacher', family, split, first, second, flush=True)
    data = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez(output/'dataset.npz', **data)
    atomic_json(output/'dataset_records.json', rows)
    return data, {row['signature'] for row in rows if row['split'] == 'train'}


def train_controller(data: dict, kind: str, output: Path, steps: int = 500) -> tuple[dict, dict]:
    """Same full-batch Adam budget; heldout targets never choose a checkpoint."""
    train = data['split'] == 'train'
    mean, scale = data['tokens'][train].mean((0, 1)), data['tokens'][train].std((0, 1))
    scale = np.maximum(scale, .1)
    x = jnp.asarray((data['tokens']-mean)/scale)
    g, h, teacher = (jnp.asarray(data[name]) for name in ('gram', 'energy', 'teacher'))
    ids = np.flatnonzero(train)
    p = init_filter_controller(jax.random.PRNGKey(SEED), x.shape[-1], kind)
    initial = jax.tree.map(lambda x: np.asarray(x).copy(), p)
    m, v = (jax.tree.map(jnp.zeros_like, p) for _ in range(2))
    def objective(p):
        a = jax.vmap(lambda t: apply_filter_controller(p, t, kind, coefficient_bound=2.))(x[ids])
        energy = jax.vmap(jax.vmap(filter_trace_energy))(g[ids], h[ids], a)
        return jnp.mean(energy-teacher[ids])
    @jax.jit
    def step(p, m, v, count):
        loss, grad = jax.value_and_grad(objective)(p)
        m = jax.tree.map(lambda a, b: .9*a+.1*b, m, grad)
        v = jax.tree.map(lambda a, b: .999*a+.001*b*b, v, grad)
        p = jax.tree.map(lambda a, b, c: a-.02*b/(1-.9**count)/(jnp.sqrt(c/(1-.999**count))+1e-8), p, m, v)
        return p, m, v, loss
    started = perf_counter(); cpu = process_time(); history = []
    cold = None
    for count in range(1, steps+1):
        p, m, v, loss = step(p, m, v, jnp.asarray(count)); history.append(float(loss))
        if count == 1: cold = perf_counter()-started
    path = output/f'{kind}_controller.pkl'
    metadata = dict(kind=kind, seed=SEED, mean=mean, scale=scale, steps=steps,
        dataset_sha256=file_sha256(output/'dataset.npz'), rank_safe_coefficient_bound=2.)
    save_checkpoint(path, p, optimizer_state=dict(m=m, v=v, step=steps, rng=jax.random.PRNGKey(SEED)), metadata=metadata)
    saved = load_checkpoint(path); opt = saved['optimizer_state']
    aa, bb = step(p, m, v, jnp.asarray(steps+1)), step(saved['params'], opt['m'], opt['v'], jnp.asarray(steps+1))
    replay = max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a, b in zip(jax.tree.leaves(aa), jax.tree.leaves(bb), strict=True))
    coefficients = np.asarray(jax.vmap(lambda t: apply_filter_controller(p, t, kind, coefficient_bound=2.))(x))
    energies = np.asarray(jax.vmap(jax.vmap(filter_trace_energy))(g, h, coefficients))
    if not np.isfinite(energies).all() or replay != 0: raise ArithmeticError('controller training/checkpoint replay failed')
    np.savez(output/f'{kind}_training.npz', loss=np.asarray(history), coefficients=coefficients, energy=energies)
    report = dict(steps=steps, seed=SEED, parameters=sum(a.size for a in jax.tree.leaves(p)),
        first_loss=history[0], final_preupdate_loss=history[-1],
        train_mean_trace_regret=float((energies-data['teacher'])[train].mean()),
        heldout_boundary_mean_trace_regret=float((energies-data['teacher'])[~train].mean()),
        wall_seconds=perf_counter()-started, CPU_seconds=process_time()-cpu, cold_step_seconds=cold,
        next_step_replay_error=replay, checkpoint_sha256=file_sha256(path), parameter_sha256=pytree_sha256(p),
        parameter_update_norms={name: float(np.linalg.norm(np.asarray(value)-initial[name])) for name, value in p.items()},
        initial_parameter_sha256=pytree_sha256(initial), equal_q_weighting=True,
        coefficient_std_over_train_context_and_q=coefficients[train].std((0,1)).tolist(),
        coefficient_std_over_train_context_per_q=coefficients[train].std(0).tolist(),
        no_target_size_or_energy_selection=True)
    return dict(params=p, mean=mean, scale=scale, kind=kind), report


def verify_state(source: dict, state: dict, folder: Path) -> dict:
    """Fresh root H, exact all-edge norm, scalar CG/decoder, once-only layers."""
    tree = triangular_schur_hierarchy(state['length']); _, joins, bonds = split_bond_ledger(tree.length, .125)
    left, right, parent = tree.binary_merges[-1]
    root = project_sector_maps(state['blocks'][left], state['blocks'][right], joins[-1], maps=state['maps'][len(joins)-1],
        compact_boundary=True, max_output_dimension=1, memory_limit_bytes=768*1024**2)
    energy = float(root.h_by_q[0][0, 0].real)
    error = abs(energy-float(state['blocks'][parent].h_by_q[0][0, 0].real))
    env, nodes = active_state_environment(source, state)
    def expectation(index, matrices):
        return sum((q+1)*float(np.trace(h @ env.downward[nodes[index]][q]).real)
            for q, h in matrices.items())/env.norm_squared
    leaf = sum(expectation(i, b.h_by_q) for i, b in enumerate(state['blocks'][:len(tree.blocks)]))
    layers = {}; node_energy = []
    for node, (l, r, p) in enumerate(tree.binary_merges):
        internal = {q: w.conj().T @ internal_action(state['blocks'][l], state['blocks'][r], state['channels'][node][q], w)
            for q, w in state['maps'][node].items()}
        value = expectation(p, {q: state['blocks'][p].h_by_q[q]-h for q, h in internal.items()})
        size = state['blocks'][p].n_sites; layers[size] = layers.get(size, 0.)+value
        node_energy.append(dict(node=node, parent_sites=size, cross_energy=value,
            exact_training_signature_match=state['records'][node]['exact_training_signature_match']))
    ledger_error = abs(leaf+sum(layers.values())-energy)
    gram_error = max(float(np.max(abs(w.conj().T @ w-np.eye(w.shape[1])))) for maps in state['maps'].values() for w in maps.values())
    if not np.isfinite([energy, env.norm_squared, error, ledger_error, gram_error]).all() or max(error, ledger_error, gram_error, abs(env.norm_squared-1)) > 2e-8:
        raise ArithmeticError('fresh root-H/norm/once-only layer ledger failed')
    bank, profile = active_bank_from_state(source, state); cfg = MixedClebschConfig(tree.length, 20, 10)
    rng = np.random.default_rng(SEED+tree.length); rows = []
    for _ in range(4):
        row = np.zeros(tree.n_sites, np.int32)
        for i, block in enumerate(tree.blocks):
            row[list(block.sites)] = rng.permutation(np.r_[np.ones(4+i % 2, np.int32), np.zeros(5-i % 2, np.int32)])
        rows.append(row)
    query = jax.jit(make_feature_lut_amplitude(cfg, active_top_widths=profile)); packed = prepare_feature_lut_bank(bank, cfg)
    tick = perf_counter(); actual = np.asarray([float(query(packed, row)) for row in rows]); decoder_seconds = perf_counter()-tick
    expected = np.asarray([active_amplitude(source, state, row) for row in rows])
    relative = float(np.max(abs(actual-expected))/max(np.max(abs(expected)), 1e-300))
    if not np.any(expected != 0) or not np.isfinite(actual).all() or relative > 2e-10:
        raise ArithmeticError('scalar CG/active decoder replay failed')
    save_checkpoint(folder/'active_bank.pkl', bank, metadata=dict(model=dict(length=tree.length, multiplicity_cap=20, maximum_q=10),
        active_top_widths=profile, raw_tree_energy_per_site=energy/tree.n_sites, A1_energy_known=False, physical_variance_known=False))
    np.savez(folder/'raw_decoder_replay.npz', configurations=np.asarray(rows), scalar=expected, decoder=actual)
    return dict(raw_energy_total=energy, raw_energy_per_site=energy/tree.n_sites, raw_norm=env.norm_squared,
        root_H_error=error, ledger_error=ledger_error, isometry_error=gram_error,
        edge_norm_error=env.maximum_relative_edge_norm_error, decoder_relative_error=relative, decoder_seconds=decoder_seconds,
        leaf_energy=leaf, layer_cross_energy=layers, by_node=node_energy,
        heldout_ge72_cross_energy=sum(v for k, v in layers.items() if k >= 72),
        physical_bonds=len(bonds), weight_sum=sum(g for _, _, g in bonds), all_bonds_once=True,
        root_child_sites=[len(tree.coarse_supports[i]) for i in (left, right)],
        all_nonroot_q_sectors=Q, active_checkpoint_sha256=file_sha256(folder/'active_bank.pkl'),
        no_A1_energy_or_physical_variance_claim=True, verification_H_projection_columns=1)


def build_state(source: dict, length: int, arm: str, model: dict | None, output: Path,
                train_signatures: set, leaf_cache: dict, node_cache: dict, sketch_cache: dict, costs: dict) -> dict:
    tree = triangular_schur_hierarchy(length); leaves, joins, bonds = split_bond_ledger(length, .125)
    folder = output/arm/f'L{length}'; folder.mkdir(parents=True)
    state = dict(length=length, j2=.125, arm=arm, active=4, maps={}, channels={}, records=[], blocks=[],
        n_leaves=len(tree.blocks), n_merges=len(joins), completed_nodes=0, physical_bonds=len(bonds))
    started = perf_counter(); cpu = process_time()
    for block, inside in zip(tree.blocks, leaves, strict=True):
        key = (block.base_index, inside)
        if key not in leaf_cache: leaf_cache[key] = _leaf_block(source['local'][block.base_index], inside)
        state['blocks'].append(leaf_cache[key])
    for node, (li, ri, parent) in enumerate(tree.binary_merges):
        seed_start = perf_counter()
        left, right = state['blocks'][li], state['blocks'][ri]; cross = joins[node]
        root = node == len(joins)-1; signature = operator_signature(left, right, cross)
        seeds = seed_info = None; seed_hash = None
        if not root and arm != 'physical':
            pair = (tree.blocks[li].base_index, tree.blocks[ri].base_index) if li < len(tree.blocks) and ri < len(tree.blocks) else None
            seeds, seed_info = seed_maps(source, left, right, pair)
            seed_hash = hashlib.sha256(b''.join(np.ascontiguousarray(seeds[q]).tobytes() for q in Q)).hexdigest()
        key = (arm, signature, root, seed_hash); tick = perf_counter(); cached = key in node_cache
        seed_seconds = perf_counter()-seed_start
        if cached:
            merged, maps, diagnostics = node_cache[key]
        else:
            if root or arm == 'physical':
                retain = retained_widths(left.q_widths, right.q_widths, 4, root)
                solved = solve_sector_merge(left, right, cross, retain=retain, seed=SEED+node, tol=1e-8,
                    maxiter=200, propagate_operators=False, compact_boundary=True,
                    memory_limit_bytes=768*1024**2, max_output_dimension=144, dense_reference_limit=64)
                maps = {q: _real_matrix(w) for q, w in solved.reduced_maps.items()}
                calls = sum(v['matvec_count'] for v in solved.diagnostics['sectors'].values())
                costs['physical_solve_H_matvecs'] += calls
                diagnostics = dict(kind='shared_exact_root' if root else 'full_physical_solve', H_matvecs=calls)
            else:
                if arm == 'copy':
                    maps, diagnostics = seeds, dict(seed=seed_info, kind='CT6_copy_plus_separable_upper', H_columns=0)
                else:
                    sketch_key = (signature, seed_hash); sketch_hit = sketch_key in sketch_cache
                    if not sketch_hit:
                        sketch_cache[sketch_key] = bundle(left, right, cross, seeds)
                        costs['target_sketch_H_columns'] += sketch_cache[sketch_key][2]['H_columns']
                    sketches, tokens, sketch_report = sketch_cache[sketch_key]
                    infer_start = perf_counter()
                    coefficients = None
                    if model is not None:
                        coefficients = np.asarray(apply_filter_controller(model['params'], jnp.asarray((tokens-model['mean'])/model['scale']), model['kind'], coefficient_bound=2.))
                    maps, reports = {}, {}
                    for index, q in enumerate(Q):
                        maps[q], reports[q] = (sketch_ritz_columns(sketches[q]) if arm == 'ritz'
                            else filtered_columns(sketches[q], coefficients[index]))
                        maps[q] = _real_matrix(maps[q])
                    diagnostics = dict(seed=seed_info, sketch_cache_hit=sketch_hit, sketch=sketch_report,
                        output=reports, controller_and_selection_seconds=perf_counter()-infer_start,
                        coefficients=None if coefficients is None else coefficients.tolist())
            project_start = perf_counter()
            merged = project_sector_maps(left, right, cross, maps=maps, compact_boundary=True,
                max_output_dimension=144, memory_limit_bytes=768*1024**2)
            diagnostics['boundary_and_H_projection_seconds'] = perf_counter()-project_start
            costs['target_projection_H_columns'] += sum(w.shape[1] for w in maps.values())
            node_cache[key] = merged, maps, diagnostics
        if parent != len(state['blocks']): raise ArithmeticError('target topological order changed')
        state['blocks'].append(merged); state['maps'][node] = maps
        state['channels'][node] = {q: source_channels(left.q_widths, right.q_widths, q) for q in maps}
        state['completed_nodes'] += 1
        state['records'].append(dict(node=node, child_sites=[left.n_sites, right.n_sites],
            parent_sites=merged.n_sites, cross_bonds=len(cross), signature=signature,
            exact_training_signature_match=signature in train_signatures,
            cache_hit=cached, seconds=perf_counter()-tick, seed_and_signature_seconds=seed_seconds,
            standalone_equivalent_sketch_H_columns=120 if not cached and not root and arm not in ('copy','physical') else 0,
            diagnostics=diagnostics))
        atomic_json(folder/'progress.json', dict(completed_nodes=node+1, target_nodes=len(joins), costs=costs))
        print('build', arm, length, node, merged.n_sites, cached, flush=True)
    save_checkpoint(folder/'state.pkl', state, metadata=dict(initializer_controller=arm, optimizer_reset_required=True))
    verification = verify_state(source, state, folder); costs['verification_H_projection_columns'] += 1
    return dict(status='complete', verification=verification, wall_seconds=perf_counter()-started,
        CPU_seconds=process_time()-cpu, records=state['records'], state_sha256=file_sha256(folder/'state.pkl'))


def native_A1_replay(source: dict, output: Path) -> dict:
    """Actual L12 native export, complete-A1 amplitude and all-bond H replay.

    The caller's single outer wall timer remains active: no nested timeout
    cancels it. These deterministic kernel centers are not Born/energy data.
    """
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.triangular_clebsch_active_layout import repack_active_clebsch_bank
    from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_projected_amplitude
    from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action
    folder = output/'attention/L12'; state = load_checkpoint(folder/'state.pkl')['params']
    stored = load_checkpoint(folder/'active_bank.pkl'); active = stored['params']; profile = stored['metadata']['active_top_widths']
    cfg = MixedClebschConfig(12,20,10); row = np.load(folder/'raw_decoder_replay.npz')['configurations'][0]
    started = perf_counter(); timings = {}
    def timed(name, callback):
        tick = perf_counter(); result = jax.tree.map(np.asarray, callback())
        if not all(np.isfinite(a).all() for a in jax.tree.leaves(result)): raise ArithmeticError('nonfinite A1 replay')
        timings[name] = perf_counter()-tick
        return result
    native, embedding = native_bank_from_state(source, state)
    save_checkpoint(folder/'native_bank.pkl', native, metadata=dict(embedding=embedding,
        model=dict(length=12,multiplicity_cap=20,maximum_q=10), initializer='attention_boundary_filter'))
    raw_full = jax.jit(make_feature_lut_amplitude(cfg))
    raw_value = float(timed('native_full_width_raw', lambda: raw_full(prepare_feature_lut_bank(native,cfg),row)))
    expected_raw = active_amplitude(source,state,row)
    raw_error = abs(raw_value-expected_raw)/max(abs(expected_raw),1e-300)
    if raw_error > 2e-10: raise ArithmeticError('native full-width amplitude differs from scalar CG state')
    packed = prepare_feature_lut_bank(active,cfg)
    restored = prepare_feature_lut_bank(repack_active_clebsch_bank(native,cfg,profile),cfg)
    images = timed('full_group_construction', lambda: np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))),np.int32))
    bonds = triangular_j1j2_bonds(12,12,j2=.125,periodic=True)
    amplitude = jax.jit(make_feature_lut_projected_amplitude(cfg,images,chunk=4,active_top_widths=profile))
    phi = float(timed('complete_A1_amplitude',lambda:amplitude(packed,row)))
    tick = perf_counter(); action = make_cached_projected_heisenberg_action(cfg,images,bonds,projection_chunk=4,active_top_widths=profile)
    timings['H_factory'] = perf_counter()-tick
    values = [timed(name,lambda bank=bank:action(bank,row[None])) for name,bank in
        (('active_full_H_cold',packed),('active_full_H_warm',packed),('native_export_full_H',restored))]
    arrays = np.asarray([[float(x[0]) for x in value] for value in values])
    scale = max(float(np.max(abs(arrays))),1e-300)
    replay = float(np.max(abs(arrays-arrays[0]))/scale)
    amplitude_error = abs(arrays[0,0]-phi)/max(abs(phi),1e-300)
    if max(replay,amplitude_error)>2e-10: raise ArithmeticError('complete A1/H native replay failed')
    report = dict(status='complete_A1_and_all_H_export_replay_only', native_checkpoint_sha256=file_sha256(folder/'native_bank.pkl'),
        embedding=embedding, raw_native_scalar_relative_error=raw_error,
        full_A1_H_native_relative_error=replay, independent_A1_amplitude_relative_error=amplitude_error,
        action_psi_Hpsi=arrays.tolist(), images=len(images), physical_bonds=len(bonds),
        full_H_centers=3, amplitude_only_centers=1, timings=timings, wall_seconds=perf_counter()-started,
        no_Born_samples_no_energy_or_variance_estimate=True)
    atomic_json(folder/'complete_A1_replay.json',report)
    return report


def run(output: Path, *, seconds_cap: float = 2700.) -> dict:
    """One frozen bounded experiment, with no target-data model selection."""
    if not 300 <= seconds_cap <= 2700: raise ValueError('numerical cap must be300..2700seconds')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('fresh isolated output required')
    sources = [Path(__file__), ROOT/'su2_nqs/frontier/operator_filter_learning.py',
        Path(__file__).with_name('triangle_hierarchy_initialization.py'), Path(__file__).with_name('triangle_sector_transfer.py'),
        ROOT/'su2_nqs/operators/su2_sector_merge.py', ROOT/'su2_nqs/lattices/triangular_schur_hierarchy.py',
        ROOT/'su2_nqs/models/triangular_clebsch_feature_lut.py']
    registration = dict(task='actual_CT6_covariant_operator_filter_attention_L12_L18_L24_v1', lengths=[12,18,24],
        j2=.125, width=4, operator_words=['Qperp_BV','Qperp_commutator_AB_V_over2','Qperp_B_squared_V'],
        q=Q, root_q=0, arms=ARMS, controller_steps=500, learning_rate=.02, seed=SEED,
        train_cross_scales=TRAIN_SCALES, heldout_cross_scales=HELD_SCALES, coefficient_bound=2.,
        filter_Gram_eigenvalue_bounds=[1.,37.], total_numerical_seconds_cap=seconds_cap,
        training_sources='actual_CT6_learned9site_and18site_modules_only', target_ge72_never_trains=True,
        loss='true_Gram_gauge_invariant_mean_projector_trace_energy_minus_same_sketch_Ritz',
        no_eigenvector_coordinate_MSE=True, no_raw_environment_prediction=True,
        exact_signature_dedup=True, no_target_energy_checkpoint_selection=True,
        root_exact_solve_shared_by_all_arms=True, source_CT6_singlet_root_never_copied_to_internal_node=True,
        covariance_conditional_on_transported_seed_at_degenerate_separable_cutoff=True,
        each_sketch_pays_five_times_retained_width_H_columns=True,
        no_query_saving_claim_versus_same_sketch_Ritz=True, no_remote_submission_or_production_change=True,
        no_new_ansatz_head=True, equal_q_local_trace_not_full_complement_objective=True,
        target_sketch_requires_target_operators=True, root_exact_solve_algorithm_shared_not_identical_problem=True,
        source_training_is_shared_sunk_GPU_cost_not_local_CPU_time=True,
        context_gate='if attention context coefficient std<=1e-3 OR context-ablation trace change<=1e-6, skip only redundant linear/MLP target builds; still build attention/constant/copy/Ritz/physical on all sizes',
        raw_initialization_not_A1_energy_or_variance=True,
        source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in sources})
    atomic_json(output/'registration.json', registration)
    frozen = output/'registered_sources'; frozen.mkdir()
    for file in sources: shutil.copy2(file, frozen/file.name)
    started = perf_counter(); cpu = process_time()
    costs = dict(teacher_sketch_H_columns=0, dataset_projection_H_columns=0, target_sketch_H_columns=0,
        target_projection_H_columns=0, physical_solve_H_matvecs=0, verification_H_projection_columns=0)
    result = dict(status='running', registration=registration, costs=costs, models={}, states={})
    try:
        with _node_timeout(seconds_cap):
            source, provenance = load_source(.125); result['source'] = provenance
            save_checkpoint(output/'canonical_CT6_source.pkl', source, metadata=provenance)
            _, result['source_occupation_seed'] = learned_source_seeds(source)
            tick = perf_counter(); data, train_signatures = make_dataset(source, output, costs)
            result['teacher_extraction_seconds'] = perf_counter()-tick
            models = {}
            for kind in ('constant','linear','mlp','attention'):
                models[kind], result['models'][kind] = train_controller(data, kind, output)
            trained = data['split'] == 'train'
            attention = np.load(output/'attention_training.npz')
            constant = np.load(output/'constant_training.npz')
            mean_coeff = attention['coefficients'][trained].mean(0)
            ablated_coeff = np.broadcast_to(mean_coeff, attention['coefficients'].shape)
            ablated_energy = np.asarray(jax.vmap(jax.vmap(filter_trace_energy))(
                jnp.asarray(data['gram']), jnp.asarray(data['energy']), jnp.asarray(ablated_coeff)))
            context_std = float(attention['coefficients'][trained].std(0).max())
            ablation_change = float(np.mean(ablated_energy[trained]-attention['energy'][trained]))
            gate = context_std > 1e-3 and abs(ablation_change) > 1e-6
            result['training_only_context_gate'] = dict(maximum_same_q_context_coefficient_std=context_std,
                context_mean_replacement_trace_change=ablation_change,
                attention_minus_optimized_constant_train_trace=float(np.mean(attention['energy'][trained]-constant['energy'][trained])),
                nontrivial_context_gate_pass=gate, no_target_data_seen=True)
            selected_arms = ARMS if gate else tuple(arm for arm in ARMS if arm not in ('linear','mlp'))
            result['realized_arms'] = selected_arms
            atomic_json(output/'frozen_models_before_targets.json', dict(models=result['models'],
                dataset_sha256=file_sha256(output/'dataset.npz'), context_gate=result['training_only_context_gate'],
                realized_arms=selected_arms, no_target_state_built_yet=True))
            leaf_cache, node_cache, sketch_cache = {}, {}, {}
            for arm in selected_arms:
                result['states'][arm] = {}
                for length in registration['lengths']:
                    result['states'][arm][str(length)] = build_state(source, length, arm, models.get(arm), output,
                        train_signatures, leaf_cache, node_cache, sketch_cache, costs)
                    result['wall_seconds'] = perf_counter()-started
                    atomic_json(output/'results.json', result)
            remaining = seconds_cap-(perf_counter()-started)
            result['L12_attention_complete_A1_cost'] = (native_A1_replay(source,output) if remaining >= 90 else
                dict(status='not_run_remaining_budget_below_preregistered_90second_gate', remaining_seconds=remaining))
            result['status'] = 'complete_trained_controllers_and_all_three_target_sizes'
    except (TimeoutError, MemoryError) as error:
        result.update(status='bounded_incomplete', error=str(error))
    except Exception as error:
        result.update(status='failed_validation', error=repr(error)); raise
    finally:
        result.update(wall_seconds=perf_counter()-started, CPU_seconds=process_time()-cpu,
            peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        atomic_json(output/'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--seconds-cap', type=float, default=2700.)
    args = parser.parse_args(); run(args.output, seconds_cap=args.seconds_cap)
