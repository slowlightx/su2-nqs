"""Frozen CT6 complete-A1 capacity tangents and cached full-H residuals.

This is an offline directional diagnostic, not a new ansatz or energy audit.
Every direction changes a complete SU(2) multiplet in a physical local Schur
basis. The parent Hamiltonian ratios are sealed old audit records; they give
the energy derivative, but NOT finite-candidate energy, variance, or H delta.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_amplitude, make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_clebsch_gauge import (
    canonicalize_triangular_bank, triangular_bank_reduced_tree,
    triangular_edge_spectrum,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout
from su2_nqs.su2.reduced_tree_environment import reduced_tree_environments
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.checkpoint import load_checkpoint


def tangent_moments(scores: np.ndarray, energies: np.ndarray,
                    probabilities: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return real physical FS metric and 2 Re Cov(conj(O), E_L).

    Uniform probabilities are appropriate for actual Born records. Explicit
    probabilities support independent exhaustive small-system validation.
    Center both quantities; a pure normalization/phase gauge has zero force.
    """
    o, e = np.asarray(scores, np.complex128), np.asarray(energies, np.complex128)
    if o.ndim != 2 or e.shape != (len(o),) or not np.isfinite(o).all() or not np.isfinite(e).all():
        raise ValueError('finite scores[N,D] and energies[N] required')
    p = np.full(len(o), 1/len(o)) if probabilities is None else np.asarray(probabilities, float)
    if p.shape != e.shape or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-12):
        raise ValueError('normalized nonnegative record probabilities required')
    centered = o-p @ o
    residual = e-p @ e
    metric = (centered.conj().T @ (p[:, None]*centered)).real
    force = 2*np.real(centered.conj().T @ (p*residual))
    return (metric+metric.T)/2, force


def _pair_operators(config: MixedClebschConfig) -> dict[int, np.ndarray]:
    """Fixed physical Schur representation of all36 scalar pair operators."""
    bits = (np.arange(512)[:, None] >> np.arange(8, -1, -1)) & 1
    mags = np.sum(2*bits-1, axis=1)
    pairs = tuple(combinations(range(9), 2))
    operators = {}
    for q in config.local_q:
        table = triangle_block_tables()['nine'][q]*(mags == q)[:, None]
        rows = []
        for a, b in pairs:
            opposite = bits[:, a] != bits[:, b]
            packed = np.arange(512) ^ (opposite.astype(int)*((1 << (8-a))+(1 << (8-b))))
            acted = np.where(opposite[:, None], -.25, .25)*table + .5*opposite[:, None]*table[packed]
            rows.append(table.T @ acted)
        operators[q] = np.asarray(rows)
    return operators


def _pair_profiles(bank: dict, config: MixedClebschConfig) -> np.ndarray:
    """Cheap exact RAW-tree local pair scalars, not full-A1 environments."""
    state = triangular_bank_reduced_tree(bank, config)
    env = reduced_tree_environments(state)
    operators = _pair_operators(config)
    profiles = []
    for node, local in zip(state.tree.leaves, bank['local'], strict=True):
        values = np.zeros(36)
        for q, w in zip(config.local_q, local, strict=True):
            sigma = (q+1)*w @ env.downward[node][q] @ w.conj().T/env.norm_squared
            values += np.einsum('pij,ji->p', operators[q], sigma).real
        profiles.append(values)
    return np.asarray(profiles)


def raw_pair_direction_responses(bank: dict, deltas: dict,
                                 config: MixedClebschConfig) -> np.ndarray:
    """Cheap direction-specific RAW pair responses, including global norm.

    One24x36 array describes how the local region of each directional edit
    responds. This is NOT its full-A1 response or H-residual target. Both the
    changed-q density and normalization changes in ALL other q blocks enter.
    """
    state = triangular_bank_reduced_tree(bank, config)
    env = reduced_tree_environments(state)
    operators = _pair_operators(config)
    profiles = _pair_profiles(bank, config)
    response = np.zeros((24, 36))
    for edge, (node, blocks, changes) in enumerate(zip(
            state.tree.leaves, bank['local'], deltas['local'], strict=True)):
        norm_derivative = np.zeros(24)
        unnormalized_response = np.zeros((24, 36))
        for q, w, change in zip(config.local_q, blocks, changes, strict=True):
            first = (q+1)*np.einsum('drw,wv,sv->drs', change, env.downward[node][q], w.conj())/env.norm_squared
            derivative = first+first.conj().swapaxes(1, 2)
            norm_derivative += np.trace(derivative, axis1=1, axis2=2).real
            unnormalized_response += np.einsum('pij,dji->dp', operators[q], derivative).real
        response += unnormalized_response-norm_derivative[:, None]*profiles[edge][None]
    return response


def physical_mode_rotation(w: np.ndarray, virtual_mode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fix a Schmidt mode and unused direction in the PHYSICAL frame.

    The largest component of phi=Wv is positive. The unused direction is
    the longest column of I-WW^T (first index breaks ties), not an arbitrary
    complete-QR column. Thus the physical infinitesimal map z phi^T is
    invariant under orthogonal virtual-coordinate changes, away from exact
    degenerate-Schmidt or max-column ties. Real orthonormal W is required.
    """
    w, vector = np.asarray(w), np.asarray(virtual_mode)
    if not np.isrealobj(w) or not np.isrealobj(vector) or not np.isfinite(w).all() or not np.isfinite(vector).all():
        raise ValueError('finite real physical Schur matrices and virtual modes required')
    w, vector = w.astype(float), vector.astype(float).copy()
    if w.ndim != 2 or vector.shape != (w.shape[1],) or w.shape[0] <= w.shape[1]:
        raise ValueError('rectangular local W with a physical unused complement required')
    if not np.allclose(w.T @ w, np.eye(w.shape[1]), atol=1e-11, rtol=0):
        raise ValueError('orthonormal local Schur columns required')
    if np.linalg.norm(vector) <= np.finfo(float).tiny:
        raise ValueError('nonzero virtual mode required')
    vector /= np.linalg.norm(vector)
    physical = w @ vector
    if physical[np.argmax(abs(physical))] < 0:
        vector *= -1
    projector = np.eye(w.shape[0])-w @ w.T
    index = int(np.argmax(np.diag(projector)))
    complement = projector[:, index]/np.linalg.norm(projector[:, index])
    if complement[index] < 0:
        complement *= -1
    return vector, complement


def capacity_coordinate_family(bank: dict, config: MixedClebschConfig) -> tuple[dict, dict, dict]:
    """24 fixed-width physical directions, independent of H residual targets.

    Four nine-site edges x q=(1,3,5) x {delete,rotate}. The 16th raw Schmidt
    copy is either removed infinitesimally or rotated into a deterministic
    unused physical Schur direction. Rotation is NOT a reversible virtual
    gauge and is NOT a newly added channel. Parent tensors remain fixed.
    Native costs count BOTH leaf and incident-parent coefficients per copy;
    fixed padded storage remains unchanged by all diagnostic directions.
    """
    if config.length != 6 or config.multiplicity_cap < 16:
        raise ValueError('registered four-block CT6 family requires width >=16')
    canonical, gauge = canonicalize_triangular_bank(bank, config)
    canonical = jax.tree.map(np.asarray, canonical)
    hierarchy = triangular_schur_hierarchy(6)
    profiles = _pair_profiles(canonical, config)
    centers = np.asarray([np.mean(b.lifted_coordinates, axis=0) for b in hierarchy.blocks])
    layout = mixed_clebsch_layout(config)
    count = 24
    deltas = [[np.zeros((count, *w.shape), dtype=np.float64) for w in node]
              for node in canonical['local']]
    tokens, features, costs, geometry = [], [], [], []
    names = ['q', 'mode_weight', 'sector_weight', 'sector_entropy', 'sector_effective_rank',
             'retained_width', 'physical_multiplicity', 'kind_rotate', 'x', 'y'] + [f'pair_{i}' for i in range(36)]
    for edge in range(4):
        spectrum = triangular_edge_spectrum(canonical, config, edge)
        parent = next(row for row in layout if edge in (row['left'], row['right']))
        is_left = parent['left'] == edge
        for q in (1, 3, 5):
            qi = config.local_q.index(q)
            w = canonical['local'][edge][qi]
            block = spectrum.blocks[q]
            # Canonical leaf Gram=I, so the exact Schmidt gauge is unitary.
            vector = np.asarray(block.gauge[:, 15])
            if np.max(abs(vector.imag)) > 1e-11:
                raise ValueError('this real-parent registered diagnostic requires real Schmidt modes')
            vector, complement = physical_mode_rotation(w, vector.real)
            physical = w @ vector
            if np.linalg.norm(w.T @ complement) > 1e-10:
                raise ArithmeticError('rotation is not orthogonal to retained physical Schur span')
            native_cost = w.shape[0]
            for group in parent['groups']:
                for ql, qr, _ in group['keys']:
                    if (ql if is_left else qr) == q:
                        native_cost += group['shape'][1]*group['shape'][3 if is_left else 2]
            weights = np.asarray(block.multiplicity_weights)
            sector_weight = float(weights.sum())
            probs = weights/max(sector_weight, np.finfo(float).tiny)
            entropy = -float(np.sum(probs[probs > 0]*np.log(probs[probs > 0])))
            rank = float(1/np.sum(probs**2))
            for kind in ('delete', 'rotate'):
                index = len(tokens)
                direction = -np.outer(physical, vector) if kind == 'delete' else np.outer(complement, vector)
                deltas[edge][qi][index] = direction
                tokens.append(dict(index=index, edge=edge, q=q, kind=kind, schmidt_index=15,
                                   physical_multiplet=q+1, marginal_native_scalars=int(native_cost)))
                features.append([q, float(weights[15]), sector_weight, entropy, rank,
                                 w.shape[1], w.shape[0], int(kind == 'rotate'), *centers[edge], *profiles[edge]])
                costs.append(native_cost)
                geometry.append(centers[edge])
    relations = np.zeros((count, count, 4))
    bondsets = [triangular_j1j2_bonds(6, 6, j2=0., periodic=True),
                [(a, b, j) for a, b, j in triangular_j1j2_bonds(6, 6, j2=1., periodic=True)
                 if (min(a, b), max(a, b)) not in {(min(i, j), max(i, j)) for i, j, _ in
                                                  triangular_j1j2_bonds(6, 6, j2=0., periodic=True)}]]
    for i, left in enumerate(tokens):
        a = set(hierarchy.blocks[left['edge']].sites)
        for j, right in enumerate(tokens):
            b = set(hierarchy.blocks[right['edge']].sites)
            relations[i, j, 0] = left['edge'] == right['edge']
            relations[i, j, 1] = left['q'] == right['q']
            for k, bonds in enumerate(bondsets):
                relations[i, j, k+2] = sum((u in a and v in b) or (u in b and v in a) for u, v, _ in bonds)
    arrays = dict(token_features=np.asarray(features), token_cost=np.asarray(costs),
                  token_nodes=np.asarray([t['edge'] for t in tokens]), token_q=np.asarray([t['q'] for t in tokens]),
                  token_kind=np.asarray([int(t['kind'] == 'rotate') for t in tokens]),
                  physical_multiplet_cost=np.asarray([t['q']+1 for t in tokens]),
                  positions=np.asarray(geometry), pair_relations=relations)
    metadata = dict(tokens=tokens, feature_names=names, canonical_report=gauge,
                    support_sites=[list(b.sites) for b in hierarchy.blocks],
                    relation_names=['same_edge', 'same_q', 'J1_connecting_bonds', 'J2_connecting_bonds'],
                    physical_rotation_phase='dominant_phi_component_positive; longest_physical_unused_projector_column',
                    exact_degenerate_Schmidt_modes_not_unique=True,
                    fixed_padded_storage=True, native_cost_is_per_edge_copy_not_rotation_added_cost=True)
    return canonical, dict(local=tuple(tuple(v) for v in deltas)), dict(arrays=arrays, metadata=metadata)


def coordinate_bank(canonical: dict, deltas: dict, theta: np.ndarray) -> dict:
    """Reconstruct the literal physical bank for independent full-H checks."""
    theta = np.asarray(theta)
    if theta.shape != (24,) or not np.isrealobj(theta) or not np.isfinite(theta).all():
        raise ValueError('24 finite real directional coefficients required')
    local = tuple(tuple(base+np.einsum('d,drw->rw', theta, delta)
                        for base, delta in zip(node, dn, strict=True))
                  for node, dn in zip(canonical['local'], deltas['local'], strict=True))
    return dict(local=local, top=canonical['top'])


def make_capacity_amplitude(canonical: dict, deltas: dict, config: MixedClebschConfig,
                            *, project: bool = True) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Exact full-A1 amplitude(coordinates,bits), differentiating only24 scalars.

    Local512-row tables and tangent tables are precontracted ONCE. They are
    linear functions of legal W tensors, so all24 derivatives remain exact.
    No full parameter Jacobian or hidden truncated spatial projector is used.
    """
    prepared = prepare_feature_lut_bank(canonical, config)
    tables = triangle_block_tables()['nine']
    changes = tuple(tuple(jnp.asarray(np.einsum('pr,drw->dpw', tables[q], np.asarray(delta)))
                          for q, delta in zip(config.local_q, node, strict=True))
                    for node in deltas['local'])
    group = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    projected = (make_feature_lut_projected_amplitude(config, group, chunk=4) if project
                 else make_feature_lut_amplitude(config))

    def amplitude(theta, bits):
        local = tuple(tuple(base+jnp.einsum('d,dpw->pw', theta, change)
                            for base, change in zip(node, dn, strict=True))
                      for node, dn in zip(prepared['local_features'], changes, strict=True))
        return projected(dict(local_features=local, top=prepared['top']), bits)
    return amplitude


def run(parent: Path, output: Path, *, records_per_chain: int = 4,
        seconds_cap: float = 600.) -> dict:
    """Cache a bounded reused-audit tangent dataset; never change the parent."""
    if not jax.config.x64_enabled:
        raise RuntimeError('enable x64 before loading the frozen checkpoint')
    if output.exists() or not 1 <= records_per_chain <= 8 or seconds_cap <= 0:
        raise ValueError('fresh output, 1..8 records/chain and positive cap required')
    start = perf_counter()
    saved = load_checkpoint(parent/'checkpoint.pkl')
    source_registration = json.loads((parent/'registration.json').read_text())
    registration = source_registration.get('config', source_registration)
    audit = json.loads((parent/'terminal_audit.json').read_text())
    bank = saved['params']
    if registration.get('arity', registration.get('arm')) != 'ternary_joint' or pytree_sha256(bank) != audit['params_sha256']:
        raise ValueError('audited joint-ternary parent identity required')
    config = MixedClebschConfig(**registration['model'])
    with np.load(parent/'terminal_audit.npz') as data:
        labels = data['chain_labels']
        chosen = np.concatenate([np.flatnonzero(labels == c)[np.linspace(0, np.sum(labels == c)-1,
                                  records_per_chain, dtype=int)] for c in np.unique(labels)])
        bits, energies, labels = data['bits'][chosen], data['local_energies'].reshape(-1)[chosen], labels[chosen]
    if len(bits) > 1024 or len(np.unique(labels)) not in (64, 128):
        raise ValueError('registered probe uses64/128 original audit chains and <=1024 records')
    canonical, deltas, data = capacity_coordinate_family(bank, config)
    chains = np.unique(labels)
    fit_count = 3*len(chains)//4
    fit_chains = np.random.default_rng(74120).permutation(chains)[:fit_count]
    fit_mask = np.isin(labels, fit_chains)
    plan = dict(task='actual_CT6_full_A1_residual_capacity_tangents_v2', model=vars(config),
                parent_checkpoint=str(parent/'checkpoint.pkl'),
                parent_checkpoint_sha256=file_sha256(parent/'checkpoint.pkl'),
                parent_params_sha256=pytree_sha256(bank),
                canonical_bank_sha256=pytree_sha256(canonical),
                parent_audit_sha256=file_sha256(parent/'terminal_audit.npz'),
                j2=registration['j2_over_j1'], records_per_chain=records_per_chain,
                max_records=1024, seconds_cap=seconds_cap, complete_A1_images=432,
                split_seed=74120, fit_chain_count=fit_count, audit_chain_count=len(chains)-fit_count,
                candidate_selection_independent_of_H_residual=True,
                reused_old_audit_not_new_independent_energy_audit=True,
                no_finite_candidate_energy_or_variance=True, full_H_calls=0,
                **data['metadata'])
    plan['sha256'] = canonical_sha256(plan)
    output.mkdir(parents=True)
    atomic_json(output/'registration.json', plan)
    print('registered', registration['j2_over_j1'], len(bits), flush=True)
    amplitude = make_capacity_amplitude(canonical, deltas, config)
    raw_amplitude = make_capacity_amplitude(canonical, deltas, config, project=False)
    zero = jnp.zeros(24, jnp.float64)
    evaluate = jax.jit(lambda rows: jax.lax.map(lambda row: (
        jax.value_and_grad(amplitude)(zero, row),
        jax.value_and_grad(raw_amplitude)(zero, row)), rows))
    amplitudes, derivatives, raw_amplitudes, raw_derivatives = [], [], [], []
    batches = []
    for first in range(0, len(bits), 16):
        if perf_counter()-start > seconds_cap:
            atomic_json(output/'censored.json', dict(rows_completed=first, seconds=perf_counter()-start,
                        reason='fixed local wall budget; no partial fit/heldout comparison claimed'))
            raise TimeoutError('registered local diagnostic budget reached')
        t = perf_counter()
        (values, grads), (raw_values, raw_grads) = evaluate(jnp.asarray(bits[first:first+16]))
        values, grads = np.asarray(values), np.asarray(grads)
        if np.any(values == 0) or not np.isfinite(values).all() or not np.isfinite(grads).all():
            raise ArithmeticError('nonfinite/null parent amplitudes or tangents')
        amplitudes.extend(values); derivatives.extend(grads)
        raw_amplitudes.extend(np.asarray(raw_values)); raw_derivatives.extend(np.asarray(raw_grads))
        batches.append(perf_counter()-t)
        print('rows', first+len(values), 'batch_seconds', batches[-1], flush=True)
    amplitudes, derivatives = np.asarray(amplitudes), np.asarray(derivatives)
    scores = derivatives/amplitudes[:, None]
    arrays = dict(**data['arrays'], bits=bits, row_indices=chosen, amplitudes=amplitudes,
                  amplitude_derivatives=derivatives, scores=scores,
                  raw_amplitudes=np.asarray(raw_amplitudes), raw_amplitude_derivatives=np.asarray(raw_derivatives),
                  local_energies=energies, chain_labels=labels, fit_mask=fit_mask)
    arrays['raw_log_derivative_valid'] = abs(arrays['raw_amplitudes']) > np.finfo(float).tiny
    arrays['raw_scores_common_projected_measure'] = np.divide(
        arrays['raw_amplitude_derivatives'], arrays['raw_amplitudes'][:, None],
        out=np.full_like(arrays['raw_amplitude_derivatives'], np.nan),
        where=arrays['raw_log_derivative_valid'][:, None])
    for name, mask in [('fit', fit_mask), ('audit', ~fit_mask)]:
        arrays[name+'_metric'], arrays[name+'_force'] = tangent_moments(scores[mask], energies[mask])
    np.savez_compressed(output/'dataset.npz', **arrays)
    result = dict(status='completed', registration_sha256=plan['sha256'],
                  dataset_sha256=file_sha256(output/'dataset.npz'),
                  rows=len(bits), directions=24, wall_seconds=perf_counter()-start,
                  batch_seconds=batches, complete_A1_images=432, full_H_calls=0,
                  parent_unchanged=pytree_sha256(load_checkpoint(parent/'checkpoint.pkl')['params']) == plan['parent_params_sha256'],
                  qualifications=['tangent energy slopes only; no finite candidate energy or variance',
                                  'raw environment features are not projected environments',
                                  'same padded arrays; no demonstrated capacity or speed win',
                                  'raw scores use the SAME projected Born rows, not raw-tree Born measure',
                                  'old reused Born records cannot certify newly opened exact-node support'])
    atomic_json(output/'results.json', result)
    return result


def run_panel(output: Path, *, seconds_cap: float = 600.) -> dict:
    """Six old training parents and two later whole-parent holdouts.

    These are related checkpoints, NOT independent physical families. The
    manifest is frozen before controller fitting; latest two cached tangents
    may already exist but have not supplied labels to the training procedure.
    """
    root = Path(__file__).resolve().parents[3]
    results = root/'projects/clebsch_tree_nqs/results'
    cases = []
    for tag in ('j0', 'j0125'):
        sources = [
            ('w16', results/f'triangle_capacity_20261017/remote_snapshot/{tag}/ternary_joint', 'train', 8),
            ('w18', results/f'remote_review_20261021/ct6_width_20261020/{tag}_ternary_joint_w18_control', 'train', 4),
            ('w20', results/f'remote_review_20261021/ct6_width_20261020/{tag}_ternary_joint_w20', 'train', 4),
            ('latest', results/('remote_review_20261021/optimized_endpoints/ct6_continue_20261021/j0_ternary_joint_w20_continue32'
                               if tag == 'j0' else 'remote_review_20261022/ct6_continue_20261021/j0125_ternary_joint_w20_continue32'),
             'whole_parent_holdout', 4),
        ]
        for stage, parent, split, records in sources:
            cases.append(dict(name=tag if stage == 'latest' else f'{tag}_{stage}',
                              parent=str(parent.relative_to(root)), split=split,
                              records_per_chain=records, checkpoint_sha256=file_sha256(parent/'checkpoint.pkl'),
                              audit_sha256=file_sha256(parent/'terminal_audit.npz')))
    plan = dict(task='six_old_training_two_late_checkpoint_holdout_capacity_v2', cases=cases,
                total_seconds_cap=seconds_cap, only_related_training_lineages=True,
                latest_two_forbid_controller_fitting=True, controller_seeds=[74121, 74122, 74123],
                controller_steps=250, no_H_calls=True)
    plan['sha256'] = canonical_sha256(plan)
    if (output/'panel_registration.json').exists():
        raise FileExistsError('panel manifests are immutable')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'panel_registration.json', plan)
    start = perf_counter()
    summary = []
    for case in cases:
        destination = output/case['name']
        if destination.exists():
            registered = json.loads((destination/'registration.json').read_text())
            if registered['task'] != 'actual_CT6_full_A1_residual_capacity_tangents_v2':
                raise ValueError('only phase-fixed v2 existing caches may be reused')
            result = json.loads((destination/'results.json').read_text())
        else:
            remaining = seconds_cap-(perf_counter()-start)
            if remaining <= 0:
                raise TimeoutError('fixed panel budget reached')
            result = run(root/case['parent'], destination, records_per_chain=case['records_per_chain'], seconds_cap=remaining)
        summary.append(dict(case=case['name'], split=case['split'], **result))
        atomic_json(output/'panel_results.json', dict(cases=summary, wall_seconds=perf_counter()-start))
    return dict(cases=summary, wall_seconds=perf_counter()-start)


def supplement_direction_features(panel: Path) -> dict:
    """Seal one logically identified missing-feature supplement, not targets.

    Initial v2 scores and registrations stay immutable. Original46-feature
    controllers remain the baseline; this prespecified supplement supplies36
    raw pair derivatives of the actual physical mode. It need not determine
    inter-block coherence, which is precisely why full-A1 residuals are tested.
    """
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((panel/'panel_registration.json').read_text())
    start = perf_counter()
    rows = []
    for case in manifest['cases']:
        directory = panel/case['name']
        destination = directory/'raw_pair_direction_response.npz'
        if destination.exists():
            raise FileExistsError('direction feature supplements are immutable')
        parent = root/case['parent']
        saved = load_checkpoint(parent/'checkpoint.pkl')
        registration = json.loads((directory/'registration.json').read_text())
        if pytree_sha256(saved['params']) != registration['parent_params_sha256']:
            raise ValueError('physical direction supplement parent identity mismatch')
        cfg = MixedClebschConfig(**registration['model'])
        canonical, deltas, _ = capacity_coordinate_family(saved['params'], cfg)
        if pytree_sha256(canonical) != registration['canonical_bank_sha256']:
            raise ValueError('canonical parent gauge changed since score generation')
        response = raw_pair_direction_responses(canonical, deltas, cfg)
        np.savez_compressed(destination, raw_pair_direction_response=response)
        result = dict(feature='36 normalized RAW local pair directional derivatives',
                      shape=list(response.shape), parent_params_sha256=registration['parent_params_sha256'],
                      initial_score_dataset_sha256=file_sha256(directory/'dataset.npz'),
                      feature_dataset_sha256=file_sha256(destination),
                      workflow_source_sha256=file_sha256(Path(__file__)),
                      response_includes_all_q_normalization=True, full_A1_or_H_calls=0,
                      not_full_projected_response_or_residual_target=True,
                      exact_interblock_coherence_not_determined=True)
        atomic_json(directory/'raw_pair_direction_response.json', result)
        rows.append(dict(case=case['name'], **result))
    summary = dict(cases=rows, wall_seconds=perf_counter()-start)
    atomic_json(panel/'direction_feature_supplements.json', summary)
    return summary


def screening_context(bits: np.ndarray, fit_mask: np.ndarray,
                      support_sites: list[list[int]], token_nodes: np.ndarray,
                      pair_response: np.ndarray) -> dict[str, np.ndarray]:
    """Raw local-mode form factors screened by FIT-only projected correlations.

    f_i=sum_j delta<S_i.S_j> has no onsite derivative. C=3 Cov(Sz_i,Sz_j)
    estimates connected vector-spin correlations of the strict singlet parent.
    R_ab=f_a^T C_ab f_b/(||f_a||||f_b||) is an input context, NOT a full-A1
    environment, susceptibility, residual-force target, or new energy result.
    """
    bits, mask = np.asarray(bits), np.asarray(fit_mask)
    nodes = np.asarray(token_nodes)
    response = np.asarray(pair_response, dtype=np.float64)
    supports = np.asarray(support_sites)
    if (bits.ndim != 2 or not np.issubdtype(bits.dtype, np.integer)
            or np.any((bits != 0) & (bits != 1)) or mask.dtype != bool
            or mask.shape != (len(bits),) or np.count_nonzero(mask) < 2
            or supports.ndim != 2 or not np.issubdtype(supports.dtype, np.integer)
            or np.any(supports < 0) or np.any(supports >= bits.shape[1])
            or nodes.shape != (len(response),) or not np.issubdtype(nodes.dtype, np.integer)
            or np.any(nodes < 0) or np.any(nodes >= len(supports))):
        raise ValueError('valid physical bits, nonempty FIT mask, supports and directional node indices required')
    pairs = tuple(combinations(range(supports.shape[1]), 2))
    if response.ndim != 2 or response.shape[1] != len(pairs) or not np.isfinite(response).all():
        raise ValueError('one finite raw pair directional profile per token required')
    incidence = np.zeros((len(pairs), supports.shape[1]))
    for row, (i, j) in enumerate(pairs):
        incidence[row, (i, j)] = 1.
    profiles = response @ incidence
    norms = np.linalg.norm(profiles, axis=1)
    z = (2*bits[mask]-1)/2
    mean = z.mean(axis=0)
    connected = 3*(z.T @ z/len(z)-np.outer(mean, mean))
    normalized = np.divide(profiles, norms[:, None], out=np.zeros_like(profiles), where=norms[:, None] > 0)
    embedded = np.zeros((len(response), bits.shape[1]))
    embedded[np.arange(len(response))[:, None], supports[nodes]] = normalized
    matrix = embedded @ connected @ embedded.T
    permutation = np.random.default_rng(74130).permutation(len(supports))
    if np.array_equal(permutation, np.arange(len(supports))) and len(supports) > 1:
        permutation = np.roll(permutation, 1)
    scrambled = np.zeros_like(embedded)
    scrambled[np.arange(len(response))[:, None], supports[permutation[nodes]]] = normalized
    return dict(raw_spin_response_profiles=profiles, raw_spin_response_norms=norms,
                projected_fit_connected_spin_matrix=connected, fit_site_Sz_mean=mean,
                screening_context=matrix, screening_context_scrambled=scrambled @ connected @ scrambled.T,
                geometry_scramble=permutation)


def supplement_screening_context(panel: Path) -> dict:
    """Immutable physics-derived cross-region input, with no amplitude/H calls."""
    source = json.loads((panel/'panel_registration.json').read_text())
    path = panel/'screening_context_registration.json'
    if path.exists():
        raise FileExistsError('screening-context registrations are immutable')
    plan = dict(task='raw_local_mode_projected_fit_spin_screening_context_v1',
                workflow_source_sha256=file_sha256(Path(__file__)),
                cases=[dict(name=c['name'], split=c['split'],
                            dataset_sha256=file_sha256(panel/c['name']/'dataset.npz'),
                            raw_direction_feature_sha256=file_sha256(panel/c['name']/'raw_pair_direction_response.npz'))
                       for c in source['cases']],
                no_new_amplitude_H_or_neural_training=True, only_FIT_rows_for_C=True,
                f_definition='f_i=sum_j delta<S_i.S_j> in RAW local nine-site region; onsite derivative zero',
                C_definition='3*(mean(z_i*z_j)-mean(z_i)*mean(z_j)); z=(2*bit-1)/2 on FIT rows only',
                R_definition='f_a.T C[block_a,block_b] f_b/(norm(f_a)*norm(f_b)); zero only if a form factor is zero',
                factor_three_requires_rotationally_invariant_parent=True,
                raw_f_and_projected_C_mixed_scope_explicit=True, geometry_scramble_seed=74130,
                not_full_projected_environment_or_H_force=True)
    plan['sha256'] = canonical_sha256(plan)
    atomic_json(path, plan)
    start = perf_counter()
    results = []
    for case in source['cases']:
        directory = panel/case['name']
        registered = json.loads((directory/'registration.json').read_text())
        with np.load(directory/'dataset.npz') as data, np.load(directory/'raw_pair_direction_response.npz') as extra:
            arrays = screening_context(data['bits'], data['fit_mask'], registered['support_sites'],
                                       data['token_nodes'], extra['raw_pair_direction_response'])
            fit_rows = np.flatnonzero(data['fit_mask'])
            arrays.update(fit_cache_row_indices=fit_rows,
                          fit_original_audit_row_indices=data['row_indices'][fit_rows],
                          fit_chain_labels=data['chain_labels'][fit_rows])
            nodes, kinds = np.asarray(data['token_nodes']), np.asarray(data['token_kind'])
        offblock = nodes[:, None] != nodes[None, :]
        report = dict(case=case['name'], source_registration_sha256=plan['sha256'],
                      fit_rows_sha256=pytree_sha256(arrays['fit_original_audit_row_indices']),
                      fits=len(fit_rows), fit_chains=len(np.unique(arrays['fit_chain_labels'])),
                      symmetry_defect=float(np.max(abs(arrays['screening_context']-arrays['screening_context'].T))),
                      offblock_R_max_abs=float(np.max(abs(arrays['screening_context'][offblock]))),
                      offblock_R_std=float(np.std(arrays['screening_context'][offblock])),
                      scramble_max_abs_difference=float(np.max(abs(arrays['screening_context']-arrays['screening_context_scrambled']))),
                      raw_rotation_sum_f_max_abs=float(np.max(abs(arrays['raw_spin_response_profiles'][kinds == 1].sum(axis=1)))),
                      raw_f_not_projected_mode_profile=True, C_from_projected_Born_fit_only=True,
                      full_A1_amplitude_calls=0, full_H_calls=0, neural_fits=0)
        destination = directory/'screening_context.npz'
        if destination.exists():
            raise FileExistsError('screening-context arrays are immutable')
        np.savez_compressed(destination, **arrays)
        report['dataset_sha256'] = file_sha256(destination)
        atomic_json(directory/'screening_context.json', report)
        results.append(report)
    summary = dict(cases=results, wall_seconds=perf_counter()-start)
    atomic_json(panel/'screening_context_results.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--panel', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records-per-chain', type=int, default=4)
    parser.add_argument('--seconds-cap', type=float, default=600.)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    if args.panel == bool(args.parent):
        parser.error('choose exactly one of --panel or --parent')
    result = (run_panel(args.output, seconds_cap=args.seconds_cap) if args.panel else
              run(args.parent, args.output, records_per_chain=args.records_per_chain, seconds_cap=args.seconds_cap))
    print(json.dumps(result, indent=2))
