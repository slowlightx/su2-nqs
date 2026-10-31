"""One bounded cache-only collective-query probe, not a production update.

Frozen old-parent controllers supply scalar token embeddings.  Their rank-r
column spaces are sealed before target residual scores/energies are loaded.
No controller is fitted, no new H query occurs, and all rank/seed controls are
reported.  Replaying a complete old tangent cache is a correctness/reference
diagnostic, NOT a certificate of production query or GPU-time savings.
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

from su2_nqs.benchmarks.collective_queries import (
    collective_subspace, fit_collective_update, audit_collective_update,
)
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.models.capacity_attention import CapacityControllerConfig, predict_capacity_coefficients
from su2_nqs.vmc.checkpoint import load_checkpoint
from .projected_capacity_attention import _screening_feature_means

SEEDS = (74121, 74122, 74123)
RANKS = (2, 4)
MODES = ('linear', 'signed_graph', 'attention', 'attention_scrambled')


def preserve_prior_cost_report(source: Path, output: Path) -> None:
    """Archive a preexisting deterministic-Q timing, never relabel as new."""
    report = dict(original_source=str(source),
                  original_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  newly_measured=False,
                  attribution='Previous same-day4-configuration full432-A1 deterministic-Q CPU canary. '
                    'Not timings of these learned Q subspaces or complete training.',
                  original_report=json.loads(source.read_text()))
    atomic_json(output/'prior_cost_canary.json', report)


def _scalar_inputs(cache: Path) -> dict:
    """Whitelist only scalar inputs; never read target scores or energy."""
    registration = json.loads((cache/'registration.json').read_text())
    with np.load(cache/'dataset.npz', allow_pickle=False) as source:
        panel = {name: source[name] for name in (
            'token_features', 'token_nodes', 'token_q', 'token_kind', 'pair_relations')}
    raw = panel['token_features']
    with np.load(cache/'raw_pair_direction_response.npz') as source:
        responses = source['raw_pair_direction_response']
    with np.load(cache/'screening_context.npz') as source:
        context = source['screening_context']
        scrambled = source['screening_context_scrambled']
    base = np.column_stack((raw, np.full(len(raw), registration['j2']), responses))
    panel.update(name=cache.name, path=str(cache), raw_features=raw,
                 features=np.column_stack((base, _screening_feature_means(panel, context))),
                 scrambled_features=np.column_stack((base, _screening_feature_means(panel, scrambled))),
                 relations=np.concatenate((panel['pair_relations'], context[..., None]), axis=-1),
                 scrambled_relations=np.concatenate((panel['pair_relations'], scrambled[..., None]), axis=-1))
    return panel


def _controller_embedding(panel: dict, mode: str, seed: int, controllers: Path) -> np.ndarray:
    """Readout Jacobian is the frozen learned action embedding, not a force."""
    restored = load_checkpoint(controllers/f'shared_{mode}_{seed}.pkl')
    metadata, params = restored['metadata'], restored['params']
    features = panel['scrambled_features'] if mode == 'attention_scrambled' else panel['features']
    relations = panel['scrambled_relations'] if mode == 'attention_scrambled' else panel['relations']
    features = np.where(np.asarray(metadata['active']),
        (features-np.asarray(metadata['mean']))/np.asarray(metadata['scale']), 0.)
    mixing = 'attention' if mode == 'attention_scrambled' else mode
    config = CapacityControllerConfig(features.shape[-1], width=4,
        relation_dim=relations.shape[-1], mixing=mixing)
    def output(readout):
        altered = dict(params, readout=readout)
        return predict_capacity_coefficients(altered, jnp.asarray(features), config,
                                             relations=jnp.asarray(relations))
    return np.asarray(jax.jacfwd(output)(params['readout']))


def run(cache_root: Path, controllers: Path, output: Path) -> dict:
    """Freeze Q using scalar inputs, then fit and audit only its r queries."""
    if output.exists() or not jax.config.x64_enabled:
        raise ValueError('fresh output path and x64 required')
    started = perf_counter()
    previous = json.loads((controllers/'registration.json').read_text())
    paths = [Path(p).parent for p in previous['cache_paths']]
    if any(not p.resolve().is_relative_to(cache_root.resolve()) for p in paths):
        raise ValueError('all frozen caches must belong to the declared cache root')
    training = paths[:previous['fit_count']]
    targets = paths[previous['fit_count']:]
    if len(training) != 6 or sorted(p.name for p in targets) != ['j0', 'j0125']:
        raise ValueError('six old parents and two frozen later parents required')
    diagonals, raw_train = [], []
    for path in training:
        with np.load(path/'dataset.npz') as source:
            scores = source['scores'][source['fit_mask'].astype(bool)]
            diagonals.append(np.mean(abs(scores-scores.mean(axis=0))**2, axis=0))
            raw_train.append(source['token_features'])
    diagonal = np.mean(diagonals, axis=0)
    # Frozen old-parent unit conversion only. No target full diagonal/force.
    coordinate_scale = 1/np.sqrt(np.maximum(diagonal, np.max(diagonal)*1e-12))
    raw_train = np.stack(raw_train)
    raw_mean, raw_std = raw_train.mean(axis=(0, 1)), raw_train.std(axis=(0, 1))
    raw_active = raw_std > 1e-12
    raw_std = np.where(raw_active, raw_std, 1.)
    subspaces, plans = {}, []
    for path in targets:
        panel = _scalar_inputs(path)
        embeddings = {'raw_0': np.where(raw_active,
            (panel['raw_features']-raw_mean)/raw_std, 0.)}
        for seed in SEEDS:
            embeddings[f'random_{seed}'] = np.random.default_rng(seed).normal(size=(24, 4))
            for mode in MODES:
                embeddings[f'{mode}_{seed}'] = _controller_embedding(panel, mode, seed, controllers)
        for label, embedding in embeddings.items():
            # Constant/bias direction is made available consistently to ALL
            # controls. Every vector generally touches every legal action.
            embedding = coordinate_scale[:, None]*np.column_stack((embedding, np.ones(24)))
            for rank in RANKS:
                q = collective_subspace(embedding, rank)
                key = f'{path.name}__{label}__r{rank}'
                subspaces[key] = q
                plans.append(dict(key=key, parent=path.name, label=label, rank=rank,
                    subspace_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                    all_24_actions_may_participate=True))
    output.mkdir(parents=True)
    np.savez_compressed(output/'frozen_subspaces.npz', **subspaces)
    registration = dict(task='frozen_collective_query_probe_v1', seeds=SEEDS, ranks=RANKS,
        fs_radius=.001, ridge=.01, train_parents=[str(p) for p in training],
        target_parents=[str(p) for p in targets], plans=plans,
        Q_uses_target_residual_scores_or_energies=False,
        target_screening_context_uses_fit_spin_samples=True,
        projector='existing verified coherent full432-image A1',
        action_dictionary='24 legal reduced-tensor delete/rotate directions; phase fixed',
        tokens='edge, doubled q and operation; scale metadata is constant9sites for these CT6 caches',
        Q_formula='left singular vectors_r(D_old^(-1/2) [d controller_output/d readout, 1])',
        raw_control='same old-parent standardization of46 local raw features, plus constant',
        linear_control='all89 v4 scalar features including local response/screening pooled features, plus constant; variance-truncated not energy-optimal',
        graph_and_attention='same89 scalar features and signed screening pair relation; same frozen250step old-parent fit',
        random_control='three frozen isotropic Gaussian action maps, plus constant',
        nonlinear_controller_hidden_width=4, controller_seeds=3,
        arm_count_per_rank_parent=16,
        full_reference_regularization='newtrace-isotropic ridge within all24 Euclidean coordinates; not prior diagonal-normalized SR baseline',
        coordinate_scale=coordinate_scale.tolist(), capacity_budget_applies=False,
        no_new_H_calls=True, no_NN_training=True, no_finite_energy_or_variance_claim=True,
        no_actual_query_saving_or_speedup_claim=True,
        qualifications=['All target H and scores are reused old cache, not fresh independent samples.',
            'Whole-chain fit/audit split from old cache; neither Q nor hyperparameters tuned with audit.',
            'Cached complete24 columns exist for exact-reference comparison; restricted SR sees OQ only.',
            'Q rotation invariance is NOT evidence of actual virtual U(d_q) gauge-equivariant learning.'])
    atomic_json(output/'registration.json', registration)
    # Q is now sealed on disk. Target derivative/energy data read only below.
    rows, fitted_directions = [], {}
    for path in targets:
        with np.load(path/'dataset.npz') as source:
            scores, energies, fit, labels = (source[name] for name in
                ('scores', 'local_energies', 'fit_mask', 'chain_labels'))
        fit = fit.astype(bool)
        fit_chains, audit_chains = set(labels[fit].tolist()), set(labels[~fit].tolist())
        if fit_chains & audit_chains:
            raise ValueError('whole-chain split is contaminated')
        for plan in [p for p in plans if p['parent'] == path.name]+[
                dict(key=f'{path.name}__full_reference__r24', parent=path.name,
                     label='full_reference', rank=24)]:
            q = subspaces[plan['key']] if plan['rank'] != 24 else np.eye(24)
            queried = scores @ q
            fitted = fit_collective_update(queried[fit], energies[fit], ridge=.01, fs_radius=.001)
            audit = audit_collective_update(fitted.coefficients, queried[~fit], energies[~fit], labels[~fit])
            actual_fit_fs = np.sqrt(max(float(fitted.coefficients @ fitted.gram @ fitted.coefficients), 0.))
            fit_slope = float(2*fitted.force @ fitted.coefficients)
            direction = q @ fitted.coefficients
            # Independent cache-reference identity; not used by the fit.
            reference = scores @ direction
            identity = float(np.max(abs(reference-queried @ fitted.coefficients)))
            fitted_directions[plan['key']] = direction
            rows.append(dict(**plan, fit_fs=float(actual_fit_fs), fit_slope_total=fit_slope,
                audit=audit, audit_fit_fs_ratio=audit['fs_tangent']/actual_fit_fs if actual_fit_fs else 0.,
                actual_audit_fs_matched_25percent=bool(abs(audit['fs_tangent']/.001-1) <= .25),
                exact_cache_restriction_max_error=identity,
                physical_fit_queries=plan['rank'], shared_cached_parent_H_centers=int(fit.sum())))
    np.savez_compressed(output/'fitted_directions.npz', **fitted_directions)
    paired = []
    for parent in (p.name for p in targets):
        for rank in RANKS:
            lookup = {row['label']: row for row in rows if row['parent'] == parent and row['rank'] == rank}
            for control in ('raw', 'linear', 'signed_graph', 'random', 'attention_scrambled'):
                differences, jackknives = [], []
                for seed in SEEDS:
                    first = lookup[f'attention_{seed}']['audit']
                    second = lookup[f'{control}_{0 if control == "raw" else seed}']['audit']
                    differences.append(first['energy_slope_per_fs_total']-second['energy_slope_per_fs_total'])
                    jackknives.append(np.asarray(first['energy_slope_per_fs_total_chain_jackknife'])
                                      -np.asarray(second['energy_slope_per_fs_total_chain_jackknife']))
                jackknife = np.mean(jackknives, axis=0)
                error = np.sqrt((len(jackknife)-1)/len(jackknife)*np.sum((jackknife-jackknife.mean())**2))
                paired.append(dict(parent=parent, rank=rank, control=control,
                    seed_differences= differences, seed_average_difference=float(np.mean(differences)),
                    paired_chain_SE=float(error), negative_favors_attention=True))
    result = dict(registration=registration, rows=rows, paired_attention_controls=paired,
                  wall_seconds=perf_counter()-started,
                  maximum_cache_identity_error=max(r['exact_cache_restriction_max_error'] for r in rows),
                  production_updates=0, new_H_calls=0,
                  conclusion='Development-only sampled tangent comparison. No finite energies or variances, speedup, or variational win established.')
    atomic_json(output/'results.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--controllers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-cost-report', type=Path)
    args = parser.parse_args()
    result = run(args.cache_root, args.controllers, args.output)
    if args.prior_cost_report is not None:
        preserve_prior_cost_report(args.prior_cost_report, args.output)
    print(json.dumps(dict(seconds=result['wall_seconds'], rows=len(result['rows']),
                          exact_error=result['maximum_cache_identity_error']), indent=2))


if __name__ == '__main__':
    main()
