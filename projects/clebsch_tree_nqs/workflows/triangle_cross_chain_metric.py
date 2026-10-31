"""Amplitude-only frozen-panel cross-chain metric diagnostic at a saved CT.

Reuse existing audited physical configurations and cached H ratios; never
sample, evaluate H, update/checkpoint parameters, or alter a remote job.
All tangent and finite-amplitude queries use the complete coherent A1 sum.
The four columns below are a DIAGNOSTIC panel, not additional model heads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.cross_chain_tangent import (
    chain_fold_metrics, consensus_tangent_direction, fixed_direction_report,
)
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics


def frozen_panel(params: dict) -> tuple[dict, ...]:
    """Three sector rescalings and a known constant-amplitude gauge direction."""
    directions = []
    for qindex in (0, 1):
        delta = jax.tree.map(jnp.zeros_like, params)
        delta['local'] = tuple(tuple(value if index == qindex else jnp.zeros_like(value)
            for index, value in enumerate(node)) for node in params['local'])
        directions.append(delta)
    delta = jax.tree.map(jnp.zeros_like, params)
    top = list(delta['top']); root = params['top'][-1][0]
    # Root packed doubled-charge order is (0,2,4,6,8,10), exact CG convention.
    if root.shape[0] != 6:
        raise ValueError('this frozen probe requires the registered qmax10 root')
    top[-1] = (jnp.zeros_like(root).at[1].set(root[1]),)
    delta['top'] = tuple(top); directions.append(delta)
    delta = jax.tree.map(jnp.zeros_like, params)
    top = list(delta['top']); top[-1] = params['top'][-1]
    delta['top'] = tuple(top); directions.append(delta)
    return tuple(directions)


def run(parent: Path, output: Path, *, records_per_chain: int = 1,
        cached_panel: Path | None = None) -> dict:
    """Frozen 128-chain cached-record probe; no new samples or production release."""
    start = perf_counter()
    if not jax.config.x64_enabled or output.exists() or records_per_chain not in (1, 8):
        raise ValueError('x64 before loading and a fresh output directory required')
    registration = json.loads((parent/'registration.json').read_text())
    checkpoint = load_checkpoint(parent/'checkpoint.pkl'); params = checkpoint['params']
    audit = json.loads((parent/'terminal_audit.json').read_text())
    cfg = MixedClebschConfig(**registration['model'])
    if (cfg.length != 6 or cfg.multiplicity_cap != 20 or cfg.maximum_q != 10
            or registration['arity'] != 'ternary_joint'
            or registration['j2_over_j1'] != .125
            or pytree_sha256(params) != audit['params_sha256']):
        raise ValueError('probe requires the sealed latest joint w20 J.125 parent')
    data = np.load(parent/'terminal_audit.npz')
    count = 128*records_per_chain
    bits = data['bits'][:count]; labels = data['chain_labels'][:count]
    energies = data['local_energies'].reshape(-1)[:count]
    if (bits.shape != (count, 36) or not np.array_equal(labels, np.tile(np.arange(128), records_per_chain))
            or not np.isfinite(energies).all()):
        raise ValueError('need one pre-existing record per independent labelled audit chain')
    fit = np.flatnonzero(labels < 96); test = np.flatnonzero(labels >= 96)
    folds = tuple(tuple(range(first, first+24)) for first in range(0, 96, 24))
    panel = frozen_panel(params)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    assert len(images) == 432
    prepare = lambda p: prepare_feature_lut_bank(arity_clebsch_bank(
        p, cfg, 'ternary_joint', registration['binary_rank']), cfg)
    projected = make_feature_lut_projected_amplitude(cfg, images, chunk=4)

    @jax.jit
    def score_rows(p, rows):
        prepared, linear = jax.linearize(prepare, p)
        tangents = tuple(linear(direction) for direction in panel)
        def one(row):
            value, action = jax.linearize(lambda bank: projected(bank, row), prepared)
            return value, jnp.stack([action(tangent)/value for tangent in tangents])
        return jax.lax.map(one, rows, batch_size=1)

    @jax.jit
    def evaluate(p, rows):
        prepared = prepare(p)
        return jax.lax.map(lambda row: projected(prepared, row), rows, batch_size=1)

    before = pytree_sha256(params); tick = perf_counter()
    if cached_panel is None:
        values, scores = score_rows(params, jnp.asarray(bits))
        jax.block_until_ready(scores)
    else:
        old = json.loads((cached_panel/'results.json').read_text())
        old_data = np.load(cached_panel/'tangent_panel.npz')
        if (old['params_sha256'] != before or old['samples'] != count
                or old['checkpoint_sha256'] != file_sha256(parent/'checkpoint.pkl')
                or not np.array_equal(old_data['bits'], bits)
                or not np.array_equal(old_data['chain_labels'], labels)
                or old_data['scores'].shape != (count, 4)):
            raise ValueError('cached complete-amplitude tangent panel identity differs')
        values, scores = old_data['amplitudes'], old_data['scores']
    score_seconds = perf_counter()-tick
    values = np.asarray(values); scores = np.asarray(scores)
    if not np.isfinite(scores).all() or np.any(values == 0):
        raise FloatingPointError('complete physical tangent at a carrier node/nonfinite; no floor')
    gauge_defect = float(np.max(np.abs(scores[:, -1]-1.)))
    if gauge_defect > 2e-11:
        raise ArithmeticError('global root rescaling must have constant physical score1')
    metrics = chain_fold_metrics(scores[fit], labels[fit], folds)
    centered = scores[fit]-scores[fit].mean(axis=0)
    force = 2*np.real(centered.conj().T @ (energies[fit]-energies[fit].mean()))/len(fit)
    frontier = []
    for strength in (0., 1., 4.):
        direction = consensus_tangent_direction(metrics, force,
            disagreement_strength=strength, target_tangent_fs=.001)
        c = direction.coefficients
        proposed = jax.tree.map(lambda original, *terms: original+
            sum(coefficient*term for coefficient, term in zip(c, terms, strict=True)), params, *panel)
        tick = perf_counter(); changed = np.asarray(evaluate(proposed, jnp.asarray(bits)))
        finite = {}
        for name, rows in (('fit', fit), ('reserved_test', test)):
            finite[name] = jax.tree.map(lambda v: float(v), sampled_overlap_diagnostics(
                jnp.log(jnp.asarray(changed[rows]/values[rows], jnp.complex128))))
        # Diagnose whether a few chains carry the estimated physical metric.
        leverage = {}
        for name, rows in (('fit', fit), ('reserved_test', test)):
            tangent = scores[rows] @ c
            weights = np.abs(tangent-tangent.mean())**2
            chain_weights = np.array([weights[labels[rows] == chain].sum()
                                      for chain in np.unique(labels[rows])])
            leverage[name] = dict(max_chain_metric_fraction=float(chain_weights.max()/chain_weights.sum()),
                largest_five_chains_metric_fraction=float(np.sort(chain_weights)[-5:].sum()/chain_weights.sum()),
                metric_contribution_effective_chains=float(chain_weights.sum()**2/np.sum(chain_weights**2)),
                qualification='leverage diagnostic only, not Markov-chain effective sample size')
        frontier.append(dict(strength=strength, coefficients=c.tolist(),
            physical_rank=direction.physical_rank, null_force_norm=direction.null_force_norm,
            fit_fold_tangent_fs=direction.predicted_fold_fs.tolist(),
            common_tangent_fs=direction.predicted_common_fs,
            fit_cached_H_energy_slope_total=direction.directional_slope,
            whitened_disagreement_eigenvalues=direction.whitened_disagreement_eigenvalues.tolist(),
            reserved_linear_report=fixed_direction_report(scores[test], c, reference_fs=.001),
            finite_amplitude=finite, chain_metric_leverage=leverage,
            finite_amplitude_seconds=perf_counter()-tick))
    assert pytree_sha256(params) == before
    output.mkdir(parents=True)
    np.savez_compressed(output/'tangent_panel.npz', bits=bits, chain_labels=labels,
        cached_local_energies=energies, amplitudes=values, scores=scores,
        fit_chain_labels=np.arange(96), reserved_test_chain_labels=np.arange(96, 128),
        fold_metrics=metrics, fit_force=force)
    report = dict(task='frozen_chain_metric_consensus_v2_common_center', parent=str(parent),
        checkpoint_sha256=file_sha256(parent/'checkpoint.pkl'), params_sha256=before,
        absolute_completed_steps=checkpoint['metadata']['absolute_completed_steps'],
        audit_sha256=file_sha256(parent/'terminal_audit.npz'),
        samples=count, cached_records_per_chain=records_per_chain,
        fit_chains=96, reserved_test_chains=32, coherent_A1_images=len(images),
        frozen_panel=['all_local_q1_rescaling', 'all_local_q3_rescaling',
                      'root_q2_rescaling', 'global_root_scale_GAUGE'],
        strengths_frozen_before_test=[0., 1., 4.], target_tangent_fs=.001,
        gauge_constant_score_max_defect=gauge_defect, score_seconds=score_seconds,
        cached_tangent_source=None if cached_panel is None else str(cached_panel),
        common_center_preserves_between_chain_means=True,
        maximum_mean_fold_vs_pooled_metric_error=float(np.max(np.abs(metrics.mean(axis=0)-
            ((scores[fit]-scores[fit].mean(axis=0)).conj().T @
             (scores[fit]-scores[fit].mean(axis=0))).real/len(fit)))),
        frontier=frontier, full_H_new_calls=0, training_updates=0,
        parent_unchanged=True, wall_seconds=perf_counter()-start,
        qualification='Development only. Existing audit chains reused; test means reserved within this new diagnostic, not globally untouched. No candidate H/energy audit, no selection on reserved chains, no production changes. A frozen three-dimensional tangent panel is not complete-network SR.')
    atomic_json(output/'results.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--records-per-chain', type=int, choices=(1, 8), default=1)
    parser.add_argument('--cached-panel', type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600 second no-H development ceiling')))
    signal.alarm(600)
    print(json.dumps(run(args.parent, args.output, records_per_chain=args.records_per_chain,
                         cached_panel=args.cached_panel), indent=2))


if __name__ == '__main__':
    main()
