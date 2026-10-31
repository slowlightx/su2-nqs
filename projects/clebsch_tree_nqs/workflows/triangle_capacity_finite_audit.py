"""Local full-H checks of frozen projected-capacity controller proposals.

The input decisions must be frozen before this diagnostic. Reuse old parent
Born configurations with proper importance weights and newly evaluated H on
EACH finite candidate; parent local energies alone are not candidate actions.
This is not new independent sampling, a production update, or an energy
benchmark endpoint. All candidates and rejected calibrations are retained.
"""
from __future__ import annotations

import argparse
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
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.carrier_protection import physical_moments
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

from .triangle_projected_capacity import (
    capacity_coordinate_family, coordinate_bank, make_capacity_amplitude,
)


def paired_product_statistics(parent: np.ndarray, hparent: np.ndarray,
                              candidate: np.ndarray, hcandidate: np.ndarray,
                              chains: np.ndarray, *, n_sites: int = 36) -> dict:
    """Full-product energy/variance differences with a paired chain jackknife.

    Parent Born support is reused; weight on exact parent nodes is not
    certified. This method does NOT substitute r^2 times OLD local energy.
    """
    p, hp, c, hc = [np.asarray(x, complex) for x in (parent, hparent, candidate, hcandidate)]
    labels = np.asarray(chains)
    if (p.ndim != 1 or any(a.shape != p.shape for a in (hp, c, hc, labels))
            or not all(np.isfinite(a).all() for a in (p, hp, c, hc)) or np.any(p == 0)):
        raise ValueError('matching finite full products and nonzero parent samples required')
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 4 or np.any(counts != counts[0]):
        raise ValueError('four or more equal independent chain groups required')
    density = abs(p)**2
    before = physical_moments(p, hp, density, n_sites)
    after = physical_moments(c, hc, density, n_sites)
    change = np.array([after['energy_per_site']-before['energy_per_site'],
                       after['variance_per_site']-before['variance_per_site']])
    leave = []
    for chain in unique:
        keep = labels != chain
        a = physical_moments(p[keep], hp[keep], density[keep], n_sites)
        b = physical_moments(c[keep], hc[keep], density[keep], n_sites)
        leave.append([b['energy_per_site']-a['energy_per_site'],
                      b['variance_per_site']-a['variance_per_site']])
    leave = np.asarray(leave)
    se = np.sqrt((len(unique)-1)/len(unique)*np.sum((leave-leave.mean(axis=0))**2, axis=0))
    return dict(parent=before, candidate=after,
                energy_change_per_site=float(change[0]), variance_change_per_site=float(change[1]),
                paired_energy_change_SE=float(se[0]), paired_variance_change_SE=float(se[1]),
                samples=len(p), chains=len(unique))


def run(decisions: Path, output: Path, *, maximum_seconds: float = 900.) -> dict:
    """One frozen local comparison, using a new H-action for every candidate."""
    if not jax.config.x64_enabled or output.exists() or maximum_seconds <= 0:
        raise ValueError('x64, fresh output and positive bounded runtime required')
    started = perf_counter()
    chosen = json.loads(decisions.read_text())
    cache = Path(chosen['cache_directory'])
    registration = json.loads((cache/'registration.json').read_text())
    parent = Path(registration['parent_checkpoint']).parent
    saved = load_checkpoint(parent/'checkpoint.pkl')
    config = MixedClebschConfig(**registration['model'])
    if (pytree_sha256(saved['params']) != registration['parent_params_sha256']
            or file_sha256(parent/'checkpoint.pkl') != registration['parent_checkpoint_sha256']):
        raise ValueError('actual parent identity changed')
    canonical, deltas, _ = capacity_coordinate_family(saved['params'], config)
    if pytree_sha256(canonical) != registration['canonical_bank_sha256']:
        raise ValueError('canonical coordinates differ from the frozen tangent dataset')
    with np.load(cache/'dataset.npz') as source:
        bits, labels, fit = source['bits'], source['chain_labels'], source['fit_mask']
        old_amplitudes, cached_energy = source['amplitudes'], source['local_energies']
    audit_ids = np.array([np.flatnonzero((labels == chain) & ~fit)[0]
                          for chain in np.unique(labels[~fit])])
    if len(audit_ids) < 4 or len(audit_ids) > 32:
        raise ValueError('one fixed record per four to32 held-out chains required')
    proposed = {name: np.asarray(values, float) for name, values in chosen['coefficients'].items()}
    if (not 1 <= len(proposed) <= 4 or any(theta.shape != (24,) or not np.isfinite(theta).all()
                                           for theta in proposed.values())):
        raise ValueError('at most four frozen24-dimensional candidates required')
    plan = dict(task='finite_full_H_capacity_diagnostic_v1', decision_sha256=file_sha256(decisions),
                parent_params_sha256=registration['parent_params_sha256'],
                cached_dataset_sha256=file_sha256(cache/'dataset.npz'),
                candidate_names=list(proposed), candidate_coefficients={k: v.tolist() for k, v in proposed.items()},
                target_fit_fs=.001, fit_relative_tolerance=.02, maximum_fit_queries=4,
                calibration_policy='fit-only scalar normalization; never tune with held-out rows',
                heldout_record_indices=audit_ids.tolist(), maximum_seconds=maximum_seconds,
                reuse_old_audit=True, no_production_update=True)
    plan['sha256'] = canonical_sha256(plan)
    output.mkdir(parents=True)
    atomic_json(output/'registration.json', plan)
    amp = make_capacity_amplitude(canonical, deltas, config)
    evaluate = jax.jit(lambda theta, rows: jax.lax.map(lambda row: amp(theta, row), rows))
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = triangular_j1j2_bonds(6, 6, j2=registration['j2'], periodic=True)
    action = make_cached_projected_heisenberg_action(config, images, bonds, projection_chunk=4)

    @jax.jit
    def complete_action(prepared, rows):
        return jax.lax.map(lambda row: jnp.stack([v[0] for v in action(prepared, row[None])]), rows)

    def products(theta: np.ndarray) -> np.ndarray:
        bank = coordinate_bank(canonical, deltas, theta)
        return np.asarray(complete_action(prepare_feature_lut_bank(bank, config),
                                         jnp.asarray(bits[audit_ids])).block_until_ready())

    parent_tick = perf_counter()
    parent_products = products(np.zeros(24))
    identity = float(np.max(abs(parent_products[:, 1]/parent_products[:, 0]-cached_energy[audit_ids])))
    if identity > 2e-10:
        raise ArithmeticError('recomputed parent full H disagrees with the original sealed audit')
    parent_seconds = perf_counter()-parent_tick
    outcomes, arrays = [], dict(parent_products=parent_products, audit_bits=bits[audit_ids],
                                audit_chain_labels=labels[audit_ids])
    fit_rows = jnp.asarray(bits[fit]); fit_old = old_amplitudes[fit]
    for name, theta in proposed.items():
        if perf_counter()-started > maximum_seconds:
            outcomes.append(dict(name=name, status='not_run_local_budget_exhausted'))
            continue
        trials = []
        factor, accepted = 1., False
        for _ in range(plan['maximum_fit_queries']):
            values = np.asarray(evaluate(jnp.asarray(theta*factor), fit_rows).block_until_ready())
            diagnostics = jax.tree.map(float, sampled_overlap_diagnostics(
                jnp.log(jnp.asarray(values/fit_old, jnp.complex128))))
            trials.append(dict(factor=factor, **diagnostics))
            fs = diagnostics['sampled_fs_sine']
            if abs(fs/.001-1) <= .02 and diagnostics['reweighting_ess_fraction'] >= .98:
                accepted = True
                break
            if not np.isfinite(fs) or fs < 1e-12:
                break
            factor *= float(np.clip(.001/fs, .25, 4.))
        if not accepted:
            outcomes.append(dict(name=name, status='fit_radius_failed', fit_trials=trials))
            continue
        tick = perf_counter(); candidate = products(theta*factor); seconds = perf_counter()-tick
        overlap = jax.tree.map(float, sampled_overlap_diagnostics(jnp.log(jnp.asarray(
            candidate[:, 0]/parent_products[:, 0], jnp.complex128))))
        statistics = paired_product_statistics(parent_products[:, 0], parent_products[:, 1],
            candidate[:, 0], candidate[:, 1], labels[audit_ids])
        outcomes.append(dict(name=name, status='full_H_diagnostic_complete', factor=factor,
                             fit_trials=trials, heldout_distance=overlap,
                             comparable_heldout_FS=abs(overlap['sampled_fs_sine']/.001-1) <= .25,
                             **statistics, full_H_centers=len(audit_ids), H_seconds=seconds))
        arrays[name+'_products'] = candidate
        arrays[name+'_finite_coordinates'] = theta*factor
        print(name, statistics['energy_change_per_site'], statistics['variance_change_per_site'], flush=True)
    np.savez_compressed(output/'full_products.npz', **arrays)
    report = dict(status='completed_bounded_finite_H_diagnostic', registration_sha256=plan['sha256'],
                  parent_full_H_relative_identity_max_absolute_error=identity,
                  parent_H_seconds_including_compile=parent_seconds, results=outcomes,
                  full_H_centers=len(audit_ids)*(1+sum(r['status'] == 'full_H_diagnostic_complete' for r in outcomes)),
                  wall_seconds=perf_counter()-started, production_updates=0,
                  qualification='Old held-out parent-Born records, not new independent chains. '
                  'Actual H applied to every finite candidate. Finite support and32chains limit inference; '
                  'no method/energy benchmark win from selecting after these results.')
    atomic_json(output/'results.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--decisions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--maximum-seconds', type=float, default=900.)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    print(json.dumps(run(args.decisions, args.output, maximum_seconds=args.maximum_seconds), indent=2))


if __name__ == '__main__':
    main()
