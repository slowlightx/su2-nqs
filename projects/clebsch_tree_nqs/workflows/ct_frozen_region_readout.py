"""Complete-A1 CT physical region readout on an immutable IID audit prefix.

No training, new sampling or H query. Evaluate actual off-diagonal region
amplitudes on the saved proposal's environments; retain exact nodes, original
Q and endpoint identity. This is a new exploratory observable calculation on
consumed audit rows, NOT a fresh independent energy/variance certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

from su2_nqs.benchmarks.cached_region_amplitudes import (
    cached_region_readout, reconstruct_region_queries, region_completion_rows,
)
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_projected_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from .ct6_native_direction_calibration import normalized_raw_bank

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'projects/clebsch_tree_nqs/results/ct_status_20261028/verified_next_parents'
CAMPAIGN = ROOT/'projects/clebsch_tree_nqs/results/su2_mechanism_local_20261028'
SITES = (12, 13, 14, 18, 19, 20)


def run(output: Path, *, records: int = 512, seconds_cap: float = 600.,
        couplings: tuple[float, ...] = (0., .125)) -> dict:
    """Read fixed J0/global466 and J.125/global492, full432 spatial images."""
    if records != 512 or seconds_cap != 600. or couplings not in ((0., .125), (0.,), (.125,)):
        raise ValueError('registered exploratory prefix512 and600CPUwall seconds required')
    if output.exists():
        raise FileExistsError('new immutable output required')
    output.mkdir(parents=True)
    sources = dict(
        operators=CAMPAIGN/'interventions/frozen_operators.npz',
        reference_projector=CAMPAIGN/'regions/frozen_projectors.npz')
    operator_seals = {k: file_sha256(v) for k, v in sources.items()}
    source_manifest = transitive_local_source_manifest([Path(__file__)], root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    with np.load(sources['operators']) as data:
        operators = {k: data[k] for k in data.files}
    with np.load(sources['reference_projector']) as data:
        operators['frozen_DMRG_singlet_main_projector'] = data['compact6_q0_reference_main_projector']
    atomic_json(output/'registration.json', dict(task='CT_full_A1_cached_compact6_readout_v1',
        sites=SITES, records=records, groups=32, seconds_cap=seconds_cap,
        endpoints={('ct6_j0' if j2 == 0. else 'ct6_j0125'): (466 if j2 == 0. else 492) for j2 in couplings},
        operators=operator_seals, source_manifest=source_manifest,
        new_sampler_records=0, new_H_centers=0, PSD_repair=False,
        not_a_new_untouched_audit=True, existing_rows_consumed_for_physics_exploration=True))
    config = MixedClebschConfig(6, 28, 10)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    if len(images) != 432:
        raise ArithmeticError('complete registered space group missing')
    scalar = make_feature_lut_projected_amplitude(config, images, chunk=4)
    batch = jax.jit(lambda bank, rows: jax.lax.map(lambda row: scalar(bank, row), rows, batch_size=4))
    started = perf_counter(); results = []
    for j2 in couplings:
        tag, step = ('ct6_j0', 466) if j2 == 0. else ('ct6_j0125', 492)
        item = dict(tag=tag, global_step=step, status='started', timings={})
        directory = BASE/tag; checkpoint = directory/'parent.pkl'
        audit_path = directory/'parent_snapshot/audit.json'
        product_path = directory/'parent_snapshot/products.npz'
        saved = load_checkpoint(checkpoint); native = saved['params']
        audit = json.loads(audit_path.read_text())
        if audit['endpoint_params_sha256'] != pytree_sha256(native):
            raise ValueError('saved IID endpoint does not match the physical CT parent')
        arity = 'binary_truncated' if tag.endswith('j0125') else 'ternary_joint'
        tick = perf_counter()
        bank, note = normalized_raw_bank(native, dict(cfg=config,
            bank=lambda p: arity_clebsch_bank(p, config, arity, 8)))
        prepared = prepare_feature_lut_bank(bank, config)
        item['timings']['canonical_preparation'] = perf_counter()-tick
        with np.load(product_path) as data:
            rows = data['rows'][:records]; golden = data['psi'][:records, 1]
            q = data['proposal'][:records]
            guide = data['guide_components'][:records]
            reconstructed_q = full_support_sz0_density(guide[:, :, 0].mean(1), guide[:, :, 1].mean(1),
                n_sites=36, epsilon=.125)
            if not np.allclose(q, reconstructed_q, atol=0., rtol=2e-14):
                raise ValueError('original IID proposal density identity changed')
        tick = perf_counter(); values = np.asarray(batch(prepared, rows[:4])); item['timings']['compile_four_rows'] = perf_counter()-tick
        relative = float(np.max(abs(values-golden[:4]))/max(np.max(abs(golden[:4])), 1e-300))
        if not np.isfinite(relative) or relative > 2e-11:
            raise ArithmeticError('complete-A1 cache amplitude/normalization replay failed')
        tick = perf_counter(); np.asarray(batch(prepared, rows[:4])); warm = (perf_counter()-tick)/4
        queries, lookup = region_completion_rows(rows, SITES)
        item.update(complete_A1_images=len(images), cached_parent_replay_relative_error=relative,
            estimated_warm_seconds_per_amplitude=warm, unique_region_amplitude_queries=len(queries),
            checkpoint_sha256=file_sha256(checkpoint), source_products_sha256=file_sha256(product_path),
            source_audit_sha256=file_sha256(audit_path), parameter_sha256=pytree_sha256(native),
            normalization=note)
        atomic_json(output/(tag+'_progress.json'), item)
        if perf_counter()-started + 1.5*warm*len(queries) > seconds_cap-15:
            item['status'] = 'cost_gate_before_region_queries'; results.append(item); continue
        tick = perf_counter(); pieces = []
        for first in range(0, len(queries), 128):
            if perf_counter()-started > seconds_cap-10:
                item['status'] = 'bounded_prefix_not_physics_estimate'; break
            pieces.append(np.asarray(batch(prepared, queries[first:first+128])))
        item['timings']['offdiagonal_region_amplitudes'] = perf_counter()-tick
        if item.get('status') == 'bounded_prefix_not_physics_estimate':
            results.append(item); continue
        query_values = np.concatenate(pieces)
        phi = reconstruct_region_queries(query_values, lookup)
        packed = rows[:, SITES] @ (1 << np.arange(5, -1, -1))
        error = float(np.max(abs(phi[np.arange(records), packed]-golden))/max(np.max(abs(golden)), 1e-300))
        if error > 2e-11:
            raise ArithmeticError('all512 cached parent amplitudes must replay before physics')
        report, rho = cached_region_readout(rows, SITES, phi, q, operators, groups=32)
        np.savez_compressed(output/(tag+'_physical_region.npz'), rows=rows, sites=np.asarray(SITES),
            proposal=q, region_amplitudes=phi, rho_Hermitian=rho,
            query_rows=queries, query_values=query_values)
        item.update(status='complete_exploratory_full_A1_readout', all_parent_amplitude_relative_error=error,
            readout=report, actual_new_H_queries=0, actual_new_sampler_records=0)
        results.append(item); atomic_json(output/(tag+'_results.json'), item)
    report = dict(status='local_complete_or_explicit_cost_gate', results=results,
        wall_seconds=perf_counter()-started, strict_SU2_state=True,
        point_RDM_not_forced_SU2_or_PSD=True, comparison_scope='same physical compact6 operators, not raw tree spectra',
        source_operators_unchanged=all(file_sha256(sources[k]) == h for k, h in operator_seals.items()),
        numerical_sources_unchanged=all(file_sha256(ROOT/k) == h for k, h in source_manifest.items()))
    atomic_json(output/'results.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--j2', type=float, choices=(0., .125))
    args = parser.parse_args()
    print(json.dumps(run(args.output, couplings=(args.j2,) if args.j2 is not None else (0., .125))), flush=True)
