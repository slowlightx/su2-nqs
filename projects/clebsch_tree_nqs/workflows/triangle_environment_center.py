"""Two-sided 9->18 response expansion, full-center solve and same-center q-SVD.

The actual 126-site complement is used, and is itself expanded inside its
18+108 child support. The complete finite projected Hamiltonian is solved;
physical leakage outside those retained children and coherent A1 quality
remain separate measurements. No attention or small-center Ritz surrogate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from scipy.linalg import eigh

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.su2_complement import descend_singlet_complement, restore_singlet_parent
from su2_nqs.operators.su2_conditional_response import (
    ConditionalResponseFactor, make_conditional_cut_response, nested_response_expansion,
    root_coefficient_matrices,
)
from su2_nqs.operators.su2_sector_merge import project_sector_maps
from su2_nqs.operators.su2_singlet_center import SingletCutOperator, compress_singlet_center
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_boundary_response_enrichment import TREE
from .triangle_hierarchy_initialization import native_bank_from_state, split_bond_ledger, source_channels
from .triangle_lower_path import ROOT, clone, cross_bonds, real_maps, reproject
from .triangle_sector_transfer import load_source
from .triangle_upper_sweep import raw_layer_energy


def transpose_response(response: ConditionalResponseFactor) -> ConditionalResponseFactor:
    """Exchange ket regions: (-1)^q Y.T, NOT the Hermitian adjoint of Y."""
    phase = (-1)**response.target_q
    return ConditionalResponseFactor(response.target_q, response.bath_channels,
        response.target_channels, tuple((phase*z.T, a.T) for a, z in response.terms),
        response.source_coefficients, dict(response.diagnostics, ket_singlet_regions_exchanged=True))


def run(output: Path, *, new_width: int = 16) -> dict:
    """Freeze then completely optimize one two-sided finite-support center."""
    if new_width not in (12, 16, 20):
        raise ValueError('registered new low-q multiplicity width is12,16,or20')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('use a new immutable result directory')
    parent_path = ROOT/'projects/clebsch_tree_nqs/results/ct_boundary_multisector_20261026/boundary_q024_plus4/state.pkl'
    registry = dict(task='CT12_two_sided_environment_full_center_then_same_center_q_SVD',
        j2=.125, target_node=0, target_sites=18, environment_sites=126,
        low_q=(0, 2, 4), target_new_width=new_width,
        parent=str(parent_path), parent_sha256=file_sha256(parent_path),
        old_columns_protected_before_optimization=True,
        response='complete crossing H inside 9x9 target and18x108 environment support; includes simultaneous two-side leakage',
        physical_bath_leakage_outside_these_children=False,
        solver='complete reduced singlet matrix, direct Hermitian lowest eigenpair',
        no_small_Krylov_surrogate=True, no_MC_or_full_A1_queries=True,
        source_sha256={str(p): file_sha256(p) for p in (Path(__file__),
            ROOT/'su2_nqs/operators/su2_singlet_center.py', ROOT/'tests/test_su2_singlet_center.py')})
    atomic_json(output/'registration.json', registry)
    start = perf_counter(); cpu = process_time()
    result = dict(registration=registry, stages=[], endpoints={})
    def stage(name: str, **values):
        result['stages'].append(dict(name=name, seconds=perf_counter()-start, **values))
        atomic_json(output/'progress.json', result)
        print(name, round(perf_counter()-start, 2), flush=True)

    source, provenance = load_source(.125)
    result['source'] = provenance
    parent = load_checkpoint(parent_path)['params']
    supports = TREE.coarse_supports
    _, joins, bonds = split_bond_ledger(12, .125)
    initial = raw_layer_energy(parent, source, .125)
    result['parent_raw_energy'] = initial
    roots = root_coefficient_matrices(parent['maps'][14][0], parent['channels'][14][0])
    env36 = descend_singlet_complement(parent['blocks'][24], parent['blocks'][25], parent['blocks'][29],
        parent['maps'][12], roots, cross_bonds(supports[25], supports[29], bonds))
    support108 = supports[25]+supports[29]
    env18 = descend_singlet_complement(parent['blocks'][16], parent['blocks'][17], env36.complement,
        parent['maps'][8], env36.coefficients, cross_bonds(supports[17], support108, bonds))
    support126 = supports[17]+support108
    stage('complete_current_complements_ready')
    old_widths = {q: w.shape[1] for q, w in parent['maps'][0].items()}
    a_maps = {q: w.copy() for q, w in parent['maps'][0].items()}
    e_maps = {q: w.copy() for q, w in env18.complement_maps.items()}
    responses = {}
    for q in registry['low_q']:
        response = make_conditional_cut_response(a_children=(parent['blocks'][0], parent['blocks'][1]),
            a_source=parent['blocks'][16], a_maps=parent['maps'][0], b=parent['blocks'][17],
            d=env36.complement, parent_maps=parent['maps'][8], root_coefficients=env36.coefficients,
            cross_bonds_ab=joins[8], cross_bonds_ad=cross_bonds(supports[16], support108, bonds),
            target_q=q, center=False, memory_limit_bytes=768*1024**2)
        a = nested_response_expansion(q, a_maps[q], response,
            additional=new_width-a_maps[q].shape[1], seed=2610279300+q)
        e = nested_response_expansion(q, e_maps[q], transpose_response(response),
            additional=new_width-e_maps[q].shape[1], seed=2610279400+q)
        a_maps[q] = real_maps({q: a.frame})[q]
        e_maps[q] = real_maps({q: e.frame})[q]
        # Joint QQ leakage is explicitly present in the response used to
        # obtain BOTH frames; neither side was projected into the old other.
        qae = response.matmat(e_maps[q])
        qq = qae-a_maps[q][:, :old_widths[q]]@(a_maps[q][:, :old_widths[q]].T@qae)
        responses[q] = dict(factor=dict(response.diagnostics), target=dict(a.diagnostics),
            environment=dict(e.diagnostics), target_added_singular_values=a.added_singular_values.tolist(),
            environment_added_singular_values=e.added_singular_values.tolist(),
            target_Q_response_in_expanded_environment_norm=float(np.linalg.norm(qq)))
        stage('two_sided_response_q'+str(q), response_shape=response.shape)
    result['responses'] = responses
    a_block = project_sector_maps(parent['blocks'][0], parent['blocks'][1], joins[0], maps=a_maps,
        compact_boundary=True, max_output_dimension=new_width*36, memory_limit_bytes=768*1024**2)
    e_block = project_sector_maps(parent['blocks'][17], env36.complement,
        cross_bonds(supports[17], support108, bonds), maps=e_maps, compact_boundary=True,
        max_output_dimension=new_width*36, memory_limit_bytes=768*1024**2)
    expanded_step = replace(env18, complement_maps=e_maps, complement=e_block)
    operator = SingletCutOperator(a_block, e_block, cross_bonds(supports[16], support126, bonds))
    old_embedded = {q: np.pad(c, ((0, a_maps[q].shape[1]-c.shape[0]),
        (0, e_maps[q].shape[1]-c.shape[1]))) for q, c in env18.coefficients.items()}
    parent_center = operator.encode(old_embedded)
    energy_before = float(np.vdot(parent_center, operator.matrix@parent_center).real)
    if abs(energy_before-initial['total']) > 2e-8:
        raise ArithmeticError('complete enlarged center does not reproduce physical parent H')
    stage('full_center_ready', dimension=operator.dimension,
          parent_energy_error=abs(energy_before-initial['total']))
    values, vectors = eigh(operator.matrix, subset_by_index=(0, 1), check_finite=True, driver='evr')
    center = vectors[:, 0]
    if np.vdot(parent_center, center).real < 0:
        center = -center
    energy = float(np.vdot(center, operator.matrix@center).real)
    residual = float(np.linalg.norm(operator.matrix@center-energy*center))
    if residual > 1e-8 or energy > energy_before+1e-9:
        raise ArithmeticError('full center eigensolve failed explicit residual/variational check')
    optimized_coefficients = operator.decode(center)
    result['solver'] = dict(center_dimension=operator.dimension, second_eigenvalue=float(values[1]),
        raw_energy_total_before=energy_before, raw_energy_total_optimized=energy,
        raw_energy_per_site_optimized=energy/144, projected_center_residual_norm=residual,
        projected_center_residual_squared=residual**2,
        projected_center_gradient_norm=2*residual,
        hermiticity_error=operator.hermiticity_error,
        physical_full_H_variance_known=False, complete_A1_quality_unknown=True,
        all_center_coefficients_optimized=True,
        fixed_children_target_sites=[9,9], fixed_children_environment_sites=[18,108])
    np.savez_compressed(output/'full_center.npz', H=operator.matrix, parent=parent_center, optimized=center,
        **{f'C_q{q}': c for q, c in optimized_coefficients.items()})
    stage('full_center_solved', residual=residual, energy_per_site=energy/144)
    cfg = MixedClebschConfig(12, 20, 10)
    parent_bank, _ = native_bank_from_state(source, parent)

    def export(name, maps18, coefficients, compression):
        candidate = clone(parent)
        candidate['maps'][0] = real_maps(maps18)
        reproject(candidate, 0, joins)
        maps36, coefficients36, restore18 = restore_singlet_parent(expanded_step, coefficients,
            {q: w.shape[1] for q, w in maps18.items()},
            parent_widths={q: w.shape[1] for q, w in parent['maps'][8].items()})
        maps72, newroots, restore36 = restore_singlet_parent(env36, coefficients36,
            {q: w.shape[1] for q, w in maps36.items()},
            parent_widths={q: w.shape[1] for q, w in parent['maps'][12].items()})
        candidate['maps'][8] = real_maps(maps36); candidate['maps'][12] = real_maps(maps72)
        for node in (8,12):
            left, right, _ = TREE.binary_merges[node]
            candidate['channels'][node] = {q: source_channels(candidate['blocks'][left].q_widths,
                candidate['blocks'][right].q_widths, q) for q in candidate['maps'][node]}
            reproject(candidate, node, joins)
        candidate['maps'][14][0] = real_maps({0: np.asarray([newroots[q][i,j]
            for q,i,_,j in candidate['channels'][14][0]])[:,None]})[0]
        reproject(candidate, 14, joins)
        layer = raw_layer_energy(candidate, source, .125)
        # Independently use the SAME full-center H on the compressed state;
        # ancestors are separately rebuilt from the physical bond ledger.
        in_expanded = {q: np.asarray(a_maps[q]).T@np.asarray(maps18[q])@coefficients[q]
                       for q in coefficients}
        vector = operator.encode(in_expanded)
        direct = float(np.vdot(vector, operator.matrix@vector).real)
        error = abs(layer['total']-direct)
        if error > 2e-8:
            raise ArithmeticError('independent restored root H disagrees with complete center')
        bank, embedding = native_bank_from_state(source, candidate)
        norm = triangular_raw_overlap(bank, bank, cfg)
        if abs(norm-1) > 2e-10:
            raise ArithmeticError('exported native raw state not normalized')
        endpoint = output/name
        save_checkpoint(endpoint/'state.pkl', candidate, metadata=dict(registration=registry, raw_only=True))
        save_checkpoint(endpoint/'initialized_bank.pkl', bank, metadata=dict(registration=registry,
            model=dict(length=12, multiplicity_cap=20, maximum_q=10), j2=.125, embedding=embedding,
            optimizer_reset_required=True, complete_A1_quality_unknown=True))
        result['endpoints'][name] = dict(raw_energy_per_site=direct/144, raw_layer_energy=layer,
            compression=compression, restore18=restore18, restore36=restore36,
            independent_root_H_error=error, native_raw_norm_real=float(norm.real),
            native_raw_norm_imaginary=float(norm.imag),
            parent_raw_fidelity_squared=float(abs(triangular_raw_overlap(parent_bank, bank, cfg))**2),
            bank_path=str((endpoint/'initialized_bank.pkl').resolve()),
            bank_sha256=file_sha256(endpoint/'initialized_bank.pkl'), embedding=embedding,
            complete_A1_quality_unknown=True, physical_full_H_variance_known=False)
        stage('exported_'+name, raw_energy_per_site=direct/144)
    export('optimized_full', a_maps, optimized_coefficients, dict(no_compression=True))
    for name, widths in (('compressed_old', old_widths), ('compressed_larger',
            {q: w.shape[1] for q, w in a_maps.items()})):
        rotations, coefficients, info = compress_singlet_center(optimized_coefficients, widths)
        maps18 = {q: a_maps[q]@rotations[q] for q in a_maps}
        export(name, maps18, coefficients, info)
    bank, embedding = native_bank_from_state(source, parent)
    save_checkpoint(output/'parent'/'initialized_bank.pkl', bank, metadata=dict(registration=registry,
        model=dict(length=12,multiplicity_cap=20,maximum_q=10), j2=.125, embedding=embedding,
        optimizer_reset_required=True, complete_A1_quality_unknown=True))
    result['endpoints']['parent'] = dict(raw_energy_per_site=initial['total']/144,
        bank_path=str((output/'parent'/'initialized_bank.pkl').resolve()),
        bank_sha256=file_sha256(output/'parent'/'initialized_bank.pkl'), embedding=embedding)
    result.update(status='complete_full_center_and_same_center_rank_comparison',
        wall_seconds=perf_counter()-start, CPU_seconds=process_time()-cpu,
        full_physical_H_queries=0, complete_A1_queries=0, new_MC_records=0)
    atomic_json(output/'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--new-width', type=int, default=16)
    options = parser.parse_args()
    run(options.output, new_width=options.new_width)
