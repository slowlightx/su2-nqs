"""Frozen-parent raw-edge compression checked in the complete physical A1 state.

This offline, amplitude-only diagnostic never updates a production checkpoint.
Raw-tree Schmidt optimality does not imply optimality after coherent space
projection. Born-sample overlap estimates also cannot certify missing support
at exact nodes of the parent. No energy or Hamiltonian call is made here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_clebsch_gauge import (
    canonicalize_triangular_bank, compress_triangular_edge,
    triangular_edge_spectrum, triangular_raw_overlap,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, make_mixed_projected_amplitude,
)
from su2_nqs.su2.reduced_tree_environment import allocate_multiplets
from su2_nqs.vmc.checkpoint import load_checkpoint


def overlap_statistics(ratios: np.ndarray, chains: np.ndarray) -> dict:
    """Ratio overlap and whole-chain jackknife; never interpret as full norm.

    Ratios may be signed or complex: replacing them by absolute values loses
    destructive interference. All chains must have the same record count.
    """
    r = np.asarray(ratios, dtype=np.complex128)
    labels = np.asarray(chains)
    if r.ndim != 1 or labels.shape != r.shape or not np.isfinite(r).all():
        raise ValueError('finite one-dimensional ratios and matching labels required')
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 4 or np.any(counts != counts[0]):
        raise ValueError('at least four equally weighted independent chain labels required')

    def moments(values: np.ndarray) -> tuple[float, float]:
        w = abs(values)**2
        if not np.isfinite(w.mean()) or w.mean() <= 0:
            raise ValueError('nonzero finite candidate sampled weight required')
        f = float(abs(values.mean())**2 / w.mean())
        return f, float(w.mean()**2 / np.mean(w*w))

    fidelity, ess = moments(r)
    leave = np.array([moments(r[labels != label])[0] for label in unique])
    error = float(np.sqrt((len(leave)-1)/len(leave)*np.sum((leave-leave.mean())**2)))
    folds = [moments(r[np.isin(labels, part)])[0] for part in np.array_split(unique, 2)]
    return dict(sampled_squared_overlap=fidelity, sampled_infidelity=1-fidelity,
                chain_jackknife_infidelity_SE=error,
                sampled_fs_sine=float(np.sqrt(max(0., 1-fidelity))),
                reweighting_ess_fraction=ess,
                disjoint_chain_half_infidelities=[1-f for f in folds],
                samples=len(r), chains=len(unique))


def paired_overlap_difference(left: np.ndarray, right: np.ndarray,
                              chains: np.ndarray) -> dict:
    """Paired infidelity difference on the SAME parent configurations.

    A positive value means the right candidate better preserves the parent.
    Whole-chain leave-outs preserve correlation between the two candidates;
    adding two separately estimated variances would discard that pairing.
    """
    a, b, labels = np.asarray(left), np.asarray(right), np.asarray(chains)
    overlap_statistics(a, labels)
    overlap_statistics(b, labels)

    def difference(x: np.ndarray, y: np.ndarray) -> float:
        return float(abs(y.mean())**2/np.mean(abs(y)**2) -
                     abs(x.mean())**2/np.mean(abs(x)**2))

    value = difference(a, b)
    unique = np.unique(labels)
    leave = np.array([difference(a[labels != label], b[labels != label])
                      for label in unique])
    error = float(np.sqrt((len(leave)-1)/len(leave)*np.sum((leave-leave.mean())**2)))
    return dict(left_infidelity_minus_right_infidelity=value,
                paired_chain_jackknife_SE=error, samples=len(a), chains=len(unique))


def analyze_saved(directory: Path) -> dict:
    """Pair saved amplitude results and expose reduced-parameter cost caveats.

    Native scalar counts are hypothetical one-edge unpadding counts, NOT
    measurements of the still-padded implementation or a full-H speedup.
    """
    import json
    registration = json.loads((directory/'registration.json').read_text())
    if (registration['edge'] != 0 or registration['config'] !=
            dict(length=6, multiplicity_cap=20, maximum_q=10)):
        raise ValueError('native scalar accounting currently validated for CT6 w20 edge0')
    result = json.loads((directory/'results.json').read_text())
    from su2_nqs.models.triangular_mixed_clebsch import mixed_clebsch_layout
    from su2_nqs.su2.triangle_blocks import triangle_block_tables, nine_site_output
    config = MixedClebschConfig(**registration['config'])
    layout = mixed_clebsch_layout(config)[0]
    widths = {q: nine_site_output(q, 20).width for q in config.local_q}
    stored_scalars = (config.length**2//9 * sum(
        triangle_block_tables()['nine'][q].shape[1]*widths[q] for q in config.local_q) +
        sum(int(np.prod(group['shape'])) for row in mixed_clebsch_layout(config)
            for group in row['groups']))
    costs = {q: int(triangle_block_tables()['nine'][q].shape[1] +
                   sum(g['shape'][1]*widths[qr] for g in layout['groups']
                       for ql, qr, _ in g['keys'] if ql == q)) for q in config.local_q}
    original_affected = sum(costs[q]*widths[q] for q in costs)
    hypothetical_scalars = {}
    for row in result['results']:
        counts = {int(q): k for q, k in row['kept_by_q'].items()}
        hypothetical_scalars[row['arm']] = stored_scalars-original_affected + sum(
            costs[q]*counts[q] for q in costs)
    with np.load(directory/'amplitudes.npz') as saved:
        labels, parent = saved['chain_labels'], saved['parent']
        names = registration['arms']
        ratios = [saved[name]/parent for name in names]
    report = dict(parent_params_sha256=registration['parent_params_sha256'],
                  source_amplitudes_sha256=file_sha256(directory/'amplitudes.npz'),
                  uniform_minus_environment=paired_overlap_difference(ratios[0], ratios[1], labels),
                  coefficient_SVD_minus_environment=paired_overlap_difference(ratios[2], ratios[1], labels),
                  actual_padded_stored_scalars=stored_scalars,
                  hypothetical_native_unpadded_scalars=hypothetical_scalars,
                  cost_qualification='Equal physical D=266 is not equal reduced-parameter/FLOP cost. '
                  'Current arrays are padded; no actual memory/H-action saving is claimed.',
                  full_H_calls=0, additional_amplitude_calls=0)
    atomic_json(directory/'paired_comparison.json', report)
    return report


def run(parent: Path, output: Path, *, edge: int = 0, cap: int = 16,
        records_per_chain: int = 4) -> dict:
    """Compare frozen complete-multiplet compressions of one CT6 joint bank."""
    if not jax.config.x64_enabled:
        raise RuntimeError('enable JAX x64 before loading checkpoints')
    if output.exists():
        raise FileExistsError('use a fresh isolated output; never overwrite evidence')
    started = perf_counter()
    checkpoint = load_checkpoint(parent/'checkpoint.pkl')
    import json
    registration = json.loads((parent/'registration.json').read_text())
    audit = json.loads((parent/'terminal_audit.json').read_text())
    bank = checkpoint['params']
    if (registration['arity'] != 'ternary_joint' or registration['model']['length'] != 6
            or pytree_sha256(bank) != audit['params_sha256']
            or pytree_sha256(bank) != checkpoint['metadata']['params_sha256']):
        raise ValueError('an audited CT6 joint parent with exact parameter identity is required')
    config = MixedClebschConfig(**registration['model'])
    if (not isinstance(edge, int) or not 0 <= edge < 4 or not isinstance(cap, int)
            or not 1 <= cap < config.multiplicity_cap or records_per_chain < 1):
        raise ValueError('one nine-site edge and a strict smaller positive cap required')
    with np.load(parent/'terminal_audit.npz') as saved:
        bits = saved['bits']
        labels = saved['chain_labels']
    unique = np.unique(labels)
    selected = []
    for label in unique:
        ids = np.flatnonzero(labels == label)
        if len(ids) < records_per_chain:
            raise ValueError('insufficient records in a chain')
        selected.extend(ids[np.linspace(0, len(ids)-1, records_per_chain, dtype=int)])
    selected = np.array(selected)
    rows, labels = bits[selected], labels[selected]
    canonical, gauge_report = canonicalize_triangular_bank(bank, config)
    spectrum = triangular_edge_spectrum(canonical, config, edge)
    uniform = {q: min(cap, len(b.multiplicity_weights)) for q, b in spectrum.blocks.items()}
    budget = sum((q+1)*k for q, k in uniform.items())
    allocation = allocate_multiplets(spectrum, budget, exact_budget=False)
    plan = dict(task='frozen_CT6_complete_A1_edge_compression_v1',
                parent_checkpoint_sha256=file_sha256(parent/'checkpoint.pkl'),
                parent_params_sha256=audit['params_sha256'],
                parent_audit_sha256=file_sha256(parent/'terminal_audit.npz'),
                config=vars(config), j2_over_j1=registration['j2_over_j1'],
                edge=edge, uniform_cap=cap, physical_bond_budget=budget,
                records_per_chain=records_per_chain,
                arms=['uniform_counts_environment_orientation',
                      'environment_counts_environment_orientation',
                      'environment_counts_coefficient_SVD_orientation'],
                no_model_fitting=True, no_H_calls=True,
                selection='evenly spaced original audit records within every chain')
    plan['sha256'] = canonical_sha256(plan)
    output.mkdir(parents=True)
    atomic_json(output/'registration.json', plan)
    uniform_bank, uniform_report = compress_triangular_edge(canonical, config, edge, uniform)
    optimal_bank, optimal_report = compress_triangular_edge(
        canonical, config, edge, allocation.copies_by_q)
    # The bare-W baseline deliberately uses the ORIGINAL nonorthogonal gauge.
    # In QR coordinates W is isometric and its singular vectors are degenerate.
    naive = jax.tree.map(np.array, bank)
    local = [list(node) for node in naive['local']]
    for i, q in enumerate(config.local_q):
        w = np.asarray(bank['local'][edge][i])
        u, s, vh = np.linalg.svd(w, full_matrices=False)
        keep = allocation.copies_by_q[q]
        local[edge][i] = (u[:, :keep]*s[:keep]) @ vh[:keep]
    naive['local'] = tuple(tuple(node) for node in local)
    arms = [uniform_bank, optimal_bank, naive]
    counts = [uniform, dict(allocation.copies_by_q), dict(allocation.copies_by_q)]
    group = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    one = make_mixed_projected_amplitude(config, group, chunk=4)
    evaluate = jax.jit(lambda p, x: jax.lax.map(lambda row: one(p, row), x))
    begin = perf_counter()
    amplitudes = np.asarray(evaluate(bank, jnp.asarray(rows)).block_until_ready())
    parent_seconds = perf_counter()-begin
    if np.any(amplitudes == 0) or not np.isfinite(amplitudes).all():
        raise ArithmeticError('parent samples have invalid or zero amplitudes')
    canonical_amplitudes = np.asarray(evaluate(canonical, jnp.asarray(rows)).block_until_ready())
    qr_defect = float(np.linalg.norm(canonical_amplitudes-amplitudes)/np.linalg.norm(amplitudes))
    if qr_defect > 1e-10:
        raise ArithmeticError('QR changed the full A1 state')
    norm = triangular_raw_overlap(bank, bank, config).real
    results = []
    values = {}
    for name, candidate, kept in zip(plan['arms'], arms, counts, strict=True):
        begin = perf_counter()
        a = np.asarray(evaluate(candidate, jnp.asarray(rows)).block_until_ready())
        seconds = perf_counter()-begin
        cross = triangular_raw_overlap(bank, candidate, config)
        candidate_norm = triangular_raw_overlap(candidate, candidate, config).real
        raw_fidelity = float(abs(cross)**2/(norm*candidate_norm))
        statistics = overlap_statistics(a/amplitudes, labels)
        # A1 and literal S+ identities use new fixed physical configurations,
        # not the Born records used for overlap estimation.
        rng = np.random.default_rng(2026102209)
        bra = rng.permutation([0]*17+[1]*19)
        ward = np.repeat(bra[None], 19, axis=0)
        ward[np.arange(19), np.flatnonzero(bra)] = 0
        ward_a = np.asarray(evaluate(candidate, jnp.asarray(ward)))
        ward_defect = float(abs(ward_a.sum())/max(np.linalg.norm(ward_a), np.finfo(float).tiny))
        image_rows = np.array([rows[0][g] for g in group[[0, 1, 37, 211]]])
        symmetry_a = np.asarray(evaluate(candidate, jnp.asarray(image_rows)))
        symmetry_defect = float(np.max(abs(symmetry_a-symmetry_a[0])) /
                                max(abs(symmetry_a[0]), np.finfo(float).tiny))
        if max(ward_defect, symmetry_defect) > 1e-10:
            raise ArithmeticError('complete-multiplet compression failed symmetry checks')
        results.append(dict(arm=name, kept_by_q=kept, raw_squared_fidelity=raw_fidelity,
                            physical_bond_dimension=sum((q+1)*n for q, n in kept.items()),
                            full_A1_overlap=statistics, amplitude_seconds=seconds,
                            Splus_relative_defect=ward_defect, A1_relative_defect=symmetry_defect))
        values[name] = a
    report = dict(registration_sha256=plan['sha256'], results=results,
                  canonical_report=gauge_report, full_A1_QR_relative_defect=qr_defect,
                  parent_amplitude_seconds_including_compile=parent_seconds,
                  wall_seconds=perf_counter()-started, full_H_calls=0,
                  production_checkpoint_modified=False, source_audit_reused=True,
                  limits=['Raw Schmidt optimality does not certify projected-state optimality.',
                          'Finite Born ratios cannot certify newly created weight on parent nodes.',
                          'Padded storage is unchanged; there is no claimed H speedup.',
                          'This is fidelity preservation, not an energy optimization or attention win.'])
    np.savez_compressed(output/'amplitudes.npz', parent=amplitudes, chain_labels=labels,
                        bits=rows, **values)
    atomic_json(output/'results.json', report)
    if edge == 0 and config == MixedClebschConfig(length=6, multiplicity_cap=20, maximum_q=10):
        analyze_saved(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--edge', type=int, default=0)
    parser.add_argument('--cap', type=int, default=16)
    parser.add_argument('--records-per-chain', type=int, default=4)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    import json
    print(json.dumps(run(args.parent, args.output, edge=args.edge, cap=args.cap,
                         records_per_chain=args.records_per_chain), indent=2))


if __name__ == '__main__':
    main()
