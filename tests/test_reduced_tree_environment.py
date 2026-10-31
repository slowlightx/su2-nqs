"""Independent physical Schmidt checks for nonorthogonal reduced TTN edges."""
from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.reduced_tree_environment import (
    ReducedSingletTree, reduced_tree_environments, canonical_edge_spectrum,
    gauge_transform_edge, allocate_multiplets, truncate_internal_edge,
)


def _fixture():
    """Four physical spin-half leaves, redundant rank-two virtual copies."""
    rng = np.random.default_rng(20261008151)
    tree = FusionTree.balanced(4)
    first = {(1, 1, q): rng.normal(size=(2, 1, 1)) + 1j*rng.normal(size=(2, 1, 1)) for q in (0, 2)}
    second = {(1, 1, q): rng.normal(size=(2, 1, 1)) + 1j*rng.normal(size=(2, 1, 1)) for q in (0, 2)}
    root = {(q, q, 0): rng.normal(size=(2, 2, 2)) + 1j*rng.normal(size=(2, 2, 2)) for q in (0, 2)}
    return ReducedSingletTree(tree, tuple({1: np.eye(1)} for _ in range(4)),
                             (first, second, root), np.asarray([1+.2j, -.4+.7j]))


def _physical_amplitudes(state, leaf_embeddings):
    """Contract all physical/magnetic indices, rather than reduced metrics."""
    values = dict(zip(state.tree.leaves, leaf_embeddings, strict=True))
    for node, kernels in zip(state.tree.internal_nodes, state.kernels, strict=True):
        values[node] = {}
        for (ql, qr, qt), tensor in kernels.items():
            cg = np.asarray([[[clebsch_gordan(ql, ml, qr, mr, qt, mt)
                               for mt in range(-qt, qt+1, 2)]
                              for mr in range(-qr, qr+1, 2)]
                             for ml in range(-ql, ql+1, 2)])
            a, b = values[node.left][ql], values[node.right][qr]
            joined = np.einsum("sml,tnr,klr,mnM->stMk", a, b, tensor, cg, optimize=True)
            joined = joined.reshape(a.shape[0]*b.shape[0], qt+1, tensor.shape[0])
            values[node][qt] = values[node].get(qt, np.zeros_like(joined)) + joined
    return values[state.tree.root][0][:, 0, :] @ state.root_vector


def _spin_half_embeddings():
    return tuple({1: np.eye(2)[:, :, None].astype(complex)} for _ in range(4))


def _full_spectrum(spectrum):
    return np.sort(np.concatenate([np.repeat(block.magnetic_eigenvalues, q+1)
                                   for q, block in spectrum.blocks.items()]))[::-1]


def test_exact_environment_schmidt_spectrum_with_rank_deficient_redundant_virtual_basis():
    state = _fixture()
    amplitude = _physical_amplitudes(state, _spin_half_embeddings())
    norm = np.vdot(amplitude, amplitude).real
    environments = reduced_tree_environments(state)
    np.testing.assert_allclose(environments.norm_squared, norm, rtol=3e-14)
    assert environments.maximum_relative_edge_norm_error < 3e-14
    spectrum = canonical_edge_spectrum(environments, state.tree.internal_nodes[0])
    expected = np.linalg.svd(amplitude.reshape(4, 4)/np.sqrt(norm), compute_uv=False)**2
    actual = _full_spectrum(spectrum)
    np.testing.assert_allclose(actual[:4], expected, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(actual[4:], 0., atol=3e-14)
    assert all(block.metric_rank == 1 for block in spectrum.blocks.values())
    assert spectrum.physical_bond_dimension == 8  # redundant stored D, rank <=4
    assert spectrum.normalized_weight_error < 3e-14
    entropy = -np.sum(expected[expected > 0]*np.log(expected[expected > 0]))
    np.testing.assert_allclose(spectrum.entropy, entropy, atol=3e-13, rtol=0)


def test_complex_gauge_and_canonical_null_space_are_reversible_in_physical_amplitudes():
    state = _fixture()
    amplitude = _physical_amplitudes(state, _spin_half_embeddings())
    gauge = np.asarray([[2+.1j, .3+.4j], [.2-.1j, .8+.2j]])
    gauged = gauge_transform_edge(state, 0, {q: gauge*(1+.2*q) for q in (0, 2)})
    np.testing.assert_allclose(_physical_amplitudes(gauged, _spin_half_embeddings()), amplitude, atol=3e-13, rtol=3e-13)
    old = canonical_edge_spectrum(reduced_tree_environments(state), state.tree.internal_nodes[0])
    new = canonical_edge_spectrum(reduced_tree_environments(gauged), state.tree.internal_nodes[0])
    for q in old.blocks:
        np.testing.assert_allclose(new.blocks[q].multiplicity_weights, old.blocks[q].multiplicity_weights, atol=3e-13, rtol=0)
    canonical = gauge_transform_edge(state, 0, {q: block.gauge for q, block in old.blocks.items()})
    rebuilt = gauge_transform_edge(canonical, 0, {q: block.gauge_inverse for q, block in old.blocks.items()})
    np.testing.assert_allclose(_physical_amplitudes(canonical, _spin_half_embeddings()), amplitude, atol=3e-13, rtol=3e-13)
    for before, after in zip(state.kernels, rebuilt.kernels, strict=True):
        for key in before:
            np.testing.assert_allclose(after[key], before[key], atol=3e-13, rtol=3e-13)


def test_whole_multiplet_knapsack_and_actual_projection_norm_overlap_match():
    state = _fixture()
    environment = reduced_tree_environments(state)
    spectrum = canonical_edge_spectrum(environment, state.tree.internal_nodes[0])
    allocation = allocate_multiplets(spectrum, 3, exact_budget=True)
    # Exhaustively independent knapsack oracle, including redundant zero modes.
    alternatives = [(a, b) for a in range(3) for b in range(3) if a+3*b == 3]
    weights = [float(spectrum.blocks[0].multiplicity_weights[:a].sum()
                     + spectrum.blocks[2].multiplicity_weights[:b].sum()) for a, b in alternatives]
    np.testing.assert_allclose(allocation.retained_weight, max(weights), atol=3e-14, rtol=0)
    assert allocation.physical_bond_dimension == 3
    truncated = truncate_internal_edge(state, 0, spectrum, allocation)
    old = _physical_amplitudes(state, _spin_half_embeddings())
    new = _physical_amplitudes(truncated, _spin_half_embeddings())
    retained = np.vdot(new, new).real/np.vdot(old, old).real
    np.testing.assert_allclose(retained, allocation.retained_weight, atol=3e-13, rtol=0)
    np.testing.assert_allclose(np.vdot(old, new)/np.vdot(old, old), retained, atol=3e-13, rtol=0)
    np.testing.assert_allclose(reduced_tree_environments(truncated).norm_squared,
                               np.vdot(new, new).real, rtol=3e-13)
    with pytest.raises(ValueError, match="not realizable"):
        allocate_multiplets(spectrum, 9)


def test_environment_validation_rejects_bad_charges_widths_and_root():
    state = _fixture()
    first = dict(state.kernels[0])
    first[(1, 1, 1)] = np.ones((2, 1, 1))
    with pytest.raises(ValueError, match="triangle or parity"):
        ReducedSingletTree(state.tree, state.leaf_grams, (first, *state.kernels[1:]), state.root_vector)
    with pytest.raises(ValueError, match="singlet root"):
        ReducedSingletTree(state.tree, state.leaf_grams, state.kernels, np.ones(1))
    with pytest.raises(ValueError, match="internal nonroot"):
        gauge_transform_edge(state, 2, {})
    with pytest.raises(ValueError, match="positive integer"):
        allocate_multiplets(canonical_edge_spectrum(reduced_tree_environments(state), state.tree.internal_nodes[0]), True)


def test_existing_finite_metric_adapter_matches_n16_exact_rdm_and_same_d_projection():
    from dataclasses import replace
    import jax.numpy as jnp
    from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import physical_config
    from su2_nqs.models.finite_schur_decoder import FiniteSchurDecoderConfig, reduced_entry_metadata, free_reduced_bank
    from su2_nqs.models.finite_schur_capacity import finite_bank_reduced_tree, reduced_tree_finite_bank
    from su2_nqs.models.schur_metric_retraction import schur_norm_environments
    from su2_nqs.su2.local_schur import cell_schur_tensor, cell_schur_spec

    cfg = FiniteSchurDecoderConfig(replace(physical_config(4, 4, 2), maximum_intermediate_q=4), 2, 2)
    rng = np.random.default_rng(20261008152)
    entries = rng.normal(size=(reduced_entry_metadata(cfg)["complex_entries"], 2))
    bank = free_reduced_bank({"entries": jnp.asarray(entries)}, cfg)
    state = finite_bank_reduced_tree(bank, cfg)
    spec = cell_schur_spec('square_plaquette')
    table = cell_schur_tensor('square_plaquette')
    embeddings = tuple({q: table[:, q][:, np.arange(-q, q+1, 2)+4, :spec.multiplicity(q)]
                        for q in spec.q_values} for _ in range(4))
    amplitude = _physical_amplitudes(state, embeddings)
    norm = np.vdot(amplitude, amplitude).real
    environment = reduced_tree_environments(state)
    old_grams, old_environments = schur_norm_environments(bank, cfg)
    for node in state.tree.internal_nodes:
        for q, gram in environment.upward[node].items():
            np.testing.assert_allclose(gram, old_grams[node][q], rtol=3e-13, atol=3e-13)
            np.testing.assert_allclose(environment.downward[node][q], old_environments[node][q].T/(q+1),
                                       rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(environment.norm_squared, norm, rtol=3e-13)
    spectrum = canonical_edge_spectrum(environment, state.tree.internal_nodes[0])
    singular = np.linalg.svd(amplitude.reshape(256, 256)/np.sqrt(norm), compute_uv=False)**2
    actual = _full_spectrum(spectrum)
    np.testing.assert_allclose(actual, singular[:len(actual)], atol=3e-13, rtol=3e-13)
    assert singular[len(actual):].sum() < 3e-25
    allocation = allocate_multiplets(spectrum, 12)
    assert allocation.physical_bond_dimension == sum((q+1)*copies for q, copies in allocation.copies_by_q.items()) == 12
    truncated = truncate_internal_edge(state, 0, spectrum, allocation)
    reduced_tree_finite_bank(truncated, cfg)  # Keeps the unchanged public decoder layout.
    result = _physical_amplitudes(truncated, embeddings)
    np.testing.assert_allclose(np.vdot(result, result).real/norm, allocation.retained_weight, atol=3e-13, rtol=0)
    np.testing.assert_allclose(np.vdot(amplitude, result)/norm, allocation.retained_weight, atol=3e-13, rtol=0)


def test_saved_capacity_readout_seals_same_d_and_actual_n16_projector_energy():
    import hashlib
    import itertools
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_schur_capacity_readout_v1'
    if not (directory/'completion.json').exists():
        pytest.skip('optional immutable saved capacity readout absent')
    result = json.loads((directory/'results.json').read_text())
    registration = json.loads((directory/'registration.json').read_text())
    completion = json.loads((directory/'completion.json').read_text())
    for name, key in (('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                      ('readouts.npz', 'readouts_sha256')):
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == completion[key]
    for path, digest in registration['input_sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest
    for path, digest in registration['source_manifest'].items():
        assert hashlib.sha256((directory/'source_snapshots'/path).read_bytes()).hexdigest() == digest
    for payload, key in ((registration, 'registration_sha256'), (result, 'result_sha256')):
        digest = hashlib.sha256(json.dumps({k:v for k,v in payload.items() if k != key},
                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert digest == payload[key]
    assert result['new_training'] == result['new_samples'] == result['new_ED'] == 0
    assert result['new_head'] is False
    assert result['stationarity_or_variational_win_claim'] is False
    assert result['padded_speedup_claim'] is False
    assert result['raw_neural_free_banks_identical'] is False
    for row in result['raw_decoder_comparisons'].values():
        assert row['saved_attempts'] == 8 and row['saved_accepted_updates'] == 1
        simple, optimal = [row['comparisons'][key] for key in ('simple_same_D', 'spectrum_same_D')]
        assert simple['physical_D'] == optimal['physical_D'] == 12
        assert optimal['retained_weight'] >= simple['retained_weight'] - 3e-14
        assert optimal['padded_bank_storage_unchanged']
        assert row['no_inference_of_missing_multiplicities']
    settings = registration['config']
    with np.load(root/settings['target_state']) as source:
        basis = source['basis']; psi = source[settings['target_vector']]
    with np.load(directory/'readouts.npz') as data:
        sigmas = {q:data[f'target_sigma_q{q}'] for q in (0,2,4,6,8)}
        matrices = {key:data[key+'_target_projector'] for key in ('simple_same_D', 'spectrum_same_D')}
    weights = {q:np.linalg.eigvalsh(sigma)[::-1] for q,sigma in sigmas.items()}
    candidates = itertools.product(*(range(min(len(weights[q]), 18//(q+1))+1) for q in weights))
    oracle = max(sum(float(weights[q][:copies].sum()) for q,copies in zip(weights, counts, strict=True))
                 for counts in candidates if sum((q+1)*copies for q,copies in zip(weights, counts, strict=True)) == 18)
    n = basis.shape[1]
    packed = basis.astype(np.int64) @ (1 << np.arange(n-1, -1, -1))
    full = np.zeros(1 << n, complex); full[packed] = psi
    sites = tuple(settings['target_region']); rest = tuple(i for i in range(n) if i not in sites)
    permutation = sites+rest
    physical_matrix = full.reshape((2,)*n).transpose(permutation).reshape(2**len(sites), -1)
    region_result = result['trusted_N16_target']
    for key, projector in matrices.items():
        row = region_result['comparisons'][key]
        assert row['physical_D'] == 18
        np.testing.assert_allclose(np.trace(projector), 18., atol=2e-13, rtol=0)
        np.testing.assert_allclose(projector@projector, projector, atol=2e-13, rtol=0)
        projected = (projector @ physical_matrix).reshape((2,)*n).transpose(np.argsort(permutation)).ravel()
        actual_norm = np.vdot(projected, projected).real
        np.testing.assert_allclose(row['actual_norm_ratio'], actual_norm, atol=3e-13, rtol=0)
        np.testing.assert_allclose(row['retained_weight'], actual_norm, atol=3e-13, rtol=0)
        # Independent periodic 120-degree metric/basis action: no production
        # graph, H kernel, global matrix or ground-state diagonalization.
        hpsi = np.zeros_like(projected)
        integers = np.arange(1 << n)
        for first in range(n):
            ra, ca = divmod(first, 4)
            for second in range(first+1, n):
                rb, cb = divmod(second, 4)
                distances = [(rb-ra+4*x)**2+(cb-ca+4*y)**2-(rb-ra+4*x)*(cb-ca+4*y)
                             for x in (-1,0,1) for y in (-1,0,1)]
                coupling = 1. if 1 in distances else .125 if 3 in distances else 0.
                if coupling == 0:
                    continue
                a, b = 1 << (n-first-1), 1 << (n-second-1)
                opposite = ((integers & a) != 0) != ((integers & b) != 0)
                hpsi += coupling*np.where(opposite, -.25, .25)*projected
                hpsi += .5*coupling*opposite*projected[integers ^ (a|b)]
        energy = np.vdot(projected, hpsi)/actual_norm
        variance = np.vdot(hpsi-energy*projected, hpsi-energy*projected).real/actual_norm
        np.testing.assert_allclose(row['physical_after']['energy_per_site'], energy.real/n, atol=3e-13, rtol=0)
        np.testing.assert_allclose(row['physical_after']['variance_per_site'], variance/n, atol=3e-13, rtol=0)
        assert row['full_A1_not_preserved_by_regional_compression']
        assert row['no_variational_training_or_energy_win_claim']
    np.testing.assert_allclose(region_result['comparisons']['spectrum_same_D']['retained_weight'], oracle, atol=3e-13, rtol=0)
