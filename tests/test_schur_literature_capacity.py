"""Larger q/chi keeps the same independent full-Schur amplitude oracle."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import physical_config
from su2_nqs.models.cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from su2_nqs.models.finite_schur_decoder import (
    FiniteSchurDecoderConfig, reduced_entry_metadata, free_reduced_bank,
    make_finite_schur_amplitude,
)
from su2_nqs.benchmarks.schur_tensor_decoder import four_cell_tensor_decoder_rows
from su2_nqs.su2.local_schur_fusion import four_cell_schur_states

jax.config.update('jax_enable_x64', True)


def test_q8_chi16_matches_independent_complete_n16_schur_amplitudes():
    cfg = FiniteSchurDecoderConfig(replace(physical_config(4, 4, 16),
        maximum_intermediate_q=8), 2, 2, embedding_dim=8, depth=1)
    meta = reduced_entry_metadata(cfg)
    entries = np.random.default_rng(2026101461).normal(size=(meta['complex_entries'], 2))/4
    bank = free_reduced_bank({'entries': jnp.asarray(entries)}, cfg)
    layouts = cell_multiplicity_layouts(cfg.tree)
    arrays = jax.tree.map(np.asarray, bank)
    coefficients = []
    for state in four_cell_schur_states('square_plaquette'):
        q, a, t = state.cell_q, state.cell_a, state.pair_q
        left = arrays['nodes'][0][layouts[0].keys.index((q[0], q[1], t)), :, a[0], a[1]]
        right = arrays['nodes'][1][layouts[1].keys.index((q[2], q[3], t)), :, a[2], a[3]]
        root = arrays['nodes'][2][layouts[2].keys.index((t, t, 0))]
        coefficients.append(np.einsum('l,r,klr->', left, right, root)/4)
    rng = np.random.default_rng(2026101462)
    bits = np.stack([rng.permutation([0]*8+[1]*8) for _ in range(13)]).astype(np.int32)
    oracle = four_cell_tensor_decoder_rows(bits, 'square_plaquette', cfg.tree.cells)@np.asarray(coefficients)
    actual = np.asarray(jax.jit(jax.vmap(make_finite_schur_amplitude(cfg),
        in_axes=(None, 0)))(bank, bits))
    np.testing.assert_allclose(actual, oracle, atol=2e-12, rtol=2e-12)
    assert np.linalg.norm(actual-oracle)/np.linalg.norm(oracle) < 2e-12


def test_registered_capacity_is_not_depth_only_or_a_native_reduced_vit():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    folder = root/'projects/clebsch_tree_nqs/configs/remote/scale_capacity_20261014'
    s = json.loads((folder/'cg_train.json').read_text())
    assert s['maximum_q'] == 8 and s['multiplicity_width'] == 16
    assert s['depth'] == 2
    assert s['steps']*s['chains']*s['records_per_step'] == 12800
    assert s['carrier_checkpoint']['sha256']
    assert s['SR']['maximum_overlap_distance'] == .01
    cfg = FiniteSchurDecoderConfig(replace(physical_config(6, 6, 16),
        maximum_intermediate_q=8), 3, 3, 32, 2)
    assert sum(q+1 for q in range(0, 9, 2))*16 == 400
    assert reduced_entry_metadata(cfg)['free_real_parameters'] == 1146624
