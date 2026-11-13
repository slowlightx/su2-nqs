"""Actual shape compression preserves every retained doubled-q multiplet."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_clebsch_gauge import (
    canonicalize_triangular_bank, compress_triangular_edge, triangular_edge_spectrum,
    triangular_raw_overlap,
)
from su2_nqs.models.triangular_compact_transfer import (
    compress_compact_triangular_bank, repack_compact_triangular_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, make_mixed_clebsch_amplitude,
    make_mixed_projected_amplitude, mixed_clebsch_layout,
)
from su2_nqs.su2.triangle_blocks import nine_site_output

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def example():
    source = MixedClebschConfig(multiplicity_cap=12, maximum_q=3)
    target = MixedClebschConfig(multiplicity_cap=9, maximum_q=3)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(2026110301), source)["ternary_free"]
    bank, _ = canonicalize_triangular_bank(bank, source)
    compact, report = compress_compact_triangular_bank(bank, source, target)
    rng = np.random.default_rng(2026110302)
    rows = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(6)], np.int32)
    return source, target, bank, compact, report, rows


def test_compact_shapes_and_norm_match_independent_padded_compression(example):
    source, target, bank, compact, report, rows = example
    # Independent reference uses the old padded one-edge API, in explicit
    # hierarchy order. It does not call the new repacking implementation.
    padded = bank
    widths = [{q: nine_site_output(q, target.multiplicity_cap).width for q in target.local_q}]*4
    widths += [dict(r["widths"]) for r in mixed_clebsch_layout(target)]
    for edge, keeps in enumerate(widths[:-1]):
        padded, _ = compress_triangular_edge(padded, source, edge, keeps)
    old_amp = jax.jit(jax.vmap(make_mixed_clebsch_amplitude(source), in_axes=(None, 0)))
    new_amp = jax.jit(jax.vmap(make_mixed_clebsch_amplitude(target), in_axes=(None, 0)))
    np.testing.assert_allclose(new_amp(compact, rows), old_amp(padded, rows), atol=5e-17, rtol=2e-10)
    old_norm = triangular_raw_overlap(bank, bank, source).real
    new_norm = triangular_raw_overlap(compact, compact, target).real
    np.testing.assert_allclose(report["raw_norm_ratio"], new_norm/old_norm, atol=2e-12)
    assert report["target_parameters"] < report["source_parameters"]
    assert report["padded_storage_unchanged"] is False
    assert report["all_source_q_sectors_retained"]
    assert report["complete_magnetic_multiplets_retained"]
    assert 0 < report["raw_squared_fidelity"] < 1


def test_compact_raw_and_full_A1_amplitudes_match_padded_version(example):
    source, target, bank, compact, _, rows = example
    # Zero-embedding the ACTUAL smaller tensors is independent of the raw
    # Schmidt choices and checks all three parent axes after bucket regrouping.
    wide = jax.tree.map(lambda x: np.zeros_like(x), bank)
    wide["local"] = tuple(tuple(np.pad(np.asarray(x), ((0, 0), (0, y.shape[1]-x.shape[1])))
                                for x, y in zip(a, b, strict=True))
                          for a, b in zip(compact["local"], bank["local"], strict=True))
    small_nodes = [{key: np.asarray(x) for group, values in zip(row["groups"], node, strict=True)
                    for key, x in zip(group["keys"], values, strict=True)}
                   for row, node in zip(mixed_clebsch_layout(target), compact["top"], strict=True)]
    wide["top"] = tuple(tuple(np.stack([np.pad(small_nodes[i][key],
                              tuple((0, old-new) for old, new in zip(group["shape"][1:], small_nodes[i][key].shape, strict=True)))
                              for key in group["keys"]]) for group in row["groups"])
                         for i, row in enumerate(mixed_clebsch_layout(source)))
    group = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    old = jax.jit(make_mixed_projected_amplitude(source, group, chunk=4))
    new = jax.jit(make_mixed_projected_amplitude(target, group, chunk=4))
    np.testing.assert_allclose(new(compact, rows[0]), old(wide, rows[0]), atol=5e-17, rtol=2e-10)
    # A literal S+ Ward sum verifies the resulting physical amplitudes,
    # rather than reading the root label from metadata.
    bra = np.random.default_rng(79).permutation([0]*17+[1]*19)
    lowered = np.repeat(bra[None, :], 19, axis=0)
    lowered[np.arange(19), np.flatnonzero(bra)] = 0
    evaluate = jax.jit(jax.vmap(make_mixed_clebsch_amplitude(target), in_axes=(None, 0)))
    values = np.asarray(evaluate(compact, lowered))
    assert abs(values.sum())/np.linalg.norm(values) < 3e-11


def test_boundary_response_span_is_kept_even_when_weakly_occupied(example):
    source, target, bank, _, _, rows = example
    spectrum = triangular_edge_spectrum(bank, source, 0)
    q = 1
    # Protect the LAST, rather than most occupied, physical Schmidt vector.
    response = spectrum.blocks[q].gauge[:, [-1]]
    compact, report = compress_compact_triangular_bank(
        bank, source, target, protected_subspaces={0: {q: response}})
    first = report["edge_selections"][0]["q_blocks"][q]
    assert first["protected_rank"] == 1
    assert first["protected_projector_defect"] < 2e-12
    original_features = np.asarray(bank["local"][0][0]) @ response
    selected_features = np.asarray(compact["local"][0][0])
    coefficient, *_ = np.linalg.lstsq(selected_features, original_features, rcond=1e-12)
    np.testing.assert_allclose(selected_features @ coefficient, original_features, atol=2e-12)
    assert np.isfinite(np.asarray(make_mixed_clebsch_amplitude(target)(compact, rows[0])))


def test_invalid_layout_tails_and_protected_rank_fail_closed(example):
    source, target, bank, _, _, _ = example
    with pytest.raises(ValueError, match="nonzero discarded"):
        repack_compact_triangular_bank(bank, source, target)
    with pytest.raises(ValueError, match="same geometry"):
        compress_compact_triangular_bank(bank, source, MixedClebschConfig(multiplicity_cap=9, maximum_q=5))
    with pytest.raises(ValueError, match="nonroot integer"):
        compress_compact_triangular_bank(bank, source, target, protected_subspaces={True: {}})
    with pytest.raises(ValueError, match="protected response rank"):
        compress_compact_triangular_bank(bank, source, target,
                                         protected_subspaces={0: {1: np.eye(12)}})
    with pytest.raises(ValueError, match="absent doubled-q"):
        compress_compact_triangular_bank(bank, source, target, protected_subspaces={0: {2: np.ones((12, 1))}})


def test_equal_width_is_a_real_copy_without_reoptimization(example):
    source, _, bank, _, _, rows = example
    same, report = compress_compact_triangular_bank(bank, source, source)
    assert report["edge_selections"] == []
    np.testing.assert_allclose(report["raw_squared_fidelity"], 1., atol=3e-15)
    for old, new in zip(jax.tree.leaves(bank), jax.tree.leaves(same), strict=True):
        np.testing.assert_array_equal(old, new)
