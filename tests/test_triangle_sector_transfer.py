"""Independent geometry and native-map checks for target-sector transfer."""
import jax
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_sector_transfer import (
    complete_with_copy, extract_merge_map, replace_merge_map, target_child_blocks,
)
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_layout,
)
from su2_nqs.su2.quantum_numbers import coupling_range


def test_native_map_pack_roundtrip_uses_exact_source_channels():
    cfg = MixedClebschConfig(6, 9, 10)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(261023101), cfg)["ternary_free"]
    bank = jax.tree.map(np.asarray, bank)
    widths = {q: values.shape[1] for q, values in zip(cfg.local_q, bank["local"][0], strict=True)}
    channels = tuple((ql, al, qr, ar) for ql in widths for qr in widths
        if 2 in coupling_range(ql, qr) for al in range(widths[ql]) for ar in range(widths[qr]))
    matrix = extract_merge_map(bank, cfg, 0, 2, channels)
    updated = replace_merge_map(bank, cfg, 0, 2, channels, matrix)
    for before, after in zip(jax.tree.leaves(bank), jax.tree.leaves(updated), strict=True):
        np.testing.assert_array_equal(before, after)
    changed = replace_merge_map(bank, cfg, 0, 2, channels, .7*matrix)
    np.testing.assert_allclose(extract_merge_map(changed, cfg, 0, 2, channels), .7*matrix, atol=1e-15)
    channels0 = tuple((ql, al, qr, ar) for ql in widths for qr in widths
        if 0 in coupling_range(ql, qr) for al in range(widths[ql]) for ar in range(widths[qr]))
    np.testing.assert_array_equal(extract_merge_map(changed, cfg, 0, 0, channels0),
                                  extract_merge_map(bank, cfg, 0, 0, channels0))


def test_complete_map_keeps_new_subspace_and_native_width():
    rng = np.random.default_rng(261023102)
    old = np.linalg.qr(rng.normal(size=(43, 20)))[0]
    low = np.linalg.qr(rng.normal(size=(43, 2)))[0]
    new = complete_with_copy(low, old)
    assert new.shape == old.shape
    np.testing.assert_allclose(new.T @ new, np.eye(20), atol=2e-14)
    np.testing.assert_allclose(new @ new.T @ low, low, atol=2e-14)


def test_target_child_geometry_matches_independent_periodic_metric():
    cfg = MixedClebschConfig(6, 9, 10)
    raw = init_mixed_clebsch_family(jax.random.PRNGKey(261023103), cfg)["ternary_free"]
    bank, _ = canonicalize_triangular_bank(raw, cfg)
    children, cross, notes = target_child_blocks(bank, .125, cap=2)
    hierarchy = triangular_schur_hierarchy(12)
    supports = [list(block.sites) for block in hierarchy.blocks[:2]]
    expected = []
    for i, a in enumerate(supports[0]):
        ra, ca = divmod(a, 12)
        for j, b in enumerate(supports[1]):
            rb, cb = divmod(b, 12)
            length = min((ra-rb+12*x)**2+(ca-cb+12*y)**2-(ra-rb+12*x)*(ca-cb+12*y)
                         for x in (-1, 0, 1) for y in (-1, 0, 1))
            if length in (1, 3):
                expected.append((i, j, 1. if length == 1 else .125))
    assert sorted(cross) == sorted(expected)
    assert set(supports[0]).isdisjoint(supports[1])
    for child, note in zip(children, notes, strict=True):
        assert child.isometry.n_sites == 9
        assert len(child.labels) == 50
        assert note["projection_diagnostics"]["scalar_error"] < 3e-12
        assert note["projection_diagnostics"]["ladder_error"] < 3e-12
