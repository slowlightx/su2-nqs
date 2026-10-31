"""Exact invariants of the isolated hierarchy initialization workflow."""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_initialization import (
    active_amplitude, complete_active_columns, native_bank_from_state,
    retained_widths, source_channels, split_bond_ledger,
)
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, make_mixed_clebsch_amplitude,
)
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank
from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_reference import schmidt_source_maps
from projects.clebsch_tree_nqs.workflows.triangle_sector_transfer import extract_merge_map
from su2_nqs.models.triangular_mixed_clebsch import mixed_clebsch_layout


def test_target_periodic_bond_ledger_is_exact():
    for j2, expected in ((0., 432), (.125, 864)):
        leaves, joins, bonds = split_bond_ledger(12, j2)
        assert len(leaves) == 16 and len(joins) == 15
        assert sum(map(len, leaves))+sum(map(len, joins)) == len(bonds) == expected
        np.testing.assert_allclose(sum(x[2] for rows in leaves+joins for x in rows),
                                   sum(x[2] for x in bonds), atol=1e-13)


def test_completion_preserves_active_phase_and_isometry():
    rng = np.random.default_rng(712)
    old = rng.normal(size=(21, 9))+1j*rng.normal(size=(21, 9))
    active = np.linalg.qr(old[:, :3])[0]*np.exp(1j*np.array([.2, -.6, 1.1]))
    finished = complete_active_columns(active, old)
    np.testing.assert_array_equal(finished[:, :3], active)
    np.testing.assert_allclose(finished.conj().T @ finished, np.eye(9), atol=5e-15)


def test_native_embedding_preserves_variable_upper_tree_state():
    """No snapshot: random legal tensors, every upper q>0 retained."""
    cfg6 = MixedClebschConfig(6, 20, 10)
    source = init_mixed_clebsch_family(jax.random.PRNGKey(72), cfg6)["ternary_free"]
    source, _ = canonicalize_triangular_bank(source, cfg6)
    tree = triangular_schur_hierarchy(12)
    widths = [{q: w.shape[1] for q, w in zip(cfg6.local_q, source["local"][i % 4], strict=True)}
              for i in range(16)]
    rng = np.random.default_rng(73); maps = {}; channels = {}
    for node, (left, right, _) in enumerate(tree.binary_merges):
        keep = retained_widths(widths[left], widths[right], 2, node == 14)
        channels[node] = {q: source_channels(widths[left], widths[right], q) for q in keep}
        maps[node] = {q: np.linalg.qr(rng.normal(size=(len(channels[node][q]), d)))[0]
                      for q, d in keep.items()}
        widths.append(keep)
    assert set(widths[-2]) == {0, 2, 4, 6, 8, 10} and widths[-1] == {0: 1}
    state = dict(maps=maps, channels=channels, active=2)
    native, diagnostics = native_bank_from_state(source, state)
    assert diagnostics["maximum_reduced_isometry_error"] < 1e-12
    query = make_mixed_clebsch_amplitude(MixedClebschConfig(12, 20, 10))
    for _ in range(2):
        row = rng.permutation(np.r_[np.zeros(72, dtype=np.int32), np.ones(72, dtype=np.int32)])
        exact = active_amplitude(source, state, row)
        actual = float(query(native, row))
        np.testing.assert_allclose(actual, exact, atol=1e-30, rtol=3e-12)


def test_schmidt_reference_keeps_complex_right_transpose_convention():
    cfg = MixedClebschConfig(6, 20, 10)
    source = init_mixed_clebsch_family(jax.random.PRNGKey(750), cfg)["ternary_free"]
    source, _ = canonicalize_triangular_bank(source, cfg)
    rng = np.random.default_rng(751)
    top = list(source["top"])
    top[2] = tuple(np.asarray(x)*(1+1j*rng.normal(size=x.shape)) for x in top[2])
    source = dict(source, top=tuple(top))
    maps, singular, diagnostics = schmidt_source_maps(source, 4)
    layout = mixed_clebsch_layout(cfg)
    rootchannels = source_channels(layout[0]["widths"], layout[1]["widths"], 0)
    root = extract_merge_map(source, cfg, 2, 0, rootchannels)[:, 0]
    lookup = dict(zip(rootchannels, root)); overlap = norm = 0.
    for q, values in singular.items():
        rotations = []
        for child in (0, 1):
            row = layout[child]
            widths = lambda i: {qi: w.shape[1] for qi, w in zip(cfg.local_q, source["local"][i], strict=True)}
            channels = source_channels(widths(row["left"]), widths(row["right"]), q)
            original = extract_merge_map(source, cfg, child, q, channels)
            rotations.append(original.conj().T @ maps[child][q])
        coefficients = np.asarray([[lookup[q, a, q, b] for b in range(layout[1]["widths"][q])]
                                   for a in range(layout[0]["widths"][q])])
        reconstructed = rotations[0] @ np.diag(values) @ rotations[1].T
        overlap += np.vdot(coefficients, reconstructed)
        norm += float(np.vdot(reconstructed, reconstructed).real)
    np.testing.assert_allclose(norm, 1., atol=2e-14)
    np.testing.assert_allclose(abs(overlap)**2/diagnostics["source_raw_norm_squared"],
                               diagnostics["normalized_raw_source_fidelity"], atol=2e-14)
