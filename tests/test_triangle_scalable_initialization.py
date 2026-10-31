"""Independent bond, parity, and active-decoder checks for L=6*k trees."""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_initialization import (
    active_amplitude, active_bank_from_state, active_state_environment,
    retained_widths, source_channels, split_bond_ledger,
)
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, init_mixed_clebsch_family
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank


@pytest.mark.parametrize("length", [12, 18, 24])
@pytest.mark.parametrize("j2", [0., .125])
def test_actual_target_bonds_are_reconstructed_once_in_global_coordinates(length, j2):
    tree = triangular_schur_hierarchy(length)
    supports = tree.coarse_supports
    leaves, joins, bonds = split_bond_ledger(length, j2)
    recovered = []
    for sites, rows in zip(supports, leaves):
        recovered.extend((*sorted((sites[a], sites[b])), value) for a, b, value in rows)
    for (left, right, _), rows in zip(tree.binary_merges, joins, strict=True):
        recovered.extend((*sorted((supports[left][a], supports[right][b])), value) for a, b, value in rows)
    assert sorted(recovered) == sorted(bonds)
    assert len(recovered) == (3 if j2 == 0 else 6)*length**2
    assert sum(value for _, _, value in recovered) == 3*length**2*(1+j2)
    widths = [{1: 20, 3: 20, 5: 20, 7: 8, 9: 1} for _ in tree.blocks]
    for node, (left, right, parent) in enumerate(tree.binary_merges):
        keep = retained_widths(widths[left], widths[right], 4, node == len(tree.binary_merges)-1)
        assert all(q % 2 == len(supports[parent]) % 2 and q <= min(10, len(supports[parent])) for q in keep)
        if node != len(tree.binary_merges)-1:
            assert set(keep) == {0, 2, 4, 6, 8, 10}
        widths.append(keep)
    assert widths[-1] == {0: 1}
    # A 144-site subtree of a larger system must not inherit a CT12 singlet root.
    assert all(any(q > 0 for q in widths[i]) for i, sites in enumerate(supports[:-1]) if len(sites) == 144)


def test_width_selector_handles_odd_regions_and_rejects_invalid_labels():
    assert retained_widths({1: 2}, {0: 1, 2: 1}, 4, False) == {1: 4, 3: 2}
    with pytest.raises(ValueError): retained_widths({1: 1}, {0: 1}, 4, True)
    for invalid in (True, 1., -1):
        with pytest.raises((ValueError, TypeError)): retained_widths({invalid: 1}, {1: 1}, 4, False)


def test_nonpower_active_bank_matches_independent_scalar_CG_contraction():
    cfg = MixedClebschConfig(18, 20, 10)
    source = init_mixed_clebsch_family(jax.random.PRNGKey(830), MixedClebschConfig(6, 20, 10))["ternary_free"]
    source, _ = canonicalize_triangular_bank(source, MixedClebschConfig(6, 20, 10))
    tree = triangular_schur_hierarchy(18)
    widths = [{q: w.shape[1] for q, w in zip(cfg.local_q, source["local"][block.base_index], strict=True)}
              for block in tree.blocks]
    state = dict(length=18, maps={}, channels={}); rng = np.random.default_rng(831)
    for node, (left, right, _) in enumerate(tree.binary_merges):
        keep = retained_widths(widths[left], widths[right], 1, node == len(tree.binary_merges)-1)
        state["channels"][node] = {q: source_channels(widths[left], widths[right], q) for q in keep}
        state["maps"][node] = {q: np.linalg.qr(rng.normal(size=(len(state["channels"][node][q]), d)))[0]
                                for q, d in keep.items()}
        widths.append(keep)
    bank, profile = active_bank_from_state(source, state)
    env, nodes = active_state_environment(source, state)
    assert len(nodes) == 71
    np.testing.assert_allclose(env.norm_squared, 1., atol=8e-13)
    assert env.maximum_relative_edge_norm_error < 8e-13
    query = make_feature_lut_amplitude(cfg, active_top_widths=profile)
    row = np.zeros(tree.n_sites, np.int32)
    for i, block in enumerate(tree.blocks):
        row[list(block.sites)] = rng.permutation(np.r_[np.ones(4+i % 2, np.int32), np.zeros(5-i % 2, np.int32)])
    expected = active_amplitude(source, state, row)
    assert expected != 0
    prepared = prepare_feature_lut_bank(bank, cfg)
    actual = float(query(prepared, row))
    np.testing.assert_allclose(actual/expected, 1., atol=3e-11, rtol=0)
    # Independently check message invalidation on the unequal-depth Morton
    # top, including the physical root cut and the 72|108 child split.
    _, joins, _ = split_bond_ledger(18, .125)
    pairs = []
    for node in range(len(tree.binary_merges)-4, len(tree.binary_merges)):
        left, right, _ = tree.binary_merges[node]
        candidates = [(tree.coarse_supports[left][a], tree.coarse_supports[right][b])
                      for a, b, _ in joins[node]]
        pairs.append(next(pair for pair in candidates if row[pair[0]] != row[pair[1]]))
    cached = make_feature_lut_exchange_queries(cfg, pairs, active_top_widths=profile)
    base, swapped = cached(prepared, row)
    direct = []
    for a, b in pairs:
        changed = row.copy(); changed[a], changed[b] = row[b], row[a]
        direct.append(active_amplitude(source, state, changed))
    np.testing.assert_allclose(float(base)/expected, 1., atol=3e-11, rtol=0)
    scale = max(abs(np.asarray(direct)))
    np.testing.assert_allclose(np.asarray(swapped)/scale, np.asarray(direct)/scale, atol=3e-11, rtol=0)
