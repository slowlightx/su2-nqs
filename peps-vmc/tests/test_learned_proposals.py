"""Independent finite-state normalization, causality and sampling oracles."""
from itertools import combinations

import jax
import numpy as np
import pytest

from experiments.learned_proposals import DefensiveLearnedProposal, FixedMagnetizationAR
from peps_vmc.proposals import CoveringBridgeProposal
from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix

jax.config.update("jax_enable_x64", True)


def rows_at_zero_m(n: int) -> np.ndarray:
    rows = np.zeros((len(list(combinations(range(n), n//2))), n), np.int32)
    for row, sites in zip(rows, combinations(range(n), n//2)):
        row[list(sites)] = 1
    return rows


@pytest.mark.parametrize("kind", ["attention", "gru"])
@pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
def test_exhaustive_normalization_and_spin_flip(kind, n):
    model = FixedMagnetizationAR(n, kind)
    parameters = model.initialize(91001)
    rows = rows_at_zero_m(n)
    law = np.exp(model.log_probability(parameters, rows))
    symmetric = np.exp(model.symmetric_log_probability(parameters, rows))
    assert abs(law.sum()-1) < 1e-12
    assert abs(symmetric.sum()-1) < 1e-12
    np.testing.assert_allclose(symmetric,
        np.exp(model.symmetric_log_probability(parameters, 1-rows)), atol=1e-15, rtol=0)
    assert np.isneginf(np.asarray(model.log_probability(parameters, np.zeros((1, n), np.int32)))[0])


@pytest.mark.parametrize("kind", ["attention", "gru"])
def test_causality_and_independently_enumerated_mask(kind):
    n = 8
    model = FixedMagnetizationAR(n, kind)
    parameters = model.initialize(91002)
    rows = rows_at_zero_m(n)
    raw = np.asarray(model.logits(parameters, rows))
    for i in range(n):
        for prefix in set(map(tuple, rows[:, :i])):
            ids = [k for k, row in enumerate(rows) if tuple(row[:i]) == prefix]
            for action in (0, 1):
                expected = any(tuple(row[:i+1]) == prefix+(action,) for row in rows)
                assert np.all(np.isfinite(raw[ids, i, action]) == expected)
            np.testing.assert_allclose(raw[ids, i], np.broadcast_to(raw[ids[0], i], (len(ids), 2)), atol=1e-14)


@pytest.mark.parametrize("kind", ["attention", "gru"])
def test_sampling_law_and_defensive_bridge(kind):
    n = 4
    model = FixedMagnetizationAR(n, kind)
    p = model.initialize(91003)
    rows = rows_at_zero_m(n)
    allpairs = np.asarray(list(combinations(range(n), 2)), np.int32)
    bridge = CoveringBridgeProposal(np.asarray([[[0, 1], [2, 3]]]), allpairs,
                                   np.full(len(allpairs), 1/len(allpairs)))
    proposal = DefensiveLearnedProposal(model, p, bridge)
    density = proposal.density(rows)
    assert abs(density.sum()-1) < 1e-12
    assert density.min() >= proposal.uniform_floor
    draws, q = proposal.draw(24_000, np.random.default_rng(91004))
    np.testing.assert_array_equal(draws.sum(1), np.full(len(draws), n//2))
    empirical = np.asarray([np.mean(np.all(draws == row, axis=1)) for row in rows])
    np.testing.assert_allclose(empirical, density, atol=.012, rtol=0)
    np.testing.assert_allclose(q[:128], proposal.density(draws[:128]), atol=1e-16)


@pytest.mark.parametrize("n", [0, 3, True])
def test_invalid_site_count(n):
    with pytest.raises(ValueError):
        FixedMagnetizationAR(n, "attention")


def test_defensive_law_keeps_exact_H_squared_contributions_at_target_nodes():
    model = FixedMagnetizationAR(4, "attention")
    rows = rows_at_zero_m(4)[::-1]  # ascending computational-basis integers
    pairs = np.asarray(list(combinations(range(4), 2)), np.int32)
    bridge = CoveringBridgeProposal(np.asarray([[[0, 1], [2, 3]]]), pairs, np.full(6, 1/6))
    proposal = DefensiveLearnedProposal(model, model.initialize(91005), bridge)
    density = proposal.density(rows)
    # Dense reference is ordered by ascending computational-basis integer.
    ham, _ = heisenberg_sz_matrix(4, [(0, 1, 1.), (1, 2, .7), (2, 3, 1.1)], n_up=2)
    psi = np.asarray([1., 0., 0., 0., 0., 0.])
    hp = ham@psi
    assert np.sum(abs(hp[psi == 0])**2) > 0
    np.testing.assert_allclose(np.sum(density*abs(hp)**2/density), psi@(ham@ham)@psi, atol=1e-15)
    assert np.all(density[psi == 0] >= proposal.uniform_floor)


def test_training_exchange_kernel_is_symmetric_and_Born_stationary():
    """Exact enumeration of the finite MC teacher transition, including a node."""
    rows = rows_at_zero_m(4)
    ids = {tuple(row): i for i, row in enumerate(rows)}
    bonds = [(0, 1), (1, 2), (2, 3), (0, 3)]
    bond_weights = np.asarray([.1, .2, .3, .4])
    proposal = np.zeros((len(rows), len(rows)))
    for old, row in enumerate(rows):
        # Long-range conditional opposite-spin selection has 2*2 pairs.
        for a in np.flatnonzero(row == 0):
            for b in np.flatnonzero(row == 1):
                changed = row.copy()
                changed[a], changed[b] = row[b], row[a]
                proposal[ids[tuple(changed)], old] += .5/4
        # Bond draws include null swaps, as in the actual teacher code.
        for (a, b), weight in zip(bonds, bond_weights):
            changed = row.copy()
            changed[a], changed[b] = row[b], row[a]
            proposal[ids[tuple(changed)], old] += .5*weight
    np.testing.assert_allclose(proposal, proposal.T, atol=1e-16)
    np.testing.assert_allclose(proposal.sum(axis=0), 1, atol=1e-16)
    target = np.asarray([.1, 0., .2, 1., .5, .03])**2
    target /= target.sum()
    transition = np.zeros_like(proposal)
    for old in range(len(rows)):
        for new in range(len(rows)):
            if new == old:
                continue
            accept = min(1., target[new]/target[old]) if target[old] else float(target[new] > 0)
            transition[new, old] = proposal[new, old]*accept
        transition[old, old] = 1-transition[:, old].sum()
    np.testing.assert_allclose(transition@target, target, atol=1e-16)
    np.testing.assert_allclose(transition*target[None, :], (transition*target[None, :]).T, atol=1e-16)
