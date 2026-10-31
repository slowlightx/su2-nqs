"""Independent operator oracles for conditional spin/chirality bookkeeping."""

import numpy as np

from projects.clebsch_tree_nqs.explorations.su2_handedness_redteam_20261028 import (
    NAMES,
    MATRICES,
    shapley,
    statistics,
)
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


def test_conditional_statistics_agree_with_commuting_operator_trace():
    """Compare algebraic ratios to a physical six-spin conditional RDM."""
    rng = np.random.default_rng(17028)
    z = rng.normal(size=(64, 9)) + 1j * rng.normal(size=(64, 9))
    rho = z @ z.conj().T
    rho /= np.trace(rho)
    t, kappa, p0, p1 = MATRICES
    d = p0 + p1
    u = sum(spin_dot_matrix(6, i, j) for i in (0, 3, 4) for j in (1, 5, 2))
    np.testing.assert_allclose(t @ t, d, atol=2e-14)
    np.testing.assert_allclose(t @ u, u @ t, atol=2e-14)
    np.testing.assert_allclose(3 * t @ u, kappa, atol=2e-14)
    rd = d @ rho @ d
    rd /= np.trace(rd)
    et = np.trace(rd @ t).real
    eu = np.trace(rd @ u).real
    cov = np.trace(rd @ t @ u).real - et * eu
    vt = np.trace(rd @ t @ t).real - et**2
    vu = np.trace(rd @ u @ u).real - eu**2
    x = np.einsum("ab,kba->k", rho, MATRICES).real
    result = statistics(x)
    np.testing.assert_allclose(result[NAMES.index("cov_T_U")], cov, atol=2e-14)
    np.testing.assert_allclose(result[NAMES.index("corr_T_U")], cov / np.sqrt(vt * vu), atol=2e-14)


def test_symmetric_decomposition_closes_for_independent_states():
    """Occupation plus conditional terms reproduce actual delta kappa."""
    rng = np.random.default_rng(18028)
    rhos = []
    for _ in range(2):
        z = rng.normal(size=(64, 4)) + 1j * rng.normal(size=(64, 4))
        rho = z @ z.conj().T
        rhos.append(rho / np.trace(rho))
    a, b = np.einsum("iab,kba->ik", np.asarray(rhos), MATRICES).real
    total, occupation, internal, fraction = shapley(a, b)
    np.testing.assert_allclose(total, b[1] - a[1], atol=1e-14)
    np.testing.assert_allclose(total, occupation + internal, atol=1e-14)
    np.testing.assert_allclose(total * fraction, internal, atol=1e-14)


def test_undefined_screening_channel_is_not_reported_as_zero():
    """No doublet support gives no conditional statistic, rather than zero."""
    values = statistics(np.zeros(4))
    for name in ("conditional_T", "conditional_U", "cov_T_U", "corr_T_U"):
        assert np.isnan(values[NAMES.index(name)])


def test_swap_energy_relabeling_and_complete_triangular_pbc_bond_ledger():
    """Check swap formula against an independent explicit Hilbert matrix."""
    rng = np.random.default_rng(19028)
    state = rng.normal(size=64) + 1j * rng.normal(size=64)
    state /= np.linalg.norm(state)
    rho = np.outer(state, state.conj())
    bonds = ((0, 1, 1.0), (0, 3, 1.0), (1, 5, 0.125), (2, 4, 0.125))
    h = sum(g * spin_dot_matrix(6, i, j) for i, j, g in bonds)
    swap = np.zeros((64, 64))
    for column in range(64):
        bits = list(f"{column:06b}")
        bits[0], bits[3] = bits[3], bits[0]
        swap[int("".join(bits), 2), column] = 1
    perm = np.arange(6)
    perm[[0, 3]] = perm[[3, 0]]
    direct = np.trace(rho @ (swap @ h @ swap - h)).real / 2
    pair_sum = sum(g * np.trace(rho @ (
        spin_dot_matrix(6, perm[i], perm[j]) - spin_dot_matrix(6, i, j)
    )).real for i, j, g in bonds) / 2
    np.testing.assert_allclose(direct, pair_sum, atol=2e-14)
    nn = triangular_j1j2_bonds(6, 6, j1=1.0, j2=0.0, periodic=True)
    nnn = triangular_j1j2_bonds(6, 6, j1=0.0, j2=1.0, periodic=True)
    assert len(nn) == len(nnn) == 108
    assert len({(min(i, j), max(i, j)) for i, j, _ in nn + nnn}) == 216
