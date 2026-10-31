"""Geometry/cache guards and independent physical root-refinement oracles."""
from dataclasses import replace

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_boundary_response_enrichment import (
    TREE, NODE, _assemble_response_sites, boundary_coupling_matrix, root_ritz_probe,
)
from su2_nqs.frontier.retained_boundary_response import (
    correlated_bath_response_factor, fixed_q_response_frame,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.operators.su2_sector_merge import SectorMergeOperator, reduced_block_from_projected


@pytest.mark.parametrize("j2,inside,outside,bonds", [(0., 25, 31, 56), (.125, 33, 51, 140)])
def test_actual_cut_includes_every_nonzero_J1_J2_bond(j2, inside, outside, bonds):
    support = TREE.coarse_supports[16+NODE]
    physical = triangular_j1j2_bonds(12, 12, j2=j2, periodic=True)
    J, a, b = boundary_coupling_matrix(support, physical)
    assert J.shape == (inside, outside)
    assert np.count_nonzero(J) == bonds
    assert set(a).issubset(support) and not set(b).intersection(support)
    expected = sum(g for i, j, g in physical if (i in support) != (j in support))
    assert J.sum() == pytest.approx(expected)
    if j2:
        _, old, _ = boundary_coupling_matrix(support,
            triangular_j1j2_bonds(12, 12, j2=0., periodic=True))
        assert set(a)-set(old) == {1, 2, 4, 14, 28, 29, 42, 135}
    else:
        K = J @ J.T
        assert np.linalg.matrix_rank(J) == 23
        assert np.sum(abs(K-np.diag(np.diag(K)))) == 74


def test_cut_reversal_is_not_a_second_physical_bond():
    with pytest.raises(ValueError, match="duplicate"):
        boundary_coupling_matrix((0, 1), [(0, 2, 1.), (2, 0, 1.)])
    J, a, b = boundary_coupling_matrix((0, 1), [(2, 0, -.5), (1, 3, 0.)])
    np.testing.assert_array_equal(J, [[-.5]])
    assert a == (0,) and b == (2,)


def test_response_cache_assembly_preserves_site_order_and_complex_column_gauge():
    reference = np.arange(30).reshape(5, 3, 2)+1j*np.arange(30, 60).reshape(5, 3, 2)
    cached = np.concatenate((reference[3], reference[0], reference[4]), axis=1)
    added = np.concatenate((reference[2], reference[1]), axis=1)
    actual = _assemble_response_sites(cached, (3, 0, 4), added, (2, 1), range(5), 2)
    np.testing.assert_array_equal(actual, reference)
    assert np.iscomplexobj(actual)
    with pytest.raises(ValueError, match="partition"):
        _assemble_response_sites(cached, (3, 0, 4), added, (4, 1), range(5), 2)
    with pytest.raises(ValueError, match="shape"):
        _assemble_response_sites(cached[:, :-1], (3, 0, 4), added, (2, 1), range(5), 2)


def test_diagonal_control_keeps_degree_weights_but_removes_only_cross_site_coherence():
    rng = np.random.default_rng(1023)
    R = rng.normal(size=(3, 10, 2))+1j*rng.normal(size=(3, 10, 2))
    J = np.array([[1., 1.], [1., 0.], [0., .125]])
    K = J @ J.T
    coherent, _ = correlated_bath_response_factor(2, R, couplings=J, bath_gram=np.eye(2))
    diagonal, _ = correlated_bath_response_factor(2, R,
        couplings=np.diag(np.sqrt(np.diag(K))), bath_gram=np.eye(3))
    expected = sum(K[i, j]*R[i] @ R[j].conj().T for i in range(3) for j in range(3))
    expected_diagonal = sum(K[i, i]*R[i] @ R[i].conj().T for i in range(3))
    np.testing.assert_allclose(coherent @ coherent.conj().T, expected, atol=8e-14)
    np.testing.assert_allclose(diagonal @ diagonal.conj().T, expected_diagonal, atol=8e-14)
    protected, _ = np.linalg.qr(rng.normal(size=(10, 8)))
    for factor in (coherent, diagonal):
        result = fixed_q_response_frame(2, protected, factor, protect=6)
        assert result.frame.shape == protected.shape
        np.testing.assert_allclose(result.frame[:, :6], protected[:, :6], atol=2e-14)
        np.testing.assert_allclose(result.frame.conj().T @ result.frame, np.eye(8), atol=2e-14)


def _physical_spins(n):
    one = (np.array([[0, 1], [1, 0]], complex)/2,
           np.array([[0, 1j], [-1j, 0]], complex)/2, np.diag([-.5, .5]))
    result = []
    for i in range(n):
        row = []
        for s in one:
            x = np.ones((1, 1))
            for j in range(n):
                x = np.kron(x, s if i == j else np.eye(2))
            row.append(x)
        result.append(row)
    return result


@pytest.mark.parametrize("complex_gauge", [False, True])
def test_same_root_Ritz_and_cross_energy_match_exact_four_spin_physical_H(complex_gauge):
    q0 = np.array([0., -1., 1., 0.])[:, None, None]/np.sqrt(2)
    q2 = np.zeros((4, 1, 3)); q2[0, 0, 0] = q2[3, 0, 2] = 1.
    q2[1:3, 0, 1] = 1/np.sqrt(2)
    if complex_gauge:
        q0 = q0*np.exp(.31j); q2 = q2*np.exp(-.63j)
    spin2 = _physical_spins(2)
    h2 = sum(a @ b for a, b in zip(*spin2, strict=True))
    child = project_block(BlockIsometry(2, {0: q0, 2: q2}), h2)
    left = reduced_block_from_projected(child); right = replace(left, h_by_q={q: .7*h for q, h in left.h_by_q.items()})
    cross = [(0, 0, .3), (0, 1, .9), (1, 0, 1.1)]
    op = SectorMergeOperator(left, right, cross, 0)
    initial = np.array([1., .25]); initial /= np.linalg.norm(initial)
    root, report = root_ritz_probe(left, right, cross, initial, seed=20261023)
    psi = np.kron(child.basis, child.basis) @ op.expand(root[:, 0]).ravel()
    spin = _physical_spins(4)
    h_cross = sum(g*sum(spin[i][axis] @ spin[j+2][axis] for axis in range(3)) for i, j, g in cross)
    physical = h_cross+sum(spin[0][axis] @ spin[1][axis]+.7*spin[2][axis] @ spin[3][axis] for axis in range(3))
    expected = float(np.vdot(psi, physical @ psi).real)
    np.testing.assert_allclose(report["raw_energy_after"], expected, atol=2e-14)
    np.testing.assert_allclose(report["raw_energy_after"], np.linalg.eigvalsh(physical)[0], atol=2e-14)
    np.testing.assert_allclose(report["root_cross_energy"], np.vdot(psi, h_cross @ psi), atol=2e-14)
    np.testing.assert_allclose(physical @ psi, expected*psi, atol=2e-14)
    assert report["raw_energy_after"] <= report["raw_energy_before"]
    assert report["reduced_product_space_residual_norm"] < 2e-14
    assert report["physical_full_H_variance_known"] is False
    assert report["full_A1_energy_known"] is False
