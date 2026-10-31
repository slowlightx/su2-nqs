import numpy as np
import pytest

from su2_nqs.operators.cell_spin_channels import cell_casimir_action, weighted_cell_exchange_action
from su2_nqs.operators.cell_spin_transitions import cell_spin_preserving_action, four_site_dimer_action
from su2_nqs.su2.local_schur import cell_schur_tensor


def _pauli_reference():
    spin = []
    for i in range(4):
        components = []
        for pauli in (np.asarray([[0, 1], [1, 0]]) / 2,
                      np.asarray([[0, 1j], [-1j, 0]]) / 2,
                      np.diag([-.5, .5])):
            value = np.ones((1, 1))
            for j in range(4):
                value = np.kron(value, pauli if i == j else np.eye(2))
            components.append(value)
        spin.append(components)
    pairs = {(i,j): sum(spin[i][k] @ spin[j][k] for k in range(3)) for i in range(4) for j in range(i+1,4)}
    return spin, lambda v,i,j: pairs[tuple(sorted((i,j)))] @ v


def test_pinching_matches_independent_spectral_blocks_and_preserves_su2():
    spin, pair = _pauli_reference()
    eye = np.eye(16)
    cells = ((0, 1), (2, 3))
    action = lambda v: weighted_cell_exchange_action(v, pair, *cells, (1., .2), (.3, 1.))
    raw = action(eye)
    keep = cell_spin_preserving_action(eye, action, pair, cells)
    change = raw - keep
    casimirs = [sum((spin[c[0]][k] + spin[c[1]][k]) @ (spin[c[0]][k] + spin[c[1]][k]) for k in range(3)) for c in cells]
    # Two-site Casimir eigenvalues are 0 and 2; direct spectral reference.
    pa, pb = [[eye - q2/2, q2/2] for q2 in casimirs]
    reference = sum(a@b@raw@a@b for a in pa for b in pb)
    np.testing.assert_allclose(keep, reference, atol=2e-14)
    assert np.linalg.norm(keep) > .1 and np.linalg.norm(change) > .1
    for q2 in casimirs:
        assert np.linalg.norm(q2@keep - keep@q2) < 2e-14
    for a in pa:
        for b in pb:
            assert np.linalg.norm(a@b@change@a@b) < 2e-14
    for matrix in (keep, change):
        assert np.linalg.norm(matrix-matrix.conj().T) < 2e-14
        for k in range(3):
            total = sum(spin[i][k] for i in range(4))
            assert np.linalg.norm(total@matrix-matrix@total) < 2e-14
    with pytest.raises(ValueError, match="disjoint"):
        cell_spin_preserving_action(eye, action, pair, ((0,1),(1,2)))


@pytest.mark.parametrize("mode", (0, 1))
def test_dimer_mode_is_pure_reduced_action_identical_in_all_magnetic_copies(mode):
    spin, pair = _pauli_reference()
    operator = four_site_dimer_action(np.eye(16), pair, tuple(range(4)), mode)
    tensor = cell_schur_tensor("square_plaquette")
    norms = []
    for q, multiplicity in ((0, 2), (2, 3), (4, 1)):
        blocks = []
        for m in range(-q,q+1,2):
            u = tensor[:,q,m+4,:multiplicity]
            block = u.T @ operator @ u
            blocks.append(block)
            assert abs(np.trace(block)) < 3e-14
            np.testing.assert_allclose(operator@u, u@block, atol=3e-14)
        for block in blocks:
            np.testing.assert_allclose(block, blocks[0], atol=3e-14)
        norms.append(np.linalg.norm(blocks[0]))
    assert norms[0] > .1 and norms[1] > .1 and norms[2] < 3e-14
    for k in range(3):
        total = sum(spin[i][k] for i in range(4))
        assert np.linalg.norm(total@operator-operator@total) < 3e-14
    # Neither mode is merely a scalar function of the cell Casimir.
    assert np.linalg.norm(operator) > 1
