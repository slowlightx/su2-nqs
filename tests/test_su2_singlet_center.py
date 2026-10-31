"""Independent physical singlet-H and q-Schmidt oracles through six spins."""
import numpy as np
import pytest

from su2_nqs.operators.su2_singlet_center import SingletCutOperator, compress_singlet_center
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from test_su2_three_block_center import child, physical_action


@pytest.mark.parametrize('sites', [2, 3])
@pytest.mark.parametrize('complex_gauge', [False, True])
def test_singlet_center_matches_complete_physical_H(sites, complex_gauge):
    internal = [(0, 1, 1.)]+([(1, 2, .7), (0, 2, .125)] if sites == 3 else [])
    a, ua = child(sites, internal, 921 if complex_gauge else None)
    e, ue = child(sites, internal, 923 if complex_gauge else None)
    cross = [(0, sites-1, 1.2), (1, 0, .3)]
    op = SingletCutOperator(a, e, cross)
    columns = []
    for q in op.slices:
        for ai in range(a.q_widths[q]):
            for ei in range(e.q_widths[q]):
                value = np.zeros(2**(2*sites), complex)
                for mi, m in enumerate(range(-q, q+1, 2)):
                    value += clebsch_gordan(q, m, q, -m, 0, 0)*np.kron(
                        ua[q][:, ai, mi], ue[q][:, ei, (-m+q)//2])
                columns.append(value)
    u = np.column_stack(columns)
    bonds = internal+[(i+sites, j+sites, g) for i, j, g in internal]
    bonds += [(i, j+sites, g) for i, j, g in cross]
    reference = u.conj().T@physical_action(u, 2*sites, bonds)
    assert np.max(abs(op.matrix-reference)) < 4e-13
    assert np.max(abs(op.matrix-op.matrix.conj().T)) < 4e-13
    c = np.arange(op.dimension)+1j*np.arange(op.dimension)[::-1]
    assert np.array_equal(op.encode(op.decode(c)), c)


def test_q_schmidt_compress_complex_ket_norm_fidelity_and_rank():
    rng = np.random.default_rng(260927)
    c = {q: rng.normal(size=(5, 6))+1j*rng.normal(size=(5, 6)) for q in (0, 2, 4)}
    norm = np.sqrt(sum(np.linalg.norm(v)**2 for v in c.values()))
    c = {q: v/norm for q, v in c.items()}
    rotations, reduced, info = compress_singlet_center(c, {q: 3 for q in c})
    reconstructed = {q: rotations[q]@reduced[q] for q in c}
    overlap = sum(np.vdot(c[q], reconstructed[q]) for q in c)
    assert abs(abs(overlap)**2-info['retained_norm_squared']) < 2e-15
    assert abs(sum(np.linalg.norm(v)**2 for v in reconstructed.values())-1) < 2e-15
    fullu, fullc, fullinfo = compress_singlet_center(c, {q: 5 for q in c})
    assert max(np.max(abs(fullu[q]@fullc[q]-c[q])) for q in c) < 2e-15
    assert fullinfo['discarded_norm_squared'] < 2e-15


@pytest.mark.parametrize('q', [True, -1, 2.])
def test_q_schmidt_compression_rejects_invalid_doubled_spin(q):
    with pytest.raises((ValueError, TypeError)):
        compress_singlet_center({q: np.ones((1, 1))}, {q: 1})
