from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


def test_triangular_j1j2_periodic_shell_counts_on_6x6():
    j1 = triangular_j1j2_bonds(6, 6, j1=1.0, j2=0.0, periodic=True)
    j2 = triangular_j1j2_bonds(6, 6, j1=0.0, j2=1.0, periodic=True)
    assert len(j1) == 3 * 36
    assert len(j2) == 3 * 36
    assert all(first < second for first, second, _ in j1 + j2)


def test_triangular_j1j2_open_has_no_wrapped_edges():
    bonds = triangular_j1j2_bonds(2, 3, j1=1.0, j2=0.0)
    assert len(bonds) == 9
    assert (0, 5, 1.0) not in bonds
