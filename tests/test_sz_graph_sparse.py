import numpy as np

from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix
from su2_nqs.operators.computational_graph_sparse import (
    heisenberg_computational_graph_sparse,
)
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse


def test_sparse_sz_graph_matches_dense_reference_and_basis_order():
    bonds = ((0, 5, 0.7), (1, 4, -0.2), (2, 3, 1.0))
    sparse, configurations = heisenberg_sz_graph_sparse(6, bonds, n_up=3)
    dense, integer_basis = heisenberg_sz_matrix(6, bonds, n_up=3)
    integer_positions = {state: index for index, state in enumerate(integer_basis)}
    order = [
        integer_positions[
            sum(bit << (5 - site) for site, bit in enumerate(configuration))
        ]
        for configuration in configurations
    ]
    np.testing.assert_allclose(sparse.toarray(), dense[np.ix_(order, order)], atol=1e-15)


def test_sparse_full_graph_matches_dense_reference_and_msb_basis_order():
    bonds = ((0, 4, 0.7), (1, 3, -0.2), (2, 4, 1.0))
    sparse, configurations = heisenberg_computational_graph_sparse(5, bonds)
    dense, integer_basis = heisenberg_sz_matrix(5, bonds)
    assert integer_basis == tuple(range(1 << 5))
    assert configurations[0] == (0, 0, 0, 0, 0)
    assert configurations[-1] == (1, 1, 1, 1, 1)
    np.testing.assert_allclose(sparse.toarray(), dense, atol=1e-15)
