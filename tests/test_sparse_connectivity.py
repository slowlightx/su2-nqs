import numpy as np
from scipy import sparse

from su2_nqs.operators.sparse_connectivity import SparseBasisConnectivity


def test_sparse_basis_connectivity_uses_new_old_column_convention():
    matrix = sparse.csr_matrix(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    connectivity = SparseBasisConnectivity(matrix, [(0,), (1,)])
    states, values = connectivity.get_conn((1,))
    assert states == ((0,), (1,))
    np.testing.assert_allclose(values, [2.0, 4.0])
