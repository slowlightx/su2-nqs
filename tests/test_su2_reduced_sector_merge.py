"""All-q reduced scalar action versus independent magnetic CG expansion."""
from dataclasses import replace

import numpy as np
import pytest

from su2_nqs.operators.su2_reduced_sector_merge import (
    ReducedSectorMergeOperator, scalar_vector_recoupling,
)
from su2_nqs.operators.su2_sector_merge import (
    SectorMergeOperator, compact_reduced_block, reduced_block_from_projected,
)
from su2_nqs.operators.su2_singlet_center import SingletCutOperator
from test_su2_sector_merge import _children


@pytest.mark.parametrize("complex_gauge", [False, True])
def test_reduced_scalar_all_q_matches_full_magnetic_product_and_physical_H(complex_gauge):
    a, b, bonds, physical = _children(complex_gauge)
    left, right = [compact_reduced_block(reduced_block_from_projected(x)) for x in (a, b)]
    physical_basis = np.kron(a.basis, b.basis)
    reference = physical_basis.conj().T @ physical @ physical_basis
    rng = np.random.default_rng(2026110317)
    for q in (0, 2, 4, 6):
        old = SectorMergeOperator(left, right, bonds, q)
        new = ReducedSectorMergeOperator(left, right, bonds, q)
        assert new.channels == old.channels
        vector = rng.normal(size=new.dimension)+1j*rng.normal(size=new.dimension)
        observed = new.matvec(vector)
        np.testing.assert_allclose(observed, old.matvec(vector), atol=2e-13)
        for m in range(-q, q+1, 2):
            x = old.expand(vector, m)
            expected = old.project((reference @ x.ravel()).reshape(x.shape), m)
            np.testing.assert_allclose(observed, expected, atol=3e-13)
        dense = new.dense_reference()
        np.testing.assert_allclose(dense, dense.conj().T, atol=2e-13)
        np.testing.assert_allclose(dense, old.dense_reference(), atol=2e-13)
        assert new.diagnostics["magnetic_product_vectors_constructed"] is False


def test_q0_reduces_to_independent_singlet_center_formula():
    a, b, bonds, _ = _children(True)
    left, right = [compact_reduced_block(reduced_block_from_projected(x)) for x in (a, b)]
    actual = ReducedSectorMergeOperator(left, right, bonds, 0)
    reference = SingletCutOperator(left, right, bonds)
    np.testing.assert_allclose(actual.dense_reference(), reference.matrix, atol=2e-13)
    for qo, qi in ((1, 1), (1, 3), (3, 1), (3, 3)):
        np.testing.assert_allclose(scalar_vector_recoupling(qo, qo, qi, qi, 0),
                                   -np.sqrt((qo+1)/(qi+1)), atol=3e-15)


def test_complex_offdiagonal_internal_H_is_kept_and_duplicate_bonds_add():
    a, b, bonds, _ = _children(True)
    left, right = [compact_reduced_block(reduced_block_from_projected(x)) for x in (a, b)]
    h = {q: x.copy() for q, x in left.h_by_q.items()}
    h[1] = h[1]+np.asarray([[.1, .2j], [-.2j, -.1]])
    left = replace(left, h_by_q=h)
    duplicated = bonds+[(0, 0, -.025), (0, 0, .025)]
    for q in (0, 2, 4, 6):
        new = ReducedSectorMergeOperator(left, right, duplicated, q)
        old = SectorMergeOperator(left, right, bonds, q)
        np.testing.assert_allclose(new.dense_reference(), old.dense_reference(), atol=3e-13)


def test_bad_quantum_numbers_shapes_bounds_and_spin_data_fail_closed():
    a, b, bonds, _ = _children()
    left, right = [compact_reduced_block(reduced_block_from_projected(x)) for x in (a, b)]
    with pytest.raises(ValueError, match="integer"):
        scalar_vector_recoupling(True, 1, 1, 1, 0)
    with pytest.raises(ValueError, match="absent"):
        ReducedSectorMergeOperator(left, right, bonds, 1)
    with pytest.raises(MemoryError):
        ReducedSectorMergeOperator(left, right, bonds, 0, memory_limit_bytes=1)
    with pytest.raises(ValueError, match="valid child site"):
        ReducedSectorMergeOperator(left, right, [(9, 0, 1.)], 0)
    op = ReducedSectorMergeOperator(left, right, bonds, 2)
    with pytest.raises(ValueError, match="finite reduced"):
        op.matvec(np.ones(op.dimension+1))
    with pytest.raises(ValueError, match="small-system bound"):
        op.dense_reference(max_dimension=1)
