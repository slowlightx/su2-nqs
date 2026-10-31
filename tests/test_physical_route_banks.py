from __future__ import annotations

import pytest

from su2_nqs.lattices.physical_route_banks import (
    triangular_diagonal_dimer_order,
    triangular_dimer_complete_route_orders,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


@pytest.mark.parametrize("rows,columns", [(4, 4), (6, 6)])
def test_diagonal_dimer_route_completes_third_nn_orientation(
    rows: int, columns: int
) -> None:
    order = triangular_diagonal_dimer_order(rows, columns)
    assert tuple(sorted(order)) == tuple(range(rows * columns))
    j1 = {
        tuple(sorted((first, second)))
        for first, second, _weight in triangular_j1j2_bonds(
            rows, columns, j1=1.0, j2=0.0, periodic=True
        )
    }
    sibling_pairs = {
        tuple(sorted(order[index : index + 2]))
        for index in range(0, len(order), 2)
    }
    assert len(sibling_pairs) == rows * columns // 2
    assert sibling_pairs <= j1

    bank = triangular_dimer_complete_route_orders(rows, columns)
    assert "diagonal_dimers" in bank
    assert len(set(bank.values())) == len(bank)


def test_diagonal_dimer_route_rejects_odd_rows() -> None:
    with pytest.raises(ValueError, match="even rows"):
        triangular_diagonal_dimer_order(3, 4)
