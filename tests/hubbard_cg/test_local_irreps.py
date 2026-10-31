import pytest

from su2_nqs.su2.hubbard_cg_irrep import (
    HubbardLocalIrrep,
    hubbard_local_irrep,
    token_quantum_number_tables,
)


def test_four_physical_tokens_have_frozen_charge_spin_labels():
    assert tuple(hubbard_local_irrep(token) for token in range(4)) == (
        HubbardLocalIrrep(0, 0, 0, 0, 0),
        HubbardLocalIrrep(1, 1, 1, 1, 1),
        HubbardLocalIrrep(2, 1, 1, -1, 1),
        HubbardLocalIrrep(3, 2, 0, 0, 2),
    )
    assert token_quantum_number_tables() == (
        (0, 1, 1, 2),
        (0, 1, 1, 0),
        (0, 1, -1, 0),
        (0, 1, 1, 2),
    )


@pytest.mark.parametrize("token", [-1, 4, 0.5, True])
def test_invalid_physical_tokens_are_rejected(token):
    with pytest.raises((TypeError, ValueError)):
        hubbard_local_irrep(token)
