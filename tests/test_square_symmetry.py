import pytest

from su2_nqs.lattices.square_symmetry import (
    square_momentum_label,
    square_site_generators,
)


def test_square_site_generators_use_active_row_major_convention() -> None:
    generators = square_site_generators(4)
    assert generators["tx"][0] == 1
    assert generators["ty"][0] == 4
    assert generators["c4"][1] == 4
    assert generators["mx"][1] == 3
    assert all(sorted(image) == list(range(16)) for image in generators.values())


@pytest.mark.parametrize(
    ("momentum", "label"),
    (((0, 0), "Gamma"), ((2, 2), "M"), ((2, 0), "X"), ((0, 2), "Y")),
)
def test_square_momentum_labels(momentum: tuple[int, int], label: str) -> None:
    assert square_momentum_label(momentum, 4) == label
