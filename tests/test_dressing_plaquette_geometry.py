from su2_nqs.benchmarks.spatial_tree_orbit import rectangular_site_permutations
from su2_nqs.dressing import (
    square_plaquettes,
    symmetry_expand_ordered_plaquettes,
    triangular_rhombi,
)


def test_square_and_triangular_cell_counts_on_three_by_four_patch():
    assert len(square_plaquettes(3, 4)) == 6
    assert len(triangular_rhombi(3, 4)) == 13
    assert len({frozenset(cell) for cell in triangular_rhombi(3, 4)}) == 13


def test_ordered_symmetry_expansion_is_closed_without_sorting_local_slots():
    group = rectangular_site_permutations(3, 4)
    expanded = symmetry_expand_ordered_plaquettes(
        square_plaquettes(3, 4), group
    )
    expanded_set = set(expanded)
    for image in group.values():
        assert {
            tuple(image[site] for site in cell) for cell in expanded
        } == expanded_set
    assert any(tuple(sorted(cell)) != cell for cell in expanded)


def test_triangular_rhombi_use_four_distinct_in_range_sites():
    for cell in triangular_rhombi(3, 4):
        assert len(set(cell)) == 4
        assert all(0 <= site < 12 for site in cell)
