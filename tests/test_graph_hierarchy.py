from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from su2_nqs.benchmarks.multiplet_flux_adaptation import weighted_boundary_score
from su2_nqs.benchmarks.tree_spin_channels import (
    laminar_family_audit,
    physical_internal_supports,
)
from su2_nqs.lattices.graph_hierarchy import (
    EDGE_AGGREGATION_CONTRACT,
    PERIODIC_ALIAS_LIMITATION,
    RELABEL_COVARIANCE_BOUNDARY,
    hamiltonian_graph_hierarchy,
    hierarchy_boundary_score,
    square_pbc_graph_hierarchy,
    triangular_hnf_graph_hierarchy,
    triangular_pbc_graph_hierarchy,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.su2.fusion_tree import FusionNode, FusionTree, Leaf


def _assert_legal(plan, n_sites: int) -> None:
    assert plan.tree.n_leaves == n_sites
    assert tuple(sorted(plan.site_order)) == tuple(range(n_sites))
    assert sorted(site for cell in plan.cells for site in cell) == list(range(n_sites))
    supports = physical_internal_supports(plan.tree, plan.site_order)
    assert len(supports) == n_sites - 2
    assert laminar_family_audit(supports)["is_laminar"]


def test_triangular_n18_recovers_frozen_boundary_minimum_without_permutations() -> None:
    bonds = triangular_j1j2_bonds(
        3, 6, j1=1.0, j2=0.125, periodic=True
    )
    plan = triangular_pbc_graph_hierarchy(3, 6, j1=1.0, j2=0.125)

    _assert_legal(plan, 18)
    assert plan.realized_coarsening == "triangle"
    assert len(plan.cells) == 6
    assert plan.site_order == (
        0, 1, 7,
        2, 3, 9,
        8, 14, 15,
        6, 12, 13,
        4, 5, 11,
        10, 16, 17,
    )
    assert plan.hierarchy_boundary_score == pytest.approx(223.5, abs=1.0e-12)

    supports = physical_internal_supports(plan.tree, plan.site_order)
    independent = sum(
        weighted_boundary_score(row["physical_support"], bonds)
        for row in supports
    )
    assert plan.hierarchy_boundary_score == pytest.approx(independent, abs=1.0e-12)
    assert hierarchy_boundary_score(plan.tree, plan.site_order, bonds) == pytest.approx(
        independent, abs=1.0e-12
    )

    source = (
        Path(__file__).resolve().parents[1]
        / "su2_nqs/lattices/graph_hierarchy.py"
    ).read_text()
    assert "itertools import permutations" not in source
    assert "factorial" not in source


@pytest.mark.parametrize(
    "factory,n_sites,motif",
    (
        (lambda: square_pbc_graph_hierarchy(4, 6, j2=0.5), 24, "square_cycle"),
        (lambda: triangular_pbc_graph_hierarchy(6, 6, j2=0.125), 36, "triangle"),
        (lambda: square_pbc_graph_hierarchy(6, 6, j2=0.5), 36, "square_cycle"),
    ),
)
def test_square_and_triangular_pbc_hierarchies_are_legal(factory, n_sites, motif) -> None:
    plan = factory()
    _assert_legal(plan, n_sites)
    assert plan.realized_coarsening == motif


def test_hnf_and_rectangular_six_by_six_graphs_give_the_same_hierarchy() -> None:
    rectangular = triangular_pbc_graph_hierarchy(6, 6, j2=0.125)
    hnf = triangular_hnf_graph_hierarchy(((6, 0), (0, 6)), j2=0.125)
    _assert_legal(hnf, 36)
    assert hnf.graph_sha256 == rectangular.graph_sha256
    assert hnf.site_order == rectangular.site_order
    assert hnf.cells == rectangular.cells
    assert hnf.hierarchy_boundary_score == rectangular.hierarchy_boundary_score


def test_periodic_pair_multiplicity_and_small_torus_alias_are_explicit() -> None:
    triangular_small = triangular_j1j2_bonds(
        3, 6, j2=0.125, periodic=True
    )
    triangular_large = triangular_j1j2_bonds(
        6, 6, j2=0.125, periodic=True
    )
    triangular_hnf = triangular_hnf_j1j2_bonds(
        ((6, 0), (0, 6)), j2=0.125
    )
    square_large = square_j1j2_bonds(6, 6, j2=0.5, periodic=True)

    for bonds in (triangular_small, triangular_large, triangular_hnf, square_large):
        pairs = [tuple(sorted((first, second))) for first, second, _ in bonds]
        assert len(pairs) == len(set(pairs))

    # On the narrow three-row torus, two distinct J2 images can connect the
    # same pair.  The lattice builder correctly exposes their summed physical
    # Hamiltonian weight; the graph hierarchy intentionally preserves it.
    assert Counter(weight for _, _, weight in triangular_small) == {
        1.0: 54,
        0.125: 18,
        0.25: 18,
    }

    # The N=36 production canaries contain no such pair alias.  Rectangular
    # and HNF descriptions expose precisely the same 6x6 triangular graph.
    expected_triangle = {1.0: 108, 0.125: 108}
    assert Counter(weight for _, _, weight in triangular_large) == expected_triangle
    assert Counter(weight for _, _, weight in triangular_hnf) == expected_triangle
    assert triangular_hnf == triangular_large
    assert Counter(weight for _, _, weight in square_large) == {
        1.0: 72,
        0.5: 72,
    }


@pytest.mark.parametrize(
    "bonds,n_sites,motif",
    (
        (triangular_j1j2_bonds(3, 6, j2=0.125, periodic=True), 18, "triangle"),
        (square_j1j2_bonds(4, 6, j2=0.5, periodic=True), 24, "square_cycle"),
    ),
)
def test_determinism_includes_bond_row_order_invariance(bonds, n_sites, motif) -> None:
    forward = hamiltonian_graph_hierarchy(n_sites, bonds, motif=motif)
    repeated = hamiltonian_graph_hierarchy(n_sites, bonds, motif=motif)
    reversed_rows = hamiltonian_graph_hierarchy(
        n_sites, tuple(reversed(bonds)), motif=motif
    )
    assert forward == repeated == reversed_rows


def test_duplicate_edge_reduction_is_row_order_independent() -> None:
    first = ((0, 1, 0.75), (1, 0, 0.25), (1, 2, 1.0), (2, 3, 1.0))
    second = tuple(reversed(first))
    left = hamiltonian_graph_hierarchy(4, first)
    right = hamiltonian_graph_hierarchy(4, second)
    assert left == right


def test_mixed_sign_duplicate_contract_is_canonical_then_absolute() -> None:
    tree = FusionTree(
        FusionNode(
            FusionNode(Leaf(0), Leaf(1)),
            FusionNode(Leaf(2), Leaf(3)),
        )
    )
    raw = ((0, 2, 4.0), (2, 0, -1.0), (1, 3, 2.0))
    canonical = ((0, 2, 3.0), (1, 3, 2.0))
    supports = physical_internal_supports(tree, tuple(range(4)))
    raw_discovery = sum(
        weighted_boundary_score(row["physical_support"], raw) for row in supports
    )
    canonical_discovery = sum(
        weighted_boundary_score(row["physical_support"], canonical)
        for row in supports
    )

    assert raw_discovery == pytest.approx(14.0)
    assert canonical_discovery == pytest.approx(10.0)
    assert hierarchy_boundary_score(tree, tuple(range(4)), raw) == pytest.approx(
        canonical_discovery
    )

    plan = hamiltonian_graph_hierarchy(4, raw)
    assert plan.edge_aggregation_contract == EDGE_AGGREGATION_CONTRACT
    assert not plan.periodic_image_provenance_available
    assert plan.periodic_alias_limitation == PERIODIC_ALIAS_LIMITATION
    assert "cannot reconstruct" in plan.periodic_alias_limitation


def test_incomplete_triangle_cover_falls_back_to_legal_vertex_bisection() -> None:
    # The canonical greedy triangle pass intentionally does not solve an
    # NP-hard exact-cover problem.  This 4x6 torus exercises its safe fallback.
    bonds = triangular_j1j2_bonds(4, 6, j2=0.125, periodic=True)
    plan = hamiltonian_graph_hierarchy(24, bonds, motif="triangle")
    _assert_legal(plan, 24)
    assert plan.requested_motif == "triangle"
    assert plan.realized_coarsening == "vertices"
    assert all(len(cell) == 1 for cell in plan.cells)


def test_relabelling_boundary_is_explicit_for_symmetric_graphs() -> None:
    plan = triangular_pbc_graph_hierarchy(3, 6, j2=0.125)
    assert plan.canonical_label_frame_used
    assert plan.relabel_covariance_boundary == RELABEL_COVARIANCE_BOUNDARY
    assert "arbitrary site-relabelling covariance is not guaranteed" in (
        plan.relabel_covariance_boundary
    )
    assert plan.work.label_tie_breaks > 0


def test_n36_polynomial_work_canary_is_small_and_deterministic() -> None:
    triangle = triangular_pbc_graph_hierarchy(6, 6, j2=0.125)
    square = square_pbc_graph_hierarchy(6, 6, j2=0.5)
    assert triangle.hierarchy_boundary_score == pytest.approx(563.0, abs=1.0e-12)
    assert square.hierarchy_boundary_score == pytest.approx(478.0, abs=1.0e-12)
    for plan in (triangle, square):
        _assert_legal(plan, 36)
        # Operation counts, rather than machine-dependent wall time, are the
        # reproducible scalability canary.  The observed work is far below
        # even N^3 and contains no cell-order factorial.
        assert plan.work.motif_candidates < 36**2
        assert plan.work.motif_feasibility_checks < 36**4
        assert plan.work.bisection_cut_evaluations < 36**3
        assert plan.work.bisection_edge_accumulations < 36**4
        assert plan.work.bisection_swap_candidates < 36**3
        assert plan.work.accepted_balanced_swaps < 36**2


@pytest.mark.parametrize(
    "n_sites,bonds,match",
    (
        (1, ((0, 0, 1.0),), "at least two"),
        (4, ((0, 4, 1.0),), "outside"),
        (4, ((0, 1, float("nan")),), "finite"),
        (4, (), "nonzero edge"),
    ),
)
def test_invalid_graphs_fail_closed(n_sites, bonds, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        hamiltonian_graph_hierarchy(n_sites, bonds)
