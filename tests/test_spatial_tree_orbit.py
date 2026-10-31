import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.spatial_tree_orbit import (
    bond_automorphisms,
    coherent_tree_orbit_analysis,
    rectangular_site_permutations,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.permutation import (
    apply_site_permutation_fusion,
    site_permutation_fusion_matrix,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def _computational_permutation_apply(
    vectors: np.ndarray, image: tuple[int, ...]
) -> np.ndarray:
    result = np.zeros_like(vectors)
    n = len(image)
    for old_index in range(1 << n):
        new_index = 0
        for old_site, new_site in enumerate(image):
            bit = (old_index >> (n - old_site - 1)) & 1
            new_index |= bit << (n - new_site - 1)
        result[new_index] = vectors[old_index]
    return result


@pytest.mark.parametrize("n,target_q", ((4, 0), (6, 0), (6, 2)))
def test_reusable_site_permutation_matches_computational_reference(
    n: int, target_q: int
) -> None:
    rng = np.random.default_rng(20260908 + 10 * n + target_q)
    image = tuple(int(value) for value in rng.permutation(n))
    paths = enumerate_paths(n, target_q)
    coupled = coupled_basis_matrix(paths, target_q, target_q)
    expected = coupled.conj().T @ _computational_permutation_apply(coupled, image)
    actual = site_permutation_fusion_matrix(image, target_q=target_q).toarray()
    np.testing.assert_allclose(actual, expected, atol=4.0e-13)
    np.testing.assert_allclose(
        apply_site_permutation_fusion(
            np.eye(len(paths)), image, target_q=target_q
        ),
        expected,
        atol=4.0e-13,
    )


def test_open_square_and_triangular_rectangles_have_expected_automorphisms() -> None:
    candidates = rectangular_site_permutations(3, 4)
    square = bond_automorphisms(
        candidates,
        square_j1j2_bonds(3, 4, j1=1.0, j2=0.6, periodic=False),
    )
    triangular = bond_automorphisms(
        candidates,
        triangular_j1j2_bonds(3, 4, j1=1.0, j2=0.125, periodic=False),
    )
    assert tuple(square) == (
        "identity",
        "reflect_rows",
        "reflect_columns",
        "rotate_180",
    )
    assert tuple(triangular) == ("identity", "rotate_180")


def test_coherent_orbit_projection_improves_without_new_components() -> None:
    swap = np.asarray(((0.0, 1.0), (1.0, 0.0)))
    hamiltonian = -swap
    representative = np.asarray((np.sqrt(0.8), np.sqrt(0.2)))
    ground = np.asarray((1.0, 1.0)) / np.sqrt(2.0)
    result = coherent_tree_orbit_analysis(
        representative,
        {"identity": np.eye(2), "swap": swap},
        hamiltonian,
        reference=ground,
    )

    assert result["component_count"] == 2
    assert result["orbit_effective_rank"] == 2
    assert result["representative"]["energy"] == pytest.approx(-0.8)
    assert result["equal_coherent"]["energy"] == pytest.approx(-1.0)
    assert result["equal_coherent"]["reference_fidelity"] == pytest.approx(1.0)
    assert result["orbit_ritz"]["energy"] == pytest.approx(-1.0)
    assert result["maximum_hamiltonian_covariance_error"] < 1.0e-14
    assert result["maximum_equal_orbit_closure_error"] < 1.0e-14
    assert result["projector_hermiticity_error"] < 1.0e-14
    assert result["projector_idempotency_error"] < 1.0e-14
    assert result["symmetry_sector_decomposition"][
        "trivial_sector_weight"
    ] == pytest.approx(0.9)
    assert result["symmetry_sector_decomposition"][
        "raw_energy_error_removed_fraction"
    ] == pytest.approx(1.0)
    assert result["mixture_proposal"]["population_ess_fraction"] == pytest.approx(
        1.0
    )
