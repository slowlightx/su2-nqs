import numpy as np

from su2_nqs.benchmarks.spatial_tree_orbit import rectangular_site_permutations
from su2_nqs.dressing import (
    plaquette_generators,
    projective_real_tangent_matrix,
    reynolds_average_generators,
    tangent_extension_diagnostics,
)


def _computational_permutation_matrix(image):
    n_sites = len(image)
    result = np.zeros((2**n_sites, 2**n_sites))
    for old_index in range(2**n_sites):
        new_index = 0
        for old_site, new_site in enumerate(image):
            bit = (old_index >> (n_sites - old_site - 1)) & 1
            new_index |= bit << (n_sites - new_site - 1)
        result[new_index, old_index] = 1.0
    return result


def test_projective_tangents_remove_norm_and_phase_directions():
    state = np.asarray((1.0, 1.0j)) / np.sqrt(2.0)
    physical = np.asarray((1.0, -1.0j)) / np.sqrt(2.0)
    tangents = np.column_stack((state, 1.0j * state, physical))
    projected = projective_real_tangent_matrix(state, tangents)
    assert np.linalg.norm(projected[:, :2]) < 1.0e-14
    assert np.linalg.matrix_rank(projected, tol=1.0e-12) == 1


def test_tangent_extension_reports_incremental_rank_and_capture():
    state = np.asarray((1.0, 0.0, 0.0), dtype=np.complex128)
    base = np.asarray(((0.0,), (1.0,), (0.0,)), dtype=np.complex128)
    added = np.asarray(((0.0,), (0.0,), (1.0,)), dtype=np.complex128)
    result = tangent_extension_diagnostics(
        state,
        base,
        added,
        residuals={"third": np.asarray((0.0, 0.0, 1.0))},
    )
    assert result["base_rank"] == 1
    assert result["incremental_rank"] == 1
    assert result["added_perpendicular_fraction"] == 1.0
    assert result["residual_capture"]["third"]["base_fraction"] == 0.0
    assert result["residual_capture"]["third"]["combined_fraction"] == 1.0


def test_square_d4_reynolds_projection_kills_all_continuous_real_generators():
    actions = tuple(
        _computational_permutation_matrix(image)
        for image in rectangular_site_permutations(2, 2).values()
    )
    averaged = reynolds_average_generators(plaquette_generators(), actions)
    assert max(np.linalg.norm(generator) for generator in averaged) < 1.0e-13
