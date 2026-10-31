"""Independent physical-vector overlap oracle for zero-query A1 postprocessing."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_overlap import paired_projected_overlap


def test_unequal_norm_complex_overlap_matches_exhaustive_physical_vectors():
    p = np.asarray([[1., 2j], [0., .4], [1j, 1.], [.2, -.7j]], complex)
    rows = np.tile(np.arange(4), 8)
    groups = np.repeat(np.arange(8), 4)
    result = paired_projected_overlap(p[rows], np.full(32, .25), groups, seed=91)
    cross = np.vdot(p[:, 0], p[:, 1])
    overlap = cross/np.prod(np.linalg.norm(p, axis=0))
    expected = [cross.real, cross.imag, overlap.real, overlap.imag,
                abs(overlap), np.arccos(abs(overlap))]
    np.testing.assert_allclose(result['values'], expected, atol=2e-15)
    np.testing.assert_allclose(result['independent_block_jackknife_SE'], 0, atol=2e-15)


def test_invalid_density_is_rejected_without_dropping_target_nodes():
    p = np.ones((16, 2), complex)
    groups = np.repeat(np.arange(4), 4)
    q = np.ones(16)
    for invalid in (0., np.nan):
        q[0] = invalid
        with pytest.raises(ValueError):
            paired_projected_overlap(p, q, groups, seed=2)
