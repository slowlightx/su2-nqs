import numpy as np

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.local_indistinguishability import (
    pure_state_transition_partial_trace,
)
from su2_nqs.dressing import (
    apply_local_gate,
    multiplicity_component_atlas,
    plaquette_unitary,
    real_multiplicity_overlap_ceiling,
    schur_transition_blocks,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def _random_singlet(n_sites: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paths = enumerate_paths(n_sites, 0)
    coupled = coupled_basis_matrix(paths, 0, 0)
    coordinates = rng.normal(size=len(paths)) + 1.0j * rng.normal(size=len(paths))
    coordinates /= np.linalg.norm(coordinates)
    return coupled @ coordinates


def test_schur_transition_reduction_is_exact_for_global_singlets():
    left = _random_singlet(6, 1201)
    right = _random_singlet(6, 1202)
    transition = pure_state_transition_partial_trace(left, right, 6, (0, 2, 4, 5))
    _, audit = schur_transition_blocks(transition)
    assert audit["relative_non_scalar_block_norm"] < 2.0e-13
    assert audit["maximum_m_slice_error"] < 2.0e-13


def test_identity_ceiling_equals_bare_fidelity():
    left = _random_singlet(6, 1301)
    right = _random_singlet(6, 1302)
    transition = pure_state_transition_partial_trace(left, right, 6, (0, 1, 4, 5))
    result = real_multiplicity_overlap_ceiling(transition, active_q=())
    fidelity = abs(np.vdot(right, left)) ** 2
    np.testing.assert_allclose(result["maximum_fidelity"], fidelity, atol=3.0e-13)


def test_connected_real_gate_ceiling_recovers_a_generated_target():
    base = _random_singlet(4, 1401)
    target = apply_local_gate(
        base,
        plaquette_unitary((0.31, -0.27, 0.44, 0.19)),
        (0, 1, 2, 3),
    )
    target *= np.exp(0.37j)
    transition = pure_state_transition_partial_trace(base, target, 4, (0, 1, 2, 3))
    atlas = multiplicity_component_atlas(transition)
    assert atlas["connected_SO2xSO3"]["maximum_fidelity"] > 1.0 - 2.0e-12
    assert atlas["identity"]["maximum_fidelity"] < 0.999
