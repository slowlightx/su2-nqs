import numpy as np

from su2_nqs.benchmarks.common_basis_gamma_a1 import (
    triangular_hnf_gamma_a1_circuit,
)
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.fusion_spatial_bridge import (
    FixedMLeftFusionTransform,
    certify_fusion_gamma_a1,
    computational_reynolds_project,
    spin_raising_relative_norm,
    triangular_hnf_120_teacher,
    triangular_square_pi_zero_dsl_teacher,
    triangular_square_stripe_teacher,
)
from su2_nqs.benchmarks.projected_singlet_agp import (
    triangular_hnf_gamma_a1_group,
)


def _configuration_integer(configuration):
    result = 0
    for bit in configuration:
        result = (result << 1) | bit
    return result


def test_fixed_m_transform_matches_independent_full_computational_columns():
    transform = FixedMLeftFusionTransform.build(6, target_q=0, total_m=0)
    full = coupled_basis_matrix(transform.paths, total_q=0, total_m=0)
    rows = [_configuration_integer(state) for state in transform.configurations]
    np.testing.assert_allclose(transform.matrix, full[rows], atol=2.0e-15)
    assert transform.orthonormality_error() < 2.0e-15

    rng = np.random.default_rng(2026091602)
    fusion = rng.normal(size=transform.fusion_dimension) + 1j * rng.normal(
        size=transform.fusion_dimension
    )
    computational = transform.fusion_to_computational(fusion)
    np.testing.assert_allclose(
        transform.computational_to_fusion(computational), fusion, atol=4.0e-15
    )
    assert spin_raising_relative_norm(computational, transform.configurations) < 3.0e-15


def test_computational_and_fixed_fusion_a1_projectors_intertwine():
    # The 2x2 triangular quotient has no q=0 Gamma.A1 copy, so use the
    # smallest registered D1 teacher torus with a nonzero target sector.
    supercell = ((6, 4), (0, 2))
    transform = FixedMLeftFusionTransform.build(12, target_q=0, total_m=0)
    circuit = triangular_hnf_gamma_a1_circuit(supercell, target_q=0)
    group = triangular_hnf_gamma_a1_group(supercell)
    rng = np.random.default_rng(2026091603)
    fusion = rng.normal(size=transform.fusion_dimension) + 1j * rng.normal(
        size=transform.fusion_dimension
    )
    audit = certify_fusion_gamma_a1(fusion, transform, circuit, group)
    assert audit["input_q_sector_reconstruction_relative_error"] < 8.0e-15
    assert audit["input_Splus_relative_norm"] < 8.0e-15
    assert audit["projector_intertwining_relative_error"] < 2.0e-13
    assert audit["projected_Gamma_A1_relative_residual"] < 2.0e-13
    assert audit["projected_Splus_relative_norm"] < 2.0e-13


def test_registered_small_physical_teacher_compactions_match_reynolds():
    supercell12 = ((6, 4), (0, 2))
    transform12 = FixedMLeftFusionTransform.build(12)
    group12 = triangular_hnf_gamma_a1_group(supercell12)
    raw120, compact120 = triangular_hnf_120_teacher(
        transform12.configurations, supercell12
    )
    reynolds120 = computational_reynolds_project(
        raw120, transform12.configurations, group12
    )
    np.testing.assert_allclose(compact120, reynolds120, atol=3.0e-15)
    assert spin_raising_relative_norm(reynolds120, transform12.configurations) < 1.0e-13

    transform4 = FixedMLeftFusionTransform.build(4)
    raw_stripe, compact_stripe, components = triangular_square_stripe_teacher(
        transform4.configurations, 2
    )
    reynolds_stripe = computational_reynolds_project(
        raw_stripe,
        transform4.configurations,
        triangular_hnf_gamma_a1_group(((2, 0), (0, 2))),
    )
    assert components.shape == (6, 3)
    np.testing.assert_allclose(compact_stripe, reynolds_stripe, atol=3.0e-15)


def test_n16_dsl_helper_has_registered_closed_shell_orbit():
    # Four configurations independently exercise the determinant signs and
    # all three boundary sectors without making the unit suite exhaustive.
    configurations = (
        (1,) * 8 + (0,) * 8,
        tuple(site % 2 for site in range(16)),
        tuple((site // 4) % 2 for site in range(16)),
        tuple((site + site // 4) % 2 for site in range(16)),
    )
    raw, compact, components, gaps = triangular_square_pi_zero_dsl_teacher(
        configurations, 4
    )
    assert raw.shape == compact.shape == (4,)
    assert components.shape == (4, 3)
    np.testing.assert_allclose(gaps, np.full(3, 4.0), atol=4.0e-14)
    np.testing.assert_allclose(compact, np.mean(components, axis=1), atol=1.0e-16)
