from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.adaptive_projected_pair_carrier import (
    batched_singlet_agp_amplitudes,
)
from su2_nqs.benchmarks.clebsch_tree_exact import reynolds_project_state
from su2_nqs.benchmarks.four_cell_additive_carrier import _case_data
from su2_nqs.benchmarks.paired_z2_carrier import paired_z2_kernel_bank


@pytest.mark.parametrize(
    "geometry",
    ("triangular_n12", "triangular_n16", "triangular_n36", "square_n16"),
)
def test_paired_z2_bank_is_gapped_symmetric_and_has_discrete_igg(
    geometry: str,
) -> None:
    bank = paired_z2_kernel_bank(geometry)  # type: ignore[arg-type]
    assert len(bank) == 3
    assert len({carrier.label for carrier in bank}) == len(bank)
    for carrier in bank:
        assert carrier.quasiparticle_gap > 0.39
        assert carrier.u_condition_number < 50.0
        assert carrier.pairing_symmetry_residual < 2.0e-13
        assert carrier.continuous_igg_nullity == 0
        assert carrier.smallest_nonzero_igg_singular_value > 0.1
        np.testing.assert_allclose(carrier.pairing, carrier.pairing.T, atol=2.0e-14)
        np.testing.assert_allclose(np.diag(carrier.pairing), 0.0, atol=0.0)
        np.testing.assert_allclose(np.linalg.norm(carrier.pairing), 1.0, atol=2.0e-14)


@pytest.mark.parametrize(
    ("case_name", "geometry"),
    (
        ("triangular_n12_j2_0125", "triangular_n12"),
        ("square_n16_j2_05", "square_n16"),
    ),
)
def test_projected_paired_z2_carriers_lie_in_exact_su2_a1_space(
    case_name: str,
    geometry: str,
) -> None:
    data = _case_data(case_name)
    singlet_basis = np.asarray(data["singlet_basis"])
    # _case_data has the same deterministic fixed-Sz ordering as the columns
    # of the singlet basis; recover its binary rows from the registered basis.
    from su2_nqs.benchmarks.four_cell_schur import (
        four_cell_schur_sector_basis_matrix,
    )
    from su2_nqs.benchmarks.four_cell_schur_frontier import (
        SQUARE_N16_SUPPORTS,
        TRIANGULAR_N12_SUPPORTS,
    )

    kind = "triangular_triangle" if geometry == "triangular_n12" else "square_plaquette"
    supports = TRIANGULAR_N12_SUPPORTS if geometry == "triangular_n12" else SQUARE_N16_SUPPORTS
    _basis, rows = four_cell_schur_sector_basis_matrix(
        kind, 0, target_m2=0, supports=supports
    )
    states = np.asarray(rows, dtype=np.int8)
    for carrier in paired_z2_kernel_bank(geometry):  # type: ignore[arg-type]
        raw = batched_singlet_agp_amplitudes(states, carrier.pairing)
        projected = np.asarray(
            reynolds_project_state(jnp.asarray(raw), data["indices"])
        )
        assert np.linalg.norm(projected) > 1.0e-10
        reduced = singlet_basis.conj().T @ projected
        residual = np.linalg.norm(projected - singlet_basis @ reduced) / np.linalg.norm(
            projected
        )
        assert residual < 3.0e-13
        np.testing.assert_allclose(
            projected,
            np.asarray(
                reynolds_project_state(jnp.asarray(projected), data["indices"])
            ),
            atol=3.0e-13,
        )


def test_triangular_n16_projected_paired_z2_carriers_are_exact_su2_a1() -> None:
    from su2_nqs.benchmarks.clebsch_tree_exact import (
        physical_permutation_index_table,
    )
    from su2_nqs.benchmarks.fusion_spatial_bridge import (
        spin_raising_relative_norm,
    )
    from su2_nqs.benchmarks.qsl_three_arm import (
        triangular_pi_flux_operator_components,
    )
    from su2_nqs.lattices.spatial_symmetry import (
        triangular_hnf_gamma_a1_group,
    )

    data = triangular_pi_flux_operator_components(0.125)
    states = np.asarray(data.configurations, dtype=np.int8)
    group = triangular_hnf_gamma_a1_group(((4, 0), (0, 4)))
    indices = physical_permutation_index_table(states, group)
    for carrier in paired_z2_kernel_bank("triangular_n16"):
        raw = batched_singlet_agp_amplitudes(states, carrier.pairing)
        projected = np.mean(raw[indices], axis=0)
        projected /= np.linalg.norm(projected)
        assert spin_raising_relative_norm(projected, states) < 5.0e-13
        assert np.linalg.norm(np.mean(projected[indices], axis=0) - projected) < 5.0e-13
