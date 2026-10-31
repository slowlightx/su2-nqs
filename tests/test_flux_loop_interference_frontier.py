from __future__ import annotations

import pytest

from su2_nqs.benchmarks.flux_loop_interference_frontier import (
    flux_loop_interference_case,
    triangular_n12_saturation_control,
)


def test_triangular_n12_is_explicitly_rejected_as_a_mechanism_gate() -> None:
    result = triangular_n12_saturation_control()
    assert result["carrier_bank_rank"] == 4
    assert result["exact_s0_gamma_a1_dimension"] == 4
    assert result["saturated"]


@pytest.mark.parametrize(
    "case",
    ("triangular_n16_j2_0.125", "square_n16_j2_0.5"),
)
def test_flux_loop_mechanism_beats_new_scrambles_but_not_matched_krylov(
    case: str,
) -> None:
    result = flux_loop_interference_case(
        case,  # type: ignore[arg-type]
        scramble_seeds=(2026092501, 2026092502, 2026092503),
    )
    gates = result["gates"]
    assert gates["exact_s0_gamma_a1"]
    assert gates["phase_bank_improves_two_lanczos_energy_and_variance"]
    assert gates["dimer_resonance_improves_phase_bank_energy_and_variance"]
    assert gates["physical_dimer_beats_every_scramble_energy_and_variance"]
    assert not gates[
        "physical_dimer_beats_dimension_matched_krylov_energy_and_variance"
    ]
    assert result["phase_bank_plus_dimer_resonance"]["rank"] == 9
    assert all(row["rank"] == 9 for row in result["scrambled_controls"])
