"""Fresh-seed confirmation of the additive-carrier mechanism."""

from __future__ import annotations

from .four_cell_additive_carrier import additive_carrier_frontier


def additive_carrier_confirmation(
    *,
    seeds: tuple[int, ...] = (101, 102, 103),
    steps: int = 1000,
    triangular_numerical_floor: float = 1.0e-10,
) -> dict[str, object]:
    """Apply a development-informed gate to untouched initialization seeds.

    The triangular raw arm already reached machine precision in development,
    so confirmation uses numerical noninferiority rather than demanding a
    mathematically impossible strict improvement over zero error.
    """

    suite = additive_carrier_frontier(seeds=seeds, steps=steps)
    triangular = suite["cases"]["triangular_n12_j2_0125"]["summary"]
    square = suite["cases"]["square_n16_j2_05"]["summary"]
    tri_physical = triangular["physical_additive_carrier"]
    square_physical = square["physical_additive_carrier"]
    square_raw = square["no_carrier"]
    square_scrambled = square["scrambled_additive_carrier"]
    gates = {
        "triangular_physical_numerically_noninferior": tri_physical[
            "maximum_energy_error_total"
        ]
        < float(triangular_numerical_floor),
        "square_physical_all_seeds_below_1e-5": square_physical[
            "maximum_energy_error_total"
        ]
        < 1.0e-5,
        "square_physical_improves_median_over_no_carrier": square_physical[
            "median_energy_error_total"
        ]
        < square_raw["median_energy_error_total"],
        "square_physical_improves_worst_seed_over_no_carrier": square_physical[
            "maximum_energy_error_total"
        ]
        < square_raw["maximum_energy_error_total"],
        "square_physical_beats_scrambled_median": square_physical[
            "median_energy_error_total"
        ]
        < square_scrambled["median_energy_error_total"],
    }
    return {
        "development_informed": True,
        "fresh_seeds": list(seeds),
        "triangular_numerical_floor": float(triangular_numerical_floor),
        "suite": suite,
        "confirmation_gates": gates,
        "confirmation_passed": bool(all(gates.values())),
        "claim_boundary": (
            "Fresh initialization seeds confirm or reject local carrier-basin "
            "robustness. Passing licenses construction of a sampled N36 "
            "canary only; the carrier bank still lacks a paired-Z2 member."
        ),
    }


__all__ = ["additive_carrier_confirmation"]
