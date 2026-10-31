"""Independent noisy polynomial scaling law with a non-Ising exponent."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from fnqs.artifacts import source_hashes, write_json
from fnqs.scaling import collapse_fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous numerical witnesses")
    sizes = np.repeat([16, 24, 32, 48, 64], 41).astype(float)
    coordinate = np.tile(np.linspace(-1, 1, 41), 5)
    fields = 1.07 + coordinate / sizes ** (1 / .9)
    mean = .03 * sizes ** (2 / .9) * (1 - .25 * coordinate ** 2
        + .05 * coordinate ** 4 + (.4 - .1 * coordinate ** 2) / sizes)
    sigma = .01 * mean
    rows = []
    for seed in [17, 29, 43]:
        measured = mean + sigma * np.random.default_rng(seed).normal(size=len(sizes))
        fit = collapse_fit(sizes, fields, measured, degree=4, chi_se=sigma)
        assert abs(fit["nu"] - .9) < .005
        assert abs(fit["hc"] - 1.07) < .0003
        assert not fit["at_bound"]
        rows.append({"seed": seed, "fit": fit, "nu_abs_error": abs(fit["nu"] - .9),
                     "hc_abs_error": abs(fit["hc"] - 1.07)})
    write_json(args.output, {"kind": "free_exponent_noisy_scaling_numerical_validation",
        "source_sha256": source_hashes(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "true_nu": .9, "true_hc": 1.07, "relative_noise_sigma": .01,
        "nu_abs_tolerance": .005, "hc_abs_tolerance": .0003,
        "max_nu_abs_error": max(row["nu_abs_error"] for row in rows),
        "max_hc_abs_error": max(row["hc_abs_error"] for row in rows),
        "rows": rows, "passed": True, "neural_result": False})


if __name__ == "__main__":
    main()
