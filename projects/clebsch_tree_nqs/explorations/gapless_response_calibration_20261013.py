"""Read-only physical-response calibration from saved N16 states; no ED.

Print a JSON artifact.  Spin labels use q=2S.  The physical scalar dimer
sources have three NN orientations, and M=(1/2,0) is legal on this 4x4 torus.
There is deliberately no claim about K/X, which are absent on this torus.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse


def main() -> None:
    start = time.monotonic()
    path = Path("projects/clebsch_tree_nqs/results/spin_qsl_symmetry_diagnostics_v1/states.npz")
    with np.load(path) as data:
        saved = {key: np.array(data[key]) for key in data.files}
    hamiltonian, basis = heisenberg_sz_graph_sparse(
        16, triangular_j1j2_bonds(4, 4, j2=0.125, periodic=True)
    )
    positions = {tuple(row): index for index, row in enumerate(saved["basis"])}
    ordering = np.array([positions[tuple(row)] for row in basis])
    bits = np.array(basis, dtype=np.int8)
    actions = FixedSzPairActions(basis)
    coordinates = np.array([divmod(site, 4) for site in range(16)])
    phase = np.exp(2j * np.pi * coordinates[:, 0] * 0.5)
    spin_diagonal = (bits.astype(float) - 0.5) @ phase / 4.0
    displacements = ((1, 0), (0, 1), (1, 1))

    def dimer(vector: np.ndarray) -> np.ndarray:
        columns = []
        for dr, dc in displacements:
            image = np.zeros_like(vector, dtype=np.complex128)
            for site, (row, column) in enumerate(coordinates):
                other = ((row + dr) % 4) * 4 + (column + dc) % 4
                image += phase[site] * actions.pair(vector, site, other) / 4.0
            columns.append(image)
        return np.column_stack(columns)

    def s2(vector: np.ndarray) -> np.ndarray:
        image = 12.0 * vector
        for first in range(16):
            for second in range(first + 1, 16):
                image = image + 2.0 * actions.pair(vector, first, second)
        return image

    translation_indices = []
    for dr, dc in ((1, 0), (0, 1)):
        mapping = np.array([((r + dr) % 4) * 4 + (c + dc) % 4 for r, c in coordinates])
        translated = np.empty_like(bits)
        translated[:, mapping] = bits
        lookup = {tuple(row): index for index, row in enumerate(basis)}
        translation_indices.append(np.array([lookup[tuple(row)] for row in translated]))

    rows = {}
    for name in ("exact_ground", "bare_pi_flux", "pi_flux_fixed_sz_gamma_a1_spatial_vit", "pi_flux_exact_su2_a1_operator_vit"):
        psi = np.asarray(saved[name][ordering], dtype=np.complex128)
        psi = psi / np.linalg.norm(psi)
        hpsi = hamiltonian @ psi
        energy = float(np.vdot(psi, hpsi).real)
        residual = hpsi - energy * psi
        images = dimer(psi)
        # At nonzero M a Gamma parent is orthogonal to these physical sources;
        # retain this measured overlap rather than silently reprojecting them.
        gram = images.conj().T @ images
        himages = hamiltonian @ images
        first = images.conj().T @ himages
        second = himages.conj().T @ himages
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        if eigenvalues[0] <= 1e-12:
            raise RuntimeError("response Gram is singular; no silent rank deletion")
        whitening = eigenvectors / np.sqrt(eigenvalues)[None, :]
        ritz_energy, ritz_coordinates = np.linalg.eigh(whitening.conj().T @ first @ whitening)
        coefficient = whitening @ ritz_coordinates[:, 0]
        response = images @ coefficient
        hresponse = himages @ coefficient
        ritz_variance = float(np.linalg.norm(hresponse - ritz_energy[0] * response) ** 2)
        scalar_q0_residual = float(np.linalg.norm(s2(response)))
        diagonal_rayleigh = np.diag(first).real / np.diag(gram).real
        # A physical scalar source preserves S, but it need not remove an
        # approximate parent's non-singlet component.  No singlet projection.
        if name == "exact_ground":
            assert scalar_q0_residual < 1e-10
        spin_image = spin_diagonal * psi
        spin_norm = float(np.vdot(spin_image, spin_image).real)
        spin_energy = float(np.vdot(spin_image, hamiltonian @ spin_image).real / spin_norm)
        # O=S^z_M is Hermitian.  Compare the true response energy with the
        # double commutator and the approximate-parent covariance defect.
        apsi = np.abs(spin_diagonal) ** 2 * psi
        covariance = complex(np.vdot(apsi, residual))
        variance_a = max(float(np.vdot(apsi, apsi).real - spin_norm**2), 0.0)
        f = float(np.vdot(spin_image, hamiltonian @ spin_image).real - np.vdot(apsi, hpsi).real)
        translation_residual = max(float(np.linalg.norm(response[index] - sign * response)) for index, sign in zip(translation_indices, (-1.0, 1.0)))
        identity_error = abs(spin_energy - energy - (f + covariance.real) / spin_norm)
        assert identity_error < 1e-10
        assert abs(covariance.real) <= np.sqrt(variance_a) * np.linalg.norm(residual) + 1e-12
        assert translation_residual < 1e-10
        rows[name] = {
            "parent_energy_total": energy,
            "parent_residual_norm": float(np.linalg.norm(residual)),
            "parent_s2_action_norm": float(np.linalg.norm(s2(psi))),
            "dimer_gram_real": gram.real.tolist(),
            "dimer_gram_max_imaginary": float(np.max(np.abs(gram.imag))),
            "dimer_gram_eigenvalues": eigenvalues.tolist(),
            "dimer_energy_matrix_real": first.real.tolist(),
            "dimer_second_energy_moment_real": second.real.tolist(),
            "dimer_ritz_energy_total": ritz_energy.tolist(),
            "dimer_ritz_minus_parent_energy": (ritz_energy - energy).tolist(),
            "dimer_single_orientation_minus_parent_energy": (diagonal_rayleigh - energy).tolist(),
            "dimer_lowest_ritz_coefficients_real": coefficient.real.tolist(),
            "dimer_lowest_ritz_coefficients_imaginary": coefficient.imag.tolist(),
            "dimer_response_normalization_error": abs(float(np.vdot(response, response).real) - 1.0),
            "dimer_energy_matrix_hermiticity_error": float(np.max(np.abs(first - first.conj().T))),
            "dimer_lowest_ritz_residual_norm": float(np.sqrt(max(ritz_variance, 0.0))),
            "dimer_lowest_q0_s2_action_norm": scalar_q0_residual,
            "dimer_momentum_character_residual": translation_residual,
            "dimer_max_parent_overlap": float(np.max(np.abs(psi.conj() @ images))),
            "spin_M_z_norm_squared": spin_norm,
            "spin_M_z_rayleigh_energy_total": spin_energy,
            "spin_M_z_minus_parent_energy": spin_energy - energy,
            "spin_M_z_q2_relative_s2_action_error": float(np.linalg.norm(s2(spin_image) - 2 * spin_image) / np.sqrt(spin_norm)),
            "spin_static_f_over_norm": f / spin_norm,
            "spin_response_covariance_defect_over_norm": float(covariance.real / spin_norm),
            "spin_covariance_CS_bound_over_norm": float(np.sqrt(variance_a) * np.linalg.norm(residual) / spin_norm),
            "spin_energy_identity_error": identity_error,
        }
    bonds36 = triangular_j1j2_bonds(6, 6, j2=0.125, periodic=True)
    dimer36 = triangular_j1j2_bonds(6, 6, j2=0.0, periodic=True)
    masks_h = [(1 << first) ^ (1 << second) for first, second, _ in bonds36]
    masks_d = [(1 << first) ^ (1 << second) for first, second, _ in dimer36]
    masks_hd = {0, *masks_h, *masks_d, *(first ^ second for first in masks_h for second in masks_d)}
    artifact = {
        "source_file": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "N": 16, "geometry": "triangular_4x4_periodic_120degree", "j2": 0.125,
        "momentum_fractional": [0.5, 0.0], "momentum_name": "M",
        "source_normalization": "1/sqrt(N); site-anchored three NN directions",
        "repeated_ED": False, "new_training": False, "new_sampling": False,
        "K_or_X_physics_calibrated": False, "thermodynamic_phase_claim": False,
        "rows": rows,
        "N36_direct_response_query_envelope": {
            "H_bonds": len(bonds36), "D_bonds": len(dimer36),
            "naive_H_D_exchange_paths_per_record": len(bonds36) * len(dimer36),
            "distinct_structural_toggle_masks_including_parent": len(masks_hd),
            "interpretation": "structural upper envelope only; antiparallel-spin gates reduce reachable states, cache reuse and model evaluation wall time not measured",
        },
        "elapsed_seconds": time.monotonic() - start,
        "max_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
