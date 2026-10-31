"""Actual physical state-span audit; no ED, optimized head, or N36 queries.

The saved N16 carrier and fixed paired kernels are a VARIATIONAL bank, not
four asserted topological sectors. Full transition RDMs, not only diagonal
RDMs, test the complete local matrix-unit algebra. Singlet one-site blindness
and an exactly soluble Z2 code are calibrated separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from su2_nqs.benchmarks.adaptive_projected_pair_carrier import batched_singlet_agp_amplitudes
from su2_nqs.benchmarks.clebsch_tree_exact import physical_permutation_index_table
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.paired_z2_carrier import paired_z2_kernel_bank
from su2_nqs.benchmarks.spin_rotation_calibration import (
    kagome_rvb_reference, lieb_mattis_singlet, manifold_region_algebra,
    singlet_covering_state,
)
from su2_nqs.benchmarks.triangular_induced_parton import induced_representation_audit
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import C6, MX, hnf_site_permutation
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds

BASE = Path("projects/clebsch_tree_nqs/results")


def whiten(columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the actual positive Gram; reject singularity without rank repair."""
    columns = np.asarray(columns, dtype=np.complex128)
    if columns.ndim != 2 or not np.isfinite(columns).all():
        raise ValueError("finite state columns required")
    gram = columns.conj().T @ columns
    values, vectors = np.linalg.eigh(gram)
    if values.min() <= 1e-12:
        raise ValueError("singular actual state span; no sector invented by whitening")
    return columns @ (vectors / np.sqrt(values)), gram, values


def restriction(columns: np.ndarray, transformed: np.ndarray) -> dict:
    """Physical U_g restriction and leakage; no guessed virtual matrix."""
    matrix = columns.conj().T @ transformed
    return dict(matrix_real=matrix.real.tolist(), matrix_imaginary=matrix.imag.tolist(),
                closure_residual_frobenius=float(np.linalg.norm(transformed-columns@matrix)),
                unitarity_error=float(np.linalg.norm(matrix.conj().T@matrix-np.eye(matrix.shape[0]))))


def local_report(columns: np.ndarray, regions: list[list[int]]) -> dict:
    """All a,b transition RDMs and local matrix units, not selected scalars."""
    answer = {}
    for region in regions:
        data = manifold_region_algebra(columns, region)
        answer[str(region)] = dict(defect_frobenius=data["local_algebra_defect_frobenius"],
            largest_transition_defect=data["maximum_transition_defect"],
            Gram_whitening_orthonormality_error=data["orthonormality_error"],
            all_offdiagonal_transitions=True)
    return answer


def toric_code_states(length: int = 3) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Exact 3x3 toric-code ground states, 18 edge qubits; NOT SU2 spins.

    Star-group orbits construct physical wavefunctions directly, not ED.
    Each star/plaquette has weight four; noncontractible X/Z Wilson loops
    are separate known logical operators, unlike microscopic spin parity.
    """
    if length != 3:
        raise ValueError("this bounded independent calibration uses L=3 only")
    n = 2*length**2
    def edge(r: int, c: int, direction: int) -> int:
        return 2*((r % length)*length+(c % length))+direction
    def mask(edges: list[int]) -> int:
        return sum(1 << (n-1-e) for e in edges)
    stars = [mask([edge(r,c,0),edge(r,c-1,0),edge(r,c,1),edge(r-1,c,1)])
             for r in range(length) for c in range(length)]
    plaquettes = [mask([edge(r,c,0),edge(r+1,c,0),edge(r,c,1),edge(r,c+1,1)])
                  for r in range(length) for c in range(length)]
    logical_x = [mask([edge(r,0,0) for r in range(length)]),
                 mask([edge(0,c,1) for c in range(length)])]
    logical_z = [mask([edge(0,c,0) for c in range(length)]),
                 mask([edge(r,0,1) for r in range(length)])]
    orbit = [0]
    for star in stars[:-1]:
        orbit += [packed ^ star for packed in orbit]
    assert len(set(orbit)) == 2**(length**2-1)
    states = np.zeros((2**n,4), dtype=np.complex128)
    for sector in range(4):
        seed = (logical_x[0] if sector & 1 else 0) ^ (logical_x[1] if sector & 2 else 0)
        states[np.array(orbit)^seed,sector] = 1/np.sqrt(len(orbit))
    return states, stars, plaquettes, logical_z


def main() -> None:
    start = time.monotonic()
    saved = BASE / "spin_qsl_symmetry_diagnostics_v1/states.npz"
    with np.load(saved) as data:
        bits = np.array(data["basis"])
        carrier = np.array(data["bare_pi_flux"], dtype=np.complex128)
    group = triangular_hnf_gamma_a1_group(((4,0),(0,4)))
    indices = physical_permutation_index_table(bits, group)
    kernels = paired_z2_kernel_bank("triangular_n16")
    paired = [np.mean(batched_singlet_agp_amplitudes(bits,k.pairing)[indices],axis=0)
              for k in kernels]
    columns = np.column_stack((carrier,*paired))
    columns /= np.linalg.norm(columns,axis=0)
    orthogonal, gram, values = whiten(columns)
    packed = bits @ (1 << np.arange(15,-1,-1))
    full = np.zeros((2**16,4), dtype=np.complex128)
    full[packed] = columns
    h, basis = heisenberg_sz_graph_sparse(16, triangular_j1j2_bonds(4,4,j2=.125,periodic=True))
    lookup = {row:i for i,row in enumerate(basis)}
    order = np.array([lookup[tuple(row)] for row in bits])
    h = h[order][:,order]
    hq = h @ orthogonal
    restricted_h = orthogonal.conj().T @ hq
    energies, rotations = np.linalg.eigh(restricted_h)
    ritz = orthogonal @ rotations
    residuals = np.linalg.norm(h@ritz-ritz*energies,axis=0)
    frozen = json.loads((BASE / "spin_flux_loop_interference_confirmation_v1/results.json").read_text())
    frozen_gram = np.array(frozen["suite"]["triangular_n16"]["carrier_overlap_matrix"])
    pair_local = local_report(full,[[0],[0,1],[0,5],[0,2],[0,1,4,5]])
    # All physical group operations were projected to A1: the resulting
    # restricted matrix is identity, hence cannot supply relative Qi-Fu signs.
    symmetry_error = max(float(np.linalg.norm(orthogonal[row]-orthogonal)) for row in indices)
    physical_operations = {
        "T1": hnf_site_permutation(((4,0),(0,4)),translation=(1,0)),
        "T2": hnf_site_permutation(((4,0),(0,4)),translation=(0,1)),
        "C6": hnf_site_permutation(((4,0),(0,4)),operation=C6),
        "mirror": hnf_site_permutation(((4,0),(0,4)),operation=MX),
    }
    spatial_matrices = {name: restriction(orthogonal,orthogonal[indices[list(group).index(image)]])
                        for name,image in physical_operations.items()}
    singlet_residuals = [spin_raising_relative_norm(column,bits) for column in columns.T]
    dimer = np.column_stack((singlet_covering_state(4,[(0,1),(2,3)]),
                             singlet_covering_state(4,[(0,2),(1,3)])))
    ordered = np.column_stack((lieb_mattis_singlet(8,[0,2,4,6]),
                               lieb_mattis_singlet(8,[0,1,2,3])))
    rvb = kagome_rvb_reference(2,2)
    toric, stars, plaquettes, logical_z = toric_code_states()
    packed_toric = np.arange(len(toric))
    stabilizer_error = max(float(np.linalg.norm(toric[packed_toric^s]-toric)) for s in stars)
    for p in plaquettes:
        signs = np.array([(-1.)**int((int(x)&p).bit_count()) for x in packed_toric])
        stabilizer_error = max(stabilizer_error,float(np.linalg.norm(signs[:,None]*toric-toric)))
    logical = []
    for z in logical_z:
        signs = np.array([(-1.)**int((int(x)&z).bit_count()) for x in packed_toric])
        logical.append(restriction(toric,signs[:,None]*toric))
    induced = induced_representation_audit()
    crossfit = json.loads((BASE / "spin_n36_relative_flux_crossfit_v1/results.json").read_text())["crossfit"]
    result = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        saved_state_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
        N16_variational_bank=dict(labels=["pi_zero_flux",*[k.label for k in kernels]],
            Gram_real=gram.real.tolist(), Gram_imaginary=gram.imag.tolist(),
            Gram_eigenvalues=values.tolist(), rank_at_absolute_threshold_1e_minus12=int(np.sum(values>1e-12)),
            absolute_Gram_replay_error=float(np.max(abs(abs(gram)-abs(frozen_gram)))),
            maximum_Splus_residual=max(singlet_residuals), all_192_A1_symmetry_closure_error=symmetry_error,
            actual_physical_space_group_restrictions=spatial_matrices,
            restricted_H_eigenvalues_total=energies.tolist(), true_Ritz_residual_norms=residuals.tolist(),
            complete_local_transition_algebra=pair_local,
            same_physical_H=True, common_boundary_family=False, asserted_topological_sectors=False),
        N36_induced_boundary_orbit=dict(projector=induced["projector"].tolist(),
            A1_projector_rank=induced["projector_rank"], A1_vector_error=induced["a1_vector_error"],
            interpretation="PP/AP/PA collapse to one A1 direction; AA open shell omitted. This is a Dirac boundary orbit, not four gapped Z2 ground sectors."),
        N36_sampled_relative_flux=dict(development_eigenvalues=crossfit["development_frozen_basis"]["eigenvalues_descending"],
            retained_modes=crossfit["development_frozen_basis"]["retained_mode_count"],
            audit_Gram_identity_Frobenius=crossfit["audit_frozen_mode_gram"]["identity_frobenius_deviation"],
            crossfit_nonkrylov_residual_fraction=crossfit["cross_fitted_dimer_response"]["audit_joint_residual_fraction"],
            physical_Gram_interval_certificate=False, support_completeness_certificate=False,
            topological_manifold=False),
        negative_controls=dict(dimer_S0_span=local_report(dimer,[[0],[0,1]]),
            magnetic_singlet_span=local_report(ordered,[[0],[0,1],[0,2]]),
            singlet_one_site_gate_is_kinematic=True,
            physical_2pi_center_is_region_parity=True),
        positive_family_finite_kagome_NNRVB=dict(n_sites=12,Gram_eigenvalues=np.linalg.eigvalsh(rvb["gram"]).tolist(),
            local_algebra=local_report(rvb["states"],[[0],[0,1],[0,3]]),
            known_Z2_family_from_literature=True, tiny_state_is_topology_certificate=False),
        exact_Z2_code_control=dict(n_edge_qubits=18,SU2_model=False,
            maximum_stabilizer_residual=stabilizer_error,
            local_algebra=local_report(toric,[[0],[0,1],[0,2],[0,1,2]]),
            noncontractible_loop_region_algebra=local_report(toric,[[0,2,4]]),
            physical_Wilson_Z_restrictions=logical,
            interpretation="Exactly soluble toric-code ground manifold calibrates local error detection only; it does not calibrate physical SU2 fractionalization."),
        unique_missing_physical_input="At least a second amplitude-accessible branch from one fixed paired kernel differing only in noncontractible holonomy, without forcing all branches to A1, together with its same-H cross matrices. Current optimized AGP endpoint is a single S0/A1 state, and different J2 endpoints use different H.",
        no_new_ED=True,no_new_N36_amplitude_queries=True,no_new_training=True,no_new_sampling=True,
        QSL_claim=False,fractionalization_claim=False,
        elapsed_seconds=time.monotonic()-start,peak_RSS_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
