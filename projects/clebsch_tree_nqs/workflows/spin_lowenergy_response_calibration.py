"""Bounded source-Krylov calibration using saved N16 states, no new ED state.

M is legal on this 4x4 torus; K is absent, so this cannot decide 120-degree
order. N36 query envelopes are measured on saved auxiliary proposal records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.ground_state_response_krylov import response_krylov
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse


def _query_envelope(bits: np.ndarray, bonds: tuple) -> dict:
    """Count genuinely reachable two-H configurations and Q1 density queries."""
    n = len(bits)
    powers = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
    word = int(bits @ powers)
    masks = [(int(powers[i]), int(powers[j])) for i, j, _ in bonds]

    def neighbors(x: int) -> set[int]:
        return {x, *(x ^ first ^ second for first, second in masks
                     if bool(x & first) != bool(x & second))}

    first = neighbors(word)
    second = set().union(*(neighbors(x) for x in first))
    qbank = {word, *(word ^ int(powers[i]) ^ int(powers[j]) for i in range(n)
                    for j in range(i + 1, n) if bits[i] != bits[j])}
    return dict(one_H_unique_queries=len(first), two_H_unique_queries=len(second),
                Q1_density_unique_queries=len(qbank),
                two_H_plus_Q1_unique_queries=len(second | qbank))


def run(output: Path) -> dict:
    start = perf_counter()
    states_file = Path('projects/clebsch_tree_nqs/results/spin_qsl_symmetry_diagnostics_v1/states.npz')
    with np.load(states_file) as archive:
        saved = {key: np.array(archive[key]) for key in archive.files}
    h, basis = heisenberg_sz_graph_sparse(16, triangular_j1j2_bonds(4, 4, j2=.125, periodic=True))
    positions = {tuple(row): k for k, row in enumerate(saved['basis'])}
    ordering = np.array([positions[tuple(row)] for row in basis])
    bits = np.array(basis)
    phase = (-1.)**(np.arange(16)//4)
    diagonal = (bits - .5) @ phase / 4
    actions = FixedSzPairActions(basis)

    def s2(vector):
        result = 12.*vector
        for i in range(16):
            for j in range(i + 1, 16):
                result = result + vector[actions.indices(i, j)] - .5*vector
        return result

    rows = {}
    for name in ('exact_ground', 'bare_pi_flux', 'pi_flux_fixed_sz_gamma_a1_spatial_vit',
                 'pi_flux_exact_su2_a1_operator_vit'):
        psi = saved[name][ordering].astype(complex)
        psi /= np.linalg.norm(psi)
        energy = float(np.vdot(psi, h @ psi).real)
        source = diagonal*psi
        source /= np.linalg.norm(source)
        # Shift reduces powers' cancellation but retains the exact H action.
        apply = lambda vector: h @ vector - energy*vector
        response = response_krylov(apply, source, max_dimension=32)
        slices = {}
        for dimension in (1, 2, 4, 8, 16, 32):
            q = response.basis[:, :dimension]
            hq = response.h_basis[:, :dimension]
            projected = q.conj().T @ hq
            eig, u = np.linalg.eigh((projected + projected.conj().T) / 2)
            states = q @ u
            residuals = np.linalg.norm(hq @ u - states*eig[None, :], axis=0)
            triplet_defect = np.linalg.norm(s2(states) - 2*states, axis=0)
            slices[str(dimension)] = dict(actual_dimension=len(eig),
                relative_ritz_energies=eig.tolist(), source_weights=abs(u[0])**2,
                residual_norms=residuals.tolist(), q2_casimir_action_residual=triplet_defect.tolist(),
                lowest_relative_ritz_energy=float(eig[0]), lowest_source_weight=float(abs(u[0, 0])**2),
                lowest_residual=float(residuals[0]), lowest_q2_defect=float(triplet_defect[0]))
            slices[str(dimension)]['source_weights'] = slices[str(dimension)]['source_weights'].tolist()
        rows[name] = dict(parent_energy_per_site=energy/16,
            parent_H_residual_norm=float(np.linalg.norm(h @ psi - energy*psi)),
            source_centroid=float(np.vdot(source, apply(source)).real),
            source_width_squared=float(np.linalg.norm(apply(source))**2 - np.vdot(source, apply(source)).real**2),
            source_q2_defect=float(np.linalg.norm(s2(source)-2*source)),
            orthonormality_error=response.orthonormality_error,
            projected_hermiticity_error=response.projected_hermiticity_error,
            dimensions=slices)
    moment_counterexamples = []
    for energy, weight in (([.15, 1.6875], [25/41, 16/41]), ([.5, 3.], [.9, .1])):
        e = np.array(energy); w = np.array(weight)
        moment_counterexamples.append(dict(poles=energy, weights=weight,
            moments=[float(w @ e**r) for r in range(4)],
            interpretation='Positive spin-source spectral measures with the same first two energy moments, different lowest poles.'))

    root = Path('projects/clebsch_tree_nqs/results/spin_response_moments_20261014')
    cost = {}
    inputs = {str(states_file): file_sha256(states_file)}
    for key, j2 in (('j0_s1000', 0.), ('j0125_s200', .125)):
        proposal = root/key/'proposal.npz'
        with np.load(proposal) as archive:
            configurations = archive['states'][np.linspace(0, len(archive['states'])-1, 8, dtype=int)]
        envelopes = [_query_envelope(x, triangular_j1j2_bonds(6, 6, j2=j2, periodic=True)) for x in configurations]
        record = json.loads((root/key/'results.json').read_text())
        throughput = record['projected_queries_this_run']/record['seconds_this_run']
        mean = float(np.mean([e['two_H_plus_Q1_unique_queries'] for e in envelopes]))
        cost[key] = dict(sampled_configuration_envelopes=envelopes,
            existing_run_projected_queries_per_second=throughput,
            N36_m3_queries_per_record_mean=mean,
            projected_query_only_seconds_for_4096_records=4096*mean/throughput,
            interpretation='Actual reachable-query count on eight saved configurations; extrapolated from previous one-swap throughput. New compilation, plan construction, numerical moment errors, larger memory, and higher-tail variance still require a bounded preflight.')
        inputs[str(proposal)] = file_sha256(proposal)
        inputs[str(root/key/'results.json')] = file_sha256(root/key/'results.json')
    result = dict(source_inputs=inputs, new_ground_state_ED=False, new_training=0, new_sampling=0,
        source_momentum='M=(1/2,0); triangular 4x4 does not contain K',
        physical_question='Can source Krylov distinguish a low pole from the first-moment center?',
        rows=rows, equal_low_moment_counterexamples=moment_counterexamples,
        N36_m3_query_cost=cost, elapsed_seconds=perf_counter()-start,
        claim_boundary='Saved-state exact-H source calibration, not N36 gaps, order classification, or fractionalization.')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('retain previous calibration artifacts')
    atomic_json(output, result)
    print(json.dumps(dict(output=str(output), elapsed_seconds=result['elapsed_seconds'],
        exact_ground={d:{k:v for k,v in row.items() if k.startswith('lowest_')}
                      for d,row in rows['exact_ground']['dimensions'].items()}, cost=cost), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
