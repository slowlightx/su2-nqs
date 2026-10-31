"""Bounded exact-small test of conditional resonance *inside* a strong carrier.

This is a local-operator mechanism experiment, not a full Spatial-ViT or a
production energy comparison. Saved physical carriers and Hamiltonians are
reused. No ED eigensystem is fitted or recomputed. A single local filter still
lies in a finite operator-generated span; that ceiling is reported explicitly.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.krylov_anchored_full_vit import a1_orbit_reduction, a1_orbit_hamiltonian
from su2_nqs.benchmarks.singlet_vap import singlet_isometry
from su2_nqs.benchmarks.conditional_carrier_resonance import (
    compile_conditional_carrier_queries, contract_conditional_carrier_queries,
    conditional_carrier_amplitudes,
)
from su2_nqs.benchmarks.triangular_induced_parton import build_n36_pi_zero_flux_orbit
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig, projected_singlet_agp_amplitudes
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.conditional_carrier_resonance import (
    conditional_filter_coefficients, factorized_to_conditional,
    diagonal_conditioned_to_conditional, packed_scalar_rows,
)
from .spin import _prepare_case, _groups

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_conditional_carrier_resonance.json'
ARMS = ('factorized', 'diagonal_conditioned', 'conditional', 'site_scrambled')
PARAMETERS = dict(factorized=16, diagonal_conditioned=46, conditional=60, site_scrambled=60)


def _npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = path.with_suffix('.npz.tmp')
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def local_indices(configurations: np.ndarray, support: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Map the fixed-M support to local rows and all allowed input rows.

    Site zero is the most significant bit. No nonzero local contribution is
    discarded, including inputs on which the carrier amplitude is zero.
    """
    bits = np.asarray(configurations, dtype=np.int64)
    n = bits.shape[1]
    if len(support) != 8 or len(set(support)) != 8 or min(support) < 0 or max(support) >= n:
        raise ValueError('eight distinct in-range sites required')
    table = packed_scalar_rows()
    local = bits[:, support] @ (1 << np.arange(7, -1, -1))
    integers = bits @ (1 << np.arange(n-1, -1, -1))
    columns = table.columns[local]
    inputs = np.broadcast_to(integers[:, None], columns.shape).copy()
    for k, site in enumerate(support):
        inputs += (((columns >> (7-k)) & 1) - bits[:, site, None]) * (1 << (n-1-site))
    lookup = np.full(1 << n, -1, dtype=np.int32)
    lookup[integers] = np.arange(len(bits))
    indices = lookup[inputs]
    if np.any((indices < 0) & table.mask[local]):
        raise ValueError('physical support is not complete under the local action')
    return local, np.maximum(indices, 0)


def carrier_columns(phi: np.ndarray, carrier: np.ndarray, configurations: np.ndarray,
                    support: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    """Contract the complete local scalar bank on C, then project once to A1.

    Phi is an independently sealed complete S=0/A1 basis. It is used solely
    as an exact-small output projector, never inserted between local factors.
    """
    tick = perf_counter()
    local, indices = local_indices(configurations, support)
    table = packed_scalar_rows()
    source = (phi @ carrier)[indices]
    physical = np.stack([np.sum(weights[local] * source, axis=1)
                         for weights in table.coefficients], axis=1)
    columns = phi.conj().T @ physical
    return columns, dict(seconds=perf_counter()-tick, local_support=list(support),
                         column_count=columns.shape[1], rank=int(np.linalg.matrix_rank(columns, tol=1e-10)),
                         maximum_connections=int(table.mask.sum(axis=1).max()))


def conditional_coherence(physical: np.ndarray, configurations: np.ndarray,
                          support: tuple[int, ...]) -> dict:
    """Measure actual environment-resolved two-dimer singlet coherence.

    X=|ss><(tt)0|+h.c. in the active Q=0 block. The local Schur table uses
    A1/B1; rotate X back explicitly. These are signed expectation values,
    not probabilities or neural attention weights.
    """
    local, indices = local_indices(configurations, support)
    table = packed_scalar_rows()
    v = physical / np.linalg.norm(physical)
    expectation = np.asarray([np.vdot(v, np.sum(weights[local]*v[indices], axis=1)).real
                              for weights in table.coefficients]).reshape(6, 10)
    probability = expectation[:, [0, 1, 3, 4, 5, 9]].sum(axis=1)
    coherence = np.sqrt(3)/2*(expectation[:, 0]-expectation[:, 1])-expectation[:, 2]/np.sqrt(2)
    return dict(spectator_probabilities=probability.tolist(),
                joint_singlet_triplet_coherence=coherence.tolist(),
                conditional_coherence=np.divide(coherence, probability, out=np.zeros(6), where=probability>1e-12).tolist())


def load_case(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load old amplitudes; optionally reconstruct the S2-polynomial isometry.

    The square builder uses space-group orbits and a polynomial in S2, not
    Hamiltonian eigenvectors. H itself is only assembled as a sparse operator.
    """
    with np.load(ROOT/case['carrier_bundle'], allow_pickle=False) as bundle:
        physical = bundle[case['carrier_key']]
        bits = bundle[case['basis_key']].astype(np.int64)
    if case.get('sector_bundle'):
        with np.load(ROOT/case['sector_bundle'], allow_pickle=False) as bundle:
            phi, h = bundle[case['phi_key']], bundle[case['h_key']]
        # These historical artifacts use OPPOSITE fixed-M row orderings.
        # Global spin-flip symmetry can hide an accidental reversal; resolve
        # bit tuples explicitly, never infer ordering from overlap alone.
        canonical = fixed_m_spin_basis(16, 0)
        lookup = {tuple(x): i for i, x in enumerate(canonical)}
        permutation = np.asarray([lookup[tuple(x)] for x in bits])
        phi = phi[permutation]
    else:
        data = _prepare_case(case['operator_case'], 'fixed_m0')
        if not np.array_equal(bits, np.asarray(data['basis'])):
            raise RuntimeError('saved square basis differs from sparse reference')
        orbit = a1_orbit_reduction(data['tables']['gamma_a1'])
        coo = data['spin_squared'].tocoo()
        s2 = a1_orbit_hamiltonian(coo.row, coo.col, coo.data, orbit)
        reduced = singlet_isometry(s2, n_sites=16)
        phi = reduced[orbit.orbit_id] / np.sqrt(orbit.orbit_sizes[orbit.orbit_id, None])
        h = phi.conj().T @ (data['hamiltonian'] @ phi)
        if np.max(abs(data['hamiltonian'] @ phi - phi @ h)) > 1e-10:
            raise RuntimeError('square H/isometry replay failed')
    return phi, h, physical, bits


def generator(theta: jax.Array, arm: str) -> jax.Array:
    if arm == 'factorized':
        return factorized_to_conditional(theta)
    if arm == 'diagonal_conditioned':
        return diagonal_conditioned_to_conditional(theta)
    return theta.reshape(6, 10)


def state_function(carrier: np.ndarray, columns: np.ndarray, arm: str):
    c, bank = jnp.asarray(carrier), jnp.asarray(columns)
    def state(theta):
        # The primitive returns exp(G)-I, so C is exactly retained at G=0.
        v = c + bank @ conditional_filter_coefficients(generator(theta, arm))
        return v / jnp.linalg.norm(v)
    return state


def metrics(state: np.ndarray, h: np.ndarray, carrier: np.ndarray) -> dict:
    v = state / np.linalg.norm(state)
    hv = h @ v
    energy = float(np.vdot(v, hv).real)
    return dict(energy_per_site=energy/16,
                variance_per_site=float(np.linalg.norm(hv-energy*v)**2)/16,
                distance_from_carrier=float(np.arccos(np.clip(abs(np.vdot(carrier, v)), 0, 1))))


def _rank_and_capture(tangents: np.ndarray, reference: np.ndarray) -> dict:
    # All generator parameters are REAL, even for a complex saved carrier.
    tangents = np.vstack((tangents.real, tangents.imag))
    reference = np.vstack((reference.real, reference.imag))
    u, s, _ = np.linalg.svd(reference, full_matrices=False)
    keep = s > max(s[0]*1e-9, 1e-12) if len(s) else np.zeros(0, bool)
    remainder = tangents - u[:, keep] @ (u[:, keep].conj().T @ tangents)
    return dict(reference_rank=int(keep.sum()),
                conditional_rank=int(np.linalg.matrix_rank(tangents, tol=1e-9)),
                outside_reference_frobenius_fraction=float(np.linalg.norm(remainder)**2 / max(np.linalg.norm(tangents)**2, 1e-300)))


def profile_n36(spec: dict) -> dict:
    """Fixed-query amplitude/cached-gradient cost, not an energy experiment."""
    group = _groups('triangular', 6, 6)['gamma_a1']
    x = np.zeros(36, dtype=np.int32)
    x[np.random.default_rng(spec['seed']).choice(36, 18, replace=False)] = 1
    states = np.stack((x, x[np.asarray(group[spec['image_index']])]))
    t = perf_counter()
    plan = compile_conditional_carrier_queries(states, group,
        active_sites=(0, 1, 6, 7), spectator_sites=(2, 3, 8, 9))
    planning = perf_counter()-t
    if len(plan.configurations) > spec['maximum_unique_queries']:
        return dict(status='query_budget_exceeded_before_evaluation', planning_seconds=planning,
                    diagnostics=plan.diagnostics, full_H_calls=0)
    orbit = build_n36_pi_zero_flux_orbit()
    cfg = ProjectedSingletAGPConfig(36, (tuple(range(36)),))
    pairings = [jnp.asarray(p) for p in orbit.pairings]
    def carrier(bits):
        columns = jnp.stack([projected_singlet_agp_amplitudes({'pairing_real': p}, bits, cfg).amplitudes
                             for p in pairings], axis=1)
        return columns @ jnp.asarray([1., -1., -1.])/3
    carrier = jax.jit(carrier)
    t = perf_counter()
    carrier(jnp.asarray(plan.configurations[:256])).block_until_ready()
    compile_seconds = perf_counter()-t
    t = perf_counter()
    values = np.concatenate([np.asarray(carrier(jnp.asarray(plan.configurations[i:i+256])))
                             for i in range(0, len(plan.configurations), 256)])
    carrier_seconds = perf_counter()-t
    t = perf_counter()
    batch = contract_conditional_carrier_queries(plan, values)
    contraction = perf_counter()-t
    c = jnp.asarray(batch.carrier_amplitudes)
    responses = jnp.asarray(batch.responses)
    theta = jnp.asarray(.02*np.sin(np.arange(60)))
    evaluator = jax.jit(lambda p: conditional_carrier_amplitudes(p, c, responses))
    gradient = jax.jit(jax.jacfwd(evaluator))
    t = perf_counter()
    psi = np.asarray(evaluator(theta)); jac = np.asarray(gradient(theta))
    cached_compile = perf_counter()-t
    t = perf_counter()
    gradient(theta).block_until_ready()
    cached_gradient = perf_counter()-t
    zero_error = float(np.max(abs(np.asarray(evaluator(jnp.zeros(60)))-np.asarray(c))))
    a1_error = float(abs(psi[0]-psi[1]) / max(np.max(abs(psi)), 1e-300))
    if zero_error != 0 or a1_error > 1e-9 or not np.isfinite(jac).all():
        raise RuntimeError('N36 fixed-query identity/A1/gradient check failed')
    return dict(status='fixed_query_profile_only', configurations=states.tolist(), diagnostics=plan.diagnostics,
                planning_seconds=planning, carrier_compile_seconds=compile_seconds,
                warm_carrier_seconds=carrier_seconds, response_contraction_seconds=contraction,
                cached_gradient_compile_seconds=cached_compile, cached_gradient_seconds=cached_gradient,
                identity_absolute_error=zero_error, a1_relative_error=a1_error,
                relative_gradient_norm=float(np.linalg.norm(jac)/np.linalg.norm(psi)),
                carrier_determinants=3*len(plan.configurations), full_H_calls=0,
                caveat='one configuration plus a symmetry image; not representative throughput, training, H-ratio, or energy. Cached gradient excludes new carrier queries.')


def run(config_path: Path, output: Path) -> dict:
    spec = json.loads(config_path.read_text())
    if spec['task'] != 'spin_conditional_carrier_resonance' or spec['remote_allowed']:
        raise ValueError('only the bounded local mechanism experiment is allowed')
    if tuple(spec['arms']) != ARMS or spec['iterations'] > 120 or len(spec['starts']) != 3:
        raise ValueError('registered arm/endpoint budget changed')
    if output.exists():
        raise FileExistsError(output)
    paths = [ROOT/item for case in spec['cases'] for item in case['inputs']]
    for case in spec['cases']:
        for path, expected in case['inputs'].items():
            if file_sha256(ROOT/path) != expected:
                raise ValueError(f'changed source artifact: {path}')
    output.mkdir(parents=True)
    reg = dict(config=spec, config_sha256=file_sha256(config_path),
               inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in paths},
               runtime=numerical_runtime_identity(),
               sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
                    package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    reg['sha256'] = canonical_sha256(reg)
    atomic_json(output/'registration.json', reg)
    start = perf_counter()
    results = dict(registration_sha256=reg['sha256'], cases={}, new_ed_diagonalizations=0,
                   remote_jobs=0, claim='exact-small local filter mechanism; not full ViT or scalable variational win')
    saved = {}
    for case in spec['cases']:
        phi, h, physical, bits = load_case(case)
        carrier = phi.conj().T @ physical
        retained_weight = float(np.vdot(carrier, carrier).real / np.vdot(physical, physical).real)
        carrier /= np.linalg.norm(carrier)
        if phi.shape[0] != len(bits) or np.max(abs(phi.conj().T @ phi-np.eye(phi.shape[1]))) > 1e-10:
            raise ValueError('invalid sector isometry')
        supports = dict(physical=tuple(case['support']), scrambled=tuple(case['scrambled_support']))
        banks, costs = {}, {}
        for label, support in supports.items():
            banks[label], costs[label] = carrier_columns(phi, carrier, bits, support)
        row = dict(carrier=metrics(carrier, h, carrier), sector_dimension=phi.shape[1],
                   construction=costs, arms={}, exact_reference_energy_per_site=case['reference_energy_per_site'],
                   carrier_projection_retained_weight=retained_weight, carrier_role=case['carrier_role'])
        row['carrier_coherence'] = conditional_coherence(phi@carrier, bits, supports['physical'])
        tangents = {}
        for arm in ARMS:
            bank = banks['scrambled' if arm == 'site_scrambled' else 'physical']
            state = state_function(carrier, bank, arm)
            h_jax = jnp.asarray(h)
            def loss(theta):
                v = state(theta)
                return jnp.vdot(v, h_jax @ v).real / 16
            value_grad = jax.jit(jax.value_and_grad(loss))
            zero = np.zeros(PARAMETERS[arm])
            if np.max(abs(np.asarray(state(zero))-carrier)) > 2e-12:
                raise RuntimeError('identity does not replay carrier')
            jac = np.asarray(jax.jacfwd(state)(zero))
            tangents[arm] = jac-carrier[:, None]*(carrier.conj() @ jac)[None, :]
            row['arms'][arm] = []
            for init in spec['starts']:
                if perf_counter()-start > spec['wall_seconds_cap']:
                    raise TimeoutError('bounded experiment wall budget exhausted; partial checkpoints preserved')
                theta0 = (np.zeros(PARAMETERS[arm]) if init['scale'] == 0 else
                          np.random.default_rng(init['seed']).normal(0, init['scale'], PARAMETERS[arm]))
                history, calls = [], [0]
                def objective(theta):
                    if perf_counter()-start > spec['wall_seconds_cap']:
                        raise TimeoutError('bounded experiment wall budget exhausted')
                    v, g = value_grad(jnp.asarray(theta))
                    calls[0] += 1
                    if not np.isfinite(v) or not np.all(np.isfinite(g)):
                        raise FloatingPointError('nonfinite objective or gradient')
                    return float(v), np.asarray(g, dtype=float)
                def callback(theta):
                    # Fixed endpoint only; no best-checkpoint selection.
                    v = np.asarray(state(theta))
                    history.append(metrics(v, h, carrier))
                tick = perf_counter()
                opt = minimize(objective, theta0, jac=True, method='L-BFGS-B',
                               bounds=[tuple(spec['parameter_bounds'][arm])]*len(theta0), callback=callback,
                               options=dict(maxiter=spec['iterations'], maxfun=spec['maximum_function_calls'],
                                            ftol=1e-14, gtol=1e-9, maxls=30))
                elapsed = perf_counter()-tick
                v = np.asarray(state(opt.x))
                record = dict(metrics(v, h, carrier), start=init, parameters=len(opt.x),
                              iterations=int(opt.nit), objective_calls=calls[0], success=bool(opt.success),
                              message=str(opt.message), gradient_max=float(np.max(abs(opt.jac))),
                              seconds=elapsed, spin_raising_residual=spin_raising_relative_norm(phi@v, tuple(map(tuple, bits))),
                              coherence=conditional_coherence(phi@v, bits, supports['physical']),
                              history=history)
                row['arms'][arm].append(record)
                label = f"{case['name']}_{arm}_{init['seed']}"
                saved[label+'_parameters'] = opt.x
                saved[label+'_state'] = v
                saved[label+'_generator'] = np.asarray(generator(opt.x, arm))
                _npz(output/'states.npz', **saved)
                atomic_json(output/'progress.json', dict(case=case['name'], arm=arm, record=record))
                print(case['name'], arm, init['seed'], record['energy_per_site'], record['variance_per_site'], opt.message, flush=True)
        row['identity_tangent'] = {arm: _rank_and_capture(tangents['conditional'], tangents[arm])
                                   for arm in ('factorized', 'diagonal_conditioned')}
        # Variational span ceiling, not a new ED reference or a trained network.
        u, s, _ = np.linalg.svd(np.column_stack((carrier, banks['physical'])), full_matrices=False)
        u = u[:, s > max(s[0]*1e-10, 1e-12)]
        if u.shape[1] == phi.shape[1]:
            ceiling = dict(energy_per_site=case['reference_energy_per_site'], variance_per_site=0.,
                           reference_reused_without_diagonalization=True)
        else:
            _, vecs = np.linalg.eigh(u.conj().T @ h @ u)
            ceiling = metrics(u@vecs[:, 0], h, carrier)
        row['single_filter_linear_span_ceiling'] = dict(ceiling,
            rank=u.shape[1], saturates_sector=bool(u.shape[1] == phi.shape[1]),
            role='complex-linear outer-envelope ceiling; NOT the real nonlinear model optimum, a learned ansatz or new ED reference')
        results['cases'][case['name']] = row
        atomic_json(output/'partial_results.json', results)
    results['n36_cost_profile'] = profile_n36(spec['n36_cost_profile'])
    results['seconds'] = perf_counter()-start
    results['states_sha256'] = file_sha256(output/'states.npz')
    results['result_sha256'] = canonical_sha256(results)
    atomic_json(output/'results.json', results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, default=ROOT/'projects/clebsch_tree_nqs/results/spin_conditional_carrier_resonance_v1')
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
