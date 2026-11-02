"""Physical-spin Spatial-ViT adapters and coherent readouts, without a carrier.

Fixed doubled magnetization is not SU(2). No isotropy assumption, RDM twirl,
positivity clipping, or entanglement-spectrum interpretation is made here.
The archived model uses a 60-degree coordinate metric; a column reflection
maps the repository's physical 120-degree triangular coordinates to it.
This explicit relabelling leaves every archived model/source untouched.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np

from ..models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig, fixed_sz_spatial_vit_log_amplitude,
)
from ..vmc.regional_density import RegionalQueryPlan
from .spatial_vit_sampling import projected_log_function


@dataclass(frozen=True)
class ScalarStructureQueryPlan:
    """Offline, deduplicated physical exchange queries for scalar S(k).

    Same-spin pairs have no transverse matrix element and request no swapped
    amplitude. Centers are included once. Bits are down/up, site zero MSB.
    """
    configurations: np.ndarray
    center_indices: np.ndarray
    pair_indices: np.ndarray
    opposite_spins: np.ndarray
    states: np.ndarray
    pairs: np.ndarray


def scalar_structure_query_plan(states: np.ndarray) -> ScalarStructureQueryPlan:
    """Coalesce ALL physical pair exchanges without changing the observable."""
    bits = np.asarray(states)
    if (bits.ndim != 2 or not len(bits) or bits.shape[1] < 2
            or bits.dtype.kind not in 'biu' or not np.isin(bits, (0, 1)).all()):
        raise ValueError('nonempty physical down/up bit matrix required')
    pairs = np.array(list(combinations(range(bits.shape[1]), 2)), dtype=np.int32)
    opposite = bits[:, pairs[:, 0]] != bits[:, pairs[:, 1]]
    records, edges = np.nonzero(opposite)
    requests = bits[records].copy()
    first, second = pairs[edges].T
    requests[np.arange(len(requests)), first] = bits[records, second]
    requests[np.arange(len(requests)), second] = bits[records, first]
    unique, inverse = np.unique(np.concatenate((bits, requests)), axis=0,
                                 return_inverse=True)
    indices = np.zeros(opposite.shape, np.int64)
    indices[opposite] = inverse[len(bits):]
    arrays = (unique, inverse[:len(bits)], indices, opposite, bits.copy(), pairs)
    for array in arrays:
        array.setflags(write=False)
    return ScalarStructureQueryPlan(*arrays)


def scalar_structure_records(
    plan: ScalarStructureQueryPlan, log_amplitudes: np.ndarray,
    coordinates: np.ndarray, momenta: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Full scalar and SzSz Fourier records from cached coherent log ratios.

    S(k)=sum_ij exp[2pi i k.(r_i-r_j)] <S_i.S_j>/N, including the
    3/4 self term. S(Gamma)*N=<S_total²>. No isotropy or three-times-SzSz
    substitution. Nodes on connected inputs are legal; sampled centers must
    be nonzero. This function performs no amplitude/H/sampling queries.
    """
    logs = np.asarray(log_amplitudes, np.complex128)
    coords = np.asarray(coordinates, float)
    ks = np.asarray(list(momenta.values()), float)
    n = plan.states.shape[1]
    if (logs.shape != (len(plan.configurations),) or np.isnan(logs).any()
            or np.isposinf(logs.real).any() or not np.isfinite(logs.imag).all()
            or not np.isfinite(logs[plan.center_indices].real).all()
            or coords.shape != (n, 2) or ks.ndim != 2 or ks.shape[1] != 2
            or not np.isfinite(coords).all() or not np.isfinite(ks).all()):
        raise ValueError('valid physical coordinates/momenta and coherent logs required')
    pairs, bits = plan.pairs, plan.states
    dots = ((bits[:, pairs[:, 0]]-.5)*(bits[:, pairs[:, 1]]-.5)).astype(complex)
    center = np.broadcast_to(logs[plan.center_indices, None], dots.shape)
    differences = logs[plan.pair_indices[plan.opposite_spins]]-center[plan.opposite_spins]
    with np.errstate(over='raise', invalid='raise'):
        dots[plan.opposite_spins] += .5*np.exp(differences)
    weights = np.cos(2*np.pi*(coords[pairs[:, 0]]-coords[pairs[:, 1]])@ks.T)
    scalar = .75+(2./n)*(dots@weights)
    phases = np.exp(2j*np.pi*coords@ks.T)
    zz = abs((bits-.5)@phases)**2/n
    if not np.isfinite(scalar).all():
        raise FloatingPointError('nonfinite scalar ratio; no clipping or repair')
    return scalar, zz


def physical_to_model_permutation(config: FixedSzSpatialViTConfig) -> np.ndarray:
    """Map physical 120-degree triangular bits to the model's 60-degree grid.

    The returned site gather is involutive. Square coordinates are unchanged.
    Physical Hamiltonians, momentum labels and regional site order NEVER
    undergo this relabelling; only the input to the pure neural model does.
    """
    grid = np.arange(config.n_sites).reshape(config.rows, config.columns)
    if config.geometry == 'triangular':
        grid = grid[:, (-np.arange(config.columns)) % config.columns]
    return grid.ravel().astype(np.int32)


def make_physical_spatial_vit_log(
    config: FixedSzSpatialViTConfig, group: Sequence[Sequence[int]] | None = None,
) -> tuple[Callable, int]:
    """Build a fixed-Sz physical log amplitude, optionally exact lattice-A1.

    ``group`` contains physical gathers ``bits[g]``. Conjugating each gather
    by the coordinate map preserves the COMPLETE coherent Reynolds sum.
    A missing group means no external space-group projection, not singlet.
    """
    mapping = physical_to_model_permutation(config)
    inverse = np.argsort(mapping)
    if group is None:
        raw = lambda p, x: fixed_sz_spatial_vit_log_amplitude(p, x, config)
        count = 1
    else:
        permutations = np.asarray(group, dtype=np.int32)
        if (permutations.ndim != 2 or permutations.shape[1] != config.n_sites
                or not np.all(np.sort(permutations, axis=1) == np.arange(config.n_sites))):
            raise ValueError('one physical site permutation per group row required')
        model_group = inverse[permutations[:, mapping]]
        raw, count = projected_log_function(config, model_group)
    gather = jnp.asarray(mapping)
    return lambda p, x: raw(p, x[gather]), count


def make_scalar_structure_factors(
    log_amplitude: Callable, coordinates: np.ndarray,
    momenta: Mapping[str, Sequence[float]], *, connector_batch: int = 4,
) -> Callable:
    """Return full scalar S(k) AND diagonal SzSz records for arbitrary states.

    Uses S_i.S_j = Sz_i Sz_j + (S_i+ S_j- + S_i- S_j+)/2 and coherent
    exchange ratios. NEVER substitutes 3*SzSz for a fixed-Sz neural state.
    The output columns follow ``momenta`` insertion order. S(Gamma)*N is
    total <S^2>, with doubled-spin convention q(q+2)/4 for an eigenstate.
    Off-diagonal queries are additional observable work, NOT full-H calls.
    """
    coords = np.asarray(coordinates, float)
    ks = np.asarray(list(momenta.values()), float)
    if (coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2
            or ks.ndim != 2 or ks.shape[1] != 2
            or not np.isfinite(coords).all() or not np.isfinite(ks).all()
            or isinstance(connector_batch, bool) or connector_batch < 1):
        raise ValueError('finite coordinates/momenta and positive connector batch required')
    n = len(coords)
    edges = np.array(list(combinations(range(n), 2)), np.int32)
    phases = np.exp(2j*np.pi*coords@ks.T)
    weights = np.cos(2*np.pi*(coords[edges[:, 0]]-coords[edges[:, 1]])@ks.T)
    j_edges, j_weights, j_phases = map(jnp.asarray, (edges, weights, phases))

    def one(params, bits):
        center = log_amplitude(params, bits)
        def pair(edge):
            i, j = edge
            swapped = bits.at[i].set(bits[j]).at[j].set(bits[i])
            ratio = jnp.exp(log_amplitude(params, swapped)-center)
            return ((bits[i]-.5)*(bits[j]-.5)
                    + .5*jnp.where(bits[i] != bits[j], ratio, 0.))
        dots = jax.lax.map(pair, j_edges, batch_size=connector_batch)
        scalar = .75 + (2./n)*(dots@j_weights)
        zz = abs((bits-.5)@j_phases)**2/n
        return scalar, zz
    return jax.jit(jax.vmap(one, in_axes=(None, 0)))


def born_conditional_regional_matrices(
    plan: RegionalQueryPlan, log_amplitudes: np.ndarray,
) -> np.ndarray:
    """Coherent conditional RDMs for environments drawn from THIS state's Born law.

    Each record is psi(A,env) psi(A,env)^dagger / sum_A |psi(A,env)|^2.
    Averaging over equilibrium Born environments is an unbiased RDM estimator.
    Exact regional nodes are permitted; zero-marginal environments fail closed.
    A record-wise log shift is a numerical gauge and cancels exactly. This
    estimator is NOT valid for samples drawn from another wavefunction.
    """
    logs = np.asarray(log_amplitudes, np.complex128)
    if (logs.shape != (len(plan.configurations),) or np.isnan(logs).any()
            or np.isposinf(logs.real).any() or not np.isfinite(logs.imag).all()):
        raise ValueError('one finite-or-zero-node log amplitude per regional query required')
    local_logs = np.where(plan.valid, logs[plan.indices], -np.inf+0j)
    shifts = np.max(local_logs.real, axis=1)
    if not np.isfinite(shifts).all():
        raise ValueError('Born environment has zero conditional norm')
    vectors = np.exp(local_logs-shifts[:, None])
    norms = np.sum(abs(vectors)**2, axis=1)
    return vectors[:, :, None]*vectors[:, None, :].conj()/norms[:, None, None]


__all__ = ['physical_to_model_permutation', 'make_physical_spatial_vit_log',
           'make_scalar_structure_factors', 'born_conditional_regional_matrices',
           'ScalarStructureQueryPlan', 'scalar_structure_query_plan',
           'scalar_structure_records']
