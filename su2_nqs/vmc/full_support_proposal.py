"""IID fixed-doubled-projection proposals with an exact uniform component.

The proposal is not the target wavefunction. For even ``n_sites`` it is
``(1-epsilon)/2*(Q_orbit+Q_Hpush)+epsilon/binom(n_sites,n_sites/2)`` in
the doubled-projection ``m=0`` computational sector. No amplitude clipping,
wavefunction floor, approximate Hamiltonian, or empirical branch reweighting
is introduced. Bits use0=down and site zero is the most significant bit.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from math import comb
from numbers import Integral, Real
from typing import Any

import numpy as np

DEFAULT_UNIFORM_FRACTION = 1 / 8
_STREAM_ROLES = ("branch", "base", "uniform")


def _settings(n_sites: int, epsilon: float) -> tuple[int, float, float]:
    if (isinstance(n_sites, bool) or not isinstance(n_sites, Integral)
            or n_sites < 2 or n_sites % 2):
        raise ValueError("an even integer site count >=2 is required for doubled m=0")
    if (isinstance(epsilon, bool) or not isinstance(epsilon, Real)
            or not np.isfinite(epsilon) or not 0 < epsilon < 1):
        raise ValueError("the fixed uniform mixture weight must lie in(0,1)")
    try:
        uniform = 1.0 / comb(int(n_sites), int(n_sites) // 2)
    except OverflowError as error:
        raise ValueError("fixed-m uniform probability is not representable in float64") from error
    if not np.isfinite(uniform) or float(epsilon) * uniform <= 0:
        raise ValueError("the positive uniform component must be representable in float64")
    return int(n_sites), float(epsilon), uniform


def full_support_sz0_constants(
    n_sites: int, *, epsilon: float = DEFAULT_UNIFORM_FRACTION,
) -> dict[str, Any]:
    """Return exact combinatorial and fixed mixture constants for doubled m=0.

    ``minimum_density`` is a proposal-support bound, not a target-state floor
    or a promise that the finite sample resolves every important contribution.
    """
    n_sites, epsilon, uniform = _settings(n_sites, epsilon)
    return dict(n_sites=n_sites, doubled_projection=0,
        sector_dimension=comb(n_sites, n_sites // 2), uniform_probability=uniform,
        uniform_fraction=epsilon, orbit_weight=(1-epsilon)/2,
        Hpush_weight=(1-epsilon)/2, minimum_density=epsilon*uniform,
        fixed_weights_not_empirical_branch_frequencies=True,
        target_wavefunction_changed=False, density_clipping=False)


def full_support_sz0_density(
    orbit_density: np.ndarray, Hpush_density: np.ndarray, *, n_sites: int,
    epsilon: float = DEFAULT_UNIFORM_FRACTION,
) -> np.ndarray:
    """Combine normalized Qorbit/QHpush with the analytic doubled-m=0 term.

    Both inputs must already be separately normalized in the same fixed-spin
    computational sector. For several normalized raw banks, average their
    orbit and push densities before this function. Positive support follows
    from the fixed epsilon term even when both input densities are zero.
    Input values are validated, never clipped. This host utility is outside
    JAX tracing; JAX kernels may use the returned fixed constants directly.
    """
    constants = full_support_sz0_constants(n_sites, epsilon=epsilon)
    orbit, pushed = np.asarray(orbit_density), np.asarray(Hpush_density)
    if (orbit.shape != pushed.shape or np.iscomplexobj(orbit)
            or np.iscomplexobj(pushed) or not np.isfinite(orbit).all()
            or not np.isfinite(pushed).all() or np.any(orbit < 0)
            or np.any(pushed < 0)):
        raise ValueError("matching finite nonnegative real proposal densities required")
    return ((orbit.astype(np.float64) + pushed.astype(np.float64))
            * constants["orbit_weight"] + constants["minimum_density"])


def _new_streams(seed: int) -> tuple[dict[str, np.random.Generator], dict[str, Any]]:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("a nonnegative integer seed is required")
    children = np.random.SeedSequence(int(seed)).spawn(len(_STREAM_ROLES))
    streams = {role: np.random.Generator(np.random.PCG64(child))
               for role, child in zip(_STREAM_ROLES, children, strict=True)}
    identity = {role: dict(entropy=int(child.entropy), spawn_key=list(child.spawn_key),
                          pool_size=int(child.pool_size))
                for role, child in zip(_STREAM_ROLES, children, strict=True)}
    return streams, identity


def _snapshot(
    streams: Mapping[str, np.random.Generator], identity: dict[str, Any], *,
    n_sites: int, epsilon: float, batches: int, records: int,
) -> dict[str, Any]:
    return dict(version=1, bit_generator="PCG64", n_sites=n_sites,
        epsilon=epsilon, batches=batches, records=records,
        stream_identities=deepcopy(identity),
        streams={role: deepcopy(streams[role].bit_generator.state) for role in _STREAM_ROLES})


def draw_full_support_sz0(
    draw_bridge: Callable[[int, int], tuple[np.ndarray, dict[str, Any]]],
    count: int, *, n_sites: int, seed: int | None = None,
    rng_state: Mapping[str, Any] | None = None,
    epsilon: float = DEFAULT_UNIFORM_FRACTION,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw genuine IID rows from a fixed bridge/uniform proposal mixture.

    ``draw_bridge(base_count,base_seed)`` must return exact IID rows from
    ``(Qorbit+QHpush)/2`` plus its latent/RNG/query report. The CT adapter is
    ``diagnostic.draw_orbit_rows([normalized_bank],rt,k,s,bridge=True)``.
    Independent PCG64 substreams decide outer branches, seed that callback,
    and generate uniform random permutations of N/2 zeros and N/2 ones.
    A uniform permutation is uniform over configurations despite duplicate
    bits, since every configuration has the same(N/2)!**2 permutations.

    Returned arrays identify each row's branch and callback row index;
    callback metadata is retained under ``base_draw``. Save those arrays
    separately when serializing the report to JSON. ``rng_after`` supports
    exact continuation at this batch boundary. Reproducing arbitrary changes
    to the batching plan is not promised because the callback takes one seed
    per batch. Neither branch frequencies nor these latents define density:
    always use the fixed weights in ``full_support_sz0_density``.
    """
    n_sites, epsilon, _ = _settings(n_sites, epsilon)
    if (isinstance(count, bool) or not isinstance(count, Integral) or count < 1
            or not callable(draw_bridge) or (seed is None) == (rng_state is None)):
        raise ValueError("positive record count, callback and exactly one seed/RNG state required")
    count = int(count)
    if rng_state is None:
        streams, identity = _new_streams(seed)
        batches, records = 0, 0
    else:
        state = deepcopy(dict(rng_state))
        if (state.get("version") != 1 or state.get("bit_generator") != "PCG64"
                or state.get("n_sites") != n_sites or state.get("epsilon") != epsilon
                or set(state.get("streams", {})) != set(_STREAM_ROLES)
                or set(state.get("stream_identities", {})) != set(_STREAM_ROLES)
                or any(isinstance(state.get(k), bool) or not isinstance(state.get(k), Integral)
                       or state[k] < 0 for k in ("batches", "records"))):
            raise ValueError("RNG state does not match this full-support sampler")
        streams = {role: np.random.Generator(np.random.PCG64()) for role in _STREAM_ROLES}
        for role in _STREAM_ROLES:
            streams[role].bit_generator.state = state["streams"][role]
        identity = state["stream_identities"]
        batches, records = int(state["batches"]), int(state["records"])
    before = _snapshot(streams, identity, n_sites=n_sites, epsilon=epsilon,
                       batches=batches, records=records)
    uniform_mask = streams["branch"].random(count) < epsilon
    uniform_ids, base_ids = np.flatnonzero(uniform_mask), np.flatnonzero(~uniform_mask)
    base_seed = int(streams["base"].integers(0, np.iinfo(np.int64).max))
    rows = np.empty((count, n_sites), np.int32)
    base_report: dict[str, Any] = {}
    if len(base_ids):
        base, base_report = draw_bridge(len(base_ids), base_seed)
        base = np.asarray(base)
        if (base.shape != (len(base_ids), n_sites) or not np.issubdtype(base.dtype, np.integer)
                or np.any((base != 0) & (base != 1))
                or np.any(base.sum(axis=1) != n_sites // 2)
                or not isinstance(base_report, dict)):
            raise ValueError("bridge callback must return exact fixed-doubled-m=0 integer-bit rows")
        rows[base_ids] = base
    template = np.repeat(np.array([0, 1], np.int32), n_sites // 2)
    for index in uniform_ids:
        rows[index] = streams["uniform"].permutation(template)
    callback_index = np.full(count, -1, np.int64)
    callback_index[base_ids] = np.arange(len(base_ids))
    after = _snapshot(streams, identity, n_sites=n_sites, epsilon=epsilon,
                      batches=batches+1, records=records+count)
    report = dict(**full_support_sz0_constants(n_sites, epsilon=epsilon),
        records=count, base_records=int(len(base_ids)), uniform_records=int(len(uniform_ids)),
        is_uniform=uniform_mask, bridge_row_indices=base_ids, uniform_row_indices=uniform_ids,
        callback_row_index=callback_index, base_seed=base_seed, base_draw=deepcopy(base_report),
        rng_before=before, rng_after=after, exact_IID_requires_exact_IID_bridge_callback=True,
        continuation_granularity="same_batch_call_plan", empirical_branch_weights_used=False,
        work=dict(outer_branch_labels=count, exact_bridge_draws=int(len(base_ids)),
            uniform_permutations=int(len(uniform_ids)), target_amplitude_queries=0,
            full_H_queries=0, callback_work_retained_in_base_draw=True))
    return rows, report
