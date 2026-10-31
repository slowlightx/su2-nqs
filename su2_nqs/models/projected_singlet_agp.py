"""Pure-JAX strict singlet ``Gamma.A1`` AGP amplitudes.

This is an alternative computational-basis carrier, not a fusion-path NQS.
For a symmetric pairing matrix ``F`` and a half-filled spin configuration,
the Gutzwiller-projected AGP amplitude is a signed determinant.  A Reynolds
average over physical site permutations then puts the state in the trivial
spatial irrep exactly (up to floating-point roundoff).

The evaluator is intentionally unnormalised and has no Born sampler.  VMC
code must supply an independent positive, full-support proposal.  Nodes are
kept as exact zeros; callers that require log derivatives must use the host
audit below, which fails closed instead of inserting an amplitude floor.

Bits follow the repository convention: zero is down, one is up, and site zero
is the most-significant bit.  A group image is active,
``image[old_site] == new_site``; evaluating ``U_image psi`` therefore pulls a
target configuration back as ``target[image[old_site]]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Literal, Mapping, NamedTuple, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


SitePermutation = tuple[int, ...]
ParameterMode = Literal["real", "complex"]


def _permutation_parity(image: SitePermutation) -> int:
    inversions = sum(
        image[first] > image[second]
        for first in range(len(image))
        for second in range(first + 1, len(image))
    )
    return -1 if inversions % 2 else 1


def _validated_group(
    group_images: Sequence[Sequence[int]], n_sites: int
) -> tuple[SitePermutation, ...]:
    if not group_images:
        raise ValueError("group_images must be nonempty")
    group: list[SitePermutation] = []
    expected = tuple(range(n_sites))
    for raw in group_images:
        if len(raw) != n_sites or any(
            isinstance(site, bool) or not isinstance(site, Integral)
            for site in raw
        ):
            raise ValueError("each group image must be an integer site permutation")
        image = tuple(int(site) for site in raw)
        if tuple(sorted(image)) != expected:
            raise ValueError("each group image must contain every site exactly once")
        group.append(image)
    if len(set(group)) != len(group):
        raise ValueError("group_images must not contain duplicate maps")
    elements = set(group)
    if expected not in elements:
        raise ValueError("group_images must contain the identity")
    for left in group:
        for right in group:
            if tuple(left[right[site]] for site in range(n_sites)) not in elements:
                raise ValueError("group_images must be closed under composition")
    return tuple(group)


@dataclass(frozen=True)
class ProjectedSingletAGPConfig:
    """Static, serialisable contract for a strict projected AGP carrier.

    ``parameter_mode='real'`` is the fail-closed time-reversal-invariant path.
    Complex symmetric pairings remain exact spin singlets and exact
    ``Gamma.A1`` states, but are explicitly marked time-reversal unconstrained.
    """

    n_sites: int
    group_images: tuple[SitePermutation, ...]
    parameter_mode: ParameterMode = "real"
    cancellation_tolerance: float = 1.0e-12
    node_relative_tolerance: float = 1.0e-13
    time_reversal_tolerance: float = 2.0e-11

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_sites, bool)
            or not isinstance(self.n_sites, Integral)
            or int(self.n_sites) < 2
            or int(self.n_sites) % 2
        ):
            raise ValueError("n_sites must be a positive even integer")
        n_sites = int(self.n_sites)
        group = _validated_group(self.group_images, n_sites)
        if self.parameter_mode not in ("real", "complex"):
            raise ValueError("parameter_mode must be 'real' or 'complex'")
        for name, value in (
            ("cancellation_tolerance", self.cancellation_tolerance),
            ("node_relative_tolerance", self.node_relative_tolerance),
            ("time_reversal_tolerance", self.time_reversal_tolerance),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "group_images", group)

    @property
    def pair_count(self) -> int:
        return self.n_sites // 2

    @property
    def parameter_count_per_real_component(self) -> int:
        return self.n_sites * (self.n_sites - 1) // 2

    @property
    def upper_triangle_indices(self) -> tuple[jax.Array, jax.Array]:
        first, second = np.triu_indices(self.n_sites, k=1)
        return jnp.asarray(first, dtype=jnp.int32), jnp.asarray(
            second, dtype=jnp.int32
        )

    @property
    def group_order(self) -> int:
        return len(self.group_images)

    @property
    def enforces_time_reversal(self) -> bool:
        return self.parameter_mode == "real"

    @property
    def all_group_permutations_even(self) -> bool:
        return all(_permutation_parity(image) == 1 for image in self.group_images)

    @property
    def group_array(self) -> jax.Array:
        return jnp.asarray(self.group_images, dtype=jnp.int32)

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return deterministic JSON-compatible carrier metadata."""

        encoded = json.dumps(
            self.group_images, separators=(",", ":"), sort_keys=False
        ).encode("utf-8")
        return {
            "schema": "projected_singlet_agp_jax_v1",
            "carrier_role": "alternative_strict_sector_carrier_not_fusion_nqs",
            "n_sites": self.n_sites,
            "pair_count": self.pair_count,
            "pairing_parameterization": "packed_strict_upper_triangle",
            "parameter_count_per_real_component": (
                self.parameter_count_per_real_component
            ),
            "group_order": self.group_order,
            "group_images_sha256": hashlib.sha256(encoded).hexdigest(),
            "parameter_mode": self.parameter_mode,
            "time_reversal": (
                "enforced_by_real_pairing"
                if self.enforces_time_reversal
                else "unconstrained"
            ),
            "spatial_sector": "Gamma.A1_physical_Reynolds",
            "spin_sector": "q=0_exact_symmetric_pairing",
            "normalization": "unknown",
            "born_sampler": "none",
            "required_sampling_contract": "external_positive_full_support_proposal",
            "cancellation_tolerance": self.cancellation_tolerance,
            "node_relative_tolerance": self.node_relative_tolerance,
            "time_reversal_tolerance": self.time_reversal_tolerance,
        }


class ProjectedAGPBatch(NamedTuple):
    """JAX-pytree output for a batch of projected amplitudes."""

    amplitudes: jax.Array
    log_absolute: jax.Array
    phases: jax.Array
    cancellation_ratios: jax.Array
    exact_nodes: jax.Array


def init_projected_singlet_agp_parameters(
    key: jax.Array,
    config: ProjectedSingletAGPConfig,
    *,
    scale: float | None = None,
) -> dict[str, jax.Array]:
    """Initialise checkpoint-friendly unconstrained pairing parameters."""

    if scale is None:
        scale = config.n_sites**-0.5
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if config.parameter_mode == "real":
        return {
            "pairing_real": scale
            * jax.random.normal(
                key,
                (config.parameter_count_per_real_component,),
                dtype=jnp.float64,
            )
        }
    real_key, imag_key = jax.random.split(key)
    return {
        "pairing_real": scale
        * jax.random.normal(
            real_key,
            (config.parameter_count_per_real_component,),
            dtype=jnp.float64,
        ),
        "pairing_imag": scale
        * jax.random.normal(
            imag_key,
            (config.parameter_count_per_real_component,),
            dtype=jnp.float64,
        ),
    }


def symmetric_pairing_from_parameters(
    parameters: Mapping[str, jax.Array],
    config: ProjectedSingletAGPConfig,
) -> jax.Array:
    """Construct an exactly symmetric, zero-diagonal JAX pairing matrix."""

    expected = (config.parameter_count_per_real_component,)
    if set(parameters) != (
        {"pairing_real"}
        if config.parameter_mode == "real"
        else {"pairing_real", "pairing_imag"}
    ):
        raise ValueError("parameter keys do not match the configured parameter mode")
    real = jnp.asarray(parameters["pairing_real"], dtype=jnp.float64)
    if real.shape != expected:
        raise ValueError("pairing_real has the wrong shape")
    if config.parameter_mode == "complex":
        imag = jnp.asarray(parameters["pairing_imag"], dtype=jnp.float64)
        if imag.shape != expected:
            raise ValueError("pairing_imag has the wrong shape")
        raw = real + 1.0j * imag
    else:
        raw = real
    first, second = config.upper_triangle_indices
    pairing = jnp.zeros((config.n_sites, config.n_sites), dtype=raw.dtype)
    pairing = pairing.at[first, second].set(raw)
    return pairing.at[second, first].set(raw)


def _configuration_signs(configurations: jax.Array) -> jax.Array:
    left_down = jnp.cumsum(1 - configurations, axis=-1) - (
        1 - configurations
    )
    inversion_parity = jnp.mod(
        jnp.sum(configurations * left_down, axis=-1), 2
    )
    pair_count = configurations.shape[-1] // 2
    pfaffian_parity = (pair_count * (pair_count - 1) // 2) % 2
    parity = jnp.mod(inversion_parity + pfaffian_parity, 2)
    return jnp.where(parity == 0, 1.0, -1.0)


def _base_agp_slogdet(
    pairing: jax.Array, configurations: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return direct determinants plus stable phase/log diagnostics.

    The direct determinant is retained because its polynomial autodiff remains
    defined at singular blocks.  Differentiating a ``slogdet`` through a node
    would instead invoke a matrix inverse and produce a false NaN gradient.
    """

    n_sites = configurations.shape[-1]
    pair_count = n_sites // 2
    # At half filling, a stable sort yields increasing down indices followed by
    # increasing up indices without dynamic-size indexing inside JIT.
    site_order = jnp.argsort(configurations, axis=-1, stable=True)
    down = site_order[..., :pair_count]
    up = site_order[..., pair_count:]
    blocks = pairing[up[..., :, None], down[..., None, :]]
    signs = _configuration_signs(configurations)
    direct_determinants = signs * jnp.linalg.det(blocks)
    determinant_phase, log_absolute = jnp.linalg.slogdet(blocks)
    return direct_determinants, signs * determinant_phase, log_absolute


@jax.jit
def _projected_agp_batch_kernel(
    pairing: jax.Array,
    configurations: jax.Array,
    group_images: jax.Array,
    cancellation_tolerance: jax.Array,
) -> ProjectedAGPBatch:
    source_configurations = jnp.take(configurations, group_images, axis=1)
    batch_size, group_order, n_sites = source_configurations.shape
    flat_sources = source_configurations.reshape((batch_size * group_order, n_sites))
    direct_determinants, determinant_phase, log_absolute = _base_agp_slogdet(
        pairing, flat_sources
    )
    direct_determinants = direct_determinants.reshape((batch_size, group_order))
    determinant_phase = determinant_phase.reshape((batch_size, group_order))
    log_absolute = log_absolute.reshape((batch_size, group_order))

    finite = jnp.isfinite(log_absolute)
    any_finite = jnp.any(finite, axis=1)
    maximum_log = jnp.max(
        jnp.where(finite, log_absolute, -jnp.inf), axis=1
    )
    safe_maximum_log = jnp.where(any_finite, maximum_log, 0.0)
    scaled_terms = jnp.where(
        finite,
        determinant_phase * jnp.exp(log_absolute - safe_maximum_log[:, None]),
        0.0,
    )
    scaled_sum = jnp.sum(scaled_terms, axis=1)
    scaled_l1 = jnp.sum(jnp.abs(scaled_terms), axis=1)
    projected_scaled_absolute = jnp.abs(scaled_sum)
    amplitudes = jnp.mean(direct_determinants, axis=1)
    exact_nodes = jnp.abs(amplitudes) == 0.0
    projected_log_absolute = jnp.where(
        exact_nodes,
        -jnp.inf,
        safe_maximum_log
        + jnp.log(projected_scaled_absolute)
        - jnp.log(jnp.asarray(group_order, dtype=safe_maximum_log.dtype)),
    )
    phases = jnp.where(
        exact_nodes, 0.0, scaled_sum / projected_scaled_absolute
    )
    cancellation_ratios = jnp.where(
        scaled_l1 > 0.0, projected_scaled_absolute / scaled_l1, 0.0
    )
    # Keep the threshold in the compiled graph.  It is not used to alter the
    # wavefunction; the host audit interprets it and fails closed for log VMC.
    cancellation_ratios = cancellation_ratios + 0.0 * cancellation_tolerance
    return ProjectedAGPBatch(
        amplitudes=amplitudes,
        log_absolute=projected_log_absolute,
        phases=phases,
        cancellation_ratios=cancellation_ratios,
        exact_nodes=exact_nodes,
    )


def _validate_configurations_shape(
    configurations: jax.Array, config: ProjectedSingletAGPConfig
) -> None:
    if configurations.ndim != 2 or configurations.shape[1] != config.n_sites:
        raise ValueError("configurations must have shape (batch, n_sites)")
    if configurations.shape[0] < 1:
        raise ValueError("configuration batch must be nonempty")


def projected_singlet_agp_amplitudes(
    parameters: Mapping[str, jax.Array],
    configurations: jax.Array | np.ndarray | Sequence[Sequence[int]],
    config: ProjectedSingletAGPConfig,
    *,
    microbatch_size: int | None = None,
) -> ProjectedAGPBatch:
    """Evaluate exact Reynolds-projected amplitudes in JAX microbatches.

    The returned arrays remain differentiable with respect to ``parameters``.
    Shape checks are static; binary and half-filling value checks belong to the
    fail-closed host audit because traced configuration values cannot be
    converted to Python during differentiation/JIT.
    """

    pairing = symmetric_pairing_from_parameters(parameters, config)
    raw_states = jnp.asarray(configurations)
    if isinstance(raw_states, jax.core.Tracer):
        states = raw_states.astype(jnp.int8)
    else:
        states = jnp.asarray(
            _validated_host_configurations(configurations, config),
            dtype=jnp.int8,
        )
    _validate_configurations_shape(states, config)
    batch_size = int(states.shape[0])
    if microbatch_size is None:
        microbatch_size = batch_size
    if (
        isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, Integral)
        or int(microbatch_size) < 1
    ):
        raise ValueError("microbatch_size must be a positive integer")
    microbatch_size = int(microbatch_size)
    group = config.group_array
    tolerance = jnp.asarray(config.cancellation_tolerance, dtype=jnp.float64)
    pieces: list[ProjectedAGPBatch] = []
    for start in range(0, batch_size, microbatch_size):
        pieces.append(
            _projected_agp_batch_kernel(
                pairing,
                states[start : min(start + microbatch_size, batch_size)],
                group,
                tolerance,
            )
        )
    return ProjectedAGPBatch(
        *(jnp.concatenate([piece[field] for piece in pieces], axis=0) for field in range(5))
    )


def spatially_invariant_pairing_from_parameters(
    parameters: Mapping[str, jax.Array],
    config: ProjectedSingletAGPConfig,
) -> jax.Array:
    """Average ``F`` over the site group, yielding a restricted pairing."""

    pairing = symmetric_pairing_from_parameters(parameters, config)
    group = config.group_array
    transformed = pairing[group[:, :, None], group[:, None, :]]
    return jnp.mean(transformed, axis=0)


def invariant_pairing_fast_amplitudes(
    parameters: Mapping[str, jax.Array],
    configurations: jax.Array | np.ndarray | Sequence[Sequence[int]],
    config: ProjectedSingletAGPConfig,
    *,
    microbatch_size: int | None = None,
) -> ProjectedAGPBatch:
    """Evaluate the low-expressivity invariant-pairing control without ``|G|``.

    A site permutation of the fermionic AGP induces its permutation parity in
    the physical spin basis.  Consequently this shortcut is ``A1`` only when
    every retained site permutation is even.  Odd groups fail closed instead
    of silently returning the wrong one-dimensional character.
    """

    if not config.all_group_permutations_even:
        raise ValueError(
            "invariant-pairing fast path is not physical A1: the group contains "
            "an odd site permutation"
        )
    pairing = spatially_invariant_pairing_from_parameters(parameters, config)
    states = jnp.asarray(configurations, dtype=jnp.int8)
    _validate_configurations_shape(states, config)
    identity_config = ProjectedSingletAGPConfig(
        n_sites=config.n_sites,
        group_images=(tuple(range(config.n_sites)),),
        parameter_mode=config.parameter_mode,
        cancellation_tolerance=config.cancellation_tolerance,
        node_relative_tolerance=config.node_relative_tolerance,
        time_reversal_tolerance=config.time_reversal_tolerance,
    )
    # Feed the already constrained pairing through a raw matrix whose
    # symmetrisation is idempotent.
    first, second = config.upper_triangle_indices
    fast_parameters: dict[str, jax.Array] = {
        "pairing_real": jnp.real(pairing[first, second])
    }
    if config.parameter_mode == "complex":
        fast_parameters["pairing_imag"] = jnp.imag(pairing[first, second])
    return projected_singlet_agp_amplitudes(
        fast_parameters,
        states,
        identity_config,
        microbatch_size=microbatch_size,
    )


@dataclass(frozen=True)
class ProjectedAGPTrainabilityAudit:
    """Host-side sampled-domain audit; never claims global Hilbert support."""

    batch_size: int
    nonfinite_count: int
    exact_node_count: int
    near_node_count: int
    severe_cancellation_count: int
    minimum_cancellation_ratio: float
    time_reversal_residual: float
    time_reversal_pass: bool
    sampled_log_vmc_safe: bool
    scope: str = "provided_configurations_and_spin_flips_only"

    def require_log_vmc_safe(self) -> None:
        if not self.sampled_log_vmc_safe:
            raise FloatingPointError(
                "projected AGP log-VMC batch is unsafe: nonfinite amplitude, "
                "node, severe Reynolds cancellation, or time-reversal failure"
            )


def _validated_host_configurations(
    configurations: jax.Array | np.ndarray | Sequence[Sequence[int]],
    config: ProjectedSingletAGPConfig,
) -> np.ndarray:
    states = np.asarray(configurations)
    if states.ndim != 2 or states.shape[1] != config.n_sites or states.shape[0] < 1:
        raise ValueError("configurations must have shape (nonzero_batch, n_sites)")
    if states.dtype == np.bool_:
        raise ValueError("configurations must use integer binary labels, not bool")
    if np.any((states != 0) & (states != 1)):
        raise ValueError("configurations must be binary")
    if np.any(np.sum(states, axis=1) != config.pair_count):
        raise ValueError("every configuration must have exactly n_sites/2 up spins")
    return np.asarray(states, dtype=np.int8)


def audit_projected_singlet_agp_batch(
    parameters: Mapping[str, jax.Array],
    configurations: jax.Array | np.ndarray | Sequence[Sequence[int]],
    config: ProjectedSingletAGPConfig,
    *,
    microbatch_size: int | None = None,
) -> ProjectedAGPTrainabilityAudit:
    """Audit nodes, Reynolds cancellation, finiteness, and time reversal.

    Passing this test establishes log-derivative safety only on the supplied
    configurations and their global spin flips.  It is not a proof of full
    support on the exponentially large fixed-``Sz`` basis.
    """

    states = _validated_host_configurations(configurations, config)
    result = projected_singlet_agp_amplitudes(
        parameters, states, config, microbatch_size=microbatch_size
    )
    flipped_result = projected_singlet_agp_amplitudes(
        parameters, 1 - states, config, microbatch_size=microbatch_size
    )
    amplitudes = np.asarray(result.amplitudes)
    flipped = np.asarray(flipped_result.amplitudes)
    cancellations = np.asarray(result.cancellation_ratios, dtype=np.float64)
    nonfinite = ~(np.isfinite(amplitudes.real) & np.isfinite(amplitudes.imag))
    scale = max(float(np.max(np.abs(amplitudes))), np.finfo(np.float64).tiny)
    near_node = np.abs(amplitudes) <= config.node_relative_tolerance * scale
    exact_nodes = np.asarray(result.exact_nodes, dtype=bool)
    severe_cancellation = cancellations <= config.cancellation_tolerance

    theta = ((-1) ** config.pair_count) * np.conj(flipped)
    norm = float(np.linalg.norm(amplitudes))
    theta_norm = float(np.linalg.norm(theta))
    if norm == 0.0 or theta_norm == 0.0:
        time_reversal_residual = float("inf")
    else:
        overlap = np.vdot(amplitudes, theta)
        phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
        time_reversal_residual = float(
            np.linalg.norm(theta - phase * amplitudes)
            / max(norm, theta_norm)
        )
    time_reversal_pass = (
        not config.enforces_time_reversal
        or time_reversal_residual <= config.time_reversal_tolerance
    )
    sampled_safe = bool(
        not np.any(nonfinite)
        and not np.any(exact_nodes)
        and not np.any(near_node)
        and not np.any(severe_cancellation)
        and time_reversal_pass
    )
    return ProjectedAGPTrainabilityAudit(
        batch_size=len(states),
        nonfinite_count=int(np.sum(nonfinite)),
        exact_node_count=int(np.sum(exact_nodes)),
        near_node_count=int(np.sum(near_node)),
        severe_cancellation_count=int(np.sum(severe_cancellation)),
        minimum_cancellation_ratio=float(np.min(cancellations)),
        time_reversal_residual=time_reversal_residual,
        time_reversal_pass=bool(time_reversal_pass),
        sampled_log_vmc_safe=sampled_safe,
    )


def checked_projected_singlet_agp_log_amplitudes(
    parameters: Mapping[str, jax.Array],
    configurations: jax.Array | np.ndarray | Sequence[Sequence[int]],
    config: ProjectedSingletAGPConfig,
    *,
    microbatch_size: int | None = None,
) -> jax.Array:
    """Return log amplitudes only after the sampled-domain fail-closed audit."""

    audit = audit_projected_singlet_agp_batch(
        parameters,
        configurations,
        config,
        microbatch_size=microbatch_size,
    )
    audit.require_log_vmc_safe()
    result = projected_singlet_agp_amplitudes(
        parameters,
        configurations,
        config,
        microbatch_size=microbatch_size,
    )
    return result.log_absolute + 1.0j * jnp.angle(result.phases)


__all__ = [
    "ProjectedAGPBatch",
    "ProjectedAGPTrainabilityAudit",
    "ProjectedSingletAGPConfig",
    "audit_projected_singlet_agp_batch",
    "checked_projected_singlet_agp_log_amplitudes",
    "init_projected_singlet_agp_parameters",
    "invariant_pairing_fast_amplitudes",
    "projected_singlet_agp_amplitudes",
    "spatially_invariant_pairing_from_parameters",
    "symmetric_pairing_from_parameters",
]
