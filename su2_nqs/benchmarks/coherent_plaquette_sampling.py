"""Matrix-free sampling diagnostics for coherent plaquette-frame carriers.

Configurations are proposed from the equally weighted mixture of normalized
plaquette-product Born distributions.  The coherent target is evaluated as a
signed sum of those same frame amplitudes.  This gives an exact, tractable
importance proposal without enumerating the spin Hilbert space.

For a canonical Schur tree, an inter-cell label has ``q>0`` exactly when the
corresponding block has nonzero total-spin Casimir.  We therefore use sampled
``<S_A^2>`` on every non-root inter-cell block as a basis-independent witness
of nonzero virtual-spin support.  This avoids pretending that a physical-bit
sample is itself a sample of fusion labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..su2.many_cell_schur import ManyCellSchurLayout
from ..su2.tree_rotations import leaf_signature
from .analytic_plaquette_carrier import aligned_plaquette_a1_local_data
from .local_multiplicity_carriers import square_plaquette_singlet_carrier


Frame = tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class CoherentFrameSampleBatch:
    """Mixture samples and exact target/proposal values."""

    configurations: np.ndarray
    proposal_probability: np.ndarray
    coherent_amplitude: np.ndarray
    importance_weight: np.ndarray
    component_index: np.ndarray


@dataclass(frozen=True)
class LinearResonanceSampledAction:
    """Sampled ``base + lambda*response`` amplitudes and Hamiltonian actions."""

    proposal_probability: np.ndarray
    base_amplitude: np.ndarray
    response_amplitude: np.ndarray
    h_base_amplitude: np.ndarray
    h_response_amplitude: np.ndarray
    n_sites: int
    bond_count: int
    full_h_configuration_calls: int


@dataclass(frozen=True)
class ResonanceBankSampledAction:
    """Sampled amplitudes and exact-H actions for every local-B1 count.

    Column ``k`` is the full spatial average of products containing exactly
    ``k`` local ``B1`` singlets and ``n_cells-k`` local ``A1`` singlets.  The
    columns therefore change the many-plaquette singlet-resonance manifold
    without introducing magnetic-index amplitudes.
    """

    proposal_probability: np.ndarray
    amplitudes: np.ndarray
    h_amplitudes: np.ndarray
    n_sites: int
    bond_count: int
    full_h_configuration_calls: int


def _checked_frames(frames: Sequence[Sequence[Sequence[int]]]) -> tuple[Frame, ...]:
    checked = tuple(
        tuple(tuple(int(site) for site in cell) for cell in frame)
        for frame in frames
    )
    if not checked:
        raise ValueError("at least one plaquette frame is required")
    n_sites = sum(len(cell) for cell in checked[0])
    for frame in checked:
        if any(len(cell) != 4 for cell in frame):
            raise ValueError("every plaquette cell must contain four sites")
        flattened = tuple(site for cell in frame for site in cell)
        if tuple(sorted(flattened)) != tuple(range(n_sites)):
            raise ValueError("each frame must partition every site exactly once")
    return checked


def ordered_frame_orbit_with_weights(
    cells: Sequence[Sequence[int]],
    group: Sequence[Sequence[int]],
) -> tuple[tuple[Frame, ...], np.ndarray]:
    """Compress a full group orbit without discarding internal cell ordering.

    Cell order is irrelevant to a product or an elementary-symmetric local
    excitation and is sorted.  Site order *inside* each cell is retained,
    because it carries the local ``A1/B1`` transformation law.
    """

    primary = tuple(tuple(int(site) for site in cell) for cell in cells)
    n_sites = sum(len(cell) for cell in primary)
    if tuple(sorted(site for cell in primary for site in cell)) != tuple(
        range(n_sites)
    ):
        raise ValueError("primary cells must partition every site")
    counts: dict[Frame, int] = {}
    for raw_image in group:
        image = tuple(int(site) for site in raw_image)
        if tuple(sorted(image)) != tuple(range(n_sites)):
            raise ValueError("group contains an invalid site permutation")
        frame = tuple(
            sorted(tuple(image[site] for site in cell) for cell in primary)
        )
        counts[frame] = counts.get(frame, 0) + 1
    frames = tuple(sorted(counts))
    weights = np.asarray([counts[frame] for frame in frames], dtype=np.float64)
    weights /= np.sum(weights)
    weights.setflags(write=False)
    return frames, weights


def plaquette_frame_component_amplitudes(
    configurations: np.ndarray,
    frames: Sequence[Sequence[Sequence[int]]],
) -> np.ndarray:
    """Evaluate every normalized plaquette-product component."""

    checked = _checked_frames(frames)
    spins = np.asarray(configurations, dtype=np.int8)
    n_sites = sum(len(cell) for cell in checked[0])
    if spins.ndim != 2 or spins.shape[1] != n_sites:
        raise ValueError(f"configurations must have shape (batch,{n_sites})")
    if np.any((spins != 0) & (spins != 1)):
        raise ValueError("configurations must be binary")
    local, _ = aligned_plaquette_a1_local_data()
    weights = np.asarray((8, 4, 2, 1), dtype=np.int64)
    result = np.ones((len(spins), len(checked)), dtype=np.complex128)
    for frame_index, frame in enumerate(checked):
        for cell in frame:
            indices = spins[:, np.asarray(cell, dtype=np.int64)] @ weights
            result[:, frame_index] *= local[indices]
    return result


def sample_coherent_plaquette_frames(
    frames: Sequence[Sequence[Sequence[int]]],
    *,
    sample_count: int,
    seed: int,
) -> CoherentFrameSampleBatch:
    """Sample the frame-mixture proposal and attach exact importance data."""

    checked = _checked_frames(frames)
    if isinstance(sample_count, bool) or int(sample_count) < 2:
        raise ValueError("sample_count must be an integer of at least two")
    sample_count = int(sample_count)
    n_sites = sum(len(cell) for cell in checked[0])
    local, _ = aligned_plaquette_a1_local_data()
    probabilities = np.abs(local) ** 2
    active = np.flatnonzero(probabilities > 0.0)
    active_probability = probabilities[active] / np.sum(probabilities[active])
    rng = np.random.default_rng(int(seed))
    components = rng.integers(0, len(checked), size=sample_count)
    configurations = np.zeros((sample_count, n_sites), dtype=np.int8)
    local_choices = rng.choice(
        active,
        size=(sample_count, len(checked[0])),
        replace=True,
        p=active_probability,
    )
    local_bits = (
        (local_choices[..., None] >> np.arange(3, -1, -1, dtype=np.int64)) & 1
    ).astype(np.int8)
    for row, frame_index in enumerate(components):
        for cell_index, cell in enumerate(checked[int(frame_index)]):
            configurations[row, np.asarray(cell, dtype=np.int64)] = local_bits[
                row, cell_index
            ]
    amplitudes = plaquette_frame_component_amplitudes(configurations, checked)
    proposal = np.mean(np.abs(amplitudes) ** 2, axis=1)
    coherent = np.mean(amplitudes, axis=1)
    if np.any(proposal <= 0.0):
        raise RuntimeError("frame-mixture proposal vanished on its own sample")
    importance = np.abs(coherent) ** 2 / proposal
    return CoherentFrameSampleBatch(
        configurations=configurations,
        proposal_probability=np.asarray(proposal, dtype=np.float64),
        coherent_amplitude=np.asarray(coherent, dtype=np.complex128),
        importance_weight=np.asarray(importance, dtype=np.float64),
        component_index=np.asarray(components, dtype=np.int32),
    )


def importance_diagnostics(batch: CoherentFrameSampleBatch) -> dict[str, float]:
    """Return proposal quality and coherent-interference diagnostics."""

    weight = np.asarray(batch.importance_weight, dtype=np.float64)
    amplitudes = np.asarray(batch.coherent_amplitude)
    if np.sum(weight) <= 0.0:
        raise RuntimeError("coherent target has zero sampled norm")
    ess = float(np.sum(weight) ** 2 / np.sum(weight**2))
    return {
        "importance_effective_sample_fraction": ess / len(weight),
        "mean_importance_weight": float(np.mean(weight)),
        "zero_target_fraction": float(np.mean(np.abs(amplitudes) <= 1.0e-15)),
    }


def _coherent_amplitudes(
    configurations: np.ndarray, frames: tuple[Frame, ...]
) -> np.ndarray:
    return np.mean(plaquette_frame_component_amplitudes(configurations, frames), axis=1)


def coherent_pair_resonance_channels(
    configurations: np.ndarray,
    frames: Sequence[Sequence[Sequence[int]]],
    frame_weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the coherent ``A1^cells`` and pair-``B1`` singlet channels.

    The response is the sum over exactly two local ``B1`` substitutions in
    every ordered frame.  The full spatial average makes both returned
    channels ``Gamma-A1``; every local factor is a four-spin singlet, so both
    channels are also exact total-spin singlets.
    """

    checked = _checked_frames(frames)
    weights = np.asarray(frame_weights, dtype=np.float64)
    if weights.shape != (len(checked),) or np.any(weights < 0.0):
        raise ValueError("one nonnegative weight is required per ordered frame")
    if not np.isfinite(weights).all() or np.sum(weights) <= 0.0:
        raise ValueError("frame weights must have positive finite sum")
    weights = weights / np.sum(weights)
    spins = np.asarray(configurations, dtype=np.int8)
    n_sites = sum(len(cell) for cell in checked[0])
    if spins.ndim != 2 or spins.shape[1] != n_sites:
        raise ValueError(f"configurations must have shape (batch,{n_sites})")
    local = np.asarray(square_plaquette_singlet_carrier().vectors)
    a1 = local[:, 0]
    b1 = local[:, 1]
    bit_weights = np.asarray((8, 4, 2, 1), dtype=np.int64)
    base = np.zeros((len(spins),), dtype=np.complex128)
    response = np.zeros_like(base)
    for frame_weight, frame in zip(weights, checked, strict=True):
        coefficient0 = np.ones((len(spins),), dtype=np.complex128)
        coefficient1 = np.zeros_like(coefficient0)
        coefficient2 = np.zeros_like(coefficient0)
        for cell in frame:
            indices = spins[:, np.asarray(cell, dtype=np.int64)] @ bit_weights
            local_a = a1[indices]
            local_b = b1[indices]
            coefficient2 = coefficient2 * local_a + coefficient1 * local_b
            coefficient1 = coefficient1 * local_a + coefficient0 * local_b
            coefficient0 = coefficient0 * local_a
        base += frame_weight * coefficient0
        response += frame_weight * coefficient2
    return base, response


def coherent_b1_count_channels(
    configurations: np.ndarray,
    frames: Sequence[Sequence[Sequence[int]]],
    frame_weights: Sequence[float],
) -> np.ndarray:
    """Evaluate all exact-singlet, spatially averaged local-B1 counts.

    For one frame the columns are the coefficients of

    ``prod_cell (A_cell + z B_cell)``.

    Averaging the complete ordered frame orbit makes every coefficient a
    ``Gamma-A1`` state.  The construction includes the old coherent carrier
    at ``k=0`` and the previously tested pair-``B1`` response at ``k=2``.
    """

    checked = _checked_frames(frames)
    weights = np.asarray(frame_weights, dtype=np.float64)
    if weights.shape != (len(checked),) or np.any(weights < 0.0):
        raise ValueError("one nonnegative weight is required per ordered frame")
    if not np.isfinite(weights).all() or np.sum(weights) <= 0.0:
        raise ValueError("frame weights must have positive finite sum")
    weights = weights / np.sum(weights)
    spins = np.asarray(configurations, dtype=np.int8)
    n_sites = sum(len(cell) for cell in checked[0])
    if spins.ndim != 2 or spins.shape[1] != n_sites:
        raise ValueError(f"configurations must have shape (batch,{n_sites})")
    if np.any((spins != 0) & (spins != 1)):
        raise ValueError("configurations must be binary")
    local = np.asarray(square_plaquette_singlet_carrier().vectors)
    a1 = local[:, 0]
    b1 = local[:, 1]
    bit_weights = np.asarray((8, 4, 2, 1), dtype=np.int64)
    n_cells = len(checked[0])
    result = np.zeros((len(spins), n_cells + 1), dtype=np.complex128)
    for frame_weight, frame in zip(weights, checked, strict=True):
        coefficients = np.zeros_like(result)
        coefficients[:, 0] = 1.0
        degree = 0
        for cell in frame:
            indices = spins[:, np.asarray(cell, dtype=np.int64)] @ bit_weights
            local_a = a1[indices]
            local_b = b1[indices]
            updated = np.zeros_like(coefficients)
            updated[:, : degree + 1] += (
                coefficients[:, : degree + 1] * local_a[:, None]
            )
            updated[:, 1 : degree + 2] += (
                coefficients[:, : degree + 1] * local_b[:, None]
            )
            coefficients = updated
            degree += 1
        result += frame_weight * coefficients
    return result


def build_linear_resonance_sampled_action(
    batch: CoherentFrameSampleBatch,
    ordered_frames: Sequence[Sequence[Sequence[int]]],
    frame_weights: Sequence[float],
    bonds: Sequence[tuple[int, int, float]],
) -> LinearResonanceSampledAction:
    """Apply a Heisenberg graph to both exact-singlet carrier channels."""

    frames = _checked_frames(ordered_frames)
    configurations = np.asarray(batch.configurations, dtype=np.int8)
    base, response = coherent_pair_resonance_channels(
        configurations, frames, frame_weights
    )
    h_base = np.zeros_like(base)
    h_response = np.zeros_like(response)
    checked_bonds = tuple(
        (int(first), int(second), float(coupling))
        for first, second, coupling in bonds
        if coupling
    )
    for first, second, coupling in checked_bonds:
        unequal = configurations[:, first] != configurations[:, second]
        diagonal = np.where(unequal, -0.25, 0.25)
        h_base += coupling * diagonal * base
        h_response += coupling * diagonal * response
        if np.any(unequal):
            swapped = configurations[unequal].copy()
            temporary = swapped[:, first].copy()
            swapped[:, first] = swapped[:, second]
            swapped[:, second] = temporary
            swapped_base, swapped_response = coherent_pair_resonance_channels(
                swapped, frames, frame_weights
            )
            h_base[unequal] += 0.5 * coupling * swapped_base
            h_response[unequal] += 0.5 * coupling * swapped_response
    return LinearResonanceSampledAction(
        proposal_probability=np.asarray(batch.proposal_probability),
        base_amplitude=base,
        response_amplitude=response,
        h_base_amplitude=h_base,
        h_response_amplitude=h_response,
        n_sites=int(configurations.shape[1]),
        bond_count=len(checked_bonds),
        full_h_configuration_calls=len(configurations),
    )


def build_resonance_bank_sampled_action(
    batch: CoherentFrameSampleBatch,
    ordered_frames: Sequence[Sequence[Sequence[int]]],
    frame_weights: Sequence[float],
    bonds: Sequence[tuple[int, int, float]],
) -> ResonanceBankSampledAction:
    """Apply a Heisenberg graph to the complete local-B1-count bank."""

    frames = _checked_frames(ordered_frames)
    configurations = np.asarray(batch.configurations, dtype=np.int8)
    amplitudes = coherent_b1_count_channels(
        configurations, frames, frame_weights
    )
    h_amplitudes = np.zeros_like(amplitudes)
    checked_bonds = tuple(
        (int(first), int(second), float(coupling))
        for first, second, coupling in bonds
        if coupling
    )
    for first, second, coupling in checked_bonds:
        unequal = configurations[:, first] != configurations[:, second]
        diagonal = np.where(unequal, -0.25, 0.25)
        h_amplitudes += coupling * diagonal[:, None] * amplitudes
        if np.any(unequal):
            swapped = configurations[unequal].copy()
            temporary = swapped[:, first].copy()
            swapped[:, first] = swapped[:, second]
            swapped[:, second] = temporary
            swapped_amplitudes = coherent_b1_count_channels(
                swapped, frames, frame_weights
            )
            h_amplitudes[unequal] += 0.5 * coupling * swapped_amplitudes
    return ResonanceBankSampledAction(
        proposal_probability=np.asarray(batch.proposal_probability),
        amplitudes=amplitudes,
        h_amplitudes=h_amplitudes,
        n_sites=int(configurations.shape[1]),
        bond_count=len(checked_bonds),
        full_h_configuration_calls=len(configurations),
    )


def resonance_bank_energy_variance(
    sampled: ResonanceBankSampledAction,
    coefficients: Sequence[complex] | np.ndarray,
) -> dict[str, float]:
    """Return self-normalized estimates for one fixed bank coefficient vector."""

    values = np.asarray(coefficients, dtype=np.complex128)
    if values.shape != (sampled.amplitudes.shape[1],):
        raise ValueError("coefficients must contain one value per bank column")
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be finite")
    amplitude = sampled.amplitudes @ values
    action = sampled.h_amplitudes @ values
    proposal = sampled.proposal_probability
    norm_terms = np.abs(amplitude) ** 2 / proposal
    norm = float(np.sum(norm_terms))
    if norm <= 0.0:
        raise RuntimeError("resonance-bank state has zero sampled norm")
    energy = complex(np.sum(np.conj(amplitude) * action / proposal) / norm)
    if abs(energy.imag) > 2.0e-9 * max(abs(energy.real), 1.0):
        raise RuntimeError("sampled resonance-bank energy developed an imaginary part")
    residual = action - energy.real * amplitude
    variance = float(np.sum(np.abs(residual) ** 2 / proposal) / norm)
    ess = float(np.sum(norm_terms) ** 2 / np.sum(norm_terms**2))
    return {
        "energy_total": float(energy.real),
        "energy_per_site": float(energy.real / sampled.n_sites),
        "variance_total": variance,
        "variance_per_site": variance / sampled.n_sites,
        "importance_effective_sample_fraction": ess / len(amplitude),
    }


def linear_resonance_energy_variance(
    sampled: LinearResonanceSampledAction,
    coefficient: float,
) -> dict[str, float]:
    """Return self-normalized importance estimates for one coefficient."""

    value = float(coefficient)
    if not np.isfinite(value):
        raise ValueError("coefficient must be finite")
    amplitude = sampled.base_amplitude + value * sampled.response_amplitude
    action = sampled.h_base_amplitude + value * sampled.h_response_amplitude
    proposal = sampled.proposal_probability
    norm_terms = np.abs(amplitude) ** 2 / proposal
    norm = float(np.sum(norm_terms))
    if norm <= 0.0:
        raise RuntimeError("linear resonance state has zero sampled norm")
    energy_terms = np.conj(amplitude) * action / proposal
    energy = complex(np.sum(energy_terms) / norm)
    if abs(energy.imag) > 2.0e-9 * max(abs(energy.real), 1.0):
        raise RuntimeError("sampled resonance energy developed an imaginary part")
    residual = action - energy.real * amplitude
    variance = float(np.sum(np.abs(residual) ** 2 / proposal) / norm)
    ess = float(np.sum(norm_terms) ** 2 / np.sum(norm_terms**2))
    return {
        "energy_total": float(energy.real),
        "energy_per_site": float(energy.real / sampled.n_sites),
        "variance_total": variance,
        "variance_per_site": variance / sampled.n_sites,
        "importance_effective_sample_fraction": ess / len(amplitude),
    }


def _weighted_pair_expectations(
    batch: CoherentFrameSampleBatch,
    frames: tuple[Frame, ...],
    pairs: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], float]:
    configurations = np.asarray(batch.configurations, dtype=np.int8)
    center = np.asarray(batch.coherent_amplitude, dtype=np.complex128)
    proposal = np.asarray(batch.proposal_probability, dtype=np.float64)
    weights = np.asarray(batch.importance_weight, dtype=np.float64)
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise RuntimeError("coherent target has zero sampled norm")
    result: dict[tuple[int, int], float] = {}
    for first, second in pairs:
        unequal = configurations[:, first] != configurations[:, second]
        diagonal = np.where(unequal, -0.25, 0.25).astype(np.float64)
        # Evaluate the off-diagonal estimator in the stable bilinear form
        # |psi|^2 * psi(swapped)/psi = conj(psi)*psi(swapped).
        offdiag_numerator = np.zeros((len(configurations),), dtype=np.complex128)
        if np.any(unequal):
            swapped = configurations[unequal].copy()
            temporary = swapped[:, first].copy()
            swapped[:, first] = swapped[:, second]
            swapped[:, second] = temporary
            swapped_amplitude = _coherent_amplitudes(swapped, frames)
            offdiag_numerator[unequal] = (
                np.conj(center[unequal]) * swapped_amplitude / proposal[unequal]
            )
        numerator = np.sum(weights * diagonal) + 0.5 * np.sum(offdiag_numerator)
        if abs(numerator.imag) > 2.0e-9 * max(abs(numerator.real), 1.0):
            raise RuntimeError("pair-spin estimator developed an imaginary expectation")
        result[(first, second)] = float(numerator.real / denominator)
    return result


def intercell_casimir_support(
    batch: CoherentFrameSampleBatch,
    frames: Sequence[Sequence[Sequence[int]]],
    layout: ManyCellSchurLayout,
) -> dict[str, object]:
    """Measure non-root inter-cell ``q>0`` support through block Casimirs."""

    checked = _checked_frames(frames)
    if layout.n_sites != batch.configurations.shape[1]:
        raise ValueError("layout and sample batch sizes disagree")
    blocks = []
    for node_index in layout.intercell_indices:
        model_sites = leaf_signature(layout.tree.internal_nodes[node_index])
        physical_sites = tuple(sorted(layout.site_order[index] for index in model_sites))
        if len(physical_sites) == layout.n_sites:
            continue
        blocks.append(physical_sites)
    if not blocks:
        raise ValueError("layout has no non-root inter-cell blocks")
    pairs = tuple(
        sorted(
            {
                (left, right)
                for block in blocks
                for offset, left in enumerate(block)
                for right in block[offset + 1 :]
            }
        )
    )
    pair_values = _weighted_pair_expectations(batch, checked, pairs)
    rows = []
    for block in blocks:
        value = 0.75 * len(block)
        value += 2.0 * sum(
            pair_values[(left, right)]
            for offset, left in enumerate(block)
            for right in block[offset + 1 :]
        )
        maximum_spin = 0.5 * len(block)
        maximum_casimir = maximum_spin * (maximum_spin + 1.0)
        rows.append(
            {
                "site_count": len(block),
                "sites": list(block),
                "casimir": float(value),
                "casimir_fraction_of_maximum": float(value / maximum_casimir),
            }
        )
    return {
        "nonroot_intercell_block_count": len(rows),
        "unique_pair_estimator_count": len(pairs),
        "blocks": rows,
        "mean_intercell_casimir": float(np.mean([row["casimir"] for row in rows])),
        "mean_intercell_casimir_fraction_of_maximum": float(
            np.mean([row["casimir_fraction_of_maximum"] for row in rows])
        ),
        "interpretation": (
            "Each block Casimir is zero on q=0 and positive on every q>0 irrep; "
            "a positive expectation is therefore a direct witness of nonzero "
            "inter-cell q support, not an estimate of its probability."
        ),
    }


__all__ = [
    "CoherentFrameSampleBatch",
    "LinearResonanceSampledAction",
    "ResonanceBankSampledAction",
    "build_linear_resonance_sampled_action",
    "build_resonance_bank_sampled_action",
    "coherent_b1_count_channels",
    "coherent_pair_resonance_channels",
    "importance_diagnostics",
    "intercell_casimir_support",
    "linear_resonance_energy_variance",
    "ordered_frame_orbit_with_weights",
    "plaquette_frame_component_amplitudes",
    "resonance_bank_energy_variance",
    "sample_coherent_plaquette_frames",
]
