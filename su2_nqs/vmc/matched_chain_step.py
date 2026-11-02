"""Bounded scalar search using the worst ACTUAL FS distance across regions."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import numpy as np

from .carrier_protection import physical_distance
from .matched_state_step import matched_state_step


@dataclass(frozen=True)
class MatchedChainStep:
    """One cached candidate; all region distances and every search record."""
    alpha: float
    amplitudes: np.ndarray
    distances: dict
    search: dict
    records: tuple[dict, ...]

    def as_dict(self) -> dict:
        return dict(alpha=self.alpha, distances=self.distances, search=self.search, records=list(self.records),
                    matched=self.search['matched'], evaluation_count=self.search['evaluation_count'],
                    matching_metric='maximum actual fixed-Q FS over declared regions; not equal FS in every chain')


def matched_chain_state_step(candidate: Callable[[float], np.ndarray], old: np.ndarray,
                              q: np.ndarray, regions: Mapping[str, np.ndarray], target_radius: float,
                              max_alpha: float, bisection_steps: int, *, relative_tolerance: float = .01
                              ) -> MatchedChainStep:
    """Reuse fixed-budget scalar matching; never alter the parameter direction.

    A two-vector represents ONLY the scalar max-FS search coordinate. Report
    original physical amplitudes and every region's measured distances, not
    that auxiliary vector as a physical state. Regions must cover all records;
    include the full set as well to detect relative phase changes across chains.
    """
    parent, density = np.asarray(old, complex), np.asarray(q, float)
    physical_distance(parent, parent, density)
    masks = {name: np.asarray(mask) for name, mask in regions.items()}
    if (not masks or any(not isinstance(name, str) or mask.dtype != bool or mask.shape != parent.shape
                         or np.sum(mask) < 2 for name, mask in masks.items())
            or not any(np.all(mask) for mask in masks.values())):
        raise ValueError('named boolean nonempty regions, each >=2 records, including full support required')
    parents = {name: physical_distance(parent[mask], parent[mask], density[mask]) for name, mask in masks.items()}
    cache = {0.: (parent.copy(), parents)}
    records = []

    def coordinate(alpha):
        values = np.asarray(candidate(alpha), complex)
        if values.shape != parent.shape:
            raise ValueError('candidate changed physical amplitude shape')
        if not np.isfinite(values).all() or not np.any(values):
            records.append(dict(alpha=alpha, valid=False, distances=None))
            return np.array([np.nan, 0.])
        try:
            distances = {name: physical_distance(parent[mask], values[mask], density[mask])
                         for name, mask in masks.items()}
        except (ValueError, FloatingPointError):
            records.append(dict(alpha=alpha, valid=False, distances=None))
            return np.array([np.nan, 0.])
        maximum = max(d['fs'] for d in distances.values())
        cache[alpha] = (values.copy(), distances)
        records.append(dict(alpha=alpha, valid=True, distances=distances, maximum_fs=maximum))
        return np.array([np.cos(maximum), np.sin(maximum)])

    answer = matched_state_step(coordinate, np.array([1., 0.]), np.ones(2),
        target_radius, max_alpha, bisection_steps, relative_tolerance=relative_tolerance)
    values, distances = cache[answer.alpha]
    return MatchedChainStep(answer.alpha, values, distances, answer.as_dict(), tuple(records))


__all__ = ['MatchedChainStep', 'matched_chain_state_step']
