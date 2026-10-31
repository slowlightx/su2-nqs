"""Opt-in scalar-bond physical seed with an explicit momentum/little irrep.

An actual coherent Gamma-A1 singlet is acted on by a symmetry-projected
scalar bond. This preserves doubled total q=0. The source is not assumed to
be a low-energy eigenstate, a monopole, or a converged X excitation. No
production defaults, tensor truncation, or ground-state checkpoint is changed.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from numbers import Integral
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.models.triangular_sector_amplitude import triangular_sector_projector
from su2_nqs.vmc.checkpoint import load_checkpoint
from .ct_physical_spin_seed_20261029 import PARENTS
from ..workflows.ct6_native_direction_calibration import normalized_raw_bank
from ..workflows.ct_sector_excitation_pilot import deadline


def projected_scalar_bonds(
    projector: PhysicalSpatialProjector,
    seed_bonds: Sequence[tuple[int, int, complex]],
) -> tuple[tuple[int, int, complex], ...]:
    """Return B such that B|Gamma-A1> = P_character B_seed|Gamma-A1>.

    B is a sum of S_i.S_j, hence an SU(2) rank-zero operator at every size.
    The site's pullback convention is exactly that of PhysicalSpatialProjector.
    Complex coefficients are not made real; cancellations are not repaired.
    """
    if projector.fermionic or not seed_bonds:
        raise ValueError("a spin projector and nonempty scalar-bond seed required")
    combined: dict[tuple[int, int], complex] = {}
    for a, b, weight in seed_bonds:
        if (any(isinstance(i, bool) or not isinstance(i, Integral) for i in (a, b))
                or not 0 <= a < projector.n_sites or not 0 <= b < projector.n_sites
                or a == b or not np.isfinite(weight)):
            raise ValueError("distinct integer physical sites and finite coefficients required")
        for image, character in zip(projector.site_permutations, projector.characters, strict=True):
            edge = tuple(sorted((image[a], image[b])))
            combined[edge] = combined.get(edge, 0j) + np.conj(character)*weight/projector.group_order
    return tuple((a, b, complex(weight)) for (a, b), weight in sorted(combined.items()))


def source_covariance_residual(source: np.ndarray, transformed: np.ndarray,
                               character: complex) -> float:
    """Reject nonfinite or null sources before a relative character check."""
    original, changed = np.asarray(source), np.asarray(transformed)
    if (original.ndim != 1 or original.shape != changed.shape or not len(original)
            or not np.isfinite(original).all() or not np.isfinite(changed).all()
            or not np.isfinite(character) or abs(abs(character)-1) > 1e-12):
        raise ArithmeticError('finite matching source vectors and a unit character required')
    norm = float(np.linalg.norm(original))
    if not np.isfinite(norm) or norm <= 0:
        raise ArithmeticError('a null source cannot validate a symmetry sector')
    residual = float(np.linalg.norm(changed-character*original)/norm)
    if not np.isfinite(residual):
        raise ArithmeticError('nonfinite relative character residual')
    return residual


def make_coherent_scalar_source(
    query_factory: Callable,
    parent_images: np.ndarray,
    source_bonds: Sequence[tuple[int, int, complex]],
    *,
    projection_chunk: int = 4,
) -> tuple[Callable, dict]:
    """Return (parent, B parent) amplitudes using shared exact raw exchanges.

    This computes B psi, not H B psi. It returns amplitudes without ratios,
    nodes, norm floors or discarded magnetic components. Parent images must
    specify the actual coherent Gamma-A1 carrier, not an orbit probability.
    """
    maps = np.asarray(parent_images)
    if (maps.ndim != 2 or not len(maps) or maps.dtype.kind not in 'iu'
            or not np.all(np.sort(maps, axis=1) == np.arange(maps.shape[1]))
            or isinstance(projection_chunk, bool) or not isinstance(projection_chunk, Integral)
            or projection_chunk < 1 or len(maps) % projection_chunk or not source_bonds):
        raise ValueError("valid physical image rows and a positive dividing chunk required")
    n_sites = maps.shape[1]
    combined: dict[tuple[int, int], complex] = {}
    for a, b, weight in source_bonds:
        if (any(isinstance(i, bool) or not isinstance(i, Integral) for i in (a, b))
                or not 0 <= a < n_sites or not 0 <= b < n_sites or a == b
                or not np.isfinite(weight)):
            raise ValueError("finite scalar-bond coefficients on legal integer edges required")
        edge = tuple(sorted((int(a), int(b))))
        combined[edge] = combined.get(edge, 0j) + weight
    edges = np.asarray(sorted(combined), np.int32)
    coefficients = jnp.asarray([combined[tuple(edge)] for edge in edges], jnp.complex128)
    inverse = np.argsort(maps, axis=1)
    query_edges, reverse = np.unique(np.sort(inverse[:, edges], axis=-1).reshape(-1, 2),
                                     axis=0, return_inverse=True)
    reindex = reverse.reshape(len(maps), len(edges))
    query = query_factory(query_edges.astype(np.int32))
    groups = jnp.asarray(maps.reshape(-1, projection_chunk, n_sites))
    indices = jnp.asarray(reindex.reshape(-1, projection_chunk, len(edges)))

    def one(bank, bits):
        bits = jnp.asarray(bits, jnp.int32)

        def image_value(image, index):
            center, changed = query(bank, bits[image])
            return jnp.concatenate((jnp.asarray(center)[None], changed[index]))

        def partial(items):
            image, index = items
            return jnp.sum(jax.vmap(image_value)(image, index), axis=0)

        coherent = jnp.sum(jax.lax.map(jax.checkpoint(partial), (groups, indices)), axis=0)/len(maps)
        parent, swapped = coherent[0], coherent[1:]
        # For spin half S_i.S_j = P_ij/2 - I/4, including equal-spin rows.
        source = jnp.dot(coefficients, .5*swapped-.25*parent)
        return jnp.stack((parent, source))

    evaluate = jax.jit(lambda bank, rows: jax.lax.map(lambda x: one(bank, x), rows))
    return evaluate, dict(parent_images=len(maps), physical_source_bonds=len(edges),
                         raw_exchange_queries=len(query_edges), target_q=0, target_m2=0,
                         strict_spin_requires_actual_singlet_parent=True,
                         full_H_source_implemented=False, no_wavefunction_division=True)


def run(output: Path) -> dict:
    """Measure an actual CT6 X.A seed amplitude, not its energy or gap."""
    if output.exists():
        raise FileExistsError("immutable new output directory required")
    output.mkdir(parents=True)
    source = PARENTS/'ct6_j0125'/'checkpoint.pkl'
    parent = load_checkpoint(source)
    binding = json.loads((PARENTS/'independent_review.json').read_text())['runs']['ct6_j0125']
    if (file_sha256(source) != binding['checkpoint_sha256']
            or pytree_sha256(parent['params']) != binding['parameter_sha256']):
        raise ValueError('audited singlet parent identity differs')
    cfg = MixedClebschConfig(6, 28, 10)
    x = ((-1, 6), (-1, 6))
    projector = triangular_sector_projector(cfg, x)
    bonds = projected_scalar_bonds(projector, ((0, 1, 6.),))
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    registration = dict(task='actual_CT6_scalar_X_A_physical_seed_amplitude_only', j2=.125,
        target_q=0, target_m2=0, momentum=[[-1, 6], [-1, 6]], point_characters='all +1',
        little_projector_images=projector.group_order, parent_images=len(images),
        parent_global_step=binding['absolute_step'], parent_parameter_sha256=pytree_sha256(parent['params']),
        parent_checkpoint_sha256=file_sha256(source), seed=3930302001,
        rows=4, wall_cap_seconds=120, energy_or_gap_not_evaluated=True,
        no_training_or_remote_changes=True)
    atomic_json(output/'registration.json', registration)
    started = perf_counter()
    with deadline(120):
        bank, _ = normalized_raw_bank(parent['params'], dict(cfg=cfg,
            bank=lambda p: arity_clebsch_bank(p, cfg, 'binary_truncated', 8)))
        prepared = prepare_feature_lut_bank(bank, cfg)
        evaluate, metadata = make_coherent_scalar_source(
            lambda edges: make_feature_lut_exchange_queries(cfg, edges), images, bonds)
        rng = np.random.default_rng(registration['seed'])
        rows = np.asarray([rng.permutation(np.r_[np.zeros(18, np.int32), np.ones(18, np.int32)])
                           for _ in range(4)])
        # Fixed fresh rows only diagnose amplitudes and symmetry, not Born moments.
        tick = perf_counter(); values = np.asarray(evaluate(prepared, jnp.asarray(rows)))
        first_seconds = perf_counter()-tick
        source_covariance_residual(values[:, 1], values[:, 1], 1.)
        if not np.isfinite(values).all():
            raise ArithmeticError('nonfinite parent or source amplitude')
        tick = perf_counter(); replay = np.asarray(evaluate(prepared, jnp.asarray(rows)))
        warm_seconds = perf_counter()-tick
        tests = []
        # Check a generating set rather than paying all 72 image rows again.
        from su2_nqs.lattices.triangular_hnf import hnf_site_permutation
        from su2_nqs.lattices.spatial_symmetry import triangular_hnf_reciprocal_little_group
        little = triangular_hnf_reciprocal_little_group(((6, 0), (0, 6)), (30, 30))
        point_names = set(little)
        by_image = dict(zip(projector.site_permutations, projector.characters, strict=True))
        generators = [hnf_site_permutation(((6, 0), (0, 6)), translation=t)
                      for t in ((1, 0), (0, 1))]
        # Include a nontranslation representative for every little point operation.
        from su2_nqs.lattices.triangular_hnf import POINT_GROUP
        operations = dict(POINT_GROUP)
        for name in point_names:
            generators.append(hnf_site_permutation(((6, 0), (0, 6)), operation=operations[name]))
        for image in generators:
            changed = np.asarray(evaluate(prepared, jnp.asarray(rows[:, np.asarray(image)])))
            relative = source_covariance_residual(values[:, 1], changed[:, 1], by_image[tuple(image)])
            tests.append(dict(character=[by_image[tuple(image)].real, by_image[tuple(image)].imag],
                              relative_source_residual=relative))
            if relative > 2e-10:
                raise ArithmeticError('actual scalar X seed failed its supplied character')
        np.savez_compressed(output/'amplitudes.npz', rows=rows, parent_and_source=values,
                            bonds=np.asarray(bonds), repeat=replay)
        report = dict(registration, query_metadata=metadata, first_seconds=first_seconds,
            warm_batch_seconds=warm_seconds, warm_seconds_per_center=warm_seconds/len(rows),
            scalar_amplitude_nonzero_rows=int(np.count_nonzero(abs(values[:, 1]) > 1e-18)),
            sampled_source_norm_not_a_normalization=float(np.linalg.norm(values[:, 1])),
            replay_maximum_absolute_difference=float(np.max(abs(values-replay))),
            symmetry_checks=tests, wall_seconds=perf_counter()-started,
            no_H_or_optimized_S0_X_state_claim=True)
        atomic_json(output/'results.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == '__main__':
    main()
