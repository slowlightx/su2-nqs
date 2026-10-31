"""Bounded complete-H cost of the frozen physical S=0,X.A bond source.

The source is fixed, so complete Hsource can be cached per future training
configuration. This reference deduplicates physical H connections and streams
source amplitudes; it does not replace Hsource by a ground-state energy times
source and does not report a one-configuration local energy as a gap.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from numbers import Integral
from pathlib import Path
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.models.triangular_sector_amplitude import triangular_sector_projector
from su2_nqs.vmc.checkpoint import load_checkpoint
from .ct_physical_spin_seed_20261029 import PARENTS
from .ct_scalar_X_seed_20261030 import make_coherent_scalar_source, projected_scalar_bonds
from ..workflows.ct6_native_direction_calibration import normalized_raw_bank
from ..workflows.ct_sector_excitation_pilot import deadline


def full_H_connections(bits: np.ndarray, bonds: Sequence[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Return rows y and coefficients H[x,y], including combined repeated bonds.

    Bits are down/up, site zero MSB. Every generated row preserves doubled
    magnetization. Equal-spin swaps contribute only their diagonal term.
    """
    row = np.asarray(bits)
    if (row.ndim != 1 or len(row) < 2 or row.dtype.kind not in 'iu'
            or not np.all((row == 0) | (row == 1)) or not bonds):
        raise ValueError('one legal integer-bit row and nonempty real Heisenberg bonds required')
    keys: dict[bytes, tuple[np.ndarray, float]] = {}
    original = row.astype(np.int32)
    keys[original.tobytes()] = (original, 0.)
    diagonal = 0.
    for a, b, coupling in bonds:
        if (any(isinstance(i, bool) or not isinstance(i, Integral) for i in (a, b))
                or not 0 <= a < len(row) or not 0 <= b < len(row) or a == b
                or not np.isfinite(coupling) or np.iscomplexobj(coupling)):
            raise ValueError('legal distinct integer endpoints and finite real couplings required')
        unlike = original[a] != original[b]
        diagonal += coupling*(-.25 if unlike else .25)
        if unlike:
            changed = original.copy()
            changed[a], changed[b] = changed[b], changed[a]
            key = changed.tobytes()
            keys[key] = (changed, keys.get(key, (changed, 0.))[1]+.5*coupling)
    keys[original.tobytes()] = (original, diagonal)
    return np.stack([value[0] for value in keys.values()]), np.asarray([value[1] for value in keys.values()])


def stream_full_H(
    evaluate: Callable[[np.ndarray], np.ndarray], rows: np.ndarray,
    coefficients: np.ndarray, *, batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate all distinct physical connections, retaining zero-amplitude rows.

    evaluate returns complex [batch,channels]. Padding is an execution detail
    only and is removed before summation. No target wavefunction division.
    """
    rows, coefficients = np.asarray(rows), np.asarray(coefficients)
    if (rows.ndim != 2 or len(rows) == 0 or coefficients.shape != (len(rows),)
            or not np.isfinite(coefficients).all() or isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral) or batch_size < 1):
        raise ValueError('matching finite H coefficients and positive batch size required')
    values = []
    requested = 0
    for first in range(0, len(rows), batch_size):
        chunk = rows[first:first+batch_size]
        length = len(chunk)
        if length < batch_size:
            chunk = np.concatenate((chunk, np.repeat(chunk[:1], batch_size-length, axis=0)))
        batch = np.asarray(evaluate(chunk))
        requested += len(chunk)
        if batch.ndim != 2 or len(batch) != batch_size or not np.isfinite(batch).all():
            raise ArithmeticError('finite amplitude channels required at every physical H connection')
        values.append(batch[:length])
    values = np.concatenate(values)
    return values[0], coefficients@values, dict(distinct_physical_rows=len(rows),
        executed_rows_including_padding=requested, node_safe=True, wavefunction_division=False)


def run(output: Path) -> dict:
    """One fixed original row: complete source H and measured cost, not energy."""
    if output.exists():
        raise FileExistsError('new immutable directory required')
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    seed_dir = root/'projects/clebsch_tree_nqs/results/ct_scalar_X_seed_20261030/run_v2'
    old = json.loads((seed_dir/'results.json').read_text())
    source_path = PARENTS/'ct6_j0125'/'checkpoint.pkl'
    saved = load_checkpoint(source_path)
    if (file_sha256(source_path) != old['parent_checkpoint_sha256']
            or pytree_sha256(saved['params']) != old['parent_parameter_sha256']):
        raise ValueError('frozen singlet parent differs')
    with np.load(seed_dir/'amplitudes.npz') as panel:
        bits = panel['rows'][0]
        reference = panel['parent_and_source'][0]
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    rows, coefficients = full_H_connections(bits, bonds)
    registration = dict(task='frozen_CT6_scalar_X_A_complete_H_single_center_cost',
        j2=.125, target_q=0, target_m2=0, momentum=[[-1, 6], [-1, 6]],
        parent_step=500, parent_parameter_sha256=old['parent_parameter_sha256'],
        parent_checkpoint_sha256=old['parent_checkpoint_sha256'],
        diagnostic_center='first fixed uniform-M0 row from the frozen amplitude probe',
        diagnostic_row_not_selected_by_energy=True, full_H_bonds=len(bonds),
        wall_cap_seconds=180, batch_size=4, no_training_or_random_draws=True,
        no_energy_or_gap_estimator=True, source_code_sha256=file_sha256(Path(__file__)))
    atomic_json(output/'registration.json', registration)
    started = perf_counter()
    try:
        with deadline(180):
            cfg = MixedClebschConfig(6, 28, 10)
            bank, _ = normalized_raw_bank(saved['params'], dict(cfg=cfg,
                bank=lambda p: arity_clebsch_bank(p, cfg, 'binary_truncated', 8)))
            prepared = prepare_feature_lut_bank(bank, cfg)
            projector = triangular_sector_projector(cfg, ((-1, 6), (-1, 6)))
            source_bonds = projected_scalar_bonds(projector, ((0, 1, 6.),))
            images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
            evaluate, metadata = make_coherent_scalar_source(
                lambda edges: make_feature_lut_exchange_queries(cfg, edges), images, source_bonds)
            compiled = lambda x: np.asarray(evaluate(prepared, jnp.asarray(x)))
            tick = perf_counter()
            psi, hpsi, work = stream_full_H(compiled, rows, coefficients, batch_size=4)
            seconds = perf_counter()-tick
            np.testing.assert_allclose(psi, reference, rtol=2e-11, atol=1e-18)
            np.savez_compressed(output/'complete_H_products.npz', center=bits,
                physical_H_rows=rows, physical_H_coefficients=coefficients,
                psi=psi, hpsi=hpsi)
            report = dict(registration, status='complete_source_H_cost_measured',
                query_metadata=metadata, work=work, H_seconds_including_compilation=seconds,
                wall_seconds=perf_counter()-started, peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,
                amplitude_replay_maximum_difference=float(np.max(abs(psi-reference))),
                full_H_source_nonzero=bool(abs(hpsi[1]) > 0),
                scalar_center_count=1, no_statistical_energy_or_lowest_gap_claim=True)
            atomic_json(output/'results.json', report)
            return report
    except Exception as error:
        atomic_json(output/'partial.json', dict(registration, status='bounded_incomplete',
            wall_seconds=perf_counter()-started, error=repr(error)))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == '__main__':
    main()
