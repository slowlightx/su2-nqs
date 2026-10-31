"""Fresh complete-A1 amplitude/full-H cost probes for a scaled active tree.

No Born records, optimization or energy/variance estimate are made. The
stored raw replay configuration is only a deterministic kernel-cost probe.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from time import perf_counter, process_time

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action
from .triangle_hierarchy_initialization import _node_timeout


def run(directory: Path, *, seconds_cap: float = 300.) -> dict:
    """Two amplitude and two full-H centers; stop at the declared wall gate."""
    if not 30 <= seconds_cap <= 300:
        raise ValueError("bounded cost gate requires 30..300 seconds")
    output = directory/"complete_A1_cost"
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("fresh isolated cost output required")
    checkpoint = directory/"active_bank.pkl"
    payload = load_checkpoint(checkpoint); bank = payload["params"]; metadata = payload["metadata"]
    cfg = MixedClebschConfig(**metadata["model"])
    profile = metadata["active_top_widths"]
    row = np.load(directory/"raw_decoder_replay.npz")["configurations"][0]
    registration = dict(task="scaled_active_tree_complete_A1_cost_only_v1",
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_sha256(checkpoint),
        replay_sha256=file_sha256(directory/"raw_decoder_replay.npz"),
        workflow_sha256=file_sha256(Path(__file__)), length=cfg.length,
        maximum_wall_seconds=seconds_cap, projected_amplitude_centers_planned=2,
        complete_physical_H_centers_planned=2, projection_chunk=4,
        no_MC_or_optimization=True, no_energy_or_variance_estimate=True,
        fixed_active_layout_not_native_channel_opening=True)
    atomic_json(output/"registration.json", registration)
    shutil.copy2(__file__, output/"registered_workflow.py")
    started = perf_counter(); cpu = process_time()
    result = dict(registration=registration, timings={}, projected_amplitude_centers=0,
                  complete_physical_H_centers=0, status="running")
    def timed(label, callback):
        tick = perf_counter(); value = callback()
        value = jax.tree.map(lambda x: np.asarray(x), value)
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(value)):
            raise FloatingPointError("nonfinite projected cost-probe output")
        result["timings"][label] = perf_counter()-tick
        atomic_json(output/"progress.json", result)
        return value
    try:
        with _node_timeout(seconds_cap):
            images = timed("complete_group_construction", lambda: np.asarray(
                triangular_hnf_gamma_a1_group(((cfg.length, 0), (0, cfg.length))), np.int32))
            bonds = triangular_j1j2_bonds(cfg.length, cfg.length, j2=.125, periodic=True)
            result.update(group_images=len(images), target_physical_bonds=len(bonds))
            prepared = prepare_feature_lut_bank(bank, cfg)
            amplitude = jax.jit(make_feature_lut_projected_amplitude(cfg, images,
                chunk=4, active_top_widths=profile))
            first = float(timed("complete_A1_amplitude_cold", lambda: amplitude(prepared, row)))
            result["projected_amplitude_centers"] += 1
            second = float(timed("complete_A1_amplitude_warm", lambda: amplitude(prepared, row)))
            result["projected_amplitude_centers"] += 1
            result["A1_amplitude"] = second
            if first != second:
                raise ArithmeticError("cold/warm A1 amplitude mismatch")
            tick = perf_counter()
            action = make_cached_projected_heisenberg_action(cfg, images, bonds,
                projection_chunk=4, active_top_widths=profile)
            result["timings"]["full_H_factory_including_weighted_group_validation"] = perf_counter()-tick
            outputs = []
            for label in ("full_H_cold", "full_H_warm"):
                values = timed(label, lambda: action(prepared, row[None]))
                result["complete_physical_H_centers"] += 1
                outputs.append([float(x[0]) for x in values])
            error = max(abs(values[0]-second) for values in outputs)/max(abs(second), 1e-300)
            if error > 2e-10 or outputs[0] != outputs[1]:
                raise ArithmeticError("full-H center amplitude failed independent A1 replay")
            result.update(status="complete_cost_probe_only", action_psi_Hpsi=outputs,
                action_vs_independent_A1_relative_error=error,
                measured_full_H_seconds_per_center=result["timings"]["full_H_warm"],
                forecast_256_full_H_centers_seconds=256*result["timings"]["full_H_warm"],
                forecast_excludes_sampling_and_compile=True)
    except (TimeoutError, MemoryError) as error:
        result.update(status="bounded_incomplete_cost_probe", interruption=str(error),
                      no_uncompleted_cost_extrapolated_as_measured=True)
    finally:
        result.update(wall_seconds=perf_counter()-started, CPU_seconds=process_time()-cpu,
            raw_initialization_energy_is_not_A1_energy=True, physical_variance_known=False,
            zero_Born_samples=True)
        atomic_json(output/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seconds-cap", type=float, default=300.)
    args = parser.parse_args()
    run(args.directory, seconds_cap=args.seconds_cap)
