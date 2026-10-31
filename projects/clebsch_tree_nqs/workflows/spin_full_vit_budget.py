"""Reusable bounded full-ViT capacity benchmark, with cached ED scalar anchors.

The existing factored ViT, orbit projector, sparse Hamiltonian, spin audit,
and checkpoint helpers are reused. Coarse-translation representative reuse
is exact for this model and independently replayed before every training run.
No carrier or restricted residual is present; all network weights train.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.krylov_anchored_full_vit import (
    a1_orbit_reduction, a1_orbit_coordinates, a1_orbit_hamiltonian,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact, load_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit,
    fixed_sz_spatial_vit_amplitudes, fixed_sz_spatial_vit_parameter_count,
)
from su2_nqs.vmc.bounded_exact import train_bounded_exact
from .spin import _prepare_case
from .spin_spatial_vit import _spin_projection_diagnostics

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_full_vit_budget.json"
DEFAULT_OUTPUT = ROOT / "projects/clebsch_tree_nqs/results/spin_full_vit_budget_v1"


def prepare_objective(data: dict, config: FixedSzSpatialViTConfig) -> dict:
    """Build exact Gamma-A1 objective with coarse translation reuse, not ED."""
    groups = data["groups"]["translation"]
    selected = [i for i, p in enumerate(groups)
                if (p[0] // config.columns) % config.patch_rows == 0
                and (p[0] % config.columns) % config.patch_columns == 0]
    coarse = a1_orbit_reduction(data["tables"]["translation"][selected])
    _, representatives = np.unique(coarse.orbit_id, return_index=True)
    basis = jnp.asarray(data["basis"], dtype=jnp.int32)
    reduced_basis = basis[representatives]
    a1 = a1_orbit_reduction(data["tables"]["gamma_a1"])
    matrix = a1_orbit_hamiltonian(data["h_rows"], data["h_columns"], data["h_values"], a1)
    h = jnp.asarray(matrix)

    def state_fn(params):
        raw = fixed_sz_spatial_vit_amplitudes(params, reduced_basis, config)
        vector = a1_orbit_coordinates(raw[jnp.asarray(coarse.orbit_id)], a1)
        return vector / jnp.linalg.norm(vector)

    def loss(params):
        state = state_fn(params)
        return jnp.vdot(state, h @ state).real

    def physical(vector):
        return np.asarray(vector)[a1.orbit_id] / np.sqrt(a1.orbit_sizes[a1.orbit_id])

    def metrics(params):
        state = np.asarray(jax.jit(state_fn)(params))
        hs = matrix @ state
        energy = float(np.vdot(state, hs).real)
        return dict(energy_per_site=energy/config.n_sites,
                    energy_error_per_site=(energy-data["exact_energy"])/config.n_sites,
                    variance_per_site=float(np.linalg.norm(hs-energy*state)**2)/config.n_sites)

    def validate(params):
        raw = np.array(jax.jit(lambda p: fixed_sz_spatial_vit_amplitudes(p, basis, config))(params))
        raw /= np.linalg.norm(raw)
        coarse_raw = raw[representatives][coarse.orbit_id]
        expected = np.mean(raw[data["tables"]["gamma_a1"]], axis=0)
        expected /= np.linalg.norm(expected)
        got = physical(jax.jit(state_fn)(params))
        action = data["hamiltonian"] @ got
        errors = dict(coarse_translation_replay_max=float(np.max(abs(raw-coarse_raw))),
                      projected_state_replay_max=float(np.max(abs(expected-got))),
                      full_H_action_replay_max=float(np.max(abs(action-physical(matrix @ np.asarray(jax.jit(state_fn)(params)))))),
                      normalization_error=float(abs(np.vdot(got, got)-1)))
        if max(errors.values()) > 2e-10:
            raise RuntimeError(f"exact objective replay failed: {errors}")
        return errors

    return dict(state_fn=jax.jit(state_fn), loss=loss, physical=physical,
                metrics=metrics, validate=validate, a1_dimension=a1.reduced_dimension,
                raw_evaluations=len(representatives), a1=a1)


def run_trial(config_path: Path, output: Path, case_index: int, arm_index: int,
              seed_index: int, resume: bool = False) -> dict:
    """Run one registered capacity/seed; refuse overwriting completed artifacts."""
    spec = json.loads(config_path.read_text())
    if spec["task"] != "spin_full_vit_budget" or not spec["preregistered_before_run"]:
        raise ValueError("expected registered full-ViT budget")
    case, arm, seed = spec["cases"][case_index], spec["arms"][arm_index], spec["seeds"][seed_index]
    name = f'{case["case_id"]}_{arm["id"]}_s{seed}'
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/"results.json").exists():
        raise FileExistsError(destination/"results.json")
    started = perf_counter()
    data = _prepare_case(case, "fixed_m0")
    config = FixedSzSpatialViTConfig(geometry=case["geometry"], rows=case["rows"],
        columns=case["columns"], **spec["model"], **{k:v for k,v in arm.items() if k != "id"})
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(seed), config)
    initial_digest = pytree_sha256(params)
    problem = prepare_objective(data, config)
    checks = dict(initial=problem["validate"](params))
    preparation_seconds = perf_counter()-started
    print(name, "prepared", preparation_seconds, "seconds", flush=True)

    def callback(p, row):
        metrics = problem["metrics"](p)
        print(name, row["stage"], row["step"], metrics, flush=True)
        return metrics

    trained, training = train_bounded_exact(params, loss_fn=problem["loss"],
        identity=dict(config_sha256=canonical_sha256(spec), trial=name),
        checkpoint=destination/"checkpoint.pkl", callback=callback, resume=resume,
        **spec["optimization"])
    final = problem["metrics"](trained)
    checks["final"] = problem["validate"](trained)
    physical = problem["physical"](problem["state_fn"](trained))
    spin = _spin_projection_diagnostics(physical, data, raw_energy=final["energy_per_site"]*config.n_sites)
    artifact = destination/"parameters.json"
    save_parameter_artifact(artifact, trained, identity=dict(trial=name))
    loaded = load_parameter_artifact(artifact, params, expected_identity=dict(trial=name))
    checks["checkpoint_amplitude_replay_max"] = float(np.max(abs(np.asarray(problem["state_fn"](loaded))-np.asarray(problem["state_fn"](trained)))))
    result = dict(task=spec["task"], trial=name, config=spec, model_config=asdict(config),
        config_sha256=canonical_sha256(spec), seed=seed, arm=arm["id"], case=case,
        parameter_count=fixed_sz_spatial_vit_parameter_count(trained),
        initial_params_sha256=initial_digest, final_params_sha256=pytree_sha256(trained),
        all_parameters_trained=True, carrier=None, checks=checks, training=training,
        metrics=final, spin=spin, preparation_seconds=preparation_seconds,
        wall_seconds=perf_counter()-started, runtime=numerical_runtime_identity(),
        a1_dimension=problem["a1_dimension"], raw_evaluations_per_full_H=problem["raw_evaluations"],
        hamiltonian_diagonalization_performed=False, remote_submitted=False,
        source_manifest=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs")))
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination/"results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--arm", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_trial(args.config, args.output, args.case, args.arm, args.seed, args.resume)


if __name__ == "__main__":
    main()
