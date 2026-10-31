"""Preregistered N16 same-backbone variation-after-singlet-projection test.

Reuses saved physical-input ViTs and exact sparse operator construction; never
diagonalizes H. Ordinary continuation, frozen-parent projection and PAV are
the controls. Exhaustive support is an audit, not a statistical held-out set.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import json

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.krylov_anchored_full_vit import a1_orbit_hamiltonian
from su2_nqs.benchmarks.singlet_vap import singlet_isometry, physical_cut_metrics
from su2_nqs.benchmarks.spin_sector import singlet_projector_action
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from su2_nqs.vmc.exact_fs_adam import train_exact_fs_adam
from .spin import _prepare_case
from .spin_full_vit_budget import prepare_objective

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT/"projects/clebsch_tree_nqs/configs/validation/spin_full_vit_vap.json"


def prepare_vap(data: dict, config: FixedSzSpatialViTConfig) -> dict:
    """Keep exactly the same raw network, spatial reduction and Hamiltonian."""
    raw = prepare_objective(data, config)
    a1 = raw["a1"]
    h = a1_orbit_hamiltonian(data["h_rows"], data["h_columns"], data["h_values"], a1)
    coo = data["spin_squared"].tocoo()
    s2 = a1_orbit_hamiltonian(coo.row, coo.col, coo.data, a1)
    columns = singlet_isometry(s2, n_sites=config.n_sites)
    p0 = columns@columns.conj().T
    columns_jax = jnp.asarray(columns)

    def projected(params):
        coefficients = columns_jax.conj().T @ raw["state_fn"](params)
        norm = jnp.linalg.norm(coefficients)
        # No hidden normalization floor: zero projection is invalid, not a state.
        return columns_jax @ jnp.where(norm > 1e-10, coefficients/norm, jnp.nan+0j)

    checks = dict(projector_idempotence=float(np.max(abs(p0@p0-p0))),
                  projector_hermiticity=float(np.max(abs(p0-p0.conj().T))),
                  singlet_residual=float(np.max(abs(s2@p0))),
                  H_commutator=float(np.max(abs(h@p0-p0@h))))
    if max(checks.values()) > 1e-9:
        raise RuntimeError(f"projector invariant failed: {checks}")
    return dict(raw=raw, h=h, s2=s2, p0=p0, projected=jax.jit(projected), checks=checks)


def state_metrics(vector: np.ndarray, problem: dict, data: dict, *, cuts: bool = False) -> dict:
    """Independently audit normalized orbit state with full physical H and S2."""
    psi = problem["raw"]["physical"](vector)
    psi = psi/np.linalg.norm(psi)
    action = data["hamiltonian"]@psi
    energy = float(np.vdot(psi, action).real)
    s2psi = data["spin_squared"]@psi
    projected = singlet_projector_action(psi, data["spin_squared"], maximum_spin=data["n_sites"]//2)
    result = dict(energy_per_site=energy/data["n_sites"],
                  energy_error_per_site=(energy-data["exact_energy"])/data["n_sites"],
                  variance_per_site=float(np.linalg.norm(action-energy*psi)**2)/data["n_sites"],
                  spin_squared=float(np.vdot(psi,s2psi).real),
                  singlet_weight=float(np.vdot(projected,projected).real),
                  singlet_residual_norm=float(np.linalg.norm(s2psi)),
                  full_H_replay_max=float(np.max(abs(action-problem["raw"]["physical"](problem["h"]@vector)))),
                  a1_replay_max=float(np.max(abs(psi[data["tables"]["gamma_a1"]]-psi))))
    if cuts:
        result["cuts"] = {name: physical_cut_metrics(psi, np.asarray(data["basis"]), sites)
            for name,sites in {"plaquette":(0,1,4,5), "row":(0,1,2,3)}.items()}
    return result


def registration(config: dict) -> dict:
    """Seal all parents and transitive numeric sources before real training."""
    files = {}
    for trial in config["parents"]:
        for name in ("results.json", "parameters.json", "parameters.npz"):
            path = ROOT/config["parent_root"]/trial/name
            files[str(path.relative_to(ROOT))] = file_sha256(path)
    record = dict(config=config, parent_files=files,
                  source_manifest=transitive_local_source_manifest((Path(__file__),), root=ROOT,
                      package_prefixes=("su2_nqs","projects.clebsch_tree_nqs")),
                  runtime=numerical_runtime_identity())
    record["registration_sha256"] = canonical_sha256(record)
    return record


def run(config_path: Path, trial: str, arm: str, *, resume: bool = False) -> dict:
    """Run one of eight fixed-budget continuations, without selecting seeds."""
    config = json.loads(config_path.read_text())
    output = ROOT/config["output"]
    registered = json.loads((output/"registration.json").read_text())
    if registered != registration(config):
        raise ValueError("registration, sources, parents or runtime changed")
    if trial not in config["parents"] or arm not in ("raw", "vap"):
        raise ValueError("unregistered trial or arm")
    destination = output/f"{trial}_{arm}"
    if (destination/"results.json").exists():
        raise FileExistsError(destination/"results.json")
    started = perf_counter()
    parent_dir = ROOT/config["parent_root"]/trial
    parent = json.loads((parent_dir/"results.json").read_text())
    model_config = FixedSzSpatialViTConfig(**parent["model_config"])
    template = init_fixed_sz_spatial_vit(jax.random.PRNGKey(0), model_config)
    params = load_parameter_artifact(parent_dir/"parameters.json", template, expected_identity=dict(trial=trial))
    if pytree_sha256(params) != parent["final_params_sha256"]:
        raise ValueError("parent result and parameters disagree")
    data = _prepare_case(parent["case"], "fixed_m0")
    problem = prepare_vap(data, model_config)
    validate = problem["raw"]["validate"](params)
    raw_parent = np.asarray(problem["raw"]["state_fn"](params))
    projected_parent = np.asarray(problem["projected"](params))
    parent_metrics = state_metrics(raw_parent, problem, data, cuts=True)
    frozen_metrics = state_metrics(projected_parent, problem, data, cuts=True)
    replay = abs(parent_metrics["energy_per_site"]-parent["metrics"]["energy_per_site"])
    if replay > 2e-11:
        raise RuntimeError("saved parent energy not replayed")
    # Independent full-support polynomial, rather than the orbit projector.
    independent = singlet_projector_action(problem["raw"]["physical"](raw_parent),
        data["spin_squared"], maximum_spin=data["n_sites"]//2)
    independent /= np.linalg.norm(independent)
    validate["physical_projector_replay_max"] = float(np.max(abs(independent-problem["raw"]["physical"](projected_parent))))
    validate["parent_energy_replay_error"] = replay
    if validate["physical_projector_replay_max"] > 1e-9:
        raise RuntimeError("independent P0 replay failed")
    state_fn = problem["projected"] if arm == "vap" else problem["raw"]["state_fn"]
    tangent = jax.tree_util.tree_map(lambda x: jnp.ones_like(x)*.01, params)
    loss = lambda p: jnp.vdot(state_fn(p), jnp.asarray(problem["h"])@state_fn(p)).real
    derivative = float(jax.jvp(loss, (params,), (tangent,))[1])
    shifts = [jax.tree_util.tree_map(lambda p,t: p+sign*1e-4*t, params, tangent) for sign in (-1,1)]
    finite_diff = float((loss(shifts[1])-loss(shifts[0]))/2e-4)
    validate["directional_gradient_error"] = abs(derivative-finite_diff)
    if validate["directional_gradient_error"] > 2e-7:
        raise RuntimeError("projected gradient check failed")
    diagnostics = (json.loads((destination/"diagnostics.json").read_text())
        if resume and (destination/"diagnostics.json").exists() else [])

    def callback(p, row):
        metrics = state_metrics(np.asarray(state_fn(p)), problem, data)
        diagnostics.append(dict(step=row["step"], **metrics))
        atomic_json(destination/"diagnostics.json", diagnostics)
        print(trial, arm, row["step"], metrics["energy_error_per_site"], metrics["variance_per_site"], flush=True)

    print(trial, arm, "prepared", perf_counter()-started, "seconds", flush=True)
    trained, training = train_exact_fs_adam(params, state_fn=state_fn, hamiltonian=problem["h"],
        identity=dict(registration_sha256=registered["registration_sha256"], trial=trial, arm=arm),
        checkpoint=destination/"checkpoint.pkl", callback=callback, resume=resume, **config["optimization"])
    terminal_raw = np.asarray(problem["raw"]["state_fn"](trained))
    terminal_projected = np.asarray(problem["projected"](trained))
    final = state_metrics(terminal_projected if arm == "vap" else terminal_raw, problem, data, cuts=True)
    pav = state_metrics(terminal_projected, problem, data, cuts=True)
    validate.update(problem["checks"])
    # Audit the underlying network too; VAP output itself is singlet by construction.
    raw_terminal_metrics = state_metrics(terminal_raw, problem, data)
    save_parameter_artifact(destination/"parameters.json", trained, identity=dict(trial=trial, arm=arm))
    loaded = load_parameter_artifact(destination/"parameters.json", template, expected_identity=dict(trial=trial, arm=arm))
    validate["checkpoint_amplitude_replay_max"] = float(np.max(abs(np.asarray(state_fn(loaded))-np.asarray(state_fn(trained)))))
    state_path = destination/"states.npz"
    temporary = destination/"states.pending.npz"
    np.savez_compressed(temporary, basis=np.asarray(data["basis"],dtype=np.int8),
        raw_parent=problem["raw"]["physical"](raw_parent),
        projected_parent=problem["raw"]["physical"](projected_parent),
        raw_terminal=problem["raw"]["physical"](terminal_raw),
        projected_terminal=problem["raw"]["physical"](terminal_projected))
    temporary.replace(state_path)
    result = dict(trial=trial, arm=arm, case=parent["case"], seed=parent["seed"], parameter_count=parent["parameter_count"],
        registration_sha256=registered["registration_sha256"], parent_params_sha256=parent["final_params_sha256"],
        final_params_sha256=pytree_sha256(trained), parent=parent_metrics, frozen_projection=frozen_metrics,
        physical_states_sha256=file_sha256(state_path),
        final=final, pav=pav, raw_terminal=raw_terminal_metrics, training=training, trajectory=diagnostics,
        checks=validate, singlet_a1_dimension=int(round(np.trace(problem["p0"]).real)),
        a1_dimension=problem["raw"]["a1_dimension"], raw_evaluations_per_state=problem["raw"]["raw_evaluations"],
        training_projection_calls=training["full_H_calls"] if arm == "vap" else 0,
        diagnostic_full_H_calls=6+len(diagnostics), diagnostic_reduced_H_calls=3,
        wall_seconds=perf_counter()-started, hamiltonian_diagonalization_performed=False,
        remote_submitted=False, all_parameters_trained=True,
        interpretation="Exact-support finite-size mechanism test. Equal steps/capacity, not equal CPU-work or FS distance. No new random seeds, sampled audit or scalable SU2 decoder claim.")
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination/"results.json", result)
    print(trial, arm, "DONE", final["energy_error_per_site"], "wall", result["wall_seconds"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--trial")
    parser.add_argument("--arm", choices=("raw", "vap"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.register:
        path = ROOT/config["output"]/"registration.json"
        if path.exists():
            raise FileExistsError(path)
        record = registration(config)
        atomic_json(path, record)
        print(record["registration_sha256"])
    else:
        run(args.config, args.trial, args.arm, resume=args.resume)


if __name__ == "__main__":
    main()
