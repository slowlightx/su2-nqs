"""Bounded local direct training of collective query spaces on saved CT data.

This is a development-only continuation of the frozen-embedding probe.  It
trains rank-two spaces on old parents rather than reusing a coefficient
network's readout Jacobian.  No CT tensor, physical state, H kernel, or remote
job is changed.  Later-parent targets and their audit chains are reused
development data, not a new confirmation set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.collective_queries import (
    collective_subspace, fit_collective_update, audit_collective_update,
)
from su2_nqs.benchmarks.projected_capacity import tangent_moments
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.models.capacity_attention import (
    CapacityControllerConfig, init_capacity_controller, predict_capacity_coefficients,
)
from su2_nqs.vmc.adam import _manual_adam_init, _manual_adam_update
from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint
from .triangle_collective_queries import _scalar_inputs


def projected_gain(q: jax.Array, gram: jax.Array, force: jax.Array) -> jax.Array:
    """Regularized tangent gain; q columns are Euclidean orthonormal.

    The supplied force is Re Cov(O,E), half the real energy derivative.
    This is a training surrogate, not a finite energy improvement.
    """
    sub = q.T @ gram @ q
    rhs = q.T @ force
    ridge = .01*jnp.trace(sub)/q.shape[1]+1e-20
    return rhs @ jnp.linalg.solve(sub+ridge*jnp.eye(q.shape[1]), rhs)


def run(previous: Path, output: Path) -> dict:
    """Train from six old parents, seal all target Q, then read target forces."""
    if output.exists():
        raise ValueError("fresh output path required")
    started = perf_counter()
    old = json.loads(previous.read_text())
    training = [Path(p) for p in old["train_parents"]]
    targets = [Path(p) for p in old["target_parents"]]
    if len(training) != 6 or len(targets) != 2:
        raise ValueError("the fixed six-parent/two-later-parent split is required")
    seeds, steps, rank = (74321, 74322, 74323), 300, 2
    modes = ("linear", "signed_graph", "attention", "attention_scrambled", "shared_free")
    panels = [_scalar_inputs(p) for p in training+targets]
    grams, forces = [], []
    for path in training:
        with np.load(path/"dataset.npz") as data:
            fit = data["fit_mask"].astype(bool)
            g, f = tangent_moments(data["scores"][fit], data["local_energies"][fit])
        grams.append(g)
        forces.append(f)
    grams, forces = np.asarray(grams), np.asarray(forces)
    scale = 1/np.sqrt(np.maximum(np.mean(np.diagonal(grams, axis1=1, axis2=2), axis=0), 1e-24))
    # A fixed scalar per old parent prevents one high-force parent dominating.
    normalizers = np.asarray([max(float(f @ np.linalg.pinv(g+.01*np.trace(g)/len(f)*np.eye(len(f))) @ f), 1e-20)
                              for g, f in zip(grams, forces)])
    output.mkdir(parents=True)
    registration = dict(task="direct_collective_subspace_training_development_v1",
        seeds=seeds, steps=steps, rank=rank, modes=modes, learning_rate=.005,
        train_parents=[str(p) for p in training], target_parents=[str(p) for p in targets],
        no_new_H_calls=True, production_updates=0, finite_energy_claim=False,
        target_data_previously_used=True, independent_confirmation=False,
        source_registration=str(previous), source_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
        objectives="maximize regularized physical tangent gain inside a learned rank2 space",
        target_Q_uses_target_force_or_score=False,
        context="same 89 scalar features and signed screening relation as prior v4; fit-chain spin context",
        costs="offline replay only; not proof of actual query or GPU savings",
        symmetry="linear combinations of previously validated exact SU2, full432-A1 scalar tangents",
        heldout="old whole-chain split; no test metric selects step, seed, rank, or learning rate")
    atomic_json(output/"registration.json", registration)
    receipts, subspaces = [], {}
    for mode in modes:
        scrambled = mode == "attention_scrambled"
        feature_key = "scrambled_features" if scrambled else "features"
        relation_key = "scrambled_relations" if scrambled else "relations"
        features = np.asarray([p[feature_key] for p in panels])
        mean, std = features[:6].mean(axis=(0, 1)), features[:6].std(axis=(0, 1))
        active = std > 1e-12
        std = np.where(active, std, 1.)
        features = np.where(active, (features-mean)/std, 0.)
        relations = np.asarray([p[relation_key] for p in panels])
        config = CapacityControllerConfig(features.shape[-1], width=4,
            relation_dim=relations.shape[-1], mixing="attention" if scrambled else "linear" if mode == "shared_free" else mode)
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            if mode == "shared_free":
                params = dict(free=jax.random.normal(key, (24, rank), dtype=jnp.float64))
            else:
                params = init_capacity_controller(key, config)
                params["readout"] = .1*jax.random.normal(jax.random.fold_in(key, 1),
                    (rank, params["readout"].size), dtype=jnp.float64)
                params["bias"] = jnp.zeros(rank, jnp.float64)

            def make_q(p, x, relation):
                if mode == "shared_free":
                    embedding = p["free"]
                else:
                    embedding = jax.vmap(lambda w, b: predict_capacity_coefficients(
                        dict(p, readout=w, bias=b), x, config, relations=relation))(
                            p["readout"], p["bias"]).T
                return jnp.linalg.qr(jnp.asarray(scale)[:, None]*embedding, mode="reduced")[0]

            def loss(p):
                qs = jax.vmap(lambda x, r: make_q(p, x, r))(
                    jnp.asarray(features[:6]), jnp.asarray(relations[:6]))
                gains = jax.vmap(projected_gain)(qs, jnp.asarray(grams), jnp.asarray(forces))
                return -jnp.mean(gains/jnp.asarray(normalizers))

            @jax.jit
            def update(p, opt):
                value, grad = jax.value_and_grad(loss)(p)
                p, opt = _manual_adam_update(p, grad, opt, .005)
                return p, opt, value

            opt = _manual_adam_init(params)
            losses = []
            arm_start = perf_counter()
            for step in range(steps):
                params, opt, value = update(params, opt)
                if step in (0, 99, 199, 299):
                    losses.append(float(value))
            jax.block_until_ready(params)
            if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(params)):
                raise ValueError("nonfinite controller; no post-hoc altered settings")
            path = output/f"{mode}_{seed}.pkl"
            save_checkpoint(path, params, optimizer_state=opt,
                metadata=dict(seed=seed, steps=steps, mode=mode, rank=rank,
                    mean=mean.tolist(), std=std.tolist(), active=active.tolist()))
            restored = load_checkpoint(path)
            live = update(params, opt)
            replay = update(restored["params"], restored["optimizer_state"])
            error = max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a, b in zip(jax.tree.leaves(live), jax.tree.leaves(replay)))
            for index, target in enumerate(targets, 6):
                q = np.asarray(make_q(params, jnp.asarray(features[index]), jnp.asarray(relations[index])))
                subspaces[f"{target.name}__{mode}_{seed}"] = q
            receipts.append(dict(mode=mode, seed=seed, losses=losses,
                parameters=sum(int(a.size) for a in jax.tree.leaves(params)),
                seconds=perf_counter()-arm_start, optimizer_replay_error=error))
            print(mode, seed, losses[-1], flush=True)
    for target in targets:
        for seed in seeds:
            subspaces[f"{target.name}__random_{seed}"] = collective_subspace(
                scale[:, None]*np.random.default_rng(seed).normal(size=(24, rank)), rank)
        subspaces[f"{target.name}__full_reference_0"] = np.eye(24)
    np.savez_compressed(output/"frozen_subspaces.npz", **subspaces)
    atomic_json(output/"controller_receipts.json", receipts)
    # All learned Q are now frozen. Target forces are first read below.
    rows = []
    for target in targets:
        with np.load(target/"dataset.npz") as data:
            scores, energies, fit, labels = (data[k] for k in ("scores", "local_energies", "fit_mask", "chain_labels"))
        fit = fit.astype(bool)
        if set(labels[fit]) & set(labels[~fit]):
            raise ValueError("chain leakage")
        for key, q in subspaces.items():
            if not key.startswith(target.name+"__"):
                continue
            queried = scores @ q
            fitted = fit_collective_update(queried[fit], energies[fit])
            audit = audit_collective_update(fitted.coefficients, queried[~fit], energies[~fit], labels[~fit])
            rows.append(dict(parent=target.name, arm=key.split("__")[1], rank=q.shape[1],
                fit_slope=float(2*fitted.force @ fitted.coefficients), audit=audit,
                audit_fit_fs_ratio=audit["fs_tangent"]/.001,
                cache_identity_error=float(np.max(abs(scores @ (q @ fitted.coefficients)-queried @ fitted.coefficients)))))
    paired = []
    for target in targets:
        lookup = {r["arm"]:r for r in rows if r["parent"] == target.name}
        for control in ("linear", "signed_graph", "shared_free", "random", "attention_scrambled"):
            diffs, deletes = [], []
            for seed in seeds:
                a, b = lookup[f"attention_{seed}"]["audit"], lookup[f"{control}_{seed}"]["audit"]
                diffs.append(a["energy_slope_per_fs_total"]-b["energy_slope_per_fs_total"])
                deletes.append(np.asarray(a["energy_slope_per_fs_total_chain_jackknife"])-np.asarray(b["energy_slope_per_fs_total_chain_jackknife"]))
            delete = np.mean(deletes, axis=0)
            paired.append(dict(parent=target.name, control=control, seed_differences=diffs,
                mean_difference=float(np.mean(diffs)), paired_chain_SE=float(np.sqrt((len(delete)-1)/len(delete)*np.sum((delete-delete.mean())**2))),
                negative_favors_attention=True))
    result = dict(registration=registration, rows=rows, paired=paired,
        receipts=receipts, seconds=perf_counter()-started,
        maximum_identity_error=max(r["cache_identity_error"] for r in rows),
        conclusion="Development-only tangent transfer; no finite-state energy/variance or speed advantage established.")
    atomic_json(output/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.output)
    print(json.dumps(dict(seconds=result["seconds"], paired=result["paired"]), indent=2))
