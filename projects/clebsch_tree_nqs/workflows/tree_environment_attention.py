"""Bounded, static compression-controller test on actual stored CT parents.

No wavefunction is optimized. Targets are physical nine-spin reduced blocks
of one RAW singlet tree, not the coherent A1 sum. Entire checkpoints/couplings
are held out; CT144 is never used to train a CT36 controller. Attention here
is explicitly a learned Gaussian environment-memory kernel, not Spatial-ViT.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.environment_compression_attention import environment_memory_weights
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_gauge import triangular_bank_reduced_tree
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.su2.reduced_tree_environment import reduced_tree_environments
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (73021, 73022, 73023)
PAIRS = tuple(itertools.combinations(range(9), 2))
RETAINED = 12


def source_parents() -> list[tuple[Path, str, float]]:
    """Fixed selection made before inspecting any prediction outcomes."""
    base = ROOT/"projects/clebsch_tree_nqs/results"
    rows = []
    for group in ("triangle_capacity_20261017/remote_snapshot",
                  "triangle_arity_followup_20261016/remote_snapshot"):
        for tag, coupling in (("j0", 0.), ("j0125", .125)):
            for mode in ("binary_truncated", "ternary_joint"):
                rows.append((base/group/tag/mode/"checkpoint.pkl", "train", coupling))
    for tag, coupling in (("j0", 0.), ("j0125", .125)):
        for mode in ("binary_truncated", "ternary_joint"):
            rows.append((base/"remote_review_20261021/ct6_width_20261020"/
                         f"{tag}_{mode}_w18_control/checkpoint.pkl", "train", coupling))
            rows.append((base/"remote_review_20261021/ct6_width_20261020"/
                         f"{tag}_{mode}_w20/checkpoint.pkl", "later_checkpoint", coupling))
        for mode in ("binary", "joint"):
            rows.append((base/"remote_review_20261021/ct12_resume_20261020"/
                         f"{tag}_{mode}/checkpoint.pkl", "larger_size", coupling))
    return rows


def build_dataset(output: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Exact raw-tree densities in a shared physical, not virtual, q=1 frame."""
    tables = triangle_block_tables()["nine"]
    bits = (np.arange(512)[:, None] >> np.arange(8, -1, -1)) & 1
    mags = (2*bits-1).sum(1)
    pair_ops = {}
    for q, table in tables.items():
        u = table*(mags == q)[:, None]
        values = []
        for i, j in PAIRS:
            opposite = bits[:, i] != bits[:, j]
            swapped = np.arange(512) ^ ((opposite.astype(int) << (8-i)) |
                                       (opposite.astype(int) << (8-j)))
            acted = np.where(opposite[:, None], -.25, .25)*u + .5*opposite[:, None]*u[swapped]
            values.append(u.T @ acted)
        pair_ops[q] = np.asarray(values)
    records, features, scrambled, sigmas, weights, parents, splits, couplings, sizes, frames = [], [], [], [], [], [], [], [], [], []
    for parent_id, (path, split, coupling) in enumerate(source_parents()):
        if not path.is_file():
            raise FileNotFoundError(f"Preregistered source is missing: {path}")
        saved = load_checkpoint(path)
        raw = saved["params"]
        count = len(raw["local"])
        length = int(np.sqrt(9*count))
        cap = raw["local"][0][0].shape[1]
        config = MixedClebschConfig(length=length, multiplicity_cap=cap, maximum_q=10)
        mode = "binary_truncated" if "six" in raw else "ternary_joint"
        bank = jax.tree.map(np.asarray, arity_clebsch_bank(raw, config, mode, 8))
        state = triangular_bank_reduced_tree(bank, config)
        env_started = time.perf_counter()
        env = reduced_tree_environments(state)
        environment_seconds = time.perf_counter()-env_started
        physical_started = time.perf_counter()
        assert env.maximum_relative_edge_norm_error < 1e-10
        local_corr, local_sigma, local_weight = [], [], []
        for node, blocks in zip(state.tree.leaves, bank["local"], strict=True):
            physical = {q: (q+1)*w @ env.downward[node][q] @ w.T/env.norm_squared
                        for q, w in zip(config.local_q, blocks, strict=True)}
            total = sum(float(np.trace(a).real) for a in physical.values())
            if abs(total-1) > 1e-10 or max(np.max(abs(a.imag)) for a in physical.values()) > 1e-12:
                raise ArithmeticError("physical scalar RDM normalization/reality failed")
            corr = sum(np.einsum("pij,ji->p", pair_ops[q], a.real) for q, a in physical.items())
            sigma = physical[1].real
            weight = float(np.trace(sigma))
            local_corr.append(corr); local_sigma.append(sigma/weight); local_weight.append(weight)
        corr = np.asarray(local_corr)
        hierarchy = triangular_schur_hierarchy(length)
        centers = np.asarray([np.mean(b.lifted_coordinates, axis=0) for b in hierarchy.blocks])
        delta = centers[:, None]-centers[None]
        distance = np.min([((delta[..., 0]+a*length)**2+(delta[..., 1]+b*length)**2
                            -(delta[..., 0]+a*length)*(delta[..., 1]+b*length))
                           for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)**.5
        near = np.exp(-distance/3); far = np.exp(distance/3-distance.max()/3)
        np.fill_diagonal(near, 0); np.fill_diagonal(far, 0)
        near /= near.sum(1, keepdims=True); far /= far.sum(1, keepdims=True)
        perm = np.random.default_rng(73020+length).permutation(count)
        features.extend(np.column_stack((corr, near @ corr, far @ corr)))
        scrambled.extend(np.column_stack((corr, near[:, perm] @ corr, far[:, perm] @ corr)))
        sigmas.extend(local_sigma); weights.extend(local_weight)
        parents.extend([parent_id]*count); splits.extend([split]*count)
        couplings.extend([coupling]*count); sizes.extend([length]*count)
        frames.extend([b.base_index for b in hierarchy.blocks])
        records.append(dict(parent_id=parent_id, checkpoint=str(path.relative_to(ROOT)),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            parameter_sha256=pytree_sha256(raw), mode=mode, cap=int(cap), length=length,
            split=split, j2=coupling, blocks=count,
            global_steps=saved.get("metadata", {}).get("completed_steps"),
            raw_environment_seconds=environment_seconds,
            physical_density_plus_feature_seconds=time.perf_counter()-physical_started,
            maximum_raw_edge_norm_error=env.maximum_relative_edge_norm_error))
        print("extracted", parent_id, split, length, coupling, mode, flush=True)
    data = dict(features=np.asarray(features), scrambled_features=np.asarray(scrambled),
                sigma=np.asarray(sigmas), q1_weight=np.asarray(weights), parent=np.asarray(parents),
                split=np.asarray(splits), j2=np.asarray(couplings), length=np.asarray(sizes),
                local_frame=np.asarray(frames))
    np.savez_compressed(output/"dataset.npz", **data)
    atomic_json(output/"sources.json", records)
    return data, records


def metrics(data: dict, indices: np.ndarray, prediction: np.ndarray) -> tuple[dict, np.ndarray]:
    """Actual one-edge raw-tree squared fidelity, keeping other q complete."""
    values, vectors = np.linalg.eigh((prediction+prediction.swapaxes(1, 2))/2)
    frames = vectors[:, :, -RETAINED:]
    projectors = frames @ frames.swapaxes(1, 2)
    retained = np.einsum("bij,bji->b", projectors, data["sigma"][indices])
    oracle = np.linalg.eigvalsh(data["sigma"][indices])[:, -RETAINED:].sum(1)
    fidelity = 1-data["q1_weight"][indices]*(1-retained)
    regret = data["q1_weight"][indices]*(oracle-retained)
    parent_means = [float(regret[data["parent"][indices] == p].mean()) for p in np.unique(data["parent"][indices])]
    return dict(mean_raw_squared_fidelity=float(fidelity.mean()),
                mean_oracle_fidelity=float(np.mean(1-data["q1_weight"][indices]*(1-oracle))),
                mean_excess_discarded_weight=float(regret.mean()),
                maximum_excess_discarded_weight=float(regret.max()),
                per_parent_mean_excess=parent_means, per_block_fidelity=fidelity.tolist(),
                per_block_excess_discarded_weight=regret.tolist()), projectors


def fit_kernel(x: np.ndarray, target: np.ndarray, groups: np.ndarray, mode: str,
               seed: int, steps: int, output: Path, tag: str) -> tuple[dict, dict]:
    """Fixed-budget leave-whole-parent-out metric learning, with exact replay."""
    x = jnp.asarray(x)
    allowed = jnp.asarray(groups[:, None] != groups[None, :])
    assert np.asarray(allowed).any(1).all()
    gram = jnp.asarray(np.einsum("aij,bij->ab", target, target))
    dim = x.shape[1]
    key = jax.random.PRNGKey(seed)
    p = dict(metric=jax.random.normal(key, (dim, 4), dtype=jnp.float64)/np.sqrt(dim*4),
             bias=jnp.zeros(4, jnp.float64))
    if mode == "diagonal_metric":
        p = dict(log_scale=jnp.zeros(dim, jnp.float64))
    def embedding(p, a):
        if mode == "diagonal_metric":
            return a*jnp.exp(jnp.clip(p["log_scale"], -3., 3.))/np.sqrt(dim)
        z = a @ p["metric"]+p["bias"]
        return jnp.tanh(z) if mode == "mlp_kernel" else z
    def objective(p):
        z = embedding(p, x)
        w = environment_memory_weights(z, z, jnp.eye(z.shape[1]), allowed=allowed)
        loss = jnp.mean(jnp.einsum("ab,bc,ac->a", w, gram, w)-2*jnp.sum(w*gram, axis=1))
        return loss+1e-4*sum(jnp.mean(a*a) for a in jax.tree.leaves(p))
    @jax.jit
    def step(p, m, v, k):
        loss, g = jax.value_and_grad(objective)(p)
        m = jax.tree.map(lambda a, b: .9*a+.1*b, m, g)
        v = jax.tree.map(lambda a, b: .999*a+.001*b*b, v, g)
        p = jax.tree.map(lambda a, b, c: a-.01*b/(1-.9**k)/(jnp.sqrt(c/(1-.999**k))+1e-8), p, m, v)
        return p, m, v, loss
    m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
    started = time.perf_counter()
    for k in range(1, steps+1):
        p, m, v, loss = step(p, m, v, jnp.asarray(k))
    jax.block_until_ready(p)
    checkpoint = output/"controller_checkpoints.pkl"
    case = f"{tag}_{mode}_{seed}"
    bundle = (load_checkpoint(checkpoint) if checkpoint.exists() else
              dict(params={}, optimizer_state={}, metadata={}))
    bundle["params"][case] = p
    bundle["optimizer_state"][case] = dict(m=m, v=v, step=steps, rng=key)
    save_checkpoint(checkpoint, bundle["params"], optimizer_state=bundle["optimizer_state"],
                    metadata=dict(source="static_scalar_environments", cases=tuple(bundle["params"])))
    restored = load_checkpoint(checkpoint)
    a = step(p, m, v, jnp.asarray(steps+1))
    state = restored["optimizer_state"][case]
    b = step(restored["params"][case], state["m"], state["v"], jnp.asarray(state["step"]+1))
    replay_error = max(float(np.max(abs(np.asarray(u)-np.asarray(v))))
                       for u, v in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))
    assert replay_error == 0
    return dict(params=p, embedding=embedding), dict(mode=mode, seed=seed,
        parameters=sum(a.size for a in jax.tree.leaves(p)), steps=steps,
        seconds=time.perf_counter()-started, loss=float(loss), replay_max_error=replay_error,
        checkpoint=checkpoint.name, checkpoint_case=case)


def run(output: Path, steps: int, protocols: list[str] | None = None) -> None:
    """Preregister, extract real parents, then evaluate every fixed endpoint."""
    output.mkdir(parents=True, exist_ok=True)
    registration = dict(seeds=SEEDS, training_steps=steps, retained_q1_copies=RETAINED,
        q=1, complete_physical_q1_multiplicity=42, other_q="kept complete",
        hypotheses=["environment-memory metric may improve on constant compression",
                    "attention-specific advantage requires beating strong fixed-kernel and linear/quadratic controls",
                    "CT36-to-CT144 generalization cannot be inferred from checkpoint interpolation"],
        features="36 intra-nine-site physical pair correlations plus distance-weighted near/far block contexts",
        target="fixed complete physical Schur-frame q=1 reduced density; not raw virtual gauge coordinates",
        scope="raw-tree one-edge static compression, no full-A1 energy or throughput claim",
        source_paths=[str(a.relative_to(ROOT)) for a, _, _ in source_parents()],
        protocols=protocols or ["history_prediction", "whole_coupling_holdout"],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), ROOT/"su2_nqs/models/environment_compression_attention.py")},
        steps_full_H=0, state_optimization_steps=0,
        limits=["Parents are related training lineages, not independent samples.",
                "History prediction tests future checkpoints, not unseen physical families.",
                "Larger-size parents are weak development states.",
                "Gaussian attention is mathematically kernel regression; labels are not novelty.",
                "Environment extraction cost is reported but is not an H-action speedup.",
                "Scalar-feature count is incomplete/low-dimensional, but these reference features are extracted using exact raw environments, not a cheaper independent estimator.",
                "Replacing an already cheap exact raw environment has no established computational value; future learning should target costly projected-A1/H-residual environments or parameter/capacity transfer."])
    atomic_json(output/"registration.json", registration)
    started = time.perf_counter()
    data, sources = build_dataset(output)
    extraction_seconds = time.perf_counter()-started
    results = []
    predictions = {}
    for protocol in registration["protocols"]:
        train_mask = data["split"] == "train"
        if protocol == "whole_coupling_holdout":
            train_mask &= data["j2"] == 0
        elif protocol == "larger_size_whole_coupling_holdout":
            train_mask = (data["split"] == "larger_size") & (data["j2"] == 0)
        elif protocol != "history_prediction":
            raise ValueError("unknown preregistered split")
        train_idx = np.flatnonzero(train_mask)
        targets = data["sigma"][train_idx]
        for scramble in (False, True):
            feature_key = "scrambled_features" if scramble else "features"
            mean = data[feature_key][train_idx].mean(0)
            scale = data[feature_key][train_idx].std(0)+1e-6
            xall = (data[feature_key]-mean)/scale
            x = xall[train_idx]
            test_groups = (["larger_size"] if protocol == "larger_size_whole_coupling_holdout"
                           else ["later_checkpoint", "larger_size"])
            methods = {}
            methods["static_mean"] = lambda z: np.broadcast_to(targets.mean(0), (len(z), 42, 42))
            methods["nearest_parent"] = lambda z: targets[np.argmin(((z[:, None]-x[None])**2).sum(-1), axis=1)]
            methods["gaussian_fixed"] = lambda z: np.asarray(jnp.einsum("bm,mij->bij",
                environment_memory_weights(jnp.asarray(z), jnp.asarray(x), jnp.eye(x.shape[1])/np.sqrt(x.shape[1])),
                jnp.asarray(targets)))
            # No post-hoc bandwidth/ridge scan. Both fixed strong controls use all scalar inputs.
            for degree in (1, 2):
                kernel = (1+x @ x.T/x.shape[1])**degree
                alpha = np.linalg.solve(kernel+.1*np.eye(len(x)), targets.reshape(len(x), -1))
                methods[f"ridge_degree{degree}"] = lambda z, d=degree, a=alpha: (
                    (1+z @ x.T/x.shape[1])**d @ a).reshape(len(z), 42, 42)
            for mode in ("diagonal_metric", "bilinear_metric", "mlp_kernel"):
                for seed in SEEDS:
                    fit, note = fit_kernel(x, targets, data["parent"][train_idx], mode, seed, steps,
                                          output, protocol+("_scramble" if scramble else ""))
                    embedded_memory = fit["embedding"](fit["params"], jnp.asarray(x))
                    def predictor(z, f=fit, memory=embedded_memory):
                        query = f["embedding"](f["params"], jnp.asarray(z))
                        w = environment_memory_weights(query, memory, jnp.eye(memory.shape[1]))
                        return np.asarray(jnp.einsum("bm,mij->bij", w, jnp.asarray(targets)))
                    methods[f"{mode}_seed{seed}"] = predictor
                    note.update(protocol=protocol, scrambled_geometry=scramble)
                    results.append(dict(training=note))
            for split in test_groups:
                mask = data["split"] == split
                if protocol in ("whole_coupling_holdout", "larger_size_whole_coupling_holdout"):
                    mask &= data["j2"] == .125
                indices = np.flatnonzero(mask)
                # A fixed local-frame copy is an especially important strong
                # control when tree frames retain a common bottom geometry.
                fixed_frame_prediction = np.asarray([targets[data["local_frame"][train_idx] == f].mean(0)
                    for f in data["local_frame"][indices]])
                frame_metric, frame_projectors = metrics(data, indices, fixed_frame_prediction)
                frame_name = f"{protocol}_{split}_{'scrambled' if scramble else 'physical'}_fixed_local_frame"
                predictions.update({frame_name+"/"+key:value for key, value in dict(indices=indices,
                    parent_ids=data["parent"][indices], predicted_density=fixed_frame_prediction,
                    retained_projectors=frame_projectors, target_density=data["sigma"][indices],
                    q1_weight=data["q1_weight"][indices]).items()})
                results.append(dict(protocol=protocol, split=split, scrambled_geometry=scramble,
                    method="fixed_local_frame", metrics=frame_metric, projectors="predictions.npz",
                    prediction_namespace=frame_name))
                for method, predict in methods.items():
                    prediction = predict(xall[indices])
                    metric, projectors = metrics(data, indices, prediction)
                    name = f"{protocol}_{split}_{'scrambled' if scramble else 'physical'}_{method}"
                    predictions.update({name+"/"+key:value for key, value in dict(indices=indices,
                        parent_ids=data["parent"][indices], predicted_density=prediction,
                        retained_projectors=projectors, target_density=data["sigma"][indices],
                        q1_weight=data["q1_weight"][indices]).items()})
                    row = dict(protocol=protocol, split=split, scrambled_geometry=scramble,
                               method=method, metrics=metric, projectors="predictions.npz",
                               prediction_namespace=name)
                    results.append(row)
                    print(protocol, split, scramble, method, metric["mean_raw_squared_fidelity"],
                          metric["mean_excess_discarded_weight"], flush=True)
            atomic_json(output/"results.json", dict(registration=registration, sources=sources,
                extraction_seconds=extraction_seconds, wall_seconds=time.perf_counter()-started, results=results))
    np.savez_compressed(output/"predictions.npz", **predictions)
    atomic_json(output/"results.json", dict(status="completed", registration=registration,
        sources=sources, extraction_seconds=extraction_seconds,
        geometry_diagnostic=dict(max_CT36_physical_scramble_feature_difference=float(np.max(abs(
            data["features"][data["length"] == 6]-data["scrambled_features"][data["length"] == 6]))),
            caveat="All four CT36 block centers are equidistant: near/far pools are uniform. CT36 alone cannot teach nontrivial geometric context weights."),
        wall_seconds=time.perf_counter()-started, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--protocols", nargs="+")
    args = parser.parse_args()
    run(args.output, args.steps, args.protocols)
