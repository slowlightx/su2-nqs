"""Bounded physical-block initialization and retained-subspace transfer probe.

The physical test is a 4x3 OPEN triangular cluster, not a production torus.
Complete three-spin multiplets form six-spin blocks. Only multiplicity axes
are compressed; the twelve-spin Ritz state is evaluated with the full sparse
physical Hamiltonian, including its variance outside the retained space.
Learned controllers predict projectors in a fixed physical Schur frame, never
eigenvector coordinates or virtual gauges. This is an offline small-block
reference and does not claim an attention speedup or 6-to-12 production win.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy.sparse.linalg import eigsh

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.retained_boundary_response import fixed_q_response_frame
from su2_nqs.operators.su2_block_transfer import (
    BlockIsometry, canonicalize_block, merge_projected_blocks, project_block,
)
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[3]
RETAIN = {0: 2, 2: 2, 4: 1, 6: 1}
LEARN_Q = (0, 2, 4)
SEEDS = (26102341, 26102342, 26102343)
# Two elementary triangles in one 2x3 open tile, in physical-bit order.
COORDS6 = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2))
DISPLACEMENTS = (((1, 0), 1), ((0, 1), 1), ((1, 1), 1),
                 ((1, -1), 3), ((1, 2), 3), ((2, 1), 3))


def physical_pair_operator(n_sites: int, first: int, second: int) -> sparse.csr_matrix:
    """S_first dot S_second with site zero the most significant bit."""
    if not 0 <= first < n_sites or not 0 <= second < n_sites or first == second:
        raise ValueError("distinct valid physical sites required")
    rows = np.arange(1 << n_sites)
    opposite = ((rows >> (n_sites-1-first)) & 1) != ((rows >> (n_sites-1-second)) & 1)
    swaps = rows[opposite] ^ ((1 << (n_sites-1-first)) | (1 << (n_sites-1-second)))
    return sparse.coo_matrix((np.r_[np.where(opposite, -.25, .25), np.full(opposite.sum(), .5)],
        (np.r_[rows, rows[opposite]], np.r_[rows, swaps])), shape=(1 << n_sites,)*2).tocsr()


def physical_bonds(coords: tuple[tuple[int, int], ...], j2: float,
                   cross_scale: float = 1., anisotropy: float = 0.) -> list[tuple[int, int, float]]:
    """Actual first/second-shell scalar exchanges, with a declared tile join."""
    position = {x: i for i, x in enumerate(coords)}
    rows = []
    for i, (r, c) in enumerate(coords):
        for direction, ((dr, dc), shell) in enumerate(DISPLACEMENTS):
            j = position.get((r+dr, c+dc))
            if j is None:
                continue
            strength = (1. if shell == 1 else float(j2)) * (1+anisotropy*((direction % 3)-1))
            if (i < 3) != (j < 3) and len(coords) == 6:
                strength *= cross_scale
            rows.append((min(i, j), max(i, j), strength))
    return rows


def schur_multiplets(n_sites: int) -> dict[int, np.ndarray]:
    """Complete three/six-spin multiplets in fixed Condon--Shortley frames."""
    if n_sites not in (3, 6):
        raise ValueError("only the complete three/six-site reference is supported")
    tables = triangle_block_tables()["three" if n_sites == 3 else "six"]
    m = np.asarray([2*i.bit_count()-n_sites for i in range(1 << n_sites)])
    return {q: np.stack([table*(m == value)[:, None]
                        for value in range(-q, q+1, 2)], axis=-1) for q, table in tables.items()}


def _hamiltonian(n: int, bonds: list[tuple[int, int, float]]) -> sparse.csr_matrix:
    return sum((strength*physical_pair_operator(n, i, j) for i, j, strength in bonds),
               sparse.csr_matrix((1 << n,)*2))


def _frames_from_projectors(prediction: np.ndarray, dimensions: tuple[int, ...]) -> dict[int, np.ndarray]:
    frames = {}; offset = 0
    for q, d in zip(LEARN_Q, dimensions, strict=True):
        matrix = prediction[offset:offset+d*d].reshape(d, d)
        _, vectors = np.linalg.eigh((matrix+matrix.T)/2)
        frames[q] = vectors[:, -RETAIN[q]:]
        offset += d*d
    frames[6] = np.ones((1, 1))
    return frames


def _projector_vector(frames: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([(frames[q] @ frames[q].T).reshape(-1) for q in LEARN_Q])


@lru_cache(maxsize=8)
def _twelve_reference(j2: float, cross_scale: float, anisotropy: float) -> tuple:
    """Build each common physical target and its exact reference only once."""
    started = perf_counter()
    coords12 = COORDS6+tuple((r+2, c) for r, c in COORDS6)
    internal = physical_bonds(COORDS6, j2, cross_scale, anisotropy)
    full_bonds = internal+[(i+6, j+6, value) for i, j, value in internal]+[
        (i, j, value) for i, j, value in physical_bonds(coords12, j2, anisotropy=anisotropy)
        if (i < 6) != (j < 6)]
    h = _hamiltonian(12, full_bonds)
    m0 = np.asarray([i for i in range(4096) if i.bit_count() == 6])
    values, vectors = eigsh(h[m0][:, m0], k=1, which="SA", tol=2e-12,
                            v0=np.random.default_rng(26102340).normal(size=len(m0)))
    exact = np.zeros(4096); exact[m0] = vectors[:, 0]
    return h, full_bonds, float(values[0]), exact, perf_counter()-started


def build_dataset() -> dict[str, object]:
    """Freeze whole coupling tuples as train/holdout before any model fitting."""
    multiplets = schur_multiplets(6)
    dimensions = tuple(multiplets[q].shape[1] for q in LEARN_Q)
    edge_rows = physical_bonds(COORDS6, 1.)
    operators = [physical_pair_operator(6, i, j) for i, j, _ in edge_rows]
    cases = [(split, j2, lam, aniso) for split, js, ls in (
        ("train", (0., .05, .125, .175), (.25, .6, 1.)),
        ("heldout", (.075, .15), (.425, .85)))
        for j2 in js for lam in ls for aniso in (-.12, .12)]
    cases += [("physical_target", j2, 1., 0.) for j2 in (0., .125)]
    inputs = []; targets = []; notes = []; frames_all = []
    for split, j2, lam, aniso in cases:
        bonds = physical_bonds(COORDS6, j2, lam, aniso)
        h = sum((value*op for (_, _, value), op in zip(bonds, operators, strict=True)),
                sparse.csr_matrix((64, 64)))
        frames = {}; target_energies = {}
        for q in multiplets:
            u = multiplets[q][:, :, -1]
            values, vectors = np.linalg.eigh(u.T @ (h @ u))
            frames[q] = vectors[:, :RETAIN[q]]
            target_energies[q] = values[:RETAIN[q]].tolist()
        tokens = []
        for i, j, strength in bonds:
            a, b = COORDS6[i], COORDS6[j]
            dr, dc = b[0]-a[0], b[1]-a[1]
            shell = dr*dr+dc*dc-dr*dc
            tokens.append((strength, shell/3, dr/2, dc/2, float((i < 3) != (j < 3)),
                           (a[0]+b[0])/4, (a[1]+b[1])/4, 1.))
        inputs.append(tokens); targets.append(_projector_vector(frames)); frames_all.append(frames)
        notes.append(dict(split=split, j2=j2, cross_scale=lam, anisotropy=aniso,
                          retained_energies=target_energies))
    parent_h = _hamiltonian(6, physical_bonds(COORDS6, 0., .6, 0.))
    parent_frames = {q: np.linalg.eigh(u[:, :, -1].T @ (parent_h @ u[:, :, -1]))[1][:, :RETAIN[q]]
                     for q, u in multiplets.items()}
    return dict(inputs=np.asarray(inputs), targets=np.asarray(targets), notes=notes,
                frames=frames_all, parent_frames=parent_frames, dimensions=dimensions,
                multiplets=multiplets)


def init_controller(kind: str, key: jax.Array, token_shape: tuple[int, int], output: int) -> dict:
    """Small full-output controllers, with the same physical token input."""
    keys = jax.random.split(key, 8); t, d = token_shape
    normal = lambda key, shape: jax.random.normal(key, shape, dtype=jnp.float64)/np.sqrt(shape[0])
    if kind == "linear":
        return dict(w=.01*normal(keys[0], (t*d, output)), b=jnp.zeros(output))
    if kind == "mlp":
        return dict(w=normal(keys[0], (t*d, 16)), b=jnp.zeros(16),
                    out=.01*normal(keys[1], (16, output)), ob=jnp.zeros(output))
    if kind not in ("attention", "scrambled_attention"):
        raise ValueError("unknown controller")
    return dict(embed=normal(keys[0], (d, 12)), query=normal(keys[1], (12, 12)),
                key=normal(keys[2], (12, 12)), value=normal(keys[3], (12, 12)),
                out=.01*normal(keys[4], (12, output)), b=jnp.zeros(output))


def controller(params: dict, x: jax.Array, kind: str) -> jax.Array:
    """Predict scalar reduced-projector matrices in the fixed physical frame."""
    if kind == "linear":
        return x.reshape(-1) @ params["w"]+params["b"]
    if kind == "mlp":
        return jnp.tanh(x.reshape(-1) @ params["w"]+params["b"]) @ params["out"]+params["ob"]
    if kind == "scrambled_attention":
        # Scramble geometric columns only; strengths remain on their edge.
        x = x.at[:, 1:7].set(jnp.roll(x[:, 1:7], 5, axis=0))
    z = jnp.tanh(x @ params["embed"])
    weights = jax.nn.softmax((z @ params["query"]) @ (z @ params["key"]).T/np.sqrt(12), axis=-1)
    pooled = jnp.mean(jnp.tanh(z+weights @ (z @ params["value"])), axis=0)
    return pooled @ params["out"]+params["b"]


def fit_controller(data: dict, kind: str, seed: int, steps: int, output: Path) -> tuple[np.ndarray, dict]:
    """Fixed-budget minibatch Adam; save and replay parameters, moments and RNG."""
    train = np.asarray([i for i, row in enumerate(data["notes"]) if row["split"] == "train"])
    # Normalize each physical FEATURE globally, retaining differences between
    # edge positions. Per-edge centering would erase all fixed geometry.
    x = np.asarray(data["inputs"]); mean, scale = x[train].mean((0, 1)), x[train].std((0, 1))
    scale = np.maximum(scale, .1)
    x = jnp.asarray((x-mean)/scale); y = jnp.asarray(data["targets"])
    parent = jnp.asarray(_projector_vector(data["parent_frames"]))
    key, init = jax.random.split(jax.random.PRNGKey(seed))
    p = init_controller(kind, init, x.shape[1:], y.shape[1])
    zero = jax.tree.map(jnp.zeros_like, p)
    state = (p, zero, zero, key, jnp.asarray(0))
    train_ids = jnp.asarray(train)

    @jax.jit
    def update(state):
        p, m, v, key, count = state
        key, sample = jax.random.split(key)
        ids = jax.random.choice(sample, train_ids, shape=(12,), replace=False)
        def loss(p):
            prediction = parent+jax.vmap(lambda row: controller(p, row, kind))(x[ids])
            return jnp.mean((prediction-y[ids])**2)
        value, grad = jax.value_and_grad(loss)(p)
        count = count+1
        m = jax.tree.map(lambda a, g: .9*a+.1*g, m, grad)
        v = jax.tree.map(lambda a, g: .999*a+.001*g*g, v, grad)
        p = jax.tree.map(lambda a, b, c: a-.01*(b/(1-.9**count))/(jnp.sqrt(c/(1-.999**count))+1e-8), p, m, v)
        return (p, m, v, key, count), value

    started = perf_counter()
    losses = []
    for _ in range(steps):
        state, loss = update(state); losses.append(float(loss))
    prediction = np.asarray(parent+jax.vmap(lambda row: controller(state[0], row, kind))(x))
    seconds = perf_counter()-started
    path = output/f"{kind}_seed{seed}.pkl"
    save_checkpoint(path, state[0], optimizer_state=dict(m=state[1], v=state[2], key=state[3], count=state[4]),
                    metadata=dict(kind=kind, seed=seed, mean=mean, scale=scale, steps=steps))
    saved = load_checkpoint(path); opt = saved["optimizer_state"]
    restored = (saved["params"], opt["m"], opt["v"], opt["key"], opt["count"])
    replay = update(restored)[0]; expected = update(state)[0]
    replay_error = max(float(np.max(abs(np.asarray(a)-np.asarray(b))))
                       for a, b in zip(jax.tree.leaves(replay), jax.tree.leaves(expected), strict=True))
    error = []
    for pred, target in zip(prediction, np.asarray(data["targets"]), strict=True):
        frames = _frames_from_projectors(pred, data["dimensions"])
        error.append(float(np.linalg.norm(_projector_vector(frames)-target)**2/(2*sum(RETAIN[q] for q in LEARN_Q))))
    return prediction, dict(kind=kind, seed=seed, steps=steps, training_seconds=seconds,
        parameters=sum(int(a.size) for a in jax.tree.leaves(state[0])),
        first_loss=losses[0], final_loss=losses[-1], next_step_replay_error=replay_error,
        train_projector_error=float(np.mean(np.asarray(error)[train])),
        heldout_projector_error=float(np.mean([v for v, note in zip(error, data["notes"], strict=True)
                                               if note["split"] == "heldout"])),
        per_case_projector_error=error, parameter_sha256=pytree_sha256(state[0]))


def evaluate_twelve(frames: dict[int, np.ndarray], j2: float, cross_scale: float,
                    anisotropy: float = 0.,
                    *, q0_only: bool = False) -> dict:
    """Full physical energy/variance of a two-block Ritz singlet, OBC 4x3."""
    base = schur_multiplets(6)
    six = BlockIsometry(6, {q: np.einsum("pak,ab->pbk", base[q], frame)
                            for q, frame in frames.items() if not q0_only or q == 0})
    h6 = _hamiltonian(6, physical_bonds(COORDS6, j2, cross_scale, anisotropy))
    projected = project_block(six, h6.toarray())
    h, full_bonds, exact_energy, exact, reference_seconds = _twelve_reference(j2, cross_scale, anisotropy)
    cross = [(i, j-6, value) for i, j, value in full_bonds if i < 6 <= j]
    start = perf_counter()
    merged = merge_projected_blocks(projected, projected, cross, retain={0: 1})
    psi = merged.block.isometry.sectors[0][:, 0, 0]
    hp = h @ psi
    energy = float(np.vdot(psi, hp).real)
    variance = float(np.vdot(hp, hp).real-energy**2)
    if variance < -1e-11*max(1., float(np.vdot(hp, hp).real)):
        raise ArithmeticError("physical variance is negative beyond roundoff")
    reduced = psi.reshape(64, 64) @ psi.reshape(64, 64).conj().T
    weights = {q: float(sum(np.trace(base[q][:, :, k].T @ reduced @ base[q][:, :, k]).real
                            for k in range(q+1))) for q in base}
    return dict(j2=j2, cross_scale=cross_scale, anisotropy=anisotropy, energy_per_site=energy/12,
                variance_per_site=max(0., variance)/12,
                raw_variance_per_site=variance/12,
                variance_roundoff_clamped=variance < 0.,
                compressed_H_variance_per_site=float(np.linalg.norm(
                    merged.block.hamiltonian-merged.block.hamiltonian[0, 0])**2)/12,
                exact_reference_energy_per_site=exact_energy/12,
                exact_squared_overlap=float(abs(np.vdot(exact, psi))**2),
                q_weights=weights, q0_only=q0_only, retained_dimension=len(projected.labels),
                merge_and_full_H_seconds=perf_counter()-start,
                common_cached_physical_reference_seconds=reference_seconds,
                maximum_CG_or_scalar_error=max(float(merged.diagnostics[k]) for k in
                                               ("cg_unitarity_error", "scalar_error")))


def saved_ct9_bridge(output: Path) -> dict:
    """Export actual saved w20 CT local multiplets, H and boundary spins."""
    from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
    from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
    from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
    source = ROOT/"projects/clebsch_tree_nqs/results/remote_review_20261021/optimized_endpoints/ct6_continue_20261021/j0125_binary_truncated_w20_continue32/checkpoint.pkl"
    if not source.exists():
        return dict(available=False, expected_source=str(source))
    start = perf_counter(); raw = load_checkpoint(source)["params"]
    config = MixedClebschConfig(6, 20, 10)
    bank = arity_clebsch_bank(raw, config, "binary_truncated", 8)
    bank, diagnostic = canonicalize_triangular_bank(bank, config)
    local = bank["local"][0]; tables = triangle_block_tables()["nine"]
    mags = np.asarray([2*i.bit_count()-9 for i in range(512)])
    sectors = {q: np.stack([(tables[q] @ np.asarray(w))*(mags == m)[:, None]
                           for m in range(-q, q+1, 2)], axis=-1)
               for q, w in zip(config.local_q, local, strict=True)}
    iso, _, qr = canonicalize_block(9, sectors)
    coords = triangular_schur_hierarchy(6).blocks[0].lifted_coordinates
    h = _hamiltonian(9, physical_bonds(coords, .125))
    projected = project_block(iso, h.toarray())
    np.savez_compressed(output/"saved_ct9_operators.npz", hamiltonian=projected.hamiltonian,
        site_spin=np.asarray(projected.site_spin), labels=np.asarray(projected.labels),
        **{f"U_q{q}": value for q, value in iso.sectors.items()})
    return dict(available=True, source=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), q_widths={q: u.shape[1] for q, u in iso.sectors.items()},
        retained_dimension=len(projected.labels), seconds=perf_counter()-start,
        canonical_diagnostics=diagnostic, block_QR=qr, projection_diagnostics=projected.diagnostics,
        full_global_VMC_H_calls=0, local_H_projection_count=1, local_site_vector_matrices=27,
        statement="One raw local block; no global A1 environment or production-state change.")


def twelve_singlet_basis() -> np.ndarray:
    """Complete 132-column six-plus-six Schur singlet reference, N=12 only."""
    columns = []
    for q, u in sorted(schur_multiplets(6).items()):
        for a in range(u.shape[1]):
            for b in range(u.shape[1]):
                value = sum(((-1)**((q-m)//2)/np.sqrt(q+1)*np.kron(
                    u[:, a, (m+q)//2], u[:, b, (-m+q)//2])
                    for m in range(-q, q+1, 2)), np.zeros(4096))
                columns.append(value)
    basis = np.column_stack(columns)
    if np.max(abs(basis.T @ basis-np.eye(132))) > 2e-13:
        raise ArithmeticError("complete twelve-spin Schur singlet columns are not orthonormal")
    return basis


def tree_coefficient_vector(theta: jax.Array) -> jax.Array:
    """Normalized complete-reference coefficients of the same 78-parameter tree.

    Both six-spin frames and the singlet root are trained. The four retained
    multiplicities are (2,2,1,1), and full source multiplicities (5,9,5,1).
    """
    theta = jnp.asarray(theta)
    if theta.shape != (78,):
        raise ValueError("the fixed-capacity tree has exactly 78 real parameters")
    left = []; right = []; offset = 0
    for destination in (left, right):
        for source, keep in ((5, 2), (9, 2), (5, 1), (1, 1)):
            destination.append(theta[offset:offset+source*keep].reshape(source, keep))
            offset += source*keep
    vectors = []
    for l, r, keep in zip(left, right, (2, 2, 1, 1), strict=True):
        root = theta[offset:offset+keep*keep].reshape(keep, keep)
        offset += keep*keep
        vectors.append((l @ root @ r.T).reshape(-1))
    result = jnp.concatenate(vectors)
    return result/jnp.linalg.norm(result)


def initial_tree_parameters(frames: dict[int, np.ndarray], h_reduced: np.ndarray,
                            *, q0_only: bool = False) -> np.ndarray:
    """Ritz-optimize the initial root, preserving every nonzero child frame."""
    maps = block_diag(*[np.kron(frames[q], frames[q]) for q in (0, 2, 4, 6)])
    root_h = maps.T @ h_reduced @ maps
    if q0_only:
        root = np.zeros(10); root[:4] = np.linalg.eigh(root_h[:4, :4])[1][:, 0]
    else:
        root = np.linalg.eigh(root_h)[1][:, 0]
    local = np.concatenate([frames[q].reshape(-1) for q in (0, 2, 4, 6)])
    return np.r_[local, local, root]


def refinement_measure(theta: np.ndarray, h_reduced: np.ndarray,
                       h_physical_basis: np.ndarray) -> tuple[dict, np.ndarray]:
    """Physical energy and variance using full H Phi, with exact physical FS."""
    coefficients = np.asarray(tree_coefficient_vector(jnp.asarray(theta)))
    energy = float(coefficients @ h_reduced @ coefficients)
    hp = h_physical_basis @ coefficients
    raw_variance = float(np.vdot(hp, hp).real-energy**2)
    if raw_variance < -1e-11*max(1., float(np.vdot(hp, hp).real)):
        raise ArithmeticError("full physical variance became negative")
    return dict(energy_per_site=energy/12, variance_per_site=max(0., raw_variance)/12,
                raw_variance_per_site=raw_variance/12), coefficients


def make_refinement_step(h_reduced: np.ndarray, h_physical_basis: np.ndarray):
    """Exact FS natural gradient with a fixed ridge and finite-step guard."""
    coefficient_and_jacobian = jax.jit(lambda t: (
        tree_coefficient_vector(t), jax.jacfwd(tree_coefficient_vector)(t)))

    def step(theta: np.ndarray) -> tuple[np.ndarray, dict]:
        current, v = refinement_measure(theta, h_reduced, h_physical_basis)
        _, jacobian = coefficient_and_jacobian(jnp.asarray(theta))
        jacobian = np.asarray(jacobian)
        fisher = jacobian.T @ jacobian
        force = jacobian.T @ (h_reduced @ v-12*current["energy_per_site"]*v)
        ridge = 1e-3*max(float(np.trace(fisher))/78, 1e-12)
        direction = -np.linalg.solve(fisher+ridge*np.eye(78), force)
        tangent = float(np.sqrt(max(0., direction @ fisher @ direction)))
        direction *= min(1., .02/max(tangent, 1e-300))
        chosen = theta; chosen_info = current; accepted = False; actual_fs = 0.
        evaluations = 1
        for backtrack in range(9):
            candidate = theta+(2.**(-backtrack))*direction
            info, value = refinement_measure(candidate, h_reduced, h_physical_basis)
            evaluations += 1
            fs = float(np.sqrt(max(0., 1-float(np.dot(v, value))**2)))
            if np.isfinite(info["energy_per_site"]) and fs <= .020000000001 and (
                    info["energy_per_site"] <= current["energy_per_site"]-1e-15):
                chosen, chosen_info, accepted, actual_fs = candidate, info, True, fs
                break
        return chosen, dict(**chosen_info, accepted=accepted, actual_FS_sine=actual_fs,
            ridge=ridge, parameter_gradient_norm=float(np.linalg.norm(2*force)),
            cached_physical_Hpsi_reconstructions=evaluations,
            complete_reference_coefficients_used_only_for_exact_N12_metric=True)
    return step


def run_refinement(source: Path, output: Path, steps: int = 12) -> dict:
    """Frozen-budget followup of ALL v1 starts, including every learned seed."""
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 16:
        raise ValueError("followup budget must be between 1 and 16 updates")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing followup")
    old = json.loads((source/"results.json").read_text())
    dataset = np.load(source/"dataset.npz")
    cases = json.loads((source/"cases.json").read_text())
    targets = {str(j): next(row for row in old["physical_evaluation"]
                           if row["j2"] == j and row["arm"] == "physical") for j in (0., .125)}
    registration = dict(task="bounded_same_capacity_78_parameter_tree_refinement_v1",
        workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_results_sha256=hashlib.sha256((source/"results.json").read_bytes()).hexdigest(),
        geometry="4x3 OPEN triangular cluster; isotropic J1/J2; no global A1 projection",
        couplings=(0., .125), retained_multiplicities=RETAIN, parameters=78,
        optimizer="exact physical FS natural gradient, ridge=.001*trace(F)/78",
        maximum_FS_sine=.02, backtrack_attempts=9, fixed_updates=steps,
        seeds=SEEDS, targets=targets, target_frozen_from_v1_not_reselected=True,
        full_variance="H Phi coefficients in 4096 physical rows, not retained Ritz H squared",
        work_units="132 physical H columns once per target; subsequent Hpsi are exact cached reconstructions, not new H queries",
        shared_exact_reference_H_columns=132,
        arms=("copy", "physical", "random", "physical_q0_start_unlocked",
              "linear", "mlp", "attention", "scrambled_attention"),
        model_selection="all seeds and all starts retained; no controller retraining")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"registration.json", registration)
    started = perf_counter(); basis = twelve_singlet_basis(); reports = []; endpoints = {}
    setup_started = perf_counter(); data = build_dataset(); setup_seconds = perf_counter()-setup_started
    shared_seconds = 0.
    for j2 in (0., .125):
        stage_started = perf_counter()
        h, _, exact_energy, _, _ = _twelve_reference(j2, 1., 0.)
        hp_basis = h @ basis
        reduced = basis.T @ hp_basis
        shared_h_seconds = perf_counter()-stage_started; shared_seconds += shared_h_seconds
        step = make_refinement_step(reduced, hp_basis)
        case = next(i for i, row in enumerate(cases) if row["j2"] == j2
                    and row["cross_scale"] == 1. and row["anisotropy"] == 0.)
        candidates = {"copy": data["parent_frames"],
                      "physical": _frames_from_projectors(dataset["targets"][case], data["dimensions"])}
        rng = np.random.default_rng(26102390)
        candidates["random"] = {q: np.linalg.qr(rng.normal(size=(u.shape[1], RETAIN[q])))[0]
                                for q, u in data["multiplets"].items()}
        costs = {name: dict(teacher_dataset_seconds=0., controller_training_seconds=0.) for name in candidates}
        costs["physical"]["teacher_dataset_seconds"] = old["teacher_seconds"]
        for kind in ("linear", "mlp", "attention", "scrambled_attention"):
            for seed in SEEDS:
                inference_started = perf_counter()
                saved = load_checkpoint(source/f"{kind}_seed{seed}.pkl")
                meta = saved["metadata"]
                x = (dataset["inputs"][case]-meta["mean"])/meta["scale"]
                prediction = _projector_vector(data["parent_frames"])+np.asarray(
                    controller(saved["params"], jnp.asarray(x), kind))
                name = f"{kind}_seed{seed}"
                candidates[name] = _frames_from_projectors(prediction, data["dimensions"])
                fit = next(row for row in old["training"] if row["kind"] == kind and row["seed"] == seed)
                costs[name] = dict(teacher_dataset_seconds=old["teacher_seconds"],
                    controller_training_seconds=fit["training_seconds"],
                    controller_restore_and_inference_seconds=perf_counter()-inference_started)
        candidates["physical_q0_start_unlocked"] = candidates["physical"]
        costs["physical_q0_start_unlocked"] = costs["physical"]
        for arm, frames in candidates.items():
            arm_started = perf_counter()
            theta = initial_tree_parameters(frames, reduced, q0_only=arm == "physical_q0_start_unlocked")
            initial_theta = theta.copy()
            initial, coefficients = refinement_measure(theta, reduced, hp_basis)
            original_arm = "physical_q0_only" if arm == "physical_q0_start_unlocked" else arm
            original = next(row for row in old["physical_evaluation"] if row["j2"] == j2 and row["arm"] == original_arm)
            replay_error = max(abs(initial[k]-original[k]) for k in ("energy_per_site", "variance_per_site"))
            if replay_error > 1e-11:
                raise ArithmeticError("v1 physical initial-state moments do not replay")
            initial_psi = basis @ coefficients
            history = [dict(step=0, elapsed_seconds=perf_counter()-arm_started,
                            cumulative_cached_Hpsi_reconstructions=1, **initial)]
            evaluations = 1
            for iteration in range(1, steps+1):
                theta, info = step(theta); evaluations += info["cached_physical_Hpsi_reconstructions"]
                history.append(dict(step=iteration, elapsed_seconds=perf_counter()-arm_started,
                                    cumulative_cached_Hpsi_reconstructions=evaluations, **info))
            checkpoint = output/f"j{str(j2).replace('.', 'p')}_{arm}.pkl"
            save_checkpoint(checkpoint, theta, optimizer_state=dict(completed_steps=steps,
                algorithm="stateless exact natural gradient", RNG=np.random.default_rng(26102340).bit_generator.state),
                metadata=dict(arm=arm, j2=j2, initial_theta=initial_theta, initial_physical_psi=initial_psi))
            restored = load_checkpoint(checkpoint)
            continued, replay_info = step(restored["params"]); direct, direct_info = step(theta)
            restore_error = float(np.max(abs(continued-direct)))
            final, final_coefficients = refinement_measure(theta, reduced, hp_basis)
            endpoints[f"j{j2}_{arm}_initial_theta"] = initial_theta
            endpoints[f"j{j2}_{arm}_theta"] = theta
            endpoints[f"j{j2}_{arm}_initial_psi"] = initial_psi
            endpoints[f"j{j2}_{arm}_psi"] = basis @ final_coefficients
            target = targets[str(j2)]
            crossing = next((row for row in history if row["energy_per_site"] <= target["energy_per_site"]+1e-12), None)
            joint = next((row for row in history if row["energy_per_site"] <= target["energy_per_site"]+1e-12
                          and row["variance_per_site"] <= target["variance_per_site"]+1e-12), None)
            overhead = sum(costs[arm].values())+shared_h_seconds
            report = dict(arm=arm, j2=j2, **costs[arm], shared_132_H_columns_seconds=shared_h_seconds,
                initial_moment_replay_error=replay_error, next_step_restore_error=restore_error,
                history=history, initial=initial, final=final,
                exact_reference_energy_per_site=exact_energy/12,
                first_energy_target_step=None if crossing is None else crossing["step"],
                first_joint_energy_variance_target_step=None if joint is None else joint["step"],
                total_CPU_seconds_to_energy_target=None if crossing is None else overhead+crossing["elapsed_seconds"],
                shared_new_N12_physical_H_columns=132,
                cached_Hpsi_reconstructions_to_energy_target=None if crossing is None else crossing["cumulative_cached_Hpsi_reconstructions"],
                uncommitted_restore_cached_Hpsi_reconstructions=replay_info["cached_physical_Hpsi_reconstructions"]+direct_info["cached_physical_Hpsi_reconstructions"],
                final_checkpoint_and_restore_seconds=perf_counter()-arm_started)
            reports.append(report)
            print("refined", j2, arm, initial["energy_per_site"], final["energy_per_site"],
                  report["first_energy_target_step"], flush=True)
    np.savez_compressed(output/"physical_endpoints.npz", **endpoints)
    result = dict(registration=registration, reports=reports,
        shared_reference_seconds=shared_seconds, fixed_dataset_reconstruction_seconds=setup_seconds,
        total_seconds=perf_counter()-started,
        interpretation="All 78 parameters trained for exactly the fixed number of attempts; root-only Ritz was initialization. CPU/full-H-vector accounting is an exact N12 reference, not MC-center or A100 throughput. Shared132 physical H columns charged once per coupling; controller teacher and fit costs retained.")
    atomic_json(output/"results.json", result)
    return result


def run(output: Path, steps: int = 200) -> dict:
    """Register, execute and retain every fixed-seed local result."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing experiment")
    output.mkdir(parents=True, exist_ok=True)
    registration = dict(task="bounded_open_cluster_SU2_merge_and_projector_transfer_v1",
        geometry="2x3 six-spin blocks; 4x3 OPEN twelve-spin target; no A1 projection",
        retain=RETAIN, seeds=SEEDS, steps=steps,
        train_j2=(0., .05, .125, .175), train_cross_scale=(.25, .6, 1.),
        heldout_j2=(.075, .15), heldout_cross_scale=(.425, .85), anisotropy=(-.12, .12),
        controls=("linear", "mlp", "attention", "scrambled_attention"),
        final_cases=((.075, .425, -.12), (.15, .85, -.12), (0., 1., 0.), (.125, 1., 0.)),
        target="retained projectors, fixed physical Schur frame; same fixed capacity",
        physical_variance="full sparse Hpsi, not the square of compressed H",
        no_production_or_size_scaling_claim=True)
    atomic_json(output/"registration.json", registration)
    started = perf_counter(); data = build_dataset(); teacher_seconds = perf_counter()-started
    np.savez_compressed(output/"dataset.npz", inputs=data["inputs"], targets=data["targets"],
                        split=np.asarray([x["split"] for x in data["notes"]]))
    atomic_json(output/"cases.json", data["notes"])
    predictions = {}; training = []
    for kind in registration["controls"]:
        for seed in SEEDS:
            predicted, report = fit_controller(data, kind, seed, steps, output)
            predictions[kind, seed] = predicted; training.append(report)
            print(kind, seed, report["heldout_projector_error"], flush=True)
    evaluation = []
    for j2, lam, aniso in registration["final_cases"]:
        case = next(i for i, row in enumerate(data["notes"])
                    if row["j2"] == j2 and row["cross_scale"] == lam and row["anisotropy"] == aniso)
        candidates = {"copy": data["parent_frames"], "physical": data["frames"][case]}
        rng = np.random.default_rng(26102390)
        candidates["random"] = {q: np.linalg.qr(rng.normal(size=(u.shape[1], RETAIN[q])))[0]
                                for q, u in data["multiplets"].items()}
        for kind in registration["controls"]:
            for seed in SEEDS:
                candidates[f"{kind}_seed{seed}"] = _frames_from_projectors(predictions[kind, seed][case], data["dimensions"])
        for name, frames in candidates.items():
            result = evaluate_twelve(frames, j2, lam, aniso)
            evaluation.append(dict(arm=name, **result)); print("physical", name, j2, result["energy_per_site"], flush=True)
        evaluation.append(dict(arm="physical_q0_only", **evaluate_twelve(candidates["physical"], j2, lam, aniso, q0_only=True)))
    bridge = saved_ct9_bridge(output)
    result = dict(registration=registration, teacher_seconds=teacher_seconds, training=training,
        physical_evaluation=evaluation, saved_CT9_bridge=bridge,
        total_seconds=perf_counter()-started, new_full_global_VMC_H_calls=0,
        interpretation="Small OBC reference only. Learned inference uses no teacher H projection; all teacher/training costs reported. No automatic production integration.")
    atomic_json(output/"results.json", result)
    return result


def reference_site_spins(n_sites: int) -> tuple[tuple[np.ndarray, ...], ...]:
    """Dense x,y,z reference operators, 0=down and site zero MSB; N<=6."""
    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or not 1 <= n_sites <= 6:
        raise ValueError("bounded reference requires 1<=N<=6")
    d = 1 << n_sites; states = np.arange(d); answer = []
    for site in range(n_sites):
        mask = 1 << (n_sites-1-site); down = states[(states & mask) == 0]
        plus = np.zeros((d, d), complex); plus[down | mask, down] = 1.
        answer.append(((plus+plus.T)/2, (plus-plus.T)/(2j),
                       np.diag(np.where((states & mask) == 0, -.5, .5))))
    return tuple(answer)


def boundary_residual_coefficients(
    coefficients: np.ndarray, left_spin: tuple, right_spin: tuple,
    bonds: list[tuple[int, int, float]], *, center: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Exact bounded boundary action on a physical ket coefficient matrix.

    ``C[a,b]`` multiplies |a>_A|b>_B, so right action is S.T, NOT S.H.
    The centered vector is (V-<V>)|psi>, not the complete H residual.
    """
    c = np.asarray(coefficients)
    if c.ndim != 2 or max(c.shape) > 64 or not np.isfinite(c).all():
        raise ValueError("finite bounded physical coefficient matrix required")
    norm = float(np.vdot(c, c).real)
    if norm <= 0:
        raise ValueError("nonzero parent norm required")
    action = np.zeros_like(c, dtype=np.result_type(c, complex))
    for i, j, coupling in bonds:
        if (not 0 <= i < len(left_spin) or not 0 <= j < len(right_spin)
                or not np.isfinite(coupling) or np.iscomplexobj(coupling)):
            raise ValueError("finite real boundary coupling and valid local indices required")
        for a, b in zip(left_spin[i], right_spin[j], strict=True):
            if a.shape != (c.shape[0],)*2 or b.shape != (c.shape[1],)*2:
                raise ValueError("site operators do not match coefficient dimensions")
            action += coupling*(a @ c @ b.T)
    expectation = np.vdot(c, action)/norm
    if abs(expectation.imag) > 2e-12*(1+abs(expectation)):
        raise ArithmeticError("Hermitian boundary action has complex expectation")
    residual = action-expectation.real*c if center else action
    return residual, dict(parent_norm=norm, boundary_expectation=float(expectation.real),
                          boundary_action_norm_squared=float(np.vdot(action, action).real),
                          residual_norm_squared=float(np.vdot(residual, residual).real))


def multiplicity_density(rho: np.ndarray, multiplets: dict[int, np.ndarray]) -> tuple[dict, dict]:
    """Trace complete doubled-q magnetic indices, verify scalar reconstruction.

    Returns sigma_q with Tr(sigma_q)=sector weight, so rho_q=I/(q+1)*sigma_q.
    Only an exact <=64-dimensional reference is supported.
    """
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or len(rho) > 64 or not np.isfinite(rho).all():
        raise ValueError("finite square reference density of dimension<=64 required")
    scale = max(float(np.linalg.norm(rho)), np.finfo(float).tiny)
    if np.linalg.norm(rho-rho.conj().T) > 2e-11*scale:
        raise ValueError("density must be Hermitian")
    blocks = {}; reconstruction = np.zeros_like(rho, dtype=complex)
    for q, u in sorted(multiplets.items()):
        if u.shape[0] != len(rho) or u.shape[2] != q+1:
            raise ValueError("complete magnetic columns must match doubled q")
        block = sum((u[:, :, m].conj().T @ rho @ u[:, :, m] for m in range(q+1)),
                    np.zeros((u.shape[1],)*2, complex))
        blocks[q] = (block+block.conj().T)/2
        reconstruction += sum((u[:, :, m] @ blocks[q] @ u[:, :, m].conj().T/(q+1)
                               for m in range(q+1)), np.zeros_like(rho, dtype=complex))
    error = float(np.linalg.norm(rho-reconstruction)/scale)
    if error > 2e-10:
        raise ArithmeticError("density does not obey strict SU(2) scalar decomposition")
    return blocks, dict(scalar_reconstruction_relative_error=error,
                       trace=float(np.trace(rho).real),
                       sector_weights={int(q): float(np.trace(b).real) for q, b in blocks.items()})


def _boundary_product_density(rho: np.ndarray, other_rho: np.ndarray,
                              spins: tuple, other_spins: tuple,
                              couplings: np.ndarray, *, diagonal_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """Product bath approximation, with actual inter-boundary spin correlations.

    K=J G J.T, G_jk=<S_j dot S_k>/3. All physical components are used;
    no highest-weight conversion factors are needed. diagonal_only isolates
    the independent-site approximation with identical diagonal weights.
    """
    g = np.asarray([[sum(np.trace(other_rho @ (a @ b)) for a, b in zip(x, y, strict=True))/3
                     for y in other_spins] for x in other_spins])
    if np.max(abs(g.imag)) > 2e-12:
        raise ArithmeticError("scalar spin covariance unexpectedly complex")
    k = couplings @ g.real @ couplings.T
    k = (k+k.T)/2
    if np.linalg.eigvalsh(k).min(initial=0) < -2e-11:
        raise ArithmeticError("bath covariance is not positive semidefinite")
    selected = np.diag(np.diag(k)) if diagonal_only else k
    answer = np.zeros_like(rho, dtype=complex)
    for i, si in enumerate(spins):
        for j, sj in enumerate(spins):
            if selected[i, j] != 0:
                for a, b in zip(si, sj, strict=True):
                    answer += selected[i, j]*(a @ rho @ b.conj().T)
    return (answer+answer.conj().T)/2, k


def _complete_density_factor(rho: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((rho+rho.conj().T)/2)
    tolerance = 2e-11*max(float(np.linalg.norm(rho)), np.finfo(float).tiny)
    if values.min(initial=0) < -tolerance:
        raise ArithmeticError("response density has negative eigenvalue beyond roundoff")
    return vectors*np.sqrt(np.maximum(values, 0.))


def _boundary_probe_ritz(left: dict, right: dict, full_basis: np.ndarray,
                         full_h_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    maps = block_diag(*[np.kron(left[q], right[q]) for q in sorted(left)])
    gram_error = float(np.max(abs(maps.conj().T @ maps-np.eye(maps.shape[1]))))
    if gram_error > 2e-11:
        raise ArithmeticError("retained singlet columns are not isometric")
    reduced = maps.conj().T @ (full_basis.conj().T @ full_h_basis) @ maps
    values, vectors = np.linalg.eigh((reduced+reduced.conj().T)/2)
    coefficients = maps @ vectors[:, 0]
    psi, hp = full_basis @ coefficients, full_h_basis @ coefficients
    e = np.vdot(psi, hp)/np.vdot(psi, psi)
    variance = float(np.vdot(hp-e*psi, hp-e*psi).real/np.vdot(psi, psi).real)
    return coefficients, maps, dict(energy_per_site=float(e.real/12),
        imaginary_energy_per_site=float(e.imag/12), variance_per_site=variance/12,
        retained_singlet_dimension=maps.shape[1], isometry_error=gram_error,
        root_Ritz_residual=float(np.linalg.norm(reduced @ vectors[:, 0]-values[0]*vectors[:, 0])),
        variance_is_full_physical_H_not_compressed_H_squared=True)


def fixed_gap_boundary_correction(h0: np.ndarray, retained: np.ndarray, response: np.ndarray,
                                 *, reference_energy: float, gap: float = 1.) -> tuple[np.ndarray, dict]:
    """Q(H0-Eref+gap)^-1 Q response in a complete small singlet reference.

    The protected family must commute with H0. This is a fixed positive
    excitation-cost filter, NOT a ground-state solve or energy guarantee.
    """
    h0, retained, response = np.asarray(h0), np.asarray(retained), np.asarray(response)
    if response.ndim != 1:
        raise ValueError("one finite response vector required")
    n = len(response)
    if (not 1 <= n <= 132 or h0.shape != (n,n)
            or retained.ndim != 2 or retained.shape[0] != n or not 0 < retained.shape[1] < n
            or not all(np.isfinite(x).all() for x in (h0,retained,response))
            or not isinstance(reference_energy, (int, float, np.integer, np.floating))
            or not isinstance(gap, (int, float, np.integer, np.floating))
            or not np.isfinite(reference_energy) or not np.isfinite(gap) or gap <= 0):
        raise ValueError("finite bounded complete singlet matrices and positive fixed gap required")
    if (np.max(abs(h0-h0.conj().T)) > 2e-11
            or np.max(abs(retained.conj().T@retained-np.eye(retained.shape[1]))) > 2e-11):
        raise ValueError("Hermitian H0 and orthonormal protected family required")
    p = retained@retained.conj().T; q = np.eye(n)-p
    commutator = float(np.linalg.norm(h0@p-p@h0))
    if commutator > 2e-10*max(1.,float(np.linalg.norm(h0))):
        raise ValueError("protected low-energy family must commute with H0")
    operator = h0+(gap-reference_energy)*np.eye(n)
    lower = float(np.linalg.eigvalsh(operator).min())
    if lower <= 0:
        raise ValueError("fixed resolvent denominator must be positive definite")
    rhs = q@response
    solved = np.linalg.solve(operator,rhs); delta = q@solved
    error = float(np.linalg.norm(q@(operator@delta)-rhs))
    if error > 2e-10*max(1.,float(np.linalg.norm(rhs))):
        raise ArithmeticError("projected resolvent solve failed")
    return delta,dict(reference_energy=float(reference_energy),fixed_gap=float(gap),
        minimum_denominator=lower,protected_H0_commutator=commutator,
        projected_solve_residual=error,protected_leakage=float(np.linalg.norm(retained.conj().T@delta)),
        unfiltered_response_norm_squared=float(np.vdot(rhs,rhs).real),
        filtered_response_norm_squared=float(np.vdot(delta,delta).real))


def run_boundary_environment_probe(output: Path, seconds_cap: float = 120.) -> dict:
    """One registered N12 exact comparison of four equally expanded q=2 frames.

    Parent low-energy frames (2,2,1,1) remain wholly contained in every arm;
    only doubled q=2 expands from2 to4. Each arm reoptimizes its singlet root
    in the same22-dimensional family. There is no new full ground-state ED,
    sampling, learning, attention, torus A1 or production update.
    """
    if not np.isfinite(seconds_cap) or not 0 < seconds_cap <= 120:
        raise ValueError("numerical wall cap must lie in(0,120] seconds")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite existing experiment")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes(); source_hash = hashlib.sha256(source).hexdigest()
    # Snapshot is a registered source artifact, not executable code generation.
    from shutil import copyfile
    copyfile(__file__, output/"registered_workflow.py")
    dependencies = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        ROOT/"su2_nqs/frontier/retained_boundary_response.py",
        ROOT/"su2_nqs/su2/triangle_blocks.py")}
    arms = ("isolated_low_energy", "independent_site", "geometry_product_bath", "coherent_boundary_residual")
    registration = dict(task="exact_small_correlated_boundary_environment_v1", source_sha256=source_hash,
        dependencies_sha256=dependencies, numerical_seconds_cap=seconds_cap, j2=(0., .125),
        geometry="4x3 OPEN triangular12; same six-plus-six cut and physical bond ledger",
        parent_retained_multiplicities=RETAIN, new_retained_multiplicities={0: 2, 2: 4, 4: 1, 6: 1},
        arms=arms, parent_embedding="all original directions retained exactly",
        independent_site="diagonal of same product-bath K; matched diagonal covariance weights",
        coherent="full (V-<V>) parent ket, all occupied source q, not product marginals",
        no_new_ground_ED=True, no_MC=True, no_attention=True, no_A1_projection=True,
        root_Ritz_dimension=22, complex_gauge_test_seed=2610234901,
        source_q_diagnostics="target q2 unretained response from source q0,2,4,6; include interference")
    atomic_json(output/"registration.json", registration)
    started = perf_counter(); previous = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame):
        raise TimeoutError("registered120second reference budget exhausted")
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, seconds_cap)
    result = dict(registration=registration, cases=[], new_global_MC_H_centers=0,
                  physical_H_reference_columns=0, boundary_action_vectors=0, status="started")
    arrays = {}
    try:
        base = schur_multiplets(6); spins = reference_site_spins(6)
        full_basis = twelve_singlet_basis()
        generator = sum((s[0]+1j*s[1] for s in spins), np.zeros((64, 64), complex))
        for j2 in registration['j2']:
            case_started = perf_counter(); coords = COORDS6+tuple((r+2, c) for r, c in COORDS6)
            bonds = physical_bonds(coords, j2); h = _hamiltonian(12, bonds)
            cross = [(i, j-6, coupling) for i, j, coupling in bonds if i < 6 <= j and coupling != 0]
            h6 = _hamiltonian(6, physical_bonds(COORDS6, j2)).toarray()
            eigframes = {q: np.linalg.eigh(u[:, :, -1].conj().T @ h6 @ u[:, :, -1])[1]
                         for q, u in base.items()}
            old = {q: eigframes[q][:, :RETAIN[q]].copy() for q in base}
            low = {q: eigframes[q][:, :(4 if q == 2 else RETAIN[q])].copy() for q in base}
            hp_basis = h @ full_basis; result['physical_H_reference_columns'] += full_basis.shape[1]
            parent, parent_map, parent_report = _boundary_probe_ritz(old, old, full_basis, hp_basis)
            c = (full_basis @ parent).reshape(64, 64)
            rho_a, rho_b = c @ c.conj().T, c.T @ c.conj()
            residual, residual_info = boundary_residual_coefficients(c, spins, spins, cross)
            result['boundary_action_vectors'] += 1
            d_a, d_b = residual @ residual.conj().T, residual.T @ residual.conj()
            couplings = np.zeros((6, 6))
            for i, j, value in cross:
                couplings[i, j] += value
            densities = {'coherent_boundary_residual': (d_a, d_b)}; kernels = []
            for name, diagonal in (("independent_site", True), ("geometry_product_bath", False)):
                a, ka = _boundary_product_density(rho_a, rho_b, spins, spins, couplings, diagonal_only=diagonal)
                b, kb = _boundary_product_density(rho_b, rho_a, spins, spins, couplings.T, diagonal_only=diagonal)
                densities[name] = (a, b)
                if not diagonal:
                    kernels = [ka, kb]
            candidates = {'isolated_low_energy': (low, low)}; scalar_errors = []; density_weights = {}
            for name, pair in densities.items():
                frames = []; density_weights[name] = []
                for density in pair:
                    blocks, scalar = multiplicity_density(density, base)
                    scalar_errors.append(scalar['scalar_reconstruction_relative_error'])
                    density_weights[name].append(scalar['sector_weights'])
                    frame = {q: value.copy() for q, value in old.items()}
                    frame[2] = fixed_q_response_frame(2, low[2], _complete_density_factor(blocks[2]), protect=2).frame
                    frames.append(frame)
                candidates[name] = tuple(frames)
            unretained = sum((base[2][:, :, m] @ (np.eye(9)-old[2] @ old[2].conj().T)
                             @ base[2][:, :, m].conj().T for m in range(3)), np.zeros((64,64), complex))
            source_vectors = []
            for q, u in sorted(base.items()):
                pq = sum((u[:, :, m] @ u[:, :, m].conj().T for m in range(q+1)), np.zeros((64,64),complex))
                source_c = pq @ c
                acted, _ = boundary_residual_coefficients(source_c, spins, spins, cross, center=False)
                result['boundary_action_vectors'] += 1
                acted -= residual_info['boundary_expectation']*source_c
                source_vectors.append(unretained @ acted)
            interference = np.asarray([[np.vdot(x,y) for y in source_vectors] for x in source_vectors])
            sum_error = float(np.linalg.norm(sum(source_vectors)-unretained @ residual))
            case = dict(j2=j2, physical_bonds=bonds, cross_bonds=cross, parent=parent_report,
                boundary_residual=residual_info, maximum_density_SU2_error=max(scalar_errors),
                response_sector_weights=density_weights, arms=[],
                source_q_order=sorted(base), source_q_response_Gram_real=interference.real.tolist(),
                source_q_response_Gram_imag=interference.imag.tolist(),
                source_sum_error=sum_error, source_diagonal_norm=float(np.trace(interference).real),
                source_interference_norm=float(interference.sum().real-np.trace(interference).real),
                source_q6_forbidden_norm=float(interference[-1,-1].real))
            tag = 'j0' if j2 == 0 else 'j0125'
            arrays[f'{tag}_parent'] = c; arrays[f'{tag}_boundary_residual'] = residual
            arrays[f'{tag}_K_left'], arrays[f'{tag}_K_right'] = kernels
            for name in arms:
                left, right = candidates[name]
                coeff, maps, report = _boundary_probe_ritz(left, right, full_basis, hp_basis)
                embedded = maps @ (maps.conj().T @ parent)
                embedding_error = float(np.linalg.norm(embedded-parent))
                embedding_fidelity = float(abs(np.vdot(parent,embedded))**2/np.vdot(embedded,embedded).real)
                if embedding_error > 2e-11 or report['energy_per_site'] > parent_report['energy_per_site']+2e-12:
                    raise ArithmeticError("same-width arm lost parent or failed variational nesting")
                psi = (full_basis @ coeff).reshape(64,64)
                total_plus_error = float(np.linalg.norm(generator @ psi+psi @ generator.T))
                pa = sum((base[2][:, :, m] @ left[2] @ left[2].conj().T @ base[2][:, :, m].conj().T
                          for m in range(3)),np.zeros((64,64),complex))
                pb = sum((base[2][:, :, m] @ right[2] @ right[2].conj().T @ base[2][:, :, m].conj().T
                          for m in range(3)),np.zeros((64,64),complex))
                target_a = sum((base[2][:,:,m] @ base[2][:,:,m].conj().T for m in range(3)),np.zeros((64,64),complex))
                q2_response = target_a @ residual @ target_a.T
                captured = pa @ residual @ pb.T
                denominator = float(np.linalg.norm(q2_response)**2)
                case['arms'].append(dict(arm=name, **report, parent_embedding_error=embedding_error,
                    parent_embedding_fidelity=embedding_fidelity, total_Splus_error=total_plus_error,
                    boundary_q2_joint_capture=float(np.linalg.norm(captured)**2/denominator),
                    energy_change_per_site=report['energy_per_site']-parent_report['energy_per_site'],
                    variance_change_per_site=report['variance_per_site']-parent_report['variance_per_site']))
                arrays[f'{tag}_{name}_psi'] = psi
                for side, frames in (('left',left),('right',right)):
                    for q, frame in frames.items():
                        arrays[f'{tag}_{name}_{side}_q{q}'] = frame
            case['seconds'] = perf_counter()-case_started; result['cases'].append(case)
            atomic_json(output/'results.json', result)
        np.savez_compressed(output/'physical_states_and_frames.npz', **arrays)
        result.update(status='completed_exact_small_reference',
            interpretation='All four expanded arms contain the same parent and use the same22-dimensional root Ritz solve. Relative comparisons isolate selected q2 directions; energy gains over the narrower parent include added capacity. No thermodynamic, A1, attention, learned transfer or production quality claim.')
        return result
    except TimeoutError as error:
        result.update(status='bounded_incomplete', error=str(error)); return result
    except Exception as error:
        result.update(status='failed_validation', error=repr(error)); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        result['total_seconds'] = perf_counter()-started
        atomic_json(output/'results.json', result)


def run_boundary_resolvent_probe(reference: Path, output: Path, seconds_cap: float = 119.8) -> dict:
    """One serial, fixed-mu energy-aware extension of the frozen four-arm test."""
    if not np.isfinite(seconds_cap) or not 0 < seconds_cap <= 119.8:
        raise ValueError("remaining numerical budget must lie in(0,119.8] seconds")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite existing evidence")
    reference = Path(reference); source_result = json.loads((reference/'results.json').read_text())
    if source_result.get('status') != 'completed_exact_small_reference':
        raise ValueError("completed frozen four-arm reference required")
    source_data = reference/'physical_states_and_frames.npz'
    output.mkdir(parents=True,exist_ok=True)
    from shutil import copyfile
    copyfile(__file__,output/'registered_workflow.py')
    registration = dict(task='single_fixed_gap_boundary_resolvent_extension_v1',
        workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependencies_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            ROOT/'su2_nqs/frontier/retained_boundary_response.py',ROOT/'su2_nqs/su2/triangle_blocks.py')},
        previous_result=str(reference/'results.json'),
        previous_result_sha256=hashlib.sha256((reference/'results.json').read_bytes()).hexdigest(),
        previous_states=str(source_data),previous_states_sha256=hashlib.sha256(source_data.read_bytes()).hexdigest(),
        previous_numerical_seconds=source_result['total_seconds'],numerical_seconds_cap=seconds_cap,
        fixed_gap_mu_J1=1.,formula='delta=Q(H_A+H_B-2E6_ground+1)^(-1)Q V psi; Q removes whole old10-dimensional singlet family',
        capacity='same q2 multiplicity2to4, same22-dimensional root Ritz as all four frozen arms',
        no_outcome_selected_gap=True,no_new_N12_ground_ED=True,no_MC=True,no_attention=True,
        geometry=source_result['registration']['geometry'],j2=[c['j2'] for c in source_result['cases']])
    atomic_json(output/'registration.json',registration)
    started=perf_counter(); previous=signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('remaining fixed-gap reference budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds_cap)
    result=dict(registration=registration,cases=[],physical_full_H_reference_columns=0,
                internal_H0_reference_columns=0,new_boundary_action_vectors=0,status='started')
    arrays={}
    try:
        base=schur_multiplets(6);basis=twelve_singlet_basis()
        with np.load(source_data) as loaded:
            for oldcase in source_result['cases']:
                j2=oldcase['j2'];tag='j0' if j2==0 else 'j0125'
                old={};low={}
                for q in base:
                    low[q]=np.array(loaded[f'{tag}_isolated_low_energy_left_q{q}'])
                    old[q]=low[q][:,:RETAIN[q]].copy()
                old_maps=block_diag(*[np.kron(old[q],old[q]) for q in sorted(old)])
                parent=np.array(loaded[f'{tag}_parent']);boundary=np.array(loaded[f'{tag}_boundary_residual'])
                parent_coeff=basis.conj().T@parent.ravel();response_coeff=basis.conj().T@boundary.ravel()
                physical_bond_ledger=oldcase['physical_bonds']
                h0=_hamiltonian(12,[(i,j,v) for i,j,v in physical_bond_ledger if (i<6)==(j<6)])
                h0_basis=h0@basis;result['internal_H0_reference_columns']+=132
                h0_reduced=basis.conj().T@h0_basis
                h6=_hamiltonian(6,physical_bonds(COORDS6,j2))
                local_ground=base[0][:,:,0]@old[0][:,0]
                e6=float(np.vdot(local_ground,h6@local_ground).real)
                delta,filter_info=fixed_gap_boundary_correction(h0_reduced,old_maps,response_coeff,
                    reference_energy=2*e6,gap=1.)
                correction=(basis@delta).reshape(64,64)
                densities=(correction@correction.conj().T,correction.T@correction.conj())
                selected=[];scalar=[]
                for density in densities:
                    blocks,notes=multiplicity_density(density,base);scalar.append(notes)
                    frames={q:v.copy() for q,v in old.items()}
                    frames[2]=fixed_q_response_frame(2,low[2],_complete_density_factor(blocks[2]),protect=2).frame
                    selected.append(frames)
                h=_hamiltonian(12,physical_bond_ledger);hp_basis=h@basis
                result['physical_full_H_reference_columns']+=132
                coefficients,maps,quality=_boundary_probe_ritz(selected[0],selected[1],basis,hp_basis)
                embedding=maps@(maps.conj().T@parent_coeff)
                error=float(np.linalg.norm(embedding-parent_coeff))
                if error>2e-11 or quality['energy_per_site']>oldcase['parent']['energy_per_site']+2e-12:
                    raise ArithmeticError('fixed-gap arm failed exact parent containment')
                comparisons={a['arm']:dict(energy_per_site=quality['energy_per_site']-a['energy_per_site'],
                    variance_per_site=quality['variance_per_site']-a['variance_per_site']) for a in oldcase['arms']}
                result['cases'].append(dict(j2=j2,arm='fixed_gap_resolvent',**quality,
                    parent_embedding_error=error,parent_embedding_fidelity=float(abs(np.vdot(parent_coeff,embedding))**2/np.vdot(embedding,embedding).real),
                    filter=filter_info,scalar_density_checks=scalar,minus_each_frozen_arm=comparisons))
                arrays[f'{tag}_correction']=correction;arrays[f'{tag}_psi']=(basis@coefficients).reshape(64,64)
                for side,frames in zip(('left','right'),selected,strict=True):
                    for q,frame in frames.items():arrays[f'{tag}_{side}_q{q}']=frame
        np.savez_compressed(output/'states_and_frames.npz',**arrays)
        result.update(status='completed_single_fixed_gap_extension',
            interpretation='One fixed physical excitation-cost filter, not a gap search. Original four arms remain immutable; no full-A1 or production claim. No further filters released by this result.')
        return result
    except Exception as error:
        result.update(status='failed_or_bounded',error=repr(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result['total_seconds']=perf_counter()-started
        result['combined_with_original_probe_seconds']=source_result['total_seconds']+result['total_seconds']
        atomic_json(output/'results.json',result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--refine-from", type=Path)
    parser.add_argument("--boundary-environment", action="store_true")
    parser.add_argument("--boundary-resolvent-from", type=Path)
    parser.add_argument("--seconds-cap", type=float, default=120.)
    args = parser.parse_args()
    if args.boundary_resolvent_from is not None:
        if args.refine_from is not None or args.boundary_environment:
            parser.error("one frozen experiment mode at a time")
        run_boundary_resolvent_probe(args.boundary_resolvent_from,args.output,min(args.seconds_cap,119.8))
    elif args.boundary_environment:
        if args.refine_from is not None:
            parser.error("boundary environment and previous refinement are distinct frozen modes")
        run_boundary_environment_probe(args.output, args.seconds_cap)
    elif args.refine_from is None:
        run(args.output, args.steps)
    else:
        run_refinement(args.refine_from, args.output, args.steps)
