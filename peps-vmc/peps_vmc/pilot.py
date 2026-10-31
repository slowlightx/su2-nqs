"""Bounded local PEPS-VMC experiment, with disjoint fit/geometry/audit data.

This is a real-coefficient, free-site D=3 PEPS, not a neural network and not
an A1-projected state. Contraction is exact. The sampler is a known positive
IID importance proposal, NOT a PEPS Born sampler. No remote submission occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr

from .contraction import make_heisenberg_action, make_square_torus_plan
from .diagnostics import finite_FS, paired_moments
from .model import SquareGraphSU2PEPS
from .proposals import CoveringBridgeProposal, torus_coverings
from .symmetry import intertwiner_labels


def jsonable(value):
    """Convert numeric diagnostics, never silently convert NaN to a number."""
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)): return jsonable(np.asarray(value).tolist())
    if isinstance(value, np.generic): return value.item()
    return value


def write_json(path: Path, payload: dict) -> None:
    """Atomic small JSON artifact; invalid numerical results fail loudly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_manifest() -> dict[str, str]:
    """Hash all isolated core and the small number of directly reused modules."""
    root = Path(__file__).resolve().parents[2]
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    files += [root/"su2_nqs"/part for part in (
        "su2/clebsch_gordan.py", "su2/quantum_numbers.py", "lattices/triangular_j1j2.py",
        "vmc/checkpoint.py", "vmc/importance_amplitude_sr.py")]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate_config(c: dict) -> None:
    for key in ("rows", "columns", "steps", "fit_records", "geometry_records", "audit_records", "audit_groups"):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(key+" must be a positive integer")
    if c["steps"] > 4 or c["fit_records"] < 2 or c["geometry_records"] < 4:
        raise ValueError("this local pilot permits 1-4 updates and nontrivial panels")
    if c["audit_groups"] < 4 or c["audit_records"] % c["audit_groups"]:
        raise ValueError("audit must have >=4 equal independent groups")
    seeds = [c[k] for k in ("initial_seed", "covering_seed", "training_seed", "ward_seed", "audit_seed")]
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("disjoint predeclared nonnegative seeds required")
    for key in ("damping", "learning_rate", "tangent_FS_radius", "maximum_finite_FS", "wall_limit_seconds", "maximum_RSS_GiB"):
        if not np.isfinite(c[key]) or c[key] <= 0: raise ValueError(key+" must be finite positive")
    fractions = c["candidate_fractions"]
    if not fractions or fractions != sorted(set(fractions), reverse=True) or not all(0 < x <= 1 for x in fractions):
        raise ValueError("finite candidate fractions must decrease in (0,1]")
    if not np.isfinite(c["j2"]) or c["j2"] < 0: raise ValueError("nonnegative finite J2 required")


class Experiment:
    """Numerical plumbing, not an alternate wavefunction or optimizer."""

    def __init__(self, config: dict, output: Path):
        validate_config(config)
        if jax.default_backend() != "cpu": raise RuntimeError("this registered pilot is CPU-only")
        self.c = config; self.output = output
        self.model = SquareGraphSU2PEPS(config["rows"], config["columns"], shared=False)
        self.plan = make_square_torus_plan(config["rows"], config["columns"], 3)
        self.bonds = triangular_j1j2_bonds(config["rows"], config["columns"], j1=1., j2=config["j2"], periodic=True)
        pairs = np.asarray([(a, b) for a, b, _ in self.bonds], np.int32)
        couplings = np.asarray([v for _, _, v in self.bonds], np.float64)
        self.action = make_heisenberg_action(self.plan, pairs.tolist(), couplings,
                                             max_intermediate_elements=20_000_000)
        self.proposal = CoveringBridgeProposal(torus_coverings(config["rows"], config["columns"],
            count=config["coverings"], seed=config["covering_seed"]), pairs,
            np.abs(couplings)/np.abs(couplings).sum(), epsilon=config["uniform_fraction"])
        self.prepare = jax.jit(lambda params: self.model.tensors(params["w"]))
        self.amp = jax.jit(self.plan.amplitude)
        self.hp = jax.jit(self.action.products)
        self.initial = {"w": jnp.asarray(self.model.initial_coefficients(config["initial_seed"],
            three_leg_scale=config["three_leg_scale"], site_noise=config["site_noise"]))}
        self.update = make_importance_amplitude_sr(lambda p, x: self.plan.amplitude(self.model.tensors(p["w"]), x),
            self.initial, damping=config["damping"], score_chunk=1,
            parameter_chunk_size=512, sr_tangent_fs_radius=config["tangent_FS_radius"],
            prepare=lambda p: self.model.tensors(p["w"]), prepared_amplitude=self.plan.amplitude)
        self.registration = output/"registration.json"
        self.products_count = 0; self.amplitudes_count = 0

    def check_resources(self) -> None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        gib = rss/(1024**3 if sys.platform == "darwin" else 1024**2)
        if gib > self.c["maximum_RSS_GiB"]: raise RuntimeError("registered peak RSS cap exceeded")
        if self.registration.exists():
            start = json.loads(self.registration.read_text())["started_epoch"]
            if time.time()-start > self.c["wall_limit_seconds"]:
                raise TimeoutError("registered total train/replay/audit wall cap reached")

    def evaluate(self, params, rows: np.ndarray, *, full_H: bool) -> np.ndarray:
        bank = self.prepare(params); result = []
        for i, row in enumerate(rows):
            if i % 16 == 0: self.check_resources()
            values = self.hp(bank, jnp.asarray(row)) if full_H else self.amp(bank, jnp.asarray(row))
            result.append(np.asarray(values))
        if full_H: self.products_count += len(rows)
        else: self.amplitudes_count += len(rows)
        result = np.asarray(result)
        if not np.isfinite(result).all(): raise FloatingPointError("nonfinite exact contraction")
        return result

    def checkpoint(self, path: Path, params, moments, rng, step: int) -> None:
        save_checkpoint(path, jax.tree.map(lambda x: np.asarray(x), params),
            optimizer_state=jax.tree.map(lambda x: np.asarray(x), moments), metadata=dict(
                step=step, next_rng=rng.bit_generator.state, config_sha256=digest(self.c),
                source_sha256=digest(source_manifest()), proposal_sha256=self.proposal_hash(),
                optimizer="direct_importance_amplitude_SR_counter_and_inherited_zero_moments",
                params_sha256=digest(params)))

    def proposal_hash(self) -> str:
        return digest(dict(coverings=self.proposal.coverings, bonds=self.proposal.bonds,
            probabilities=self.proposal.exchange_probabilities, epsilon=self.proposal.epsilon))

    def warmup_and_oracles(self) -> dict:
        """Actual N36 amplitude, gradient, full-H and S+ cancellation checks."""
        rng = np.random.default_rng(self.c["ward_seed"])
        rows, _ = self.proposal.draw(6, rng); bank = self.prepare(self.initial)
        timing = {}
        for name, fn in (("amplitude", lambda: self.amp(bank, jnp.asarray(rows[0]))),
                         ("full_H", lambda: self.hp(bank, jnp.asarray(rows[0]))),
                         ("full_native_gradient", lambda: jax.jit(jax.grad(
                             lambda w: self.plan.amplitude(self.model.tensors(w), jnp.asarray(rows[0]))))(self.initial["w"]))):
            start = time.monotonic(); result = fn(); jax.block_until_ready(result)
            timing[name+"_first_seconds"] = time.monotonic()-start
            if name != "full_native_gradient":
                start = time.monotonic(); result = fn(); jax.block_until_ready(result)
                timing[name+"_warm_seconds"] = time.monotonic()-start
        gradient = np.asarray(result)
        if not np.isfinite(gradient).all() or np.linalg.norm(gradient) == 0:
            raise ArithmeticError("full native gradient is zero or nonfinite")
        direction = rng.normal(size=gradient.shape); direction /= np.linalg.norm(direction)
        eps = 1e-5
        plus = self.evaluate({"w": self.initial["w"]+eps*direction}, rows[:1], full_H=False)[0]
        minus = self.evaluate({"w": self.initial["w"]-eps*direction}, rows[:1], full_H=False)[0]
        analytic = float(np.sum(gradient*direction)); difference = float((plus-minus)/(2*eps))
        derivative_relative = abs(difference-analytic)/max(abs(difference), abs(analytic))
        if derivative_relative > 2e-6: raise ArithmeticError("native gradient finite difference failed")
        # Independent H oracle: direct amplitude on EVERY opposite-spin swap,
        # rather than the boundary-H contraction recurrence.
        row = rows[0]; psi = float(self.amp(bank, jnp.asarray(row)))
        diagonal = sum(v*.25*(1-2*int(row[a] != row[b])) for a, b, v in self.bonds)
        reference = diagonal*psi
        for a, b, coupling in self.bonds:
            if row[a] != row[b]:
                changed = row.copy(); changed[a], changed[b] = row[b], row[a]
                reference += .5*coupling*float(self.amp(bank, jnp.asarray(changed)))
        hp = float(self.hp(bank, jnp.asarray(row))[1])
        H_relative = abs(reference-hp)/max(abs(reference), abs(hp))
        if H_relative > 1e-10: raise ArithmeticError("full triangular H oracle failed")
        extra = np.asarray([sum(channel.virtual_q) == 3 for channel in intertwiner_labels(4)])
        ward_rows = rows.copy()
        for row in ward_rows:
            row[int(rng.choice(np.flatnonzero(row == 0)))] = 1
        np.savez_compressed(self.output/"ward_configurations.npz", rows=ward_rows)
        ward = self.ward(self.initial, ward_rows)
        return dict(timings=timing, full_H_bonds=len(self.bonds), amplitude=psi,
            Hpsi=hp, independent_Hpsi=reference, H_relative_discrepancy=H_relative,
            directional_gradient=analytic, finite_difference_gradient=difference,
            gradient_relative_discrepancy=derivative_relative,
            native_gradient_norm=float(np.linalg.norm(gradient)),
            three_leg_channel_gradient_norm=float(np.linalg.norm(gradient[:, extra])),
            nonzero_three_leg_gradient_entries=int(np.count_nonzero(gradient[:, extra])),
            S_plus_Ward=ward)

    def ward(self, params, target_rows) -> dict:
        residuals = []; undefined = 0
        for target in target_rows:
            if target.sum() != self.model.n_sites//2+1: raise ValueError("Ward target must have doubled M=2")
            terms = []
            for site in np.flatnonzero(target):
                parent = target.copy(); parent[site] = 0; terms.append(parent)
            values = self.evaluate(params, np.asarray(terms), full_H=False)
            denominator = float(np.sum(abs(values)))
            if denominator == 0: undefined += 1
            else: residuals.append(float(abs(values.sum())/denominator))
        if not residuals or max(residuals) > 1e-11: raise ArithmeticError("nontrivial S+ Ward check failed")
        return dict(relative_residuals=residuals, maximum_relative_residual=max(residuals),
            zero_all_terms_cases=undefined, diagnostic_not_global_S_squared_bound=True)

    def finite_update(self, parent, proposal, rows, density, guard_rows, guard_density):
        """Select on fresh amplitudes; independent guard may veto, not retune."""
        selection_parent = self.evaluate(parent, rows, full_H=False)
        selection = []; selected = None; fraction = 0.
        for f in self.c["candidate_fractions"]:
            candidate = jax.tree.map(lambda x, y: x+f*(y-x), parent, proposal)
            amplitudes = self.evaluate(candidate, rows, full_H=False)
            geometry = finite_FS(np.stack((selection_parent, amplitudes), 1), density)
            selection.append(dict(fraction=f, **geometry))
            if geometry["FS_angle"] <= self.c["maximum_finite_FS"]:
                selected = candidate; fraction = f; break
        if selected is None:
            return parent, dict(selection=selection, accepted=False, selected_fraction=0.,
                reason="no selection fraction passed")
        guard = finite_FS(np.stack((self.evaluate(parent, guard_rows, full_H=False),
            self.evaluate(selected, guard_rows, full_H=False)), 1), guard_density)
        accepted = guard["FS_angle"] <= self.c["maximum_finite_FS"]
        return (selected if accepted else parent), dict(selection=selection, guard=guard,
            accepted=accepted, selected_fraction=fraction,
            reason="fresh finite geometry only; not an energy-improvement certificate")

    def train(self) -> dict:
        if self.registration.exists(): raise FileExistsError("experiment already registered; choose a new output")
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.registration, dict(started_epoch=time.time(), config=self.c,
            config_sha256=digest(self.c), sources=source_manifest(),
            source_sha256=digest(source_manifest()), proposal_sha256=self.proposal_hash(),
            model=self.model.metadata(), contraction=self.plan.metadata(), H_action=self.action.metadata(),
            endpoint_policy="fixed two attempted updates; no audit-based endpoint selection",
            data_roles="disjoint IID fit, selection, veto, and frozen-endpoint audit",
            inference_scope="engineering feasibility, not a competitive energy or phase claim"))
        preflight = self.warmup_and_oracles(); write_json(self.output/"preflight.json", preflight)
        print(json.dumps(dict(event="preflight", **jsonable(preflight))), flush=True)
        rng = np.random.default_rng(self.c["training_seed"])
        params = self.initial
        moments = (jnp.asarray(0, jnp.int64), jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))
        self.checkpoint(self.output/"initial.pkl", params, moments, rng, 0)
        history = []
        for step in range(1, self.c["steps"]+1):
            start = time.monotonic(); folder = self.output/f"step_{step:02d}"; folder.mkdir()
            self.checkpoint(folder/"before.pkl", params, moments, rng, step-1)
            rows, density = self.proposal.draw(self.c["fit_records"], rng)
            values = self.evaluate(params, rows, full_H=True)
            np.savez_compressed(folder/"fit.npz", rows=rows, density=density, products=values)
            proposed, next_moments, sr_info = self.update(params, moments, jnp.asarray(rows),
                jnp.asarray(values[:, 1]), jnp.asarray(density), self.c["learning_rate"])
            jax.block_until_ready(proposed)
            if not bool(sr_info["accepted"]): raise ArithmeticError("invalid native SR proposal")
            selection_rows, selection_q = self.proposal.draw(self.c["geometry_records"], rng)
            guard_rows, guard_q = self.proposal.draw(self.c["geometry_records"], rng)
            np.savez_compressed(folder/"geometry.npz", selection_rows=selection_rows,
                selection_q=selection_q, guard_rows=guard_rows, guard_q=guard_q)
            previous = params
            params, finite = self.finite_update(params, proposed, selection_rows, selection_q, guard_rows, guard_q)
            if finite["accepted"]: moments = next_moments
            self.checkpoint(folder/"after.pkl", params, moments, rng, step)
            transition = dict(step=step, seconds=time.monotonic()-start, SR=sr_info, finite_update=finite,
                native_parameter_delta_norm=float(np.linalg.norm(np.asarray(params["w"]-previous["w"]))),
                endpoint_sha256=digest(params), next_rng_sha256=digest(rng.bit_generator.state))
            write_json(folder/"transition.json", transition); history.append(jsonable(transition))
            print(json.dumps(dict(event="update", **jsonable(transition))), flush=True)
        self.checkpoint(self.output/"frozen_endpoint.pkl", params, moments, rng, self.c["steps"])
        endpoint_ward = self.ward(params, np.load(self.output/"ward_configurations.npz")["rows"])
        result = dict(history=history, endpoint_S_plus_Ward=endpoint_ward,
            full_H_centers=self.products_count, scalar_amplitude_queries=self.amplitudes_count,
            seconds_since_registration=time.time()-json.loads(self.registration.read_text())["started_epoch"])
        write_json(self.output/"training.json", result)
        return result

    def verify_registered(self) -> dict:
        registration = json.loads(self.registration.read_text())
        if registration["config_sha256"] != digest(self.c) or registration["source_sha256"] != digest(source_manifest()):
            raise RuntimeError("registered configuration or source changed")
        if registration["proposal_sha256"] != self.proposal_hash(): raise RuntimeError("proposal changed")
        self.check_resources()
        return registration

    def replay(self) -> dict:
        """Fresh-process optimizer/parameter/RNG replay of the final transition."""
        self.verify_registered(); folder = self.output/f"step_{self.c['steps']:02d}"
        before = load_checkpoint(folder/"before.pkl"); after = load_checkpoint(folder/"after.pkl")
        frozen = load_checkpoint(self.output/"frozen_endpoint.pkl")
        rng = np.random.default_rng(); rng.bit_generator.state = before["metadata"]["next_rng"]
        fit = np.load(folder/"fit.npz"); geometry = np.load(folder/"geometry.npz")
        rows, density = self.proposal.draw(self.c["fit_records"], rng)
        if not np.array_equal(rows, fit["rows"]) or not np.array_equal(density, fit["density"]):
            raise ArithmeticError("restored RNG did not regenerate fit data")
        selection_rows, selection_q = self.proposal.draw(self.c["geometry_records"], rng)
        guard_rows, guard_q = self.proposal.draw(self.c["geometry_records"], rng)
        for name, actual in (("selection_rows", selection_rows), ("selection_q", selection_q),
                             ("guard_rows", guard_rows), ("guard_q", guard_q)):
            if not np.array_equal(actual, geometry[name]): raise ArithmeticError("restored geometry RNG mismatch")
        params = jax.tree.map(jnp.asarray, before["params"]); moments = jax.tree.map(jnp.asarray, before["optimizer_state"])
        proposed, next_moments, info = self.update(params, moments, jnp.asarray(rows),
            jnp.asarray(fit["products"][:, 1]), jnp.asarray(density), self.c["learning_rate"])
        replayed, finite = self.finite_update(params, proposed, selection_rows, selection_q, guard_rows, guard_q)
        actual_moments = next_moments if finite["accepted"] else moments
        parameter_error = float(np.max(abs(np.asarray(replayed["w"])-after["params"]["w"])))
        optimizer_error = max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a, b in zip(
            jax.tree.leaves(actual_moments), jax.tree.leaves(after["optimizer_state"])))
        rng_exact = digest(rng.bit_generator.state) == digest(after["metadata"]["next_rng"])
        if parameter_error > 2e-11 or optimizer_error > 2e-11 or not rng_exact:
            raise ArithmeticError("training transition replay failed")
        if digest(frozen["params"]) != digest(after["params"]): raise ArithmeticError("frozen endpoint differs")
        # Recompute real H, not only saved gradients. Two rows keep this bounded.
        values = self.evaluate(params, rows[:2], full_H=True); reference = fit["products"][:2]
        product_error = float(np.max(abs(values-reference))/np.max(abs(reference)))
        if product_error > 2e-11: raise ArithmeticError("H-product replay failed")
        result = dict(passed=True, parameter_max_abs=parameter_error, optimizer_max_abs=optimizer_error,
            next_rng_exact=rng_exact, H_product_relative_error=product_error,
            endpoint_sha256=digest(frozen["params"]), replay_uses_cached_fit_H_but_recomputes_two_H_centers=True,
            independently_compiled_process=True)
        write_json(self.output/"replay.json", result)
        return result

    def audit(self) -> dict:
        self.verify_registered()
        if (self.output/"audit.json").exists(): raise FileExistsError("do not reuse the untouched audit")
        replay = json.loads((self.output/"replay.json").read_text())
        endpoint = load_checkpoint(self.output/"frozen_endpoint.pkl")
        if not replay["passed"] or replay["endpoint_sha256"] != digest(endpoint["params"]):
            raise RuntimeError("a passed frozen-endpoint replay is required")
        rows, density = self.proposal.draw(self.c["audit_records"], np.random.default_rng(self.c["audit_seed"]))
        # From this point the audit panel is spent, even if interrupted.
        marker = self.output/"audit_started.json"
        if marker.exists(): raise FileExistsError("audit seed was already opened")
        write_json(marker, dict(endpoint_sha256=digest(endpoint["params"]), rows_sha256=digest(rows),
            seed=self.c["audit_seed"], endpoint_frozen_before_audit=True))
        params = jax.tree.map(jnp.asarray, endpoint["params"])
        start = time.monotonic(); products = []
        for first in range(0, len(rows), 32):
            chunk = rows[first:first+32]
            products.append(np.stack((self.evaluate(self.initial, chunk, full_H=True),
                                      self.evaluate(params, chunk, full_H=True)), 1))
            print(json.dumps(dict(event="audit_progress", records=first+len(chunk),
                seconds=time.monotonic()-start)), flush=True)
        products = np.concatenate(products)
        np.savez_compressed(self.output/"untouched_audit.npz", rows=rows, density=density, products=products)
        result = paired_moments(products, density, n_sites=self.model.n_sites, groups=self.c["audit_groups"])
        result.update(seconds=time.monotonic()-start, full_H_centers=2*len(rows),
            endpoint_sha256=digest(params), geometry_A1_enforced=False,
            contraction_exact=True, precision_claim="small engineering pilot; not a competitive energy estimate",
            finite_panel_geometry=finite_FS(products[:, :, 0], density))
        write_json(self.output/"audit.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "replay", "audit"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = Experiment(json.loads(args.config.read_text()), args.output)
    result = getattr(experiment, args.phase)()
    print(json.dumps(dict(phase=args.phase, result=jsonable(result)), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
