"""Residualized environment test beyond the sealed three-exchange baseline."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.benchmarks.spectral_error_tomography import lowest_subspace_state, orthonormal_column_space
from su2_nqs.benchmarks.su2_entanglement_spectrum import su2_resolved_entanglement
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.operators.cell_spin_channels import four_site_spin_modes, weighted_cell_exchange_action
from su2_nqs.operators.exchange_environment import spectator_environment_action
from .spin_channel_attention import _krylov
from .spin_channel_mechanism import _observables, rotate_complement
from .spin_exchange_environment import _reference_bundle
from .spin_n16_physical_fmove_carrier_gate import ROOT, _load_source, _fixed_sz_configurations, _metrics


def orthogonal_residual(direction: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a twice-reorthogonalized residual and its squared norm fraction."""
    vector = np.asarray(direction, dtype=np.complex128)
    q = np.asarray(basis, dtype=np.complex128)
    if vector.ndim != 1 or q.ndim != 2 or q.shape[0] != len(vector):
        raise ValueError("incompatible direction/basis")
    if not np.allclose(q.conj().T@q, np.eye(q.shape[1]), atol=1e-12, rtol=0):
        raise ValueError("basis must be orthonormal")
    if not np.isfinite(vector).all() or not np.isfinite(q).all():
        raise ValueError("finite arrays required")
    residual = vector.copy()
    for _ in range(2):
        residual -= q@(q.conj().T@residual)
    norm = float(np.vdot(vector, vector).real)
    fraction = float(np.vdot(residual, residual).real/norm) if norm else 0.
    return residual, fraction


def partition_operators(phi: np.ndarray, basis: tuple, cells: tuple) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the fixed dimer environment and collective OF before projection."""
    pairs = FixedSzPairActions(basis)
    pair = lambda v,i,j: .5*v[pairs.indices(i,j)]-.25*v
    a,b,c,d = cells
    fv = spectator_environment_action(phi, pair, c, d, "dimer_correlation")
    image = sum(weighted_cell_exchange_action(fv,pair,a,b,w,w)
                for w in four_site_spin_modes()[1:])/np.sqrt(3)
    f, of = phi.conj().T@fv, phi.conj().T@image
    checks = {
        "hermiticity_max": max(float(np.max(np.abs(m-m.conj().T))) for m in (f,of)),
        "singlet_raising_relative": spin_raising_relative_norm(image[:,0],basis),
    }
    return (f+f.conj().T)/2, (of+of.conj().T)/2, checks


def run_spin_environment_complement(config_path: Path, output: Path, *, overwrite: bool=False) -> dict:
    """Test one fixed OF correction beyond the entire stronger exchange space."""
    spec = json.loads(config_path.read_text())
    if spec["task"] != "spin_environment_complement" or not spec["preregistered_before_run"] or spec["remote_training_on_pass"]:
        raise ValueError("expected registered local complement test")
    if spec["environment_feature"] != "dimer_correlation" or spec["active_exchange"] != "collective" or spec["maximum_rank"] != 6:
        raise ValueError("registered single-feature trial changed")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists() and not overwrite:
        raise FileExistsError(output/"results.json")
    _, arrays = _load_source(spec)
    previous = _reference_bundle(spec)
    ref = spec["environment_reference"]
    directory = ROOT/ref["directory"]
    old = json.loads((directory/"results.json").read_text())
    if old["result_sha256"] != ref["result_sha256"] or canonical_sha256({k:v for k,v in old.items() if k!="result_sha256"}) != ref["result_sha256"]:
        raise RuntimeError("old environment result seal changed")
    if file_sha256(directory/"states.npz") != old["bundle_sha256"]:
        raise RuntimeError("old environment bundle changed")
    with np.load(directory/"states.npz",allow_pickle=False) as bundle:
        o = [bundle[f"O_mode_{i}"] for i in (1,2,3)]
        original_f, original_of = bundle["F_dimer_correlation"], bundle["OF_collective_dimer_correlation"]
    phi, basis = arrays["physical_a1_basis"], _fixed_sz_configurations()
    h1 = arrays["coupling_0_h_a1"]
    h2 = (arrays["coupling_0.125_h_a1"]-h1)/.125
    pi = arrays["coupling_0.125_carrier_a1"]
    tolerance = float(spec["relative_rank_tolerance"])
    cases, stored, diagnostics = {}, {}, {}
    for partition, cells_raw in spec["partitions"].items():
        cells = tuple(tuple(x) for x in cells_raw)
        if sorted(sum(cells,())) != list(range(16)):
            raise ValueError("partition must cover each spin exactly once")
        f, of, checks = partition_operators(phi,basis,cells)
        if partition == "original":
            checks["sealed_F_replay_max"] = float(np.max(np.abs(f-original_f)))
            checks["sealed_OF_replay_max"] = float(np.max(np.abs(of-original_of)))
        if max(checks.values()) > 2e-10:
            raise RuntimeError("physical operator validation failed")
        diagnostics[partition] = checks
        stored[f"{partition}_F"], stored[f"{partition}_OF"] = f,of
        for coupling in spec["couplings"]:
            h = h1+coupling*h2
            k2 = _krylov(pi,h,3)
            carrier,_ = lowest_subspace_state(k2,h)
            strong, brank, _ = orthonormal_column_space(np.column_stack((k2,*[m@carrier for m in o])),relative_tolerance=tolerance)
            base,_ = lowest_subspace_state(strong,h)
            residual, fraction = orthogonal_residual(of@carrier,strong)
            state, rank = lowest_subspace_state(np.column_stack((strong,residual)),h,relative_tolerance=tolerance)
            if rank > 6 or rank >= len(pi):
                raise RuntimeError("non-saturation cap failed")
            additive, arank = lowest_subspace_state(np.column_stack((strong,f@carrier)),h,relative_tolerance=tolerance)
            krylov, krank = lowest_subspace_state(_krylov(pi,h,rank),h)
            _, eigenvectors = np.linalg.eigh(h)
            exact = eigenvectors[:,0]
            a,b = cells[:2]
            reference = [su2_resolved_entanglement(phi@exact,basis,cut) for cut in (a,a+b)]
            def describe(v):
                return dict(_metrics(v,h),physics=_observables(v,phi,basis,reference,h1,h2,previous["operator_bank"],previous))
            row = dict(describe(state),rank=rank,base=dict(describe(base),rank=brank),
                       additive=dict(describe(additive),rank=arank),matched_krylov=dict(describe(krylov),rank=krank),
                       exact_energy_per_site=_metrics(exact,h)["energy_per_site"],outside_norm_fraction=fraction,
                       residual_orthogonality_max=float(np.max(np.abs(strong.conj().T@residual))))
            gradient = h@base-np.vdot(base,h@base)*base
            norm = np.linalg.norm(residual)
            row["hamiltonian_residual_capture_fraction"] = float(abs(np.vdot(residual,gradient))**2/(norm**2*np.vdot(gradient,gradient).real)) if norm>1e-14 else 0.
            row["energy_change_from_strong_per_site"] = row["energy_per_site"]-row["base"]["energy_per_site"]
            row["energy_minus_matched_krylov_per_site"] = row["energy_per_site"]-row["matched_krylov"]["energy_per_site"]
            row["additive_rank_matched"] = rank == arank
            row["scrambles"] = []
            errors=[]
            for seed in spec["scramble_seeds"]:
                rotation=rotate_complement(strong,seed)
                candidate,srank=lowest_subspace_state(np.column_stack((strong,rotation@residual)),h,relative_tolerance=tolerance)
                if srank != rank:
                    raise RuntimeError("scramble rank changed")
                errors.append(float(np.max(np.abs(rotation@strong-strong))))
                row["scrambles"].append(dict(_metrics(candidate,h),rank=srank,seed=seed))
            row["scramble_anchor_max_error"] = max(errors)
            row["scramble_joint_wins"] = sum(row["energy_per_site"]<r["energy_per_site"] and row["variance_total"]<r["variance_total"] for r in row["scrambles"])
            row["wins_krylov_energy_and_variance"] = row["energy_minus_matched_krylov_per_site"] < -1e-10 and row["variance_total"] <= row["matched_krylov"]["variance_total"]
            row["connected_OF"] = {}
            collective = sum(o)/np.sqrt(3)
            for label,v in (("base",base),("dressed",state),("exact",exact)):
                row["connected_OF"][label] = float((np.vdot(v,of@v)-np.vdot(v,collective@v)*np.vdot(v,f@v)).real)
            key=f"{partition}/j{coupling}"
            cases[key]=row
            stored[key+"/base"],stored[key+"/dressed"],stored[key+"/residual"] = base,state,residual
            print(f"{key}: rank {brank}->{rank}; outside={fraction:.4g}; gain={row['energy_change_from_strong_per_site']:+.4g}; deltaK={row['energy_minus_matched_krylov_per_site']:+.4g}; scramble={row['scramble_joint_wins']}/5; additive rank={arank}",flush=True)
    np.savez_compressed(output/"states.npz",**stored)
    result={"task":spec["task"],"status":"complete","config":spec,"config_sha256":canonical_sha256(spec),
            "created_utc":datetime.now(timezone.utc).isoformat(),"cases":cases,"diagnostics":diagnostics,
            "remote_training_licensed":False,"bundle_sha256":file_sha256(output/"states.npz"),
            "claim_boundary":"One residualized dimer-environment direction beyond full K2+three exchanges; exact finite cluster, no neural training, no efficiency claim, all partitions reported.",
            "source_manifest":transitive_local_source_manifest((Path(__file__).resolve(),),root=ROOT,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))}
    result["result_sha256"]=canonical_sha256(result)
    atomic_json(output/"results.json",result)
    return result
