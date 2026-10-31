"""Low-rank test of environment-dependent complete exchange coefficients.

Reuse the sealed carrier and physics diagnostics. The main collective mode
is fixed before running; the three resolved modes are robustness controls.
The affine coefficient oracle is not a ViT or a scalable sampled evaluator.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.benchmarks.spectral_error_tomography import lowest_subspace_state, orthonormal_column_space
from su2_nqs.benchmarks.su2_entanglement_spectrum import su2_resolved_entanglement
from su2_nqs.benchmarks.triangular_state_manifold import TRIANGULAR_N16_RHOMBUS_SUPPORTS
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.operators.cell_spin_channels import four_site_spin_modes, weighted_cell_exchange_action
from su2_nqs.operators.exchange_environment import ENVIRONMENT_FEATURES, spectator_environment_action
from .spin_channel_attention import _krylov
from .spin_channel_mechanism import _observables, rotate_complement
from .spin_n16_physical_fmove_carrier_gate import ROOT, _load_source, _fixed_sz_configurations, _metrics


def build_environment_bank(phi: np.ndarray, basis: tuple) -> tuple[dict, dict, dict, dict]:
    """Form O F physically before compressing, and audit forbidden shortcuts."""
    pairs=FixedSzPairActions(basis)
    counters={"pair_calls":0,"vector_equivalents":0}
    def pair(v,i,j):
        counters["pair_calls"]+=1
        counters["vector_equivalents"]+=v.shape[1] if v.ndim==2 else 1
        return .5*v[pairs.indices(i,j)]-.25*v
    a,b,c,d=TRIANGULAR_N16_RHOMBUS_SUPPORTS
    weights=four_site_spin_modes()[1:]
    checks={"hermiticity_max":0.,"spin_raising_relative_max":0.,"commuting_action_order_max":0.}
    costs={}
    def compress(image):
        matrix=phi.conj().T@image
        checks["hermiticity_max"]=max(checks["hermiticity_max"],float(np.max(np.abs(matrix-matrix.conj().T))))
        checks["spin_raising_relative_max"]=max(checks["spin_raising_relative_max"],spin_raising_relative_norm(image[:,0],basis))
        return (matrix+matrix.conj().T)/2
    def measured(label,action):
        start,before=perf_counter(),dict(counters)
        image=action()
        costs[label]={k:counters[k]-before[k] for k in counters}
        costs[label]["wall_seconds"]=perf_counter()-start
        return image
    raw_images={f"mode_{mu}":measured(f"O_mode_{mu}",lambda w=w:weighted_cell_exchange_action(phi,pair,a,b,w,w)) for mu,w in enumerate(weights,1)}
    raw_images["collective"]=sum(raw_images.values())/np.sqrt(3)
    operators={name:compress(v) for name,v in raw_images.items()}
    envs,products={},{}
    shortcut={}
    for feature in ENVIRONMENT_FEATURES:
        fv=measured(f"F_{feature}",lambda:spectator_environment_action(phi,pair,c,d,feature))
        envs[feature]=compress(fv)
        images={}
        for mu,w in enumerate(weights,1):
            name=f"mode_{mu}"
            images[name]=measured(f"O_F_{name}_{feature}_environment_cached",lambda:weighted_cell_exchange_action(fv,pair,a,b,w,w))
            reverse=spectator_environment_action(raw_images[name],pair,c,d,feature)
            checks["commuting_action_order_max"]=max(checks["commuting_action_order_max"],float(np.max(np.abs(reverse-images[name]))))
        images["collective"]=sum(images.values())/np.sqrt(3)
        products[feature]={name:compress(v) for name,v in images.items()}
        shortcut[feature]={name:float(np.linalg.norm(products[feature][name]-operators[name]@envs[feature])) for name in images}
    return operators,envs,products,{"checks":checks,"costs":costs,"premature_A1_projection_frobenius_error":shortcut}


def _reference_bundle(spec: dict) -> dict:
    reference=spec["mechanism_reference"]
    directory=ROOT/reference["directory"]
    result=json.loads((directory/"results.json").read_text())
    expected=reference["result_sha256"]
    if result["result_sha256"]!=expected or canonical_sha256({k:v for k,v in result.items() if k!="result_sha256"})!=expected:
        raise RuntimeError("mechanism reference seal changed")
    if file_sha256(directory/"states.npz")!=result["bundle_sha256"]:
        raise RuntimeError("mechanism reference bundle changed")
    with np.load(directory/"states.npz",allow_pickle=False) as bundle:
        return {k:bundle[k] for k in ("operator_bank","a0","a1","b0","b1")}


def run_spin_exchange_environment(config_path: Path, output: Path, *, overwrite: bool=False) -> dict:
    """Run every registered environment/control without selecting a mode."""
    spec=json.loads(config_path.read_text())
    if spec.get("task")!="spin_exchange_environment" or spec.get("preregistered_before_run") is not True or spec.get("remote_training_on_pass") is not False:
        raise ValueError("expected registered local-only environment comparison")
    if tuple(spec["environment_features"])!=ENVIRONMENT_FEATURES or spec["active_exchanges"]!=["collective","mode_1","mode_2","mode_3"] or spec["maximum_rank"]!=5:
        raise ValueError("do not silently change the registered model family")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists() and not overwrite:
        raise FileExistsError(output/"results.json")
    _,arrays=_load_source(spec)
    previous=_reference_bundle(spec)
    phi,basis=arrays["physical_a1_basis"],_fixed_sz_configurations()
    operators,envs,products,diagnostics=build_environment_bank(phi,basis)
    replay=max(float(np.max(np.abs(operators[f"mode_{mu+1}"]-previous["operator_bank"][mu]-previous["operator_bank"][mu+3]))) for mu in range(3))
    diagnostics["checks"]["old_unsplit_operator_replay_max"]=replay
    if max(diagnostics["checks"].values())>2e-10:
        raise RuntimeError("exact operator gate failed")
    h1=arrays["coupling_0_h_a1"]
    h2=(arrays["coupling_0.125_h_a1"]-h1)/.125
    pi=arrays["coupling_0.125_carrier_a1"]
    tolerance=float(spec["relative_rank_tolerance"])
    cases,stored={},{}
    for coupling in spec["couplings"]:
        h=h1+float(coupling)*h2
        anchor=_krylov(pi,h,3)
        carrier,_=lowest_subspace_state(anchor,h)
        _,eigenvectors=np.linalg.eigh(h)
        exact=eigenvectors[:,0]
        a,b=TRIANGULAR_N16_RHOMBUS_SUPPORTS[:2]
        reference=[su2_resolved_entanglement(phi@exact,basis,cut) for cut in (a,a+b)]
        case={"modes":{},"exact_energy_per_site":_metrics(exact,h)["energy_per_site"],"a1_dimension":h.shape[0]}
        stored[f"j{coupling}_exact"]=exact
        def describe(state):
            return dict(_metrics(state,h),physics=_observables(state,phi,basis,reference,h1,h2,previous["operator_bank"],previous))
        for name in spec["active_exchanges"]:
            op=operators[name]
            constant_columns=np.column_stack((anchor,op@carrier))
            constant,crank=lowest_subspace_state(constant_columns,h,relative_tolerance=tolerance)
            constant_basis,_,_=orthonormal_column_space(constant_columns,relative_tolerance=tolerance)
            mode={"constant":dict(describe(constant),rank=crank),"environments":{}}
            stored[f"j{coupling}_{name}_constant"]=constant
            for feature in spec["environment_features"]:
                f,of=envs[feature],products[feature][name]
                direction=of@carrier
                columns=np.column_stack((constant_columns,direction))
                state,rank=lowest_subspace_state(columns,h,relative_tolerance=tolerance)
                if rank>spec["maximum_rank"] or rank>=h.shape[0]:
                    raise RuntimeError("registered non-saturation bound exceeded")
                overlap=np.vdot(carrier,state)
                if abs(overlap)>1e-12:
                    state=state*np.conj(overlap)/abs(overlap)
                additive,arank=lowest_subspace_state(np.column_stack((constant_columns,f@carrier)),h,relative_tolerance=tolerance)
                krylov,krank=lowest_subspace_state(_krylov(pi,h,rank),h)
                row=dict(describe(state),rank=rank)
                row["environment_only_addition"]=dict(describe(additive),rank=arank)
                row["matched_krylov"]=dict(describe(krylov),rank=krank)
                coefficients,_,_,singular=np.linalg.lstsq(columns,state,rcond=tolerance)
                fit_error=float(np.linalg.norm(columns@coefficients-state))
                identifiable=rank==columns.shape[1] and singular[-1]>1e-8*singular[0] and abs(overlap)>1e-12
                row["affine_coefficient_fit"]={
                    "unique_with_fixed_carrier_phase":bool(identifiable),
                    "relative_smallest_singular_value":float(singular[-1]/singular[0]),
                    "reconstruction_error":fit_error,
                    "alpha":float(coefficients[-2].real) if identifiable else None,
                    "beta":float(coefficients[-1].real) if identifiable else None,
                    "imaginary_coefficient_max":float(np.max(np.abs(coefficients.imag))),
                    "meaning":"alpha O + beta O F around the full K2 polynomial anchor; coefficients are not probabilities",
                }
                for label,v in (("constant",constant),("modulated",state),("exact",exact)):
                    mean_o=np.vdot(v,op@v).real
                    mean_f=np.vdot(v,f@v).real
                    mean_of=np.vdot(v,of@v).real
                    row[f"correlation_{label}"]={"O":float(mean_o),"F":float(mean_f),"OF":float(mean_of),"connected_OF":float(mean_of-mean_o*mean_f)}
                row["energy_change_from_constant_per_site"]=row["energy_per_site"]-mode["constant"]["energy_per_site"]
                row["energy_minus_additive_environment_per_site"]=row["energy_per_site"]-row["environment_only_addition"]["energy_per_site"]
                row["energy_minus_matched_krylov_per_site"]=row["energy_per_site"]-row["matched_krylov"]["energy_per_site"]
                row["wins_additive_energy_and_variance"]=rank==arank and row["energy_minus_additive_environment_per_site"]<-1e-10 and row["variance_total"]<=row["environment_only_addition"]["variance_total"]+1e-12
                row["wins_krylov_energy_and_variance"]=rank==krank and row["energy_minus_matched_krylov_per_site"]<-1e-10 and row["variance_total"]<=row["matched_krylov"]["variance_total"]+1e-12
                row["scrambles"]=[]
                anchor_error=gram_error=0.
                for seed in spec["scramble_seeds"]:
                    rotation=rotate_complement(constant_basis,seed)
                    rotated=rotation@direction
                    anchor_error=max(anchor_error,float(np.max(np.abs(rotation@constant_basis-constant_basis))))
                    all_cols=np.column_stack((constant_basis,direction))
                    gram_error=max(gram_error,float(np.max(np.abs(all_cols.conj().T@all_cols-(rotation@all_cols).conj().T@(rotation@all_cols)))))
                    candidate,srank=lowest_subspace_state(np.column_stack((constant_columns,rotated)),h,relative_tolerance=tolerance)
                    if srank!=rank:
                        raise RuntimeError("scramble changed subspace rank")
                    row["scrambles"].append(dict(_metrics(candidate,h),seed=seed,rank=srank))
                row["scramble_anchor_max_error"]=anchor_error
                row["scramble_gram_max_error"]=gram_error
                row["scramble_joint_wins"]=sum(row["energy_per_site"]<s["energy_per_site"] and row["variance_total"]<s["variance_total"] for s in row["scrambles"])
                mode["environments"][feature]=row
                stored[f"j{coupling}_{name}_{feature}"]=state
                stored[f"j{coupling}_{name}_{feature}_additive"]=additive
                print(f"J2={coupling} {name}/{feature}: r={rank}, E/N={row['energy_per_site']:.12f}, gain={row['energy_change_from_constant_per_site']:+.3g}, deltaK={row['energy_minus_matched_krylov_per_site']:+.3g}, deltaAdd={row['energy_minus_additive_environment_per_site']:+.3g}, scramble={row['scramble_joint_wins']}/5",flush=True)
            case["modes"][name]=mode
        cases[str(coupling)]=case
    for name,matrix in operators.items(): stored[f"O_{name}"]=matrix
    for feature,matrix in envs.items(): stored[f"F_{feature}"]=matrix
    for feature,items in products.items():
        for name,matrix in items.items(): stored[f"OF_{name}_{feature}"]=matrix
    np.savez_compressed(output/"states.npz",**stored)
    result={
        "task":"spin_exchange_environment","status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),
        "config":spec,"diagnostics":diagnostics,"cases":cases,
        "model":"triangular 4x4 PBC J1-J2, exact S=0 Gamma-A1",
        "claim_boundary":"A fixed low-rank affine operator-coefficient oracle, not ViT training or sampled/GPU scaling. OF is formed in physical amplitudes before A1 compression. No mode selection, no QSL phase claim, no remote release.",
        "remote_training_licensed":False,
        "bundle_sha256":file_sha256(output/"states.npz"),
        "source_manifest":transitive_local_source_manifest((Path(__file__).resolve(),),root=ROOT,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs")),
    }
    result["result_sha256"]=canonical_sha256(result)
    atomic_json(output/"results.json",result)
    return result
