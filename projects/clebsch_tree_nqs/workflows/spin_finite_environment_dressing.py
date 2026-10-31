"""Finite-step, strict-A1 physical-amplitude dressing with fresh paired audits."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from su2_nqs.benchmarks.finite_scalar_dressing import finite_step_moments
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.parton_swap_actions import ClosedShellSwapEvaluator
from su2_nqs.benchmarks.projected_singlet_agp import triangular_hnf_gamma_a1_group
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax,
    triangular_pi_zero_flux_hopping, BOUNDARY_ORBIT, A1_COEFFICIENTS,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.lattices.triangular_hnf import C6, MX, hnf_site_permutation
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.environment_permutations import environment_permutation_bank
from su2_nqs.operators.projected_permutations import reynolds_permutation_bank, disjoint_swap_masks, evaluate_swap_bank
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from su2_nqs.vmc.spin_physical import padded_spin_connectivity
from .spin_n16_physical_fmove_carrier_gate import ROOT, _load_source, _fixed_sz_configurations


def _bank(length: int, cells: list) -> tuple:
    p,c=environment_permutation_bank(length**2,cells)
    active=c[:,-1]!=0
    group=np.asarray(triangular_hnf_gamma_a1_group(((length,0),(0,length))))
    return reynolds_permutation_bank(p[active],c[active,-1:],group)


def _validate_n16(spec: dict) -> dict:
    _,arrays=_load_source(spec)
    ref=spec["environment_reference"]; directory=ROOT/ref["directory"]
    r=json.loads((directory/"results.json").read_text())
    if r["result_sha256"]!=ref["result_sha256"] or canonical_sha256({k:v for k,v in r.items() if k!="result_sha256"})!=ref["result_sha256"] or file_sha256(directory/"states.npz")!=r["bundle_sha256"]:
        raise RuntimeError("N16 reference seal changed")
    with np.load(directory/"states.npz") as b: operator=b["OF_collective_dimer_correlation"]
    phi=arrays["physical_a1_basis"];pi=arrays["coupling_0.125_carrier_a1"]
    h=arrays["coupling_0.125_h_a1"]
    basis=_fixed_sz_configurations();states=np.asarray(basis,dtype=np.int8)
    psi=phi@pi; direction=phi@(operator@pi)
    powers=1<<np.arange(15,-1,-1)
    table=np.zeros(2**16,dtype=complex);table[states@powers]=psi
    p,c,cost=_bank(4,[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]])
    rng=np.random.default_rng(spec["validation_seed"])
    selected=rng.choice(len(states),spec["n16_validation_amplitudes"],replace=False)
    observed,_=evaluate_swap_bank(states[selected],lambda s:table[s@powers],disjoint_swap_masks(p),c)
    checks={"amplitude_relative_error":float(np.linalg.norm(observed[:,0]-direction[selected])/np.linalg.norm(direction[selected])),
            "singlet_raising_relative":spin_raising_relative_norm(direction,basis)}
    rows=[]
    for epsilon in spec["epsilon_grid"]:
        physical=psi+epsilon*direction; reduced=pi+epsilon*operator@pi
        z=float(np.vdot(physical,physical).real)
        expected=float(np.vdot(reduced,reduced).real)
        checks["normalization_error"]=max(checks.get("normalization_error",0.),abs(z-expected),abs(np.sum(abs(physical/np.sqrt(z))**2)-1))
        m=finite_step_moments(psi,direction,phi@(h@pi),phi@(h@operator@pi),np.ones(len(psi)),epsilon)
        rows.append(m)
    if max(checks.values())>2e-10:
        raise RuntimeError("N16 finite-amplitude/normalization gate failed")
    return {"checks":checks,"bank":cost,"epsilon_results":rows}


def run_spin_finite_environment_dressing(config_path: Path, output: Path, *, overwrite: bool=False) -> dict:
    """Run exact-A1 amplitude/H gates, then immutable finite-step paired streams."""
    spec=json.loads(config_path.read_text())
    if spec["task"]!="spin_finite_environment_dressing" or not spec["preregistered_before_run"] or spec["optimization_steps"] or spec["remote_training_on_pass"]:
        raise ValueError("expected fixed local finite-step protocol")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists() and not overwrite:
        raise FileExistsError(output/"results.json")
    started=perf_counter(); config_hash=canonical_sha256(spec)
    manifest=transitive_local_source_manifest((Path(__file__).resolve(),),root=ROOT,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))
    manifest_hash=canonical_sha256(manifest)
    n16=_validate_n16(spec)
    print("N16 gates:",n16["checks"],flush=True)
    orbit=build_n36_pi_zero_flux_orbit()
    carrier=jax.jit(lambda x:projected_pi_zero_flux_a1_amplitudes_jax(x,orbit=orbit))
    batch_size=spec["amplitude_batch_size"]
    def amplitude(states):
        result=[]
        for start in range(0,len(states),batch_size):
            batch=np.asarray(states[start:start+batch_size],dtype=np.int8)
            padded=np.repeat(batch[:1],batch_size,axis=0);padded[:len(batch)]=batch
            result.append(np.asarray(carrier(padded))[:len(batch)])
        return np.concatenate(result)
    p,c,bank_cost=_bank(6,spec["cells"])
    masks=disjoint_swap_masks(p)
    orbitals=np.asarray([np.linalg.eigh(triangular_pi_zero_flux_hopping(6,boundary_x=x,boundary_y=y))[1][:,:18] for x,y in BOUNDARY_ORBIT])
    fast=ClosedShellSwapEvaluator(orbitals,A1_COEFFICIENTS,p,c,amplitude,
        chunk_size=spec["swap_chunk_size"],condition_limit=spec["condition_limit"])
    bonds=triangular_j1j2_bonds(6,6,j1=1.,j2=spec["j2_over_j1"],periodic=True)
    def h_images(states, *, direct: bool=False):
        connected,elements,valid,_=padded_spin_connectivity(states,bonds)
        source=connected[valid]
        hvalues=np.zeros(elements.shape,dtype=complex);dvalues=np.zeros_like(hvalues)
        hvalues[valid]=amplitude(source)
        if direct:
            observed,cost=evaluate_swap_bank(source,amplitude,masks,c)
        else:
            observed,cost=fast(source)
        dvalues[valid]=observed[:,0]
        return np.sum(elements*hvalues,axis=1),np.sum(elements*dvalues,axis=1),cost
    rng=np.random.default_rng(spec["validation_seed"])
    validation=np.zeros((2,36),dtype=np.int8)
    for row in validation:row[rng.choice(36,18,replace=False)]=1
    direct,_=evaluate_swap_bank(validation,amplitude,masks,c)
    observed,_=fast(validation)
    check={"low_rank_amplitude_relative_error":float(np.linalg.norm(observed-direct)/np.linalg.norm(direct))}
    for name,operation,translation in (("Tx",None,(1,0)),("Ty",None,(0,1)),("C6",C6,(0,0)),("Mx",MX,(0,0))):
        kwargs={"translation":translation}
        if operation is not None:kwargs["operation"]=operation
        image=hnf_site_permutation(((6,0),(0,6)),**kwargs)
        rotated,_=fast(validation[:,image])
        check[f"A1_{name}_relative_error"]=float(np.linalg.norm(rotated-observed)/np.linalg.norm(observed))
    t=perf_counter();_,hd_direct,_=h_images(validation[:1],direct=True);direct_h_seconds=perf_counter()-t
    t=perf_counter();_,hd_fast,_=h_images(validation[:1]);fast_h_seconds=perf_counter()-t
    check["low_rank_H_direction_relative_error"]=float(np.linalg.norm(hd_fast-hd_direct)/np.linalg.norm(hd_direct))
    if max(check.values())>2e-9:
        raise RuntimeError("N36 strict-A1/H replay gate failed")
    print("N36 gates:",check,"H direct/fast seconds",direct_h_seconds,fast_h_seconds,flush=True)
    rng=np.random.default_rng(spec["calibration_seed"])
    uniform=np.zeros((spec["calibration_uniform_count"],36),dtype=np.int8)
    for row in uniform:row[rng.choice(36,18,replace=False)]=1
    kappa=float(spec["proposal_floor_fraction"]*np.mean(abs(amplitude(uniform))**2))
    if not np.isfinite(kappa) or kappa<=0:raise RuntimeError("invalid proposal calibration")
    proposal=lambda s:np.sqrt(abs(amplitude(s))**2+kappa)
    rows=[];streams=[]
    for index,seed in enumerate(spec["replicate_seeds"]):
        meta_path=output/f"replicate_{index}.json";bundle_path=output/f"replicate_{index}.npz"
        if meta_path.exists() and not overwrite:
            meta=json.loads(meta_path.read_text())
            if meta["config_sha256"]!=config_hash or meta["source_manifest_sha256"]!=manifest_hash or file_sha256(bundle_path)!=meta["bundle_sha256"]:
                raise RuntimeError("checkpoint identity mismatch")
            with np.load(bundle_path) as b:streams.append({k:b[k] for k in b.files})
            rows.append(meta);continue
        rep_start=perf_counter()
        samples=sample_fixed_wavefunction_exchange(proposal,n_sites=36,n_up=18,chain_count=spec["chain_count"],
            burn_sweeps=spec["burn_exchange_steps_per_chain"],measurement_blocks=spec["measurement_blocks"],
            decorrelation_sweeps=spec["decorrelation_exchange_steps"],seed=seed)
        states=samples.configurations;psi=amplitude(states)
        direction,center_cost=fast(states);direction=direction[:,0]
        hpsi=[];hdirection=[];fallback=center_cost["direct_fallback_count"];max_condition=center_cost["maximum_component_condition"]
        for start in range(0,len(states),1):
            if perf_counter()-started>spec["wall_limit_seconds"]:raise TimeoutError("local wall cap reached; completed streams remain checkpointed")
            hv,dv,cost=h_images(states[start:start+1])
            hpsi.extend(hv);hdirection.extend(dv)
            fallback+=cost["direct_fallback_count"];max_condition=max(max_condition,cost["maximum_component_condition"])
            if (start+1)%8==0:print(f"stream {index}: exact full-H dressed actions {start+1}/{len(states)}; elapsed {perf_counter()-rep_start:.1f}s",flush=True)
        data={"configurations":states,"psi":psi,"direction":direction,"hpsi":np.asarray(hpsi),"hdirection":np.asarray(hdirection),"proposal_squared":abs(psi)**2+kappa}
        moments=[finite_step_moments(data["psi"],data["direction"],data["hpsi"],data["hdirection"],data["proposal_squared"],eps) for eps in spec["epsilon_grid"]]
        np.savez_compressed(bundle_path,**data)
        meta={"index":index,"seed":seed,"config_sha256":config_hash,"source_manifest_sha256":manifest_hash,
              "bundle_sha256":file_sha256(bundle_path),"sample_count":len(states),"moments":moments,
              "acceptance_fraction":samples.acceptance_fraction,"direct_fallback_count":fallback,
              "maximum_component_condition":max_condition,"full_H_sampled_centers":len(states),"wall_seconds":perf_counter()-rep_start}
        atomic_json(meta_path,meta);rows.append(meta);streams.append(data)
        print(f"stream {index} checkpoint saved; {meta['wall_seconds']:.1f}s",flush=True)
    data={k:np.concatenate([s[k] for s in streams]) for k in streams[0]}
    def moments(indices):
        return [finite_step_moments(*(data[k][indices] for k in ("psi","direction","hpsi","hdirection","proposal_squared")),eps) for eps in spec["epsilon_grid"]]
    aggregate=moments(np.arange(len(data["psi"])))
    keys=("norm_ratio","energy_total","variance_total","energy_change_total","variance_change_total","carrier_fidelity")
    nc,nb,nr=spec["chain_count"],spec["measurement_blocks"],len(streams)
    chains=np.arange(nr*nc*nb).reshape(nr,nb,nc).transpose(0,2,1).reshape(nr*nc,nb)
    rng=np.random.default_rng(spec["bootstrap_seed"]);boots=[]
    for _ in range(spec["bootstrap_chain_replicates"]):
        ix=chains[rng.integers(0,len(chains),len(chains))].ravel()
        boots.append([[r[k] for k in keys] for r in moments(ix)])
    stderr=np.std(boots,axis=0,ddof=1)
    for row,se in zip(aggregate,stderr):row["paired_chain_bootstrap_se"]=dict(zip(keys,map(float,se)))
    result={"task":spec["task"],"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),
            "config":spec,"config_sha256":config_hash,"source_manifest":manifest,"n16_validation":n16,
            "n36_validation":check,"bank_cost":bank_cost,"direct_H_seconds_one_center":direct_h_seconds,
            "low_rank_H_seconds_one_center":fast_h_seconds,"proposal_kappa":kappa,"replicates":rows,
            "aggregate":aggregate,"sample_count":len(data["psi"]),"independent_chain_count":len(chains),
            "wall_seconds_this_invocation":perf_counter()-started,"runtime_identity":numerical_runtime_identity(),
            "remote_training_licensed":False,"hamiltonian_diagonalization_performed":False,
            "claim_boundary":"Finite steps on one fixed projected zero-Lanczos carrier, full deterministic A1 and H. Exact N16 normalization, sampled N36 relative normalization. Paired full-support importance audit; no epsilon selection, no NN training, no comparison to optimized Krylov/ViT. Short-chain and finite-sample bias remain."}
    result["result_sha256"]=canonical_sha256(result);atomic_json(output/"results.json",result)
    return result
