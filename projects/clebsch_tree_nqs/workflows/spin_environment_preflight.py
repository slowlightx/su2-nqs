"""Bounded local N36 environment-action and first-derivative feasibility test."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from su2_nqs.benchmarks.environment_tangent import importance_tangent_moments
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.environment_permutations import (
    ENVIRONMENT_ACTION_LABELS, environment_permutation_bank, evaluate_environment_actions,
)
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
from su2_nqs.vmc.spin_physical import padded_spin_connectivity

ROOT = Path(__file__).resolve().parents[3]


def run_spin_environment_preflight(config_path: Path, output: Path, *, overwrite: bool=False) -> dict:
    """Run or resume bounded independent streams, never optimizing energy."""
    spec=json.loads(config_path.read_text())
    if spec["task"]!="spin_environment_preflight" or not spec["preregistered_before_run"] or spec["linear_size"]!=6 or spec["optimization_steps"]!=0 or spec["remote_training_on_pass"]:
        raise ValueError("expected registered local N36 diagnostic")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists() and not overwrite:
        raise FileExistsError(output/"results.json")
    started=perf_counter()
    manifest=transitive_local_source_manifest((Path(__file__).resolve(),),root=ROOT,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))
    manifest_hash=canonical_sha256(manifest)
    config_hash=canonical_sha256(spec)
    orbit=build_n36_pi_zero_flux_orbit()
    compiled=jax.jit(lambda x:projected_pi_zero_flux_a1_amplitudes_jax(x,orbit=orbit))
    batch_size=int(spec["amplitude_batch_size"])
    counts={"requested_amplitudes":0,"evaluated_padded_amplitudes":0}
    def amplitude(states):
        if perf_counter()-started > spec["wall_limit_seconds"]:
            raise TimeoutError("local diagnostic wall cap reached; completed streams are checkpointed")
        states=np.asarray(states,dtype=np.int8)
        values=[]
        for start in range(0,len(states),batch_size):
            batch=states[start:start+batch_size]
            padded=np.repeat(batch[:1],batch_size,axis=0)
            padded[:len(batch)]=batch
            values.append(np.asarray(compiled(padded))[:len(batch)])
            counts["requested_amplitudes"]+=len(batch)
            counts["evaluated_padded_amplitudes"]+=batch_size
        return np.concatenate(values)
    rng=np.random.default_rng(spec["calibration_seed"])
    calibration=np.zeros((spec["calibration_uniform_count"],36),dtype=np.int8)
    for row in calibration:
        row[rng.choice(36,18,replace=False)]=1
    warm_start=perf_counter()
    calibration_amplitudes=amplitude(calibration)
    warm_seconds=perf_counter()-warm_start
    reference=projected_pi_zero_flux_a1_amplitudes(calibration[:16],orbit=orbit).amplitudes
    replay=float(np.linalg.norm(calibration_amplitudes[:16]-reference)/np.linalg.norm(reference))
    if replay>2e-12:
        raise RuntimeError("carrier replay failed")
    kappa=float(spec["proposal_floor_fraction_of_uniform_mean_weight"]*np.mean(np.abs(calibration_amplitudes)**2))
    if not np.isfinite(kappa) or kappa<=0:
        raise RuntimeError("full-support proposal calibration failed")
    proposal=lambda x:np.sqrt(np.abs(amplitude(x))**2+kappa)
    permutations,coefficients=environment_permutation_bank(36,spec["cells"])
    bonds=triangular_j1j2_bonds(6,6,j1=1.,j2=spec["j2_over_j1"],periodic=True)
    rows,streams=[],[]
    print(f"N36 amplitude replay={replay:.3g}; warmup={warm_seconds:.2f}s; permutations={len(permutations)}; proposal kappa={kappa:.3g}",flush=True)
    for index,seed in enumerate(spec["replicate_seeds"]):
        meta_path=output/f"replicate_{index}.json"
        bundle_path=output/f"replicate_{index}.npz"
        if meta_path.exists() and not overwrite:
            meta=json.loads(meta_path.read_text())
            if meta["config_sha256"]!=config_hash or meta["source_manifest_sha256"]!=manifest_hash or meta["bundle_sha256"]!=file_sha256(bundle_path):
                raise RuntimeError("checkpoint identity differs")
            with np.load(bundle_path,allow_pickle=False) as data:
                streams.append({k:data[k] for k in data.files})
            rows.append(meta)
            continue
        rep_start,before=perf_counter(),dict(counts)
        samples=sample_fixed_wavefunction_exchange(proposal,n_sites=36,n_up=18,
            chain_count=spec["chain_count"],burn_sweeps=spec["burn_exchange_steps_per_chain"],
            measurement_blocks=spec["measurement_blocks"],decorrelation_sweeps=spec["decorrelation_exchange_steps"],seed=seed)
        sampling_seconds=perf_counter()-rep_start
        states=samples.configurations
        psi=amplitude(states)
        action_start=perf_counter()
        actions,cost=evaluate_environment_actions(states,amplitude,permutations,coefficients,batch_size=spec["operator_sample_batch_size"])
        operator_seconds=perf_counter()-action_start
        h_start=perf_counter()
        hvalues=[]
        connected_requests=0
        for start in range(0,len(states),8):
            connected,elements,valid,_=padded_spin_connectivity(states[start:start+8],bonds)
            values=np.zeros(elements.shape,dtype=complex)
            values[valid]=amplitude(connected[valid])
            connected_requests+=int(np.count_nonzero(valid))
            hvalues.append(np.sum(elements*values,axis=1))
        hpsi=np.concatenate(hvalues)
        h_seconds=perf_counter()-h_start
        q=np.abs(psi)**2+kappa
        moments=importance_tangent_moments(psi,hpsi,actions,q)
        nonzero=np.abs(psi)>0
        ratios=hpsi[nonzero]/psi[nonzero]
        if not np.isfinite(ratios).all():
            raise FloatingPointError("non-finite H ratios on nonzero carrier support")
        data={"configurations":states,"psi":psi,"hpsi":hpsi,"actions":actions,"proposal_squared":q}
        np.savez_compressed(bundle_path,**data)
        meta={"index":index,"seed":seed,"sample_count":len(states),"config_sha256":config_hash,
              "source_manifest_sha256":manifest_hash,"bundle_sha256":file_sha256(bundle_path),
              "moments":moments,"acceptance_fraction":samples.acceptance_fraction,
              "burn_acceptance_fraction":samples.burn_acceptance_fraction,
              "carrier_zero_samples":int(np.sum(~nonzero)),"max_abs_H_ratio_on_nonzero_support":float(np.max(np.abs(ratios))),
              "operator_cost":cost,"H_connected_requests":connected_requests,"full_H_calls":len(states),
              "sampling_seconds":sampling_seconds,"operator_seconds":operator_seconds,"H_seconds":h_seconds,
              "wall_seconds":perf_counter()-rep_start,
              "amplitude_work":{k:counts[k]-before[k] for k in counts}}
        atomic_json(meta_path,meta)
        rows.append(meta); streams.append(data)
        print(f"replicate={index} E/N={moments['energy_total']/36:.7f}; OF derivative/N={moments['energy_derivative_total'][-1]/36:+.4g}; ESS={moments['importance_ess_fraction']:.3f}; {meta['wall_seconds']:.2f}s",flush=True)
    data={k:np.concatenate([s[k] for s in streams]) for k in streams[0]}
    aggregate=importance_tangent_moments(data["psi"],data["hpsi"],data["actions"],data["proposal_squared"])
    # Resample whole persistent chains, never treat correlated snapshots as iid.
    nchains,nblocks,nrep=spec["chain_count"],spec["measurement_blocks"],len(streams)
    chain_indices=np.arange(nrep*nblocks*nchains).reshape(nrep,nblocks,nchains).transpose(0,2,1).reshape(nrep*nchains,nblocks)
    boot=[]
    rng=np.random.default_rng(spec["bootstrap_seed"])
    for _ in range(spec["bootstrap_chain_replicates"]):
        ix=chain_indices[rng.integers(0,len(chain_indices),len(chain_indices))].ravel()
        m=importance_tangent_moments(data["psi"][ix],data["hpsi"][ix],data["actions"][ix],data["proposal_squared"][ix])
        boot.append([m["energy_total"],m["variance_total"],*m["energy_derivative_total"]])
    stderr=np.std(boot,axis=0,ddof=1)
    aggregate["chain_bootstrap_se"]={"energy_total":float(stderr[0]),"variance_total":float(stderr[1]),"energy_derivative_total":stderr[2:].tolist()}
    aggregate["sample_count"]=len(data["psi"])
    aggregate["independent_persistent_chain_count"]=len(chain_indices)
    result={"task":spec["task"],"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),
            "config":spec,"config_sha256":config_hash,"carrier_replay_relative_error":replay,
            "warmup_seconds":warm_seconds,"proposal_kappa":kappa,"action_labels":ENVIRONMENT_ACTION_LABELS,
            "replicates":rows,"aggregate":aggregate,"wall_seconds_this_invocation":perf_counter()-started,
            "runtime_identity":numerical_runtime_identity(),"source_manifest":manifest,
            "remote_training_licensed":False,"hamiltonian_diagonalization_performed":False,
            "claim_boundary":"CPU feasibility and projected first derivatives at a zero-Lanczos exact-A1 carrier only. No projected metric or trained/dressed energy. Full-support proposal includes nodal contributions; finite self-normalization and short-chain bias remain. No N36 beyond-three-mode claim."}
    result["result_sha256"]=canonical_sha256(result)
    atomic_json(output/"results.json",result)
    return result
