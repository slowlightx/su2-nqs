"""Reproducible Phase 0 reference/neural runs and Phase 1 ED reconnaissance."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import csv
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import load_checkpoint, save_checkpoint, source_hashes, write_json
from .configuration import validate_config
from .initialization import initialize_run
from .exact import TFIMReference
from .basis import all_bits
from .hamiltonians import IsingCouplings, ground_state
from .models import GRUConfig, coupling_scores, init_model, log_psi, sample
from .observables import correlations_from_state, fidelity_metric, quantum_geometric_tensor, renyi2_from_state, sampled_correlations
from .scaling import collapse_fit, fit_c, fit_eta, phase0_gate, power_fit
from .training import Task, init_optimizer, local_energies, train


def provenance(config: dict) -> dict:
    """Record UTC time, versions, exact config and hashes with every result."""
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
            "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()], "source_sha256": source_hashes(),
            "conventions": {"operators": "Pauli", "bits": "0=-1,1=+1,site0=MSB",
                            "couplings": ["j1","j2","jx","h"], "chi": "total", "entropy": "Renyi2,natural_log"}}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_reference(config: dict, output: Path) -> dict:
    """Calibrate extraction at research sizes against scalable exact TFIM data."""
    from scipy.optimize import minimize_scalar

    validate_config("reference",config)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    sizes = config["sizes"]
    rows, critical, peaks = [], [], []
    for n in sizes:
        for x in np.linspace(-config["window_x"], config["window_x"], config["points"]):
            h = float(1 + x / n)
            ref = TFIMReference(n,h)
            rows.append({"n": n,"h": h,"energy_total": ref.energy(),
                         "chi_total": ref.fidelity_susceptibility(),"chi_per_site": ref.fidelity_susceptibility()/n})
        ref = TFIMReference(n,1)
        correlations = np.array([ref.correlation(r) for r in range(n)])
        ells = np.arange(max(4,n//4),n//2+1)
        entropy = np.array([ref.entropy(int(ell),2) for ell in ells])
        eta = fit_eta(n,correlations,max(4,n//8)).to_dict()
        c = fit_c(n,ells,entropy,2).to_dict()
        critical.append({"n":n,"correlations":correlations.tolist(),"ells":ells.tolist(),"renyi2":entropy.tolist(),"eta_fit":eta,"c_fit":c})
        peak = minimize_scalar(lambda h: -TFIMReference(n,float(h)).fidelity_susceptibility(),
                               bounds=(.7,1.1),method="bounded",options={"xatol":1e-12})
        peaks.append({"n":n,"h_peak":float(peak.x),"chi_peak":float(-peak.fun)})
    collapse = collapse_fit([r["n"] for r in rows],[r["h"] for r in rows],[r["chi_total"] for r in rows])
    # Fit-window, polynomial, and leave-smallest-size-out stability are diagnostics.
    sensitivity = []
    for degree in [4,6]:
        for drop in [False,True]:
            subset = [r for r in rows if not drop or r["n"]>min(sizes)]
            sensitivity.append({"drop_smallest_size":drop,**collapse_fit([r["n"] for r in subset],[r["h"] for r in subset],[r["chi_total"] for r in subset],degree=degree)})
    slope = power_fit(np.asarray(sizes),np.array([TFIMReference(n,1).fidelity_susceptibility() for n in sizes]))
    # Finite TFIM peak drift is O(1/L^2), not assumed generic for Phase 1.
    peak_coefficients = np.polyfit(1/np.array(sizes,dtype=float)**2,[p["h_peak"] for p in peaks],1)
    audit = {"max_energy_abs_error":0.,"max_correlation_abs_error":0.,"max_renyi2_abs_error":0.,"max_ed_residual":0.}
    for n in [4,6,8,10]:
        for h in [.7,1.,1.3]:
            energy,state,residual = ground_state(n,IsingCouplings(h=h))
            ref = TFIMReference(n,h)
            corr = correlations_from_state(state,n)
            audit["max_energy_abs_error"] = max(audit["max_energy_abs_error"],abs(energy-ref.energy()))
            audit["max_ed_residual"] = max(audit["max_ed_residual"],residual)
            audit["max_correlation_abs_error"] = max(audit["max_correlation_abs_error"],max(abs(corr[r]-ref.correlation(r)) for r in range(n)))
            audit["max_renyi2_abs_error"] = max(audit["max_renyi2_abs_error"],max(abs(renyi2_from_state(state,n,ell)-ref.entropy(ell,2)) for ell in range(1,n)))
    _plot_reference(output,rows,critical,collapse)
    result = {**provenance(config), "kind":"exact_reference_calibration", "neural_result":False,
              "collapse":collapse,"sensitivity":sensitivity,"critical":critical,"peaks":peaks,
              "hc_peak_extrapolation":float(peak_coefficients[1]),"nu_effective_peak_height":2/slope.value,
              "independent_ed_audit":audit,"tolerance_abs":4e-11,
              "elapsed_seconds":time.monotonic()-started}
    write_json(output/"reference.json",result)
    _write_csv(output/"reference.csv",rows)
    return result


def _plot_reference(output: Path, rows: list[dict], critical: list[dict], collapse: dict) -> None:
    """Export a standalone research figure, independent of notebook state."""
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"fnqs-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout="constrained")
    for data in critical:
        n=data["n"];subset=[r for r in rows if r["n"]==n]
        h=np.array([r["h"] for r in subset]);chi=np.array([r["chi_total"] for r in subset])
        x=(h-collapse["hc"])*n**(1/collapse["nu"])
        scaled=chi/n**(2/collapse["nu"])
        correction=np.polynomial.chebyshev.chebval(x/collapse["x_scale"],collapse["correction_coefficients"])/n
        line,=axes[0].plot(x,scaled-correction,label=f"L={n}")
        axes[0].plot(x,scaled,":",color=line.get_color(),alpha=.25)
        r=np.arange(max(4,n//8),n//2+1);chord=n/np.pi*np.sin(np.pi*r/n)
        axes[1].loglog(chord,np.asarray(data["correlations"])[r],".-",label=f"L={n}")
        ell=np.asarray(data["ells"])
        axes[2].plot(np.log(n/np.pi*np.sin(np.pi*ell/n)),data["renyi2"],".-",label=f"L={n}")
    axes[0].set(xlabel=r"$(h-h_c)L^{1/\nu}$",ylabel=r"$\chi_F/L^{2/\nu}$ (1/L correction removed)",title=f"Exact reference: hc={collapse['hc']:.5f}, nu={collapse['nu']:.4f}")
    axes[1].set(xlabel="PBC chord distance",ylabel="C(r)",title="Critical correlations")
    axes[2].set(xlabel="log(PBC chord length)",ylabel="S2",title="Renyi-2: slope = c/4")
    axes[0].legend(fontsize=8)
    fig.savefig(output/"reference.png",dpi=180)
    plt.close(fig)


def _audit_exact(params: dict, config: GRUConfig, n: int, g: IsingCouplings) -> dict:
    bits=jnp.asarray(all_bits(n));vector=jnp.asarray(g.vector())
    logs=np.asarray(log_psi(params,bits,vector,config));psi=np.exp(logs)
    local=np.asarray(local_energies(params,bits,vector,config));p=psi**2
    energy=float(p@local);variance=float(p@(local-energy)**2)
    ref=TFIMReference(n,g.h,g.j1)
    correlations=correlations_from_state(psi,n)
    scores=np.asarray(coupling_scores(params,bits,vector,config))
    metric=quantum_geometric_tensor(scores,p).real
    delta=1e-4
    plus=np.exp(np.asarray(log_psi(params,bits,vector.at[3].add(delta),config)))
    minus=np.exp(np.asarray(log_psi(params,bits,vector.at[3].add(-delta),config)))
    finite=fidelity_metric(psi,plus,minus,delta)
    _,ed_state,residual=ground_state(n,g)
    return {"n":n,"h":g.h,"energy_total":energy,"energy_relative_error":abs(energy-ref.energy())/abs(ref.energy()),
            "variance_total":variance,"norm_error":abs(float(p.sum())-1),
            "infidelity":max(0.,float(1-abs(np.vdot(ed_state,psi))**2)),"ed_residual":residual,
            "correlations":correlations.tolist(),"max_correlation_abs_error":max(abs(correlations[r]-ref.correlation(r)) for r in range(n)),
            "chi_total":float(metric[3,3]),"chi_reference":ref.fidelity_susceptibility(),
            "chi_relative_error":abs(float(metric[3,3])-ref.fidelity_susceptibility())/ref.fidelity_susceptibility(),
            "coupling_metric":metric.tolist(),"ad_fidelity_abs_discrepancy":abs(float(metric[3,3])-finite),
            "renyi2_half":renyi2_from_state(psi,n,n//2),"renyi2_half_reference":ref.entropy(n//2,2)}


@partial(jax.jit, static_argnames=("config",))
def _sampled_audit_values(params: dict, bits: jax.Array, vector: jax.Array,
                          config: GRUConfig) -> tuple[jax.Array, jax.Array]:
    """Reuse compilation across fields and seeds; all numerical inputs vary."""
    return local_energies(params, bits, vector, config), coupling_scores(params, bits, vector, config)


def _audit_sampled(params: dict, config: GRUConfig, n: int, g: IsingCouplings,
                   key: jax.Array, count: int, chunk: int = 64) -> dict:
    """Fresh independent audit; chunk connected-state evaluation to bound memory."""
    vector=jnp.asarray(g.vector())
    bits=sample(params,key,count,n,vector,config)
    values=[_sampled_audit_values(params,bits[start:start+chunk],vector,config)
            for start in range(0,count,chunk)]
    local=np.concatenate([np.asarray(value[0]) for value in values])
    scores=np.concatenate([np.asarray(value[1]) for value in values])
    metric=quantum_geometric_tensor(scores).real
    correlations,errors=sampled_correlations(np.asarray(bits))
    ref=TFIMReference(n,g.h,g.j1)
    # Whole-independent-draw bootstrap for the diagonal coupling covariance.
    rng=np.random.default_rng(int(np.asarray(key)[-1]))
    chi_bootstrap=np.array([np.var(scores[rng.integers(count,size=count),3]) for _ in range(200)])
    chi_se=float(chi_bootstrap.std(ddof=1))
    return {"n":n,"h":g.h,"energy_total":float(local.mean()),
            "energy_se":float(local.std(ddof=1)/np.sqrt(count)),"energy_relative_error":abs(float(local.mean())-ref.energy())/abs(ref.energy()),
            "variance_total":float(local.var()),"correlations":correlations.tolist(),"correlation_se":errors.tolist(),
            "max_correlation_abs_error":max(abs(correlations[r]-ref.correlation(r)) for r in range(n)),
            "chi_total":float(metric[3,3]),"chi_se":chi_se,"chi_reference":ref.fidelity_susceptibility(),
            "chi_relative_error":abs(float(metric[3,3])-ref.fidelity_susceptibility())/ref.fidelity_susceptibility(),
            "coupling_metric":metric.tolist(),"audit_samples":count,"audit_independent_of_training":True}


def run_neural(config: dict, output: Path, resume: bool = False) -> dict:
    """Train shared multi-L states and audit withheld h values at fixed theta."""
    validate_config("train",config)
    started=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    model=GRUConfig(**config["model"])
    tasks=[Task(n,IsingCouplings(h=h)) for n in config["sizes"] for h in config["train_fields"]]
    expected_signature={"model":config["model"],"sizes":config["sizes"],"train_fields":config["train_fields"],
                        "mode":config["mode"],"optimizer":config["optimizer"],"learning_rate":config["learning_rate"],
                        "samples_per_task":config["samples_per_task"],"sr_shift":config.get("sr_shift",1e-3)}
    training_options={name:config.get(name,default) for name,default in {
        "sr_solver":"dense","sr_tolerance":1e-8,"sr_maxiter":500,
        "sr_max_step_norm":None,"sr_backtracking":False,"tasks_per_step":None}.items()}
    expected_signature.update(training_options)
    warm_path=Path(config["warm_start_checkpoint"]) if config.get("warm_start_checkpoint") else None
    if warm_path is not None:
        import hashlib
        expected_signature["warm_start_sha256"]=hashlib.sha256(warm_path.read_bytes()).hexdigest()
    arms=[]
    for seed in config["seeds"]:
        checkpoint=output/f"seed_{seed}.npz"
        past_history=[]
        if resume:
            params,state,key,metadata=load_checkpoint(checkpoint)
            if metadata["signature"]!=expected_signature or metadata["seed"]!=seed:
                raise ValueError("checkpoint training configuration differs from requested run")
            if metadata["source_sha256"]!=source_hashes():
                raise ValueError("project sources changed since checkpoint; resume requires identical numerical code")
            past_history=metadata["history"]
            warm_origin=metadata.get("warm_start")
        else:
            if checkpoint.exists():
                raise FileExistsError(f"checkpoint already exists: {checkpoint}; use --resume or a new output")
            params,state,key,warm_origin=initialize_run(seed,model,warm_path)
        initial_step=int(state["step"])
        params,state,key,history=train(params,tasks,model,key,steps=config["steps"],learning_rate=config["learning_rate"],
                                     mode=config["mode"],optimizer=config["optimizer"],n_samples=config["samples_per_task"],
                                     sr_shift=config.get("sr_shift",1e-3),state=state,
                                     **training_options,
                                     callback=lambda record:print(f"seed={seed} step={record['step']} E/N={record['energy_per_site']:.8f}",flush=True))
        metadata={"seed":seed,"signature":expected_signature,"source_sha256":source_hashes(),"history":past_history+history,
                  "warm_start":warm_origin}
        # Save training key BEFORE audit; resume does not depend on audit sample count.
        save_checkpoint(checkpoint,params,state,key,metadata)
        rows=[];audit_key=jax.random.fold_in(key,0xA0D17)
        audit_sizes=sorted(set(config["sizes"]+config.get("test_sizes",[])))
        for index,(n,h) in enumerate((n,h) for n in audit_sizes for h in sorted(set(config["train_fields"]+config["test_fields"]))):
            if config["mode"]=="exact":
                row=_audit_exact(params,model,n,IsingCouplings(h=h))
            else:
                row=_audit_sampled(params,model,n,IsingCouplings(h=h),jax.random.fold_in(audit_key,index),config["audit_samples"])
            row["split"]="train" if h in config["train_fields"] else "held_field"
            if n not in config["sizes"]:row["split"]="held_size_"+row["split"]
            rows.append(row)
        extracted={"hc":None,"nu":None,"eta":None}
        fit_rows=[r for r in rows if r["n"] in config["sizes"] and abs((r["h"]-1)*r["n"])<=config.get("collapse_window_x",1.0)]
        if min(config["sizes"])>=16 and len(set(r["n"] for r in fit_rows))>=3 and len(fit_rows)>=18:
            fit=collapse_fit([r["n"] for r in fit_rows],[r["h"] for r in fit_rows],[r["chi_total"] for r in fit_rows],
                             chi_se=np.asarray([r["chi_se"] for r in fit_rows]) if all("chi_se" in r for r in fit_rows) else None)
            extracted.update(hc=fit["hc"],nu=fit["nu"],collapse=fit)
            critical_rows=[r for r in fit_rows if abs(r["h"]-1)<1e-12]
            if critical_rows:
                eta_fits=[fit_eta(r["n"],r["correlations"],max(4,r["n"]//8)).value for r in critical_rows]
                extracted["eta"]=float(np.mean(eta_fits))
        else:
            extracted["reason"]="Insufficient research sizes or critical-window grid; no exponent estimate."
        # Observable accuracy is stricter than energy alone. Thresholds are
        # declared diagnostics, not statistical confidence or universal laws.
        accurate=True
        for row in rows:
            ref=TFIMReference(row["n"],row["h"])
            correlation_error=max(abs(row["correlations"][r]-ref.correlation(r))
                                  +2*row.get("correlation_se",[0.]*row["n"])[r] for r in range(row["n"]))
            energy_bound=row["energy_relative_error"]+2*row.get("energy_se",0.)/abs(ref.energy())
            chi_bound=row["chi_relative_error"]+2*row.get("chi_se",0.)/row["chi_reference"]
            accurate=accurate and chi_bound<=.1 and correlation_error<=.03 and energy_bound<=.002
            accurate=accurate and row.get("norm_error",0.)<=1e-10
            accurate=accurate and row.get("ad_fidelity_abs_discrepancy",0.)<=1e-5*max(1.,row["chi_total"])
        arms.append({"seed":seed,"initial_step":initial_step,"final_step":int(state["step"]),"history":history,
                     "warm_start":warm_origin,
                     "training_work":state["work"],
                     "rows":rows,"extracted":extracted,"observable_checks_passed":accurate})
        write_json(output/"neural.partial.json",{
            **provenance(config),"kind":"shared_neural_phase0_partial","arms":arms,
            "complete":False,"expected_seeds":config["seeds"],
            "phase0_scientific_gate":{"passed":False,"reason":"awaiting completed batch"}})
    extracted=[arm["extracted"] for arm in arms]
    mean=lambda name:float(np.mean([r[name] for r in extracted])) if all(r[name] is not None for r in extracted) else None
    gate=phase0_gate(mean("hc"),mean("nu"),mean("eta"),all(arm["observable_checks_passed"] for arm in arms),config["sizes"],len(arms))
    individual=[]
    for arm in arms:
        fit=arm["extracted"]
        individual.append(phase0_gate(fit["hc"],fit["nu"],fit["eta"],arm["observable_checks_passed"],config["sizes"],len(arms))["passed"]
                          and fit.get("collapse",{}).get("success",False) and not fit.get("collapse",{}).get("at_bound",True))
    gate["checks"]["every_seed_extraction_passes"]=all(individual)
    gate["passed"]=gate["passed"] and all(individual)
    gate["missing"]=[name for name,passed in gate["checks"].items() if not passed]
    result={**provenance(config),"kind":"shared_neural_phase0", "arms":arms,"phase0_scientific_gate":gate,
            "ensemble_objective":"equal-task mean energy per site; uniform task minibatches when configured",
            "tasks_per_update":config.get("tasks_per_step") or len(tasks),
            "gradient_hamiltonians_per_seed":config["steps"]*(config.get("tasks_per_step") or len(tasks)),
            "total_samples_per_update":(config.get("tasks_per_step") or len(tasks))*config["samples_per_task"] if config["mode"]=="vmc" else 0,
            "observable_thresholds":{"energy_relative":.002,"chi_relative":.1,"correlation_absolute":.03,
                                     "sampled_error_margin":"absolute error plus 2 standard errors; descriptive, not simultaneous confidence"},
            "elapsed_seconds":time.monotonic()-started,"phase1_training_authorized_by_gate":gate["passed"]}
    write_json(output/"neural.json",result)
    return result


def run_reconnaissance(config: dict, output: Path) -> dict:
    """Phase 1 ED reconnaissance only: no universality/foundation claim."""
    validate_config("recon",config)
    started=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    cuts=[]
    for family in config["families"]:
        for n in config["sizes"]:
            fields=np.asarray(config["fields"],dtype=float)
            if len(fields)<5 or np.any(np.diff(fields)<=0):
                raise ValueError("Phase 1 fields must be strictly increasing with >=5 points")
            states,rows=[],[]
            for h in fields:
                g=IsingCouplings(j2=family.get("j2",0.),jx=family.get("jx",0.),h=float(h))
                energy,state,residual=ground_state(n,g)
                correlations=correlations_from_state(state,n)
                states.append(state)
                rows.append({"h":float(h),"energy_total":energy,"ed_residual":residual,
                             "m2":float(correlations.mean()),"c_half":float(correlations[n//2]),
                             "renyi2_half":renyi2_from_state(state,n,n//2)})
            susceptibility=[]
            for i in range(1,len(fields)-1):
                plus,minus=fields[i+1]-fields[i],fields[i]-fields[i-1]
                susceptibility.append(float(((1-abs(np.vdot(states[i],states[i+1]))**2)/plus**2+
                                             (1-abs(np.vdot(states[i],states[i-1]))**2)/minus**2)/2))
            peak=int(np.argmax(susceptibility))+1
            cuts.append({"family":family["name"],"j2":family.get("j2",0.),"jx":family.get("jx",0.),"n":n,
                         "rows":rows,"fields_chi":fields[1:-1].tolist(),"chi_finite_difference":susceptibility,
                         "peak_field":float(fields[peak]),"peak_bracket":[float(fields[peak-1]),float(fields[peak+1])],
                         "peak_on_grid_edge":peak in [1,len(fields)-2]})
    _plot_reconnaissance(output,cuts)
    result={**provenance(config),"kind":"phase1_ed_reconnaissance","neural_result":False,
            "universality_established":False,"cuts":cuts,"elapsed_seconds":time.monotonic()-started,
            "next":"Refine peak brackets and obtain large-L independent references before interpreting Family B/C exponents."}
    write_json(output/"reconnaissance.json",result)
    return result


def _plot_reconnaissance(output: Path, cuts: list[dict]) -> None:
    """Finite-size reconnaissance curves, explicitly labeled without hc claims."""
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"fnqs-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    largest=max(cut["n"] for cut in cuts)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for cut in cuts:
        if cut["n"]!=largest:continue
        axes[0].plot(cut["fields_chi"],cut["chi_finite_difference"],label=cut["family"])
        axes[1].plot([r["h"] for r in cut["rows"]],[r["m2"] for r in cut["rows"]],label=cut["family"])
    axes[0].set(xlabel="h (Pauli units)",ylabel="Total finite-difference fidelity susceptibility",title=f"ED L={largest}: peaks are finite-size brackets")
    axes[1].set(xlabel="h (Pauli units)",ylabel=r"$m^2=L^{-2}\sum_{ij}\langle Z_iZ_j\rangle$",title="No universality assignment")
    axes[0].legend(fontsize=8)
    fig.savefig(output/"reconnaissance.png",dpi=180)
    plt.close(fig)
