"""Independent host replay of sealed seven-point AGP audit products.

No optimization, samples or many-body diagonalization are generated.  The
NumPy signed-determinant reference and an explicit bond sum verify one saved
configuration at each initial/final quality checkpoint.  Raw importance
moments, pooled whole-chain errors, and physical RDM legality are replayed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_agp_seven_point import (
    ROOT, lattice_contract, read_sealed, sealed_json,
)
from su2_nqs.benchmarks.projected_singlet_agp import ProjectedSingletAGP
from su2_nqs.benchmarks.spin_operator_information import three_bond_readout
from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint


def independent_moments(psi: np.ndarray, hpsi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return E/N and Var(H)/N via direct first/second moments, including nodes."""
    if (psi.ndim != 1 or psi.shape != hpsi.shape or q.shape != psi.shape or np.any(q <= 0)
            or not np.isfinite(psi).all() or not np.isfinite(hpsi).all() or not np.isfinite(q).all()):
        raise ValueError("matching amplitudes and positive proposal density required")
    norm = np.sum(abs(psi)**2/q)
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("resolved physical norm required")
    energy = np.sum((psi.conj()*hpsi).real/q)/norm
    h2 = np.sum(abs(hpsi)**2/q)/norm
    return np.array([energy/36, (h2-energy**2)/36])


def independent_comparison(arrays: dict, q: np.ndarray, chains: int) -> dict:
    """Replay pooled ratios and delete-whole-chain paired errors independently."""
    labels = list(arrays); groups = np.arange(len(q)) % chains
    central = {key: independent_moments(*arrays[key], q) for key in labels}
    deletes = {key: np.stack([independent_moments(arrays[key][0][groups != c],
        arrays[key][1][groups != c], q[groups != c]) for c in range(chains)]) for key in labels}
    def se(values):
        return np.sqrt((chains-1)*np.mean((values-values.mean(0))**2, axis=0))
    pairs = list(zip(labels[:-1], labels[1:]))
    if len(labels) > 2:
        pairs.append((labels[0], labels[-1]))
    return dict(states={key: dict(energy=central[key][0], variance=central[key][1],
        energy_chain_SE=se(deletes[key])[0], variance_chain_SE=se(deletes[key])[1]) for key in labels},
        paired={after+"-minus-"+before: dict(energy=(central[after]-central[before])[0],
            variance=(central[after]-central[before])[1],
            energy_chain_SE=se(deletes[after]-deletes[before])[0],
            variance_chain_SE=se(deletes[after]-deletes[before])[1])
            for before, after in pairs})


def regional_quality_records(raw, report: dict, regions: dict, labels: list[str]) -> dict:
    """Read same-record physical p(q)/scalar quality changes with paired errors."""
    result = {}
    for name, spec in regions.items():
        def observe(rho):
            block = physical_spin_block_readout(rho)
            values = {"p_q"+q: value for q, value in block["p_q"].items()}
            if len(spec["sites"]) == 6:
                triple = three_bond_readout(rho, spec["left"])
                values.update(high_q3=triple["high_q3"], Y_connected=triple["orbits"]["Y"]["connected"])
            return values
        central = {label: observe(raw[name+"_"+label+"_rho"]) for label in labels}
        jack_values = {label: [observe(r) for r in raw[name+"_"+label+"_jackknife_rhos"]]
                       for label in labels}
        deletes = {label: {key: np.array([row[key] for row in jack_values[label]])
                          for key in central[label]} for label in labels}
        chains = len(next(iter(deletes[labels[0]].values())))
        def se(values):
            return float(np.sqrt((chains-1)*np.mean((values-values.mean())**2)))
        initial, endpoint = labels[0], labels[-1]
        result[name] = dict(states={label: {key: dict(value=value, chain_SE=se(deletes[label][key]))
            for key, value in central[label].items()} for label in labels},
            endpoint_minus_initial={key: dict(value=central[endpoint][key]-central[initial][key],
                paired_chain_SE=se(deletes[endpoint][key]-deletes[initial][key])) for key in central[initial]},
            actual_RDM_SU2_defects={label: report["regions"][name][label]["block"]["su2_factorization_frobenius"]
                                   for label in labels},
            strict_multiplicity_spectrum_claim=False, input_twirl=False)
    return result


def explicit_host_action(reference: ProjectedSingletAGP, bits: np.ndarray,
                         bonds: tuple, scale: float) -> tuple[complex, complex]:
    """Direct physical H[new,old] bond action, without query-plan code or division."""
    amplitude = reference.amplitude(bits)/scale
    action = 0.+0.j
    for first, second, coupling in bonds:
        sz_first, sz_second = int(bits[first])-.5, int(bits[second])-.5
        action += coupling*sz_first*sz_second*amplitude
        if bits[first] != bits[second]:
            flipped = bits.copy()
            flipped[first], flipped[second] = bits[second], bits[first]
            action += .5*coupling*reference.amplitude(flipped)/scale
    return amplitude, action


def audit_case(folder: Path, budget: dict, regions: dict) -> dict:
    """Audit one complete immutable finite run and fail closed on any seal/error."""
    result = read_sealed(folder/"results.json")
    path = folder/"independent_audit_v2.json"
    if path.exists():
        old = read_sealed(path)
        if old["experiment_result_sha256"] != result["result_sha256"]:
            raise ValueError("independently audited experiment changed")
        return old
    completion = read_sealed(folder/"completion.json")
    for name, digest in completion["outputs"].items():
        if file_sha256(folder/name) != digest:
            raise ValueError(f"immutable experiment product changed: {name}")
    if not result["complete_bounded_audit"] or result["completed_steps"] != 12:
        raise ValueError("complete fixed twelve-step and two-audit products required")
    labels = list(result["frozen_checkpoints"])
    errors = []
    raw = []
    regional_records = []
    for index in (0, 1):
        report = read_sealed(folder/f"audit_{index}.json")
        z = np.load(folder/f"audit_{index}.npz")
        if file_sha256(folder/f"audit_{index}.npz") != report["raw_file_sha256"]:
            raise ValueError("raw audit seal changed")
        if np.any(z["configurations"].sum(1) != 18) or np.any(z["density"] <= 0):
            raise ValueError("illegal support or proposal density")
        replay = independent_comparison({key: (z[key+"_psi"], z[key+"_Hpsi"])
                                         for key in labels}, z["density"], budget["chains"])
        for label in labels:
            for observed_key, reference_key in (("energy", "energy_per_site"),
                    ("variance", "variance_per_site"), ("energy_chain_SE", "energy_chain_SE"),
                    ("variance_chain_SE", "variance_chain_SE")):
                errors.append(abs(replay["states"][label][observed_key]-report["states"][label][reference_key]))
        for pair in report["paired"]:
            errors.extend(abs(replay["paired"][pair][key]-report["paired"][pair][key])
                          for key in replay["paired"][pair])
        for region, states in report["regions"].items():
            for label, saved in states.items():
                block = physical_spin_block_readout(z[region+"_"+label+"_rho"])
                if saved["input_twirl"] or saved["positivity_repair"] or saved["strict_multiplicity_spectrum_reported"]:
                    raise ValueError("sampled physical RDM was repaired or given an ineligible spectrum")
                errors.append(abs(block["su2_factorization_frobenius"]-saved["block"]["su2_factorization_frobenius"]))
                errors.extend(abs(block["p_q"][key]-saved["block"]["p_q"][key]) for key in block["p_q"])
        regional_records.append(regional_quality_records(z, report, regions, labels))
        raw.append(z)
    def interleave(items):
        return np.concatenate([a.reshape(budget["audit_blocks"], budget["chains"])
                               for a in items], axis=1).reshape(-1)
    pooled = independent_comparison({key: (interleave([z[key+"_psi"] for z in raw]),
        interleave([z[key+"_Hpsi"] for z in raw])) for key in labels},
        interleave([z["density"] for z in raw]), 2*budget["chains"])
    for label in labels:
        for observed_key, reference_key in (("energy", "energy_per_site"),
                ("variance", "variance_per_site"), ("energy_chain_SE", "energy_chain_SE"),
                ("variance_chain_SE", "variance_chain_SE")):
            errors.append(abs(pooled["states"][label][observed_key]-result["pooled"]["states"][label][reference_key]))
    for pair in result["pooled"]["paired"]:
        errors.extend(abs(pooled["paired"][pair][key]-result["pooled"]["paired"][pair][key])
                      for key in pooled["paired"][pair])
    case = result["case"]
    group, bonds = lattice_contract(case["lattice"], 6, case["j2_over_j1"])
    registration = read_sealed(folder/"registration.json")
    host_errors = []
    for label in (labels[0], labels[-1]):
        checkpoint = load_checkpoint(folder/(label+".pkl"))
        if file_sha256(folder/(label+".pkl")) != result["frozen_checkpoints"][label]:
            raise ValueError("frozen pairing changed")
        p = np.asarray(checkpoint["params"]["pairing_real"])
        matrix = np.zeros((36, 36)); first, second = np.triu_indices(36, 1)
        matrix[first, second] = p; matrix[second, first] = p
        reference = ProjectedSingletAGP(matrix, group)
        psi, hpsi = explicit_host_action(reference, raw[0]["configurations"][0],
                                         bonds, registration["amplitude_scale"])
        cached_psi, cached_hpsi = raw[0][label+"_psi"][0], raw[0][label+"_Hpsi"][0]
        host_errors.append(dict(checkpoint=label, amplitude_absolute_error=float(abs(psi-cached_psi)),
            H_absolute_error=float(abs(hpsi-cached_hpsi)),
            amplitude_relative_error=float(abs(psi-cached_psi)/max(abs(psi), 1e-300)),
            H_relative_error=float(abs(hpsi-cached_hpsi)/max(abs(hpsi), 1e-300)),
            raw_audit_record=0, group_order=len(group), bond_count=len(bonds)))
    for z in raw:
        z.close()
    maximum = float(max(errors))
    if maximum > 5e-11 or any(max(row["amplitude_relative_error"], row["H_relative_error"]) > 2e-10
                             for row in host_errors):
        raise ValueError("independent saved-moment or explicit NumPy-H replay failed")
    return sealed_json(path, dict(experiment_result_sha256=result["result_sha256"],
        independent_audit_source_sha256=file_sha256(Path(__file__)),
        experiment_completion_sha256=file_sha256(folder/"completion.json"),
        maximum_saved_moment_SE_RDM_absolute_error=maximum, saved_moment_tolerance=5e-11,
        explicit_numpy_H_replays=host_errors, explicit_H_relative_tolerance=2e-10,
        pooled_moments=pooled, new_samples=0, new_training_steps=0, new_many_body_ED=0,
        regional_quality_by_independent_audit=regional_records,
        strict_multiplicity_spectrum_claim=False, passed=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text()); output = ROOT/spec["output"]
    registration = read_sealed(output/"registration.json")
    for path, digest in registration["sources"].items():
        if file_sha256(output/"source_snapshot"/path) != digest:
            raise ValueError("registered experiment source snapshot changed")
    rows = []
    for case in spec["cases"]:
        folder = output/case["name"]
        if (folder/"completion.json").exists():
            row = audit_case(folder, spec["budget"], spec["regions"])
            rows.append(dict(case=case["name"], **row))
            print(json.dumps(dict(case=case["name"], passed=row["passed"],
                moment_error=row["maximum_saved_moment_SE_RDM_absolute_error"],
                H_replays=row["explicit_numpy_H_replays"])), flush=True)
    sealed_json(output/"independent_audit_v2.json", dict(rows=rows, requested_cases=7,
        completed_audits=len(rows), sources=transitive_local_source_manifest((Path(__file__),),
        root=ROOT, package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs")),
        new_samples=0, new_many_body_ED=0))


if __name__ == "__main__":
    main()
