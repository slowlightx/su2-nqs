"""Bounded offline readout of existing CT banks and physical regional caches.

No training, sampling, spatial twirl, RDM repair, or remote access. Raw-tree
edge spectra are explicitly separated from physical coherent-A1 readouts.
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns, su2_multiplicity_readout
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank, triangular_bank_reduced_tree, triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.su2.reduced_tree_environment import reduced_tree_environments, canonical_edge_spectrum
from su2_nqs.vmc.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "projects/clebsch_tree_nqs/results"
OUT = RESULTS / "su2_mechanism_local_20261028/regions"


def _entropy(values: np.ndarray) -> float:
    positive = np.asarray(values)[np.asarray(values) > 1e-14]
    return float(-np.sum(positive * np.log(positive)))


def raw_ct(path: Path, arity: str, j2: float) -> dict:
    """Exact raw-state spectra; q labels are doubled and no A1 claim is made."""
    tick = perf_counter()
    saved = load_checkpoint(path)
    config = MixedClebschConfig(length=6, multiplicity_cap=28, maximum_q=10)
    bank = arity_clebsch_bank(saved["params"], config, arity, 8)
    oldnorm = triangular_raw_overlap(bank, bank, config).real
    canonical, qr = canonicalize_triangular_bank(bank, config)
    newnorm = triangular_raw_overlap(canonical, canonical, config).real
    overlap = triangular_raw_overlap(canonical, bank, config)
    fidelity_error = abs(abs(overlap)**2 / (oldnorm * newnorm) - 1)
    if fidelity_error > 1e-10:
        raise ArithmeticError("canonicalization changed the raw physical state")
    state = triangular_bank_reduced_tree(canonical, config)
    env = reduced_tree_environments(state)
    hierarchy = triangular_schur_hierarchy(6)
    nodes = list(state.tree.leaves) + list(state.tree.internal_nodes)
    reports = []
    for edge, sites in enumerate(hierarchy.coarse_supports[:-1]):
        spectrum = canonical_edge_spectrum(env, nodes[edge])
        spin = magnetic = internal = 0.0
        blocks = {}
        for q, block in spectrum.blocks.items():
            weights = block.multiplicity_weights
            p = float(weights.sum())
            conditional = weights / p if p > 1e-14 else np.zeros_like(weights)
            spin += -p * np.log(p) if p > 1e-14 else 0.0
            magnetic += p * np.log(q + 1)
            internal += p * _entropy(conditional)
            blocks[str(q)] = dict(q=q, p=p, stored_width=len(weights), metric_rank=block.metric_rank,
                multiplicity_weights=weights.tolist(), conditional_multiplicity_spectrum=conditional.tolist(),
                conditional_entropy_nats=_entropy(conditional), effective_multiplicity=float(np.exp(_entropy(conditional))))
        reports.append(dict(edge=edge, sites=list(sites), size=len(sites), sectors=blocks,
            entropy_nats=spectrum.entropy, spin_weight_entropy=spin, magnetic_entropy=magnetic,
            multiplicity_entropy=internal, entropy_identity_error=abs(spectrum.entropy-spin-magnetic-internal),
            spin_squared_mean=sum(x["p"]*x["q"]*(x["q"]+2)/4 for x in blocks.values()),
            normalization_error=spectrum.normalized_weight_error))
    return dict(method="ClebschTree", j2=j2, arity=arity,
        parent_checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=file_sha256(path),
        parameter_sha256=pytree_sha256(saved["params"]), absolute_step=saved["metadata"]["absolute_completed_steps"],
        interpretation="EXACT RAW UNPROJECTED TREE ONLY; not the coherent A1 Schmidt spectrum", coherent_A1_spectrum=False,
        region_order="nine-site hierarchy blocks; not DMRG prefix18", canonicalization=qr,
        raw_norm_squared=float(oldnorm), gauge_fidelity_error=float(fidelity_error),
        maximum_edge_norm_error=env.maximum_relative_edge_norm_error, edges=reports,
        wall_seconds=perf_counter()-tick)


def regional(rho: np.ndarray, jackknife: np.ndarray | None = None) -> dict:
    """Legal invariant algebra readout even if rho has symmetry/sampling error.

    The magnetic-traced conditional multiplicity density is legitimate even
    without covariance, but is NOT called a physical SU2 entanglement spectrum.
    The latter and entropy decomposition require strict factorization checks.
    """
    n = int(round(np.log2(rho.shape[0])))
    u, labels = regional_schur_columns(n)
    transformed = u.conj().T @ rho @ u
    jack_transformed = u.conj().T @ jackknife @ u if jackknife is not None else None
    raising = np.zeros_like(rho)
    for packed in range(len(rho)):
        for site in range(n):
            mask = 1 << site
            if not packed & mask:
                raising[packed | mask, packed] += 1
    sectors = {}
    input_psd = float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min()) >= -1e-10
    for q in range(n % 2, n+1, 2):
        ids = [i for i, label in enumerate(labels) if label[0] == q]
        projector = u[:, ids] @ u[:, ids].conj().T
        p = float(np.trace(projector @ rho).real)
        se = None
        if jackknife is not None:
            vals = np.einsum("ij,bji->b", projector, jackknife).real
            se = float(np.sqrt((len(vals)-1)/len(vals)*np.sum((vals-vals.mean())**2)))
        local = [ [i for i, label in enumerate(labels) if label[:2] == (q,m)]
                  for m in range(-q,q+1,2) ]
        reduced = sum(transformed[np.ix_(ids, ids)] for ids in local)
        conditional = None
        spectrum_se = None
        if input_psd and p > 1e-14:
            conditional = (np.linalg.eigvalsh((reduced+reduced.conj().T)/2)[::-1]/p).tolist()
            if jack_transformed is not None:
                values = []
                for matrix in jack_transformed:
                    sub = sum(matrix[np.ix_(ids, ids)] for ids in local)
                    probability = float(np.trace(sub).real)
                    if probability <= 1e-14:
                        break
                    values.append(np.linalg.eigvalsh((sub+sub.conj().T)/2)[::-1]/probability)
                if len(values) == len(jack_transformed):
                    values = np.asarray(values)
                    spectrum_se = np.sqrt((len(values)-1)/len(values)*np.sum((values-values.mean(axis=0))**2, axis=0)).tolist()
        sectors[str(q)] = dict(q=q, p=p, independent_chain_jackknife_se=se,
            conditional_multiplicity_reduced_spectrum=conditional,
            conditional_spectrum_chain_jackknife_se=spectrum_se,
            conditional_spectrum_interpretation="Physical local-charge postselection followed by magnetic trace; not a strict SU2 entanglement spectrum unless covariance separately passes",
            spectral_uncertainty_not_estimated=spectrum_se is None,
            spectrum_bias_warning="Conditional normalization and eigenvalue plugin bias are not removed by this jackknife SE")
    minimum = float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min())
    try:
        strict = su2_multiplicity_readout(rho, tolerance=1e-9)
        failure = None
    except ValueError as error:
        strict, failure = None, str(error)
    return dict(region_size=n, spin_sector_weights=sectors,
        minimum_eigenvalue=minimum, trace_error=float(abs(np.trace(rho)-1)),
        hermiticity_error=float(np.linalg.norm(rho-rho.conj().T)),
        raising_commutator_frobenius=float(np.linalg.norm(rho@raising-raising@rho)),
        strict_SU2_spectrum_accepted=strict is not None, strict_report=strict,
        withholding_reason=failure, input_repaired=False)


def physical_assets() -> list[dict]:
    """Actual same-six-site caches; independent chains retained for ViT errors."""
    paths = [
        ("ViT", 0., RESULTS/"coupling_quality_review_20261012/latest_inputs/readouts/tri_j0_continue_step00750_regional/physics.npz"),
        ("ViT", 0., RESULTS/"coupling_quality_review_20261012/latest_inputs/readouts/tri_j0_continue_step01000_regional/physics.npz"),
        ("ViT", .125, RESULTS/"spin_spatial_vit_gpu_continuation_6600_v1/analysis/six_site_readout/physics.npz"),
        ("ViT", .125, RESULTS/"spin_spatial_vit_gpu_continuation_6600_v1/analysis/local_hexagon6_readout_v2/physics.npz"),
        ("ViT", .125, RESULTS/"spin_spatial_vit_gpu_continuation_7600_v1/analysis/witness_regions_v1/physics.npz"),
        ("DMRG", .125, RESULTS/"spin_tensor_network_continuation_v2/chi4096/rdms.npz"),
        ("DMRG", .125, RESULTS/"spin_tensor_network_continuation_v2/chi5120_probe/rdms.npz"),
        ("DMRG", .125, RESULTS/"coupling_quality_review_20261012/latest_inputs/dmrg/work/chi10240_from8192_v1/rdms.npz"),
        ("DMRG", .125, RESULTS/"coupling_quality_review_20261012/latest_inputs/dmrg/readouts/chi10240_sweep32_hexagon6_env_v1/regions/local_hexagon6/rdm.npz"),
    ]
    output = []
    for method, j2, path in paths:
        if not path.exists():
            continue
        tick = perf_counter()
        provenance = {}
        for sidecar_name in ("results.json", "registration.json", "readout.json"):
            sidecar = path.parent/sidecar_name
            if sidecar.exists():
                payload = json.loads(sidecar.read_text())
                provenance[sidecar_name] = dict(sha256=file_sha256(sidecar),
                    recorded_identity={k:payload[k] for k in ("params_sha256", "parameter_sha256", "checkpoint_sha256", "step", "physical_problem_signature", "registration_sha256") if k in payload})
        with np.load(path, allow_pickle=False) as f:
            for key in f.files:
                rho = f[key]
                if rho.shape not in [(64,64), (16,16)]:
                    continue
                if not (key.endswith("__rho") or key.startswith("chi") or key == "rho"):
                    continue
                region = key.removesuffix("__rho").split("_", 1)[-1] if key.startswith("chi") else key.removesuffix("__rho")
                if region not in ("compact6", "winding6", "local_hexagon6", "rho", "compact4"):
                    continue
                jackkey = key.replace("__rho", "__jackknife_rhos") if "__rho" in key else "jackknife_rhos"
                jack = f[jackkey] if jackkey in f.files else None
                siteskey = key.replace("__rho", "__region_sites") if "__rho" in key else "region_sites"
                sites = f[siteskey].tolist() if siteskey in f.files else None
                if sites is None:
                    sites = dict(compact6=[12,13,14,18,19,20], winding6=[12,13,14,15,16,17]).get(region)
                if region == "rho" and path.parent.name == "local_hexagon6":
                    region = "local_hexagon6"
                    request = next(path.parent.glob("request_*.json"), None)
                    if request is not None:
                        payload = json.loads(request.read_text())
                        sites = payload.get("region", {}).get("sites", payload.get("sites", payload.get("region_sites", sites)))
                output.append(dict(method=method, j2=j2, source=str(path.relative_to(ROOT)),
                    source_sha256=file_sha256(path), region=region, sites=sites,
                    provenance_sidecars=provenance,
                    physical_not_raw_tree=True, readout=regional(rho, jack), wall_seconds=perf_counter()-tick))
    return output


def dmrg_large_cuts() -> list[dict]:
    """Summarize cached physical Casimir/spatial measurements, not excitation E."""
    output = []
    for name in ["ordered_J0_chi2048_cut18", "ordered_J0_chi4096_cut18", "chi8192_cut18", "chi10240_cut18"]:
        path = RESULTS/"rotor_families_20261015/mps"/name/"results.json"
        full = json.loads(path.read_text())
        r = full["readout"]["readout"]
        rows = {}
        for m, values in r["leading_spin_content"]["sectors"].items():
            spatial = r["leading_spatial_content"][m]
            levels = []
            for i, level in enumerate(values["leading_levels"]):
                casimir = level["physical_casimir_expectation"]
                nearest_q = int(round(np.sqrt(max(0,1+4*casimir))-1))
                levels.append(dict(level_index=i, eigenvalue=level["eigenvalue"],
                    entanglement_energy=level["entanglement_energy"], casimir_expectation=casimir,
                    nearest_doubled_q=nearest_q,
                    casimir_mismatch=abs(casimir-nearest_q*(nearest_q+2)/4),
                    strict_q_assignment=False,
                    cut_translation_weights=spatial["momentum_weights"][i]))
            rows[m] = dict(doubled_m=int(m), physical_density_weight=values["physical_density_weight"], levels=levels)
        output.append(dict(name=name, j2=0. if "ordered" in name else .125,
            source=str(path.relative_to(ROOT)), source_sha256=file_sha256(path),
            checkpoint_sha256=full["readout"]["checkpoint_sha256"], region_sites=list(range(18)),
            region="physical MPS prefix18; different from CT18 hierarchy supports",
            physical_purity=r["purity"], relative_rotation_covariance_squared=r["purity_normalized_commutator_squared"],
            strict_SU2_verified=r["strict_su2_verified"], measured_fixed_m_blocks=rows,
            warning="Entanglement levels, not global excitation energies. Fixed m does not equal fixed S. Approximate Casimir alone is not certified q assignment."))
    return output


def frozen_projector_audit() -> dict:
    """DMRG-only reference fixes a physical q=0 projector before ViT tests.

    All scalar means are linear before the explicitly labelled conditional
    ratio. Joint chain jackknife covariance retains numerator/denominator
    dependence. No ViT data choose the projector or its sign.
    """
    reference = RESULTS/"coupling_quality_review_20261012/latest_inputs/dmrg/work/chi10240_from8192_v1/rdms.npz"
    with np.load(reference) as f:
        rho_ref = f['chi10240_compact6']
    u, labels = regional_schur_columns(6)
    ids = [i for i,label in enumerate(labels) if label[:2]==(0,0)]
    u0 = u[:,ids]
    reduced = u0.conj().T @ rho_ref @ u0
    values, vectors = np.linalg.eigh((reduced+reduced.conj().T)/2)
    vector = u0 @ vectors[:,-1]
    target = np.outer(vector, vector.conj())
    p0 = u0 @ u0.conj().T
    raising = np.zeros_like(target)
    for packed in range(64):
        for site in range(6):
            mask = 1 << site
            if not packed & mask:
                raising[packed|mask, packed] += 1
    operators = np.stack([target, p0])
    frozen_path = OUT/'frozen_projectors.npz'
    np.savez(frozen_path, compact6_q0_reference_main_projector=target, compact6_q0_projector=p0,
             compact6_sites=np.asarray([12,13,14,18,19,20]), q0_reference_multiplicity_vector=vectors[:,-1])
    panels = [
        ("ViT", 0., RESULTS/"coupling_quality_review_20261012/latest_inputs/readouts/tri_j0_continue_step00750_regional/physics.npz", 'compact6__rho'),
        ("ViT", 0., RESULTS/"coupling_quality_review_20261012/latest_inputs/readouts/tri_j0_continue_step01000_regional/physics.npz", 'compact6__rho'),
        ("ViT", .125, RESULTS/"spin_spatial_vit_gpu_continuation_6600_v1/analysis/six_site_readout/physics.npz", 'compact6__rho'),
        ("ViT", .125, RESULTS/"spin_spatial_vit_gpu_continuation_7600_v1/analysis/witness_regions_v1/physics.npz", 'compact6__rho'),
        ("DMRG", .125, RESULTS/"spin_tensor_network_continuation_v2/chi4096/rdms.npz", 'chi4096_compact6'),
        ("DMRG", .125, RESULTS/"spin_tensor_network_continuation_v2/chi5120_probe/rdms.npz", 'chi5120_compact6'),
        ("DMRG", .125, reference, 'chi10240_compact6'),
    ]
    reports = []
    for method,j2,path,key in panels:
        with np.load(path) as f:
            means = np.einsum('kij,ji->k',operators,f[key]).real
            ratio = float(means[0]/means[1])
            covariance, se, ratio_se = None, None, None
            jackkey = key.replace('__rho','__jackknife_rhos')
            if '__rho' in key and jackkey in f.files:
                joint = np.einsum('kij,bji->bk',operators,f[jackkey]).real
                centered = joint-joint.mean(axis=0)
                covariance = (len(joint)-1)/len(joint)*(centered.T@centered)
                se = np.sqrt(np.diag(covariance))
                ratios = joint[:,0]/joint[:,1]
                ratio_se = float(np.sqrt((len(ratios)-1)/len(ratios)*np.sum((ratios-ratios.mean())**2)))
            reports.append(dict(method=method,j2=j2,source=str(path.relative_to(ROOT)),source_sha256=file_sha256(path),
                frozen_linear_target_weight=float(means[0]),q0_weight=float(means[1]),conditional_ratio=ratio,
                linear_chain_jackknife_se=se.tolist() if se is not None else None,
                paired_numerator_denominator_chain_jackknife_covariance=covariance.tolist() if covariance is not None else None,
                conditional_ratio_chain_jackknife_se=ratio_se,
                statistical_scope="paired within-chain numerator/denominator; different states are independent panels, not paired trajectories"))
    a,b = reports[1],reports[3]
    contrast = dict(label='J.125_step7600_minus_J0_step1000',
        linear_target_weight=b['frozen_linear_target_weight']-a['frozen_linear_target_weight'],
        linear_target_weight_independent_panel_se=float(np.hypot(a['linear_chain_jackknife_se'][0],b['linear_chain_jackknife_se'][0])),
        conditional_ratio=b['conditional_ratio']-a['conditional_ratio'],
        conditional_ratio_independent_panel_se=float(np.hypot(a['conditional_ratio_chain_jackknife_se'],b['conditional_ratio_chain_jackknife_se'])))
    return dict(operator_definition='Projector onto leading q=0 multiplicity eigenvector of DMRG chi10240 compact6; fixed before all ViT evaluations',
        reference=str(reference.relative_to(ROOT)),reference_sha256=file_sha256(reference),
        physical_sites=[12,13,14,18,19,20],reference_eigenvalue_gap=float(values[-1]-values[-2]),
        operator_artifact=str(frozen_path.relative_to(ROOT)),operator_artifact_sha256=file_sha256(frozen_path),
        idempotence_defect=float(np.linalg.norm(target@target-target)),raising_commutator_defect=float(np.linalg.norm(target@raising-raising@target)),
        panels=reports,contrast=contrast,
        warning='No spectral target chosen from ViT noise. Conditional ratio still has finite-sample ratio bias. This local spin-singlet projector is not a QSL-exclusive witness.')


def main() -> None:
    tick = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    base = RESULTS/"ct_status_20261028/verified_next_parents"
    ct = [raw_ct(base/name/"parent.pkl", arity, j2) for name,arity,j2 in
          [("ct6_j0", "ternary_joint", 0.), ("ct6_j0125", "binary_truncated", .125)]]
    atomic_json(OUT/"ct_raw_edge_spectra.json", dict(states=ct))
    physical = physical_assets()
    atomic_json(OUT/"physical_regional_readouts.json", dict(states=physical))
    ordered = next(s for s in physical if s['method']=='ViT' and s['region']=='compact6'
                   and 'step01000' in s['source'])
    frustrated = next(s for s in physical if s['method']=='ViT' and s['region']=='compact6' and s['j2']==.125)
    values = [s['readout']['spin_sector_weights']['0'] for s in (ordered, frustrated)]
    p_delta = values[1]['p']-values[0]['p']
    p_se = float(np.hypot(*(v['independent_chain_jackknife_se'] for v in values)))
    largest = [v['conditional_multiplicity_reduced_spectrum'][0] for v in values]
    largest_delta = largest[1]-largest[0]
    largest_se = float(np.hypot(*(v['conditional_spectrum_chain_jackknife_se'][0] for v in values)))
    comparisons = dict(region="compact6", sites=ordered['sites'], doubled_q=0,
        ordered_source=ordered['source'], frustrated_source=frustrated['source'],
        same_sites_verified=ordered['sites']==frustrated['sites'],
        delta_p0_frustrated_minus_ordered=p_delta, delta_p0_independent_panel_se=p_se,
        largest_conditional_multiplicity_eigenvalue=largest,
        delta_largest_frustrated_minus_ordered=largest_delta,
        delta_largest_independent_panel_se=largest_se,
        delta_largest_over_se=largest_delta/largest_se,
        interpretation="Historical per-state adaptive eigenvalue comparison only. Do not use this as a final significance claim; primary control is frozen_projector_audit.json including ViT7600 quality drift.",
        warnings=["Independent-panel SE assumes the stored coupling panels are independent.",
                  "No multiple-comparison correction; this is a candidate, not a preregistered final discovery.",
                  "Conditional and eigenvalue plugin bias is not removed; linear frozen-projector validation is a useful next test.",
                  "These are physical reduced multiplicity densities; strict SU2 physical entanglement spectra were withheld."])
    atomic_json(OUT/"physical_comparison.json", comparisons)
    frozen = frozen_projector_audit()
    atomic_json(OUT/"frozen_projector_audit.json", frozen)
    large = dmrg_large_cuts()
    atomic_json(OUT/"dmrg_physical_cut18_family_inventory.json", dict(states=large))
    summary = dict(status="bounded_local_readout_complete", raw_CT_state_count=len(ct),
        analysis_source_sha256=file_sha256(Path(__file__)),
        physical_RDM_count=len(physical), cached_DMRG_cut18_count=len(large),
        primary_frozen_projector_contrast=frozen['contrast'],
        primary_conclusion="A strict physical projector and cross-method interface are now available, but the latest fixed-projector coupling contrast is unresolved (about1.50SE conditional,1.08SE unconditioned). Per-state maximum-eigenvalue differences are not a substitute for this control.",
        no_new_training=True, no_new_sampling=True, no_remote_access=True,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
        warnings=["Raw CT spectra are not full coherent A1 spectra.",
                  "CT hierarchy18 and DMRG prefix18 are different physical cuts; no cross-method spectrum distance is computed.",
                  "No complete global S0/S1/S2/S3 excitation family was optimized in this readout.",
                  "Failure of strict SU2/positivity checks withholds strict physical entropy/spectra; legal projector means and independent-chain SE remain reported."],
        wall_seconds=perf_counter()-tick)
    atomic_json(OUT/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
