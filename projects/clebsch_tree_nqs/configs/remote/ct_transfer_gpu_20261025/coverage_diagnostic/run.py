"""One preregistered H-coverage diagnosis, never a quality-gate release."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_audit import run_h_bridge_endpoint_audit

CONFIG = Path(__file__).resolve().relative_to(Path.cwd().resolve()).parent
BASE = Path('projects/clebsch_tree_nqs/results/ct_transfer_gpu_audit_20261025')
OUTPUT = BASE/'coverage_diagnostic'


def prepare():
    if (CONFIG/'plan.json').exists():
        raise FileExistsError('Immutable diagnostic already registered')
    source = CONFIG.parent/'chunk64/plan.json'
    spec = json.loads(source.read_text())
    spec.update(task='separate_fixed_H_coverage_diagnostic_after_failed_support',
        audit_purpose='coverage_diagnostic', separately_registered_diagnostic=True,
        no_benchmark=True, no_attention=True, chains=8, records=32,
        burn_proposals=128, proposals_between=128, seconds_cap=3300.,
        seed=2610258300,
        fresh_audit_rng_protocol='Previously registered, unused audit seed2610258300; no reused support RNG',
        no_energy_or_variance_estimate=False, no_automatic_H_release=True)
    spec.pop('initial_rng_key', None)
    for key in ('support_samples', 'support_results', 'support_registration'):
        filename = {'support_samples': 'support_prefix.npz', 'support_results': 'results.json',
                    'support_registration': 'registration.json'}[key]
        path = BASE/'chunk64/support10240'/filename
        spec[key+'_path'] = str(path)
        spec[key+'_sha256'] = file_sha256(path)
    atomic_json(CONFIG/'plan.json', spec)
    atomic_json(CONFIG/'registration.json', dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Measure whether full energy/second-moment integrands share the observed slow K mode',
        original_support_gate_passed=False, original_support_logQ_split_rhat=1.1036581468763,
        observed_K_split_rhat=1.19155, observed_K_half_drift=-.31607,
        observed_K_half_drift_chain_se=.12423,
        no_threshold_change=True, all_eight_terminal_chains=True,
        records_per_chain=32, burn_proposals=128, proposals_between=128,
        fresh_seed=2610258300, maximum_full_H_centers=516,
        fixed_endpoints=True, no_energy_selection=True,
        no_attention_or_benchmark_release_regardless_of_result=True,
        finite_burn_ratio_diagnostic_not_variational_bound=True,
        jackknife_cannot_include_unknown_equilibration_bias=True,
        report='All chains and first/last16 records: norm, E, H2, residual variance, paired differences; bridge mass balance, exact nodes, tails and base/pushed K correlations',
        prior_actual_GPU_seconds=2123, requested_GPU_seconds=3600,
        cumulative_actual_plus_reservation_GPU_hours=5723/3600,
        original_total_actual_GPU_hours_cap=3., no_adaptive_extension=True))


def seal():
    spec = json.loads((CONFIG/'plan.json').read_text())
    manifest = transitive_local_source_manifest([Path(__file__)], root=Path.cwd(),
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    extras = [CONFIG/'plan.json', CONFIG/'registration.json', CONFIG/'run.sbatch', Path('AGENTS.md')]
    extras += [Path(spec[k]) for k in ('control_checkpoint', 'candidate_checkpoint',
               'support_samples_path', 'support_results_path', 'support_registration_path')]
    for path in extras:
        manifest[str(path)] = file_sha256(path)
    atomic_json(CONFIG/'manifest.json', manifest)
    (CONFIG/'sync_files.txt').write_text('\n'.join(sorted(manifest))+'\n'+str(CONFIG/'manifest.json')+'\n')
    print(json.dumps(dict(files=len(manifest), bytes=sum(Path(p).stat().st_size for p in manifest))))


def execute():
    for path, digest in json.loads((CONFIG/'manifest.json').read_text()).items():
        if file_sha256(Path(path)) != digest:
            raise ArithmeticError('Frozen input/source changed: '+path)
    result = run_h_bridge_endpoint_audit(json.loads((CONFIG/'plan.json').read_text()), OUTPUT/'H_products')
    print(json.dumps({k:v for k,v in result.items() if k != 'timings'}), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'seal': seal, 'execute': execute}[sys.argv[1]]()
