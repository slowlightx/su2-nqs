"""One-time capture of the two already-authorized midpoint audit parents."""
import importlib.util
import json
from pathlib import Path
import statistics
import sys

SOURCE = Path('/home/yx789/nqs-ct6-wide-production-v2-20261023')
WORK = Path('/home/yx789/nqs-ct6-midpoint-audit-20261023')
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('midpoint_driver', WORK / 'audit_driver.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.jax.config.update('jax_enable_x64', True)

folder = Path('projects/clebsch_tree_nqs/configs/remote/ct6_wide_production_20261023')
manifest = folder / 'source_manifest.json'
campaign = json.loads((SOURCE / folder / 'campaign.json').read_text())
for path, digest in json.loads((SOURCE / manifest).read_text()).items():
    if audit.file_sha256(SOURCE / path) != digest:
        raise ValueError('sealed source changed before capture: ' + path)

used_seeds = {2610218112, 2610218122}
def collect_seeds(value, key=''):
    if isinstance(value, dict):
        for name, child in value.items(): collect_seeds(child, key + '/' + name)
    elif isinstance(value, list):
        for child in value: collect_seeds(child, key)
    elif isinstance(value, int) and 'seed' in key.lower(): used_seeds.add(value)
for path in (SOURCE / 'projects/clebsch_tree_nqs/configs').rglob('*.json'):
    collect_seeds(json.loads(path.read_text()))

receipt = dict(status='snapshots_frozen_not_submitted',
    source_root=str(SOURCE), audit_workspace=str(WORK),
    wrapper_sha256=audit.file_sha256(WORK / 'audit_driver.py'),
    source_manifest_sha256=audit.file_sha256(SOURCE / manifest),
    new_requested_A100_hours=3., aggregate_reserved_A100_hours=campaign['aggregate_reserved_A100_hours'] + 3,
    CT6_total_budget_A100_hours=240, maximum_simultaneous_extra_audit_GPUs=1,
    temporary_capacity_exception='Two original training GPUs unchanged; at most one independent audit GPU',
    deadline_utc=campaign['deadline_utc'], cases={})
for tag, seed in [('j0125', 2610239201), ('j0', 2610239101)]:
    source_path = SOURCE / folder / (tag + '_w28.json')
    plan = json.loads(source_path.read_text())
    live = Path(plan['output'])
    case = WORK / tag
    case.mkdir()
    registration = json.loads((live / 'registration.json').read_text())
    if registration['plan_sha256'] != audit.canonical_sha256(plan):
        raise ValueError('source checkpoint registration and plan disagree')
    snapshot = audit.capture_checkpoint(live / 'checkpoint.pkl', case / 'checkpoint.pkl', registration)
    (case / 'source_registration.json').write_text(json.dumps(registration, indent=2) + '\n')
    (case / 'source_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    state, meta = audit.shared.restore_state(case / 'checkpoint.pkl', registration['sha256'])
    recent = meta['history'][-16:]
    h_mean = statistics.mean(row['H_seconds'] for row in recent)
    sample_mean = statistics.mean(row['sampling_seconds'] for row in recent)
    # Existing training sampling includes 149760 train +27648 calibration proposals.
    forecast = 4 * h_mean + (958464 / 177408) * sample_mean
    frozen = dict(task='ct6_immutable_midpoint_audit_v1', case=tag,
        source_root=str(SOURCE), source_manifest_path=str(manifest),
        source_manifest_sha256=receipt['source_manifest_sha256'],
        source_plan_path=str(case / 'source_plan.json'), source_plan_sha256=audit.canonical_sha256(plan),
        wrapper_sha256=receipt['wrapper_sha256'], snapshot=snapshot,
        output=str(case / 'audit'), audit=dict(audit.AUDIT_SPEC), seed=seed,
        forbidden_seeds=sorted(used_seeds), maximum_seconds=5400, requested_A100_hours=1.5,
        campaign_total_A100_hours=240, aggregate_reserved_A100_hours=receipt['aggregate_reserved_A100_hours'],
        campaign_deadline_utc=campaign['deadline_utc'], cost_forecast_seconds=forecast,
        cost_forecast=dict(source='last16 saved full updates from this exact width28 state lineage',
            full_H_seconds_per4096=h_mean, combined_sampling_seconds_per177408_proposals=sample_mean,
            midpoint_H_centers=16384, midpoint_proposal_centers=958464,
            additional_compile_and_save_margin_seconds=690,
            measured_full_audit_runtime=False),
        interpretation='Time-captured midpoint, not an endpoint selected by energy. Original128-step training and final65536 audit remain unchanged. Fresh uniform fixed-Sz chains, full coherent432A1 and all physical H bonds, float64. No clipping, optimizer update, or replacement of any old audit.')
    audit.validate_plan(frozen, plan)
    audit.verify_sources(frozen)
    audit.atomic_json(case / 'plan.json', frozen)
    audit.atomic_json(case / 'snapshot.json', snapshot)
    for name in ('source_registration.json', 'source_plan.json', 'plan.json', 'snapshot.json'):
        (case / name).chmod(0o444)
    receipt['cases'][tag] = dict(plan_path=str(case / 'plan.json'), plan_sha256=audit.canonical_sha256(frozen),
        snapshot=snapshot, cost_forecast_seconds=forecast, seed=seed)
audit.atomic_json(WORK / 'capture_receipt.json', receipt)
print(json.dumps(receipt, indent=2), flush=True)
