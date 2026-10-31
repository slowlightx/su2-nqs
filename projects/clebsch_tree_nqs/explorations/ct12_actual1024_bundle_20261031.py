"""Pure plan construction for a root-submitted, code-only CT12 Git deployment.

No upload, Slurm action, checkpoint copy or source-tree mutation occurs here.
All numerical inputs use immutable absolute EXTERNAL artifact paths.
"""
from __future__ import annotations
import json
from pathlib import Path

from projects.clebsch_tree_nqs.workflows.ct12_actual1024_cached_gate_20261031 import (
    ROOT,TASK,ROLES,GEOMETRY_FACTORS,validate_plan)


def make_plan() -> dict:
    """Freeze scientific counts and existing global45 artifact identities."""
    old=json.loads((ROOT/'configs/remote/ct12_w28_gpu_preflight_20261030/plan.json').read_text())
    previous=Path('/home/yx789/nqs-ct12-w28-gpu-preflight-20261030')
    inputs={key:dict(path=str(previous/old['inputs'][name]['path']),sha256=old['inputs'][name]['sha256'])
            for key,name in (('parent','parent'),('expanded','expanded'),
                             ('parent_registration','parent_registration'),('diagnostic_rows','rows'))}
    inputs['saved_H_products']=dict(
        path='/home/yx789/nqs-wavefunctions/nqs/ct12_w28_gpu_preflight_20261030/run01/full_H_products.npz',
        # Previously independently downloaded/hashed GPU replay fixture. Pure
        # plan construction and unit tests must not open ignored result assets.
        sha256='b6aae674bc0b5c7fd524e1a866722d3682dd4905a76f82adad043e90536ad131')
    plan=dict(task=TASK,execution_authorized=True,j2=.125,parent_global_step=45,fit_records=1024,
        calibration_records=1024,calibration_split=512,bilateral_records=512,bilateral_groups=32,
        confirmation_records=1024,confirmation_groups=32,audit_records=4096,audit_groups=64,steps=8,
        damping=.5,learning_rate=.03,epsilon=.125,target_FS=.015,maximum_FS=.02,
        complete_A1_images=1728,physical_H_bonds=864,original_parameters=1495080,wide_parameters=20748160,
        original_model=dict(length=12,multiplicity_cap=20,maximum_q=10),
        wide_model=dict(length=12,multiplicity_cap=28,maximum_q=10),
        maximum_peak_device_bytes=70*1024**3,maximum_host_array_bytes=240*1024**3,requested_host_GiB=320,
        compare_wall_cap_seconds=21600,original_gate_wall_cap_seconds=28800,wide_gate_wall_cap_seconds=64800,
        original_pilot_wall_cap_seconds=64800,wide_pilot_wall_cap_seconds=172800,
        restore_reserve_seconds=1800,save_reserve_seconds=300,derivative_row_batch=2,cached_parameter_chunk=16384,
        H_chunk=8,projection_chunk=16,action_chunk=4,score_chunk=1,
        gate_geometry_factors=GEOMETRY_FACTORS.tolist(),minimum_guard_FS=.0075,
        pilot_geometry_factors=[1.,.5,.25,.125,0.],
        pilot_cache_policy='volatile_host_exact_cache_per_current_parent',
        automatically_submit=False,automatically_start_pilot=False,
        seeds={name:3930317001+i for i,name in enumerate(ROLES)},inputs=inputs,
        host_cache_note='exact1024x20748160 float64 cache158.296GiB; one persisted gate cache only; never /tmp',
        gate_scope='each arm own certificate; original may proceed if wide fails; no automatic pilot',
        coordinate_lr_note='.03 is legacy direction-cost metadata; NEW trust profile tangent.015 and held-out finiteFSmax.02, not unchanged historical tiny steps',
        runtime_python='/home/yx789/nqs-vit-quality/.conda-netket/bin/python',
        deployment='full pinned code-only Git commit; inputs/output/J/cache absolute outside readonlyrelease',
        authorization_note='User explicitly requests continuing CT12 optimization with actual1024 direction validation; root alone submits bounded jobs and releases at most one conditional pilot.')
    validate_plan(plan,executing=True);return plan


if __name__=='__main__':print(json.dumps(make_plan(),indent=2))
