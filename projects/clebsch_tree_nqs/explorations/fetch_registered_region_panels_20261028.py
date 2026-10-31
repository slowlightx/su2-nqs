"""Fetch already-existing registered RDM panels; never run a remote readout.

Only accepts exact source/destination/SHA entries of two frozen local
inventories (eight middle-coupling and three late-J.125 files). Existing
local files are never overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import tempfile

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256

ROOT=Path(__file__).resolve().parents[3]
INVENTORY=ROOT/'projects/clebsch_tree_nqs/results/su2_mechanism_local_20261028/coupling_series/missing_inputs.json'
LATE_INVENTORY=ROOT/'projects/clebsch_tree_nqs/results/su2_mechanism_local_20261028/pending_late_j125_panels.json'


def fetch(item: dict) -> dict:
    """Copy a fixed remote file to a new local path only after exact SHA replay."""
    source=item['original_registered_physics_path']
    target=(ROOT/item['local_destination']).resolve()
    allowed=(ROOT/'projects/clebsch_tree_nqs/results/coupling_quality_review_20261012/latest_inputs/readouts').resolve()
    if (not source.startswith(('/home/yx789/nqs-wavefunctions/readouts/',
                               '/share/kim/yx789/nqs-physics-readouts/'))
            or not target.is_relative_to(allowed)):
        raise ValueError('outside the registered read-only source/local readout scope')
    report=dict(source=source,destination=str(target.relative_to(ROOT)),expected_sha256=item['expected_sha256'])
    if target.exists():
        if file_sha256(target)!=item['expected_sha256']:
            raise ValueError('existing local file differs; do not overwrite '+str(target))
        return dict(report,status='already_exact',bytes=target.stat().st_size)
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='registered-fetch-',suffix='.npz',dir=target.parent)
    os.close(fd)
    tmp=Path(temporary)
    try:
        reply=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',
                              'unicorn:'+source,str(tmp)],capture_output=True,text=True,timeout=240)
        if reply.returncode:
            return dict(report,status='not_fetched',error=reply.stderr[-1800:])
        digest=file_sha256(tmp)
        if digest!=item['expected_sha256']:
            return dict(report,status='SHA_mismatch_not_installed',actual_sha256=digest)
        if target.exists():
            raise FileExistsError('destination appeared during fetch; preserve it')
        os.replace(tmp,target)
        return dict(report,status='fetched_exact',bytes=target.stat().st_size,actual_sha256=digest)
    finally:
        tmp.unlink(missing_ok=True)


def run(output: Path, inventory: Path = INVENTORY) -> dict:
    """Save a new local copy/identity receipt without modifying the inventory."""
    if output.exists():
        raise FileExistsError('new output receipt directory required')
    inventory=inventory.resolve()
    if inventory not in (INVENTORY.resolve(),LATE_INVENTORY.resolve()):
        raise ValueError('only the two explicit frozen local inventories are permitted')
    document=json.loads(inventory.read_text())
    items=document['items'] if inventory==LATE_INVENTORY.resolve() else document['existing_registered_files_needed_for_covariance']
    if len(items)!=(3 if inventory==LATE_INVENTORY.resolve() else 8):
        raise ValueError('the registered file inventory changed')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(fetch,items))
    report=dict(inventory_sha256=file_sha256(inventory),files=results,remote_mutations=False,
                new_H_queries=0,new_sampler_records=0,new_training=False)
    output.mkdir(parents=True)
    atomic_json(output/'receipt.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,default=INVENTORY)
    args=parser.parse_args()
    result=run(args.output,args.inventory)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    print(json.dumps([dict(source=x['source'],status=x['status'],bytes=x.get('bytes')) for x in result['files']]))
