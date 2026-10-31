"""Seed-isolated scheduling of the already frozen, unchanged numeric recipe.

Each worker records its own launcher and protocol before initialization, uses
the original three-size field schedule, and writes disjoint canonical seed
folders. The original sequential runner's partial-run guard prevents a second
optimizer from entering an active folder; after workers finish it can seal the
original nine-endpoint campaign by validating/skipping accepted nodes.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
from fnqs.artifacts import write_json
from fnqs.pilots.lineage import ROOT,sha
from fnqs.scripts.run_frozen_confirmation import verified,train_all


def execute(output: Path,seed: int) -> None:
    """Record scheduling provenance; preserve every numeric protocol value."""
    original=verified(output)
    if seed not in original['seeds']:raise ValueError('seed must be predeclared')
    if any((output/'runs'/f'L{n}_seed_{seed}').exists() for n in original['sizes']):
        raise ValueError('worker must start before this seed has any trajectory')
    child=output/f'parallel_seed_{seed}'
    if child.exists():raise FileExistsError('fresh worker declaration required')
    child.mkdir();p=copy.deepcopy(original)
    p['seeds']=[seed];p['schedule_sha256']={str(seed):original['schedule_sha256'][str(seed)]}
    launcher='fnqs/scripts/parallel_frozen_seed.py'
    p['source_sha256'][launcher]=sha(ROOT/launcher)
    p['parallel_schedule']=dict(parent_protocol_sha256=sha(output/'protocol.json'),
        numeric_recipe_unchanged=True,canonical_seed_output=str((output/'runs').resolve()),
        all_nine_endpoints_required_before_any_held_reference=True,worker_opens_references=False)
    # Check that scheduling is the entire change, including the base schedule.
    check=copy.deepcopy(p);check.pop('parallel_schedule');check['seeds']=original['seeds']
    check['schedule_sha256']=original['schedule_sha256'];check['source_sha256'].pop(launcher)
    if check!=original:raise ValueError('numeric recipe changed in parallel scheduling')
    for name in p['source_sha256']:
        target=child/'runtime'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(output/f'schedule_{seed}.npy',child/f'schedule_{seed}.npy')
    (child/'runs').symlink_to((output/'runs').resolve(),target_is_directory=True)
    write_json(child/'protocol.json',p)
    train_all(child)
    write_json(child/'completion.json',dict(parent_protocol_sha256=sha(output/'protocol.json'),
        protocol_sha256=sha(child/'protocol.json'),numeric_recipe_unchanged=True,
        all_sizes_complete=True,held_references_opened=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int);args=parser.parse_args();execute(args.output,args.seed)
