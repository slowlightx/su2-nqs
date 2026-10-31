"""Predeclared large-L physical-response residual supplement; no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from fnqs.artifacts import load_checkpoint,write_json
from fnqs.diagnostics.response import bootstrap_response
from fnqs.pilots.measurements import sampled_records
from fnqs.pilots.engine import draw
from fnqs.pilots.lineage import sha,read_node
from fnqs.variants.size_features import SizeFeatureConfig
from fnqs.scripts.run_frozen_confirmation import verified


def prepare(output: Path) -> None:
    """Bind residual diagnostics before the parent exposes confirmation labels."""
    p=verified(output)
    if (output/'endpoint_seal.json').exists():raise ValueError('supplement must be declared before endpoint completion')
    dest=output/'response_supplement/protocol.json'
    if dest.exists():raise FileExistsError('already declared')
    write_json(dest,dict(parent_protocol_sha256=sha(output/'protocol.json'),source_sha256=sha(Path(__file__)),
        sizes=[24,32],seeds=p['seeds'],field_indices=[0,19,39],samples=8192,chunk=128,
        bootstrap=200,seed_namespace=0xAE5018,kind='physical_response_diagnostic_only',
        relative_residual_is_not_chi_error_bound=True,new_training=False,
        expected_cost=dict(born_samples=147456,hamiltonian_actions=36,source_actions=18),
        reference_labels_used=False))


def run(output: Path) -> None:
    """Compute Q[(H-E)tau+Vpsi], HF defect and both physical FD checks."""
    p=verified(output);dest=output/'response_supplement';a=json.loads((dest/'protocol.json').read_text())
    if a['parent_protocol_sha256']!=sha(output/'protocol.json') or a['source_sha256']!=sha(Path(__file__)):
        raise ValueError('supplement declaration changed')
    seal=json.loads((output/'endpoint_seal.json').read_text())['nodes'];cfg=SizeFeatureConfig(**p['model']);rows=[]
    for seed in a['seeds']:
        for n in a['sizes']:
            folder=output/'runs'/f'L{n}_seed_{seed}';node=folder/'fast/node.json';read_node(node)
            if seal[folder.name]!=sha(node):raise ValueError('selected endpoint changed')
            params,_,_,_=load_checkpoint(folder/'fast/checkpoint.npz')
            for index in a['field_indices']:
                archive=dest/f'L{n}_seed_{seed}_point_{index}.npz';receipt=archive.with_suffix('.json')
                if receipt.exists():
                    result=json.loads(receipt.read_text())
                    if sha(archive)!=result['archive_sha256']:raise ValueError('supplement archive changed')
                else:
                    g=p['reserved_points'][index]
                    key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed),a['seed_namespace']),n),index)
                    bits=draw(params,key,a['samples'],n,jnp.asarray(g),cfg)
                    blocks=[sampled_records(params,bits[i:i+a['chunk']],jnp.asarray(g),cfg) for i in range(0,a['samples'],a['chunk'])]
                    raw={k:np.concatenate([np.asarray(b[k]) for b in blocks]) for k in blocks[0]}
                    result=bootstrap_response(raw,seed+n+index,a['bootstrap'])
                    if max(result['fd_score_relative_rms_error'])>5e-4:raise RuntimeError('supplement FD guard failed')
                    np.savez_compressed(archive,bits=np.asarray(bits),couplings=g,**raw)
                    result.update(n=n,seed=seed,couplings=g,archive_sha256=sha(archive),checkpoint_sha256=sha(folder/'fast/checkpoint.npz'))
                    write_json(receipt,result)
                rows.append(result);print('response',n,seed,index,result['response_relative_residual'],flush=True)
    write_json(dest/'review.json',dict(rows=rows,audit_cost=a['expected_cost'],training_cost=0,
        supplement_protocol_sha256=sha(dest/'protocol.json'),diagnostic_not_accuracy_certificate=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run']);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();(prepare if args.command=='prepare' else run)(args.output)
