"""Read-only actual CT12 active-state/tangent/recovery check; no training/H."""
from pathlib import Path
from time import perf_counter
import argparse

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_active_train import lift_active_train_bank, canonical_active_sampling_bank
from su2_nqs.models.triangular_clebsch_active_layout import repack_active_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank, make_feature_lut_amplitude
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint


def run(checkpoint: Path, fixture: Path, output: Path) -> dict:
    if output.exists():raise FileExistsError('immutable preparation report required')
    start=perf_counter(); saved=load_checkpoint(checkpoint)
    cfg=MixedClebschConfig(12,20,10)
    p=saved['metadata']['embedding']['active_top_widths']
    profile=tuple({int(q):int(d) for q,d in p[i if i in p else str(i)].items()} for i in range(15))
    active=repack_active_clebsch_bank(saved['params'],cfg,profile)
    rng=np.random.default_rng(2610279991)
    delta=jax.tree.map(lambda x:jnp.asarray(rng.normal(size=x.shape))*.01,active)
    row=jnp.asarray(np.load(fixture)['raw_rows'][0])
    raw=make_feature_lut_amplitude(cfg,active_top_widths=profile)
    native=make_feature_lut_amplitude(cfg)
    fa=lambda bank:raw(prepare_feature_lut_bank(bank,cfg),row)
    fn=lambda bank:native(prepare_feature_lut_bank(lift_active_train_bank(bank,cfg,profile),cfg),row)
    value,tangent=jax.block_until_ready(jax.jvp(fa,(active,),(delta,)))
    other,other_tangent=jax.block_until_ready(jax.jvp(fn,(active,),(delta,)))
    amplitude_error=float(abs(value-other)/max(abs(float(value)),1e-300))
    tangent_error=float(abs(tangent-other_tangent)/max(abs(float(tangent)),1e-300))
    canonical,notes=canonical_active_sampling_bank(active,cfg,profile)
    canonical_error=float(abs(fa(canonical)*np.sqrt(notes['original_raw_norm'])-value)/max(abs(float(value)),1e-300))
    if max(amplitude_error,tangent_error,canonical_error)>2e-11 or float(value)==0 or float(tangent)==0:
        raise ArithmeticError('actual active/native amplitude/tangent/canonical replay failed')
    result=dict(status='passed_actual_CT12_no_training_or_H',checkpoint_sha256=file_sha256(checkpoint),
        fixture_sha256=file_sha256(fixture),fixture_use='execution identity only; not a new physical audit',
        active_real_parameters=sum(np.size(x) for x in jax.tree.leaves(active)),
        native_padded_real_parameters=sum(np.size(x) for x in jax.tree.leaves(saved['params'])),
        active_profile=profile,amplitude_relative_error=amplitude_error,
        tangent_relative_error=tangent_error,canonical_scale_relative_error=canonical_error,
        amplitude=float(value),tangent=float(tangent),canonical_notes=notes,
        full_H_queries=0,training_updates=0,raw_amplitude_primal_or_tangent_replays=3,
        wall_seconds=perf_counter()-start)
    atomic_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(run(a.checkpoint,a.fixture,a.output))
