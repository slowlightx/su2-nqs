"""Bounded single-J exact-small SU2 tensor decoder + convolution/attention.

One candidate, two preset parents. Fully train all neural weights with an
additive strong saved singlet carrier. Continue the original VAP as a strong
optimization control; unequal parameter counts/cost preclude a fairness or
SOTA claim. This complete-basis oracle is deliberately barred from N36.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import signal
from time import perf_counter
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.schur_tensor_decoder import four_cell_tensor_decoder_rows
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact, load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.coupling_conditioned_spatial_vit import wrap_spatial_vit_parameters
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from su2_nqs.models.reduced_conv_attention import ReducedConvAttentionConfig, init_reduced_conv_attention, reduced_conv_attention_coefficients
from su2_nqs.su2.local_schur_fusion import four_cell_schur_states
from su2_nqs.vmc.exact_state_sr import train_product_state_sr_trust_region
from .spin_coupling_full_tangent import _sealed_result, SOURCE, ROOT
from .spin_coupling_stationarity import _saved_operator_state
from .spin_overlap_attention_cost import _save_npz

jax.config.update('jax_enable_x64', True)
OUTPUT = ROOT/'projects/clebsch_tree_nqs/results/spin_reduced_conv_attention_v2'
PROTOCOL = dict(task='spin_reduced_conv_attention_single_J', geometry='square_4x4_PBC', J2_over_J1=.5,
    parents=['square_d16_l2_s2026092901', 'square_d16_l2_s2026092902'],
    neural_seeds=[2026100101, 2026100102], arms=['reduced_CNN_attention', 'continued_original_VAP'],
    architecture='complete configuration-dependent reduced coefficients, fixed local Schur+CG decoder; full A1 linear orbit map',
    steps=32, initial_learning_rate=1., initial_damping=1e-2, maximum_mean_fs_infidelity=1e-4,
    maximum_backtracks=14, linear_solver='dual_dense', maximum_linear_relative_residual=1e-8,
    wall_seconds_cap=300, new_ED_calls=0, new_samples=0,
    scope='N16 exact-small amplitude/energy test only; no new small head, no foundation/response/size claim',
    scalable_contraction_implemented=False, n36_production_allowed=False,
    fairness='same initial physical states, steps and physical trust cap; parameter counts, H calls and CPU times must be reported, not declared equal',
    technical_revision='v1 control dual-SR failed relative residual 1.09157e-8 > 1e-8; retain failed run, increase common damping from .001 to .01, do not relax residual tolerance or select v1 neural result')


def _metrics(p: np.ndarray, h: np.ndarray) -> dict:
    p = np.asarray(p); p = p/np.linalg.norm(p)
    hp = h@p; energy = float(np.vdot(p, hp).real)
    return dict(energy_per_site=energy/16, variance_per_site=float(np.linalg.norm(hp-energy*p)**2/16))


def run() -> dict:
    source = _sealed_result(SOURCE)
    for path, sha in source['artifacts'].items():
        if file_sha256(SOURCE/path) != sha:
            raise ValueError('sealed source artifact changed')
    cfg = ReducedConvAttentionConfig()
    inputs = {str((SOURCE/name).relative_to(ROOT)): file_sha256(SOURCE/name)
              for name in ('results.json', 'operators.npz', 'states.npz')}
    registration = dict(protocol=PROTOCOL, model_config=asdict(cfg), inputs=inputs,
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    registration['registration_sha256'] = canonical_sha256(registration)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError('immutable single-coupling pilot exists')
    OUTPUT.mkdir(parents=True, exist_ok=True); atomic_json(OUTPUT/'registration.json', registration)
    started = perf_counter(); records = {}; arrays = {}
    def timeout(*_):
        raise TimeoutError('fixed single-coupling prototype cap; no extension')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(PROTOCOL['wall_seconds_cap'])
    try:
        with np.load(SOURCE/'operators.npz', allow_pickle=False) as archive:
            op = {k: archive[k].copy() for k in archive.files}
        with np.load(SOURCE/'states.npz', allow_pickle=False) as archive:
            physical = {k: archive[k].copy() for k in archive.files}
        cells = ((0, 1, 4, 5), (2, 3, 6, 7), (8, 9, 12, 13), (10, 11, 14, 15))
        states = four_cell_schur_states('square_plaquette')
        labels = jnp.asarray(np.stack(([s.cell_q for s in states], [s.cell_a for s in states]), axis=-1))
        internal = jnp.asarray([s.pair_q for s in states])
        decode = np.zeros((153, 1430))
        for first in range(0, len(op['basis']), 256):
            last = min(first+256, len(op['basis']))
            rows = four_cell_tensor_decoder_rows(op['basis'][first:last], 'square_plaquette', cells)
            ids = op['orbit_id'][first:last]
            np.add.at(decode, ids, rows/np.sqrt(op['orbit_sizes'][ids])[:, None])
        errors = dict(decoder_range=float(np.max(abs(decode@decode.T-op['p0']))),
                      decoder_spin=float(np.max(abs(op['s2']@decode))))
        if errors['decoder_range'] > 1e-10 or errors['decoder_spin'] > 1e-10:
            raise ValueError('fixed Schur/CG decoder range check failed')
        vals, vecs = np.linalg.eigh(op['p0'])  # Projector only, NOT H.
        iso = vecs[:, vals > .5]; d = jnp.asarray(iso.conj().T@decode)
        h = iso.conj().T@(op['h0']+.5*op['h2'])@iso
        arrays['decoder_A1'] = decode; arrays['singlet_isometry'] = iso; arrays['hamiltonian'] = h
        oldcfg = FixedSzSpatialViTConfig(**source['model_config'])
        oldtemplate = wrap_spatial_vit_parameters(init_fixed_sz_spatial_vit(jax.random.PRNGKey(0), oldcfg))
        _, oldunravel = ravel_pytree(oldtemplate)
        oldfn = _saved_operator_state(oldcfg, op, oldunravel)
        for parent, seed in zip(PROTOCOL['parents'], PROTOCOL['neural_seeds'], strict=True):
            run_name = parent+'__singlet_vap__point_0.50'
            v = physical[run_name+'__0.5']; carrier = np.zeros(153, complex)
            np.add.at(carrier, op['orbit_id'], v/np.sqrt(op['orbit_sizes'][op['orbit_id']]))
            carrier = iso.conj().T@carrier; carrier /= np.linalg.norm(carrier)
            params = init_reduced_conv_attention(jax.random.PRNGKey(seed), cfg)
            coefficients = lambda p: reduced_conv_attention_coefficients(p, labels, internal, cfg)
            reference = jax.lax.stop_gradient(coefficients(params))
            scale = 1/float(jnp.linalg.norm(d@reference))
            def neural_state(p):
                value = jnp.asarray(carrier)+scale*d@(coefficients(p)-reference)
                return value/jnp.linalg.norm(value)
            oldparams = load_parameter_artifact(SOURCE/run_name/'parameters.json', oldtemplate)
            def old_state(p):
                x, _ = ravel_pytree(p)
                return jnp.asarray(iso).conj().T@oldfn(x, .5, True)
            for arm, initial, fn in (('reduced_CNN_attention', params, neural_state),
                                     ('continued_original_VAP', oldparams, old_state)):
                path = OUTPUT/parent/arm; path.mkdir(parents=True)
                identity = dict(task=PROTOCOL['task'], parent=parent, arm=arm,
                                registration_sha256=registration['registration_sha256'])
                save_parameter_artifact(path/'initial.json', initial, identity=identity|{'stage':'initial'})
                p0 = np.asarray(fn(initial)); initial_error = float(np.max(abs(p0-carrier)))
                if initial_error > 1e-10:
                    raise ValueError('strong-carrier initial physical replay failed')
                tick = perf_counter()
                trained, training = train_product_state_sr_trust_region(initial,
                    states_fn=lambda p: fn(p)[None], hamiltonian_action=lambda v: v@jnp.asarray(h).T,
                    **{k:PROTOCOL[k] for k in ('steps', 'initial_learning_rate', 'initial_damping',
                        'maximum_mean_fs_infidelity', 'maximum_backtracks', 'linear_solver', 'maximum_linear_relative_residual')})
                seconds = perf_counter()-tick; final = np.asarray(fn(trained))
                save_parameter_artifact(path/'parameters.json', trained, identity=identity|{'stage':'final'})
                # Sealed final THEN exact physical energy/variance and symmetry audit.
                expanded = iso@final
                record = dict(parent=parent, arm=arm, seed=seed if arm=='reduced_CNN_attention' else None,
                    parameter_count=sum(np.asarray(v).size for v in jax.tree.leaves(initial)),
                    initial=_metrics(p0, h), final=_metrics(final, h), training=training,
                    carrier_replay_error=initial_error, raw_singlet_error=float(np.linalg.norm(op['s2']@expanded)),
                    A1_by_construction=True, seconds=seconds,
                    nonlinearities_only_on_reduced_labels=arm=='reduced_CNN_attention')
                atomic_json(path/'results.json', record); records[parent+'__'+arm] = record
                arrays[parent+'__'+arm+'__initial'] = p0; arrays[parent+'__'+arm+'__final'] = final
                print(json.dumps(dict(parent=parent, arm=arm, final=record['final'], seconds=seconds)), flush=True)
        _save_npz(OUTPUT/'states.npz', **arrays)
        result = dict(status='complete', protocol=PROTOCOL, model_config=asdict(cfg), records=records,
            exact_decoder_errors=errors, seconds=perf_counter()-started,
            registration_sha256=file_sha256(OUTPUT/'registration.json'),
            artifacts={'states.npz':file_sha256(OUTPUT/'states.npz')},
            new_ED_calls=0, new_samples=0, n36_production_allowed=False,
            variational_win_claimed=False, independent_evidence=False,
            interpretation='Single-J complete-basis prototype and strong continued optimization control; no scalable contraction or statistically independent architecture win')
        result['result_sha256'] = canonical_sha256(result); atomic_json(OUTPUT/'results.json', result)
        return result
    except Exception as error:
        atomic_json(OUTPUT/'failure.json', dict(error=repr(error), seconds=perf_counter()-started))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    run()
