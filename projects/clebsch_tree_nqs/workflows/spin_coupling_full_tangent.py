"""Read-only complete shared tangents at original and product-SR endpoints.

Reuses sealed H0/H2 and the unchanged 4008-parameter VAP state definition.
No endpoint selection, update, new sample or Hamiltonian diagonalization.
The eigendecomposition below factors the SAVED projector, not Hamiltonian.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
from time import perf_counter
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.coupling_conditioned_spatial_vit import wrap_spatial_vit_parameters
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from su2_nqs.vmc.product_tangent_diagnostic import diagnose_product_tangent
from .spin_coupling_stationarity import _saved_operator_state
from .spin_overlap_attention_cost import _save_npz

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'projects/clebsch_tree_nqs/results/spin_coupling_transfer_v1'
SR_SOURCE = ROOT/'projects/clebsch_tree_nqs/results/spin_coupling_product_sr_gate_v4'
OUTPUT = ROOT/'projects/clebsch_tree_nqs/results/spin_coupling_full_tangent_v1'
PROTOCOL = dict(task='spin_coupling_full_tangent', all_original_and_SR_singlet_parents=True,
    all_couplings=[.2, .35, .5, .65, .8], training_indices=[0, 2, 4],
    relative_cutoffs=[1e-6, 1e-8, 1e-10], damping=1e-3,
    groups=['all', 'common_backbone', 'conditional_direction'], wall_seconds_cap=240,
    training_updates_committed=0, new_samples=0, new_ED_calls=0, new_H_constructions=0,
    interpretation='Posthoc complete local differential rank; not global capacity, held-response or variational-win evidence')


def _sealed_result(folder: Path) -> dict:
    result = json.loads((folder/'results.json').read_text())
    bare = dict(result); seal = bare.pop('result_sha256')
    if canonical_sha256(bare) != seal or result['status'] != 'complete':
        raise ValueError('invalid parent seal: '+str(folder))
    return result


def run() -> dict:
    source, sr = _sealed_result(SOURCE), _sealed_result(SR_SOURCE)
    for relative, digest in source['artifacts'].items():
        if file_sha256(SOURCE/relative) != digest:
            raise ValueError('source artifact changed: '+relative)
    inputs = {str((folder/file).relative_to(ROOT)): file_sha256(folder/file)
              for folder, files in ((SOURCE, ['results.json', 'operators.npz', 'states.npz']),
                                    (SR_SOURCE, ['results.json'])) for file in files}
    for parent in sr['config']['parents']:
        for folder, relative in ((SOURCE, parent+'__singlet_vap__shared'), (SR_SOURCE, parent)):
            for suffix in ('json', 'npz'):
                path = folder/relative/('parameters.'+suffix)
                inputs[str(path.relative_to(ROOT))] = file_sha256(path)
    registration = dict(protocol=PROTOCOL, inputs=inputs, runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    registration['registration_sha256'] = canonical_sha256(registration)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError('immutable tangent attempt exists')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT/'registration.json', registration)
    started = perf_counter(); records = {}; arrays = {}
    def timeout(*_):
        raise TimeoutError('fixed full-tangent diagnostic cap')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(PROTOCOL['wall_seconds_cap'])
    try:
        with np.load(SOURCE/'operators.npz', allow_pickle=False) as archive:
            operators = {k: archive[k].copy() for k in archive.files}
        cfg = FixedSzSpatialViTConfig(**source['model_config'])
        template = wrap_spatial_vit_parameters(init_fixed_sz_spatial_vit(jax.random.PRNGKey(0), cfg))
        flat, unravel = ravel_pytree(template)
        mark = jax.tree.map(jnp.zeros_like, template); mark['coupling_bias'] = jnp.ones(16)
        conditional, _ = ravel_pytree(mark); conditional = np.asarray(conditional, dtype=bool)
        gauge = jnp.asarray(conditional/np.sqrt(conditional.sum()))
        canonicalize = lambda x: x-gauge*jnp.vdot(gauge, x).real
        eigvals, eigvecs = np.linalg.eigh(operators['p0'])
        isometry = eigvecs[:, eigvals > .5]
        factor_error = float(np.max(abs(isometry@isometry.conj().T-operators['p0'])))
        if isometry.shape != (153, 31) or factor_error > 1e-10:
            raise ValueError('saved singlet range does not factor')
        old_state = _saved_operator_state(cfg, operators, unravel)
        state = jax.jit(lambda x, j: jnp.asarray(isometry).conj().T@old_state(canonicalize(x), j, True))
        def real_state(x, j):
            p = state(x, j)
            return jnp.concatenate((p.real, p.imag))
        jac = jax.jit(jax.jacrev(real_state))
        h0, h2 = (isometry.conj().T@operators[k]@isometry for k in ('h0', 'h2'))
        max_energy_replay = 0.; max_gradient_replay = 0.; max_fd = 0.; max_gauge = 0.
        for parent in sr['config']['parents']:
            for stage, folder, relative in (('original', SOURCE, parent+'__singlet_vap__shared'),
                                           ('product_sr', SR_SOURCE, parent)):
                params = load_parameter_artifact(folder/relative/'parameters.json', template)
                expected = (source['training'][relative]['final_parameters_sha256'] if stage == 'original'
                            else sr['records'][parent]['final_parameters_sha256'])
                if pytree_sha256(params) != expected:
                    raise ValueError('parameter endpoint changed')
                x, _ = ravel_pytree(params); x = canonicalize(x)
                key = parent+'__'+stage; ps = []; hs = []; ts = []; pointwise = {}
                for j in PROTOCOL['all_couplings']:
                    p = np.asarray(state(x, j)); h = (h0+j*h2)@p
                    real_jac = np.asarray(jac(x, j)); t = real_jac[:31]+1j*real_jac[31:]
                    pointwise[str(j)] = {}
                    for group, columns in (('all', np.ones(len(flat), bool)),
                                           ('common_backbone', ~conditional), ('conditional_direction', conditional)):
                        analysis = diagnose_product_tangent(p[None], h[None], t[:, columns][None], n_sites=16,
                            relative_cutoffs=tuple(PROTOCOL['relative_cutoffs']), damping=PROTOCOL['damping'])
                        pointwise[str(j)][group] = analysis.metrics
                    energy = float(np.vdot(p, h).real/16)
                    # Compare independent sealed SR audit values at BOTH stages.
                    audit = sr['records'][parent]['before' if stage == 'original' else 'after'][str(j)]
                    max_energy_replay = max(max_energy_replay, abs(energy-audit['energy_per_site']))
                    gradient = jax.grad(lambda v: jnp.vdot(state(v, j), jnp.asarray(h0+j*h2)@state(v, j)).real/16)(x)
                    residual = h-np.vdot(p, h).real*p
                    max_gradient_replay = max(max_gradient_replay, float(np.max(abs(np.asarray(gradient)-2*np.real(t.conj().T@residual)/16))))
                    max_gauge = max(max_gauge, float(np.max(abs(t@np.asarray(gauge)))))
                    ps.append(p); hs.append(h); ts.append(t)
                p, h, t = map(np.asarray, (ps, hs, ts))
                indices = PROTOCOL['training_indices']; joint = {}
                for group, columns in (('all', np.ones(len(flat), bool)),
                                       ('common_backbone', ~conditional), ('conditional_direction', conditional)):
                    analysis = diagnose_product_tangent(p[indices], h[indices], t[indices][:, :, columns], n_sites=16,
                        relative_cutoffs=tuple(PROTOCOL['relative_cutoffs']), damping=PROTOCOL['damping'])
                    joint[group] = analysis.metrics
                    if group == 'all':
                        arrays[key+'__design'] = analysis.design
                        arrays[key+'__target'] = analysis.target
                        arrays[key+'__projection'] = analysis.projected_target
                # A fixed deterministic central-difference check; no direction
                # selected from energy, no finite candidate committed.
                direction = np.sin(np.arange(len(flat))+1.); direction /= np.linalg.norm(direction)
                for k, j in enumerate(PROTOCOL['all_couplings']):
                    fd = (np.asarray(state(x+1e-5*direction, j))-np.asarray(state(x-1e-5*direction, j)))/2e-5
                    max_fd = max(max_fd, float(np.max(abs(fd-t[k]@direction))))
                arrays[key+'__states'] = p; arrays[key+'__actions'] = h
                arrays[key+'__jacobians'] = t
                records[key] = dict(parent=parent, stage=stage, parameter_sha256=pytree_sha256(params),
                    pointwise=pointwise, training_product=joint,
                    interpretation='Complete local tangent, threshold-sensitive; held rays are diagnostics, not training targets')
                atomic_json(OUTPUT/'partial_results.json', dict(records=records, seconds=perf_counter()-started))
                print(json.dumps(dict(endpoint=key, joint_rank=joint['all']['spectra']['column_equilibrated']['cutoffs']['1e-10']['rank'],
                    capture=joint['all']['spectra']['column_equilibrated']['cutoffs']['1e-10']['projected_residual_fraction'],
                    seconds=perf_counter()-started)), flush=True)
        _save_npz(OUTPUT/'tangents.npz', **arrays)
        result = dict(status='complete', protocol=PROTOCOL, records=records,
            registration_sha256=file_sha256(OUTPUT/'registration.json'),
            saved_projector_factor_error=factor_error, max_energy_replay_error=max_energy_replay,
            max_gradient_identity_error=max_gradient_replay, max_JVP_FD_error=max_fd,
            max_uniform_w_gauge_tangent=max_gauge, full_Jacobian_evaluations=20,
            training_updates_committed=0, new_samples=0, new_ED_calls=0, new_H_constructions=0,
            independent_evidence=False, variational_win_claimed=False, seconds=perf_counter()-started,
            artifacts={'tangents.npz': file_sha256(OUTPUT/'tangents.npz')})
        if max_energy_replay > 1e-11 or max_gradient_replay > 1e-9 or max_fd > 1e-7 or max_gauge > 1e-10:
            raise RuntimeError('full tangent validation failed')
        result['result_sha256'] = canonical_sha256(result); atomic_json(OUTPUT/'results.json', result)
        return result
    except Exception as error:
        atomic_json(OUTPUT/'failure.json', dict(error=repr(error), seconds=perf_counter()-started))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    run()
