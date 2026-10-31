"""No-fit cache audit of the frozen six-body scalar and CT physical readout."""
from pathlib import Path
import json
from time import perf_counter
import numpy as np

from su2_nqs.benchmarks.su2_maximum_entropy import invariant_constraint_space
from su2_nqs.benchmarks.su2_six_body_chirality import six_body_chirality_operators
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT/'projects/clebsch_tree_nqs/results'
OUT = RESULTS/'su2_mechanism_local_20261028/regions'


def main():
    start = perf_counter()
    base = RESULTS/'spin_spatial_vit_gpu_continuation_6600_v1/analysis/invariant_maximum_entropy_v1'
    done = json.loads((base/'completion.json').read_text())
    for name in ('results', 'registration', 'readouts'):
        suffix = '.npz' if name=='readouts' else '.json'
        assert file_sha256(base/(name+suffix)) == done[name+'_sha256']
    registration = json.loads((base/'registration.json').read_text())
    space = invariant_constraint_space(6)
    assert registration['physical_problem']['regions']['compact6']==[12,13,14,18,19,20]
    assert registration['constraint_space']==space.diagnostics
    bank = six_body_chirality_operators()
    reduced = {}
    recon_errors = {}
    for key, op in bank.items():
        reduced[key] = np.zeros((20,20),complex)
        reconstructed = np.zeros((64,64),complex)
        for sl, us in zip(space.slices, space.magnetic_columns):
            block = us[-1].conj().T @ op @ us[-1]
            reduced[key][sl,sl] = block
            for u in us:
                reconstructed += u @ block @ u.conj().T
        recon_errors[key] = float(np.linalg.norm(op-reconstructed))
    comparison = json.loads((RESULTS/'spin_spatial_vit_gpu_continuation_6600_v1/analysis/six_site_quality_comparison/results.json').read_text())['registration']['config']
    dmrg_dir = ROOT/comparison['reference_protocol']
    dmrg_key = '__'.join((comparison['reference_problem'], comparison['reference_family'], 'chi4096_sweep0024', 'compact6'))+'_raw_rho'
    panels = [
        ('ViT__5600__compact6',RESULTS/'spin_spatial_vit_gpu_recovery_v6/analysis/six_site_readout/physics.npz','compact6__rho'),
        ('ViT__6600__compact6',RESULTS/'spin_spatial_vit_gpu_continuation_6600_v1/analysis/six_site_readout/physics.npz','compact6__rho'),
        ('DMRG__chi4096_sweep0024__compact6',dmrg_dir/'readouts.npz',dmrg_key),
    ]
    records = []
    with np.load(base/'readouts.npz') as cache:
        for prefix,path,key in panels:
            assert registration['inputs'][str(path.relative_to(ROOT))] == file_sha256(path)
            with np.load(path) as f:
                rho = f[key]
            omega = cache[prefix+'__omega']; model = cache[prefix+'__maxent']
            identity = float(np.max(abs(space.data(rho)-omega)))
            assert identity < 2e-12
            row = dict(prefix=prefix, physical_source=str(path.relative_to(ROOT)),
                physical_source_sha256=file_sha256(path), cached_omega_replay_max_error=identity,
                model_minimum_eigenvalue=float(np.linalg.eigvalsh(model).min()), observables={})
            for name,op in reduced.items():
                actual = float(np.trace(omega@op).real)
                physical = float(np.trace(rho@bank[name]).real)
                me = float(np.trace(model@op).real)
                row['observables'][name] = dict(actual=actual, maximum_entropy=me,
                    actual_minus_maximum_entropy=actual-me,
                    invariant_reduction_error=abs(actual-physical),
                    residual_SE=None, residual_SE_reason='cached ME delete-chain matrices are not stored; no refits were authorized')
            records.append(row)
    maxent = dict(task='cached_maximum_entropy_six_body_redteam',
        source_readouts=str((base/'readouts.npz').relative_to(ROOT)), source_sha256=file_sha256(base/'readouts.npz'),
        constraint_set=registration['config']['protocol'], constraint_space=space.diagnostics,
        physical_sites=[12,13,14,18,19,20], local_groups=dict(A=[0,3,4],B=[1,5,2]),
        invariant_operator_reconstruction_errors=recon_errors, records=records,
        no_new_fits=True,no_new_queries=True,
        warnings=['These are old matching5600/6600 andchi4096/sweep24 states, not7600/chi10240.',
                  'A finite ME residual is failure of this specific exponential model, not proof that every positive state with those low moments must disagree.',
                  'No ME-residual statistical significance is claimed without matched refit/influence uncertainty.'])
    atomic_json(OUT/'cached_maxent_six_body_audit.json',maxent)
    ct = RESULTS/'su2_mechanism_local_20261028/ct_A1_compact6'
    with np.load(RESULTS/'su2_mechanism_local_20261028/interventions/frozen_operators.npz') as f:
        ops = {k:f[k] for k in ('vector_chirality_A_dot_B','vector_chirality_qtotal0','vector_chirality_qtotal2')}
    with np.load(OUT/'frozen_projectors.npz') as f:
        ops['frozen_DMRG_singlet_main_projector'] = f['compact6_q0_reference_main_projector']
    ops.update({k:bank[k] for k in ('T','T_qtotal0','T_qtotal2')})
    npz = ct/'ct6_j0_physical_region.npz'
    original = json.loads((ct/'ct6_j0_results.json').read_text())
    with np.load(npz) as f:
        rows,sites,phi,q,rho = [f[k] for k in ('rows','sites','region_amplitudes','proposal','rho_Hermitian')]
    packed = rows[:,sites]@(2**np.arange(5,-1,-1))
    center = phi[np.arange(len(rows)),packed]
    mass = abs(center)**2/q
    independent = {}
    for name,O in ops.items():
        action = (phi@O.T)[np.arange(len(rows)),packed]
        numerator = center.conj()*action/q
        value = numerator.sum()/mass.sum()
        group_n = numerator.reshape(32,-1).sum(axis=1)
        group_d = mass.reshape(32,-1).sum(axis=1)
        leave = (numerator.sum()-group_n)/(mass.sum()-group_d)
        se = float(np.sqrt(31/32*np.sum(abs(leave-leave.mean())**2)))
        old = original['readout']['observables'].get(name)
        independent[name] = dict(real=float(value.real),imaginary=float(value.imag),group_jackknife_SE=se,
            original_point_difference=float(abs(value.real-old['real'])) if old else None,
            original_SE_difference=float(abs(se-old['group_jackknife_SE'])) if old else None,
            RDM_consistency_error=float(abs(value.real-np.trace(rho@O).real)))
    ct_result = dict(task='independent_CT_J0_compact6_cache_recalculation',
        source=str(npz.relative_to(ROOT)),source_sha256=file_sha256(npz),
        records=len(rows),groups=32,complete_A1_images=original['complete_A1_images'],
        observables=independent,no_new_amplitude_queries=True,
        cached_rows_not_a_new_independent_audit=True,
        wall_seconds_total=perf_counter()-start)
    atomic_json(OUT/'ct_J0_A1_independent_cache_review.json',ct_result)
    print(json.dumps(dict(maxent_panels=len(records),ct_observables=independent,seconds=perf_counter()-start),indent=2))


if __name__=='__main__':
    main()
