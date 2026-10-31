"""Saved-state/operator replay and an independent compressed-operator oracle."""
import json
from pathlib import Path
import numpy as np
from su2_nqs.frontier.provenance import file_sha256
from projects.clebsch_tree_nqs.workflows.spin_readout_information import restricted_span
from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split

ROOT=Path(__file__).resolve().parents[1]


def test_restricted_operator_against_full_dense_embedding():
    n=6; full=np.arange(64);basis=((full[:,None]>>np.arange(5,-1,-1))&1).astype(np.int8)
    rng=np.random.default_rng(2026100607);u=np.linalg.qr(rng.normal(size=(64,3)))[0]
    splits={'1':joint_spin_operator_split(6,(0,2,4),(1,1,0))}
    result=restricted_span(basis,u,list(range(6)),splits)
    # A generic 3-dimensional subspace is already covered by the scalar span;
    # this checks the dimension-dependent, not universally nonzero, remainder.
    assert result['targets']['1']['relative_residual'] < 1e-12
    assert result['restricted_low_order_rank'] == 6


def test_saved_six_body_readout_and_sources_replay():
    folder=ROOT/'projects/clebsch_tree_nqs/results/spin_readout_information_v1'
    reg=json.loads((folder/'registration.json').read_text());r=json.loads((folder/'results.json').read_text())
    for p,h in reg['inputs'].items():assert file_sha256(ROOT/p)==h
    for p,h in reg['sources'].items():assert file_sha256(folder/'source_snapshots'/p)==h
    assert r['restricted_sector_check']['saved_subspace_dimension']==14
    assert r['restricted_sector_check']['restricted_low_order_rank']==22
    assert abs(r['coupling_j0_to_j0125']['3']['high']-.00211490890123733) < 1e-13
    assert not r['phase_claim'] and not r['size_evolution_released']
    assert all(c['exhaustive_max_element_error'] < 1e-13 for c in r['sampling_calibration'].values())
    assert all(r[k]==0 for k in ('new_states','new_training','new_ED'))


def test_N36_attention_amplitudes_match_preexisting_full_H_feature_cache():
    import jax.numpy as jnp
    from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
    from projects.clebsch_tree_nqs.workflows.spin_nqs_region_readout import SavedStates,DEFAULT
    spec=json.loads(DEFAULT.read_text());model=SavedStates(spec,Path('/private/tmp'))
    source=ROOT/spec['attention_source']
    with np.load(source/'audit_samples.npz') as z:x=z['states'][:2]
    with np.load(source/'audit_features.npz') as z:c=z['carrier'][:2];features=z['features'][:2]
    cc=model.carrier(x);ratio=cc/c
    np.testing.assert_allclose(ratio,ratio[0],atol=1e-11,rtol=0)
    template=dict(parameters=jnp.zeros(730),generator_table=jnp.zeros((6,6,10)),coefficients=jnp.zeros(1176))
    expected=[]
    for name in ('initial0.json','attention0_endpoint.json'):
        old=load_parameter_artifact(source/name,template)
        expected.append((c+features@np.asarray(old['coefficients']))*ratio[0])
    np.testing.assert_allclose(model.attention(x),np.column_stack(expected),atol=2e-12,rtol=2e-10)


def test_old_optimized_pair_readout_replay_without_repeating_optimizer():
    from su2_nqs.frontier.artifacts import canonical_sha256
    from su2_nqs.benchmarks.xdiag_physical_seed_krylov import spin_raising_relative_norm
    folder=ROOT/'projects/clebsch_tree_nqs/results/spin_readout_information_v1/optimized_carrier_reuse'
    reg=json.loads((folder/'registration.json').read_text());r=json.loads((folder/'results.json').read_text())
    for p,h in reg['inputs'].items():assert file_sha256(ROOT/p)==h
    for p,h in reg['sources'].items():assert file_sha256(folder/'source_snapshots'/p)==h
    old=json.loads((ROOT/'projects/clebsch_tree_nqs/results/spin_adaptive_projected_pair_carrier_v3/results.json').read_text())
    digest=old.pop('result_sha256');assert canonical_sha256(old)==digest
    expected=next(row for row in old['rows'] if row['j2_over_j1']==.125)['physical_adaptive_carrier']
    # Registration identifies the frozen physical isometry/H directly; do not
    # construct or diagonalize another many-body Hamiltonian to test a replay.
    bundle=next(ROOT/p for p in reg['inputs'] if p.endswith('.npz'))
    with np.load(bundle) as z:u,h=z['physical_a1_basis'],z['h_a1']
    with np.load(folder/'states.npz') as z:
        bits=z['basis'];integers=bits@(1<<np.arange(15,-1,-1))
        for degree in (0,2):
            name=f'optimized_pair_k{degree}';psi=z[name];c=u.conj().T@psi
            assert np.linalg.norm(psi-u@c)<1e-13
            assert spin_raising_relative_norm(psi,integers.tolist(),16)[0]<1e-12
            energy=np.vdot(c,h@c).real/16
            variance=np.linalg.norm(h@c-16*energy*c)**2/16
            np.testing.assert_allclose([energy,variance],
                [expected[str(degree)][key] for key in ('energy_per_site','variance_per_site')],atol=2e-13,rtol=0)
            np.testing.assert_allclose(joint_spin_operator_split(6,(0,2,4),(3,3,0)).measure(z[name+'_rho'])['high'],
                r['states'][name]['signals']['3']['high'],atol=1e-14,rtol=0)
    assert all(r[key]==0 for key in ('new_training','new_optimizer_evaluations','new_many_body_ED','new_samples'))
    assert not r['old_optimizer_repeated'] and not r['distinct_QSL_claim']


def test_bounded_pair_quality_artifact_replays_without_another_optimizer():
    from su2_nqs.benchmarks.spin_operator_information import three_bond_readout
    from su2_nqs.benchmarks.xdiag_physical_seed_krylov import spin_raising_relative_norm
    folder=ROOT/'projects/clebsch_tree_nqs/results/spin_pair_quality_v1'
    registration=json.loads((folder/'registration.json').read_text())
    result=json.loads((folder/'results.json').read_text())
    completion=json.loads((folder/'completion.json').read_text())
    for path,digest in registration['inputs'].items():assert file_sha256(ROOT/path)==digest
    for path,digest in registration['sources'].items():assert file_sha256(folder/'source_snapshots'/path)==digest
    for path,digest in completion['outputs'].items():assert file_sha256(folder/path)==digest
    bundle=next(ROOT/p for p in registration['inputs'] if p.endswith('.npz'))
    with np.load(bundle) as z:u,h=z['physical_a1_basis'],z['h_a1']
    with np.load(folder/'states_readouts.npz') as z:
        basis=z['basis'];integers=(basis@(1<<np.arange(15,-1,-1))).tolist()
        for trajectory in result['trajectories']:
            name=trajectory['final_label'];psi=z[name+'_state'];c=u.conj().T@psi
            assert np.linalg.norm(psi-u@c)<1e-13
            assert np.max(abs(psi.imag))<1e-13
            assert spin_raising_relative_norm(psi.real,integers,16)[0]<2e-12
            e=np.vdot(c,h@c).real/16
            variance=np.linalg.norm(h@c-16*e*c)**2/16
            expected=result['states'][name]
            np.testing.assert_allclose([e,variance],
                [expected['energy_per_site'],expected['variance_per_site']],atol=2e-13,rtol=0)
            bond=three_bond_readout(z[name+'_rho'],(0,2,4))
            assert abs(bond['high_q3']-expected['three_bond']['high_q3'])<1e-13
            assert expected['readout']['strict_su2_eligible']
            assert trajectory['iterations']<=96 and trajectory['objective_calls']<=160
    assert result['optimized_pair_family_count']==1
    assert result['new_many_body_ED']==result['new_MC_samples']==result['new_heads']==0
    assert not result['phase_claim'] and not result['size_evolution_released']
    assert result['maximum_explicit_projection_error']<1e-13
    assert max(result['gradient_fd_errors'])<2e-7
