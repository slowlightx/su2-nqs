"""Collective-singlet controls and same-coupling distance checks."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_competing_state_readout as w
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
from su2_nqs.benchmarks.regional_su2_readout import su2_multiplicity_readout
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.su2.sz_basis import enumerate_sz_configs


@pytest.mark.parametrize('blocks,cross', [(((0,1),(2,3),(4,5)),-.25),(((0,1,2),(3,4,5)),-5/12)])
def test_collective_singlet_matches_independent_casimir_sum(blocks,cross):
    b=np.array(enumerate_sz_configs(6,3))
    v=w.collective_sublattice_singlet(b,blocks)
    a=FixedSzPairActions(b)
    assert np.vdot(v,v).real==pytest.approx(1.,abs=2e-15)
    assert spin_raising_relative_norm(v,b)<2e-14
    assert np.vdot(v,a.pair(v,blocks[0][0],blocks[0][1])).real==pytest.approx(.25,abs=2e-15)
    assert np.vdot(v,a.pair(v,blocks[0][0],blocks[1][0])).real==pytest.approx(cross,abs=2e-15)
    rho=reduced_density_matrix(v,b,(0,1,2))
    r=su2_multiplicity_readout(rho)
    assert r['diagnostics']['factorization_frobenius']<1e-14
    assert max(w._distance(rho,rho,r,r).values())<1e-14


def test_three_maximal_spins_not_reachable_are_rejected():
    b=np.array(enumerate_sz_configs(6,3))
    with pytest.raises(ValueError,match='cannot couple'):
        w.collective_sublattice_singlet(b,((0,1,2,3),(4,),(5,)))


def test_region_protocol_and_new_work_are_bounded():
    import json
    spec=json.loads(w.CONFIG.read_text())
    assert [len(s) for s in spec['regions'].values()]==[2,4,6,8]
    cuts=list(spec['regions'].values())
    assert all(set(a)<=set(b) for a,b in zip(cuts,cuts[1:]))
    assert not any(spec[k] for k in ('new_ed','new_samples','new_training'))


def test_physical_basis_reindexing_changes_no_amplitudes_or_rdm():
    b=np.array(enumerate_sz_configs(6,3))
    v=w.collective_sublattice_singlet(b,((0,1),(2,3),(4,5)))
    order=np.random.default_rng(3).permutation(len(b))
    np.testing.assert_allclose(w.align_basis_vector(v[order],b[order],b),v,atol=0)
    np.testing.assert_allclose(reduced_density_matrix(v[order],b[order],(0,1)),
                               reduced_density_matrix(v,b,(0,1)),atol=1e-15)
    with pytest.raises(ValueError,match='identical and unique'):
        w.align_basis_vector(v,np.zeros_like(b),b)


def test_completed_readout_preserves_sources_and_region_factorization():
    import json
    path=w.ROOT/'projects/clebsch_tree_nqs/results/spin_competing_state_readout_v1'
    if not (path/'results.json').exists():
        pytest.skip('local readout not computed yet')
    r=json.loads((path/'results.json').read_text())
    assert w.canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'})==r['result_sha256']
    assert w.file_sha256(path/'readout.npz')==r['readout_sha256']
    for p,sha in r['registration']['inputs'].items():
        assert w.file_sha256(w.ROOT/p)==sha
    snapshots=json.loads((path/'source_snapshots/manifest.json').read_text())['original_to_snapshot']
    for p,sha in r['sources'].items():
        producer=path/snapshots[p] if p in snapshots else w.ROOT/p
        assert w.file_sha256(producer)==sha,p
    with np.load(path/'readout.npz') as z:
        for name,record in r['states'].items():
            if not record['strict_su2_eligible']:
                continue
            for cut,sites in r['config']['regions'].items():
                rho=reduced_density_matrix(z[name+'_state'],z['basis'],sites)
                np.testing.assert_allclose(rho,z[name+'_'+cut+'_rho'],atol=1e-14)
                replay=su2_multiplicity_readout(rho)
                assert replay['multiplicity_entropy']==pytest.approx(record['regions'][cut]['multiplicity_entropy'],abs=2e-13)


def test_derived_cyclic_readout_preserves_inputs_and_q_marginals():
    import json
    from su2_nqs.benchmarks.regional_su2_readout import su2_cyclic_entanglement_readout
    path=w.ROOT/'projects/clebsch_tree_nqs/results/spin_competing_state_readout_v1'
    if not (path/'cyclic_readout.json').exists():
        pytest.skip('derived readout not computed yet')
    r=json.loads((path/'cyclic_readout.json').read_text())
    parent=json.loads((path/'results.json').read_text())
    assert w.canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'})==r['result_sha256']
    assert w.file_sha256(path/'results.json')==r['parent_result_sha256']
    assert w.file_sha256(path/'readout.npz')==r['parent_readout_sha256']
    for p,sha in r['sources'].items():
        assert w.file_sha256(w.ROOT/p)==sha,p
    with np.load(path/'readout.npz') as z:
        for name,record in r['states'].items():
            if record['status']!='eligible':
                continue
            replay=su2_cyclic_entanglement_readout(z[name+'_strip_8_rho'],tuple(r['active_permutation']),order=4)
            for q,sectors in replay['sectors'].items():
                assert sum(s['p'] for s in sectors.values())==pytest.approx(parent['states'][name]['regions']['strip_8']['sectors'][q]['p'],abs=2e-13)
                for k,sector in sectors.items():
                    assert sector['p']==pytest.approx(record['readout']['sectors'][q][k]['p'],abs=2e-13)
    change=r['coupling_distances_from_j0']['0.125']['joint_q_k_total_variation']
    assert change>=parent['coupling_distances_from_j0']['0.125']['strip_8']['p_q_total_variation']
    assert not any(r[k] for k in ('new_ED','new_training','new_samples','new_full_H_actions'))
