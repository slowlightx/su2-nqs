"""Actual magnetic FCS: dense, gauge, noncanonical, and physical controls."""
import numpy as np
import pytest

from su2_nqs.benchmarks.mps_spin_rotation import (
    estimate_prefix_rotation_memory, exact_spin_half_charge_distribution,
    finite_mps_prefix_charge_distribution, physical_spin_rotation_summary,
    sampled_prefix_charge_distribution,
)
from su2_nqs.benchmarks.spin_rotation_calibration import (
    kagome_rvb_reference, lieb_mattis_singlet, singlet_covering_state,
)


def test_dimer_crossing_cuts_and_ordered_singlet_are_non_QSL_FCS_controls():
    angles = np.linspace(0, 2*np.pi, 41)
    state = singlet_covering_state(8, [(0,4),(1,5),(2,3),(6,7)])
    m, p = exact_spin_half_charge_distribution(state, [0,1,2,3])
    summary = physical_spin_rotation_summary(m,p,n_region=4,angles=angles)
    np.testing.assert_allclose(summary['rotation_real'],np.cos(angles/2)**2,atol=2e-14,rtol=0)
    assert summary['variance_sz'] == pytest.approx(.5,abs=2e-14)
    assert not summary['qsl_specific'] and not summary['schmidt_spectrum_measured']
    ordered = lieb_mattis_singlet(8, [0,1,2,3])
    _, ordered_p = exact_spin_half_charge_distribution(ordered, [0,1,2,3])
    np.testing.assert_allclose(ordered_p,np.full(5,.2),atol=2e-14,rtol=0)
    assert physical_spin_rotation_summary(m,ordered_p,n_region=4,angles=angles)['variance_sz'] == pytest.approx(2,abs=2e-14)


def test_physical_center_is_only_region_parity_not_fractionalization():
    for n in (2,4,6):
        state = singlet_covering_state(n, [(i,i+1) for i in range(0,n,2)])
        for cut in range(1,n):
            m,p = exact_spin_half_charge_distribution(state,range(cut))
            z = physical_spin_rotation_summary(m,p,n_region=cut,angles=np.array([2*np.pi]))
            assert z['physical_2pi_center_error'] < 2e-14
            assert z['rotation_real'][0] == pytest.approx((-1)**cut,abs=2e-14)


@pytest.mark.parametrize('bad', [np.array([-1.,1.]),np.array([-2,1]),np.array([-1,-1])])
def test_physical_magnetic_labels_reject_float_bad_parity_and_duplicates(bad):
    with pytest.raises(ValueError):
        physical_spin_rotation_summary(bad,np.array([.5,.5]),n_region=1,angles=np.array([0.]))


def test_bounded_memory_estimate_does_not_call_charge_readout_free():
    estimate = estimate_prefix_rotation_memory(10240,3_205_484_368)
    assert 25 < estimate['estimated_total_bytes']/1024**3 < 26
    assert not estimate['estimate_is_hard_guarantee']
    assert not estimate['sparsity_assumed']


def test_saved_born_histogram_matches_explicit_two_chain_count_without_SE_claim():
    bits = np.array([[[0,1,1,0],[1,1,0,0]],
                     [[1,0,0,1],[1,1,1,0]]], dtype=np.int8)
    m,p = sampled_prefix_charge_distribution(bits,3)
    np.testing.assert_array_equal(m,[-3,-1,1,3])
    np.testing.assert_array_equal(p,[0,.25,.5,.25])
    with pytest.raises(ValueError):
        sampled_prefix_charge_distribution(bits.astype(float),3)


def _small_mps(n, total_m, seed):
    pytest.importorskip('tenpy')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    site = SpinHalfSite(conserve='Sz')
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    labels = np.array([2*i.bit_count()-n for i in range(2**n)])
    state[labels != total_m] = 0
    state /= np.linalg.norm(state)
    tensor = npc.Array.from_ndarray(state.reshape([2]*n),[site.leg]*n,
                qtotal=[total_m],labels=[f'p{i}' for i in range(n)])
    psi = MPS.from_full([site]*n,tensor,bc='finite',unit_cell_width=n)
    return psi,state


@pytest.mark.parametrize('total_m',[0,2,-2])
def test_actual_two_sided_distribution_matches_dense_and_complement(total_m):
    psi,state = _small_mps(8,total_m,42)
    for cut in (1,3,4,7):
        m,p,meta = finite_mps_prefix_charge_distribution(psi,cut)
        exact_m,exact_p = exact_spin_half_charge_distribution(state,range(cut))
        np.testing.assert_array_equal(m,exact_m)
        np.testing.assert_allclose(p,exact_p,atol=2e-14,rtol=0)
        cm,cp = exact_spin_half_charge_distribution(state,range(cut,8))
        for charge,weight in zip(m,p):
            match = cp[cm==total_m-charge]
            assert weight == pytest.approx(float(match.sum()),abs=2e-14)
        assert meta['physical_total_doubled_sz'] == total_m
        assert meta['stored_schmidt_weights_used'] is False


def test_virtual_charge_gauges_cannot_change_actual_physical_prefix_distribution():
    psi,state = _small_mps(6,2,11)
    exact_m,exact_p = exact_spin_half_charge_distribution(state,range(3))
    for gauge in ([5],np.array([[1],[-2],[3],[1],[0],[4]])):
        shifted = psi.copy()
        shifted.gauge_total_charge(gauge)
        m,p,meta = finite_mps_prefix_charge_distribution(shifted,3)
        np.testing.assert_array_equal(m,exact_m)
        np.testing.assert_allclose(p,exact_p,atol=2e-14,rtol=0)
        assert meta['physical_total_doubled_sz'] == 2
    assert meta['virtual_charge_to_physical_offset'] == 2


def test_noncanonical_parent_measured_unchanged_and_canonical_copy_agrees():
    psi,state = _small_mps(8,0,13)
    # Physical diagonal filter keeps U(1), breaks canonicality. No S data
    # updated: a raw S^2 shortcut is now wrong, whereas true environments work.
    filter_values = np.array([.25,2.])
    psi._B[2] = psi._B[2].scale_axis(filter_values,'p')
    before = [b.to_ndarray().copy() for b in psi._B]
    stored_s = [np.asarray(s).copy() for s in psi._S]
    filtered = state*filter_values[(np.arange(2**8) >> 5) & 1]
    m,p,_ = finite_mps_prefix_charge_distribution(psi,4)
    exact_m,exact_p = exact_spin_half_charge_distribution(filtered,range(4))
    np.testing.assert_array_equal(m,exact_m)
    np.testing.assert_allclose(p,exact_p,atol=2e-14,rtol=0)
    # On a separate copy only, full canonicalization must restore the genuine
    # Schmidt-based magnetic distribution and keep the same physical FCS.
    canonical = psi.copy()
    canonical.canonical_form()
    _,copy_p,_ = finite_mps_prefix_charge_distribution(canonical,4)
    np.testing.assert_allclose(copy_p,p,atol=2e-14,rtol=0)
    charges = psi.get_B(4,form=None).get_leg('vL').to_qflat()[:,0]
    raw = np.array([sum(np.asarray(psi.get_SL(4))**2*(charges==q)) for q in m])
    assert np.max(abs(raw-p)) > .02
    for b,value in zip(psi._B,before):
        np.testing.assert_array_equal(b.to_ndarray(),value)
    for s,value in zip(psi._S,stored_s):
        np.testing.assert_array_equal(s,value)


def test_shifted_left_boundary_and_site_tensor_totals_preserve_physical_prefix():
    psi,state = _small_mps(6,-2,16)
    from tenpy.linalg import np_conserved as npc
    old = psi.get_B(0,form=None).get_leg('vL')
    new = npc.LegCharge.from_qflat(psi.chinfo,old.to_qflat()+7,qconj=1)
    shifted = psi.copy()
    shifted.gauge_total_charge(np.array([[2],[1],[-4],[0],[3],[5]]),vL_leg=new)
    m,p,meta = finite_mps_prefix_charge_distribution(shifted,3)
    expected_m,expected_p = exact_spin_half_charge_distribution(state,range(3))
    np.testing.assert_array_equal(m,expected_m)
    np.testing.assert_allclose(p,expected_p,atol=2e-14,rtol=0)
    assert meta['physical_total_doubled_sz'] == -2
    assert meta['virtual_charge_to_physical_offset'] == -8


@pytest.mark.parametrize('total_m,filtered,gauged',[(0,False,False),(2,False,True),
                                                 (-2,True,False),(0,True,True)])
def test_charged_renyi2_matches_actual_complex_dense_RDM_not_raw_S(total_m,filtered,gauged):
    psi,state = _small_mps(8,total_m,65)
    if filtered:
        values = np.array([.2,1.7])
        psi._B[1] = psi._B[1].scale_axis(values,'p')
        state = state*values[(np.arange(2**8) >> 6) & 1]
        # Break BOTH Gram metrics; a canonical/identity RP would fail to
        # expose an accidental transpose for a complex wavefunction.
        right_values = np.array([1.4,.3])
        psi._B[6] = psi._B[6].scale_axis(right_values,'p')
        state = state*right_values[(np.arange(2**8) >> 1) & 1]
    if gauged:
        psi.gauge_total_charge(np.array([[2],[-1],[3],[1],[-5],[4],[2],[8]]))
    state /= np.linalg.norm(state)
    cut = 4
    # Independent actual partial trace in physical computational ordering.
    wavefunction = state.reshape(2**cut,-1)
    rho = wavefunction @ wavefunction.conj().T
    labels = np.array([2*i.bit_count()-cut for i in range(2**cut)])
    m,p,meta = finite_mps_prefix_charge_distribution(psi,cut,renyi2=True)
    exact = np.array([np.trace(rho[np.ix_(labels==q,labels==q)] @
                             rho[np.ix_(labels==q,labels==q)]).real for q in m])
    np.testing.assert_allclose(meta['renyi2']['sector_purities'],exact,atol=2e-14,rtol=0)
    assert np.all(exact <= p**2+2e-14)
    assert meta['renyi2']['purity'] == pytest.approx(np.trace(rho@rho).real,abs=2e-14)
    assert meta['renyi2']['gram_norm_closure_error'] < 2e-14
    canonical = psi.copy(); canonical.canonical_form()
    _,_,reference = finite_mps_prefix_charge_distribution(canonical,cut,renyi2=True)
    np.testing.assert_allclose(reference['renyi2']['sector_purities'],exact,atol=2e-14,rtol=0)


@pytest.mark.parametrize('kind',['finite_RVB','three_crossing_dimers','ordered_singlet'])
def test_positive_family_and_negative_controls_use_same_actual_Renyi_oracle(kind):
    pytest.importorskip('tenpy')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    if kind == 'finite_RVB':
        data = kagome_rvb_reference()
        state = data['total_state']
        n,cut = 12,6
        assert not data['intrinsic_topology_certified']
    else:
        n,cut = 6,3
        state = (singlet_covering_state(6,[(0,3),(1,4),(2,5)])
                 if kind == 'three_crossing_dimers' else lieb_mattis_singlet(6,[0,1,2]))
    site = SpinHalfSite(conserve='Sz')
    tensor = npc.Array.from_ndarray(state.reshape([2]*n),[site.leg]*n,
                                  qtotal=[0],labels=[f'p{i}' for i in range(n)])
    psi = MPS.from_full([site]*n,tensor,bc='finite',unit_cell_width=n)
    _,_,meta = finite_mps_prefix_charge_distribution(psi,cut,renyi2=True)
    rho = state.reshape(2**cut,-1) @ state.reshape(2**cut,-1).conj().T
    assert meta['renyi2']['purity'] == pytest.approx(np.trace(rho@rho).real,abs=2e-14)
    assert meta['renyi2']['strict_su2_irrep_resolution'] is False
    if kind == 'three_crossing_dimers':
        assert meta['renyi2']['purity'] == pytest.approx(1/8,abs=2e-14)
    elif kind == 'ordered_singlet':
        assert meta['renyi2']['purity'] == pytest.approx(1/4,abs=2e-14)


def test_bounded_workflow_preserves_parent_and_seals_result(tmp_path):
    psi,state = _small_mps(6,0,23)
    import hashlib
    import json
    import h5py
    import tenpy
    from tenpy.tools import hdf5_io
    from projects.clebsch_tree_nqs.workflows.mps_spin_rotation_probe import run_probe
    checkpoint = tmp_path/'checkpoint.h5'
    with h5py.File(checkpoint,'w') as handle:
        hdf5_io.save_to_hdf5(handle,dict(psi=psi,tenpy_version=tenpy.__version__))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    spec = dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,n_sites=6,
                chi_upper_bound=8,cut=3,threads=1,wall_seconds=30,memory_gib=2,renyi2=True)
    config = tmp_path/'config.json'
    config.write_text(json.dumps(spec))
    output = tmp_path/'readout'
    result = run_probe(config,output)
    assert result['status'] == 'complete'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    child = result['readout']
    assert child['parent_fingerprint_before'] == child['parent_fingerprint_after']
    expected_m,expected_p = exact_spin_half_charge_distribution(state,range(3))
    np.testing.assert_allclose(child['readout']['probabilities'],expected_p,atol=2e-14,rtol=0)
    np.testing.assert_array_equal(child['readout']['magnetic_labels'],expected_m)
    assert child['new_training'] == child['new_samples'] == child['new_ED'] == 0
    assert not child['strict_su2_schmidt_spectrum_measured']
    assert result['renyi2_attempt']['status'] == 'complete'
    second = result['renyi2_attempt']['readout']
    rho = state.reshape(8,8) @ state.reshape(8,8).conj().T
    assert second['charged_renyi2']['normalization'] == pytest.approx(np.trace(rho@rho).real,abs=2e-14)
    assert second['charged_renyi2']['contains_off_diagonal_coherence'] is True
    assert second['worker_rss_watchdog_observed'] is True
    with pytest.raises(FileExistsError):
        run_probe(config,output)


def test_bounded_workflow_refuses_nonphysical_cut_and_unsealed_input(tmp_path):
    from projects.clebsch_tree_nqs.workflows.mps_spin_rotation_probe import _validated_spec
    valid = dict(checkpoint='state.h5',checkpoint_sha256='a'*64,n_sites=36,
                 chi_upper_bound=10240,cut=18,threads=32,wall_seconds=1200,memory_gib=128)
    assert _validated_spec(valid)['angle_count'] == 81
    for changes in (dict(cut=36),dict(cut=True),dict(checkpoint_sha256='x'),
                    dict(threads=0),dict(angle_count=80),dict(memory_gib=float('inf')),dict(renyi2=1)):
        with pytest.raises(ValueError):
            _validated_spec({**valid,**changes})
