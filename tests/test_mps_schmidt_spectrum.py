"""Physical Schmidt spectra: independent dense/gauge/noncanonical oracles."""
import numpy as np
import pytest

from su2_nqs.benchmarks.mps_schmidt_spectrum import (
    finite_mps_prefix_schmidt_spectrum, gram_schmidt_spectrum,
)


@pytest.mark.parametrize('complex_state', [False, True])
def test_two_sided_noncanonical_gram_spectrum_matches_physical_SVD(complex_state):
    rng = np.random.default_rng(104)
    left = rng.normal(size=(5, 4))
    right = rng.normal(size=(7, 4))
    if complex_state:
        left = left+1j*rng.normal(size=left.shape)
        right = right+1j*rng.normal(size=right.shape)
    wavefunction = left @ right.T
    singular_values = np.linalg.svd(wavefunction, compute_uv=False)
    expected = singular_values**2/np.sum(singular_values**2)
    # Four-dimensional virtual basis all carries the same legal m=0.
    report = gram_schmidt_spectrum(left.conj().T @ left,
        (right.conj().T @ right).T, np.zeros(4, dtype=int), n_region=4)
    actual = np.asarray(report['sectors']['0']['eigenvalues'])
    np.testing.assert_allclose(actual, expected[:4], atol=5e-15, rtol=0)
    assert report['purity'] == pytest.approx(expected @ expected, abs=5e-15)
    assert report['spectral_trace_error'] < 5e-15
    assert not report['q_labels_assigned']


def test_magnetic_block_conservation_and_positive_metrics_are_validated():
    labels = np.array([-2, 0, 0, 2])
    report = gram_schmidt_spectrum(np.eye(4), np.eye(4), labels, n_region=2)
    assert report['magnetic_spin_inversion_spectrum_l1'] == 0
    for left, right in [(np.ones((4,4)), np.eye(4)),
                        (np.diag([-1.,1.,1.,1.]), np.eye(4)),
                        (np.eye(4)+1j*np.eye(4), np.eye(4))]:
        with pytest.raises(ValueError):
            gram_schmidt_spectrum(left, right, labels, n_region=2)


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
    tensor = npc.Array.from_ndarray(state.reshape([2]*n), [site.leg]*n,
                qtotal=[total_m], labels=[f'p{i}' for i in range(n)])
    return MPS.from_full([site]*n, tensor, bc='finite', unit_cell_width=n), state


@pytest.mark.parametrize('total_m,filtered,gauged', [(0,False,False), (2,True,False),
                                                  (-2,False,True), (0,True,True)])
def test_actual_MPS_spectrum_matches_dense_partial_trace(total_m, filtered, gauged):
    psi, state = _small_mps(8, total_m, 116)
    if filtered:
        for site, values in [(1, [.3,1.7]), (6, [1.4,.2])]:
            psi._B[site] = psi._B[site].scale_axis(np.asarray(values), 'p')
            state *= np.asarray(values)[(np.arange(256) >> (7-site)) & 1]
    if gauged:
        psi.gauge_total_charge(np.array([[2],[-1],[4],[1],[-3],[5],[0],[2]]))
    state /= np.linalg.norm(state)
    tensors_before = [b.to_ndarray().copy() for b in psi._B]
    schmidt_before = [np.asarray(s).copy() for s in psi._S]
    for cut in (3,4,5):
        wavefunction = state.reshape(2**cut, -1)
        rho = wavefunction @ wavefunction.conj().T
        charges = np.array([2*i.bit_count()-cut for i in range(2**cut)])
        report = finite_mps_prefix_schmidt_spectrum(psi, cut)
        for m, sector in report['sectors'].items():
            exact = np.linalg.eigvalsh(rho[np.ix_(charges == int(m), charges == int(m))])[::-1]
            actual = np.asarray(sector['eigenvalues'])
            size = max(len(actual),len(exact))
            np.testing.assert_allclose(np.pad(actual,(0,size-len(actual))),
                                       np.pad(exact,(0,size-len(exact))),atol=8e-15,rtol=0)
        assert report['purity'] == pytest.approx(np.trace(rho @ rho).real, abs=8e-15)
        assert report['physical_total_doubled_sz'] == total_m
        assert report['tensor_contraction_closure_relative_error'] < 8e-15
    for before,b in zip(tensors_before,psi._B):
        np.testing.assert_array_equal(before,b.to_ndarray())
    for before,s in zip(schmidt_before,psi._S):
        np.testing.assert_array_equal(before,s)


def test_same_charge_degeneracies_do_not_claim_SU2_or_a_tower():
    # Two crossing dimers have a maximally mixed four-spin subspace on their
    # endpoints. A multiplet-looking degeneracy is already present and is
    # not unique to magnetic order or a QSL.
    report = gram_schmidt_spectrum(np.eye(4), np.eye(4),
                                  np.array([-2,0,0,2]), n_region=2)
    assert report['magnetic_spin_inversion_spectrum_l1'] == 0
    assert not report['strict_su2_verified']
    assert not report['entanglement_tower_identified']


def test_bounded_schmidt_worker_seals_unmodified_parent(tmp_path):
    psi, state = _small_mps(6, 0, 182)
    import hashlib
    import json
    import h5py
    import tenpy
    from tenpy.tools import hdf5_io
    from projects.clebsch_tree_nqs.workflows.mps_schmidt_probe import run_probe
    checkpoint = tmp_path/'checkpoint.h5'
    with h5py.File(checkpoint, 'w') as handle:
        hdf5_io.save_to_hdf5(handle, dict(psi=psi, tenpy_version=tenpy.__version__))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(checkpoint=str(checkpoint),
        checkpoint_sha256=digest, n_sites=6, chi_upper_bound=8, cut=3,
        threads=1, wall_seconds=30, memory_gib=2)))
    result = run_probe(config, tmp_path/'output')
    assert result['status'] == 'complete'
    worker = result['readout']
    assert worker['parent_fingerprint_before'] == worker['parent_fingerprint_after']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    wavefunction = state.reshape(8,8)
    rho = wavefunction @ wavefunction.conj().T
    assert worker['readout']['purity'] == pytest.approx(np.trace(rho @ rho).real, abs=8e-15)
    assert not worker['strict_su2_irrep_spectrum_measured']
    with pytest.raises(FileExistsError):
        run_probe(config, tmp_path/'output')
