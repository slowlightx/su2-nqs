"""Independent exact RDM/order and bounded acquisition safety checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.dmrg_region_probe import (
    TRIANGULAR_REGIONS, bounded_subprocess, collect_regions, estimate_region_memory, raw_rdm_diagnostics,
    materialize_completed_parent,
)


def _write_checkpoint(path, psi):
    import h5py
    import tenpy
    from tenpy.tools import hdf5_io
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        hdf5_io.save_to_hdf5(handle, dict(psi=psi, tenpy_version=tenpy.__version__))


def _config(tmp_path, checkpoint, *, n, chi, dtype, regions=None, **extra):
    spec = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                n_sites=n, chi_upper_bound=chi, dtype=dtype, output=str(tmp_path/'readout'),
                limits=dict(memory_gib=2., wall_seconds=30., threads=1))
    if regions is not None:
        spec['regions'] = regions
    spec.update(extra)
    config = tmp_path/'config.json'
    config.write_text(json.dumps(spec))
    return config


def _rho_exact(vector, sites, n):
    complement = [i for i in range(n) if i not in sites]
    amplitudes = vector.reshape([2]*n).transpose(list(sites)+complement).reshape(2**len(sites), -1)
    return amplitudes@amplitudes.conj().T


def test_dense_noncontiguous_memory_estimate_cannot_be_confused_with_output_rdm_size():
    for k, expected in [(6, 128*1024**3), (7, 512*1024**3), (8, 2*1024**4)]:
        sites = [0, 2, 4, 6, 8, 10, 12, 14][:k]
        result = estimate_region_memory(sites, 4096, itemsize=8)
        assert result['principal_dense_bytes'] == expected
        assert result['estimated_total_bytes'] > 2*expected
        assert not result['consecutive'] and not result['sparsity_assumed']
    contiguous = estimate_region_memory(list(range(6)), 4096, itemsize=8)
    assert contiguous['principal_dense_bytes'] == 8*2**6*4096**2
    assert contiguous['consecutive']
    with pytest.raises(ValueError):
        estimate_region_memory([0, 0], 4, itemsize=8)


def test_invalid_raw_rdm_keeps_diagnostics_without_repair_or_fake_spectrum():
    matrix = np.diag([-.03, .2, .3, .53]).astype(complex)
    saved = matrix.copy()
    report = raw_rdm_diagnostics(matrix, tolerance=1e-9)
    assert not report['physical_matrix_valid']
    assert not report['strict_su2_eligible']
    assert not report['strict_entanglement_spectrum_reported']
    assert not report['input_repaired']
    assert 'p_q' not in report
    assert report['minimum_eigenvalue_of_hermitian_part'] == pytest.approx(-.03)
    np.testing.assert_array_equal(saved, matrix)


def test_global_observable_probe_is_fixed_parent_and_saves_each_task(tmp_path):
    pytest.importorskip('tenpy')
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    psi=MPS.from_singlets(SpinHalfSite(conserve='Sz'),4,[(0,1),(2,3)],bc='finite')
    checkpoint=tmp_path/'parent'/'checkpoint.h5';_write_checkpoint(checkpoint,psi)
    rows=[dict(name='spin',sites=[0],observable=dict(kind='spin_correlations',rows=2,columns=2)),
          dict(name='dimers',sites=[0],observable=dict(kind='dimer_correlations',bonds=[[0,1],[2,3]])),
          dict(name='variance',sites=[0],observable=dict(kind='hamiltonian_variance',physical_bonds=[[0,1,1.],[2,3,1.]]))]
    config=_config(tmp_path,checkpoint,n=4,chi=2,dtype='complex128',regions=rows)
    result=collect_regions(config)
    assert result['status']=='complete' and result['new_H_calls']==2 and result['new_ED']==0
    assert all(r['mps_unchanged'] and r['payload_kind']=='fixed_mps_observables' for r in result['regions'])
    assert result['regions'][0]['diagnostics']['spin_squared']==pytest.approx(0,abs=3e-14)
    assert result['regions'][2]['diagnostics']['energy_per_site']==pytest.approx(-.375,abs=3e-14)
    assert result['regions'][2]['diagnostics']['variance_per_site']==pytest.approx(0,abs=3e-14)
    with np.load(tmp_path/'readout/regions/dimers/rdm.npz') as z:
        np.testing.assert_allclose(z['bond_means'],[-.75,-.75],atol=3e-14,rtol=0)
        np.testing.assert_allclose(z['connected'],0,atol=3e-14,rtol=0)
    assert collect_regions(config)['regions']==result['regions']


def test_future_parent_materialization_never_accepts_a_live_incomplete_generation(tmp_path):
    parent=tmp_path/'archive';parent.mkdir()
    checkpoint=parent/'checkpoint.h5';checkpoint.write_bytes(b'trusted-test-checkpoint')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    physical=dict(chi=8192,rows=6,columns=6,j2=.125)
    (parent/'registration.json').write_text(json.dumps(dict(config=physical)))
    (parent/'archive.json').write_text(json.dumps(dict(checkpoint_sha256=digest)))
    results=dict(status='paused',global_completed_sweeps=27,checkpoint_sha256=digest)
    (parent/'results.json').write_text(json.dumps(results))
    spec=dict(completed_parent=dict(directory=str(parent),sweep=28,**physical),
              materialized_config=str(tmp_path/'materialized.json'),n_sites=36)
    path=tmp_path/'plan.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='incomplete'):materialize_completed_parent(path)
    results.update(status='complete',global_completed_sweeps=28)
    (parent/'results.json').write_text(json.dumps(results))
    target=materialize_completed_parent(path)
    assert json.loads(target.read_text())['checkpoint_sha256']==digest
    assert materialize_completed_parent(path)==target
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='archive/checkpoint'):materialize_completed_parent(path)


def test_standalone_import_needs_no_jax_or_tenpy_and_child_wallclock_is_enforced(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = ('import sys; import projects.clebsch_tree_nqs.workflows.dmrg_region_probe; '
            'assert "jax" not in sys.modules and "tenpy" not in sys.modules')
    subprocess.run([sys.executable, '-c', code], cwd=root, check=True)
    env = os.environ.copy()
    report = bounded_subprocess([sys.executable, '-c', 'import time; time.sleep(30)'],
                                wall_seconds=.08, memory_bytes=2*1024**3,
                                log_path=tmp_path/'timeout.log', env=env)
    assert report['supervisor_stop_reason'] == 'wallclock_limit'
    assert report['exit_code'] != 0 and report['supervisor_seconds'] < 2.


def test_worker_fails_closed_when_darwin_has_neither_address_limit_nor_rss_monitor(tmp_path, monkeypatch):
    import projects.clebsch_tree_nqs.workflows.dmrg_region_probe as probe
    request = tmp_path/'request.json'
    request.write_text(json.dumps(dict(region=dict(limits=dict(memory_gib=2.)))))
    monkeypatch.setattr(probe.sys, 'platform', 'darwin')
    monkeypatch.setattr(probe, '_rss_bytes', lambda pid: 0)
    with pytest.raises(RuntimeError, match='no unbounded fallback'):
        probe._worker(request)
    assert not list(tmp_path.glob('*.npz'))
    assert not (tmp_path/'readout.json').exists()


def test_random_complex_mps_matches_dense_oracle_in_requested_six_seven_site_order_and_resumes(tmp_path):
    pytest.importorskip('tenpy')
    import tenpy.linalg.np_conserved as npc
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    from su2_nqs.benchmarks.tensor_network_readout import tenpy_region_density

    n = 8
    rng = np.random.default_rng(2026100907)
    vector = rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    vector /= np.linalg.norm(vector)
    site = SpinHalfSite(conserve=None)
    perm = np.argsort(np.diag(site.Sz.to_ndarray()).real)
    bits = ((np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1)
    native = perm[bits]@(2**np.arange(n-1, -1, -1))
    native_vector = np.zeros_like(vector); native_vector[native] = vector
    theta = npc.Array.from_ndarray_trivial(native_vector.reshape([2]*n), labels=[f'p{i}' for i in range(n)])
    psi = MPS.from_full([site]*n, theta, cutoff=1e-15)
    checkpoint = tmp_path/'parent'/'checkpoint.h5'
    _write_checkpoint(checkpoint, psi)
    parent_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    six = [6, 0, 4, 2, 5, 1]
    seven = six+[3]
    config = _config(tmp_path, checkpoint, n=n, chi=max(psi.chi), dtype='complex128',
                     regions=[dict(name='six', sites=six), dict(name='seven', sites=seven)])
    result = collect_regions(config)
    assert result['status'] == 'complete'
    assert result['new_optimization'] == result['new_H_calls'] == result['new_ED'] == 0
    for sites, row in zip((six, seven), result['regions'], strict=True):
        name = row['region']['name']
        with np.load(tmp_path/'readout'/'regions'/name/'rdm.npz') as archive:
            actual = archive['rho']
            np.testing.assert_array_equal(archive['sites'], sites)
        np.testing.assert_allclose(actual, _rho_exact(vector, sites, n), atol=3e-14, rtol=0)
        np.testing.assert_allclose(actual, tenpy_region_density(psi, sites), atol=3e-14, rtol=0)
        assert row['mps_unchanged'] and row['no_jax_import'] and row['raw_rdm_unmodified']
        assert row['diagnostics']['minimum_eigenvalue'] > -3e-14
        assert not row['diagnostics']['strict_su2_eligible']
        assert not row['diagnostics']['strict_entanglement_spectrum_reported']
        assert row['worker_peak_rss_bytes'] < 2*1024**3
    # The center is last in the requested order: trace it without reordering.
    rho7 = _rho_exact(vector, seven, n).reshape(64, 2, 64, 2)
    np.testing.assert_allclose(np.trace(rho7, axis1=1, axis2=3), _rho_exact(vector, six, n), atol=3e-14, rtol=0)
    files = [tmp_path/'readout'/'regions'/name/leaf for name in ('six', 'seven') for leaf in ('readout.json', 'rdm.npz')]
    prior = {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    resumed = collect_regions(config)
    assert resumed['regions'] == result['regions']
    assert all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in prior.items())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == parent_hash
    # A damaged completed file is rejected, never silently recomputed.
    files[1].write_bytes(files[1].read_bytes()+b'bad')
    with pytest.raises(ValueError, match='completed region seal mismatch'):
        collect_regions(config)


def test_charge_sorted_singlet_mps_has_legal_raw_weights_without_reported_spectrum(tmp_path):
    pytest.importorskip('tenpy')
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    site = SpinHalfSite(conserve='Sz')
    psi = MPS.from_singlets(site, 8, [(0, 1), (2, 3), (4, 5), (6, 7)], bc='finite')
    checkpoint = tmp_path/'parent'/'checkpoint.h5'
    _write_checkpoint(checkpoint, psi)
    config = _config(tmp_path, checkpoint, n=8, chi=max(psi.chi), dtype='complex128',
                     regions=[dict(name='singlet6', sites=[6, 0, 1, 4, 5, 7])])
    result = collect_regions(config)
    row = result['regions'][0]
    singlet = np.asarray([0., -1., 1., 0.])/np.sqrt(2)
    vector = singlet
    for _ in range(3):
        vector = np.kron(vector, singlet)
    with np.load(tmp_path/'readout'/'regions'/'singlet6'/'rdm.npz') as archive:
        np.testing.assert_allclose(archive['rho'], _rho_exact(vector, [6, 0, 1, 4, 5, 7], 8), atol=3e-14, rtol=0)
    assert row['diagnostics']['strict_su2_eligible']
    assert row['diagnostics']['p_q']['0'] == pytest.approx(1., abs=3e-14)
    assert not row['diagnostics']['strict_entanglement_spectrum_reported']
    assert 'invariant_readout' not in row['diagnostics']


def test_actual_hexagon_order_defaults_are_small_fixed_parent_probes_and_seven_reduces_to_six(tmp_path):
    pytest.importorskip('tenpy')
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    site = SpinHalfSite(conserve='Sz')
    pattern = ['up' if i%2 else 'down' for i in range(36)]
    psi = MPS.from_product_state([site]*36, pattern, bc='finite')
    checkpoint = tmp_path/'parent'/'checkpoint.h5'
    _write_checkpoint(checkpoint, psi)
    config = _config(tmp_path, checkpoint, n=36, chi=1, dtype='float64')
    result = collect_regions(config)
    assert result['status'] == 'complete'
    for name, sites in TRIANGULAR_REGIONS:
        with np.load(tmp_path/'readout'/'regions'/name/'rdm.npz') as archive:
            rho = archive['rho']
            index = sum((i%2)*2**(len(sites)-1-j) for j, i in enumerate(sites))
            expected = np.zeros_like(rho); expected[index, index] = 1.
            np.testing.assert_array_equal(rho, expected)
    with np.load(tmp_path/'readout'/'regions'/'local_hexagon6'/'rdm.npz') as a, np.load(tmp_path/'readout'/'regions'/'local_hexagon7'/'rdm.npz') as b:
        np.testing.assert_array_equal(np.trace(b['rho'].reshape(64, 2, 64, 2), axis1=1, axis2=3), a['rho'])


def test_dense_preflight_refusal_and_worker_memory_failure_preserve_parent_and_partial_results(tmp_path):
    pytest.importorskip('tenpy')
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    psi = MPS.from_product_state([SpinHalfSite(conserve=None)]*8, ['up']*8, bc='finite')
    checkpoint = tmp_path/'parent'/'checkpoint.h5'
    _write_checkpoint(checkpoint, psi)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = _config(tmp_path, checkpoint, n=8, chi=8192, dtype='complex128',
                     regions=[dict(name='small', sites=[0], dense_estimate_override=True),
                              dict(name='too_big', sites=[0, 1, 2, 4, 5, 6])])
    result = collect_regions(config)
    assert result['status'] == 'incomplete'
    assert result['regions'][0]['status'] == 'complete'
    assert result['regions'][1]['status'] == 'preflight_refused'
    assert not (tmp_path/'readout'/'regions'/'too_big'/'readout.json').exists()
    first_hash = hashlib.sha256((tmp_path/'readout'/'regions'/'small'/'rdm.npz').read_bytes()).hexdigest()
    again = collect_regions(config)
    assert again['status'] == 'incomplete'
    assert hashlib.sha256((tmp_path/'readout'/'regions'/'small'/'rdm.npz').read_bytes()).hexdigest() == first_hash
    # Bypass only the estimate to test the actual address-space guard; no
    # successful RDM may be published by a worker with a 32MiB address cap.
    spec = json.loads(config.read_text())
    spec['output'] = str(tmp_path/'memory_failure')
    spec['regions'] = [dict(name='bounded', sites=[0], dense_estimate_override=True,
                            limits=dict(memory_gib=1/32, wall_seconds=10.))]
    config2 = tmp_path/'memory_config.json'; config2.write_text(json.dumps(spec))
    failed = collect_regions(config2)
    assert failed['status'] == 'incomplete' and failed['regions'][0]['status'] == 'failed'
    assert not (tmp_path/'memory_failure'/'regions'/'bounded'/'readout.json').exists()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    spec['checkpoint_sha256'] = '0'*64
    config2.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='input checkpoint SHA'):
        collect_regions(config2)
