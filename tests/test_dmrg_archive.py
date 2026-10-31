"""Storage tests only: no new quantum states, optimization, or exact diagonalization."""
import hashlib
import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows.dmrg_archive import archive_stage, copy_open_checkpoint


def _stage(path: Path, sweep: int = 20) -> str:
    h5py = pytest.importorskip('h5py')
    path.mkdir(parents=True, exist_ok=True)
    with h5py.File(path/'checkpoint.h5', 'w') as f:
        f['schema'] = 1
        f.create_group('psi')
        f['resume_data/sweeps'] = sweep
        f['metadata/protocol_sha256'] = 'test-protocol'
    registration = {'protocol_sha256': 'test-protocol', 'sources': {}}
    (path/'registration.json').write_text(json.dumps(registration))
    digest = hashlib.sha256((path/'checkpoint.h5').read_bytes()).hexdigest()
    (path/'progress.json').write_text(json.dumps(dict(checkpoint_sha256=digest,
        sweeps=sweep, energy_per_site=-.51)))
    (path/'checkpoint.tmp.h5').write_bytes(b'INCOMPLETE: MUST NOT ARCHIVE')
    return digest


def test_open_generation_survives_atomic_producer_replacement(tmp_path):
    source, replacement, target = [tmp_path/n for n in ('source', 'replacement', 'copy')]
    source.write_bytes(b'old-complete-boundary')
    replacement.write_bytes(b'new-complete-boundary')
    with source.open('rb') as handle:
        replacement.replace(source)
        digest = copy_open_checkpoint(handle, target)
    assert target.read_bytes() == b'old-complete-boundary'
    assert source.read_bytes() == b'new-complete-boundary'
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()


def test_boundary_snapshot_is_full_immutable_and_does_not_copy_temporary(tmp_path):
    source, destination = tmp_path/'live', tmp_path/'archive'
    digest = _stage(source)
    before = (source/'checkpoint.h5').read_bytes()
    report = archive_stage(source, destination)
    assert report['checkpoint_sha256'] == digest and report['global_sweep'] == 20
    assert report['source_status'] == 'boundary'
    assert (destination/'latest/checkpoint.h5').read_bytes() == before
    assert not list(destination.rglob('checkpoint.tmp.h5'))
    assert (source/'checkpoint.h5').read_bytes() == before
    first = (destination/'latest').resolve()
    sealed_receipt = (first/'archive.json').read_bytes()
    assert archive_stage(source, destination) == report
    assert (first/'archive.json').read_bytes() == sealed_receipt
    _stage(source, sweep=21)
    newer = archive_stage(source, destination)
    assert newer['global_sweep'] == 21 and first.exists()
    assert (first/'checkpoint.h5').read_bytes() == before
    assert (destination/'latest').resolve() != first


def test_completed_result_and_raw_rdm_seals_are_required(tmp_path):
    source, destination = tmp_path/'live', tmp_path/'archive'
    digest = _stage(source)
    raw = b'raw coherent RDM'
    (source/'rdms.npz').write_bytes(raw)
    result = dict(status='complete', checkpoint_sha256=digest,
        registration=json.loads((source/'registration.json').read_text()),
        energy_per_site=-.514, rdms_sha256=hashlib.sha256(raw).hexdigest())
    (source/'results.json').write_text(json.dumps(result))
    report = archive_stage(source, destination)
    assert report['source_status'] == 'complete'
    assert (destination/'latest/results.json').read_bytes() == (source/'results.json').read_bytes()
    (source/'rdms.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='RDM seal'):
        archive_stage(source, tmp_path/'bad-archive')


def test_rejects_protocol_mismatch_and_live_subdirectory(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'live'
    _stage(source)
    with pytest.raises(ValueError, match='outside'):
        archive_stage(source, source/'archive')
    with h5py.File(source/'checkpoint.h5', 'a') as f:
        del f['metadata/protocol_sha256']
        f['metadata/protocol_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='protocol'):
        archive_stage(source, tmp_path/'archive')


def test_older_boundary_does_not_replace_newer_latest(tmp_path):
    source, destination = tmp_path/'live', tmp_path/'archive'
    _stage(source, sweep=21)
    archive_stage(source, destination)
    newest = (destination/'latest').resolve()
    _stage(source, sweep=20)
    older = archive_stage(source, destination)
    assert older['global_sweep'] == 20
    assert (destination/'latest').resolve() == newest


def test_frozen_runtime_hash_mismatch_cannot_publish_an_archive(tmp_path):
    source, destination = tmp_path/'live', tmp_path/'archive'
    _stage(source)
    registration = json.loads((source/'registration.json').read_text())
    registration['sources'] = {'module.py': hashlib.sha256(b'expected').hexdigest()}
    (source/'registration.json').write_text(json.dumps(registration))
    snapshot = source/'source_snapshots/module.py'
    snapshot.parent.mkdir()
    snapshot.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source seal'):
        archive_stage(source, destination)
    assert not (destination/'latest').exists()
    assert not list(destination.glob('.archive-*.h5'))
