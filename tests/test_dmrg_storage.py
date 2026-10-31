"""Exact byte/seal tests for storage routing, without DMRG or MPS loads."""
import hashlib
import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows.dmrg_archive import archive_stage
from projects.clebsch_tree_nqs.workflows.dmrg_storage import (
    prepare_scratch_config, relocate_completed, tree_manifest, validate_runtime_paths)


def _complete(path: Path) -> str:
    path.mkdir(parents=True)
    (path/'checkpoint.h5').write_bytes(b'sealed full MPS and optimizer')
    (path/'chi4096.h5').write_bytes((path/'checkpoint.h5').read_bytes())
    (path/'empty.log').touch()
    digest = hashlib.sha256((path/'checkpoint.h5').read_bytes()).hexdigest()
    registration = dict(protocol_sha256='physical-seal', config={'chi': 4096})
    (path/'registration.json').write_text(json.dumps(registration))
    (path/'results.json').write_text(json.dumps(dict(status='complete',
        registration=registration, checkpoint_sha256=digest)))
    return digest


def test_completed_home_stage_keeps_all_bytes_and_deduplicates_durable_hdf(tmp_path):
    source, destination = tmp_path/'home/stage', tmp_path/'work/stage'
    _complete(source)
    before = tree_manifest(source)
    archived = tmp_path/'work/snapshot/checkpoint.h5'
    archived.parent.mkdir(parents=True)
    archived.write_bytes((source/'checkpoint.h5').read_bytes())
    record = relocate_completed(source, destination, reuse_checkpoints=(archived,))
    assert source.is_symlink() and source.resolve() == destination
    assert tree_manifest(source) == before | {
        '.storage-migration.json': tree_manifest(source)['.storage-migration.json']}
    assert archived.stat().st_ino == (destination/'checkpoint.h5').stat().st_ino
    assert archived.stat().st_ino == (destination/'chi4096.h5').stat().st_ino
    assert record['bytes_removed_from_home'] == sum(v['bytes'] for v in before.values())
    assert not list(source.parent.glob('*.verified-home-migration'))
    assert relocate_completed(source, destination) == record


def test_uncompleted_or_changed_state_never_reclaimed(tmp_path):
    source = tmp_path/'home/stage'
    _complete(source)
    (source/'checkpoint.h5').write_bytes(b'changed')
    with pytest.raises(ValueError, match='sealed'):
        relocate_completed(source, tmp_path/'work/stage')
    assert not source.is_symlink() and (source/'chi4096.h5').exists()


def test_live_nested_link_and_overlapping_roots_are_rejected(tmp_path):
    source = tmp_path/'home/stage'
    _complete(source)
    with pytest.raises(ValueError, match='disjoint'):
        relocate_completed(source, source/'archive')
    (source/'live-data').symlink_to(tmp_path/'scratch')
    with pytest.raises(ValueError, match='nested symlink'):
        relocate_completed(source, tmp_path/'work/stage')


def test_legacy_probe_requires_explicit_authorization(tmp_path):
    source = tmp_path/'home/probe'
    source.mkdir(parents=True)
    (source/'chi1024.h5').write_bytes(b'old probe')
    with pytest.raises(ValueError, match='explicit authorization'):
        relocate_completed(source, tmp_path/'work/probe')
    relocate_completed(source, tmp_path/'work/probe', legacy_probe=True)
    assert (source/'chi1024.h5').read_bytes() == b'old probe'


def _archived_parent(tmp_path: Path) -> Path:
    h5py = pytest.importorskip('h5py')
    live = tmp_path/'live'
    live.mkdir()
    with h5py.File(live/'checkpoint.h5', 'w') as f:
        f['schema'] = 1
        f.create_group('psi')
        f['resume_data/sweeps'] = 20
        f['metadata/protocol_sha256'] = 'physical-seal'
    raw = b'raw unmodified RDM'
    (live/'rdms.npz').write_bytes(raw)
    registration = dict(protocol_sha256='physical-seal', sources={})
    (live/'registration.json').write_text(json.dumps(registration))
    (live/'results.json').write_text(json.dumps(dict(status='complete',
        registration=registration, energy_per_site=-.514,
        checkpoint_sha256=hashlib.sha256((live/'checkpoint.h5').read_bytes()).hexdigest(),
        rdms_sha256=hashlib.sha256(raw).hexdigest())))
    archive = archive_stage(live, tmp_path/'work/parent')
    return Path(archive['snapshot'])


def test_fresh_config_stages_verified_input_output_in_scratch_without_resealing(tmp_path):
    snapshot = _archived_parent(tmp_path)
    template, config = tmp_path/'home/template.json', tmp_path/'home/new.json'
    template.parent.mkdir()
    template.write_text(json.dumps(dict(output='old', chi=4096, sweeps=4,
        regions={'compact4': [0, 1, 2, 3]}, parent_config='old-frozen-config.json')))
    original = template.read_bytes()
    record = prepare_scratch_config(template, snapshot, tmp_path/'scratch',
        'projects/test/results/new-stage', config)
    spec = json.loads(config.read_text())
    assert template.read_bytes() == original
    assert spec['chi'] == 4096 and spec['sweeps'] == 4
    assert Path(spec['output']).is_relative_to(tmp_path/'scratch')
    assert Path(spec['initial_checkpoint']).parent == Path(spec['parent_output'])
    assert Path(spec['initial_checkpoint']).read_bytes() == (snapshot/'checkpoint.h5').read_bytes()
    assert spec['parent_config'] == 'old-frozen-config.json' and record['submitted'] is False
    with pytest.raises(FileExistsError, match='fresh config'):
        prepare_scratch_config(template, snapshot, tmp_path/'scratch', 'other', config)


def test_prepare_rejects_uncompleted_archive_and_predeclared_parent_mismatch(tmp_path):
    snapshot = _archived_parent(tmp_path)
    template = tmp_path/'template.json'
    template.write_text(json.dumps(dict(initial_checkpoint_sha256='another-parent')))
    with pytest.raises(ValueError, match='different'):
        prepare_scratch_config(template, snapshot, tmp_path/'scratch', 'new', tmp_path/'new.json')
    record = json.loads((snapshot/'archive.json').read_text())
    record['source_status'] = 'boundary'
    (snapshot/'archive.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='completed'):
        prepare_scratch_config(template, snapshot, tmp_path/'scratch', 'new', tmp_path/'new.json')


def test_launch_rejects_relative_home_and_symlink_escape(tmp_path):
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    config = tmp_path/'config.json'
    spec = dict(output=str(scratch/'output'), parent_output=str(scratch/'inputs'),
                initial_checkpoint=str(scratch/'inputs/checkpoint.h5'))
    config.write_text(json.dumps(spec))
    assert validate_runtime_paths(config, scratch)['passed']
    for bad in ('projects/test/results', str(tmp_path/'home/output')):
        config.write_text(json.dumps(spec | {'output': bad}))
        with pytest.raises(ValueError, match='absolute scratch'):
            validate_runtime_paths(config, scratch)
    (scratch/'escape').symlink_to(tmp_path/'home', target_is_directory=True)
    config.write_text(json.dumps(spec | {'output': str(scratch/'escape/output')}))
    with pytest.raises(ValueError, match='absolute scratch'):
        validate_runtime_paths(config, scratch)
