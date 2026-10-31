"""Keep DMRG state I/O in scratch and verified, retained states in WORK.

Independent of the frozen DMRG optimizer. Existing scientific registrations
are not rewritten: completed HOME bundles become compatibility symlinks.
Fresh continuation configs stage a sealed WORK input in scratch and use
absolute scratch paths. This module never submits jobs or loads an MPS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

from .dmrg_archive import copy_open_checkpoint, file_sha256, _relative, _save_bytes


def tree_manifest(directory: Path) -> dict[str, dict]:
    """Inventory regular files, rejecting links and special files."""
    manifest = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'nested symlink not allowed in a migration: {path}')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f'not a regular file: {path}')
        manifest[str(path.relative_to(directory))] = dict(
            bytes=path.stat().st_size, sha256=file_sha256(path))
    if not manifest:
        raise ValueError('cannot migrate an empty directory')
    return manifest


def verify_tree(directory: Path, manifest: dict[str, dict]) -> None:
    """Verify every original file before publishing links or reclaiming HOME."""
    for name, expected in manifest.items():
        path = directory/_relative(name)
        if (not path.is_file() or path.stat().st_size != expected['bytes']
                or file_sha256(path) != expected['sha256']):
            raise ValueError(f'file size/hash mismatch: {path}')


def _copy_files(source: Path, target: Path, manifest: dict[str, dict],
                reuse: dict[str, Path] | None = None) -> None:
    """Bounded, fsynced copies; identical completed HDFs may share an inode."""
    known = dict(reuse or {})
    for name, expected in manifest.items():
        src, dst = source/_relative(name), target/_relative(name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        prior = known.get(expected['sha256']) if src.suffix == '.h5' else None
        linked = False
        if prior is not None:
            if file_sha256(prior) != expected['sha256']:
                raise ValueError('reused durable checkpoint changed')
            try:
                os.link(prior, dst)  # Same WORK filesystem; otherwise copy below.
                linked = True
            except OSError:
                pass
        if not linked:
            with src.open('rb') as handle:
                digest = copy_open_checkpoint(handle, dst) if expected['bytes'] else None
            if expected['bytes'] == 0:
                with dst.open('xb') as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
            elif digest != expected['sha256']:
                raise ValueError('source changed during storage copy')
            shutil.copystat(src, dst)
        if src.suffix == '.h5':
            known[expected['sha256']] = dst


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def relocate_completed(source: Path, destination: Path, *,
                       reuse_checkpoints: tuple[Path, ...] = (),
                       legacy_probe: bool = False) -> dict:
    """Verify a retained WORK bundle, replace the old HOME data with a link.

    The exact source directory must be a completed, non-writing stage. A
    separately authorized old MPS-only probe can use ``legacy_probe=True``.
    All original filenames/bytes and scientific seals survive. Duplicate
    final/checkpoint HDFs share WORK storage rather than doubling its usage.
    """
    source = source.absolute()
    destination = destination.absolute()
    if source.is_symlink():
        if source.resolve() != destination.resolve():
            raise ValueError('existing source link has another target')
        record = json.loads((destination/'.storage-migration.json').read_text())
        verify_tree(destination, record['files'])
        return record
    if not source.is_dir() or destination.exists():
        raise ValueError('source must exist and destination must be fresh')
    if (destination.resolve().is_relative_to(source.resolve())
            or source.resolve().is_relative_to(destination.resolve())):
        raise ValueError('migration roots must be disjoint')
    manifest = tree_manifest(source)
    if not any(Path(name).suffix == '.h5' for name in manifest):
        raise ValueError('not a physical-state bundle')
    if (source/'checkpoint.h5').exists():
        result = json.loads((source/'results.json').read_text())
        registration = json.loads((source/'registration.json').read_text())
        if (result['status'] != 'complete' or result['registration'] != registration
                or result['checkpoint_sha256'] != manifest['checkpoint.h5']['sha256']):
            raise ValueError('only sealed, completed stages may leave HOME')
    elif not legacy_probe:
        raise ValueError('MPS-only legacy probe needs explicit authorization')
    reuse = {file_sha256(path): path for path in reuse_checkpoints}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.storage-copy-', dir=destination.parent))
    # On failure keep the uncommitted copy for inspection; never delete source.
    _copy_files(source, temporary, manifest, reuse)
    verify_tree(temporary, manifest)
    if tree_manifest(source) != manifest:
        raise ValueError('source changed; do not relocate a live writer')
    record = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source), destination=str(destination), files=manifest,
        bytes_removed_from_home=sum(item['bytes'] for item in manifest.values()),
        scientific_files_modified=False, new_training=0, new_H_calls=0,
        old_path_is_compatibility_symlink=True)
    _save_bytes(temporary/'.storage-migration.json', json.dumps(record, indent=2).encode()+b'\n')
    _sync_directory(temporary)
    temporary.rename(destination)
    _sync_directory(destination.parent)
    # Completed parents have no active writers. Keep a recoverable copy until
    # publication succeeds, then reclaim only the exact verified file inventory.
    backup = source.with_name(source.name+'.verified-home-migration')
    link = source.with_name(source.name+'.work-link')
    if backup.exists() or link.exists() or link.is_symlink():
        raise FileExistsError('migration recovery paths already exist')
    link.symlink_to(destination, target_is_directory=True)
    source.rename(backup)
    try:
        link.replace(source)
    except BaseException:
        backup.rename(source)
        link.unlink()
        raise
    _sync_directory(source.parent)
    verify_tree(backup, manifest)
    for name in manifest:
        (backup/_relative(name)).unlink()
    for path in sorted(backup.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        path.rmdir()
    backup.rmdir()
    return record


def prepare_scratch_config(template: Path, parent_snapshot: Path,
                           scratch_root: Path, stage: str, config_output: Path) -> dict:
    """Prepare, NOT run, a new continuation with staged scratch input/output.

    Use an already authorized fresh-stage template, not a registered/resumable
    config. Past config files are immutable. The parent must be a complete
    WORK archive; the current scratch boundary alone is not a completed parent.
    """
    if config_output.exists() or config_output.resolve() == template.resolve():
        raise FileExistsError('write a fresh config; never rewrite a past seal')
    if not scratch_root.is_absolute() or not parent_snapshot.is_absolute():
        raise ValueError('data roots must be absolute')
    parent_snapshot = parent_snapshot.resolve()  # Pin a mutable `latest` link once.
    scratch_root = scratch_root.resolve()
    archive = json.loads((parent_snapshot/'archive.json').read_text())
    if archive['source_status'] != 'complete':
        raise ValueError('parent archive is not a completed stage')
    manifest = {name: dict(bytes=(parent_snapshot/_relative(name)).stat().st_size,
        sha256=digest) for name, digest in archive['files_sha256'].items()}
    verify_tree(parent_snapshot, manifest)
    result = json.loads((parent_snapshot/'results.json').read_text())
    registration = json.loads((parent_snapshot/'registration.json').read_text())
    if (result['status'] != 'complete' or result['registration'] != registration
            or result['checkpoint_sha256'] != archive['checkpoint_sha256']):
        raise ValueError('parent completion seal mismatch')
    spec = json.loads(template.read_text())
    for key, expected in (('initial_checkpoint_sha256', archive['checkpoint_sha256']),
                          ('parent_protocol_sha256', archive['protocol_sha256'])):
        if key in spec and spec[key] != expected:
            raise ValueError(f'new template declares a different {key}')
    output = scratch_root/_relative(stage)
    if output.resolve().is_relative_to(scratch_root/'inputs'):
        raise ValueError('output must not overwrite staged immutable inputs')
    if output.exists() or output.is_symlink():
        raise FileExistsError('new scratch output must be fresh')
    inputs = scratch_root/'inputs'/archive['checkpoint_sha256']
    inputs.parent.mkdir(parents=True, exist_ok=True)
    if inputs.exists():
        verify_tree(inputs, manifest)
    else:
        temporary = Path(tempfile.mkdtemp(prefix='.stage-input-', dir=inputs.parent))
        _copy_files(parent_snapshot, temporary, manifest)
        verify_tree(temporary, manifest)
        _sync_directory(temporary)
        temporary.rename(inputs)
        _sync_directory(inputs.parent)
    spec.update(output=str(output), parent_output=str(inputs),
        initial_checkpoint=str(inputs/'checkpoint.h5'),
        initial_checkpoint_sha256=archive['checkpoint_sha256'],
        parent_protocol_sha256=archive['protocol_sha256'])
    _save_bytes(config_output, json.dumps(spec, indent=2).encode()+b'\n')
    return dict(config=str(config_output), scratch_input=str(inputs),
                scratch_output=str(output), work_parent=str(parent_snapshot),
                config_sha256=file_sha256(config_output), submitted=False)


def validate_runtime_paths(config: Path, scratch_root: Path) -> dict:
    """Reject HOME state I/O at a future launch, without changing the config."""
    spec = json.loads(config.read_text())
    root = scratch_root.resolve()
    for key in ('output', 'parent_output', 'initial_checkpoint'):
        path = Path(spec[key])
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f'{key} must be an absolute scratch path')
    if Path(spec['initial_checkpoint']).resolve() != (
            Path(spec['parent_output'])/'checkpoint.h5').resolve():
        raise ValueError('scratch parent/checkpoint disagree')
    return dict(config=str(config), config_sha256=file_sha256(config),
                physical_state_io_root=str(root), passed=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    move = commands.add_parser('relocate')
    move.add_argument('--source', type=Path, required=True)
    move.add_argument('--destination', type=Path, required=True)
    move.add_argument('--reuse-checkpoint', type=Path, action='append', default=[])
    move.add_argument('--legacy-probe', action='store_true')
    move.add_argument('--receipt', type=Path, required=True)
    stage = commands.add_parser('prepare')
    stage.add_argument('--template', type=Path, required=True)
    stage.add_argument('--parent-snapshot', type=Path, required=True)
    stage.add_argument('--scratch-root', type=Path, required=True)
    stage.add_argument('--stage', required=True)
    stage.add_argument('--config-output', type=Path, required=True)
    check = commands.add_parser('check')
    check.add_argument('--config', type=Path, required=True)
    check.add_argument('--scratch-root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'relocate':
        record = relocate_completed(args.source, args.destination,
            reuse_checkpoints=tuple(args.reuse_checkpoint), legacy_probe=args.legacy_probe)
        _save_bytes(args.receipt, json.dumps(record, indent=2).encode()+b'\n')
        print(json.dumps({key: record[key] for key in (
            'source', 'destination', 'bytes_removed_from_home')}))
    elif args.command == 'prepare':
        print(json.dumps(prepare_scratch_config(args.template, args.parent_snapshot,
            args.scratch_root, args.stage, args.config_output)))
    else:
        print(json.dumps(validate_runtime_paths(args.config, args.scratch_root)))


if __name__ == '__main__':
    main()
