"""Archive full DMRG sweep boundaries without loading or changing an MPS.

Readers hold the source file descriptor across the copy: the producer's atomic
rename cannot mix two checkpoint generations. Temporary/incomplete HDFs are
never inputs. Archives are immutable; only the stage's ``latest`` link changes.
This utility is independent of all frozen training/DMRG runtime sources.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import BinaryIO


def file_sha256(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def copy_open_checkpoint(source: BinaryIO, target: Path) -> str:
    """Copy one already-open generation, reject in-place mutation, and fsync."""
    before = os.fstat(source.fileno())
    if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
        raise ValueError('checkpoint must be a nonempty regular file')
    source.seek(0)
    digest = hashlib.sha256()
    with target.open('wb') as output:
        while block := source.read(4 * 1024**2):
            output.write(block)
            digest.update(block)
        output.flush()
        os.fsync(output.fileno())
    after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('source was mutated in-place, not atomically replaced')
    if target.stat().st_size != before.st_size or file_sha256(target) != digest.hexdigest():
        raise RuntimeError('checkpoint copy failed its byte/hash verification')
    return digest.hexdigest()


def _relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('archive names must be nonempty relative paths')
    return path


def _save_bytes(target: Path, data: bytes) -> str:
    """Commit an immutable small file, or verify an existing identical copy."""
    digest = hashlib.sha256(data).hexdigest()
    if target.exists():
        if file_sha256(target) != digest:
            raise FileExistsError(f'archived file differs: {target}')
        return digest
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    return digest


def _publish_latest(destination: Path, snapshot: Path, receipt: dict) -> None:
    """Advance only a same-protocol discovery link, never an archived file."""
    latest = destination/'latest'
    if latest.exists():
        prior = json.loads((latest/'archive.json').read_text())
        if prior['protocol_sha256'] != receipt['protocol_sha256']:
            raise ValueError('do not mix scientific protocols in one archive stage')
        if prior['global_sweep'] > receipt['global_sweep']:
            return
    if latest.exists() and not latest.is_symlink():
        raise FileExistsError('latest must be an archive-managed symbolic link')
    link = destination/f'.latest-{os.getpid()}'
    link.symlink_to(snapshot.relative_to(destination), target_is_directory=True)
    link.replace(latest)


def archive_stage(source: Path, destination: Path) -> dict:
    """Archive a completed/current full boundary plus sealed small provenance.

    ``destination`` is a stage directory in durable WORK storage, NOT the
    live output. Optional h5py reads scalar metadata only; no TeNPy state load,
    contractions, sampling or optimization. ``latest/checkpoint.h5`` points
    to the most recent archived boundary. Older boundaries remain untouched.
    """
    import h5py

    source, destination = source.resolve(), destination.resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('archive destination must be outside the live source')
    registration_bytes = (source/'registration.json').read_bytes()
    registration = json.loads(registration_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.archive-', suffix='.h5', dir=destination)
    os.close(fd)
    temporary = Path(name)
    try:
        with (source/'checkpoint.h5').open('rb') as handle:
            digest = copy_open_checkpoint(handle, temporary)
        with h5py.File(temporary, 'r') as handle:
            if int(handle['schema'][()]) != 1 or 'psi' not in handle:
                raise ValueError('not a full schema1 DMRG checkpoint')
            sweep = int(handle['resume_data/sweeps'][()])
            protocol = handle['metadata/protocol_sha256'][()]
            if isinstance(protocol, bytes):
                protocol = protocol.decode()
            if protocol != registration['protocol_sha256']:
                raise ValueError('checkpoint and registration protocol differ')

        files = {'registration.json': registration_bytes}
        mode, energy = 'boundary', None
        for label in ('results.json', 'chunk_status.json', 'progress.json', 'transition.json'):
            path = source/label
            if not path.exists():
                continue
            data = path.read_bytes()
            record = json.loads(data)
            if label == 'transition.json' or record.get('checkpoint_sha256') == digest:
                files[label] = data
                if label == 'results.json':
                    if record['registration'] != registration or record['status'] != 'complete':
                        raise ValueError('completed result registration/status mismatch')
                    mode, energy = 'complete', record['energy_per_site']
                    raw = (source/'rdms.npz').read_bytes()
                    if hashlib.sha256(raw).hexdigest() != record['rdms_sha256']:
                        raise ValueError('final raw RDM seal mismatch')
                    files['rdms.npz'] = raw
                elif label == 'chunk_status.json' and mode != 'complete':
                    mode = 'paused' if record['status'] == 'paused' else 'boundary'
                elif label == 'progress.json' and energy is None:
                    energy = record['energy_per_site']
        for path in sorted((source/'sweep_readouts').glob('sweep*.json')):
            data = path.read_bytes()
            record = json.loads(data)
            if record['global_sweep'] > sweep:
                continue
            raw = path.with_suffix('.npz').read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['rdms_sha256']:
                raise ValueError('completed-sweep raw RDM seal mismatch')
            files[f'sweep_readouts/{path.name}'] = data
            files[f'sweep_readouts/{path.with_suffix(".npz").name}'] = raw
        for relative, expected in registration['sources'].items():
            path = _relative(relative)
            data = (source/'source_snapshots'/path).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('frozen runtime source seal mismatch')
            files[str(Path('source_snapshots')/path)] = data

        snapshot = destination/'snapshots'/f'sweep{sweep:04d}_{digest[:12]}_{mode}'
        snapshot.mkdir(parents=True, exist_ok=True)
        receipt_path = snapshot/'archive.json'
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            if prior['checkpoint_sha256'] != digest or prior['protocol_sha256'] != protocol:
                raise ValueError('existing archive receipt differs from captured boundary')
            for relative, expected in prior['files_sha256'].items():
                if file_sha256(snapshot/_relative(relative)) != expected:
                    raise ValueError('existing immutable archive seal mismatch')
            _publish_latest(destination, snapshot, prior)
            return prior
        checkpoint = snapshot/'checkpoint.h5'
        if checkpoint.exists():
            if file_sha256(checkpoint) != digest:
                raise FileExistsError('existing archived checkpoint differs')
        else:
            temporary.replace(checkpoint)
        hashes = {'checkpoint.h5': digest}
        for relative, data in files.items():
            hashes[relative] = _save_bytes(snapshot/_relative(relative), data)
        receipt = dict(source=str(source), destination=str(destination),
            snapshot=str(snapshot), checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_sha256=digest, global_sweep=sweep, source_status=mode,
            energy_per_site=energy, protocol_sha256=protocol, files_sha256=hashes,
            source_modified=False, full_optimizer_checkpoint=True,
            no_temporary_hdf_copied=True, new_training=0, new_H_calls=0)
        _save_bytes(receipt_path, json.dumps(receipt, indent=2).encode()+b'\n')
        _publish_latest(destination, snapshot, receipt)
        return receipt
    finally:
        if temporary.exists():
            temporary.unlink()  # Only our private uncommitted copy, never a source.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--stage', action='append', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for stage in args.stage:
        record = archive_stage(args.source_root/_relative(stage),
                               args.archive_root/_relative(stage))
        records.append(record)
        print(json.dumps({key: record[key] for key in (
            'snapshot', 'checkpoint_bytes', 'global_sweep', 'source_status',
            'checkpoint_sha256')}), flush=True)
    report = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
        archives=records, bytes_in_this_receipt=sum(r['checkpoint_bytes'] for r in records),
        purpose='Durable full-checkpoint backup; no optimization or source mutation.')
    _save_bytes(args.receipt, json.dumps(report, indent=2).encode()+b'\n')
    print(json.dumps(dict(archived_stages=len(records), receipt=str(args.receipt),
        checkpoint_bytes=report['bytes_in_this_receipt'])), flush=True)


if __name__ == '__main__':
    main()
