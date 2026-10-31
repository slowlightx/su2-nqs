"""Prepare an explicit code import and scan candidates or actual staged bytes.

No Git mutation or credential-file access is performed. Findings never contain
matched values, context snippets, or token hashes. Scientific artifacts and
credential/session trees are pruned before any file is opened.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'configs/git_import_20261031/generated'
MAX_BYTES = 10 * 1024**2
ROOT_FILES = frozenset({'.gitignore', 'AGENTS.md', 'README.md',
    'DIRECT_FUSION_LEGACY.md', 'pyproject.toml', 'environment-nqs.yml',
    'benchmark-requirements.txt', 'SQUARE_J1J2_RESULTS.ipynb',
    'TRIANGULAR_J1J2_RESULTS.ipynb'})
PREFIXES = ('su2_nqs/', 'tests/', 'fnqs/', 'peps-vmc/',
    'projects/clebsch_tree_nqs/', 'projects/hubbard_cg/', 'configs/')
TEXT_SUFFIXES = frozenset({'.py', '.md', '.json', '.toml', '.yml', '.yaml',
    '.sh', '.sbatch', '.txt', '.files'})
DENIED_PARTS = frozenset({'results', 'logs', 'archive', 'data', 'checkpoints',
    '__pycache__', '.pytest_cache', '.git', '.aws', '.ssh', '.codex', '.agents',
    '.config', '.venv', 'venv', 'node_modules', 'site-packages', 'sessions',
    'conversations', 'transcripts', 'build', 'dist'})
EXACT_KEYS = (
    '22bc977a758e8bae385c06d088a2a2f64852c4ed98fc00ca81402c0a57150ec8',
    '4dda5f1320adddae8d5b949c2ad62e2211fdb01ba8c385beef261003657fea22',
    '589afffc9a65f7f5566c23d30c4538981be87608da6ded9b4ec822b54b828626',
    '90ad32fb79fedc5ad1ae1747aa56c57a7861dbcbc7cd4511b23586b430294c29',
    'c43719b538f9cedbba58a7197583477c3b9155fa9b5b38d65258563b582ce708',
    'dd27eec56bb056dedd4c37eced96c6e20841c7f4d0bfcd7be6a5bdb2e5b532c4')
NUMERIC_FIXTURES = frozenset(
    ['projects/clebsch_tree_nqs/initializations/g1_operator_vit_h_ring.npz'] +
    ['projects/clebsch_tree_nqs/references/exact_cache/' + k + '.npz' for k in EXACT_KEYS])
FIGURES = frozenset(
    'figures/' + path + ext
    for path in ('square_j1j2/energy_methods_vs_size',
        'triangular_j1j2/energy_methods_vs_size',
        'triangular_j1j2/energy_methods_vs_size_zoom',
        'triangular_j1j2/published_spatial_vit_scaling',
        'triangular_j1j2/ct6_width20_to28_fixed_endpoint')
    for ext in ('.png', '.pdf'))
REQUIRED = frozenset({'projects/clebsch_tree_nqs/SU2NQS_all.md',
    'SQUARE_J1J2_RESULTS.ipynb', 'TRIANGULAR_J1J2_RESULTS.ipynb',
    'su2_nqs/vmc/__init__.py', 'su2_nqs/vmc/local_energy.py'})

# No generic entropy/hex rule: source seals, SHA256 and runtime package hashes
# are provenance, not secrets. The assignment rule requires a credential name.
PATTERNS = {
    'private_key_header': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'openai_style_key': re.compile(r'(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}'),
    'github_token': re.compile(r'(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
    'aws_access_key_id': re.compile(r'(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])'),
    'credential_assignment': re.compile(
        r'''(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd)["']?\s*[:=]\s*["']([^"'\s]{16,})["']'''),
    'credential_in_url': re.compile(r'(?i)https?://[^\s/:@]+:[^\s/@]{8,}@'),
}
PLACEHOLDER = re.compile(r'(?i)(?:example|placeholder|dummy|redacted|your[_-]|replace[_-]|\$\{|^\$|^<|^os\.|^process\.)')


def denied_reason(path: str) -> str | None:
    """Reject artifact, credential and conversation paths without opening them."""
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts:
        return 'outside_repository'
    if any(part in DENIED_PARTS or part.startswith('.conda') for part in p.parts):
        return 'artifact_environment_or_private_directory'
    if path.startswith('configs/git_import_20261031/generated/'):
        return 'generated_import_receipt'
    if path.startswith('projects/clebsch_tree_nqs/notebooks/'):
        return 'superseded_generated_notebook'
    name = p.name.lower()
    if (name.startswith(('.env', 'id_rsa', 'id_ed25519', 'credentials', 'secrets'))
            or name.endswith(('.pem', '.key', '.env'))):
        return 'credential_path_do_not_open'
    if any(s in name for s in ('conversation_history', 'chat_history', 'session_history')):
        return 'historical_conversation_do_not_open'
    if name == '.ds_store' or name.endswith(('.log', '.err', '.out', '.xml', '.lock')):
        return 'generated_output_or_lock'
    return None


def candidate_kind(path: str) -> str | None:
    """Classify an explicit authored-source or small reviewed binary exception."""
    if denied_reason(path):
        return None
    if path in NUMERIC_FIXTURES:
        return 'explicit_numeric_fixture'
    if path in FIGURES:
        return 'explicit_publication_figure'
    if path in ROOT_FILES or path == 'fnqs/.gitignore':
        return 'root_document_notebook_or_packaging'
    if path.startswith(PREFIXES) and PurePosixPath(path).suffix in TEXT_SUFFIXES:
        return 'authored_source_config_or_document'
    return None


def scan_text(path: str, text: str) -> list[dict[str, Any]]:
    """Return only path/line/rule metadata, never matched values or snippets."""
    findings = []
    for rule, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            if rule == 'credential_assignment' and PLACEHOLDER.search(match.group(1)):
                continue
            findings.append(dict(path=path, line=text.count('\n', 0, match.start())+1, rule=rule))
    return findings


def scan_bytes(path: str, data: bytes) -> tuple[list[dict], dict]:
    """Scan raw bytes and notebook string fields; validate numerical exceptions."""
    kind = candidate_kind(path)
    findings = scan_text(path, data.decode('utf-8', errors='replace'))
    details: dict[str, Any] = {}
    if kind == 'explicit_numeric_fixture':
        import numpy as np
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            arrays = {}
            for name in archive.files:
                value = archive[name]
                if value.dtype.hasobject or value.dtype.kind not in 'biufc':
                    raise ValueError('non-numeric fixture requires separate review: ' + path)
                arrays[name] = dict(dtype=str(value.dtype), shape=list(value.shape))
            details['numeric_arrays'] = arrays
    elif path.endswith('.ipynb'):
        notebook = json.loads(data)
        strings: list[str] = []
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values(): visit(child)
            elif isinstance(value, list):
                if value and all(isinstance(x, str) for x in value): strings.append(''.join(value))
                else:
                    for child in value: visit(child)
            elif isinstance(value, str): strings.append(value)
        visit(notebook)
        for index, value in enumerate(strings):
            for item in scan_text(path, value):
                item['notebook_string_field'] = index
                findings.append(item)
        details['notebook_cells'] = len(notebook.get('cells', []))
        details['source_output_metadata_strings_scanned'] = len(strings)
    return findings, details


def external_fixture_risk(path: str, text: str) -> bool:
    """Flag test source referencing external asset paths; not a runtime proof."""
    if not (path.startswith('tests/') or '/tests/' in path) or not path.endswith('.py'):
        return False
    try:
        strings = (node.value for node in ast.walk(ast.parse(text))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str))
        return any(re.search(r'(?:^|/)(?:results|archive|reference_wavefunctions)(?:/|$)', x)
                   for x in strings)
    except SyntaxError:
        return True


def write_json(path: Path, value: Any) -> None:
    """Write only generated audit outputs, never source or Git metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8')


def prepare() -> dict:
    """Hash/scan approved candidates, pruning denied trees before file reads."""
    entries = []; excluded = []; pruned = []; findings = []; risks = []
    for current, directories, files in os.walk(ROOT, followlinks=False):
        directories.sort(); files.sort()
        for name in list(directories):
            absolute = Path(current)/name; relative = absolute.relative_to(ROOT).as_posix()
            if absolute.is_symlink() or denied_reason(relative+'/'):
                directories.remove(name); pruned.append(relative)
        for name in files:
            absolute = Path(current)/name; relative = absolute.relative_to(ROOT).as_posix()
            kind = candidate_kind(relative)
            if absolute.is_symlink():
                excluded.append(dict(path=relative, reason='symlink_not_followed')); continue
            if kind is None:
                excluded.append(dict(path=relative, reason=denied_reason(relative) or 'not_in_source_allowlist')); continue
            size = absolute.stat().st_size
            if size > MAX_BYTES:
                excluded.append(dict(path=relative, reason='over_10MiB_requires_explicit_review', bytes=size)); continue
            data = absolute.read_bytes()
            items, details = scan_bytes(relative, data); findings.extend(items)
            entries.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), kind=kind, **details))
            if external_fixture_risk(relative, data.decode('utf-8', errors='replace')): risks.append(relative)
    entries.sort(key=lambda x:x['path']); paths = [x['path'] for x in entries]
    missing = sorted(REQUIRED-set(paths))
    manifest = dict(schema='su2_nqs_explicit_source_import_v1', generated_UTC=datetime.now(timezone.utc).isoformat(),
        root=str(ROOT), maximum_candidate_bytes=MAX_BYTES, file_count=len(entries),
        total_bytes=sum(x['bytes'] for x in entries), entries=entries,
        required_source_and_notebooks=sorted(REQUIRED), missing_required=missing,
        explicit_numeric_fixtures=sorted(NUMERIC_FIXTURES), explicit_publication_figures=sorted(FIGURES),
        generated_outputs_not_self_hashed=True, no_Git_mutation=True)
    scan = dict(mode='working_tree_candidates_not_staged_index', files_scanned=len(entries),
        total_bytes=manifest['total_bytes'], findings=findings, missing_required=missing,
        passed=not findings and not missing, rules=list(PATTERNS),
        no_generic_entropy_or_SHA256_heuristic=True, private_paths_pruned_before_open=True,
        no_secret_values_or_context_recorded=True,
        limitations=['not a credential-safety proof', 'archives/results/credentials not opened',
                    'actual staged index scan is separately required before commit'])
    write_json(OUTPUT/'source_manifest.json', manifest)
    write_json(OUTPUT/'candidate_secret_scan.json', scan)
    write_json(OUTPUT/'excluded_files.json', dict(pruned_directories=pruned, excluded_files=excluded))
    write_json(OUTPUT/'external_fixture_risks.json', dict(
        detection='test AST string constants referencing results/archive/reference_wavefunctions; non-exhaustive',
        count=len(risks), test_files=sorted(risks),
        no_test_weakened=True, all_external_scientific_assets_retained_in_place=True))
    (OUTPUT/'source_paths.nul').write_bytes(b''.join(p.encode()+b'\0' for p in paths))
    (OUTPUT/'source_paths.txt').write_text('\n'.join(paths)+'\n')
    result = dict(mode=scan['mode'], files=len(entries), bytes=manifest['total_bytes'],
        findings=len(findings), missing_required=missing, external_fixture_risk_tests=len(risks),
        by_kind=dict(Counter(x['kind'] for x in entries)), output=str(OUTPUT), passed=scan['passed'])
    print(json.dumps(result, indent=2)); return result


def scan_staged() -> dict:
    """Read actual index blobs; fail if they differ from the approved candidates."""
    approved = json.loads((OUTPUT/'source_manifest.json').read_text())
    known = {row['path']: row for row in approved['entries']}
    names = subprocess.check_output(['git', 'ls-files', '--cached', '-z'], cwd=ROOT).split(b'\0')
    paths = [name.decode('utf-8') for name in names if name]
    findings = []; violations = []; bytes_scanned = 0
    for path in paths:
        reason = denied_reason(path)
        if reason or candidate_kind(path) is None:
            violations.append(dict(path=path, rule=reason or 'not_in_source_allowlist')); continue
        data = subprocess.check_output(['git', 'show', ':'+path], cwd=ROOT)
        if len(data)>MAX_BYTES:
            violations.append(dict(path=path, rule='over_size_limit')); continue
        bytes_scanned += len(data)
        if path not in known or hashlib.sha256(data).hexdigest()!=known[path]['sha256']:
            violations.append(dict(path=path, rule='staged_blob_not_approved_manifest_content'))
        items, _ = scan_bytes(path, data); findings.extend(items)
    missing = sorted(set(known)-set(paths))
    result = dict(mode='actual_staged_index_blobs', files_scanned=len(paths), bytes_scanned=bytes_scanned,
        findings=findings, path_or_content_violations=violations, missing_approved_candidates=missing,
        required_notebooks_and_main_SU2NQS_present=not (REQUIRED-set(paths)),
        no_secret_values_or_context_recorded=True, no_Git_mutation=True,
        passed=bool(paths) and not findings and not violations and not missing,
        source_manifest_sha256=hashlib.sha256((OUTPUT/'source_manifest.json').read_bytes()).hexdigest())
    write_json(OUTPUT/'staged_secret_scan.json', result)
    print(json.dumps(dict(mode=result['mode'], files=len(paths), bytes=bytes_scanned,
        findings=len(findings), violations=len(violations), missing=len(missing), passed=result['passed']), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-staged', action='store_true')
    args = parser.parse_args()
    result = scan_staged() if args.scan_staged else prepare()
    if not result['passed']: raise SystemExit(2)
